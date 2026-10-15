"""
SkipGCN Model (порт HCP_GCN).

Порт модели SkipGCN из проекта HCP_GCN (``src/models.py``):
графовая свертка с тремя слоями ``GCNConv`` и skip-соединениями
через конкатенацию, глобальный пулинг и линейный классификатор.
Класс ``SkipGCNForCFMSR`` адаптирует модель под графовый пайплайн
проекта (батч ``torch_geometric.data.Data`` с ``edge_attr``).

Отклонения от оригинала (исправления):
- ``super().__init__()`` вызывается первым;
- ``pooling_type`` реально используется (в оригинале всегда max-pool);
- ``dropout`` отключён в eval-режиме (в оригинале ``F.dropout``
  без флага ``training`` ронял признаки и при инференсе).
"""

# 1. Standard library
from typing import Literal, Optional

# 2. External libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn.conv import GCNConv


SUPPORTED_POOLING = ("global_max_pool", "global_mean_pool")


class SkipGCN(nn.Module):
    """
    SkipGCN.

    Графовая сверточная сеть со skip-соединениями для классификации графов.
    Каждый следующий слой получает конкатенацию выхода предыдущего слоя
    и остатка, что сохраняет локальные и глобальные структуры графа.

    Parameters
    ----------
    hidden_channels : int
        Размерность скрытого представления вершины.
    dropout : float
        Вероятность dropout после свёрток.
    num_node_features : int
        Размерность входных признаков узла.
    pooling_type : str, optional
        Тип глобального пулинга: ``global_max_pool`` или ``global_mean_pool``.
    """

    def __init__(
        self,
        hidden_channels: int,
        dropout: float,
        num_node_features: int,
        pooling_type: str = "global_max_pool",
    ) -> None:
        """
        Инициализация слоёв SkipGCN.

        Parameters
        ----------
        hidden_channels : int
            Размерность скрытого представления вершины.
        dropout : float
            Вероятность dropout после свёрток.
        num_node_features : int
            Размерность входных признаков узла.
        pooling_type : str, optional
            Тип глобального пулинга.
        """
        super().__init__()

        if int(hidden_channels) < 1:
            raise ValueError(
                f"hidden_channels должен быть >= 1, получено {hidden_channels}"
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(
                f"dropout должен быть в [0, 1), получено {dropout}"
            )
        if int(num_node_features) < 1:
            raise ValueError(
                f"num_node_features должен быть >= 1, получено {num_node_features}"
            )
        if str(pooling_type) not in SUPPORTED_POOLING:
            raise ValueError(
                f"pooling_type должен быть одним из {SUPPORTED_POOLING}, "
                f"получено {pooling_type!r}"
            )

        hidden_channels = int(hidden_channels)
        self.drop_p = float(dropout)
        self.pooling_type = str(pooling_type)

        # Значения эмбеддингов по умолчанию до первого forward.
        self.embeddings = np.ones(2 * hidden_channels + num_node_features)

        self.conv1 = GCNConv(num_node_features, hidden_channels, normalize=False)
        # Skip connection: hidden_channels + num_node_features для остатка.
        self.conv2 = GCNConv(
            hidden_channels + num_node_features,
            hidden_channels + num_node_features,
            normalize=False,
        )
        # Skip connection: hidden_channels + hidden_channels + остатки.
        self.conv3 = GCNConv(
            2 * hidden_channels + num_node_features,
            2 * hidden_channels + num_node_features,
            normalize=False,
        )
        self.bn_conv1 = nn.BatchNorm1d(num_features=hidden_channels)
        self.bn_conv2 = nn.BatchNorm1d(
            num_features=hidden_channels + num_node_features
        )
        self.bn_conv3 = nn.BatchNorm1d(
            num_features=2 * hidden_channels + num_node_features
        )
        self.bn1 = nn.BatchNorm1d(
            num_features=2 * hidden_channels + num_node_features
        )
        self.lin = nn.Linear(
            in_features=2 * hidden_channels + num_node_features, out_features=1
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        pooling_type: Optional[str],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Прямой проход модели.

        Parameters
        ----------
        x : torch.Tensor
            Признаки узлов формы ``(N, F)``.
        edge_index : torch.Tensor
            Индексы рёбер формы ``(2, E)``.
        edge_weight : Optional[torch.Tensor]
            Веса рёбер формы ``(E,)`` или None.
        pooling_type : Optional[str]
            Переопределение пулинга на один проход; None — конфиг модели.
        batch : torch.Tensor
            Вектор принадлежности узлов графам батча формы ``(N,)``.

        Returns
        -------
        torch.Tensor
            Логиты предсказаний формы ``(B, 1)``.
        """
        pooling = self.pooling_type if pooling_type is None else str(pooling_type)
        if pooling not in SUPPORTED_POOLING:
            raise ValueError(
                f"pooling_type должен быть одним из {SUPPORTED_POOLING}, "
                f"получено {pooling!r}"
            )

        residual_0 = x
        # 1. Получение эмбеддингов узлов.
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.bn_conv1(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        residual_1 = x
        x = torch.cat([x, residual_0], dim=1)
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.bn_conv2(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = torch.cat([x, residual_1], dim=1)
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn_conv3(x)
        # 2. Readout-слой: глобальный пулинг.
        if pooling == "global_mean_pool":
            x = global_mean_pool(x, batch=batch)
        else:
            x = global_max_pool(x, batch=batch)
        # BatchNorm эмбеддингов.
        x = self.bn1(x)
        self.embeddings = x.detach().to("cpu").numpy()

        # 3. Финальный классификатор.
        x = self.lin(x)
        return x

    def get_embeddings(self) -> np.ndarray:
        """
        Get embeddings.

        Возвращает эмбеддинги графов с последнего прямого прохода.

        Returns
        -------
        np.ndarray
            Массив эмбеддингов формы ``(B, 2 * hidden + F)``.
        """
        return self.embeddings


def to_edge_weight(edge_attr: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Convert edge_attr to edge_weight.

    Приводит признаки рёбер CF-MSR к весам рёбер SkipGCN.

    Parameters
    ----------
    edge_attr : Optional[torch.Tensor]
        Признаки рёбер формы ``(E, 1)``/``(E,)`` или None.

    Returns
    -------
    Optional[torch.Tensor]
        Веса рёбер формы ``(E,)`` или None.
    """
    if edge_attr is None:
        return None
    if edge_attr.dim() == 2 and edge_attr.size(1) == 1:
        return edge_attr.squeeze(-1).float()
    if edge_attr.dim() == 1:
        return edge_attr.float()
    raise ValueError(
        "edge_attr должен иметь форму (E, 1) или (E,), "
        f"получено {tuple(edge_attr.shape)}"
    )


class SkipGCNForCFMSR(nn.Module):
    """
    SkipGCNForCFMSR.

    Адаптер SkipGCN под пайплайн проекта: принимает батч
    ``torch_geometric.data.Data`` (``x``, ``edge_index``, ``edge_attr``)
    и возвращает логиты ``(B, 1)``, как ``CFMSR``.

    Parameters
    ----------
    training_cfg : DictConfig
        Ветка ``training`` конфигурации; читается подсекция ``skipgcn``
        (``hidden_channels``, ``dropout``, ``pooling_type``).
    num_nodes : int
        Число узлов графа (для совместимости фабрики; топология берётся
        из батча, значение не используется напрямую).
    input_dim : int, optional
        Размерность признаков узла ``x``, по умолчанию 1.
    """

    def __init__(
        self,
        training_cfg: DictConfig,
        num_nodes: int,
        input_dim: int = 1,
    ) -> None:
        """
        Инициализация адаптера SkipGCN.

        Parameters
        ----------
        training_cfg : DictConfig
            Ветка ``training`` конфигурации.
        num_nodes : int
            Число узлов графа.
        input_dim : int, optional
            Размерность признаков узла.
        """
        super().__init__()

        skip_cfg = training_cfg.get("skipgcn", {})
        hidden_channels = int(skip_cfg.get("hidden_channels", 15))
        dropout = float(skip_cfg.get("dropout", 0.2))
        pooling_type = str(skip_cfg.get("pooling_type", "global_max_pool"))

        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.pooling_type: Literal["global_max_pool", "global_mean_pool"] = (
            pooling_type  # type: ignore[assignment]
        )
        self.gcn = SkipGCN(
            hidden_channels=hidden_channels,
            dropout=dropout,
            num_node_features=int(input_dim),
            pooling_type=pooling_type,
        )

    def forward(self, batch: Data) -> torch.Tensor:
        """
        Прямой проход адаптера.

        Parameters
        ----------
        batch : Data
            Пакет графовых данных из PyTorch Geometric.

        Returns
        -------
        torch.Tensor
            Логиты предсказаний формы ``(B, 1)``.
        """
        if batch.x is None:
            raise ValueError("В батче отсутствуют признаки узлов (batch.x is None)")
        x = batch.x.float()
        edge_weight = to_edge_weight(getattr(batch, "edge_attr", None))
        batch_vec = getattr(batch, "batch", None)
        if batch_vec is None:
            batch_vec = torch.zeros(
                x.size(0), dtype=torch.long, device=x.device
            )
        return self.gcn(
            x=x,
            edge_index=batch.edge_index,
            edge_weight=edge_weight,
            pooling_type=self.pooling_type,
            batch=batch_vec,
        )

    def get_embeddings(self) -> np.ndarray:
        """
        Get embeddings.

        Возвращает эмбеддинги графов с последнего прямого прохода.

        Returns
        -------
        np.ndarray
            Массив эмбеддингов.
        """
        return self.gcn.get_embeddings()
