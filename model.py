"""
CFMSR Model Implementation.

Реализация модели CFMSR для обработки графовых данных.
Модель использует механизмы внимания с учетом признаков ребер
и множественные seed-токены для агрегации информации на уровне графа.
"""

# 1. Standard library
from typing import Literal, Optional, Tuple

# 2. External libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import scatter, to_dense_adj, to_dense_batch

# 3. Local
from config_utils import parse_optional_float


def drop_edges(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    p: float,
    training: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    DropEdge.

    Случайно удаляет рёбра с вероятностью ``p`` в режиме train.

    Parameters
    ----------
    edge_index : torch.Tensor
        Индексы рёбер формы ``(2, E)``.
    edge_attr : Optional[torch.Tensor]
        Признаки рёбер формы ``(E, F)`` или None.
    p : float
        Вероятность удаления ребра.
    training : bool
        Флаг режима обучения.

    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.Tensor]]
        Обновлённые ``edge_index`` и ``edge_attr``.
    """
    if (not training) or p <= 0.0 or edge_index.numel() == 0:
        return edge_index, edge_attr

    if not 0.0 <= float(p) < 1.0:
        raise ValueError(f"drop_edge должен быть в [0, 1), получено {p}")

    e = edge_index.size(1)
    keep = torch.rand(e, device=edge_index.device) >= float(p)
    if not keep.any():
        # Оставляем хотя бы одно ребро для численной стабильности.
        keep[int(torch.randint(0, e, (1,), device=edge_index.device).item())] = True

    new_index = edge_index[:, keep]
    new_attr = edge_attr[keep] if edge_attr is not None else None
    return new_index, new_attr


def add_gaussian_noise(
    t: torch.Tensor,
    std: float,
    training: bool,
    relative: bool = False,
) -> torch.Tensor:
    """
    Add Gaussian noise.

    Добавляет гауссов шум в режиме train.

    Parameters
    ----------
    t : torch.Tensor
        Входной тензор.
    std : float
        Стандартное отклонение шума (абсолютное или относительное).
    training : bool
        Флаг режима обучения.
    relative : bool, optional
        Если True, ``std`` умножается на ``t.std()``.

    Returns
    -------
    torch.Tensor
        Тензор с шумом или исходный тензор.
    """
    if (not training) or std <= 0.0 or t.numel() == 0:
        return t

    if float(std) < 0.0:
        raise ValueError(f"noise std должен быть >= 0, получено {std}")

    scale = float(std)
    if relative:
        data_std = float(t.detach().std().clamp_min(1e-6).item())
        scale = scale * data_std

    return t + torch.randn_like(t) * scale


class EdgeMultiHeadAttention(nn.Module):
    """
    EdgeMultiHeadAttention.

    Многоголовое внимание с учетом признаков ребер графа.
    Признаки ребер преобразуются в аддитивное смещение (bias)
    для матрицы оценок внимания.
    """

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float,
            edge_dim: int = 1,
            use_edge_norm: bool = True,
    ) -> None:
        """
        Инициализация слоя многоголового внимания с ребрами.

        Parameters
        ----------
        hidden_dim : int
            Размерность скрытого представления.
        num_heads : int
            Количество голов внимания.
        dropout : float
            Вероятность dropout для весов внимания.
        edge_dim : int, optional
            Размерность признаков ребер, по умолчанию 1.
        use_edge_norm : bool, optional
            Использовать ли ``BatchNorm1d`` для признаков рёбер.
        """
        super().__init__()

        # Валидация размерностей
        assert hidden_dim % num_heads == 0, (
            "hidden_dim должен быть кратен num_heads"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Линейные проекции для Q, K, V и выхода
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)

        # Нормализация и преобразование признаков ребер в bias
        self.edge_norm = (
            nn.BatchNorm1d(edge_dim)
            if (edge_dim > 0 and use_edge_norm)
            else None
        )
        self.edge_to_bias = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: Optional[torch.Tensor],
            batch: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Прямой проход слоя внимания.

        Parameters
        ----------
        x : torch.Tensor
            Входные признаки узлов, форма (B, N, H).
        edge_index : torch.Tensor
            Индексы ребер, форма (2, E).
        edge_attr : Optional[torch.Tensor]
            Признаки ребер, форма (E, edge_dim).
        batch : torch.Tensor
            Вектор батчей для узлов, форма (N,).
        mask : torch.Tensor
            Булева маска валидных узлов, форма (B, N).

        Returns
        -------
        torch.Tensor
            Выходные признаки узлов, форма (B, N, H).
        """
        B, N, H = x.shape
        h = self.num_heads
        d = self.head_dim

        # Вычисление Q, K, V и приведение к форме (B, h, N, d)
        q = self.q_proj(x).view(B, N, h, d).transpose(1, 2)
        k = self.k_proj(x).view(B, N, h, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, h, d).transpose(1, 2)

        # Матрица оценок внимания (B, h, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)

        # Добавление смещения на основе признаков ребер
        if (
                edge_attr is not None
                and edge_index.numel() > 0
                and edge_attr.size(0) > 0
        ):
            e = edge_attr
            if e.dim() == 1:
                e = e.unsqueeze(-1)

            # Нормализация признаков ребер (с защитой от batch_size=1)
            if self.edge_norm is not None:
                if self.training and e.size(0) < 2:
                    pass
                else:
                    e = self.edge_norm(e)

            # Преобразование признаков ребер в bias
            edge_bias = self.edge_to_bias(e)

            # Приведение к плотному виду (B, N, N, h) -> (B, h, N, N)
            bias_dense = to_dense_adj(
                edge_index,
                batch=batch,
                edge_attr=edge_bias,
                max_num_nodes=N,
            )
            if bias_dense.dim() == 3:
                bias_dense = bias_dense.unsqueeze(-1)
            bias_dense = bias_dense.permute(0, 3, 1, 2)

            scores = scores + bias_dense

        # Маскирование невалидных ключей и запросов
        key_mask = (~mask).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(key_mask, -1e9)

        query_mask = (~mask).unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(query_mask, 0.0)

        # Вычисление весов внимания и применение к значениям
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, H)
        out = self.out_proj(out)

        return out


class CFMSRBlock(nn.Module):
    """
    CFMSRBlock.

    Трансформерный блок, состоящий из слоя внимания с ребрами
    и полносвязной сети (FFN) с остаточными связями и нормализацией.
    """

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            ffn_mult: float,
            dropout: float,
            edge_dim: int,
            use_edge_norm: bool = True,
    ) -> None:
        """
        Инициализация трансформерного блока.

        Parameters
        ----------
        hidden_dim : int
            Размерность скрытого представления.
        num_heads : int
            Количество голов внимания.
        ffn_mult : float
            Множитель размерности для скрытого слоя FFN.
        dropout : float
            Вероятность dropout.
        edge_dim : int
            Размерность признаков ребер.
        use_edge_norm : bool, optional
            Пробрасывается в ``EdgeMultiHeadAttention``.
        """
        super().__init__()

        ffn_dim = max(1, int(hidden_dim * ffn_mult))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = EdgeMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            use_edge_norm=use_edge_norm,
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: Optional[torch.Tensor],
            batch: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Прямой проход трансформерного блока.

        Parameters
        ----------
        x : torch.Tensor
            Входные признаки узлов, форма (B, N, H).
        edge_index : torch.Tensor
            Индексы ребер, форма (2, E).
        edge_attr : Optional[torch.Tensor]
            Признаки ребер, форма (E, edge_dim).
        batch : torch.Tensor
            Вектор батчей для узлов, форма (N,).
        mask : torch.Tensor
            Булева маска валидных узлов, форма (B, N).

        Returns
        -------
        torch.Tensor
            Выходные признаки узлов, форма (B, N, H).
        """
        # Остаточная связь для слоя внимания
        x = x + self.attn(
            self.norm1(x),
            edge_index,
            edge_attr,
            batch,
            mask,
        )

        # Остаточная связь для FFN
        x = x + self.ffn(self.norm2(x))

        return x


class MultiSeedReadout(nn.Module):
    """
    MultiSeedReadout.

    Механизм чтения графа с использованием множественных seed-токенов.
    Seed-токены агрегируют информацию из узлов графа через механизм
    кросс-внимания, после чего признаки конкатенируются со статистиками.
    """

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_seeds: int,
            dropout: float,
    ) -> None:
        """
        Инициализация механизма чтения графа.

        Parameters
        ----------
        hidden_dim : int
            Размерность скрытого представления.
        num_heads : int
            Количество голов внимания.
        num_seeds : int
            Количество seed-токенов для агрегации.
        dropout : float
            Вероятность dropout для весов внимания.
        """
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            "hidden_dim должен быть кратен num_heads"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_seeds = num_seeds

        # Обучаемые seed-токены
        self.seeds = nn.Parameter(
            torch.randn(1, num_seeds, hidden_dim) * 0.02
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Проекция для кросс-внимания
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Итоговая размерность: flat + mean + max
        self.out_dim = hidden_dim * num_seeds + 2 * hidden_dim

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Прямой проход механизма чтения.

        Parameters
        ----------
        x : torch.Tensor
            Входные признаки узлов, форма (B, N, H).
        mask : torch.Tensor
            Булева маска валидных узлов, форма (B, N).

        Returns
        -------
        torch.Tensor
            Векторное представление графа, форма (B, out_dim).
        """
        B, N, H = x.shape
        K = self.num_seeds
        h = self.num_heads
        d = self.head_dim

        x_norm = self.norm(x)

        # Расширение seed-токенов на размер батча
        seeds = self.seeds.expand(B, -1, -1)

        # Вычисление Q для seeds и K, V для узлов
        q = self.q_proj(seeds).view(B, K, h, d).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, h, d).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, h, d).transpose(1, 2)

        # Оценки внимания и маскирование
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        key_mask = (~mask).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(key_mask, -1e9)

        # Применение softmax и dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Агрегация информации
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, K, H)
        out = self.out_proj(out)

        # Формирование итогового вектора графа
        flat = out.flatten(1)
        mean = out.mean(dim=1)
        maxv = out.max(dim=1).values

        return torch.cat([flat, mean, maxv], dim=-1)


class EdgeReadout(nn.Module):
    """
    EdgeReadout.

    Агрегация признаков рёбер в вектор графа (mean/max или gated).
    """

    def __init__(
            self,
            edge_dim: int,
            out_dim: int,
            mode: Literal["mean", "gated"],
            dropout: float,
    ) -> None:
        """
        Initialize edge readout.

        Parameters
        ----------
        edge_dim : int
            Размерность признаков рёбер.
        out_dim : int
            Размерность проекции ребра (до concat mean/max).
        mode : Literal["mean", "gated"]
            Режим агрегации.
        dropout : float
            Вероятность dropout.
        """
        super().__init__()
        if mode not in ("mean", "gated"):
            raise ValueError(
                f"edge_readout mode должен быть mean|gated, получено {mode!r}"
            )
        if out_dim < 1:
            raise ValueError(f"edge_readout_dim должен быть >= 1, получено {out_dim}")

        self.mode = mode
        self.proj_dim = int(out_dim)
        self.out_dim = 2 * self.proj_dim
        self.dropout = nn.Dropout(dropout)

        if mode == "mean":
            self.proj = nn.Linear(edge_dim, self.proj_dim)
            self.gate = None
            self.val = None
        else:
            self.proj = None
            self.gate = nn.Linear(edge_dim, 1)
            self.val = nn.Linear(edge_dim, self.proj_dim)

    def forward(
            self,
            edge_attr: torch.Tensor,
            edge_batch: torch.Tensor,
            num_graphs: int,
    ) -> torch.Tensor:
        """
        Forward edge readout.

        Parameters
        ----------
        edge_attr : torch.Tensor
            Признаки рёбер ``(E, edge_dim)``.
        edge_batch : torch.Tensor
            Индекс графа для каждого ребра ``(E,)``.
        num_graphs : int
            Число графов в батче.

        Returns
        -------
        torch.Tensor
            Вектор ``(num_graphs, out_dim)``.
        """
        if edge_attr.numel() == 0 or edge_attr.size(0) == 0:
            return edge_attr.new_zeros((num_graphs, self.out_dim))

        e = edge_attr
        if e.dim() == 1:
            e = e.unsqueeze(-1)

        if self.mode == "mean":
            assert self.proj is not None
            h = self.dropout(self.proj(e))
            mean_v = global_mean_pool(h, edge_batch, size=num_graphs)
            max_v = global_max_pool(h, edge_batch, size=num_graphs)
            return torch.cat([mean_v, max_v], dim=-1)

        assert self.gate is not None and self.val is not None
        gate = torch.sigmoid(self.gate(e))
        val = self.dropout(self.val(e))
        weighted = gate * val
        sum_w = scatter(weighted, edge_batch, dim=0, dim_size=num_graphs, reduce="sum")
        sum_g = scatter(gate, edge_batch, dim=0, dim_size=num_graphs, reduce="sum")
        gated = sum_w / sum_g.clamp_min(1e-6)
        mean_v = global_mean_pool(val, edge_batch, size=num_graphs)
        return torch.cat([gated, mean_v], dim=-1)


class CFMSR(nn.Module):
    """
    CFMSR.

    Главная архитектура модели CFMSR.
    Объединяет проекцию входных признаков, опциональное встраивание узлов,
    стек трансформерных блоков с учетом ребер, механизм чтения
    и финальный классификатор.
    """

    def __init__(
            self,
            training_cfg: DictConfig,
            num_nodes: int,
            input_dim: int = 1,
            edge_dim: int = 1,
            use_node_embedding: Optional[bool] = None,
    ) -> None:
        """
        Инициализация модели CFMSR.

        Parameters
        ----------
        training_cfg : DictConfig
            Конфигурация параметров обучения и архитектуры.
        num_nodes : int
            Общее количество уникальных узлов в графе (для эмбеддингов).
        input_dim : int, optional
            Размерность входных признаков узлов, по умолчанию 1.
        edge_dim : int, optional
            Размерность признаков ребер, по умолчанию 1.
        use_node_embedding : Optional[bool], optional
            Флаг эмбеддингов узлов; ``None`` — из ``training_cfg``.
        """
        super().__init__()

        hidden_dim = int(training_cfg.hidden_dim)
        num_layers = int(training_cfg.num_layers)
        num_heads = int(training_cfg.num_heads)
        ffn_mult = float(training_cfg.ffn_mult)
        dropout = float(training_cfg.dropout)
        num_seeds = int(training_cfg.num_seeds)

        classifier_dropout = parse_optional_float(
            training_cfg.get("classifier_dropout", None)
        )

        assert hidden_dim % num_heads == 0, (
            "hidden_dim должен быть кратен num_heads"
        )

        if classifier_dropout is None:
            classifier_dropout = dropout

        if use_node_embedding is None:
            use_node_embedding = bool(
                training_cfg.get("use_node_embedding", True)
            )

        drop_edge = float(training_cfg.get("drop_edge", 0.0) or 0.0)
        node_noise_std = float(training_cfg.get("node_noise_std", 0.0) or 0.0)
        edge_noise_std = float(training_cfg.get("edge_noise_std", 0.0) or 0.0)

        if not 0.0 <= drop_edge < 1.0:
            raise ValueError(
                f"drop_edge должен быть в [0, 1), получено {drop_edge}"
            )
        if node_noise_std < 0.0:
            raise ValueError(
                f"node_noise_std должен быть >= 0, получено {node_noise_std}"
            )
        if edge_noise_std < 0.0:
            raise ValueError(
                f"edge_noise_std должен быть >= 0, получено {edge_noise_std}"
            )

        self.drop_edge_p = drop_edge
        self.node_noise_std = node_noise_std
        self.edge_noise_std = edge_noise_std

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.edge_dim = int(edge_dim)

        use_edge_norm = bool(training_cfg.get("edge_batchnorm", True))

        edge_readout_mode = str(
            training_cfg.get("edge_readout", "none")
        ).strip().lower()
        edge_readout_dim = int(training_cfg.get("edge_readout_dim", 16))
        if edge_readout_mode not in ("none", "null", "", "mean", "gated"):
            raise ValueError(
                f"edge_readout должен быть none|mean|gated, "
                f"получено {edge_readout_mode!r}"
            )

        # Входная проекция и нормализация
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        else:
            self.node_embedding = None

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Стек трансформерных блоков
        self.layers = nn.ModuleList(
            [
                CFMSRBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    use_edge_norm=use_edge_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # Механизм чтения и классификатор
        self.readout = MultiSeedReadout(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_seeds=num_seeds,
            dropout=dropout,
        )

        if edge_readout_mode in ("mean", "gated"):
            self.edge_readout: Optional[EdgeReadout] = EdgeReadout(
                edge_dim=edge_dim,
                out_dim=edge_readout_dim,
                mode=edge_readout_mode,  # type: ignore[arg-type]
                dropout=dropout,
            )
        else:
            self.edge_readout = None

        readout_dim = self.readout.out_dim
        if self.edge_readout is not None:
            readout_dim = readout_dim + self.edge_readout.out_dim
        clf_hidden = max(1, hidden_dim // 2)

        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, clf_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(clf_hidden, 1),
        )

        # Инициализация весов
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Инициализация весов модулей модели.

        Parameters
        ----------
        module : nn.Module
            Инициализируемый модуль PyTorch.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, batch: Data) -> torch.Tensor:
        """
        Прямой проход модели.

        Parameters
        ----------
        batch : Data
            Пакет графовых данных из PyTorch Geometric.

        Returns
        -------
        torch.Tensor
            Логиты предсказаний, форма (B, 1).
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = getattr(batch, "edge_attr", None)
        batch_idx = batch.batch

        # Augmentations только в train, до проекции и стека блоков.
        x = add_gaussian_noise(
            x,
            std=self.node_noise_std,
            training=self.training,
            relative=False,
        )
        edge_index, edge_attr = drop_edges(
            edge_index,
            edge_attr,
            p=self.drop_edge_p,
            training=self.training,
        )
        if edge_attr is not None:
            edge_attr = add_gaussian_noise(
                edge_attr,
                std=self.edge_noise_std,
                training=self.training,
                relative=True,
            )

        # Приведение к плотному представлению для работы с масками
        x_dense, mask = to_dense_batch(x, batch_idx)

        x_dense = self.input_proj(x_dense)

        # Добавление позиционных/узловых эмбеддингов
        if self.node_embedding is not None:
            B, N, _ = x_dense.shape
            node_ids = (
                torch.arange(N, device=x_dense.device)
                .unsqueeze(0)
                .expand(B, N)
            )
            x_dense = x_dense + self.node_embedding(node_ids)

        x_dense = self.input_norm(x_dense)
        x_dense = self.input_dropout(x_dense)

        # Прогон через стек трансформерных блоков
        for layer in self.layers:
            x_dense = layer(
                x=x_dense,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch_idx,
                mask=mask,
            )

        # Чтение графа и классификация
        graph_vec = self.readout(x_dense, mask)

        if self.edge_readout is not None:
            num_graphs = int(x_dense.size(0))
            if (
                    edge_attr is not None
                    and edge_index.numel() > 0
                    and edge_attr.size(0) > 0
            ):
                edge_batch = batch_idx[edge_index[0]]
                edge_vec = self.edge_readout(
                    edge_attr, edge_batch, num_graphs=num_graphs,
                )
            else:
                edge_vec = graph_vec.new_zeros(
                    (num_graphs, self.edge_readout.out_dim)
                )
            graph_vec = torch.cat([graph_vec, edge_vec], dim=-1)

        logits = self.classifier(graph_vec)

        return logits
