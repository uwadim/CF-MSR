"""
Model factory.

Фабрика графовых моделей: выбор архитектуры по ``training.model_name``.
Единая точка создания модели для single-split обучения (``train.py``),
кросс-валидации (``crossval.py``) и HPO (``hpo.py``).
"""

# 1. Standard library
from typing import Tuple

# 2. External libraries
import torch.nn as nn
from omegaconf import DictConfig

# 3. Local
from model import CFMSR
from skipgcn_model import SkipGCNForCFMSR


SUPPORTED_MODEL_NAMES: Tuple[str, ...] = ("cfmsr", "skipgcn")


def get_model_name(training_cfg: DictConfig) -> str:
    """
    Get model name.

    Читает селектор архитектуры из ветки ``training``.

    Parameters
    ----------
    training_cfg : DictConfig
        Ветка ``training`` конфигурации.

    Returns
    -------
    str
        Имя модели в нижнем регистре без пробелов.
    """
    raw = training_cfg.get("model_name", "cfmsr")
    return str(raw).strip().lower()


def build_model(
    training_cfg: DictConfig,
    num_nodes: int,
    input_dim: int = 1,
) -> nn.Module:
    """
    Build graph model.

    Создаёт ``CFMSR`` или ``SkipGCNForCFMSR`` по ``training.model_name``.
    Обе модели принимают батч ``Data`` и возвращают логиты ``(B, 1)``.

    Parameters
    ----------
    training_cfg : DictConfig
        Ветка ``training`` конфигурации.
    num_nodes : int
        Число узлов графа (для эмбеддингов ``CFMSR``).
    input_dim : int, optional
        Размерность признаков узла, по умолчанию 1.

    Returns
    -------
    nn.Module
        Экземпляр выбранной модели.
    """
    name = get_model_name(training_cfg)
    if name == "cfmsr":
        return CFMSR(
            training_cfg=training_cfg,
            num_nodes=num_nodes,
            input_dim=input_dim,
        )
    if name == "skipgcn":
        return SkipGCNForCFMSR(
            training_cfg=training_cfg,
            num_nodes=num_nodes,
            input_dim=input_dim,
        )
    raise ValueError(
        f"training.model_name должен быть одним из {SUPPORTED_MODEL_NAMES}, "
        f"получено {name!r}"
    )
