"""
Tests for SkipGCN port (HCP_GCN) and CF-MSR adapter.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from model_factory import build_model, get_model_name
from skipgcn_model import SkipGCN, SkipGCNForCFMSR, to_edge_weight


def _skipgcn_cfg(**overrides: object) -> OmegaConf:
    base = {
        "model_name": "skipgcn",
        "skipgcn": {
            "hidden_channels": 8,
            "dropout": 0.0,
            "pooling_type": "global_max_pool",
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _batch(
    n_graphs: int = 2,
    input_dim: int = 3,
    with_edge_attr: bool = True,
) -> Batch:
    graphs = []
    for i in range(n_graphs):
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )
        kwargs = {
            "x": torch.randn(3, input_dim) + float(i),
            "edge_index": edge_index,
            "y": torch.tensor([float(i % 2)]),
        }
        if with_edge_attr:
            kwargs["edge_attr"] = torch.randn(edge_index.size(1), 1)
        graphs.append(Data(**kwargs))
    return Batch.from_data_list(graphs)


def test_forward_returns_graph_logits() -> None:
    """Forward возвращает логиты (B, 1) для многомерного x."""
    model = SkipGCNForCFMSR(
        training_cfg=_skipgcn_cfg(), num_nodes=3, input_dim=3
    )
    model.eval()
    out = model(_batch(n_graphs=2, input_dim=3))
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_forward_train_mode_runs() -> None:
    """Forward в train-режиме (dropout/BatchNorm) не падает."""
    cfg = _skipgcn_cfg()
    cfg.skipgcn.dropout = 0.5
    model = SkipGCNForCFMSR(training_cfg=cfg, num_nodes=3, input_dim=2)
    model.train()
    out = model(_batch(n_graphs=4, input_dim=2))
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()


def test_eval_is_deterministic() -> None:
    """В eval-режиме dropout выключен: два прохода совпадают."""
    cfg = _skipgcn_cfg()
    cfg.skipgcn.dropout = 0.5
    model = SkipGCNForCFMSR(training_cfg=cfg, num_nodes=3, input_dim=2)
    model.eval()
    batch = _batch(n_graphs=2, input_dim=2)
    first = model(batch)
    second = model(batch)
    assert torch.equal(first, second)


def test_mean_pooling_variant() -> None:
    """Пулинг global_mean_pool поддерживается и даёт (B, 1)."""
    cfg = _skipgcn_cfg()
    cfg.skipgcn.pooling_type = "global_mean_pool"
    model = SkipGCNForCFMSR(training_cfg=cfg, num_nodes=3, input_dim=2)
    model.eval()
    out = model(_batch(n_graphs=2, input_dim=2))
    assert out.shape == (2, 1)


def test_invalid_pooling_raises() -> None:
    """Неизвестный pooling_type отклоняется fail-fast."""
    with pytest.raises(ValueError, match="pooling_type"):
        SkipGCN(
            hidden_channels=8,
            dropout=0.0,
            num_node_features=2,
            pooling_type="global_add_pool",
        )


def test_to_edge_weight_shapes() -> None:
    """Конвертация edge_attr (E, 1)/(E,) и None."""
    w = to_edge_weight(torch.randn(4, 1))
    assert w is not None and w.shape == (4,)
    w = to_edge_weight(torch.randn(4))
    assert w is not None and w.shape == (4,)
    assert to_edge_weight(None) is None
    with pytest.raises(ValueError, match="edge_attr"):
        to_edge_weight(torch.randn(4, 2))


def test_single_graph_without_batch_vector() -> None:
    """Одиночный Data без batch-вектора обрабатывается как один граф."""
    model = SkipGCNForCFMSR(
        training_cfg=_skipgcn_cfg(), num_nodes=3, input_dim=2
    )
    model.eval()
    single = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, 1),
        y=torch.tensor([1.0]),
    )
    out = model(single)
    assert out.shape == (1, 1)


def test_embeddings_updated_after_forward() -> None:
    """get_embeddings возвращает массив (B, 2*hidden + F)."""
    model = SkipGCNForCFMSR(
        training_cfg=_skipgcn_cfg(), num_nodes=3, input_dim=2
    )
    model.eval()
    model(_batch(n_graphs=2, input_dim=2))
    emb = model.get_embeddings()
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (2, 2 * 8 + 2)


def test_factory_dispatch() -> None:
    """Фабрика строит CFMSR по умолчанию и SkipGCN по имени."""
    from model import CFMSR

    default_cfg = OmegaConf.create(
        {
            "hidden_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "ffn_mult": 2.0,
            "dropout": 0.0,
            "num_seeds": 2,
            "classifier_dropout": 0.0,
            "use_node_embedding": False,
            "drop_edge": 0.0,
            "node_noise_std": 0.0,
            "edge_noise_std": 0.0,
            "edge_batchnorm": True,
            "edge_readout": "none",
            "edge_readout_dim": 8,
        }
    )
    assert get_model_name(default_cfg) == "cfmsr"
    assert isinstance(
        build_model(default_cfg, num_nodes=3, input_dim=1), CFMSR
    )
    skip_cfg = _skipgcn_cfg()
    assert get_model_name(skip_cfg) == "skipgcn"
    assert isinstance(
        build_model(skip_cfg, num_nodes=3, input_dim=2), SkipGCNForCFMSR
    )
    bad_cfg = OmegaConf.create({"model_name": "gat"})
    with pytest.raises(ValueError, match="model_name"):
        build_model(bad_cfg, num_nodes=3, input_dim=1)
