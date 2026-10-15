"""
Tests for EdgeReadout branch (Ш3).
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from model import CFMSR, EdgeReadout


def _tiny_cfg(**overrides: object) -> OmegaConf:
    base = {
        "hidden_dim": 16,
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
    base.update(overrides)
    return OmegaConf.create(base)


def _batch(n_graphs: int = 2, n_edges: int = 4) -> Batch:
    graphs = []
    for _ in range(n_graphs):
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long,
        )[:, :n_edges]
        graphs.append(
            Data(
                x=torch.randn(3, 1),
                edge_index=edge_index,
                edge_attr=torch.randn(edge_index.size(1), 1),
                y=torch.tensor([1.0]),
            )
        )
    return Batch.from_data_list(graphs)


def test_edge_readout_none_keeps_parameter_count() -> None:
    """При edge_readout=none число параметров как у базовой модели."""
    cfg_none = _tiny_cfg(edge_readout="none")
    cfg_implicit = _tiny_cfg()
    # Убираем ключ edge_readout, чтобы проверить дефолт через get.
    del cfg_implicit["edge_readout"]
    m1 = CFMSR(training_cfg=cfg_none, num_nodes=3, use_node_embedding=False)
    m2 = CFMSR(training_cfg=cfg_implicit, num_nodes=3, use_node_embedding=False)
    n1 = sum(p.numel() for p in m1.parameters())
    n2 = sum(p.numel() for p in m2.parameters())
    assert n1 == n2
    assert m1.edge_readout is None


def test_edge_readout_shapes() -> None:
    """Для mean/gated выход (B, out_dim)."""
    edge_attr = torch.randn(6, 1)
    edge_batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    for mode in ("mean", "gated"):
        module = EdgeReadout(edge_dim=1, out_dim=8, mode=mode, dropout=0.0)
        out = module(edge_attr, edge_batch, num_graphs=2)
        assert out.shape == (2, module.out_dim)
        assert module.out_dim == 16


def test_edge_readout_permutation_invariant() -> None:
    """Перестановка рёбер внутри графа не меняет выход."""
    edge_attr = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32,
    )
    edge_batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    module = EdgeReadout(edge_dim=1, out_dim=4, mode="mean", dropout=0.0)
    module.eval()
    out1 = module(edge_attr, edge_batch, num_graphs=1)
    perm = torch.tensor([2, 0, 3, 1])
    out2 = module(edge_attr[perm], edge_batch[perm], num_graphs=1)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_cfmsr_forward_with_edge_readout() -> None:
    """Логиты (B, 1), вход классификатора правильной размерности."""
    for mode in ("mean", "gated"):
        cfg = _tiny_cfg(edge_readout=mode, edge_readout_dim=8)
        model = CFMSR(training_cfg=cfg, num_nodes=3, use_node_embedding=False)
        model.eval()
        batch = _batch()
        logits = model(batch)
        assert logits.shape == (2, 1)
        expected_in = model.readout.out_dim + model.edge_readout.out_dim
        assert model.classifier[0].in_features == expected_in


def test_edge_readout_empty_edges() -> None:
    """edge_index формы (2, 0) → forward не падает."""
    cfg = _tiny_cfg(edge_readout="gated", edge_readout_dim=8)
    model = CFMSR(training_cfg=cfg, num_nodes=3, use_node_embedding=False)
    model.eval()
    data = Data(
        x=torch.randn(3, 1),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 1),
        y=torch.tensor([0.0]),
    )
    batch = Batch.from_data_list([data])
    logits = model(batch)
    assert logits.shape == (1, 1)
