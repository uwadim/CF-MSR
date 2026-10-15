"""
Tests for DropEdge, noise and label smoothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from model import CFMSR, add_gaussian_noise, drop_edges
from train import SmoothedBCEWithLogitsLoss


def test_drop_edges_noop_in_eval() -> None:
    """В eval-режиме drop_edges ничего не меняет."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.randn(3, 1)
    new_index, new_attr = drop_edges(edge_index, edge_attr, p=0.9, training=False)
    assert torch.equal(new_index, edge_index)
    assert new_attr is not None
    assert torch.equal(new_attr, edge_attr)


def test_noise_noop_in_eval() -> None:
    """В eval-режиме шум не применяется."""
    t = torch.randn(5, 3)
    out = add_gaussian_noise(t, std=1.0, training=False, relative=False)
    assert torch.equal(out, t)


def test_drop_edges_approx_ratio_in_train() -> None:
    """В train доля удалённых рёбер примерно равна p."""
    torch.manual_seed(0)
    e = 10_000
    edge_index = torch.stack(
        [torch.arange(e), (torch.arange(e) + 1) % e],
        dim=0,
    )
    p = 0.3
    new_index, _ = drop_edges(edge_index, None, p=p, training=True)
    kept_ratio = new_index.size(1) / e
    assert abs(kept_ratio - (1.0 - p)) < 0.05


def test_label_smoothing_zero_matches_bce() -> None:
    """При label_smoothing=0 loss совпадает с BCEWithLogitsLoss."""
    torch.manual_seed(0)
    logits = torch.randn(16)
    target = torch.randint(0, 2, (16,)).float()
    pos_weight = torch.tensor([1.5])

    smoothed = SmoothedBCEWithLogitsLoss(pos_weight=pos_weight, smoothing=0.0)
    plain = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    assert torch.allclose(smoothed(logits, target), plain(logits, target))


def test_cfmsr_without_node_embedding() -> None:
    """CFMSR с use_node_embedding=false не содержит node_embedding."""
    cfg = OmegaConf.create(
        {
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "ffn_mult": 2.0,
            "dropout": 0.1,
            "num_seeds": 2,
            "classifier_dropout": 0.1,
            "use_node_embedding": False,
            "drop_edge": 0.0,
            "node_noise_std": 0.0,
            "edge_noise_std": 0.0,
        }
    )
    model = CFMSR(training_cfg=cfg, num_nodes=5, use_node_embedding=False)
    assert model.node_embedding is None
