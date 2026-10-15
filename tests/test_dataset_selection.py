"""
Tests for node/edge selection on WeightedFullGraphDataset.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from dataset import CFMSRDataModule, WeightedFullGraphDataset


def _tiny_dataset() -> WeightedFullGraphDataset:
    # 4 samples, 4 nodes, 6 directed edges (3 undirected).
    node_features = np.array(
        [
            [1.0, 0.0, 5.0, 0.1],
            [1.1, 0.0, 4.5, 0.2],
            [0.0, 2.0, 0.0, 0.0],
            [0.1, 2.1, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 0, 2, 1, 2],
            [1, 0, 2, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    # edge_attr: (E, n_samples)
    edge_attr = np.array(
        [
            [1.0, 1.1, 0.0, 0.0],
            [1.0, 1.1, 0.0, 0.0],
            [0.5, 0.4, 0.0, 0.0],
            [0.5, 0.4, 0.0, 0.0],
            [0.1, 0.1, 2.0, 2.1],
            [0.1, 0.1, 2.0, 2.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return WeightedFullGraphDataset(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        labels=labels,
    )


def test_select_nodes_reindexes_edges() -> None:
    """После select_nodes индексы рёбер лежат в [0, N_new)."""
    ds = _tiny_dataset()
    keep = np.array([0, 2], dtype=np.int64)
    out = ds.select_nodes(keep)
    assert out.node_features.shape[1] == 2
    assert int(out.edge_index.max()) < 2
    assert int(out.edge_index.min()) >= 0


def test_select_edges_keeps_both_directions() -> None:
    """select_edges оставляет оба направления выбранных пар."""
    ds = _tiny_dataset()
    # Оставляем только первую undirected пару (рёбра 0 и 1).
    mask = np.array([True, True, False, False, False, False])
    out = ds.select_edges(mask)
    assert out.edge_index.size(1) == 2


def test_selection_stats_use_train_only() -> None:
    """Статистики отбора считаются только по train: смена val не влияет."""
    ds = _tiny_dataset()

    # Минимальный data_cfg / training_cfg для вызова _select_nodes.
    data_cfg = OmegaConf.create(
        {
            "data_csv": "unused.csv",
            "train_valid_graphs_csv": "unused.csv",
            "target_col": "y",
            "batch_size": 2,
            "val_ratio": 0.5,
            "num_workers": 0,
            "data_row_shift": 0,
            "feature_selection": {"method": "variance", "top_k": 2},
            "edge_selection": {"method": "none"},
        }
    )
    training_cfg = OmegaConf.create({"seed": 0})

    # Обходим полный __init__ (читает CSV): создаём объект без вызова.
    dm = object.__new__(CFMSRDataModule)
    dm.data_cfg = data_cfg
    dm.training_cfg = training_cfg
    dm.seed = 0
    dm.num_nodes = 4
    dm.feature_cols = ["a", "b", "c", "d"]
    dm._all_feature_cols = list(dm.feature_cols)
    dm._all_num_nodes = 4

    train_idx = np.array([0, 1], dtype=np.int64)
    keep1 = dm._select_nodes(ds, train_idx)

    # Меняем val-метки (samples 2,3) — keep не должен измениться.
    ds2 = ds.replace()
    ds2.labels[2] = 1.0 - ds2.labels[2]
    ds2.labels[3] = 1.0 - ds2.labels[3]
    keep2 = dm._select_nodes(ds2, train_idx)
    assert keep1 is not None and keep2 is not None
    assert np.array_equal(keep1, keep2)


def test_kept_nodes_applied_consistently() -> None:
    """Один и тот же keep применяется к двум датасетам одинаково."""
    ds = _tiny_dataset()
    keep = np.array([0, 2], dtype=np.int64)
    a = ds.select_nodes(keep)
    b = ds.select_nodes(keep)
    assert a.node_features.shape == b.node_features.shape
    assert torch.equal(a.edge_index, b.edge_index)
