"""
Tests for edge-derived node features (Ш1).
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from dataset import (
    CFMSRDataModule,
    WeightedFullGraphDataset,
    compute_edge_node_features,
)
from model import CFMSR


def _tiny_dataset() -> WeightedFullGraphDataset:
    """3 узла, 2 образца, 4 направленных ребра (2 undirected)."""
    node_features = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5],
        ],
        dtype=np.float32,
    )
    # Рёбра: 0-1 и 1-2 (оба направления).
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    # edge_attr: (E, S)
    edge_attr = np.array(
        [
            [1.0, 2.0],  # 0->1
            [1.0, 2.0],  # 1->0
            [3.0, 4.0],  # 1->2
            [3.0, 4.0],  # 2->1
        ],
        dtype=np.float32,
    )
    labels = np.array([1.0, 0.0], dtype=np.float32)
    return WeightedFullGraphDataset(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        labels=labels,
    )


def test_compute_mean_matches_manual() -> None:
    """mean по src совпадает с ручным расчётом."""
    ds = _tiny_dataset()
    edge_index = ds.edge_index.numpy()
    out = compute_edge_node_features(
        edge_index, ds.edge_attr, num_nodes=3, stats=["mean"],
    )
    # sample 0: node0 mean=1.0; node1 mean=(1+3)/2=2.0; node2 mean=3.0
    assert out.shape == (2, 3, 1)
    np.testing.assert_allclose(out[0, :, 0], [1.0, 2.0, 3.0], rtol=1e-5)
    # sample 1: 2.0, (2+4)/2=3.0, 4.0
    np.testing.assert_allclose(out[1, :, 0], [2.0, 3.0, 4.0], rtol=1e-5)


def test_compute_unknown_stat_raises() -> None:
    """Неизвестная статистика → ValueError."""
    ds = _tiny_dataset()
    try:
        compute_edge_node_features(
            ds.edge_index.numpy(), ds.edge_attr, 3, stats=["median"],
        )
        assert False, "ожидался ValueError"
    except ValueError:
        pass


def _bare_dm(
        enabled: bool = False,
        stats: list[str] | None = None,
        include_raw: bool = True,
        edge_scaler: str = "none",
) -> CFMSRDataModule:
    """Минимальный CFMSRDataModule без чтения CSV."""
    data_cfg = OmegaConf.create(
        {
            "data_csv": "unused.csv",
            "train_valid_graphs_csv": "unused.csv",
            "target_col": "y",
            "batch_size": 2,
            "val_ratio": 0.5,
            "num_workers": 0,
            "data_row_shift": 0,
            "feature_selection": {"method": "none"},
            "edge_selection": {"method": "none"},
            "edge_node_features": {
                "enabled": enabled,
                "stats": stats or ["mean"],
                "include_raw": include_raw,
            },
            "edge_scaler": edge_scaler,
        }
    )
    training_cfg = OmegaConf.create({"seed": 0})
    dm = object.__new__(CFMSRDataModule)
    dm.data_cfg = data_cfg
    dm.training_cfg = training_cfg
    dm.seed = 0
    dm.num_nodes = 3
    dm.feature_cols = ["a", "b", "c"]
    dm._all_feature_cols = list(dm.feature_cols)
    dm._all_num_nodes = 3
    dm.edge_node_enabled = enabled
    dm.edge_node_stats = list(stats or ["mean"])
    dm.edge_node_include_raw = include_raw
    dm.edge_scaler_mode = edge_scaler
    dm.node_mean = None
    dm.node_std = None
    dm.extra_mean = None
    dm.extra_std = None
    dm.edge_mean = None
    dm.edge_std = None
    dm.kept_nodes = None
    dm.edge_mask = None
    dm.pos_weight = 1.0
    dm.group_col = None
    return dm


def test_disabled_keeps_input_dim_one() -> None:
    """enabled=false → node_input_dim == 1, x.shape[-1] == 1."""
    ds = _tiny_dataset()
    dm = _bare_dm(enabled=False)
    dm._raw_train_valid = ds
    dm._raw_test = None
    dm.prepare_split(
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([1], dtype=np.int64),
    )
    assert dm.node_input_dim == 1
    x = dm.train_valid_dataset.get(0).x
    assert x.shape[-1] == 1


def test_enabled_input_dim_and_get_shape() -> None:
    """stats=[mean, std], include_raw=true → node_input_dim == 3."""
    ds = _tiny_dataset()
    dm = _bare_dm(enabled=True, stats=["mean", "std"], include_raw=True)
    dm._raw_train_valid = ds
    dm._raw_test = None
    dm.prepare_split(
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([1], dtype=np.int64),
    )
    assert dm.node_input_dim == 3
    x = dm.train_valid_dataset.get(0).x
    assert x.shape == (3, 3)


def test_select_edges_resets_node_extra() -> None:
    """select_edges сбрасывает node_extra."""
    ds = _tiny_dataset()
    extra = compute_edge_node_features(
        ds.edge_index.numpy(), ds.edge_attr, 3, stats=["mean"],
    )
    ds = ds.replace(node_extra=extra)
    assert ds.node_extra is not None
    mask = np.array([True, True, False, False])
    out = ds.select_edges(mask)
    assert out.node_extra is None


def test_select_nodes_filters_node_extra() -> None:
    """select_nodes фильтрует node_extra по keep."""
    ds = _tiny_dataset()
    extra = compute_edge_node_features(
        ds.edge_index.numpy(), ds.edge_attr, 3, stats=["mean"],
    )
    ds = ds.replace(node_extra=extra)
    keep = np.array([0, 2], dtype=np.int64)
    out = ds.select_nodes(keep)
    assert out.node_extra is not None
    assert out.node_extra.shape == (2, 2, 1)


def test_extra_scaler_uses_train_only() -> None:
    """extra_mean/std считаются только по train."""
    ds = _tiny_dataset()
    dm = _bare_dm(enabled=True, stats=["mean"], include_raw=True)
    dm._raw_train_valid = ds
    dm._raw_test = None

    dm.prepare_split(
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([1], dtype=np.int64),
    )
    mean_a = dm.extra_mean.copy()

    # Меняем val-образец в сырых данных — статистики не должны измениться.
    ds2 = _tiny_dataset()
    ds2.edge_attr[:, 1] = 100.0
    dm._raw_train_valid = ds2
    dm.prepare_split(
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([1], dtype=np.int64),
    )
    np.testing.assert_allclose(dm.extra_mean, mean_a, rtol=1e-5)


def test_cfmsr_forward_input_dim_3() -> None:
    """Forward на батче с x формы (N, 3) → логиты (B, 1)."""
    cfg = OmegaConf.create(
        {
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
        }
    )
    model = CFMSR(
        training_cfg=cfg, num_nodes=3, input_dim=3, use_node_embedding=False,
    )
    model.eval()

    from torch_geometric.data import Data

    data = Data(
        x=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, 1),
        y=torch.tensor([1.0]),
    )
    batch = Batch.from_data_list([data, data])
    logits = model(batch)
    assert logits.shape == (2, 1)
