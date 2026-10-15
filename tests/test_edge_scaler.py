"""
Tests for edge scaler and edge_batchnorm (Ш2).
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from dataset import CFMSRDataModule, WeightedFullGraphDataset
from model import CFMSR


def _tiny_dataset() -> WeightedFullGraphDataset:
    node_features = np.array(
        [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=np.float32,
    )
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long,
    )
    edge_attr = np.array(
        [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]], dtype=np.float32,
    )
    labels = np.array([1.0, 0.0], dtype=np.float32)
    return WeightedFullGraphDataset(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        labels=labels,
    )


def _bare_dm(edge_scaler: str = "none") -> CFMSRDataModule:
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
                "enabled": False,
                "stats": ["mean"],
                "include_raw": True,
            },
            "edge_scaler": edge_scaler,
        }
    )
    dm = object.__new__(CFMSRDataModule)
    dm.data_cfg = data_cfg
    dm.training_cfg = OmegaConf.create({"seed": 0})
    dm.seed = 0
    dm.num_nodes = 3
    dm.feature_cols = ["a", "b", "c"]
    dm._all_feature_cols = list(dm.feature_cols)
    dm._all_num_nodes = 3
    dm.edge_node_enabled = False
    dm.edge_node_stats = ["mean"]
    dm.edge_node_include_raw = True
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


def test_edge_scaler_none_is_identity() -> None:
    """edge_scaler=none не меняет edge_attr."""
    ds = _tiny_dataset()
    original = ds.edge_attr.copy()
    dm = _bare_dm(edge_scaler="none")
    dm._raw_train_valid = ds
    dm._raw_test = None
    dm.prepare_split(
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([1], dtype=np.int64),
    )
    np.testing.assert_allclose(
        dm.train_valid_dataset.edge_attr, original, rtol=1e-5,
    )


def test_edge_scaler_global_train_only() -> None:
    """global-статистика не зависит от значений val-образцов."""
    ds = _tiny_dataset()
    dm = _bare_dm(edge_scaler="global")
    dm._raw_train_valid = ds
    dm._raw_test = None
    dm.prepare_split(
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([1], dtype=np.int64),
    )
    mean_a = float(dm.edge_mean)
    std_a = float(dm.edge_std)

    ds2 = _tiny_dataset()
    ds2.edge_attr[:, 1] = 999.0
    dm._raw_train_valid = ds2
    dm.prepare_split(
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([1], dtype=np.int64),
    )
    assert float(dm.edge_mean) == mean_a
    assert float(dm.edge_std) == std_a


def test_edge_scaler_per_edge_shapes() -> None:
    """per_edge даёт векторы длины E."""
    ds = _tiny_dataset()
    e = ds.edge_attr.shape[0]
    dm = _bare_dm(edge_scaler="per_edge")
    dm._raw_train_valid = ds
    dm._raw_test = None
    dm.prepare_split(
        train_idx=np.array([0, 1], dtype=np.int64),
        val_idx=np.array([], dtype=np.int64),
    )
    assert isinstance(dm.edge_mean, np.ndarray)
    assert dm.edge_mean.shape == (e,)
    assert dm.edge_std.shape == (e,)


def test_cfmsr_edge_batchnorm_false_has_no_batchnorm() -> None:
    """edge_batchnorm=false → у всех блоков edge_norm is None."""
    cfg = OmegaConf.create(
        {
            "hidden_dim": 16,
            "num_layers": 2,
            "num_heads": 2,
            "ffn_mult": 2.0,
            "dropout": 0.0,
            "num_seeds": 2,
            "classifier_dropout": 0.0,
            "use_node_embedding": False,
            "drop_edge": 0.0,
            "node_noise_std": 0.0,
            "edge_noise_std": 0.0,
            "edge_batchnorm": False,
            "edge_readout": "none",
        }
    )
    model = CFMSR(training_cfg=cfg, num_nodes=4, use_node_embedding=False)
    assert all(layer.attn.edge_norm is None for layer in model.layers)
