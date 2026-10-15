"""
Tests for CV split helpers.
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from cv_splits import make_cv_splits


def test_stratified_splits_disjoint_and_both_classes() -> None:
    """Train/val не пересекаются; оба класса в каждом fold."""
    rng = np.random.default_rng(0)
    n = 100
    labels = np.array([0] * 50 + [1] * 50, dtype=np.float32)
    rng.shuffle(labels)

    cfg = OmegaConf.create(
        {"scheme": "stratified_kfold", "n_splits": 5, "n_repeats": 1}
    )
    splits = make_cv_splits(labels, folds=None, groups=None, cv_cfg=cfg, seed=0)
    assert len(splits) == 5

    for train_idx, val_idx in splits:
        assert len(set(train_idx) & set(val_idx)) == 0
        assert set(np.unique(labels[train_idx].astype(int))) == {0, 1}
        assert set(np.unique(labels[val_idx].astype(int))) == {0, 1}


def test_group_kfold_no_group_leakage() -> None:
    """Ни одна группа не попадает в обе части fold."""
    # В каждой группе оба класса — иначе StratifiedGroupKFold даёт одноклассовый val.
    labels_list = []
    groups_list = []
    for g in range(8):
        labels_list.extend([0, 1, 0, 1])
        groups_list.extend([g, g, g, g])
    labels = np.asarray(labels_list, dtype=np.float32)
    groups = np.asarray(groups_list, dtype=np.int64)

    cfg = OmegaConf.create(
        {"scheme": "group_kfold", "n_splits": 4, "n_repeats": 1}
    )
    splits = make_cv_splits(
        labels, folds=None, groups=groups, cv_cfg=cfg, seed=0,
    )
    for train_idx, val_idx in splits:
        g_tr = set(groups[train_idx].tolist())
        g_va = set(groups[val_idx].tolist())
        assert g_tr.isdisjoint(g_va)
        assert set(np.unique(labels[val_idx].astype(int))) == {0, 1}


def test_fold_column_scheme() -> None:
    """fold_column использует готовые номера fold."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    folds = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    cfg = OmegaConf.create({"scheme": "fold_column", "n_splits": 2, "n_repeats": 1})
    splits = make_cv_splits(labels, folds=folds, groups=None, cv_cfg=cfg, seed=0)
    assert len(splits) == 4
