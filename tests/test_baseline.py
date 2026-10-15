"""
Tests for baseline models.

Проверки на синтетике: шум → AUC≈0.5; сигнал → AUC>0.8; undirected mask.
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from baseline_models import (
    MeanScoreClassifier,
    _unique_undirected_mask,
    make_model,
    permutation_auc,
    repeated_cv_auc,
)


def test_unique_undirected_mask_keeps_half() -> None:
    """Оставляет ровно половину направленных рёбер."""
    # 3 undirected edges → 6 directed
    edge_index = np.array(
        [
            [0, 1, 0, 2, 1, 2],
            [1, 0, 2, 0, 2, 1],
        ],
        dtype=np.int64,
    )
    mask = _unique_undirected_mask(edge_index)
    assert mask.sum() == 3
    assert mask.dtype == bool


def test_random_features_auc_near_chance() -> None:
    """Случайные признаки дают AUC около 0.5 и p > 0.05."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 10))
    y = rng.integers(0, 2, size=n)
    # Гарантируем оба класса.
    y[0] = 0
    y[1] = 1

    cfg = OmegaConf.create(
        {
            "logreg_c_grid": [0.1, 1.0],
            "rf": {"n_estimators": 50, "max_depth": 3, "min_samples_leaf": 5},
            "hist_gb": {"max_depth": 2, "learning_rate": 0.1, "max_iter": 50},
        }
    )

    def factory():
        return make_model("logreg", cfg, seed=0)

    cv_aucs = repeated_cv_auc(factory, x, y, n_splits=5, n_repeats=2, seed=0)
    mean_auc = float(np.nanmean(cv_aucs))
    assert 0.35 <= mean_auc <= 0.65

    perm = permutation_auc(factory, x, y, n=30, seed=0)
    perm_valid = perm[np.isfinite(perm)]
    p_value = float(np.mean(perm_valid >= mean_auc))
    assert p_value > 0.05


def test_synthetic_signal_high_auc() -> None:
    """Синтетика с заложенным сигналом даёт AUC > 0.8."""
    rng = np.random.default_rng(42)
    n = 300
    signal = rng.normal(size=n)
    y = (signal > 0).astype(int)
    x = np.column_stack([signal + rng.normal(scale=0.3, size=n), rng.normal(size=(n, 5))])

    cfg = OmegaConf.create(
        {
            "logreg_c_grid": [0.01, 0.1, 1.0, 10.0],
        }
    )

    def factory():
        return make_model("logreg", cfg, seed=42)

    cv_aucs = repeated_cv_auc(factory, x, y, n_splits=5, n_repeats=2, seed=42)
    assert float(np.nanmean(cv_aucs)) > 0.8


def test_edge_mean_classifier_scores_are_row_means() -> None:
    """Score edge_mean = среднее признаков по строке."""
    x = np.array(
        [
            [0.2, 0.4, 0.6],
            [0.0, 1.0, 0.5],
        ],
        dtype=np.float64,
    )
    clf = MeanScoreClassifier()
    clf.fit(x, np.array([0, 1]))
    proba = clf.predict_proba(x)
    expected = x.mean(axis=1)
    np.testing.assert_allclose(proba[:, 1], expected, rtol=1e-6)
    assert not isinstance(make_model("edge_mean", OmegaConf.create({}), 0), type(None))
