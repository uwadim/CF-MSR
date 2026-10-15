"""
Tests for CV objective helpers.
"""

from __future__ import annotations

import math

import pytest
from omegaconf import OmegaConf

from crossval import (
    ObjectiveSpec,
    compute_objective,
    fold_objective_value,
    parse_objective_spec,
)

METRICS = ["accuracy", "roc_auc"]


def _spec(**kwargs: object) -> ObjectiveSpec:
    base = {"metric": "roc_auc", "scope": "fold", "kind": "mean", "std_penalty": 0.5}
    base.update(kwargs)
    return ObjectiveSpec(**base)  # type: ignore[arg-type]


def test_default_spec_is_legacy_monitor() -> None:
    """По умолчанию objective — score выбора эпохи."""
    spec = parse_objective_spec(OmegaConf.create({}), METRICS)
    assert spec.metric is None
    assert spec.scope == "fold"
    assert spec.kind == "mean"


def test_parse_metric_alias_and_null_strings() -> None:
    """Алиасы нормализуются, строки null/none означают отсутствие метрики."""
    cfg = OmegaConf.create({"objective_metric": "ROC-AUC"})
    assert parse_objective_spec(cfg, METRICS).metric == "roc_auc"
    cfg = OmegaConf.create({"objective_metric": "none"})
    assert parse_objective_spec(cfg, METRICS).metric is None


@pytest.mark.parametrize(
    "cv_cfg",
    [
        {"objective": "median"},
        {"objective_scope": "pool"},
        {"objective_metric": "f1"},  # нет в training.metrics
        {"objective_scope": "oof"},  # oof без метрики
        {"objective_metric": "loss", "objective_scope": "oof"},
    ],
)
def test_parse_invalid_raises(cv_cfg: dict) -> None:
    """Недопустимые настройки objective приводят к ValueError."""
    with pytest.raises(ValueError):
        parse_objective_spec(OmegaConf.create(cv_cfg), METRICS)


def test_fold_value_variants() -> None:
    """Метрика берётся из val_out, loss — с обратным знаком, None — fit_score."""
    val_out = {"loss": 0.8, "roc_auc": 0.62}
    assert fold_objective_value(val_out, fit_score=-0.9, metric="roc_auc") == 0.62
    assert fold_objective_value(val_out, fit_score=-0.9, metric="loss") == -0.8
    assert fold_objective_value(val_out, fit_score=-0.9, metric=None) == -0.9
    assert math.isnan(fold_objective_value({}, fit_score=0.0, metric="roc_auc"))


def test_compute_fold_mean_and_penalty() -> None:
    """scope=fold: среднее, для mean_minus_std вычитается std·penalty."""
    values = [0.6, 0.7]
    assert compute_objective(values, {}, _spec()) == pytest.approx(0.65)
    penalized = compute_objective(values, {}, _spec(kind="mean_minus_std"))
    assert penalized == pytest.approx(0.65 - 0.5 * 0.05)


def test_compute_oof_scope_uses_pooled_metric() -> None:
    """scope=oof: берётся метрика по пулу OOF, штраф — по разбросу fold."""
    value = compute_objective([0.6, 0.7], {"roc_auc": 0.66}, _spec(scope="oof"))
    assert value == pytest.approx(0.66)
    penalized = compute_objective(
        [0.6, 0.7],
        {"roc_auc": 0.66},
        _spec(scope="oof", kind="mean_minus_std"),
    )
    assert penalized == pytest.approx(0.66 - 0.5 * 0.05)


def test_compute_ignores_nan_folds() -> None:
    """NaN в отдельном fold не ломает среднее."""
    assert compute_objective([0.6, float("nan"), 0.8], {}, _spec()) == pytest.approx(0.7)
