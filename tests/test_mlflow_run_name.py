"""
Tests for MLflow run name resolution.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from mlflow_utils import _resolve_run_name


def test_explicit_run_name_has_priority() -> None:
    """Явное mlflow.run_name перекрывает автоматическое имя."""
    cfg = OmegaConf.create({"mlflow": {"run_name": " best_fold_column "}})
    assert _resolve_run_name(cfg) == "best_fold_column"


def test_empty_run_name_falls_back_to_default() -> None:
    """Пустое или null-имя не используется; без Hydra имя — ``train``."""
    for value in (None, "", "   "):
        cfg = OmegaConf.create({"mlflow": {"run_name": value}})
        assert _resolve_run_name(cfg) == "train"


def test_hpo_without_hydra_returns_none() -> None:
    """В HPO без явного имени MLflow сам генерирует имя."""
    cfg = OmegaConf.create({"mlflow": {"run_name": None}, "hpo": {"n_trials": 1}})
    assert _resolve_run_name(cfg) is None
