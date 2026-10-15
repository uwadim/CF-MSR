"""
Cross-validation training.

Кросс-валидация CF-MSR с OOF-предсказаниями и ансамблем holdout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from config_utils import resolve_path
from cv_splits import make_cv_splits
from dataset import CFMSRDataModule
from mlflow_utils import (
    log_artifact_path,
    log_hparams,
    log_split_metrics,
    mlflow_run_context,
)
from model_factory import build_model
from train import (
    build_criteria,
    evaluate,
    fit_model,
    format_metrics,
    get_metric_names,
    get_metric_threshold,
    get_score_metric,
    log_metrics_block,
    normalize_metric_name,
    safe_metrics,
    set_seed,
    _is_mlflow_active,
)

log = logging.getLogger(__name__)

OBJECTIVE_KINDS = ("mean", "mean_minus_std")
OBJECTIVE_SCOPES = ("fold", "oof")


@dataclass(frozen=True)
class ObjectiveSpec:
    """
    Objective specification.

    Описание целевой функции CV для Optuna.

    Attributes
    ----------
    metric : Optional[str]
        Имя метрики (``roc_auc``, ``balanced_accuracy``, ..., ``loss``).
        ``None`` — прежнее поведение: score выбора эпохи (``monitor_metric``).
    scope : str
        ``fold`` — среднее метрики по fold/seed, ``oof`` — метрика по пулу OOF.
    kind : str
        ``mean`` или ``mean_minus_std``.
    std_penalty : float
        Коэффициент штрафа за разброс по fold/seed.
    """

    metric: Optional[str]
    scope: str
    kind: str
    std_penalty: float


def parse_objective_spec(
    cv_cfg: Mapping[str, Any],
    metric_names: Sequence[str],
) -> ObjectiveSpec:
    """
    Parse objective spec.

    Читает и валидирует настройки целевой функции из ветки ``cv``.

    Parameters
    ----------
    cv_cfg : Mapping[str, Any]
        Ветка ``cv`` конфигурации.
    metric_names : Sequence[str]
        Метрики, вычисляемые на валидации (``training.metrics``).

    Returns
    -------
    ObjectiveSpec
        Проверенная спецификация целевой функции.

    Raises
    ------
    ValueError
        Если ``objective``, ``objective_scope`` или ``objective_metric``
        имеют недопустимое значение.
    """
    kind = str(cv_cfg.get("objective", "mean")).strip().lower()
    if kind not in OBJECTIVE_KINDS:
        raise ValueError(
            f"cv.objective={kind!r} не поддерживается, допустимо: {OBJECTIVE_KINDS}"
        )

    scope = str(cv_cfg.get("objective_scope", "fold")).strip().lower()
    if scope not in OBJECTIVE_SCOPES:
        raise ValueError(
            f"cv.objective_scope={scope!r} не поддерживается, "
            f"допустимо: {OBJECTIVE_SCOPES}"
        )

    raw_metric = cv_cfg.get("objective_metric", None)
    metric: Optional[str] = None
    if raw_metric is not None and str(raw_metric).strip().lower() not in {"", "none", "null"}:
        metric = normalize_metric_name(raw_metric)

    if metric is None:
        if scope == "oof":
            raise ValueError("cv.objective_scope=oof требует cv.objective_metric")
    elif metric == "loss":
        if scope == "oof":
            raise ValueError("cv.objective_metric=loss недоступен для scope=oof")
    elif metric not in metric_names:
        raise ValueError(
            f"cv.objective_metric={metric!r} не входит в training.metrics: "
            f"{list(metric_names)}"
        )

    return ObjectiveSpec(
        metric=metric,
        scope=scope,
        kind=kind,
        std_penalty=float(cv_cfg.get("std_penalty", 0.5)),
    )


def fold_objective_value(
    val_out: Mapping[str, Any],
    fit_score: float,
    metric: Optional[str],
) -> float:
    """
    Fold objective value.

    Значение целевой метрики для одного fold/seed (больше — лучше).

    Parameters
    ----------
    val_out : Mapping[str, Any]
        Результат ``evaluate`` на валидационной части fold лучшей моделью.
    fit_score : float
        Score выбора эпохи из ``fit_model`` (``FitResult.best_score``).
    metric : Optional[str]
        Имя метрики; ``None`` — использовать ``fit_score``.

    Returns
    -------
    float
        Значение метрики; для ``loss`` берётся с обратным знаком.
    """
    if metric is None:
        return float(fit_score)
    if metric == "loss":
        return -float(val_out["loss"])
    return float(val_out.get(metric, float("nan")))


def compute_objective(
    fold_values: Sequence[float],
    oof_metrics: Mapping[str, Any],
    spec: ObjectiveSpec,
) -> float:
    """
    Compute objective.

    Итоговое значение целевой функции CV для Optuna (maximize).

    Parameters
    ----------
    fold_values : Sequence[float]
        Значения метрики по fold/seed (см. ``fold_objective_value``).
    oof_metrics : Mapping[str, Any]
        Метрики по пулу OOF-предсказаний.
    spec : ObjectiveSpec
        Спецификация целевой функции.

    Returns
    -------
    float
        ``mean`` (или OOF-метрика) либо ``mean − std_penalty·std``.
    """
    values = np.asarray(fold_values, dtype=np.float64)
    spread = float(np.nanstd(values))

    if spec.scope == "oof":
        assert spec.metric is not None
        base = float(oof_metrics.get(spec.metric, float("nan")))
    else:
        base = float(np.nanmean(values))

    if spec.kind == "mean_minus_std":
        return base - spec.std_penalty * spread
    return base


def run_cv_training(cfg: DictConfig) -> float:
    """
    Run CV training.

    Цикл по fold/seed: ``prepare_split`` → модель → ``fit_model`` → оценка.
    Собирает OOF и ансамбль holdout.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.

    Returns
    -------
    float
        Objective для Optuna (maximize), см. ``parse_objective_spec``:
        ``cv.objective_metric`` (по умолчанию score выбора эпохи),
        ``cv.objective_scope`` (fold | oof), ``cv.objective`` (mean | mean_minus_std).
    """
    data_cfg = cfg.data
    training_cfg = cfg.training
    cv_cfg = cfg.cv

    if data_cfg.get("val_fold", None) is not None:
        raise ValueError(
            "Нельзя одновременно использовать data.val_fold и cv.enabled=true"
        )

    log_hparams(cfg)

    seed = int(training_cfg.seed)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_names = get_metric_names(training_cfg)
    metric_threshold = get_metric_threshold(training_cfg)
    score_metric = get_score_metric(training_cfg, metric_names)
    # Fail-fast: ошибки в настройке objective выявляем до обучения.
    objective_spec = parse_objective_spec(cv_cfg, metric_names)
    log.info(
        "CV objective: metric=%s scope=%s kind=%s",
        objective_spec.metric or f"monitor({score_metric})",
        objective_spec.scope,
        objective_spec.kind,
    )

    dm = CFMSRDataModule(data_cfg=data_cfg, training_cfg=training_cfg)
    dm.setup()
    assert dm._raw_train_valid is not None
    raw = dm._raw_train_valid

    splits = make_cv_splits(
        labels=raw.labels,
        folds=raw.folds,
        groups=getattr(raw, "groups", None),
        cv_cfg=cv_cfg,
        seed=seed,
    )

    ensemble_seeds = int(cv_cfg.get("ensemble_seeds", 1))
    save_models = bool(cv_cfg.get("save_models", False))
    evaluate_holdout = bool(cv_cfg.get("evaluate_holdout", True))
    n_samples = len(raw)
    oof_prob = np.full(n_samples, np.nan, dtype=np.float64)
    oof_counts = np.zeros(n_samples, dtype=np.float64)

    fold_scores: List[float] = []
    objective_values: List[float] = []
    holdout_prob_sum: Optional[np.ndarray] = None
    holdout_n = 0
    model_count = 0

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        for seed_i in range(ensemble_seeds):
            run_seed = seed + fold_i * 1009 + seed_i * 17
            set_seed(run_seed)

            nested_name = f"fold_{fold_i}_seed_{seed_i}"
            with mlflow_run_context(cfg, run_name=nested_name, nested=True):
                log.info(
                    "CV fold=%d seed_i=%d train=%d val=%d",
                    fold_i,
                    seed_i,
                    len(train_idx),
                    len(val_idx),
                )

                dm.prepare_split(train_idx, val_idx)
                train_loader = dm.train_loader()
                val_loader = dm.val_loader()
                test_loader = dm.test_loader()

                model = build_model(
                    training_cfg=training_cfg,
                    num_nodes=dm.num_nodes,
                    input_dim=dm.node_input_dim,
                ).to(device)

                pos_weight = torch.tensor(
                    [dm.pos_weight],
                    dtype=torch.float,
                    device=device,
                )
                label_smoothing = float(
                    training_cfg.get("label_smoothing", 0.0) or 0.0
                )
                train_crit, eval_crit = build_criteria(
                    pos_weight=pos_weight,
                    label_smoothing=label_smoothing,
                )

                fit_result = fit_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=train_crit,
                    training_cfg=training_cfg,
                    device=device,
                    metric_names=metric_names,
                    metric_threshold=metric_threshold,
                    eval_criterion=eval_crit,
                )
                fold_scores.append(float(fit_result.best_score))

                if _is_mlflow_active():
                    import mlflow

                    mlflow.log_metric("best_score", float(fit_result.best_score))
                    mlflow.log_metric("best_epoch", float(fit_result.best_epoch))

                # OOF predictions on val fold.
                assert val_loader is not None
                val_out = evaluate(
                    model=model,
                    loader=val_loader,
                    criterion=eval_crit,
                    device=device,
                    metric_names=metric_names,
                    metric_threshold=metric_threshold,
                )
                log_metrics_block(
                    f"cv_fold_{fold_i}_val",
                    val_out,
                    metric_names,
                )
                log_split_metrics(
                    f"cv_fold_{fold_i}_val",
                    val_out,
                    metric_names,
                )
                objective_values.append(
                    fold_objective_value(
                        val_out=val_out,
                        fit_score=fit_result.best_score,
                        metric=objective_spec.metric,
                    )
                )

                pred_df = val_out.get("predictions")
                if isinstance(pred_df, pd.DataFrame):
                    # PyG index_select вызывает get(original_idx), поэтому
                    # sample_idx уже глобальный индекс в raw-датасете.
                    probs = pred_df["prob"].to_numpy(dtype=np.float64)
                    global_idx = pred_df["sample_idx"].to_numpy(dtype=np.int64)
                    for g_i, prob in zip(global_idx, probs):
                        g_i = int(g_i)
                        if 0 <= g_i < n_samples:
                            if np.isnan(oof_prob[g_i]):
                                oof_prob[g_i] = prob
                                oof_counts[g_i] = 1.0
                            else:
                                oof_prob[g_i] += prob
                                oof_counts[g_i] += 1.0

                if evaluate_holdout and test_loader is not None:
                    test_out = evaluate(
                        model=model,
                        loader=test_loader,
                        criterion=eval_crit,
                        device=device,
                        metric_names=metric_names,
                        metric_threshold=metric_threshold,
                    )
                    tpred = test_out.get("predictions")
                    if isinstance(tpred, pd.DataFrame):
                        probs = tpred["prob"].to_numpy(dtype=np.float64)
                        if holdout_prob_sum is None:
                            holdout_prob_sum = np.zeros_like(probs)
                        holdout_prob_sum = holdout_prob_sum + probs
                        holdout_n += 1

                if save_models:
                    path = resolve_path(f"cv_fold{fold_i}_seed{seed_i}.pt")
                    if path is None:
                        path = f"cv_fold{fold_i}_seed{seed_i}.pt"
                    torch.save(model.state_dict(), path)
                    log_artifact_path(cfg, path)

                model_count += 1

    # Усреднение OOF по числу моделей на объект.
    mask = oof_counts > 0
    oof_prob[mask] = oof_prob[mask] / oof_counts[mask]

    y_true = raw.labels
    oof_metrics = safe_metrics(
        y_true=y_true.tolist(),
        y_score=oof_prob.tolist(),
        metric_names=metric_names,
        threshold=metric_threshold,
    )
    log.info("OOF metrics: %s", format_metrics(oof_metrics, metric_names))
    log_split_metrics("oof", oof_metrics, metric_names)

    oof_df = pd.DataFrame(
        {
            "sample_idx": np.arange(n_samples),
            "sample_id": raw.sample_ids,
            "label": y_true,
            "oof_prob": oof_prob,
        }
    )
    oof_out = resolve_path(str(cv_cfg.get("oof_out", "cv_oof_predictions.csv")))
    if oof_out is None:
        oof_out = "cv_oof_predictions.csv"
    oof_df.to_csv(oof_out, index=False)
    log_artifact_path(cfg, oof_out)

    if holdout_prob_sum is not None and holdout_n > 0 and dm.test_dataset is not None:
        holdout_prob = holdout_prob_sum / float(holdout_n)
        y_te = dm.test_dataset.labels
        holdout_metrics = safe_metrics(
            y_true=y_te.tolist(),
            y_score=holdout_prob.tolist(),
            metric_names=metric_names,
            threshold=metric_threshold,
        )
        log.info(
            "Holdout ensemble metrics: %s",
            format_metrics(holdout_metrics, metric_names),
        )
        log_split_metrics("ensemble_test", holdout_metrics, metric_names)

        pred_path = resolve_path(
            training_cfg.get("predictions_out", "holdout_predictions.csv")
        )
        if pred_path is None:
            pred_path = "holdout_predictions.csv"
        pd.DataFrame(
            {
                "sample_idx": np.arange(len(holdout_prob)),
                "sample_name": dm.test_dataset.sample_ids,
                "prob": holdout_prob,
                "pred": (holdout_prob >= metric_threshold).astype(int),
                "label": y_te,
            }
        ).to_csv(pred_path, index=False)
        log_artifact_path(cfg, pred_path)

    scores = np.asarray(fold_scores, dtype=np.float64)
    mean_score = float(np.nanmean(scores))
    std_score = float(np.nanstd(scores))
    log.info(
        "CV scores: mean=%.4f std=%.4f n=%d",
        mean_score,
        std_score,
        len(scores),
    )

    objective_value = compute_objective(
        fold_values=objective_values,
        oof_metrics=oof_metrics,
        spec=objective_spec,
    )
    objective_arr = np.asarray(objective_values, dtype=np.float64)
    log.info(
        "CV objective: value=%.4f fold_mean=%.4f fold_std=%.4f",
        objective_value,
        float(np.nanmean(objective_arr)),
        float(np.nanstd(objective_arr)),
    )

    if _is_mlflow_active():
        import mlflow

        # cv_mean/std_score — по score выбора эпохи (monitor_metric), для
        # совместимости; cv_objective* — по целевой метрике Optuna.
        mlflow.log_metric("cv_mean_score", mean_score)
        mlflow.log_metric("cv_std_score", std_score)
        mlflow.log_metric("cv_objective", objective_value)
        mlflow.log_metric("cv_objective_mean", float(np.nanmean(objective_arr)))
        mlflow.log_metric("cv_objective_std", float(np.nanstd(objective_arr)))
        if "roc_auc" in oof_metrics and np.isfinite(oof_metrics["roc_auc"]):
            mlflow.log_metric("oof_roc_auc", float(oof_metrics["roc_auc"]))

    return float(objective_value)
