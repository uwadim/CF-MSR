"""
Baseline models.

Sklearn-базлайны для проверки наличия сигнала в данных CF-MSR.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config_utils import resolve_path
from dataset import CFMSRDataModule, WeightedFullGraphDataset
from mlflow_utils import log_artifact_path, log_hparams, mlflow_run_context
from train import get_metric_names, get_metric_threshold, safe_metrics, set_seed

log = logging.getLogger(__name__)


class MeanScoreClassifier(BaseEstimator, ClassifierMixin):
    """
    Mean Score Classifier.

    Score = среднее признаков по строке; ``fit`` — no-op.
    """

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
    ) -> "MeanScoreClassifier":
        """
        Fit classifier.

        Parameters
        ----------
        x : np.ndarray
            Матрица признаков.
        y : np.ndarray
            Метки классов.

        Returns
        -------
        MeanScoreClassifier
            Self.
        """
        y_arr = np.asarray(y)
        classes = np.unique(y_arr[np.isfinite(y_arr)])
        self.classes_ = classes.astype(np.int64) if classes.size > 0 else np.array(
            [0, 1], dtype=np.int64,
        )
        self.n_features_in_ = int(np.asarray(x).shape[1]) if np.asarray(x).ndim == 2 else 0
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.

        Parameters
        ----------
        x : np.ndarray
            Матрица признаков ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Вероятности формы ``(n_samples, 2)``.
        """
        scores = np.asarray(x, dtype=np.float64).mean(axis=1)
        scores = np.clip(scores, 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict labels.

        Parameters
        ----------
        x : np.ndarray
            Матрица признаков.

        Returns
        -------
        np.ndarray
            Метки ``0/1``.
        """
        proba = self.predict_proba(x)
        return (proba[:, 1] >= 0.5).astype(np.int64)


def _unique_undirected_mask(edge_index: np.ndarray) -> np.ndarray:
    """
    Unique undirected mask.

    Оставляет одно направление для каждой неориентированной пары ``u < v``.

    Parameters
    ----------
    edge_index : np.ndarray
        Индексы рёбер формы ``(2, E)``.

    Returns
    -------
    np.ndarray
        Булева маска длины ``E``.
    """
    src = edge_index[0]
    dst = edge_index[1]
    return src < dst


def build_feature_matrix(
    dm: CFMSRDataModule,
    dataset: WeightedFullGraphDataset,
    feature_set: str,
) -> np.ndarray:
    """
    Build feature matrix.

    Собирает матрицу признаков из узлов и/или рёбер датасета.

    Parameters
    ----------
    dm : CFMSRDataModule
        Модуль данных (для совместимости API; сейчас не используется).
    dataset : WeightedFullGraphDataset
        Графовый датасет.
    feature_set : str
        ``nodes`` | ``edges`` | ``nodes_edges`` | ``edge_nodes``.

    Returns
    -------
    np.ndarray
        Матрица формы ``(n_samples, n_features)``.
    """
    del dm  # API-совместимость с планом
    kind = str(feature_set).strip().lower()
    node_x = np.asarray(dataset.node_features, dtype=np.float32)

    if kind == "nodes":
        return node_x

    if kind == "edge_nodes":
        if dataset.node_extra is None:
            raise ValueError(
                "feature_set=edge_nodes требует dataset.node_extra; "
                "включите data.edge_node_features.enabled=true"
            )
        extra = np.asarray(dataset.node_extra, dtype=np.float32)
        return extra.reshape(extra.shape[0], -1)

    edge_index = dataset.edge_index.detach().cpu().numpy()
    mask = _unique_undirected_mask(edge_index)
    # edge_attr: (E, n_samples) → (n_samples, E_undirected)
    edge_x = np.asarray(dataset.edge_attr[mask], dtype=np.float32).T

    if kind == "edges":
        return edge_x
    if kind == "nodes_edges":
        return np.concatenate([node_x, edge_x], axis=1)

    raise ValueError(
        f"Неизвестный feature_set={feature_set!r}; "
        "ожидается nodes | edges | nodes_edges | edge_nodes"
    )


def make_model(
    name: str,
    baseline_cfg: DictConfig,
    seed: int,
) -> Any:
    """
    Make sklearn model.

    Возвращает Pipeline(StandardScaler, estimator).

    Parameters
    ----------
    name : str
        Имя модели: ``dummy``, ``edge_mean``, ``logreg``, ``random_forest``,
        ``hist_gb``.
    baseline_cfg : DictConfig
        Ветка ``baseline`` конфигурации.
    seed : int
        Random seed.

    Returns
    -------
    Any
        Sklearn estimator / Pipeline.
    """
    key = str(name).strip().lower()

    if key == "edge_mean":
        # Без StandardScaler: score = среднее исходных весов.
        return MeanScoreClassifier()

    if key == "dummy":
        clf: Any = DummyClassifier(strategy="prior")
    elif key == "logreg":
        c_grid = list(baseline_cfg.get("logreg_c_grid", [0.001, 0.01, 0.1, 1.0, 10.0]))
        clf = LogisticRegressionCV(
            Cs=c_grid,
            cv=5,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=int(seed),
        )
    elif key in ("random_forest", "rf"):
        rf = baseline_cfg.get("rf", {})
        clf = RandomForestClassifier(
            n_estimators=int(rf.get("n_estimators", 500)),
            max_depth=int(rf.get("max_depth", 4)),
            min_samples_leaf=int(rf.get("min_samples_leaf", 5)),
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=-1,
        )
    elif key in ("hist_gb", "histgradientboosting", "hgb"):
        hgb = baseline_cfg.get("hist_gb", {})
        clf = HistGradientBoostingClassifier(
            max_depth=int(hgb.get("max_depth", 3)),
            learning_rate=float(hgb.get("learning_rate", 0.05)),
            max_iter=int(hgb.get("max_iter", 100)),
            random_state=int(seed),
        )
    else:
        raise ValueError(f"Неизвестная baseline-модель: {name!r}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def _predict_scores(model: Any, x: np.ndarray) -> np.ndarray:
    """Возвращает вероятности положительного класса или decision scores."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=np.float64)
        return np.asarray(proba[:, 0], dtype=np.float64)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=np.float64)
    return np.asarray(model.predict(x), dtype=np.float64)


def evaluate_on_split(
    model: Any,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ev: np.ndarray,
    y_ev: np.ndarray,
    metric_names: List[str],
    threshold: float,
) -> Dict[str, float]:
    """
    Evaluate on split.

    Обучает модель на train и считает метрики на evaluation-сплите.

    Parameters
    ----------
    model : Any
        Sklearn estimator.
    x_tr : np.ndarray
        Признаки train.
    y_tr : np.ndarray
        Метки train.
    x_ev : np.ndarray
        Признаки evaluation.
    y_ev : np.ndarray
        Метки evaluation.
    metric_names : List[str]
        Список метрик.
    threshold : float
        Порог бинарных предсказаний.

    Returns
    -------
    Dict[str, float]
        Словарь метрик.
    """
    model.fit(x_tr, y_tr)
    scores = _predict_scores(model, x_ev)
    return safe_metrics(
        y_true=y_ev.astype(int).tolist(),
        y_score=scores.tolist(),
        metric_names=metric_names,
        threshold=threshold,
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int,
    seed: int,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for AUC.

    Parameters
    ----------
    y_true : np.ndarray
        Истинные метки.
    y_score : np.ndarray
        Скоры.
    n : int
        Число bootstrap-реплик.
    seed : int
        Random seed.

    Returns
    -------
    Tuple[float, float, float]
        ``(auc, ci_low, ci_high)`` — точечная оценка и 95% CI.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    valid = (y_true >= 0) & (y_true <= 1) & np.isfinite(y_score)
    y_true = y_true[valid].astype(int)
    y_score = y_score[valid]

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")

    point = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(int(seed))
    aucs: List[float] = []
    n_samples = len(y_true)

    for _ in range(int(n)):
        idx = rng.integers(0, n_samples, size=n_samples)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(float(roc_auc_score(yt, y_score[idx])))

    if not aucs:
        return point, float("nan"), float("nan")

    low, high = np.percentile(aucs, [2.5, 97.5])
    return point, float(low), float(high)


def permutation_auc(
    model_factory: Callable[[], Any],
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    """
    Permutation AUC.

    Считает AUC при перемешанных метках (потолок шума / p-value).

    Parameters
    ----------
    model_factory : Callable[[], Any]
        Фабрика моделей без аргументов.
    x : np.ndarray
        Признаки.
    y : np.ndarray
        Метки.
    n : int
        Число перестановок.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Массив AUC длины ``n`` (с возможными NaN).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(int(seed))
    y = np.asarray(y).astype(int)
    out = np.full(int(n), np.nan, dtype=np.float64)

    for i in range(int(n)):
        y_perm = rng.permutation(y)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed) + i)
        fold_aucs: List[float] = []
        try:
            for tr, te in skf.split(x, y_perm):
                model = model_factory()
                model.fit(x[tr], y_perm[tr])
                scores = _predict_scores(model, x[te])
                if len(np.unique(y_perm[te])) < 2:
                    continue
                fold_aucs.append(float(roc_auc_score(y_perm[te], scores)))
        except Exception:
            continue
        if fold_aucs:
            out[i] = float(np.mean(fold_aucs))

    return out


def repeated_cv_auc(
    model_factory: Callable[[], Any],
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> np.ndarray:
    """
    Repeated stratified CV AUC.

    Parameters
    ----------
    model_factory : Callable[[], Any]
        Фабрика моделей.
    x : np.ndarray
        Признаки.
    y : np.ndarray
        Метки.
    n_splits : int
        Число fold.
    n_repeats : int
        Число повторов.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        AUC по каждому fold×repeat.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    rskf = RepeatedStratifiedKFold(
        n_splits=int(n_splits),
        n_repeats=int(n_repeats),
        random_state=int(seed),
    )
    aucs: List[float] = []

    for tr, te in rskf.split(x, y):
        model = model_factory()
        model.fit(x[tr], y[tr])
        scores = _predict_scores(model, x[te])
        if len(np.unique(y[te])) < 2:
            aucs.append(float("nan"))
            continue
        aucs.append(float(roc_auc_score(y[te], scores)))

    return np.asarray(aucs, dtype=np.float64)


def _valid_label_mask(labels: np.ndarray) -> np.ndarray:
    return (labels >= 0.0) & (labels <= 1.0) & np.isfinite(labels)


def run_baselines(cfg: DictConfig) -> pd.DataFrame:
    """
    Run baselines.

    Строит CFMSRDataModule, обучает sklearn-модели и пишет таблицу результатов.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.

    Returns
    -------
    pd.DataFrame
        Таблица метрик по моделям.
    """
    log.info("Baseline config:\n%s", OmegaConf.to_yaml(cfg))
    baseline_cfg = cfg.baseline
    training_cfg = cfg.training
    seed = int(training_cfg.seed)
    set_seed(seed)

    metric_names = get_metric_names(training_cfg)
    threshold = get_metric_threshold(training_cfg)
    feature_set = str(baseline_cfg.get("feature_set", "nodes"))
    models = list(baseline_cfg.get("models", ["dummy", "logreg"]))

    with mlflow_run_context(cfg):
        log_hparams(cfg)

        dm = CFMSRDataModule(data_cfg=cfg.data, training_cfg=training_cfg)
        dm.setup()

        assert dm.train_valid_dataset is not None
        assert dm.train_idx is not None
        assert dm.val_idx is not None

        x_all = build_feature_matrix(dm, dm.train_valid_dataset, feature_set)
        y_all = np.asarray(dm.train_valid_dataset.labels, dtype=np.float32)

        train_idx = dm.train_idx
        val_idx = dm.val_idx

        tr_mask = _valid_label_mask(y_all[train_idx])
        va_mask = _valid_label_mask(y_all[val_idx])
        x_tr = x_all[train_idx][tr_mask]
        y_tr = y_all[train_idx][tr_mask].astype(int)
        x_va = x_all[val_idx][va_mask]
        y_va = y_all[val_idx][va_mask].astype(int)

        # Объединение train+val для CV / permutation.
        tv_idx = np.concatenate([train_idx, val_idx])
        tv_mask = _valid_label_mask(y_all[tv_idx])
        x_tv = x_all[tv_idx][tv_mask]
        y_tv = y_all[tv_idx][tv_mask].astype(int)

        x_te: Optional[np.ndarray] = None
        y_te: Optional[np.ndarray] = None
        if dm.test_dataset is not None:
            x_te = build_feature_matrix(dm, dm.test_dataset, feature_set)
            y_te_raw = np.asarray(dm.test_dataset.labels, dtype=np.float32)
            te_mask = _valid_label_mask(y_te_raw)
            x_te = x_te[te_mask]
            y_te = y_te_raw[te_mask].astype(int)

        cv_cfg = baseline_cfg.get("cv", {})
        n_splits = int(cv_cfg.get("n_splits", 5))
        n_repeats = int(cv_cfg.get("n_repeats", 5))
        bootstrap_n = int(baseline_cfg.get("bootstrap_n", 1000))
        permutation_n = int(baseline_cfg.get("permutation_n", 200))

        rows: List[Dict[str, Any]] = []

        for model_name in models:
            log.info("Baseline model: %s", model_name)

            def factory(name: str = model_name) -> Any:
                return make_model(name, baseline_cfg, seed)

            model = factory()
            val_metrics = evaluate_on_split(
                model, x_tr, y_tr, x_va, y_va, metric_names, threshold,
            )

            test_metrics: Dict[str, float] = {}
            test_scores: Optional[np.ndarray] = None
            if x_te is not None and y_te is not None and len(y_te) > 0:
                model_te = factory()
                test_metrics = evaluate_on_split(
                    model_te, x_tr, y_tr, x_te, y_te, metric_names, threshold,
                )
                test_scores = _predict_scores(model_te, x_te)

            cv_aucs = repeated_cv_auc(
                factory, x_tv, y_tv, n_splits, n_repeats, seed,
            )
            cv_mean = float(np.nanmean(cv_aucs))
            cv_std = float(np.nanstd(cv_aucs))

            # Bootstrap CI по holdout, иначе по val.
            if test_scores is not None and y_te is not None and len(y_te) > 0:
                auc_point, ci_low, ci_high = bootstrap_ci(
                    y_te, test_scores, bootstrap_n, seed,
                )
            else:
                model_va = factory()
                model_va.fit(x_tr, y_tr)
                va_scores = _predict_scores(model_va, x_va)
                auc_point, ci_low, ci_high = bootstrap_ci(
                    y_va, va_scores, bootstrap_n, seed,
                )

            perm = permutation_auc(factory, x_tv, y_tv, permutation_n, seed)
            observed = cv_mean
            perm_valid = perm[np.isfinite(perm)]
            if len(perm_valid) > 0 and np.isfinite(observed):
                p_value = float(np.mean(perm_valid >= observed))
            else:
                p_value = float("nan")

            row: Dict[str, Any] = {
                "model": model_name,
                "feature_set": feature_set,
                "val_roc_auc": val_metrics.get("roc_auc", float("nan")),
                "test_roc_auc": test_metrics.get("roc_auc", float("nan")),
                "cv_auc_mean": cv_mean,
                "cv_auc_std": cv_std,
                "auc_point": auc_point,
                "auc_ci_low": ci_low,
                "auc_ci_high": ci_high,
                "permutation_p": p_value,
            }
            for name in metric_names:
                row[f"val_{name}"] = val_metrics.get(name, float("nan"))
                row[f"test_{name}"] = test_metrics.get(name, float("nan"))

            rows.append(row)
            log.info(
                "%s | val_auc=%.4f test_auc=%.4f cv=%.4f±%.4f "
                "CI=[%.4f, %.4f] p=%.4f",
                model_name,
                row["val_roc_auc"],
                row["test_roc_auc"],
                cv_mean,
                cv_std,
                ci_low,
                ci_high,
                p_value,
            )

        results = pd.DataFrame(rows)
        out_path = resolve_path(
            str(baseline_cfg.get("results_out", "baseline_results.csv"))
        )
        if out_path is None:
            out_path = "baseline_results.csv"
        results.to_csv(out_path, index=False)
        log.info("Baseline results saved to %s", out_path)
        log_artifact_path(cfg, out_path)

        if _mlflow_active():
            import mlflow

            for _, row in results.iterrows():
                prefix = f"baseline_{row['model']}_"
                for col in (
                    "val_roc_auc",
                    "test_roc_auc",
                    "cv_auc_mean",
                    "cv_auc_std",
                    "auc_ci_low",
                    "auc_ci_high",
                    "permutation_p",
                ):
                    try:
                        val = float(row[col])
                    except Exception:
                        continue
                    if np.isfinite(val):
                        mlflow.log_metric(f"{prefix}{col}", val)

        return results


def _mlflow_active() -> bool:
    try:
        import mlflow

        return mlflow.active_run() is not None
    except Exception:
        return False
