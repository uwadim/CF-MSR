"""Интеграция MLflow для обычного обучения и HPO."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def get_mlflow_cfg(cfg: DictConfig) -> Optional[DictConfig]:
    """
    Возвращает ветку ``mlflow`` из Hydra-конфига.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.

    Returns
    -------
    Optional[DictConfig]
        Ветка ``mlflow`` или ``None``.
    """
    if "mlflow" not in cfg:
        return None
    return cfg.mlflow


def _is_mlflow_enabled(cfg: DictConfig) -> bool:
    mlflow_cfg = get_mlflow_cfg(cfg)
    if mlflow_cfg is None:
        return False
    return bool(mlflow_cfg.get("enabled", False))


def _project_root() -> Path:
    """Корневая директория запуска (до hydra.job.chdir)."""
    try:
        from hydra.utils import get_original_cwd

        return Path(get_original_cwd()).resolve()
    except Exception:
        return Path.cwd().resolve()


def resolve_sqlite_uri(uri: str) -> str:
    """
    Преобразует относительный ``sqlite:///...`` в абсолютный URI.

    Parameters
    ----------
    uri : str
        Исходный tracking/storage URI.

    Returns
    -------
    str
        URI с абсолютным путём к файлу БД.
    """
    if not uri.startswith("sqlite:///"):
        return uri

    db_path = uri[len("sqlite:///"):]
    path = Path(db_path).expanduser()
    if not path.is_absolute():
        path = _project_root() / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _ensure_parent_dir_sqlite(uri: str) -> None:
    resolve_sqlite_uri(uri)


def _resolve_run_name(cfg: DictConfig) -> Optional[str]:
    """
    Имя родительского run.

    Приоритет: явное ``mlflow.run_name`` → ``trial_N`` для HPO → ``train_<job>``.
    ``mlflow.run_name`` не предназначено для HPO: имя будет одинаковым у всех trials.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.

    Returns
    -------
    Optional[str]
        Имя run или ``None`` (MLflow сгенерирует имя сам).
    """
    mlflow_cfg = get_mlflow_cfg(cfg)
    if mlflow_cfg is not None:
        explicit = mlflow_cfg.get("run_name", None)
        if explicit is not None and str(explicit).strip() != "":
            return str(explicit).strip()

    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        if "hpo" in cfg:
            return f"trial_{hydra_cfg.job.num}"
        return f"train_{hydra_cfg.job.name}"
    except Exception:
        if "hpo" in cfg:
            return None
        return "train"


@contextmanager
def mlflow_run_context(
    cfg: DictConfig,
    run_name: Optional[str] = None,
    nested: bool = False,
) -> Iterator[Optional[Any]]:
    """
    Открывает MLflow run, если интеграция включена в ветке ``mlflow``.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.
    run_name : Optional[str], optional
        Явное имя run; если None — из Hydra/HPO.
    nested : bool, optional
        Если True, создаёт вложенный run (для CV fold).

    Yields
    ------
    Optional[Any]
        Активный MLflow run или ``None``.
    """
    if not _is_mlflow_enabled(cfg):
        yield None
        return

    import mlflow

    mlflow_cfg = get_mlflow_cfg(cfg)
    assert mlflow_cfg is not None

    tracking_uri = resolve_sqlite_uri(
        str(mlflow_cfg.get("tracking_uri", "sqlite:///artifacts/mlflow.db"))
    )
    experiment_name = str(mlflow_cfg.get("experiment_name", "cf_msr"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    resolved_name = run_name if run_name is not None else _resolve_run_name(cfg)

    with mlflow.start_run(run_name=resolved_name, nested=nested) as run:
        log.info(
            "MLflow run started: experiment=%s, run_id=%s, nested=%s",
            experiment_name,
            run.info.run_id,
            nested,
        )
        yield run


def _flatten_branch(
    branch: Any,
    prefix: str,
    flat: Dict[str, Any],
) -> None:
    """Рекурсивно уплощает вложенный dict в ``prefix.key``."""
    if not isinstance(branch, dict):
        flat[prefix] = branch
        return
    for key, value in branch.items():
        if key == "metrics":
            continue
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_branch(value, full, flat)
        else:
            flat[full] = value


def log_hparams(cfg: DictConfig) -> None:
    """
    Логирует гиперпараметры data/training/cv в MLflow.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.
    """
    if not _is_mlflow_enabled(cfg):
        return

    import mlflow

    flat: Dict[str, Any] = {}

    for branch_name in ("data", "training", "cv", "baseline"):
        if branch_name not in cfg:
            continue
        branch = OmegaConf.to_container(cfg[branch_name], resolve=True)
        if not isinstance(branch, dict):
            continue
        _flatten_branch(branch, branch_name, flat)

    if "hpo" in cfg:
        hpo_branch = OmegaConf.to_container(cfg.hpo, resolve=True)
        if isinstance(hpo_branch, dict):
            _flatten_branch(hpo_branch, "hpo", flat)

    # MLflow ограничивает длину/число params — режем слишком длинные строки.
    safe = {}
    for key, value in flat.items():
        s = _mlflow_safe_value(value)
        if isinstance(s, str) and len(s) > 250:
            s = s[:247] + "..."
        safe[key] = s

    mlflow.log_params(safe)


def log_split_metrics(
    split_name: str,
    metrics: Dict[str, Any],
    metric_names: List[str],
    step: Optional[int] = None,
) -> None:
    """
    Логирует метрики split в MLflow с префиксом ``train_`` / ``val_`` / ``test_``.

    Parameters
    ----------
    split_name : str
        Имя split: ``train``, ``val`` или ``test``.
    metrics : Dict[str, Any]
        Словарь метрик, включая ``loss``.
    metric_names : List[str]
        Список имён метрик из ``training.metrics``.
    step : Optional[int], optional
        Номер эпохи (step) для кривых обучения.
    """
    import mlflow

    if mlflow.active_run() is None:
        return

    prefix = f"{split_name}_"
    kwargs: Dict[str, Any] = {}
    if step is not None:
        kwargs["step"] = int(step)

    if "loss" in metrics:
        try:
            loss = float(metrics["loss"])
            if not pd.isna(loss):
                mlflow.log_metric(f"{prefix}loss", loss, **kwargs)
        except Exception:
            pass

    for name in metric_names:
        if name not in metrics:
            continue
        try:
            value = float(metrics[name])
        except Exception:
            continue
        if pd.isna(value):
            continue
        mlflow.log_metric(f"{prefix}{name}", value, **kwargs)


def log_overfit_gap_metrics(
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    metric_names: List[str],
    step: int,
) -> None:
    """
    Логирует разрыв train−val для контроля переобучения.

    Parameters
    ----------
    train_metrics : Dict[str, Any]
        Метрики обучающей выборки.
    val_metrics : Dict[str, Any]
        Метрики валидации.
    metric_names : List[str]
        Список имён метрик из ``training.metrics``.
    step : int
        Номер эпохи.
    """
    import mlflow

    if mlflow.active_run() is None:
        return

    names = ["loss", *metric_names]
    seen = set()

    for name in names:
        if name in seen:
            continue
        seen.add(name)

        if name not in train_metrics or name not in val_metrics:
            continue

        try:
            train_value = float(train_metrics[name])
            val_value = float(val_metrics[name])
        except Exception:
            continue

        if pd.isna(train_value) or pd.isna(val_value):
            continue

        # Для loss: val - train (рост = переобучение).
        # Для остальных метрик (выше = лучше): train - val.
        if name == "loss":
            gap = val_value - train_value
        else:
            gap = train_value - val_value

        mlflow.log_metric(f"gap_{name}", float(gap), step=int(step))


def log_artifact_path(cfg: DictConfig, path: Optional[str]) -> None:
    """
    Логирует файл как MLflow artifact.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.
    path : Optional[str]
        Путь к файлу.
    """
    if not _is_mlflow_enabled(cfg) or path is None:
        return

    mlflow_cfg = get_mlflow_cfg(cfg)
    if mlflow_cfg is None or not bool(mlflow_cfg.get("log_artifacts", True)):
        return

    file_path = Path(path)
    if not file_path.is_file():
        return

    import mlflow

    if mlflow.active_run() is None:
        return

    mlflow.log_artifact(str(file_path))


def _mlflow_safe_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
