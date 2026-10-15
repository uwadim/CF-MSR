"""
Training Pipeline.
Обучение модели CF-MSR, валидация, сохранение лучшей модели и прогнозы на holdout.
"""

# 1. Standard library
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 2. External libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

# 3. Local
from config_utils import resolve_path
from dataset import CFMSRDataModule
from mlflow_utils import (
    log_artifact_path,
    log_hparams,
    log_overfit_gap_metrics,
    log_split_metrics,
    mlflow_run_context,
)
from model_factory import build_model


log = logging.getLogger(__name__)


# ==========================================================
# Метрики
# ==========================================================

DEFAULT_METRIC_NAMES: List[str] = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

SUPPORTED_METRIC_NAMES = set(DEFAULT_METRIC_NAMES)
SUPPORTED_SCORE_NAMES = set(SUPPORTED_METRIC_NAMES) | {"loss", "neg_loss"}

METRIC_ALIASES: Dict[str, str] = {
    "acc": "accuracy",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "balanced_acc": "balanced_accuracy",
    "bal_acc": "balanced_accuracy",
    "balancedaccuracy": "balanced_accuracy",
    "precision": "precision",
    "precision_score": "precision",
    "recall": "recall",
    "recall_score": "recall",
    "f1": "f1",
    "f1_score": "f1",
    "roc_auc": "roc_auc",
    "rocauc": "roc_auc",
    "auc": "roc_auc",
    "roc_auc_score": "roc_auc",
    "loss": "loss",
    "neg_loss": "loss",
}


def normalize_metric_name(name: Any) -> str:
    """
    Normalize metric name.
    Приводит имя метрики к каноническому виду.
    """
    s = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return METRIC_ALIASES.get(s, s)


def get_metric_names(training_cfg: DictConfig) -> List[str]:
    """
    Get metric names from config.
    Возвращает список метрик, которые нужно вычислять.
    """
    if "metrics" not in training_cfg:
        return DEFAULT_METRIC_NAMES.copy()

    raw = training_cfg.get("metrics", None)

    # Если metrics: null, считаем все метрики.
    if raw is None:
        return DEFAULT_METRIC_NAMES.copy()

    try:
        raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass

    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, dict):
        items = [str(key) for key, value in raw.items() if bool(value)]
    elif isinstance(raw, (list, tuple)):
        items = [str(item) for item in raw]
    else:
        items = [str(raw)]

    metrics: List[str] = []
    for item in items:
        name = normalize_metric_name(item)
        if name not in SUPPORTED_METRIC_NAMES:
            log.warning(
                "Метрика '%s' не поддерживается. "
                "Поддерживаемые метрики: %s",
                item,
                sorted(SUPPORTED_METRIC_NAMES),
            )
            continue
        if name not in metrics:
            metrics.append(name)

    return metrics


def get_metric_threshold(training_cfg: DictConfig) -> float:
    """
    Get metric threshold.
    Возвращает порог для бинарных предсказаний.
    """
    raw = training_cfg.get("metric_threshold", 0.5)
    if raw is None:
        return 0.5

    try:
        value = float(raw)
    except Exception:
        return 0.5

    if np.isnan(value):
        return 0.5

    return value


def get_score_metric(training_cfg: DictConfig, metric_names: List[str]) -> str:
    """
    Get score metric.
    Возвращает имя метрики, используемой для выбора лучшей модели.
    """
    raw = training_cfg.get("score_metric", "roc_auc")
    if raw is None:
        raw = "roc_auc"

    name = normalize_metric_name(raw)

    if name not in SUPPORTED_SCORE_NAMES:
        log.warning(
            "score_metric='%s' не поддерживается. Используется 'roc_auc'.",
            raw,
        )
        name = "roc_auc"

    if metric_names and name not in metric_names and name not in {"loss", "neg_loss"}:
        log.warning(
            "score_metric='%s' не входит в список metrics=%s. "
            "Если метрика не будет вычислена, будет использован fallback.",
            name,
            metric_names,
        )

    return name


def format_metrics(
    metrics: Dict[str, Any],
    metric_names: Optional[List[str]] = None,
) -> str:
    """
    Format metrics for logging.
    Формирует строку метрик для логов.
    """
    parts: List[str] = []

    if "loss" in metrics:
        try:
            loss = float(metrics["loss"])
            if pd.isna(loss):
                parts.append("loss=nan")
            else:
                parts.append(f"loss={loss:.4f}")
        except Exception:
            parts.append("loss=nan")

    if metric_names is None:
        names = [key for key in metrics.keys() if key != "loss"]
    else:
        names = metric_names

    for name in names:
        if name not in metrics:
            continue
        try:
            value = float(metrics[name])
            if pd.isna(value):
                parts.append(f"{name}=nan")
            else:
                parts.append(f"{name}={value:.4f}")
        except Exception:
            parts.append(f"{name}=nan")

    return " ".join(parts)


def select_score(
    metrics: Dict[str, Any],
    score_metric: str,
) -> Tuple[float, str]:
    """
    Select score for model selection.
    Выбирает score для early stopping и scheduler.
    """
    if score_metric in {"loss", "neg_loss"}:
        try:
            loss = float(metrics.get("loss", float("nan")))
        except Exception:
            loss = float("nan")

        if not pd.isna(loss):
            return -loss, f"neg_loss={-loss:.4f}"

    candidates = [
        score_metric,
        "roc_auc",
        "f1",
        "balanced_accuracy",
        "precision",
        "recall",
        "accuracy",
    ]

    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)

        if name in {"loss", "neg_loss"}:
            continue

        if name not in metrics:
            continue

        try:
            value = float(metrics[name])
        except Exception:
            continue

        if not pd.isna(value):
            return value, f"{name}={value:.4f}"

    try:
        loss = float(metrics.get("loss", float("nan")))
    except Exception:
        loss = float("nan")

    if pd.isna(loss):
        return -np.inf, "score=-inf"

    return -loss, f"neg_loss={-loss:.4f}"


def _manual_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Manual classification metrics.
    Считает accuracy, balanced_accuracy, precision, recall, f1 без sklearn.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())

    total = tp + fp + fn + tn

    accuracy = (tp + tn) / total if total > 0 else float("nan")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Если в выборке есть только один класс, используем recall этого класса.
    # Это ближе к поведению sklearn balanced_accuracy_score,
    # чем принудительное усреднение с нулём для отсутствующего класса.
    if (tp + fn) == 0:
        balanced_accuracy = tnr
    elif (tn + fp) == 0:
        balanced_accuracy = tpr
    else:
        balanced_accuracy = 0.5 * (tpr + tnr)

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _manual_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Manual ROC-AUC.
    Считает ROC-AUC через ранги, если sklearn недоступен.
    """
    y_true = y_true.astype(int)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]

    ranks = np.empty(y_score.size, dtype=np.float64)

    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1

        # Ранги 1-based, для одинаковых score используем средний ранг.
        avg_rank = (i + j + 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(ranks[y_true == 1].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)

    return float(np.clip(auc, 0.0, 1.0))


def safe_metrics(
    y_true: List[int],
    y_score: List[float],
    metric_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute metrics safely.
    Вычисляет выбранные метрики, игнорируя некорректные метки и NaN-скоры.
    """
    if metric_names is None:
        metric_names = DEFAULT_METRIC_NAMES.copy()

    result: Dict[str, float] = {name: float("nan") for name in metric_names}

    if len(metric_names) == 0:
        return result

    try:
        threshold_value = float(threshold)
    except Exception:
        threshold_value = 0.5

    if np.isnan(threshold_value):
        threshold_value = 0.5

    y_true_np = np.asarray(y_true, dtype=np.float32)
    y_score_np = np.asarray(y_score, dtype=np.float32)

    # Оставляем только корректные бинарные метки и валидные скоры.
    valid = (
        (y_true_np >= 0.0)
        & (y_true_np <= 1.0)
        & (~np.isnan(y_score_np))
    )

    y_true_np = y_true_np[valid]
    y_score_np = y_score_np[valid]

    if y_true_np.size == 0:
        return result

    y_int = y_true_np.astype(int)
    y_pred = (y_score_np >= threshold_value).astype(int)

    # Сначала пробуем sklearn, если он доступен.
    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        for name in metric_names:
            try:
                if name == "accuracy":
                    result[name] = float(accuracy_score(y_int, y_pred))

                elif name == "balanced_accuracy":
                    result[name] = float(
                        balanced_accuracy_score(y_int, y_pred)
                    )

                elif name == "precision":
                    result[name] = float(
                        precision_score(
                            y_int,
                            y_pred,
                            zero_division=0,
                        )
                    )

                elif name == "recall":
                    result[name] = float(
                        recall_score(
                            y_int,
                            y_pred,
                            zero_division=0,
                        )
                    )

                elif name == "f1":
                    result[name] = float(
                        f1_score(
                            y_int,
                            y_pred,
                            zero_division=0,
                        )
                    )

                elif name == "roc_auc":
                    # ROC-AUC не определен, если в выборке присутствует только один класс.
                    if len(np.unique(y_int)) >= 2:
                        result[name] = float(
                            roc_auc_score(y_int, y_score_np)
                        )

            except Exception:
                # Если конкретная метрика упала, ниже будет manual fallback.
                pass

    except Exception:
        # sklearn недоступен — используем ручной расчёт.
        pass

    # Manual fallback для метрик, которые остались NaN.
    manual = _manual_classification_metrics(y_int, y_pred)

    for name in metric_names:
        if not pd.isna(result[name]):
            continue

        if name == "roc_auc":
            if len(np.unique(y_int)) >= 2:
                result[name] = _manual_roc_auc(y_int, y_score_np)
        elif name in manual:
            result[name] = manual[name]

    return result


def set_seed(seed: int) -> None:
    """
    Set random seed.
    Фиксирует генераторы случайных чисел для воспроизводимости обучения.

    Parameters
    ----------
    seed : int
        Значение random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Делаем CUDA-вычисления детерминированными.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # For hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_auc(y_true: List[int], y_score: List[float]) -> float:
    """
    Compute ROC-AUC safely.
    Вычисляет ROC-AUC, предварительно отфильтровывая некорректные метки
    и NaN-значения скоров.

    Parameters
    ----------
    y_true : List[int]
        Истинные метки классов.
    y_score : List[float]
        Вероятности или скоры положительного класса.

    Returns
    -------
    float
        Значение ROC-AUC или ``nan``, если метрика не может быть вычислена.
    """
    y_true_np = np.asarray(y_true, dtype=np.float32)
    y_score_np = np.asarray(y_score, dtype=np.float32)

    # Оставляем только корректные бинарные метки и валидные скоры.
    valid = (
        (y_true_np >= 0.0)
        & (y_true_np <= 1.0)
        & (~np.isnan(y_score_np))
    )

    y_true_np = y_true_np[valid]
    y_score_np = y_score_np[valid]

    if len(y_true_np) == 0:
        return float("nan")

    # ROC-AUC не определен, если в выборке присутствует только один класс.
    if len(np.unique(y_true_np)) < 2:
        return float("nan")

    try:
        # Локальный импорт, чтобы падение sklearn не прерывало весь pipeline.
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true_np, y_score_np))
    except Exception:
        # Любая ошибка расчета метрики не должна останавливать обучение.
        return float("nan")


def safe_accuracy(y_true: List[int], y_score: List[float]) -> float:
    """
    Compute accuracy safely.
    Вычисляет accuracy, предварительно отфильтровывая некорректные метки
    и NaN-значения скоров.

    Parameters
    ----------
    y_true : List[int]
        Истинные метки классов.
    y_score : List[float]
        Вероятности или скоры положительного класса.

    Returns
    -------
    float
        Значение accuracy или ``nan``, если метрика не может быть вычислена.
    """
    y_true_np = np.asarray(y_true, dtype=np.float32)
    y_score_np = np.asarray(y_score, dtype=np.float32)

    # Оставляем только корректные бинарные метки и валидные скоры.
    valid = (
        (y_true_np >= 0.0)
        & (y_true_np <= 1.0)
        & (~np.isnan(y_score_np))
    )

    y_true_np = y_true_np[valid]
    y_score_np = y_score_np[valid]

    if len(y_true_np) == 0:
        return float("nan")

    # Порог принятия решения для бинарной классификации.
    preds = (y_score_np >= 0.5).astype(int)
    return float((preds == y_true_np.astype(int)).mean())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
    metric_names: Optional[List[str]] = None,
    metric_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Train one epoch.
    Выполняет одну эпоху обучения модели и возвращает агрегированные метрики.

    Parameters
    ----------
    model : nn.Module
        Обучаемая модель.
    loader : DataLoader
        Загрузчик обучающих графов.
    optimizer : torch.optim.Optimizer
        Оптимизатор параметров модели.
    criterion : nn.Module
        Функция потерь.
    device : torch.device
        Устройство для вычислений.
    grad_clip : float, optional
        Максимальная норма градиентов, по умолчанию 1.0.
    metric_names : Optional[List[str]], optional
        Список метрик для вычисления. Если None, используются все метрики по умолчанию.
    metric_threshold : float, optional
        Порог для бинарных предсказаний, по умолчанию 0.5.

    Returns
    -------
    Dict[str, float]
        Словарь метрик: ``loss``, ``acc``, ``auc``.
        При использовании конфигурации metrics дополнительно возвращает
        accuracy, balanced_accuracy, precision, recall, f1, roc_auc.
    """
    model.train()

    total_loss = 0.0
    total_valid = 0

    all_y: List[int] = []
    all_p: List[float] = []

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(batch).squeeze(-1)
        y = batch.y.squeeze(-1)

        # Для batch_size=1 squeeze может вернуть скаляр.
        # Приводим результат к одномерному виду.
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)

        # Метки около -1 считаются отсутствующими и исключаются из loss.
        valid = (y >= -0.5) & (y <= 1.5)
        if not valid.any():
            continue

        logits_valid = logits[valid]
        y_valid = y[valid].float()

        loss = criterion(logits_valid, y_valid)
        loss.backward()

        # Ограничение градиентов для стабилизации обучения.
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n_valid = int(valid.sum().item())
        total_loss += float(loss.item()) * n_valid
        total_valid += n_valid

        probs = torch.sigmoid(logits_valid).detach().cpu().numpy()

        all_y.extend(y_valid.detach().cpu().numpy().astype(int).tolist())
        all_p.extend(probs.tolist())

    avg_loss = total_loss / max(1, total_valid)

    # Вычисление метрик, заданных конфигурацией Hydra.
    metrics = safe_metrics(
        y_true=all_y,
        y_score=all_p,
        metric_names=metric_names,
        threshold=metric_threshold,
    )

    out: Dict[str, float] = {"loss": avg_loss}
    out.update(metrics)

    # Обратная совместимость со старыми ключами acc/auc.
    if "accuracy" in out:
        out["acc"] = out["accuracy"]
    if "roc_auc" in out:
        out["auc"] = out["roc_auc"]

    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metric_names: Optional[List[str]] = None,
    metric_threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Evaluate model.
    Вычисляет метрики на датасете и формирует DataFrame прогнозов.

    Parameters
    ----------
    model : nn.Module
        Оцениваемая модель.
    loader : DataLoader
        Загрузчик графов.
    criterion : nn.Module
        Функция потерь.
    device : torch.device
        Устройство для вычислений.
    metric_names : Optional[List[str]], optional
        Список метрик для вычисления. Если None, используются все метрики по умолчанию.
    metric_threshold : float, optional
        Порог для бинарных предсказаний, по умолчанию 0.5.

    Returns
    -------
    Dict[str, object]
        Словарь с метриками и прогнозами:
        ``loss``, ``acc``, ``auc``, ``predictions``.
        При использовании конфигурации metrics дополнительно возвращает
        accuracy, balanced_accuracy, precision, recall, f1, roc_auc.
    """
    model.eval()

    total_loss = 0.0
    total_valid = 0

    all_y: List[int] = []
    all_p: List[float] = []
    all_sample_idx: List[int] = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch).squeeze(-1)
        y = batch.y.squeeze(-1)

        # Для batch_size=1 squeeze может вернуть скаляр.
        # Приводим результат к одномерному виду.
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_p.extend(probs.tolist())

        # Пропущенные метки сохраняем как -1 для последующей диагностики.
        y_np = y.detach().cpu().numpy()
        y_np = np.nan_to_num(y_np, nan=-1.0)
        all_y.extend(y_np.astype(int).tolist())

        # Сохраняем идентификаторы образцов, если они есть в батче.
        if hasattr(batch, "sample_idx"):
            all_sample_idx.extend(
                batch.sample_idx.view(-1).detach().cpu().tolist()
            )
        else:
            all_sample_idx.extend(list(range(int(y.numel()))))

        # Loss считается только по объектам с валидной меткой.
        valid = (y >= -0.5) & (y <= 1.5)
        if valid.any():
            logits_valid = logits[valid]
            y_valid = y[valid].float()

            loss = criterion(logits_valid, y_valid)

            n_valid = int(valid.sum().item())
            total_loss += float(loss.item()) * n_valid
            total_valid += n_valid

    avg_loss = total_loss / max(1, total_valid)

    # Вычисление метрик, заданных конфигурацией Hydra.
    metrics = safe_metrics(
        y_true=all_y,
        y_score=all_p,
        metric_names=metric_names,
        threshold=metric_threshold,
    )

    try:
        threshold_value = float(metric_threshold)
    except Exception:
        threshold_value = 0.5

    if np.isnan(threshold_value):
        threshold_value = 0.5

    # Порог принятия решения для бинарной классификации.
    preds = (np.asarray(all_p) >= threshold_value).astype(int).tolist()

    predictions = pd.DataFrame(
        {
            "sample_idx": all_sample_idx,
            "prob": all_p,
            "pred": preds,
            "label": all_y,
        }
    )

    out: Dict[str, object] = {"loss": avg_loss}
    out.update(metrics)

    # Обратная совместимость со старыми ключами acc/auc.
    if "accuracy" in out:
        out["acc"] = out["accuracy"]
    if "roc_auc" in out:
        out["auc"] = out["roc_auc"]

    out["predictions"] = predictions

    return out


def log_metrics_block(
    split_name: str,
    metrics: Dict[str, Any],
    metric_names: List[str],
) -> None:
    """
    Log metrics block.
    Выводит метрики split в лог в формате training.metrics.

    Parameters
    ----------
    split_name : str
        Имя split, например ``validation`` или ``test``.
    metrics : Dict[str, Any]
        Словарь метрик.
    metric_names : List[str]
        Список имён метрик из конфигурации.
    """
    log.info(
        "%s metrics: %s",
        split_name,
        format_metrics(metrics, metric_names),
    )


@dataclass
class FitResult:
    """
    Fit result.

    Результат цикла обучения модели.
    """

    best_score: float
    best_epoch: int
    best_state: Dict[str, torch.Tensor]
    history: List[Dict[str, Any]] = field(default_factory=list)


class LrSchedulerWrapper:
    """
    LR scheduler wrapper.

    Обёртка над LR-scheduler с единым методом ``step(score)``.
    """

    def __init__(
            self,
            scheduler: Any,
            needs_score: bool = True,
    ) -> None:
        """
        Initialize wrapper.

        Parameters
        ----------
        scheduler : Any
            Объект scheduler из ``torch.optim.lr_scheduler``.
        needs_score : bool, optional
            Если True, в ``step`` передаётся метрика (plateau).
        """
        self.scheduler = scheduler
        self.needs_score = needs_score

    def step(self, score: Optional[float] = None) -> None:
        """
        Step scheduler.

        Parameters
        ----------
        score : Optional[float], optional
            Значение метрики для ReduceLROnPlateau.
        """
        if self.needs_score:
            self.scheduler.step(score)
        else:
            self.scheduler.step()


def _resolve_output_paths(training_cfg: DictConfig) -> Tuple[str, str]:
    """
    Resolve output paths.

    Возвращает пути для сохранения модели и прогнозов holdout.

    Parameters
    ----------
    training_cfg : DictConfig
        Ветка ``training`` конфигурации.

    Returns
    -------
    Tuple[str, str]
        Пара ``(model_out, predictions_out)``.
    """
    model_out = resolve_path(training_cfg.get("model_out", "cf_msr_best.pt"))
    predictions_out = resolve_path(
        training_cfg.get("predictions_out", "holdout_predictions.csv")
    )

    if model_out is None:
        model_out = resolve_path("cf_msr_best.pt")
    if predictions_out is None:
        predictions_out = resolve_path("holdout_predictions.csv")

    if model_out is None:
        model_out = "cf_msr_best.pt"
    if predictions_out is None:
        predictions_out = "holdout_predictions.csv"

    return str(model_out), str(predictions_out)


def build_optimizer(
    model: nn.Module,
    training_cfg: DictConfig,
) -> torch.optim.Optimizer:
    """
    Build optimizer.

    Создаёт AdamW. При ``training.adamw_no_decay_bias_norm=true``
    bias, LayerNorm и Embedding исключаются из weight decay.

    Parameters
    ----------
    model : nn.Module
        Обучаемая модель.
    training_cfg : DictConfig
        Ветка ``training`` с ``lr`` и ``weight_decay``.

    Returns
    -------
    torch.optim.Optimizer
        Оптимизатор AdamW.
    """
    lr = float(training_cfg.lr)
    weight_decay = float(training_cfg.weight_decay)
    use_no_decay = bool(training_cfg.get("adamw_no_decay_bias_norm", False))

    if not use_no_decay:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_l = name.lower()
        if (
            name_l.endswith("bias")
            or "layernorm" in name_l
            or "layer_norm" in name_l
            or "norm.weight" in name_l
            or "embedding" in name_l
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: DictConfig,
) -> LrSchedulerWrapper:
    """
    Build LR scheduler.

    Создаёт обёртку над ReduceLROnPlateau или cosine+warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Оптимизатор.
    training_cfg : DictConfig
        Ветка ``training``.

    Returns
    -------
    LrSchedulerWrapper
        Обёртка с методом ``step(score)``.
    """
    kind = str(training_cfg.get("lr_scheduler", "plateau")).strip().lower()
    patience = int(training_cfg.patience)
    epochs = int(training_cfg.epochs)
    warmup_epochs = int(training_cfg.get("warmup_epochs", 0) or 0)

    if kind in ("", "plateau", "reduce_on_plateau", "reducelronplateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(5, patience // 2),
        )
        return LrSchedulerWrapper(scheduler, needs_score=True)

    if kind == "cosine":
        warmup = max(0, warmup_epochs)
        cosine_epochs = max(1, epochs - warmup)
        schedulers: List[Any] = []
        milestones: List[int] = []

        if warmup > 0:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-2,
                    end_factor=1.0,
                    total_iters=warmup,
                )
            )
            milestones.append(warmup)

        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
            )
        )

        if len(schedulers) == 1:
            return LrSchedulerWrapper(schedulers[0], needs_score=False)

        sequential = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
        )
        return LrSchedulerWrapper(sequential, needs_score=False)

    raise ValueError(f"Неизвестный lr_scheduler: {kind!r}")


def moving_average(values: List[float], window: int) -> float:
    """
    Moving average.

    Сглаживает последние ``window`` значений.

    Parameters
    ----------
    values : List[float]
        Последовательность значений.
    window : int
        Размер окна (>= 1).

    Returns
    -------
    float
        Среднее по последним ``window`` элементам (или меньше, если данных мало).
    """
    if window < 1:
        raise ValueError(f"window должен быть >= 1, получено {window}")
    if not values:
        return float("nan")
    tail = values[-window:]
    return float(np.mean(tail))


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: nn.Module,
    training_cfg: DictConfig,
    device: torch.device,
    metric_names: List[str],
    metric_threshold: float,
    eval_criterion: Optional[nn.Module] = None,
) -> FitResult:
    """
    Fit model.

    Запускает цикл эпох с early stopping и сохранением лучшего состояния.

    Parameters
    ----------
    model : nn.Module
        Обучаемая модель.
    train_loader : DataLoader
        Загрузчик обучающих данных.
    val_loader : Optional[DataLoader]
        Загрузчик валидации или None.
    criterion : nn.Module
        Функция потерь для train.
    training_cfg : DictConfig
        Ветка ``training``.
    device : torch.device
        Устройство вычислений.
    metric_names : List[str]
        Список метрик.
    metric_threshold : float
        Порог бинарных предсказаний.
    eval_criterion : Optional[nn.Module], optional
        Функция потерь для val; если None — используется ``criterion``.

    Returns
    -------
    FitResult
        Лучший score, эпоха, state_dict и история.
    """
    optimizer = build_optimizer(model, training_cfg)
    scheduler = build_scheduler(optimizer, training_cfg)
    val_criterion = eval_criterion if eval_criterion is not None else criterion

    patience = int(training_cfg.patience)
    grad_clip = float(training_cfg.grad_clip)
    min_delta = float(training_cfg.get("early_stop_min_delta", 0.0) or 0.0)
    smooth_window = int(training_cfg.get("monitor_smoothing_window", 1) or 1)

    metric_names_local = metric_names
    score_metric = get_score_metric(training_cfg, metric_names_local)
    monitor_raw = training_cfg.get("monitor_metric", None)
    if monitor_raw is None:
        monitor_metric = score_metric
    else:
        monitor_metric = get_score_metric(
            OmegaConf.create({"score_metric": monitor_raw}),
            metric_names_local,
        )

    best_score = -np.inf
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []
    raw_scores: List[float] = []

    for epoch in range(1, int(training_cfg.epochs) + 1):
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=grad_clip,
            metric_names=metric_names_local,
            metric_threshold=metric_threshold,
        )

        log_split_metrics("train", train_metrics, metric_names_local, step=epoch)

        val_metrics: Optional[Dict[str, Any]] = None
        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=val_criterion,
                device=device,
                metric_names=metric_names_local,
                metric_threshold=metric_threshold,
            )

            score, score_desc = select_score(val_metrics, monitor_metric)
            scheduler.step(score)

            log_split_metrics("val", val_metrics, metric_names_local, step=epoch)
            log_overfit_gap_metrics(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                metric_names=metric_names_local,
                step=epoch,
            )

            if _is_mlflow_active():
                import mlflow

                mlflow.log_metric("score", float(score), step=epoch)

            log.info(
                "Epoch %03d | train %s | val %s | score=%s",
                epoch,
                format_metrics(train_metrics, metric_names_local),
                format_metrics(val_metrics, metric_names_local),
                score_desc,
            )
        else:
            score, score_desc = select_score(train_metrics, monitor_metric)
            scheduler.step(score)

            if _is_mlflow_active():
                import mlflow

                mlflow.log_metric("score", float(score), step=epoch)

            log.info(
                "Epoch %03d | train %s | score=%s",
                epoch,
                format_metrics(train_metrics, metric_names_local),
                score_desc,
            )

        if np.isnan(score):
            score = -np.inf

        raw_scores.append(float(score))
        smoothed = moving_average(raw_scores, smooth_window)

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "score": float(score),
                "smoothed_score": smoothed,
            }
        )

        if smoothed > best_score + min_delta:
            best_score = smoothed
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience:
            log.info("Early stopping after %d epochs", epoch)
            break

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_epoch = int(training_cfg.epochs)

    model.load_state_dict(best_state)
    model.to(device)

    return FitResult(
        best_score=float(best_score),
        best_epoch=int(best_epoch),
        best_state=best_state,
        history=history,
    )


def build_criteria(
    pos_weight: torch.Tensor,
    label_smoothing: float = 0.0,
) -> Tuple[nn.Module, nn.Module]:
    """
    Build train/eval criteria.

    Для train — опционально сглаженный BCE; для val/test — обычный BCE.

    Parameters
    ----------
    pos_weight : torch.Tensor
        Вес положительного класса.
    label_smoothing : float, optional
        Коэффициент сглаживания меток; 0 — без сглаживания.

    Returns
    -------
    Tuple[nn.Module, nn.Module]
        Пара ``(train_criterion, eval_criterion)``.
    """
    eval_criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eps = float(label_smoothing)
    if eps <= 0.0:
        return eval_criterion, eval_criterion
    return SmoothedBCEWithLogitsLoss(pos_weight=pos_weight, smoothing=eps), eval_criterion


class SmoothedBCEWithLogitsLoss(nn.Module):
    """
    Smoothed BCE with logits.

    Цель: ``y * (1 - eps) + 0.5 * eps``; поддерживает ``pos_weight``.
    """

    def __init__(
            self,
            pos_weight: Optional[torch.Tensor] = None,
            smoothing: float = 0.0,
    ) -> None:
        """
        Initialize loss.

        Parameters
        ----------
        pos_weight : Optional[torch.Tensor], optional
            Вес положительного класса.
        smoothing : float, optional
            Коэффициент сглаживания меток.
        """
        super().__init__()
        if not 0.0 <= float(smoothing) < 1.0:
            raise ValueError(
                f"label_smoothing должен быть в [0, 1), получено {smoothing}"
            )
        self.smoothing = float(smoothing)
        self._bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward.

        Parameters
        ----------
        logits : torch.Tensor
            Логиты модели.
        target : torch.Tensor
            Бинарные метки.

        Returns
        -------
        torch.Tensor
            Значение loss.
        """
        eps = self.smoothing
        if eps <= 0.0:
            return self._bce(logits, target)
        soft = target * (1.0 - eps) + 0.5 * eps
        return self._bce(logits, soft)


def run_training(cfg: DictConfig) -> float:
    """
    Run training.
    Запускает полный цикл обучения модели CF-MSR:
    подготовку данных, обучение, валидацию, сохранение модели
    и формирование прогнозов на holdout-выборке.

    Parameters
    ----------
    cfg : DictConfig
        Hydra-конфигурация с ветками ``data`` и ``training``.
    Returns
    -------
    float
        Лучший score на validation (или train, если val отсутствует).
        Используется Optuna Sweeper как objective.
    """
    log.info("Hydra config:\n%s", OmegaConf.to_yaml(cfg))

    cv_cfg = cfg.get("cv", None)
    if cv_cfg is not None and bool(cv_cfg.get("enabled", False)):
        from crossval import run_cv_training

        with mlflow_run_context(cfg):
            return run_cv_training(cfg)

    with mlflow_run_context(cfg):
        return _run_training_impl(cfg)


def _run_training_impl(cfg: DictConfig) -> float:
    """
    Run single-split training.

    Оркестратор: данные → модель → fit_model → финальная оценка.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.

    Returns
    -------
    float
        Лучший validation score.
    """
    data_cfg = cfg.data
    training_cfg = cfg.training

    log_hparams(cfg)

    seed = int(training_cfg.seed)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_out, predictions_out = _resolve_output_paths(training_cfg)

    dm = CFMSRDataModule(
        data_cfg=data_cfg,
        training_cfg=training_cfg,
    )
    dm.setup()

    train_loader = dm.train_loader()
    val_loader = dm.val_loader()
    test_loader = dm.test_loader()

    metric_names = get_metric_names(training_cfg)
    metric_threshold = get_metric_threshold(training_cfg)
    score_metric = get_score_metric(training_cfg, metric_names)

    log.info("=" * 80)
    log.info("Данные подготовлены")
    log.info("=" * 80)
    log.info("num_nodes: %d", dm.num_nodes)
    log.info("node_input_dim: %d", dm.node_input_dim)
    log.info(
        "train samples: %d",
        len(dm.train_idx) if dm.train_idx is not None else 0,
    )
    log.info(
        "val samples: %d",
        len(dm.val_idx) if dm.val_idx is not None else 0,
    )
    log.info(
        "test samples: %d",
        len(dm.test_dataset) if dm.test_dataset is not None else 0,
    )
    log.info("pos_weight: %.4f", dm.pos_weight)
    log.info("device: %s", device)
    log.info("metrics: %s", metric_names if metric_names else "[]")
    log.info("metric_threshold: %.4f", metric_threshold)
    log.info("score_metric: %s", score_metric)
    log.info("=" * 80)

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
    label_smoothing = float(training_cfg.get("label_smoothing", 0.0) or 0.0)
    train_criterion, eval_criterion = build_criteria(
        pos_weight=pos_weight,
        label_smoothing=label_smoothing,
    )

    fit_result = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=train_criterion,
        training_cfg=training_cfg,
        device=device,
        metric_names=metric_names,
        metric_threshold=metric_threshold,
        eval_criterion=eval_criterion,
    )

    best_score = fit_result.best_score
    torch.save(model.state_dict(), model_out)
    log.info("Лучшая модель сохранена в %s", model_out)
    log.info("Лучший score: %.4f (epoch %d)", best_score, fit_result.best_epoch)

    if val_loader is not None:
        val_final_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=eval_criterion,
            device=device,
            metric_names=metric_names,
            metric_threshold=metric_threshold,
        )
        log.info("=" * 80)
        log.info("Validation results (best model)")
        log.info("=" * 80)
        log_metrics_block("validation", val_final_metrics, metric_names)
        log_split_metrics("best_val", val_final_metrics, metric_names)

    if test_loader is not None:
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=eval_criterion,
            device=device,
            metric_names=metric_names,
            metric_threshold=metric_threshold,
        )

        log.info("=" * 80)
        log.info("Holdout results (best model)")
        log.info("=" * 80)
        log_metrics_block("test", test_metrics, metric_names)
        log_split_metrics("best_test", test_metrics, metric_names)

        predictions_obj = test_metrics.get("predictions")
        if not isinstance(predictions_obj, pd.DataFrame):
            raise TypeError("evaluate() вернул некорректный объект predictions")

        pred_df = predictions_obj.copy()

        if dm.test_dataset is not None and hasattr(dm.test_dataset, "sample_ids"):
            sample_ids = dm.test_dataset.sample_ids
            pred_df["sample_name"] = pred_df["sample_idx"].map(
                lambda i: sample_ids[i] if 0 <= i < len(sample_ids) else None
            )

        pred_df.to_csv(predictions_out, index=False)
        log.info("Прогнозы сохранены в %s", predictions_out)

    log_artifact_path(cfg, model_out)
    log_artifact_path(cfg, predictions_out if test_loader is not None else None)

    if _is_mlflow_active():
        import mlflow

        mlflow.log_metric("best_score", float(best_score))
        mlflow.log_metric("best_epoch", float(fit_result.best_epoch))

    return float(best_score)


def _is_mlflow_active() -> bool:
    try:
        import mlflow

        return mlflow.active_run() is not None
    except Exception:
        return False
