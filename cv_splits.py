"""
CV splits.

Чистые функции построения fold-разбиений для кросс-валидации.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from omegaconf import DictConfig


def make_cv_splits(
    labels: np.ndarray,
    folds: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    cv_cfg: DictConfig,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Make CV splits.

    Строит список пар ``(train_idx, val_idx)`` по схеме из ``cv_cfg``.

    Parameters
    ----------
    labels : np.ndarray
        Метки образцов.
    folds : Optional[np.ndarray]
        Номера fold из колонок графа (для ``fold_column``).
    groups : Optional[np.ndarray]
        Идентификаторы групп/пациентов (для ``group_kfold``).
    cv_cfg : DictConfig
        Ветка ``cv`` конфигурации.
    seed : int
        Random seed.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        Список разбиений.
    """
    n_splits = int(cv_cfg.get("n_splits", 5))
    n_repeats = int(cv_cfg.get("n_repeats", 1))
    scheme = str(cv_cfg.get("scheme", "stratified_kfold")).strip().lower()

    if n_splits < 2:
        raise ValueError(f"cv.n_splits должен быть >= 2, получено {n_splits}")

    labels = np.asarray(labels)
    valid = np.isfinite(labels) & (labels >= 0.0) & (labels <= 1.0)
    if valid.sum() < n_splits:
        raise ValueError(
            f"Недостаточно объектов с валидной меткой ({valid.sum()}) "
            f"для n_splits={n_splits}"
        )

    indices = np.arange(len(labels))

    if scheme == "fold_column":
        if folds is None:
            raise ValueError("scheme=fold_column требует dataset.folds")
        folds_arr = np.asarray(folds)
        unique = sorted({int(f) for f in folds_arr.tolist()})
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for fold_id in unique:
            val_idx = indices[folds_arr == fold_id]
            train_idx = indices[folds_arr != fold_id]
            _validate_both_classes(labels, train_idx, val_idx, fold_id)
            if len(val_idx) == 0 or len(train_idx) == 0:
                continue
            splits.append((train_idx, val_idx))
        if len(splits) < 2:
            raise ValueError("fold_column: получено меньше 2 непустых fold")
        return splits

    if scheme == "stratified_kfold":
        from sklearn.model_selection import RepeatedStratifiedKFold

        y = labels.copy()
        # Для сплита заменяем NaN временной меткой, но индексы берём все;
        # объекты без метки лучше исключить.
        valid_idx = indices[valid]
        y_valid = labels[valid].astype(int)
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=int(seed),
        )
        splits = []
        for fold_i, (tr_rel, va_rel) in enumerate(rskf.split(valid_idx, y_valid)):
            train_idx = valid_idx[tr_rel]
            val_idx = valid_idx[va_rel]
            _validate_both_classes(labels, train_idx, val_idx, fold_i)
            splits.append((train_idx, val_idx))
        return splits

    if scheme == "group_kfold":
        if groups is None:
            raise ValueError(
                "scheme=group_kfold требует data.group_col и dataset.groups"
            )
        from sklearn.model_selection import StratifiedGroupKFold

        valid_idx = indices[valid]
        y_valid = labels[valid].astype(int)
        g_valid = np.asarray(groups)[valid]
        # StratifiedGroupKFold не имеет n_repeats — повторяем с разными seed.
        splits = []
        for rep in range(max(1, n_repeats)):
            sgkf = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=int(seed) + rep,
            )
            for fold_i, (tr_rel, va_rel) in enumerate(
                sgkf.split(valid_idx, y_valid, g_valid)
            ):
                train_idx = valid_idx[tr_rel]
                val_idx = valid_idx[va_rel]
                _validate_group_disjoint(groups, train_idx, val_idx)
                _validate_both_classes(labels, train_idx, val_idx, fold_i)
                splits.append((train_idx, val_idx))
        return splits

    raise ValueError(
        f"Неизвестный cv.scheme={scheme!r}; "
        "ожидается fold_column | stratified_kfold | group_kfold"
    )


def _validate_both_classes(
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_id: Any,
) -> None:
    """Проверяет, что в train и val есть оба класса."""
    for name, idx in (("train", train_idx), ("val", val_idx)):
        y = labels[idx]
        y = y[np.isfinite(y) & (y >= 0.0) & (y <= 1.0)]
        uniq = set(np.unique(y.astype(int)).tolist())
        if uniq != {0, 1}:
            raise ValueError(
                f"Fold {fold_id}: в {name} нет обоих классов, uniq={uniq}"
            )


def _validate_group_disjoint(
    groups: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    """Проверяет, что группы не пересекаются между train и val."""
    g_tr = set(np.asarray(groups)[train_idx].tolist())
    g_va = set(np.asarray(groups)[val_idx].tolist())
    overlap = g_tr & g_va
    if overlap:
        raise ValueError(f"group_kfold: группы пересекаются: {sorted(overlap)[:5]}")
