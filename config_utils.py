"""
Утилиты разбора Hydra-конфигурации.

Разрешение путей, разбор списков признаков и опциональных скалярных значений.
"""

# 1. Standard library
import os
from typing import List, Optional

# 2. External libraries
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def resolve_path(path: object) -> Optional[str]:
    """
    Разрешает путь относительно оригинальной рабочей директории запуска.

    Hydra меняет текущую рабочую директорию на run dir.
    Чтобы пути к данным и результатам оставались предсказуемыми,
    используется ``to_absolute_path``.

    Parameters
    ----------
    path : object
        Значение пути из конфигурации.

    Returns
    -------
    Optional[str]
        Абсолютный путь или ``None``, если путь пустой либо null-подобный.
    """
    if path is None:
        return None

    path_str = str(path).strip()

    # Пустые и null-подобные значения считаем отсутствующим путем.
    if path_str == "" or path_str.lower() in {"null", "none"}:
        return None

    try:
        resolved = to_absolute_path(path_str)
    except Exception:  # noqa: BLE001 - безопасный fallback для путей
        # При сбое Hydra-резолвера переходим на стандартный абсолютный путь.
        resolved = os.path.abspath(path_str)

    parent = os.path.dirname(resolved)
    if parent:
        # Создаем родительские директории для сохранения результатов.
        os.makedirs(parent, exist_ok=True)

    return resolved


def parse_feature_cols(cfg: DictConfig) -> Optional[List[str]]:
    """
    Достает feature_cols из ветки data-конфига.

    Поддерживает варианты:
    - ``feature_cols: null``
    - ``feature_cols: ["a", "b"]``
    - ``feature_cols: a,b,c``

    Parameters
    ----------
    cfg : DictConfig
        Ветка data-конфига.

    Returns
    -------
    Optional[List[str]]
        Список имен признаков или ``None``, если список не задан.

    Raises
    ------
    ValueError
        Если ``feature_cols`` задан словарем.
    """
    raw: object = cfg.get("feature_cols", None)

    if raw is None:
        return None

    # Строковый вариант: перечисление колонок через запятую.
    if isinstance(raw, str):
        raw = raw.strip()

        if raw == "" or raw.lower() in {"null", "none"}:
            return None

        items = [item.strip() for item in raw.split(",")]
        return [item for item in items if item]

    # Приводим OmegaConf-узел к обычному Python-контейнеру.
    container: object = OmegaConf.to_container(raw, resolve=True)

    if container is None:
        return None

    if isinstance(container, str):
        container = container.strip()

        if container == "" or container.lower() in {"null", "none"}:
            return None

        return [container]

    if isinstance(container, (list, tuple)):
        return [str(item) for item in container]

    if isinstance(container, dict):
        raise ValueError(
            "feature_cols должен быть списком или строкой со списком через запятую."
        )

    # Одиночные скалярные значения оборачиваем в список.
    return [str(container)]


def parse_optional_int(value: object) -> Optional[int]:
    """
    Преобразует значение конфига в int или None.

    Parameters
    ----------
    value : object
        Значение из конфигурации.

    Returns
    -------
    Optional[int]
        Целое число или ``None``.

    Raises
    ------
    TypeError
        Если значение не может быть преобразовано к int.
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        # Пустые и null-подобные строки считаем отсутствующим значением.
        if value == "" or value.lower() in {"null", "none"}:
            return None

        return int(value)

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    raise TypeError(f"Не удалось преобразовать значение к int: {value!r}")


def parse_optional_float(value: object) -> Optional[float]:
    """
    Преобразует значение конфига в float или None.

    Parameters
    ----------
    value : object
        Значение из конфигурации.

    Returns
    -------
    Optional[float]
        Вещественное число или ``None``.

    Raises
    ------
    TypeError
        Если значение не может быть преобразовано к float.
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        # Пустые и null-подобные строки считаем отсутствующим значением.
        if value == "" or value.lower() in {"null", "none"}:
            return None

        return float(value)

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, int):
        return float(value)

    if isinstance(value, float):
        return value

    raise TypeError(f"Не удалось преобразовать значение к float: {value!r}")


def parse_optional_str(value: object) -> Optional[str]:
    """
    Преобразует значение конфига в str или None.

    Parameters
    ----------
    value : object
        Значение из конфигурации.

    Returns
    -------
    Optional[str]
        Строка или ``None``.
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        # Пустые и null-подобные строки считаем отсутствующим значением.
        if value == "" or value.lower() in {"null", "none"}:
            return None

        return value

    return str(value)
