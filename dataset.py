"""
Dataset Module.

Загрузка и подготовка графовых данных для модели CF-MSR.
"""

# 1. Standard library
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 2. External libraries
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# 3. Local
from config_utils import (
    parse_feature_cols,
    parse_optional_int,
    parse_optional_str,
    resolve_path,
)

# Регулярные выражения для разбора имён sample-колонок в csv-файлах графов
SAMPLE_COL_RE = re.compile(r"^(\d+)(?:[_-](.*))?$")
SAMPLE_COL_FOLD_SUFFIX_RE = re.compile(r"^fold[_-]?(\d+)$", re.IGNORECASE)


def parse_sample_column(name: str) -> Optional[Tuple[int, int]]:
    """
    Parse sample column name.

    Разбирает имя sample-колонки графа на номер объекта и fold.
    Номер объекта — ведущие цифры в имени колонки. Содержимое после
    первого разделителя ``_``/``-`` учитывается только для извлечения fold.

    Parameters
    ----------
    name : str
        Имя колонки из csv-файла графа.

    Returns
    -------
    Optional[Tuple[int, int]]
        Пара ``(sample_num, fold)`` или ``None``, если колонка не sample-колонка.
    """
    name = str(name).strip()
    match = SAMPLE_COL_RE.match(name)

    if not match:
        return None

    sample_num = int(match.group(1))
    suffix = match.group(2)
    fold = 0

    # Извлечение номера fold из суффикса, если он присутствует
    if suffix is not None:
        fold_match = SAMPLE_COL_FOLD_SUFFIX_RE.match(suffix.strip())
        if fold_match:
            fold = int(fold_match.group(1))

    return sample_num, fold


def strip_num_prefix(name: str) -> str:
    """
    Strip num prefix.

    Удаляет префикс `num__` из имени признака.

    Parameters
    ----------
    name : str
        Исходное имя признака или узла графа.

    Returns
    -------
    str
        Имя без префикса num__ или исходное имя, если префикса нет.
    """
    name = str(name).strip()
    if name.startswith("num__"):
        return name[5:]
    return name


def build_node_map(feature_cols: List[str]) -> Dict[str, int]:
    """
    Build node map.

    Строит отображение имён признаков в индексы узлов графа.
    Для каждого признака регистрируются варианты написания с префиксом
    ``num__`` и без него, чтобы сопоставлять узлы из csv-файлов графов.

    Parameters
    ----------
    feature_cols : List[str]
        Список имён признаковых колонок табличного датасета.

    Returns
    -------
    Dict[str, int]
        Словарь ``имя_признака -> индекс_узла``.
    """
    node_map: Dict[str, int] = {}

    for idx, col in enumerate(feature_cols):
        col = str(col).strip()
        stripped = strip_num_prefix(col)

        # Генерируем все возможные варианты написания имени узла
        candidates = {
            col,
            stripped,
            f"num__{col}",
            f"num__{stripped}",
        }

        for cand in candidates:
            if cand and cand not in node_map:
                node_map[cand] = idx

    return node_map


def resolve_node(name: Any, node_map: Dict[str, int]) -> Optional[int]:
    """
    Resolve node index.

    Находит индекс узла по имени из колонок `p1`/`p2` графа.

    Parameters
    ----------
    name : Any
        Имя узла из csv-файла графа.
    node_map : Dict[str, int]
        Отображение имён признаков в индексы узлов.

    Returns
    -------
    Optional[int]
        Индекс узла или ``None``, если сопоставление не найдено.
    """
    try:
        if pd.isna(name):
            return None
    except TypeError:
        pass

    name = str(name).strip()

    # Проверяем различные варианты написания имени узла
    candidates = [
        name,
        strip_num_prefix(name),
        f"num__{name}",
        name.replace("num__", ""),
    ]

    for cand in candidates:
        cand = cand.strip()
        if cand in node_map:
            return node_map[cand]

    return None


_EDGE_NODE_STATS = frozenset({"mean", "std", "max", "min"})


def compute_edge_node_features(
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        num_nodes: int,
        stats: Sequence[str],
) -> np.ndarray:
    """
    Compute edge-derived node features.

    Для каждого узла считает статистики весов инцидентных рёбер
    (агрегация по ``edge_index[0]``; оба направления уже в индексе).

    Parameters
    ----------
    edge_index : np.ndarray
        Индексы рёбер формы ``(2, E)``.
    edge_attr : np.ndarray
        Веса рёбер формы ``(E, S)``.
    num_nodes : int
        Число узлов.
    stats : Sequence[str]
        Подмножество ``mean | std | max | min``.

    Returns
    -------
    np.ndarray
        Массив формы ``(S, num_nodes, len(stats))``.
    """
    if len(stats) == 0:
        raise ValueError("stats не должен быть пустым")

    unknown = [s for s in stats if str(s) not in _EDGE_NODE_STATS]
    if unknown:
        raise ValueError(
            f"Неизвестные статистики: {unknown}; "
            f"ожидается подмножество {_EDGE_NODE_STATS}"
        )

    edge_index = np.asarray(edge_index)
    edge_attr = np.asarray(edge_attr, dtype=np.float64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index должен иметь форму (2, E), получено {edge_index.shape}"
        )
    if edge_attr.ndim != 2:
        raise ValueError(
            f"edge_attr должен иметь форму (E, S), получено {edge_attr.shape}"
        )
    if edge_index.shape[1] != edge_attr.shape[0]:
        raise ValueError(
            f"Число рёбер в edge_index ({edge_index.shape[1]}) не совпадает "
            f"с edge_attr ({edge_attr.shape[0]})"
        )
    if num_nodes < 0:
        raise ValueError(f"num_nodes должен быть >= 0, получено {num_nodes}")

    src = edge_index[0].astype(np.int64, copy=False)
    n_edges, n_samples = edge_attr.shape
    out = np.zeros((n_samples, num_nodes, len(stats)), dtype=np.float32)

    degree = np.zeros(num_nodes, dtype=np.float64)
    if n_edges > 0:
        np.add.at(degree, src, 1.0)

    for s_i in range(n_samples):
        w = edge_attr[:, s_i]
        for j, name in enumerate(stats):
            name = str(name)
            if name == "mean":
                sums = np.zeros(num_nodes, dtype=np.float64)
                if n_edges > 0:
                    np.add.at(sums, src, w)
                vals = sums / np.maximum(degree, 1.0)
                vals[degree == 0] = 0.0
            elif name == "std":
                sum_w = np.zeros(num_nodes, dtype=np.float64)
                sum_w2 = np.zeros(num_nodes, dtype=np.float64)
                if n_edges > 0:
                    np.add.at(sum_w, src, w)
                    np.add.at(sum_w2, src, w * w)
                mean = sum_w / np.maximum(degree, 1.0)
                mean_sq = sum_w2 / np.maximum(degree, 1.0)
                vals = np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))
                vals[degree == 0] = 0.0
            elif name == "max":
                vals = np.full(num_nodes, -np.inf, dtype=np.float64)
                if n_edges > 0:
                    np.maximum.at(vals, src, w)
                vals[~np.isfinite(vals)] = 0.0
            else:  # min
                vals = np.full(num_nodes, np.inf, dtype=np.float64)
                if n_edges > 0:
                    np.minimum.at(vals, src, w)
                vals[~np.isfinite(vals)] = 0.0
            out[s_i, :, j] = vals.astype(np.float32)

    return out


class WeightedFullGraphDataset(Dataset):
    """
    Weighted Full Graph Dataset.

    PyG-датасет полных графов с индивидуальными весами рёбер.
    Каждый элемент — один образец (граф): общая топология ``edge_index``,
    но собственные веса рёбер ``edge_attr`` и признаки узлов ``x``.
    """

    def __init__(
            self,
            node_features: np.ndarray,
            edge_index: torch.Tensor,
            edge_attr: np.ndarray,
            labels: np.ndarray,
            sample_ids: Optional[List[str]] = None,
            folds: Optional[np.ndarray] = None,
            node_extra: Optional[np.ndarray] = None,
            include_raw_node_feature: bool = True,
    ) -> None:
        """
        Initialize dataset.

        Инициализирует датасет графовых образцов.

        Parameters
        ----------
        node_features : np.ndarray
            Матрица признаков узлов формы ``(num_samples, num_nodes)``.
        edge_index : torch.Tensor
            Индексы рёбер общего графа формы ``(2, num_edges)``.
        edge_attr : np.ndarray
            Веса рёбер формы ``(num_edges, num_samples)``.
        labels : np.ndarray
            Метки классов формы ``(num_samples,)``.
        sample_ids : Optional[List[str]], optional
            Имена sample-колонок; по умолчанию — строковые индексы.
        folds : Optional[np.ndarray], optional
            Номера fold для каждого образца.
        node_extra : Optional[np.ndarray], optional
            Доп. признаки узлов ``(num_samples, num_nodes, F_extra)``.
        include_raw_node_feature : bool, optional
            Включать ли табличный признак узла первым каналом ``x``.
        """
        super().__init__()

        self.node_features = node_features.astype(np.float32)
        self.edge_index = edge_index.long()
        self.edge_attr = edge_attr.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.include_raw_node_feature = bool(include_raw_node_feature)

        if sample_ids is None:
            sample_ids = [str(i) for i in range(len(self.labels))]
        self.sample_ids = sample_ids

        self.folds: Optional[np.ndarray] = (
            folds.astype(np.int64) if folds is not None else None
        )

        assert self.node_features.shape[0] == self.labels.shape[0]
        assert self.edge_attr.shape[1] == self.labels.shape[0]
        assert self.edge_index.shape[1] == self.edge_attr.shape[0]

        if node_extra is not None:
            node_extra = np.asarray(node_extra, dtype=np.float32)
            if node_extra.ndim != 3:
                raise ValueError(
                    f"node_extra должен иметь форму (S, N, F), "
                    f"получено {node_extra.shape}"
                )
            if node_extra.shape[:2] != self.node_features.shape:
                raise ValueError(
                    f"node_extra.shape[:2]={node_extra.shape[:2]} не совпадает "
                    f"с node_features.shape={self.node_features.shape}"
                )
            self.node_extra: Optional[np.ndarray] = node_extra
        else:
            self.node_extra = None

        if (not self.include_raw_node_feature) and self.node_extra is None:
            raise ValueError(
                "include_raw_node_feature=false требует заданный node_extra"
            )

        self.groups: Optional[np.ndarray] = None

    @property
    def node_input_dim(self) -> int:
        """
        Node input dimension.

        Размерность признака узла ``x`` в ``get``.

        Returns
        -------
        int
            Число каналов на узел.
        """
        f_extra = 0 if self.node_extra is None else int(self.node_extra.shape[-1])
        if self.include_raw_node_feature:
            return 1 + f_extra
        return f_extra

    def replace(
            self,
            node_features: Optional[np.ndarray] = None,
            edge_index: Optional[torch.Tensor] = None,
            edge_attr: Optional[np.ndarray] = None,
            node_extra: Optional[np.ndarray] = None,
    ) -> "WeightedFullGraphDataset":
        """
        Replace arrays.

        Создаёт копию датасета с опциональной подменой массивов.
        ``None`` для любого массива означает «копировать существующий».
        Чтобы сбросить ``node_extra``, задайте его после ``replace``
        (см. ``select_edges``).

        Parameters
        ----------
        node_features : Optional[np.ndarray], optional
            Новая матрица признаков узлов; если None — копируется текущая.
        edge_index : Optional[torch.Tensor], optional
            Новые индексы рёбер; если None — копируется текущий.
        edge_attr : Optional[np.ndarray], optional
            Новые веса рёбер; если None — копируется текущий.
        node_extra : Optional[np.ndarray], optional
            Новые доп. признаки узлов; если None — копируется текущий.

        Returns
        -------
        WeightedFullGraphDataset
            Новый датасет с теми же метками, sample_ids и folds.
        """
        new_node_features = (
            node_features
            if node_features is not None
            else self.node_features.copy()
        )
        new_edge_index = (
            edge_index
            if edge_index is not None
            else self.edge_index.clone()
        )
        new_edge_attr = (
            edge_attr if edge_attr is not None else self.edge_attr.copy()
        )
        if node_extra is not None:
            new_node_extra: Optional[np.ndarray] = node_extra
        elif self.node_extra is not None:
            new_node_extra = self.node_extra.copy()
        else:
            new_node_extra = None

        out = WeightedFullGraphDataset(
            node_features=new_node_features,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            labels=self.labels.copy(),
            sample_ids=list(self.sample_ids),
            folds=None if self.folds is None else self.folds.copy(),
            node_extra=new_node_extra,
            include_raw_node_feature=self.include_raw_node_feature,
        )
        if self.groups is not None:
            out.groups = self.groups.copy()
        return out

    def select_nodes(self, keep: np.ndarray) -> "WeightedFullGraphDataset":
        """
        Select nodes.

        Фильтрует признаки узлов, переиндексирует ``edge_index``,
        удаляет рёбра к убранным узлам и фильтрует ``edge_attr``.

        Parameters
        ----------
        keep : np.ndarray
            Индексы оставляемых узлов (в исходной нумерации).

        Returns
        -------
        WeightedFullGraphDataset
            Новый датасет с уменьшенным числом узлов.
        """
        keep = np.asarray(keep, dtype=np.int64)
        if keep.ndim != 1 or keep.size == 0:
            raise ValueError("keep должен быть непустым 1D массивом индексов узлов")

        n_old = self.node_features.shape[1]
        if keep.min() < 0 or keep.max() >= n_old:
            raise ValueError(
                f"keep вне диапазона [0, {n_old}): "
                f"min={keep.min()}, max={keep.max()}"
            )

        # old_id -> new_id; -1 для удалённых.
        mapping = np.full(n_old, -1, dtype=np.int64)
        mapping[keep] = np.arange(len(keep), dtype=np.int64)

        src = self.edge_index[0].detach().cpu().numpy()
        dst = self.edge_index[1].detach().cpu().numpy()
        edge_keep = (mapping[src] >= 0) & (mapping[dst] >= 0)

        new_src = mapping[src[edge_keep]]
        new_dst = mapping[dst[edge_keep]]
        new_edge_index = torch.from_numpy(
            np.stack([new_src, new_dst], axis=0)
        ).long()
        new_edge_attr = self.edge_attr[edge_keep]
        new_node_features = self.node_features[:, keep]
        new_node_extra = (
            None if self.node_extra is None else self.node_extra[:, keep, :]
        )

        return self.replace(
            node_features=new_node_features,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            node_extra=new_node_extra,
        )

    def select_edges(self, edge_mask: np.ndarray) -> "WeightedFullGraphDataset":
        """
        Select edges.

        Выбирает рёбра по маске. Ожидается, что маска согласована
        для обоих направлений одной неориентированной пары.
        ``node_extra`` сбрасывается: агрегаты зависят от набора рёбер
        и пересчитываются в ``prepare_split``.

        Parameters
        ----------
        edge_mask : np.ndarray
            Булева маска длины ``E``.

        Returns
        -------
        WeightedFullGraphDataset
            Датасет с подмножеством рёбер.
        """
        edge_mask = np.asarray(edge_mask, dtype=bool)
        if edge_mask.shape[0] != self.edge_index.size(1):
            raise ValueError(
                f"edge_mask длины {edge_mask.shape[0]} не совпадает "
                f"с числом рёбер {self.edge_index.size(1)}"
            )
        if not edge_mask.any():
            raise ValueError("edge_mask не оставляет ни одного ребра")

        new_edge_index = self.edge_index[:, edge_mask]
        new_edge_attr = self.edge_attr[edge_mask]
        out = self.replace(edge_index=new_edge_index, edge_attr=new_edge_attr)
        # Агрегаты рёбер невалидны после смены топологии.
        out.node_extra = None
        out.include_raw_node_feature = True
        return out

    def len(self) -> int:
        """
        Get dataset length.

        Возвращает число графовых образцов.
        """
        return len(self.labels)

    def get(self, idx: int) -> Data:
        """
        Get graph sample.

        Формирует один объект ``Data`` для PyG.

        Parameters
        ----------
        idx : int
            Индекс образца в датасете.

        Returns
        -------
        Data
            Граф с признаками узлов, весами рёбер, меткой и индексом образца.
        """
        if self.include_raw_node_feature:
            raw = (
                torch.from_numpy(self.node_features[idx])
                .float()
                .unsqueeze(-1)
            )
            if self.node_extra is not None:
                extra = torch.from_numpy(self.node_extra[idx]).float()
                x = torch.cat([raw, extra], dim=-1)
            else:
                x = raw
        else:
            if self.node_extra is None:
                raise RuntimeError(
                    "include_raw_node_feature=false, но node_extra is None"
                )
            x = torch.from_numpy(self.node_extra[idx]).float()

        edge_attr = torch.from_numpy(self.edge_attr[:, idx]).float().unsqueeze(-1)

        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
        )
        data.y = torch.tensor([float(self.labels[idx])], dtype=torch.float)
        data.sample_idx = torch.tensor([idx], dtype=torch.long)

        if self.folds is not None:
            data.fold = torch.tensor([int(self.folds[idx])], dtype=torch.long)

        return data


class CFMSRDataModule:
    """
    CF-MSR Data Module.

    Модуль подготовки данных: таблица признаков + csv-файлы графов.
    Загружает табличный датасет и графы, строит PyG-датасеты,
    выполняет train/val-сплит, нормализацию признаков узлов
    и предоставляет ``DataLoader``-ы для обучения и оценки.
    """

    def __init__(
            self,
            data_cfg: DictConfig,
            training_cfg: DictConfig,
    ) -> None:
        """
        Initialize data module.

        Инициализирует модуль данных из веток Hydra-конфига.

        Parameters
        ----------
        data_cfg : DictConfig
            Ветка ``data`` конфигурации: пути, признаки, сплит.
        training_cfg : DictConfig
            Ветка ``training`` конфигурации: seed и прочие параметры обучения.
        """
        self.data_cfg = data_cfg
        self.training_cfg = training_cfg

        # Разрешение путей к файлам данных
        self.data_csv = resolve_path(data_cfg.data_csv)
        self.train_valid_graph_csv = resolve_path(data_cfg.train_valid_graphs_csv)
        self.holdout_graph_csv = resolve_path(data_cfg.get("holdout_graphs_csv", None))

        # Параметры чтения CSV и целевой переменной
        self.separator = str(data_cfg.get("separator", ","))
        self.target_col = str(data_cfg.target_col)
        self.positive_label = parse_optional_str(data_cfg.get("positive_label", None))

        # Параметры сплита и загрузки данных
        self.batch_size = int(data_cfg.batch_size)
        self.val_ratio = float(data_cfg.val_ratio)
        self.val_fold = parse_optional_int(data_cfg.get("val_fold", None))
        self.seed = int(training_cfg.seed)
        self.data_row_shift = int(data_cfg.data_row_shift)
        self.num_workers = int(data_cfg.num_workers)

        # Валидация обязательных путей
        if self.data_csv is None:
            raise ValueError("data_csv не задан в конфигурации data")
        if self.train_valid_graph_csv is None:
            raise ValueError("train_valid_graphs_csv не задан в конфигурации data")

        # Загрузка табличных данных
        self.data_df = pd.read_csv(self.data_csv, sep=self.separator)
        self.data_df.columns = [str(c).strip() for c in self.data_df.columns]

        # Определение признаковых колонок
        feature_cols = parse_feature_cols(data_cfg)
        if feature_cols is None:
            exclude = {self.target_col}
            # Служебные колонки, которые не являются признаками узлов
            service_cols = {"id", "row", "index", "unnamed: 0"}
            feature_cols = [
                c
                for c in self.data_df.columns
                if c not in exclude and str(c).strip().lower() not in service_cols
            ]
        else:
            feature_cols = [c for c in feature_cols if c != self.target_col]

        if len(feature_cols) == 0:
            raise ValueError("Не найдено ни одного признакового столбца в data.csv")

        self._all_feature_cols = [str(c).strip() for c in feature_cols]
        self.feature_cols = list(self._all_feature_cols)

        # Приведение признаков к числовому виду
        self.data_df[self.feature_cols] = self.data_df[self.feature_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        self._all_num_nodes = len(self.feature_cols)
        self.num_nodes = self._all_num_nodes
        self.node_map = build_node_map(self.feature_cols)
        self.labels_encoded = self._encode_target()

        # Инициализация атрибутов для датасетов и статистик
        self._raw_train_valid: Optional[WeightedFullGraphDataset] = None
        self._raw_test: Optional[WeightedFullGraphDataset] = None
        self.train_valid_dataset: Optional[WeightedFullGraphDataset] = None
        self.test_dataset: Optional[WeightedFullGraphDataset] = None
        self.train_idx: Optional[np.ndarray] = None
        self.val_idx: Optional[np.ndarray] = None
        self.node_mean: Optional[np.ndarray] = None
        self.node_std: Optional[np.ndarray] = None
        self.extra_mean: Optional[np.ndarray] = None
        self.extra_std: Optional[np.ndarray] = None
        self.edge_mean: Optional[Any] = None
        self.edge_std: Optional[Any] = None
        self.pos_weight = 1.0
        self.kept_nodes: Optional[np.ndarray] = None
        self.edge_mask: Optional[np.ndarray] = None
        self._is_setup = False
        self.group_col = parse_optional_str(data_cfg.get("group_col", None))

        # Признаки узлов из инцидентных рёбер (Ш1).
        enf = data_cfg.get("edge_node_features", None)
        if enf is None:
            self.edge_node_enabled = False
            self.edge_node_stats: List[str] = ["mean"]
            self.edge_node_include_raw = True
        else:
            self.edge_node_enabled = bool(enf.get("enabled", False))
            stats_raw = enf.get("stats", ["mean"])
            self.edge_node_stats = [str(s) for s in list(stats_raw)]
            self.edge_node_include_raw = bool(enf.get("include_raw", True))

        if (not self.edge_node_enabled) and (not self.edge_node_include_raw):
            raise ValueError(
                "edge_node_features.include_raw=false при enabled=false "
                "даёт пустой x; включите enabled или include_raw"
            )

        # Нормировка рёбер по train (Ш2).
        self.edge_scaler_mode = str(
            data_cfg.get("edge_scaler", "none")
        ).strip().lower()

    def _encode_target(self) -> np.ndarray:
        """
        Encode target variable.

        Кодирует целевую переменную в бинарные метки ``0.0``/``1.0``.

        Returns
        -------
        np.ndarray
            Массив меток формы ``(num_rows,)``; ``NaN`` — отсутствующая метка.
        """
        if self.target_col not in self.data_df.columns:
            warnings.warn(
                f"Колонка таргета '{self.target_col}' не найдена. "
                "Все метки будут NaN. Обучение на train_valid будет невозможно."
            )
            return np.full(len(self.data_df), np.nan, dtype=np.float32)

        s = self.data_df[self.target_col]

        # Обработка явного указания положительного класса
        if self.positive_label is not None:
            s_num = pd.to_numeric(s, errors="coerce")
            pos_num = pd.to_numeric(
                pd.Series([self.positive_label]),
                errors="coerce",
            ).iloc[0]

            if not pd.isna(pos_num) and s_num.notna().sum() == s.notna().sum():
                arr = np.full(len(s), np.nan, dtype=np.float32)
                notna = s_num.notna().to_numpy()
                arr[notna] = 0.0
                arr[(s_num == pos_num).to_numpy()] = 1.0
                return arr

            s_str = s.astype("string").str.strip()
            arr = np.full(len(s), np.nan, dtype=np.float32)
            notna = s_str.notna().to_numpy()
            arr[notna] = 0.0
            arr[s_str.isin([str(self.positive_label)]).to_numpy()] = 1.0
            return arr

        # Автоматическое кодирование числового таргета
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() == s.notna().sum():
            uniq = sorted(s_num.dropna().unique().tolist())
            if len(uniq) == 0:
                return np.full(len(s), np.nan, dtype=np.float32)
            if set(uniq).issubset({0.0, 1.0}):
                return s_num.astype(np.float32).to_numpy()
            if len(uniq) == 2:
                warnings.warn(
                    f"Таргет числовой, но не 0/1. Уникальные значения: {uniq}. "
                    f"Будет отображено: {uniq[0]} -> 0, {uniq[1]} -> 1. "
                    "Если это неверно, задайте positive_label."
                )
                mapping = {uniq[0]: 0.0, uniq[1]: 1.0}
                return s_num.map(mapping).astype(np.float32).to_numpy()

        # Автоматическое кодирование строкового таргета
        s_str = s.astype("string").str.strip()
        uniq = sorted(s_str.dropna().unique().tolist())

        if len(uniq) == 0:
            return np.full(len(s), np.nan, dtype=np.float32)

        if len(uniq) == 1:
            warnings.warn(
                f"В таргете только один уникальный класс: {uniq[0]}. "
                "Он будет отображен в 0. Бинарное обучение будет вырожденным."
            )
            arr = np.full(len(s), np.nan, dtype=np.float32)
            arr[s_str.notna().to_numpy()] = 0.0
            return arr

        if len(uniq) == 2:
            warnings.warn(
                f"Найдено 2 класса: {uniq}. "
                f"Будет отображено: {uniq[0]} -> 0, {uniq[1]} -> 1. "
                "Если это неверно, задайте positive_label."
            )
            mapping = {uniq[0]: 0.0, uniq[1]: 1.0}
            return s_str.map(mapping).astype(np.float32).to_numpy()

        raise ValueError(
            f"Таргет '{self.target_col}' содержит более 2 уникальных значений: {uniq}. "
            "Для бинарной классификации задайте positive_label."
        )

    def _resolve_data_index(
            self,
            sample_num: int,
            require_target: bool = False,
    ) -> Optional[int]:
        """
        Resolve data index.

        Находит индекс строки в табличном датасете по номеру объекта.
        Сначала проверяется основное смещение ``data_row_shift``, затем
        запасные варианты на случай расхождения нумерации.

        Parameters
        ----------
        sample_num : int
            Номер объекта из имени sample-колонки графа.
        require_target : bool, optional
            Если ``True``, возвращает индекс только при наличии метки.

        Returns
        -------
        Optional[int]
            Индекс строки в ``data_df`` или ``None``.
        """
        base = sample_num - self.data_row_shift

        # Кандидаты индексов: основной и запасные эвристики
        candidates = [
            base,
            sample_num - 1,
            sample_num - 3,
            sample_num - 4,
            sample_num,
        ]

        # Убираем дубликаты, сохраняя порядок
        seen = set()
        ordered = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)

        for idx in ordered:
            if 0 <= idx < len(self.data_df):
                if require_target:
                    label = self.labels_encoded[idx]
                    if not np.isnan(label):
                        return idx
                else:
                    return idx

        return None

    def _label_at(self, idx: int) -> Optional[float]:
        """
        Get label at index.

        Возвращает метку объекта по индексу строки в табличном датасете.

        Parameters
        ----------
        idx : int
            Индекс строки в ``data_df``.

        Returns
        -------
        Optional[float]
            Метка ``0.0``/``1.0`` или ``None``, если метка отсутствует.
        """
        val = self.labels_encoded[idx]
        if np.isnan(val):
            return None
        return float(val)

    def _build_edges(
            self,
            graph_df: pd.DataFrame,
    ) -> Tuple[torch.Tensor, List[List[int]], set]:
        """
        Build graph edges.

        Строит ориентированный ``edge_index`` и группы строк для каждого ребра.
        Несколько строк csv с одной парой узлов объединяются в одно ребро;
        веса таких строк усредняются при формировании ``edge_attr``.

        Parameters
        ----------
        graph_df : pd.DataFrame
            Таблица рёбер с колонками ``p1``, ``p2`` и sample-колонками.

        Returns
        -------
        Tuple[torch.Tensor, List[List[int]], set]
            ``edge_index``, список групп исходных строк на ребро
            и множество несопоставленных имён узлов.
        """
        p1_idx = graph_df["p1"].apply(lambda x: resolve_node(x, self.node_map))
        p2_idx = graph_df["p2"].apply(lambda x: resolve_node(x, self.node_map))

        unmapped = set()
        pairs_to_rows: Dict[Tuple[int, int], List[int]] = {}

        # Группировка строк по парам узлов
        for pos in range(len(graph_df)):
            u = p1_idx.iat[pos]
            v = p2_idx.iat[pos]

            if u is None or pd.isna(u):
                unmapped.add(str(graph_df["p1"].iat[pos]).strip())
                continue
            if v is None or pd.isna(v):
                unmapped.add(str(graph_df["p2"].iat[pos]).strip())
                continue

            u, v = int(u), int(v)

            # Неориентированная пара: (min, max) для группировки дубликатов
            if u == v:
                key = (u, v)
            else:
                key = (min(u, v), max(u, v))

            pairs_to_rows.setdefault(key, []).append(pos)

        directed_edges: List[Tuple[int, int]] = []
        edge_groups: List[List[int]] = []

        # Формирование направленных рёбер (дублирование для неориентированного графа)
        for (u, v), rows in pairs_to_rows.items():
            directed_edges.append((u, v))
            edge_groups.append(rows)
            if u != v:
                directed_edges.append((v, u))
                edge_groups.append(rows)

        # Fallback: полный граф без петель, если рёбра не распознаны
        if len(directed_edges) == 0:
            warnings.warn(
                "Не удалось распарсить ни одного ребра. "
                "Создается полный ориентированный граф без петель с нулевыми весами."
            )
            for u in range(self.num_nodes):
                for v in range(self.num_nodes):
                    if u != v:
                        directed_edges.append((u, v))
                        edge_groups.append([])

        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        return edge_index, edge_groups, unmapped

    def _build_dataset(
            self,
            graph_csv: str,
            require_target: bool,
    ) -> WeightedFullGraphDataset:
        """
        Build weighted dataset.

        Собирает ``WeightedFullGraphDataset`` из csv-файла графа.

        Parameters
        ----------
        graph_csv : str
            Путь к csv-файлу с колонками ``p1``, ``p2`` и sample-колонками.
        require_target : bool
            Если ``True``, пропускает образцы без метки (для train/valid).

        Returns
        -------
        WeightedFullGraphDataset
            Готовый датасет графовых образцов.
        """
        graph_df = pd.read_csv(graph_csv, sep=self.separator)
        graph_df.columns = [str(c).strip() for c in graph_df.columns]

        if "p1" not in graph_df.columns or "p2" not in graph_df.columns:
            raise ValueError(f"В файле {graph_csv} должны быть колонки p1 и p2")

        sample_cols: List[str] = []
        sample_meta: List[Tuple[str, int, int, float]] = []
        ignored_cols: List[str] = []

        # Парсинг sample-колонок и сопоставление с табличными данными
        for col in graph_df.columns:
            if col in {"p1", "p2"}:
                continue

            col_clean = str(col).strip()
            parsed = parse_sample_column(col_clean)

            if parsed is None:
                ignored_cols.append(col_clean)
                continue

            sample_num, fold = parsed
            expected_idx = sample_num - self.data_row_shift

            data_idx = self._resolve_data_index(
                sample_num,
                require_target=require_target,
            )

            if data_idx is None:
                warnings.warn(
                    f"Не удалось найти строку в data.csv для колонки '{col_clean}' "
                    f"(sample_num={sample_num}, data_row_shift={self.data_row_shift}). "
                    "Колонка пропущена."
                )
                continue

            if data_idx != expected_idx:
                warnings.warn(
                    f"Для колонки '{col_clean}' использован index={data_idx}, "
                    f"хотя ожидаемый index при data_row_shift={self.data_row_shift}: {expected_idx}. "
                    "Проверьте соответствие нумерации строк."
                )

            label = self._label_at(data_idx)
            if require_target and label is None:
                warnings.warn(
                    f"Для колонки '{col_clean}' найден индекс data.csv={data_idx}, "
                    "но таргет отсутствует. Колонка пропущена."
                )
                continue

            if label is None:
                # Holdout-образцы без метки получают маркер -1.0
                label = -1.0

            sample_cols.append(col)
            sample_meta.append((col_clean, data_idx, fold, float(label)))

        if len(ignored_cols) > 0:
            warnings.warn(
                f"В {graph_csv} проигнорированы {len(ignored_cols)} колонок, "
                f"не похожих на sample-колонки (формат '###' или '###_...'). "
                f"Примеры: {ignored_cols[:5]}"
            )

        if len(sample_cols) == 0:
            raise ValueError(
                f"В файле {graph_csv} не найдено ни одной валидной sample-колонки "
                f"(формат '###' или '###_...', где ### — номер объекта в data.csv)"
            )

        edge_index, edge_groups, unmapped = self._build_edges(graph_df)

        if len(unmapped) > 0:
            warnings.warn(
                f"В {graph_csv} найдено {len(unmapped)} уникальных имен узлов, "
                f"которые не сопоставлены с feature_cols. Примеры: {list(unmapped)[:10]}"
            )

        # Извлечение весов рёбер и приведение к числовому виду
        weights_df = graph_df[sample_cols].apply(pd.to_numeric, errors="coerce")
        weights = weights_df.to_numpy(dtype=np.float32)

        num_edges = edge_index.size(1)
        num_samples = len(sample_cols)
        edge_attr = np.zeros((num_edges, num_samples), dtype=np.float32)

        # Агрегация весов для каждого ребра (усреднение дубликатов)
        for e, rows in enumerate(edge_groups):
            if len(rows) == 0:
                continue
            if len(rows) == 1:
                vals = weights[rows[0], :]
            else:
                with np.errstate(all="ignore"):
                    vals = np.nanmean(weights[rows, :], axis=0)

            edge_attr[e, :] = np.nan_to_num(
                vals,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        # Формирование массивов меток, fold и индексов
        data_indices = [m[1] for m in sample_meta]
        labels = np.array([m[3] for m in sample_meta], dtype=np.float32)
        folds = np.array([m[2] for m in sample_meta], dtype=np.int64)
        sample_ids = [m[0] for m in sample_meta]

        # Извлечение признаков узлов для выбранных образцов
        node_features = self.data_df.iloc[data_indices][self.feature_cols].to_numpy(
            dtype=np.float32
        )
        node_features = np.nan_to_num(
            node_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        dataset = WeightedFullGraphDataset(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            labels=labels,
            sample_ids=sample_ids,
            folds=folds,
        )

        if self.group_col is not None:
            if self.group_col not in self.data_df.columns:
                raise ValueError(
                    f"data.group_col={self.group_col!r} отсутствует в data.csv"
                )
            groups_series = self.data_df.iloc[data_indices][self.group_col]
            # Кодируем группы в int64 для StratifiedGroupKFold.
            codes, _ = pd.factorize(groups_series.astype("string"), sort=False)
            dataset.groups = np.asarray(codes, dtype=np.int64)

        return dataset

    def _load_raw(self) -> None:
        """
        Load raw datasets.

        Загружает сырые train/valid и holdout датасеты без нормализации
        и без отбора признаков.
        """
        self._raw_train_valid = self._build_dataset(
            self.train_valid_graph_csv,
            require_target=True,
        )

        if self.holdout_graph_csv and os.path.exists(self.holdout_graph_csv):
            self._raw_test = self._build_dataset(
                self.holdout_graph_csv,
                require_target=False,
            )
        else:
            self._raw_test = None

    def prepare_split(
            self,
            train_idx: np.ndarray,
            val_idx: np.ndarray,
    ) -> None:
        """
        Prepare split-dependent data.

        На копиях сырых данных выполняет отбор признаков/рёбер (если задан),
        нормализацию узлов и расчёт ``pos_weight``.

        Parameters
        ----------
        train_idx : np.ndarray
            Индексы обучающей выборки.
        val_idx : np.ndarray
            Индексы валидационной выборки.
        """
        if self._raw_train_valid is None:
            raise RuntimeError("Сначала вызовите setup() или _load_raw()")

        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.val_idx = np.asarray(val_idx, dtype=np.int64)

        # Сброс размерности к полному набору признаков (важно для CV).
        self.feature_cols = list(self._all_feature_cols)
        self.num_nodes = self._all_num_nodes
        self.node_map = build_node_map(self.feature_cols)

        # Копии сырых данных: исходные массивы не изменяются.
        prepared = self._raw_train_valid.replace()
        prepared_test = (
            None if self._raw_test is None else self._raw_test.replace()
        )

        # Отбор узлов/рёбер (п4): по умолчанию method=none — no-op.
        keep_nodes = self._select_nodes(prepared, self.train_idx)
        if keep_nodes is not None:
            prepared = prepared.select_nodes(keep_nodes)
            if prepared_test is not None:
                prepared_test = prepared_test.select_nodes(keep_nodes)
            self.kept_nodes = keep_nodes
            self.num_nodes = int(keep_nodes.shape[0])
            self.feature_cols = [
                self._all_feature_cols[i] for i in keep_nodes.tolist()
            ]
            self.node_map = build_node_map(self.feature_cols)
        else:
            self.kept_nodes = None

        edge_mask = self._select_edges(prepared, self.train_idx)
        if edge_mask is not None:
            prepared = prepared.select_edges(edge_mask)
            if prepared_test is not None:
                prepared_test = prepared_test.select_edges(edge_mask)
            self.edge_mask = edge_mask
        else:
            self.edge_mask = None

        # Порядок: edge scaler → edge_node_features.
        # Агрегаты Ш1 считаются по нормированным рёбрам, затем
        # стандартизуются собственным scaler'ом. Для интерпретируемости
        # агрегатов в единицах вероятности поменять шаги местами.
        self._fit_edge_scaler(prepared, self.train_idx)
        prepared = self._apply_edge_scaler(prepared)
        if prepared_test is not None:
            prepared_test = self._apply_edge_scaler(prepared_test)

        prepared = self._add_edge_node_features(prepared)
        if prepared_test is not None:
            prepared_test = self._add_edge_node_features(prepared_test)

        self._fit_node_scaler(prepared, self.train_idx)
        prepared = self._apply_node_scaler(prepared)
        if prepared_test is not None:
            prepared_test = self._apply_node_scaler(prepared_test)

        self.train_valid_dataset = prepared
        self.test_dataset = prepared_test
        self._compute_pos_weight()

    def setup(self) -> None:
        """
        Setup datasets.

        Загружает сырые данные, делает дефолтный train/val-сплит
        и вызывает ``prepare_split``. Повторный вызов безопасен.
        """
        if self._is_setup:
            return

        self._load_raw()
        assert self._raw_train_valid is not None
        self.train_idx, self.val_idx = self._split_train_val(
            self._raw_train_valid
        )
        self.prepare_split(self.train_idx, self.val_idx)
        self._is_setup = True

    @property
    def node_input_dim(self) -> int:
        """
        Node input dimension.

        Делегирует в ``train_valid_dataset.node_input_dim``.

        Returns
        -------
        int
            Размерность ``x`` на узел.
        """
        if self.train_valid_dataset is None:
            return 1
        return self.train_valid_dataset.node_input_dim

    def _add_edge_node_features(
            self,
            dataset: WeightedFullGraphDataset,
    ) -> WeightedFullGraphDataset:
        """
        Add edge-derived node features.

        Считает агрегаты инцидентных рёбер и выставляет ``include_raw``.

        Parameters
        ----------
        dataset : WeightedFullGraphDataset
            Датасет после отбора узлов/рёбер и edge scaler.

        Returns
        -------
        WeightedFullGraphDataset
            Датасет с ``node_extra`` (если enabled) и флагом include_raw.
        """
        if not self.edge_node_enabled:
            out = dataset.replace()
            out.include_raw_node_feature = True
            out.node_extra = None
            return out

        edge_index = dataset.edge_index.detach().cpu().numpy()
        num_nodes = int(dataset.node_features.shape[1])
        node_extra = compute_edge_node_features(
            edge_index=edge_index,
            edge_attr=dataset.edge_attr,
            num_nodes=num_nodes,
            stats=self.edge_node_stats,
        )
        out = dataset.replace(node_extra=node_extra)
        out.include_raw_node_feature = self.edge_node_include_raw
        return out

    def _select_nodes(
            self,
            dataset: WeightedFullGraphDataset,
            train_idx: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Select node features.

        Считает оценки только по ``train_idx`` и возвращает индексы
        оставляемых узлов или None, если отбор отключён.

        Parameters
        ----------
        dataset : WeightedFullGraphDataset
            Датасет для оценки статистик отбора.
        train_idx : np.ndarray
            Индексы train (только по ним считаются статистики).

        Returns
        -------
        Optional[np.ndarray]
            Индексы узлов или None.
        """
        fs = self.data_cfg.get("feature_selection", None)
        if fs is None:
            return None
        method = str(fs.get("method", "none")).strip().lower()
        if method in ("", "none", "null"):
            return None

        top_k = fs.get("top_k", None)
        top_k = None if top_k in (None, "null") else int(top_k)
        n_features = dataset.node_features.shape[1]
        if top_k is not None and top_k > n_features:
            raise ValueError(
                f"feature_selection.top_k={top_k} больше числа признаков "
                f"{n_features}"
            )

        x_train = dataset.node_features[train_idx]
        y_train = dataset.labels[train_idx]
        valid_y = np.isfinite(y_train) & (y_train >= 0.0) & (y_train <= 1.0)
        x_train = x_train[valid_y]
        y_train = y_train[valid_y].astype(int)

        if method == "variance":
            scores = np.nanvar(x_train, axis=0)
        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif

            scores = mutual_info_classif(
                np.nan_to_num(x_train),
                y_train,
                random_state=self.seed,
            )
        elif method == "l1":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(np.nan_to_num(x_train))
            clf = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                random_state=self.seed,
                max_iter=2000,
            )
            clf.fit(x_scaled, y_train)
            scores = np.abs(clf.coef_).ravel()
        else:
            raise ValueError(
                f"Неизвестный feature_selection.method={method!r}; "
                "ожидается none | variance | mutual_info | l1"
            )

        scores = np.nan_to_num(scores, nan=-np.inf)
        if top_k is None:
            # Оставляем признаки со score > 0 (для l1/variance/mi).
            keep = np.where(scores > 0)[0]
            if keep.size == 0:
                keep = np.argsort(scores)[::-1][: max(1, n_features // 2)]
        else:
            keep = np.argsort(scores)[::-1][:top_k]

        keep = np.sort(keep.astype(np.int64))
        return keep

    def _select_edges(
            self,
            dataset: WeightedFullGraphDataset,
            train_idx: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Select edges.

        Считает оценки только по ``train_idx``. Для неориентированной пары
        оценка одна; в маску попадают оба направления.

        Parameters
        ----------
        dataset : WeightedFullGraphDataset
            Датасет для оценки статистик отбора.
        train_idx : np.ndarray
            Индексы train (только по ним считаются статистики).

        Returns
        -------
        Optional[np.ndarray]
            Булева маска рёбер или None.
        """
        es = self.data_cfg.get("edge_selection", None)
        if es is None:
            return None
        method = str(es.get("method", "none")).strip().lower()
        if method in ("", "none", "null"):
            return None

        top_k = es.get("top_k", None)
        keep_ratio = es.get("keep_ratio", None)
        top_k = None if top_k in (None, "null") else int(top_k)
        keep_ratio = (
            None if keep_ratio in (None, "null") else float(keep_ratio)
        )
        if top_k is not None and keep_ratio is not None:
            raise ValueError(
                "edge_selection: нельзя задавать top_k и keep_ratio одновременно"
            )

        edge_index = dataset.edge_index.detach().cpu().numpy()
        src, dst = edge_index[0], edge_index[1]
        e = edge_index.shape[1]

        # Неориентированные пары: канонический ключ (min, max).
        a = np.minimum(src, dst)
        b = np.maximum(src, dst)
        # Уникальные undirected рёбра.
        undirected_keys = np.stack([a, b], axis=1)
        # Индексы первого вхождения каждой пары.
        _, first_idx, inv = np.unique(
            undirected_keys,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        n_undirected = len(first_idx)

        if top_k is not None and top_k > n_undirected:
            raise ValueError(
                f"edge_selection.top_k={top_k} больше числа undirected рёбер "
                f"{n_undirected}"
            )

        # Признаки рёбер по train: (E, n_train)
        w_train = dataset.edge_attr[:, train_idx]
        y_train = dataset.labels[train_idx]
        valid_y = np.isfinite(y_train) & (y_train >= 0.0) & (y_train <= 1.0)
        w_train = w_train[:, valid_y]
        y_train = y_train[valid_y].astype(int)

        # Агрегируем веса двух направлений в undirected-признак.
        undirected_feat = np.zeros((n_undirected, w_train.shape[1]), dtype=np.float32)
        for e_i in range(e):
            u_i = inv[e_i]
            undirected_feat[u_i] += w_train[e_i]
        # Усреднение по числу направлений (обычно 2).
        counts = np.bincount(inv, minlength=n_undirected).astype(np.float32)
        undirected_feat = undirected_feat / np.maximum(counts[:, None], 1.0)

        if method == "variance":
            scores = np.nanvar(undirected_feat, axis=1)
        elif method == "abs_mean":
            scores = np.nanmean(np.abs(undirected_feat), axis=1)
        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif

            scores = mutual_info_classif(
                np.nan_to_num(undirected_feat.T),
                y_train,
                random_state=self.seed,
            )
            # mutual_info_classif ожидает (n_samples, n_features);
            # undirected_feat.T → (n_train, n_undirected) ✓
        else:
            raise ValueError(
                f"Неизвестный edge_selection.method={method!r}; "
                "ожидается none | variance | abs_mean | mutual_info"
            )

        scores = np.nan_to_num(scores, nan=-np.inf)
        order = np.argsort(scores)[::-1]

        if top_k is not None:
            keep_u = order[:top_k]
        elif keep_ratio is not None:
            if not 0.0 < keep_ratio <= 1.0:
                raise ValueError(
                    f"keep_ratio должен быть в (0, 1], получено {keep_ratio}"
                )
            k = max(1, int(round(n_undirected * keep_ratio)))
            keep_u = order[:k]
        else:
            keep_u = order[scores[order] > 0]
            if keep_u.size == 0:
                keep_u = order[: max(1, n_undirected // 2)]

        keep_u_set = set(keep_u.tolist())
        edge_mask = np.array([inv[i] in keep_u_set for i in range(e)], dtype=bool)
        return edge_mask

    def _split_train_val(
            self,
            dataset: WeightedFullGraphDataset,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split train and validation sets.

        Делит train/valid датасет на обучающую и валидационную выборки.
        Приоритет: сплит по ``val_fold``, затем стратифицированный sklearn,
        затем случайный сплит с фиксированным seed.

        Parameters
        ----------
        dataset : WeightedFullGraphDataset
            Датасет train/valid с заполненным полем ``folds``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Пара массивов индексов ``(train_idx, val_idx)``.
        """
        n = len(dataset)
        indices = np.arange(n)

        # Попытка сплита по заранее заданным fold
        if self.val_fold is not None and dataset.folds is not None:
            folds = dataset.folds
            val_idx = np.where(folds == self.val_fold)[0]
            train_idx = np.where(folds != self.val_fold)[0]
            if len(val_idx) > 0 and len(train_idx) > 0:
                return train_idx, val_idx
            warnings.warn(
                f"Сплит по val_fold={self.val_fold} не удался. "
                "Будет использован случайный сплит."
            )

        if self.val_ratio <= 0:
            return indices, np.array([], dtype=np.int64)

        labels = dataset.labels

        # Групповой сплит, если задан group_col и groups заполнены.
        if self.group_col is not None and getattr(dataset, "groups", None) is not None:
            try:
                from sklearn.model_selection import GroupShuffleSplit

                gss = GroupShuffleSplit(
                    n_splits=1,
                    test_size=self.val_ratio,
                    random_state=self.seed,
                )
                train_idx, val_idx = next(
                    gss.split(indices, labels, groups=dataset.groups)
                )
                return np.asarray(train_idx), np.asarray(val_idx)
            except Exception as e:
                warnings.warn(
                    f"Групповой сплит не удался: {e}. "
                    "Будет использован стратифицированный сплит."
                )

        # Попытка стратифицированного сплита через sklearn
        try:
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(
                indices,
                test_size=self.val_ratio,
                random_state=self.seed,
                stratify=labels.astype(int),
            )
            return np.asarray(train_idx), np.asarray(val_idx)
        except Exception as e:
            warnings.warn(
                f"Стратифицированный sklearn-сплит не удался: {e}. "
                "Будет использован случайный сплит."
            )

        # Fallback: случайный сплит
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        n_val = int(round(n * self.val_ratio))

        if n > 1:
            n_val = max(1, min(n_val, n - 1))
        else:
            n_val = 0

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        return train_idx, val_idx

    def _fit_node_scaler(
            self,
            dataset: WeightedFullGraphDataset,
            train_idx: np.ndarray,
    ) -> None:
        """
        Fit node scaler.

        Оценивает mean/std признаков узлов (и ``node_extra``) по train.

        Parameters
        ----------
        dataset : WeightedFullGraphDataset
            Датасет с матрицей ``node_features``.
        train_idx : np.ndarray
            Индексы образцов, используемых для оценки статистик.
        """
        if len(train_idx) == 0:
            self.node_mean = np.zeros(self.num_nodes, dtype=np.float32)
            self.node_std = np.ones(self.num_nodes, dtype=np.float32)
            if dataset.node_extra is not None:
                f_extra = int(dataset.node_extra.shape[-1])
                self.extra_mean = np.zeros(
                    (self.num_nodes, f_extra), dtype=np.float32,
                )
                self.extra_std = np.ones(
                    (self.num_nodes, f_extra), dtype=np.float32,
                )
            else:
                self.extra_mean = None
                self.extra_std = None
            return

        train_feats = dataset.node_features[train_idx]

        with np.errstate(all="ignore"):
            self.node_mean = np.nanmean(train_feats, axis=0)
            self.node_std = np.nanstd(train_feats, axis=0)

        # Защита от NaN и бесконечностей, а также от нулевого std
        self.node_mean = np.nan_to_num(
            self.node_mean, nan=0.0, posinf=0.0, neginf=0.0,
        )
        self.node_std = np.nan_to_num(
            self.node_std, nan=1.0, posinf=1.0, neginf=1.0,
        )
        self.node_std[self.node_std < 1e-6] = 1.0

        if dataset.node_extra is not None:
            train_extra = dataset.node_extra[train_idx]
            with np.errstate(all="ignore"):
                self.extra_mean = np.nanmean(train_extra, axis=0)
                self.extra_std = np.nanstd(train_extra, axis=0)
            self.extra_mean = np.nan_to_num(
                self.extra_mean, nan=0.0, posinf=0.0, neginf=0.0,
            ).astype(np.float32)
            self.extra_std = np.nan_to_num(
                self.extra_std, nan=1.0, posinf=1.0, neginf=1.0,
            ).astype(np.float32)
            self.extra_std[self.extra_std < 1e-6] = 1.0
        else:
            self.extra_mean = None
            self.extra_std = None

    def _apply_node_scaler(
            self,
            dataset: WeightedFullGraphDataset,
    ) -> WeightedFullGraphDataset:
        """
        Apply node scaler.

        Возвращает новый датасет с нормализованными признаками узлов
        и ``node_extra``. Исходный датасет не изменяется.

        Parameters
        ----------
        dataset : WeightedFullGraphDataset
            Датасет, к которому применяется нормализация.

        Returns
        -------
        WeightedFullGraphDataset
            Датасет с нормализованными ``node_features`` / ``node_extra``.
        """
        if self.node_mean is None or self.node_std is None:
            return dataset.replace()

        if dataset.node_features.shape[1] != len(self.node_mean):
            warnings.warn(
                "Размерность узлов в датасете не совпадает с node_mean/node_std. "
                "Нормализация пропущена."
            )
            return dataset.replace()

        scaled = (dataset.node_features - self.node_mean) / self.node_std
        scaled = np.nan_to_num(
            scaled,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        new_extra: Optional[np.ndarray] = None
        if (
                dataset.node_extra is not None
                and self.extra_mean is not None
                and self.extra_std is not None
        ):
            if dataset.node_extra.shape[1:] != self.extra_mean.shape:
                warnings.warn(
                    "Размерность node_extra не совпадает с extra_mean/extra_std. "
                    "Нормализация node_extra пропущена."
                )
                new_extra = dataset.node_extra.copy()
            else:
                new_extra = (
                    (dataset.node_extra - self.extra_mean) / self.extra_std
                )
                new_extra = np.nan_to_num(
                    new_extra, nan=0.0, posinf=0.0, neginf=0.0,
                ).astype(np.float32)

        if new_extra is not None:
            return dataset.replace(node_features=scaled, node_extra=new_extra)
        return dataset.replace(node_features=scaled)

    def _fit_edge_scaler(
            self,
            dataset: WeightedFullGraphDataset,
            train_idx: np.ndarray,
    ) -> None:
        """
        Fit edge scaler.

        Оценивает статистики весов рёбер только по ``train_idx``.

        Parameters
        ----------
        dataset : WeightedFullGraphDataset
            Датасет с ``edge_attr``.
        train_idx : np.ndarray
            Индексы train-образцов.
        """
        mode = self.edge_scaler_mode
        if mode in ("", "none", "null"):
            self.edge_mean = None
            self.edge_std = None
            return

        if len(train_idx) == 0:
            if mode == "global":
                self.edge_mean = np.float32(0.0)
                self.edge_std = np.float32(1.0)
            elif mode == "per_edge":
                e = dataset.edge_attr.shape[0]
                self.edge_mean = np.zeros(e, dtype=np.float32)
                self.edge_std = np.ones(e, dtype=np.float32)
            else:
                raise ValueError(
                    f"Неизвестный edge_scaler={mode!r}; "
                    "ожидается none | global | per_edge"
                )
            return

        w_train = dataset.edge_attr[:, train_idx]

        if mode == "global":
            with np.errstate(all="ignore"):
                mean_v = np.nanmean(w_train)
                std_v = np.nanstd(w_train)
            mean_v = float(np.nan_to_num(mean_v, nan=0.0, posinf=0.0, neginf=0.0))
            std_v = float(np.nan_to_num(std_v, nan=1.0, posinf=1.0, neginf=1.0))
            if std_v < 1e-6:
                std_v = 1.0
            self.edge_mean = np.float32(mean_v)
            self.edge_std = np.float32(std_v)
        elif mode == "per_edge":
            with np.errstate(all="ignore"):
                self.edge_mean = np.nanmean(w_train, axis=1).astype(np.float32)
                self.edge_std = np.nanstd(w_train, axis=1).astype(np.float32)
            self.edge_mean = np.nan_to_num(
                self.edge_mean, nan=0.0, posinf=0.0, neginf=0.0,
            )
            self.edge_std = np.nan_to_num(
                self.edge_std, nan=1.0, posinf=1.0, neginf=1.0,
            )
            self.edge_std[self.edge_std < 1e-6] = 1.0
        else:
            raise ValueError(
                f"Неизвестный edge_scaler={mode!r}; "
                "ожидается none | global | per_edge"
            )

    def _apply_edge_scaler(
            self,
            dataset: WeightedFullGraphDataset,
    ) -> WeightedFullGraphDataset:
        """
        Apply edge scaler.

        Нормализует ``edge_attr`` train-статистиками.

        Parameters
        ----------
        dataset : WeightedFullGraphDataset
            Датасет для нормализации.

        Returns
        -------
        WeightedFullGraphDataset
            Датасет с нормализованными весами рёбер.
        """
        if self.edge_mean is None or self.edge_std is None:
            return dataset.replace()

        if isinstance(self.edge_mean, np.ndarray):
            if self.edge_mean.shape[0] != dataset.edge_attr.shape[0]:
                warnings.warn(
                    "Число рёбер в датасете не совпадает с edge_mean/edge_std. "
                    "Нормализация рёбер пропущена."
                )
                return dataset.replace()
            scaled = (
                (dataset.edge_attr - self.edge_mean[:, None])
                / self.edge_std[:, None]
            )
        else:
            scaled = (dataset.edge_attr - float(self.edge_mean)) / float(
                self.edge_std
            )

        scaled = np.nan_to_num(
            scaled.astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return dataset.replace(edge_attr=scaled)

    def _compute_pos_weight(self) -> None:
        """
        Compute positive weight.

        Вычисляет вес положительного класса для функции потерь.
        """
        if self.train_valid_dataset is None or self.train_idx is None:
            self.pos_weight = 1.0
            return

        train_labels = np.array(
            [self.train_valid_dataset.labels[i] for i in self.train_idx],
            dtype=np.float32,
        )
        train_labels = train_labels[
            (train_labels >= 0.0) & (train_labels <= 1.0)
            ]

        pos = float((train_labels == 1.0).sum())
        neg = float((train_labels == 0.0).sum())

        if pos > 0:
            self.pos_weight = neg / pos if neg > 0 else 1.0
        else:
            self.pos_weight = 1.0

    def train_loader(self) -> DataLoader:
        """
        Get train DataLoader.

        Возвращает DataLoader для обучающей выборки.

        Returns
        -------
        DataLoader
            Загрузчик обучающих данных.
        """
        if self.train_valid_dataset is None or self.train_idx is None:
            raise RuntimeError("Сначала вызовите setup()")

        train_dataset = self.train_valid_dataset.index_select(self.train_idx.tolist())
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_loader(self) -> Optional[DataLoader]:
        """
        Get validation DataLoader.

        Возвращает DataLoader для валидационной выборки.

        Returns
        -------
        Optional[DataLoader]
            Загрузчик валидационных данных или None, если выборка пуста.
        """
        if (
                self.train_valid_dataset is None
                or self.val_idx is None
                or len(self.val_idx) == 0
        ):
            return None

        val_dataset = self.train_valid_dataset.index_select(self.val_idx.tolist())
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_loader(self) -> Optional[DataLoader]:
        """
        Get test DataLoader.

        Возвращает DataLoader для тестовой (holdout) выборки.

        Returns
        -------
        Optional[DataLoader]
            Загрузчик тестовых данных или None, если датасет не задан.
        """
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
