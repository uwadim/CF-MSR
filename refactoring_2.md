# План 2: прямой путь сигнала рёбер в CF-MSR

Продолжение `refactoring.md` (этап 0, п1–п7 выполнены). План составлен по результатам п1 (baseline).

Рекомендуемый порядок: Ш1 → Ш2 → Ш4 → Ш3 → Ш5 → Ш6.

- Ш1 и Ш2 — минимальные изменения, после которых GNN получает тот же вход, что и logreg‑по‑рёбрам.
- Ш4 (протокол оценки) нужен до любых сравнений, иначе выводы будут смещены оптимизмом train+val.
- Ш3 — более тяжёлая архитектурная ветка; делать, только если Ш1–Ш2 не вывели GNN на уровень baseline.
- Ш5 — пересборка пресетов и HPO под новые ключи.
- Ш6 — вне CF‑MSR (Synolytic), отдельная задача.

Все новые ключи конфигурации получают значения по умолчанию, воспроизводящие текущее поведение.
Старые запуски, `hpo.py`, `baseline.py` и тесты продолжают работать без правок.

---

## Диагноз (по п1 и дополнительной проверке)

| Признаки | Модель | holdout AUC | CV AUC (train+val) |
|---|---|---|---|
| nodes | logreg / rf / hist_gb | 0.54 / 0.61 / 0.62 | 0.47–0.52, p > 0.05 |
| **edges** | **logreg** | **0.62** [0.55; 0.69] | **0.74 ± 0.04**, p = 0.0 |
| edges | hist_gb | 0.54 | 0.64 ± 0.06 |
| nodes_edges | logreg | 0.62 | 0.74 ± 0.04 |
| — | mean(edges), 0 параметров | 0.64 | 0.50 |
| — | **GNN CF‑MSR** (лог 22‑43‑13) | **0.55** | val 0.53 |

Выводы, на которых построен план:

1. Сигнал есть только в рёбрах, он аддитивный (взвешенная сумма многих слабых рёбер). Узлы — шум.
2. Честный уровень — holdout AUC ≈ 0.62–0.64. CV внутри train+val завышен на ≈ 0.1,
   потому что рёбра train+val — OOF‑предсказания Synolytic, обученные на метках других образцов из того же пула.
3. Распределения рёбер train+val (5 fold‑моделей, OOF) и holdout (одна refit‑модель) различаются:
   на holdout 161 ребро с per‑edge AUC > 0.6, на train+val — 6.
4. В `EdgeMultiHeadAttention.forward` рёбра входят только как аддитивный bias к `scores`.
   Значения `v` берутся из узлов (шум). Сигнал рёбер попадает в readout только косвенно через `node_embedding`.
   Поэтому GNN (0.55) проигрывает агрегатору с нулём параметров (0.64).

---

## Ш1. Признаки узлов из инцидентных рёбер (`dataset.py`)

Идея: для узла *u* в образце *s* признак = статистика весов рёбер, инцидентных *u*, в графе *s*.
Тогда `x` становится информативным, а архитектура модели не меняется (меняется только `input_dim`).

### `conf/data.yaml` — новая секция

```yaml
# Признаки узлов из весов инцидентных рёбер (считаются после отбора узлов/рёбер).
edge_node_features:
  enabled: false
  stats: [mean]        # подмножество: mean | std | max | min
  include_raw: true    # оставить исходный табличный признак узла первым каналом
```

При `enabled: false` поведение не меняется: `x` имеет форму `(num_nodes, 1)`.
`include_raw: false` + `enabled: true` — узлы описываются только агрегатами рёбер (ablation «без табличных признаков»).

### `dataset.py` — новая функция уровня модуля

```python
def compute_edge_node_features(
    edge_index: np.ndarray,       # (2, E)
    edge_attr: np.ndarray,        # (E, S)
    num_nodes: int,
    stats: Sequence[str],
) -> np.ndarray:                  # (S, num_nodes, len(stats))
```

- Агрегация по `edge_index[0]` (оба направления уже присутствуют в `edge_index`, поэтому src покрывает всех соседей).
- `mean` — `np.add.at` по src и деление на степень; `std` — через `E[w²] − mean²` с `clip(0)`;
  `max`/`min` — `np.maximum.at` / `np.minimum.at` с инициализацией `-inf`/`+inf` и заменой на 0 для изолированных узлов.
- Валидация: неизвестное имя статистики → `ValueError`; пустой `stats` → `ValueError`.
- Чистая функция без I/O — покрывается тестом напрямую.

### `WeightedFullGraphDataset`

- `__init__(..., node_extra: Optional[np.ndarray] = None)` — массив `(S, num_nodes, F_extra)` или `None`.
  Проверка формы: `node_extra.shape[:2] == node_features.shape`.
- `replace(node_features=None, edge_index=None, edge_attr=None, node_extra=None)` — дополнительный аргумент;
  `None` означает «копировать существующий».
- `select_nodes(keep)` — фильтровать `node_extra[:, keep, :]`, если задан.
- `select_edges(edge_mask)` — `node_extra` **сбрасывать в `None`**: агрегаты зависят от набора рёбер
  и пересчитываются в `prepare_split`.
- Новое свойство `node_input_dim -> int`: `1 + F_extra` или `F_extra` (при `include_raw=false`), или `1`.
  Для этого датасет хранит флаг `include_raw_node_feature: bool = True`.
- `get(idx)` — сборка `x`:
  - `include_raw=true`, `node_extra=None` → как сейчас, `(num_nodes, 1)`;
  - `include_raw=true`, `node_extra` задан → `concat([node_features[idx][:, None], node_extra[idx]], axis=-1)`;
  - `include_raw=false` → только `node_extra[idx]`.

### `CFMSRDataModule`

- `__init__`: прочитать `data_cfg.edge_node_features` в поля `self.edge_node_enabled`, `self.edge_node_stats`,
  `self.edge_node_include_raw`. Валидация: `include_raw=false` при `enabled=false` → `ValueError`
  (иначе `x` окажется пустым).
- Новый метод `_add_edge_node_features(dataset) -> WeightedFullGraphDataset`:
  вызывает `compute_edge_node_features`, возвращает `dataset.replace(node_extra=...)` и выставляет флаг `include_raw`.
- `prepare_split`: порядок шагов становится

  1. `replace()` сырых данных;
  2. `_select_nodes` → `select_nodes`;
  3. `_select_edges` → `select_edges`;
  4. **новое** `_add_edge_node_features` для train_valid и test (если `enabled`);
  5. `_fit_node_scaler(prepared, train_idx)` — расширить: дополнительно считать
     `self.extra_mean`, `self.extra_std` формы `(num_nodes, F_extra)` по train;
  6. `_apply_node_scaler` — дополнительно нормировать `node_extra` теми же train‑статистиками;
  7. `_compute_pos_weight`.
- Новое свойство `node_input_dim -> int` — делегирует в `train_valid_dataset.node_input_dim`.

### `model.py`

Без изменений в логике. `CFMSR.__init__` уже принимает `input_dim`; `input_proj = nn.Linear(input_dim, hidden_dim)`
работает для любого `input_dim`. `add_gaussian_noise(x, ...)` в `forward` применяется ко всем каналам — допустимо,
так как после scaler все каналы стандартизованы.

### `train.py`, `crossval.py`

В обоих местах создания модели добавить аргумент:

```python
model = CFMSR(
    training_cfg=training_cfg,
    num_nodes=dm.num_nodes,
    input_dim=dm.node_input_dim,
).to(device)
```

`train.py::_run_training_impl` — строка `model = CFMSR(...)`; `crossval.py::run_cv_training` — внутри цикла по fold.
В лог «Данные подготовлены» добавить `node_input_dim`.

### `baseline_models.py`

- `build_feature_matrix`: новый `feature_set = "edge_nodes"` → `dataset.node_extra.reshape(S, -1)`
  (`(S, num_nodes · F_extra)`); `ValueError`, если `node_extra is None`.
  `nodes_edge_nodes` не добавлять (YAGNI).
- `conf/config_baseline.yaml`: комментарий к `feature_set` дополнить `edge_nodes`.

Это даёт прямую проверку: logreg на 133 признаках «средний вес инцидентных рёбер» против logreg на 8778 рёбрах.

### Тесты — новый `tests/test_edge_node_features.py`

- `test_compute_mean_matches_manual` — граф из 3 узлов, 2 образца, сверка с ручным расчётом.
- `test_compute_unknown_stat_raises`.
- `test_disabled_keeps_input_dim_one` — `CFMSRDataModule` с `enabled=false` → `node_input_dim == 1`, `x.shape[-1] == 1`.
- `test_enabled_input_dim_and_get_shape` — `stats=[mean, std]`, `include_raw=true` → `node_input_dim == 3`.
- `test_select_edges_resets_node_extra` и `test_select_nodes_filters_node_extra`.
- `test_extra_scaler_uses_train_only` — по аналогии с `test_selection_stats_use_train_only`.
- `test_cfmsr_forward_input_dim_3` — forward на батче с `x` формы `(N, 3)`, логиты `(B, 1)`.

### Проверка

`python main.py +experiment=regularized data.edge_node_features.enabled=true cv.enabled=true`.
Ожидание: средний CV ROC‑AUC заметно выше 0.53; первый ориентир — результат logreg на `feature_set=edge_nodes`.

---

## Ш2. Нормировка рёбер по train‑статистике (`dataset.py`, `model.py`)

Сейчас рёбра нормирует только `BatchNorm1d(edge_dim)` внутри каждого `EdgeMultiHeadAttention`,
то есть статистика батча из 8 графов. Это шумно и не защищает от сдвига train+val ↔ holdout.

### `conf/data.yaml`

```yaml
# Нормировка весов рёбер по статистикам train.
edge_scaler: none        # none | global | per_edge
```

### `conf/training.yaml`

```yaml
# Нормализация рёбер внутри attention (BatchNorm1d). При data.edge_scaler != none можно отключить.
edge_batchnorm: true
```

### `dataset.py`

- `_fit_edge_scaler(dataset, train_idx) -> None`:
  - `global` → скаляры `self.edge_mean`, `self.edge_std` по всем рёбрам и всем train‑образцам;
  - `per_edge` → векторы формы `(E,)` по train‑образцам;
  - `none` → `self.edge_mean = self.edge_std = None`.
  - Та же защита от NaN и `std < 1e-6 → 1.0`, что в `_fit_node_scaler`.
- `_apply_edge_scaler(dataset) -> WeightedFullGraphDataset` — `dataset.replace(edge_attr=scaled)`; при `None` — `replace()`.
  Проверка совпадения `E` для `per_edge` с предупреждением, как в `_apply_node_scaler`.
- `prepare_split`: вызывать между шагом 3 (отбор рёбер) и шагом 4 (`_add_edge_node_features`).
  Агрегаты Ш1 тогда считаются по нормированным рёбрам; это допустимо, потому что они потом стандартизуются
  собственным scaler’ом ещё раз. Если нужна интерпретируемость агрегатов в единицах вероятности —
  поменять местами шаги 3а и 4 (решение зафиксировать комментарием в коде).

### `model.py`

- `EdgeMultiHeadAttention.__init__(..., use_edge_norm: bool = True)`:
  `self.edge_norm = nn.BatchNorm1d(edge_dim) if (edge_dim > 0 and use_edge_norm) else None`.
  `forward` уже обрабатывает `self.edge_norm is None` — не менять.
- `CFMSRBlock.__init__(..., use_edge_norm: bool = True)` — проброс.
- `CFMSR.__init__`: `use_edge_norm = bool(training_cfg.get("edge_batchnorm", True))`, передать в блоки.

### Тесты — дополнить `tests/test_dataset_selection.py` или новый `tests/test_edge_scaler.py`

- `test_edge_scaler_none_is_identity`.
- `test_edge_scaler_global_train_only` — статистика не зависит от значений val‑образцов.
- `test_edge_scaler_per_edge_shapes`.
- `test_cfmsr_edge_batchnorm_false_has_no_batchnorm` — `all(m.edge_norm is None for m in blocks)`.

### Проверка

Сравнить через CV: `edge_scaler=none` vs `global` vs `per_edge` при включённом Ш1. Побочная польза:
`training.edge_noise_std` становится интерпретируемым (std в единицах стандартизованных весов).

---

## Ш3. Ветка readout по рёбрам (`model.py`)

Делать, если после Ш1–Ш2 GNN в CV всё ещё ниже logreg‑по‑рёбрам. Даёт рёбрам путь к классификатору,
не зависящий от узлов вообще.

### `conf/training.yaml`

```yaml
# Дополнительная ветка агрегации рёбер в вектор графа.
edge_readout: none       # none | mean | gated
edge_readout_dim: 16     # размерность проекции ребра в ветке edge_readout
```

### `model.py` — новый класс

```python
class EdgeReadout(nn.Module):
    def __init__(self, edge_dim: int, out_dim: int, mode: Literal["mean", "gated"], dropout: float) -> None: ...
    def forward(self, edge_attr: torch.Tensor, edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor: ...
    out_dim: int
```

- `mode="mean"`: `proj = Linear(edge_dim, out_dim)` → `global_mean_pool(proj(e), edge_batch, size=num_graphs)`;
  дополнительно `global_max_pool` → `out_dim = 2 * out_dim`.
- `mode="gated"`: `gate = sigmoid(Linear(edge_dim, 1))`, `val = Linear(edge_dim, out_dim)`;
  взвешенная сумма `scatter(gate·val) / scatter(gate)` + mean‑pool → `out_dim = 2 * out_dim`.
- `edge_batch = batch_idx[edge_index[0]]` — вычисляется в `CFMSR.forward` **после** `drop_edges`,
  чтобы маска рёбер была согласована.
- Пулы брать из `torch_geometric.nn` (`global_mean_pool`, `global_max_pool`) или `torch_geometric.utils.scatter`.
  Без новых зависимостей.

### `CFMSR`

- `__init__`: прочитать `edge_readout`, `edge_readout_dim`; валидация значений (`ValueError`).
  `self.edge_readout = EdgeReadout(...)` или `None`.
  Вход классификатора: `readout_dim = self.readout.out_dim + (self.edge_readout.out_dim if ... else 0)`.
- `forward`: после `graph_vec = self.readout(x_dense, mask)`:

  ```python
  if self.edge_readout is not None and edge_attr is not None:
      edge_batch = batch_idx[edge_index[0]]
      edge_vec = self.edge_readout(edge_attr, edge_batch, num_graphs=x_dense.size(0))
      graph_vec = torch.cat([graph_vec, edge_vec], dim=-1)
  ```

- Пустой `edge_attr` (все рёбра удалены / нет рёбер) → нулевой вектор нужной размерности, без исключения.

### Тесты — новый `tests/test_edge_readout.py`

- `test_edge_readout_none_keeps_parameter_count` — число параметров совпадает с текущим при `edge_readout=none`.
- `test_edge_readout_shapes` — для `mean`/`gated` выход `(B, out_dim)`.
- `test_edge_readout_permutation_invariant` — перестановка рёбер внутри графа не меняет выход.
- `test_cfmsr_forward_with_edge_readout` — логиты `(B, 1)`, classifier вход правильной размерности.
- `test_edge_readout_empty_edges` — `edge_index` формы `(2, 0)` → forward не падает.

### Проверка

Ablation через CV: `edge_readout=none | mean | gated` при фиксированных Ш1–Ш2.
Дополнительно полезен прогон `num_layers=1 use_node_embedding=false edge_readout=gated` как «почти линейная» модель:
если она даёт CV ≈ 0.74, значит глубина не нужна.

---

## Ш4. Протокол оценки (без изменения модели)

### `baseline_models.py`, `baseline.py`

- Новая модель `edge_mean` — безпараметрический агрегатор. Класс в `baseline_models.py`:

  ```python
  class MeanScoreClassifier(BaseEstimator, ClassifierMixin):
      """Score = среднее признаков; fit — no-op."""
      def fit(self, x, y) -> "MeanScoreClassifier": ...
      def predict_proba(self, x) -> np.ndarray: ...
  ```

  В `make_model` для `edge_mean` возвращать estimator **без** `StandardScaler` (иначе это уже не «среднее весов»).
  `_predict_scores`, `evaluate_on_split`, `bootstrap_ci`, `permutation_auc`, `repeated_cv_auc` не менять — работают через `predict_proba`.
- `conf/config_baseline.yaml`:
  - `models: [dummy, edge_mean, logreg, random_forest]` — `hist_gb` убрать из дефолта (55 мин на запуск,
    стабильно хуже logreg). Оставить доступным через CLI.
  - `hist_gb.max_iter: 100`.
  - комментарий к `feature_set`: `nodes | edges | nodes_edges | edge_nodes`.
- `tests/test_baseline.py`: `test_edge_mean_classifier_scores_are_row_means`.

### `conf/cv/default.yaml`, `crossval.py`

```yaml
# Оценивать ли holdout ансамблем fold-моделей. Для подбора модели ставить false:
# holdout смотреть один раз в конце.
evaluate_holdout: true
```

В `run_cv_training` обернуть блок с `test_loader`/`holdout_prob_sum` условием. Дефолт `true` сохраняет поведение.

### `README.md` — новый раздел «Протокол оценки»

- CV внутри train+val завышен на ≈ 0.1 относительно holdout (причина: OOF‑рёбра Synolytic). Использовать CV только для
  **ранжирования** вариантов, с `cv.evaluate_holdout=false`.
- Holdout — один раз, для финальной модели.
- Ориентиры: logreg‑по‑рёбрам CV 0.74 ± 0.04 ↔ holdout 0.62; `edge_mean` holdout 0.64.
- Разница меньше std по fold’ам улучшением не считается (повтор правила из `refactoring.md`).

### `conf/data.yaml` — исправить комментарий

`# Например, 355_fold1 -> data_df.iloc[353]` → `# Например, при data_row_shift=0: 355_fold1 -> data_df.iloc[355]`.
Текущий комментарий соответствует shift=2 и вводит в заблуждение.

---

## Ш5. Пресеты и HPO под новые ключи

### Новый `conf/experiment/edge_signal.yaml` (`# @package _global_`)

`regularized.yaml` не трогать — он нужен как контрольная точка.

```yaml
data:
  edge_node_features: {enabled: true, stats: [mean, std], include_raw: true}
  edge_scaler: global
  edge_selection: {method: variance, keep_ratio: 0.3}
training:
  hidden_dim: 32
  num_heads: 4
  num_layers: 1
  num_seeds: 2
  ffn_mult: 2.0
  dropout: 0.4
  classifier_dropout: 0.5
  weight_decay: 1.0e-2
  lr: 3.0e-4
  patience: 15
  monitor_metric: loss
  monitor_smoothing_window: 3
  drop_edge: 0.3
  edge_noise_std: 0.3
  edge_batchnorm: false
  edge_readout: none        # поднять до gated после Ш3
cv:
  enabled: true
  ensemble_seeds: 3
  evaluate_holdout: false
```

Примечание: в `regularized.yaml` сейчас `patience: 100` — с таким значением early stopping фактически отключён
(лог 22‑43‑13: переобучение с эпохи ~18). В новом пресете `patience: 15` + мониторинг loss.

### `conf/config_hpo.yaml`

- Добавить в `hydra.sweeper.params`:

  ```yaml
  data.edge_node_features.enabled: choice(true, false)
  data.edge_scaler: choice(none, global)
  data.edge_selection.keep_ratio: choice(0.1, 0.3, 1.0)
  training.edge_readout: choice(none, mean, gated)     # после Ш3
  ```

- `data.edge_selection.method` зафиксировать `variance` в defaults HPO‑конфига (`keep_ratio=1.0` ≡ все рёбра).
- Сузить дублирующие оси: `training.use_node_embedding: choice(true, false)` оставить —
  после Ш1 эмбеддинг перестаёт быть единственным каналом сигнала, и ответ может измениться.
- `conf/hpo/default.yaml`: `study_name: cf_msr_hpo_v3`.

### `README.md`

Разделы «Конфигурация», «Пресеты экспериментов», «Подбор гиперпараметров»: новые ключи
`edge_node_features`, `edge_scaler`, `edge_batchnorm`, `edge_readout*`, `cv.evaluate_holdout`, пресет `edge_signal`,
feature_set `edge_nodes`, модель `edge_mean`.

---

## Ш6. Synolytic: согласовать распределения train+val и holdout (вне CF‑MSR)

Проект `/home/vadim/workspace/Институт/Archive/Synolytic_network`, отдельный репозиторий и отдельная задача.

- `main.py`: вызов `predict_holdout(..., method="refit")` заменить на значение из конфига;
  `conf/training/config.yaml`: `holdout_method: average`.
  `_predict_average` уже реализован — это бэггинг пяти fold‑моделей, его распределение ближе к OOF‑рёбрам train+val.
- Перегенерировать `results/holdout_graph_active.csv`, затем повторить `python baseline.py baseline.feature_set=edges`
  в CF‑MSR и сравнить holdout AUC logreg (сейчас 0.62) и `edge_mean` (0.64). Если разрыв CV ↔ holdout сузится,
  оставить `average`.
- `data_row_shift`, формат колонок `{idx}_fold{k}`, `StratifiedKFold(5)` — не менять (проверено ранее).

---

## Что менять не нужно

- **`model.py`**: `drop_edges`, `add_gaussian_noise`, `EdgeMultiHeadAttention.forward` (включая `to_dense_adj`‑bias),
  `CFMSRBlock.forward`, `MultiSeedReadout` целиком, `CFMSR._init_weights`, схема augmentations в начале `CFMSR.forward`.
  В `EdgeMultiHeadAttention.__init__` и `CFMSRBlock.__init__` — только новый флаг `use_edge_norm` (Ш2).
- **`dataset.py`**: `parse_sample_column`, `strip_num_prefix`, `build_node_map`, `resolve_node`, `_encode_target`,
  `_resolve_data_index`, `_label_at`, `_build_edges`, `_build_dataset`, `_load_raw`, `_select_nodes`, `_select_edges`,
  `_split_train_val`, `_compute_pos_weight`, `train_loader`/`val_loader`/`test_loader`, `setup()`.
  Меняются только `WeightedFullGraphDataset` (новое поле), `prepare_split` (порядок шагов) и scaler‑методы.
- **`train.py`**: всё ядро метрик (`safe_metrics`, `_manual_*`, `select_score`, `get_*`), `set_seed`, `train_epoch`,
  `evaluate`, `fit_model`, `build_optimizer`, `build_scheduler`, `LrSchedulerWrapper`, `moving_average`,
  `build_criteria`, `SmoothedBCEWithLogitsLoss`, `run_training`, `_resolve_output_paths`.
  В `_run_training_impl` — одна строка (`input_dim=`) и одна строка лога.
- **`crossval.py`**: разбиения, OOF‑сборка по `sample_idx`, nested MLflow runs, объектив `mean | mean_minus_std`.
  Меняются: `input_dim=` и условие `evaluate_holdout`.
- **`cv_splits.py`**, **`config_utils.py`**, **`mlflow_utils.py`** — без изменений.
- **`baseline.py`** (точка входа), `evaluate_on_split`, `bootstrap_ci`, `permutation_auc`, `repeated_cv_auc`,
  `_unique_undirected_mask` — без изменений.
- **Точки входа** `main.py`, `hpo.py` — без изменений.
- **Конфиги**: `conf/experiment/regularized.yaml` (контрольная точка), `conf/mlflow/default.yaml`, `conf/config.yaml`,
  `hydra.run.dir`, пути и `data_row_shift: 0` в `conf/data.yaml` (меняется только комментарий).
- **Существующие тесты** (19 шт.) — не редактировать; новые ключи с дефолтами их не ломают.
- **Зависимости**: `requirements.txt` не расширяется — `torch_geometric.nn` пулы и sklearn `BaseEstimator` уже доступны.
- **Synolytic**: `synolytic.py`, формат графов, `data_row_shift`‑логика маппинга.
- Каталоги `outputs/`, `multirun/`, `mlruns/`, `artifacts/`.

---

## Сводка по файлам

| Файл | Действие | Шаг |
|---|---|---|
| `dataset.py` | `compute_edge_node_features`; `node_extra`, `node_input_dim` в `WeightedFullGraphDataset`; `_add_edge_node_features`, `_fit/_apply_edge_scaler`, расширение node scaler, порядок в `prepare_split` | Ш1, Ш2 |
| `model.py` | `use_edge_norm` в attention/block/CFMSR; класс `EdgeReadout`; конкатенация в `CFMSR.forward` | Ш2, Ш3 |
| `train.py` | `input_dim=dm.node_input_dim` в `_run_training_impl` | Ш1 |
| `crossval.py` | `input_dim=`; условие `cv.evaluate_holdout` | Ш1, Ш4 |
| `baseline_models.py` | `feature_set=edge_nodes`; `MeanScoreClassifier`, модель `edge_mean` | Ш1, Ш4 |
| `conf/data.yaml` | `edge_node_features`, `edge_scaler`; исправить комментарий про `iloc` | Ш1, Ш2, Ш4 |
| `conf/training.yaml` | `edge_batchnorm`, `edge_readout`, `edge_readout_dim` | Ш2, Ш3 |
| `conf/cv/default.yaml` | `evaluate_holdout` | Ш4 |
| `conf/config_baseline.yaml` | модели по умолчанию, `hist_gb.max_iter`, комментарии | Ш4 |
| `conf/experiment/edge_signal.yaml` | создать | Ш5 |
| `conf/config_hpo.yaml`, `conf/hpo/default.yaml` | новые оси, `study_name: cf_msr_hpo_v3` | Ш5 |
| `tests/test_edge_node_features.py`, `tests/test_edge_scaler.py`, `tests/test_edge_readout.py` | создать | Ш1–Ш3 |
| `tests/test_baseline.py` | один тест на `edge_mean` | Ш4 |
| `README.md` | «Протокол оценки», новые ключи, пресет | Ш4, Ш5 |
| Synolytic `main.py`, `conf/training/config.yaml` | `holdout_method` | Ш6 |

Коммиты — по одному на шаг, на русском с префиксом
(`feat: признаки узлов из инцидентных рёбер`, `feat: edge scaler по train`, `feat: EdgeReadout`, `docs: протокол оценки`).

---

## Критерии успеха

Все сравнения — `cv.enabled=true`, `ensemble_seeds=3`, `cv.evaluate_holdout=false`; holdout — один раз в конце.

1. После Ш1 (+Ш2): средний CV ROC‑AUC GNN ≥ CV logreg на `feature_set=edge_nodes`.
   Если нет — ошибка в пайплайне (scaler, порядок шагов, `input_dim`), а не в модели.
2. После Ш3: CV GNN в пределах std от logreg‑по‑рёбрам (0.74 ± 0.04).
3. Финальный holdout GNN ≥ 0.62 (logreg) и желательно ≥ 0.64 (`edge_mean`). Если GNN не превосходит `edge_mean`
   на holdout, для задачи достаточно линейной модели — это тоже результат.
4. Разрыв train ROC‑AUC − val ROC‑AUC на лучшей эпохе < 0.15 (сейчас ≈ 0.35).
