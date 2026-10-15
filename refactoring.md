# План: борьба с переобучением в CF-MSR

Номера пунктов (п1–п7) соответствуют списку предложений по уменьшению переобучения.

Рекомендуемый порядок работ: этап 0 → п1 → п2 → п5 → п3 → п6 → п4 → п7.

- Этап 0 нужен перед п4 и п6: сейчас `CFMSRDataModule` и `_run_training_impl` нельзя вызвать повторно для другого разбиения.
- П1 идёт первым: он покажет, есть ли в данных сигнал вообще.

Все новые параметры получают значения по умолчанию, которые воспроизводят текущее поведение. Тогда старые запуски и `hpo.py` продолжат работать без правок.

## Диагноз (последний прогон `outputs/2026-10-13/17-30-43`)

- Train ROC-AUC растёт с 0.46 до 0.85 (эпоха ~44), val loss растёт с 0.89 до 1.7.
- Val ROC-AUC всё время около 0.45–0.55, лучшее значение 0.55 выбрано из шума; итоговый test ROC-AUC = 0.53.
- Данных мало: 468 train / 118 val, при этом ~133 признака узлов и ≈17.5 тыс. весов рёбер на граф.
- `weight_decay` в HPO ищется в диапазоне 1e-6…1e-3, то есть практически без регуляризации.

---

## Этап 0. Рефакторинг без изменения поведения

**`train.py`.** Сейчас `_run_training_impl` занимает около 265 строк и делает всё сразу. Выделить из него:

- `_resolve_output_paths(training_cfg) -> Tuple[str, str]`. Заменяет блок с четырьмя проверками `None`.
- `build_optimizer(model, training_cfg) -> Optimizer`. Вынести создание AdamW. Если нужно, добавить группы параметров без weight decay для bias, LayerNorm и Embedding.
- `build_scheduler(optimizer, training_cfg) -> LrSchedulerWrapper`. Класс-обёртка с методом `step(score)`. Пока внутри только `ReduceLROnPlateau`.
- `fit_model(model, train_loader, val_loader, criterion, training_cfg, device, metric_names, metric_threshold) -> FitResult`. Это эпоха-цикл с ранней остановкой и сохранением лучшего состояния.
- `@dataclass FitResult`: `best_score`, `best_epoch`, `best_state`, `history`.
- `_run_training_impl` остаётся оркестратором из четырёх шагов: данные → модель → `fit_model` → финальная оценка и сохранение.

**`dataset.py`.** `setup()` и `_apply_node_scaler` сейчас изменяют датасет на месте. Для кросс-валидации нужны неизменяемые «сырые» данные:

- В `setup()` оставить только загрузку сырых данных (`_load_raw`). Сырые `train_valid_dataset` и `test_dataset` не должны меняться.
- Новый метод `prepare_split(train_idx, val_idx) -> None` делает всё, что зависит от разбиения: отбор признаков, scaler, `pos_weight`. Он работает на копиях.
- Новый метод `WeightedFullGraphDataset.replace(node_features=None, edge_index=None, edge_attr=None) -> WeightedFullGraphDataset` создаёт копию с подменёнными массивами.
- `_apply_node_scaler` возвращает новый датасет вместо изменения на месте.
- Публичное поведение `setup()`, `train_loader()`, `val_loader()`, `test_loader()` сохраняется.

**Проверка:** прогон `python main.py` с тем же seed даёт те же метрики, что и до рефакторинга (эпоха 1: val roc_auc=0.4452).

---

## П1. Базовые модели (проверка наличия сигнала)

### Новые файлы

| Файл | Назначение |
|---|---|
| `baseline.py` | Тонкая точка входа: `@hydra.main(config_name="config_baseline")` вызывает `run_baselines(cfg)`. Устроена как `main.py` и `hpo.py`. |
| `baseline_models.py` | Вся логика. Вынесена отдельно, чтобы её можно было тестировать. |
| `conf/config_baseline.yaml` | Корневой конфиг: defaults `data`, `training`, `mlflow: default`. `training` подключается ради `seed`, `metrics`, `metric_threshold`, `score_metric`. Своя секция `baseline:` лежит внутри этого же файла. |
| `tests/test_baseline.py` | Тесты на синтетике. |

### Секция `baseline:` в `conf/config_baseline.yaml`

```yaml
mlflow:
  experiment_name: cf_msr_baseline
hydra:
  run:
    dir: artifacts/baseline_runs/${now:%Y-%m-%d_%H-%M-%S}
baseline:
  models: [dummy, logreg, random_forest, hist_gb]
  feature_set: nodes            # nodes | edges | nodes_edges
  logreg_c_grid: [0.001, 0.01, 0.1, 1.0, 10.0]
  rf: {n_estimators: 500, max_depth: 4, min_samples_leaf: 5}
  hist_gb: {max_depth: 3, learning_rate: 0.05, max_iter: 200}
  cv: {n_splits: 5, n_repeats: 5}
  bootstrap_n: 1000
  permutation_n: 200
  results_out: baseline_results.csv
```

### Функции в `baseline_models.py`

Сигнатуры с типами и numpy-docstring:

- `build_feature_matrix(dm, dataset, feature_set) -> np.ndarray`. Берёт `node_features`, а для рёбер `edge_attr`. `edge_attr` дублирует направленные рёбра, поэтому нужна `_unique_undirected_mask(edge_index)`: оставить пары `u<v`.
- `make_model(name, baseline_cfg, seed) -> sklearn estimator`. Это `Pipeline(StandardScaler, …)`. `logreg` — `LogisticRegressionCV` с L2 и `class_weight="balanced"`. `hist_gb` — `HistGradientBoostingClassifier`. `dummy` — `DummyClassifier(strategy="prior")`. Новые зависимости не нужны, scikit-learn уже есть.
- `evaluate_on_split(model, X_tr, y_tr, X_ev, y_ev, metric_names, threshold) -> Dict[str, float]`. Использует `safe_metrics` из `train.py`.
- `bootstrap_ci(y_true, y_score, n, seed) -> Tuple[float, float, float]`. Доверительный интервал AUC.
- `permutation_auc(model_factory, X, y, n, seed) -> np.ndarray`. AUC при перемешанных метках. Даёт p-value и «потолок шума».
- `repeated_cv_auc(model_factory, X, y, n_splits, n_repeats, seed) -> np.ndarray`. `RepeatedStratifiedKFold` на объединении train+val, 586 объектов.
- `run_baselines(cfg) -> pd.DataFrame`. Строит `CFMSRDataModule(...).setup()` и использует `dm.train_valid_dataset`, `dm.train_idx`, `dm.val_idx`, `dm.test_dataset`. Разбиение и z-score те же, что у GNN, поэтому результаты сопоставимы. Для holdout берёт только образцы с валидной меткой. Для каждой модели считает метрики val, test, CV mean±std, bootstrap 95% CI и p-value по перестановкам. Пишет таблицу в лог и в CSV, логирует в MLflow через `mlflow_run_context` с отдельным экспериментом.

### Правило интерпретации

- Если 95% CI для AUC включает 0.5, а CV AUC ≈ 0.5, то сигнал слабый. Тогда сначала меняйте данные, таргет и признаки, а регуляризацию подбирайте потом.
- Если базовая модель стабильно даёт AUC ≥ 0.6, а GNN нет, переобучение GNN реально. Продолжайте с п3 и далее.

### Тесты

- Случайные признаки дают AUC около 0.5, p-value больше 0.05.
- Синтетика с заложенным сигналом даёт AUC больше 0.8.
- `_unique_undirected_mask` оставляет ровно половину направленных рёбер.

### Утечки

- `StandardScaler` стоит внутри `Pipeline`, поэтому внутри CV он обучается по каждому fold отдельно.
- `dm.setup()` уже стандартизирует по `train_idx`. Это безвредно, но при CV может слегка утекать. После этапа 0 `baseline.py` может брать сырые признаки.

### Запуск

```bash
python baseline.py
python baseline.py baseline.feature_set=nodes_edges baseline.models='[logreg,hist_gb]'
```

---

## П2. Малая модель + сильная регуляризация (только конфиги)

Создать `conf/experiment/regularized.yaml` с первой строкой `# @package _global_`:

```yaml
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
```

- Запуск: `python main.py +experiment=regularized`. Для HPO можно использовать тот же файл.
- В `conf/config.yaml` и `conf/config_hpo.yaml` пресет в `defaults` не подключать, только через `+experiment=`. Тогда текущие значения по умолчанию остаются нетронутыми.
- `hidden_dim=32` кратно `num_heads=4`, а в `CFMSR.__init__` есть `assert hidden_dim % num_heads == 0`.
- `README.md`: добавить раздел про пресеты.

---

## П3. Дополнительная регуляризация (код + конфиг)

### `conf/training.yaml`

Новые ключи с нейтральными значениями по умолчанию:

```yaml
drop_edge: 0.0
node_noise_std: 0.0
edge_noise_std: 0.0
label_smoothing: 0.0
use_node_embedding: true
```

### `model.py`

- Новые чистые функции:
  - `drop_edges(edge_index, edge_attr, p, training) -> Tuple[Tensor, Optional[Tensor]]` для DropEdge.
  - `add_gaussian_noise(t, std, training, relative: bool) -> Tensor`. Для рёбер шум масштабируется на `t.std()`, потому что веса рёбер пока не нормализованы.
- `CFMSR.__init__`: читает `drop_edge`, `node_noise_std`, `edge_noise_std`. Параметр `use_node_embedding: Optional[bool] = None`: если `None`, берётся из `training_cfg.get("use_node_embedding", True)`. Валидация диапазонов, `ValueError`.
- `CFMSR.forward`: в режиме train применить шум к `x` и DropEdge плюс шум к `edge_attr`, до входной проекции и до стека блоков. В eval ничего не меняется.
- `EdgeMultiHeadAttention.edge_norm` — `BatchNorm1d` по рёбрам. После DropEdge статистики считаются по оставшимся рёбрам, это приемлемо, но стоит проверить.

### `train.py`

- Новый `class SmoothedBCEWithLogitsLoss(nn.Module)`: цель `y*(1-ε)+0.5ε`, поддерживает `pos_weight`.
- Новая `build_criteria(pos_weight, label_smoothing) -> Tuple[nn.Module, nn.Module]`. Для train возвращает сглаженный loss, для val/test обычный BCE. Так val loss остаётся сопоставимым.
- Сигнатуры `train_epoch` и `evaluate` не меняются.

### Тесты `tests/test_regularization.py`

- В eval-режиме `drop_edges` и шум ничего не меняют.
- В train-режиме доля удалённых рёбер примерно равна `p`.
- При `label_smoothing=0` loss совпадает с `BCEWithLogitsLoss`.
- `CFMSR` с `use_node_embedding=false` не содержит `node_embedding`.

---

## П4. Сокращение размерности входа

Нужно, чтобы выбор признаков и рёбер происходил только по train-части, иначе будет утечка. Поэтому он живёт в `prepare_split` из этапа 0.

### `conf/data.yaml`

Новые ключи:

```yaml
feature_selection:
  method: none            # none | variance | mutual_info | l1
  top_k: null
edge_selection:
  method: none            # none | variance | abs_mean | mutual_info
  top_k: null
  keep_ratio: null
```

### `dataset.py`

- Новый `WeightedFullGraphDataset.select_nodes(keep: np.ndarray) -> WeightedFullGraphDataset`. Фильтрует `node_features`, переиндексирует `edge_index`, удаляет рёбра к убранным узлам и фильтрует `edge_attr`.
- Новый `WeightedFullGraphDataset.select_edges(edge_mask: np.ndarray) -> WeightedFullGraphDataset`. Выбирает неориентированные пары и оставляет оба направления.
- В `CFMSRDataModule`: `_select_nodes(dataset, train_idx) -> np.ndarray` и `_select_edges(dataset, train_idx) -> np.ndarray`. Считают оценки только по `train_idx`. Результат (`kept_nodes`, `edge_mask`) хранится в модуле и применяется и к val, и к holdout.
- `prepare_split` вызывает в порядке: выбор узлов → выбор рёбер → scaler → `pos_weight`.
- После выбора обновить `self.num_nodes` и `self.feature_cols`, потому что `CFMSR` строит `nn.Embedding(num_nodes, …)` по `dm.num_nodes`.
- Валидация: `top_k` и `keep_ratio` одновременно задавать нельзя. `top_k` больше числа признаков тоже ошибка.

### Тесты `tests/test_dataset_selection.py`

На маленьком синтетическом датасете:

- После `select_nodes` индексы рёбер лежат в `[0, N_new)`.
- Статистики отбора считаются только по train: изменение val-меток не меняет результат.
- Для holdout используются те же `kept_nodes`.

---

## П5. Ранняя остановка и расписание lr

### `conf/training.yaml`

Новые ключи:

```yaml
monitor_metric: null       # null -> score_metric; можно: loss
early_stop_min_delta: 0.0
monitor_smoothing_window: 1
lr_scheduler: plateau      # plateau | cosine
warmup_epochs: 0
```

### `train.py`

- В `fit_model`: `monitor_metric` берётся из конфига, иначе из `score_metric`. Значение считается через уже существующий `select_score(val_metrics, monitor_metric)`, у которого есть fallback на `-loss`.
- Новая чистая функция `moving_average(values, window) -> float` для сглаживания по последним `window` эпохам.
- Улучшение считается как `score > best_score + early_stop_min_delta`.
- `build_scheduler` (из этапа 0): `plateau` оставляет текущий `ReduceLROnPlateau(mode="max")`. `cosine` — это `SequentialLR(LinearLR warmup, CosineAnnealingLR)`, и там `step()` вызывается без метрики.
- `FitResult.best_epoch` пишется в MLflow метрикой `best_epoch`.
- Значение `patience` по умолчанию лучше поставить 15, но это решение пользователя: у текущих запусков оно 30.

### Тесты `tests/test_early_stopping.py`

- `moving_average` с окном 1 равен самому значению.
- Остановка срабатывает ровно через `patience` эпох без улучшения.
- При `monitor_metric=loss` выбирается эпоха с минимальным val loss.

---

## П6. Кросс-валидация, ансамбль, group split

### `conf/data.yaml`

Ключ `group_col: null`. Это имя колонки с id пациента, если один пациент может встречаться в нескольких образцах.

### Новый `conf/cv/default.yaml` (`# @package cv`)

```yaml
enabled: false
scheme: stratified_kfold     # fold_column | stratified_kfold | group_kfold
n_splits: 5
n_repeats: 1
ensemble_seeds: 1
save_models: false
oof_out: cv_oof_predictions.csv
objective: mean              # mean | mean_minus_std
std_penalty: 0.5
```

В `defaults` подключить `- cv: default` в `conf/config.yaml` и `conf/config_hpo.yaml`.

### Новый `cv_splits.py`

- `make_cv_splits(labels, folds, groups, cv_cfg, seed) -> List[Tuple[np.ndarray, np.ndarray]]`. Чистая функция.
  - `fold_column` использует `dataset.folds`.
  - `stratified_kfold` — `RepeatedStratifiedKFold`.
  - `group_kfold` — `StratifiedGroupKFold`.
- Валидация: `n_splits >= 2`, `group_col` обязателен для `group_kfold`, в каждом fold есть оба класса.

### Новый `crossval.py`

- `run_cv_training(cfg) -> float`. Строит данные один раз и в цикле по fold-ам вызывает `dm.prepare_split` → `build_model` → `fit_model` → оценка. Собирает OOF-предсказания и предсказания holdout от каждого fold и seed.
  - Метрики: среднее ± std по fold-ам и общий OOF ROC-AUC.
  - Ансамбль: усреднение вероятностей holdout по всем моделям.
  - Возвращаемое значение: `mean` или `mean − std_penalty·std` по `cv.objective`. Это целевое значение для Optuna.
- Не использовать `val_fold` одновременно с `cv.enabled=true`; это конфликт и нужна ошибка.

### `dataset.py`

- `_build_dataset`: если задан `data.group_col`, сохранить `dataset.groups`.
- `_split_train_val`: при заданном `group_col` использовать групповой сплит.

### `train.py`

- `run_training`: в начале `if cfg.get("cv") and cfg.cv.enabled: return run_cv_training(cfg)`. Тогда `main.py` и `hpo.py` не меняются.

### `mlflow_utils.py`

- `mlflow_run_context(cfg, run_name=None, nested=False)`. Для CV: родительский run плюс вложенные `fold_k`. Внутри HPO trial он тоже вложенный.
- Проверить, что `log_hparams` корректно логирует вложенные ключи `cv.*`.

### Тесты `tests/test_cv_splits.py`

- Объекты train и val одного fold-а не пересекаются.
- В `group_kfold` ни одна группа не попадает в обе части.
- Все fold-ы содержат оба класса.

### Что проверить до реализации

Колонки вида `###_fold#` означают, что графы строились по fold-ам. Если веса рёбер вычислялись с использованием меток всех образцов, кросс-валидация по этим графам может быть смещена. Нужно выяснить, как именно строился Synolytic-граф.

---

## П7. Пространство поиска HPO

### `conf/config_hpo.yaml`, секция `hydra.sweeper.params`

```yaml
training.weight_decay: tag(log, interval(1e-4, 1e-1))
training.hidden_dim: choice(16, 32, 64)
training.num_heads: choice(2, 4)
training.num_layers: range(1, 3)
training.dropout: interval(0.2, 0.6)
training.classifier_dropout: interval(0.2, 0.6)
training.drop_edge: interval(0.0, 0.5)
training.node_noise_std: interval(0.0, 0.3)
training.edge_noise_std: interval(0.0, 0.3)
training.label_smoothing: interval(0.0, 0.2)
training.use_node_embedding: choice(true, false)
```

- `lr` оставить как есть и проверить, что `range(1, 3)` в Hydra-Optuna включает верхнюю границу.
- Остальные ключи (`ffn_mult`, `num_seeds`) уменьшить до `choice(2.0)` и `choice(2, 4)`.
- **Обязательно** сменить `study_name` в `conf/hpo/default.yaml` (например, `cf_msr_hpo_v2`) или удалить `artifacts/optuna.db`. Optuna ругается на несовместимые распределения параметров в существующем study.
- HPO запускать с `cv.enabled=true`, иначе она снова подгонится под одну выборку из 118 объектов.
- Тестовую выборку для выбора модели не использовать.

---

## Что менять не нужно

- **`model.py`**: `EdgeMultiHeadAttention.forward`, `CFMSRBlock`, `MultiSeedReadout`, `_init_weights`. Внутри `CFMSR` меняется только `__init__` и `forward` (см. п3).
- **`dataset.py`**: `parse_sample_column`, `strip_num_prefix`, `build_node_map`, `resolve_node`, `_encode_target`, `_resolve_data_index`, `_label_at`, `_build_edges`, `_compute_pos_weight` (логика). В `_build_dataset` добавляется только `groups`.
- **`train.py`**: `safe_metrics`, `_manual_*`, `format_metrics`, `select_score`, `get_*`, `set_seed`, `train_epoch`, `evaluate` (сигнатуры и поведение), `log_metrics_block`. Это ядро метрик, и `baseline.py` использует их же.
- **`mlflow_utils.py`**: `resolve_sqlite_uri`, `get_mlflow_cfg`, `log_split_metrics`, `log_overfit_gap_metrics`, `log_artifact_path`.
- **Точки входа** `main.py` и `hpo.py`: CV подключается внутри `run_training`.
- **Конфиги**: `conf/mlflow/default.yaml`, `conf/hpo/default.yaml` (кроме `study_name`), `hydra.run.dir` в обоих корневых конфигах, пути и параметры в `conf/data.yaml` (кроме новых ключей).
- **Зависимости**: `requirements.txt` не расширяется, scikit-learn уже есть. Добавить только `pytest` (отдельным dev-файлом или строкой).
- Каталоги `outputs/`, `multirun/`, `mlruns/`, `artifacts/` — артефакты запусков, не трогать.

---

## Сводка по файлам

| Файл | Действие |
|---|---|
| `baseline.py`, `baseline_models.py`, `conf/config_baseline.yaml` | создать (п1) |
| `cv_splits.py`, `crossval.py`, `conf/cv/default.yaml` | создать (п6) |
| `conf/experiment/regularized.yaml` | создать (п2) |
| `tests/` (файлы выше) | создать |
| `train.py` | рефакторинг, `fit_model`, `build_*`, loss со сглаживанием, диспетчер CV |
| `dataset.py` | `replace`, `select_nodes`, `select_edges`, `prepare_split`, группы |
| `model.py` | `drop_edges`, `add_gaussian_noise`, правки `CFMSR` |
| `mlflow_utils.py` | вложенные runs |
| `conf/training.yaml`, `conf/data.yaml`, `conf/config.yaml`, `conf/config_hpo.yaml`, `conf/hpo/default.yaml` | новые ключи |
| `README.md`, `requirements.txt` | разделы про baseline, CV и пресеты; `pytest` |

Коммиты — по одному на этап, на русском с префиксом (например, `refactor: вынес fit_model из train.py`, `feat: добавил baseline.py`).

## Как сравнивать варианты

Запускайте каждый вариант с `cv.enabled=true` и тремя seed. Таблица: исходная модель → малая модель → + weight decay → + DropEdge → + шум → + отбор признаков. Сравнивайте по среднему CV ROC-AUC ± std и по OOF-AUC, а не по одному val-разбиению.

Если разница меньше std, улучшением её считать нельзя.
