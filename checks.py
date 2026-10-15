"""Просмотр топ-trials и важностей параметров Optuna study."""

from __future__ import annotations

from pathlib import Path

import optuna
import pandas as pd
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator

# Абсолютный путь: скрипт можно запускать из любой cwd.
_ROOT = Path(__file__).resolve().parent
_STORAGE = f"sqlite:///{_ROOT / 'artifacts' / 'optuna.db'}"
_STUDY_NAME = "cf_msr_hpo_v4"
_OUT_XLSX = _ROOT / "artifacts" / f"{_STUDY_NAME}_summary.xlsx"


def main() -> None:
    """
    Load study and export top trials / importances to Excel.

    Returns
    -------
    None
        Пишет ``artifacts/<study>_summary.xlsx`` и дублирует в stdout.
    """
    st = optuna.load_study(study_name=_STUDY_NAME, storage=_STORAGE)
    df = st.trials_dataframe().sort_values("value", ascending=False)
    param_cols = [c for c in df.columns if c.startswith("params_")]
    top10 = df.head(10)[["number", "value", "state", *param_cols]].copy()
    print(top10)

    # fANOVA (дефолт Optuna 2.10) падает на NumPy 2.x:
    # ValueError: setting an array element with a sequence.
    importances = optuna.importance.get_param_importances(
        st,
        evaluator=MeanDecreaseImpurityImportanceEvaluator(),
    )
    print(importances)

    imp_df = pd.DataFrame(
        {
            "param": list(importances.keys()),
            "importance": list(importances.values()),
        }
    )
    all_trials = df.copy()

    _OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(_OUT_XLSX, engine="openpyxl") as writer:
        top10.to_excel(writer, sheet_name="top10", index=False)
        all_trials.to_excel(writer, sheet_name="all_trials", index=False)
        imp_df.to_excel(writer, sheet_name="importances", index=False)

    print(f"Saved: {_OUT_XLSX}")


if __name__ == "__main__":
    main()
