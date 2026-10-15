"""
Hyperparameter optimization entry point.

Точка входа для поиска гиперпараметров через Hydra Optuna Sweeper.
"""

import hydra
from omegaconf import DictConfig

from train import run_training


@hydra.main(version_base=None, config_path="conf", config_name="config_hpo")
def main(cfg: DictConfig) -> float:
    """
    Hydra + Optuna entry point.

    Пример запуска:

        python hpo.py -m

    Пример с переопределением числа trials:

        python hpo.py -m hpo.n_trials=10

    Важно: используется Optuna Sweeper, а не grid-multirun Hydra.
    """
    return run_training(cfg)


if __name__ == "__main__":
    main()
