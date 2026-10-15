"""
SkipGCN entry point.

Отдельная точка входа для запуска SkipGCN (порт HCP_GCN)
на данных проекта. Использует корневой конфиг ``config_skipgcn``,
который подключает пресет ``experiment/skipgcn``.
"""

import hydra
from omegaconf import DictConfig

from train import run_training


@hydra.main(version_base=None, config_path="conf", config_name="config_skipgcn")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point.

    Пример запуска:

        python main_skipgcn.py

    Пример переопределения параметров:

        python main_skipgcn.py training.lr=3e-4 training.skipgcn.hidden_channels=32

    Пример кросс-валидации:

        python main_skipgcn.py cv.enabled=true

    MLflow:

        python main_skipgcn.py mlflow.experiment_name=my_skipgcn_run
        python main_skipgcn.py mlflow.enabled=false
    """
    run_training(cfg)


if __name__ == "__main__":
    main()
