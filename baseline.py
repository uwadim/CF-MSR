"""
Baseline entry point.

Точка входа для запуска sklearn-базлайнов против тех же данных, что и GNN.
"""

import hydra
from omegaconf import DictConfig

from baseline_models import run_baselines


@hydra.main(version_base=None, config_path="conf", config_name="config_baseline")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point for baselines.

    Parameters
    ----------
    cfg : DictConfig
        Hydra-конфигурация с ветками ``data``, ``training``, ``baseline``.
    """
    run_baselines(cfg)


if __name__ == "__main__":
    main()
