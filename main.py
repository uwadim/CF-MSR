import hydra
from omegaconf import DictConfig

from train import run_training


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point.

    Пример запуска:

        python main.py

    Пример переопределения параметров:

        python main.py training.lr=1e-3 training.hidden_dim=256 training.num_layers=6

    Пример указания списка признаков:

        python main.py data.feature_cols='["Зрительная","Стволовая","Пирамидная"]'

    MLflow:

        python main.py mlflow.experiment_name=cf_msr_train
        python main.py mlflow.enabled=false

    Для HPO используйте:

        python hpo.py -m
    """
    run_training(cfg)


if __name__ == "__main__":
    main()