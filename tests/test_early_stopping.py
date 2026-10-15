"""
Tests for early stopping helpers.
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from train import fit_model, moving_average, select_score


def test_moving_average_window_one() -> None:
    """Окно 1 возвращает само значение."""
    assert moving_average([0.1, 0.5, 0.9], 1) == 0.9


def test_moving_average_window_three() -> None:
    """Окно 3 усредняет последние три."""
    assert abs(moving_average([1.0, 2.0, 3.0, 4.0], 3) - 3.0) < 1e-9


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, batch: Any) -> torch.Tensor:
        if hasattr(batch, "x"):
            x = batch.x
        else:
            x = batch[0]
        return self.lin(x)


class _Batch:
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y
        self.batch = torch.zeros(x.size(0), dtype=torch.long)
        self.edge_index = torch.zeros(2, 0, dtype=torch.long)
        self.sample_idx = torch.arange(x.size(0))

    def to(self, device: torch.device) -> "_Batch":
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self


class _ListLoader:
    def __init__(self, batches: List[_Batch]) -> None:
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def test_early_stop_after_patience() -> None:
    """Остановка ровно через patience эпох без улучшения."""
    torch.manual_seed(0)
    # Константные батчи: score не будет расти → early stop.
    x = torch.randn(8, 2)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    loader = _ListLoader([_Batch(x, y)])

    # Патчим train_epoch/evaluate через крошечную модель и короткий patience.
    # Используем monitor_metric=loss: score = -loss; без улучшения → stop.
    cfg = OmegaConf.create(
        {
            "epochs": 50,
            "lr": 1e-6,
            "weight_decay": 0.0,
            "patience": 3,
            "grad_clip": 1.0,
            "score_metric": "loss",
            "monitor_metric": "loss",
            "early_stop_min_delta": 1e6,  # любое улучшение недостаточно
            "monitor_smoothing_window": 1,
            "lr_scheduler": "plateau",
            "warmup_epochs": 0,
            "metrics": ["roc_auc"],
            "adamw_no_decay_bias_norm": False,
        }
    )

    model = _TinyModel()
    # Подмена forward для совместимости с train_epoch (ожидает batch.y / model(batch)).
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")

    # Monkeypatch: train_epoch и evaluate работают с нашим _Batch,
    # но CFMSR-стиль model(batch) уже есть. Нужно чтобы squeeze работал.
    result = fit_model(
        model=model,
        train_loader=loader,  # type: ignore[arg-type]
        val_loader=loader,  # type: ignore[arg-type]
        criterion=criterion,
        training_cfg=cfg,
        device=device,
        metric_names=["roc_auc"],
        metric_threshold=0.5,
    )

    # Первый epoch всегда "улучшение" (best=-inf). Затем 3 без улучшения → stop на 4.
    assert len(result.history) == 4
    assert result.best_epoch == 1


def test_monitor_metric_loss_picks_min_loss_epoch() -> None:
    """select_score для loss возвращает -loss (больше = лучше)."""
    metrics = {"loss": 0.5, "roc_auc": 0.4}
    score, desc = select_score(metrics, "loss")
    assert abs(score - (-0.5)) < 1e-9
    assert "neg_loss" in desc
