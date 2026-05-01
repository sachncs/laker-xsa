"""Tests for training modules (losses, trainer).

Covers label_smoothing_cross_entropy, TrainingConfig, and Trainer.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.model.full_model import XSALAKERTransformer
from laker_xsa.training.losses import label_smoothing_cross_entropy
from laker_xsa.training.trainer import Trainer, TrainingConfig


# ---------------------------------------------------------------------------
# label_smoothing_cross_entropy
# ---------------------------------------------------------------------------

class TestLabelSmoothingCrossEntropy:
    """Tests for label_smoothing_cross_entropy."""

    def test_no_smoothing(self) -> None:
        logits = torch.randn(32, 100)
        labels = torch.randint(0, 100, (32,))
        loss = label_smoothing_cross_entropy(logits, labels, smoothing=0.0)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_with_smoothing(self) -> None:
        logits = torch.randn(32, 100)
        labels = torch.randint(0, 100, (32,))
        loss = label_smoothing_cross_entropy(logits, labels, smoothing=0.1)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_3d_input_flattened(self) -> None:
        logits = torch.randn(2, 16, 50)
        labels = torch.randint(0, 50, (2, 16))
        loss = label_smoothing_cross_entropy(logits, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_ignore_index(self) -> None:
        logits = torch.randn(4, 10)
        labels = torch.tensor([1, 2, -1, 3])
        loss = label_smoothing_cross_entropy(logits, labels, ignore_index=-1)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_all_ignored(self) -> None:
        logits = torch.randn(4, 10)
        labels = torch.full((4,), -1)
        loss = label_smoothing_cross_entropy(logits, labels, ignore_index=-1)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_perfect_prediction_no_smoothing(self) -> None:
        logits = torch.zeros(4, 10)
        logits[:, 3] = 100.0  # Very confident on class 3
        labels = torch.full((4,), 3)
        loss = label_smoothing_cross_entropy(logits, labels, smoothing=0.0)
        assert loss.item() < 0.01  # Near-zero loss

    def test_invalid_smoothing_negative(self) -> None:
        with pytest.raises(ValueError, match="Label smoothing must be"):
            label_smoothing_cross_entropy(
                torch.randn(4, 10), torch.randint(0, 10, (4,)), smoothing=-0.1,
            )

    def test_invalid_smoothing_one(self) -> None:
        with pytest.raises(ValueError, match="Label smoothing must be"):
            label_smoothing_cross_entropy(
                torch.randn(4, 10), torch.randint(0, 10, (4,)), smoothing=1.0,
            )

    def test_gradient_flow(self) -> None:
        logits = torch.randn(4, 10, requires_grad=True)
        labels = torch.randint(0, 10, (4,))
        loss = label_smoothing_cross_entropy(logits, labels, smoothing=0.0)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.num_epochs == 10
        assert cfg.learning_rate == 1e-3
        assert cfg.seed == 42
        assert cfg.label_smoothing == 0.1

    def test_custom_values(self) -> None:
        cfg = TrainingConfig(num_epochs=5, learning_rate=1e-4, label_smoothing=0.0)
        assert cfg.num_epochs == 5
        assert cfg.learning_rate == 1e-4
        assert cfg.label_smoothing == 0.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def tiny_model(self) -> XSALAKERTransformer:
        config = XSA_LAKER_Config(d_model=32, num_heads=2, dropout=0.0)
        return XSALAKERTransformer(config, num_layers=1, vocab_size=50, max_seq_len=16)

    @pytest.fixture
    def trainer(self, tiny_model: XSALAKERTransformer) -> Trainer:
        cfg = TrainingConfig()
        return Trainer(tiny_model, cfg, torch.device("cpu"))

    def test_train_step_returns_loss(self, trainer: Trainer) -> None:
        batch = (
            torch.randint(0, 50, (2, 16)),
            torch.randint(0, 50, (2, 16)),
        )
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert metrics["loss"] > 0
        assert trainer.step_count == 1

    def test_compute_loss(self, trainer: Trainer) -> None:
        logits = torch.randn(2, 16, 50)
        labels = torch.randint(0, 50, (2, 16))
        loss = trainer.compute_loss(logits, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_evaluate(self, trainer: Trainer, tiny_model: XSALAKERTransformer) -> None:
        data = TensorDataset(
            torch.randint(0, 50, (8, 16)),
            torch.randint(0, 50, (8, 16)),
        )
        loader = DataLoader(data, batch_size=2)
        metrics = trainer.evaluate(loader)
        assert "eval_loss" in metrics
        assert metrics["eval_loss"] > 0

    def test_train_epoch(self, trainer: Trainer) -> None:
        data = TensorDataset(
            torch.randint(0, 50, (8, 16)),
            torch.randint(0, 50, (8, 16)),
        )
        loader = DataLoader(data, batch_size=2)
        metrics = trainer.train_epoch(loader)
        assert "epoch_loss" in metrics
        assert "elapsed" in metrics
        assert metrics["steps"] > 0

    def test_multiple_steps_no_nan(self, trainer: Trainer) -> None:
        for _ in range(5):
            batch = (
                torch.randint(0, 50, (2, 16)),
                torch.randint(0, 50, (2, 16)),
            )
            metrics = trainer.train_step(batch)
            assert torch.isfinite(torch.tensor(metrics["loss"]))
