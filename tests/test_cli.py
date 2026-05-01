"""Smoke tests for CLI entry points (train, benchmark, evaluate)."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

from laker_xsa.cli.benchmark import main as benchmark_main
from laker_xsa.cli.train import main as train_main

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class TestCLITrain:
    """Smoke tests for training CLI."""

    def test_train_runs_minimal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", [
            "train",
            "--d-model", "32",
            "--num-heads", "2",
            "--num-layers", "1",
            "--vocab-size", "20",
            "--num-epochs", "1",
            "--batch-size", "2",
            "--seq-len", "8",
            "--num-samples", "10",
            "--attention-type", "standard",
            "--seed", "42",
        ])
        train_main()

    def test_train_with_xsa(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", [
            "train",
            "--d-model", "32",
            "--num-heads", "2",
            "--num-layers", "1",
            "--vocab-size", "20",
            "--num-epochs", "1",
            "--batch-size", "2",
            "--seq-len", "8",
            "--num-samples", "10",
            "--attention-type", "xsa",
            "--seed", "42",
        ])
        train_main()

    def test_train_with_kernel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", [
            "train",
            "--d-model", "32",
            "--num-heads", "2",
            "--num-layers", "1",
            "--vocab-size", "20",
            "--num-epochs", "1",
            "--batch-size", "2",
            "--seq-len", "8",
            "--num-samples", "10",
            "--attention-type", "kernel",
            "--seed", "42",
        ])
        train_main()

    def test_train_with_fused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", [
            "train",
            "--d-model", "32",
            "--num-heads", "2",
            "--num-layers", "1",
            "--vocab-size", "20",
            "--num-epochs", "1",
            "--batch-size", "2",
            "--seq-len", "8",
            "--num-samples", "10",
            "--attention-type", "fused",
            "--seed", "42",
        ])
        train_main()

    def test_train_with_fused_v2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", [
            "train",
            "--d-model", "32",
            "--num-heads", "2",
            "--num-layers", "1",
            "--vocab-size", "20",
            "--num-epochs", "1",
            "--batch-size", "2",
            "--seq-len", "8",
            "--num-samples", "10",
            "--attention-type", "fused_v2",
            "--seed", "42",
        ])
        train_main()


class TestCLIBenchmark:
    """Smoke tests for benchmark CLI."""

    def test_benchmark_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            monkeypatch.setattr(sys, "argv", [
                "benchmark",
                "--d-model", "64",
                "--num-heads", "2",
                "--num-runs", "2",
                "--output", output_path,
            ])
            benchmark_main()

            with open(output_path) as f:
                data = json.load(f)
            assert "config" in data
            assert "results" in data
        finally:
            os.unlink(output_path)


class TestCLIEvaluate:
    """Smoke tests for evaluate CLI (requires checkpoint fixture)."""

    def test_evaluate_needs_checkpoint(self) -> None:
        # evaluate requires --checkpoint; skipped if no checkpoint
        from laker_xsa.cli.evaluate import main as evaluate_main

        # Just confirm the module imports correctly
        assert callable(evaluate_main)
