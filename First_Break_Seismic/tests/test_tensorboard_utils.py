"""Tests for TensorBoardManager utility."""

from collections.abc import Iterator  # Və ya typing import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.utils.tensorboard_utils import TensorBoardManager


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for TensorBoard logs."""
    return tmp_path / "logs"

@pytest.fixture
def tb_manager(log_dir: Path) -> Iterator[TensorBoardManager]:
    """Fixture providing a TensorBoardManager instance."""
    manager = TensorBoardManager(log_dir=str(log_dir), experiment_name="test_exp")
    yield manager
    manager.close()


def test_tensorboard_manager_initialization(log_dir: Path) -> None:
    """Test TensorBoardManager initialization and directory creation."""
    exp_name = "init_test_exp"
    expected_path = log_dir / exp_name

    manager = TensorBoardManager(log_dir=str(log_dir), experiment_name=exp_name)

    assert expected_path.exists()
    assert expected_path.is_dir()
    assert manager.log_dir == expected_path
    assert manager.step == 0

    manager.close()


def test_set_step(tb_manager: TensorBoardManager) -> None:
    """Test updating the step counter."""
    tb_manager.set_step(42)
    assert tb_manager.step == 42


def test_log_scalar(tb_manager: TensorBoardManager) -> None:
    """Test logging a scalar value writes to SummaryWriter correctly."""
    with patch.object(tb_manager.writer, "add_scalar") as mock_add_scalar:
        tb_manager.log_scalar("loss/train", 0.5, step=10)
        mock_add_scalar.assert_called_once_with("loss/train", 0.5, 10)

        # Test default step fallback
        tb_manager.set_step(5)
        tb_manager.log_scalar("loss/val", 0.8)
        mock_add_scalar.assert_called_with("loss/val", 0.8, 5)


def test_log_scalars(tb_manager: TensorBoardManager) -> None:
    """Test logging multiple scalars."""
    metrics = {"train": 0.2, "val": 0.4}
    with patch.object(tb_manager.writer, "add_scalars") as mock_add_scalars:
        tb_manager.log_scalars("Metrics/loss", metrics, step=1)
        mock_add_scalars.assert_called_once_with("Metrics/loss", metrics, 1)


def test_log_seismogram_without_predictions(tb_manager: TensorBoardManager) -> None:
    """Test log_seismogram without predictions parameter."""
    n_traces, n_samples = 20, 100
    data = np.random.randn(n_traces, n_samples).astype(np.float32)
    mask = np.random.randint(0, 3, size=(n_traces, n_samples))

    with patch.object(tb_manager.writer, "add_figure") as mock_add_figure:
        tb_manager.log_seismogram(data=data, mask=mask, predictions=None, shot_id=101, step=2)

        mock_add_figure.assert_called_once()
        args, _ = mock_add_figure.call_args
        assert args[0] == "Seismogram/Shot_101"
        assert args[2] == 2


def test_log_seismogram_with_predictions(tb_manager: TensorBoardManager) -> None:
    """Test log_seismogram with predictions parameter."""
    n_traces, n_samples = 20, 100
    data = np.random.randn(n_traces, n_samples).astype(np.float32)
    mask = np.random.randint(0, 3, size=(n_traces, n_samples))
    predictions = np.random.randint(0, 3, size=(n_traces, n_samples))

    with patch.object(tb_manager.writer, "add_figure") as mock_add_figure:
        tb_manager.log_seismogram(
            data=data,
            mask=mask,
            predictions=predictions,
            shot_id=202,
            step=3,
        )

        mock_add_figure.assert_called_once()
        args, _ = mock_add_figure.call_args
        assert args[0] == "Seismogram/Shot_202"
        assert args[2] == 3


def test_log_weights_histograms(tb_manager: TensorBoardManager) -> None:
    """Test log_weights_histograms iterates over trainable layers."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1),
    )

    # Freeze one parameter to verify requires_grad filtering
    next(iter(model.parameters())).requires_grad = False

    with patch.object(tb_manager.writer, "add_histogram") as mock_add_histogram:
        tb_manager.log_weights_histograms(model, step=15)

        # 4 total params (2 weights, 2 biases) - 1 frozen = 3 expected calls
        assert mock_add_histogram.call_count == 3


def test_additional_log_methods(tb_manager: TensorBoardManager) -> None:
    """Test remaining log utility methods (image, histogram, graph, etc.)."""
    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
    dummy_hist = np.array([1.0, 2.0, 3.0])
    dummy_model = torch.nn.Linear(2, 1)
    dummy_input = torch.randn(1, 2)

    with (
        patch.object(tb_manager.writer, "add_image") as mock_img,
        patch.object(tb_manager.writer, "add_histogram") as mock_hist,
        patch.object(tb_manager.writer, "add_graph") as mock_graph,
    ):
        tb_manager.log_image("img_tag", dummy_img, step=1)
        mock_img.assert_called_once_with("img_tag", dummy_img, 1, dataformats="HWC")

        tb_manager.log_histogram("hist_tag", dummy_hist, step=1)
        mock_hist.assert_called_once_with("hist_tag", dummy_hist, 1)

        tb_manager.log_graph(dummy_model, dummy_input)
        mock_graph.assert_called_once_with(dummy_model, dummy_input)


def test_flush_and_close(tb_manager: TensorBoardManager) -> None:
    """Test flush and close methods correctly invoke SummaryWriter methods."""
    tb_manager.writer.flush = MagicMock()
    tb_manager.writer.close = MagicMock()

    tb_manager.flush()
    tb_manager.writer.flush.assert_called_once()

    tb_manager.close()
    tb_manager.writer.close.assert_called_once()
