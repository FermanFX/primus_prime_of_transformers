"""
Professional unit test suite for scripts/export_model.py.
Covers CLI validation, checkpoint loading, model export (TorchScript & ONNX),
config parsing, and error handling using pytest and unittest.mock.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import click.testing
import pytest

from scripts.export_model import main

@pytest.fixture
def cli_runner() -> click.testing.CliRunner:
    """Fixture providing an isolated Click CLI runner."""
    return click.testing.CliRunner()


@pytest.fixture
def mock_model() -> MagicMock:
    """Fixture providing a mock PyTorch model object."""
    model = MagicMock()
    model.to.return_value = model
    return model


def test_main_should_exit_with_error_when_no_export_format_specified(
    cli_runner: click.testing.CliRunner,
) -> None:
    """Test that main exits with status code 1 when neither --onnx nor --torchscript is provided."""
    # Arrange
    # Act
    result = cli_runner.invoke(main, ["--model", "dummy_checkpoint.pt"])

    # Assert
    assert result.exit_code == 1
    assert "ERROR: Please specify at least one export format" in result.output


@patch("scripts.export_model.create_model")
@patch("torch.load")
@patch("torch.jit.trace")
@patch("torch.jit.save")
def test_main_should_export_torchscript_successfully_when_valid_checkpoint_provided(
    mock_jit_save: MagicMock,
    mock_jit_trace: MagicMock,
    mock_torch_load: MagicMock,
    mock_create_model: MagicMock,
    cli_runner: click.testing.CliRunner,
    mock_model: MagicMock,
    tmp_path: Path,
) -> None:
    """Test successful TorchScript export workflow under normal conditions (Happy Path)."""
    # Arrange
    mock_create_model.return_value = mock_model
    mock_torch_load.return_value = {
        "model_state_dict": {"layer.weight": None},
        "epoch": 10,
        "val_loss": 0.05,
    }

    scripted_model_mock = MagicMock()
    mock_jit_trace.return_value = scripted_model_mock

    output_dir = tmp_path / "exported_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    scripted_path = output_dir / "unet_model_scripted.pt"
    scripted_path.write_bytes(b"dummy binary data")

    # Act
    result = cli_runner.invoke(
        main,
        [
            "--model",
            "dummy_checkpoint.pt",
            "--output",
            str(output_dir),
            "--torchscript",
        ],
    )

    # Assert
    assert result.exit_code == 0
    mock_create_model.assert_called_once_with("unet", in_channels=1, out_channels=3)
    mock_torch_load.assert_called_once()
    mock_jit_trace.assert_called_once()
    mock_jit_save.assert_called_once_with(scripted_model_mock, scripted_path)


@patch("scripts.export_model.create_model")
@patch("torch.load")
@patch("torch.onnx.export")
def test_main_should_export_onnx_successfully_when_valid_checkpoint_provided(
    mock_onnx_export: MagicMock,
    mock_torch_load: MagicMock,
    mock_create_model: MagicMock,
    cli_runner: click.testing.CliRunner,
    mock_model: MagicMock,
    tmp_path: Path,
) -> None:
    """Test successful ONNX export workflow under normal conditions (Happy Path)."""
    # Arrange
    mock_create_model.return_value = mock_model
    mock_torch_load.return_value = {"model_state_dict": {}}

    output_dir = tmp_path / "exported_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_file = output_dir / "unet_model.onnx"
    onnx_file.write_bytes(b"dummy onnx data")

    # Act
    result = cli_runner.invoke(
        main,
        [
            "--model",
            "dummy_checkpoint.pt",
            "--output",
            str(output_dir),
            "--onnx",
        ],
    )

    # Assert
    assert result.exit_code == 0
    mock_onnx_export.assert_called_once()


@patch("scripts.export_model.create_model")
@patch("torch.load")
@patch("torch.jit.trace")
@patch("torch.jit.save")
def test_main_should_handle_direct_state_dict_checkpoint_when_key_missing(
    mock_jit_save: MagicMock,
    mock_jit_trace: MagicMock,
    mock_torch_load: MagicMock,
    mock_create_model: MagicMock,
    cli_runner: click.testing.CliRunner,
    mock_model: MagicMock,
    tmp_path: Path,
) -> None:
    """Test edge case where checkpoint is a direct state dictionary without 'model_state_dict' wrapper."""
    # Arrange
    mock_create_model.return_value = mock_model
    # Checkpoint is directly the state dict, lacking 'model_state_dict' key
    mock_torch_load.return_value = {"weight": "value"}

    output_dir = tmp_path / "exported_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    scripted_path = output_dir / "unet_model_scripted.pt"
    scripted_path.write_bytes(b"dummy data")

    # Act
    result = cli_runner.invoke(
        main,
        [
            "--model",
            "flat_checkpoint.pt",
            "--output",
            str(output_dir),
            "--torchscript",
        ],
    )

    # Assert
    assert result.exit_code == 0
    mock_model.load_state_dict.assert_called_once_with({"weight": "value"})


@patch("torch.load")
def test_main_should_exit_with_error_when_checkpoint_loading_fails(
    mock_torch_load: MagicMock,
    cli_runner: click.testing.CliRunner,
) -> None:
    """Test error handling when loading the checkpoint file triggers an exception (Error Handling)."""
    # Arrange
    mock_torch_load.side_effect = RuntimeError("Invalid file format")

    # Act
    result = cli_runner.invoke(
        main,
        [
            "--model",
            "corrupted.pt",
            "--torchscript",
        ],
    )

    # Assert
    assert result.exit_code == 1
    assert "Failed to load model" in result.output


@pytest.mark.parametrize("model_type", ["unet", "mpslight"])
@patch("scripts.export_model.create_model")
@patch("torch.load")
@patch("torch.jit.trace")
@patch("torch.jit.save")
def test_main_should_support_parameterized_model_types(
    mock_jit_save: MagicMock,
    mock_jit_trace: MagicMock,
    mock_torch_load: MagicMock,
    mock_create_model: MagicMock,
    model_type: str,
    cli_runner: click.testing.CliRunner,
    mock_model: MagicMock,
    tmp_path: Path,
) -> None:
    """Test model export with different architecture types using parameterization."""
    # Arrange
    mock_create_model.return_value = mock_model
    mock_torch_load.return_value = {"model_state_dict": {}}

    output_dir = tmp_path / "exported_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    scripted_path = output_dir / f"{model_type}_model_scripted.pt"
    scripted_path.write_bytes(b"dummy data")

    # Act
    result = cli_runner.invoke(
        main,
        [
            "--model",
            "dummy.pt",
            "--output",
            str(output_dir),
            "--torchscript",
            "--model-type",
            model_type,
        ],
    )

    # Assert
    assert result.exit_code == 0
    mock_create_model.assert_called_once_with(model_type, in_channels=1, out_channels=3)
