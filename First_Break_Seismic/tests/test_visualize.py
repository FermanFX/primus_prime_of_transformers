"""
Unit tests for the scripts.visualize module.
"""

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from click.testing import CliRunner
from pytest_mock import MockerFixture

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visualize import main


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing Click CLI Runner."""
    return CliRunner()


@pytest.fixture
def mock_base_dependencies(mocker: MockerFixture) -> dict[str, Any]:
    """
    Fixture to mock I/O, logging, configurations, and matplotlib plotting.
    Returns dictionary containing mocked objects for verification.
    """
    # 1. Mock Open and Safe Load
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch(
        "scripts.visualize.yaml.safe_load",
        return_value={
            "dataset_name": "Halfmile",
            "chunk_dir": "/mock/chunks",
            "model_name": "mpslight",
        },
    )

    # 2. Mock SeismicConfig Object
    mock_cfg = mocker.patch("scripts.visualize.SeismicConfig").return_value
    mock_cfg.dataset_name = "Halfmile"
    mock_cfg.chunk_dir = "/mock/chunks"
    mock_cfg.device = "cpu"
    mock_cfg.model_name = "mpslight"

    # 3. Mock Logging & Task Name
    mocker.patch("scripts.visualize.create_task_name", return_value="task_123")
    mocker.patch("scripts.visualize.setup_logger")

    # 4. Mock File System Operations
    mocker.patch("scripts.visualize.Path.exists", return_value=True)
    mocker.patch("scripts.visualize.Path.mkdir")
    mocker.patch("scripts.visualize.load_manifest", return_value={"mock": "manifest"})

    # 5. Mock Matplotlib Plotting
    mock_fig = mocker.MagicMock()
    mock_axes = [mocker.MagicMock(), mocker.MagicMock(), mocker.MagicMock()]
    mock_plt = mocker.patch("scripts.visualize.plt")
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    return {"cfg": mock_cfg, "plt": mock_plt}


@pytest.fixture
def mock_dataset_and_model(mocker: MockerFixture) -> dict[str, Any]:
    """Fixture to mock Dataset items and Model inference behavior."""
    # 1. Mock Test Dataset
    mock_dataset = mocker.MagicMock()
    mock_dataset.__len__.return_value = 3
    mock_dataset.get_shot_id.side_effect = ["shot_001", "shot_002", "shot_003"]

    # Valid Tensors: data shape (1, 10, 10), mask shape (10, 10)
    mock_data = torch.rand(1, 10, 10)
    mock_mask = torch.zeros(10, 10)
    mock_dataset.__getitem__.return_value = (mock_data, mock_mask)

    # 2. Mock Data Manager
    mock_data_manager = mocker.patch(
        "scripts.visualize.ChunkedDataManager"
    ).return_value
    mock_data_manager.get_dataset.return_value = mock_dataset

    # 3. Mock Neural Network Model
    mock_model = mocker.MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    # Inference output tensor shape (batch=1, classes=3, H=10, W=10)
    mock_model.return_value = torch.rand(1, 3, 10, 10)
    mocker.patch("scripts.visualize.create_model", return_value=mock_model)

    return {"dataset": mock_dataset, "model": mock_model}


def test_main_should_visualize_successfully_when_valid_inputs_provided(
    cli_runner: CliRunner,
    mock_base_dependencies: dict[str, Any],
    mock_dataset_and_model: dict[str, Any],
    mocker: MockerFixture,
) -> None:
    """Happy Path: Verifies successful visualization loop execution."""
    # Arrange
    mocker.patch(
        "scripts.visualize.torch.load",
        return_value={"model_state_dict": {"dummy": "weights"}},
    )
    mock_plt = mock_base_dependencies["plt"]

    # Act
    result = cli_runner.invoke(
        main, ["-c", "configs/default.yaml", "-m", "model.pt", "-n", "3", "-d", "cpu"]
    )

    # Assert
    assert result.exit_code == 0
    assert mock_plt.subplots.call_count == 3
    assert mock_plt.savefig.call_count == 3
    assert mock_plt.close.call_count == 3


def test_main_should_exit_with_error_when_manifest_not_found(
    cli_runner: CliRunner,
    mock_base_dependencies: dict[str, Any],
    mocker: MockerFixture,
) -> None:
    """Error Handling: Verifies script exits gracefully when manifest is missing."""
    # Arrange
    mocker.patch("scripts.visualize.Path.exists", return_value=False)

    # Act
    result = cli_runner.invoke(
        main, ["-c", "configs/default.yaml", "-m", "model.pt", "-n", "3", "-d", "cpu"]
    )

    # Assert
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)


def test_main_should_cap_visualization_to_dataset_length_when_n_samples_exceeds_it(
    cli_runner: CliRunner,
    mock_base_dependencies: dict[str, Any],
    mock_dataset_and_model: dict[str, Any],
    mocker: MockerFixture,
) -> None:
    """Edge Case: Verifies sample count is bounded by actual dataset length."""
    # Arrange
    mocker.patch("scripts.visualize.torch.load", return_value={})
    mock_plt = mock_base_dependencies["plt"]

    mock_dataset = mock_dataset_and_model["dataset"]
    mock_dataset.__len__.return_value = 2

    # Act
    result = cli_runner.invoke(
        main, ["-c", "configs/default.yaml", "-m", "model.pt", "-n", "10", "-d", "cpu"]
    )

    # Assert
    assert result.exit_code == 0
    assert mock_plt.savefig.call_count == 2


@pytest.mark.parametrize(
    "checkpoint_data, expected_key",
    [
        ({"model_state_dict": {"layer1": "weights"}}, True),
        ({"layer1": "weights"}, False),
    ],
    ids=["nested_state_dict", "raw_state_dict"],
)
def test_main_should_load_model_correctly_with_different_checkpoint_formats(
    cli_runner: CliRunner,
    mock_base_dependencies: dict[str, Any],
    mock_dataset_and_model: dict[str, Any],
    mocker: MockerFixture,
    checkpoint_data: dict[str, Any],
    expected_key: bool,
) -> None:
    """Edge Case: Tests both nested and raw state_dict loading branches."""
    # Arrange
    mocker.patch("scripts.visualize.torch.load", return_value=checkpoint_data)
    mock_model = mock_dataset_and_model["model"]

    # Act
    result = cli_runner.invoke(
        main, ["-c", "configs/default.yaml", "-m", "model.pt", "-n", "1", "-d", "cpu"]
    )

    # Assert
    assert result.exit_code == 0
    if expected_key:
        mock_model.load_state_dict.assert_called_once_with(
            checkpoint_data["model_state_dict"]
        )
    else:
        mock_model.load_state_dict.assert_called_once_with(checkpoint_data)


def test_script_entrypoint_execution(mocker: MockerFixture) -> None:
    """Coverage: Tests direct module execution (__name__ == '__main__')."""
    # Arrange: Patch sys.argv to prevent Click from reading pytest's flags
    mocker.patch.object(sys, "argv", ["visualize.py", "--help"])

    # Act & Assert
    import runpy

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("scripts.visualize", run_name="__main__")

    assert exc_info.value.code == 0
