"""
Tests for MLflow integration.

Covers:
- MLflow run creation
- Parameter and tag logging
- Metrics and artifacts
- Model registry and versioning
- Model aliases
- Run information retrieval
- Model loading
- Model name utilities
- Context manager behavior
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.utils.mlflow_utils as mlflow_utils
from src.utils.mlflow_utils import (
    MLflowManager,
    format_model_name,
    format_registered_model_name,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def manager():
    """
    Create a lightweight MLflowManager for unit tests.

    The real __init__ is intentionally skipped so tests do not
    create a real MLflow database, enable autologging, or start
    system metric monitoring.
    """
    obj = object.__new__(MLflowManager)

    obj.experiment_name = "test-experiment"
    obj.experiment_id = "test-experiment-id"
    obj.current_run = None
    obj.run_id = None
    obj.client = MagicMock()

    return obj


# ============================================================
# Run / Parameters
# ============================================================


def test_start_run_logs_parameters_and_tags(monkeypatch, manager):
    """Verify configuration parameters and tags are logged."""

    fake_run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="run-123"
        )
    )

    start_run = MagicMock(return_value=fake_run)
    log_params = MagicMock()
    set_tags = MagicMock()
    set_tag = MagicMock()

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "start_run",
        start_run,
    )

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_params",
        log_params,
    )

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "set_tags",
        set_tags,
    )

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "set_tag",
        set_tag,
    )

    config = {
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 10,
    }

    tags = {
        "dataset": "Halfmile",
        "model": "unet",
    }

    run_id = manager.start_run(
        config_dict=config,
        run_name="test-run",
        tags=tags,
    )

    assert run_id == "run-123"
    assert manager.run_id == "run-123"
    assert manager.current_run is fake_run

    start_run.assert_called_once_with(
        run_name="test-run"
    )

    log_params.assert_called_once_with(config)
    set_tags.assert_called_once_with(tags)

    # start_run creates exactly one start_time tag.
    assert set_tag.call_count == 1
    assert set_tag.call_args.args[0] == "start_time"


def test_start_run_generates_deterministic_name(monkeypatch, manager):
    """Verify a run name is generated when no name is supplied."""

    fake_run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="run-456"
        )
    )

    start_run = MagicMock(return_value=fake_run)

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "start_run",
        start_run,
    )

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_params",
        MagicMock(),
    )

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "set_tag",
        MagicMock(),
    )

    config = {
        "batch_size": 8,
        "learning_rate": 0.0001,
    }

    manager.start_run(config)

    generated_name = start_run.call_args.kwargs["run_name"]

    assert generated_name.startswith("seismic_")
    assert len(generated_name) == len("seismic_") + 8


# ============================================================
# Metrics and Artifacts
# ============================================================


def test_log_metrics(monkeypatch, manager):
    """Verify metrics are logged with the correct step."""

    log_metrics = MagicMock()

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_metrics",
        log_metrics,
    )

    metrics = {
        "train_loss": 0.25,
        "val_loss": 0.30,
        "mean_iou": 0.72,
    }

    manager.log_metrics(
        metrics,
        step=5,
    )

    log_metrics.assert_called_once_with(
        metrics,
        step=5,
    )


def test_log_artifact(monkeypatch, manager, tmp_path):
    """Verify artifact files are logged to MLflow."""

    artifact = tmp_path / "prediction.png"
    artifact.write_text("test artifact")

    log_artifact = MagicMock()

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_artifact",
        log_artifact,
    )

    manager.log_artifact(
        str(artifact),
        "predictions",
    )

    log_artifact.assert_called_once_with(
        str(artifact),
        "predictions",
    )


# ============================================================
# Model Registry / Versioning
# ============================================================


def test_log_model_with_registry(monkeypatch, manager):
    """
    Verify model logging with registry support.

    Checks:
    - model name formatting
    - epoch suffix
    - registered model name
    - model version
    - model version tags
    """

    fake_model_info = SimpleNamespace(
        model_id="model-123",
        model_uri="models:/test_model/1",
        registered_model_version="1",
    )

    log_model = MagicMock(
        return_value=fake_model_info
    )

    monkeypatch.setattr(
        mlflow_utils.mlflow.pytorch,
        "log_model",
        log_model,
    )

    manager.client.set_model_version_tag = MagicMock()

    model = MagicMock()

    result = manager.log_model_with_registry(
        model=model,
        model_name="unet/test",
        dataset_name="Halfmile",
        step=10,
        registered_model_name="halfmile-unet",
        tags={"environment": "test"},
    )

    assert result["model_id"] == "model-123"
    assert result["model_uri"] == "models:/test_model/1"
    assert result["registered_model_version"] == "1"

    log_model.assert_called_once()

    kwargs = log_model.call_args.kwargs

    assert kwargs["pytorch_model"] is model
    assert kwargs["name"] == "unet_test_epoch_10"
    assert kwargs["registered_model_name"] == "halfmile-unet"
    assert kwargs["serialization_format"] == "pickle"

    # The implementation creates:
    # dataset, model_type, step, timestamp + custom tag.
    assert manager.client.set_model_version_tag.call_count == 5

    logged_tags = {
        call.kwargs["key"]: call.kwargs["value"]
        for call in manager.client.set_model_version_tag.call_args_list
    }

    assert logged_tags["dataset"] == "Halfmile"

    # Current implementation only splits model_name by "_".
    # "unet/test" therefore remains "unet/test".
    assert logged_tags["model_type"] == "unet/test"

    assert logged_tags["step"] == "10"
    assert logged_tags["environment"] == "test"
    assert "timestamp" in logged_tags


def test_log_model_without_registry(monkeypatch, manager):
    """Verify model logging works when no registry name is provided."""

    fake_model_info = SimpleNamespace(
        model_id="model-456",
        model_uri="models:/test_model",
    )

    log_model = MagicMock(
        return_value=fake_model_info
    )

    monkeypatch.setattr(
        mlflow_utils.mlflow.pytorch,
        "log_model",
        log_model,
    )

    model = MagicMock()

    result = manager.log_model_with_registry(
        model=model,
        model_name="unet",
        dataset_name="Brunswick",
    )

    assert result["model_id"] == "model-456"
    assert result["model_uri"] == "models:/test_model"

    kwargs = log_model.call_args.kwargs

    assert kwargs["pytorch_model"] is model
    assert kwargs["name"] == "unet"

    # The current implementation explicitly passes None.
    assert kwargs["registered_model_name"] is None

    assert kwargs["serialization_format"] == "pickle"


# ============================================================
# Model Alias Management
# ============================================================


def test_set_model_alias(manager):
    """Verify an alias is assigned to the requested model version."""

    manager.client.set_registered_model_alias = MagicMock()

    manager.set_model_alias(
        registered_model_name="halfmile-unet",
        alias="champion",
        version=3,
    )

    manager.client.set_registered_model_alias.assert_called_once_with(
        name="halfmile-unet",
        alias="champion",
        version="3",
    )



def test_get_model_by_alias(manager):
    """Verify a model version can be retrieved using an alias."""

    expected_model = SimpleNamespace(
        name="halfmile-unet",
        version="3",
        aliases=["champion"],
    )

    manager.client.get_model_version_by_alias = MagicMock(
        return_value=expected_model
    )

    result = manager.get_model_by_alias(
        "halfmile-unet",
        "champion",
    )

    assert result is expected_model
    assert result.version == "3"

    manager.client.get_model_version_by_alias.assert_called_once_with(
        name="halfmile-unet",
        alias="champion",
    )


def test_get_model_by_alias_returns_none_on_failure(manager):
    """Verify alias lookup handles MLflow errors safely."""

    manager.client.get_model_version_by_alias = MagicMock(
        side_effect=OSError("MLflow unavailable")
    )

    result = manager.get_model_by_alias(
        "halfmile-unet",
        "champion",
    )

    assert result is None


# ============================================================
# Run Information
# ============================================================


def test_get_run_metrics(manager):
    """Verify metrics, parameters, tags and run info are returned."""

    fake_run = SimpleNamespace(
        data=SimpleNamespace(
            metrics={
                "val_loss": 0.25,
                "mean_iou": 0.80,
            },
            params={
                "learning_rate": "0.001",
                "batch_size": "16",
            },
            tags={
                "dataset": "Halfmile",
            },
        ),
        info=SimpleNamespace(
            run_id="run-123"
        ),
    )

    manager.client.get_run = MagicMock(
        return_value=fake_run
    )

    result = manager.get_run_metrics("run-123")

    assert result["metrics"]["val_loss"] == 0.25
    assert result["metrics"]["mean_iou"] == 0.80

    assert result["params"]["learning_rate"] == "0.001"
    assert result["params"]["batch_size"] == "16"

    assert result["tags"]["dataset"] == "Halfmile"

    assert result["info"].run_id == "run-123"


def test_get_run_metrics_returns_empty_dict_on_failure(manager):
    """Verify failed run lookup returns an empty dictionary."""

    manager.client.get_run = MagicMock(
        side_effect=OSError("connection error")
    )

    result = manager.get_run_metrics("invalid-run")

    assert result == {}


# ============================================================
# Compare Runs
# ============================================================


def test_compare_runs(manager, monkeypatch):
    """Verify selected metrics are compared between runs."""

    def fake_get_run_metrics(run_id):
        data = {
            "run-1": {
                "metrics": {
                    "val_loss": 0.30,
                    "mean_iou": 0.70,
                }
            },
            "run-2": {
                "metrics": {
                    "val_loss": 0.20,
                    "mean_iou": 0.80,
                }
            },
        }

        return data[run_id]

    monkeypatch.setattr(
        manager,
        "get_run_metrics",
        fake_get_run_metrics,
    )

    result = manager.compare_runs(
        run_ids=["run-1", "run-2"],
        metric_names=["val_loss", "mean_iou"],
    )

    assert result["run-1"]["val_loss"] == 0.30
    assert result["run-1"]["mean_iou"] == 0.70

    assert result["run-2"]["val_loss"] == 0.20
    assert result["run-2"]["mean_iou"] == 0.80


# ============================================================
# Model Loading
# ============================================================


def test_load_model_from_uri(monkeypatch, manager):
    """Verify generic MLflow model loading."""

    expected_model = MagicMock()

    fake_pyfunc = MagicMock()
    fake_pyfunc.load_model.return_value = expected_model

    # Replace the whole pyfunc object so the real MLflow registry
    # is never contacted.
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "pyfunc",
        fake_pyfunc,
    )

    result = manager.load_model_from_uri(
        "models:/halfmile-unet/3"
    )

    assert result is expected_model

    fake_pyfunc.load_model.assert_called_once_with(
        "models:/halfmile-unet/3"
    )


def test_load_pytorch_model(monkeypatch, manager):
    """Verify PyTorch model loading."""

    expected_model = MagicMock()

    load_model = MagicMock(
        return_value=expected_model
    )

    monkeypatch.setattr(
        mlflow_utils.mlflow.pytorch,
        "load_model",
        load_model,
    )

    result = manager.load_pytorch_model(
        "models:/halfmile-unet/3"
    )

    assert result is expected_model

    load_model.assert_called_once_with(
        "models:/halfmile-unet/3"
    )


# ============================================================
# Utility Functions
# ============================================================


def test_format_model_name():
    """Verify model name formatting."""

    assert (
        format_model_name(
            "unet",
            "Halfmile",
        )
        == "unet_Halfmile"
    )

    assert (
        format_model_name(
            "unet",
            "Halfmile",
            "best",
        )
        == "unet_Halfmile_best"
    )


def test_format_registered_model_name():
    """Verify registered model names are converted to lowercase."""

    assert (
        format_registered_model_name("HalfMile")
        == "halfmile"
    )

    assert (
        format_registered_model_name("BRUNSWICK")
        == "brunswick"
    )


# ============================================================
# End Run
# ============================================================


def test_end_run(monkeypatch, manager):
    """Verify the active MLflow run is ended."""

    manager.current_run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="run-123"
        )
    )

    set_tag = MagicMock()
    end_run = MagicMock()

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "set_tag",
        set_tag,
    )

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "end_run",
        end_run,
    )

    manager.end_run()

    assert set_tag.call_count == 1
    assert set_tag.call_args.args[0] == "end_time"

    end_run.assert_called_once()


def test_end_run_without_active_run(monkeypatch, manager):
    """Verify end_run does nothing without an active run."""

    end_run = MagicMock()

    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "end_run",
        end_run,
    )

    manager.current_run = None

    manager.end_run()

    end_run.assert_not_called()


# ============================================================
# Context Manager
# ============================================================


def test_context_manager_calls_end_run(manager):
    """Verify MLflowManager works as a context manager."""

    manager.end_run = MagicMock()

    with manager:
        pass

    manager.end_run.assert_called_once()
