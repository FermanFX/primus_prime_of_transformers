"""Unit tests for the seismic FBP evaluation script."""

from __future__ import annotations

import builtins
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts import evaluate


def _main_callback(**kwargs: Any) -> Any:
    """Invoke the Click command callback with a type-safe callable lookup."""
    callback = getattr(evaluate.main, "callback", None)
    if callback is None:
        raise AssertionError("evaluate.main does not expose a callback")
    return cast(Callable[..., Any], callback)(**kwargs)


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# MLflow test stub
# ---------------------------------------------------------------------------

if "mlflow" not in sys.modules:
    mlflow_stub = ModuleType("mlflow")
    mlflow_pytorch_stub = ModuleType("mlflow.pytorch")

    mlflow_stub.pytorch = mlflow_pytorch_stub  # type: ignore[attr-defined]

    sys.modules["mlflow"] = mlflow_stub
    sys.modules["mlflow.pytorch"] = mlflow_pytorch_stub


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeTensor:
    """Small tensor stub used to isolate evaluation logic from PyTorch."""

    def __init__(self, array: object = None) -> None:
        self.array = array

    def to(self, device: object) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> object:
        return self.array


class FakeModel:
    """Minimal model stub with the methods used by the evaluation script."""

    def __init__(self) -> None:
        self.eval_called = False
        self.loaded_state_dict: object = None
        self.device: object = None

    def to(self, device: object) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> FakeModel:
        self.eval_called = True
        return self

    def load_state_dict(self, state_dict: object) -> None:
        self.loaded_state_dict = state_dict

    def __call__(self, x: FakeTensor) -> FakeTensor:
        return FakeTensor([[0]])


class FakeDataset:
    """Dataset stub that provides the interfaces used by evaluation."""

    def __init__(
        self,
        length: int = 2,
        shot_ids: list[str] | None = None,
    ) -> None:
        self.length = length
        self.shot_ids = shot_ids

    def __len__(self) -> int:
        return self.length

    def get_shot_id(self, index: int) -> str:
        if self.shot_ids is None:
            raise AttributeError("shot_id is not available")

        return self.shot_ids[index]


class FakeSegmentationMetrics:
    """Metric stub with deterministic results for assertions."""

    instances: ClassVar[list[FakeSegmentationMetrics]] = []

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.update_calls: list[tuple[object, object]] = []
        self.__class__.instances.append(self)

    def update(
        self,
        predictions: object,
        targets: object,
    ) -> None:
        self.update_calls.append((predictions, targets))

    def compute(self) -> dict[str, object]:
        return {
            "accuracy": float(np.float32(0.95)),
            "mean_iou": float(np.float64(0.85)),
            "mean_f1": float(np.float64(0.90)),
            "iou_per_class": [
                float(np.float64(0.90)),
                float(np.float64(0.80)),
                float(np.float64(0.85)),
            ],
        }


class FakeFirstBreakMetrics:
    """First-break metric stub with deterministic results for assertions."""

    instances: ClassVar[list[FakeFirstBreakMetrics]] = []

    def __init__(self, tolerance_samples: int) -> None:
        self.tolerance_samples = tolerance_samples
        self.update_calls: list[tuple[object, object]] = []
        self.__class__.instances.append(self)

    def update(
        self,
        predicted: object,
        true: object,
    ) -> None:
        self.update_calls.append((predicted, true))

    def compute(self) -> dict[str, object]:
        return {
            "mean_absolute_error": float(np.float64(1.5)),
            "std_absolute_error": float(np.float64(0.5)),
            "median_absolute_error": float(np.float64(1.0)),
            "accuracy_within_tolerance": float(np.float64(0.8)),
            "total_traces": int(np.int64(4)),
        }


class FakeDataFrame:
    """DataFrame stub that records CSV output requests."""

    instances: ClassVar[list[FakeDataFrame]] = []

    def __init__(self, data: object) -> None:
        self.data = data
        self.to_csv_calls: list[tuple[object, bool]] = []
        self.__class__.instances.append(self)

    def to_csv(
        self,
        path: object,
        index: bool = True,
    ) -> None:
        self.to_csv_calls.append((path, index))


class FakeConfig:
    """Configuration stub exposing the attributes used by the script."""

    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

        self.chunk_dir = "/chunks"
        self.dataset_name = values.get(
            "dataset_name",
            "Halfmile",
        )
        self.device = "cpu"
        self.batch_size = 2
        self.model_name = "mpslight"


# ---------------------------------------------------------------------------
# Shared test setup
# ---------------------------------------------------------------------------


def _patch_common_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Patch dependencies shared by most main-function tests."""

    logger: MagicMock = MagicMock()

    data_manager: MagicMock = MagicMock()

    dataset = FakeDataset(
        length=2,
        shot_ids=["shot-001", "shot-002"],
    )

    data_manager.get_dataset.return_value = dataset

    model = FakeModel()

    mock_create_task_name: MagicMock = MagicMock(
        return_value="task-name",
    )

    mock_setup_logger: MagicMock = MagicMock(
        return_value=logger,
    )

    mock_load_manifest: MagicMock = MagicMock(
        return_value={"test": []},
    )

    mock_chunked_data_manager: MagicMock = MagicMock(
        return_value=data_manager,
    )

    mock_create_model: MagicMock = MagicMock(
        return_value=model,
    )

    extract_picks: MagicMock = MagicMock()

    def extract_picks_side_effect(
        _: object,
    ) -> list[int]:
        # evaluate.main calls this function once for predictions and once for
        # targets. Return one pick list per call, rather than a tuple containing
        # both lists.
        if extract_picks.call_count % 2 == 1:
            return [10, 20]

        return [11, 19]

    extract_picks.side_effect = extract_picks_side_effect

    mock_torch_argmax: MagicMock = MagicMock(
        return_value=FakeTensor(),
    )

    mock_torch_load: MagicMock = MagicMock(
        return_value={},
    )

    mock_mlflow_load_model: MagicMock = MagicMock(
        return_value=model,
    )

    mock_data_loader = lambda dataset, **_: [(FakeTensor(), FakeTensor())]

    monkeypatch.setattr(
        evaluate,
        "SeismicConfig",
        FakeConfig,
    )

    monkeypatch.setattr(
        evaluate,
        "create_task_name",
        mock_create_task_name,
    )

    monkeypatch.setattr(
        evaluate,
        "setup_logger",
        mock_setup_logger,
    )

    monkeypatch.setattr(
        evaluate,
        "load_manifest",
        mock_load_manifest,
    )

    monkeypatch.setattr(
        evaluate,
        "ChunkedDataManager",
        mock_chunked_data_manager,
    )

    monkeypatch.setattr(
        evaluate,
        "create_model",
        mock_create_model,
    )

    monkeypatch.setattr(
        evaluate,
        "SegmentationMetrics",
        FakeSegmentationMetrics,
    )

    monkeypatch.setattr(
        evaluate,
        "FirstBreakMetrics",
        FakeFirstBreakMetrics,
    )

    monkeypatch.setattr(
        evaluate,
        "extract_picks_from_mask",
        extract_picks,
    )

    monkeypatch.setattr(
        evaluate.Path,
        "exists",
        lambda _: True,
    )

    monkeypatch.setattr(
        evaluate.torch,
        "argmax",
        mock_torch_argmax,
    )

    monkeypatch.setattr(
        evaluate.torch.utils.data,
        "DataLoader",
        mock_data_loader,
    )

    monkeypatch.setattr(
        evaluate.pd,
        "DataFrame",
        FakeDataFrame,
    )

    monkeypatch.setattr(
        evaluate.torch,
        "load",
        mock_torch_load,
        raising=False,
    )

    monkeypatch.setattr(
        evaluate.mlflow.pytorch,
        "load_model",
        mock_mlflow_load_model,
        raising=False,
    )

    return {
        "logger": logger,
        "data_manager": data_manager,
        "dataset": dataset,
        "model": model,
        "create_model": mock_create_model,
        "load_model": mock_mlflow_load_model,
        "extract_picks": extract_picks,
    }


# ---------------------------------------------------------------------------
# Main evaluation tests
# ---------------------------------------------------------------------------


def test_main_should_evaluate_requested_split_and_save_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The default evaluation path should evaluate one split and save JSON results."""

    dependencies = _patch_common_dependencies(monkeypatch)

    yaml_load: MagicMock = MagicMock(
        return_value={"dataset_name": "Halfmile"},
    )

    monkeypatch.setattr(
        evaluate.yaml,
        "safe_load",
        yaml_load,
    )

    monkeypatch.setattr(
        builtins,
        "open",
        MagicMock(),
    )

    output_dir = tmp_path / "results"

    mkdir_mock: MagicMock = MagicMock()

    monkeypatch.setattr(
        evaluate.Path,
        "mkdir",
        mkdir_mock,
    )

    dump_mock: MagicMock = MagicMock()

    monkeypatch.setattr(
        evaluate.json,
        "dump",
        dump_mock,
    )

    monkeypatch.setattr(
        evaluate.torch,
        "device",
        lambda value: f"device:{value}",
    )

    _main_callback(
        config="config.yaml",
        model="model.pt",
        output=str(output_dir),
        device="cpu",
        batch_size=2,
        dataset=None,
        split="test",
        detailed=False,
    )

    yaml_load.assert_called_once()

    assert dependencies["data_manager"].get_dataset.call_count == 1

    assert dependencies["data_manager"].get_dataset.call_args.args == ("test",)

    assert dependencies["model"].eval_called is True

    dump_mock.assert_called_once()

    saved_results = dump_mock.call_args.args[0]

    assert saved_results["dataset"] == "Halfmile"

    assert saved_results["split_results"]["test"]["n_shots"] == 2

    assert isinstance(
        saved_results["split_results"]["test"]["segmentation"]["mean_iou"],
        float,
    )

    assert isinstance(
        saved_results["split_results"]["test"]["first_break"]["total_traces"],
        int,
    )

    mkdir_mock.assert_called_once_with(
        parents=True,
        exist_ok=True,
    )


def test_main_should_override_dataset_and_evaluate_all_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dataset override and all-splits branch should be applied to every split."""

    dependencies = _patch_common_dependencies(monkeypatch)

    monkeypatch.setattr(
        evaluate.yaml,
        "safe_load",
        MagicMock(return_value={}),
    )

    monkeypatch.setattr(
        builtins,
        "open",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.json,
        "dump",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.torch,
        "device",
        lambda value: f"device:{value}",
    )

    monkeypatch.setattr(
        evaluate.Path,
        "mkdir",
        MagicMock(),
    )

    _main_callback(
        config="config.yaml",
        model="model.pt",
        output="results",
        device="cpu",
        batch_size=8,
        dataset="CustomDataset",
        split="all",
        detailed=False,
    )

    assert dependencies["data_manager"].get_dataset.call_count == 3

    assert [
        call.args[0] for call in dependencies["data_manager"].get_dataset.call_args_list
    ] == [
        "train",
        "val",
        "test",
    ]

    assert dependencies["model"].eval_called is True


def test_main_should_save_detailed_errors_and_fallback_to_index_for_missing_shot_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detailed evaluation should create per-shot data and use batch index fallback."""

    dependencies = _patch_common_dependencies(monkeypatch)

    dependencies["dataset"] = FakeDataset(
        length=2,
        shot_ids=None,
    )

    dependencies["data_manager"].get_dataset.return_value = dependencies["dataset"]

    monkeypatch.setattr(
        evaluate.yaml,
        "safe_load",
        MagicMock(return_value={}),
    )

    monkeypatch.setattr(
        builtins,
        "open",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.json,
        "dump",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.torch,
        "device",
        lambda value: f"device:{value}",
    )

    monkeypatch.setattr(
        evaluate.Path,
        "mkdir",
        MagicMock(),
    )

    _main_callback(
        config="config.yaml",
        model="model.pt",
        output="results",
        device="cpu",
        batch_size=2,
        dataset=None,
        split="test",
        detailed=True,
    )

    detailed_frames = FakeDataFrame.instances

    assert detailed_frames

    data = next(
        frame.data
        for frame in reversed(detailed_frames)
        if isinstance(frame.data, dict) and "shot_id" in frame.data
    )

    assert data["shot_id"] == [0, 1]
    assert data["error_samples"] == [1, 1]

    assert np.array_equal(
        data["error_ms"],
        np.array([2, 2]),
    )

    assert detailed_frames[-1].to_csv_calls

    assert detailed_frames[-1].to_csv_calls[-1][1] is False


def test_main_should_load_local_checkpoint_with_state_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local checkpoint containing model_state_dict should load into model."""

    dependencies = _patch_common_dependencies(monkeypatch)

    checkpoint = {
        "model_state_dict": {
            "weight": 123,
        }
    }

    torch_load: MagicMock = MagicMock(
        return_value=checkpoint,
    )

    monkeypatch.setattr(
        evaluate.torch,
        "load",
        torch_load,
    )

    monkeypatch.setattr(
        evaluate.torch,
        "device",
        lambda value: f"device:{value}",
    )

    monkeypatch.setattr(
        evaluate.yaml,
        "safe_load",
        MagicMock(return_value={}),
    )

    monkeypatch.setattr(
        builtins,
        "open",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.json,
        "dump",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.Path,
        "mkdir",
        MagicMock(),
    )

    _main_callback(
        config="config.yaml",
        model="checkpoint.pt",
        output="results",
        device="cpu",
        batch_size=2,
        dataset=None,
        split="test",
        detailed=False,
    )

    torch_load.assert_called_once_with(
        "checkpoint.pt",
        map_location="device:cpu",
    )

    assert dependencies["model"].loaded_state_dict == {
        "weight": 123,
    }


def test_main_should_load_local_checkpoint_without_wrapper_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain state dictionary should be passed directly to the model."""

    dependencies = _patch_common_dependencies(monkeypatch)

    checkpoint = {
        "bias": 42,
    }

    monkeypatch.setattr(
        evaluate.torch,
        "load",
        MagicMock(return_value=checkpoint),
    )

    monkeypatch.setattr(
        evaluate.torch,
        "device",
        lambda value: f"device:{value}",
    )

    monkeypatch.setattr(
        evaluate.yaml,
        "safe_load",
        MagicMock(return_value={}),
    )

    monkeypatch.setattr(
        builtins,
        "open",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.json,
        "dump",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.Path,
        "mkdir",
        MagicMock(),
    )

    _main_callback(
        config="config.yaml",
        model="checkpoint.pt",
        output="results",
        device="cpu",
        batch_size=2,
        dataset=None,
        split="test",
        detailed=False,
    )

    assert dependencies["model"].loaded_state_dict == checkpoint


def test_main_should_load_mlflow_model_for_models_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A models URI should bypass local checkpoint creation."""

    dependencies = _patch_common_dependencies(monkeypatch)

    load_model: MagicMock = MagicMock(
        return_value=dependencies["model"],
    )

    create_model: MagicMock = MagicMock()

    monkeypatch.setattr(
        evaluate.mlflow.pytorch,
        "load_model",
        load_model,
    )

    monkeypatch.setattr(
        evaluate,
        "create_model",
        create_model,
    )

    monkeypatch.setattr(
        evaluate.torch,
        "device",
        lambda value: f"device:{value}",
    )

    monkeypatch.setattr(
        evaluate.yaml,
        "safe_load",
        MagicMock(return_value={}),
    )

    monkeypatch.setattr(
        builtins,
        "open",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.json,
        "dump",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.Path,
        "mkdir",
        MagicMock(),
    )

    _main_callback(
        config="config.yaml",
        model="models:/registered/model",
        output="results",
        device="cpu",
        batch_size=2,
        dataset=None,
        split="test",
        detailed=False,
    )

    load_model.assert_called_once_with(
        "models:/registered/model",
    )

    create_model.assert_not_called()


def test_main_should_use_champion_model_when_best_alias_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The best-model branch should prefer champion alias when available."""

    dependencies = _patch_common_dependencies(monkeypatch)

    manager = MagicMock()

    manager.get_model_by_alias.return_value = SimpleNamespace(
        model_id="champion",
    )

    monkeypatch.setattr(
        evaluate,
        "get_mlflow_manager",
        MagicMock(return_value=manager),
    )

    format_name: MagicMock = MagicMock(
        return_value="registered.Halfmile",
    )

    monkeypatch.setattr(
        evaluate,
        "format_registered_model_name",
        format_name,
    )

    load_model: MagicMock = MagicMock(
        return_value=dependencies["model"],
    )

    monkeypatch.setattr(
        evaluate.mlflow.pytorch,
        "load_model",
        load_model,
    )

    monkeypatch.setattr(
        evaluate.torch,
        "device",
        lambda value: f"device:{value}",
    )

    monkeypatch.setattr(
        evaluate.yaml,
        "safe_load",
        MagicMock(return_value={}),
    )

    monkeypatch.setattr(
        builtins,
        "open",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.json,
        "dump",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.Path,
        "mkdir",
        MagicMock(),
    )

    _main_callback(
        config="config.yaml",
        model="best",
        output="results",
        device="cpu",
        batch_size=2,
        dataset=None,
        split="test",
        detailed=False,
    )

    format_name.assert_called_once_with(
        "Halfmile",
    )

    manager.get_model_by_alias.assert_called_once_with(
        registered_model_name="registered.Halfmile",
        alias="champion",
    )

    load_model.assert_called_once_with(
        "models:/registered.Halfmile@champion",
    )


def test_main_should_fall_back_to_best_mlflow_model_when_champion_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The best-model branch should search by validation IoU when no champion."""

    dependencies = _patch_common_dependencies(monkeypatch)

    manager = MagicMock()

    manager.get_model_by_alias.return_value = None

    manager.search_models.return_value = [SimpleNamespace(model_id="model-123")]

    monkeypatch.setattr(
        evaluate,
        "get_mlflow_manager",
        MagicMock(return_value=manager),
    )

    monkeypatch.setattr(
        evaluate,
        "format_registered_model_name",
        MagicMock(return_value="registered.Halfmile"),
    )

    load_model: MagicMock = MagicMock(
        return_value=dependencies["model"],
    )

    monkeypatch.setattr(
        evaluate.mlflow.pytorch,
        "load_model",
        load_model,
    )

    monkeypatch.setattr(
        evaluate.torch,
        "device",
        lambda value: f"device:{value}",
    )

    monkeypatch.setattr(
        evaluate.yaml,
        "safe_load",
        MagicMock(return_value={}),
    )

    monkeypatch.setattr(
        builtins,
        "open",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.json,
        "dump",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.Path,
        "mkdir",
        MagicMock(),
    )

    _main_callback(
        config="config.yaml",
        model="best",
        output="results",
        device="cpu",
        batch_size=2,
        dataset=None,
        split="test",
        detailed=False,
    )

    manager.search_models.assert_called_once_with(
        filter_string="tags.dataset = 'Halfmile'",
        order_by=[
            {
                "field_name": "metrics.val_iou",
                "ascending": False,
            }
        ],
        max_results=1,
    )

    load_model.assert_called_once_with(
        "models:/model-123",
    )


def test_main_should_exit_when_manifest_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation should stop with exit code 1 when manifest is missing."""

    monkeypatch.setattr(
        evaluate.yaml,
        "safe_load",
        MagicMock(return_value={}),
    )

    monkeypatch.setattr(
        builtins,
        "open",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate,
        "SeismicConfig",
        FakeConfig,
    )

    monkeypatch.setattr(
        evaluate,
        "create_task_name",
        MagicMock(return_value="task-name"),
    )

    monkeypatch.setattr(
        evaluate,
        "setup_logger",
        MagicMock(return_value=MagicMock()),
    )

    monkeypatch.setattr(
        evaluate.Path,
        "exists",
        lambda _: False,
    )

    with pytest.raises(SystemExit) as exc_info:
        _main_callback(
            config="config.yaml",
            model="model.pt",
            output="results",
            device="cpu",
            batch_size=2,
            dataset=None,
            split="test",
            detailed=False,
        )

    assert exc_info.value.code == 1


@pytest.mark.parametrize(
    ("model", "load_error"),
    [
        ("best", RuntimeError("MLflow unavailable")),
        (
            "models:/registered/model",
            RuntimeError("Model download failed"),
        ),
        (
            "checkpoint.pt",
            RuntimeError("Corrupt checkpoint"),
        ),
    ],
)
def test_main_should_exit_when_model_loading_fails(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    load_error: Exception,
) -> None:
    """Every model loading route should convert failures into exit code 1."""

    _patch_common_dependencies(monkeypatch)

    monkeypatch.setattr(
        evaluate.yaml,
        "safe_load",
        MagicMock(return_value={}),
    )

    monkeypatch.setattr(
        builtins,
        "open",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.Path,
        "exists",
        lambda _: True,
    )

    monkeypatch.setattr(
        evaluate.Path,
        "mkdir",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.torch,
        "device",
        lambda value: f"device:{value}",
    )

    manager = MagicMock()

    manager.get_model_by_alias.return_value = SimpleNamespace(
        model_id="champion",
    )

    monkeypatch.setattr(
        evaluate,
        "get_mlflow_manager",
        MagicMock(return_value=manager),
    )

    monkeypatch.setattr(
        evaluate,
        "format_registered_model_name",
        MagicMock(return_value="registered.Halfmile"),
    )

    monkeypatch.setattr(
        evaluate.mlflow.pytorch,
        "load_model",
        MagicMock(side_effect=load_error),
    )

    monkeypatch.setattr(
        evaluate.torch,
        "load",
        MagicMock(side_effect=load_error),
    )

    with pytest.raises(SystemExit) as exc_info:
        _main_callback(
            config="config.yaml",
            model=model,
            output="results",
            device="cpu",
            batch_size=2,
            dataset=None,
            split="test",
            detailed=False,
        )

    assert exc_info.value.code == 1


def test_main_should_exit_when_no_best_mlflow_model_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The best-model branch should exit when no model exists."""

    _patch_common_dependencies(monkeypatch)

    manager = MagicMock()

    manager.get_model_by_alias.return_value = None
    manager.search_models.return_value = []

    monkeypatch.setattr(
        evaluate,
        "get_mlflow_manager",
        MagicMock(return_value=manager),
    )

    monkeypatch.setattr(
        evaluate,
        "format_registered_model_name",
        MagicMock(return_value="registered.Halfmile"),
    )

    monkeypatch.setattr(
        evaluate.yaml,
        "safe_load",
        MagicMock(return_value={}),
    )

    monkeypatch.setattr(
        builtins,
        "open",
        MagicMock(),
    )

    monkeypatch.setattr(
        evaluate.Path,
        "exists",
        lambda _: True,
    )

    monkeypatch.setattr(
        evaluate.torch,
        "device",
        lambda value: f"device:{value}",
    )

    with pytest.raises(SystemExit) as exc_info:
        _main_callback(
            config="config.yaml",
            model="best",
            output="results",
            device="cpu",
            batch_size=2,
            dataset=None,
            split="test",
            detailed=False,
        )

    assert exc_info.value.code == 1
