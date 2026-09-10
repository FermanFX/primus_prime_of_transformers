"""Unit tests for ``predict.py`` (seismic model inference CLI).

Every external boundary that ``main()`` touches — YAML config parsing,
checkpoint / MLflow model loading, HDF5 I/O, and pick extraction — is
mocked so the suite is fast, deterministic, and does not require real
model weights or seismic data files on disk.

"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path
from types import TracebackType
from typing import Any, Literal
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Dynamic import of the module under test.
#
# ``predict.py`` may live at the project root or inside a ``scripts/``
# directory (as in this project's layout). We search a few conventional
# locations and add the likely project root to ``sys.path`` so that the
# module's own ``from src... import ...`` statements resolve correctly.
# ---------------------------------------------------------------------------


def _load_predict_module() -> Any:
    here = Path(__file__).resolve().parent

    for path in (here, here.parent, here.parent.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    candidates = [
        here / "predict.py",
        here / "scripts" / "predict.py",
        here.parent / "scripts" / "predict.py",
        here.parent / "predict.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("predict", candidate)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules["predict"] = module
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        "Could not locate predict.py. Update the `candidates` list in "
        "_load_predict_module() to point at its actual location."
    )


predict = _load_predict_module()


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeH5File:
    """Stand-in for ``h5py.File`` supporting the context-manager protocol
    and the nested ``__getitem__`` chain used by ``predict.py``.

    Calling the instance mimics ``h5py.File(path, mode)``: it records the
    call arguments (for assertions) and either raises a configured error
    (to simulate a corrupt/missing file) or returns itself as the context
    manager, whose ``__enter__`` yields the fake nested payload.
    """

    def __init__(self, payload: dict, raise_on_open: Exception | None = None) -> None:
        self._payload = payload
        self._raise_on_open = raise_on_open
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> _FakeH5File:
        self.calls.append((args, kwargs))
        if self._raise_on_open is not None:
            raise self._raise_on_open
        return self

    def __enter__(self) -> dict:
        return self._payload

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        return False


def _assert_success(result: Any) -> None:
    """Assert a CLI invocation succeeded, surfacing the *full* traceback
    of any unexpected exception raised inside main().

    Click's CliRunner swallows exceptions and only exposes them via
    ``result.exception`` / ``result.exc_info``; without this, a failing
    test only shows the outer assertion, not the actual line inside
    predict.py that raised the error.
    """
    if result.exit_code != 0:
        if result.exception is not None:
            tb = "".join(traceback.format_exception(*result.exc_info))
            pytest.fail(f"CLI command failed with an exception:\n{tb}")
        pytest.fail(
            f"CLI command failed with exit code {result.exit_code}. "
            f"Output:\n{result.output}"
        )


def _cli_args(
    model: str,
    input_: str,
    output: str,
    config: Path,
    device: str = "cpu",
    batch_size: int = 4,
    chunk_size: int = 100,
) -> list[str]:
    return [
        "--model",
        model,
        "--input",
        input_,
        "--output",
        output,
        "--config",
        str(config),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--chunk-size",
        str(chunk_size),
    ]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """A Click test runner used to invoke the ``main`` command."""
    return CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A real (content-irrelevant, since ``yaml.safe_load`` is mocked)
    config file so ``open(config, "r")`` succeeds."""
    path = tmp_path / "config.yaml"
    path.write_text("model_name: tiny_unet\n")
    return path


@pytest.fixture
def make_h5_payload():
    """Factory building the nested dict structure that mimics the HDF5
    group layout ``f["TRACE_DATA"]["DEFAULT"]["data_array"]``."""

    def _make(data: np.ndarray) -> dict:
        return {"TRACE_DATA": {"DEFAULT": {"data_array": data}}}

    return _make


@pytest.fixture
def fake_traces() -> np.ndarray:
    """A small deterministic seismic trace array: 7 traces x 50 samples."""
    return np.arange(7 * 50, dtype=np.float32).reshape(7, 50)


@pytest.fixture
def mock_model() -> MagicMock:
    """A fake model returning random logits shaped (batch, 3, samples).

    ``.to()`` and ``.eval()`` return the same mock so the reassignment
    pattern in ``main()`` (``model_obj = model_obj.to(device)``) keeps
    referring to this fixture's mock.
    """
    model = MagicMock(name="model")
    model.to.return_value = model
    model.eval.return_value = model
    model.load_state_dict.return_value = None

    def _forward(chunk_tensor: torch.Tensor) -> torch.Tensor:
        batch, _channels, samples = chunk_tensor.shape
        return torch.randn(batch, 3, samples)

    model.side_effect = _forward
    return model


@pytest.fixture
def patch_common(monkeypatch: pytest.MonkeyPatch, mock_model: MagicMock) -> dict:
    """Patch every external dependency ``main()`` touches except HDF5
    I/O, which individual tests configure with a specific fake payload.
    """
    fake_cfg = MagicMock(model_name="tiny_unet")

    monkeypatch.setattr(
        predict.yaml, "safe_load", MagicMock(return_value={"model_name": "tiny_unet"})
    )
    monkeypatch.setattr(predict, "SeismicConfig", MagicMock(return_value=fake_cfg))
    monkeypatch.setattr(predict, "create_model", MagicMock(return_value=mock_model))
    monkeypatch.setattr(
        predict.torch, "load", MagicMock(return_value={"model_state_dict": {}})
    )
    monkeypatch.setattr(
        predict,
        "extract_picks_from_mask",
        MagicMock(side_effect=lambda pred: list(range(pred.shape[0]))),
    )
    monkeypatch.setattr(predict, "logger", MagicMock())

    return {"cfg": fake_cfg, "model": mock_model}


# ---------------------------------------------------------------------------
# Config / model loading
# ---------------------------------------------------------------------------


def test_main_parses_config_and_instantiates_seismicconfig_should_forward_model_name(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_traces: np.ndarray,
    make_h5_payload,
) -> None:
    """SeismicConfig should be built from the parsed YAML dict, and
    create_model should receive cfg.model_name plus the fixed channel
    counts."""
    # Arrange
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(fake_traces)))
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
        ),
    )

    # Assert
    _assert_success(result)
    predict.yaml.safe_load.assert_called_once()
    predict.SeismicConfig.assert_called_once_with(model_name="tiny_unet")
    predict.create_model.assert_called_once_with(
        "tiny_unet", in_channels=1, out_channels=3
    )


def test_main_loads_local_checkpoint_when_model_is_not_mlflow_uri_should_call_torch_load(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_traces: np.ndarray,
    make_h5_payload,
) -> None:
    """When --model is a plain file path, torch.load must be used and
    the resulting state dict loaded into the freshly created model."""
    # Arrange
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(fake_traces)))
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
            device="cpu",
        ),
    )

    # Assert
    _assert_success(result)
    predict.torch.load.assert_called_once_with("checkpoint.pt", map_location="cpu")
    patch_common["model"].load_state_dict.assert_called_once_with({})


def test_main_loads_model_via_mlflow_when_uri_is_prefixed_should_skip_torch_load(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_traces: np.ndarray,
    make_h5_payload,
) -> None:
    """When --model starts with 'models:/', the model must be fetched via
    mlflow.pytorch.load_model instead of torch.load."""
    # Arrange: inject a fake mlflow module, since it is imported lazily
    # inside main() only for this branch.
    fake_load_model = MagicMock(return_value=patch_common["model"])
    fake_mlflow_pytorch = MagicMock(load_model=fake_load_model)
    fake_mlflow = MagicMock(pytorch=fake_mlflow_pytorch)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.pytorch", fake_mlflow_pytorch)
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(fake_traces)))
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="models:/my-model/1",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
        ),
    )

    # Assert
    _assert_success(result)
    fake_load_model.assert_called_once_with("models:/my-model/1")
    predict.torch.load.assert_not_called()


def test_main_raises_key_error_when_checkpoint_missing_model_state_dict(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_traces: np.ndarray,
    make_h5_payload,
) -> None:
    """A malformed checkpoint (missing the expected key) should surface
    as a KeyError rather than being silently swallowed."""
    # Arrange
    predict.torch.load.return_value = {}  # no "model_state_dict" key
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(fake_traces)))
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
        ),
    )

    # Assert
    assert result.exit_code != 0
    assert isinstance(result.exception, KeyError)


# ---------------------------------------------------------------------------
# HDF5 input reading
# ---------------------------------------------------------------------------


def test_main_reads_trace_data_from_expected_hdf5_group_should_open_input_path(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_traces: np.ndarray,
    make_h5_payload,
) -> None:
    """main() must open the given input path in read mode and pull data
    from TRACE_DATA/DEFAULT/data_array."""
    # Arrange
    fake_file = _FakeH5File(make_h5_payload(fake_traces))
    monkeypatch.setattr(predict.h5py, "File", fake_file)
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
        ),
    )

    # Assert
    _assert_success(result)
    assert fake_file.calls == [(("input.h5", "r"), {})]
    log_messages = [c.args[0] for c in predict.logger.info.call_args_list]
    assert any("7 traces" in msg for msg in log_messages)


def test_main_propagates_error_when_hdf5_file_is_invalid(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt/missing HDF5 file should propagate its OSError rather
    than being masked."""
    # Arrange
    monkeypatch.setattr(
        predict.h5py,
        "File",
        _FakeH5File({}, raise_on_open=OSError("Unable to open file")),
    )
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="bad.h5",
            output=str(output_path),
            config=config_file,
        ),
    )

    # Assert
    assert result.exit_code != 0
    assert isinstance(result.exception, OSError)


def test_main_fails_when_config_file_does_not_exist(
    runner: CliRunner, patch_common: dict, tmp_path: Path
) -> None:
    """A missing --config path should raise FileNotFoundError from the
    plain `open()` call before any parsing happens."""
    # Arrange
    missing_config = tmp_path / "does_not_exist.yaml"
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=missing_config,
        ),
    )

    # Assert
    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


def test_main_saves_output_as_npy_when_output_path_ends_with_npy(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_traces: np.ndarray,
    make_h5_payload,
) -> None:
    """A ``.npy`` output path should be saved via np.save with one pick
    per input trace."""
    # Arrange
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(fake_traces)))
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
        ),
    )

    # Assert
    _assert_success(result)
    assert output_path.exists()
    saved = np.load(output_path)
    np.testing.assert_array_equal(saved, np.arange(7))


def test_main_saves_output_as_csv_with_expected_columns(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_traces: np.ndarray,
    make_h5_payload,
) -> None:
    """A ``.csv`` output path should produce trace_idx/pick_sample
    columns via pandas."""
    # Arrange
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(fake_traces)))
    output_path = tmp_path / "picks.csv"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
        ),
    )

    # Assert
    _assert_success(result)
    df = pd.read_csv(output_path)
    assert list(df.columns) == ["trace_idx", "pick_sample"]
    np.testing.assert_array_equal(df["trace_idx"].to_numpy(), np.arange(7))
    np.testing.assert_array_equal(df["pick_sample"].to_numpy(), np.arange(7))


def test_main_falls_back_to_csv_when_output_extension_is_unrecognized(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_traces: np.ndarray,
    make_h5_payload,
) -> None:
    """Documents current behaviour: any extension other than '.npy'
    falls through to the CSV branch, even e.g. '.dat'."""
    # Arrange
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(fake_traces)))
    output_path = tmp_path / "picks.dat"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
        ),
    )

    # Assert
    _assert_success(result)
    df = pd.read_csv(output_path)
    assert list(df.columns) == ["trace_idx", "pick_sample"]
    assert len(df) == 7


def test_main_creates_missing_output_parent_directories(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_traces: np.ndarray,
    make_h5_payload,
) -> None:
    """Nested, not-yet-existing output directories should be created
    automatically."""
    # Arrange
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(fake_traces)))
    output_path = tmp_path / "nested" / "dir" / "picks.npy"
    assert not output_path.parent.exists()

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
        ),
    )

    # Assert
    _assert_success(result)
    assert output_path.parent.exists()
    assert output_path.exists()


# ---------------------------------------------------------------------------
# Chunking behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_traces", "chunk_size", "expected_forward_calls"),
    [
        (10, 5, 2),  # evenly divisible
        (11, 5, 3),  # trailing partial chunk
        (1, 100, 1),  # fewer traces than chunk size
        (100, 10, 10),  # many equal chunks
    ],
)
def test_main_processes_traces_in_expected_number_of_chunks(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_h5_payload,
    n_traces: int,
    chunk_size: int,
    expected_forward_calls: int,
) -> None:
    """The number of forward passes through the model should equal
    ceil(n_traces / chunk_size), and every trace should get exactly one
    pick in the output."""
    # Arrange
    data = np.arange(n_traces * 20, dtype=np.float32).reshape(n_traces, 20)
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(data)))
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
            chunk_size=chunk_size,
        ),
    )

    # Assert
    _assert_success(result)
    assert patch_common["model"].call_count == expected_forward_calls
    saved = np.load(output_path)
    assert len(saved) == n_traces


def test_main_handles_zero_traces_without_error(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_h5_payload,
) -> None:
    """Empty input data (0 traces) should not crash and should produce
    an empty picks array without ever invoking the model."""
    # Arrange
    data = np.empty((0, 20), dtype=np.float32)
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(data)))
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
        ),
    )

    # Assert
    _assert_success(result)
    saved = np.load(output_path)
    assert saved.shape == (0,)
    patch_common["model"].assert_not_called()


def test_main_logs_progress_every_ten_chunks(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_h5_payload,
) -> None:
    """Progress should be logged whenever the running total is a
    multiple of chunk_size * 10, matching the code's own condition."""
    # Arrange
    chunk_size = 10
    n_traces = chunk_size * 25  # 250 traces -> 25 chunks
    data = np.zeros((n_traces, 20), dtype=np.float32)
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(data)))
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
            chunk_size=chunk_size,
        ),
    )

    # Assert
    _assert_success(result)
    progress_logs = [
        call.args[0]
        for call in predict.logger.info.call_args_list
        if "Processed" in call.args[0]
    ]
    # Triggers when (i + chunk_size) % (chunk_size * 10) == 0 -> i in {90, 190}
    assert len(progress_logs) == 2


def test_main_accepts_batch_size_option_without_affecting_chunk_count(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_h5_payload,
) -> None:
    """Documents current behaviour: --batch-size is accepted by the CLI
    but the chunking loop is driven solely by --chunk-size."""
    # Arrange
    data = np.zeros((6, 20), dtype=np.float32)
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(data)))
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
            batch_size=2,
            chunk_size=3,
        ),
    )

    # Assert
    _assert_success(result)
    assert patch_common["model"].call_count == 2  # 6 traces / chunk_size 3


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


def test_main_passes_selected_device_to_checkpoint_loading(
    runner: CliRunner,
    patch_common: dict,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_traces: np.ndarray,
    make_h5_payload,
) -> None:
    """The --device option should be forwarded to torch.load's
    map_location argument."""
    # Arrange
    monkeypatch.setattr(predict.h5py, "File", _FakeH5File(make_h5_payload(fake_traces)))
    output_path = tmp_path / "picks.npy"

    # Act
    result = runner.invoke(
        predict.main,
        _cli_args(
            model="checkpoint.pt",
            input_="input.h5",
            output=str(output_path),
            config=config_file,
            device="cpu",
        ),
    )

    # Assert
    _assert_success(result)
    predict.torch.load.assert_called_once_with("checkpoint.pt", map_location="cpu")
