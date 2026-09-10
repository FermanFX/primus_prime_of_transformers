"""
Unit tests for :mod:`scripts.batch_train`.

Covers:
    * ``load_batch_config`` — YAML loading, defaults, dataset overrides.
    * ``is_memory_error`` — MPS / CUDA / non-memory / MLflow detection.
    * ``is_real_error`` — MLflow INFO/WARNING filtering.
    * ``check_memory_usage`` — system memory reporting.
    * ``train_dataset`` — subprocess command, env, success/error handling.

The tests use :mod:`unittest.mock` to avoid spawning real subprocesses and
to keep behaviour deterministic across machines.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

# Ensure the repository root is importable when running pytest from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import batch_train

# ============================================================
# HELPERS
# ============================================================


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    """Write a YAML file and return its path."""
    path.write_text(yaml.safe_dump(data))
    return path


# ============================================================
# 1. load_batch_config — YAML LOADING
# ============================================================


class TestLoadBatchConfigYaml:
    """Tests for :func:`batch_train.load_batch_config` (YAML loading)."""

    def test_loads_yaml_file(self, tmp_path: Path) -> None:
        """A valid YAML file must be loaded and returned as a dict."""
        config_file = _write_yaml(
            tmp_path / "batch_config.yaml",
            {
                "global": {
                    "epochs": 42,
                    "device": "cpu",
                    "verbose": True,
                },
                "datasets": {
                    "Halfmile": {"epochs": 10},
                },
            },
        )

        config = batch_train.load_batch_config(str(config_file))

        assert config["global"]["epochs"] == 42
        assert config["global"]["device"] == "cpu"
        assert config["global"]["verbose"] is True
        assert config["datasets"]["Halfmile"]["epochs"] == 10

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing YAML file must raise ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            batch_train.load_batch_config(str(tmp_path / "does_not_exist.yaml"))

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Malformed YAML must raise ``yaml.YAMLError``."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("global: [unclosed")

        with pytest.raises(yaml.YAMLError):
            batch_train.load_batch_config(str(bad))


# ============================================================
# 2. load_batch_config — DEFAULTS
# ============================================================


class TestLoadBatchConfigDefaults:
    """Tests that :func:`load_batch_config` applies default values."""

    def test_empty_yaml_gets_all_defaults(self, tmp_path: Path) -> None:
        """An empty YAML file must produce a config with all defaults set."""
        config_file = _write_yaml(tmp_path / "empty.yaml", {})

        config = batch_train.load_batch_config(str(config_file))

        # Global defaults
        g = config["global"]
        assert g["epochs"] == 30
        assert g["device"] == "mps"
        assert g["log_memory"] is False
        assert g["verbose"] is False
        assert g["log_level"] == "INFO"
        assert g["preprocess"] is False
        assert g["checkpoint_every"] == 5
        assert g["early_stopping"] == 5
        assert g["timeout_seconds"] == 7200
        assert g["skip_failed"] is True
        assert g["max_retries"] == 3
        assert g["clear_memory_between_datasets"] is True
        assert g["pause_between_datasets"] == 2

        # Other sections default to empty/typical values
        assert config["variants"] == []
        assert "monitoring" in config
        assert config["monitoring"]["memory_warning_threshold_gb"] == 16.0
        assert config["monitoring"]["memory_critical_threshold_gb"] == 20.0
        assert config["monitoring"]["system_memory_percent_warning"] == 80
        assert config["monitoring"]["system_memory_percent_critical"] == 90

    def test_partial_global_keeps_missing_defaults(self, tmp_path: Path) -> None:
        """Values absent from ``global`` must fall back to defaults."""
        config_file = _write_yaml(
            tmp_path / "partial.yaml",
            {"global": {"epochs": 7}},
        )

        config = batch_train.load_batch_config(str(config_file))

        assert config["global"]["epochs"] == 7
        # Default not overridden
        assert config["global"]["device"] == "mps"
        assert config["global"]["verbose"] is False

    def test_missing_global_section_gets_defaults(self, tmp_path: Path) -> None:
        """A YAML file with no ``global`` key must still get defaults."""
        config_file = _write_yaml(
            tmp_path / "no_global.yaml",
            {"datasets": {"Halfmile": {"epochs": 3}}},
        )

        config = batch_train.load_batch_config(str(config_file))

        assert config["global"]["epochs"] == 30
        assert config["global"]["device"] == "mps"
        assert config["datasets"]["Halfmile"]["epochs"] == 3


# ============================================================
# 3. load_batch_config — DATASET OVERRIDES
# ============================================================


class TestLoadBatchConfigDatasetOverrides:
    """Tests that dataset overrides survive loading unchanged."""

    def test_dataset_overrides_preserved(self, tmp_path: Path) -> None:
        """Per-dataset values must be preserved in the returned config."""
        config_file = _write_yaml(
            tmp_path / "overrides.yaml",
            {
                "global": {"epochs": 30},
                "datasets": {
                    "Brunswick": {"epochs": 40, "log_memory": True},
                    "Sudbury": {"epochs": 25, "device": "cpu"},
                },
            },
        )

        config = batch_train.load_batch_config(str(config_file))

        assert config["datasets"]["Brunswick"]["epochs"] == 40
        assert config["datasets"]["Brunswick"]["log_memory"] is True
        assert config["datasets"]["Sudbury"]["epochs"] == 25
        assert config["datasets"]["Sudbury"]["device"] == "cpu"

    def test_auto_section_preserved(self, tmp_path: Path) -> None:
        """The ``auto`` section must pass through unchanged."""
        config_file = _write_yaml(
            tmp_path / "auto.yaml",
            {
                "auto": {
                    "strategy": "smart",
                    "memory_usage": 0.85,
                    "loss_overrides": {
                        "Lalor": {"loss_function": "dice"},
                    },
                },
            },
        )

        config = batch_train.load_batch_config(str(config_file))

        assert config["auto"]["strategy"] == "smart"
        assert config["auto"]["memory_usage"] == 0.85
        assert config["auto"]["loss_overrides"]["Lalor"]["loss_function"] == "dice"


# ============================================================
# 4. is_memory_error
# ============================================================


class TestIsMemoryError:
    """Tests for :func:`batch_train.is_memory_error`."""

    def test_mps_out_of_memory(self) -> None:
        """An MPS out-of-memory message must be detected."""
        msg = "RuntimeError: MPS out of memory. Tried to allocate 2.5 GB."
        assert batch_train.is_memory_error(msg) is True

    def test_cuda_out_of_memory(self) -> None:
        """A CUDA out-of-memory message must be detected."""
        msg = (
            "torch.cuda.OutOfMemoryError: CUDA out of memory. "
            "Tried to allocate 1.2 GiB."
        )
        assert batch_train.is_memory_error(msg) is True

    def test_generic_out_of_memory(self) -> None:
        """A generic ``out of memory`` message must be detected."""
        assert (
            batch_train.is_memory_error(
                "RuntimeError: out of memory while allocating tensor"
            )
            is True
        )

    def test_memory_error_class_name(self) -> None:
        """``MemoryError`` must be detected."""
        assert batch_train.is_memory_error("MemoryError: cannot allocate") is True

    def test_non_memory_error(self) -> None:
        """A non-memory error must NOT be flagged as a memory error."""
        msg = "RuntimeError: shape mismatch in conv2d"
        assert batch_train.is_memory_error(msg) is False

    def test_mlflow_info_message_not_memory_error(self) -> None:
        """Pure MLflow INFO lines must not be treated as memory errors."""
        msg = (
            "2024-01-01 12:00:00 | INFO | mlflow.tracking | Run started\n"
            "2024-01-01 12:00:01 | INFO | mlflow.tracking | Logging metrics"
        )
        assert batch_train.is_memory_error(msg) is False

    def test_mlflow_warning_not_memory_error(self) -> None:
        """MLflow WARNING lines must not be treated as memory errors."""
        msg = "2024-01-01 12:00:00 | WARNING | mlflow.tracking | Autologging failed"
        assert batch_train.is_memory_error(msg) is False

    def test_empty_string_not_memory_error(self) -> None:
        """An empty message must not be a memory error."""
        assert batch_train.is_memory_error("") is False


# ============================================================
# 5. is_real_error
# ============================================================


class TestIsRealError:
    """Tests for :func:`batch_train.is_real_error`."""

    def test_real_python_exception(self) -> None:
        """A genuine Python exception must be treated as a real error."""
        output = "Traceback (most recent call last):\nRuntimeError: boom"
        assert batch_train.is_real_error(output) is True

    def test_value_error(self) -> None:
        """``ValueError`` must be detected."""
        assert batch_train.is_real_error("ValueError: bad value") is True

    def test_file_not_found_error(self) -> None:
        """``FileNotFoundError`` must be detected."""
        assert (
            batch_train.is_real_error("FileNotFoundError: [Errno 2] No such file")
            is True
        )

    def test_mlflow_info_only_not_real_error(self) -> None:
        """A stream containing only MLflow INFO lines must not be a real error."""
        output = (
            "2024-01-01 12:00:00 | INFO | mlflow.tracking | Run started\n"
            "2024-01-01 12:00:01 | INFO | mlflow.tracking | Metrics logged\n"
            "2024-01-01 12:00:02 | INFO | mlflow.tracking | Run ended"
        )
        assert batch_train.is_real_error(output) is False

    def test_mlflow_warning_only_not_real_error(self) -> None:
        """MLflow WARNING lines alone must not be a real error."""
        output = (
            "2024-01-01 12:00:00 | WARNING | mlflow.tracking | System metrics disabled"
        )
        assert batch_train.is_real_error(output) is False

    def test_mlflow_info_and_real_error(self) -> None:
        """A mixed stream with a real error must be a real error."""
        output = (
            "2024-01-01 12:00:00 | INFO | mlflow.tracking | Run started\n"
            "RuntimeError: shape mismatch\n"
            "2024-01-01 12:00:01 | INFO | mlflow.tracking | Run ended"
        )
        assert batch_train.is_real_error(output) is True

    def test_mlflow_info_and_warning_mixed_not_real_error(self) -> None:
        """MLflow INFO/WARNING mixed without a real error must not flag."""
        output = (
            "2024-01-01 12:00:00 | INFO | mlflow.tracking | Run started\n"
            "2024-01-01 12:00:01 | WARNING | mlflow.tracking | Retry\n"
            "2024-01-01 12:00:02 | INFO | mlflow.tracking | Run ended"
        )
        assert batch_train.is_real_error(output) is False

    def test_empty_output_not_real_error(self) -> None:
        """An empty output must not be a real error."""
        assert batch_train.is_real_error("") is False

    def test_exit_code_1_detected(self) -> None:
        """An explicit ``exit(1)`` / ``sys.exit(1)`` must be detected."""
        assert batch_train.is_real_error("sys.exit(1)") is True
        assert batch_train.is_real_error("exit(1)") is True

    def test_failed_with_exit_code_detected(self) -> None:
        """A ``failed with exit code`` message must be detected."""
        assert batch_train.is_real_error("Command failed with exit code 1") is True


# ============================================================
# 6. check_memory_usage
# ============================================================


class TestCheckMemoryUsage:
    """Tests for :func:`batch_train.check_memory_usage`."""

    def test_returns_required_keys(self) -> None:
        """The returned dict must contain the documented keys."""
        usage = batch_train.check_memory_usage()

        for key in ("total_gb", "available_gb", "used_gb", "percent"):
            assert key in usage, f"Missing key: {key}"

    def test_total_gb_positive(self) -> None:
        """``total_gb`` must be positive on any real machine."""
        usage = batch_train.check_memory_usage()
        assert usage["total_gb"] > 0

    def test_available_gb_non_negative(self) -> None:
        """``available_gb`` must be non-negative."""
        usage = batch_train.check_memory_usage()
        assert usage["available_gb"] >= 0

    def test_used_gb_non_negative(self) -> None:
        """``used_gb`` must be non-negative."""
        usage = batch_train.check_memory_usage()
        assert usage["used_gb"] >= 0

    def test_percent_in_range(self) -> None:
        """``percent`` must be in ``[0, 100]``."""
        usage = batch_train.check_memory_usage()
        assert 0 <= usage["percent"] <= 100

    def test_used_plus_available_le_total(self) -> None:
        """``used_gb + available_gb`` must not exceed ``total_gb`` by much."""
        usage = batch_train.check_memory_usage()
        # Allow small rounding slack because values come from separate fields.
        assert usage["used_gb"] + usage["available_gb"] <= usage["total_gb"] * 1.05

    def test_gpu_key_present(self) -> None:
        """The ``gpu`` sub-dict must always be present (possibly empty)."""
        usage = batch_train.check_memory_usage()
        assert "gpu" in usage
        assert isinstance(usage["gpu"], dict)


# ============================================================
# 7. train_dataset — COMMAND BUILDING
# ============================================================


class TestTrainDatasetCommandBuilding:
    """Tests that :func:`train_dataset` builds the expected command."""

    def _run_with_captured_cmd(
        self,
        dataset_name: str,
        config_variant: dict[str, Any],
        global_config: dict[str, Any],
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Run ``train_dataset`` with a mocked subprocess and return the cmd."""
        captured: dict[str, list[str]] = {}

        def fake_run(cmd: list[str], **kwargs: Any) -> mock.Mock:
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs  # type: ignore[assignment]
            result = mock.Mock()
            result.returncode = 0
            result.stdout = "OK"
            result.stderr = ""
            return result

        with mock.patch.object(batch_train.subprocess, "run", side_effect=fake_run):
            batch_train.train_dataset(
                dataset_name=dataset_name,
                config_variant=config_variant,
                global_config=global_config,
                extra_args=extra_args,
            )

        return captured["cmd"]

    def test_basic_command_structure(self) -> None:
        """The command must start with ``python3.12 scripts/train.py``."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Halfmile",
            config_variant={"model": "unet"},
            global_config={},
        )

        assert cmd[0] == "python3.12"
        assert cmd[1] == "scripts/train.py"
        assert "--config" in cmd
        assert "--model" in cmd

    def test_uses_dataset_config_file(self) -> None:
        """The ``--config`` value must match the dataset's YAML path."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Brunswick",
            config_variant={"model": "mpslight"},
            global_config={},
        )

        idx = cmd.index("--config")
        assert cmd[idx + 1] == batch_train.DATASET_CONFIGS["Brunswick"]["config_file"]

    def test_class_weights_split_into_three_args(self) -> None:
        """``class_weights`` string must be split into three separate args."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Halfmile",
            config_variant={
                "model": "unet",
                "class_weights": "0.1,0.1,0.8",
            },
            global_config={},
        )

        idx = cmd.index("--class-weights")
        assert cmd[idx + 1 : idx + 4] == ["0.1", "0.1", "0.8"]

    def test_batch_size_override(self) -> None:
        """``batch_size`` must be forwarded as ``--batch-size``."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Halfmile",
            config_variant={"model": "unet", "batch_size": 2},
            global_config={},
        )

        idx = cmd.index("--batch-size")
        assert cmd[idx + 1] == "2"

    def test_global_epochs_forwarded(self) -> None:
        """Global epochs must be forwarded as ``--epochs``."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Halfmile",
            config_variant={"model": "unet"},
            global_config={"epochs": 12},
        )

        idx = cmd.index("--epochs")
        assert cmd[idx + 1] == "12"

    def test_global_device_forwarded(self) -> None:
        """Global device must be forwarded as ``--device``."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Halfmile",
            config_variant={"model": "unet"},
            global_config={"device": "cpu"},
        )

        idx = cmd.index("--device")
        assert cmd[idx + 1] == "cpu"

    def test_log_memory_flag(self) -> None:
        """``log_memory=True`` must add ``--log-memory``."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Halfmile",
            config_variant={"model": "unet"},
            global_config={"log_memory": True},
        )
        assert "--log-memory" in cmd

    def test_verbose_flag(self) -> None:
        """``verbose=True`` must add ``--verbose``."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Halfmile",
            config_variant={"model": "unet"},
            global_config={"verbose": True},
        )
        assert "--verbose" in cmd

    def test_log_level_forwarded_when_non_default(self) -> None:
        """A non-default log level must be forwarded."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Halfmile",
            config_variant={"model": "unet"},
            global_config={"log_level": "DEBUG"},
        )
        idx = cmd.index("--log-level")
        assert cmd[idx + 1] == "DEBUG"

    def test_log_level_default_not_forwarded(self) -> None:
        """The default ``INFO`` level must NOT be forwarded."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Halfmile",
            config_variant={"model": "unet"},
            global_config={"log_level": "INFO"},
        )
        assert "--log-level" not in cmd

    def test_preprocess_flag(self) -> None:
        """``preprocess=True`` must add ``--preprocess``."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Halfmile",
            config_variant={"model": "unet"},
            global_config={"preprocess": True},
        )
        assert "--preprocess" in cmd

    def test_extra_args_appended(self) -> None:
        """Extra args must be appended at the end of the command."""
        cmd = self._run_with_captured_cmd(
            dataset_name="Halfmile",
            config_variant={"model": "unet"},
            global_config={},
            extra_args=["--custom-flag", "value"],
        )
        assert cmd[-2:] == ["--custom-flag", "value"]


# ============================================================
# 8. train_dataset — ENVIRONMENT VARIABLES
# ============================================================


class TestTrainDatasetEnvironment:
    """Tests that :func:`train_dataset` sets expected env vars."""

    def _run_and_capture_env(
        self,
        config_variant: dict[str, Any],
    ) -> dict[str, str]:
        captured: dict[str, Any] = {}

        def fake_run(cmd: list[str], **kwargs: Any) -> mock.Mock:
            captured["env"] = kwargs.get("env", {})
            result = mock.Mock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with mock.patch.object(batch_train.subprocess, "run", side_effect=fake_run):
            batch_train.train_dataset(
                dataset_name="Halfmile",
                config_variant=config_variant,
                global_config={},
            )

        return captured["env"]

    def test_default_mps_memory_limit(self) -> None:
        """Without ``memory_limit_gb``, the default 8 GB must be used."""
        env = self._run_and_capture_env({"model": "unet"})
        assert env["PYTORCH_MPS_MEMORY_LIMIT"] == str(int(8 * 1e9))

    def test_custom_mps_memory_limit(self) -> None:
        """A custom ``memory_limit_gb`` must be reflected in the env var."""
        env = self._run_and_capture_env({"model": "unet", "memory_limit_gb": 4})
        assert env["PYTORCH_MPS_MEMORY_LIMIT"] == str(int(4 * 1e9))

    def test_cuda_alloc_conf_set(self) -> None:
        """``PYTORCH_CUDA_ALLOC_CONF`` must be set."""
        env = self._run_and_capture_env({"model": "unet"})
        assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


# ============================================================
# 9. train_dataset — SUCCESS HANDLING
# ============================================================


class TestTrainDatasetSuccess:
    """Tests that :func:`train_dataset` handles success correctly."""

    def _run_with_result(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> dict[str, Any]:
        def fake_run(cmd: list[str], **kwargs: Any) -> mock.Mock:
            result = mock.Mock()
            result.returncode = returncode
            result.stdout = stdout
            result.stderr = stderr
            return result

        with mock.patch.object(batch_train.subprocess, "run", side_effect=fake_run):
            return batch_train.train_dataset(
                dataset_name="Halfmile",
                config_variant={"model": "unet"},
                global_config={},
            )

    def test_success_returncode_zero(self) -> None:
        """A clean exit must be reported as success."""
        result = self._run_with_result(0, "Training complete", "")

        assert result["success"] is True
        assert result["return_code"] == 0
        assert result["dataset"] == "Halfmile"
        assert result["error"] is None

    def test_success_includes_output(self) -> None:
        """The result must include a truncated output snippet."""
        result = self._run_with_result(0, "All good", "")
        assert "All good" in result["output"]

    def test_success_includes_duration(self) -> None:
        """The result must include a non-negative duration."""
        result = self._run_with_result(0, "OK", "")
        assert result["duration"] >= 0


# ============================================================
# 10. train_dataset — ERROR HANDLING
# ============================================================


class TestTrainDatasetErrorHandling:
    """Tests that :func:`train_dataset` handles errors correctly."""

    def _run_with_result(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> dict[str, Any]:
        def fake_run(cmd: list[str], **kwargs: Any) -> mock.Mock:
            result = mock.Mock()
            result.returncode = returncode
            result.stdout = stdout
            result.stderr = stderr
            return result

        with mock.patch.object(batch_train.subprocess, "run", side_effect=fake_run):
            return batch_train.train_dataset(
                dataset_name="Halfmile",
                config_variant={"model": "unet"},
                global_config={},
            )

    def test_nonzero_returncode_with_real_error(self) -> None:
        """A non-zero return code with a real error must be a failure."""
        result = self._run_with_result(1, "", "RuntimeError: boom")

        assert result["success"] is False
        assert result["return_code"] == 1
        assert result["error"] is not None

    def test_nonzero_returncode_with_only_mlflow_info_is_success(self) -> None:
        """A non-zero return code with only MLflow noise must be a success."""
        mlflow_only = (
            "2024-01-01 12:00:00 | INFO | mlflow.tracking | Run ended\n"
            "2024-01-01 12:00:01 | WARNING | mlflow.tracking | cleanup"
        )
        result = self._run_with_result(1, mlflow_only, "")

        assert result["success"] is True
        assert result["error"] is None

    def test_exception_during_run_is_failure(self) -> None:
        """An exception raised by ``subprocess.run`` must produce a failure."""
        with mock.patch.object(
            batch_train.subprocess,
            "run",
            side_effect=OSError("cannot spawn process"),
        ):
            result = batch_train.train_dataset(
                dataset_name="Halfmile",
                config_variant={"model": "unet"},
                global_config={},
            )

        assert result["success"] is False
        assert result["return_code"] == -1
        assert "cannot spawn process" in result["error"]

    def test_error_message_is_captured(self) -> None:
        """The error message must include the combined output."""
        result = self._run_with_result(
            1,
            "some stdout",
            "ValueError: invalid shape",
        )
        assert result["error"] is not None
        assert "ValueError" in result["error"]

    def test_output_is_truncated_to_2000_chars(self) -> None:
        """The ``output`` field must be truncated to 2000 characters."""
        long_stdout = "x" * 10_000
        result = self._run_with_result(0, long_stdout, "")
        assert len(result["output"]) <= 2000
