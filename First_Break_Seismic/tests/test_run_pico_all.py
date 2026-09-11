"""
Unit tests for :mod:`scripts.run_pico_all`.

Covers:
    * Command building — correct ``train.py`` invocation per dataset.
    * Dataset iteration — every dataset processed in order.
    * Subprocess behavior — success and failure returncodes.
    * Log file creation — timestamped log file per dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_pico_all

# ============================================================
# HELPERS
# ============================================================


class _FakeProc:
    """A minimal stand-in for the ``subprocess.Popen`` return value."""

    def __init__(self, lines: list[str], returncode: int = 0) -> None:
        self.stdout = iter(lines)
        self.returncode = returncode
        self.wait = mock.MagicMock(return_value=returncode)


def _run_main_with_mocks() -> tuple[list[list[str]], list[str]]:
    """Execute ``run_pico_all.main`` with subprocess/input/open mocked.

    Returns:
        captured_commands: All commands passed to ``Popen``.
        opened_log_paths: All log file paths passed to ``open(..., "w")``.
    """
    captured_commands: list[list[str]] = []
    opened_log_paths: list[str] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        captured_commands.append(list(cmd))
        return _FakeProc(["line 1\n", "line 2\n"], returncode=0)

    real_open = open

    def fake_open(path: Any, mode: str = "r", *args: Any, **kwargs: Any):
        if isinstance(path, str) and path.startswith("logs/pico_runs/"):
            opened_log_paths.append(path)
            return mock.MagicMock()
        return real_open(path, mode, *args, **kwargs)

    with (
        mock.patch.object(run_pico_all.subprocess, "Popen", side_effect=fake_popen),
        mock.patch.object(run_pico_all, "input", return_value=""),
        mock.patch("builtins.open", side_effect=fake_open),
    ):
        run_pico_all.main()

    return captured_commands, opened_log_paths


# ============================================================
# 1. DATASET CONFIGURATION
# ============================================================


class TestDatasetConfiguration:
    """Tests for the module-level ``DATASETS`` constant."""

    def test_datasets_is_a_list_of_strings(self) -> None:
        """``DATASETS`` must be a non-empty list of strings."""
        assert isinstance(run_pico_all.DATASETS, list)
        assert len(run_pico_all.DATASETS) > 0
        for dataset in run_pico_all.DATASETS:
            assert isinstance(dataset, str)

    def test_datasets_order_is_preserved(self) -> None:
        """The four expected datasets must appear in order."""
        assert run_pico_all.DATASETS == [
            "Halfmile",
            "Sudbury",
            "Brunswick",
            "Lalor",
        ]


# ============================================================
# 2. COMMAND BUILDING
# ============================================================


class TestCommandBuilding:
    """Tests that per-dataset commands are built correctly."""

    def test_one_command_per_dataset(self) -> None:
        """One ``Popen`` call must be made per dataset."""
        commands, _ = _run_main_with_mocks()
        assert len(commands) == len(run_pico_all.DATASETS)

    def test_command_starts_with_python(self) -> None:
        """Each command must begin with ``python3.12 scripts/train.py``."""
        commands, _ = _run_main_with_mocks()
        for cmd in commands:
            assert cmd[0] == "python3.12"
            assert cmd[1] == "scripts/train.py"

    def test_command_uses_pico_model(self) -> None:
        """Each command must use ``--model pico``."""
        commands, _ = _run_main_with_mocks()
        for cmd in commands:
            idx = cmd.index("--model")
            assert cmd[idx + 1] == "pico"

    def test_command_uses_correct_config_per_dataset(self) -> None:
        """Each command must reference the dataset's YAML config."""
        commands, _ = _run_main_with_mocks()
        expected = [f"configs/{d.lower()}.yaml" for d in run_pico_all.DATASETS]

        for cmd, expected_cfg in zip(commands, expected):
            idx = cmd.index("--config")
            assert cmd[idx + 1] == expected_cfg

    def test_command_includes_epochs_and_flags(self) -> None:
        """Each command must include epochs, verbose, log-memory, log-level."""
        commands, _ = _run_main_with_mocks()
        for cmd in commands:
            assert cmd[cmd.index("--epochs") + 1] == "1"
            assert "--verbose" in cmd
            assert "--log-memory" in cmd
            assert cmd[cmd.index("--log-level") + 1] == "DEBUG"


# ============================================================
# 3. DATASET ITERATION
# ============================================================


class TestDatasetIteration:
    """Tests that every dataset is processed in order."""

    def test_all_datasets_processed_in_order(self) -> None:
        """All four datasets must be processed in order."""
        commands, _ = _run_main_with_mocks()
        configs = [cmd[cmd.index("--config") + 1] for cmd in commands]
        assert configs == [
            "configs/halfmile.yaml",
            "configs/sudbury.yaml",
            "configs/brunswick.yaml",
            "configs/lalor.yaml",
        ]

    def test_input_called_between_datasets(self) -> None:
        """``input()`` must be called once per dataset to pause between runs."""
        with (
            mock.patch.object(
                run_pico_all.subprocess,
                "Popen",
                return_value=_FakeProc(["x\n"]),
            ),
            mock.patch.object(run_pico_all, "input", return_value="") as mocked_input,
            mock.patch("builtins.open", side_effect=mock.mock_open()),
        ):
            run_pico_all.main()

        assert mocked_input.call_count == len(run_pico_all.DATASETS)

    def test_process_wait_called_per_dataset(self) -> None:
        """Each spawned process must have ``wait()`` called exactly once."""
        procs: list[_FakeProc] = []

        def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
            p = _FakeProc(["x\n"])
            procs.append(p)
            return p

        with (
            mock.patch.object(run_pico_all.subprocess, "Popen", side_effect=fake_popen),
            mock.patch.object(run_pico_all, "input", return_value=""),
            mock.patch("builtins.open", side_effect=mock.mock_open()),
        ):
            run_pico_all.main()

        assert len(procs) == len(run_pico_all.DATASETS)
        for p in procs:
            assert p.wait.call_count == 1


# ============================================================
# 4. LOG FILE CREATION
# ============================================================


class TestLogFileCreation:
    """Tests that each dataset gets its own timestamped log file."""

    def test_log_file_opened_per_dataset(self) -> None:
        """One log file must be opened per dataset."""
        _, opened = _run_main_with_mocks()
        assert len(opened) == len(run_pico_all.DATASETS)

    def test_log_file_paths_reference_dataset_names(self) -> None:
        """Log file names must include each dataset name."""
        _, opened = _run_main_with_mocks()
        joined = " ".join(opened)
        for dataset in run_pico_all.DATASETS:
            assert f"pico_{dataset}_" in joined

    def test_log_files_under_pico_runs_dir(self) -> None:
        """Log files must live under ``logs/pico_runs/``."""
        _, opened = _run_main_with_mocks()
        for path in opened:
            assert path.startswith("logs/pico_runs/")


# ============================================================
# 5. SUBPROCESS SUCCESS / FAILURE
# ============================================================


class TestSubprocessHandling:
    """Tests that success and failure returncodes are handled."""

    def test_success_returncode_observable(self) -> None:
        """A subprocess with returncode 0 must not raise."""
        with (
            mock.patch.object(
                run_pico_all.subprocess,
                "Popen",
                return_value=_FakeProc(["ok\n"], returncode=0),
            ),
            mock.patch.object(run_pico_all, "input", return_value=""),
            mock.patch("builtins.open", side_effect=mock.mock_open()),
        ):
            run_pico_all.main()  # must not raise

    def test_failure_returncode_does_not_raise(self) -> None:
        """A subprocess with a non-zero returncode must not crash the runner."""
        with (
            mock.patch.object(
                run_pico_all.subprocess,
                "Popen",
                return_value=_FakeProc(["err\n"], returncode=1),
            ),
            mock.patch.object(run_pico_all, "input", return_value=""),
            mock.patch("builtins.open", side_effect=mock.mock_open()),
        ):
            run_pico_all.main()  # must not raise
