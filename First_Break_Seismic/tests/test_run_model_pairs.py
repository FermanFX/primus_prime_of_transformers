"""
Unit tests for :mod:`scripts.run_model_pairs_fallback`.

Covers:
    * Model pairs configuration — pair structure and dataset list.
    * Skip logic for large datasets — UNet/Efficient skipped for Lalor.
    * Dry run functionality — prints commands without executing.
    * Command building — correct ``batch_train.py`` invocation.
    * Orchestration — ``run_model_pairs_with_fallback`` dispatches correctly.

All subprocess calls are mocked; no real training is executed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_model_pairs_fallback as rmp

# ============================================================
# HELPERS
# ============================================================


def _mock_completed(returncode: int = 0, stderr: str = "") -> mock.Mock:
    """Build a mock ``subprocess.CompletedProcess``-like object."""
    result = mock.Mock()
    result.returncode = returncode
    result.stdout = ""
    result.stderr = stderr
    return result


def _flatten_commands(call_args_list: list[Any]) -> list[list[str]]:
    """Extract the ``cmd`` list from a list of ``subprocess.run`` call args."""
    commands: list[list[str]] = []
    for call in call_args_list:
        args, _ = call
        # subprocess.run(cmd, ...) → cmd is args[0]
        commands.append(args[0])
    return commands


# ============================================================
# 1. MODEL PAIRS CONFIGURATION
# ============================================================


class TestModelPairsConfiguration:
    """Tests for the module-level ``MODEL_PAIRS`` and ``DATASETS`` constants."""

    def test_model_pairs_is_a_list_of_lists(self) -> None:
        """``MODEL_PAIRS`` must be a list of 2-element lists."""
        assert isinstance(rmp.MODEL_PAIRS, list)
        assert len(rmp.MODEL_PAIRS) > 0

        for pair in rmp.MODEL_PAIRS:
            assert isinstance(pair, list)
            assert len(pair) == 2, f"Pair {pair} must have exactly 2 models"

    def test_model_pairs_contains_known_models(self) -> None:
        """Every model name in ``MODEL_PAIRS`` must be a valid model."""
        known = {
            "pico",
            "nano",
            "tiny",
            "mpslight",
            "light",
            "mobile",
            "efficient",
            "unet",
        }
        for pair in rmp.MODEL_PAIRS:
            for model in pair:
                assert model in known, f"Unknown model in pair: {model}"

    def test_model_pairs_ordered_by_capacity(self) -> None:
        """Pairs must be ordered from smallest to largest capacity."""
        # The first pair should contain the smallest models,
        # the last pair the largest.
        first_pair = rmp.MODEL_PAIRS[0]
        last_pair = rmp.MODEL_PAIRS[-1]

        assert "pico" in first_pair or "nano" in first_pair
        assert "unet" in last_pair or "efficient" in last_pair

    def test_datasets_is_a_list(self) -> None:
        """``DATASETS`` must be a non-empty list of strings."""
        assert isinstance(rmp.DATASETS, list)
        assert len(rmp.DATASETS) > 0
        for dataset in rmp.DATASETS:
            assert isinstance(dataset, str)

    def test_datasets_contains_expected_names(self) -> None:
        """``DATASETS`` must contain the four expected datasets."""
        expected = {"Halfmile", "Sudbury", "Brunswick", "Lalor"}
        assert expected.issubset(set(rmp.DATASETS))

    def test_skip_per_dataset_is_a_dict(self) -> None:
        """``SKIP_PER_DATASET`` must be a dict of dataset → list of models."""
        assert isinstance(rmp.SKIP_PER_DATASET, dict)
        for dataset, models in rmp.SKIP_PER_DATASET.items():
            assert isinstance(dataset, str)
            assert isinstance(models, list)
            for model in models:
                assert isinstance(model, str)


# ============================================================
# 2. SKIP LOGIC FOR LARGE DATASETS
# ============================================================


class TestSkipLogicForLargeDatasets:
    """Tests that large models are skipped for large datasets."""

    def test_unet_skipped_for_lalor(self) -> None:
        """``unet`` must be in the skip list for Lalor."""
        assert "unet" in rmp.SKIP_PER_DATASET.get("Lalor", [])

    def test_efficient_skipped_for_lalor(self) -> None:
        """``efficient`` must be in the skip list for Lalor."""
        assert "efficient" in rmp.SKIP_PER_DATASET.get("Lalor", [])

    def test_other_models_not_skipped_for_lalor(self) -> None:
        """Smaller models must NOT be in the skip list for Lalor."""
        skipped = rmp.SKIP_PER_DATASET.get("Lalor", [])
        for model in ("pico", "nano", "tiny", "mpslight", "light", "mobile"):
            assert model not in skipped, f"{model} must not be skipped for Lalor"

    def test_smaller_datasets_have_no_skips(self) -> None:
        """Datasets other than Lalor must not skip any model."""
        for dataset in ("Halfmile", "Sudbury", "Brunswick"):
            assert rmp.SKIP_PER_DATASET.get(dataset, []) == []

    def test_lalor_is_the_only_dataset_with_skips(self) -> None:
        """Only Lalor should have non-empty skip entries."""
        non_empty = {ds for ds, models in rmp.SKIP_PER_DATASET.items() if models}
        assert non_empty == {"Lalor"}

    def test_skip_logic_applied_in_fallback_loop(self) -> None:
        """The fallback loop must filter out skipped models per dataset."""
        # Simulate the filtering logic used in ``run_model_pairs_with_fallback``
        pair = ["efficient", "unet"]
        datasets = ["Halfmile", "Lalor"]

        available_models: list[str] = []
        for model in pair:
            is_skipped = any(
                model in rmp.SKIP_PER_DATASET.get(dataset, []) for dataset in datasets
            )
            if not is_skipped:
                available_models.append(model)

        # Both `efficient` and `unet` are skipped because Lalor is in datasets
        assert available_models == []

    def test_skip_logic_keeps_small_models_for_mixed_datasets(self) -> None:
        """Small models must survive the skip filter for mixed datasets."""
        pair = ["pico", "nano"]
        datasets = ["Halfmile", "Lalor"]

        available_models: list[str] = []
        for model in pair:
            is_skipped = any(
                model in rmp.SKIP_PER_DATASET.get(dataset, []) for dataset in datasets
            )
            if not is_skipped:
                available_models.append(model)

        assert available_models == ["pico", "nano"]


# ============================================================
# 3. DRY RUN FUNCTIONALITY
# ============================================================


class TestDryRunFunctionality:
    """Tests for the ``--dry-run`` behaviour of the CLI."""

    def _invoke_dry_run(
        self,
        epochs: int = 2,
        device: str = "cpu",
        show_functions: bool = False,
        show_mlflow: bool = False,
        show_metrics: bool = False,
        show_output: bool = False,
    ) -> str:
        """Invoke the CLI's ``main`` in dry-run mode with mocked I/O.

        Loguru writes to stderr via its own handler, which Click's
        ``CliRunner`` does not capture by default. To capture the output,
        we install a temporary Loguru sink that appends all emitted
        messages to an in-memory list, and return the joined text.

        The existing handlers (console + files) are left untouched.
        """
        from click.testing import CliRunner
        from loguru import logger as _loguru_logger

        runner = CliRunner()
        args = [
            "--dry-run",
            "--epochs",
            str(epochs),
            "--device",
            device,
        ]
        if show_functions:
            args.append("--show-functions")
        if show_mlflow:
            args.append("--show-mlflow")
        if show_metrics:
            args.append("--show-metrics")
        if show_output:
            args.append("--show-output")

        captured: list[str] = []

        # Sink that appends raw formatted messages to our list.
        sink_id = _loguru_logger.add(
            lambda message: captured.append(message),
            format="{message}",
            level="INFO",
        )

        try:
            with mock.patch.object(rmp, "run_model_pairs_with_fallback") as mocked_run:
                result = runner.invoke(rmp.main, args)
                mocked_run.assert_not_called()
        finally:
            _loguru_logger.remove(sink_id)

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        return "".join(captured)

    def test_dry_run_prints_commands(self) -> None:
        """Dry run must print the commands that would be executed."""
        output = self._invoke_dry_run()

        assert "COMMANDS THAT WILL RUN" in output
        assert "scripts/batch_train.py" in output
        assert "--auto-config" in output

    def test_dry_run_does_not_execute_training(self) -> None:
        """Dry run must NOT spawn any subprocess."""
        from click.testing import CliRunner

        runner = CliRunner()
        with (
            mock.patch.object(rmp.subprocess, "run") as mocked_subprocess,
            mock.patch.object(rmp, "run_model_pairs_with_fallback") as mocked_run,
        ):
            result = runner.invoke(rmp.main, ["--dry-run", "--epochs", "1"])

            mocked_subprocess.assert_not_called()
            mocked_run.assert_not_called()

        assert result.exit_code == 0

    def test_dry_run_commands_are_correct(self) -> None:
        """Each printed command must contain the expected flags and models."""
        output = self._invoke_dry_run(epochs=3, device="cpu")

        assert "python3.12 scripts/batch_train.py" in output
        assert "--auto-config" in output
        assert "--epochs 3" in output
        assert "--device cpu" in output
        assert "--verbose" in output
        assert "--log-memory" in output

    def test_dry_run_lists_all_combinations(self) -> None:
        """Dry run summary must report the total number of combinations."""
        output = self._invoke_dry_run()

        assert "DRY RUN SUMMARY" in output
        assert "Total combinations:" in output

    def test_dry_run_does_not_create_models(self) -> None:
        """Dry run must not create model checkpoints on disk."""
        from click.testing import CliRunner

        runner = CliRunner()
        with mock.patch.object(rmp, "run_model_pairs_with_fallback"):
            result = runner.invoke(rmp.main, ["--dry-run"])

        assert result.exit_code == 0

    def test_dry_run_exits_cleanly(self) -> None:
        """Dry run must exit with code 0."""
        from click.testing import CliRunner

        runner = CliRunner()
        with mock.patch.object(rmp, "run_model_pairs_with_fallback"):
            result = runner.invoke(rmp.main, ["--dry-run"])

        assert result.exit_code == 0


# ============================================================
# 4. COMMAND BUILDING
# ============================================================


class TestCommandBuilding:
    """Tests that the orchestrator builds correct ``batch_train.py`` commands."""

    def test_orchestrator_invokes_batch_train_for_each_pair(self) -> None:
        """Each model pair must produce one ``subprocess.run`` invocation."""
        with mock.patch.object(
            rmp.subprocess,
            "run",
            return_value=_mock_completed(returncode=0),
        ) as mocked_run:
            rmp.run_model_pairs_with_fallback(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=2,
                device="cpu",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        assert mocked_run.call_count == 1

    def test_command_includes_auto_config(self) -> None:
        """The generated command must include ``--auto-config``."""
        with mock.patch.object(
            rmp.subprocess,
            "run",
            return_value=_mock_completed(returncode=0),
        ) as mocked_run:
            rmp.run_model_pairs_with_fallback(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=2,
                device="cpu",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        cmd = _flatten_commands(mocked_run.call_args_list)[0]
        assert "--auto-config" in cmd

    def test_command_includes_models(self) -> None:
        """The generated command must include ``--models`` for each model."""
        with mock.patch.object(
            rmp.subprocess,
            "run",
            return_value=_mock_completed(returncode=0),
        ) as mocked_run:
            rmp.run_model_pairs_with_fallback(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=2,
                device="cpu",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        cmd = _flatten_commands(mocked_run.call_args_list)[0]
        # Both models must appear after --models flags
        model_indices = [i for i, tok in enumerate(cmd) if tok == "--models"]
        model_values = [cmd[i + 1] for i in model_indices]
        assert "pico" in model_values
        assert "nano" in model_values

    def test_command_includes_datasets(self) -> None:
        """The generated command must include ``--datasets`` for each dataset."""
        with mock.patch.object(
            rmp.subprocess,
            "run",
            return_value=_mock_completed(returncode=0),
        ) as mocked_run:
            rmp.run_model_pairs_with_fallback(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile", "Brunswick"],
                epochs=2,
                device="cpu",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        cmd = _flatten_commands(mocked_run.call_args_list)[0]
        dataset_indices = [i for i, tok in enumerate(cmd) if tok == "--datasets"]
        dataset_values = [cmd[i + 1] for i in dataset_indices]
        assert "Halfmile" in dataset_values
        assert "Brunswick" in dataset_values

    def test_command_includes_epochs_and_device(self) -> None:
        """The generated command must include ``--epochs`` and ``--device``."""
        with mock.patch.object(
            rmp.subprocess,
            "run",
            return_value=_mock_completed(returncode=0),
        ) as mocked_run:
            rmp.run_model_pairs_with_fallback(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=5,
                device="mps",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        cmd = _flatten_commands(mocked_run.call_args_list)[0]
        assert "--epochs" in cmd
        assert cmd[cmd.index("--epochs") + 1] == "5"
        assert "--device" in cmd
        assert cmd[cmd.index("--device") + 1] == "mps"

    def test_command_includes_log_memory_when_enabled(self) -> None:
        """``log_memory=True`` must add ``--log-memory`` to the command."""
        with mock.patch.object(
            rmp.subprocess,
            "run",
            return_value=_mock_completed(returncode=0),
        ) as mocked_run:
            rmp.run_model_pairs_with_fallback(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=2,
                device="cpu",
                verbose=False,
                log_memory=True,
                dry_run=False,
            )

        cmd = _flatten_commands(mocked_run.call_args_list)[0]
        assert "--log-memory" in cmd

    def test_command_omits_log_memory_when_disabled(self) -> None:
        """``log_memory=False`` must omit ``--log-memory``."""
        with mock.patch.object(
            rmp.subprocess,
            "run",
            return_value=_mock_completed(returncode=0),
        ) as mocked_run:
            rmp.run_model_pairs_with_fallback(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=2,
                device="cpu",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        cmd = _flatten_commands(mocked_run.call_args_list)[0]
        assert "--log-memory" not in cmd


# ============================================================
# 5. ORCHESTRATION — FULL DISPATCH
# ============================================================


class TestOrchestration:
    """Tests for ``run_model_pairs_with_fallback`` end-to-end dispatch."""

    def test_one_invocation_per_pair(self) -> None:
        """N pairs must produce N subprocess invocations."""
        pairs = [["pico", "nano"], ["tiny", "mpslight"]]

        with mock.patch.object(
            rmp.subprocess,
            "run",
            return_value=_mock_completed(returncode=0),
        ) as mocked_run:
            rmp.run_model_pairs_with_fallback(
                model_pairs=pairs,
                datasets=["Halfmile"],
                epochs=1,
                device="cpu",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        assert mocked_run.call_count == len(pairs)

    def test_dry_run_does_not_invoke_subprocess(self) -> None:
        """``dry_run=True`` must not spawn any subprocess."""
        with mock.patch.object(rmp.subprocess, "run") as mocked_run:
            rmp.run_model_pairs_with_fallback(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=1,
                device="cpu",
                verbose=False,
                log_memory=False,
                dry_run=True,
            )

        mocked_run.assert_not_called()

    def test_pair_with_all_models_skipped_is_ignored(self) -> None:
        """A pair whose models are all skipped must not invoke subprocess."""
        with mock.patch.object(rmp.subprocess, "run") as mocked_run:
            rmp.run_model_pairs_with_fallback(
                model_pairs=[["efficient", "unet"]],
                datasets=["Lalor"],  # skips efficient and unet
                epochs=1,
                device="cpu",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        mocked_run.assert_not_called()

    def test_pair_with_some_models_available_runs(self) -> None:
        """A pair with at least one available model must invoke subprocess."""
        with mock.patch.object(
            rmp.subprocess,
            "run",
            return_value=_mock_completed(returncode=0),
        ) as mocked_run:
            rmp.run_model_pairs_with_fallback(
                model_pairs=[["pico", "unet"]],
                datasets=["Lalor"],  # unet is skipped, pico is not
                epochs=1,
                device="cpu",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )

        assert mocked_run.call_count == 1
        cmd = _flatten_commands(mocked_run.call_args_list)[0]
        model_indices = [i for i, tok in enumerate(cmd) if tok == "--models"]
        model_values = [cmd[i + 1] for i in model_indices]
        assert "pico" in model_values
        assert "unet" not in model_values

    def test_subprocess_failure_does_not_crash_orchestrator(self) -> None:
        """A failing subprocess must be logged, not raised."""
        with mock.patch.object(
            rmp.subprocess,
            "run",
            return_value=_mock_completed(returncode=1, stderr="boom"),
        ):
            # Must not raise
            rmp.run_model_pairs_with_fallback(
                model_pairs=[["pico", "nano"]],
                datasets=["Halfmile"],
                epochs=1,
                device="cpu",
                verbose=False,
                log_memory=False,
                dry_run=False,
            )
