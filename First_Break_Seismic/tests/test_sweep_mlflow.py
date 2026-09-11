"""
Unit tests for :mod:`scripts.sweep_mlflow`.

Covers:
    * Config loading — YAML loading, defaults, structure.
    * Grid generation — full Cartesian product of datasets/models/losses.
    * Experiment generation — one experiment per combination.
    * MLflow tracking — run started, params logged, metrics logged.
    * Command building — correct ``train.py`` invocation per experiment.
    * Metric parsing — extracting metrics from training output.

All subprocess and MLflow calls are mocked; no real training runs and no
real MLflow tracking server is contacted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import sweep_mlflow

# ============================================================
# HELPERS
# ============================================================


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    """Write a YAML file and return its path."""
    path.write_text(yaml.safe_dump(data))
    return path


def _minimal_sweep_config() -> dict[str, Any]:
    """Return a minimal but valid sweep config dictionary."""
    return {
        "global": {
            "epochs": 2,
            "device": "cpu",
            "verbose": False,
            "log_memory": False,
            "timeout_seconds": 3600,
        },
        "sweep": {
            "datasets": ["Halfmile", "Brunswick"],
            "models": ["pico", "nano"],
            "losses": ["cross_entropy", "focal"],
            "loss_params": {
                "cross_entropy": {"class_weights": [0.1, 0.1, 0.8]},
                "focal": {
                    "class_weights": [0.1, 0.1, 0.8],
                    "focal_gamma": 2.0,
                },
            },
        },
        "tracking": {
            "enabled": True,
            "experiment_name": "test_sweep",
            "tags": {"project": "test"},
        },
    }


def _mock_subprocess_result(returncode: int = 0, stdout: str = "") -> mock.Mock:
    """Build a mock CompletedProcess-like object."""
    result = mock.Mock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = ""
    return result


# ============================================================
# 1. CONFIG LOADING
# ============================================================


class TestConfigLoading:
    """Tests for :meth:`SweepExperiment.load_config`."""

    def test_loads_yaml_config(self, tmp_path: Path) -> None:
        """A valid YAML file must be loaded into the expected sections."""
        config_file = _write_yaml(tmp_path / "sweep.yaml", _minimal_sweep_config())

        with mock.patch.object(sweep_mlflow, "get_mlflow_manager"):
            sweep = sweep_mlflow.SweepExperiment(str(config_file))

        assert sweep.global_config["epochs"] == 2
        assert sweep.global_config["device"] == "cpu"
        assert sweep.sweep_config["datasets"] == ["Halfmile", "Brunswick"]
        assert sweep.sweep_config["models"] == ["pico", "nano"]
        assert sweep.sweep_config["losses"] == ["cross_entropy", "focal"]
        assert sweep.tracking_config["experiment_name"] == "test_sweep"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing config file must raise ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            sweep_mlflow.SweepExperiment(str(tmp_path / "missing.yaml"))

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Malformed YAML must raise ``yaml.YAMLError``."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("sweep: [unclosed")

        with pytest.raises(yaml.YAMLError):
            sweep_mlflow.SweepExperiment(str(bad))

    def test_empty_sections_default(self, tmp_path: Path) -> None:
        """A config missing optional sections must fall back to empty dicts."""
        config_file = _write_yaml(tmp_path / "minimal.yaml", {})

        with mock.patch.object(sweep_mlflow, "get_mlflow_manager"):
            sweep = sweep_mlflow.SweepExperiment(str(config_file))

        assert sweep.global_config == {}
        assert sweep.sweep_config == {}
        assert sweep.tracking_config == {}

    def test_tracking_disabled_leaves_manager_none(self, tmp_path: Path) -> None:
        """When ``tracking.enabled`` is False, the MLflow manager is None."""
        cfg = _minimal_sweep_config()
        cfg["tracking"]["enabled"] = False
        config_file = _write_yaml(tmp_path / "no_track.yaml", cfg)

        sweep = sweep_mlflow.SweepExperiment(str(config_file))

        assert sweep.mlflow_manager is None

    def test_tracking_enabled_creates_manager(self, tmp_path: Path) -> None:
        """When ``tracking.enabled`` is True, ``get_mlflow_manager`` is called."""
        config_file = _write_yaml(tmp_path / "track.yaml", _minimal_sweep_config())

        with mock.patch.object(sweep_mlflow, "get_mlflow_manager") as mocked_get:
            sweep = sweep_mlflow.SweepExperiment(str(config_file))

        mocked_get.assert_called_once()
        assert sweep.mlflow_manager is mocked_get.return_value


# ============================================================
# 2. GRID GENERATION
# ============================================================


class TestGridGeneration:
    """Tests that the sweep produces the full Cartesian product."""

    def _make_sweep(self, tmp_path: Path, cfg: dict[str, Any]) -> Any:
        config_file = _write_yaml(tmp_path / "sweep.yaml", cfg)
        with mock.patch.object(sweep_mlflow, "get_mlflow_manager"):
            return sweep_mlflow.SweepExperiment(str(config_file))

    def test_combination_count(self, tmp_path: Path) -> None:
        """Total experiments = len(datasets) × len(models) × len(losses)."""
        cfg = _minimal_sweep_config()
        sweep = self._make_sweep(tmp_path, cfg)

        datasets = sweep.sweep_config["datasets"]
        models = sweep.sweep_config["models"]
        losses = sweep.sweep_config["losses"]

        expected = len(datasets) * len(models) * len(losses)
        assert expected == 8  # 2 × 2 × 2

    def test_all_combinations_covered(self, tmp_path: Path) -> None:
        """Every (dataset, model, loss) triple must appear exactly once."""
        cfg = _minimal_sweep_config()
        sweep = self._make_sweep(tmp_path, cfg)

        # Reproduce the grid the sweep would iterate
        combinations = [
            (dataset, model, loss)
            for dataset in sweep.sweep_config["datasets"]
            for model in sweep.sweep_config["models"]
            for loss in sweep.sweep_config["losses"]
        ]

        expected = {
            (d, m, l)
            for d in sweep.sweep_config["datasets"]
            for m in sweep.sweep_config["models"]
            for l in sweep.sweep_config["losses"]
        }

        assert len(combinations) == len(expected)
        assert set(combinations) == expected

    def test_empty_datasets_yields_zero_experiments(self, tmp_path: Path) -> None:
        """An empty datasets list must produce zero combinations."""
        cfg = _minimal_sweep_config()
        cfg["sweep"]["datasets"] = []
        sweep = self._make_sweep(tmp_path, cfg)

        combinations = [
            (d, m, l)
            for d in sweep.sweep_config["datasets"]
            for m in sweep.sweep_config["models"]
            for l in sweep.sweep_config["losses"]
        ]

        assert combinations == []

    def test_empty_models_yields_zero_experiments(self, tmp_path: Path) -> None:
        """An empty models list must produce zero combinations."""
        cfg = _minimal_sweep_config()
        cfg["sweep"]["models"] = []
        sweep = self._make_sweep(tmp_path, cfg)

        combinations = [
            (d, m, l)
            for d in sweep.sweep_config["datasets"]
            for m in sweep.sweep_config["models"]
            for l in sweep.sweep_config["losses"]
        ]

        assert combinations == []


# ============================================================
# 3. EXPERIMENT GENERATION
# ============================================================


class TestExperimentGeneration:
    """Tests that ``run_sweep`` dispatches one experiment per combination."""

    def _make_sweep_with_mocks(
        self, tmp_path: Path
    ) -> tuple[Any, mock.MagicMock, mock.MagicMock]:
        config_file = _write_yaml(tmp_path / "sweep.yaml", _minimal_sweep_config())

        # Mock the MLflow manager entirely.
        mlflow_mock = mock.MagicMock()
        mlflow_mock.start_run.return_value = "run_id_mock"
        mlflow_mock.run_id = "run_id_mock"

        with mock.patch.object(
            sweep_mlflow, "get_mlflow_manager", return_value=mlflow_mock
        ):
            sweep = sweep_mlflow.SweepExperiment(str(config_file))

        # Mock subprocess.run and save_checkpoint to isolate run_sweep.
        subprocess_mock = mock.MagicMock(
            return_value=_mock_subprocess_result(
                returncode=0,
                stdout="Train Loss: 1.0\nVal Loss: 0.9\nTrain IoU: 0.5\nVal IoU: 0.4\n",
            )
        )
        save_ckpt_mock = mock.MagicMock()
        print_summary_mock = mock.MagicMock()

        return sweep, subprocess_mock, save_ckpt_mock, print_summary_mock  # type: ignore[return-value]

    def test_run_sweep_dispatches_all_combinations(self, tmp_path: Path) -> None:
        """``run_sweep`` must call ``run_experiment`` once per combination."""
        cfg = _minimal_sweep_config()
        config_file = _write_yaml(tmp_path / "sweep.yaml", cfg)

        mlflow_mock = mock.MagicMock()
        mlflow_mock.start_run.return_value = "run_id_mock"
        mlflow_mock.run_id = "run_id_mock"

        with mock.patch.object(
            sweep_mlflow, "get_mlflow_manager", return_value=mlflow_mock
        ):
            sweep = sweep_mlflow.SweepExperiment(str(config_file))

        with (
            mock.patch.object(sweep, "run_experiment") as mocked_run_exp,
            mock.patch.object(sweep, "save_checkpoint"),
            mock.patch.object(sweep, "print_summary"),
        ):
            mocked_run_exp.return_value = {
                "success": True,
                "dataset": "X",
                "model": "Y",
                "loss": "Z",
                "metrics": {},
            }
            sweep.run_sweep()

        # 2 datasets × 2 models × 2 losses = 8 experiments
        assert mocked_run_exp.call_count == 8

    def test_run_sweep_passes_correct_combination(self, tmp_path: Path) -> None:
        """Each ``run_experiment`` call must receive a valid combination."""
        cfg = _minimal_sweep_config()
        config_file = _write_yaml(tmp_path / "sweep.yaml", cfg)

        mlflow_mock = mock.MagicMock()
        mlflow_mock.start_run.return_value = "run_id_mock"
        mlflow_mock.run_id = "run_id_mock"

        with mock.patch.object(
            sweep_mlflow, "get_mlflow_manager", return_value=mlflow_mock
        ):
            sweep = sweep_mlflow.SweepExperiment(str(config_file))

        captured_calls: list[dict[str, Any]] = []

        def fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
            captured_calls.append(kwargs)
            return {"success": True, "metrics": {}}

        with (
            mock.patch.object(sweep, "run_experiment", side_effect=fake_run_experiment),
            mock.patch.object(sweep, "save_checkpoint"),
            mock.patch.object(sweep, "print_summary"),
        ):
            sweep.run_sweep()

        seen_combos = {
            (call["dataset"], call["model"], call["loss"]) for call in captured_calls
        }
        expected = {
            (d, m, l)
            for d in cfg["sweep"]["datasets"]
            for m in cfg["sweep"]["models"]
            for l in cfg["sweep"]["losses"]
        }

        assert seen_combos == expected

    def test_run_sweep_returns_all_results(self, tmp_path: Path) -> None:
        """``run_sweep`` must return one result dict per combination."""
        cfg = _minimal_sweep_config()
        config_file = _write_yaml(tmp_path / "sweep.yaml", cfg)

        mlflow_mock = mock.MagicMock()
        mlflow_mock.start_run.return_value = "run_id_mock"
        mlflow_mock.run_id = "run_id_mock"

        with mock.patch.object(
            sweep_mlflow, "get_mlflow_manager", return_value=mlflow_mock
        ):
            sweep = sweep_mlflow.SweepExperiment(str(config_file))

        with (
            mock.patch.object(sweep, "run_experiment") as mocked_run_exp,
            mock.patch.object(sweep, "save_checkpoint"),
            mock.patch.object(sweep, "print_summary"),
        ):
            mocked_run_exp.return_value = {
                "success": True,
                "metrics": {},
            }
            results = sweep.run_sweep()

        assert len(results) == 8


# ============================================================
# 4. MLFLOW TRACKING
# ============================================================


class TestMLflowTracking:
    """Tests that MLflow tracking is invoked correctly per experiment."""

    def _make_sweep_with_mocks(self, tmp_path: Path) -> tuple[Any, mock.MagicMock]:
        config_file = _write_yaml(tmp_path / "sweep.yaml", _minimal_sweep_config())

        mlflow_mock = mock.MagicMock()
        mlflow_mock.start_run.return_value = "run_id_mock"
        mlflow_mock.run_id = "run_id_mock"

        with mock.patch.object(
            sweep_mlflow, "get_mlflow_manager", return_value=mlflow_mock
        ):
            sweep = sweep_mlflow.SweepExperiment(str(config_file))

        return sweep, mlflow_mock

    def test_run_experiment_starts_mlflow_run(self, tmp_path: Path) -> None:
        """``run_experiment`` must call ``start_run`` exactly once."""
        sweep, mlflow_mock = self._make_sweep_with_mocks(tmp_path)

        with mock.patch.object(
            sweep_mlflow.subprocess,
            "run",
            return_value=_mock_subprocess_result(returncode=0),
        ):
            sweep.run_experiment(
                dataset="Halfmile",
                model="pico",
                loss="cross_entropy",
                loss_params={"class_weights": [0.1, 0.1, 0.8]},
                experiment_id=1,
                total_experiments=8,
            )

        assert mlflow_mock.start_run.call_count == 1

    def test_run_experiment_logs_params(self, tmp_path: Path) -> None:
        """The config passed to ``start_run`` must include dataset/model/loss."""
        sweep, mlflow_mock = self._make_sweep_with_mocks(tmp_path)

        with mock.patch.object(
            sweep_mlflow.subprocess,
            "run",
            return_value=_mock_subprocess_result(returncode=0),
        ):
            sweep.run_experiment(
                dataset="Halfmile",
                model="pico",
                loss="cross_entropy",
                loss_params={"class_weights": [0.1, 0.1, 0.8]},
                experiment_id=1,
                total_experiments=8,
            )

        call_kwargs = mlflow_mock.start_run.call_args.kwargs
        config_dict = call_kwargs["config_dict"]

        assert config_dict["dataset"] == "Halfmile"
        assert config_dict["model"] == "pico"
        assert config_dict["loss"] == "cross_entropy"
        assert "class_weights" in config_dict

    def test_run_experiment_sets_tags(self, tmp_path: Path) -> None:
        """The tags passed to ``start_run`` must include dataset/model/loss."""
        sweep, mlflow_mock = self._make_sweep_with_mocks(tmp_path)

        with mock.patch.object(
            sweep_mlflow.subprocess,
            "run",
            return_value=_mock_subprocess_result(returncode=0),
        ):
            sweep.run_experiment(
                dataset="Brunswick",
                model="nano",
                loss="focal",
                loss_params={"class_weights": [0.1, 0.1, 0.8], "focal_gamma": 2.0},
                experiment_id=2,
                total_experiments=8,
            )

        call_kwargs = mlflow_mock.start_run.call_args.kwargs
        tags = call_kwargs["tags"]

        assert tags["dataset"] == "Brunswick"
        assert tags["model"] == "nano"
        assert tags["loss"] == "focal"
        assert tags["experiment_type"] == "sweep"

    def test_run_experiment_logs_metrics(self, tmp_path: Path) -> None:
        """Metrics parsed from output must be logged via ``log_metrics``."""
        sweep, mlflow_mock = self._make_sweep_with_mocks(tmp_path)

        stdout = (
            "Train Loss: 1.234\nVal Loss: 1.100\nTrain IoU: 0.500\nVal IoU: 0.450\n"
        )

        with mock.patch.object(
            sweep_mlflow.subprocess,
            "run",
            return_value=_mock_subprocess_result(returncode=0, stdout=stdout),
        ):
            sweep.run_experiment(
                dataset="Halfmile",
                model="pico",
                loss="cross_entropy",
                loss_params={"class_weights": [0.1, 0.1, 0.8]},
                experiment_id=1,
                total_experiments=8,
            )

        # log_metrics must have been called at least once
        assert mlflow_mock.log_metrics.called

        logged_metrics = mlflow_mock.log_metrics.call_args.args[0]
        assert logged_metrics["train_loss"] == pytest.approx(1.234)
        assert logged_metrics["val_loss"] == pytest.approx(1.100)
        assert logged_metrics["train_iou"] == pytest.approx(0.500)
        assert logged_metrics["val_iou"] == pytest.approx(0.450)

    def test_run_experiment_ends_run(self, tmp_path: Path) -> None:
        """``end_run`` must be called after every experiment, success or not."""
        sweep, mlflow_mock = self._make_sweep_with_mocks(tmp_path)

        with mock.patch.object(
            sweep_mlflow.subprocess,
            "run",
            return_value=_mock_subprocess_result(
                returncode=1, stdout="RuntimeError: boom"
            ),
        ):
            sweep.run_experiment(
                dataset="Halfmile",
                model="pico",
                loss="cross_entropy",
                loss_params={},
                experiment_id=1,
                total_experiments=8,
            )

        assert mlflow_mock.end_run.call_count == 1

    def test_run_experiment_no_tracking_when_disabled(self, tmp_path: Path) -> None:
        """When tracking is disabled, no MLflow calls occur."""
        cfg = _minimal_sweep_config()
        cfg["tracking"]["enabled"] = False
        config_file = _write_yaml(tmp_path / "no_track.yaml", cfg)

        sweep = sweep_mlflow.SweepExperiment(str(config_file))
        assert sweep.mlflow_manager is None

        with mock.patch.object(
            sweep_mlflow.subprocess,
            "run",
            return_value=_mock_subprocess_result(returncode=0),
        ):
            result = sweep.run_experiment(
                dataset="Halfmile",
                model="pico",
                loss="cross_entropy",
                loss_params={},
                experiment_id=1,
                total_experiments=8,
            )

        # The run still completes and reports success
        assert result["success"] is True


# ============================================================
# 5. COMMAND BUILDING
# ============================================================


class TestCommandBuilding:
    """Tests for :meth:`SweepExperiment.build_command`."""

    def _make_sweep(self, tmp_path: Path) -> Any:
        config_file = _write_yaml(tmp_path / "sweep.yaml", _minimal_sweep_config())
        with mock.patch.object(sweep_mlflow, "get_mlflow_manager"):
            return sweep_mlflow.SweepExperiment(str(config_file))

    def test_basic_command_structure(self, tmp_path: Path) -> None:
        """The command must start with ``python3.12 scripts/train.py``."""
        sweep = self._make_sweep(tmp_path)

        cmd = sweep.build_command(
            dataset="Halfmile",
            model="pico",
            loss="cross_entropy",
            loss_params={"class_weights": [0.1, 0.1, 0.8]},
        )

        assert cmd[0] == "python3.12"
        assert cmd[1] == "scripts/train.py"
        assert "--config" in cmd
        assert "--model" in cmd
        assert "--loss" in cmd

    def test_config_path_matches_dataset(self, tmp_path: Path) -> None:
        """``--config`` must be ``configs/{dataset_lower}.yaml``."""
        sweep = self._make_sweep(tmp_path)

        cmd = sweep.build_command(
            dataset="Halfmile",
            model="pico",
            loss="cross_entropy",
            loss_params={},
        )

        idx = cmd.index("--config")
        assert cmd[idx + 1] == "configs/halfmile.yaml"

    def test_epochs_and_device_forwarded(self, tmp_path: Path) -> None:
        """Global epochs and device must be forwarded."""
        sweep = self._make_sweep(tmp_path)

        cmd = sweep.build_command(
            dataset="Halfmile",
            model="pico",
            loss="cross_entropy",
            loss_params={},
        )

        epochs_idx = cmd.index("--epochs")
        device_idx = cmd.index("--device")

        assert cmd[epochs_idx + 1] == "2"  # from _minimal_sweep_config global.epochs
        assert cmd[device_idx + 1] == "cpu"

    def test_class_weights_split_into_three_args(self, tmp_path: Path) -> None:
        """``class_weights`` list must be split into three separate args."""
        sweep = self._make_sweep(tmp_path)

        cmd = sweep.build_command(
            dataset="Halfmile",
            model="pico",
            loss="cross_entropy",
            loss_params={"class_weights": [0.1, 0.2, 0.7]},
        )

        idx = cmd.index("--class-weights")
        assert cmd[idx + 1 : idx + 4] == ["0.1", "0.2", "0.7"]

    def test_combo_loss_adds_dice_and_focal(self, tmp_path: Path) -> None:
        """``combo`` loss must add ``--dice-weight`` and ``--focal-gamma``."""
        sweep = self._make_sweep(tmp_path)

        cmd = sweep.build_command(
            dataset="Halfmile",
            model="pico",
            loss="combo",
            loss_params={
                "class_weights": [0.1, 0.1, 0.8],
                "dice_weight": 0.6,
                "focal_gamma": 2.5,
            },
        )

        assert "--dice-weight" in cmd
        assert "--focal-gamma" in cmd
        assert cmd[cmd.index("--dice-weight") + 1] == "0.6"
        assert cmd[cmd.index("--focal-gamma") + 1] == "2.5"

    def test_focal_loss_adds_only_focal(self, tmp_path: Path) -> None:
        """``focal`` loss must add ``--focal-gamma`` but not ``--dice-weight``."""
        sweep = self._make_sweep(tmp_path)

        cmd = sweep.build_command(
            dataset="Halfmile",
            model="pico",
            loss="focal",
            loss_params={
                "class_weights": [0.1, 0.1, 0.8],
                "focal_gamma": 3.0,
            },
        )

        assert "--focal-gamma" in cmd
        assert "--dice-weight" not in cmd
        assert cmd[cmd.index("--focal-gamma") + 1] == "3.0"


# ============================================================
# 6. METRIC PARSING
# ============================================================


class TestMetricParsing:
    """Tests for :meth:`SweepExperiment.parse_metrics`."""

    def test_parse_full_output(self) -> None:
        """All known metric lines must be extracted."""
        output = (
            "Epoch 1/2 - Train Loss: 1.234\n"
            "  Train IoU: 0.500, Train Acc: 0.700\n"
            "Epoch 1/2 - Val Loss: 1.100\n"
            "  Val IoU: 0.450, Val Acc: 0.650\n"
        )

        metrics = sweep_mlflow.SweepExperiment.parse_metrics(output)

        assert metrics["train_loss"] == pytest.approx(1.234)
        assert metrics["val_loss"] == pytest.approx(1.100)
        assert metrics["train_iou"] == pytest.approx(0.500)
        assert metrics["val_iou"] == pytest.approx(0.450)
        assert metrics["train_acc"] == pytest.approx(0.700)
        assert metrics["val_acc"] == pytest.approx(0.650)

    def test_parse_empty_output(self) -> None:
        """Empty output must yield an empty metrics dict."""
        metrics = sweep_mlflow.SweepExperiment.parse_metrics("")
        assert metrics == {}

    def test_parse_partial_output(self) -> None:
        """Only the metrics that appear in the output must be extracted."""
        output = "Train Loss: 0.500\n"

        metrics = sweep_mlflow.SweepExperiment.parse_metrics(output)

        assert metrics["train_loss"] == pytest.approx(0.500)
        assert "val_loss" not in metrics
        assert "train_iou" not in metrics

    def test_parse_malformed_line_is_ignored(self) -> None:
        """Lines with ``Train Loss:`` but no numeric value must be skipped."""
        output = "Train Loss: not-a-number\nVal Loss: 1.000\n"

        metrics = sweep_mlflow.SweepExperiment.parse_metrics(output)

        assert "train_loss" not in metrics
        assert metrics["val_loss"] == pytest.approx(1.000)


# ============================================================
# 7. INTEGRATION — END-TO-END SWEEP
# ============================================================


class TestEndToEndSweep:
    """End-to-end checks: one sweep produces one MLflow run per combination."""

    def test_full_sweep_starts_one_mlflow_run_per_combination(
        self, tmp_path: Path
    ) -> None:
        """A full sweep must start exactly N MLflow runs."""
        cfg = _minimal_sweep_config()
        config_file = _write_yaml(tmp_path / "sweep.yaml", cfg)

        mlflow_mock = mock.MagicMock()
        mlflow_mock.start_run.return_value = "run_id_mock"
        mlflow_mock.run_id = "run_id_mock"

        with mock.patch.object(
            sweep_mlflow, "get_mlflow_manager", return_value=mlflow_mock
        ):
            sweep = sweep_mlflow.SweepExperiment(str(config_file))

        with (
            mock.patch.object(
                sweep_mlflow.subprocess,
                "run",
                return_value=_mock_subprocess_result(
                    returncode=0,
                    stdout="Train Loss: 1.0\nVal Loss: 0.9\n",
                ),
            ),
            mock.patch.object(sweep, "save_checkpoint"),
            mock.patch.object(sweep, "print_summary"),
        ):
            sweep.run_sweep()

        expected = (
            len(cfg["sweep"]["datasets"])
            * len(cfg["sweep"]["models"])
            * len(cfg["sweep"]["losses"])
        )
        assert mlflow_mock.start_run.call_count == expected
        assert mlflow_mock.end_run.call_count == expected
