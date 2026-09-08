"""
Tests for the preprocessing pipeline CLI and end-to-end integration.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from src.config import SeismicConfig


def create_synthetic_hdf5(
    filepath: Path, num_shots: int = 5, total_traces: int = 100, num_samples: int = 751
) -> None:
    """Creates a small synthetic HDF5 file matching TRACE_DATA/DEFAULT structure for testing."""
    with h5py.File(filepath, "w") as f:
        group = f.create_group("TRACE_DATA").create_group("DEFAULT")

        # Create dummy shot IDs (e.g., each shot has 20 traces)
        traces_per_shot = total_traces // num_shots
        shotids = np.repeat(np.arange(num_shots), traces_per_shot).astype(np.int64)

        # Ensure exact total_traces
        actual_total = len(shotids)

        group.create_dataset("SHOTID", data=shotids)
        group.create_dataset(
            "data_array",
            data=np.random.randn(actual_total, num_samples).astype(np.float32),
        )
        group.create_dataset(
            "SPARE1",
            data=np.random.randint(50, 150, size=(actual_total, 1)).astype(np.float32),
        )


@pytest.fixture
def synthetic_hdf5(tmp_path: Path) -> Path:
    """Fixture that generates and returns a temporary synthetic HDF5 file path."""
    h5_path = tmp_path / "synthetic_data.h5"
    create_synthetic_hdf5(h5_path)
    return h5_path


@pytest.fixture
def sample_config(tmp_path: Path, synthetic_hdf5: Path) -> tuple:
    """Creates a temporary YAML config file compatible with SeismicConfig."""
    config_path = tmp_path / "test_config.yaml"
    output_dir = tmp_path / "output"

    base_config_file = Path("configs/halfmile.yaml")
    if base_config_file.exists():
        with open(base_config_file, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    cfg["hdf5_path"] = str(synthetic_hdf5)
    cfg["chunk_dir"] = str(output_dir)
    cfg["chunk_size"] = 2
    cfg["target_traces"] = 30
    cfg["n_samples"] = 751

    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    return config_path, output_dir


def test_cli_help() -> None:
    """Test CLI argument parsing and help message."""
    from scripts.preprocess import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_pipeline_end_to_end(sample_config: tuple) -> None:
    """Test preprocessing pipeline from HDF5 to chunks and manifest generation."""
    from scripts.preprocess import main

    config_path, output_dir = sample_config
    runner = CliRunner()

    result = runner.invoke(main, ["--config", str(config_path)])

    assert result.exit_code == 0

    dataset_subfolder = output_dir / SeismicConfig().dataset_name
    assert dataset_subfolder.exists()

    chunk_files = list(dataset_subfolder.glob("chunk_*.pt"))
    assert len(chunk_files) > 0

    manifest_path = dataset_subfolder / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)
    assert "chunks" in manifest_data


def test_force_reprocess_flag(sample_config: tuple) -> None:
    """Verify that the --force flag correctly forces reprocessing."""
    from scripts.preprocess import main

    config_path, output_dir = sample_config
    runner = CliRunner()

    result1 = runner.invoke(main, ["--config", str(config_path)])
    assert result1.exit_code == 0

    dataset_subfolder = output_dir / SeismicConfig().dataset_name
    manifest_path = dataset_subfolder / "manifest.json"
    mtime_v1 = manifest_path.stat().st_mtime

    result3 = runner.invoke(main, ["--config", str(config_path), "--force"])
    assert result3.exit_code == 0
    assert manifest_path.stat().st_mtime >= mtime_v1


def test_invalid_hdf5_error_handling(tmp_path: Path) -> None:
    """Test error handling when an invalid or corrupted HDF5 file is provided via config."""
    from scripts.preprocess import main

    runner = CliRunner()

    invalid_file = tmp_path / "invalid.h5"
    invalid_file.write_text("not an hdf5 file")
    output_dir = tmp_path / "output"

    config_path = tmp_path / "invalid_config.yaml"
    cfg = {"hdf5_path": str(invalid_file), "chunk_dir": str(output_dir)}
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    result = runner.invoke(main, ["--config", str(config_path)])

    assert result.exit_code != 0
