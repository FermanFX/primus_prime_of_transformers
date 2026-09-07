"""
Unit tests for HDF5 utility functions in src/utils/hdf5_utils.py.
"""

from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pytest

from src.utils.hdf5_utils import (
    get_trace_counts,
    load_shot_data,
    load_shot_indices,
    validate_hdf5,
)


@pytest.fixture
def mock_valid_hdf5(tmp_path: Path) -> Path:
    """Fixture to create a valid seismic HDF5 structure."""
    file_path = tmp_path / "valid_seismic.hdf5"
    with h5py.File(file_path, "w") as f:
        group = f.create_group("TRACE_DATA").create_group("DEFAULT")

        # Mock SHOTIDs: 2 shots (3 traces for shot 101, 2 traces for shot 102)
        shotids: npt.NDArray[np.int32] = np.array(
            [101, 101, 101, 102, 102], dtype=np.int32
        )
        group.create_dataset("SHOTID", data=shotids)

        # Mock data array: 5 total traces, 751 time samples
        data_array: npt.NDArray[np.float32] = np.ones((5, 751), dtype=np.float32)
        group.create_dataset("data_array", data=data_array)

        # Mock SPARE1 (picks): 5 traces x 2 columns
        spare1: npt.NDArray[np.float32] = np.ones((5, 2), dtype=np.float32) * 50.0
        group.create_dataset("SPARE1", data=spare1)

    return file_path


@pytest.fixture
def mock_invalid_hdf5(tmp_path: Path) -> Path:
    """Fixture to create an invalid HDF5 structure missing required keys."""
    file_path = tmp_path / "invalid_seismic.hdf5"
    with h5py.File(file_path, "w") as f:
        f.create_group("WRONG_GROUP")
    return file_path


class TestHDF5UtilsLoading:
    """Test suite for data loading and shot index utilities."""

    def test_load_shot_indices_should_extract_unique_shots_and_bounds(
        self, mock_valid_hdf5: Path
    ) -> None:
        # Act
        unique_shots, start_indices, end_indices = load_shot_indices(
            str(mock_valid_hdf5)
        )

        # Assert
        np.testing.assert_array_equal(unique_shots, np.array([101, 102]))
        np.testing.assert_array_equal(start_indices, np.array([0, 3]))
        np.testing.assert_array_equal(end_indices, np.array([3, 5]))

    def test_load_shot_data_should_pad_when_actual_traces_less_than_target(
        self, mock_valid_hdf5: Path
    ) -> None:
        # Arrange - shot 101 has 3 traces, target is 5
        start_idx, end_idx = 0, 3
        target_traces = 5
        n_samples = 751

        # Act
        shot_data, shot_picks = load_shot_data(
            str(mock_valid_hdf5),
            start_idx=start_idx,
            end_idx=end_idx,
            target_traces=target_traces,
            n_samples=n_samples,
        )

        # Assert
        assert shot_data.shape == (5, 751)
        assert shot_picks.shape == (5,)
        # Check that original 3 traces are 1.0 and padded 2 traces are 0.0
        np.testing.assert_array_equal(shot_data[:3, :], 1.0)
        np.testing.assert_array_equal(shot_data[3:, :], 0.0)
        np.testing.assert_array_equal(shot_picks[:3], 50.0)
        np.testing.assert_array_equal(shot_picks[3:], 0.0)

    def test_load_shot_data_should_not_pad_when_actual_equals_target(
        self, mock_valid_hdf5: Path
    ) -> None:
        # Arrange - shot 101 has 3 traces, target is 3
        start_idx, end_idx = 0, 3
        target_traces = 3

        # Act
        shot_data, shot_picks = load_shot_data(
            str(mock_valid_hdf5),
            start_idx=start_idx,
            end_idx=end_idx,
            target_traces=target_traces,
        )

        # Assert
        assert shot_data.shape == (3, 751)
        assert shot_picks.shape == (3,)
        np.testing.assert_array_equal(shot_data, 1.0)

    def test_get_trace_counts_should_calculate_correct_differences(
        self,
    ) -> None:
        # Arrange
        unique_shots: npt.NDArray[np.int64] = np.array([101, 102])
        start_indices: npt.NDArray[np.int64] = np.array([0, 3])
        end_indices: npt.NDArray[np.int64] = np.array([3, 5])

        # Act
        trace_counts = get_trace_counts(
            "dummy_path", unique_shots, start_indices, end_indices
        )

        # Assert
        np.testing.assert_array_equal(trace_counts, np.array([3, 2]))


class TestHDF5UtilsValidationAndErrorHandling:
    """Test suite for HDF5 structure validation and error scenarios."""

    def test_validate_hdf5_should_return_true_for_valid_file(
        self, mock_valid_hdf5: Path
    ) -> None:
        # Act & Assert
        assert validate_hdf5(str(mock_valid_hdf5)) is True

    def test_validate_hdf5_should_return_false_when_trace_data_missing(
        self, mock_invalid_hdf5: Path
    ) -> None:
        # Act & Assert
        assert validate_hdf5(str(mock_invalid_hdf5)) is False

    @pytest.mark.parametrize("missing_dataset", ["data_array", "SHOTID", "SPARE1"])
    def test_validate_hdf5_should_return_false_when_required_dataset_missing(
        self, tmp_path: Path, missing_dataset: str
    ) -> None:
        # Arrange
        file_path = tmp_path / f"missing_{missing_dataset}.hdf5"
        with h5py.File(file_path, "w") as f:
            group = f.create_group("TRACE_DATA").create_group("DEFAULT")
            datasets = ["data_array", "SHOTID", "SPARE1"]
            for ds in datasets:
                if ds != missing_dataset:
                    dummy_data: npt.NDArray[np.int64] = np.array([1])
                    group.create_dataset(ds, data=dummy_data)

        # Act & Assert
        assert validate_hdf5(str(file_path)) is False

    def test_validate_hdf5_should_return_false_on_corrupted_or_non_existent_file(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        non_existent = tmp_path / "non_existent.hdf5"

        # Act & Assert
        assert validate_hdf5(str(non_existent)) is False
