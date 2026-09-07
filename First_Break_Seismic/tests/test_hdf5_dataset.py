"""
Unit tests for HDF5SeismicDataset class in src/data/hdf5_dataset.py.
Covers lazy loading, padding/cropping, mask generation, and file cleanup.
"""

from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

from src.data.hdf5_dataset import HDF5SeismicDataset


@pytest.fixture
def mock_hdf5_data(tmp_path: Path):
    """Fixture to create a mock HDF5 file with known trace and pick data."""
    file_path = tmp_path / "seismic_dataset.hdf5"

    with h5py.File(file_path, "w") as f:
        group = f.create_group("TRACE_DATA").create_group("DEFAULT")

        # Total traces = 3 + 5 + 6 = 14
        data = np.ones((14, 751), dtype=np.float32)
        group.create_dataset("data_array", data=data)

        # SPARE1 column 0 contains picks
        spare1 = np.zeros((14, 2), dtype=np.float32)
        spare1[0:3, 0] = [100.0, -1.0, 800.0]
        spare1[3:8, 0] = 50.0
        spare1[8:14, 0] = 200.0
        group.create_dataset("SPARE1", data=spare1)

    shot_indices = {
        101: (0, 3),
        102: (3, 8),
        103: (8, 14),
    }
    shot_ids = [101, 102, 103]

    return str(file_path), shot_indices, shot_ids


class TestHDF5SeismicDataset:
    """Test suite for HDF5SeismicDataset operations."""

    def test_lazy_file_opening(self, mock_hdf5_data) -> None:
        """Test that HDF5 file is opened lazily upon first __getitem__ call."""
        hdf5_path, shot_indices, shot_ids = mock_hdf5_data

        dataset = HDF5SeismicDataset(
            hdf5_path=hdf5_path,
            shot_indices=shot_indices,
            shot_ids=shot_ids,
            target_traces=5,
        )

        assert dataset.file is None
        _, _ = dataset[0]
        assert dataset.file is not None
        dataset.close()

    def test_padding_when_traces_less_than_target(self, mock_hdf5_data) -> None:
        """Test padding with zeros when shot has fewer traces than target_traces."""
        hdf5_path, shot_indices, shot_ids = mock_hdf5_data

        dataset = HDF5SeismicDataset(
            hdf5_path=hdf5_path,
            shot_indices=shot_indices,
            shot_ids=shot_ids,
            target_traces=5,
            n_samples=751,
        )

        data, mask = dataset[0]

        assert data.shape == (1, 5, 751)
        assert mask.shape == (5, 751)

        np.testing.assert_array_equal(data[0, :3, :].numpy(), 1.0)
        np.testing.assert_array_equal(data[0, 3:, :].numpy(), 0.0)

        dataset.close()

    def test_cropping_when_traces_more_than_target(self, mock_hdf5_data) -> None:
        """Test cropping when shot has more traces than target_traces."""
        hdf5_path, shot_indices, shot_ids = mock_hdf5_data

        dataset = HDF5SeismicDataset(
            hdf5_path=hdf5_path,
            shot_indices=shot_indices,
            shot_ids=shot_ids,
            target_traces=5,
            n_samples=751,
        )

        data, mask = dataset[2]

        assert data.shape == (1, 5, 751)
        assert mask.shape == (5, 751)

        dataset.close()

    def test_mask_generation_and_unlabeled_handling(self, mock_hdf5_data) -> None:
        """Test 3-class mask generation logic and handling of invalid picks."""
        hdf5_path, shot_indices, shot_ids = mock_hdf5_data

        dataset = HDF5SeismicDataset(
            hdf5_path=hdf5_path,
            shot_indices=shot_indices,
            shot_ids=shot_ids,
            target_traces=5,
            n_samples=751,
            strip_width=8,
        )

        _, mask = dataset[0]
        mask_np = mask.numpy()

        assert np.all(mask_np[0, :96] == 0)
        assert np.all(mask_np[0, 96:105] == 2)
        assert np.all(mask_np[0, 105:] == 1)

        assert np.all(mask_np[1, :] == 0)
        assert np.all(mask_np[2, :] == 0)

        dataset.close()

    def test_helper_methods(self, mock_hdf5_data) -> None:
        """Test __len__ and get_shot_id methods."""
        hdf5_path, shot_indices, shot_ids = mock_hdf5_data

        dataset = HDF5SeismicDataset(
            hdf5_path=hdf5_path,
            shot_indices=shot_indices,
            shot_ids=shot_ids,
        )

        assert len(dataset) == 3
        assert dataset.get_shot_id(1) == 102

        dataset.close()