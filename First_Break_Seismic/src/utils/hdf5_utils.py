"""
Utility functions for reading, processing, and validating seismic data
stored in HDF5 files.

The HDF5 files are expected to contain a ``TRACE_DATA/DEFAULT`` group
with the following datasets:

- ``SHOTID``: Shot identifier for each trace.
- ``data_array``: Seismic trace samples.
- ``SPARE1``: Pick value associated with each trace.
"""


import h5py  # type: ignore[import-untyped]
import numpy as np
from loguru import logger


def load_shot_indices(hdf5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load shot identifiers and their trace index ranges from an HDF5 file.

    The function reads the ``SHOTID`` dataset and identifies the unique shot
    IDs together with the start and end indices of the traces belonging to
    each shot. The returned indices can be used to efficiently retrieve
    individual shots from the trace datasets.

    Args:
        hdf5_path: Path to the HDF5 file.

    Returns:
        A tuple containing:
            unique_shots: 1-D array of unique shot IDs.
            start_indices: 1-D array containing the inclusive start index
                of each shot.
            end_indices: 1-D array containing the exclusive end index of
                each shot.

    Raises:
        KeyError: If the expected HDF5 groups or datasets are missing.
        OSError: If the HDF5 file cannot be opened.
    """
    with h5py.File(hdf5_path, "r") as f:
        group = f["TRACE_DATA"]["DEFAULT"]
        shotids = group["SHOTID"][()].flatten()

        unique_shots, indices = np.unique(shotids, return_index=True)
        end_indices = np.append(indices[1:], len(shotids))

        logger.debug(f"Found {len(unique_shots)} shots in {hdf5_path}")
        return unique_shots, indices, end_indices


def load_shot_data(
    hdf5_path: str,
    start_idx: int,
    end_idx: int,
    target_traces: int = 1578,
    n_samples: int = 751,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load seismic traces and associated picks for a single shot.

    The function reads traces from ``data_array`` and their corresponding
    pick values from ``SPARE1`` within the specified index range. If the
    shot contains fewer traces than ``target_traces``, both arrays are
    zero-padded to the requested number of traces.

    Args:
        hdf5_path: Path to the HDF5 file.
        start_idx: Inclusive start index of the shot traces.
        end_idx: Exclusive end index of the shot traces.
        target_traces: Target number of traces in the returned arrays.
            Defaults to 1578.
        n_samples: Number of samples expected per trace. Defaults to 751.

    Returns:
        A tuple containing:
            shot_data: Seismic trace data with shape
                ``(target_traces, n_samples)`` when padding is required.
            shot_picks: Pick values with shape ``(target_traces,)`` when
                padding is required.

    Raises:
        KeyError: If the expected HDF5 groups or datasets are missing.
        OSError: If the HDF5 file cannot be opened.
        ValueError: If the requested indices or dataset dimensions are
    """
        
    with h5py.File(hdf5_path, "r") as f:
        group = f["TRACE_DATA"]["DEFAULT"]

        shot_data = group["data_array"][start_idx:end_idx, :]  # (n_traces, 751)
        shot_picks = group["SPARE1"][start_idx:end_idx, 0]  # (n_traces,)

        actual_traces = shot_data.shape[0]

        if actual_traces < target_traces:
            # Pad with zeros
            data_padded = np.zeros((target_traces, n_samples), dtype=np.float32)
            picks_padded = np.zeros(target_traces, dtype=np.float32)
            data_padded[:actual_traces, :] = shot_data
            picks_padded[:actual_traces] = shot_picks
            shot_data = data_padded
            shot_picks = picks_padded

        return shot_data, shot_picks


def get_trace_counts(
    hdf5_path: str,
    unique_shots: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
) -> np.ndarray:
    """
    Calculate the number of traces associated with each shot.

    The trace count is calculated as the difference between the exclusive
    end index and the inclusive start index of each shot.

    Args:
        hdf5_path: Path to the HDF5 file. Included for API consistency;
            it is not accessed by this function.
        unique_shots: Array containing the unique shot IDs. Included for
            API consistency; it is not accessed by this function.
        start_indices: Inclusive start indices for each shot.
        end_indices: Exclusive end indices for each shot.

    Returns:
        1-D array containing the number of traces for each shot.

    Raises:
        ValueError: If ``start_indices`` and ``end_indices`` have
            incompatible shapes.
    """
    return end_indices - start_indices


def validate_hdf5(hdf5_path: str) -> bool:
    """
    Validate the required structure of a seismic HDF5 file.

    The function checks whether the file contains the ``TRACE_DATA/DEFAULT``
    group and the required ``data_array``, ``SHOTID``, and ``SPARE1``
    datasets.

    Args:
        hdf5_path: Path to the HDF5 file to validate.

    Returns:
        ``True`` if the required HDF5 structure is present; otherwise,
        ``False``. Any file access or validation error is logged and
        results in ``False``.
    """
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "TRACE_DATA" not in f:
                logger.error(f"TRACE_DATA group not found in {hdf5_path}")
                return False
            group = f["TRACE_DATA"]["DEFAULT"]
            if "data_array" not in group:
                logger.error(f"data_array not found in {hdf5_path}")
                return False
            if "SHOTID" not in group:
                logger.error(f"SHOTID not found in {hdf5_path}")
                return False
            if "SPARE1" not in group:
                logger.error(f"SPARE1 not found in {hdf5_path}")
                return False
        return True
    except Exception as e: # noqa: BLE001
        logger.error(f"Error validating HDF5: {e}")
        return False
