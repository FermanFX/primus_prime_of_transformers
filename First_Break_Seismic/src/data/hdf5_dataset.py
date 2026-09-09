"""
HDF5-backed seismic dataset with lazy loading and level-based telemetry.

The dataset keeps the HDF5 file closed until the first sample is requested,
then reuses the open file handle for subsequent reads. Each shot is loaded
on demand, padded or cropped to a fixed number of traces, and converted into
a PyTorch tensor together with a pick-based segmentation mask.

Logging:
    INFO:
        Dataset initialization.

    DEBUG:
        HDF5 file opening and closing.
        Every ``__getitem__`` call, including shot ID and slice information.
        Dataset configuration details.
"""

from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset


class HDF5SeismicDataset(Dataset):
    """
    Memory-efficient PyTorch dataset for seismic data stored in HDF5.

    Each dataset item corresponds to one shot. Shot boundaries are provided
    through ``shot_indices`` and the actual seismic traces are read lazily
    from the HDF5 file when ``__getitem__`` is called.

    The returned seismic data is normalized to a fixed shape of
    ``(1, target_traces, n_samples)``. Shots with fewer traces are zero-padded,
    while shots with more traces are cropped.

    A corresponding mask of shape ``(target_traces, n_samples)`` is generated
    from the values stored in the ``SPARE1`` HDF5 dataset:

        - ``0``: no valid pick / padding region.
        - ``2``: region around the seismic pick.
        - ``1``: region after the pick.

    The HDF5 file is opened lazily on the first access rather than during
    dataset initialization. This is particularly useful when the dataset is
    used with PyTorch ``DataLoader`` workers.

    Logging:
        - INFO: Dataset initialization.
        - DEBUG: Dataset configuration, HDF5 file opening/closing, and every
          ``__getitem__`` call with shot ID and slice information.

    Attributes:
        hdf5_path: Path to the HDF5 file.
        shot_indices: Mapping from shot ID to its ``(start, end)`` trace
            indices in the HDF5 datasets.
        shot_ids: Ordered list of shot IDs exposed by the dataset.
        target_traces: Number of traces expected for every returned sample.
        n_samples: Number of samples expected for every trace.
        strip_width: Width of the region centered around each pick.
        half_width: Half of ``strip_width`` used when constructing masks.
        file: Lazily opened HDF5 file handle.
        group: HDF5 group containing the seismic trace datasets.

    Args:
        hdf5_path: Path to the input HDF5 file.
        shot_indices: Mapping of shot IDs to ``(start_idx, end_idx)`` trace
            ranges.
        shot_ids: List of shot IDs that should be exposed by this dataset.
        target_traces: Fixed number of traces returned for every shot.
            Defaults to ``1578``.
        n_samples: Number of samples per trace. Defaults to ``751``.
        strip_width: Width of the pick-centered mask region. Defaults to ``8``.
    """

    def __init__(
        self,
        hdf5_path: str,
        shot_indices: dict[int, tuple[int, int]],
        shot_ids: list,
        target_traces: int = 1578,
        n_samples: int = 751,
        strip_width: int = 8,
    ):
        """
        Initialize the HDF5 seismic dataset.

        The HDF5 file is not opened during initialization. It is opened
        lazily when the first sample is requested.

        Args:
            hdf5_path: Path to the HDF5 file.
            shot_indices: Mapping from shot ID to its trace range as
                ``(start_idx, end_idx)``.
            shot_ids: List of shot IDs included in this dataset.
            target_traces: Number of traces in every returned shot.
            n_samples: Number of samples in each trace.
            strip_width: Width of the mask region around a pick.
        """
        self.hdf5_path = hdf5_path
        self.shot_indices = shot_indices
        self.shot_ids = shot_ids
        self.target_traces = target_traces
        self.n_samples = n_samples
        self.strip_width = strip_width
        self.half_width = strip_width // 2

        self.file: h5py.File | None = None
        self.group: Any = None

        logger.info(f"[HDF5] INIT: {len(self)} shots, file={hdf5_path}")
        logger.debug(
            f"[HDF5] target_traces={target_traces}, n_samples={n_samples}, strip_width={strip_width}"
        )

    def __len__(self) -> int:
        """
        Return the number of shots in the dataset.

        Returns:
            Number of available shot IDs.
        """
        return len(self.shot_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a single seismic shot.

        The corresponding trace range is read from the HDF5 file. The shot is
        then padded with zeros if it contains fewer than ``target_traces``
        traces, or cropped if it contains more.

        The method returns both the seismic data and a segmentation mask
        generated from the ``SPARE1`` pick values.

        Args:
            idx: Zero-based dataset index.

        Returns:
            A tuple containing:

            - ``torch.Tensor``: Seismic data with shape
              ``(1, target_traces, n_samples)`` and dtype ``float32``.
            - ``torch.Tensor``: Pick-based mask with shape
              ``(target_traces, n_samples)`` and dtype ``int64``.

        Raises:
            KeyError: If the selected shot ID is not present in
                ``shot_indices``.
            IndexError: If ``idx`` is outside the valid dataset range.
        """
        shot_id = self.shot_ids[idx]
        start_idx, end_idx = self.shot_indices[shot_id]

        logger.debug(
            f"[HDF5] GET idx={idx} → shot={shot_id}, slice={start_idx}:{end_idx}"
        )

        if self.file is None or self.group is None:
            logger.debug(f"[HDF5] Opening HDF5 file: {self.hdf5_path}")
            self.file = h5py.File(self.hdf5_path, "r", swmr=True)
            self.group = self.file["TRACE_DATA"]["DEFAULT"]

        # Read data
        shot_data = self.group["data_array"][start_idx:end_idx, :]
        shot_picks = self.group["SPARE1"][start_idx:end_idx, 0]

        # Pad/crop
        actual_traces = shot_data.shape[0]
        if actual_traces < self.target_traces:
            data_padded = np.zeros(
                (self.target_traces, self.n_samples), dtype=np.float32
            )
            picks_padded = np.zeros(self.target_traces, dtype=np.float32)
            data_padded[:actual_traces, :] = shot_data
            picks_padded[:actual_traces] = shot_picks
            shot_data = data_padded
            shot_picks = picks_padded
        elif actual_traces > self.target_traces:
            shot_data = shot_data[: self.target_traces, :]
            shot_picks = shot_picks[: self.target_traces]

        mask = self._create_mask(shot_picks)

        return (
            torch.from_numpy(shot_data).float().unsqueeze(0),
            torch.from_numpy(mask).long(),
        )

    def _create_mask(self, picks: np.ndarray) -> np.ndarray:
        """
        Create a segmentation mask from seismic pick positions.

        For every valid pick, a region centered on the pick is assigned class
        ``2`` and all samples after that region are assigned class ``1``.
        Samples before the pick region remain class ``0``.

        Invalid picks (values less than or equal to zero, or greater than or
        equal to ``n_samples``) are ignored.

        Args:
            picks: One-dimensional NumPy array containing one pick value per
                trace.

        Returns:
            Integer mask with shape ``(len(picks), n_samples)`` and values
            ``0``, ``1``, or ``2``.
        """
        n_traces = len(picks)
        mask = np.zeros((n_traces, self.n_samples), dtype=np.int64)

        for i, pick in enumerate(picks):
            if pick <= 0 or pick >= self.n_samples:
                continue
            pick_int = round(pick)
            start = max(0, pick_int - self.half_width)
            end = min(self.n_samples, pick_int + self.half_width + 1)
            mask[i, start:end] = 2
            mask[i, end:] = 1

        return mask

    def close(self) -> None:
        """
        Close the currently open HDF5 file.

        If the file has not been opened yet, this method does nothing.
        After closing, the file and group references are reset to ``None``.
        """
        if self.file is not None:
            logger.debug(f"[HDF5] Closing file: {self.hdf5_path}")
            self.file.close()
            self.file = None
            self.group = None

    def __del__(self) -> None:
        """Close the HDF5 file when the dataset object is garbage-collected."""
        self.close()

    def get_shot_id(self, idx: int) -> int:
        """
        Return the shot ID associated with a dataset index.

        Args:
            idx: Zero-based dataset index.

        Returns:
            Shot ID corresponding to ``idx``.

        Raises:
            IndexError: If ``idx`` is outside the valid dataset range.
        """
        return self.shot_ids[idx]
