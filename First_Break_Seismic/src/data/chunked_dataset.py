"""
Chunked seismic dataset utilities for memory-efficient model training.

This module provides datasets that load seismic data in chunks on demand
and keep recently accessed chunks in an LRU cache. This avoids loading the
entire dataset into memory and is especially useful for large seismic
datasets.

Classes:
    ChunkedSeismicDataset:
        PyTorch dataset that resolves global sample indices to chunk/local
        indices and loads chunks lazily from disk.

    ChunkedDataManager:
        Lazily creates and manages datasets for different data splits.
"""

from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import Dataset

from src.data.cache import LRUCache


class ChunkedSeismicDataset(Dataset):
    """
    Memory-efficient PyTorch dataset backed by chunked seismic data.

    The dataset does not load all seismic samples into memory during
    initialization. Instead, data is stored on disk in multiple chunks and
    individual chunks are loaded only when a sample from that chunk is
    requested.

    Recently accessed chunks are stored in an LRU cache. When the cache
    reaches its configured capacity, the least recently used chunk is
    evicted automatically.

    A global sample index is maintained internally and mapped to a
    ``(chunk_idx, local_idx)`` pair. This allows the dataset to expose all
    samples through the standard PyTorch ``Dataset`` interface while
    keeping the underlying data chunked on disk.

    Logging levels:
        INFO:
            Dataset initialization, chunk loading, and cache statistics.

        DEBUG:
            Individual sample accesses, cache hits/misses, and dataset
            statistics.

    Args:
        chunk_dir:
            Directory containing the serialized dataset chunks.

        manifest:
            Dataset manifest containing chunk metadata. The manifest is
            expected to contain a ``"chunks"`` list where each chunk
            provides at least ``"split"``, ``"filename"``, ``"n_shots"``,
            and ``"shot_ids"``.

        split:
            Dataset split to expose, for example ``"train"``, ``"val"``,
            or ``"test"``.

        cache_size:
            Maximum number of chunks kept in memory at the same time.

        shuffle_chunks:
            Whether chunk-level shuffling is enabled. The flag is stored
            for compatibility with the data pipeline but does not alter
            indexing inside this class.

    Raises:
        ValueError:
            If the manifest does not contain any chunks for the requested
            split.

    Example:
        >>> dataset = ChunkedSeismicDataset(
        ...     chunk_dir="data/chunks",
        ...     manifest=manifest,
        ...     split="train",
        ...     cache_size=3,
        ... )
        >>> data, mask = dataset[0]
        >>> shot_id = dataset.get_shot_id(0)
    """


    def __init__(
        self,
        chunk_dir: str | Path,
        manifest: dict[str, Any],
        split: str = "train",
        cache_size: int = 3,
        shuffle_chunks: bool = True,
    ) -> None:
        self.chunk_dir = Path(chunk_dir)
        self.manifest = manifest
        self.split = split
        self.cache_size = cache_size
        self.shuffle_chunks = shuffle_chunks

        logger.debug(f"[Dataset] INIT split={split} | cache_size={cache_size}")

        # Get chunks for this split
        self.chunks = [c for c in manifest["chunks"] if c["split"] == split]

        if not self.chunks:
            raise ValueError(f"No chunks found for split '{split}'")

        # Build global index: global_idx -> (chunk_idx, local_idx)
        self.global_index: list[int] = []
        self.chunk_indices: list[int] = []
        self.chunk_offsets: list[int] = []
        self.shot_ids: list[Any] = []

        offset = 0
        for chunk_idx, chunk in enumerate(self.chunks):
            n_shots = chunk["n_shots"]
            self.global_index.extend(list(range(offset, offset + n_shots)))
            self.chunk_indices.extend([chunk_idx] * n_shots)
            self.chunk_offsets.extend(range(n_shots))
            self.shot_ids.extend(chunk["shot_ids"])
            offset += n_shots

        self.n_samples = manifest.get("config", {}).get("n_samples", len(self.global_index))
        self.target_traces = manifest.get("config", {}).get("target_traces", None)

        # Cache: chunk_idx -> {"data": Tensor, "mask": Tensor, "shot_ids": list}
        self.cache = LRUCache(max_size=cache_size)

        logger.info(
            f"[Dataset] Ready: {len(self)} samples, {len(self.chunks)} chunks, cache_size={cache_size}"
        )

    def __len__(self) -> int:
        """
        Return the total number of samples in the selected split.

        Returns:
            Number of samples represented by the dataset's global index.
        """
        return len(self.global_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single seismic sample and its mask.

        The global dataset index is resolved to the corresponding chunk
        and local sample index. If the required chunk is not currently
        cached, it is loaded from disk and inserted into the LRU cache.

        The returned data tensor receives an additional leading dimension
        using ``unsqueeze(0)`` and is made contiguous in memory.

        Args:
            idx:
                Zero-based global sample index.

        Returns:
            A tuple ``(data, mask)`` where:

            - ``data`` is the seismic sample tensor with a leading
              singleton dimension.
            - ``mask`` is the corresponding mask tensor.

        Raises:
            IndexError:
                If ``idx`` is outside the valid dataset range.
        """
        if idx < 0 or idx >= len(self.global_index):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")

        chunk_idx = self.chunk_indices[idx]
        local_idx = self.chunk_offsets[idx]
        shot_id = self.shot_ids[idx]

        logger.debug(
            f"[Dataset] GET idx={idx} → chunk={chunk_idx}, local={local_idx}, shot={shot_id}"
        )

        # Load chunk if not in cache
        if chunk_idx not in self.cache:
            logger.debug(f"[Dataset] CHUNK {chunk_idx} NOT in cache → loading")
            self._load_chunk(chunk_idx)
        else:
            logger.debug(f"[Dataset] CHUNK {chunk_idx} in cache (hit)")

        cached_item = self.cache.get(chunk_idx)
        assert cached_item is not None, f"Chunk {chunk_idx} failed to load into cache"

        data = cached_item["data"][local_idx]
        mask = cached_item["mask"][local_idx]

        return data.unsqueeze(0).contiguous(), mask.contiguous()

    def _load_chunk(self, chunk_idx: int) -> None:
        """
        Load a dataset chunk from disk and insert it into the LRU cache.

        The chunk is loaded onto CPU memory using ``torch.load`` and only
        the fields required by the dataset are stored in the cache.

        Args:
            chunk_idx:
                Zero-based index of the chunk within the selected split.

        Raises:
            FileNotFoundError:
                If the chunk file does not exist.

            KeyError:
                If the serialized chunk does not contain the expected
                ``data``, ``mask``, or ``shot_ids`` fields.
        """
        chunk = self.chunks[chunk_idx]
        chunk_path = self.chunk_dir / chunk["filename"]

        logger.info(f"[Dataset] LOAD chunk {chunk_idx}: {chunk_path.name}")

        chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)

        self.cache.put(
            chunk_idx,
            {
                "data": chunk_data["data"],
                "mask": chunk_data["mask"],
                "shot_ids": chunk_data["shot_ids"],
            },
        )

        # Log cache stats after load
        stats = self.cache.get_stats()
        logger.info(
            f"[Dataset] CACHE: {stats['size']}/{stats['max_size']} | hit_rate={stats['hit_rate']:.1%}"
        )

    def get_shot_id(self, idx: int) -> Any:
        """
        Return the shot identifier associated with a sample.

        Args:
            idx:
                Zero-based global sample index.

        Returns:
            Identifier of the seismic shot corresponding to ``idx``.
        """
        return self.shot_ids[idx]

    def get_chunk_stats(self) -> dict[str, Any]:
        """
        Return statistics describing the dataset and chunk cache.

        The returned dictionary contains the number of chunks, total number
        of samples, current LRU cache statistics, and the size of each
        chunk.

        Returns:
            Dictionary with the following keys:

            ``total_chunks``:
                Number of chunks in the selected split.

            ``total_samples``:
                Number of samples in the dataset.

            ``cache_size``:
                Current cache statistics returned by ``LRUCache``.

            ``chunk_sizes``:
                List containing the number of samples in each chunk.
        """
        stats = {
            "total_chunks": len(self.chunks),
            "total_samples": len(self),
            "cache_size": self.cache.get_stats(),
            "chunk_sizes": [c["n_shots"] for c in self.chunks],
        }
        logger.debug(f"[Dataset] STATS: {stats}")
        return stats


class ChunkedDataManager:
    """
    Lazily manages ``ChunkedSeismicDataset`` instances for dataset splits.

    The manager provides a single access point for creating and reusing
    datasets for splits such as ``train``, ``validation``, and ``test``.
    A dataset is instantiated only when ``get_dataset`` is called for the
    first time for a particular split.

    Created datasets are retained internally, so subsequent calls for the
    same split return the existing dataset instance rather than creating
    another dataset and cache.

    Args:
        chunk_dir:
            Directory containing the serialized dataset chunks.

        manifest:
            Dataset manifest describing the available chunks and dataset
            configuration.

        cache_size:
            Maximum number of chunks cached by each dataset instance.

        shuffle_chunks:
            Whether chunk-level shuffling is enabled for the managed
            datasets.

    Example:
        >>> manager = ChunkedDataManager(
        ...     chunk_dir="data/chunks",
        ...     manifest=manifest,
        ...     cache_size=3,
        ... )
        >>> train_dataset = manager.get_dataset("train")
        >>> val_dataset = manager.get_dataset("val")
    """

    def __init__(
        self,
        chunk_dir: str | Path,
        manifest: dict[str, Any],
        cache_size: int = 3,
        shuffle_chunks: bool = True,
    ) -> None:
        self.chunk_dir = Path(chunk_dir)
        self.manifest = manifest
        self.cache_size = cache_size
        self.shuffle_chunks = shuffle_chunks

        logger.debug(f"[Manager] INIT cache_size={cache_size}")
        self._datasets: dict[str, ChunkedSeismicDataset] = {}

    def get_dataset(self, split: str) -> ChunkedSeismicDataset:
        """
        Return the dataset associated with a given split.

        If the requested split has not been created yet, a new
        ``ChunkedSeismicDataset`` is initialized and stored for reuse.

        Args:
            split:
                Name of the dataset split, such as ``"train"``, ``"val"``,
                or ``"test"``.

        Returns:
            The ``ChunkedSeismicDataset`` instance for the requested split.

        Raises:
            ValueError:
                If the manifest does not contain chunks for the requested
                split.
        """
        logger.debug(f"[Manager] GET_DATASET split={split}")

        if split not in self._datasets:
            logger.debug(f"[Manager] Creating new dataset for split={split}")
            self._datasets[split] = ChunkedSeismicDataset(
                self.chunk_dir,
                self.manifest,
                split=split,
                cache_size=self.cache_size,
                shuffle_chunks=self.shuffle_chunks,
            )
        return self._datasets[split]
