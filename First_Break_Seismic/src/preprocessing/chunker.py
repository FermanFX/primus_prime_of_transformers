"""
Utilities for assigning shots to dataset splits and grouping them into chunks.
"""

from typing import Any

import numpy as np
from loguru import logger


class Chunker:
    """Assign shot IDs to train/validation/test splits and group them into chunks.

    The class provides two main operations:
    1. Randomly split shot IDs into train, validation, and test sets.
    2. Group shot IDs into fixed-size chunks for further processing.

    Attributes:
        chunk_size: Maximum number of shots contained in a single chunk.
        train_split: Fraction of shots assigned to the training split.
        val_split: Fraction of shots assigned to the validation split.
        test_split: Fraction of shots assigned to the test split.
        random_seed: Seed used to make the random split reproducible.
    """

    def __init__(
        self,
        chunk_size: int = 69,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42,
    ):
        """Initialize a Chunker.

        Args:
            chunk_size: Maximum number of shots per chunk.
            train_split: Fraction of the dataset allocated to training.
            val_split: Fraction of the dataset allocated to validation.
            test_split: Fraction of the dataset allocated to testing.
            random_seed: Random seed used when shuffling shot IDs.

        Raises:
            ValueError: If ``chunk_size`` is not positive or if the split
                fractions are invalid.
        """
        self.chunk_size = chunk_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed

    def assign_splits(self, shot_ids: np.ndarray) -> dict[str, list[int]]:
        """Randomly assign shot IDs to train, validation, and test splits.

        The input shot IDs are shuffled using the configured random seed
        before being divided according to the configured split fractions.
        Any remaining shots after the train and validation allocations are
        assigned to the test split.

        Args:
            shot_ids: NumPy array containing the shot IDs to split.

        Returns:
            A dictionary containing three lists:
                - ``train``: Shot IDs assigned to the training split.
                - ``val``: Shot IDs assigned to the validation split.
                - ``test``: Shot IDs assigned to the test split.

        Example:
            >>> chunker = Chunker(train_split=0.8, val_split=0.1, test_split=0.1)
            >>> splits = chunker.assign_splits(np.array([1, 2, 3, 4, 5]))
            >>> set(splits) == {"train", "val", "test"}
            True
        """
        np.random.seed(self.random_seed)
        n_shots = len(shot_ids)

        # Shuffle
        shuffled_indices = np.random.permutation(n_shots)
        shuffled_shots = shot_ids[shuffled_indices]

        # Split
        n_train = int(n_shots * self.train_split)
        n_val = int(n_shots * self.val_split)

        train_shots = shuffled_shots[:n_train].tolist()
        val_shots = shuffled_shots[n_train : n_train + n_val].tolist()
        test_shots = shuffled_shots[n_train + n_val :].tolist()

        logger.info(
            f"Split assignment: Train={len(train_shots)}, Val={len(val_shots)}, Test={len(test_shots)}"
        )

        return {"train": train_shots, "val": val_shots, "test": test_shots}

    def create_chunks(self, shot_ids: list[int]) -> list[dict[str, Any]]:
        """Group shot IDs into fixed-size chunks.

        The input list is divided into consecutive chunks of at most
        ``chunk_size`` shots. Each chunk contains metadata describing its
        ID, shot IDs, number of shots, and the original start/end indices.

        Args:
            shot_ids: List of shot IDs to group into chunks. The original
                ordering is preserved.

        Returns:
            A list of dictionaries, where each dictionary contains:
                - ``id``: One-based chunk ID.
                - ``shot_ids``: Shot IDs belonging to the chunk.
                - ``n_shots``: Number of shots in the chunk.
                - ``start_idx``: Zero-based starting index in the input list.
                - ``end_idx``: Zero-based ending index in the input list.

        Example:
            >>> chunker = Chunker(chunk_size=2)
            >>> chunks = chunker.create_chunks([1, 2, 3, 4, 5])
            >>> [chunk["shot_ids"] for chunk in chunks]
            [[1, 2], [3, 4], [5]]
        """
        chunks = []
        chunk_id = 1

        for i in range(0, len(shot_ids), self.chunk_size):
            chunk_shots = shot_ids[i : i + self.chunk_size]
            chunks.append(
                {
                    "id": chunk_id,
                    "shot_ids": chunk_shots,
                    "n_shots": len(chunk_shots),
                    "start_idx": i,
                    "end_idx": i + len(chunk_shots) - 1,
                }
            )
            chunk_id += 1

        return chunks
