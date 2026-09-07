"""Tests for chunker module."""

import numpy as np

from src.preprocessing.chunker import Chunker


class TestAssignSplits:
    """Tests for assign_splits method."""

    def test_assign_splits_default_proportions(self):
        """Test default 80/10/10 split proportions."""
        chunker = Chunker()

        shot_ids = np.arange(100)
        splits = chunker.assign_splits(shot_ids)

        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_assign_splits_contains_all_shots(self):
        """Test that every shot is assigned exactly once."""
        chunker = Chunker()

        shot_ids = np.arange(100)
        splits = chunker.assign_splits(shot_ids)

        assigned = (
            splits["train"]
            + splits["val"]
            + splits["test"]
        )

        assert len(assigned) == 100
        assert sorted(assigned) == list(range(100))

    def test_same_seed_produces_same_splits(self):
        """Test reproducibility with the same random seed."""
        shot_ids = np.arange(100)

        chunker1 = Chunker(random_seed=42)
        chunker2 = Chunker(random_seed=42)

        splits1 = chunker1.assign_splits(shot_ids)
        splits2 = chunker2.assign_splits(shot_ids)

        assert splits1 == splits2

    def test_different_seed_produces_different_splits(self):
        """Test that different seeds produce different splits."""
        shot_ids = np.arange(100)

        chunker1 = Chunker(random_seed=42)
        chunker2 = Chunker(random_seed=123)

        splits1 = chunker1.assign_splits(shot_ids)
        splits2 = chunker2.assign_splits(shot_ids)

        assert splits1 != splits2

    def test_custom_split_proportions(self):
        """Test custom 70/20/10 split proportions."""
        chunker = Chunker(
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
        )

        shot_ids = np.arange(100)
        splits = chunker.assign_splits(shot_ids)

        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 20
        assert len(splits["test"]) == 10


class TestCreateChunks:
    """Tests for create_chunks method."""

    def test_default_chunk_size(self):
        """Test chunk creation with default chunk size of 69."""
        chunker = Chunker()

        shot_ids = list(range(150))
        chunks = chunker.create_chunks(shot_ids)

        assert len(chunks) == 3
        assert chunks[0]["n_shots"] == 69
        assert chunks[1]["n_shots"] == 69
        assert chunks[2]["n_shots"] == 12

    def test_custom_chunk_size(self):
        """Test chunk creation with custom chunk size of 50."""
        chunker = Chunker(chunk_size=50)

        shot_ids = list(range(120))
        chunks = chunker.create_chunks(shot_ids)

        assert len(chunks) == 3
        assert chunks[0]["n_shots"] == 50
        assert chunks[1]["n_shots"] == 50
        assert chunks[2]["n_shots"] == 20

    def test_last_chunk_contains_remaining_shots(self):
        """Test that the last chunk contains remaining shots."""
        chunker = Chunker(chunk_size=50)

        shot_ids = list(range(123))
        chunks = chunker.create_chunks(shot_ids)

        assert len(chunks) == 3
        assert chunks[-1]["shot_ids"] == list(range(100, 123))
        assert chunks[-1]["n_shots"] == 23

    def test_chunk_size_is_respected(self):
        """Test that no chunk exceeds the configured chunk size."""
        chunker = Chunker(chunk_size=10)

        shot_ids = list(range(95))
        chunks = chunker.create_chunks(shot_ids)

        for chunk in chunks:
            assert chunk["n_shots"] <= 10
            assert len(chunk["shot_ids"]) <= 10

    def test_chunk_ids_are_sequential(self):
        """Test that chunk IDs are sequential starting from 1."""
        chunker = Chunker(chunk_size=10)

        shot_ids = list(range(55))
        chunks = chunker.create_chunks(shot_ids)

        assert [chunk["id"] for chunk in chunks] == [1, 2, 3, 4, 5, 6]

    def test_chunks_preserve_shot_order(self):
        """Test that chunks preserve the input shot order."""
        chunker = Chunker(chunk_size=10)

        shot_ids = list(range(25))
        chunks = chunker.create_chunks(shot_ids)

        reconstructed = []

        for chunk in chunks:
            reconstructed.extend(chunk["shot_ids"])

        assert reconstructed == shot_ids

    def test_chunk_indices_are_correct(self):
        """Test start and end indices."""
        chunker = Chunker(chunk_size=10)

        shot_ids = list(range(25))
        chunks = chunker.create_chunks(shot_ids)

        assert chunks[0]["start_idx"] == 0
        assert chunks[0]["end_idx"] == 9
        assert chunks[1]["start_idx"] == 10
        assert chunks[1]["end_idx"] == 19
        assert chunks[2]["start_idx"] == 20
        assert chunks[2]["end_idx"] == 24

    def test_empty_input(self):
        """Test that empty input produces no chunks."""
        chunker = Chunker(chunk_size=10)

        chunks = chunker.create_chunks([])

        assert chunks == []


class TestReproducibility:
    """Tests for reproducibility."""

    def test_same_seed_produces_same_chunks(self):
        """Test that the same seed produces identical chunk assignments."""
        shot_ids = np.arange(200)

        chunker1 = Chunker(chunk_size=69, random_seed=42)
        chunker2 = Chunker(chunk_size=69, random_seed=42)

        splits1 = chunker1.assign_splits(shot_ids)
        splits2 = chunker2.assign_splits(shot_ids)

        chunks1 = {
            split: chunker1.create_chunks(ids)
            for split, ids in splits1.items()
        }

        chunks2 = {
            split: chunker2.create_chunks(ids)
            for split, ids in splits2.items()
        }

        assert chunks1 == chunks2

    def test_different_seeds_change_assignment(self):
        """Test that different seeds change the train split."""
        shot_ids = np.arange(200)

        splits1 = Chunker(random_seed=42).assign_splits(shot_ids)
        splits2 = Chunker(random_seed=123).assign_splits(shot_ids)

        assert splits1["train"] != splits2["train"]
