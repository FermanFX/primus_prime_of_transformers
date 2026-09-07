from pathlib import Path

import pytest
import torch

from src.data.chunked_dataset import ChunkedDataManager, ChunkedSeismicDataset


@pytest.fixture
def mock_dataset_env(tmp_path: Path):
    """Generates dummy chunk files and a manifest covering train/val/test splits."""
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()

    # 10 chunks total: 7 train, 2 val, 1 test (70% / 20% / 10% split ratio)
    chunks_info = []
    total_shots = 0

    for i in range(10):
        if i < 7:
            split = "train"
        elif i < 9:
            split = "val"
        else:
            split = "test"

        filename = f"chunk_{i}.pt"
        n_shots = 5
        shot_ids = [f"shot_{i}_{j}" for j in range(n_shots)]

        # Save dummy torch file
        torch.save(
            {
                "data": torch.randn(n_shots, 10, 100),
                "mask": torch.ones(n_shots, 100),
                "shot_ids": shot_ids,
            },
            chunk_dir / filename,
        )

        chunks_info.append(
            {
                "filename": filename,
                "split": split,
                "n_shots": n_shots,
                "shot_ids": shot_ids,
            }
        )
        total_shots += n_shots

    manifest = {
        "chunks": chunks_info,
        "config": {"n_samples": total_shots, "target_traces": 100},
    }

    return chunk_dir, manifest


# ============================================================================
# ISSUE #8: Test dataset splits (train/val/test)
# ============================================================================
class TestIssue8DatasetSplits:

    def test_get_dataset_splits_returns_correct_data(self, mock_dataset_env):
        chunk_dir, manifest = mock_dataset_env
        manager = ChunkedDataManager(chunk_dir, manifest)

        train_ds = manager.get_dataset("train")
        val_ds = manager.get_dataset("val")
        test_ds = manager.get_dataset("test")

        assert train_ds.split == "train"
        assert val_ds.split == "val"
        assert test_ds.split == "test"

        # Check sample counts (35 train, 10 val, 5 test)
        assert len(train_ds) == 35
        assert len(val_ds) == 10
        assert len(test_ds) == 5

    def test_splits_are_disjoint_and_cover_all_data(self, mock_dataset_env):
        chunk_dir, manifest = mock_dataset_env
        manager = ChunkedDataManager(chunk_dir, manifest)

        train_ds = manager.get_dataset("train")
        val_ds = manager.get_dataset("val")
        test_ds = manager.get_dataset("test")

        train_shots = set(train_ds.shot_ids)
        val_shots = set(val_ds.shot_ids)
        test_shots = set(test_ds.shot_ids)

        # 1. Test disjointness (no overlap between splits)
        assert train_shots.isdisjoint(val_shots)
        assert train_shots.isdisjoint(test_shots)
        assert val_shots.isdisjoint(test_shots)

        # 2. Test coverage (must cover all shot_ids from manifest)
        all_manifest_shots = set()
        for c in manifest["chunks"]:
            all_manifest_shots.update(c["shot_ids"])

        combined_shots = train_shots | val_shots | test_shots
        assert combined_shots == all_manifest_shots

    def test_different_split_ratios(self, tmp_path):
        """Test custom split ratios e.g. 80/10/10."""
        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()

        chunks_info = []
        # 8 train chunks, 1 val chunk, 1 test chunk
        for i in range(10):
            split = "train" if i < 8 else ("val" if i == 8 else "test")
            filename = f"chunk_{i}.pt"
            torch.save(
                {
                    "data": torch.randn(2, 5),
                    "mask": torch.ones(2, 5),
                    "shot_ids": [i * 2, i * 2 + 1],
                },
                chunk_dir / filename,
            )
            chunks_info.append(
                {
                    "filename": filename,
                    "split": split,
                    "n_shots": 2,
                    "shot_ids": [i * 2, i * 2 + 1],
                }
            )

        manifest = {"chunks": chunks_info}
        manager = ChunkedDataManager(chunk_dir, manifest)

        assert len(manager.get_dataset("train")) == 16  # 80%
        assert len(manager.get_dataset("val")) == 2  # 10%
        assert len(manager.get_dataset("test")) == 2  # 10%


# ============================================================================
# ISSUE #9: Test cache integration with chunked dataset
# ============================================================================
class TestIssue9CacheIntegration:

    def test_chunks_cached_on_first_load(self, mock_dataset_env):
        chunk_dir, manifest = mock_dataset_env
        dataset = ChunkedSeismicDataset(
            chunk_dir, manifest, split="train", cache_size=3
        )

        assert len(dataset.cache) == 0

        # Load item from chunk 0
        _ = dataset[0]
        assert 0 in dataset.cache
        assert len(dataset.cache) == 1

    def test_cache_eviction_and_memory_release(self, mock_dataset_env):
        chunk_dir, manifest = mock_dataset_env
        # Cache size = 2
        dataset = ChunkedSeismicDataset(
            chunk_dir, manifest, split="train", cache_size=2
        )

        # Access chunk 0 (idx=0)
        _ = dataset[0]
        # Access chunk 1 (idx=5)
        _ = dataset[5]
        assert len(dataset.cache) == 2
        assert 0 in dataset.cache and 1 in dataset.cache

        # Access chunk 2 (idx=10) -> Chunk 0 must be evicted (LRU)
        _ = dataset[10]
        assert len(dataset.cache) == 2
        assert 0 not in dataset.cache  # Evicted
        assert 1 in dataset.cache
        assert 2 in dataset.cache

    def test_evicted_chunks_reloaded_when_requested(self, mock_dataset_env):
        chunk_dir, manifest = mock_dataset_env
        dataset = ChunkedSeismicDataset(
            chunk_dir, manifest, split="train", cache_size=1
        )

        _ = dataset[0]  # Load chunk 0
        assert 0 in dataset.cache

        _ = dataset[5]  # Load chunk 1, evict chunk 0
        assert 0 not in dataset.cache
        assert 1 in dataset.cache

        _ = dataset[0]  # Reload chunk 0, evict chunk 1
        assert 0 in dataset.cache
        assert 1 not in dataset.cache

    def test_cache_hit_rate_calculation(self, mock_dataset_env):
        chunk_dir, manifest = mock_dataset_env
        dataset = ChunkedSeismicDataset(
            chunk_dir, manifest, split="train", cache_size=3
        )

        # Read 3 items from Chunk 0
        _ = dataset[0]
        _ = dataset[1]
        _ = dataset[2]

        stats = dataset.cache.get_stats()
        assert stats["misses"] == 0
        assert stats["hits"] == 3
        assert stats["hit_rate"] == pytest.approx(1.0)

    def test_cache_behavior_with_different_max_sizes(self, mock_dataset_env):
        chunk_dir, manifest = mock_dataset_env

        # Large cache (size=10)
        ds_large = ChunkedSeismicDataset(
            chunk_dir, manifest, split="train", cache_size=10
        )
        for idx in range(0, 35, 5):  # Touch all 7 chunks
            _ = ds_large[idx]
        assert len(ds_large.cache) == 7

        # Small cache (size=1)
        ds_small = ChunkedSeismicDataset(
            chunk_dir, manifest, split="train", cache_size=1
        )
        for idx in range(0, 35, 5):
            _ = ds_small[idx]
        assert len(ds_small.cache) == 1