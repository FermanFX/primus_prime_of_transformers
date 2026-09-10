"""
Unit tests for :mod:`src.preprocessing.manifest`.

Covers:
    * ``compute_checksum`` — SHA-256 truncated checksum computation.
    * ``get_next_version`` — patch version increment logic.
    * ``generate_manifest`` — manifest structure and metadata.
    * ``save_manifest`` / ``load_manifest`` — disk round-trip.
    * ``validate_manifest`` — required keys, splits, and totals.
    * ``get_chunk_paths`` — split → chunk path mapping.
    * ``get_manifest_stats`` — summary statistics.
    * Checksum corruption detection (manifest + chunk checksums).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.preprocessing.manifest import (
    compute_checksum,
    generate_manifest,
    get_chunk_paths,
    get_manifest_stats,
    get_next_version,
    load_manifest,
    save_manifest,
    validate_manifest,
)

# ============================================================
# HELPERS
# ============================================================


def _make_chunk_entry(
    chunk_id: int,
    split: str,
    shot_ids: list[int],
    start_idx: int = 0,
) -> dict:
    """Build a single chunk metadata entry."""
    return {
        "id": chunk_id,
        "shot_ids": shot_ids,
        "n_shots": len(shot_ids),
        "start_idx": start_idx,
        "end_idx": start_idx + len(shot_ids) - 1,
    }


def _make_chunks() -> dict[str, list[dict]]:
    """Return a small, valid chunk dictionary with all three splits."""
    return {
        "train": [_make_chunk_entry(1, "train", [1, 2, 3])],
        "val": [_make_chunk_entry(2, "val", [4])],
        "test": [_make_chunk_entry(3, "test", [5])],
    }


def _valid_manifest() -> dict:
    """Return a fully valid manifest dictionary for validation tests."""
    return {
        "dataset": "TestDataset",
        "version": "1.0.0",
        "created": "2026-01-01T00:00:00+00:00",
        "config": {},
        "total_shots": 6,
        "chunks": [
            {
                "id": 1,
                "filename": "chunk_001_train.pt",
                "split": "train",
                "shot_ids": [1, 2, 3],
                "n_shots": 3,
            },
            {
                "id": 2,
                "filename": "chunk_002_val.pt",
                "split": "val",
                "shot_ids": [4],
                "n_shots": 1,
            },
            {
                "id": 3,
                "filename": "chunk_003_test.pt",
                "split": "test",
                "shot_ids": [5, 6],
                "n_shots": 2,
            },
        ],
    }


# ============================================================
# 1. compute_checksum
# ============================================================


class TestComputeChecksum:
    """Tests for :func:`compute_checksum`."""

    def test_checksum_is_16_characters(self, tmp_path: Path) -> None:
        """Checksum must be a 16-character hexadecimal string."""
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello world")

        checksum = compute_checksum(filepath)

        assert isinstance(checksum, str)
        assert len(checksum) == 16
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_same_content_has_same_checksum(self, tmp_path: Path) -> None:
        """Identical content in different files must yield the same checksum."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("same content")
        file2.write_text("same content")

        assert compute_checksum(file1) == compute_checksum(file2)

    def test_different_content_has_different_checksum(self, tmp_path: Path) -> None:
        """Different content must yield different checksums."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("content A")
        file2.write_text("content B")

        assert compute_checksum(file1) != compute_checksum(file2)

    def test_empty_file_checksum(self, tmp_path: Path) -> None:
        """An empty file must still produce a valid 16-char checksum."""
        filepath = tmp_path / "empty.bin"
        filepath.write_bytes(b"")

        checksum = compute_checksum(filepath)

        assert isinstance(checksum, str)
        assert len(checksum) == 16

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Computing a checksum for a missing file must raise."""
        filepath = tmp_path / "does_not_exist.bin"

        with pytest.raises(FileNotFoundError):
            compute_checksum(filepath)

    def test_large_file_is_processed_in_chunks(self, tmp_path: Path) -> None:
        """A file larger than one read block must still produce a checksum."""
        filepath = tmp_path / "large.bin"
        # 10 KB, larger than the 4096-byte read block
        filepath.write_bytes(b"x" * 10_000)

        checksum = compute_checksum(filepath)

        assert len(checksum) == 16

    def test_checksum_detects_single_byte_change(self, tmp_path: Path) -> None:
        """A single-byte change must alter the checksum."""
        filepath = tmp_path / "data.bin"

        filepath.write_bytes(b"original content")
        original = compute_checksum(filepath)

        filepath.write_bytes(b"Original content")  # capital O
        modified = compute_checksum(filepath)

        assert original != modified


# ============================================================
# 2. get_next_version
# ============================================================


class TestGetNextVersion:
    """Tests for :func:`get_next_version`."""

    def test_missing_manifest_returns_initial_version(self, tmp_path: Path) -> None:
        """A missing manifest must yield the initial version ``1.0.0``."""
        path = tmp_path / "manifest.json"

        assert get_next_version(path) == "1.0.0"

    def test_existing_manifest_increments_patch(self, tmp_path: Path) -> None:
        """An existing manifest must have its patch component incremented."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"version": "1.0.3"}))

        assert get_next_version(path) == "1.0.4"

    def test_invalid_json_returns_initial_version(self, tmp_path: Path) -> None:
        """Invalid JSON must fall back to ``1.0.0``."""
        path = tmp_path / "manifest.json"
        path.write_text("{invalid json")

        assert get_next_version(path) == "1.0.0"

    def test_invalid_version_returns_initial_version(self, tmp_path: Path) -> None:
        """A non-semver version string must fall back to ``1.0.0``."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"version": "invalid"}))

        assert get_next_version(path) == "1.0.0"

    def test_version_with_two_parts_is_invalid(self, tmp_path: Path) -> None:
        """A version with only two components is invalid."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"version": "1.0"}))

        assert get_next_version(path) == "1.0.0"

    def test_version_with_non_numeric_parts_is_invalid(self, tmp_path: Path) -> None:
        """A version with non-numeric components is invalid."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"version": "1.x.0"}))

        assert get_next_version(path) == "1.0.0"

    def test_missing_version_key_returns_initial(self, tmp_path: Path) -> None:
        """A manifest without a ``version`` key falls back to ``1.0.0``."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"dataset": "Test"}))

        assert get_next_version(path) == "1.0.1"


# ============================================================
# 3. generate_manifest
# ============================================================


class TestGenerateManifest:
    """Tests for :func:`generate_manifest`."""

    def test_generate_manifest_without_chunk_files(self, tmp_path: Path) -> None:
        """Missing chunk files must still be listed, with ``None`` checksum."""
        chunks = _make_chunks()
        config = {"chunk_size": 3}

        manifest = generate_manifest(
            dataset_name="TestDataset",
            chunks=chunks,
            config=config,
            chunk_dir=tmp_path,
            total_shots=5,
            total_traces=100,
            increment_version=False,
        )

        assert manifest["dataset"] == "TestDataset"
        assert manifest["version"] == "1.0.0"
        assert manifest["config"] == config
        assert manifest["total_shots"] == 5
        assert manifest["total_traces"] == 100
        assert len(manifest["chunks"]) == 3

        assert manifest["chunks"][0]["filename"] == "chunk_001_train.pt"
        assert manifest["chunks"][0]["checksum"] is None

    def test_generate_manifest_uses_file_size_and_checksum(
        self, tmp_path: Path
    ) -> None:
        """Existing chunk files must contribute size and checksum metadata."""
        chunk_path = tmp_path / "chunk_001_train.pt"
        chunk_path.write_bytes(b"test chunk content")

        chunks = {
            "train": [_make_chunk_entry(1, "train", [1, 2])],
        }

        manifest = generate_manifest(
            dataset_name="TestDataset",
            chunks=chunks,
            config={},
            chunk_dir=tmp_path,
            total_shots=2,
            total_traces=10,
            increment_version=False,
        )

        chunk = manifest["chunks"][0]

        assert chunk["file_size_mb"] >= 0
        assert chunk["checksum"] == compute_checksum(chunk_path)

    def test_generate_manifest_structure(self, tmp_path: Path) -> None:
        """All required top-level keys must be present."""
        manifest = generate_manifest(
            dataset_name="TestDataset",
            chunks=_make_chunks(),
            config={"chunk_size": 3},
            chunk_dir=tmp_path,
            total_shots=5,
            total_traces=100,
            increment_version=False,
        )

        required = {
            "dataset",
            "version",
            "created",
            "config",
            "total_shots",
            "total_traces",
            "chunks",
            "manifest_checksum",
        }
        assert required.issubset(manifest.keys())

    def test_generate_manifest_chunk_metadata(self, tmp_path: Path) -> None:
        """Each chunk entry must contain all required metadata fields."""
        manifest = generate_manifest(
            dataset_name="TestDataset",
            chunks=_make_chunks(),
            config={},
            chunk_dir=tmp_path,
            total_shots=5,
            total_traces=100,
            increment_version=False,
        )

        for chunk in manifest["chunks"]:
            for key in (
                "id",
                "filename",
                "split",
                "shot_ids",
                "n_shots",
                "start_idx",
                "end_idx",
                "file_size_mb",
                "checksum",
            ):
                assert key in chunk, f"Missing key: {key}"

    def test_generate_manifest_increments_version(self, tmp_path: Path) -> None:
        """``increment_version=True`` must bump the patch version."""
        # Write an initial manifest to bump from
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"version": "1.0.5"}))

        manifest = generate_manifest(
            dataset_name="TestDataset",
            chunks=_make_chunks(),
            config={},
            chunk_dir=tmp_path,
            total_shots=5,
            total_traces=100,
            increment_version=True,
        )

        assert manifest["version"] == "1.0.6"

    def test_generate_manifest_does_not_increment_when_disabled(
        self, tmp_path: Path
    ) -> None:
        """``increment_version=False`` must always use ``1.0.0``."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"version": "1.0.5"}))

        manifest = generate_manifest(
            dataset_name="TestDataset",
            chunks=_make_chunks(),
            config={},
            chunk_dir=tmp_path,
            total_shots=5,
            total_traces=100,
            increment_version=False,
        )

        assert manifest["version"] == "1.0.0"

    def test_generate_manifest_filename_per_split(self, tmp_path: Path) -> None:
        """Chunk filenames must embed the chunk ID and split name."""
        manifest = generate_manifest(
            dataset_name="TestDataset",
            chunks=_make_chunks(),
            config={},
            chunk_dir=tmp_path,
            total_shots=5,
            total_traces=100,
            increment_version=False,
        )

        filenames = {c["split"]: c["filename"] for c in manifest["chunks"]}

        assert filenames["train"] == "chunk_001_train.pt"
        assert filenames["val"] == "chunk_002_val.pt"
        assert filenames["test"] == "chunk_003_test.pt"


# ============================================================
# 4. save_manifest / load_manifest
# ============================================================


class TestSaveLoadManifest:
    """Tests for :func:`save_manifest` and :func:`load_manifest`."""

    def test_save_manifest_creates_file(self, tmp_path: Path) -> None:
        """Saving must create the file and attach a checksum."""
        path = tmp_path / "nested" / "manifest.json"

        manifest: dict[str, Any] = {
            "dataset": "TestDataset",
            "version": "1.0.0",
            "created": "2026-01-01T00:00:00+00:00",
            "config": {},
            "total_shots": 3,
            "total_traces": 10,
            "chunks": [],
            "manifest_checksum": None,
        }

        save_manifest(manifest, path)

        assert path.exists()
        assert manifest["manifest_checksum"] is not None
        assert len(manifest["manifest_checksum"]) == 16

    def test_load_manifest_returns_saved_data(self, tmp_path: Path) -> None:
        """Loading must return the same data that was saved."""
        path = tmp_path / "manifest.json"

        manifest = {
            "dataset": "TestDataset",
            "version": "1.0.0",
            "created": "2026-01-01T00:00:00+00:00",
            "config": {"chunk_size": 3},
            "total_shots": 3,
            "total_traces": 10,
            "chunks": [
                {
                    "id": 1,
                    "filename": "chunk_001_train.pt",
                    "split": "train",
                    "shot_ids": [1, 2, 3],
                    "n_shots": 3,
                }
            ],
            "manifest_checksum": None,
        }

        save_manifest(manifest, path)
        loaded = load_manifest(path)

        assert loaded["dataset"] == "TestDataset"
        assert loaded["version"] == "1.0.0"
        assert loaded["total_shots"] == 3
        assert len(loaded["chunks"]) == 1

    def test_load_missing_manifest_raises(self, tmp_path: Path) -> None:
        """Loading a missing manifest must raise ``FileNotFoundError``."""
        path = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError):
            load_manifest(path)

    def test_save_manifest_creates_parent_directories(self, tmp_path: Path) -> None:
        """Nested parent directories must be created automatically."""
        path = tmp_path / "a" / "b" / "c" / "manifest.json"

        manifest = {
            "dataset": "TestDataset",
            "version": "1.0.0",
            "created": "2026-01-01T00:00:00+00:00",
            "config": {},
            "total_shots": 0,
            "total_traces": 0,
            "chunks": [],
            "manifest_checksum": None,
        }

        save_manifest(manifest, path)

        assert path.exists()

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        """Loading an invalid JSON file must raise ``JSONDecodeError``."""
        path = tmp_path / "manifest.json"
        path.write_text("{not valid json")

        with pytest.raises(json.JSONDecodeError):
            load_manifest(path)

    def test_load_manifest_round_trip_preserves_checksum(self, tmp_path: Path) -> None:
        """The manifest checksum must be preserved across save/load."""
        path = tmp_path / "manifest.json"

        manifest = {
            "dataset": "TestDataset",
            "version": "1.0.0",
            "created": "2026-01-01T00:00:00+00:00",
            "config": {},
            "total_shots": 0,
            "total_traces": 0,
            "chunks": [],
            "manifest_checksum": None,
        }

        save_manifest(manifest, path)
        saved_checksum = manifest["manifest_checksum"]

        loaded = load_manifest(path)

        assert loaded["manifest_checksum"] == saved_checksum


# ============================================================
# 5. validate_manifest
# ============================================================


class TestValidateManifest:
    """Tests for :func:`validate_manifest`."""

    def test_valid_manifest(self) -> None:
        """A fully valid manifest must return ``True``."""
        assert validate_manifest(_valid_manifest()) is True

    def test_missing_required_key(self) -> None:
        """Missing top-level keys must fail validation."""
        manifest = _valid_manifest()
        del manifest["dataset"]

        assert validate_manifest(manifest) is False

    def test_missing_version_key(self) -> None:
        """A missing ``version`` key must fail validation."""
        manifest = _valid_manifest()
        del manifest["version"]

        assert validate_manifest(manifest) is False

    def test_missing_created_key(self) -> None:
        """A missing ``created`` key must fail validation."""
        manifest = _valid_manifest()
        del manifest["created"]

        assert validate_manifest(manifest) is False

    def test_missing_config_key(self) -> None:
        """A missing ``config`` key must fail validation."""
        manifest = _valid_manifest()
        del manifest["config"]

        assert validate_manifest(manifest) is False

    def test_missing_chunks_key(self) -> None:
        """A missing ``chunks`` key must fail validation."""
        manifest = _valid_manifest()
        del manifest["chunks"]

        assert validate_manifest(manifest) is False

    def test_empty_chunks(self) -> None:
        """An empty ``chunks`` list must fail validation."""
        manifest = _valid_manifest()
        manifest["chunks"] = []

        assert validate_manifest(manifest) is False

    def test_invalid_split(self) -> None:
        """An unknown split name must fail validation."""
        manifest = _valid_manifest()
        manifest["chunks"][0]["split"] = "invalid"

        assert validate_manifest(manifest) is False

    def test_total_shots_mismatch(self) -> None:
        """A ``total_shots`` value that disagrees with the chunks must fail."""
        manifest = _valid_manifest()
        manifest["total_shots"] = 999

        assert validate_manifest(manifest) is False

    def test_missing_chunk_key(self) -> None:
        """A chunk missing a required key must fail validation."""
        manifest = _valid_manifest()
        del manifest["chunks"][0]["filename"]

        assert validate_manifest(manifest) is False

    def test_missing_split_is_not_fatal(self) -> None:
        """Missing splits produce a warning but do not fail validation."""
        manifest = _valid_manifest()
        manifest["chunks"] = [
            manifest["chunks"][0],
            manifest["chunks"][1],
        ]
        manifest["total_shots"] = 4

        assert validate_manifest(manifest) is True

    def test_missing_chunk_id_key(self) -> None:
        """A chunk missing the ``id`` key must fail validation."""
        manifest = _valid_manifest()
        del manifest["chunks"][0]["id"]

        assert validate_manifest(manifest) is False

    def test_missing_n_shots_key(self) -> None:
        """A chunk missing ``n_shots`` must fail validation."""
        manifest = _valid_manifest()
        del manifest["chunks"][0]["n_shots"]

        with pytest.raises(KeyError):
            assert validate_manifest(manifest) is False


# ============================================================
# 6. get_chunk_paths
# ============================================================


class TestChunkPaths:
    """Tests for :func:`get_chunk_paths`."""

    def test_get_chunk_paths(self, tmp_path: Path) -> None:
        """Chunk paths must be grouped by split."""
        manifest = {
            "chunks": [
                {
                    "id": 1,
                    "filename": "chunk_001_train.pt",
                    "split": "train",
                },
                {
                    "id": 2,
                    "filename": "chunk_002_train.pt",
                    "split": "train",
                },
                {
                    "id": 3,
                    "filename": "chunk_003_val.pt",
                    "split": "val",
                },
            ]
        }

        result = get_chunk_paths(manifest, tmp_path)

        assert result["train"] == [
            tmp_path / "chunk_001_train.pt",
            tmp_path / "chunk_002_train.pt",
        ]
        assert result["val"] == [
            tmp_path / "chunk_003_val.pt",
        ]

    def test_get_chunk_paths_empty(self, tmp_path: Path) -> None:
        """An empty ``chunks`` list must return an empty dictionary."""
        result = get_chunk_paths({"chunks": []}, tmp_path)

        assert result == {}

    def test_get_chunk_paths_all_splits(self, tmp_path: Path) -> None:
        """All three split names must be represented."""
        manifest = {
            "chunks": [
                {"id": 1, "filename": "c1.pt", "split": "train"},
                {"id": 2, "filename": "c2.pt", "split": "val"},
                {"id": 3, "filename": "c3.pt", "split": "test"},
            ]
        }

        result = get_chunk_paths(manifest, tmp_path)

        assert set(result.keys()) == {"train", "val", "test"}


# ============================================================
# 7. get_manifest_stats
# ============================================================


class TestManifestStats:
    """Tests for :func:`get_manifest_stats`."""

    def test_get_manifest_stats(self) -> None:
        """Statistics must aggregate totals and per-split counts."""
        manifest = {
            "total_shots": 10,
            "chunks": [
                {"split": "train", "n_shots": 6, "file_size_mb": 100.0},
                {"split": "val", "n_shots": 2, "file_size_mb": 20.0},
                {"split": "test", "n_shots": 2, "file_size_mb": 30.0},
            ],
        }

        stats = get_manifest_stats(manifest)

        assert stats["total_chunks"] == 3
        assert stats["total_shots"] == 10
        assert stats["total_size_mb"] == 150.0
        assert stats["total_size_gb"] == 0.15

        assert stats["split_stats"]["train"]["chunks"] == 1
        assert stats["split_stats"]["train"]["shots"] == 6
        assert stats["split_stats"]["train"]["size_mb"] == 100.0

    def test_get_manifest_stats_empty(self) -> None:
        """An empty manifest must return zero totals."""
        manifest = {"total_shots": 0, "chunks": []}

        stats = get_manifest_stats(manifest)

        assert stats["total_chunks"] == 0
        assert stats["total_shots"] == 0
        assert stats["total_size_mb"] == 0.0
        assert stats["split_stats"] == {}

    def test_get_manifest_stats_missing_size(self) -> None:
        """Chunks without ``file_size_mb`` must contribute zero size."""
        manifest = {
            "total_shots": 3,
            "chunks": [
                {"split": "train", "n_shots": 3},  # no file_size_mb
            ],
        }

        stats = get_manifest_stats(manifest)

        assert stats["total_size_mb"] == 0.0
        assert stats["split_stats"]["train"]["size_mb"] == 0.0


# ============================================================
# 8. CHECKSUM CORRUPTION DETECTION
# ============================================================


class TestManifestChecksumCorruption:
    """Verify that manifest checksums detect corruption."""

    def test_manifest_checksum_detects_tampering(self, tmp_path: Path) -> None:
        """Modifying the manifest on disk must trigger a warning on load."""
        from loguru import logger as loguru_logger

        path = tmp_path / "manifest.json"

        manifest = {
            "dataset": "TestDataset",
            "version": "1.0.0",
            "created": "2026-01-01T00:00:00+00:00",
            "config": {},
            "total_shots": 0,
            "total_traces": 0,
            "chunks": [],
            "manifest_checksum": None,
        }

        save_manifest(manifest, path)

        # Tamper with the saved file (change a value without updating checksum)
        with open(path, "r") as f:
            data = json.load(f)
        data["dataset"] = "TamperedDataset"

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # Capture Loguru records with a temporary sink
        captured: list[str] = []
        sink_id = loguru_logger.add(
            lambda message: captured.append(message),
            level="WARNING",
        )

        try:
            loaded = load_manifest(path)
        finally:
            loguru_logger.remove(sink_id)

        assert loaded["dataset"] == "TamperedDataset"
        assert any("checksum mismatch" in msg.lower() for msg in captured), (
            "Expected a checksum mismatch warning"
        )

    def test_manifest_checksum_matches_when_untampered(self, tmp_path: Path) -> None:
        """An untampered manifest must not emit a checksum warning."""
        from loguru import logger as loguru_logger

        path = tmp_path / "manifest.json"

        manifest = {
            "dataset": "TestDataset",
            "version": "1.0.0",
            "created": "2026-01-01T00:00:00+00:00",
            "config": {},
            "total_shots": 0,
            "total_traces": 0,
            "chunks": [],
            "manifest_checksum": None,
        }

        save_manifest(manifest, path)

        captured: list[str] = []
        sink_id = loguru_logger.add(
            lambda message: captured.append(message),
            level="WARNING",
        )

        try:
            load_manifest(path)
        finally:
            loguru_logger.remove(sink_id)

        assert any("checksum mismatch" in msg.lower() for msg in captured), (
            "Did not expect a checksum mismatch warning"
        )

    def test_chunk_checksum_detects_corruption(self, tmp_path: Path) -> None:
        """A chunk whose contents change must produce a different checksum."""
        chunk = tmp_path / "chunk_001_train.pt"

        chunk.write_bytes(b"original chunk data")
        original = compute_checksum(chunk)

        chunk.write_bytes(b"corrupted chunk data")
        corrupted = compute_checksum(chunk)

        assert original != corrupted

    def test_generate_manifest_records_chunk_checksum(self, tmp_path: Path) -> None:
        """``generate_manifest`` must record the correct chunk checksum."""
        chunk_path = tmp_path / "chunk_001_train.pt"
        chunk_path.write_bytes(b"chunk payload")

        chunks = {"train": [_make_chunk_entry(1, "train", [1])]}

        manifest = generate_manifest(
            dataset_name="TestDataset",
            chunks=chunks,
            config={},
            chunk_dir=tmp_path,
            total_shots=1,
            total_traces=10,
            increment_version=False,
        )

        assert manifest["chunks"][0]["checksum"] == compute_checksum(chunk_path)

    def test_generate_manifest_checksum_changes_with_content(
        self, tmp_path: Path
    ) -> None:
        """Regenerating after modifying a chunk must yield a new checksum."""
        chunk_path = tmp_path / "chunk_001_train.pt"
        chunk_path.write_bytes(b"first version")

        chunks = {"train": [_make_chunk_entry(1, "train", [1])]}

        manifest_a = generate_manifest(
            dataset_name="TestDataset",
            chunks=chunks,
            config={},
            chunk_dir=tmp_path,
            total_shots=1,
            total_traces=10,
            increment_version=False,
        )

        chunk_path.write_bytes(b"second version")

        manifest_b = generate_manifest(
            dataset_name="TestDataset",
            chunks=chunks,
            config={},
            chunk_dir=tmp_path,
            total_shots=1,
            total_traces=10,
            increment_version=False,
        )

        assert (
            manifest_a["chunks"][0]["checksum"] != manifest_b["chunks"][0]["checksum"]
        )
