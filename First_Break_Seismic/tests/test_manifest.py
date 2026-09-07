import json

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


class TestComputeChecksum:
    def test_checksum_is_16_characters(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello world")

        checksum = compute_checksum(filepath)

        assert isinstance(checksum, str)
        assert len(checksum) == 16

    def test_same_content_has_same_checksum(self, tmp_path):
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("same content")
        file2.write_text("same content")

        assert compute_checksum(file1) == compute_checksum(file2)

    def test_different_content_has_different_checksum(self, tmp_path):
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("content A")
        file2.write_text("content B")

        assert compute_checksum(file1) != compute_checksum(file2)


class TestGetNextVersion:
    def test_missing_manifest_returns_initial_version(self, tmp_path):
        path = tmp_path / "manifest.json"

        assert get_next_version(path) == "1.0.0"

    def test_existing_manifest_increments_patch(self, tmp_path):
        path = tmp_path / "manifest.json"

        path.write_text(json.dumps({"version": "1.0.3"}))

        assert get_next_version(path) == "1.0.4"

    def test_invalid_json_returns_initial_version(self, tmp_path):
        path = tmp_path / "manifest.json"

        path.write_text("{invalid json")

        assert get_next_version(path) == "1.0.0"

    def test_invalid_version_returns_initial_version(self, tmp_path):
        path = tmp_path / "manifest.json"

        path.write_text(json.dumps({"version": "invalid"}))

        assert get_next_version(path) == "1.0.0"


class TestGenerateManifest:
    def test_generate_manifest_without_chunk_files(self, tmp_path):
        chunks = {
            "train": [
                {
                    "id": 1,
                    "shot_ids": [1, 2, 3],
                    "n_shots": 3,
                    "start_idx": 0,
                    "end_idx": 2,
                }
            ],
            "val": [
                {
                    "id": 2,
                    "shot_ids": [4],
                    "n_shots": 1,
                    "start_idx": 0,
                    "end_idx": 0,
                }
            ],
            "test": [
                {
                    "id": 3,
                    "shot_ids": [5],
                    "n_shots": 1,
                    "start_idx": 0,
                    "end_idx": 0,
                }
            ],
        }

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
        self,
        tmp_path,
    ):
        chunk_path = tmp_path / "chunk_001_train.pt"
        chunk_path.write_bytes(b"test chunk content")

        chunks = {
            "train": [
                {
                    "id": 1,
                    "shot_ids": [1, 2],
                    "n_shots": 2,
                    "start_idx": 0,
                    "end_idx": 1,
                }
            ]
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


class TestSaveLoadManifest:
    def test_save_manifest_creates_file(self, tmp_path):
        path = tmp_path / "nested" / "manifest.json"

        manifest = {
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

    def test_load_manifest_returns_saved_data(self, tmp_path):
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

    def test_load_missing_manifest_raises(self, tmp_path):
        path = tmp_path / "missing.json"

        try:
            load_manifest(path)
            assert False
        except FileNotFoundError:
            pass


class TestValidateManifest:
    def _valid_manifest(self):
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

    def test_valid_manifest(self):
        manifest = self._valid_manifest()

        assert validate_manifest(manifest) is True

    def test_missing_required_key(self):
        manifest = self._valid_manifest()
        del manifest["dataset"]

        assert validate_manifest(manifest) is False

    def test_empty_chunks(self):
        manifest = self._valid_manifest()
        manifest["chunks"] = []

        assert validate_manifest(manifest) is False

    def test_invalid_split(self):
        manifest = self._valid_manifest()
        manifest["chunks"][0]["split"] = "invalid"

        assert validate_manifest(manifest) is False

    def test_total_shots_mismatch(self):
        manifest = self._valid_manifest()
        manifest["total_shots"] = 999

        assert validate_manifest(manifest) is False

    def test_missing_chunk_key(self):
        manifest = self._valid_manifest()
        del manifest["chunks"][0]["filename"]

        assert validate_manifest(manifest) is False

    def test_missing_split_is_not_fatal(self):
        manifest = self._valid_manifest()

        manifest["chunks"] = [
            manifest["chunks"][0],
            manifest["chunks"][1],
        ]
        manifest["total_shots"] = 4

        assert validate_manifest(manifest) is True


class TestChunkPaths:
    def test_get_chunk_paths(self, tmp_path):
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


class TestManifestStats:
    def test_get_manifest_stats(self):
        manifest = {
            "total_shots": 10,
            "chunks": [
                {
                    "split": "train",
                    "n_shots": 6,
                    "file_size_mb": 100.0,
                },
                {
                    "split": "val",
                    "n_shots": 2,
                    "file_size_mb": 20.0,
                },
                {
                    "split": "test",
                    "n_shots": 2,
                    "file_size_mb": 30.0,
                },
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
