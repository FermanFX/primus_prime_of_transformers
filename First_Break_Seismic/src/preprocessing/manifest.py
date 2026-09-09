"""
Utilities for generating, saving, loading, validating, and inspecting
manifests for chunked datasets.

A manifest stores dataset metadata, preprocessing configuration, semantic
version information, chunk-level metadata, file sizes, and checksums.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


def compute_checksum(filepath: Path) -> str:
    """
    Calculate a truncated SHA-256 checksum for a file.

    The file is processed incrementally in fixed-size blocks so that large
    files do not need to be loaded entirely into memory.

    Args:
        filepath: Path to the file whose checksum should be calculated.

    Returns:
        The first 16 hexadecimal characters of the SHA-256 digest.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
        OSError: If the file cannot be opened or read.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()[:16]


def get_next_version(manifest_path: Path) -> str:
    """
    Determine the next patch version based on an existing manifest.

    If the manifest exists and contains a valid semantic version in
    ``MAJOR.MINOR.PATCH`` format, only the patch component is incremented.

    Args:
        manifest_path: Path to the existing manifest JSON file.

    Returns:
        The next version string. If the manifest does not exist or its
        version cannot be parsed, ``"1.0.0"`` is returned.

    Examples:
        If the existing version is ``"1.2.3"``, this function returns
        ``"1.2.4"``.
    """
    if not manifest_path.exists():
        return "1.0.0"

    try:
        with open(manifest_path, "r") as f:
            existing = json.load(f)

        version = existing.get("version", "1.0.0")
        version_parts = version.split(".")

        if len(version_parts) != 3:
            raise ValueError("Invalid version format")

        major, minor, patch = version_parts

        if not all(part.isdigit() for part in (major, minor, patch)):
            raise ValueError("Invalid version format")

        return f"{major}.{minor}.{int(patch) + 1}"

    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        logger.warning(
            f"Could not parse version from {manifest_path}, starting at 1.0.0"
        )
        return "1.0.0"


def generate_manifest(
    dataset_name: str,
    chunks: dict[str, list[dict]],
    config: dict[str, Any],
    chunk_dir: Path,
    total_shots: int,
    total_traces: int,
    increment_version: bool = True,
) -> dict[str, Any]:
    """
    Generate a manifest containing metadata for a chunked dataset.

    For every chunk, the manifest records its identifier, filename, split,
    shot IDs, shot count, index range, file size, and checksum. The manifest
    also stores dataset-level metadata and preprocessing configuration.

    Args:
        dataset_name: Name or identifier of the processed dataset.
        chunks: Mapping of split names to lists of chunk metadata dictionaries.
        config: Configuration used during dataset preprocessing.
        chunk_dir: Directory containing the generated chunk files.
        total_shots: Total number of shots across all dataset chunks.
        total_traces: Total number of traces in the dataset.
        increment_version: If ``True``, derive the version by incrementing
            the existing manifest's patch version. If ``False``, use
            ``"1.0.0"``.

    Returns:
        A dictionary containing the complete manifest metadata.

    Notes:
        Missing chunk files are still included in the manifest. Their
        ``file_size_mb`` is set to ``0`` and their ``checksum`` is set
        to ``None``.
    """
    manifest_path = chunk_dir / "manifest.json"

    manifest: dict[str, Any] = {
        "dataset": dataset_name,
        "version": get_next_version(manifest_path) if increment_version else "1.0.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "total_shots": total_shots,
        "total_traces": total_traces,
        "chunks": [],
    }

    for split_name, chunk_list in chunks.items():
        for chunk in chunk_list:
            chunk_filename = f"chunk_{chunk['id']:03d}_{split_name}.pt"
            chunk_path = chunk_dir / chunk_filename

            file_size_mb = (
                chunk_path.stat().st_size / (1024 * 1024) if chunk_path.exists() else 0
            )

            manifest["chunks"].append(
                {
                    "id": chunk["id"],
                    "filename": chunk_filename,
                    "split": split_name,
                    "shot_ids": chunk["shot_ids"],
                    "n_shots": chunk["n_shots"],
                    "start_idx": chunk.get("start_idx", 0),
                    "end_idx": chunk.get("end_idx", 0),
                    "file_size_mb": round(file_size_mb, 2),
                    "checksum": (
                        compute_checksum(chunk_path) if chunk_path.exists() else None
                    ),
                }
            )

    manifest["manifest_checksum"] = None

    return manifest


def save_manifest(
    manifest: dict[str, Any],
    path: Path,
) -> None:
    """
    Serialize a manifest to JSON and attach its checksum.

    The manifest is first written without a checksum. The checksum of that
    exact representation is then calculated and stored in the manifest
    before the final JSON file is written.

    Args:
        manifest: Manifest dictionary to serialize.
        path: Destination path for the manifest JSON file.

    Raises:
        OSError: If the destination directory or file cannot be created
            or written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    checksum = compute_checksum(path)
    manifest["manifest_checksum"] = checksum

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest saved to {path} (checksum: {checksum})")


def load_manifest(
    path: Path,
) -> dict[str, Any]:
    """
    Load a manifest from JSON and verify its checksum.

    A checksum mismatch is logged as a warning but does not prevent
    the manifest from being returned.

    Args:
        path: Path to the manifest JSON file.

    Returns:
        Loaded manifest dictionary.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r") as f:
        manifest = json.load(f)

    if manifest.get("manifest_checksum"):
        stored_checksum = manifest["manifest_checksum"]
        manifest_copy = {k: v for k, v in manifest.items() if k != "manifest_checksum"}

        temp_path = path.with_suffix(".tmp")

        with open(temp_path, "w") as f:
            json.dump(manifest_copy, f, indent=2)

        computed_checksum = compute_checksum(temp_path)
        temp_path.unlink()

        if stored_checksum != computed_checksum:
            logger.warning(
                f"Manifest checksum mismatch: "
                f"stored={stored_checksum}, computed={computed_checksum}"
            )

    logger.info(
        f"Manifest loaded from {path} (version: {manifest.get('version', 'unknown')})"
    )

    return manifest


def validate_manifest(
    manifest: dict[str, Any],
) -> bool:
    """
    Validate the structure and consistency of a dataset manifest.

    The validation checks required fields, chunk metadata, supported
    split names, and total shot count consistency.

    Args:
        manifest: Manifest dictionary to validate.

    Returns:
        True if all fatal validation checks pass, otherwise False.

    Note:
        Missing train, validation, or test splits generate a warning
        but are not treated as fatal errors.
    """
    required_keys = [
        "dataset",
        "version",
        "created",
        "config",
        "chunks",
    ]

    for key in required_keys:
        if key not in manifest:
            logger.error(f"Missing required key in manifest: {key}")
            return False

    version = manifest.get("version", "")
    if not version:
        logger.error("Missing version")
        return False

    if not manifest["chunks"]:
        logger.error("No chunks found in manifest")
        return False

    total_shots = 0
    seen_splits = set()

    for chunk in manifest["chunks"]:
        total_shots += chunk["n_shots"]

        chunk_keys = [
            "id",
            "filename",
            "split",
            "shot_ids",
            "n_shots",
        ]

        for key in chunk_keys:
            if key not in chunk:
                logger.error(
                    f"Missing key in chunk {chunk.get('id', 'unknown')}: {key}"
                )
                return False

        if chunk["split"] not in ["train", "val", "test"]:
            logger.error(f"Invalid split in chunk {chunk['id']}: {chunk['split']}")
            return False

        seen_splits.add(chunk["split"])

    if total_shots != manifest["total_shots"]:
        logger.error(
            f"Total shots mismatch: {total_shots} vs {manifest['total_shots']}"
        )
        return False

    expected_splits = {"train", "val", "test"}
    missing_splits = expected_splits - seen_splits

    if missing_splits:
        logger.warning(f"Missing splits: {missing_splits}")

    return True


def get_chunk_paths(
    manifest: dict[str, Any],
    chunk_dir: Path,
) -> dict[str, list[Path]]:
    """
    Build file paths for all chunks grouped by dataset split.

    Args:
        manifest: Manifest containing chunk metadata and filenames.
        chunk_dir: Directory containing the chunk files.

    Returns:
        A dictionary mapping each split name to a list of corresponding
        chunk file paths.

    Example:
        ``{"train": [Path("chunk_001_train.pt")], "val": [...]}``
    """
    paths: dict[str, list[Path]] = {}

    for chunk in manifest["chunks"]:
        split = chunk["split"]
        filename = chunk["filename"]

        if split not in paths:
            paths[split] = []

        paths[split].append(chunk_dir / filename)

    return paths


def get_manifest_stats(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate summary statistics for a dataset manifest.

    Args:
        manifest: Manifest containing chunk metadata.

    Returns:
        Dictionary containing total chunks, shots, storage size,
        and statistics grouped by dataset split.
    """
    total_chunks = len(manifest["chunks"])
    total_shots = manifest["total_shots"]
    total_size_mb = sum(c.get("file_size_mb", 0) for c in manifest["chunks"])

    split_stats = {}

    for chunk in manifest["chunks"]:
        split = chunk["split"]

        if split not in split_stats:
            split_stats[split] = {
                "chunks": 0,
                "shots": 0,
                "size_mb": 0,
            }

        split_stats[split]["chunks"] += 1
        split_stats[split]["shots"] += chunk["n_shots"]
        split_stats[split]["size_mb"] += chunk.get(
            "file_size_mb",
            0,
        )

    return {
        "total_chunks": total_chunks,
        "total_shots": total_shots,
        "total_size_mb": round(total_size_mb, 2),
        "total_size_gb": round(total_size_mb / 1024, 2),
        "split_stats": split_stats,
    }
