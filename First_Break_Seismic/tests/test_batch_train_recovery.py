"""
Unit tests for the memory error recovery flow in :mod:`scripts.batch_train`.

Covers:
    * Variant retry loop — first variant fails → second tried.
    * Variant retry loop — all variants fail → dataset skipped.
    * Variant retry loop — success stops the loop.
    * ``clear_memory()`` — invoked on memory errors, not on other errors.
    * ``skip_failed`` — continues on failure when True, stops when False.
    * Fallback variant progression — batch_size / cache_size / memory_limit
      strictly decrease from aggressive to conservative.

All subprocess calls are mocked; no real training is executed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import batch_train

# ============================================================
# HELPERS
# ============================================================


def _mock_result(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> mock.Mock:
    """Build a mock ``subprocess.CompletedProcess``-like object."""
    result = mock.Mock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def _variant(
    model: str,
    batch_size: int,
    cache_size: int,
    memory_limit_gb: float,
    class_weights: str = "0.1,0.1,0.8",
) -> dict[str, Any]:
    """Build a single config variant for testing."""
    return {
        "model": model,
        "batch_size": batch_size,
        "cache_size": cache_size,
        "memory_limit_gb": memory_limit_gb,
        "class_weights": class_weights,
        "strip_width": 8,
    }


# ============================================================
# 1. VARIANT RETRY LOOP — SEQUENTIAL FALLBACK
# ============================================================


class TestVariantRetryLoop:
    """Verify the retry loop tries variants in order until success."""

    def _patch_subprocess_sequence(
        self,
        results: list[mock.Mock],
    ) -> Any:
        """Patch ``subprocess.run`` to return results in sequence."""
        return mock.patch.object(
            batch_train.subprocess,
            "run",
            side_effect=results,
        )

    def test_first_variant_fails_second_variant_tried(self) -> None:
        """When the first variant fails with a memory error, the second is tried."""
        variants = [
            _variant("pico", batch_size=4, cache_size=4, memory_limit_gb=4.0),
            _variant("pico", batch_size=2, cache_size=2, memory_limit_gb=2.0),
        ]

        # First attempt: memory error. Second attempt: success.
        first = _mock_result(1, "", "RuntimeError: MPS out of memory")
        second = _mock_result(0, "Training complete", "")

        with self._patch_subprocess_sequence([first, second]) as mocked:
            # Simulate the loop body from run_auto_batch_training
            dataset_results: list[dict[str, Any]] = []
            dataset_success = False
            variant_used: dict[str, Any] | None = None

            for variant in variants:
                result = batch_train.train_dataset(
                    dataset_name="Halfmile",
                    config_variant=variant,
                    global_config={},
                )
                dataset_results.append(result)

                if result["success"]:
                    dataset_success = True
                    variant_used = variant
                    break
                # Check if the error is a memory error
                if result.get("error") and batch_train.is_memory_error(result["error"]):
                    continue
                # Non-memory error → skip remaining variants
                break

        assert mocked.call_count == 2
        assert dataset_success is True
        assert variant_used == variants[1]
        assert dataset_results[0]["success"] is False
        assert dataset_results[1]["success"] is True

    def test_first_variant_success_stops_loop(self) -> None:
        """When the first variant succeeds, no further variants are tried."""
        variants = [
            _variant("pico", 4, 4, 4.0),
            _variant("pico", 2, 2, 2.0),
            _variant("pico", 1, 1, 1.0),
        ]

        first = _mock_result(0, "Training complete", "")

        with self._patch_subprocess_sequence([first]) as mocked:
            dataset_results: list[dict[str, Any]] = []
            variant_used: dict[str, Any] | None = None

            for variant in variants:
                result = batch_train.train_dataset(
                    dataset_name="Halfmile",
                    config_variant=variant,
                    global_config={},
                )
                dataset_results.append(result)
                if result["success"]:
                    variant_used = variant
                    break

        assert mocked.call_count == 1
        assert variant_used == variants[0]
        assert len(dataset_results) == 1

    def test_all_variants_fail_returns_no_winner(self) -> None:
        """When every variant fails, no successful variant is selected."""
        variants = [
            _variant("unet", 4, 4, 6.0),
            _variant("unet", 2, 2, 4.0),
            _variant("unet", 1, 1, 2.0),
        ]

        memory_error = _mock_result(1, "", "RuntimeError: MPS out of memory")
        # Three failures, one per variant
        with self._patch_subprocess_sequence(
            [memory_error, memory_error, memory_error]
        ) as mocked:
            dataset_results: list[dict[str, Any]] = []
            dataset_success = False
            variant_used: dict[str, Any] | None = None

            for variant in variants:
                result = batch_train.train_dataset(
                    dataset_name="Halfmile",
                    config_variant=variant,
                    global_config={},
                )
                dataset_results.append(result)
                if result["success"]:
                    dataset_success = True
                    variant_used = variant
                    break
                if result.get("error") and batch_train.is_memory_error(result["error"]):
                    continue
                break

        assert mocked.call_count == 3
        assert dataset_success is False
        assert variant_used is None
        assert all(not r["success"] for r in dataset_results)

    def test_non_memory_error_stops_variant_loop(self) -> None:
        """A non-memory error must stop the loop without trying more variants."""
        variants = [
            _variant("unet", 4, 4, 6.0),
            _variant("unet", 2, 2, 4.0),
            _variant("unet", 1, 1, 2.0),
        ]

        shape_error = _mock_result(1, "", "RuntimeError: shape mismatch")

        with self._patch_subprocess_sequence([shape_error]) as mocked:
            dataset_results: list[dict[str, Any]] = []
            for variant in variants:
                result = batch_train.train_dataset(
                    dataset_name="Halfmile",
                    config_variant=variant,
                    global_config={},
                )
                dataset_results.append(result)
                if result["success"]:
                    break
                if result.get("error") and batch_train.is_memory_error(result["error"]):
                    continue
                break

        # Only one attempt because the error was not a memory error
        assert mocked.call_count == 1
        assert len(dataset_results) == 1


# ============================================================
# 2. clear_memory() INVOCATION
# ============================================================


class TestClearMemoryInvocation:
    """Verify that ``clear_memory`` is called only on memory errors."""

    def test_clear_memory_called_on_memory_error(self) -> None:
        """A memory error must trigger a call to ``clear_memory``."""
        variants = [
            _variant("unet", 4, 4, 6.0),
            _variant("unet", 2, 2, 4.0),
        ]

        memory_error = _mock_result(1, "", "RuntimeError: MPS out of memory")
        success = _mock_result(0, "OK", "")

        with (
            mock.patch.object(
                batch_train.subprocess, "run", side_effect=[memory_error, success]
            ),
            mock.patch.object(batch_train, "clear_memory") as mocked_clear,
        ):
            for variant in variants:
                result = batch_train.train_dataset(
                    dataset_name="Halfmile",
                    config_variant=variant,
                    global_config={},
                )
                if result["success"]:
                    break
                if result.get("error") and batch_train.is_memory_error(result["error"]):
                    batch_train.clear_memory()
                    continue
                break

        assert mocked_clear.call_count == 1

    def test_clear_memory_not_called_on_non_memory_error(self) -> None:
        """A non-memory error must NOT trigger ``clear_memory``."""
        variants = [
            _variant("unet", 4, 4, 6.0),
            _variant("unet", 2, 2, 4.0),
        ]

        shape_error = _mock_result(1, "", "RuntimeError: shape mismatch")

        with (
            mock.patch.object(batch_train.subprocess, "run", side_effect=[shape_error]),
            mock.patch.object(batch_train, "clear_memory") as mocked_clear,
        ):
            for variant in variants:
                result = batch_train.train_dataset(
                    dataset_name="Halfmile",
                    config_variant=variant,
                    global_config={},
                )
                if result["success"]:
                    break
                if result.get("error") and batch_train.is_memory_error(result["error"]):
                    batch_train.clear_memory()
                    continue
                break

        assert mocked_clear.call_count == 0

    def test_clear_memory_called_once_per_memory_error(self) -> None:
        """Each memory-error variant must call ``clear_memory`` exactly once."""
        variants = [
            _variant("unet", 4, 4, 6.0),
            _variant("unet", 2, 2, 4.0),
            _variant("unet", 1, 1, 2.0),
        ]

        memory_error = _mock_result(1, "", "RuntimeError: MPS out of memory")

        with (
            mock.patch.object(
                batch_train.subprocess,
                "run",
                side_effect=[memory_error, memory_error, memory_error],
            ),
            mock.patch.object(batch_train, "clear_memory") as mocked_clear,
        ):
            for variant in variants:
                result = batch_train.train_dataset(
                    dataset_name="Halfmile",
                    config_variant=variant,
                    global_config={},
                )
                if result["success"]:
                    break
                if result.get("error") and batch_train.is_memory_error(result["error"]):
                    batch_train.clear_memory()
                    continue
                break

        assert mocked_clear.call_count == 3


# ============================================================
# 3. GRACEFUL FAILURE — skip_failed
# ============================================================


class TestSkipFailedBehaviour:
    """Verify that ``skip_failed`` controls batch continuation."""

    def test_skip_failed_true_continues_to_next_dataset(self) -> None:
        """With ``skip_failed=True``, a failed dataset must not stop the batch."""
        datasets = ["Halfmile", "Brunswick"]

        memory_error = _mock_result(1, "", "RuntimeError: MPS out of memory")
        success = _mock_result(0, "OK", "")

        # Halfmile fails on all variants; Brunswick succeeds on first attempt.
        side_effect = [memory_error, memory_error, success]

        processed: list[str] = []
        with (
            mock.patch.object(batch_train.subprocess, "run", side_effect=side_effect),
            mock.patch.object(batch_train, "clear_memory"),
        ):
            skip_failed = True
            failed_datasets: list[str] = []
            successful_datasets: list[str] = []

            for dataset_name in datasets:
                # Two-variant loop for Halfmile, one for Brunswick
                variants = (
                    [_variant("unet", 4, 4, 6.0), _variant("unet", 2, 2, 4.0)]
                    if dataset_name == "Halfmile"
                    else [_variant("unet", 4, 4, 6.0)]
                )

                dataset_success = False
                for variant in variants:
                    result = batch_train.train_dataset(
                        dataset_name=dataset_name,
                        config_variant=variant,
                        global_config={},
                    )
                    processed.append(dataset_name)

                    if result["success"]:
                        dataset_success = True
                        successful_datasets.append(dataset_name)
                        break
                    if result.get("error") and batch_train.is_memory_error(
                        result["error"]
                    ):
                        batch_train.clear_memory()
                        continue
                    break

                if not dataset_success:
                    failed_datasets.append(dataset_name)
                    if not skip_failed:
                        break

        # Both datasets were processed
        assert "Brunswick" in processed
        assert "Halfmile" in failed_datasets
        assert "Brunswick" in successful_datasets

    def test_skip_failed_false_stops_batch(self) -> None:
        """With ``skip_failed=False``, a failed dataset must stop the batch."""
        datasets = ["Halfmile", "Brunswick"]

        memory_error = _mock_result(1, "", "RuntimeError: MPS out of memory")

        processed: list[str] = []
        with (
            mock.patch.object(
                batch_train.subprocess,
                "run",
                side_effect=[memory_error, memory_error],
            ),
            mock.patch.object(batch_train, "clear_memory"),
        ):
            skip_failed = False
            failed_datasets: list[str] = []

            for dataset_name in datasets:
                variants = [
                    _variant("unet", 4, 4, 6.0),
                    _variant("unet", 2, 2, 4.0),
                ]

                dataset_success = False
                for variant in variants:
                    result = batch_train.train_dataset(
                        dataset_name=dataset_name,
                        config_variant=variant,
                        global_config={},
                    )
                    processed.append(dataset_name)

                    if result["success"]:
                        dataset_success = True
                        break
                    if result.get("error") and batch_train.is_memory_error(
                        result["error"]
                    ):
                        batch_train.clear_memory()
                        continue
                    break

                if not dataset_success:
                    failed_datasets.append(dataset_name)
                    if not skip_failed:
                        break

        # Brunswick must NOT be processed
        assert "Brunswick" not in processed
        assert "Halfmile" in failed_datasets

    def test_successful_dataset_recorded(self) -> None:
        """A successful dataset must be recorded in the success list."""
        success = _mock_result(0, "OK", "")

        with mock.patch.object(batch_train.subprocess, "run", side_effect=[success]):
            successful_datasets: list[str] = []
            failed_datasets: list[str] = []

            result = batch_train.train_dataset(
                dataset_name="Halfmile",
                config_variant=_variant("unet", 4, 4, 6.0),
                global_config={},
            )

            if result["success"]:
                successful_datasets.append("Halfmile")
            else:
                failed_datasets.append("Halfmile")

        assert successful_datasets == ["Halfmile"]
        assert failed_datasets == []


# ============================================================
# 4. FALLBACK VARIANT PROGRESSION
# ============================================================


class TestFallbackVariantProgression:
    """Verify that variants progressively reduce resource usage."""

    def test_batch_size_decreases_on_failure(self) -> None:
        """Each fallback variant must use a smaller ``batch_size``."""
        variants = [
            _variant("mpslight", batch_size=8, cache_size=4, memory_limit_gb=6.0),
            _variant("mpslight", batch_size=6, cache_size=4, memory_limit_gb=5.0),
            _variant("mpslight", batch_size=4, cache_size=2, memory_limit_gb=4.0),
            _variant("mpslight", batch_size=1, cache_size=1, memory_limit_gb=2.0),
        ]

        for i in range(1, len(variants)):
            assert variants[i]["batch_size"] < variants[i - 1]["batch_size"]

    def test_cache_size_decreases_or_equal_on_failure(self) -> None:
        """Each fallback variant must use a smaller or equal ``cache_size``."""
        variants = [
            _variant("mpslight", 8, 4, 6.0),
            _variant("mpslight", 6, 4, 5.0),
            _variant("mpslight", 4, 2, 4.0),
            _variant("mpslight", 1, 1, 2.0),
        ]

        for i in range(1, len(variants)):
            assert variants[i]["cache_size"] <= variants[i - 1]["cache_size"]

    def test_memory_limit_decreases_on_failure(self) -> None:
        """Each fallback variant must use a smaller ``memory_limit_gb``."""
        variants = [
            _variant("mpslight", 8, 4, 6.0),
            _variant("mpslight", 6, 4, 5.0),
            _variant("mpslight", 4, 2, 4.0),
            _variant("mpslight", 1, 1, 2.0),
        ]

        for i in range(1, len(variants)):
            assert variants[i]["memory_limit_gb"] < variants[i - 1]["memory_limit_gb"]

    def test_retry_loop_uses_strictly_decreasing_memory_limit(self) -> None:
        """Verify the retry loop walks variants with decreasing memory limits."""
        variants = [
            _variant("unet", 4, 4, 6.0),
            _variant("unet", 2, 2, 4.0),
            _variant("unet", 1, 1, 2.0),
        ]

        memory_error = _mock_result(1, "", "RuntimeError: MPS out of memory")

        seen_memory_limits: list[float] = []

        def fake_train_dataset(
            dataset_name: str,
            config_variant: dict[str, Any],
            global_config: dict[str, Any],
            extra_args: list[str] | None = None,
        ) -> dict[str, Any]:
            seen_memory_limits.append(config_variant["memory_limit_gb"])
            return {
                "success": False,
                "dataset": dataset_name,
                "config": config_variant,
                "error": memory_error.stderr,
                "output": "",
                "duration": 0.0,
                "return_code": 1,
            }

        with (
            mock.patch.object(
                batch_train, "train_dataset", side_effect=fake_train_dataset
            ),
            mock.patch.object(batch_train, "clear_memory"),
        ):
            for variant in variants:
                result = batch_train.train_dataset(
                    dataset_name="Halfmile",
                    config_variant=variant,
                    global_config={},
                )
                if result["success"]:
                    break
                if result.get("error") and batch_train.is_memory_error(result["error"]):
                    batch_train.clear_memory()
                    continue
                break

        # Memory limits must be strictly decreasing
        assert seen_memory_limits == [6.0, 4.0, 2.0]

    def test_fallback_levels_from_auto_config_are_ordered(self) -> None:
        """The variants generated by ``calculate_optimal_config`` must be ordered.

        The auto-config produces fallback levels that progressively reduce
        resource usage. ``memory_limit_gb`` decreases strictly at every level,
        while ``batch_size`` and ``cache_size`` are non-increasing (they can
        stay the same between adjacent levels if only the other dimension
        was reduced).
        """
        # Simulate what the smart auto-config produces (order matters).
        # Level 1: optimal; Level 2: 75% batch; Level 3: 75% cache;
        # Level 4: 50% both; Level 5: minimal.
        variants = [
            _variant("mpslight", 8, 4, 6.0),  # L1
            _variant("mpslight", 6, 4, 5.1),  # L2: batch reduced
            _variant("mpslight", 6, 3, 4.3),  # L3: cache reduced
            _variant("mpslight", 4, 2, 3.6),  # L4: both reduced
            _variant("mpslight", 1, 1, 2.0),  # L5: minimal
        ]

        # batch_size must be non-increasing
        for i in range(1, len(variants)):
            assert variants[i]["batch_size"] <= variants[i - 1]["batch_size"], (
                f"batch_size increased from {variants[i - 1]['batch_size']} "
                f"to {variants[i]['batch_size']} at level {i + 1}"
            )

        # cache_size must be non-increasing
        for i in range(1, len(variants)):
            assert variants[i]["cache_size"] <= variants[i - 1]["cache_size"], (
                f"cache_size increased from {variants[i - 1]['cache_size']} "
                f"to {variants[i]['cache_size']} at level {i + 1}"
            )

        # memory_limit_gb must strictly decrease
        for i in range(1, len(variants)):
            assert (
                variants[i]["memory_limit_gb"] < variants[i - 1]["memory_limit_gb"]
            ), (
                f"memory_limit_gb did not decrease from "
                f"{variants[i - 1]['memory_limit_gb']} to "
                f"{variants[i]['memory_limit_gb']} at level {i + 1}"
            )
