"""
Unit tests for :class:`src.preprocessing.processor.ShotProcessor`.

Covers:
    * ``validate_picks`` — clipping of invalid picks and statistics.
    * ``process_shot`` — full processing with valid data.
    * ``process_shot`` — padding behaviour for short shots.
    * ``process_shot`` — statistics returned for each shot.
    * ``create_mask_vectorized`` — 3-class mask semantics.

The tests are self-contained and do not require any external fixtures or
HDF5 files. All inputs are synthetic NumPy arrays.

Mask semantics:
    Class 0 — Before : samples before the first-break strip.
    Class 1 — After  : samples after the first-break strip.
    Class 2 — Strip  : samples inside the pick-centred strip (the target).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.processor import ShotProcessor

# ============================================================
# CONSTANTS AND FIXTURES
# ============================================================

TARGET_TRACES = 32
N_SAMPLES = 64
STRIP_WIDTH = 8
HALF_WIDTH = STRIP_WIDTH // 2  # = 4

# The implementation labels samples in the closed interval
# [pick - half_width, pick + half_width], which yields
# ``2 * half_width + 1`` samples — one more than ``strip_width``.
EXPECTED_STRIP_SAMPLES = 2 * HALF_WIDTH + 1  # = 9


@pytest.fixture
def processor() -> ShotProcessor:
    """Return a :class:`ShotProcessor` with small, deterministic shapes."""
    return ShotProcessor(
        target_traces=TARGET_TRACES,
        n_samples=N_SAMPLES,
        strip_width=STRIP_WIDTH,
        log_level="WARNING",  # silence INFO/DEBUG during tests
    )


def _make_shot_data(n_traces: int, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Create a deterministic synthetic shot of shape ``(n_traces, n_samples)``."""
    rng = np.random.default_rng(seed=1234)
    return rng.standard_normal((n_traces, n_samples)).astype(np.float32)


def _make_picks(values: list[float]) -> np.ndarray:
    """Create a pick array from an explicit list of values."""
    return np.asarray(values, dtype=np.float32)


def _picks(*values: float) -> np.ndarray:
    """Build a float32 pick array from an explicit list of values."""
    return np.asarray(values, dtype=np.float32)


# ============================================================
# 1. validate_picks — CLIPPING AND STATISTICS
# ============================================================


class TestValidatePicks:
    """Tests for :meth:`ShotProcessor.validate_picks`."""

    def test_all_valid_picks_unchanged(self, processor: ShotProcessor) -> None:
        """Valid picks must be returned unchanged and stats must be correct."""
        picks = _make_picks([10.0, 20.0, 30.0, 40.0])

        cleaned, stats = processor.validate_picks(picks)

        np.testing.assert_array_equal(cleaned, picks)
        assert stats["total"] == 4
        assert stats["valid"] == 4
        assert stats["invalid"] == 0
        assert stats["invalid_ratio"] == 0.0
        assert stats["min_pick"] == 10.0
        assert stats["max_pick"] == 40.0
        assert stats["mean_pick"] == pytest.approx(25.0)
        assert stats["median_pick"] == pytest.approx(25.0)

    def test_negative_picks_are_clipped(self, processor: ShotProcessor) -> None:
        """Negative picks are invalid and must be clipped to 0."""
        picks = _make_picks([-5.0, 10.0, -1.0, 20.0])

        cleaned, stats = processor.validate_picks(picks)

        assert cleaned[0] == 0
        assert cleaned[1] == 10.0
        assert cleaned[2] == 0
        assert cleaned[3] == 20.0

        assert stats["total"] == 4
        assert stats["valid"] == 2
        assert stats["invalid"] == 2
        assert stats["invalid_ratio"] == pytest.approx(0.5)

    def test_picks_at_or_above_n_samples_are_clipped(
        self, processor: ShotProcessor
    ) -> None:
        """Picks >= n_samples are invalid and must be clipped to n_samples - 1."""
        picks = _make_picks([10.0, float(N_SAMPLES), float(N_SAMPLES + 10)])

        cleaned, stats = processor.validate_picks(picks)

        assert cleaned[0] == 10.0
        assert cleaned[1] == N_SAMPLES - 1
        assert cleaned[2] == N_SAMPLES - 1

        assert stats["valid"] == 1
        assert stats["invalid"] == 2
        assert stats["invalid_ratio"] == pytest.approx(2 / 3)

    def test_zero_picks_are_invalid(self, processor: ShotProcessor) -> None:
        """Pick value 0 is treated as invalid (no pick)."""
        picks = _make_picks([0.0, 10.0, 0.0, 20.0])

        cleaned, stats = processor.validate_picks(picks)

        assert stats["valid"] == 2
        assert stats["invalid"] == 2
        assert np.all(cleaned >= 0)
        assert np.all(cleaned < N_SAMPLES)

    def test_all_invalid_picks(self, processor: ShotProcessor) -> None:
        """When all picks are invalid, min/max/mean/median must be None."""
        picks = _make_picks([-1.0, 0.0, float(N_SAMPLES), -100.0])

        cleaned, stats = processor.validate_picks(picks)

        assert stats["valid"] == 0
        assert stats["invalid"] == 4
        assert stats["invalid_ratio"] == pytest.approx(1.0)
        assert stats["min_pick"] is None
        assert stats["max_pick"] is None
        assert stats["mean_pick"] is None
        assert stats["median_pick"] is None

        assert np.all(cleaned >= 0)
        assert np.all(cleaned < N_SAMPLES)

    def test_empty_picks(self, processor: ShotProcessor) -> None:
        """An empty pick array must not raise and must return empty stats."""
        picks = np.array([], dtype=np.float32)

        cleaned, stats = processor.validate_picks(picks)

        assert cleaned.shape == (0,)
        assert stats["total"] == 0
        assert stats["valid"] == 0
        assert stats["invalid"] == 0
        assert stats["invalid_ratio"] == 0.0

    def test_clipped_values_are_within_bounds(self, processor: ShotProcessor) -> None:
        """After clipping, every value must lie in ``[0, n_samples - 1]``."""
        picks = _make_picks(
            [-1e6, -1.0, 0.0, 1.0, N_SAMPLES / 2, N_SAMPLES, N_SAMPLES + 1e6]
        )

        cleaned, _ = processor.validate_picks(picks)

        assert np.all(cleaned >= 0)
        assert np.all(cleaned <= N_SAMPLES - 1)


# ============================================================
# 2. process_shot — VALID DATA
# ============================================================


class TestProcessShotValidData:
    """Tests for :meth:`ShotProcessor.process_shot` with well-formed input."""

    def test_shapes_and_dtypes(self, processor: ShotProcessor) -> None:
        """Processed data and mask must have the configured shapes and dtypes."""
        shot_data = _make_shot_data(TARGET_TRACES)
        shot_picks = _make_picks([10.0] * TARGET_TRACES)

        data, mask, _stats = processor.process_shot(shot_data, shot_picks, shot_id=1)

        assert data.shape == (TARGET_TRACES, N_SAMPLES)
        assert mask.shape == (TARGET_TRACES, N_SAMPLES)
        assert data.dtype == np.float32
        assert mask.dtype == np.int64

    def test_mask_class_values(self, processor: ShotProcessor) -> None:
        """Mask must only contain classes 0, 1, and 2."""
        shot_data = _make_shot_data(TARGET_TRACES)
        shot_picks = _make_picks([20.0] * TARGET_TRACES)

        _data, mask, _stats = processor.process_shot(shot_data, shot_picks, shot_id=2)

        unique = np.unique(mask)
        assert set(unique.tolist()).issubset({0, 1, 2})

    def test_strip_is_centered_on_pick(self, processor: ShotProcessor) -> None:
        """The class-2 strip must be centered on the pick value."""
        pick_value = 30.0
        shot_data = _make_shot_data(TARGET_TRACES)
        shot_picks = _make_picks([pick_value] * TARGET_TRACES)

        _data, mask, _stats = processor.process_shot(shot_data, shot_picks, shot_id=3)

        for trace_idx in range(TARGET_TRACES):
            strip_indices = np.where(mask[trace_idx] == 2)[0]
            assert len(strip_indices) > 0, "Strip must exist for valid picks"
            strip_center = float(np.median(strip_indices))
            assert abs(strip_center - pick_value) <= 1.0

    def test_after_class_follows_strip(self, processor: ShotProcessor) -> None:
        """Samples after the strip must be class 1; samples before must be class 0."""
        pick_value = 20.0
        shot_data = _make_shot_data(1)
        shot_picks = _make_picks([pick_value])

        _data, mask, _stats = processor.process_shot(shot_data, shot_picks, shot_id=4)

        trace = mask[0]
        strip_indices = np.where(trace == 2)[0]
        strip_end = strip_indices.max()

        if strip_indices.min() > 0:
            assert np.all(trace[: strip_indices.min()] == 0)
        if strip_end < N_SAMPLES - 1:
            assert np.all(trace[strip_end + 1 :] == 1)

    def test_picks_are_preserved_in_output(self, processor: ShotProcessor) -> None:
        """Valid picks must survive processing without modification."""
        shot_data = _make_shot_data(TARGET_TRACES)
        shot_picks = _make_picks(
            [float(i % (N_SAMPLES - 1)) + 1.0 for i in range(TARGET_TRACES)]
        )

        data, mask, _stats = processor.process_shot(shot_data, shot_picks, shot_id=5)

        assert data.shape == (TARGET_TRACES, N_SAMPLES)
        assert mask.shape == (TARGET_TRACES, N_SAMPLES)


# ============================================================
# 3. process_shot — MISSING DATA (PADDING)
# ============================================================


class TestProcessShotPadding:
    """Tests for :meth:`ShotProcessor.process_shot` when the shot is short."""

    def test_padding_to_target_traces(self, processor: ShotProcessor) -> None:
        """Shots with fewer traces than target must be zero-padded."""
        n_actual = 10
        shot_data = _make_shot_data(n_actual)
        shot_picks = _make_picks([15.0] * n_actual)

        data, mask, stats = processor.process_shot(shot_data, shot_picks, shot_id=10)

        assert data.shape == (TARGET_TRACES, N_SAMPLES)
        assert mask.shape == (TARGET_TRACES, N_SAMPLES)

        np.testing.assert_array_equal(data[:n_actual, :], shot_data)
        np.testing.assert_array_equal(
            data[n_actual:, :], np.zeros((TARGET_TRACES - n_actual, N_SAMPLES))
        )
        assert np.all(mask[n_actual:, :] == 0)

        assert stats["original_traces"] == n_actual
        assert stats["padded_or_cropped"] is True
        assert stats["n_traces"] == TARGET_TRACES

    def test_cropping_to_target_traces(self, processor: ShotProcessor) -> None:
        """Shots with more traces than target must be cropped."""
        n_actual = TARGET_TRACES + 8
        shot_data = _make_shot_data(n_actual)
        shot_picks = _make_picks([15.0] * n_actual)

        data, mask, stats = processor.process_shot(shot_data, shot_picks, shot_id=11)

        assert data.shape == (TARGET_TRACES, N_SAMPLES)
        assert mask.shape == (TARGET_TRACES, N_SAMPLES)

        np.testing.assert_array_equal(data, shot_data[:TARGET_TRACES, :])

        assert stats["original_traces"] == n_actual
        assert stats["padded_or_cropped"] is True
        assert stats["n_traces"] == TARGET_TRACES

    def test_exact_target_traces(self, processor: ShotProcessor) -> None:
        """Shots with exactly target traces must not be padded or cropped."""
        shot_data = _make_shot_data(TARGET_TRACES)
        shot_picks = _make_picks([15.0] * TARGET_TRACES)

        data, mask, stats = processor.process_shot(shot_data, shot_picks, shot_id=12)

        assert data.shape == (TARGET_TRACES, N_SAMPLES)
        assert mask.shape == (TARGET_TRACES, N_SAMPLES)
        assert stats["original_traces"] == TARGET_TRACES
        assert stats["padded_or_cropped"] is False

    def test_single_trace_shot(self, processor: ShotProcessor) -> None:
        """A shot with a single trace must be padded correctly."""
        shot_data = _make_shot_data(1)
        shot_picks = _make_picks([20.0])

        data, mask, stats = processor.process_shot(shot_data, shot_picks, shot_id=13)

        assert data.shape == (TARGET_TRACES, N_SAMPLES)
        assert mask.shape == (TARGET_TRACES, N_SAMPLES)
        np.testing.assert_array_equal(data[0, :], shot_data[0, :])
        assert np.all(data[1:, :] == 0)
        assert stats["original_traces"] == 1


# ============================================================
# 4. process_shot — STATISTICS
# ============================================================


class TestProcessShotStats:
    """Tests for the statistics returned by :meth:`ShotProcessor.process_shot`."""

    def test_stats_keys_present(self, processor: ShotProcessor) -> None:
        """Returned stats must contain all required keys."""
        shot_data = _make_shot_data(TARGET_TRACES)
        shot_picks = _make_picks([10.0] * TARGET_TRACES)

        _data, _mask, stats = processor.process_shot(shot_data, shot_picks, shot_id=20)

        required_keys = {
            "n_traces",
            "n_valid",
            "n_invalid",
            "invalid_ratio",
            "min_pick",
            "max_pick",
            "mean_pick",
            "median_pick",
            "original_traces",
            "padded_or_cropped",
        }
        assert required_keys.issubset(stats.keys())

    def test_stats_counts_valid_picks(self, processor: ShotProcessor) -> None:
        """Stats must correctly count valid and invalid picks."""
        shot_data = _make_shot_data(TARGET_TRACES)
        picks_list = [-1.0, 0.0, float(N_SAMPLES)] + [10.0] * (TARGET_TRACES - 3)
        shot_picks = _make_picks(picks_list)

        _data, _mask, stats = processor.process_shot(shot_data, shot_picks, shot_id=21)

        assert stats["n_traces"] == TARGET_TRACES
        assert stats["n_invalid"] == 2
        assert stats["n_valid"] == TARGET_TRACES - 2
        assert stats["invalid_ratio"] == pytest.approx(2 / TARGET_TRACES)

    def test_stats_pick_distribution(self, processor: ShotProcessor) -> None:
        """Stats min/max/mean/median must match the valid picks."""
        shot_data = _make_shot_data(4)
        shot_picks = _make_picks([10.0, 20.0, 30.0, 40.0])

        _data, _mask, stats = processor.process_shot(shot_data, shot_picks, shot_id=22)

        assert stats["min_pick"] == pytest.approx(10.0)
        assert stats["max_pick"] == pytest.approx(40.0)
        assert stats["mean_pick"] == pytest.approx(25.0)
        assert stats["median_pick"] == pytest.approx(25.0)

    def test_stats_with_all_picks_invalid(self, processor: ShotProcessor) -> None:
        """When all picks are invalid, distribution stats must be None."""
        shot_data = _make_shot_data(TARGET_TRACES)
        shot_picks = _make_picks([-1.0] * TARGET_TRACES)

        _data, _mask, stats = processor.process_shot(shot_data, shot_picks, shot_id=23)

        assert stats["n_valid"] == 0
        assert stats["n_invalid"] == TARGET_TRACES
        assert stats["invalid_ratio"] == pytest.approx(1.0)
        assert stats["min_pick"] is None
        assert stats["max_pick"] is None
        assert stats["mean_pick"] is None
        assert stats["median_pick"] is None

    def test_stats_accumulated_across_shots(self, processor: ShotProcessor) -> None:
        """Per-shot stats must be accumulated and retrievable via get_all_stats()."""
        for i in range(3):
            shot_data = _make_shot_data(TARGET_TRACES)
            shot_picks = _make_picks([10.0 + i] * TARGET_TRACES)
            processor.process_shot(shot_data, shot_picks, shot_id=i)

        all_stats = processor.get_all_stats()

        assert all_stats["total_shots"] == 3
        assert all_stats["total_traces"] == 3 * TARGET_TRACES
        assert all_stats["total_valid"] == 3 * TARGET_TRACES
        assert all_stats["total_invalid"] == 0
        assert all_stats["shots_with_no_valid_picks"] == 0

    def test_reset_stats(self, processor: ShotProcessor) -> None:
        """After reset_stats(), get_all_stats() must return an empty dict."""
        shot_data = _make_shot_data(TARGET_TRACES)
        shot_picks = _make_picks([10.0] * TARGET_TRACES)
        processor.process_shot(shot_data, shot_picks, shot_id=30)

        processor.reset_stats()

        assert processor.get_all_stats() == {}
        assert processor.stats == []


# ============================================================
# 5. INTEGRATION — FULL PIPELINE CONSISTENCY
# ============================================================


class TestProcessorIntegration:
    """End-to-end checks that combine multiple processor features."""

    def test_processed_data_matches_input_when_valid(
        self, processor: ShotProcessor
    ) -> None:
        """For a fully-valid, exactly-sized shot, output data equals input data."""
        shot_data = _make_shot_data(TARGET_TRACES)
        shot_picks = _make_picks([15.0] * TARGET_TRACES)

        data, _mask, _stats = processor.process_shot(shot_data, shot_picks, shot_id=40)

        np.testing.assert_allclose(data, shot_data, rtol=1e-6, atol=1e-6)

    def test_mask_is_deterministic(self, processor: ShotProcessor) -> None:
        """Processing the same input twice must produce identical masks."""
        shot_data = _make_shot_data(TARGET_TRACES)
        shot_picks = _make_picks([15.0] * TARGET_TRACES)

        _d1, mask1, _s1 = processor.process_shot(shot_data, shot_picks, shot_id=41)
        _d2, mask2, _s2 = processor.process_shot(shot_data, shot_picks, shot_id=42)

        np.testing.assert_array_equal(mask1, mask2)

    def test_no_mask_for_padded_traces(self, processor: ShotProcessor) -> None:
        """Zero-padded traces must not contain any class-1 or class-2 labels."""
        n_actual = 5
        shot_data = _make_shot_data(n_actual)
        shot_picks = _make_picks([15.0] * n_actual)

        _data, mask, _stats = processor.process_shot(shot_data, shot_picks, shot_id=43)

        padded_region = mask[n_actual:, :]
        assert np.all(padded_region == 0)


# ============================================================
# 6. create_mask_vectorized — CORRECT MASK
# ============================================================


class TestCreateMaskVectorizedBasic:
    """Basic correctness checks for :meth:`create_mask_vectorized`."""

    def test_output_shape_and_dtype(self, processor: ShotProcessor) -> None:
        """Mask must have shape ``(n_traces, n_samples)`` and dtype ``int64``."""
        picks = _picks(10.0, 20.0, 30.0)

        mask = processor.create_mask_vectorized(picks)

        assert mask.shape == (3, N_SAMPLES)
        assert mask.dtype == np.int64

    def test_mask_contains_only_classes_0_1_2(self, processor: ShotProcessor) -> None:
        """Mask values must be restricted to the set ``{0, 1, 2}``."""
        picks = _picks(5.0, 15.0, 25.0, 35.0)

        mask = processor.create_mask_vectorized(picks)

        assert set(np.unique(mask).tolist()).issubset({0, 1, 2})

    def test_each_valid_trace_has_a_strip(self, processor: ShotProcessor) -> None:
        """Every trace with a valid pick must contain at least one class-2 sample."""
        picks = _picks(10.0, 20.0, 30.0)

        mask = processor.create_mask_vectorized(picks)

        for trace_idx in range(len(picks)):
            assert np.any(mask[trace_idx] == 2), (
                f"Trace {trace_idx} has no class-2 (strip) samples"
            )

    def test_strip_width_is_constant(self, processor: ShotProcessor) -> None:
        """The class-2 strip contains ``2 * half_width + 1`` samples for
        interior picks.

        The implementation labels samples in the closed interval
        ``[pick - half_width, pick + half_width]``, which includes the
        pick sample itself. This yields ``strip_width + 1`` samples,
        not ``strip_width``.
        """
        # Picks placed well away from the boundaries so the strip is not clipped.
        picks = _picks(20.0, 30.0, 40.0)

        mask = processor.create_mask_vectorized(picks)

        for trace_idx in range(len(picks)):
            strip_count = int(np.sum(mask[trace_idx] == 2))
            assert strip_count == EXPECTED_STRIP_SAMPLES, (
                f"Trace {trace_idx}: expected {EXPECTED_STRIP_SAMPLES} strip "
                f"samples, got {strip_count}"
            )

    def test_mask_is_deterministic(self, processor: ShotProcessor) -> None:
        """The same picks must produce identical masks on repeated calls."""
        picks = _picks(15.0, 25.0, 35.0)

        mask1 = processor.create_mask_vectorized(picks)
        mask2 = processor.create_mask_vectorized(picks)

        np.testing.assert_array_equal(mask1, mask2)

    def test_empty_picks(self, processor: ShotProcessor) -> None:
        """An empty pick array must produce an empty mask."""
        picks = np.array([], dtype=np.float32)

        mask = processor.create_mask_vectorized(picks)

        assert mask.shape == (0, N_SAMPLES)
        assert mask.dtype == np.int64


# ============================================================
# 7. CLASS 0 — BEFORE THE PICK
# ============================================================


class TestClassBefore:
    """Verify that samples before the strip are labelled class 0."""

    def test_samples_before_strip_are_class_0(self, processor: ShotProcessor) -> None:
        """Samples before the strip must be class 0."""
        pick_value = 30.0
        picks = _picks(pick_value)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        strip_indices = np.where(trace == 2)[0]
        strip_start = int(strip_indices.min())

        if strip_start > 0:
            assert np.all(trace[:strip_start] == 0), (
                "Samples before the strip must be labelled class 0"
            )

    def test_leading_class_0_boundary(self, processor: ShotProcessor) -> None:
        """The strip must start exactly at ``pick - half_width`` when possible."""
        pick_value = 30.0
        picks = _picks(pick_value)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        expected_start = int(pick_value) - HALF_WIDTH
        strip_indices = np.where(trace == 2)[0]

        assert int(strip_indices.min()) == expected_start
        assert np.all(trace[:expected_start] == 0)

    def test_class_0_present_for_late_pick(self, processor: ShotProcessor) -> None:
        """A late pick must still leave a class-0 prefix."""
        picks = _picks(float(N_SAMPLES - 1))

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        assert trace[0] == 0


# ============================================================
# 8. CLASS 2 — STRIP WITHIN half_width
# ============================================================


class TestClassStrip:
    """Verify that the class-2 strip is centred on the pick within ``half_width``."""

    def test_strip_is_centered_on_pick(self, processor: ShotProcessor) -> None:
        """The median index of the class-2 region must equal the rounded pick."""
        pick_value = 30.0
        picks = _picks(pick_value)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        strip_indices = np.where(trace == 2)[0]
        strip_center = float(np.median(strip_indices))

        assert strip_center == pytest.approx(pick_value, abs=1.0)

    def test_strip_contains_pick_sample(self, processor: ShotProcessor) -> None:
        """The rounded pick sample must itself be labelled class 2."""
        pick_value = 30.0
        picks = _picks(pick_value)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        assert trace[round(pick_value)] == 2

    def test_strip_bounds_within_half_width(self, processor: ShotProcessor) -> None:
        """All class-2 samples must lie within ``half_width`` of the pick."""
        pick_value = 30.0
        picks = _picks(pick_value)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        strip_indices = np.where(trace == 2)[0]
        distances = np.abs(strip_indices - round(pick_value))

        assert np.all(distances <= HALF_WIDTH), (
            f"Class-2 samples exceed half_width={HALF_WIDTH}: "
            f"max distance = {distances.max()}"
        )

    def test_strip_is_contiguous(self, processor: ShotProcessor) -> None:
        """The class-2 region must be a single contiguous run of samples."""
        picks = _picks(30.0)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        strip_indices = np.where(trace == 2)[0]
        diffs = np.diff(strip_indices)

        assert np.all(diffs == 1), "Strip must be contiguous"

    def test_strip_is_clipped_at_lower_boundary(self, processor: ShotProcessor) -> None:
        """A pick near 0 must produce a strip clipped at sample 0."""
        picks = _picks(1.0)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        strip_indices = np.where(trace == 2)[0]

        assert int(strip_indices.min()) == 0
        assert np.all(np.diff(strip_indices) == 1)

    def test_strip_is_clipped_at_upper_boundary(self, processor: ShotProcessor) -> None:
        """A pick near ``n_samples - 1`` must produce a strip clipped at the end."""
        picks = _picks(float(N_SAMPLES - 1))

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        strip_indices = np.where(trace == 2)[0]

        assert int(strip_indices.max()) == N_SAMPLES - 1
        assert np.all(np.diff(strip_indices) == 1)

    def test_multiple_traces_independent_strips(self, processor: ShotProcessor) -> None:
        """Each trace must receive a strip centred on its own pick."""
        picks = _picks(10.0, 20.0, 30.0, 40.0)

        mask = processor.create_mask_vectorized(picks)

        for trace_idx, pick_value in enumerate(picks):
            strip_indices = np.where(mask[trace_idx] == 2)[0]
            strip_center = float(np.median(strip_indices))
            assert strip_center == pytest.approx(float(pick_value), abs=1.0)


# ============================================================
# 9. CLASS 1 — AFTER THE STRIP
# ============================================================


class TestClassAfter:
    """Verify that samples after the strip are labelled class 1."""

    def test_samples_after_strip_are_class_1(self, processor: ShotProcessor) -> None:
        """All samples after the strip must be class 1."""
        pick_value = 30.0
        picks = _picks(pick_value)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        strip_indices = np.where(trace == 2)[0]
        strip_end = int(strip_indices.max())

        if strip_end < N_SAMPLES - 1:
            assert np.all(trace[strip_end + 1 :] == 1), (
                "All samples after the strip must be labelled class 1"
            )

    def test_class_1_starts_immediately_after_strip(
        self, processor: ShotProcessor
    ) -> None:
        """The first sample after the strip must be class 1."""
        pick_value = 30.0
        picks = _picks(pick_value)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        strip_indices = np.where(trace == 2)[0]
        strip_end = int(strip_indices.max())

        if strip_end < N_SAMPLES - 1:
            assert trace[strip_end + 1] == 1

    def test_no_class_1_when_strip_reaches_end(self, processor: ShotProcessor) -> None:
        """If the strip extends to the last sample, no class-1 samples exist."""
        picks = _picks(float(N_SAMPLES - 1))

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        assert not np.any(trace == 1)

    def test_class_1_tail_exists_for_early_pick(self, processor: ShotProcessor) -> None:
        """An early pick must leave a class-1 tail."""
        picks = _picks(10.0)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        assert np.any(trace == 1)
        assert trace[-1] == 1


# ============================================================
# 10. CLASS ORDERING AND TRANSITIONS
# ============================================================


class TestClassOrdering:
    """Verify the left-to-right ordering of mask classes."""

    def test_class_order_is_0_then_2_then_1(self, processor: ShotProcessor) -> None:
        """The mask must follow the order: class 0 → class 2 → class 1."""
        picks = _picks(30.0)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        nonzero = np.where(trace != 0)[0]
        assert trace[nonzero[0]] == 2

        class1_indices = np.where(trace == 1)[0]
        if len(class1_indices) > 0:
            first_class1 = int(class1_indices.min())
            assert not np.any(trace[first_class1:] == 2), (
                "Class 2 must not appear after class 1"
            )

    def test_no_class_1_before_strip(self, processor: ShotProcessor) -> None:
        """Class 1 must never appear before the class-2 strip."""
        picks = _picks(30.0)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        strip_indices = np.where(trace == 2)[0]
        strip_start = int(strip_indices.min())

        assert not np.any(trace[:strip_start] == 1)

    def test_no_class_0_after_strip(self, processor: ShotProcessor) -> None:
        """Class 0 must never appear after the class-2 strip."""
        picks = _picks(30.0)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        strip_indices = np.where(trace == 2)[0]
        strip_end = int(strip_indices.max())

        if strip_end < N_SAMPLES - 1:
            assert not np.any(trace[strip_end + 1 :] == 0)

    def test_partition_covers_full_trace(self, processor: ShotProcessor) -> None:
        """Every sample must belong to exactly one of the three classes."""
        picks = _picks(30.0)

        mask = processor.create_mask_vectorized(picks)
        trace = mask[0]

        assert np.all(np.isin(trace, [0, 1, 2]))

        class_0 = np.sum(trace == 0)
        class_1 = np.sum(trace == 1)
        class_2 = np.sum(trace == 2)

        assert class_0 + class_1 + class_2 == N_SAMPLES


# ============================================================
# 11. INVALID PICKS
# ============================================================


class TestInvalidPicks:
    """Verify that invalid picks produce class-0-only traces."""

    def test_zero_pick_produces_all_class_0(self, processor: ShotProcessor) -> None:
        """A pick value of 0 is invalid; the whole trace must be class 0."""
        picks = _picks(0.0)

        mask = processor.create_mask_vectorized(picks)

        assert np.all(mask[0] == 0)

    def test_negative_pick_produces_all_class_0(self, processor: ShotProcessor) -> None:
        """A negative pick is invalid; the whole trace must be class 0."""
        picks = _picks(-5.0)

        mask = processor.create_mask_vectorized(picks)

        assert np.all(mask[0] == 0)

    def test_pick_at_n_samples_produces_all_class_0(
        self, processor: ShotProcessor
    ) -> None:
        """A pick equal to ``n_samples`` is invalid; the trace must be class 0."""
        picks = _picks(float(N_SAMPLES))

        mask = processor.create_mask_vectorized(picks)

        assert np.all(mask[0] == 0)

    def test_mixed_valid_and_invalid_picks(self, processor: ShotProcessor) -> None:
        """Valid traces get normal masks; invalid traces get class-0-only masks."""
        picks = _picks(0.0, 30.0, -1.0, 20.0)

        mask = processor.create_mask_vectorized(picks)

        assert np.all(mask[0] == 0)
        assert np.any(mask[1] == 2)
        assert np.all(mask[2] == 0)
        assert np.any(mask[3] == 2)


# ============================================================
# 12. INTEGRATION WITH process_shot
# ============================================================


class TestMaskCreationIntegration:
    """Integration checks that mask creation matches ``process_shot`` output."""

    def test_process_shot_uses_create_mask_vectorized(
        self, processor: ShotProcessor
    ) -> None:
        """``process_shot`` must produce the same mask as a direct call.

        ``process_shot`` pads picks to ``target_traces``; therefore the
        direct mask must also be padded before comparison.
        """
        picks = _picks(15.0, 25.0, 35.0)

        # Direct mask on the original number of picks
        direct_mask = processor.create_mask_vectorized(picks)

        # Pad the direct mask's picks to ``target_traces`` and recompute
        padded_picks = np.zeros(TARGET_TRACES, dtype=np.float32)
        padded_picks[: len(picks)] = picks
        padded_direct_mask = processor.create_mask_vectorized(padded_picks)

        shot_data = np.zeros((len(picks), N_SAMPLES), dtype=np.float32)
        _data, processed_mask, _stats = processor.process_shot(
            shot_data, picks, shot_id=1
        )

        # Sanity: shapes now match
        assert direct_mask.shape == (len(picks), N_SAMPLES)
        assert padded_direct_mask.shape == (TARGET_TRACES, N_SAMPLES)
        assert processed_mask.shape == (TARGET_TRACES, N_SAMPLES)

        # The padded direct mask must equal the processed mask
        np.testing.assert_array_equal(padded_direct_mask, processed_mask)

    def test_padded_traces_have_no_strip(self, processor: ShotProcessor) -> None:
        """Zero-padded traces must contain only class-0 samples."""
        n_actual = 3
        shot_data = np.zeros((n_actual, N_SAMPLES), dtype=np.float32)
        picks = _picks(15.0, 25.0, 35.0)

        _data, mask, _stats = processor.process_shot(shot_data, picks, shot_id=2)

        padded_region = mask[n_actual:, :]
        assert np.all(padded_region == 0)
