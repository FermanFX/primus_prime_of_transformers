import numpy as np

from src.preprocessing.processor import ShotProcessor


class TestShotProcessor:
    def test_initialization(self):
        processor = ShotProcessor(
            target_traces=100,
            n_samples=200,
            strip_width=8,
        )

        assert processor.target_traces == 100
        assert processor.n_samples == 200
        assert processor.strip_width == 8
        assert processor.half_width == 4
        assert processor.stats == []

    def test_validate_picks_all_valid(self):
        processor = ShotProcessor(n_samples=100)

        picks = np.array([10, 20, 30, 40])

        cleaned, stats = processor.validate_picks(picks)

        np.testing.assert_array_equal(cleaned, picks)

        assert stats["total"] == 4
        assert stats["valid"] == 4
        assert stats["invalid"] == 0
        assert stats["invalid_ratio"] == 0
        assert stats["min_pick"] == 10.0
        assert stats["max_pick"] == 40.0
        assert stats["mean_pick"] == 25.0
        assert stats["median_pick"] == 25.0

    def test_validate_picks_invalid_values_are_clipped(self):
        processor = ShotProcessor(n_samples=100)

        picks = np.array([-10, 0, 50, 99, 100, 150])

        cleaned, stats = processor.validate_picks(picks)

        np.testing.assert_array_equal(
            cleaned,
            np.array([0, 0, 50, 99, 99, 99]),
        )

        assert stats["total"] == 6
        assert stats["valid"] == 2
        assert stats["invalid"] == 4
        assert stats["invalid_ratio"] == 4 / 6

    def test_validate_picks_empty(self):
        processor = ShotProcessor(n_samples=100)

        picks = np.array([])

        cleaned, stats = processor.validate_picks(picks)

        assert len(cleaned) == 0
        assert stats["total"] == 0
        assert stats["valid"] == 0
        assert stats["invalid"] == 0
        assert stats["invalid_ratio"] == 0
        assert stats["min_pick"] is None
        assert stats["max_pick"] is None
        assert stats["mean_pick"] is None
        assert stats["median_pick"] is None

    def test_create_mask_vectorized(self):
        processor = ShotProcessor(
            target_traces=3,
            n_samples=10,
            strip_width=4,
        )

        picks = np.array([5, 5, 5])

        mask = processor.create_mask_vectorized(picks)

        assert mask.shape == (3, 10)
        assert mask.dtype == np.int64

        # half_width = 2
        # Strip is samples 3..7
        assert np.all(mask[:, 0:3] == 0)
        assert np.all(mask[:, 3:8] == 2)
        assert np.all(mask[:, 8:] == 1)

    def test_create_mask_with_invalid_picks(self):
        processor = ShotProcessor(
            target_traces=2,
            n_samples=10,
            strip_width=4,
        )

        picks = np.array([5, 0])

        mask = processor.create_mask_vectorized(picks)

        # Valid pick
        assert np.all(mask[0, 0:3] == 0)
        assert np.all(mask[0, 3:8] == 2)
        assert np.all(mask[0, 8:] == 1)

        # Invalid pick must be class 0
        assert np.all(mask[1] == 0)

    def test_create_mask_for_pick_near_start(self):
        processor = ShotProcessor(
            target_traces=1,
            n_samples=10,
            strip_width=4,
        )

        picks = np.array([1])

        mask = processor.create_mask_vectorized(picks)

        assert np.all(mask[0, :4] == 2)
        assert np.all(mask[0, 4:] == 1)

    def test_create_mask_for_pick_near_end(self):
        processor = ShotProcessor(
            target_traces=1,
            n_samples=10,
            strip_width=4,
        )

        picks = np.array([8])

        mask = processor.create_mask_vectorized(picks)

        assert np.all(mask[0, :6] == 0)
        assert np.all(mask[0, 6:] == 2)

    def test_validate_mask_valid(self):
        processor = ShotProcessor(
            target_traces=2,
            n_samples=20,
            strip_width=4,
        )

        picks = np.array([10, 12])
        mask = processor.create_mask_vectorized(picks)

        assert processor.validate_mask(mask, picks) is True

    def test_validate_mask_without_strip(self):
        processor = ShotProcessor(
            target_traces=1,
            n_samples=20,
            strip_width=4,
        )

        mask = np.zeros((1, 20), dtype=np.int64)
        picks = np.array([10])

        assert processor.validate_mask(mask, picks) is False

    def test_get_shot_statistics_with_valid_picks(self):
        processor = ShotProcessor()

        picks = np.array([10, 20, 30, 0, 40])

        stats = processor.get_shot_statistics(picks)

        assert stats["n_traces"] == 5
        assert stats["n_valid"] == 4
        assert stats["n_invalid"] == 1
        assert stats["invalid_ratio"] == 0.2
        assert stats["min_pick"] == 10.0
        assert stats["max_pick"] == 40.0
        assert stats["mean_pick"] == 25.0
        assert stats["median_pick"] == 25.0

    def test_get_shot_statistics_with_no_valid_picks(self):
        processor = ShotProcessor()

        picks = np.zeros(5)

        stats = processor.get_shot_statistics(picks)

        assert stats["n_traces"] == 5
        assert stats["n_valid"] == 0
        assert stats["n_invalid"] == 5
        assert stats["invalid_ratio"] == 1.0
        assert stats["min_pick"] is None
        assert stats["max_pick"] is None
        assert stats["mean_pick"] is None
        assert stats["median_pick"] is None

    def test_process_shot_pads_data(self):
        processor = ShotProcessor(
            target_traces=4,
            n_samples=10,
            strip_width=2,
        )

        shot_data = np.ones((2, 10), dtype=np.float64)
        shot_picks = np.array([3, 5])

        data, mask, stats = processor.process_shot(
            shot_data,
            shot_picks,
            shot_id=1,
        )

        assert data.shape == (4, 10)
        assert mask.shape == (4, 10)
        assert data.dtype == np.float32
        assert mask.dtype == np.int64

        np.testing.assert_array_equal(data[:2], 1.0)
        np.testing.assert_array_equal(data[2:], 0.0)

        assert stats["original_traces"] == 2
        assert stats["padded_or_cropped"] is True

    def test_process_shot_crops_data(self):
        processor = ShotProcessor(
            target_traces=2,
            n_samples=10,
            strip_width=2,
        )

        shot_data = np.ones((4, 10), dtype=np.float64)
        shot_picks = np.array([3, 4, 5, 6])

        data, mask, stats = processor.process_shot(
            shot_data,
            shot_picks,
        )

        assert data.shape == (2, 10)
        assert mask.shape == (2, 10)
        assert stats["original_traces"] == 4
        assert stats["padded_or_cropped"] is True

    def test_process_shot_same_size(self):
        processor = ShotProcessor(
            target_traces=2,
            n_samples=10,
            strip_width=2,
        )

        shot_data = np.ones((2, 10), dtype=np.float64)
        shot_picks = np.array([3, 5])

        data, mask, stats = processor.process_shot(
            shot_data,
            shot_picks,
        )

        assert data.shape == (2, 10)
        assert mask.shape == (2, 10)
        assert stats["original_traces"] == 2
        assert stats["padded_or_cropped"] is False

    def test_process_shot_updates_stats(self):
        processor = ShotProcessor(
            target_traces=2,
            n_samples=10,
        )

        shot_data = np.ones((2, 10))
        shot_picks = np.array([3, 5])

        processor.process_shot(shot_data, shot_picks)
        processor.process_shot(shot_data, shot_picks)

        assert len(processor.stats) == 2

    def test_get_all_stats_empty(self):
        processor = ShotProcessor()

        assert processor.get_all_stats() == {}

    def test_get_all_stats(self):
        processor = ShotProcessor(
            target_traces=2,
            n_samples=10,
        )

        shot_data = np.ones((2, 10))
        picks1 = np.array([3, 5])
        picks2 = np.array([4, 0])

        processor.process_shot(shot_data, picks1)
        processor.process_shot(shot_data, picks2)

        stats = processor.get_all_stats()

        assert stats["total_shots"] == 2
        assert stats["total_traces"] == 4
        assert stats["total_valid"] == 3
        assert stats["total_invalid"] == 1
        assert stats["avg_valid_per_shot"] == 1.5
        assert stats["avg_invalid_per_shot"] == 0.5
        assert stats["shots_with_no_valid_picks"] == 0
        assert stats["min_pick_overall"] == 3.0
        assert stats["max_pick_overall"] == 5.0

    def test_reset_stats(self):
        processor = ShotProcessor(
            target_traces=2,
            n_samples=10,
        )

        shot_data = np.ones((2, 10))
        picks = np.array([3, 5])

        processor.process_shot(shot_data, picks)

        assert len(processor.stats) == 1

        processor.reset_stats()

        assert processor.stats == []
        assert processor.get_all_stats() == {}
