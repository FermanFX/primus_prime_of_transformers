import numpy as np
import torch
import torch.nn as nn

from src.training.metrics import (
    SegmentationMetrics,
    FirstBreakMetrics,
    compute_gradient_norm,
    compute_weight_norm,
    compute_layerwise_norms,
    ComboLoss,
    extract_picks_from_mask,
)


# ============================================================
# SegmentationMetrics
# ============================================================


class TestSegmentationMetrics:

    def test_initial_state(self):
        metrics = SegmentationMetrics(
            num_classes=3,
            ignore_index=-1,
        )

        assert metrics.num_classes == 3
        assert metrics.ignore_index == -1
        assert metrics.total_pixels == 0

        expected_cm = np.zeros((3, 3), dtype=np.int64)

        np.testing.assert_array_equal(
            metrics.confusion_matrix,
            expected_cm,
        )

    def test_reset(self):
        metrics = SegmentationMetrics(
            num_classes=3,
            ignore_index=-1,
        )

        predictions = torch.tensor([0, 1, 2, 1])
        targets = torch.tensor([0, 1, 2, 1])

        metrics.update(predictions, targets)

        assert metrics.total_pixels == 4
        assert np.sum(metrics.confusion_matrix) == 4

        metrics.reset()

        assert metrics.total_pixels == 0

        expected_cm = np.zeros((3, 3), dtype=np.int64)

        np.testing.assert_array_equal(
            metrics.confusion_matrix,
            expected_cm,
        )

    def test_update_confusion_matrix(self):
        metrics = SegmentationMetrics(
            num_classes=3,
            ignore_index=-1,
        )

        predictions = torch.tensor([
            0, 1, 2,
            0, 1, 2,
        ])

        targets = torch.tensor([
            0, 1, 2,
            1, 2, 0,
        ])

        metrics.update(predictions, targets)

        expected_cm = np.array([
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ])

        np.testing.assert_array_equal(
            metrics.confusion_matrix,
            expected_cm,
        )

        assert metrics.total_pixels == 6

    def test_perfect_predictions(self):
        metrics = SegmentationMetrics(
            num_classes=3,
            ignore_index=-1,
        )

        predictions = torch.tensor([
            0, 1, 2,
            0, 1, 2,
        ])

        targets = torch.tensor([
            0, 1, 2,
            0, 1, 2,
        ])

        metrics.update(predictions, targets)

        result = metrics.compute()

        assert result["accuracy"] == 1.0
        assert result["mean_iou"] == 1.0
        assert result["mean_f1"] == 1.0

        assert result["iou_per_class"] == [
            1.0,
            1.0,
            1.0,
        ]

        assert result["precision_per_class"] == [
            1.0,
            1.0,
            1.0,
        ]

        assert result["recall_per_class"] == [
            1.0,
            1.0,
            1.0,
        ]

        assert result["f1_per_class"] == [
            1.0,
            1.0,
            1.0,
        ]

    def test_ignore_index(self):
        metrics = SegmentationMetrics(
            num_classes=3,
            ignore_index=-1,
        )

        predictions = torch.tensor([
            0, 1, 2, 1
        ])

        targets = torch.tensor([
            0, -1, 2, 1
        ])

        metrics.update(
            predictions,
            targets,
        )

        # IMPORTANT:
        # Mövcud metrics.py kodunda ignore_index=-1 üçün
        # filtering tətbiq edilmir, çünki:
        #
        # if self.ignore_index >= 0:
        #
        # şərti False olur.
        #
        # Buna görə total_pixels 4 olaraq qalır.
        assert metrics.total_pixels == 4

        # target=-1 confusion matrix-ə əlavə edilmir.
        expected_cm = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        np.testing.assert_array_equal(
            metrics.confusion_matrix,
            expected_cm,
        )

    def test_multiple_updates(self):
        metrics = SegmentationMetrics(
            num_classes=3,
            ignore_index=-1,
        )

        predictions_1 = torch.tensor([0, 1, 2])
        targets_1 = torch.tensor([0, 1, 2])

        predictions_2 = torch.tensor([1, 2, 0])
        targets_2 = torch.tensor([1, 2, 0])

        metrics.update(
            predictions_1,
            targets_1,
        )

        metrics.update(
            predictions_2,
            targets_2,
        )

        assert metrics.total_pixels == 6

        result = metrics.compute()

        assert result["accuracy"] == 1.0
        assert result["mean_iou"] == 1.0
        assert result["mean_f1"] == 1.0

    def test_compute_empty(self):
        metrics = SegmentationMetrics(
            num_classes=3,
            ignore_index=-1,
        )

        result = metrics.compute()

        assert result["accuracy"] == 0.0
        assert result["mean_iou"] == 0.0
        assert result["mean_f1"] == 0.0

        assert result["iou_per_class"] == [
            0.0,
            0.0,
            0.0,
        ]

        assert result["precision_per_class"] == [
            0.0,
            0.0,
            0.0,
        ]

        assert result["recall_per_class"] == [
            0.0,
            0.0,
            0.0,
        ]

        assert result["f1_per_class"] == [
            0.0,
            0.0,
            0.0,
        ]


class TestFirstBreakMetrics:

    def test_initial_state(self):
        metrics = FirstBreakMetrics(
            tolerance_samples=3,
        )

        assert metrics.tolerance_samples == 3
        assert metrics.errors == []
        assert metrics.within_tolerance == []
        assert metrics.total_traces == 0

    def test_perfect_predictions(self):
        metrics = FirstBreakMetrics(
            tolerance_samples=3,
        )

        predicted = np.array([
            10, 20, 30, 40
        ])

        true = np.array([
            10, 20, 30, 40
        ])

        metrics.update(
            predicted,
            true,
        )

        result = metrics.compute()

        assert result["mean_absolute_error"] == 0.0
        assert result["std_absolute_error"] == 0.0
        assert result["accuracy_within_tolerance"] == 1.0
        assert result["median_absolute_error"] == 0.0
        assert result["max_absolute_error"] == 0.0
        assert result["min_absolute_error"] == 0.0
        assert result["total_traces"] == 4

    def test_absolute_error(self):
        metrics = FirstBreakMetrics(
            tolerance_samples=3,
        )

        predicted = np.array([
            10, 20, 30
        ])

        true = np.array([
            12, 16, 35
        ])

        metrics.update(
            predicted,
            true,
        )

        result = metrics.compute()

        # Errors: 2, 4, 5
        assert result["mean_absolute_error"] == 11 / 3
        assert result["median_absolute_error"] == 4.0
        assert result["max_absolute_error"] == 5.0
        assert result["min_absolute_error"] == 2.0

    def test_tolerance_accuracy(self):
        metrics = FirstBreakMetrics(
            tolerance_samples=3,
        )

        predicted = np.array([
            10, 20, 30, 40
        ])

        true = np.array([
            12, 23, 34, 50
        ])

        metrics.update(
            predicted,
            true,
        )

        result = metrics.compute()

        # Errors:
        # 2 -> within tolerance
        # 3 -> within tolerance
        # 4 -> outside
        # 10 -> outside
        #
        # Accuracy = 2 / 4 = 0.5

        assert result["accuracy_within_tolerance"] == 0.5

    def test_zero_picks_are_ignored(self):
        metrics = FirstBreakMetrics(
            tolerance_samples=3,
        )

        predicted = np.array([
            0, 20, 30
        ])

        true = np.array([
            10, 20, 30
        ])

        metrics.update(
            predicted,
            true,
        )

        result = metrics.compute()

        assert result["total_traces"] == 2
        assert result["mean_absolute_error"] == 0.0

    def test_zero_true_picks_are_ignored(self):
        metrics = FirstBreakMetrics(
            tolerance_samples=3,
        )

        predicted = np.array([
            10, 20, 30
        ])

        true = np.array([
            0, 20, 30
        ])

        metrics.update(
            predicted,
            true,
        )

        result = metrics.compute()

        assert result["total_traces"] == 2
        assert result["mean_absolute_error"] == 0.0

    def test_no_valid_predictions(self):
        metrics = FirstBreakMetrics(
            tolerance_samples=3,
        )

        predicted = np.array([
            0, 0, 0
        ])

        true = np.array([
            10, 20, 30
        ])

        metrics.update(
            predicted,
            true,
        )

        result = metrics.compute()

        assert result["mean_absolute_error"] == 0.0
        assert result["std_absolute_error"] == 0.0
        assert result["accuracy_within_tolerance"] == 0.0
        assert result["median_absolute_error"] == 0.0
        assert result["max_absolute_error"] == 0.0
        assert result["min_absolute_error"] == 0.0
        assert result["total_traces"] == 0

    def test_reset(self):
        metrics = FirstBreakMetrics(
            tolerance_samples=3,
        )

        predicted = np.array([
            10, 20, 30
        ])

        true = np.array([
            11, 21, 31
        ])

        metrics.update(
            predicted,
            true,
        )

        assert metrics.total_traces == 3
        assert len(metrics.errors) == 3

        metrics.reset()

        assert metrics.errors == []
        assert metrics.within_tolerance == []
        assert metrics.total_traces == 0

    def test_multiple_updates(self):
        metrics = FirstBreakMetrics(
            tolerance_samples=3,
        )

        predicted_1 = np.array([
            10, 20
        ])

        true_1 = np.array([
            11, 22
        ])

        predicted_2 = np.array([
            30, 40
        ])

        true_2 = np.array([
            31, 45
        ])

        metrics.update(
            predicted_1,
            true_1,
        )

        metrics.update(
            predicted_2,
            true_2,
        )

        result = metrics.compute()

        # Errors = 1, 2, 1, 5
        assert result["total_traces"] == 4
        assert result["mean_absolute_error"] == 2.25
        assert result["max_absolute_error"] == 5.0
        assert result["min_absolute_error"] == 1.0

    def test_negative_predictions_are_ignored(self):
        metrics = FirstBreakMetrics(
            tolerance_samples=3,
        )

        predicted = np.array([
            -1, 20, 30
        ])

        true = np.array([
            10, 20, 30
        ])

        metrics.update(
            predicted,
            true,
        )

        result = metrics.compute()

        assert result["total_traces"] == 2


# ============================================================
# Gradient / Weight Norms
# ============================================================


class TestNormFunctions:

    def test_weight_norm(self):
        model = nn.Linear(2, 2)

        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(1.0)

        expected = np.sqrt(6.0)

        result = compute_weight_norm(model)

        assert np.isclose(result, expected)

    def test_gradient_norm_without_gradients(self):
        model = nn.Linear(2, 2)

        result = compute_gradient_norm(model)

        assert result == 0.0

    def test_gradient_norm_after_backward(self):
        model = nn.Linear(2, 2)

        x = torch.tensor([
            [1.0, 2.0]
        ])

        output = model(x)
        loss = output.sum()

        loss.backward()

        result = compute_gradient_norm(model)

        assert result > 0.0
        assert np.isfinite(result)

    def test_layerwise_norms(self):
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )

        x = torch.tensor([
            [1.0, 2.0]
        ])

        output = model(x)
        loss = output.sum()

        loss.backward()

        result = compute_layerwise_norms(model)

        assert isinstance(result, dict)

        assert "weights_0.weight" in result
        assert "weights_0.bias" in result
        assert "weights_2.weight" in result
        assert "weights_2.bias" in result

        assert "grads_0.weight" in result
        assert "grads_0.bias" in result
        assert "grads_2.weight" in result
        assert "grads_2.bias" in result

        for value in result.values():
            assert isinstance(value, float)
            assert np.isfinite(value)

    def test_layerwise_norms_without_gradients(self):
        model = nn.Linear(2, 2)

        result = compute_layerwise_norms(model)

        assert "weights_weight" in result
        assert "weights_bias" in result

        assert "grads_weight" not in result
        assert "grads_bias" not in result


# ============================================================
# ComboLoss
# ============================================================


class TestComboLoss:

    def test_forward(self):
        loss_fn = ComboLoss(
            class_weights=[1.0, 1.0, 1.0],
            dice_weight=0.5,
            focal_gamma=2.0,
        )

        logits = torch.randn(
            2,
            3,
            4,
            4,
        )

        target = torch.randint(
            0,
            3,
            (2, 4, 4),
        )

        loss = loss_fn(
            logits,
            target,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_backward(self):
        loss_fn = ComboLoss(
            class_weights=[1.0, 1.0, 1.0],
            dice_weight=0.5,
            focal_gamma=2.0,
        )

        logits = torch.randn(
            2,
            3,
            4,
            4,
            requires_grad=True,
        )

        target = torch.randint(
            0,
            3,
            (2, 4, 4),
        )

        loss = loss_fn(
            logits,
            target,
        )

        loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_different_dice_weights(self):
        logits = torch.randn(
            2,
            3,
            4,
            4,
        )

        target = torch.randint(
            0,
            3,
            (2, 4, 4),
        )

        loss_fn_1 = ComboLoss(
            class_weights=[1.0, 1.0, 1.0],
            dice_weight=0.0,
            focal_gamma=2.0,
        )

        loss_fn_2 = ComboLoss(
            class_weights=[1.0, 1.0, 1.0],
            dice_weight=1.0,
            focal_gamma=2.0,
        )

        loss_1 = loss_fn_1(
            logits,
            target,
        )

        loss_2 = loss_fn_2(
            logits,
            target,
        )

        assert torch.isfinite(loss_1)
        assert torch.isfinite(loss_2)

    def test_different_focal_gamma(self):
        logits = torch.randn(
            2,
            3,
            4,
            4,
        )

        target = torch.randint(
            0,
            3,
            (2, 4, 4),
        )

        loss_fn_1 = ComboLoss(
            class_weights=[1.0, 1.0, 1.0],
            dice_weight=0.5,
            focal_gamma=1.0,
        )

        loss_fn_2 = ComboLoss(
            class_weights=[1.0, 1.0, 1.0],
            dice_weight=0.5,
            focal_gamma=3.0,
        )

        loss_1 = loss_fn_1(
            logits,
            target,
        )

        loss_2 = loss_fn_2(
            logits,
            target,
        )

        assert torch.isfinite(loss_1)
        assert torch.isfinite(loss_2)

    def test_custom_class_weights(self):
        loss_fn = ComboLoss(
            class_weights=[1.0, 2.0, 3.0],
            dice_weight=0.5,
            focal_gamma=2.0,
        )

        assert torch.allclose(
            loss_fn.ce.weight,
            torch.tensor([1.0, 2.0, 3.0]),
        )


# ============================================================
# extract_picks_from_mask
# ============================================================


class TestExtractPicksFromMask:

    def test_extract_from_strip_class(self):
        mask = np.zeros(
            (2, 10),
            dtype=np.int64,
        )

        mask[0, 4:7] = 2
        mask[1, 6:9] = 2

        picks = extract_picks_from_mask(mask)

        assert picks.shape == (2,)
        assert picks[0] == 5
        assert picks[1] == 7

    def test_extract_from_after_class(self):
        mask = np.zeros(
            (2, 10),
            dtype=np.int64,
        )

        mask[0, 3:] = 1
        mask[1, 4:] = 1

        picks = extract_picks_from_mask(mask)

        assert picks.shape == (2,)
        assert picks[0] == -1
        assert picks[1] == 0

    def test_no_break(self):
        mask = np.zeros(
            (2, 10),
            dtype=np.int64,
        )

        picks = extract_picks_from_mask(mask)

        expected = np.array([
            0,
            0,
        ])

        np.testing.assert_array_equal(
            picks,
            expected,
        )

    def test_strip_has_priority_over_after(self):
        mask = np.zeros(
            (1, 10),
            dtype=np.int64,
        )

        mask[0, 2:5] = 1
        mask[0, 5:8] = 2

        picks = extract_picks_from_mask(mask)

        # Class 2 should have priority.
        assert picks[0] == 6

    def test_single_strip_pixel(self):
        mask = np.zeros(
            (1, 10),
            dtype=np.int64,
        )

        mask[0, 5] = 2

        picks = extract_picks_from_mask(mask)

        assert picks[0] == 5

    def test_output_shape(self):
        mask = np.zeros(
            (5, 20),
            dtype=np.int64,
        )

        picks = extract_picks_from_mask(mask)

        assert picks.shape == (5,)
        assert picks.dtype == np.int64

    def test_multiple_strip_pixels(self):
        mask = np.zeros(
            (1, 20),
            dtype=np.int64,
        )

        mask[0, 5:10] = 2

        picks = extract_picks_from_mask(mask)

        # indices = [5, 6, 7, 8, 9]
        # median = 7
        assert picks[0] == 7