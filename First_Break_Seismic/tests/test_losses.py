import torch
from torch import nn

from src.training.losses import (
    FocalLoss,
    DiceLoss,
    ComboLoss,
    create_loss_function,
)


# ============================================================
# Test configuration
# ============================================================

class TestConfig:
    def __init__(
        self,
        loss_function="cross_entropy",
        class_weights=None,
        focal_gamma=2.0,
        dice_weight=0.5,
    ):
        self.loss_function = loss_function
        self.class_weights = (
            class_weights
            if class_weights is not None
            else [0.2, 0.2, 0.6]
        )
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight


# ============================================================
# Helper function
# ============================================================

def create_fake_data():
    """
    Create fake segmentation data.

    logits:
        [batch, classes, height, width]

    target:
        [batch, height, width]
    """

    torch.manual_seed(42)

    logits = torch.randn(2, 3, 4, 4)

    target = torch.randint(
        low=0,
        high=3,
        size=(2, 4, 4),
    )

    return logits, target


# ============================================================
# FocalLoss tests
# ============================================================

def test_focal_loss():
    logits, target = create_fake_data()

    loss_fn = FocalLoss(
        alpha=[0.2, 0.2, 0.6],
        gamma=2.0,
    )

    loss = loss_fn(logits, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0


def test_focal_loss_sum_reduction():
    logits, target = create_fake_data()

    loss_fn = FocalLoss(
        alpha=[0.2, 0.2, 0.6],
        gamma=2.0,
        reduction="sum",
    )

    loss = loss_fn(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0


def test_focal_loss_none_reduction():
    logits, target = create_fake_data()

    loss_fn = FocalLoss(
        alpha=[0.2, 0.2, 0.6],
        gamma=2.0,
        reduction="none",
    )

    loss = loss_fn(logits, target)

    assert loss.shape == target.shape
    assert torch.all(torch.isfinite(loss))
    assert torch.all(loss >= 0)


def test_focal_loss_backward():
    logits, target = create_fake_data()

    logits.requires_grad_(True)

    loss_fn = FocalLoss(
        alpha=[0.2, 0.2, 0.6],
        gamma=2.0,
    )

    loss = loss_fn(logits, target)

    loss.backward()

    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


# ============================================================
# DiceLoss tests
# ============================================================

def test_dice_loss():
    logits, target = create_fake_data()

    loss_fn = DiceLoss(
        smooth=1e-6,
        num_classes=3,
    )

    loss = loss_fn(logits, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # Dice loss should normally be between 0 and 1
    assert 0 <= loss <= 1


def test_dice_loss_backward():
    logits, target = create_fake_data()

    logits.requires_grad_(True)

    loss_fn = DiceLoss(
        smooth=1e-6,
        num_classes=3,
    )

    loss = loss_fn(logits, target)

    loss.backward()

    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_dice_loss_different_number_of_classes():
    torch.manual_seed(42)

    logits = torch.randn(2, 4, 4, 4)
    target = torch.randint(0, 4, (2, 4, 4))

    loss_fn = DiceLoss(
        num_classes=4,
    )

    loss = loss_fn(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ============================================================
# ComboLoss tests
# ============================================================

def test_combo_loss():
    logits, target = create_fake_data()

    loss_fn = ComboLoss(
        class_weights=[0.2, 0.2, 0.6],
        dice_weight=0.5,
        focal_gamma=2.0,
    )

    loss = loss_fn(logits, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0


def test_combo_loss_backward():
    logits, target = create_fake_data()

    logits.requires_grad_(True)

    loss_fn = ComboLoss(
        class_weights=[0.2, 0.2, 0.6],
        dice_weight=0.5,
        focal_gamma=2.0,
    )

    loss = loss_fn(logits, target)

    loss.backward()

    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


# ============================================================
# ComboLoss component tracking
# ============================================================

def test_combo_loss_returns_components():
    logits, target = create_fake_data()

    loss_fn = ComboLoss(
        class_weights=[0.2, 0.2, 0.6],
        dice_weight=0.5,
        focal_gamma=2.0,
    )

    loss_fn.return_components = True

    total_loss, components = loss_fn(logits, target)

    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(components, dict)

    assert "total" in components
    assert "ce" in components
    assert "focal" in components
    assert "dice" in components
    assert "ce_focal_combined" in components
    assert "per_class" in components


def test_combo_loss_component_values():
    logits, target = create_fake_data()

    loss_fn = ComboLoss(
        class_weights=[0.2, 0.2, 0.6],
    )

    loss_fn.return_components = True

    _total_loss, components = loss_fn(logits, target)

    assert torch.isfinite(
        torch.tensor(components["total"])
    )

    assert torch.isfinite(
        torch.tensor(components["ce"])
    )

    assert torch.isfinite(
        torch.tensor(components["focal"])
    )

    assert torch.isfinite(
        torch.tensor(components["dice"])
    )

    assert torch.isfinite(
        torch.tensor(components["ce_focal_combined"])
    )


# ============================================================
# ComboLoss per-class tests
# ============================================================

def test_combo_loss_per_class():
    logits, target = create_fake_data()

    loss_fn = ComboLoss(
        class_weights=[0.2, 0.2, 0.6],
    )

    loss_fn.return_components = True

    _, components = loss_fn(logits, target)

    per_class = components["per_class"]

    assert "class_0" in per_class
    assert "class_1" in per_class
    assert "class_2" in per_class
    assert "strip" in per_class


def test_combo_loss_strip_is_class_2():
    logits, target = create_fake_data()

    loss_fn = ComboLoss(
        class_weights=[0.2, 0.2, 0.6],
    )

    loss_fn.return_components = True

    _, components = loss_fn(logits, target)

    per_class = components["per_class"]

    assert per_class["strip"] == per_class["class_2"]


def test_combo_loss_per_class_values_are_valid():
    logits, target = create_fake_data()

    loss_fn = ComboLoss(
        class_weights=[0.2, 0.2, 0.6],
    )

    loss_fn.return_components = True

    _, components = loss_fn(logits, target)

    per_class = components["per_class"]

    for key in ["class_0", "class_1", "class_2", "strip"]:
        assert torch.isfinite(
            torch.tensor(per_class[key])
        )

        assert per_class[key] >= 0


# ============================================================
# create_loss_function() tests
# ============================================================

def test_create_cross_entropy_loss():
    config = TestConfig(
        loss_function="cross_entropy",
    )

    loss_fn = create_loss_function(config)

    assert isinstance(
        loss_fn,
        nn.CrossEntropyLoss,
    )


def test_create_focal_loss():
    config = TestConfig(
        loss_function="focal",
        focal_gamma=2.0,
    )

    loss_fn = create_loss_function(config)

    assert isinstance(
        loss_fn,
        FocalLoss,
    )

    assert loss_fn.gamma == 2.0


def test_create_dice_loss():
    config = TestConfig(
        loss_function="dice",
        class_weights=[0.2, 0.2, 0.6],
    )

    loss_fn = create_loss_function(config)

    assert isinstance(
        loss_fn,
        DiceLoss,
    )

    assert loss_fn.num_classes == 3


def test_create_combo_loss():
    config = TestConfig(
        loss_function="combo",
        class_weights=[0.2, 0.2, 0.6],
        dice_weight=0.5,
        focal_gamma=2.0,
    )

    loss_fn = create_loss_function(config)

    assert isinstance(
        loss_fn,
        ComboLoss,
    )

    assert loss_fn.dice_weight == 0.5
    assert loss_fn.focal_gamma == 2.0


def test_create_loss_unknown_type():
    config = TestConfig(
        loss_function="something_wrong",
    )

    try:
        create_loss_function(config)
        assert False, "Expected ValueError"
    except ValueError as error:
        assert "Unknown loss function" in str(error)


# ============================================================
# Integration test
# ============================================================

def test_all_loss_functions():
    """
    Make sure every supported loss function
    can process the same segmentation data.
    """

    logits, target = create_fake_data()

    loss_types = [
        "cross_entropy",
        "focal",
        "dice",
        "combo",
    ]

    for loss_type in loss_types:
        config = TestConfig(
            loss_function=loss_type,
        )

        loss_fn = create_loss_function(config)

        loss = loss_fn(logits, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss >= 0
