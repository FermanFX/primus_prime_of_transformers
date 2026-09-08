import pytest

from src.models.light_unet import LightUNet
from src.models.mobilenet import MobileUNet
from src.models.mps_light_unet import MPSLightUNet
from src.models.nano_unet import NanoUNet
from src.models.pico_unet import PicoUNet
from src.models.tiny_unet import TinyUNet
from src.models.unet import UNet


MODEL_CLASSES = [
    NanoUNet,
    PicoUNet,
    TinyUNet,
    LightUNet,
    MPSLightUNet,
    MobileUNet,
    UNet,
]


EXPECTED_PARAMETER_COUNTS = {
    NanoUNet: 7_817,
    PicoUNet: 532,
    TinyUNet: 30_655,
    LightUNet: 773_827,
    MPSLightUNet: 1_943_795,
    MobileUNet: 3_675_401,
    UNet: 7_765_475,
}


def create_model(model_class):
    """Create model with the standard test configuration."""

    return model_class(
        in_channels=1,
        out_channels=3,
    )


def count_parameters(model):
    """Count all model parameters."""

    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model):
    """Count trainable model parameters."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_parameter_count_is_positive(model_class):
    """Every model must contain parameters."""

    model = create_model(model_class)

    total_params = count_parameters(model)

    assert total_params > 0


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_parameter_count_is_consistent(model_class):
    """
    Multiple instances of the same model must have
    exactly the same parameter count.
    """

    model_1 = create_model(model_class)
    model_2 = create_model(model_class)
    model_3 = create_model(model_class)

    count_1 = count_parameters(model_1)
    count_2 = count_parameters(model_2)
    count_3 = count_parameters(model_3)

    print(
        f"\n{model_class.__name__}: "
        f"run1={count_1:,}, "
        f"run2={count_2:,}, "
        f"run3={count_3:,}"
    )

    assert count_1 == count_2
    assert count_2 == count_3


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_trainable_parameter_count_is_consistent(model_class):
    """
    Trainable parameter count must be consistent
    between model instantiations.
    """

    model_1 = create_model(model_class)
    model_2 = create_model(model_class)

    count_1 = count_trainable_parameters(model_1)
    count_2 = count_trainable_parameters(model_2)

    print(f"\n{model_class.__name__}: trainable1={count_1:,}, trainable2={count_2:,}")

    assert count_1 == count_2


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_parameter_count_matches_expected(model_class):
    """
    Verify that each model has the expected parameter count.
    """

    model = create_model(model_class)

    actual = count_parameters(model)
    expected = EXPECTED_PARAMETER_COUNTS[model_class]

    print(f"\n{model_class.__name__}: expected={expected:,}, actual={actual:,}")

    assert actual == expected, (
        f"{model_class.__name__}: expected {expected:,} parameters, got {actual:,}"
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_trainable_parameters_do_not_exceed_total(model_class):
    """Trainable parameters cannot exceed total parameters."""

    model = create_model(model_class)

    total = count_parameters(model)
    trainable = count_trainable_parameters(model)

    assert trainable <= total


def test_all_parameter_counts_are_deterministic():
    """
    Instantiate every model five times and verify that
    parameter counts never change.
    """

    for model_class in MODEL_CLASSES:
        counts = []

        for _ in range(5):
            model = create_model(model_class)
            counts.append(count_parameters(model))

        assert len(set(counts)) == 1, (
            f"{model_class.__name__}: inconsistent parameter counts: {counts}"
        )


def test_expected_parameter_counts_are_defined_for_all_models():
    """Every tested model must have an expected parameter count."""

    for model_class in MODEL_CLASSES:
        assert model_class in EXPECTED_PARAMETER_COUNTS
        assert EXPECTED_PARAMETER_COUNTS[model_class] > 0
