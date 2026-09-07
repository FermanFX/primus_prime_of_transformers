import pytest
import torch

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
    MPSLightUNet,
    MobileUNet,
    UNet,
]


STANDARD_SIZE = (1578, 751)
LARGE_SIZE = (1600, 800)
SMALL_SIZE = (1500, 700)

NON_STANDARD_SIZES = [
    (512, 1024),
    (1024, 512),
    (333, 777),
    (777, 333),
]


def create_model(model_class):
    """Create model in evaluation mode on CPU."""

    model = model_class(
        in_channels=1,
        out_channels=3,
    )

    model.eval()

    return model.cpu()


def run_model(model_class, height, width):
    """Run model with the requested input size."""

    model = create_model(model_class)

    x = torch.randn(
        1,
        1,
        height,
        width,
        device="cpu",
    )

    with torch.no_grad():
        output = model(x)

    return x, output


def assert_output_matches_input(
    output,
    height,
    width,
):
    """Verify that output spatial size matches input size."""

    expected_shape = (
        1,
        3,
        height,
        width,
    )

    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(output.shape)}"
    )

    assert output.device.type == "cpu"

    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_padding_standard_size(model_class):
    """
    Test standard input size.

    Input: 1578 x 751
    Output must be exactly 1578 x 751.
    """

    height, width = STANDARD_SIZE

    x, output = run_model(
        model_class,
        height,
        width,
    )

    print(
        f"\n{model_class.__name__}: "
        f"input={tuple(x.shape)}, "
        f"output={tuple(output.shape)}"
    )

    assert_output_matches_input(
        output,
        height,
        width,
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_padding_large_size(model_class):
    """
    Test larger input size.

    Input: 1600 x 800
    Output must be exactly 1600 x 800.
    """

    height, width = LARGE_SIZE

    x, output = run_model(
        model_class,
        height,
        width,
    )

    print(
        f"\n{model_class.__name__}: "
        f"input={tuple(x.shape)}, "
        f"output={tuple(output.shape)}"
    )

    assert_output_matches_input(
        output,
        height,
        width,
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_padding_small_size(model_class):
    """
    Test smaller input size.

    Input: 1500 x 700
    Output must be exactly 1500 x 700.
    """

    height, width = SMALL_SIZE

    x, output = run_model(
        model_class,
        height,
        width,
    )

    print(
        f"\n{model_class.__name__}: "
        f"input={tuple(x.shape)}, "
        f"output={tuple(output.shape)}"
    )

    assert_output_matches_input(
        output,
        height,
        width,
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize(
    "height,width",
    NON_STANDARD_SIZES,
)
def test_padding_non_standard_aspect_ratio(
    model_class,
    height,
    width,
):
    """
    Test non-standard aspect ratios.

    Output must preserve the exact input dimensions.
    """

    x, output = run_model(
        model_class,
        height,
        width,
    )

    print(
        f"\n{model_class.__name__}: "
        f"input={tuple(x.shape)}, "
        f"output={tuple(output.shape)}"
    )

    assert_output_matches_input(
        output,
        height,
        width,
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize(
    "height,width",
    [
        (1577, 750),
        (1579, 752),
        (1501, 701),
        (1601, 801),
        (127, 131),
        (65, 73),
    ],
)
def test_padding_non_multiple_dimensions(
    model_class,
    height,
    width,
):
    """
    Test dimensions that are not multiples of common
    U-Net downsampling factors.

    This verifies padding and final cropping.
    """

    x, output = run_model(
        model_class,
        height,
        width,
    )

    assert x.shape == (
        1,
        1,
        height,
        width,
    )

    assert_output_matches_input(
        output,
        height,
        width,
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_padding_preserves_batch_size(model_class):
    """Verify batch size and spatial dimensions are preserved."""

    height, width = STANDARD_SIZE

    model = create_model(model_class)

    x = torch.randn(
        2,
        1,
        height,
        width,
        device="cpu",
    )

    with torch.no_grad():
        output = model(x)

    expected_shape = (
        2,
        3,
        height,
        width,
    )

    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(output.shape)}"
    )

    assert output.device.type == "cpu"

    assert torch.isfinite(output).all()
