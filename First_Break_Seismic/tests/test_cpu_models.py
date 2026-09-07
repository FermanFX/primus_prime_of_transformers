import pytest
import torch

from src.models.efficient_unet import EfficientUNet
from src.models.light_unet import LightUNet
from src.models.mobilenet import MobileUNet
from src.models.mps_light_unet import MPSLightUNet
from src.models.nano_unet import NanoUNet
from src.models.pico_unet import PicoUNet
from src.models.tiny_unet import TinyUNet
from src.models.unet import UNet


# All segmentation models in the project.
MODEL_CLASSES = [
    NanoUNet,
    PicoUNet,
    TinyUNet,
    LightUNet,
    MPSLightUNet,
    MobileUNet,
    UNet,
    EfficientUNet,
]


def create_model(model_class):
    """Create model and explicitly place it on CPU."""

    model = model_class(
        in_channels=1,
        out_channels=3,
    )

    return model.cpu()


def create_dummy_input(batch_size=1):
    """Create dummy input directly on CPU."""

    return torch.randn(
        batch_size,
        1,
        64,
        64,
        device="cpu",
    )


def validate_cpu_output(output, batch_size):
    """Validate basic CPU output properties."""

    assert isinstance(output, torch.Tensor)

    # Output must be on CPU.
    assert output.device.type == "cpu"

    # Segmentation output must be 4D:
    # [batch, channels, height, width]
    assert output.ndim == 4

    # Batch dimension must be preserved.
    assert output.shape[0] == batch_size

    # Three segmentation classes.
    assert output.shape[1] == 3

    # No NaN or Inf values.
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_on_cpu(model_class):
    """Test every model can perform a forward pass on CPU."""

    model = create_model(model_class)
    model.eval()

    x = create_dummy_input()

    with torch.no_grad():
        output = model(x)

    print(
        f"\n{model_class.__name__}: "
        f"input={tuple(x.shape)}, "
        f"output={tuple(output.shape)}, "
        f"device={output.device}"
    )

    validate_cpu_output(
        output,
        batch_size=1,
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_cpu_dummy_data(model_class):
    """Test every model with dummy data on CPU."""

    model = create_model(model_class)
    model.eval()

    x = torch.randn(
        2,
        1,
        63,
        67,
        device="cpu",
    )

    with torch.no_grad():
        output = model(x)

    # Input must be on CPU.
    assert x.device.type == "cpu"

    # Output must also be on CPU.
    assert output.device.type == "cpu"

    validate_cpu_output(
        output,
        batch_size=2,
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_cpu_multiple_forward_passes(model_class):
    """Verify that models can perform repeated CPU inference."""

    model = create_model(model_class)
    model.eval()

    for _ in range(3):
        x = create_dummy_input()

        with torch.no_grad():
            output = model(x)

        validate_cpu_output(
            output,
            batch_size=1,
        )
