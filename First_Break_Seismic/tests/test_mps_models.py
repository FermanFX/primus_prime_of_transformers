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
    LightUNet,
    MPSLightUNet,
    MobileUNet,
    UNet,
]


def get_device() -> torch.device:
    """Use MPS when available, otherwise fall back to CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_model(model_class, device):
    """Create a model and move it to the selected device."""

    model = model_class(
        in_channels=1,
        out_channels=3,
    )

    return model.to(device)


def create_dummy_input(device, batch_size=1):
    """Create dummy input data on the selected device."""

    return torch.randn(
        batch_size,
        1,
        64,
        64,
        device=device,
    )


def validate_output(output, device, batch_size):
    """Validate basic model output properties."""

    assert isinstance(output, torch.Tensor)

    # Output must remain on the selected device.
    assert output.device.type == device.type

    # Segmentation output must be 4D:
    # [batch, channels, height, width]
    assert output.ndim == 4

    # Batch size must be preserved.
    assert output.shape[0] == batch_size

    # Number of output classes must be preserved.
    assert output.shape[1] == 3

    # Output must contain valid numerical values.
    assert torch.isfinite(output).all()


def test_mps_availability():
    """Verify PyTorch MPS availability detection."""

    available = torch.backends.mps.is_available()

    print(f"\nMPS available: {available}")

    if available:
        assert torch.backends.mps.is_built()


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_device(model_class):
    """
    Test every model on MPS.

    If MPS is unavailable, CPU is used automatically.
    """

    device = get_device()

    model = create_model(model_class, device)
    model.eval()

    x = create_dummy_input(device)

    with torch.no_grad():
        output = model(x)

    print(
        f"\n{model_class.__name__}: "
        f"device={device}, "
        f"input={tuple(x.shape)}, "
        f"output={tuple(output.shape)}, "
        f"output_device={output.device}"
    )

    validate_output(
        output,
        device,
        batch_size=1,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available",
)
@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_on_mps(model_class):
    """Explicitly verify every model runs on MPS."""

    device = torch.device("mps")

    model = create_model(model_class, device)
    model.eval()

    x = create_dummy_input(
        device,
        batch_size=2,
    )

    with torch.no_grad():
        output = model(x)

    print(
        f"\n{model_class.__name__}: "
        f"MPS input={tuple(x.shape)}, "
        f"output={tuple(output.shape)}"
    )

    assert output.device.type == "mps"

    validate_output(
        output,
        device,
        batch_size=2,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available",
)
@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_dummy_data_on_mps(model_class):
    """Test every model with dummy data allocated directly on MPS."""

    device = torch.device("mps")

    model = create_model(model_class, device)
    model.train()

    # Non-standard dimensions also test model padding/shape handling.
    x = torch.randn(
        2,
        1,
        63,
        67,
        device=device,
    )

    output = model(x)

    assert x.device.type == "mps"
    assert output.device.type == "mps"

    validate_output(
        output,
        device,
        batch_size=2,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available",
)
@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_model_mps_memory_usage(model_class):
    """Verify that every model can allocate and use MPS memory."""

    device = torch.device("mps")

    if hasattr(torch.mps, "reset_peak_memory_stats"):
        torch.mps.reset_peak_memory_stats()

    before = torch.mps.current_allocated_memory()

    model = create_model(model_class, device)
    model.eval()

    x = create_dummy_input(
        device,
        batch_size=1,
    )

    with torch.no_grad():
        output = model(x)

    if hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

    after = torch.mps.current_allocated_memory()

    print(
        f"\n{model_class.__name__} MPS memory: "
        f"{before / 1024**2:.2f} MB -> "
        f"{after / 1024**2:.2f} MB"
    )

    assert after >= before
    assert output.device.type == "mps"


def test_cpu_fallback_when_mps_unavailable(monkeypatch):
    """
    Simulate an environment without MPS and verify CPU fallback.
    """

    monkeypatch.setattr(
        torch.backends.mps,
        "is_available",
        lambda: False,
    )

    device = get_device()

    assert device.type == "cpu"

    model = create_model(
        NanoUNet,
        device,
    )

    model.eval()

    x = create_dummy_input(device)

    with torch.no_grad():
        output = model(x)

    assert output.device.type == "cpu"
    assert output.ndim == 4
    assert output.shape[0] == 1
    assert output.shape[1] == 3
    assert torch.isfinite(output).all()
