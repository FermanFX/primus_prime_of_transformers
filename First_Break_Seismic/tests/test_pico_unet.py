import torch

from src.models.pico_unet import PicoUNet


def test_pico_unet_forward_cpu():
    """Test PicoUNet forward pass on CPU."""

    model = PicoUNet(
        in_channels=1,
        out_channels=3,
    ).cpu()

    model.eval()

    # Use non-multiple-of-4 dimensions to also test
    # the padding/cropping logic.
    x = torch.randn(
        1,
        1,
        63,
        67,
    )

    with torch.no_grad():
        output = model(x)

    # Forward pass must complete successfully.
    assert output is not None

    # Output must remain on CPU.
    assert output.device.type == "cpu"

    # Output shape must match input spatial dimensions.
    assert output.shape == (1, 3, 63, 67)

    # Output must not contain NaN or Inf.
    assert torch.isfinite(output).all()


def test_pico_unet_parameter_count():
    """Test that PicoUNet remains a minimal model."""

    model = PicoUNet(
        in_channels=1,
        out_channels=3,
    )

    total_params = sum(parameter.numel() for parameter in model.parameters())

    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(f"\nPicoUNet total parameters: {total_params:,}")
    print(f"PicoUNet trainable parameters: {trainable_params:,}")

    # Model must contain trainable parameters.
    assert total_params > 0
    assert trainable_params > 0

    # PicoUNet is expected to be around 2K parameters.
    # Keep the limit tolerant enough for small architectural changes.
    assert total_params < 5_000


def test_pico_unet_output_shape_cpu():
    """Test PicoUNet output shape for a standard input."""

    model = PicoUNet(
        in_channels=1,
        out_channels=3,
    ).cpu()

    model.eval()

    x = torch.randn(
        2,
        1,
        64,
        64,
    )

    with torch.no_grad():
        output = model(x)

    expected_shape = (2, 3, 64, 64)

    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(output.shape)}"
    )
