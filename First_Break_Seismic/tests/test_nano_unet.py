import torch

from src.models.nano_unet import NanoUNet


def test_nano_unet_forward():
    """Test NanoUNet forward pass and output shape."""

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = NanoUNet(
        in_channels=1,
        out_channels=3,
    ).to(device)

    # Intentionally use non-multiple-of-8 dimensions
    # to test padding and cropping logic.
    x = torch.randn(
        2,
        1,
        127,
        131,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        output = model(x)

    expected_shape = (2, 3, 127, 131)

    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(output.shape)}"
    )

    assert torch.isfinite(output).all(), "Model output contains NaN or Inf values"


def test_nano_unet_parameters():
    """Test that NanoUNet has a lightweight parameter count."""

    model = NanoUNet(
        in_channels=1,
        out_channels=3,
    )

    total_params = sum(parameter.numel() for parameter in model.parameters())

    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(f"\nNanoUNet parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    assert total_params > 0
    assert trainable_params > 0

    # NanoUNet should remain lightweight.
    # Current architecture is expected to be well below 100K parameters.
    assert total_params < 100_000


def test_nano_unet_backward():
    """Test loss calculation and gradient propagation."""

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = NanoUNet(
        in_channels=1,
        out_channels=3,
    ).to(device)

    model.train()

    x = torch.randn(
        2,
        1,
        127,
        131,
        device=device,
    )

    target = torch.randint(
        0,
        3,
        (2, 127, 131),
        device=device,
    )

    output = model(x)

    loss = torch.nn.functional.cross_entropy(
        output,
        target,
    )

    print(f"\nNanoUNet test loss: {loss.item():.6f}")

    assert torch.isfinite(loss), "Loss is NaN or Inf"

    loss.backward()

    gradients_found = False

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        assert parameter.grad is not None, f"No gradient for parameter: {name}"

        assert torch.isfinite(parameter.grad).all(), (
            f"Invalid gradient for parameter: {name}"
        )

        gradients_found = True

    assert gradients_found, "No trainable parameter received gradients"


def test_nano_unet_different_input_sizes():
    """Test that NanoUNet preserves arbitrary input dimensions."""

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = NanoUNet(
        in_channels=1,
        out_channels=3,
    ).to(device)

    model.eval()

    test_sizes = [
        (64, 64),
        (128, 128),
        (127, 131),
        (65, 73),
        (100, 150),
    ]

    with torch.no_grad():
        for height, width in test_sizes:
            x = torch.randn(
                1,
                1,
                height,
                width,
                device=device,
            )

            output = model(x)

            expected_shape = (
                1,
                3,
                height,
                width,
            )

            assert output.shape == expected_shape, (
                f"Input {(height, width)}: "
                f"expected {expected_shape}, "
                f"got {tuple(output.shape)}"
            )

            assert torch.isfinite(output).all(), (
                f"Input {(height, width)} produced NaN or Inf values"
            )
