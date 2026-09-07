"""Tests for MobileNetV2 + U-Net decoder."""

import pytest
import torch
from torchvision.models import MobileNetV2

from src.models.mobilenet import MobileUNet


@pytest.fixture
def model():
    """Create MobileUNet without downloading pretrained weights."""
    return MobileUNet(
        in_channels=1,
        out_channels=3,
        pretrained=False,
    )


class TestMobileUNetForward:
    """Forward-pass tests."""

    def test_forward_pass_on_cpu(self, model):
        """MobileUNet should successfully run on CPU."""
        model.eval()

        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 64, 64)

    def test_output_shape_for_seismic_input(self, model):
        """Output should preserve the spatial dimensions."""
        model.eval()

        x = torch.randn(1, 1, 1578, 751)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 1578, 751)

    @pytest.mark.parametrize(
        "height,width",
        [
            (17, 17),
            (31, 47),
            (33, 35),
            (65, 73),
        ],
    )
    def test_padding_for_non_divisible_dimensions(self, model, height, width):
        """Padding should be removed so output matches original dimensions."""
        model.eval()

        x = torch.randn(1, 1, height, width)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, height, width)

    def test_output_is_contiguous(self, model):
        """Output should be contiguous."""
        model.eval()

        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.is_contiguous()


class TestMobileUNetParameters:
    """Parameter-count tests."""

    def test_parameter_count_is_positive(self, model):
        """Model should contain trainable parameters."""
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert params > 0

    def test_parameter_count_is_reasonable(self, model):
        """MobileUNet should have a reasonable parameter count."""
        params = sum(p.numel() for p in model.parameters())

        # MobileNetV2 itself is ~3.5M parameters.
        # Decoder + stem add additional parameters.
        assert 3_000_000 <= params <= 6_000_000


class TestMobileUNetEncoder:
    """MobileNetV2 encoder tests."""

    def test_encoder_is_mobilenet_v2(self, model):
        """Encoder should be a MobileNetV2 model."""
        assert isinstance(model.encoder, MobileNetV2)

    def test_encoder_has_mobilenet_features(self, model):
        """Encoder should expose MobileNetV2 feature blocks."""
        assert hasattr(model.encoder, "features")
        assert len(model.encoder.features) == 19

    def test_encoder_parameters_are_frozen(self, model):
        """MobileNetV2 encoder parameters should be frozen."""
        assert all(not param.requires_grad for param in model.encoder.parameters())

    def test_encoder_feature_blocks_are_defined(self, model):
        """All expected encoder stages should exist."""
        assert model.enc1 is not None
        assert model.enc2 is not None
        assert model.enc3 is not None
        assert model.enc4 is not None
        assert model.enc5 is not None


class TestMobileUNetConfiguration:
    """Configuration tests."""

    def test_custom_channels(self):
        """Model should support custom input/output channels."""
        model = MobileUNet(
            in_channels=2,
            out_channels=4,
            pretrained=False,
        )
        model.eval()

        x = torch.randn(1, 2, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 4, 64, 64)

    def test_output_channels(self, model):
        """Output channel count should match out_channels."""
        model.eval()

        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape[1] == 3
