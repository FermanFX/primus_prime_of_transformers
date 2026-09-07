"""Tests for TinyUNet."""

import pytest
import torch

from src.models.tiny_unet import TinyUNet


@pytest.fixture
def model():
    """Create the default TinyUNet."""
    return TinyUNet(
        in_channels=1,
        out_channels=3,
    )


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestTinyUNetForward:
    """Forward-pass tests."""

    def test_forward_pass_on_cpu(self, model):
        """TinyUNet should successfully run on CPU."""
        model.eval()

        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 64, 64)

    def test_output_shape_for_seismic_input(self, model):
        """Output should preserve the input spatial dimensions."""
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
    def test_padding_for_non_divisible_dimensions(
        self,
        model,
        height,
        width,
    ):
        """Padding should be removed before returning the output."""
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


class TestTinyUNetParameters:
    """Parameter-count tests."""

    def test_parameter_count_is_positive(self, model):
        """TinyUNet should have trainable parameters."""
        params = count_parameters(model)

        assert params > 0

    def test_parameter_count_is_approximately_50k(self, model):
        """TinyUNet should have a very low parameter count."""
        params = count_parameters(model)

        # The implementation documents approximately 50K parameters.
        assert 30_000 <= params <= 60_000

    def test_parameter_count_is_much_smaller_than_standard_unet(self, model):
        """TinyUNet should remain significantly smaller than a standard U-Net."""
        from src.models.unet import UNet

        tiny_params = count_parameters(model)
        unet_params = count_parameters(UNet())

        assert tiny_params < unet_params
        assert tiny_params < unet_params / 10


class TestTinyUNetConfiguration:
    """Configuration tests."""

    def test_custom_input_and_output_channels(self):
        """TinyUNet should support custom channel counts."""
        model = TinyUNet(
            in_channels=2,
            out_channels=4,
        )
        model.eval()

        x = torch.randn(1, 2, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 4, 64, 64)

    def test_output_channels_match_configuration(self):
        """Output channel count should match out_channels."""
        model = TinyUNet(
            in_channels=1,
            out_channels=5,
        )
        model.eval()

        x = torch.randn(1, 1, 32, 32)

        with torch.no_grad():
            output = model(x)

        assert output.shape[1] == 5
