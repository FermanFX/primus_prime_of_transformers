import pytest
import torch

from src.models.unet import UNet


@pytest.fixture
def model():
    """Create UNet in evaluation mode for inference tests."""
    model = UNet(in_channels=1, out_channels=3)
    model.eval()
    return model


class TestUNetForward:
    def test_forward_pass_completes_without_errors(self, model):
        """UNet should successfully process a valid input tensor."""
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output is not None

    @pytest.mark.slow
    def test_forward_pass_with_seismic_input_shape(self, model):
        """UNet should process the expected seismic input shape."""
        x = torch.randn(1, 1, 1578, 751)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 1578, 751)

    def test_output_shape_matches_input_spatial_dimensions(self, model):
        """Output should preserve input height and width."""
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 64, 64)

    @pytest.mark.parametrize(
        "height,width",
        [
            (17, 17),
            (31, 47),
            (33, 35),
            (65, 73),
            (157, 151),
        ],
    )
    def test_padding_handling_for_non_divisible_sizes(
        self,
        model,
        height,
        width,
    ):
        """
        Input dimensions that are not divisible by 16 should be
        padded internally and cropped back to the original size.
        """
        assert height % 16 != 0
        assert width % 16 != 0

        x = torch.randn(1, 1, height, width)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, height, width)

    def test_output_values_are_finite(self, model):
        """
        UNet returns raw logits, so values are not expected to be
        restricted to [0, 1]. They should, however, be finite.
        """
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert torch.isfinite(output).all()

    def test_output_has_three_classes(self, model):
        """Default UNet should produce three output channels."""
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape[1] == 3

    def test_output_is_contiguous(self, model):
        """Forward pass should return a contiguous tensor for MPS."""
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.is_contiguous()

    def test_custom_channel_configuration(self):
        """UNet should support custom input and output channel counts."""
        model = UNet(in_channels=2, out_channels=4)
        model.eval()

        x = torch.randn(1, 2, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 4, 64, 64)
        assert torch.isfinite(output).all()
