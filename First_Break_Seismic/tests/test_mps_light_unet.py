import pytest
import torch

from src.models.mps_light_unet import MPSLightUNet
from src.models.unet import UNet


@pytest.fixture
def model():
    """Create MPSLightUNet in evaluation mode."""
    model = MPSLightUNet(in_channels=1, out_channels=3)
    model.eval()
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestMPSLightUNetCPU:
    def test_forward_pass_on_cpu(self, model):
        """MPSLightUNet should successfully run on CPU."""
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 64, 64)
        assert output.device.type == "cpu"
        assert torch.isfinite(output).all()

    def test_output_shape_for_seismic_input(self, model):
        """Output should preserve the seismic input spatial dimensions."""
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
            (157, 151),
        ],
    )
    def test_padding_for_non_divisible_dimensions(
        self,
        model,
        height,
        width,
    ):
        """Non-16-divisible inputs should be padded and cropped correctly."""
        x = torch.randn(1, 1, height, width)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, height, width)
        assert torch.isfinite(output).all()

    def test_output_is_contiguous(self, model):
        """Output should be contiguous for MPS compatibility."""
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.is_contiguous()


class TestMPSLightUNetParameters:
    def test_parameter_count_is_approximately_1_7_million(self, model):
        """MPSLightUNet should have approximately 1.7M parameters."""
        params = count_parameters(model)
        # Allow a reasonable tolerance around the documented ~1.7M.
        assert 1_500_000 <= params <= 2_000_000

    def test_parameter_count_is_lower_than_unet(self):
        """MPSLightUNet should have fewer parameters than standard UNet."""
        light_model = MPSLightUNet(in_channels=1, out_channels=3)
        standard_model = UNet(in_channels=1, out_channels=3)

        light_params = count_parameters(light_model)
        standard_params = count_parameters(standard_model)

        assert light_params < standard_params


class TestMPSLightUNetMPS:
    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS is not available",
    )
    def test_forward_pass_on_mps(self, model):
        """MPSLightUNet should successfully run on Apple Silicon MPS."""
        device = torch.device("mps")
        model = model.to(device)

        x = torch.randn(
            1,
            1,
            64,
            64,
            device=device,
        )

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 64, 64)
        assert output.device.type == "mps"
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS is not available",
    )
    def test_forward_pass_non_divisible_shape_on_mps(self, model):
        """MPS should correctly handle padding and cropping."""
        device = torch.device("mps")
        model = model.to(device)

        x = torch.randn(
            1,
            1,
            65,
            73,
            device=device,
        )

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 65, 73)
        assert output.device.type == "mps"


class TestMPSLightUNetMemory:
    def test_parameter_memory_is_lower_than_standard_unet(self):
        """
        MPSLightUNet should require less parameter memory than UNet.

        This compares model parameter memory, which is deterministic and
        portable across CPU and MPS. Runtime activation memory can vary
        significantly by backend and PyTorch version.
        """
        light_model = MPSLightUNet(in_channels=1, out_channels=3)
        standard_model = UNet(in_channels=1, out_channels=3)

        light_memory = sum(
            p.numel() * p.element_size() for p in light_model.parameters()
        )
        standard_memory = sum(
            p.numel() * p.element_size() for p in standard_model.parameters()
        )

        assert light_memory < standard_memory

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS is not available",
    )
    def test_mps_memory_usage_is_reasonable(self, model):
        """
        Verify that the MPS model can execute without excessive allocation.

        This is intentionally a smoke test rather than an exact byte-level
        comparison because MPS allocator behavior is backend-dependent.
        """
        device = torch.device("mps")
        model = model.to(device)

        torch.mps.empty_cache()

        x = torch.randn(
            1,
            1,
            64,
            64,
            device=device,
        )

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 64, 64)

        torch.mps.synchronize()
