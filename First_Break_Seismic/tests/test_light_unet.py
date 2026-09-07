import pytest
import torch
from torch import nn

from src.models.light_unet import DepthwiseSeparableConv, LightUNet, NanoUNetLight


@pytest.fixture
def model():
    """Create a LightUNet model in evaluation mode."""
    model = LightUNet(in_channels=1, out_channels=3)
    model.eval()
    return model


@pytest.fixture
def train_model():
    """Create a LightUNet model in training mode."""
    return LightUNet(in_channels=1, out_channels=3)


@pytest.fixture
def nano_model():
    """Create a NanoUNetLight model in evaluation mode."""
    model = NanoUNetLight(in_channels=1, out_channels=3)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


class TestLightUNetInitialization:
    def test_default_configuration(self):
        """Default configuration should be valid."""
        model = LightUNet()

        assert isinstance(model, nn.Module)
        assert model.depth == 4

    def test_custom_input_channels(self):
        """Model should support custom input channel counts."""
        model = LightUNet(in_channels=2)

        first_conv = model.enc1[0]

        assert first_conv.in_channels == 2

    def test_custom_output_channels(self):
        """Model should support custom output channel counts."""
        model = LightUNet(out_channels=5)

        assert model.out_conv.out_channels == 5

    def test_custom_base_channels(self):
        """Model should support custom base channel width."""
        model = LightUNet(base_channels=8)

        assert model.enc1[0].out_channels == 8
        assert model.enc2[0].out_channels == 16
        assert model.enc3[0].out_channels == 32
        assert model.enc4[0].out_channels == 64
        assert model.bottleneck[0].out_channels == 128


class TestLightUNetForward:
    def test_forward_pass_on_cpu(self, model):
        """Forward pass should complete successfully on CPU."""
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 128, 128)
        assert output.device.type == "cpu"
        assert torch.isfinite(output).all()

    def test_standard_seismic_input_shape(self, model):
        """Model should support seismic input dimensions."""
        x = torch.randn(1, 1, 1578, 751)

        with torch.no_grad():
            output = model(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 3
        assert output.shape[2:] == (1578, 751)

    @pytest.mark.parametrize(
        "height,width,expected_shape",
        [
            (16, 16, (32, 32)),
            (17, 17, (17, 17)),
            (31, 47, (31, 47)),
            (32, 32, (64, 64)),
            (33, 35, (33, 35)),
            (63, 65, (63, 65)),
            (65, 73, (65, 73)),
            (127, 129, (127, 129)),
            (1578, 751, (1578, 751)),
        ],
    )
    def test_dynamic_input_dimensions(self, model, height, width, expected_shape):
        """Model should support arbitrary spatial height and width scaling."""
        x = torch.randn(1, 1, height, width)

        with torch.no_grad():
            output = model(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 3
        assert output.shape[2:] == expected_shape

    def test_single_pixel_spatial_dimensions(self, model):
        """Very small inputs should be handled correctly."""
        x = torch.randn(1, 1, 1, 1)

        with torch.no_grad():
            output = model(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 3

    def test_batch_size_two(self, model):
        """Model should support multiple samples per batch."""
        x = torch.randn(2, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 3, 128, 128)

    def test_larger_batch(self, model):
        """Model should preserve larger batch dimensions."""
        x = torch.randn(4, 1, 32, 32)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 3, 64, 64)

    def test_output_is_contiguous(self, model):
        """Output should be contiguous for downstream MPS operations."""
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.is_contiguous()

    def test_output_dtype_matches_input(self, model):
        """Output should preserve the input floating-point dtype."""
        x = torch.randn(1, 1, 64, 64, dtype=torch.float32)

        with torch.no_grad():
            output = model(x)

        assert output.dtype == x.dtype

    def test_output_contains_finite_values(self, model):
        """Forward pass should not produce NaN or Inf values."""
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert torch.isfinite(output).all()

    def test_zero_input_does_not_produce_nan(self, model):
        """Zero input should produce finite output."""
        x = torch.zeros(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert torch.isfinite(output).all()

    def test_negative_input_does_not_produce_nan(self, model):
        """Negative seismic amplitudes should be handled safely."""
        x = torch.full((1, 1, 64, 64), -1.0)

        with torch.no_grad():
            output = model(x)

        assert torch.isfinite(output).all()


class TestLightUNetTraining:
    def test_training_mode(self, train_model):
        """Model should support training mode."""
        train_model.train()

        assert train_model.training
        assert train_model.enc1[1].training

    def test_eval_mode(self, train_model):
        """Model should support evaluation mode."""
        train_model.eval()

        assert not train_model.training
        assert not train_model.enc1[1].training

    def test_backward_pass(self, train_model):
        """Model should support gradient computation."""
        x = torch.randn(1, 1, 32, 32, requires_grad=True)

        output = train_model(x)
        loss = output.mean()

        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_model_parameters_receive_gradients(self, train_model):
        """Trainable parameters should receive gradients."""
        x = torch.randn(1, 1, 32, 32)

        output = train_model(x)
        loss = output.mean()
        loss.backward()

        parameters_with_grad = [
            parameter
            for parameter in train_model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]

        assert len(parameters_with_grad) > 0

    def test_optimizer_step(self, train_model):
        """Model parameters should update after an optimizer step."""
        optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-3)

        x = torch.randn(1, 1, 32, 32)

        before = {
            name: parameter.detach().clone()
            for name, parameter in train_model.named_parameters()
        }

        output = train_model(x)
        loss = output.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = [
            not torch.equal(before[name], parameter.detach())
            for name, parameter in train_model.named_parameters()
        ]

        assert any(changed)


class TestLightUNetParameters:
    def test_parameter_count_is_positive(self, model):
        """Model should contain trainable parameters."""
        params = count_parameters(model)

        assert params > 0

    def test_parameter_count_is_lightweight(self, model):
        """LightUNet should remain lightweight."""
        params = count_parameters(model)

        assert 500_000 <= params <= 1_000_000

    def test_all_parameters_are_finite(self, model):
        """Model parameters should be initialized with finite values."""
        for parameter in model.parameters():
            assert torch.isfinite(parameter).all()

    def test_model_has_expected_output_head(self, model):
        """Output head should map base channels to three classes."""
        assert isinstance(model.out_conv, nn.Conv2d)
        assert model.out_conv.in_channels == 16
        assert model.out_conv.out_channels == 3
        assert model.out_conv.kernel_size == (1, 1)


class TestDepthwiseSeparableConv:
    def test_depthwise_groups_equal_input_channels(self):
        """Depthwise convolution should use one group per input channel."""
        block = DepthwiseSeparableConv(
            in_channels=16,
            out_channels=32,
        )

        assert block.depthwise.groups == 16
        assert block.depthwise.in_channels == 16
        assert block.depthwise.out_channels == 16

    def test_depthwise_kernel_is_3x3(self):
        """Depthwise convolution should use a 3x3 kernel by default."""
        block = DepthwiseSeparableConv(
            in_channels=16,
            out_channels=32,
        )

        assert block.depthwise.kernel_size == (3, 3)
        assert block.depthwise.padding == (1, 1)

    def test_pointwise_kernel_is_1x1(self):
        """Pointwise convolution should use a 1x1 kernel."""
        block = DepthwiseSeparableConv(
            in_channels=16,
            out_channels=32,
        )

        assert block.pointwise.kernel_size == (1, 1)

    def test_pointwise_maps_to_output_channels(self):
        """Pointwise convolution should produce requested output channels."""
        block = DepthwiseSeparableConv(
            in_channels=16,
            out_channels=32,
        )

        assert block.pointwise.in_channels == 16
        assert block.pointwise.out_channels == 32

    def test_has_batch_normalization(self):
        """Depthwise separable block should contain BatchNorm."""
        block = DepthwiseSeparableConv(
            in_channels=16,
            out_channels=32,
        )

        assert isinstance(block.bn, nn.BatchNorm2d)
        assert block.bn.num_features == 32

    def test_has_relu_activation(self):
        """Depthwise separable block should contain ReLU."""
        block = DepthwiseSeparableConv(
            in_channels=16,
            out_channels=32,
        )

        assert isinstance(block.relu, nn.ReLU)

    def test_forward_preserves_spatial_dimensions(self):
        """Depthwise separable convolution should preserve H/W."""
        block = DepthwiseSeparableConv(
            in_channels=16,
            out_channels=32,
        )
        block.eval()

        x = torch.randn(2, 16, 32, 32)

        with torch.no_grad():
            output = block(x)

        assert output.shape == (2, 32, 32, 32)

    def test_forward_supports_different_channel_counts(self):
        """Block should support arbitrary valid channel configurations."""
        block = DepthwiseSeparableConv(
            in_channels=8,
            out_channels=24,
        )
        block.eval()

        x = torch.randn(1, 8, 16, 20)

        with torch.no_grad():
            output = block(x)

        assert output.shape == (1, 24, 16, 20)

    def test_output_is_finite(self):
        """Depthwise separable block should not produce NaN or Inf."""
        block = DepthwiseSeparableConv(
            in_channels=8,
            out_channels=16,
        )
        block.eval()

        x = torch.randn(1, 8, 16, 16)

        with torch.no_grad():
            output = block(x)

        assert torch.isfinite(output).all()


class TestLightUNetConfiguration:
    def test_custom_channel_configuration(self):
        """Custom input/output channels should work."""
        model = LightUNet(
            in_channels=2,
            out_channels=4,
            base_channels=8,
            depth=4,
        )
        model.eval()

        x = torch.randn(1, 2, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 4, 128, 128)

    def test_custom_configuration_non_divisible_shape(self):
        """Custom configuration should handle padding correctly."""
        model = LightUNet(
            in_channels=2,
            out_channels=4,
            base_channels=8,
            depth=4,
        )
        model.eval()

        x = torch.randn(2, 2, 37, 53)

        with torch.no_grad():
            output = model(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 4
        assert output.shape[2:] == (37, 53)

    def test_depth_is_stored(self):
        """Configured depth should be stored on the model."""
        model = LightUNet(depth=4)

        assert model.depth == 4


class TestNanoUNetLight:
    def test_nano_unet_initialization(self):
        """NanoUNetLight initialization defaults."""
        model = NanoUNetLight()

        assert isinstance(model, nn.Module)
        assert model.out_conv.out_channels == 3

    def test_nano_unet_custom_channels(self):
        """NanoUNetLight custom channel counts."""
        model = NanoUNetLight(in_channels=2, out_channels=5)

        assert model.enc1[0].in_channels == 2
        assert model.out_conv.out_channels == 5

    def test_nano_unet_forward_even_input(self, nano_model):
        """Forward pass with 16-divisible input shape."""
        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = nano_model(x)

        assert output.shape == (1, 3, 128, 128)
        assert output.is_contiguous()
        assert torch.isfinite(output).all()

    def test_nano_unet_forward_odd_input(self, nano_model):
        """Forward pass with input requiring padding and cropping."""
        x = torch.randn(2, 1, 37, 53)

        with torch.no_grad():
            output = nano_model(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 3
        assert output.shape[2:] == (37, 53)
        assert output.is_contiguous()
        assert torch.isfinite(output).all()

    def test_nano_unet_parameter_count(self, nano_model):
        """NanoUNetLight parameter count verification."""
        params = count_parameters(nano_model)

        assert params > 0
        assert params < 1_000_000
