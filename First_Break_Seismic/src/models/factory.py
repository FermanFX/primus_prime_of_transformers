"""
Model factory for creating model instances.
Centralizes model creation to avoid duplication across scripts.
"""
import time

import torch
from torch import nn

from src.models.efficient_unet import EfficientUNet
from src.models.light_unet import LightUNet, NanoUNetLight
from src.models.mobilenet import MobileUNet
from src.models.mps_light_unet import MPSLightUNet
from src.models.nano_unet import NanoUNet as UltraNanoUNet
from src.models.pico_unet import PicoUNet
from src.models.tiny_unet import TinyUNet
from src.models.unet import UNet

# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {
    "unet": UNet,
    "mpslight": MPSLightUNet,
    "light": LightUNet,
    "nano": NanoUNetLight,
    "ultranano": UltraNanoUNet,
    "tiny": TinyUNet,
    "pico": PicoUNet,
    "mobile": MobileUNet,
    "efficient": EfficientUNet,
}


# ============================================================
# MODEL CARD CONFIGURATION
# ============================================================

MODEL_CARD_RECOMMENDATIONS = {
    "unet": {
        "use_cases": ["high-quality segmentation", "accuracy-first training"],
        "training": {
            "batch_size": "Use the largest batch size that fits available memory",
            "learning_rate": "Start with 1e-3 and tune based on validation loss",
        },
    },
    "mpslight": {
        "use_cases": ["Apple Silicon", "memory-constrained segmentation"],
        "training": {
            "batch_size": "Start with a small-to-medium batch size",
            "learning_rate": "Start with 1e-3",
        },
    },
    "light": {
        "use_cases": ["balanced accuracy and efficiency", "general segmentation"],
        "training": {
            "batch_size": "Use a medium batch size when memory permits",
            "learning_rate": "Start with 1e-3",
        },
    },
    "nano": {
        "use_cases": ["fast inference", "resource-constrained deployment"],
        "training": {
            "batch_size": "Use a larger batch size when memory permits",
            "learning_rate": "Start with 1e-3",
        },
    },
    "ultranano": {
        "use_cases": ["extremely lightweight inference", "edge deployment"],
        "training": {
            "batch_size": "Use a larger batch size when memory permits",
            "learning_rate": "Start with 1e-3",
        },
    },
    "tiny": {
        "use_cases": ["lightweight inference", "edge or CPU deployment"],
        "training": {
            "batch_size": "Use the largest feasible batch size",
            "learning_rate": "Start with 1e-3",
        },
    },
    "pico": {
        "use_cases": ["minimal-resource deployment", "very fast inference"],
        "training": {
            "batch_size": "Use a large batch size when memory permits",
            "learning_rate": "Start with 1e-3",
        },
    },
    "mobile": {
        "use_cases": ["mobile deployment", "fast inference"],
        "training": {
            "batch_size": "Use the largest batch size supported by the training device",
            "learning_rate": "Start with 1e-3",
        },
    },
    "efficient": {
        "use_cases": ["efficient segmentation", "production inference"],
        "training": {
            "batch_size": "Use a medium-to-large batch size when possible",
            "learning_rate": "Start with 1e-3",
        },
    },
}


# ============================================================
# FACTORY FUNCTION
# ============================================================


def create_model(
    model_name: str, in_channels: int = 1, out_channels: int = 3, **kwargs
) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        model_name: Name of the model (e.g., "mpslight", "unet")
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 3)
        **kwargs: Additional arguments to pass to the model

    Returns:
        nn.Module: The instantiated model

    Raises:
        ValueError: If model_name is not in the registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]
    return model_class(in_channels=in_channels, out_channels=out_channels, **kwargs)


def list_models() -> list:
    """List all available model names."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> dict:
    """
    Get information about a model.

    Args:
        model_name: Name of the model

    Returns:
        dict: Information about the model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")

    model_class = MODEL_REGISTRY[model_name]

    try:
        temp_model = model_class(in_channels=1, out_channels=3)
        params = sum(p.numel() for p in temp_model.parameters())
        del temp_model
    except Exception:  # noqa BLE001
        params = None
        print("Cannot get parameter count")

    return {
        "name": model_name,
        "class": model_class.__name__,
        "params": params,
    }


def get_model_card(
    model_name: str,
    in_channels: int = 1,
    out_channels: int = 3,
    input_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
    benchmark_runs: int = 20,
) -> dict:
    """
    Generate a model card with architecture, size, speed, use cases,
    and training recommendations.

    Args:
        model_name: Name of the model.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        input_size: Spatial input size used for the speed benchmark.
        device: Device used for the speed benchmark.
        benchmark_runs: Number of timed inference runs.

    Returns:
        dict: Model card information.

    Raises:
        ValueError: If model_name is not in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")

    model = create_model(
        model_name,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    model = model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Parameter memory assumes FP32 weights.
    parameter_memory_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024**2)

    # Benchmark inference speed using the requested input dimensions.
    speed_ms = None
    try:
        dummy_input = torch.randn(
            1,
            in_channels,
            input_size[0],
            input_size[1],
            device=device,
        )

        with torch.inference_mode():
            # Warm-up runs.
            for _ in range(5):
                model(dummy_input)

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            start = time.perf_counter()

            for _ in range(benchmark_runs):
                model(dummy_input)

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            speed_ms = (elapsed / benchmark_runs) * 1000

    except Exception:  # noqa BLE001
        print("Cannot benchmark inference speed")

    recommendations = MODEL_CARD_RECOMMENDATIONS.get(
        model_name,
        {
            "use_cases": ["general segmentation"],
            "training": {
                "batch_size": "Choose the largest batch size supported by available memory",
                "learning_rate": "Start with 1e-3 and tune using validation performance",
            },
        },
    )

    del model

    return {
        "name": model_name,
        "class": MODEL_REGISTRY[model_name].__name__,
        "parameters": {
            "total": params,
            "trainable": trainable_params,
        },
        "memory": {
            "parameter_memory_mb_fp32": round(parameter_memory_mb, 2),
        },
        "speed": {
            "device": device,
            "input_size": input_size,
            "inference_ms": round(speed_ms, 3) if speed_ms is not None else None,
            "inference_fps": (
                round(1000 / speed_ms, 2)
                if speed_ms is not None and speed_ms > 0
                else None
            ),
            "benchmark_runs": benchmark_runs,
        },
        "use_cases": recommendations["use_cases"],
        "training_configuration": recommendations["training"],
    }
