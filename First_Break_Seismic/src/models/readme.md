# Model Size Comparison

This repository contains multiple U-Net-based models designed for seismic segmentation and first-break picking. The models range from an ultra-minimal `PicoUNet` for instant testing to encoder-decoder architectures such as `UNet`, `MobileUNet`, and `EfficientUNet`.

## Model Comparison

| Model | Architecture | Parameters | Memory | Speed / Training Time | Main Use Case |
|---|---|---:|---:|---|---|
| **PicoUNet** | Minimal U-Net | ~2K | Very Low | ~10 sec/epoch | Instant testing |
| **NanoUNet** | Tiny U-Net | ~10K | Very Low | ~30 sec/epoch | Ultra-fast testing |
| **TinyUNet** | Small U-Net | ~50K | Low | ~1 min/epoch | Quick testing |
| **NanoUNetLight** | Lightweight U-Net | ~0.8M | ~50 MB | ~4× faster than original U-Net | Lightweight segmentation |
| **MPSLightUNet** | MPS-optimized U-Net | ~1.7M | ~100 MB | ~2× faster than original U-Net | Apple Silicon / MPS |
| **LightUNet** | Depthwise-separable U-Net | ~2.5M | ~150 MB | ~2× faster than original U-Net | General lightweight segmentation |
| **MobileUNet** | MobileNetV2 + U-Net decoder | TBD | TBD | TBD | Lightweight pretrained encoder |
| **EfficientUNet** | EfficientNet-B0 + U-Net decoder | TBD | TBD | TBD | Pretrained encoder / segmentation |
| **UNet** | Standard U-Net | TBD | TBD | Baseline | Full-capacity segmentation |

> **Note:** Memory and speed values are the values documented in the model implementations. `TBD` values should be replaced with benchmark results when measured on the target hardware.

## Model Details

### PicoUNet

`PicoUNet` is the smallest model in the collection and is intended for instant testing.

- Parameters: ~2K
- Training time: ~10 seconds per epoch
- Encoder channels: `1 → 2 → 4`
- Bottleneck: `4` channels
- Output: 3 classes
- Designed for extremely fast experiments and pipeline validation.

### NanoUNet

`NanoUNet` is a slightly larger minimal U-Net suitable for quick experiments.

- Parameters: ~10K
- Training time: ~30 seconds per epoch
- Encoder channels: `2 → 4 → 8`
- Bottleneck: `16` channels
- Output: 3 classes
- Designed for rapid testing while retaining the basic U-Net structure.

### TinyUNet

`TinyUNet` provides a larger capacity than `NanoUNet` while remaining lightweight.

- Parameters: ~50K
- Training time: ~1 minute per epoch
- Encoder channels: `4 → 8 → 16`
- Bottleneck: `32` channels
- Output: 3 classes
- Suitable for quick training and debugging.

### NanoUNetLight

`NanoUNetLight` is an ultra-lightweight U-Net using depthwise-separable convolutions.

- Parameters: ~0.8M
- Memory: ~50 MB
- Speed: ~4× faster than the original U-Net
- Uses depthwise + pointwise convolutions.
- Four encoder/decoder levels.
- Designed for low-resource segmentation workloads.

### MPSLightUNet

`MPSLightUNet` is specifically optimized for Apple Silicon and MPS memory constraints.

- Parameters: ~1.7M
- Memory: ~100 MB
- Speed: ~2× faster than the original U-Net
- Encoder channels: `16 → 32 → 64 → 128`
- Bottleneck: `256`
- Uses standard convolution blocks.
- Designed for Apple Silicon systems using the PyTorch MPS backend.

### LightUNet

`LightUNet` is a lightweight U-Net that reduces computation using depthwise-separable convolutions.

- Parameters: ~2.5M
- Memory: ~150 MB
- Speed: ~2× faster than the original U-Net
- Uses depthwise-separable convolutions in encoder and decoder blocks.
- Configurable `base_channels` and `depth`.
- Default configuration uses `base_channels=16` and `depth=4`.

### MobileUNet

`MobileUNet` combines a pretrained MobileNetV2 encoder with a U-Net-style decoder.

- Encoder: MobileNetV2
- Decoder: U-Net-style decoder
- Uses ImageNet-pretrained weights by default.
- Encoder weights are frozen by default.
- Single-channel seismic input is expanded to three channels before entering MobileNetV2.
- Output: 3-class segmentation mask.

This architecture is intended to provide a strong balance between pretrained feature extraction and lightweight segmentation.

### EfficientUNet

`EfficientUNet` combines an EfficientNet-B0 encoder with a U-Net decoder.

- Encoder: EfficientNet-B0
- Decoder: U-Net-style decoder
- Uses ImageNet-pretrained weights by default.
- Supports single-channel seismic input through a 3-channel stem.
- Uses skip connections between encoder and decoder stages.
- Output: 3-class segmentation mask.

EfficientNet provides a strong feature extractor while maintaining a relatively compact architecture compared with a standard large U-Net.

### UNet

`UNet` is the standard full-capacity U-Net baseline used for comparison.

- Encoder channels: `32 → 64 → 128 → 256`
- Bottleneck: `512`
- Decoder channels: `256 → 128 → 64 → 32`
- Output: 3 classes
- Input: single-channel seismic data
- Designed as the primary baseline for segmentation quality and performance comparisons.

## Size and Performance Ranking

From smallest to largest based on the documented parameter counts:

1. **PicoUNet** — ~2K parameters
2. **NanoUNet** — ~10K parameters
3. **TinyUNet** — ~50K parameters
4. **NanoUNetLight** — ~0.8M parameters
5. **MPSLightUNet** — ~1.7M parameters
6. **LightUNet** — ~2.5M parameters
7. **MobileUNet** — TBD
8. **EfficientUNet** — TBD
9. **UNet** — TBD

## Choosing a Model

| Requirement | Recommended Model |
|---|---|
| Fastest possible testing | `PicoUNet` |
| Very fast experiments | `NanoUNet` |
| Quick testing with more capacity | `TinyUNet` |
| Very low memory usage | `NanoUNetLight` |
| Apple Silicon / MPS | `MPSLightUNet` |
| Lightweight general-purpose model | `LightUNet` |
| Lightweight pretrained encoder | `MobileUNet` |
| Strong pretrained encoder | `EfficientUNet` |
| Baseline / maximum capacity | `UNet` |

## Benchmarking

The documented memory and speed values are approximate and hardware-dependent. For a reliable comparison, all models should be benchmarked using the same:

- Input resolution
- Batch size
- Device
- PyTorch version
- Precision (`FP32`, `FP16`, etc.)
- Number of training iterations
- Dataset and preprocessing pipeline

Recommended metrics include:

- Total trainable parameters
- Peak GPU/MPS memory
- Training time per epoch
- Inference time per batch
- Throughput (samples/sec)
- Validation IoU / Dice score

This allows the lightweight models to be compared fairly against the standard `UNet`, `MobileUNet`, and `EfficientUNet` architectures.

# Models Directory (`First_Break_Seismic/src/models/`)

This directory houses the neural network architectures, custom variants, and factory loaders designed for seismic trace analysis and first-break picking pipelines. It contains various scales of U-Net models optimized for different resource constraints and performance requirements.

## Model Files Overview

* **`__init__.py`**: Marks the directory as a Python package and manages module-level namespace exports for network architectures.
* **`efficient_unet.py`**: Implements an optimized variant of the U-Net architecture focused on computational efficiency and reduced parameter counts.
* **`factory.py`**: Provides a centralized factory class or function to instantiate different model architectures dynamically based on configuration files.
* **`light_unet.py`**: Contains a lightweight U-Net model configuration designed for faster training iterations and reduced memory footprints.
* **`mobilenet.py`**: Integrates MobileNet-based building blocks or encoder backbones for mobile-friendly or resource-constrained environments.
* **`mps_light_unet.py`**: Implements a light U-Net variant optimized specifically for Apple Silicon (MPS - Metal Performance Shaders) hardware acceleration.
* **`nano_unet.py`**: Features an ultra-compact nano-scale U-Net architecture built for rapid prototyping and extreme hardware limitations.
* **`pico_unet.py`**: Contains a pico-scale U-Net variant optimized for minimal memory consumption and rapid inference.
* **`readme.md`**: Provides architectural documentation outlining the structural design of the models directory and its internal network variants.
* **`tiny_unet.py`**: Implements a reduced-capacity tiny U-Net model suitable for small-scale seismic datasets and fast testing.
* **`unet.py`**: Implements the standard, full-scale U-Net segmentation architecture tailored for processing multi-channel seismic traces and generating precise pick masks.

