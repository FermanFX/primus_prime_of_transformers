# First Break Picking — Training & Scripts Documentation

## 1. Overview

This project provides a deep learning pipeline for **First Break Picking** on seismic data.

The training system supports:

- Multiple seismic datasets
- Multiple U-Net-based models
- Automatic memory-aware configuration
- Sequential batch training
- MPS / CUDA / CPU
- Memory error recovery and fallback configurations
- MLflow experiment tracking
- Model evaluation and visualization
- ONNX / TorchScript export
deploy link // https://check-deploy-ros7hqundwlxjin6ercsmr.streamlit.app/
---

## 2. Repository Structure

```text
first_break_pick/
├── configs/
│   ├── batch_config.yaml
│   ├── brunswick.yaml
│   ├── halfmile.yaml
│   ├── lalor.yaml
│   └── sudbury.yaml
├── scripts/
│   ├── batch_train.py
│   ├── train.py
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── export_model.py
│   ├── sweep_mlflow.py
│   ├── check_device_memory.py
│   ├── search_models.py
│   └── run_model_pairs.py
├── data/
│   └── chunks/
├── models/
│   └── registry/
├── logs/
│   └── batch/
└── evaluation_results/
```

---

## 3. Datasets

| Dataset   | Traces | Samples |
| --------- | -----: | ------: |
| Brunswick |   2582 |     751 |
| Halfmile  |   1578 |     751 |
| Lalor     |   2685 |    1501 |
| Sudbury   |   1138 |    1001 |

Each dataset has its own configuration under `configs/`.

---

## 4. Model Architectures

| Model       | Approx. Parameters | Description                     |
| ----------- | -----------------: | ------------------------------- |
| `pico`      |                ~2K | Minimal fallback model          |
| `nano`      |               ~10K | Very small U-Net                |
| `tiny`      |               ~50K | Small U-Net                     |
| `mpslight`  |              ~1.7M | MPS-optimized lightweight U-Net |
| `light`     |              ~2.5M | Lightweight U-Net               |
| `mobile`    |              ~3.5M | MobileNet-based U-Net           |
| `efficient` |                ~5M | EfficientNet-based U-Net        |
| `unet`      |               ~31M | Full U-Net                      |

Smaller models are useful when device memory is limited.

---

# 5. Scripts

| Script                   | Purpose                        |
| ------------------------ | ------------------------------ |
| `batch_train.py`         | Train multiple datasets/models |
| `train.py`               | Train one model on one dataset |
| `preprocess.py`          | Convert HDF5 data into chunks  |
| `evaluate.py`            | Evaluate trained models        |
| `visualize.py`           | Generate prediction images     |
| `export_model.py`        | Export models                  |
| `sweep_mlflow.py`        | Run MLflow parameter sweeps    |
| `check_device_memory.py` | Analyze available memory       |
| `search_models.py`       | Search MLflow models           |
| `run_model_pairs.py`     | Train predefined model pairs   |

---

# 6. `batch_train.py`

The main training orchestrator.

It:

1. Loads the batch configuration.
2. Selects datasets and models.
3. Detects available device memory.
4. Generates suitable training configurations.
5. Trains datasets sequentially.
6. Recovers from memory errors using fallback configurations.
7. Saves training results and logs.

### Basic usage

```bash
python scripts/batch_train.py
```

### Automatic configuration

```bash
python scripts/batch_train.py --auto-config --epochs 30
```

### Specific dataset/model

```bash
python scripts/batch_train.py \
  --auto-config \
  --datasets Halfmile \
  --models pico \
  --epochs 2
```

### Useful options

| Option            | Description                               |
| ----------------- | ----------------------------------------- |
| `--config`        | Batch configuration file                  |
| `--datasets`      | Dataset(s) to train                       |
| `--models`        | Model(s) to train                         |
| `--epochs`        | Number of epochs                          |
| `--device`        | `mps`, `cuda`, or `cpu`                   |
| `--auto-config`   | Automatically select memory configuration |
| `--loss`          | Loss function                             |
| `--class-weights` | Three class weights                       |
| `--preprocess`    | Run preprocessing                         |
| `--verbose`       | Detailed logging                          |
| `--log-memory`    | Log memory usage                          |
| `--log-level`     | Logging level                             |

### Memory recovery

If training runs out of memory:

```text
Attempt 1
   ↓
Memory Error
   ↓
Clear memory
   ↓
Attempt 2 with smaller batch/model
   ↓
Memory Error
   ↓
Attempt 3
   ↓
Success / Skip dataset
```

This prevents one failed dataset from stopping the entire batch.

---

# 7. `train.py`

Trains a single model on a single dataset.

### Example

```bash
python scripts/train.py \
  --config configs/halfmile.yaml \
  --model pico \
  --epochs 2 \
  --batch-size 8 \
  --device mps
```

### Main arguments

```text
--config
--model
--epochs
--loss
--class-weights
--batch-size
--cache-size
--device
--verbose
--log-memory
--resume
--preprocess
```

### Training flow

```text
YAML config
    ↓
SeismicConfig
    ↓
ChunkedDataManager
    ↓
Train / Validation DataLoader
    ↓
Model
    ↓
Loss + Adam
    ↓
SeismicTrainer
    ↓
Checkpoint / Metrics
```

Supported models:

```text
unet
mpslight
light
nano
tiny
pico
mobile
efficient
```

Resume training:

```bash
python scripts/train.py \
  --config configs/halfmile.yaml \
  --model pico \
  --resume models/registry/checkpoint_epoch_10.pt
```

---

# 8. `preprocess.py`

Converts raw HDF5 seismic data into chunked PyTorch tensors.

### Usage

```bash
python scripts/preprocess.py \
  --config configs/halfmile.yaml
```

### Processing pipeline

```text
HDF5
 ↓
Discover shots
 ↓
Validate shots
 ↓
Create train/val/test splits
 ↓
Create chunks
 ↓
Generate segmentation masks
 ↓
Save .pt files
 ↓
Generate manifest.json
```

Output:

```text
data/chunks/Halfmile/
├── manifest.json
├── train/*.pt
├── val/*.pt
└── test/*.pt
```

The generated masks contain three classes:

```text
0 = Before
1 = After
2 = Strip
```

---

# 9. `evaluate.py`

Evaluates a trained model on test or validation data.

### Usage

```bash
python scripts/evaluate.py \
  --config configs/halfmile.yaml \
  --model best \
  --split test
```

### Metrics

Segmentation:

- Accuracy
- Mean IoU
- Mean F1
- IoU per class

First Break:

- MAE
- Standard deviation of absolute error
- Median absolute error
- ±3 sample accuracy

Results are saved under:

```text
evaluation_results/
```

Detailed evaluation can be enabled with:

```bash
--detailed
```

---

# 10. `visualize.py`

Creates visual comparisons between:

1. Original seismic data
2. Ground-truth mask
3. Model prediction

### Usage

```bash
python scripts/visualize.py \
  --config configs/halfmile.yaml \
  --model best \
  --n-samples 5
```

Output:

```text
visualization_results/
└── shot_123_comparison.png
```

---

# 11. `export_model.py`

Exports trained models to:

- ONNX
- TorchScript

### TorchScript

```bash
python scripts/export_model.py \
  --model models/registry/model.pt \
  --torchscript
```

### ONNX

```bash
python scripts/export_model.py \
  --model models/registry/model.pt \
  --onnx
```

MLflow models can also be specified, for example:

```text
models:/halfmile@champion
```

The export script uses an example input shape.
**Verify the input dimensions for the target dataset before deployment.**

---

# 12. `sweep_mlflow.py`

Runs grid-search experiments with MLflow.

Example combinations:

```text
Datasets:
- Halfmile
- Brunswick

Models:
- pico
- mpslight

Losses:
- cross_entropy
- combo
```

### Usage

```bash
python scripts/sweep_mlflow.py \
  --config configs/sweep_config.yaml
```

Each experiment logs parameters, metrics and artifacts to MLflow.

---

# 13. `check_device_memory.py`

Checks available device memory and recommends training configurations.

```bash
python scripts/check_device_memory.py
```

Example recommendations:

```text
PICO      → batch 8
MPSLIGHT  → batch 6
UNET      → batch 3
```

Values depend on the actual device and available memory.

---

# 14. `search_models.py`

Searches and compares models registered in MLflow.

### Search

```bash
python scripts/search_models.py
```

### Dataset filter

```bash
python scripts/search_models.py \
  --dataset Halfmile
```

### IoU filter

```bash
python scripts/search_models.py \
  --dataset Halfmile \
  --min-iou 0.5
```

### Compare models

```bash
python scripts/search_models.py \
  --dataset Halfmile \
  --compare \
  --top 2
```

---

# 15. `run_model_pairs.py`

Trains predefined model pairs across datasets.

### Dry run

```bash
python scripts/run_model_pairs.py \
  --dry-run
```

### Quick training

```bash
python scripts/run_model_pairs.py \
  --epochs 2
```

### Full training

```bash
python scripts/run_model_pairs.py \
  --epochs 30
```

Datasets are processed sequentially to control memory usage.

---

# 16. `batch_config.yaml`

Controls batch training behavior.

### Global configuration

```yaml
epochs: 30
device: mps
log_memory: false
verbose: false
log_level: INFO
preprocess: false
checkpoint_every: 5
early_stopping: 5
timeout_seconds: 7200
skip_failed: true
max_retries: 3
clear_memory_between_datasets: true
pause_between_datasets: 2
```

### Dataset-specific overrides

```yaml
datasets:
  Halfmile:
    epochs: 40

  Lalor:
    batch_size_override: 2
```

### Fallback variants

```yaml
variants:
  - batch_size: 4
    model: mpslight

  - batch_size: 2
    model: mpslight

  - batch_size: 1
    model: tiny
```

These variants are used when memory limitations prevent the preferred configuration from running.

---

# 17. Outputs & Logs

Training produces:

```text
logs/batch/
├── batch_summary_*.json
├── batch_train_*.log
└── batch_train_*_errors.log
```

Model files:

```text
models/registry/
```

Evaluation:

```text
evaluation_results/
```

Preprocessed data:

```text
data/chunks/<dataset>/
```

The batch summary contains:

- Timestamp
- Configuration
- Successful datasets
- Failed datasets
- Duration
- Selected training configuration
- Errors

---

# 18. Notifications

The batch pipeline can optionally send notifications through:

- Email
- Slack

Sensitive values such as passwords and webhooks should be provided through environment variables rather than stored directly in configuration files.

---

# 19. Recommended Workflows

## Quick Test

```bash
python scripts/check_device_memory.py

python scripts/preprocess.py \
  --config configs/halfmile.yaml

python scripts/train.py \
  --config configs/halfmile.yaml \
  --model pico \
  --epochs 2 \
  --batch-size 8

python scripts/evaluate.py \
  --config configs/halfmile.yaml \
  --model best \
  --split test
```

## Full Batch Training

```bash
python scripts/batch_train.py \
  --auto-config \
  --epochs 30 \
  --verbose \
  --log-memory
```

## Evaluate

```bash
python scripts/evaluate.py \
  --config configs/halfmile.yaml \
  --model best \
  --split test
```

## Export

```bash
python scripts/export_model.py \
  --model models/registry/model.pt \
  --onnx \
  --torchscript
```

---

# 20. Troubleshooting

### Out of Memory

Use:

```text
smaller batch size
smaller model
automatic configuration
```

For example:

```bash
python scripts/batch_train.py \
  --datasets Halfmile \
  --batch-size 1
```

If supported by the current CLI/configuration version, increase timeout when necessary.

### Training timeout

Increase:

```yaml
timeout_seconds: 10800
```

### MPS memory

Clear cached memory:

```bash
python -c "import torch; torch.mps.empty_cache()"
```

### HDF5 validation errors

Check:

- HDF5 file path
- Dataset configuration
- Shot indices
- Available data

### Missing CLI option

If an option such as `--checkpoint-every` is required but not recognized, verify that the option is implemented in the corresponding script before adding it to the configuration or workflow.

---

# 21. Best Practices

- Start with **Halfmile** or **Brunswick** before larger datasets.
- Use `--auto-config` when device memory is uncertain.
- Use smaller models for limited-memory devices.
- Monitor memory during training.
- Keep batch summary JSON files.
- Use dataset-specific configuration overrides when necessary.
- Use MLflow to compare experiments.
- Validate exported models before deployment.
- Keep sensitive credentials outside YAML files.

---

## Summary

The training pipeline provides:

- Sequential multi-dataset training
- Multiple model architectures
- Automatic memory configuration
- Memory-error recovery
- Fallback training variants
- MPS / CUDA / CPU support
- MLflow experiment tracking
- Evaluation and visualization
- ONNX / TorchScript export
- Comprehensive logs and summaries
- Optional notifications

The main entry point for large-scale training is:

```bash
python scripts/batch_train.py --auto-config
```

# Seismic First-Break Picking — Development & Testing

## 1. Overview

End-to-end seismic first-break picking pipeline for:

- Halfmile
- Brunswick
- Lalor
- Sudbury

Main pipeline:

```text
HDF5 → Preprocessing → Chunks → Training → MLflow → Evaluation → Prediction/Export
```

---

## 2. Project Status

| Component          | Status         |
| ------------------ | -------------- |
| HDF5 Data Pipeline | ✅ Complete    |
| Preprocessing      | ✅ Complete    |
| Chunking           | ✅ Complete    |
| MPSLightUNet       | ✅ Working     |
| Training           | ✅ Working     |
| Evaluation         | ✅ Complete    |
| TensorBoard        | ✅ Working     |
| MLflow             | ✅ Enhanced    |
| Model Registry     | ✅ Working     |
| Batch Training     | ✅ Implemented |
| Memory Recovery    | ✅ Implemented |
| Model Search       | ✅ Implemented |
| ONNX/TorchScript   | ✅ Implemented |

---

## 3. Datasets

| Dataset   | Traces | Samples |
| --------- | -----: | ------: |
| Brunswick |   2582 |     751 |
| Halfmile  |   1578 |     751 |
| Lalor     |   2685 |    1501 |
| Sudbury   |   1138 |    1001 |

---

## 4. Models

Supported architectures:

```text
unet
mpslight
light
nano
tiny
pico
mobile
efficient
```

Approximate model sizes:

| Model         | Parameters |
| ------------- | ---------: |
| UNet          |        31M |
| MPSLightUNet  |   1.7–1.9M |
| LightUNet     |       2.5M |
| EfficientUNet |         5M |
| MobileUNet    |       3.5M |
| TinyUNet      |        50K |
| NanoUNet      |        10K |
| PicoUNet      |         2K |

---

# 5. Data Pipeline

```text
HDF5
 ↓
Shot Loading
 ↓
Validation
 ↓
Padding/Cropping
 ↓
Mask Creation
 ↓
Chunking
 ↓
Manifest
```

### HDF5Dataset

Supports:

- Lazy loading
- Mask creation
- File-handle reuse
- Configurable `target_traces`

### LRU Cache

Provides:

- Cache get/put
- Eviction
- Hit/miss statistics
- Memory cleanup

### Dataset Splits

Train/validation/test splits must be:

- Complete
- Disjoint
- Reproducible

---

# 6. Preprocessing

The segmentation mask contains three classes:

```text
0 = Before
1 = After
2 = Strip
```

`ShotProcessor` handles:

- Pick validation
- Shot processing
- Padding
- Unlabeled traces

`Chunker` handles:

- Dataset splitting
- Chunk generation
- Custom chunk sizes
- Reproducibility

Manifest stores chunk metadata and checksums.

---

# 7. Training

Training supports:

- Multiple models
- Multiple datasets
- Configurable epochs
- Class weights
- Loss selection
- Checkpoints
- Resume training
- Early stopping
- Gradient clipping
- TensorBoard
- MLflow

Training flow:

```text
Config
 ↓
Dataset
 ↓
Model
 ↓
Loss
 ↓
Train
 ↓
Validate
 ↓
Checkpoint
 ↓
MLflow
```

---

# 8. Losses & Metrics

Supported losses:

- CrossEntropy
- FocalLoss
- DiceLoss
- ComboLoss

Segmentation metrics:

- Accuracy
- IoU
- F1
- Per-class IoU

First-break metrics:

- MAE
- Median/Std Absolute Error
- Tolerance Accuracy
- Pick extraction

---

# 9. Checkpointing

Checkpoints can contain:

- Model state
- Optimizer state
- Scheduler state
- Epoch
- Metrics

Resume workflow:

```text
Checkpoint → Load → Restore State → Continue Training
```

---

# 10. Batch Training

`batch_train.py` supports:

- Multiple datasets
- Multiple models
- Sequential execution
- Dataset-specific configuration
- Automatic configuration
- Memory recovery
- Retry/fallback
- Failure handling
- Summary generation

Typical flow:

```text
Dataset
 ↓
Model
 ↓
Train
 ↓
Evaluate
 ↓
Log
 ↓
Clear Memory
 ↓
Next Run
```

---

# 11. Memory Management

OOM conditions include:

```text
OOM
out of memory
CUDA out of memory
MPS out of memory
cannot allocate
memory exhausted
```

Recovery:

```text
OOM
 ↓
Clear Memory
 ↓
Reduce Batch/Cache
 ↓
Retry
 ↓
Fallback
 ↓
Skip or Stop
```

This is especially important for large datasets such as Lalor.

---

# 12. MLflow

MLflow was expanded from basic tracking to full experiment/model management.

Current features:

- Autologging
- System metrics
- Manual metrics
- Checkpoint tracking
- Model Registry
- Model versions
- Model aliases
- Model search
- Model comparison

Workflow:

```text
Training
 ↓
MLflow Run
 ↓
Model Registry
 ↓
Version
 ↓
Alias
 ↓
Evaluation
```

---

# 13. MLflow Autologging

PyTorch autologging automatically records training information such as:

- Loss
- Learning rate
- Optimizer information
- Model information
- Training metrics

Can be disabled with:

```text
--disable-autolog
```

---

# 14. MLflow System Metrics

Depending on hardware, MLflow can track:

- CPU utilization
- GPU utilization
- Memory usage
- GPU memory
- Temperature

Useful for identifying resource bottlenecks.

---

# 15. Model Registry

Models are stored as versions:

```text
SeismicUNet_Halfmile
 ├── v1
 ├── v2
 ├── v3
 └── v4
```

Versions can contain dataset, model type, metrics and run information.

---

# 16. Model Aliases

| Alias        | Purpose                |
| ------------ | ---------------------- |
| `champion`   | Best model             |
| `challenger` | Candidate model        |
| `staging`    | Model under validation |

Best-model evaluation:

```bash
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best
```

---

# 17. Model Search

`search_models.py` allows filtering by:

- Dataset
- Model
- IoU
- Other metrics

Example:

```bash
python scripts/search_models.py \
    --dataset Halfmile \
    --top 5 \
    --compare
```

---

# 18. Evaluation

`evaluate.py` supports:

- Model loading
- Test-set evaluation
- Segmentation metrics
- First-break metrics
- Per-shot metrics
- JSON/CSV results
- MLflow model loading

---

# 19. Prediction & Export

Prediction supports:

- Model loading
- Inference
- Large-file chunking
- NPY/CSV output

Models can be exported to:

- ONNX
- TorchScript

Exports should be verified after creation.

---

# 20. Visualization

Visualization supports:

- Seismograms
- Ground truth
- Predictions
- Comparison plots
- Wiggle plots
- TensorBoard figures

---

# 21. Main Scripts

| Script                   | Purpose               |
| ------------------------ | --------------------- |
| `train.py`               | Single-model training |
| `batch_train.py`         | Batch training        |
| `preprocess.py`          | HDF5 preprocessing    |
| `evaluate.py`            | Evaluation            |
| `visualize.py`           | Visualization         |
| `export_model.py`        | Model export          |
| `search_models.py`       | MLflow search         |
| `sweep_mlflow.py`        | Experiment sweep      |
| `run_model_pairs.py`     | Model pairs           |
| `check_device_memory.py` | Memory diagnostics    |

---

# 22. Configuration

Important batch settings:

```text
epochs
device
preprocess
checkpoint_every
early_stopping
timeout
skip_failed
max_retries
clear_memory
```

Configuration priority:

```text
CLI > YAML > Defaults
```

---

# 23. Logging & Monitoring

The project uses:

- Loguru
- TensorBoard
- MLflow
- System metrics

Logs are organized in date-based directories.

TensorBoard:

```bash
tensorboard --logdir runs/Halfmile/MPSLightUNet
```

MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

# 24. MLflow Infrastructure

Current setup:

```text
SQLite
```

Backend:

```text
sqlite:///mlflow.db
```

Deferred:

- Remote MLflow Server
- PostgreSQL
- S3/MinIO
- Multi-user collaboration

These are not required for current local development.

---

# 25. Testing Plan

Total planned issues: **119**

| Person | Responsibility                  | Issues |
| ------ | ------------------------------- | -----: |
| P1     | Configuration + Data            |     19 |
| P2     | Models                          |     19 |
| P3     | Training + MLflow               |     26 |
| P4     | Utilities + Evaluation + Export |     23 |
| P5     | Preprocessing + Orchestration   |     24 |
| All    | Integration                     |      8 |

---

# 26. P1 — Configuration + Data

19 issues cover:

- Configuration loading
- Configuration overrides
- LRU cache
- HDF5 utilities
- HDF5Dataset
- Padding/cropping
- ChunkedDataset
- Dataset splits
- Cache integration
- ShotProcessor
- Three-class masks
- Unlabeled traces
- Chunker
- Reproducibility
- Manifest
- Checksums
- Preprocessing pipeline
- `--force`
- Data-layer documentation

---

# 27. P2 — Model Architectures

19 issues cover:

- Model registry
- `create_model()`
- Model information
- Invalid model handling
- UNet
- MPSLightUNet
- LightUNet
- MobileUNet
- EfficientUNet
- TinyUNet
- NanoUNet
- PicoUNet
- MPS support
- CPU fallback
- Variable input sizes
- Parameter consistency
- Model documentation
- Model cards
- Model comparison

---

# 28. P3 — Training + MLflow

26 issues cover:

- CrossEntropy
- FocalLoss
- DiceLoss
- ComboLoss
- Loss factory
- Segmentation metrics
- First-break metrics
- Pick extraction
- EarlyStopping
- ModelCheckpoint
- LoggingCallback
- Training epoch
- Validation
- Checkpoint save/load
- Resume training
- Early stopping integration
- Gradient clipping
- MLflow logging
- Model Registry
- Model aliases
- TensorBoard
- Seismogram logging
- Training CLI
- `--resume`
- `--loss`

---

# 29. P4 — Utilities + Evaluation + Export

23 issues cover:

- Logger
- Log directories
- Log levels
- Task names
- Memory monitoring
- Memory cleanup
- Low-memory detection
- Device detection
- MPS detection
- CPU fallback
- ONNX export
- TorchScript export
- Checkpoint loading
- Visualization CLI
- Seismogram plotting
- Evaluation
- MLflow model loading
- Segmentation metrics
- First-break metrics
- Per-shot metrics
- Prediction
- Large-file chunking
- Output formats

---

# 30. P5 — Preprocessing + Orchestration

24 issues cover:

- Chunker
- Chunk creation
- ShotProcessor
- Mask creation
- Manifest
- Checksums
- Batch configuration
- Memory-error detection
- Real-error filtering
- Memory usage
- `train_dataset()`
- Memory recovery
- Memory cleanup
- Graceful failure
- Fallback progression
- Model-pair runner
- Large-dataset skip logic
- Dry run
- Sweep configuration
- Experiment generation
- MLflow sweep tracking
- Pico runner
- Model search
- Dataset filtering

---

# 31. Integration Tests

Eight final integration issues:

1. End-to-end preprocessing → training → evaluation
2. Checkpoint/resume
3. Batch training
4. MLflow integration
5. Memory recovery
6. Evaluation
7. Model-pair execution
8. MLflow sweep

---

# 32. Testing Dependency

Recommended order:

```text
Configuration / Data
        ↓
Preprocessing
        ↓
Models
        ↓
Losses / Metrics
        ↓
Training
        ↓
MLflow / Evaluation
        ↓
Export / Prediction
        ↓
Integration
```

---

# 33. Development Priority

### High Priority

- Data loading
- Preprocessing
- Model creation
- Losses
- Metrics
- Training
- Checkpoints
- Memory recovery

### Medium Priority

- MLflow
- Evaluation
- Prediction
- Export
- Batch training

### Later

- Documentation
- Model cards
- Model comparison
- Remote MLflow infrastructure

---

# 34. Recommended Validation

Before starting large experiments:

```text
1 epoch training
      ↓
Check logs
      ↓
Check TensorBoard
      ↓
Check MLflow
      ↓
Check checkpoint
      ↓
Evaluate
      ↓
Run 30 epochs
```

After validation, run the full experiment across all four datasets.

---

# 35. Final System

The target system provides:

- Reliable seismic data processing
- First-break segmentation
- Multiple model architectures
- Automated training
- Checkpoint/resume
- Memory-aware batch training
- MLflow experiment tracking
- Model Registry
- Model comparison
- Evaluation
- Prediction
- Model export
- End-to-end testing

Final architecture:

```text
HDF5
 ↓
Preprocessing
 ↓
Chunks
 ↓
Model Factory
 ↓
Training
 ├── TensorBoard
 ├── MLflow
 └── Checkpoints
       ↓
 Model Registry
       ↓
 Evaluation
       ↓
 Prediction / Export
```

The main remaining focus is **testing, integration, model comparison and large-scale training across the four seismic datasets**.

# Seismic First-Break Picking

Deep learning pipeline for automatic seismic first-break detection across **Brunswick, Halfmile, Lalor and Sudbury**.

**Pipeline:**

`HDF5 → Preprocessing → Chunks → Training → MLflow → Evaluation → Prediction/Export`

---

## 1. Project Structure

```text
first_break_pick/
├── configs/
│   ├── batch_config.yaml
│   ├── default.yaml
│   └── {brunswick,halfmile,lalor,sudbury}.yaml
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── batch_train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── export_model.py
│   ├── search_models.py
│   ├── sweep_mlflow.py
│   ├── run_model_pairs.py
│   └── check_device_memory.py
├── src/
│   ├── config.py
│   ├── data/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   └── utils/
├── data/
│   ├── raw/
│   └── chunks/
├── models/registry/
├── logs/
├── runs/
├── mlflow.db
├── requirements.txt
└── README.md
```

---

## 2. Datasets

| Dataset   | Traces | Samples | Approx. Size |
| --------- | -----: | ------: | -----------: |
| Brunswick |   2582 |     751 |       1.5 GB |
| Halfmile  |   1578 |     751 |       1.2 GB |
| Lalor     |   2685 |    1501 |       3.3 GB |
| Sudbury   |   1138 |    1001 |       1.0 GB |

Raw data is stored in HDF5. Dataset-specific settings are defined in `configs/*.yaml`.

---

## 3. Data Pipeline

```text
Raw HDF5
   ↓
Shot discovery & validation
   ↓
Train / Val / Test split
   ↓
Shot processing
   ↓
3-class mask generation
   ↓
Chunking
   ↓
PyTorch tensors + manifest
   ↓
Training
```

Shots with fewer than **10 traces** are filtered.

Mask classes:

```text
0 = Before
1 = After
2 = First-break strip
```

The dataset uses lazy loading and an **LRU cache** for memory efficiency.

---

## 4. Configuration

Example `configs/halfmile.yaml`:

```yaml
dataset_name: Halfmile
hdf5_path: data/raw/Halfmile3D_add_geom_sorted.hdf5
target_traces: 1578
n_samples: 751
strip_width: 8
chunk_size: 69

train_split: 0.8
val_split: 0.1
test_split: 0.1

batch_size: 4
learning_rate: 0.001
n_epochs: 30
device: mps
class_weights: [0.1, 0.1, 0.8]
cache_size: 3
```

CLI values override YAML values.

`batch_config.yaml` controls:

- epochs/device
- preprocessing
- checkpoint frequency
- early stopping
- timeout/retries
- dataset-specific overrides
- memory recovery
- fallback models

---

## 5. Model Architectures

| Model         | Params | Main Use            |
| ------------- | -----: | ------------------- |
| PicoUNet      |    ~2K | Minimal testing     |
| NanoUNet      |   ~10K | Ultra-light testing |
| TinyUNet      |   ~50K | Lightweight testing |
| MPSLightUNet  |  ~1.7M | Main MPS model      |
| LightUNet     |  ~2.5M | Lightweight         |
| MobileUNet    |  ~3.5M | Mobile-oriented     |
| EfficientUNet |    ~5M | Efficient model     |
| UNet          |   ~31M | Full model          |

Model creation is handled through the model factory.

---

# 6. Training

### Single Model

`train.py` trains one model on one dataset.

```bash
python scripts/train.py \
  --config configs/halfmile.yaml \
  --model mpslight \
  --epochs 30
```

Supports:

- model/loss selection
- batch/cache size
- learning rate
- device
- class weights
- checkpoints
- resume
- early stopping
- memory logging
- TensorBoard
- MLflow

Resume:

```bash
python scripts/train.py \
  --config configs/halfmile.yaml \
  --model mpslight \
  --resume checkpoints/epoch_10.pt
```

### Batch Training

`batch_train.py` trains multiple datasets/models sequentially.

```bash
python scripts/batch_train.py --auto-config --epochs 30
```

It provides:

```text
Dataset
  ↓
Memory Check
  ↓
Training Variant
  ↓
Success ─────────→ Next Dataset
  │
  └─ OOM → Clear Memory → Smaller Config/Model → Retry
```

Features:

- automatic configuration
- dataset/model selection
- memory-aware batch/cache
- OOM recovery
- fallback variants
- retries
- cleanup between datasets
- training summary

---

## 7. Memory Management

The project includes:

- LRU chunk cache
- system/device memory monitoring
- automatic memory cleanup
- MPS/CUDA OOM detection
- batch-size reduction
- cache-size reduction
- smaller-model fallback

Check available memory:

```bash
python scripts/check_device_memory.py
```

For MPS, `PYTORCH_MPS_MEMORY_LIMIT` can be used when necessary.

Recommended values are **hardware-dependent** and should not be treated as universal.

---

## 8. Losses & Metrics

### Losses

- Cross Entropy
- Focal Loss
- Dice Loss
- Combo Loss

### Segmentation Metrics

- Accuracy
- IoU
- F1
- Per-class IoU

### First-Break Metrics

- MAE
- Median Absolute Error
- Standard Deviation of Error
- ±3 sample accuracy

---

## 9. Evaluation

Evaluate the best model:

```bash
python scripts/evaluate.py \
  --config configs/halfmile.yaml \
  --model best \
  --split test
```

Detailed per-shot evaluation:

```bash
python scripts/evaluate.py \
  --config configs/halfmile.yaml \
  --model best \
  --detailed
```

Outputs include JSON/CSV evaluation results and detailed errors.

---

## 10. Visualization

```bash
python scripts/visualize.py \
  --config configs/halfmile.yaml \
  --model best \
  --n_samples 10
```

Visualizations contain:

```text
Seismogram | Ground Truth | Prediction
```

---

## 11. MLflow

MLflow is used for:

- experiment tracking
- parameter/metric logging
- system metrics
- model artifacts
- model registry
- versioning
- model comparison

Local backend:

```text
sqlite:///mlflow.db
```

Start MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Model lifecycle:

```text
Training
   ↓
Candidate
   ↓
Staging
   ↓
Compare with Champion
   ├── Better → Champion
   └── Worse  → Challenger
```

Aliases:

```text
champion
challenger
staging
```

Search models:

```bash
python scripts/search_models.py \
  --dataset Halfmile \
  --top 5 \
  --compare
```

---

## 12. Experiments

### MLflow Sweep

Runs combinations of:

```text
Dataset × Model × Loss
```

```bash
python scripts/sweep_mlflow.py \
  --config configs/sweep_config.yaml
```

### Model Pairs

```bash
python scripts/run_model_pairs.py --epochs 2 --dry-run
```

Use `--epochs 30` for full training.

---

## 13. Model Export

Models can be exported to:

- ONNX
- TorchScript

```bash
python scripts/export_model.py \
  --model model.pt \
  --onnx \
  --torchscript
```

**Note:** the export example input shape must match the target dataset/model configuration; do not assume one fixed shape for all datasets.

---

## 14. Logging & Monitoring

Logging uses **Loguru**.

Available:

- DEBUG / INFO / WARNING / ERROR / CRITICAL
- structured JSON logs
- log rotation and retention
- batch logs
- TensorBoard
- MLflow system metrics

TensorBoard:

```bash
tensorboard --logdir runs/
```

---

## 15. Main Scripts

| Script                   | Purpose                  |
| ------------------------ | ------------------------ |
| `preprocess.py`          | HDF5 → chunks            |
| `train.py`               | Single-model training    |
| `batch_train.py`         | Multi-dataset training   |
| `evaluate.py`            | Evaluation               |
| `visualize.py`           | Prediction visualization |
| `export_model.py`        | ONNX/TorchScript         |
| `search_models.py`       | MLflow model search      |
| `sweep_mlflow.py`        | Experiment sweep         |
| `run_model_pairs.py`     | Model-pair experiments   |
| `check_device_memory.py` | Memory diagnostics       |

---

## 16. Quick Workflow

### Development Test

```bash
# 1. Check device
python scripts/check_device_memory.py

# 2. Preprocess
python scripts/preprocess.py \
  --config configs/halfmile.yaml

# 3. Quick training
python scripts/train.py \
  --config configs/halfmile.yaml \
  --model pico \
  --epochs 2

# 4. Evaluate
python scripts/evaluate.py \
  --config configs/halfmile.yaml \
  --model best

# 5. Visualize
python scripts/visualize.py \
  --config configs/halfmile.yaml \
  --model best
```

### Full Training

```bash
python scripts/batch_train.py \
  --auto-config \
  --epochs 30 \
  --verbose \
  --log-memory
```

Then evaluate and export the best/champion model.

---

## 17. Environment Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Optional:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export PYTORCH_MPS_MEMORY_LIMIT=8000000000
```

---

## 18. Troubleshooting

| Problem                   | Solution                                  |
| ------------------------- | ----------------------------------------- |
| Memory/OOM                | Reduce batch/cache or use smaller model   |
| MPS OOM                   | Set MPS memory limit and retry            |
| HDF5 validation failure   | Check dataset/file integrity              |
| Model not found in MLflow | Verify model registration/run             |
| CUDA unavailable          | Use CPU/MPS                               |
| CLI option missing        | Check current `train.py`/script arguments |

---

## 19. Development Priority

Recommended validation order:

```text
Configuration & Data
        ↓
Preprocessing
        ↓
Models
        ↓
Losses & Metrics
        ↓
Training & Checkpoints
        ↓
MLflow & Evaluation
        ↓
Prediction & Export
        ↓
End-to-End Integration
```

Before large experiments:

1. Run a 1–2 epoch test.
2. Verify preprocessing.
3. Check TensorBoard/MLflow.
4. Verify checkpoint creation.
5. Evaluate the model.
6. Run longer training.
7. Expand to all four datasets.

---

## 20. Current System

The project provides:

- Multi-asset seismic data processing
- First-break segmentation
- Multiple U-Net architectures
- Configurable training
- Checkpoint/resume
- Memory-aware batch training
- OOM recovery
- MLflow experiment tracking
- Model Registry
- Model comparison
- Evaluation and visualization
- ONNX/TorchScript export

**Target architecture:**

```text
             HDF5
               ↓
         Preprocessing
               ↓
            Chunks
               ↓
         Model Factory
               ↓
           Training
          /    |    \
 TensorBoard MLflow Checkpoints
               ↓
         Model Registry
               ↓
           Evaluation
               ↓
       Prediction / Export
```

# Seismic First-Break Picking

End-to-end deep learning pipeline for automatic seismic first-break picking across four real-world seismic assets:

- Brunswick
- Halfmile
- Lalor
- Sudbury

The system covers data preprocessing, memory-efficient chunking, model training, batch orchestration, evaluation, visualization, experiment tracking, model registry, and production export.

**Pipeline**

```text
HDF5
  ↓
Preprocessing
  ↓
Train / Validation / Test Split
  ↓
Chunked Dataset
  ↓
Model Training
  ↓
MLflow / TensorBoard
  ↓
Evaluation
  ↓
Model Registry
  ↓
Prediction / ONNX / TorchScript
```

---

## 1. Project Structure

```text
first_break_pick/
├── configs/
│   ├── batch_config.yaml
│   ├── sweep_config.yaml
│   ├── default.yaml
│   ├── production.yaml
│   └── {brunswick,halfmile,lalor,sudbury}.yaml
│
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── batch_train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── export_model.py
│   ├── search_models.py
│   ├── sweep_mlflow.py
│   ├── run_model_pairs.py
│   └── check_device_memory.py
│
├── src/
│   ├── config.py
│   ├── data/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   └── utils/
│
├── tests/
├── data/
│   ├── raw/
│   └── chunks/
├── models/registry/
├── logs/
├── runs/
├── mlflow.db
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Main Components

| Directory            | Purpose                                                 |
| -------------------- | ------------------------------------------------------- |
| `configs/`           | Dataset, batch and experiment configuration             |
| `scripts/`           | CLI entry points                                        |
| `src/data/`          | HDF5, chunked datasets and caching                      |
| `src/models/`        | U-Net model architectures                               |
| `src/preprocessing/` | Shot processing, chunking and manifests                 |
| `src/training/`      | Trainer, losses, metrics and callbacks                  |
| `src/utils/`         | Logging, memory, MLflow, TensorBoard and HDF5 utilities |
| `tests/`             | Automated tests                                         |
| `data/`              | Raw and preprocessed data                               |
| `models/registry/`   | Saved checkpoints                                       |
| `logs/`              | Application and batch logs                              |
| `runs/`              | TensorBoard logs                                        |

---

## 2. Datasets

| Dataset   | Traces | Samples | Approx. Size |
| --------- | -----: | ------: | -----------: |
| Brunswick |  2,582 |     751 |      ~1.5 GB |
| Halfmile  |  1,578 |     751 |      ~1.2 GB |
| Lalor     |  2,685 |   1,501 |      ~3.3 GB |
| Sudbury   |  1,138 |   1,001 |      ~1.0 GB |

Each dataset has its own configuration file:

```text
configs/
├── brunswick.yaml
├── halfmile.yaml
├── lalor.yaml
└── sudbury.yaml
```

---

## 3. Data Pipeline

```text
HDF5
 ↓
Shot Discovery & Validation
 ↓
Filter Invalid Shots
 ↓
Train / Val / Test Split
 ↓
Shot Processing
 ↓
First-Break Mask Creation
 ↓
Chunking
 ↓
PyTorch Tensors + Manifest
 ↓
Training
```

### Preprocessing

The preprocessing pipeline:

- loads seismic traces from HDF5;
- groups traces by shot;
- validates first-break labels;
- filters shots with fewer than 10 traces;
- creates train/validation/test splits;
- pads/crops traces when required;
- creates first-break segmentation masks;
- writes chunked tensors;
- generates `manifest.json`.

### Three-Class Mask

```text
0 → Before first break
1 → After first break
2 → First-break strip
```

The first-break region is controlled by `strip_width`.

### Memory-Efficient Dataset

The data loader supports:

- HDF5 lazy loading;
- file-handle reuse;
- chunked loading;
- configurable target trace count;
- LRU caching;
- cache statistics and cleanup.

This avoids loading the complete seismic dataset into RAM.

---

## 4. Configuration

Configuration is YAML-based and can be overridden from the CLI.

Example:

```yaml
dataset_name: "Halfmile"
hdf5_path: "data/raw/Halfmile3D_add_geom_sorted.hdf5"

target_traces: 1578
n_samples: 751
strip_width: 8
chunk_size: 69

train_split: 0.8
val_split: 0.1
test_split: 0.1

batch_size: 4
learning_rate: 0.001
n_epochs: 30

device: "mps"
class_weights: [0.1, 0.1, 0.8]
cache_size: 3
```

### Configuration Priority

```text
CLI arguments
    ↓
Dataset / Batch YAML
    ↓
Default values
```

### Batch Configuration

`configs/batch_config.yaml` controls:

- epochs;
- device;
- preprocessing;
- checkpoint frequency;
- early stopping;
- timeout;
- dataset-specific overrides;
- memory logging;
- retry/fallback behavior;
- memory cleanup between datasets.

Dataset-specific settings can override global values.

---

## 5. Model Architectures

| Model       | Parameters | Main Use                   |
| ----------- | ---------: | -------------------------- |
| `pico`      |        ~2K | Last-resort / testing      |
| `nano`      |       ~10K | Very lightweight testing   |
| `tiny`      |       ~50K | Fast fallback              |
| `mpslight`  |      ~1.7M | Recommended MPS model      |
| `light`     |      ~2.5M | Lightweight model          |
| `mobile`    |      ~3.5M | MobileNet-based            |
| `efficient` |        ~5M | EfficientNet-based         |
| `unet`      |       ~31M | Full U-Net / high capacity |

`MPSLightUNet` is the primary lightweight model for Apple Silicon/MPS environments.

---

## 6. Loss Functions

The training pipeline supports:

- **Cross Entropy** — baseline classification loss
- **Focal Loss** — useful for class imbalance
- **Dice Loss** — segmentation-focused optimization
- **Combo Loss** — combination of CE, Focal and Dice

Example:

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --loss combo
```

---

## 7. Training

### Single Dataset / Model

`train.py` handles:

```text
Config
 ↓
Dataset
 ↓
Model
 ↓
Loss
 ↓
Optimizer
 ↓
Training
 ↓
Validation
 ↓
Checkpoint
 ↓
MLflow / TensorBoard
```

Supported training features:

- multiple model architectures;
- multiple loss functions;
- configurable batch/cache size;
- class weights;
- learning rate;
- checkpointing;
- checkpoint resume;
- early stopping;
- gradient clipping;
- TensorBoard;
- MLflow;
- device selection.

Example:

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --loss combo \
    --verbose
```

Resume training:

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --resume models/registry/checkpoint.pt
```

---

# 8. Batch Training Pipeline

`batch_train.py` orchestrates training across multiple datasets and models.

### Batch Flow

```text
Load Configuration
        ↓
Detect Device / Memory
        ↓
Generate Training Configuration
        ↓
For Each Dataset
        ↓
Try Training Variant
        ↓
Success ──────────────→ Save Results
        │
        │ Memory Error
        ↓
Clear Memory
        ↓
Try Smaller Variant
        ↓
Success / Retry
        ↓
Next Dataset
        ↓
Generate Summary
```

### Sequential Training

Datasets are trained sequentially so that memory can be released between runs.

A failure in one dataset does not stop the remaining datasets when `skip_failed: true`.

### Automatic Memory Recovery

The system detects common memory errors such as:

```text
out of memory
MPS out of memory
CUDA out of memory
cannot allocate
memory exhausted
OOM
```

Recovery may include:

1. clearing device/cache memory;
2. reducing batch size;
3. reducing cache size;
4. switching to a smaller model;
5. retrying training;
6. skipping the dataset if all variants fail.

Example fallback strategy:

```text
Attempt 1 → batch=4, model=mpslight
     ↓ OOM
Attempt 2 → batch=2, model=mpslight
     ↓ OOM
Attempt 3 → batch=1, model=tiny
     ↓
   Success
```

The exact values are hardware/configuration dependent.

### Smart Configuration

With `--auto-config`, the system can use available device memory and dataset size to select appropriate:

- batch size;
- cache size;
- model;
- memory limit;
- fallback configuration.

---

## 9. Batch CLI

Train all datasets:

```bash
python scripts/batch_train.py --auto-config --epochs 30
```

Train selected datasets:

```bash
python scripts/batch_train.py \
    --auto-config \
    --datasets Halfmile \
    --datasets Brunswick
```

Train selected models:

```bash
python scripts/batch_train.py \
    --auto-config \
    --models pico \
    --models mpslight
```

Debug / memory monitoring:

```bash
python scripts/batch_train.py \
    --auto-config \
    --verbose \
    --log-level DEBUG \
    --log-memory
```

Force preprocessing:

```bash
python scripts/batch_train.py --preprocess
```

### Main Options

| Option                | Purpose                  |
| --------------------- | ------------------------ |
| `--config`, `-c`      | Batch configuration      |
| `--datasets`, `-d`    | Select datasets          |
| `--models`, `-m`      | Select models            |
| `--epochs`, `-e`      | Override epochs          |
| `--device`            | `cpu`, `cuda`, or `mps`  |
| `--auto-config`, `-a` | Automatic configuration  |
| `--preprocess`, `-p`  | Force preprocessing      |
| `--log-memory`        | Enable memory logging    |
| `--verbose`, `-v`     | Verbose logging          |
| `--log-level`         | Logging level            |
| `--timeout`           | Override dataset timeout |

---

## 10. Evaluation

`evaluate.py` supports segmentation and first-break metrics.

### Segmentation Metrics

- Accuracy
- Mean IoU
- Mean F1
- Per-class IoU

### First-Break Metrics

- MAE
- Median Absolute Error
- Standard Deviation of Absolute Error
- Accuracy within ±3 samples/ms, depending on label representation

Example:

```bash
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split test
```

Detailed per-shot evaluation:

```bash
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split test \
    --detailed
```

Results can be exported to JSON/CSV.

---

## 11. Visualization

`visualize.py` compares:

```text
Original Seismogram | Ground Truth | Prediction
```

Example:

```bash
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model best \
    --n_samples 10
```

Visualization is used to verify whether predicted first breaks follow the actual seismic arrival patterns.

---

## 12. MLflow & Experiment Tracking

MLflow provides:

- experiment tracking;
- parameters;
- losses and metrics;
- system metrics;
- checkpoints/artifacts;
- model versions;
- model registry;
- model comparison.

Current local backend:

```text
sqlite:///mlflow.db
```

Start MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Registry Lifecycle

```text
Training Run
     ↓
Candidate
     ↓
Staging
     ↓
Compare with Champion
     ↓
 ┌───────────────┐
 │ Better        │ → Champion
 │ Not better    │ → Challenger
 └───────────────┘
```

Useful search:

```bash
python scripts/search_models.py \
    --dataset Halfmile \
    --top 5 \
    --compare
```

---

## 13. Experiment Sweeps

`sweep_mlflow.py` performs grid-search experiments across:

```text
Dataset × Model × Loss
```

Example:

```bash
python scripts/sweep_mlflow.py \
    --config configs/sweep_config.yaml
```

A sweep configuration can specify:

```yaml
sweep:
  datasets:
    - Halfmile
  models:
    - pico
    - nano
    - tiny
    - mpslight
  losses:
    - cross_entropy
    - focal
    - dice
    - combo
```

`run_model_pairs.py` provides an additional workflow for running selected model pairs across datasets.

---

## 14. Model Export

Models can be exported to:

- ONNX
- TorchScript

Example:

```bash
python scripts/export_model.py \
    --model models/registry/model.pt \
    --onnx \
    --torchscript
```

When exporting ONNX, the example input shape must match the target dataset/configuration. Dataset dimensions are not universal.

---

## 15. Logging & Monitoring

The project uses **Loguru**, TensorBoard and MLflow.

### Logs

```text
logs/
├── YYYY-MM-DD/
├── batch/
└── latest/
```

Logs support structured output, rotation and retention.

Batch runs additionally generate:

```text
logs/batch/
├── batch_summary_*.json
├── batch_train_*.log
└── batch_train_*_errors.log
```

The batch summary records:

- successful/failed datasets;
- selected training configuration;
- duration;
- errors;
- overall batch results.

### TensorBoard

```bash
tensorboard --logdir runs/
```

---

## 16. Memory Management

Memory management is important because the datasets, especially Lalor, can be large.

The system provides:

- LRU chunk cache;
- memory usage monitoring;
- device memory detection;
- cache cleanup;
- MPS/CUDA memory recovery;
- automatic batch reduction;
- model fallback;
- dataset-level cleanup.

Check device memory:

```bash
python scripts/check_device_memory.py
```

Memory recommendations are hardware-dependent and should not be treated as fixed limits.

---

## 17. Testing

The project uses Pytest.

Run all tests:

```bash
python3.12 -m pytest tests/ -v
```

Run a specific test file:

```bash
python3.12 -m pytest tests/test_pipeline.py -v
```

Run with coverage:

```bash
python3.12 -m pytest tests/ \
    --cov=src \
    --cov-report=html
```

Testing covers:

- configuration;
- device/memory detection;
- smart configuration;
- LRU cache;
- chunking;
- manifests;
- preprocessing;
- masks;
- metrics;
- losses;
- memory error detection;
- MLflow;
- batch training sequence;
- end-to-end pipeline.

### Recommended Test Order

```text
Configuration / Data
        ↓
Preprocessing
        ↓
Models
        ↓
Losses / Metrics
        ↓
Training
        ↓
MLflow / Evaluation
        ↓
Export / Prediction
        ↓
Integration
```

---

## 18. Recommended Development Workflow

### Step 1 — Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Optional:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export PYTORCH_MPS_MEMORY_LIMIT=8000000000
```

### Step 2 — Check Device

```bash
python scripts/check_device_memory.py
```

### Step 3 — Preprocess

```bash
python scripts/preprocess.py \
    --config configs/halfmile.yaml
```

Force reprocessing:

```bash
python scripts/preprocess.py \
    --config configs/halfmile.yaml \
    --force
```

### Step 4 — Smoke Test

```bash
python scripts/batch_train.py \
    --auto-config \
    --datasets Halfmile \
    --models pico \
    --epochs 2 \
    --verbose
```

### Step 5 — Evaluate

```bash
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split test
```

### Step 6 — Visualize

```bash
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model best \
    --n_samples 10
```

### Step 7 — Full Training

```bash
python scripts/batch_train.py \
    --auto-config \
    --epochs 30 \
    --models mpslight \
    --log-memory
```

### Step 8 — Track Results

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

```bash
tensorboard --logdir runs/
```

---

## 19. Troubleshooting

| Problem                 | Typical Solution                              |
| ----------------------- | --------------------------------------------- |
| MPS/CUDA OOM            | Reduce batch/cache or use smaller model       |
| Training timeout        | Increase `--timeout`                          |
| HDF5 validation failure | Check file path/data integrity                |
| MLflow model not found  | Verify model registration and MLflow tracking |
| CUDA unavailable        | Use `--device cpu` or `--device mps`          |
| High RAM usage          | Reduce cache size and batch size              |
| Failed batch dataset    | Check `logs/batch/` summary/error logs        |

For debugging:

```bash
python scripts/check_device_memory.py
python scripts/search_models.py
tail -f logs/latest/latest.log
```

Check device availability:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'MPS:', torch.backends.mps.is_available())"
```

---

## 20. Main Scripts

| Script                   | Purpose                            |
| ------------------------ | ---------------------------------- |
| `preprocess.py`          | HDF5 → chunked training data       |
| `train.py`               | Single model/dataset training      |
| `batch_train.py`         | Multi-dataset/model orchestration  |
| `evaluate.py`            | Model and first-break evaluation   |
| `visualize.py`           | Prediction visualization           |
| `export_model.py`        | ONNX/TorchScript export            |
| `search_models.py`       | MLflow model search/comparison     |
| `sweep_mlflow.py`        | Dataset × model × loss experiments |
| `run_model_pairs.py`     | Model-pair experiments             |
| `check_device_memory.py` | Device/memory recommendations      |

---

## 21. System Architecture

```text
                 ┌──────────────────┐
                 │   Raw HDF5 Data  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │   Preprocessing  │
                 │  Shot + Masks    │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Chunked Dataset  │
                 │ + LRU Cache      │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Training Engine  │
                 │ U-Net Variants   │
                 └────────┬─────────┘
                          ↓
              ┌───────────┴───────────┐
              ↓                       ↓
        ┌───────────┐           ┌───────────┐
        │ TensorBoard│           │  MLflow   │
        └───────────┘           └─────┬─────┘
                                      ↓
                              ┌──────────────┐
                              │Model Registry│
                              └──────┬───────┘
                                     ↓
                         ┌────────────────────┐
                         │ Evaluation / Export│
                         │ ONNX / TorchScript │
                         └────────────────────┘
```

---

## 22. Current Capabilities

| Capability                   | Status        |
| ---------------------------- | ------------- |
| Multi-dataset support        | ✅ 4 datasets |
| HDF5 preprocessing           | ✅            |
| Chunked/lazy data loading    | ✅            |
| LRU memory cache             | ✅            |
| First-break segmentation     | ✅            |
| Multiple U-Net architectures | ✅ 8 models   |
| Multiple loss functions      | ✅ 4 losses   |
| Batch training               | ✅            |
| Smart configuration          | ✅            |
| Memory error recovery        | ✅            |
| Checkpoint / resume          | ✅            |
| Early stopping               | ✅            |
| TensorBoard                  | ✅            |
| MLflow tracking              | ✅            |
| Model registry               | ✅            |
| Model search/comparison      | ✅            |
| Evaluation                   | ✅            |
| Visualization                | ✅            |
| ONNX/TorchScript export      | ✅            |
| Automated tests              | ✅            |
| Experiment sweeps            | ✅            |

The resulting system provides a complete, memory-aware and experiment-trackable workflow for seismic first-break picking from raw HDF5 data through model training, evaluation and deployment.
