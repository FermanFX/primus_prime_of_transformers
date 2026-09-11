## 2. First Break Seismic Detection Project

```text
First_Break_Seismic/
├── README.md                           # Main project overview and documentation
├── checkpoints/                        # Model weights and training checkpoints
│   ├── Brunswick/                      # Saved models for Brunswick deposit
│   ├── Halfmile/                       # Saved models for Halfmile deposit
│   ├── Lalor/                          # Saved models for Lalor deposit
│   └── Sudbury/                        # Saved models for Sudbury deposit
├── configs/                            # Configuration files (YAML)
│   ├── default.yaml                    # Base/default parameters
│   ├── batch_config.yaml               # Batch training configurations
│   ├── sweep_config.yaml               # Hyperparameter sweep configurations
│   ├── production.yaml                 # Production/inference settings
│   ├── brunswick.yaml                  # Brunswick-specific site configuration
│   ├── halfmile.yaml                   # Halfmile-specific site configuration
│   ├── lalor.yaml                      # Lalor-specific site configuration
│   └── sudbury.yaml                    # Sudbury-specific site configuration
├── data/                               # Data directory (Raw & Preprocessed HDF5 files)
├── notebooks/                          # Experimental Jupyter notebooks
│   ├── README.md                       # Guidelines for notebooks
│   ├── Seismic_FirstBreak_EDA_brunswick.ipynb   # Brunswick EDA
│   ├── Seismic_FirstBreak_EDA_halfmile.ipynb    # Halfmile EDA
│   ├── Seismic_FirstBreak_EDA_Lalor_raw.ipynb   # Lalor raw data analysis
│   └── Seismic_FirstBreak_EDA_preprocessed.ipynb# Exploratory data analysis on preprocessed data
├── scripts/                            # Executable CLI scripts and entry points
│   ├── preprocess.py                   # Data preprocessing script
│   ├── train.py                        # Single model training script
│   ├── batch_train.py                  # Batch training across multiple models/datasets
│   ├── predict.py                      # Inference and prediction script
│   ├── evaluate.py                     # Model evaluation script
│   ├── export_model.py                 # Export model (ONNX / TorchScript)
│   ├── run_pico_all.py                 # Sequential execution for PicoUNet models
│   ├── run_model_pairs_fallback.py     # Execution pipeline for model pairs with fallback
│   ├── search_models.py                # Model architecture search
│   ├── sweep_mlflow.py                 # Hyperparameter optimization using MLflow
│   ├── check_device_memory.py          # Device memory analysis (GPU/MPS/RAM)
│   └── visualize.py                    # Visualization for seismic traces and picks
├── src/                                # Core source code package
│   ├── __init__.py
│   ├── config.py                       # Configuration loaders and parsers
│   ├── data/                           # PyTorch Dataset and DataLoader implementations
│   │   ├── hdf5_dataset.py             # PyTorch Dataset for HDF5 seismic data
│   │   ├── chunked_dataset.py          # Dataset reader for chunked data
│   │   └── cache.py                    # In-memory caching mechanisms
│   ├── models/                         # Neural network architectures (UNet variants)
│   │   ├── unet.py                     # Standard U-Net implementation
│   │   ├── efficient_unet.py           # EfficientNet-backed U-Net
│   │   ├── light_unet.py               # Lightweight U-Net
│   │   ├── nano_unet.py                # Nano-sized U-Net
│   │   ├── pico_unet.py                # Ultra-lightweight Pico U-Net
│   │   ├── tiny_unet.py                # Tiny U-Net
│   │   ├── mps_light_unet.py           # Apple Silicon (MPS) optimized U-Net
│   │   ├── mobilenet.py                # MobileNet-backed architectures
│   │   └── factory.py                  # Model creation factory pattern
│   ├── preprocessing/                  # Signal processing and dataset compilation
│   │   ├── processor.py                # Signal processing logic
│   │   ├── chunker.py                  # Chunking logic for large seismic volumes
│   │   ├── manifest.py                 # Dataset manifest generation and parsing
│   │   └── writer.py                   # Exporter for processed HDF5 formats
│   ├── training/                       # Model training utilities
│   │   ├── trainer.py                  # Core Trainer class & training loop
│   │   ├── losses.py                   # Loss functions (e.g., BCE, Dice loss)
│   │   ├── metrics.py                  # Evaluation metrics (MAE, RMSE, IoU)
│   │   └── callbacks.py                # Callbacks (Early Stopping, Checkpointing)
│   └── utils/                          # Helper tools and infrastructure utilities
│       ├── logger.py                   # Logging setups
│       ├── mlflow_utils.py             # MLflow integration for experiment tracking
│       ├── tensorboard_utils.py        # TensorBoard logging utilities
│       ├── hdf5_utils.py               # HDF5 file handling utilities
│       └── memory_utils.py             # Memory profiling and optimization (GPU/MPS)
└── tests/                              # Unit and integration tests (pytest)
    ├── test_train.py                   # Tests for training module
    ├── test_evaluate.py                # Tests for evaluation logic
    ├── test_hdf5_dataset.py            # Tests for HDF5 dataset loader
    ├── test_unet.py                    # Tests for UNet architecture implementations
    └── ...                             # Comprehensive test suite for modules
