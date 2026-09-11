# Utils Directory (`First_Break_Seismic/src/utils/`)

This directory contains utility modules and helper functions designed to support data processing, logging, experiment tracking, and resource management across the project pipelines.

## Utility Modules Overview

* **`__init__.py`**: Marks the directory as a Python package and manages module-level exports for convenient importing.
* **`hdf5_utils.py`**: Provides helper functions for reading, writing, and inspecting HDF5 seismic data assets, handling structural queries and dataset extractions.
* **`logger.py`**: Implements logging configurations to standardize console and file-based output tracking across training and inference runs.
* **`memory_utils.py`**: Handles resource tracking, memory optimization, and hardware monitoring utilities to prevent out-of-memory errors during large-scale data loading.
* **`mlflow_utils.py`**: Manages integration with MLflow for tracking experiments, logging metrics, parameters, and saving model artifacts.
* **`tensorboard_utils.py`**: Provides wrappers and utilities for logging training metrics, loss curves, and visualization data directly to TensorBoard.
