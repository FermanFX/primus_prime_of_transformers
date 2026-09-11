# Source Directory (`First_Break_Seismic/src/`)

This directory contains the primary source code implementation for the first-break picking pipeline, housing core packages responsible for data handling, neural network modeling, preprocessing routines, and training execution.

## Source Subdirectories Overview

* **`__init__.py`**: Marks the source directory as a Python package and manages top-level namespace initialization.
* **`config.py`**: Manages global configuration loading, parsing parameters, and setting up runtime variables across modules.
* **`data/`**: Houses datasets, caching mechanisms, and HDF5 loaders designed to efficiently stream seismic data arrays.
* **`models/`**: Contains neural network architectures, factory loaders, and U-Net variants ranging from nano to full-scale implementations.
* **`preprocessing/`**: Manages data chunking, manifest generation, trace processing, and output writers for raw seismic assets.
* **`readme.md`**: Provides architectural documentation detailing the core package organization and module layouts under the source directory.
* **`training/`**: Encapsulates the core training loops, loss functions, evaluation metrics, and custom training callback handlers.
* **`utils/`**: Provides utility modules for logging, memory optimization, HDF5 handling, and platform integrations like MLflow and TensorBoard.
