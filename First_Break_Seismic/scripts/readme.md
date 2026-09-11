# Scripts Directory (`First_Break_Seismic/scripts/`)

This directory contains executable command-line Python scripts designed to orchestrate the entire machine learning workflow, ranging from raw data preprocessing and model training to evaluation, inference, and visualization[cite: 1]. These scripts serve as the primary operational entry points for executing pipeline tasks across the different seismic assets[cite: 1].

## Script Files Overview

* **`__init__.py`**: Initializes the scripts package namespace and enables module imports across execution scripts.
* **`readme.md`**: Provides documentation detailing the purpose, execution instructions, and operational scope of each script[cite: 1].
* **`check_device_memory.py`**: Validates and monitors hardware device memory availability (CPU, GPU, or Apple Silicon MPS) prior to heavy pipeline execution[cite: 1].
* **`evaluate.py`**: Executes model evaluation routines on validation and test split chunks, calculating performance metrics to assess accuracy[cite: 1].
* **`export_model.py`**: Handles model serialization and exporting routines to convert trained model checkpoints into production-ready formats[cite: 1].
* **`predict.py`**: Runs automated inference on new or unseen seismic records to detect and predict first break arrival curves[cite: 1].
* **`preprocess.py`**: Manages the data preprocessing pipeline by reorganizing raw HDF5 trace data into separated 2D seismic images, extracting first break labels, and generating chunked datasets and manifests[cite: 1].
* **`train.py`**: Orchestrates the model optimization loops, managing epochs, loss computation, validation steps, and checkpoint saving for U-Net architectures[cite: 1].
* **`visualize.py`**: Generates visual plots and graphical representations mapping seismic traces alongside predicted first break lines[cite: 1].
