# Tests Directory (`First_Break_Seismic/tests/`)

This comprehensive test suite directory contains automated unit, integration, and end-to-end test scripts designed to ensure the reliability, structural integrity, and correctness of the entire seismic first break processing pipeline[cite: 1]. Leveraging `pytest`, the test files validate everything from individual module functions to complex CLI executions and model training loops[cite: 1].

## Test Files Overview

* **`__init__.py`**: Initializes the test package namespace and ensures proper module discovery during test execution.
* **`readme.md`**: Provides documentation outlining the organization, execution instructions, and coverage scope of the test suite[cite: 1].
* **`test_callbacks.py`**: Validates training callback behaviors, ensuring proper execution during model optimization epochs.
* **`test_check_device_memory.py`**: Tests hardware memory verification and device monitoring utilities[cite: 1].
* **`test_chunked_dataset.py`**: Verifies chunked dataset loading, batch generation, and split management logic.
* **`test_chunker.py`**: Tests data chunking reproducibility, spatial assignments, and train/val/test split distribution[cite: 1].
* **`test_config.py`**: Ensures correct parsing, validation, and dataclass initialization of configuration files[cite: 1].
* **`test_cpu_models.py`**: Validates model execution and forward-pass correctness under CPU-bound environments.
* **`test_dataset.py`**: Tests general data loading utilities and dataset behavior.
* **`test_evaluate.py`**: Verifies the evaluation pipeline script (`evaluate.py`) and performance metric computations on test splits[cite: 1].
* **`test_export_model.py`**: Tests model export routines to ensure checkpoints serialize into production-ready formats[cite: 1].
* **`test_factory.py`**: Validates the model factory pattern (`factory.py`) to ensure dynamic neural network instantiation functions correctly[cite: 1].
* **`test_hdf5_dataset.py`**: Tests lazy loading, padding, and cropping mechanisms for HDF5 seismic datasets[cite: 1].
* **`test_hdf5_utils.py`**: Verifies HDF5 file validation, index loading, and trace-reading helper functions[cite: 1].
* **`test_light_unet.py`**: Tests the architecture and tensor shapes for the Light U-Net model variant[cite: 1].
* **`test_loadsave_trainer.py`**: Verifies checkpoint saving and loading functionality within the training engine.
* **`test_logger.py`**: Tests logging utilities to ensure logs format correctly across execution modules[cite: 1].
* **`test_logging.py`**: Validates system-wide logging configuration and output streams.
* **`test_losses.py`**: Tests custom seismic loss functions to ensure gradient calculations behave as expected[cite: 1].
* **`test_manifest.py`**: Verifies manifest generation, file validation, and checksum computations[cite: 1].
* **`test_memory_utils.py`**: Tests memory management utility functions during execution routines[cite: 1].
* **`test_metrics.py`**: Validates evaluation metrics used to measure first-break prediction accuracy[cite: 1].
* **`test_mlflow.py`**: Tests MLflow integration wrappers to ensure experiment parameters and metrics log properly[cite: 1].
* **`test_mobilenet.py`**: Verifies the MobileNet-backed U-Net architecture implementation[cite: 1].
* **`test_mps_light_unet.py`**: Tests Light U-Net execution performance and compatibility on Apple Silicon (MPS) backends.
* **`test_mps_models.py`**: Validates model execution and forward-passes on Metal Performance Shaders (MPS).
* **`test_nano_unet.py`**: Tests the architecture and parameter scaling of the Nano U-Net variant[cite: 1].
* **`test_padding.py`**: Verifies trace padding and cropping operations for seismic sequence alignment.
* **`test_parameter_counts.py`**: Validates that neural network models stay within expected parameter budget limits.
* **`test_pico_unet.py`**: Tests the ultra-lightweight Pico U-Net architecture implementation[cite: 1].
* **`test_predict.py`**: Verifies inference script execution (`predict.py`) on unseen seismic records[cite: 1].
* **`test_preprocess.py`**: Tests end-to-end data preprocessing pipelines, CLI argument parsing, manifest generation, and `--force` reprocessing behavior[cite: 1].
* **`test_preprocessing_preprocess.py`**: Validates supplementary preprocessing execution steps and helper integrations.
* **`test_shot_processor.py`**: Tests individual shot processing logic and 3-class target mask generation[cite: 1].
* **`test_tensorboard_utils.py`**: Validates TensorBoard logging wrappers for experiment tracking[cite: 1].
* **`test_tiny_unet.py`**: Tests the Tiny U-Net model architecture and layer configurations[cite: 1].
* **`test_train.py`**: Verifies the training script execution (`train.py`) and loop initialization[cite: 1].
* **`test_trainer.py`**: Tests the core `Trainer` engine class across training epochs and validation cycles[cite: 1].
* **`test_training.py`**: Validates general training pipeline integrations and callback workflows[cite: 1].
* **`test_unet.py`**: Tests the standard U-Net model architecture implementation[cite: 1].
* **`test_visualize.py`**: Verifies visualization script execution (`visualize.py`) for plotting seismic traces and pick lines[cite: 1].
