# Training Directory (`First_Break_Seismic/src/training/`)

This directory houses the core training infrastructure, optimization loops, loss functions, evaluation metrics, and custom callback handlers required to train deep learning models for seismic first-break picking.

## Training Files Overview

* **`__init__.py`**: Marks the directory as a Python package and manages module-level namespace exports for core training components.
* **`callbacks.py`**: Implements custom callback classes (such as model checkpointing, early stopping, and learning rate scheduling) to monitor training dynamics.
* **`losses.py`**: Defines specialized loss functions tailored for precise seismic waveform segmentation and first-break time prediction tasks.
* **`metrics.py`**: Contains quantitative evaluation metrics used to assess model performance and pick accuracy during validation and testing.
* **`readme.md`**: Provides architectural documentation outlining the purpose of the training submodule and its internal execution components.
* **`trainer.py`**: Implements the main training loop and orchestrator class, managing data loaders, forward/backward passes, device allocation, and tracking integrations.
