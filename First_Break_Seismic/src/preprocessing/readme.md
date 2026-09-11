# Preprocessing Directory (`First_Break_Seismic/src/preprocessing/`)

This directory houses the core data preprocessing logic, structural chunking scripts, manifest generators, and batch processors used to prepare raw seismic datasets for neural network ingestion.

## Preprocessing Files Overview

* **`__init__.py`**: Marks the directory as a Python package and manages module-level namespace exports for preprocessing components.
* **`chunker.py`**: Handles the segmentation and chunking of large-scale seismic volumes into manageable blocks for efficient memory mapping and loading.
* **`manifest.py`**: Generates and manages data manifests, indexing structural metadata and trace mappings across survey files.
* **`processor.py`**: Orchestrates data processing routines, applying transformations, filtering parameters, and scaling operations to raw traces.
* **`readme.md`**: Provides architectural documentation outlining the structural design and purpose of the preprocessing modules.
* **`writer.py`**: Manages the formatting and output writing of processed tensor blocks and chunked datasets back to disk or HDF5 structures.
