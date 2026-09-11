# Configs Directory (`First_Break_Seismic/configs/`)

This directory serves as the centralized configuration management repository for the first-break picking and deep learning pipelines. It stores YAML-formatted configuration files that govern model hyperparameters, data paths, training execution flags, batch sizes, and survey-specific dataset mappings.

## Configuration Files Overview

* **`batch_config.yaml`**: Defines multi-job execution parameters and batch scheduling options for cluster or sequential model training runs.
* **`brunswick.yaml`**: Survey-specific configuration tailored for the Brunswick dataset (`Brunswick_orig_1500ms_V2.hdf5`), mapping specific input dimensions and header fields.
* **`default.yaml`**: Serves as the fallback configuration file supplying baseline global hyperparameters, network architecture defaults, and optimizer settings.
* **`experiment_001.yaml`**: Custom configuration file dedicated to tracking and executing specific experimental training runs and hyperparameter tests.
* **`halfmile.yaml`**: Survey-specific parameter configuration designed for the Halfmile dataset (`Halfmile3D_add_geom_sorted.hdf5`), handling specific trace and shot gather setups.
* **`lalor.yaml`**: Config file tailored to the Lalor land-seismic survey dataset (`Lalor_raw_z_1500ms_norp_geom_v3.hdf5`), adjusting paths and label sources accordingly.
* **`production.yaml`**: Optimized configuration profile configured specifically for deployment, inference, and production-level model execution.
* **`readme.md`**: Provides structural documentation outlining the purpose of each configuration file and guidance for parameter tuning across datasets.
* **`sudbury.yaml`**: Dedicated configuration file for the Sudbury seismic dataset (`preprocessed_Sudbury3D.hdf`), pointing to its respective paths and tensor parameters.
* **`sweep_config.yaml`**: Controls hyperparameter optimization sweeps (e.g., via Wandb or similar frameworks), defining search spaces for learning rates, batch sizes, and network depths.
