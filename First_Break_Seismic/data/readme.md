# Data Directory (`First_Break_Seismic/data/`)

This directory serves as the designated storage repository for raw input HDF5 seismic assets downloaded from cloud sources as well as processed, chunked datasets prepared specifically for deep learning model loaders[cite: 1]. It handles multi-gigabyte real-world seismic data files that undergo rigorous structural auditing, coordinate sorting, amplitude scaling, and 3-class segmentation mask generation before being fed into U-Net neural network pipelines[cite: 1, 2, 3, 4].

## Data Files Overview

* **`Brunswick_orig_1500ms_V2.hdf5`**: Raw 15.57 GB seismic asset containing 4,496,540 traces and over 100 vendor SEGY header fields, which requires external label integration and coordinate-based sorting before model training[cite: 2].
* **`Halfmile3D_add_geom_sorted.hdf5`**: Raw 3.75 GB pre-sorted seismic asset comprising 1,099,559 traces and 690 unique shot gathers, featuring 89.49% valid pick coverage in the `SPARE1` field for U-Net mask generation[cite: 4].
* **`Lalor_raw_z_1500ms_norp_geom_v3.hdf5`**: Raw 15.67 GB land-seismic survey file consisting of 2,424,923 traces organized by `SHOTID`, utilizing `SPARE1` as the primary ground-truth label source since default break-time fields are unpopulated[cite: 3].
* **`preprocessed_Sudbury3D.hdf`**: 8.08 GB HDF5 dataset containing 1,810,220 traces and 1,001 time samples per trace under `TRACE_DATA/DEFAULT`, complete with source-receiver offsets, coordinates, processing histories, and first-break pick times[cite: 5].
* **`readme.md`**: Provides architectural documentation outlining data directory structures, dataset specifications, preprocessing requirements, and pipeline ingestion guidelines[cite: 1].
`
import kagglehub

# Download latest version
path = kagglehub.dataset_download("frmanxankiiyev/first-break-picking-data")

print("Path to dataset files:", path)
`
`
https://www.kaggle.com/datasets/frmanxankiiyev/first-break-picking-data
`
