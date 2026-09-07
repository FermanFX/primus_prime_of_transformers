# HDF5 Seismic Dataset

A memory-efficient PyTorch `Dataset` implementation for loading seismic shot data lazily from an HDF5 file.

The dataset is designed for seismic machine learning workflows where the complete dataset may be too large to load into RAM. Instead of loading all shots during initialization, each shot is read from the HDF5 file only when it is requested.

## Features

* **Lazy loading** — seismic data is loaded on demand using `__getitem__`.
* **Memory efficient** — the entire HDF5 dataset does not need to fit into RAM.
* **PyTorch compatible** — implements `torch.utils.data.Dataset`.
* **Automatic padding/cropping** — every shot is converted to a fixed number of traces.
* **Pick-based mask generation** — creates a target mask from seismic pick information stored in `SPARE1`.
* **HDF5 SWMR support** — the file is opened using `swmr=True`.
* **Level-based logging** — uses Loguru for `INFO` and `DEBUG` telemetry.
* **Explicit resource management** — provides a `close()` method for closing the HDF5 file.

---

## Dataset Overview

The `HDF5SeismicDataset` class expects seismic data to be stored in an HDF5 file.

Each dataset item represents a single **shot** and contains:

1. Seismic trace data
2. A mask generated from the corresponding pick values

The returned tensors have the following shapes:

```text
Data:
(1, target_traces, n_samples)

Mask:
(target_traces, n_samples)
```

For the default configuration:

```text
Data  → (1, 1578, 751)
Mask  → (1578, 751)
```

---

## Expected HDF5 Structure

The implementation expects the following HDF5 hierarchy:

```text
HDF5 file
└── TRACE_DATA
    └── DEFAULT
        ├── data_array
        └── SPARE1
```

### `data_array`

Contains the seismic trace samples.

Expected conceptual shape:

```text
(number_of_traces, number_of_samples)
```

For example:

```text
(1578, 751)
```

Each row corresponds to one seismic trace, while each column represents a time/depth sample.

### `SPARE1`

Contains pick information associated with each trace.

The implementation reads:

```python
SPARE1[start_idx:end_idx, 0]
```

Therefore, the first column of `SPARE1` is interpreted as the pick position for each trace.

---

## Shot Indexing

The dataset receives a `shot_indices` dictionary:

```python
shot_indices: dict[int, tuple[int, int]]
```

Each shot ID maps to a start and end index:

```python
shot_indices = {
    1001: (0, 1578),
    1002: (1578, 3156),
    1003: (3156, 4734),
}
```

The tuple represents the corresponding slice in the HDF5 datasets:

```python
data_array[start_idx:end_idx, :]
SPARE1[start_idx:end_idx, 0]
```

The `shot_ids` list determines the order in which shots are exposed by the PyTorch dataset.

---

## Installation

Install the required dependencies:

```bash
pip install h5py numpy torch loguru
```

Or add them to your project's dependency manager.

---

## Basic Usage

```python
from torch.utils.data import DataLoader

dataset = HDF5SeismicDataset(
    hdf5_path="seismic_data.h5",
    shot_indices=shot_indices,
    shot_ids=shot_ids,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

for data, mask in loader:
    print(data.shape)
    print(mask.shape)
```

With the default configuration, a batch of four shots will have approximately:

```text
data → (4, 1, 1578, 751)
mask → (4, 1578, 751)
```

---

## Constructor

```python
HDF5SeismicDataset(
    hdf5_path: str,
    shot_indices: dict[int, tuple[int, int]],
    shot_ids: list,
    target_traces: int = 1578,
    n_samples: int = 751,
    strip_width: int = 8,
)
```

### Parameters

| Parameter       | Type                         | Default | Description                                    |
| --------------- | ---------------------------- | ------: | ---------------------------------------------- |
| `hdf5_path`     | `str`                        |       — | Path to the HDF5 file                          |
| `shot_indices`  | `dict[int, tuple[int, int]]` |       — | Mapping from shot ID to HDF5 start/end indices |
| `shot_ids`      | `list`                       |       — | List of shot IDs exposed by the dataset        |
| `target_traces` | `int`                        |  `1578` | Fixed number of traces returned per shot       |
| `n_samples`     | `int`                        |   `751` | Number of samples expected for each trace      |
| `strip_width`   | `int`                        |     `8` | Width of the target region around each pick    |

---

## Lazy Loading

The HDF5 file is **not opened during dataset initialization**.

Instead, it is opened the first time `__getitem__()` is called:

```python
if self.file is None or self.group is None:
    self.file = h5py.File(
        self.hdf5_path,
        "r",
        swmr=True,
    )
    self.group = self.file["TRACE_DATA"]["DEFAULT"]
```

This approach provides two important benefits:

* Dataset initialization is lightweight.
* Large HDF5 files do not need to be loaded entirely into memory.

Only the requested shot is read:

```python
shot_data = self.group["data_array"][start_idx:end_idx, :]
shot_picks = self.group["SPARE1"][start_idx:end_idx, 0]
```

---

## Padding and Cropping

Machine learning models often require inputs with a fixed shape.

The dataset therefore normalizes every shot to exactly:

```text
target_traces × n_samples
```

### When a shot has fewer traces

If:

```text
actual_traces < target_traces
```

the dataset creates zero-filled arrays:

```python
data_padded = np.zeros(
    (target_traces, n_samples),
    dtype=np.float32,
)

picks_padded = np.zeros(
    target_traces,
    dtype=np.float32,
)
```

The available traces are copied into the beginning of the arrays.

Conceptually:

```text
Original:

[ trace 1 ]
[ trace 2 ]
[ trace 3 ]
[ ...     ]

After padding:

[ trace 1 ]
[ trace 2 ]
[ trace 3 ]
[ ...     ]
[ zero    ]
[ zero    ]
[ ...     ]
```

### When a shot has more traces

If:

```text
actual_traces > target_traces
```

only the first `target_traces` traces are retained:

```python
shot_data = shot_data[:target_traces, :]
shot_picks = shot_picks[:target_traces]
```

This guarantees a consistent input size.

---

## Pick-Based Mask Generation

The `_create_mask()` method converts the pick values into a segmentation-style mask.

For every trace, the pick determines where the mask changes.

The mask initially contains zeros:

```text
0 = background / before pick
```

For a valid pick, a narrow region around the pick is assigned:

```text
2 = pick/target region
```

Everything after that region is assigned:

```text
1 = post-pick region
```

### Mask labels

| Value | Meaning                  |
| ----: | ------------------------ |
|   `0` | Before pick / background |
|   `1` | After the pick region    |
|   `2` | Pick region              |

The width of the pick region is controlled by:

```python
strip_width
```

With the default:

```python
strip_width = 8
```

the half-width is:

```python
half_width = strip_width // 2
```

The pick is rounded to the nearest integer:

```python
pick_int = round(pick)
```

and the target region is generated using:

```python
start = max(0, pick_int - self.half_width)
end = min(self.n_samples, pick_int + self.half_width + 1)
```

Then:

```python
mask[i, start:end] = 2
mask[i, end:] = 1
```

---

## Invalid Picks

Pick values are ignored when they are outside the valid sample range:

```python
if pick <= 0 or pick >= self.n_samples:
    continue
```

For these traces, the mask remains entirely zero.

This is particularly useful when zero or out-of-range values are used to indicate missing picks.

---

## Returned Values

`__getitem__()` returns:

```python
(
    torch.Tensor,
    torch.Tensor,
)
```

Specifically:

```python
data, mask = dataset[idx]
```

### Data

The seismic data is converted to a PyTorch tensor:

```python
torch.from_numpy(shot_data).float().unsqueeze(0)
```

Shape:

```text
(1, target_traces, n_samples)
```

The additional first dimension represents the input channel.

### Mask

The generated NumPy mask is converted to:

```python
torch.from_numpy(mask).long()
```

Shape:

```text
(target_traces, n_samples)
```

The `long` dtype is suitable for many PyTorch classification/segmentation losses.

---

## Logging

The dataset uses [Loguru](https://github.com/Delgan/loguru) for telemetry.

### INFO level

Dataset initialization is logged:

```text
[HDF5] INIT: 100 shots, file=seismic_data.h5
```

### DEBUG level

Additional information is available at debug level:

```text
[HDF5] target_traces=1578, n_samples=751, strip_width=8
```

Every requested shot is also logged:

```text
[HDF5] GET idx=10 → shot=1001, slice=1578:3156
```

The HDF5 file opening is logged as well:

```text
[HDF5] Opening HDF5 file: seismic_data.h5
```

To enable debug logging:

```python
from loguru import logger

logger.enable("__main__")
```

Or configure Loguru according to your application's logging setup.

---

## File Management

The dataset keeps the HDF5 file open after the first access and reuses the same file handle for subsequent requests.

The file can be explicitly closed:

```python
dataset.close()
```

The implementation also attempts to close the file when the dataset object is destroyed:

```python
def __del__(self) -> None:
    self.close()
```

---

## Getting a Shot ID

The original shot ID corresponding to a dataset index can be retrieved with:

```python
shot_id = dataset.get_shot_id(idx)
```

Example:

```python
shot_id = dataset.get_shot_id(10)

print(f"Dataset index 10 corresponds to shot {shot_id}")
```

This is useful when connecting model predictions back to the original seismic acquisition metadata.

---

## Data Flow

The complete data flow can be summarized as:

```text
                    HDF5 File
                       │
                       ▼
              TRACE_DATA / DEFAULT
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        data_array             SPARE1
             │                   │
             │                   ▼
             │              Pick values
             │                   │
             ▼                   ▼
       Shot extraction     Mask generation
             │                   │
             ▼                   ▼
      Padding / cropping     0 / 1 / 2 mask
             │                   │
             └─────────┬─────────┘
                       ▼
                PyTorch tensors
                       │
                       ▼
                  DataLoader
                       │
                       ▼
                    Model
```

---

## Example: Inspecting One Shot

```python
data, mask = dataset[0]

print("Data shape :", data.shape)
print("Mask shape :", mask.shape)
print("Data dtype :", data.dtype)
print("Mask dtype :", mask.dtype)
```

Expected output:

```text
Data shape : torch.Size([1, 1578, 751])
Mask shape : torch.Size([1578, 751])
Data dtype : torch.float32
Mask dtype : torch.int64
```

---

## Example Training Loop

The dataset can be used directly with a standard PyTorch training loop:

```python
from torch.utils.data import DataLoader

dataset = HDF5SeismicDataset(
    hdf5_path="seismic_data.h5",
    shot_indices=shot_indices,
    shot_ids=shot_ids,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
)

for data, mask in loader:
    data = data.to(device)
    mask = mask.to(device)

    prediction = model(data)

    loss = criterion(prediction, mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Multi-Worker DataLoader Considerations

When using:

```python
DataLoader(
    dataset,
    num_workers > 0,
)
```

PyTorch may create separate worker processes.

Because HDF5 file handles should generally not be shared across forked worker processes, it is recommended that each worker opens its own HDF5 handle.

The current implementation opens the file lazily, which is helpful for this pattern because the file is not opened during dataset construction.

For example:

```python
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
)
```

Depending on the operating system and multiprocessing configuration, additional worker-specific HDF5 handling may be desirable for production workloads.

---

## Design Goals

The implementation focuses on three main goals:

### 1. Low memory usage

Only one shot is loaded at a time instead of loading the entire seismic dataset into memory.

### 2. Fixed model input size

Different shot sizes are normalized using padding or cropping.

### 3. Reproducible target generation

The mask is deterministically generated from the pick values stored in the HDF5 file.

---

## Requirements

Python packages:

```text
Python 3.x
h5py
numpy
torch
loguru
```

---

## Class API

### `HDF5SeismicDataset`

```python
class HDF5SeismicDataset(Dataset):
```

### Methods

#### `__len__()`

Returns the number of available shots.

```python
len(dataset)
```

#### `__getitem__(idx)`

Loads and returns one shot and its generated mask.

```python
data, mask = dataset[idx]
```

#### `_create_mask(picks)`

Creates the integer mask from the seismic pick positions.

#### `close()`

Closes the open HDF5 file.

```python
dataset.close()
```

#### `get_shot_id(idx)`

Returns the original shot ID for a dataset index.

```python
dataset.get_shot_id(idx)
```

---

## Summary

`HDF5SeismicDataset` provides a simple interface for using large seismic HDF5 datasets with PyTorch without loading the complete dataset into memory.

Its main processing pipeline is:

```text
HDF5
 │
 ├── data_array ──► lazy shot loading
 │
 └── SPARE1 ──────► pick extraction
                       │
                       ▼
                  mask creation
                       │
                       ▼
              padding / cropping
                       │
                       ▼
                 PyTorch tensors
```

This makes the class suitable for seismic segmentation, horizon/pick detection, and other machine-learning workflows where seismic shots and their pick-derived targets are stored in HDF5.
