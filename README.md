# GeoSAM-Image-Encoder

A standalone Python package for encoding remote sensing images into feature representations compatible with [Geo-SAM](https://github.com/coolzhao/Geo-SAM). This fork includes a critical bug fix not present in the official distribution and removes the QGIS dependency. Encoded features remain compatible with QGIS for interactive labeling via the Geo-SAM plugin.

## Key Features

* Standalone Python implementation without QGIS dependencies
* Compatible with Geo-SAM for QGIS integration
* Includes critical bug fixes not in official distribution
* Supports GPU acceleration for efficient processing
* Multi-GPU support for batch processing
* Produces .npz feature files for seamless Geo-SAM integration

## Installation

### Core Package

```bash
pip install git+https://github.com/awi-response/GeoSAM-Image-Encoder.git
```

### GPU Acceleration (Recommended)

For optimal performance, install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

Ensure CUDA Toolkit is installed on your system.

## Usage

### Prerequisites

1. Download the Segment Anything Model (SAM) checkpoint file (`sam_vit_l_0b3195.pth`) following the Geo-SAM official documentation.

### Basic Example

```python
import os
import glob
import torch
import time
import numpy as np
import rasterio
import geosam
from geosam import ImageEncoder
from tqdm import tqdm
from multiprocessing import Process, current_process

# Config
checkpoint_path = '/path/to/checkpoint.pth'  # Path to the SAM checkpoint
image_folder = '/path/to/images'  # Path to the folder containing images
feature_dir = '/path/to/encodings'  # Path to save the encoded features

GPUS = [0, 1, 2, 3]

def encode_images_on_gpu(gpu_id, image_paths):
    torch.cuda.set_device(gpu_id)
    print(f"[{current_process().name}] Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    # Tell ImageEncoder which GPU to use
    img_encoder = ImageEncoder(checkpoint_path, gpu_id=gpu_id)

    for image_path in tqdm(image_paths, desc=f"GPU {gpu_id}"):
        try:
            start = time.time()
            img_encoder.encode_image(str(image_path), feature_dir)
            print(f"[GPU {gpu_id}] Encoded {os.path.basename(image_path)} in {time.time() - start:.2f}s")
        except Exception as e:
            print(f"[GPU {gpu_id} ERROR] Failed on {image_path}: {e}")


def chunk_list(lst, n):
    """Split list `lst` into `n` roughly equal chunks"""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

if __name__ == "__main__":
    # Get all .tif image paths
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.tif")))
    print(f"[INFO] Found {len(image_paths)} images in {image_folder}")
    print(f"[INFO] Available GPUs: {len(GPUS)}")

    if len(GPUS) == 0:
        raise RuntimeError("No CUDA GPUs available.")

    # Divide images among GPUs
    chunks = chunk_list(image_paths, len(GPUS))

    # Launch one process per GPU
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        p = Process(target=encode_images_on_gpu, args=(gpu_id, chunk))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("[INFO] All encoding complete.")
```

# Usage in QGis



### Script Parameters

* **checkpoint_path**: Path to the downloaded SAM checkpoint file
* **image_folder**: Directory containing input .tif images
* **feature_dir**: Output directory for saving SAM-encoded .npz features
* **GPUS**: List of available GPU IDs on your machine

## Output Format

Each processed .tif file generates a corresponding .npz file in the specified output folder. These files contain the SAM-encoded features and are directly compatible with Geo-SAM in QGIS for interactive segmentation and labeling.

## Important Notes

📝 Ensure your input images are in .tif format.

⚠️ This fork addresses a critical bug present in the official repository that could prevent encoding completion.

🔍 The encoded features can be imported into QGIS using the Geo-SAM plugin for interactive labeling workflows.