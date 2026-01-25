# Neural Methods for Scene Reconstruction and Novel View Synthesis

A 3D Gaussian Splatting pipeline for reconstructing scenes and generating novel views from custom image datasets.

## Requirements

- Python 3.8+
- CUDA-capable GPU
- FFmpeg
- COLMAP
- ImageMagick

### Python Dependencies

```
torch
plyfile
tqdm
pillow
opencv-python
numpy
matplotlib
```

## Quick Start

### 1. Clone the Gaussian Splatting Repository

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
pip install plyfile tqdm pillow opencv-python
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### 2. Prepare Your Custom Dataset

**Option A: From Images**

```bash
python prepare_data.py --input /path/to/images --output /path/to/dataset --type photos
```

**Option B: From Video**

```bash
python prepare_data.py --input /path/to/video.mp4 --output /path/to/dataset --type video --fps 2
```

### 3. Train the Model

```bash
python train.py -s /path/to/dataset -m /path/to/output/model --iterations 7000 --eval
```

For depth-guided training (if depth maps available):

```bash
python train.py -s /path/to/dataset -d /path/to/depth --eval --iterations 7000
```

### 4. Render Novel Views

```bash
python render.py -m /path/to/output/model --skip_train --iteration 7000
```

### 5. Evaluate Results

```bash
python metrics.py -m /path/to/output/model
```

## Expected Output Structure

```
output/
├── model/
│   ├── point_cloud/
│   │   └── iteration_7000/
│   │       └── point_cloud.ply
│   ├── cameras.json
│   └── results.json
└── test/
    └── ours_7000/
        ├── renders/
        └── gt/
```

## Evaluation Metrics

- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)

## Dataset Requirements

For optimal results:

- Minimum 50-100 images covering the scene from multiple angles
- Consistent lighting conditions
- Sufficient overlap between consecutive views
- Images should be in JPG or PNG format

## Troubleshooting

| Issue | Solution |
|-------|----------|
| COLMAP fails | Ensure sufficient image overlap and feature-rich scenes |
| Out of memory | Reduce resolution with `--resolution 2` or `--resolution 4` |
| Poor reconstruction | Increase image count or improve coverage |

## Dataset link for nerf
https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset