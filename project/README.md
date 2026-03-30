# BIT Thesis - Point Cloud Reconstruction

This repository contains the code for a bachelor thesis focused on 3D point cloud defect simulation and reconstruction.
It includes dataset preparation and visualization tools, model training scripts, evaluation scripts, and checkpoints.

## Table Of Contents

- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Run Dataset Visualization](#run-dataset-visualization)
- [Run Training](#run-training)
- [Project Structure](#project-structure)
- [License](#license)

## System Requirements

- OS: Linux (primary target)
- Python: 3.10
- Conda or Miniconda: recommended
- GPU: NVIDIA GPU recommended for training
- CUDA driver: compatible with CUDA 11.8 (for `pytorch-cuda=11.8` in `environment.yml`)
- Disk space:
  - Several GB for dependencies
  - Large additional space for datasets and generated outputs

Optional but useful:

- `HF_TOKEN` environment variable for ShapeNet download tooling
- Kaggle credentials for ModelNet download tooling
- `DISCORD_WEBHOOK_URL` for training notifications

## Environment Setup

### Option A: One-command setup

```bash
bash setup.sh
conda activate bit-thesis
```

### Option B: Manual Conda setup

```bash
conda env create -f environment.yml
conda activate bit-thesis
```

### Environment variables

Recommended is to create a `.env` file in the project root with the structure provided in `.env.example`.

Important is to set `ROOT_DIR` to the project root so scripts can resolve data/output paths consistently:

```bash
export ROOT_DIR="$(pwd)"
```

## Dataset Preparation

Expected default layout:

```text
data/
	ShapeNetV2/
		raw/
		processed/
	ModelNet40/
		raw/
		processed/
	ShapeNetV2_dense/
```

Notes:

- Datasets are processed on first use by dataset loaders.
- Dense point clouds are generated on demand when `--dense` is enabled.

## Run Dataset Visualization

Main script:

```bash
python ./src/dataset.py --dataset shapenet --mode advanced --generate-images
```

Useful examples:

1. Generate image gallery only:

```bash
python ./src/dataset.py \
	--dataset shapenet \
	--mode basic \
	--num-samples 8 \
	--output-dir outputs/dataset
```

2. Open interactive Polyscope viewer only:

```bash
python ./src/dataset.py \
	--dataset shapenet \
	--mode advanced \
	--open-viewer \
	--no-generate-images
```

3. Viewer + gallery generation:

```bash
python ./src/dataset.py \
	--dataset modelnet \
	--mode advanced \
	--open-viewer \
	--generate-images \
	--output-dir outputs/dataset \
	--output-name modelnet_advanced_gallery.png
```

Important flags:

- `--mode {basic,advanced}`: defect pipeline selection
- `--open-viewer`: open interactive Polyscope sample viewer
- `--generate-images` / `--no-generate-images`: control gallery export
- `--output-dir`: output folder for generated gallery image
- `--output-name`: custom output filename

## Run Training

Recommended script:

```bash
python ./src/train.py --model pcn
```

### PCN training example

```bash
python ./src/train.py \
	--model pcn \
	--dataset-variant advanced \
	--epochs 100 \
	--batch-size 64 \
	--learning-rate 1e-3 \
	--output-dir outputs \
	--run-name pcn_advanced_run
```

### PoinTr training example

```bash
python ./src/train.py \
	--model pointr \
	--dataset-variant advanced \
	--epochs 100 \
	--batch-size 32 \
	--learning-rate 3e-4 \
	--weight-decay 1e-4 \
	--output-dir outputs \
	--run-name pointr_advanced_run
```

### Resume from checkpoint

```bash
python ./src/train.py \
	--model pcn \
	--resume-checkpoint outputs/pcn_advanced_run/checkpoints/best.pt
```

### Multi-GPU selection

Use one of these options:

```bash
python ./src/train.py --model pcn --num-gpus 2
```

```bash
python ./src/train.py --model pcn --gpu-ids 0,1
```

Training outputs:

- Run directory with checkpoints
- `loss_curve.png`
- `training_summary.json`

Legacy model-specific scripts still exist in `src/train_pcn.py` and `src/train_pointr.py`, but `src/train.py` is the recommended unified entrypoint.

## Project Structure

```text
.
├── checkpoints/                # Stored model checkpoints
├── data/                       # Datasets (raw, processed, dense variants)
├── outputs/                    # Training and visualization outputs
├── src/
│   ├── core/                   # Core APIs (args, bootstrap, datasets, models, logging)
│   ├── dataset/                # Dataset loaders, defects, wrappers, download helpers
│   ├── metrics/                # Evaluation metrics (Chamfer, F-score, Hausdorff)
│   ├── models/                 # Reconstruction models (PCN, PoinTr, etc.)
│   ├── notifications/          # Discord notifier integration
│   ├── visualize/              # Gallery rendering and viewer components
│   ├── dataset.py              # Dataset visualization CLI
│   ├── train.py                # Unified training CLI
│   ├── train_pcn.py            # Legacy PCN-specific training script
│   ├── train_pointr.py         # Legacy PoinTr-specific training script
│   ├── eval_pcn.py             # PCN evaluation script
│   └── eval_pointr.py          # PoinTr evaluation script
├── environment.yml             # Conda environment definition
├── setup.sh                    # Automated environment setup script
└── README.md
```

## License

This project is licensed under the GPL-3.0 License.
See [LICENSE](LICENSE) for details.
