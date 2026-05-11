# Repairing Photogrammetric 3D Reconstruction Defects Using Machine Learning

This document is a practical user manual for running the BIT Thesis project. It provides step-by-step instructions for setting up the environment, downloading necessary assets, and executing the main scripts for dataset preparation, model training, evaluation, and inference.

- **Name:** Repairing Photogrammetric 3D Reconstruction Defects Using Machine Learning
- **Author:** Matěj Křenek (xkrenem00)
- **Contact:** xkrenem00@vutbr.cz
- **Supervisor:** doc. Ing. Michal Španěl, Ph.D.
- **Institution:** Brno University of Technology, Faculty of Information Technology
- **Year:** 2026

## Table Of Contents

- [1. Project Overview](#1-project-overview)
- [2. System Requirements](#2-system-requirements)
- [3. Project Setup](#3-project-setup)
- [4. Download Pretrained Models And Evaluation Artifacts](#4-download-pretrained-models-and-evaluation-artifacts)
- [5. Dataset Download Behavior](#5-dataset-download-behavior)
- [6. Script Reference And Example Commands](#6-script-reference-and-example-commands)
- [7. Reference Workflow (Recommended)](#7-reference-workflow-recommended)
- [8. Project Folder Structure](#8-project-folder-structure)
- [9. Troubleshooting](#9-troubleshooting)
- [10. Reproducibility Notes](#10-reproducibility-notes)
- [11. Known Limitations](#11-known-limitations)
- [12. Future Work](#12-future-work)
- [13. Acknowledgements](#13-acknowledgements)

## 1. Project Overview

This thesis addresses the analysis and automatic repair of geometric defects arising during photogrammetric reconstruction of 3D objects. Such defects typically manifest as noise, missing geometry, or spurious points and artifacts in point clouds. The work investigates the potential of deep learning methods for point cloud processing to automate the repair process, comparing architectures across a broad spectrum of approaches from classical encoder-decoder networks to transformer-based models. Specifically, the models PCN, PoinTr, and AdaPoinTr are evaluated for missing geometry completion, and PointCleanNet for noise and outlier removal. A synthetic dataset simulating typical photogrammetric reconstruction defects is constructed from the ShapeNetCore database of 3D objects.~A~patch-based approach for direct application of completion models to dense point clouds was also investigated but proved non-functional due to the architectural dependence of the models on a global object representation, and remains a topic for future work. Results are evaluated quantitatively using Chamfer Distance, Hausdorff Distance, and Density-Aware Chamfer Distance at the level of complete objects as well as through segmented evaluation distinguishing repaired and preserved regions. Based on the experiments, AdaPoinTr was identified as the most suitable approach for reconstructing missing geometry, achieving the lowest error in the repaired region while best preserving the existing geometry. The denoising pipeline based on PointCleanNet substantially reduces the presence of spurious points and noise in the input point cloud.

**Focus of the work:**

- Analysis of photogrammetric reconstruction defects and their impact on downstream applications.
- Construction of a synthetic dataset simulating typical defects in photogrammetric point clouds.
- Evaluation of deep learning models for point cloud completion and denoising, including PCN, PoinTr, AdaPoinTr, and PointCleanNet.
- Quantitative evaluation using Chamfer Distance, Hausdorff Distance, and Density-Aware Chamfer Distance.
- Investigation of a patch-based approach for applying completion models to dense point clouds, and discussion of its limitations and future directions.

## 2. System Requirements

**Reference platform:** Linux (reference: Ubuntu)
**Reference server used for training and evaluation:** sophie1.fit.vutbr.cz (GPU: 4x NVIDIA RTX A5000 24 GB VRAM)

**Required tools and hardware:**

- Miniconda (or Conda-compatible installation)
- NVIDIA GPU (required for practical training speed)
- CUDA driver compatible with environment.yml packages

**Disk requirements:**

- Python/Conda environment and outputs: several GB
- Datasets and generated variants: plan for at least 100 GB free space
- Checkpoints and evaluation artifacts: additional space based on run count

## 3. Project Setup

### Step 1: Configure .env

Create local environment file from the template and edit values based on your setup. Training and evaluation scripts rely on environment variables for configuration of paths and `HF_TOKEN` is required for downloading ShapeNetCore dataset which is used as the base for synthetic dataset generation.

```bash
cp .env.example .env
```

### Step 2: Run setup script

```bash
bash ./tools/setup.sh
```

**This script:**

- resolves conda executable
- recreates the bit-thesis conda environment from environment.yml
- verifies key packages (including torch and pytorch3d)

After setup, follow instructions printed by the script (environment activation).

## 4. Download Pretrained Models And Evaluation Artifacts

Use the download utility to pull files from the configured Hugging Face bucket.

```bash
python ./src/download.py \
  --output-dir ./outputs
```

**Structure of the outputs directory after download:**

```text
outputs/
├── models/                        # Pretrained model weights and related artifacts
│   ├── pcn/
│   ├── pointr/
│   ├── adapointr/
│   └── pointcleannet/
├── eval/                          # Reference evaluation exports and sample reports used in text of the thesis generated from eval.py script
│   ├── completion_*/
│   ├── pipeline_*/
|   └── sample_patched_*/
├── dataset/                       #  Exported galleries/samples from dataset.py script
│   ├── grid_*/
└── └── sample_*/
```

## 5. Dataset Download Behavior

Datasets are fetched/prepared automatically when first needed by project scripts.
You do not need to run a separate mandatory "download dataset" command for basic usage.

**Important:**

- Keep approximately 100 GB free disk space before first full run.
- First access can take significant time due to download and preprocessing.

## 6. Script Reference And Example Commands

### 6.1 Dataset Pipeline And Visualization

**Script:** ./src/dataset.py

**Purpose:**

- visualize dataset samples
- generate gallery images
- compare dataset corruption modes

**Supported datasets:**

| Dataset  | Status      | Notes                                                 |
| -------- | ----------- | ----------------------------------------------------- |
| shapenet | main        | Default base dataset for creation of syntetic dataset |
| modelnet | alternative | Alternative for more diverse experiments              |

Modes:

| Mode     | Description                                                                                              |
| -------- | -------------------------------------------------------------------------------------------------------- |
| pure     | No synthetic defects, original samples only                                                              |
| basic    | Lighter defect pipeline (LocalDropouts, LargeMissingRegions)                                             |
| advanced | Stronger and more complex defect pipeline (Noise, OutlierPoints, SurfaceToPlaneBridge, BelowObjectPlane) |

Key parameters:

| Parameter                                | Meaning                                                         |
| ---------------------------------------- | --------------------------------------------------------------- |
| --dataset                                | Dataset selection: shapenet or modelnet                         |
| --mode                                   | Corruption mode: pure, basic, advanced                          |
| --num-samples                            | Number of random samples for gallery                            |
| --sample-indices                         | Explicit sample selection (overrides random sampling)           |
| --dense                                  | Use dense cloud variant where available                         |
| --open-viewer                            | Open interactive Polyscope viewer                               |
| --generate-images / --no-generate-images | Enable or disable gallery export                                |
| --output-dir, --run-name                 | Output location and run folder naming                           |
| --save-clouds-format                     | Optional export format for saving generated clouds (ply or npz) |

**Example:**

```bash
python ./src/dataset.py \
  --dataset shapenet \
  --mode advanced \
  --num-samples 8 \
  --sample-indices 0,3,5 \
  --dense \
  --open-viewer \
  --generate-images \
  --output-dir outputs/dataset
  --save-clouds-format ply
```

**For full details, run:**

```bash
python ./src/dataset.py --help
```

### 6.2 Model Training Workflow

Script: ./src/train.py

Purpose:

- unified training entry point (pcn, pointr, adapointr)
- checkpointing, loss curves, summaries
- resume and finetuning support
- multi-GPU support
- optional training notifications (via Discord webhooks)
- default training configuration is used for all models so no need to specify model configuration and hyperparameters for training

**Key parameters:**

| Parameter                                   | Meaning                                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------ |
| --model                                     | Completion model: pcn, pointr, adapointr                                       |
| --target-dataset                            | Dataset selection: shapenet (main) or modelnet (alternative)                   |
| --dataset-variant                           | Dataset variant: basic or advanced                                             |
| --epochs, --batch-size                      | Core training schedule and batch sizing                                        |
| --learning-rate, --weight-decay             | Optimization hyperparameters (Optional)                                        |
| --output-dir, --run-name                    | Output location and run folder naming                                          |
| --resume-checkpoint / --finetune-checkpoint | Continue run or initialize from existing weights                               |
| --overfit, --overfit-samples                | Debugging overfitting mode with specified limited samples to check convergence |

**Example:**

```bash
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py \
  --model pcn \
  --dataset-variant basic \
  --target-dataset shapenet \
  --epochs 100 \
  --batch-size 64 \
  --output-dir outputs \
  --run-name pcn
```

**For full details, run:**

```bash
python ./src/train.py --help
```

### 6.3 Evaluation And Metrics Reporting

**Script:** ./src/eval.py

**Purpose:**

- evaluate one or more trained models
- compute metrics (chamfer, hausdorff, dcd)
- export per-sample and aggregate tables
- generate evaluation galleries

**Scenarios:**

| Scenario | Description                                                                               |
| -------- | ----------------------------------------------------------------------------------------- |
| a        | Basic evaluation on original samples with basic corruption (for quick reference)          |
| b        | Evaluation on advanced corruption with dense data (for detailed analysis)                 |
| c        | Segmented evaluation distinguishing repaired and preserved regions (for in-depth insight) |

Key parameters:

| Parameter                       | Meaning                                                            |
| ------------------------------- | ------------------------------------------------------------------ |
| --dataset                       | Dataset selection: shapenet (main) or modelnet (alternative)       |
| --mode                          | Evaluation corruption mode: basic or advanced                      |
| --model-spec                    | Repeatable model definition in format name:type:checkpoint         |
| --scenario                      | Scenario selector: a, b, c, or all                                 |
| --metrics                       | Comma-separated metric list (for example chamfer,hausdorff,dcd)    |
| --num-samples, --sample-indices | Random count or explicit sample list                               |
| --dense, --dense-root           | Optional dense data usage and location (Use for scenarios b and c) |
| --output-dir, --run-name        | Output location and run folder naming                              |
| --denoise-model-checkpoint      | Optional checkpoints for pipeline evaluation scenario (b and c)    |
| --denoise-params-checkpoint     | Optional checkpoints for pipeline evaluation scenario (b and c)    |
| --outlier-model-checkpoint      | Optional checkpoints for pipeline evaluation scenario (b and c)    |

**Example:**

```bash
CUDA_VISIBLE_DEVICES=0 python ./src/eval.py \
  --dataset shapenet \
  --mode advanced \
  --model-spec AdaPoinTr:adapointr:outputs/models/adapointr/checkpoints/best.pth \
  --scenario c \
  --metrics chamfer,hausdorff,dcd \
  --num-samples 6 \
  --output-dir eval \
  --run-name shapenet_advanced_eval \
  --dense \
  --denoise-model-checkpoint outputs/models/pointcleannet/checkpoints/best_denoise.pth \
  --denoise-params-checkpoint outputs/models/pointcleannet/checkpoints/best_denoise_params.pth \
  --outlier-model-checkpoint outputs/models/pointcleannet/checkpoints/best_outliers.pth
```

**For full details, run:**

```bash
python ./src/eval.py --help
```

### 6.4 Single-Cloud Inference Pipeline

**Script:** ./src/infer.py

**Purpose:**

- run standalone pipeline inference for one point cloud
- optional visualization of pipeline stages

**Key parameters:**

| Parameter                                                                  | Meaning                                      |
| -------------------------------------------------------------------------- | -------------------------------------------- |
| --input                                                                    | Input cloud path (.npz or .ply)              |
| --output, --report-json                                                    | Output cloud and JSON report paths           |
| --completion-model                                                         | Completion backend: pcn, pointr, adapointr   |
| --run-outlier-before, --run-denoise, --run-outlier-after, --run-completion | Enable or disable individual pipeline stages |
| --denoise-checkpoint, --outlier-checkpoint, --completion-checkpoint        | Required model checkpoints                   |
| --visualize                                                                | Open pipeline result visualization           |

**Example:**

```bash
CUDA_VISIBLE_DEVICES=0 python ./src/infer.py \
  --input /path/to/input_cloud.npz \
  --denoise-checkpoint outputs/models/pointcleannet/checkpoints/best_denoise.pth \
  --denoise-params-checkpoint outputs/models/pointcleannet/checkpoints/best_denoise_params.pth \
  --outlier-checkpoint outputs/models/pointcleannet/checkpoints/best_outliers.pth \
  --completion-checkpoint outputs/models/adapointr/checkpoints/best.pth \
  --completion-model adapointr \
  --output /path/to/output_cloud.npz \
  --visualize
```

For full details, run:

```bash
python ./src/infer.py --help
```

### 6.5 Patch-Based Reconstruction Workflow

**Script:** ./src/patchbased.py

**Purpose:**

- inspect patch decomposition
- reassemble from patches
- run patch completion and merge results

**Actions:**

| Action                  | Description                                                              |
| ----------------------- | ------------------------------------------------------------------------ |
| visualize_patches       | Visualize patch decomposition of a sample cloud in Polyscope viewer      |
| reassemble              | Reassemble original cloud from patches and visualize in Polyscope viewer |
| complete_and_reassemble | Run patch-based completion and reassembly workflow, visualize results,   |

**Key parameters:**

| Parameter                        | Meaning                                                                 |
| -------------------------------- | ----------------------------------------------------------------------- |
| --action                         | Workflow action: visualize_patches, reassemble, complete_and_reassemble |
| --dataset                        | Dataset selection: shapenet (main) or modelnet (alternative)            |
| --mode                           | Corruption mode: pure, basic, advanced                                  |
| --sample-index                   | Selected sample id                                                      |
| --dense, --dense-root            | Dense data usage and location                                           |
| --model-spec                     | Model definition (required for complete_and_reassemble)                 |
| --open-viewer, --generate-images | Visualization and output generation control                             |
| --output-dir, --run-name         | Output location and run folder naming                                   |

**Example:**

```bash
CUDA_VISIBLE_DEVICES=0 python ./src/patchbased.py \
  --action complete_and_reassemble \
  --dataset shapenet \
  --mode advanced \
  --sample-index 0 \
  --model-spec AdaPoinTr:adapointr:outputs/models/adapointr/checkpoints/best.pth \
  --generate-images \
  --output-dir outputs/patchbased
```

**For full details, run:**

```bash
python ./src/patchbased.py --help
```

### 6.6 Remote Asset Downloading

**Script:** ./src/download.py

**Purpose:**

- download remote assets from Hugging Face bucket storage

**Key parameters:**

| Parameter                           | Meaning                                |
| ----------------------------------- | -------------------------------------- |
| --source                            | Remote bucket source path              |
| --output-dir                        | Local destination directory            |
| --allow-patterns, --ignore-patterns | Include/exclude filtering patterns     |
| --force-download                    | Overwrite already existing local files |

**Minimal example:**

```bash
python ./src/download.py --output-dir ./outputs
```

**For full details, run:**

```bash
python ./src/download.py --help
```

## 7. Reference Workflow (Recommended)

1. Setup environment.
2. Download pretrained assets.
3. Generate a quick dataset gallery.
4. Train one baseline model.
5. Run evaluation on that checkpoint.
6. Run optional single-file inference.

**Suggested command sequence:**

```bash
cp .env.example .env
bash ./tools/setup.sh
python ./src/download.py --output-dir ./outputs
python ./src/dataset.py --dataset shapenet --mode advanced --num-samples 6 --generate-images
python ./src/train.py --model pcn --dataset-variant advanced --target-dataset shapenet --run-name pcn
python ./src/eval.py --dataset shapenet --mode basic --model-spec pcn:pcn:outputs/pcn/checkpoints/best.pt --scenario a
```

## 8. Project Folder Structure

Following section provides a high-level overview of the project folder structure and its key components.

```text
.
├── src/
│   ├── core/
│   ├── dataset/
│   │   ├── wrapper/              # Dataset wrappers (normalization, patching, augmentation stages)
│   │   ├── defect/               # Synthetic defect generators used for corruption pipelines
│   │   ├── downloader/           # Download backends for dataset/model assets
│   │   ├── shapenet.py           # ShapeNet dataset implementation
│   │   └── modelnet.py           # ModelNet dataset implementation
│   ├── metrics/                  # Metric implementations and evaluation utilities
│   ├── models/                   # Model architectures and utilities
│   ├── notifications/            # Optional notification utilities (for Discord webhooks)
│   ├── visualize/                # Visualization helpers
│   ├── dataset.py                # Dataset preparation, corruption and visualization script
│   ├── train.py                  # Main training script with unified interface for all completion models
│   ├── eval.py                   # Main evaluation script with metric computation and reporting
│   ├── infer.py                  # Single-cloud inference pipeline with optional stages
│   ├── patchbased.py             # Patch decomposition/reassembly and patch-based completion workflow
│   └── download.py               # Downloading pretrained models and evaluation artifacts from Hugging Face bucket
├── tools/
│   ├── setup.sh                  # Environment setup script
│   └── download_from_ssh.sh      # Optional script for SSH-based downloads from remote server
├── environment.yml               # Conda environment specification
├── .env.example                  # Template for local environment configuration
├── README.md                     # Project overview and user guide
```

## 9. Troubleshooting

**Common issues:**

- Conda not found:
  - install Miniconda and ensure conda is available in PATH and installed correctly based on guide outputed by setup.sh
- CUDA mismatch:
  - verify GPU driver compatibility with packages in environment.yml
- Slow first run:
  - expected during first dataset download/preprocessing
- Missing checkpoints:
  - verify download path and model-spec checkpoint paths using download.py

## 10. Reproducibility Notes

- Use explicit --seed values in scripts.
- Keep run names stable for easier comparison.
- Save command lines used for each experiment.
- Store checkpoints and CSV metrics together per run.

## 11. Known Limitations

- Patch-based completion workflow is non-functional due to architectural dependence of models on global object representation. This remains a topic for future work.
- Evaluation is currently limited to synthetic datasets and may not fully capture performance on real photogrammetric scans. Adjusting the dataset and evaluation pipeline for real-world data is a key area for future improvement.
- Used completion models are invariant to rotation and do not leverage color or normal information, which could potentially enhance performance if incorporated.

## 12. Future Work

**Future additions:**

- adjust the inference pipeline to be usable on real photogrammetric scans (currently focused on synthetic data)
- adjusting the completion models to be able to apply them patch-based on large real scans (currently focused on small synthetic point clouds of 8192 points)
- adding new defects to dataset pipeline based on iterative adjustment of inference pipeline for real scans

## 13. Acknowledgements

- [PointNet](https://arxiv.org/abs/1612.00593) and [PointNet++](https://arxiv.org/abs/1706.02413) for foundational point cloud processing concepts.
- [PCN](https://arxiv.org/abs/1808.00671),[PoinTr](https://arxiv.org/abs/2108.08839), and [AdaPoinTr](https://arxiv.org/abs/2301.04545) for model architectures and training strategies.
- [ShapeNet](https://www.shapenet.org/) and [ModelNet](http://modelnet.cs.princeton.edu/) for dataset resources.
- Hugging Face for hosting pretrained models and evaluation artifacts.
- [PointCleanNet](https://arxiv.org/abs/1901.01060) for denoising and outlier filtering models.
- https://github.com/mrakotosaon/pointcleannet for reference implementations and checkpoints of PointCleanNet models.
- https://github.com/yuxumin/PoinTr for reference implementations of PoinTr, AdaPoinTr and PCN.
