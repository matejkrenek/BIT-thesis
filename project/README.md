# Bachelor Thesis

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Folder Structure](#folder-structure)
- [License](#license)

## Introduction

This repository contains the implementation of model for my bachelor thesis with custom dataset inherited from ShapeNet. Model is focused on repairing defects in 3D point clouds caused by photogrammetry process.

## Installation

Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Installation Conda
Create a conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate bit-thesis
```

## Usage

To run dataset visualization script, use the following command:

```bash
python ./src/main.py
```

## Dataset

To be added soon.

## Folder Structure

The repository is organized as follows:

```
.
├── data/                   # Dataset files (ignored in .gitignore)
├── notebooks/              # Jupyter notebooks for experiments (currently not working XD)
├── src/                    # Source code
│   ├── main.py             # Main script to run the project
│   ├── logger/             # Logger implementation
│   ├── dataset/            # Dataset implementation
│   │   ├── defect/         # Defect generation entities
│   │   └── downloader/     # Dataset downloading entities
│   └── visualize/          # Utility functions
│       └── viewer/         # Visualization entities
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
