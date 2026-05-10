#!/usr/bin/env bash

# Author: Matěj Křenek (xkrenem00)
# Contact: xkrenem00@vutbr.cz
# File: setup.sh
# Responsibility: Creates and verifies the project conda environment from environment.yml.
# Usage:
#   bash tools/setup.sh
# Requirements:
#   - conda installed and on PATH (or ~/miniconda3/bin/conda)
#   - Internet connection

set -euo pipefail

ENV_NAME="bit-thesis"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/environment.yml"
CONDA_EXE=""

die() {
    echo "ERROR: $*" >&2
    exit 1
}

resolve_conda() {
    if command -v conda >/dev/null 2>&1; then
        CONDA_EXE="conda"
        return
    fi
    if [[ -f "$HOME/miniconda3/bin/conda" ]]; then
        CONDA_EXE="$HOME/miniconda3/bin/conda"
        return
    fi
    if [[ -f "$HOME/anaconda3/bin/conda" ]]; then
        CONDA_EXE="$HOME/anaconda3/bin/conda"
        return
    fi

    die "conda not found. Install Miniconda first."
}

ensure_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        die "environment.yml not found at $ENV_FILE"
    fi
}

create_environment() {
    echo ""
    echo "==> [1/2] Creating conda environment '${ENV_NAME}' ..."
    echo ""

    "$CONDA_EXE" env remove -n "$ENV_NAME" -y 2>/dev/null || true
    "$CONDA_EXE" env create -f "$ENV_FILE"
}

verify_environment() {
    echo ""
    echo "==> [2/2] Verifying installation ..."
    echo ""

    "$CONDA_EXE" run -n "$ENV_NAME" python -c "
import torch
print(f'  torch:      {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')

import pytorch3d
from pytorch3d import _C
print(f'  pytorch3d:  {pytorch3d.__version__} (CUDA OK)')
"
}

main() {
    resolve_conda
    ensure_env_file
    create_environment
    verify_environment

    echo ""
    echo "==> Done! Activate with:  conda activate ${ENV_NAME}"
    echo ""
}

main "$@"
