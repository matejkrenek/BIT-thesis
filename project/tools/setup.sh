#!/bin/bash
# =============================================================================
# setup.sh — One-command environment setup for bit-thesis
#
# Usage:
#   bash setup.sh
#
# Requirements:
#   - conda installed and on PATH (or ~/miniconda3/bin/conda)
#   - Internet connection
# =============================================================================

set -e

# Resolve conda
if command -v conda &>/dev/null; then
    CONDA_EXE="conda"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/anaconda3/bin/conda"
else
    echo "ERROR: conda not found. Install Miniconda first."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="bit-thesis"

echo ""
echo "==> [1/2] Creating conda environment '${ENV_NAME}' ..."
echo ""

$CONDA_EXE env remove -n "$ENV_NAME" -y 2>/dev/null || true
$CONDA_EXE env create -f "$SCRIPT_DIR/environment.yml"

echo ""
echo "==> [2/2] Verifying installation ..."
echo ""

$CONDA_EXE run -n "$ENV_NAME" python -c "
import torch
print(f'  torch:      {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')

import pytorch3d
from pytorch3d import _C
print(f'  pytorch3d:  {pytorch3d.__version__} (CUDA OK)')
"

echo ""
echo "==> Done! Activate with:  conda activate ${ENV_NAME}"
echo ""
