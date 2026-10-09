#!/bin/bash
#SBATCH --job-name=nlu_ft
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=runs/logs/%x_%j.log
# Fine-tune on one GPU (an en+ur run takes ~6 min on an A100). Submit from the repo root:
#   sbatch jobs/train.sbatch --name en_ur --train-locales en-US,ur-PK
# The encoder must already be in the Hugging Face cache, because compute nodes may have no internet.
set -euo pipefail
# site-specific: load a Python 3.12 module here if your cluster needs one
unset PYTHONPATH
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
.venv/bin/python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
.venv/bin/python -u -m nlu.train "$@"
