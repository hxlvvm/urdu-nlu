#!/bin/bash
#SBATCH --job-name=nlu_cpu
#SBATCH --partition=<cpu-partition>
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=runs/logs/%x_%j.log
# CPU work: baselines, ONNX export and latency. Submit from the repo root:
#   sbatch jobs/cpu.sbatch nlu.export_onnx --run runs/en_ur/best --out artifacts
set -euo pipefail
# site-specific: load a Python 3.12 module here if your cluster needs one
unset PYTHONPATH
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
.venv/bin/python -u -m "$@"
