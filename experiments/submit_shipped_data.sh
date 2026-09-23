#!/bin/bash
#SBATCH --job-name=cmicl-shipped
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_normal
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --dependency=singleton
#SBATCH --output=logs/cmicl_shipped_%j.out
#SBATCH --error=logs/cmicl_shipped_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
source experiments/_activate_env.sh

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

# The A/B against job 22599794 (_paper, our generated rows -> feasibility 0.53).
# ONLY the training rows change; the loop, MIP, calibration and cost draw are the
# same code path, reached by monkeypatching _reactor_dataset in-process.
#
# `paper` only. `fixed_ones` would add nothing: with the model fixed AND c fixed
# there is no randomness left, so it yields ONE deterministic decision, not a
# rate -- which is what the _paper run's `solved=1/1` showed.
SCHEMES="${SCHEMES:-paper}"
N_INSTANCES="${N_INSTANCES:-100}"
ALPHA="${ALPHA:-0.1}"
OUT_SUFFIX="${OUT_SUFFIX:?set OUT_SUFFIX (e.g. _shipped); a bare run overwrites the committed CSV}"
export CMICL_SHEET="${CMICL_SHEET:-$HOME/c-micl/data/unscaled_noisy_reactor_data.xlsx}"

echo "=== sheet='${CMICL_SHEET}' schemes='${SCHEMES}' n=${N_INSTANCES} alpha=${ALPHA} suffix='${OUT_SUFFIX}' ==="

python -u experiments/probe_shipped_data.py \
    --schemes ${SCHEMES} \
    --n-instances "${N_INSTANCES}" \
    --alpha "${ALPHA}" \
    --out-suffix "${OUT_SUFFIX}"

echo "Finished at $(date)"

#   OUT_SUFFIX=_shipped_smoke N_INSTANCES=5 sbatch experiments/submit_shipped_data.sh
#   OUT_SUFFIX=_shipped sbatch experiments/submit_shipped_data.sh
