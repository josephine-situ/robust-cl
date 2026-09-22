#!/bin/bash
#SBATCH --job-name=cmicl-h32
#SBATCH --partition=mit_normal
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
# Both cells share this job-name, so singleton serializes them: one Gurobi
# session at a time, well inside the two-session licence.
#SBATCH --dependency=singleton
#SBATCH --output=logs/cmicl_h32_%j.out
#SBATCH --error=logs/cmicl_h32_%j.err

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

SCHEMES="${SCHEMES:-paper}"
N_INSTANCES="${N_INSTANCES:-100}"
ALPHA="${ALPHA:-0.1}"
OUT_SUFFIX="${OUT_SUFFIX:?set OUT_SUFFIX (e.g. _h32); a bare run overwrites the committed CSV}"
export CMICL_SHIPPED="${CMICL_SHIPPED:-0}"
export CMICL_SHEET="${CMICL_SHEET:-$HOME/c-micl/data/unscaled_noisy_reactor_data.xlsx}"

echo "=== h=32x32 shipped=${CMICL_SHIPPED} n=${N_INSTANCES} alpha=${ALPHA} suffix='${OUT_SUFFIX}' ==="

python -u experiments/probe_h32.py \
    --schemes ${SCHEMES} \
    --n-instances "${N_INSTANCES}" \
    --alpha "${ALPHA}" \
    --out-suffix "${OUT_SUFFIX}"

echo "Finished at $(date)"
