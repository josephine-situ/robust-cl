#!/bin/bash
#SBATCH --job-name=cmicl-paperfit
#SBATCH --partition=mit_normal
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --dependency=singleton
#SBATCH --output=logs/cmicl_paperfit_%j.out
#SBATCH --error=logs/cmicl_paperfit_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
source experiments/_activate_env.sh

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

N_INSTANCES="${N_INSTANCES:-100}"
TAG="${TAG:-}"

# baseline runs FIRST and through the same patch path with the knobs off, so the
# n_iter_ contrast is measured under one code path rather than against a
# historical cell fitted by slightly different code.
for M in baseline paper; do
    echo
    echo "################ U_MODE=${M} ################"
    U_MODE="${M}" python -u experiments/probe_paper_units.py \
        --schemes paper --n-instances "${N_INSTANCES}" --alpha 0.1 \
        --out-suffix "_${M}fit${TAG}"
done

echo "Finished at $(date)"

#   TAG=_smoke N_INSTANCES=3 sbatch --time=1:00:00 experiments/submit_paper_units.sh
#   sbatch experiments/submit_paper_units.sh
