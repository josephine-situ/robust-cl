#!/bin/bash
#SBATCH --job-name=cmicl-utrain
#SBATCH --partition=mit_normal
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --dependency=singleton
#SBATCH --output=logs/cmicl_utrain_%j.out
#SBATCH --error=logs/cmicl_utrain_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
source experiments/_activate_env.sh

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

# All variants in ONE job, sequentially: one Gurobi session at a time, and the
# whole factorial lands in one log next to one baseline.
VARIANTS="${VARIANTS:-batch32 full2000 raw all}"
N_INSTANCES="${N_INSTANCES:-100}"
ALPHA="${ALPHA:-0.1}"
TAG="${TAG:-}"

for V in ${VARIANTS}; do
    echo
    echo "################ u-trainer variant: ${V} ################"
    U_VARIANT="${V}" python -u experiments/probe_u_trainer.py \
        --schemes paper \
        --n-instances "${N_INSTANCES}" \
        --alpha "${ALPHA}" \
        --out-suffix "_u_${V}${TAG}"
done

echo "Finished at $(date)"

#   TAG=_smoke N_INSTANCES=3 sbatch --time=1:00:00 experiments/submit_u_trainer.sh
#   sbatch experiments/submit_u_trainer.sh
