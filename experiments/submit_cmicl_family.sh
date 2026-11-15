#!/bin/bash
#SBATCH --job-name=cmicl-family
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-19
#SBATCH --output=logs/cmicl_family_%A_%a.out
#SBATCH --error=logs/cmicl_family_%A_%a.err

# Does C-MICL's Assumption 4.1 fail more for tree surrogates?
# (experiments/probe_cmicl_family_coverage.py). One task per (family, noise
# seed): task = 5*family + seed, families rf gbm mlp cart, seeds 0-4. Each task
# saves after every cost vector and resumes, so a preemption costs one cost.
#
# Needs the reference set first (ODE at 1000 points, ~1 min on 8 cores):
#   sbatch --partition=mit_quicktest --time=0:15:00 --array=0 \
#          --export=ALL,STAGE=prep experiments/submit_cmicl_family.sh
# then this array with --dependency=afterok:<that job>. Afterwards:
#   python experiments/probe_cmicl_family_coverage.py --stage collect

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date) (restart ${SLURM_RESTART_COUNT:-0})"
source experiments/_activate_env.sh

export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1          # joblib runs the ODE across processes instead
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore
export ROBCL_GRB_LOG=0

if [ "${STAGE:-run}" = "prep" ]; then
    exec python experiments/probe_cmicl_family_coverage.py --stage prep
fi

FAMILIES=(rf gbm mlp cart)
T="${SLURM_ARRAY_TASK_ID:-0}"
FAMILY="${FAMILIES[$(( T / 5 ))]}"
SEED=$(( T % 5 ))
python experiments/probe_cmicl_family_coverage.py --stage run \
    --family "$FAMILY" --seed "$SEED"
