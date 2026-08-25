#!/bin/bash
#SBATCH --job-name=chemo-table6
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/chemo_%j.out
#SBATCH --error=logs/chemo_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Working directory: $PWD"

# Env setup is shared and node-tolerant: `module load miniforge` is unknown on
# some nodes of mit_normal (it killed 2 of 6 rho-sweep array tasks on
# 2026-08-25). See experiments/_activate_env.sh -- the caller has already cd-ed
# to the repo root.
source experiments/_activate_env.sh

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u experiments/run_chemo_replication.py

echo "Finished at $(date)"
