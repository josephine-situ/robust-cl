#!/bin/bash
#SBATCH --job-name=p1-gastric
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --output=logs/p1_gastric_%A_%a.out
#SBATCH --error=logs/p1_gastric_%A_%a.err

# Probe P1 (paper/paper_plan.md): ridge on the five gastric toxicities, the
# certified radius r_hat of each of the 96 test cohorts, against the joint
# radius by enumeration, under two choices of sigma. About 20 s per cohort
# (local timing, 2026-11-19), so 24 cohorts per task fits mit_quicktest.
#
#   tasks 0-3  binomial sigma  cohorts 24*i .. 24*i+23
#   tasks 4-7  oof sigma       cohorts 24*(i-4) ..
#
# AFTER:
#   python experiments/probe_p1_gastric.py --collect

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date)"
source experiments/_activate_env.sh

export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

T="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$T" -lt 4 ]; then KIND=binomial; FIRST=$(( T * 24 ))
else                    KIND=oof;      FIRST=$(( (T - 4) * 24 ))
fi

python experiments/probe_p1_gastric.py --sigma "$KIND" --first "$FIRST" \
    --n-cohorts 24 --n-draws 1000
