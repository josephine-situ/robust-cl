#!/bin/bash
#SBATCH --job-name=wfp-trees
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_normal
#SBATCH --time=6:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-59
#SBATCH --output=logs/wfp_%A_%a.out
#SBATCH --error=logs/wfp_%A_%a.err

# WFP first look (experiments/probe_wfp.py), a DIAGNOSTIC of the instance:
# 3 tree families x {OptiCL's convex-hull trust region, none} x 10 chunks of 10
# redraws, standard splits. The smoke test found the learned constraint mostly
# slack inside the hull and misfit dominating without it; this measures both.
#   task T: chunk T % 10; Q = T / 10: family {lineardt, rf, gbm}[Q / 2],
#   trust region on if Q % 2 == 0

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date)"
source experiments/_activate_env.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export GRB_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

T="${SLURM_ARRAY_TASK_ID:-0}"
C=$(( T % 10 )); Q=$(( T / 10 ))
FAMS=(lineardt rf gbm); FAM=${FAMS[$(( Q / 2 ))]}
TR=""; if [ $(( Q % 2 )) -eq 1 ]; then TR="--no-trust-region"; fi
python experiments/probe_wfp.py --family "$FAM" --variant standard $TR --first $(( 10 * C )) --n-redraws 10
