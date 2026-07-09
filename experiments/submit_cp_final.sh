#!/bin/bash
#SBATCH --job-name=cp-final
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1%2
#SBATCH --output=logs/cp_final_%A_%a.out
#SBATCH --error=logs/cp_final_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs

echo "Job ${SLURM_JOB_ID:-local} array-task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname) at $(date)"

module load miniforge
conda activate robcl_env

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ============================================================================
# Final headline + frontier with the IMPROVED CP defaults (config.yaml now has
# robustify_objective=false, eval_mode=global -- the ablation's best config).
# Scarce data (subsample_frac=0.5), both constraint modes, common random numbers.
#
# JOB ARRAY (2 tasks, capped at 2 concurrent = the Gurobi session limit):
#   task 0 : headline confirmation, R=30 at rhs=0.6
#   task 1 : RHS frontier, R=10 across rhs in {0.4, 0.5, 0.6}
# Outputs: results/gastric/chemo_robust_{realizations,robustness_summary}_<TAG>_rhs_sweep.csv
# ============================================================================
R_CONFIRM="${R_CONFIRM:-30}"
R_SWEEP="${R_SWEEP:-10}"
FRAC="${FRAC:-0.5}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) R="${R_CONFIRM}"; RHS="0.6";         TAG="final_confirm" ;;
  1) R="${R_SWEEP}";   RHS="0.4 0.5 0.6"; TAG="final_sweep" ;;
  *) echo "unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID}: R=${R} rhs=[${RHS}] tag=${TAG} ==="
python -u experiments/run_chemo_robust.py \
    --n-realizations "${R}" --subsample-frac "${FRAC}" --rhs-grid ${RHS} \
    --methods nominal robust_reg cp \
    --output-tag "${TAG}"

echo "Finished task ${SLURM_ARRAY_TASK_ID} at $(date)"

# Notes:
#   - Uses the improved CP config from config.yaml (obj off, global master); no
#     ablation flags needed.
#   - Submit both:            sbatch experiments/submit_cp_final.sh
#   - Just the confirmation:  sbatch --array=0 experiments/submit_cp_final.sh
#   - Just the frontier:      sbatch --array=1 experiments/submit_cp_final.sh
#   - Overridable via env: R_CONFIRM, R_SWEEP, FRAC.
