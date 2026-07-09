#!/bin/bash
#SBATCH --job-name=cp-final
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5%2
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
# Final headline + both frontier axes, with the IMPROVED CP defaults (config.yaml
# now has robustify_objective=false, eval_mode=global). Both constraint modes,
# common random numbers (subsample seed depends only on the realization, so every
# task/cell is paired on the same training draws).
#
# JOB ARRAY (capped at 2 concurrent = the Gurobi session limit):
#   0 : headline confirmation   nominal/robust_reg/cp, frac=0.5, rhs=0.6, R=30
#   1 : RHS frontier            nominal/robust_reg/cp, frac=0.5, rhs 0.4..0.8, R=10
#   2 : frac frontier (scarce)  nominal/robust_reg/cp, frac=0.3, rhs=0.6, R=10
#   3 : frac frontier (plenty)  nominal/robust_reg/cp, frac=0.8, rhs=0.6, R=10
#   4 : wrapper (headline cell) wrapper,               frac=0.5, rhs=0.6, R=10
#   (frac=0.5 point of the scarcity axis is task 0)
# Outputs: results/gastric/chemo_robust_{realizations,robustness_summary}_<TAG>_rhs_sweep.csv
# ============================================================================
R_CONFIRM="${R_CONFIRM:-30}"
R_SWEEP="${R_SWEEP:-10}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) METHODS="nominal robust_reg cp"; FRAC=0.5; RHS="0.6";                 R="${R_CONFIRM}"; TAG="final_confirm" ;;
  1) METHODS="nominal robust_reg cp"; FRAC=0.5; RHS="0.4 0.5 0.6 0.7 0.8"; R="${R_SWEEP}";   TAG="final_rhs" ;;
  2) METHODS="nominal robust_reg cp"; FRAC=0.3; RHS="0.6";                 R="${R_SWEEP}";   TAG="final_frac03" ;;
  3) METHODS="nominal robust_reg cp"; FRAC=0.8; RHS="0.6";                 R="${R_SWEEP}";   TAG="final_frac08" ;;
  4) METHODS="wrapper";               FRAC=0.5; RHS="0.6";                 R="${R_SWEEP}";   TAG="final_wrapper" ;;
  *) echo "unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID}: methods='${METHODS}' frac=${FRAC} rhs=[${RHS}] R=${R} tag=${TAG} ==="
python -u experiments/run_chemo_robust.py \
    --n-realizations "${R}" --subsample-frac "${FRAC}" --rhs-grid ${RHS} \
    --methods ${METHODS} \
    --output-tag "${TAG}"

echo "Finished task ${SLURM_ARRAY_TASK_ID} at $(date)"

# Notes:
#   - Uses the improved CP config from config.yaml (obj off, global master).
#   - Submit all:             sbatch experiments/submit_cp_final.sh
#   - Subsets, e.g.:          sbatch --array=0     experiments/submit_cp_final.sh   # headline only
#                             sbatch --array=1-3%2 experiments/submit_cp_final.sh   # frontiers only
#   - Task 1 (RHS frontier, 5 x R) is the heaviest; lower R_SWEEP if it nears 12h.
#   - Overridable via env: R_CONFIRM, R_SWEEP.
