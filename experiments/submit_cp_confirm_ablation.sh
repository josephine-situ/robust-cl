#!/bin/bash
#SBATCH --job-name=cp-confirm-abl
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4
#SBATCH --output=logs/cp_confirm_abl_%A_%a.out
#SBATCH --error=logs/cp_confirm_abl_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs

echo "Job ${SLURM_JOB_ID:-local} array-task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname) at $(date)"

module load miniforge
conda activate robcl_env

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ============================================================================
# Confirmation + CP ablation at the paper's operating threshold (rhs = 0.6),
# both constraint modes (from config.yaml), scarce data (subsample_frac = 0.5).
# Common random numbers across tasks (subsample seed depends only on the
# realization), so every task is paired on the same training draws.
#
# JOB ARRAY (fits a 12h wall clock): one task per variant, run in parallel.
#   task 0        : headline confirmation (nominal / robust_reg / cp)
#   tasks 1-4     : CP ablation, 2x2 over {robustify_objective} x {eval_mode}
# Each task's heaviest work <= task 0, which is lighter than the R=10 x 4-rhs
# sweep that already completed, so each fits well under 12h.
#
# Outputs: results/gastric/chemo_robust_{realizations,robustness_summary}_<TAG>_rhs_sweep.csv
# ============================================================================
R_CONFIRM="${R_CONFIRM:-30}"
R_ABLATE="${R_ABLATE:-20}"
FRAC="${FRAC:-0.5}"
RHS="${RHS:-0.6}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) METHODS="nominal robust_reg cp"; R="${R_CONFIRM}"; CPFLAGS="";                                                 TAG="confirm" ;;
  1) METHODS="cp"; R="${R_ABLATE}"; CPFLAGS="--cp-robustify-objective true  --cp-eval-mode per_anchor_nearest"; TAG="cp_objtrue_peranchor" ;;
  2) METHODS="cp"; R="${R_ABLATE}"; CPFLAGS="--cp-robustify-objective true  --cp-eval-mode global";             TAG="cp_objtrue_global" ;;
  3) METHODS="cp"; R="${R_ABLATE}"; CPFLAGS="--cp-robustify-objective false --cp-eval-mode per_anchor_nearest"; TAG="cp_objfalse_peranchor" ;;
  4) METHODS="cp"; R="${R_ABLATE}"; CPFLAGS="--cp-robustify-objective false --cp-eval-mode global";             TAG="cp_objfalse_global" ;;
  *) echo "unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID}: methods='${METHODS}' R=${R} tag=${TAG} ${CPFLAGS} ==="
python -u experiments/run_chemo_robust.py \
    --n-realizations "${R}" --subsample-frac "${FRAC}" --rhs-grid "${RHS}" \
    --methods ${METHODS} ${CPFLAGS} \
    --output-tag "${TAG}"

echo "Finished task ${SLURM_ARRAY_TASK_ID} at $(date)"

# Notes:
#   - Submit all five tasks:  sbatch experiments/submit_cp_confirm_ablation.sh
#   - Just the confirmation:  sbatch --array=0   experiments/submit_cp_confirm_ablation.sh
#   - Just the ablation:      sbatch --array=1-4 experiments/submit_cp_confirm_ablation.sh
#   - If tasks queue rather than run in parallel, they still each fit 12h.
#   - Shared scratch (results/gastric/cp_trace.csv, prescriptions/*.csv) is
#     overwritten across concurrent CP tasks; only the *tagged* summary/realization
#     CSVs are per-variant and used for analysis.
#   - Overridable via env: R_CONFIRM, R_ABLATE, FRAC, RHS.
