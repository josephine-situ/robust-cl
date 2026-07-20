#!/bin/bash
#SBATCH --job-name=cp-final
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4%2
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
# PREREQUISITE: run stage 1 first (experiments/submit_cv_calibrate.sh) to produce
# results/cv/gastric_robustness_knobs.json. With calibration.method=cv, the headline
# tasks (0-2) build each method at its CV-selected theta*, and task 4 centers each
# method's Pareto grid on theta* (--pareto-center-cv).
#
# JOB ARRAY (capped at 2 concurrent = the Gurobi session limit):
#   0 : headline confirmation   nominal/robust_reg/cp, frac=0.5, rhs=0.6,      R=30
#   1 : RHS frontier            nominal/robust_reg/cp, frac=0.5, rhs 0.4..0.8, R=10
#   2 : frac (scarcity) frontier nominal/robust_reg/cp, rhs=0.6, frac 0.3..0.8, R=10
#   3 : wrapper (headline cell)  wrapper,               frac=0.5, rhs=0.6,      R=10
#   4 : CV-centered Pareto       nominal/robust_reg/cp, rhs=0.6, frac=0.5,
#         knob = theta* x {0.5,0.75,1,1.5,2} per method -> OS-vs-worst-case
#         feasibility frontier centered on each method's CV operating point.
# Outputs: results/gastric/chemo_robust_{realizations,robustness_summary}_<TAG>_sweep.csv
# (frontier CSVs carry rhs, frac, and/or strength columns for the figures.)
# ============================================================================
R_CONFIRM="${R_CONFIRM:-30}"
R_SWEEP="${R_SWEEP:-10}"
FRAC_GRID="${FRAC_GRID:-0.3 0.4 0.5 0.6 0.7 0.8}"   # uniform scarcity axis
CONS_GRID="${CONS_GRID:-0 0.25 0.5 0.75 1.0}"       # conservativeness strengths

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) CMD="--subsample-frac 0.5 --rhs-grid 0.6";              METHODS="nominal robust_reg cp"; R="${R_CONFIRM}"; TAG="final_confirm" ;;
  1) CMD="--subsample-frac 0.5 --rhs-grid 0.4 0.5 0.6 0.7 0.8"; METHODS="nominal robust_reg cp"; R="${R_SWEEP}"; TAG="final_rhs" ;;
  2) CMD="--rhs-grid 0.6 --frac-grid ${FRAC_GRID}";         METHODS="nominal robust_reg cp"; R="${R_SWEEP}";   TAG="final_frac" ;;
  3) CMD="--subsample-frac 0.5 --rhs-grid 0.6";             METHODS="wrapper";               R="${R_SWEEP}";   TAG="final_wrapper" ;;
  4) CMD="--subsample-frac 0.5 --rhs-grid 0.6 --pareto-center-cv";                   METHODS="nominal robust_reg cp"; R="${R_SWEEP}"; TAG="final_pareto" ;;
  *) echo "unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID}: methods='${METHODS}' [${CMD}] R=${R} tag=${TAG} ==="
python -u experiments/run_chemo_robust.py \
    --n-realizations "${R}" ${CMD} \
    --methods ${METHODS} \
    --output-tag "${TAG}"

echo "Finished task ${SLURM_ARRAY_TASK_ID} at $(date)"

# Notes:
#   - Uses the improved CP config from config.yaml (obj off, global master).
#   - Submit all:             sbatch experiments/submit_cp_final.sh
#   - Subsets, e.g.:          sbatch --array=0     experiments/submit_cp_final.sh   # headline only
#                             sbatch --array=1-2%2 experiments/submit_cp_final.sh   # frontiers only
#   - Tasks 1 (5 rhs) and 2 (6 frac) are the heaviest (~5-6 x R builds); lower
#     R_SWEEP or trim FRAC_GRID if either nears 12h.
#   - Overridable via env: R_CONFIRM, R_SWEEP, FRAC_GRID.
