#!/bin/bash
#SBATCH --job-name=cv-calib
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/cv_calib_%j.out
#SBATCH --error=logs/cv_calib_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs results/cv

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module load miniforge
conda activate robcl_env

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ============================================================================
# STAGE 1: robustness-parameter CV (temporal folds, train-only oracle).
# Selects each method's single robustness knob (CP dist_tol, robust_reg label_eps,
# wrapper alpha) by max held-out feasibility s.t. held-out OS within
# cv_calibration.os_tolerance_frac of nominal. Writes:
#   results/cv/gastric_robustness_knobs.json      (theta* per method -> stage 2)
#   results/cv/gastric_robustness_cv_scores.csv   (resumable checkpoint)
#
# Resumable: re-submitting skips already-scored (method, knob) cells. Set REFRESH=1
# to recompute from scratch. METHODS overrides which methods are calibrated
# (CP is scored first as it is the slowest / most likely to fail).
# ============================================================================
METHODS="${METHODS:-cp robust_reg}"     # add 'wrapper' if you can afford it (slow)
REFRESH_FLAG=""
if [ "${REFRESH:-0}" = "1" ]; then REFRESH_FLAG="--refresh-cv"; fi

echo "=== CV calibration: methods='${METHODS}' ${REFRESH_FLAG} ==="
python -u experiments/run_chemo_robust.py \
    --calibrate-cv ${REFRESH_FLAG} \
    --methods ${METHODS}

echo "Finished CV calibration at $(date)"
echo "Knobs:"; cat results/cv/gastric_robustness_knobs.json || true

# Notes:
#   - Submit:            sbatch experiments/submit_cv_calibrate.sh
#   - Clean recompute:   REFRESH=1 sbatch experiments/submit_cv_calibrate.sh
#   - Include wrapper:   METHODS="cp robust_reg wrapper" sbatch experiments/submit_cv_calibrate.sh
#   - Then run stage 2 (submit_cp_final.sh) which auto-loads the knobs JSON.
