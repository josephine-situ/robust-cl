#!/bin/bash
#SBATCH --job-name=synth
#SBATCH --partition=mit_normal
# 12h, not the 4h this started at. Synthetic is "one LP" but the wrapper embeds
# P=20 RF models (~375s/solve, 76% of a run) and CP separates over B=200, so one
# run_experiment is ~8 min. The noise sweep is 5 sigmas x N_REAL and the Pareto is
# 5 factors x N_REAL x 4 methods -- at N_REAL=10 that is ~14h of work, so both
# stages checkpoint and resume. Lower N_REAL if you want it inside one slot.
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/synth_%j.out
#SBATCH --error=logs/synth_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs results/cv results/synthetic

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

# Env setup is shared and node-tolerant: `module load miniforge` is unknown on
# some nodes of mit_normal (it killed 2 of 6 rho-sweep array tasks on
# 2026-08-25). See experiments/_activate_env.sh -- the caller has already cd-ed
# to the repo root.
source experiments/_activate_env.sh

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ============================================================================
# STAGE 0: the SYNTHETIC problem, run BEFORE gastric.
#
# Synthetic is one LP with one learned constraint, so the whole stage is minutes
# rather than hours -- and it is where the cheap correctness checks live:
#   * CP's worst-violation trace must be monotone over the fixed scenario bank
#   * tau must actually change the iteration count / objective across the grid
#   * d0 (the high quantile) must be stable as B varies
#   * coherent == incoherent exactly (synthetic has ONE outcome, so coherence
#     is vacuous) -- a mismatch here means the coherence flag is miswired
# Running it first means a design error costs minutes, not a 12h gastric wall.
#
# Writes:
#   results/cv/synthetic_robustness_knobs.json     (theta* per method)
#   results/cv/synthetic_robustness_cv_scores.csv  (resumable checkpoint)
#   results/synthetic/noise_sweep_results.csv      (+ figures)
#   results/synthetic/synthetic_pareto.csv
# ============================================================================
METHODS="${METHODS:-cp robust_reg wrapper}"
N_REAL="${N_REAL:-10}"
REFRESH_FLAG=""
if [ "${REFRESH:-0}" = "1" ]; then REFRESH_FLAG="--refresh-cv"; fi

echo "=== 0a. synthetic CV calibration: methods='${METHODS}' ${REFRESH_FLAG} ==="
python -u experiments/run_sweep.py \
    --calibrate-cv ${REFRESH_FLAG} \
    --methods ${METHODS}

echo "Knobs:"; cat results/cv/synthetic_robustness_knobs.json || true

# Noise sweep at the CV-calibrated knobs, over N_REAL independent data draws
# (--refresh-sweep when REFRESH=1, else it resumes the incremental checkpoint).
SWEEP_REFRESH=""
if [ "${REFRESH:-0}" = "1" ]; then SWEEP_REFRESH="--refresh-sweep"; fi

echo "=== 0b. synthetic noise sweep: n_real=${N_REAL} ${SWEEP_REFRESH} ==="
python -u experiments/run_sweep.py \
    --sweep noise --n-real "${N_REAL}" ${SWEEP_REFRESH}

echo "=== 0c. synthetic CV-centered Pareto: n_real=${N_REAL} ==="
python -u experiments/run_sweep.py \
    --pareto --n-real "${N_REAL}" \
    --methods nominal robust_reg cp wrapper

echo "Finished synthetic stage at $(date)"

# Notes:
#   - Submit alone:      sbatch experiments/submit_synthetic.sh
#   - Clean recompute:   REFRESH=1 sbatch experiments/submit_synthetic.sh
#   - Chained ahead of gastric by experiments/submit_pipeline.sh (afterok), so a
#     synthetic failure stops the gastric jobs before they consume a 12h slot.
#   - Overridable via env: METHODS, N_REAL, REFRESH.
