#!/bin/bash
#SBATCH --job-name=chemo-robust
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/chemo_robust_%j.out
#SBATCH --error=logs/chemo_robust_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Working directory: $PWD"

module load miniforge
conda activate robcl_env

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Label-noise robustness over R training subsamples (m-out-of-n, without
# replacement). Uses the full config from config.yaml: 96-row test set, both
# constraint modes, n_bootstrap=20, CP 20 iterations, 10 anchors. The GT ensemble
# oracle is refit on the full clean cohort each realization (invariant); only the
# constraint/robust model fit rows are resampled.
# NOTE: runtime scales ~linearly with (R * #RHS); raise --time / lower R if it overruns.
#
# RUN_MODE=probe (default): single subsample_frac, all methods. Writes
#   results/gastric/chemo_robust_{realizations,robustness_summary}.csv
# RUN_MODE=sweep: RHS x subsample-frac frontier with common random numbers, over
#   nominal/robust_reg/cp (wrapper excluded here -- slow; add it on final cells).
#   Writes results/gastric/chemo_robust_{realizations,robustness_summary}_rhs_sweep.csv
RUN_MODE="${RUN_MODE:-probe}"
N_REALIZATIONS="${N_REALIZATIONS:-20}"
SUBSAMPLE_FRAC="${SUBSAMPLE_FRAC:-0.8}"
RHS_GRID="${RHS_GRID:-0.3 0.4 0.5 0.6}"

if [[ "${RUN_MODE}" == "sweep" ]]; then
    python -u experiments/run_chemo_robust.py \
        --n-realizations "${N_REALIZATIONS}" \
        --subsample-frac "${SUBSAMPLE_FRAC}" \
        --rhs-grid ${RHS_GRID} \
        --methods nominal robust_reg cp
else
    python -u experiments/run_chemo_robust.py \
        --n-realizations "${N_REALIZATIONS}" \
        --subsample-frac "${SUBSAMPLE_FRAC}"
fi

# Examples:
#   sbatch experiments/submit_chemo_robust.sh                                  # probe, frac=0.8
#   SUBSAMPLE_FRAC=0.5 sbatch experiments/submit_chemo_robust.sh               # probe sensitivity
#   RUN_MODE=sweep N_REALIZATIONS=10 SUBSAMPLE_FRAC=0.5 sbatch experiments/submit_chemo_robust.sh

echo "Finished at $(date)"
