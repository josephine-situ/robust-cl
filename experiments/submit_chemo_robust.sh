#!/bin/bash
#SBATCH --job-name=chemo-robust
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
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

# Label-noise robustness probe over R training subsamples (m-out-of-n, without
# replacement). Uses the full config from config.yaml: 96-row test set, both
# constraint modes, n_bootstrap=20, CP 20 iterations, 10 anchors. The GT ensemble
# oracle is refit on the full clean cohort each realization (invariant), only the
# constraint/robust model fit rows are resampled. Writes
# results/gastric/chemo_robust_realizations.csv and
# results/gastric/chemo_robust_robustness_summary.csv.
# NOTE: runtime scales ~linearly with R; raise --time / lower R if it overruns.
N_REALIZATIONS="${N_REALIZATIONS:-20}"
SUBSAMPLE_FRAC="${SUBSAMPLE_FRAC:-0.8}"

python -u experiments/run_chemo_robust.py \
    --n-realizations "${N_REALIZATIONS}" \
    --subsample-frac "${SUBSAMPLE_FRAC}"

# Sensitivity (stronger dataset variability): uncomment to also run at 50%.
# python -u experiments/run_chemo_robust.py --n-realizations "${N_REALIZATIONS}" --subsample-frac 0.5

echo "Finished at $(date)"
