#!/bin/bash
#SBATCH --job-name=cmicl-cost-probe
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
# Matches submit_dial_sweep.sh: 16G is far above the reactor's measured peak
# (MaxRSS 0.95G on job 21224636) and 1G/core keeps the task packable.
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
# ONE Gurobi session per task and the license allows two. This is a single task,
# not an array, so the cap is structural -- but singleton still prevents a second
# submission of this same probe running concurrently with the first.
#SBATCH --dependency=singleton
#SBATCH --output=logs/cmicl_cost_probe_%j.out
#SBATCH --error=logs/cmicl_cost_probe_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
source experiments/_activate_env.sh

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

# WHY BOTH SCHEMES IN ONE JOB. `paper` is Ovalle et al.'s own cost draw
# (regression.py:713-719, signed U(-4,4) with the negative half shrunk 10x,
# drawn against their /100-scaled variables). `fixed_ones` is the production c
# this repo reports against. Running them together means the two protocols share
# one model, one calibration split and one set of width semantics, so the
# difference between the rates is the COST DISTRIBUTION and nothing else. The
# historical 0.60-vs-0.90 contrast could not say that: it compared a fixed-c
# measurement here against a sampled-c number in the paper.
#
# OUT-SUFFIX IS MANDATORY. The output path keys on alpha and --out-suffix ONLY,
# so a bare run OVERWRITES results/reactor/diagnostics/cmicl_cost_sampling_a0.1.csv
# -- the committed 2026-08-22 cell. Never submit this without a suffix.
SCHEMES="${SCHEMES:-paper fixed_ones}"
N_INSTANCES="${N_INSTANCES:-100}"
ALPHA="${ALPHA:-0.1}"
OUT_SUFFIX="${OUT_SUFFIX:?set OUT_SUFFIX (e.g. _paper); a bare run overwrites the committed CSV}"

echo "=== schemes='${SCHEMES}' n=${N_INSTANCES} alpha=${ALPHA} suffix='${OUT_SUFFIX}' ==="

python -u experiments/probe_cmicl_cost_sampling.py \
    --schemes ${SCHEMES} \
    --n-instances "${N_INSTANCES}" \
    --alpha "${ALPHA}" \
    --out-suffix "${OUT_SUFFIX}"

echo "Finished at $(date)"

# Smoke first, then the real run:
#   OUT_SUFFIX=_smoke N_INSTANCES=5 sbatch experiments/submit_cmicl_cost_probe.sh
#   OUT_SUFFIX=_paper sbatch experiments/submit_cmicl_cost_probe.sh
