#!/bin/bash
#SBATCH --job-name=cmicl-balchk
#SBATCH --partition=mit_normal
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --dependency=singleton
#SBATCH --output=logs/cmicl_balchk_%j.out
#SBATCH --error=logs/cmicl_balchk_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
source experiments/_activate_env.sh

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

# WHAT IS batch_size=32 WORTH ON THE OBJECTIVE WE ACTUALLY REPORT?
#
# Every cell behind the config change used `--schemes paper`, i.e. Ovalle et
# al.'s cost draw. Production reports `reactor.cost_vector: "balanced"`
# (= 1/span_i, config.yaml:47), which sends the optimizer somewhere else -- and
# the whole finding is that feasibility depends on u AT x*. So the 0.53 -> 0.81
# magnitude does not transfer on its own.
#
# `balanced` itself cannot be measured as a RATE here: with the model fixed and
# c fixed there is no randomness left, so it yields one deterministic decision
# (the trap `fixed_ones` fell into -- solved=1/1, x* spread exactly 0). `scaled`
# is its randomised form, c_i ~ U(0,1)/span_i: the same geometry, every variable
# contributing comparably, but varying which one leads. That is the closest
# thing to production's objective that produces a rate.
#
# BEFORE is forced back to sklearn's batch default through probe_u_trainer's
# rebuild path, so both cells run the same code under one config.
N_INSTANCES="${N_INSTANCES:-100}"
TAG="${TAG:-}"

echo
echo "################ scaled objective, batch_size=auto (BEFORE) ################"
U_VARIANT=batchauto python -u experiments/probe_u_trainer.py \
    --schemes scaled --n-instances "${N_INSTANCES}" --alpha 0.1 \
    --out-suffix "_bal_before${TAG}"

echo
echo "################ scaled objective, batch_size=32 (AFTER, the new default) ################"
python -u experiments/probe_cmicl_cost_sampling.py \
    --schemes scaled --n-instances "${N_INSTANCES}" --alpha 0.1 \
    --out-suffix "_bal_after${TAG}"

echo "Finished at $(date)"

#   TAG=_smoke N_INSTANCES=3 sbatch --time=1:00:00 experiments/submit_balanced_check.sh
#   sbatch experiments/submit_balanced_check.sh
