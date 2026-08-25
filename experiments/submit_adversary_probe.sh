#!/bin/bash
#SBATCH --job-name=adv-probe
#SBATCH --partition=mit_normal
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/adversary_probe_%j.out
#SBATCH --error=logs/adversary_probe_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs results/adversary_probe

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Working directory: $PWD"

# Env setup is shared and node-tolerant: `module load miniforge` is unknown on
# some nodes of mit_normal (it killed 2 of 6 rho-sweep array tasks on
# 2026-08-25). See experiments/_activate_env.sh -- the caller has already cd-ed
# to the repo root.
source experiments/_activate_env.sh

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Measurement only -- no CP loop, no calibration, no Table 6 evaluation.
# Three questions, all at the nominal prescriptions:
#   A. does a DIRECTED per-outcome adversary beat the best of the B random bank
#      draws, and by how much (reported as a fraction of the +eps_c anchor)?
#   B. does the L1 budget actually bind?  nnz(g) vs m = budget_frac * n.
#   C. with EVERY constraint at its own worst case simultaneously (the product
#      set, not the shared-b diagonal), is the master still feasible after ONE
#      round of cuts?
#   D. and does it survive cuts ACCUMULATING? Runs CP's loop shape with a
#      directed per-outcome oracle and no tau / no rollback / no coverage cap,
#      swept over adversary budgets, reporting the iteration at which the
#      protected anchor set first breaks. This is the one that decides whether
#      "what replaces scenario rollback" is even the right question.
#
# It also reports EMBEDDING FIDELITY up front: |sklearn - embedded| for each
# nominal model at each x*. Any cut violated by less than that gap is vacuous in
# MIP space, so it bounds how small a separation tolerance can mean anything.
#
# Cost is dominated by the bank (B * #outcomes retrains) and by Part D's master
# solves (|anchors| * |loop budgets| * iters, on a growing master). B=200 x 6
# outcomes is the same order as one CP run's bank build, so 4h/64G is generous;
# raise --time if you raise B, N_ANCHORS or LOOP_ITERS.
PROBLEM="${PROBLEM:-gastric}"
N_SCENARIOS="${N_SCENARIOS:-200}"
N_ANCHORS="${N_ANCHORS:-15}"
FW_STEPS="${FW_STEPS:-3}"
BUDGET_FRACS="${BUDGET_FRACS:-1.0 0.5 0.25 0.0}"
LOOP_ITERS="${LOOP_ITERS:-8}"
LOOP_BUDGET_FRACS="${LOOP_BUDGET_FRACS:-1.0 0.5 0.25 0.1}"
# "" = config.yaml's uncertainty.geometry. Set box_l1 / ellipsoid to override; the
# two sets are matched in L2 size, so a pair of runs at the same seed and B is a
# controlled comparison of SHAPE alone. Outputs are tagged with the geometry, so
# the pair does not overwrite itself (box_l1 keeps the historical filenames).
GEOMETRY="${GEOMETRY:-}"
# Extra flags passed straight through, e.g. EXTRA_ARGS="--skip-feasibility"
EXTRA_ARGS="${EXTRA_ARGS:-}"

python -u experiments/run_adversary_probe.py \
    --problem "${PROBLEM}" \
    --n-scenarios "${N_SCENARIOS}" \
    --n-anchors "${N_ANCHORS}" \
    --fw-steps "${FW_STEPS}" \
    --budget-fracs ${BUDGET_FRACS} \
    --loop-iters "${LOOP_ITERS}" \
    --loop-budget-fracs ${LOOP_BUDGET_FRACS} \
    ${GEOMETRY:+--geometry ${GEOMETRY}} \
    ${EXTRA_ARGS}

# Examples:
#   sbatch experiments/submit_adversary_probe.sh                          # gastric, B=200
#   PROBLEM=synthetic N_ANCHORS=1 sbatch experiments/submit_adversary_probe.sh
#   N_SCENARIOS=50 FW_STEPS=5 sbatch experiments/submit_adversary_probe.sh
#   LOOP_ITERS=0 sbatch experiments/submit_adversary_probe.sh                    # skip Part D
#   LOOP_BUDGET_FRACS="0.2 0.1 0.05" sbatch experiments/submit_adversary_probe.sh  # refine the threshold
#   # geometry A/B -- Part A/B only is the cheap half (C and D dominate the wall clock):
#   GEOMETRY=box_l1    LOOP_ITERS=0 EXTRA_ARGS=--skip-feasibility sbatch experiments/submit_adversary_probe.sh
#   GEOMETRY=ellipsoid LOOP_ITERS=0 EXTRA_ARGS=--skip-feasibility sbatch experiments/submit_adversary_probe.sh
#   EXTRA_ARGS=--skip-feasibility sbatch experiments/submit_adversary_probe.sh   # A/B/D only
#
# Outputs (results/adversary_probe/):
#   gastric_influence.csv    one row per (outcome, anchor): eps_c, nnz(g), whether
#                            the budget binds, and the prediction shift achieved by
#                            bank-best / directed / Frank-Wolfe, in eps_c units
#   gastric_feasibility.csv  one row per (anchor, t): feasible + objective
#   gastric_loop.csv         one row per (t, iteration): feasible anchors, whether
#                            the protected set is intact, cuts added/total,
#                            max normalized exceedance, mean objective

echo "Finished at $(date)"
