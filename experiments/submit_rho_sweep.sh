#!/bin/bash
#SBATCH --job-name=rho-sweep
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2%2
#SBATCH --output=logs/rho_sweep_%A_%a.out
#SBATCH --error=logs/rho_sweep_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs results/rho_sweep

echo "Job ${SLURM_JOB_ID:-local} array-task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname) at $(date)"
echo "Working directory: $PWD"

module load miniforge
conda activate robcl_env

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# The shared-D rho axis. rho is the SINGLE size parameter of the ellipsoidal D
# (R_c = rho * scale(y_c) * sqrt(n)); it defines the problem every method solves,
# so it is swept and reported, not fitted; the derived rho*(method) is then what
# the evaluation run uses, one rho* per method. Each method's own dial (CP tau,
# wrapper alpha) stays FIXED at its config value across the whole sweep, so D is
# literally shared at every rho and a gap between methods is a difference in
# method.
#
# ---------------------------------------------------------------------------
# THIS RUN (2026-08-21) changes three things from the 08-17/08-19 sweep.
#
# 1. SEEDS -- the sweep is repeated end to end at several bank seeds, ONE PER
#    ARRAY TASK. D is sampled, not enumerated: CP cuts against B=200 vertices and
#    the wrapper embeds P=20, so a curve read off a single bank cannot separate
#    "this method absorbs more rho" from "this bank missed the direction that
#    breaks it". Repeating the whole procedure, bank build included, at a new seed
#    is what turns one curve into a spread, and it is the cheapest attack on Known
#    gap #5: rho* currently carries no error bars at all.
#    --seed moves the METHODS' randomness: the bank draws, and the random_state of
#    every model fit. The second is not a side effect to engineer away -- it moves
#    the out-of-fold residual sd, so R_c = rho * scale(y_c) * sqrt(n) wobbles a few
#    percent between seeds (synthetic, fold 1: oof_sd 0.1314 at seed 7 vs 0.1238 at
#    seed 42), which is estimation noise a single-seed curve hides. D still stays
#    shared ACROSS METHODS within a task, which is what the comparison needs.
#    The DATA and the EVALUATION FOLDS keep config.yaml's
#    uncertainty.bootstrap_seed and are bit-identical across tasks. Do NOT chase
#    this by editing bootstrap_seed instead: that also reseeds
#    synthetic_nonlinear (the data) and the synthetic KFold, and the sources of
#    variation could not be told apart afterwards.
#
# 2. NO robust_reg. Its label adversary delta = R*r/||r|| maximizes TRAINING LOSS,
#    and squared loss is symmetric in the sign of the residual, so it has no
#    preferred direction in PREDICTION space; on synthetic it lands as a
#    variance-inflating rather than a conservative perturbation -- objective
#    BETTER than nominal at rho >= 0.5 while held-out feasibility is 0 (Known gap
#    #1, measured in results/rho_sweep/diagnostics/synthetic_robust_reg_surface*.csv).
#    That curve is not reporting what a rho axis is meant to report, so it is off
#    this sweep until the diagnostic resolves. Put it back with
#    METHODS="nominal cp wrapper robust_reg"; its label_eps still tracks rho.
#
# 3. Synthetic runs 5 folds, matching run_cv.py's 5-fold model-selection CV rather
#    than cv_calibration.n_kfold=4. Held-out feasibility is quantized to 1/n_folds
#    on that single-decision problem, so the fold count is exactly what the curve
#    can resolve -- and one fold count across both CV stages keeps them
#    comparable. Passed only under --problem synthetic: gastric's folds are
#    temporal and come from cv_calibration.fold_cutoffs, which --n-folds does not
#    reach.
# ---------------------------------------------------------------------------
#
# Three outputs. Each name carries a CELL suffix -- _coh/_incoh, plus _matchbank
# under MATCH_BANK, _f<n> under N_FOLDS, and _s<seed> per task -- so the cells
# below coexist instead of one resuming from and overwriting another:
#   {problem}_rho_curve<cell>.csv  PRIMARY -- feasibility/objective per (method, rho)
#   {problem}_rho_star<cell>.csv   DERIVED -- rho*(method), the largest rho still
#                                  meeting FEAS_TARGET, i.e. how much assumed
#                                  uncertainty each method absorbs before it stops
#                                  delivering, and what that costs in objective
#                                  and in time.
#   {problem}_ablations<cell>.csv  ABLATE=--ablate -- tau and alpha at ONE rho, to
#                                  show the fixed dials were not cherry-picked.
#
# Every row now carries a `seed` column, so the per-seed curves pool by plain
# concatenation. Read mean/sd of feasibility per (method, rho) across seeds, and
# the DISTRIBUTION of rho* rather than a single value -- with 3 seeds that is a
# range, not a confidence interval, and should be reported as one:
#   python experiments/pool_rho_seeds.py --problem gastric --cell _coh
#
# Every cell carries status, n_capped, and the wall clock split into the MASTER
# phase (train + build + solve to the final master; for CP the whole cut loop)
# and the TEST-POINT phase (one prescribe solve per held-out context). CP pays up
# front and prescribes from a small master; the wrapper embeds all P models and
# pays again per test point -- that trade is the point of the split. Cells with
# n_capped > 0 hit max_iterations; they are KEPT and flagged, not dropped.
#
# rho* is a REPORTING choice and can be re-derived later with no re-solving. Pass
# the same cell flags the sweep used -- INCLUDING --seed -- since they select
# which curve is read:
#   python experiments/run_rho_sweep.py --problem gastric --coherent --seed 7 \
#       --rho-star-only --feas-target 0.8 --out-suffix _t080
#
# COST is dominated by the wrapper: it embeds all P models, ~32k vars / ~101k
# constrs on synthetic, times |folds| times |rho grid|. CP is far cheaper per
# solve (one cut per iteration, ~6.4k vars) but pays for its bank build. One seed
# per array task is what keeps a full sweep inside the 12h wall clock; %2 caps
# concurrency at two Gurobi sessions.
#
# The score CSV is a resumable checkpoint keyed by (method@rho, knob), so a
# requeued task skips finished cells. That key carries NEITHER coherence, nor the
# fold count, nor the seed -- the filename suffix above is the only thing keeping
# the cells apart, so do not collapse them onto one name. Sharing a name across
# seeds is the worst of those mistakes: every seed would resume the first seed's
# rows and the spread would read as exactly zero. EXTRA_ARGS=--refresh discards
# the checkpoint.
PROBLEM="${PROBLEM:-gastric}"
RHO_GRID="${RHO_GRID:-0.05 0.1 0.2 0.3 0.5 0.75 1.0}"
# robust_reg deliberately absent -- see (2) above.
METHODS="${METHODS:-nominal cp wrapper}"
FEAS_TARGET="${FEAS_TARGET:-0.9}"
MIN_SOLVED="${MIN_SOLVED:-0.5}"
# One bank seed per array task. 42 is the config seed, so task 0 repeats the
# established cell (under its own _s42 name) and tasks 1-2 are the new draws.
# Extend this list and --array together; they must stay the same length.
SEEDS="${SEEDS:-42 7 13}"
# Synthetic only; ignored on gastric (temporal folds) -- see (3) above.
N_FOLDS="${N_FOLDS:-5}"
# Coherence cell. The two are not interchangeable and rho* is read per cell.
COHERENCE="${COHERENCE:---coherent}"
# --match-bank sets CP's bank B to the wrapper's P. Without it a rho* gap between
# CP and the wrapper is confounded with sampling density (B=200 vs P=20): the
# wrapper may need a smaller rho only because 20 draws sample D more sparsely.
# Run BOTH and report the pair; the confounded one is not wrong, just not clean.
MATCH_BANK="${MATCH_BANK:-}"
# Ablations at ONE rho, after the sweep. Default rho is the median rho* across
# methods (a point where they actually differ beats an endpoint); ABLATE_RHO
# overrides. Expect the small-tau end to report n_capped > 0.
#
# Run on the FIRST array task only: the ablation exists to show tau=0.01 and
# alpha=0.2 were not cherry-picked, which one bank answers. ABLATE_ALL_SEEDS=1
# pays for it on every task (roughly doubling a task's work).
ABLATE="${ABLATE:---ablate}"
ABLATE_ALL_SEEDS="${ABLATE_ALL_SEEDS:-}"
ABLATE_RHO="${ABLATE_RHO:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
read -r -a SEED_ARR <<< "${SEEDS}"
if (( TASK >= ${#SEED_ARR[@]} )); then
    echo "array task ${TASK} has no seed in SEEDS='${SEEDS}'"
    exit 1
fi
SEED="${SEED_ARR[${TASK}]}"

if [[ -z "${ABLATE_ALL_SEEDS}" && "${TASK}" != "0" ]]; then
    ABLATE=""
fi

# --n-folds reaches the synthetic KFold only. Passing it on gastric would suffix
# every output _f5 while changing nothing -- a cell name that lies is worse than
# a flag not passed.
FOLD_ARG=""
if [[ "${PROBLEM}" == "synthetic" && -n "${N_FOLDS}" ]]; then
    FOLD_ARG="--n-folds ${N_FOLDS}"
fi

echo "=== task ${TASK}: problem=${PROBLEM} seed=${SEED} methods='${METHODS}' ${COHERENCE} ${MATCH_BANK} ${FOLD_ARG} ${ABLATE} ==="

python -u experiments/run_rho_sweep.py \
    --problem "${PROBLEM}" \
    --rho-grid ${RHO_GRID} \
    --methods ${METHODS} \
    --seed "${SEED}" \
    --feas-target "${FEAS_TARGET}" \
    --min-solved "${MIN_SOLVED}" \
    ${FOLD_ARG} \
    ${COHERENCE} \
    ${MATCH_BANK} \
    ${ABLATE} \
    ${ABLATE_RHO:+--ablate-rho ${ABLATE_RHO}} \
    ${EXTRA_ARGS}

# Examples:
#   sbatch experiments/submit_rho_sweep.sh                                  # gastric, coherent, 3 seeds
#   sbatch --array=0 experiments/submit_rho_sweep.sh                        # config seed only
#   SEEDS="42 7 13 23 31" sbatch --array=0-4%2 experiments/submit_rho_sweep.sh
#   COHERENCE=--incoherent sbatch experiments/submit_rho_sweep.sh
#   MATCH_BANK=--match-bank sbatch experiments/submit_rho_sweep.sh          # B=P, clean rho*
#   PROBLEM=synthetic sbatch experiments/submit_rho_sweep.sh                # 5 folds, per (3)
#   RHO_GRID="0.01 0.02 0.05 0.1" sbatch experiments/submit_rho_sweep.sh    # refine the low end
#   METHODS="nominal cp" sbatch experiments/submit_rho_sweep.sh             # skip the slow wrapper
#   EXTRA_ARGS=--refresh sbatch experiments/submit_rho_sweep.sh             # ignore the checkpoint
#
# NOTE on --problem synthetic: it is single-decision, so each fold yields ONE
# prescription and held-out feasibility is quantized to 1/n_folds -- 0.2 at the
# N_FOLDS=5 above. Read the CURVE there; a rho* at a 0.9 target needs many more
# folds, or gastric.

echo "Finished task ${TASK} at $(date)"
