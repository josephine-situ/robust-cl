#!/bin/bash
#SBATCH --job-name=rho-sweep
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-5%2
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
# 1. SEEDS -- the sweep is repeated end to end at several bank seeds. Since
#    2026-08-22 an array task is one (PROBLEM, SEED) pair, seed varying fastest,
#    so the default array covers gastric x {42,7,13} then reactor x {42,7,13}. D is sampled, not enumerated: CP cuts against B=200 vertices and
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
# 3. Fold counts are PER PROBLEM on the single-decision instances, because on
#    those the fold count is exactly what the curve can resolve: one solve per
#    fold, so held-out feasibility is quantized to 1/n_folds.
#      reactor 10 -- FEAS_TARGET=0.9 is then exactly representable, so its rho*/m*
#        means what it says. Matches cv_calibration.n_kfold.
#      synthetic 5 -- matches run_cv.py's 5-fold model-selection CV, keeping the
#        two CV stages comparable, but 0.9 is NOT on that grid (only a perfect 1.0
#        clears it). Read the CURVE on synthetic, or raise N_FOLDS to 10.
#    Neither reaches gastric: its folds are temporal and come from
#    cv_calibration.fold_cutoffs, which --n-folds does not touch.
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
# BOTH default problems, one task per (problem, seed) pair. gastric is the case
# study; the reactor is the only instance with a MECHANISTIC (ODE) oracle, so it
# is the one place a feasibility number is not decided by a fitted judge's own
# error at the boundary (Known gap #8) -- which is exactly what a rho* rests on.
# Running them together means the headline table is never a single-instance claim.
# PROBLEM= (singular) still works and wins, so existing one-off invocations are
# unchanged: PROBLEM=synthetic sbatch ...
PROBLEMS="${PROBLEMS:-${PROBLEM:-gastric reactor}}"
RHO_GRID="${RHO_GRID:-0.05 0.1 0.2 0.3 0.5 0.75 1.0}"
# robust_reg deliberately absent -- see (2) above. The sweep is PER-METHOD
# PARAMETER: rho for cp/wrapper/robust_reg, the RHS margin m for `margin`, and
# nothing for cmicl (its alpha is pinned to 1 - feas_target, so it is scored once
# as a reference level). Add `margin` -- METHODS="nominal cp wrapper margin" --
# to get the feasibility-tuned nominal baseline: what a plain RHS shift buys at
# the same feasibility, i.e. the line a shared-D rho* has to beat. Its m* comes
# off the MAIN curve, so no extra ablation flag is needed; --margin-grid gives it
# values of its own. Never read an m against a rho as if they were the same
# quantity -- one is a tightening, the other an assumed radius (see param_swept).
METHODS="${METHODS:-nominal cp wrapper}"
FEAS_TARGET="${FEAS_TARGET:-0.9}"
MIN_SOLVED="${MIN_SOLVED:-0.5}"
# One bank seed per array task. 42 is the config seed, so task 0 repeats the
# established cell (under its own _s42 name) and tasks 1-2 are the new draws.
# Extend this list and --array together; they must stay the same length.
# NOTE what a seed does and does NOT move. It is the METHODS' randomness: the
# bank draws for cp/wrapper, robust_reg's refits, cmicl's calibration split. It
# does NOT reach `nominal` (solve_nominal takes no seed) and reaches `margin` only
# through the fold split behind scale(y_c) -- which on GASTRIC is temporal and so
# ignores the seed entirely. Measured: gastric margin scales are bit-identical at
# seeds 42/7/13 (dlt 0.249513, blood 0.239812); synthetic moves ~0.6%
# (0.100790/0.100581/0.101148). So a pooled spread is a spread for the SAMPLING
# methods; for nominal and gastric-margin it is exactly zero by construction, not
# evidence of stability.
SEEDS="${SEEDS:-42 7 13}"
# Fold count for the SINGLE-DECISION problems, one per problem; ignored on
# gastric (temporal folds) -- see (3) above and the WARNING at FOLD_ARG below.
# They differ on purpose. Each fold yields ONE prescription, so held-out
# feasibility is quantized to 1/n_folds, and the fold count is the RESOLUTION of
# the whole curve:
#   reactor 10 -- feasibility takes 0, 0.1, ..., 1.0, so the default
#     FEAS_TARGET=0.9 is exactly representable and a rho*/m* read there means what
#     it says. Matches config.yaml's cv_calibration.n_kfold, which is 10 for this
#     reason. Equivalent to passing nothing today (run_rho_sweep resolves the same
#     10 from config, and the cell name is _f10 either way) -- it is passed
#     explicitly so the resolution is stated at the call site rather than inherited.
#   synthetic 5 -- the cheaper established value. At 5 feasibility takes only
#     {0, 0.2, ..., 1.0} and 0.9 is NOT representable: it is met only by a perfect
#     1.0, so read the CURVE there, not the star. Raise to 10 to match the reactor.
N_FOLDS="${N_FOLDS:-5}"
N_FOLDS_REACTOR="${N_FOLDS_REACTOR:-10}"
# Coherence cell. The two are not interchangeable and rho* is read per cell.
# --incoherent is the production cell since 2026-08-21 (config.yaml says why);
# --coherent is the ablation, and the arm the alpha=0 == tau->0 check needs.
COHERENCE="${COHERENCE:---incoherent}"
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
read -r -a PROB_ARR <<< "${PROBLEMS}"
N_SEEDS=${#SEED_ARR[@]}
N_PROBS=${#PROB_ARR[@]}
N_TASKS=$(( N_PROBS * N_SEEDS ))
# Task t -> (problem t / n_seeds, seed t % n_seeds). Seed varies fastest, so a
# truncated --array still yields whole seed-sweeps of the FIRST problem rather
# than one seed of each -- a partial run is then a complete answer about one
# instance instead of an incomparable spread across two.
if (( TASK >= N_TASKS )); then
    # Not an error: narrowing PROBLEMS or SEEDS without narrowing --array leaves
    # unused slots, and failing them would put spurious red tasks in the queue for
    # a correct invocation. Exit clean and say so.
    echo "array task ${TASK} is beyond PROBLEMS('${PROBLEMS}') x SEEDS('${SEEDS}') = ${N_TASKS} tasks"
    echo "nothing to do. Narrow the array to match: --array=0-$(( N_TASKS - 1 ))"
    exit 0
fi
PROBLEM="${PROB_ARR[$(( TASK / N_SEEDS ))]}"
SEED="${SEED_ARR[$(( TASK % N_SEEDS ))]}"

# Ablate on the FIRST SEED OF EACH PROBLEM, not on global task 0 -- with two
# problems in the array, task 0 is gastric only, so keying off it would leave the
# reactor with no tau/alpha ablation at all. The ablation is a per-problem
# statement (tau and alpha were not cherry-picked ON THIS INSTANCE), so it needs
# one task per problem.
if [[ -z "${ABLATE_ALL_SEEDS}" && $(( TASK % N_SEEDS )) != "0" ]]; then
    ABLATE=""
fi

# --n-folds reaches the KFold of the SINGLE-DECISION problems (synthetic,
# reactor) only. Passing it on gastric would suffix every output _f5 while
# changing nothing -- gastric's folds are temporal -- and a cell name that lies is
# worse than a flag not passed.
#
# WARNING on the value: those two problems solve ONCE per fold, so held-out
# feasibility is quantized to 1/n_folds and a target finer than that cannot be
# met except by a perfect 1.0. See the per-problem defaults above.
#
# `if`, not `[[ ... ]] && FOLD_ARG=...`: under `set -e` a false test as the last
# command of a branch would abort the task.
FOLD_ARG=""
case "${PROBLEM}" in
    synthetic)
        if [[ -n "${N_FOLDS}" ]]; then FOLD_ARG="--n-folds ${N_FOLDS}"; fi
        ;;
    reactor)
        if [[ -n "${N_FOLDS_REACTOR}" ]]; then FOLD_ARG="--n-folds ${N_FOLDS_REACTOR}"; fi
        ;;
esac

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

# Examples. --array MUST match PROBLEMS x SEEDS; narrowing either without it
# leaves tasks that exit 0 with "nothing to do".
#   sbatch experiments/submit_rho_sweep.sh                                  # gastric + reactor, incoherent, 3 seeds each (6 tasks)
#   PROBLEM=gastric sbatch --array=0-2%2 experiments/submit_rho_sweep.sh     # gastric only, 3 seeds
#   PROBLEMS="gastric reactor synthetic" sbatch --array=0-8%2 experiments/submit_rho_sweep.sh
#   sbatch --array=0,3 experiments/submit_rho_sweep.sh                      # config seed only, BOTH problems
#   SEEDS="42 7 13 23 31" PROBLEM=gastric sbatch --array=0-4%2 experiments/submit_rho_sweep.sh
#   COHERENCE=--coherent sbatch experiments/submit_rho_sweep.sh            # the coherence ablation
#   MATCH_BANK=--match-bank sbatch experiments/submit_rho_sweep.sh          # B=P, clean rho*
#   RHO_GRID="0.01 0.02 0.05 0.1" sbatch experiments/submit_rho_sweep.sh    # refine the low end
#   METHODS="nominal cp" sbatch experiments/submit_rho_sweep.sh             # skip the slow wrapper
#   METHODS="nominal cp wrapper cmicl" sbatch experiments/submit_rho_sweep.sh # + the conformal reference line
#   METHODS="nominal cp wrapper margin" sbatch experiments/submit_rho_sweep.sh # + the tuned-nominal baseline (m* is on the main curve)
#   N_FOLDS=10 sbatch experiments/submit_rho_sweep.sh                       # raise SYNTHETIC to the reactor's resolution
#   N_FOLDS_REACTOR=20 sbatch experiments/submit_rho_sweep.sh                # finer reactor curve (20 solves/cell)
#   EXTRA_ARGS=--refresh sbatch experiments/submit_rho_sweep.sh             # ignore the checkpoint
#
# NOTE on the single-decision problems: each fold yields ONE prescription, so
# held-out feasibility is quantized to 1/n_folds. The reactor runs at 10, where
# FEAS_TARGET=0.9 is exactly representable; synthetic runs at 5, where it is not
# (0.9 is met only by a perfect 1.0) -- read the CURVE there, or set N_FOLDS=10.
# Gastric is contextual and unaffected by either.

echo "Finished task ${TASK} at $(date)"
