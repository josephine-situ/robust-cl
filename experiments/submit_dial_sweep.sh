#!/bin/bash
#SBATCH --job-name=dial-sweep
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
# 16G is 5x the measured peak. seff/sacct on job 21224636 (2026-08-26):
# MaxRSS 3.13G gastric, 0.95G reactor -- 2.4% and 0.7% of the 128G this used to
# ask for. Memory is NOT billed on mit_normal (its TRES line reads
# billing=cpu=5120, i.e. TRESBillingWeights is CPU-only), so the old request cost
# no fair-share -- it cost PACKING. Nodes here are ~102 cores / ~459G, about 4.5G
# per core; 128G against 16 cpus asked 8G/core, nearly double the node ratio, so
# the task blocked memory worth ~28 cores while using 16 and could only land in a
# large-memory gap. At 16G/16cpu it is 1G/core and fits wherever the cores do.
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-1%2
#SBATCH --output=logs/dial_sweep_%A_%a.out
#SBATCH --error=logs/dial_sweep_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs results/rho_sweep

echo "Job ${SLURM_JOB_ID:-local} array-task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname) at $(date)"
echo "Working directory: $PWD"

# Shared, node-tolerant env setup: `module load miniforge` is unknown on some
# nodes of mit_normal (it killed 2 of 6 rho-sweep array tasks on 2026-08-25).
source experiments/_activate_env.sh

# Thread caps the libraries ACTUALLY read.
#
# `export GRB_THREADS=...` stood here and was DEAD: no Python in this repo reads
# it, there is no gurobi.env, and Gurobi does not pick up arbitrary GRB_<PARAM>
# environment variables. The cores were nonetheless being used -- measured 13.4
# of 16 on the reactor (seff 21224636_1: CPU 10:46:14 over 48:21 wall, 83.5%) and
# 10.9 on gastric (2-07:40:17 over 05:07:38, 67.9%) -- but by LIBRARY DEFAULTS
# sized off the physical machine, not off this allocation. On a ~102-core node
# that lets a 4-cpu task spawn ~102 BLAS threads inside a 4-core cgroup and
# thrash, which would make any --cpus-per-task comparison meaningless.
#
# CAVEAT, deliberately taken: BLAS reductions reassociate with thread count, so
# changing --cpus-per-task can move results in the last bits. That was ALREADY
# true and worse -- the count came from whichever node the task landed on
# (node3305 vs node3308), so it was not even a function of the job script. This
# makes it a function of SLURM_CPUS_PER_TASK, which is recorded. Compare
# WALLTIME across core counts; treat float-noise differences as expected.
#
# Gurobi is NOT covered here: src/methods/cp.py sets Params.Threads = 0 (= the
# machine) and following SLURM_CPUS_PER_TASK needs a code change, not an export.
NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${NTHREADS}"
export OPENBLAS_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export NUMEXPR_NUM_THREADS="${NTHREADS}"

# ---------------------------------------------------------------------------
# THE PRIMARY AXIS: every method along its OWN dial, at a FIXED rho.
#
# run_rho_sweep.sh is the supporting experiment now. It answers "how much assumed
# uncertainty does each method absorb", which is a question about D. This answers
# the one the contribution rests on -- "at equal held-out feasibility, whose
# decisions are better" -- and that is a statement about two axes at once, read
# off a scatter in objective x feasibility space.
#
#   method      dial      rho columns                       notes
#   -------------------------------------------------------------------------
#   cp          tau       gastric {0.5,1.0}, reactor {3,4}  ONE fixed tau grid
#   wrapper     alpha     same                              P is a bank prefix
#   margin      m         --                                faces no D
#   cmicl       alpha     --                                alpha=0.1 = protocol
#   nominal     none      --                                one reference point
#
# robust_reg is NOT here and cannot be added: its dial label_eps IS D's radius, so
# at a fixed rho it has no dial to walk. It keeps its place on the rho sweep,
# where the axis is the quantity it actually moves. Naming it is an error rather
# than a silent flat line.
#
# BOTH rho columns land in ONE output CSV per (problem, coherence, seed) -- the
# rho column already tells them apart and one file per cell is what the plot
# reads. The checkpoint key is (method@rho, dial), so resume is unchanged.
#
# Three things this run does that the rho sweep did not:
#
# 1. PER-CONTEXT RECORDS. {problem}_dial_contexts<cell>.csv carries (fold,
#    context_idx, solved, feasible, objective) per cell. Primary scoring is
#    unchanged -- still conditional on each cell's OWN solved contexts, and that
#    independence is load-bearing. But the objective is the deliverable now, and a
#    conditional mean of it flatters whoever solved least; with these rows the
#    same-cohort comparison is derivable afterwards without coupling every cell to
#    every other.
#
# 2. ONE SHARED BANK per (rho column, fold). A bank is a pure function of
#    (instance, D, seed, B) -- neither tau nor alpha reaches it -- so one B=200
#    bank serves CP's whole tau grid AND the wrapper's alpha grid (the wrapper's P
#    models are a prefix of CP's B). Gastric drops from ~14 bank builds per rho
#    column to one per fold. Verified on synthetic: 2 constructions for 2 folds
#    across 5 solves, and CP at the smallest tau still equals the wrapper at
#    alpha=0 to the last digit.
#    COST: the cache holds every fold's bank at once, where the un-cached path
#    held one at a time. That is len(folds) x the models, and it is why the cache
#    is dropped between rho columns. It is NOT why MEM is what it is: measured
#    peak is 3.13G on gastric (4 folds x 200 draws x 6 small XGB/ElasticNet) and
#    0.95G on the reactor, so the 128G this script used to request was over by
#    more than an order of magnitude. See the --mem note in the header.
#
# 3. TAU IS FIXED BEFORE THE RUN. ONE absolute grid (TAU_GRID below), the same on
#    every rho column and every problem, in unexplained-sd units. tau is a
#    PARAMETER OF THE METHOD, set in advance like rho and like the margin's m --
#    it is never read back off the run, and in particular NEVER placed from an
#    iteration-0 separation distance. That was tried and removed: it made tau a
#    function of the bank, of B and of which folds were probed, so the same
#    nominal tau meant a different tolerance in every cell and the primary
#    figure's axis stopped being one quantity. Whether the top of the grid stops
#    before any cut is a PROPERTY OF THE RUN, reported by the
#    `[cp] ... max iter-0 dist=` line -- not something the grid is bent to give.
#
# 4. THE GRID IS SEARCHED, NOT WALKED (SEARCH=adaptive, the default). A dial grid
#    is ordered by robustness, and two answers say where NOT to look: a cell
#    scoring feasibility 0 means nothing LESS robust can deliver, and one below
#    MIN_SOLVED means nothing MORE robust can solve. The delivering cells are
#    therefore an interval of that order, whose least-robust end is the protocol
#    point -- robustness is what the objective is paid with. The search bisects to
#    that end, then fills the band around it with what is left of MAX_EVALS. This
#    is why the grids below could get FINER while a run got CHEAPER. It ASSUMES
#    monotonicity; violations among the scored cells are printed and land in the
#    star table, and SEARCH=grid is the fallback that assumes nothing.
# ---------------------------------------------------------------------------
#
# EXPECTATIONS worth pre-registering, so a surprise is a result and not a bug:
#
#  - GASTRIC C-MICL will be infeasible over much of its alpha grid. It is measured
#    infeasible at alpha=0.1 under BOTH multiplicity settings (half-widths
#    1.33-1.73 sd(y) on five constraints at once against rhs=0.6), and n_cal=80
#    means alpha >= 0.02 is needed for a finite conformal quantile at all. The
#    grid is extended UPWARD (0.2, 0.3, 0.5) to find where it FIRST solves; that
#    threshold is the result. Budget for it: proving the marginal case infeasible
#    took 176 s against nominal's 0.9 s.
#  - REACTOR rho=2 may still be short. Nominal misses the benzene target by ~4
#    units of F and rho=1 buys ~2.2, so 2 is right at the edge --
#    RHO_COLUMNS="1 2 3" if it is.
#
# SEEDS: the full dial grids run at seed 42 only. Repeating three seeds triples a
# grid that is already |rho columns| x |dial grid| cells; the bank-variance spread
# is cheaper to buy by re-running the PROTOCOL POINTS (each method at its dial*)
# at seeds 7 and 13, which is a separate, much smaller job. Revisit if the curves
# come out non-monotone.
#
# Outputs, all under the same cell suffix run_rho_sweep uses
# (_coh/_incoh [_matchbank] [_f<n>] [_m<model>] [_s<seed>]):
#   {problem}_dial_curve<cell>.csv    PRIMARY -- what plot_dial_sweep.py reads
#   {problem}_dial_contexts<cell>.csv per-context records
#   {problem}_dial_star<cell>.csv     DERIVED -- each series' protocol point
#   {problem}_dial_skipped<cell>.csv  cells the search did not score, and why
#   {problem}_dial_scores<cell>.csv   resume checkpoint, keyed (method@rho, dial)
#
# Then:
#   python experiments/plot_dial_sweep.py --all --suffix _incoh

# One array task per problem. gastric is the case study; the reactor is the only
# instance with a MECHANISTIC (ODE) oracle, so it is the one place a feasibility
# number is not decided by a fitted judge's own error at the boundary (Known gap
# #8) -- which is exactly what a protocol point rests on.
PROBLEMS="${PROBLEMS:-${PROBLEM:-gastric reactor}}"
METHODS="${METHODS:-nominal cp wrapper margin cmicl}"
# rho columns, per problem. Empty means run_dial_sweep's own defaults.
RHO_COLUMNS_GASTRIC="${RHO_COLUMNS_GASTRIC:-0.5 1.0}"
RHO_COLUMNS_REACTOR="${RHO_COLUMNS_REACTOR:-3 4}"
RHO_COLUMNS_SYNTHETIC="${RHO_COLUMNS_SYNTHETIC:-0.5 1.0}"
# Absolute, fixed, the same on every rho column. See (3) above.
#
# All four are FINER than they were before 2026-08-26, and each is a strict
# SUPERSET of its old values, so a checkpoint written under the old grids resumes
# into these instead of being orphaned. The length is affordable because SEARCH
# (below) does not walk them: it brackets the feasibility target and fills the
# band around it, so the resolution lands where the frontier bends and the dead
# tails cost nothing.
TAU_GRID="${TAU_GRID:-1.0 0.3 0.1 0.03 0.01 0.003 0.001}"
ALPHA_GRID="${ALPHA_GRID:-0.0 0.05 0.1 0.15 0.2 0.3 0.4 0.5}"
MARGIN_GRID="${MARGIN_GRID:-0.0 0.1 0.2 0.3 0.4 0.5 0.625 0.75 0.875 1.0 1.25 1.5}"
CMICL_ALPHA_GRID="${CMICL_ALPHA_GRID:-0.02 0.03 0.05 0.075 0.1 0.15 0.2 0.3 0.4 0.5}"
# How each series' grid is walked.
#
#   adaptive (default)  Order the grid by robustness, bisect to bracket the
#                       feasibility target, then spend what is left of MAX_EVALS
#                       filling the band around it. A cell scoring feasibility 0
#                       prunes everything LESS robust; one below MIN_SOLVED prunes
#                       everything MORE robust. Cells already on the checkpoint
#                       are free. Unscored cells go to {problem}_dial_skipped*.csv
#                       with the reason, never into the curve as NaN rows.
#   grid                Walk the whole grid.
#
# adaptive ASSUMES the dial is monotone -- feasibility rising with robustness, the
# solved fraction falling. Violations among the cells actually scored are printed
# as `[search] NON-MONOTONE` and land in the star table's `monotone_note`; that is
# the signal to re-run that series with SEARCH=grid, which assumes nothing.
SEARCH="${SEARCH:-adaptive}"
# Cells scored per series under adaptive. Empty = ceil(log2 n) + 2, floored at 4.
MAX_EVALS="${MAX_EVALS:-}"
FEAS_TARGET="${FEAS_TARGET:-0.9}"
MIN_SOLVED="${MIN_SOLVED:-0.5}"
SEED="${SEED:-42}"
# Single-decision problems only; gastric's folds are temporal and ignore this.
N_FOLDS_SYNTHETIC="${N_FOLDS_SYNTHETIC:-5}"
N_FOLDS_REACTOR="${N_FOLDS_REACTOR:-10}"
# --incoherent is production since 2026-08-21; --coherent is the ablation, and the
# arm the alpha=0 == tau->0 check is defined on.
COHERENCE="${COHERENCE:---incoherent}"
MATCH_BANK="${MATCH_BANK:-}"
# The coverage-cap ablation: hold tau at tau*, walk cp_alpha. It asks whether
# relaxing the cap lifts CP's ~0.984 feasibility ceiling and what that costs in
# solved fraction. ONE rho column -- gastric 1.0, reactor 2 -- chosen where CP's
# curve is strongest and the ceiling question is sharpest.
#
# STRUCTURALLY INERT on synthetic and the reactor: both take CP's BASIC separation
# path, which has no protected-anchor test to relax, so there is no cap there at
# all. run_dial_sweep says so and skips rather than emitting a flat curve that
# reads as a measurement. Left on for both problems anyway so the skip is on the
# record in the reactor's log.
CP_ALPHA_ABLATE="${CP_ALPHA_ABLATE:---cp-alpha-ablate}"
CP_ALPHA_RHO_GASTRIC="${CP_ALPHA_RHO_GASTRIC:-1.0}"
CP_ALPHA_RHO_REACTOR="${CP_ALPHA_RHO_REACTOR:-3}"
CP_ALPHA_GRID="${CP_ALPHA_GRID:-0.0 0.1 0.2 0.3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
read -r -a PROB_ARR <<< "${PROBLEMS}"
N_PROBS=${#PROB_ARR[@]}
if (( TASK >= N_PROBS )); then
    # Not an error: narrowing PROBLEMS without narrowing --array leaves unused
    # slots, and failing them would put spurious red tasks in the queue for a
    # correct invocation.
    echo "array task ${TASK} is beyond PROBLEMS('${PROBLEMS}') = ${N_PROBS} tasks"
    echo "nothing to do. Narrow the array to match: --array=0-$(( N_PROBS - 1 ))"
    exit 0
fi
PROBLEM="${PROB_ARR[${TASK}]}"

# --n-folds reaches the KFold of the single-decision problems only. Passing it on
# gastric would suffix every output _f<n> while changing nothing, and a cell name
# that lies is worse than a flag not passed.
FOLD_ARG=""
case "${PROBLEM}" in
    synthetic) RHO_COLUMNS="${RHO_COLUMNS_SYNTHETIC}"
               CP_ALPHA_RHO=""
               if [[ -n "${N_FOLDS_SYNTHETIC}" ]]; then FOLD_ARG="--n-folds ${N_FOLDS_SYNTHETIC}"; fi ;;
    reactor)   RHO_COLUMNS="${RHO_COLUMNS_REACTOR}"
               CP_ALPHA_RHO="${CP_ALPHA_RHO_REACTOR}"
               if [[ -n "${N_FOLDS_REACTOR}" ]]; then FOLD_ARG="--n-folds ${N_FOLDS_REACTOR}"; fi ;;
    gastric)   RHO_COLUMNS="${RHO_COLUMNS_GASTRIC}"
               CP_ALPHA_RHO="${CP_ALPHA_RHO_GASTRIC}" ;;
    *) echo "unknown problem '${PROBLEM}'"; exit 1 ;;
esac

# RUN_SWEEP=0 runs the TEST STAGE ALONE, against the dial* already on disk. Use
# it when the tuning is done and only the test is wanted: re-running the sweep
# would replay its checkpoint for free, but it would ALSO spend fresh evaluation
# budget on cells the adaptive search had not reached, which can move dial*
# underneath the test stage that is about to read it.
RUN_SWEEP="${RUN_SWEEP:-1}"
if [[ "${RUN_SWEEP}" == "1" ]]; then
echo "=== task ${TASK}: problem=${PROBLEM} rho columns='${RHO_COLUMNS}' methods='${METHODS}' seed=${SEED} ${COHERENCE} ${MATCH_BANK} ${FOLD_ARG} ${CP_ALPHA_ABLATE} ==="

python -u experiments/run_dial_sweep.py \
    --problem "${PROBLEM}" \
    --methods ${METHODS} \
    --rho-columns ${RHO_COLUMNS} \
    --tau-grid ${TAU_GRID} \
    --alpha-grid ${ALPHA_GRID} \
    --margin-grid ${MARGIN_GRID} \
    --cmicl-alpha-grid ${CMICL_ALPHA_GRID} \
    --search "${SEARCH}" \
    ${MAX_EVALS:+--max-evals ${MAX_EVALS}} \
    --feas-target "${FEAS_TARGET}" \
    --min-solved "${MIN_SOLVED}" \
    --seed "${SEED}" \
    ${FOLD_ARG} \
    ${COHERENCE} \
    ${MATCH_BANK} \
    ${CP_ALPHA_ABLATE} \
    ${CP_ALPHA_RHO:+--cp-alpha-rho ${CP_ALPHA_RHO}} \
    --cp-alpha-grid ${CP_ALPHA_GRID} \
    ${EXTRA_ARGS}
else
    echo "=== task ${TASK}: RUN_SWEEP=0 -- skipping the tuning sweep. The test"
    echo "    stage below holds each method at the dial* already on disk, in"
    echo "    results/rho_sweep/${PROBLEM}_dial_star*.csv ==="
fi

# ---------------------------------------------------------------------------
# THE TEST STAGE. The sweep above TUNES: every number in it is a fold score under
# the judge that instance tunes against, and dial* is fitted to exactly that
# column. run_dial_test.py holds each method at its own dial* and scores it again
# under a judge the dial was never fitted against -- the ODE on the reactor, the
# analytic f_true on synthetic, and on gastric the held-out X_test arms (there is
# no other judge there, so what makes it a test is the COHORT).
#
# Three phases, and they are not interchangeable:
#   folds      the sweep's own folds re-solved at dial*, truth-judged. A RATE,
#              with spread -- but dial* was chosen on these folds, so it is not
#              held out.
#   full       one refit on all rows, one decision per method. The deployed
#              procedure, and the only place the objectives are directly
#              comparable (one x* each, same data, same judge). Feasibility is
#              one bit per method here, which is why the fold rate is reported
#              beside it.
#   subsample  GASTRIC ONLY: m-out-of-n subsampling (N_REALIZATIONS draws of
#              SUBSAMPLE_FRAC of the constraint FIT rows, without replacement)
#              against the FIXED full-cohort GT oracle, each draw prescribing for
#              the 96 held-out X_test arms. It is `full` repeated over training
#              draws, and it is the only spread the test stage can honestly
#              report: the folds are fixed by construction AND are what dial* was
#              tuned on, while the training draw is neither. Same uncertainty
#              every Table 6 number in this repo is reported over, and the same
#              CRN seeds (bootstrap_seed + 1000*(r+1)), so realization r is the
#              same draw as realization r there. It reports feasibility twice:
#              conditional on each series' own solved arms, and over the
#              samestore cohort (the arms every series solved, recomputed per
#              draw). The reactor and synthetic have no held-out cohort to
#              prescribe for -- their held-out axis IS the fold, which `folds`
#              already scores under the ODE / the analytic truth -- so the phase
#              is refused there rather than silently skipped.
#
# COST: the subsample phase is N_REALIZATIONS x the `full` phase, i.e. one full
# gastric instance build (data + GT ensemble) and one master + 96 prescribe
# solves per (draw, series). At 10 draws x ~6 series that is the expensive part
# of this job; N_REALIZATIONS=3 is the cheap smoke-sized version.
#
# It reads {problem}_dial_star<cell>.csv, so the cell flags below MUST match the
# sweep's. A series with no dial* (never reached the target) is skipped with the
# reason printed -- there is no tuned dial to test.
RUN_TEST="${RUN_TEST:-1}"
# m-out-of-n is the standing gastric protocol, so it is in the default phases
# there and is not a legal phase anywhere else.
if [[ "${PROBLEM}" == "gastric" ]]; then
    TEST_PHASES="${TEST_PHASES:-folds full subsample}"
else
    TEST_PHASES="${TEST_PHASES:-folds full}"
fi
N_REALIZATIONS="${N_REALIZATIONS:-10}"
SUBSAMPLE_FRAC="${SUBSAMPLE_FRAC:-0.5}"
if [[ "${RUN_TEST}" == "1" ]]; then
    echo "=== task ${TASK}: TEST STAGE, problem=${PROBLEM}, phases='${TEST_PHASES}' ==="
    python -u experiments/run_dial_test.py \
        --problem "${PROBLEM}" \
        --methods ${METHODS} \
        --phases ${TEST_PHASES} \
        --n-realizations "${N_REALIZATIONS}" \
        --subsample-frac "${SUBSAMPLE_FRAC}" \
        --seed "${SEED}" \
        ${FOLD_ARG} \
        ${COHERENCE} \
        ${MATCH_BANK}
fi

# Examples. --array MUST match PROBLEMS; narrowing PROBLEMS without it leaves
# tasks that exit 0 with "nothing to do".
#   sbatch experiments/submit_dial_sweep.sh                                  # gastric + reactor
#   PROBLEM=gastric sbatch --array=0 experiments/submit_dial_sweep.sh        # gastric only
#   RHO_COLUMNS_REACTOR="1 2 3" sbatch experiments/submit_dial_sweep.sh      # if rho=2 is short
#   COHERENCE=--coherent sbatch experiments/submit_dial_sweep.sh             # the coherence ablation
#   MATCH_BANK=--match-bank sbatch experiments/submit_dial_sweep.sh          # B=P, clean CP-vs-wrapper
#   METHODS="nominal cp wrapper margin" sbatch experiments/submit_dial_sweep.sh  # skip the slow gastric cmicl
#   SEED=7 sbatch experiments/submit_dial_sweep.sh                           # a second bank (own _s7 cell)
#   CP_ALPHA_ABLATE= sbatch experiments/submit_dial_sweep.sh                 # sweep only
#   EXTRA_ARGS=--refresh sbatch experiments/submit_dial_sweep.sh             # ignore the checkpoint
#   RUN_TEST=0 sbatch experiments/submit_dial_sweep.sh                       # tune only, no test stage
#   RUN_SWEEP=0 PROBLEM=gastric sbatch --array=0 experiments/submit_dial_sweep.sh   # test only, at the dial* on disk
#   TEST_PHASES=full sbatch experiments/submit_dial_sweep.sh                 # skip the |folds| re-solves
#   TEST_PHASES="folds full" sbatch experiments/submit_dial_sweep.sh          # gastric without the m-out-of-n draws
#   N_REALIZATIONS=3 sbatch experiments/submit_dial_sweep.sh                  # cheaper m-out-of-n
#
# Test-stage outputs, same cell suffix:
#   {problem}_dial_test<cell>.csv         summary, one row per (method, phase)
#   {problem}_dial_test_points<cell>.csv  one row per (method, phase, fold or
#                                         realization, context)

echo "Finished task ${TASK} at $(date)"
