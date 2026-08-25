#!/bin/bash
#SBATCH --job-name=dial-sweep
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=128G
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

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

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
#   cp          tau       gastric {0.5,1.0}, reactor {1,2}  grid re-read per rho
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
#    held one at a time. That is len(folds) x the models -- the reason MEM is 128G
#    and the reason the cache is dropped between rho columns.
#
# 3. THE TAU GRID IS RE-READ PER RHO COLUMN. tau's scale tracks D. At rho=0.5 the
#    same absolute grid separates harder and more cells collapse to nominal, so
#    each column is PROBED (one CP run at a tau above every distance, reading
#    CPHistory.iter0_tau) and its grid set as fractions of that. tau_frac=1 is
#    exactly the value that stops before any cut, so that endpoint IS nominal and
#    anchors the curve -- measured on synthetic: tau=0.5205 returns the nominal
#    objective to the last digit. FIXING RHO is what makes this legitimate rather
#    than circular: the grid is placed from a statistic of the ASSUMED set, never
#    from the feasibility it is about to be scored on. The probe is cached in
#    {problem}_tau_probe<cell>.json, so a requeued task does not repay it.
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
#   {problem}_dial_scores<cell>.csv   resume checkpoint, keyed (method@rho, dial)
#   {problem}_tau_probe<cell>.json    the placement statistic per rho column
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
RHO_COLUMNS_REACTOR="${RHO_COLUMNS_REACTOR:-1 2}"
RHO_COLUMNS_SYNTHETIC="${RHO_COLUMNS_SYNTHETIC:-0.5 1.0}"
TAU_FRAC_GRID="${TAU_FRAC_GRID:-1.0 0.5 0.25 0.1 0.05 0.02}"
ALPHA_GRID="${ALPHA_GRID:-0.0 0.1 0.2 0.3 0.5}"
MARGIN_GRID="${MARGIN_GRID:-0.0 0.1 0.2 0.3 0.5 0.75 1.0}"
CMICL_ALPHA_GRID="${CMICL_ALPHA_GRID:-0.02 0.05 0.1 0.2 0.3 0.5}"
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
CP_ALPHA_RHO_REACTOR="${CP_ALPHA_RHO_REACTOR:-2}"
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

echo "=== task ${TASK}: problem=${PROBLEM} rho columns='${RHO_COLUMNS}' methods='${METHODS}' seed=${SEED} ${COHERENCE} ${MATCH_BANK} ${FOLD_ARG} ${CP_ALPHA_ABLATE} ==="

python -u experiments/run_dial_sweep.py \
    --problem "${PROBLEM}" \
    --methods ${METHODS} \
    --rho-columns ${RHO_COLUMNS} \
    --tau-frac-grid ${TAU_FRAC_GRID} \
    --alpha-grid ${ALPHA_GRID} \
    --margin-grid ${MARGIN_GRID} \
    --cmicl-alpha-grid ${CMICL_ALPHA_GRID} \
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

echo "Finished task ${TASK} at $(date)"
