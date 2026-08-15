#!/bin/bash
#SBATCH --job-name=rho-sweep
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/rho_sweep_%j.out
#SBATCH --error=logs/rho_sweep_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs results/rho_sweep

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Working directory: $PWD"

module load miniforge
conda activate robcl_env

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# The shared-D rho axis. rho is the SINGLE size parameter of the ellipsoidal D
# (R_c = rho * scale(y_c) * sqrt(n)); it defines the problem all three methods
# solve, so it is swept and reported, never selected. Each method's own dial
# (CP tau, wrapper alpha) stays FIXED at its stage-1 CV value across the whole
# sweep, so D is literally shared at every rho and a gap between methods is a
# difference in method.
#
# robust_reg is the exception and deliberately so: its label_eps IS a D radius in
# units of scale(y), so it tracks rho rather than staying fixed. Holding it fixed
# would train its adversary against a different set than the bank draws from.
#
# Three outputs:
#   {problem}_rho_curve.csv  PRIMARY -- feasibility/objective per (method, rho)
#   {problem}_rho_star.csv   DERIVED -- rho*(method), the largest rho still
#                            meeting FEAS_TARGET, i.e. how much assumed
#                            uncertainty each method absorbs before it stops
#                            delivering, and what that costs in objective and
#                            time. This is what stage-1 knob CV used to produce.
#   {problem}_ablations.csv  ABLATE=--ablate -- tau and alpha at ONE rho, to show
#                            the fixed dials were not cherry-picked
#
# Every cell carries status, n_capped, and the wall clock split into the MASTER
# phase (train + build + solve to the final master; for CP the whole cut loop)
# and the TEST-POINT phase (one prescribe solve per held-out context). CP pays up
# front and prescribes from a small master; the wrapper embeds all P models and
# pays again per test point -- that trade is the point of the split. Cells with
# n_capped > 0 hit max_iterations; they are KEPT and flagged, not dropped.
#
# rho* is a REPORTING choice and can be re-derived later with no re-solving:
#   python experiments/run_rho_sweep.py --problem gastric --rho-star-only \
#       --feas-target 0.8 --out-suffix _t080
# The curve CSV carries every column the criteria could need, and the chosen
# criteria are written back as columns so a table is never ambiguous.
#
# COST is dominated by the wrapper: it embeds all P models, ~32k vars / ~101k
# constrs on synthetic, times |folds| times |rho grid|. CP is far cheaper per
# solve (one cut per iteration, ~6.4k vars) but pays for its bank build. Budget
# roughly |grid| x (wrapper fold solves) and raise --time before raising the grid.
#
# The score CSV is a resumable checkpoint keyed by (method@rho, knob), so a
# requeued job skips finished cells. Pass EXTRA_ARGS=--refresh to discard it.
PROBLEM="${PROBLEM:-gastric}"
RHO_GRID="${RHO_GRID:-0.05 0.1 0.2 0.3 0.5 0.75 1.0}"
METHODS="${METHODS:-nominal cp wrapper robust_reg}"
FEAS_TARGET="${FEAS_TARGET:-0.9}"
MIN_SOLVED="${MIN_SOLVED:-0.5}"
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
ABLATE="${ABLATE:---ablate}"
ABLATE_RHO="${ABLATE_RHO:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

python -u experiments/run_rho_sweep.py \
    --problem "${PROBLEM}" \
    --rho-grid ${RHO_GRID} \
    --methods ${METHODS} \
    --feas-target "${FEAS_TARGET}" \
    --min-solved "${MIN_SOLVED}" \
    ${COHERENCE} \
    ${MATCH_BANK} \
    ${ABLATE} \
    ${ABLATE_RHO:+--ablate-rho ${ABLATE_RHO}} \
    ${EXTRA_ARGS}

# Examples:
#   sbatch experiments/submit_rho_sweep.sh                                  # gastric, coherent
#   COHERENCE=--incoherent sbatch experiments/submit_rho_sweep.sh
#   MATCH_BANK=--match-bank sbatch experiments/submit_rho_sweep.sh          # B=P, clean rho*
#   PROBLEM=synthetic EXTRA_ARGS="--n-folds 10" sbatch experiments/submit_rho_sweep.sh
#   RHO_GRID="0.01 0.02 0.05 0.1" sbatch experiments/submit_rho_sweep.sh    # refine the low end
#   METHODS="nominal cp" sbatch experiments/submit_rho_sweep.sh             # skip the slow wrapper
#
# NOTE on --problem synthetic: it is single-decision, so each fold yields ONE
# prescription and held-out feasibility is quantized to 1/n_folds. Read the CURVE
# there; rho* at a 0.9 target needs --n-folds well above 4, or gastric.

echo "Finished at $(date)"
