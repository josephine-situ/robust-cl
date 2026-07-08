#!/bin/bash
#SBATCH --job-name=cp-confirm-abl
#SBATCH --partition=mit_normal
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/cp_confirm_abl_%j.out
#SBATCH --error=logs/cp_confirm_abl_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Working directory: $PWD"

module load miniforge
conda activate robcl_env

export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ============================================================================
# Confirmation + CP ablation at the paper's operating threshold (rhs = 0.6),
# both constraint modes (dlt_only + all_constraints, from config.yaml), under
# scarce data (subsample_frac = 0.5). Common random numbers across variants
# (subsample seed depends only on the realization), so all runs are paired.
#
# All runs write results/gastric/chemo_robust_{realizations,robustness_summary}<TAG>.csv
# ============================================================================
R_CONFIRM="${R_CONFIRM:-30}"      # realizations for the headline confirmation
R_ABLATE="${R_ABLATE:-20}"        # realizations for the mechanism ablation (cheaper)
FRAC="${FRAC:-0.5}"
RHS="${RHS:-0.6}"

# --- 1. Headline confirmation: nominal / robust_reg / cp at rhs=0.6, R=30 ---
echo "=== [1/5] confirmation (nominal, robust_reg, cp) R=${R_CONFIRM} ==="
python -u experiments/run_chemo_robust.py \
    --n-realizations "${R_CONFIRM}" --subsample-frac "${FRAC}" --rhs-grid "${RHS}" \
    --methods nominal robust_reg cp \
    --output-tag confirm

# --- 2-5. CP ablation: 2x2 over {robustify_objective} x {eval_mode} ---
# Explains WHERE CP's worst-case gain comes from (objective robustification vs
# per-anchor masters vs the toxicity cuts alone).
for OBJ in true false; do
  for EVAL in per_anchor_nearest global; do
    TAG="cp_obj${OBJ}_${EVAL}"
    echo "=== ablation: robustify_objective=${OBJ} eval_mode=${EVAL} (tag=${TAG}) ==="
    python -u experiments/run_chemo_robust.py \
        --n-realizations "${R_ABLATE}" --subsample-frac "${FRAC}" --rhs-grid "${RHS}" \
        --methods cp \
        --cp-robustify-objective "${OBJ}" \
        --cp-eval-mode "${EVAL}" \
        --output-tag "${TAG}"
  done
done

echo "Finished at $(date)"

# Notes:
#   - Both modes run automatically (config.yaml methods.chemo.constraint_modes).
#   - Outputs: confirm -> ..._confirm_rhs_sweep.csv ; ablation -> ..._cp_objX_Y_rhs_sweep.csv
#   - If 24h is tight, split into two jobs (confirmation vs ablation) or lower R_ABLATE.
#   - Overridable: R_CONFIRM, R_ABLATE, FRAC, RHS via env, e.g.
#       R_CONFIRM=50 FRAC=0.5 sbatch experiments/submit_cp_confirm_ablation.sh
