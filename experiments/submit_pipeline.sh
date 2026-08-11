#!/bin/bash
# Submit the full three-stage pipeline with SLURM dependencies, each stage starting
# only after the previous one finishes successfully (afterok):
#
#   stage 0  SYNTHETIC   CV calibration + noise sweep + CV-centered Pareto
#   stage 1  GASTRIC     robustness-parameter CV (theta* per method x coherence)
#   stage 2  GASTRIC     CV-calibrated headline + frontiers + CV-centered Pareto
#
# Synthetic runs FIRST and gates the rest. It is one LP with one learned constraint
# (minutes, not hours) and it carries the cheap correctness checks -- monotone CP
# trace, tau actually moving the solution, d0 stable in B, coherent == incoherent on
# a single-outcome problem. If any of those break, the gastric jobs never start and
# a design error costs minutes instead of a 12h wall.
#
# Usage:
#   bash experiments/submit_pipeline.sh
#
# Env overrides:
#   METHODS   methods to CALIBRATE in stages 0 and 1 (default "cp robust_reg"; add
#             wrapper to keep stage-2's wrapper task CV-consistent -- slow).
#   REFRESH=1 recompute stage-0/1 CV from scratch (else they resume the checkpoints).
#   N_REAL    synthetic data draws in stage 0 (default 10).
#   COHERENCE which (method, coherence) cells stage 1 calibrates:
#             coherent | incoherent | both (default). "coherent" halves stage 1;
#             keep uncertainty.coherent: true so stage 2 uses the matching theta*.
#   STAGE2_ARRAY   stage-2 array spec (default "0-4%2"; e.g. "0-2,4%2" to skip wrapper).
#   SKIP_SYNTHETIC=1  start at stage 1 (use when re-chaining gastric after a wall-clock
#                     failure and synthetic has already passed).
#
# Examples:
#   METHODS="cp robust_reg wrapper" bash experiments/submit_pipeline.sh
#   STAGE2_ARRAY="0-2,4%2" bash experiments/submit_pipeline.sh   # skip stage-2 wrapper
#   REFRESH=1 bash experiments/submit_pipeline.sh
#   SKIP_SYNTHETIC=1 bash experiments/submit_pipeline.sh         # gastric only
set -euo pipefail
cd "$(dirname "$0")/.."

METHODS="${METHODS:-cp robust_reg}"
STAGE2_ARRAY="${STAGE2_ARRAY:-0-4%2}"
N_REAL="${N_REAL:-10}"

jids=""
dep=""

if [ "${SKIP_SYNTHETIC:-0}" = "1" ]; then
  echo "Stage 0 (synthetic): SKIPPED (SKIP_SYNTHETIC=1)"
else
  echo "Stage 0 (synthetic CV + noise sweep + Pareto): METHODS='${METHODS}' N_REAL=${N_REAL} REFRESH=${REFRESH:-0}"
  jid0=$(METHODS="${METHODS}" N_REAL="${N_REAL}" REFRESH="${REFRESH:-0}" \
         sbatch --parsable experiments/submit_synthetic.sh)
  echo "  submitted stage 0 as job ${jid0}"
  jids="${jid0}"
  dep="--dependency=afterok:${jid0}"
fi

echo "Stage 1 (gastric CV calibration): METHODS='${METHODS}' REFRESH=${REFRESH:-0}${dep:+, ${dep}}"
jid1=$(METHODS="${METHODS}" REFRESH="${REFRESH:-0}" COHERENCE="${COHERENCE:-both}" \
       sbatch --parsable ${dep} experiments/submit_cv_calibrate.sh)
echo "  submitted stage 1 as job ${jid1}"
jids="${jids:+${jids} }${jid1}"

echo "Stage 2 (gastric headline + frontiers + CV-centered Pareto): array=${STAGE2_ARRAY}, afterok:${jid1}"
jid2=$(sbatch --parsable --dependency="afterok:${jid1}" --array="${STAGE2_ARRAY}" \
       experiments/submit_cp_final.sh)
echo "  submitted stage 2 as job ${jid2} (waits for ${jid1})"
jids="${jids} ${jid2}"

echo
echo "Watch:   squeue -u \$USER    (queued stages show state 'Dependency' until their parent finishes)"
echo "Cancel:  scancel ${jids}"
echo
echo "Note: if a stage hits its wall, re-submit that script alone (stages 0 and 1 resume"
echo "      their checkpoints) and re-chain the downstream stages on the new job id --"
echo "      SKIP_SYNTHETIC=1 restarts the pipeline at gastric stage 1."
