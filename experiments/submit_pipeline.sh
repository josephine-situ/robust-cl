#!/bin/bash
# Submit the full two-stage CV pipeline with a SLURM dependency: stage 2 (the
# CV-calibrated headline + frontiers + CV-centered Pareto) starts only after stage 1
# (robustness-parameter CV) finishes successfully (afterok).
#
# Usage:
#   bash experiments/submit_pipeline.sh
#
# Env overrides:
#   METHODS   methods to CALIBRATE in stage 1 (default "cp robust_reg"; add wrapper
#             to keep stage-2's wrapper task CV-consistent -- slow).
#   REFRESH=1 recompute stage-1 CV from scratch (else it resumes the checkpoint).
#   STAGE2_ARRAY  stage-2 array spec (default "0-4%2"; e.g. "0-2,4%2" to skip wrapper).
#
# Examples:
#   METHODS="cp robust_reg wrapper" bash experiments/submit_pipeline.sh
#   STAGE2_ARRAY="0-2,4%2" bash experiments/submit_pipeline.sh   # skip stage-2 wrapper
#   REFRESH=1 bash experiments/submit_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

METHODS="${METHODS:-cp robust_reg}"
STAGE2_ARRAY="${STAGE2_ARRAY:-0-4%2}"

echo "Stage 1 (CV calibration): METHODS='${METHODS}' REFRESH=${REFRESH:-0}"
jid=$(METHODS="${METHODS}" REFRESH="${REFRESH:-0}" \
      sbatch --parsable experiments/submit_cv_calibrate.sh)
echo "  submitted stage 1 as job ${jid}"

echo "Stage 2 (headline + frontiers + CV-centered Pareto): array=${STAGE2_ARRAY}, afterok:${jid}"
jid2=$(sbatch --parsable --dependency="afterok:${jid}" --array="${STAGE2_ARRAY}" \
       experiments/submit_cp_final.sh)
echo "  submitted stage 2 as job ${jid2} (waits for ${jid})"

echo
echo "Watch:   squeue -u \$USER    (stage 2 shows state 'Dependency' until stage 1 finishes)"
echo "Cancel:  scancel ${jid} ${jid2}"
echo
echo "Note: if stage 1 hits the 12h wall, re-run 'sbatch experiments/submit_cv_calibrate.sh'"
echo "      (it resumes the checkpoint) and re-chain stage 2 on the new job id."
