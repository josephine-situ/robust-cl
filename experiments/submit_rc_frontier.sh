#!/bin/bash
#SBATCH --job-name=rc-frontier
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-3
#SBATCH --output=logs/rc_frontier_%A_%a.out
#SBATCH --error=logs/rc_frontier_%A_%a.err

# Paper A, H2 (paper/paper_plan.md): cubic ridge on the reactor, Ovalle et
# al.'s cost vectors, the SAME 100 redraws as R1 (job 29141177 task 30). Each
# redraw is solved on an RC radius grid (sigma units, plus the pointwise and
# simulated radii) and a constant-margin grid (flow units), and judged by the
# clean-label model and the ODE. Four tasks of 25 redraws; each saves after
# every redraw and resumes from its own file.
#
# DESIGN picks the training rows (probe_rc_ridge.design_rows):
#   full  every row (job 29223094, 2026-11-15; ran as 5 x 20 before this
#         script took DESIGN)
#   sub   a random N_TRAIN-row subset          (default 200)
#   thin  all rows with flow <= 45, THIN_KEEP of the rest (default 0.2)
#   sbatch --export=ALL,DESIGN=sub experiments/submit_rc_frontier.sh
#
# AFTER (login node, seconds):
#   python experiments/probe_rc_ridge.py --stage frontier_collect --costs paper --design $DESIGN

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date)"
source experiments/_activate_env.sh

export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export GRB_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

T="${SLURM_ARRAY_TASK_ID:-0}"
EPS="${EPS:-0.1}"
STAGE="${STAGE:-frontier}"     # or "conformal": quick C-MICL + split conformal
NPER="${NPER:-25}"             # redraws per task; with NPER=50 pass --array=0-1
# The full-design frontier (job 29223094) ran as 5 x 20 on a SHORTER grid; a
# re-run here would treat its redraws as unfinished and rewrite them.
if [ "$STAGE" = frontier ] && [ "${DESIGN:-full}" = full ]; then
    echo "refusing: the full-design frontier is committed (job 29223094)" >&2
    exit 1
fi
python experiments/probe_rc_ridge.py --stage "$STAGE" --model cubic \
    --costs paper --eps "$EPS" --first $(( T * NPER )) --n-redraws "$NPER" \
    --design "${DESIGN:-full}" --n-train "${N_TRAIN:-200}" --thin-keep "${THIN_KEEP:-0.2}"
