#!/bin/bash
#SBATCH --job-name=rc-frontier
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --output=logs/rc_frontier_%A_%a.out
#SBATCH --error=logs/rc_frontier_%A_%a.err

# Paper A, first H2 evidence (paper/paper_plan.md): cubic ridge on the reactor,
# Ovalle et al.'s cost vectors, the SAME 100 redraws as R1 (job 29141177 task
# 30). Each redraw is solved on an RC radius grid (sigma units, plus the
# pointwise and simulated radii) and a constant-margin grid (flow units), and
# judged by the clean-label model and the ODE. Five tasks of 20 redraws; each
# saves after every redraw and resumes from its own file.
#
# AFTER (login node, seconds):
#   python experiments/probe_rc_ridge.py --stage frontier_collect --costs paper

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date)"
source experiments/_activate_env.sh

export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

T="${SLURM_ARRAY_TASK_ID:-0}"
EPS="${EPS:-0.1}"
python experiments/probe_rc_ridge.py --stage frontier --model cubic \
    --costs paper --eps "$EPS" --first $(( T * 20 )) --n-redraws 20
