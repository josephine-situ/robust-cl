#!/bin/bash
#SBATCH --job-name=rc-modelfree
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-10
#SBATCH --output=logs/rc_modelfree_%A_%a.out
#SBATCH --error=logs/rc_modelfree_%A_%a.err

# Probe F-1 (paper/paper_plan.md): the RC at the MODEL-FREE radius
# sigma*sqrt(chi2(n, 0.9)) = 32.5 sigma (n = 1,000), on the same 100 redraws
# and Ovalle et al.'s cost vectors as job 29141177 (R1 cubic, R2 net_frozen).
# For a model affine in its labels this is FO-CP at radius (a). Enumeration
# (2026-11-18) says cubic cannot afford it (largest radius 18-23 sigma) and
# net_frozen can (40-45 sigma); this run gives the certified status, the cost
# and the ODE feasibility.
#
#   tasks 0-9  net_frozen  redraws 10*i .. 10*i+9
#   task  10   cubic       redraws 0 .. 99
#
# Writes modelfree_*.json only; the committed redraw_*.json files are read,
# never written. Resumes per redraw after a preemption.
# AFTER:
#   python experiments/probe_rc_ridge.py --stage modelfree_collect --costs paper

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date) (restart ${SLURM_RESTART_COUNT:-0})"
source experiments/_activate_env.sh

export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

T="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$T" -lt 10 ]; then MODEL=net_frozen; FIRST=$(( T * 10 )); N=10
else                     MODEL=cubic;      FIRST=0;            N=100
fi

python experiments/probe_rc_ridge.py --stage modelfree --model "$MODEL" \
    --costs paper --eps 0.1 --first "$FIRST" --n-redraws "$N"
