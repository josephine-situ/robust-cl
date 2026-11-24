#!/bin/bash
#SBATCH --job-name=focp-f2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-199
#SBATCH --output=logs/focp_f2_%A_%a.out
#SBATCH --error=logs/focp_f2_%A_%a.err

# Probe F2 (paper/paper_plan.md): the FO-CP loop on SS1 with the end-to-end
# network at the CV alpha, radius (c) per redraw (computed in the task if
# missing), one-step and multistart separation, a posteriori attack.
#   tasks 0-99    arm net (truth = ss1_truth_net)  redraws 0-99
#   tasks 100-199 arm ode (truth = the ODE)        redraws 0-99
# The pilot's redraws 0-9 (job 29709201) are found done and skipped.
# ONE thread: torch's threads make each retrain 33-120x slower on some nodes
# (thread test, jobs 29781412-29781414), so 1 CPU per task.
# Checkpointed per master iteration, so a preemption redoes at most one
# separation (plus the a posteriori attack if it was running).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date) (restart ${SLURM_RESTART_COUNT:-0})"
source experiments/_activate_env.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"   # torch vs conda libstdc++
export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export GRB_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

T="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$T" -lt 100 ]; then ARM=net; else ARM=ode; fi
python experiments/probe_focp.py --stage f2 --arm "$ARM" --redraw $(( T % 100 )) --threads 1
