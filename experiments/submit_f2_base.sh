#!/bin/bash
#SBATCH --job-name=focp-f2base
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --time=8:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-199
#SBATCH --output=logs/focp_f2base_%A_%a.out
#SBATCH --error=logs/focp_f2base_%A_%a.err

# F2's baselines (paper/paper_plan.md): on each (arm, redraw) of stage f2, the
# end-to-end network's nominal, constant margin grid, wrapper (P = 20) and
# C-MICL / split conformal, plus the frozen network's nominal, RC and FO-CP at
# radius (c). Same noise, cost vector and judges as f2.
#   tasks 0-99    arm net (truth = ss1_truth_net)  redraws 0-99
#   tasks 100-199 arm ode (truth = the ODE)        redraws 0-99
# theta0 and the clean fits come from fits_*.npz when f2 stored them (reruns of
# job 30530397 on); otherwise they are refit here and stored, and the run
# records whether its nominal matches f2's (paired_with_f2). Baseline fits are
# stored in f2base_*_fits.npz and rows saved as they finish, so a preemption
# resumes on the same models. ONE thread (see submit_f2.sh).

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
python experiments/probe_focp.py --stage f2_base --arm "$ARM" --redraw $(( T % 100 )) --threads 1
