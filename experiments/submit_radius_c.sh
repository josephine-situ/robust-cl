#!/bin/bash
#SBATCH --job-name=focp-radius-c
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-19
#SBATCH --output=logs/focp_radius_c_%A_%a.out
#SBATCH --error=logs/focp_radius_c_%A_%a.err

# FO-CP radius (c) by enumeration (paper/paper_plan.md, Radius, and the F2
# pilot): N = 200 noise draws per redraw, end-to-end network at the CV alpha.
#   tasks 0-9   arm ode (truth = the ODE)        redraws 0-9
#   tasks 10-19 arm net (truth = ss1_truth_net)  redraws 0-9
# About 8 min per task locally; resumable per draw, so a task cut at 15 min is
# finished by resubmitting the same array index.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date)"
source experiments/_activate_env.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"   # torch vs conda libstdc++
export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

T="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$T" -lt 10 ]; then ARM=ode; else ARM=net; fi
python experiments/probe_focp.py --stage radius_c --arm "$ARM" --redraw $(( T % 10 )) \
    --n-draws 200 --threads "${SLURM_CPUS_PER_TASK:-4}"
