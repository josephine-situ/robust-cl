#!/bin/bash
#SBATCH --job-name=focp-radius-refine
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=logs/focp_radius_refine_%A_%a.out
#SBATCH --error=logs/focp_radius_refine_%A_%a.err

# How much does enumeration understate radius (c)? Local maximization (SLSQP)
# from the best enumerated points of the 25 highest draws, then R recomputed.
# Must run where radius_c ran: theta0 is not bit-identical across machines.
# Also a determinism check: each draw's enumerated W is recomputed and printed.
#   tasks 0-2: arm ode, redraws 0, 2, 8;  tasks 3-4: arm net, redraws 0, 1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source experiments/_activate_env.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

ARMS=(ode ode ode net net); REDRAWS=(0 2 8 0 1)
T="${SLURM_ARRAY_TASK_ID:-0}"
python experiments/probe_focp.py --stage radius_refine --arm "${ARMS[$T]}" \
    --redraw "${REDRAWS[$T]}" --refine-top 25 --threads "${SLURM_CPUS_PER_TASK:-4}"
