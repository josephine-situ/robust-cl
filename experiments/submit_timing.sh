#!/bin/bash
#SBATCH --job-name=focp-timing
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/focp_timing_%j.out
#SBATCH --error=logs/focp_timing_%j.err

# Thread test (F2 ran ~20x slower on some nodes): the same separation workload
# at 1 and at 4 threads, on whichever node --nodelist names.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source experiments/_activate_env.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore
grep -m1 "model name" /proc/cpuinfo
for T in 1 4; do
    OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T GRB_THREADS=$T \
        python experiments/probe_focp.py --stage timing --threads $T | grep TIMING
done
