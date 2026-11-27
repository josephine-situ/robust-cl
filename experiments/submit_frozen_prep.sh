#!/bin/bash
#SBATCH --job-name=focp-frozenprep
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/focp_frozenprep_%j.out
#SBATCH --error=logs/focp_frozenprep_%j.err

# Train the per-arm frozen hidden layers at the end-to-end alpha, once
# (probe_focp.py --stage frozen_prep), before submit_f2_extra.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source experiments/_activate_env.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore
python experiments/probe_focp.py --stage frozen_prep --threads 1
