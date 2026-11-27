#!/bin/bash
#SBATCH --job-name=focp-f2extra
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-199
#SBATCH --output=logs/focp_f2extra_%A_%a.out
#SBATCH --error=logs/focp_f2extra_%A_%a.err

# F2's frozen network at the end-to-end alpha (step 1 of 2026-11-27): hidden
# layers from --stage frozen_prep (run first, once: submit_frozen_prep.sh, and
# pass --dependency=afterok:<its job>), last layer by ridge; nominal, RC and
# FO-CP at radius (c), on the stored theta0. Output f2extra_<arm>_r<k>_a1.json.
# A few minutes per task (smoke test: under 1 minute of solves). Preemptable,
# not quicktest: quicktest caps submitted jobs per user, and an array counts each
# task. Rows are saved as they finish, so a requeued task resumes.
#   tasks 0-99 arm net, 100-199 arm ode; redraw T % 100

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date)"
source experiments/_activate_env.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"   # torch vs conda libstdc++
export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export GRB_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

T="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$T" -lt 100 ]; then ARM=net; else ARM=ode; fi
python experiments/probe_focp.py --stage f2_extra --arm "$ARM" --redraw $(( T % 100 )) --threads 1
