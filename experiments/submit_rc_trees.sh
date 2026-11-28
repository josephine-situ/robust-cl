#!/bin/bash
#SBATCH --job-name=rc-trees
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_normal
#SBATCH --time=8:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-119
#SBATCH --output=logs/rc_trees_%A_%a.out
#SBATCH --error=logs/rc_trees_%A_%a.err

# The RC on Ovalle et al.'s tree surrogates (experiments/probe_rc_trees.py):
# 3 families x 2 split variants x 2 SS1 arms x 10 chunks of 10 redraws.
#   task T: chunk T % 10 (redraws 10c .. 10c+9); combo T / 10 over
#   family {lineardt, rf, gbm} x variant {standard, honest} x arm {ode, net}
# mit_normal (non-preemptable): a chunk takes under an hour, and the fits are
# redone on restart, so an uninterrupted run keeps each redraw on one fit.
# Resumable per redraw. ONE thread; torch is imported for arm net's truth.

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
C=$(( T % 10 )); Q=$(( T / 10 ))
FAMS=(lineardt rf gbm); VARS=(standard honest); ARMS=(ode net)
FAM=${FAMS[$(( Q / 4 ))]}; VAR=${VARS[$(( (Q / 2) % 2 ))]}; ARM=${ARMS[$(( Q % 2 ))]}
python experiments/probe_rc_trees.py --family "$FAM" --variant "$VAR" --arm "$ARM" \
    --first $(( 10 * C )) --n-redraws 10
