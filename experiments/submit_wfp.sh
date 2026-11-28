#!/bin/bash
#SBATCH --job-name=wfp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/wfp_%x_%A_%a.out
#SBATCH --error=logs/wfp_%x_%A_%a.err

# WFP on Ovalle et al.'s design (experiments/probe_wfp.py). One script, three
# stages, chosen by STAGE (pass the array range on the command line):
#   STAGE=cv    sbatch --array=0-5  -J wfp-cv    experiments/submit_wfp.sh
#               the network's L2 penalty, one alpha per task (CV_ALPHAS)
#   STAGE=trees sbatch --array=0-59 -J wfp-trees experiments/submit_wfp.sh
#               family {lineardt, rf, gbm}[T / 20] x variant {standard, honest}
#               [(T / 10) % 2] x chunk T % 10 of 10 redraws
#   STAGE=focp  sbatch --array=0-99 -J wfp-focp --dependency=afterok:<cv job> ...
#               FO-CP and its network baselines, redraw T
# mit_normal (non-preemptable; fits are stored, rows resumable). ONE thread.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} stage ${STAGE:-?} on $(hostname) at $(date)"
source experiments/_activate_env.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"   # torch vs conda libstdc++
export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export GRB_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

T="${SLURM_ARRAY_TASK_ID:-0}"
case "${STAGE:?set STAGE=cv|trees|focp}" in
  cv)
    ALPHAS=(0.01 0.1 0.3 1 3 10)
    python experiments/probe_wfp.py --stage cv --alpha "${ALPHAS[$T]}" ;;
  trees)
    FAMS=(lineardt rf gbm); VARS=(standard honest)
    python experiments/probe_wfp.py --stage trees --family "${FAMS[$(( T / 20 ))]}" \
        --variant "${VARS[$(( (T / 10) % 2 ))]}" --first $(( 10 * (T % 10) )) --n-redraws 10 ;;
  focp)
    python experiments/probe_wfp.py --stage focp --redraw "$T" --threads 1 ;;
esac
