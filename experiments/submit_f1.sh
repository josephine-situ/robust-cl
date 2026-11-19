#!/bin/bash
#SBATCH --job-name=focp-f1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_normal
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=logs/focp_f1_%A_%a.out
#SBATCH --error=logs/focp_f1_%A_%a.err

# Probe F1 (paper/paper_plan.md, FO-CP): does ascent beyond the first step find
# worse relabelings than the linearized step? End-to-end (32, 32) network at the
# CV-chosen alpha (results/paper_a/prelim/cv_alpha.json), one redraw per task,
# at the nominal decision and at one robust iterate: linearized step, multistart
# ascent, random bank (B = 200), the true relabeling, cold restarts; d_eff.
# Writes f1_r<k>_a<alpha>.json only; resumes per point.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date)"
source experiments/_activate_env.sh

# torch loads the system libstdc++ before numpy's extension can find conda's
# newer one (GLIBCXX_3.4.29); put the env's lib first.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export GRB_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

python experiments/probe_focp.py --stage f1 --redraw "${SLURM_ARRAY_TASK_ID:-0}" \
    --threads "${SLURM_CPUS_PER_TASK:-4}"
