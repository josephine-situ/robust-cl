#!/bin/bash
#SBATCH --job-name=focp-f2grid
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-599
#SBATCH --output=logs/focp_f2grid_%A_%a.out
#SBATCH --error=logs/focp_f2grid_%A_%a.err

# F2 radius grid (paper/paper_plan.md, step 1 of 2026-11-27): FO-CP with
# multistart separation at s x radius (c), s in {0.5, 0.75, 1.0}, on the stored
# theta0 (fits_*.npz), so every point is paired with the baselines of job
# 30532214. s = 1.0 repeats F2 on those fits (F2 itself shared theta0 with them
# in only about half the redraws). Radius (c) is the committed radius_c_*.json.
#   task T: scale index T / 200; arm net if T % 200 < 100, else ode; redraw T % 100
# Submit caps per user (256 on mit_preemptable, 448 on mit_normal) count every
# array task, so submit in pieces, e.g. --partition=mit_normal --array=0-399
# (scales 0.5, 0.75; runs take at most about 6 h, inside mit_normal's 12 h)
# and --array=400-599 (scale 1.0) when there is room.
# Output f2_<arm>_r<k>_a1_s<scale>.json. Checkpointed per master iteration.
# ONE thread (see submit_f2.sh).

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
SCALES=(0.5 0.75 1.0)
S="${SCALES[$(( T / 200 ))]}"
if [ $(( T % 200 )) -lt 100 ]; then ARM=net; else ARM=ode; fi
python experiments/probe_focp.py --stage f2 --arm "$ARM" --redraw $(( T % 100 )) \
    --r-scale "$S" --variants multi --threads 1
