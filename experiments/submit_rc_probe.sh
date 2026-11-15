#!/bin/bash
#SBATCH --job-name=rc-probe
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jositu@mit.edu
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-31
#SBATCH --output=logs/rc_probe_%A_%a.out
#SBATCH --error=logs/rc_probe_%A_%a.err

# Paper A go/no-go probe (paper/paper_plan.md, "Tonight"): the robust
# counterpart on the reactor, 100 redraws per model under Ovalle et al.'s cost
# vectors, each redraw solved at four radii and judged by the clean-label model
# and the ODE.
#
#   tasks  0-9   net_frozen    redraws 10*i .. 10*i+9
#   tasks 10-19  net_standard  redraws 10*(i-10) ..
#   tasks 20-29  net_honest    redraws 10*(i-20) ..   (sample-split, one dataset)
#   task  30     cubic         redraws 0 .. 99
#   task  31     radius check: solver-certified W against enumeration for the
#                frozen network, 10 noise draws (is the +0.1 sigma enough?)
#
# mit_preemptable with --requeue: each redraw task saves after every redraw and
# resumes from its own file, so a preemption costs at most the redraw in
# flight (a few minutes). The radius task (31) does not checkpoint; a
# preemption restarts it. mit_normal queued the first submission (29138280)
# for ~21 h; mit_quicktest's 15 min is shorter than a ten-redraw task.
#
# BEFORE submitting, once, on the login node (2 s; every net_frozen task must
# share one saved network). Skip it if net_frozen_weights.npz already exists:
#   python experiments/probe_rc_ridge.py --stage prep
# AFTER:
#   python experiments/probe_rc_ridge.py --stage collect --costs paper

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "Job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date) (restart ${SLURM_RESTART_COUNT:-0})"
source experiments/_activate_env.sh

export GRB_LICENSE_FILE=/orcd/software/community/001/pkg/gurobi/gurobi.lic
export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$PWD"
export PYTHONWARNINGS=ignore

T="${SLURM_ARRAY_TASK_ID:-0}"
EPS="${EPS:-0.1}"
if [ "$T" -eq 31 ]; then
    exec python experiments/probe_rc_ridge.py --stage radius --model net_frozen \
        --n-draws 10
fi
if   [ "$T" -lt 10 ]; then MODEL=net_frozen;   FIRST=$(( T * 10 ));        N=10
elif [ "$T" -lt 20 ]; then MODEL=net_standard; FIRST=$(( (T - 10) * 10 )); N=10
elif [ "$T" -lt 30 ]; then MODEL=net_honest;   FIRST=$(( (T - 20) * 10 )); N=10
else                       MODEL=cubic;        FIRST=0;                    N=100
fi

python experiments/probe_rc_ridge.py --stage redraw --model "$MODEL" \
    --costs paper --eps "$EPS" --first "$FIRST" --n-redraws "$N"
