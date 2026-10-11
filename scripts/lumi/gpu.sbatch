#!/bin/bash -l
#SBATCH --job-name=one-shui
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --hint=nomultithread
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
set -euo pipefail
: "${ONESHUI_ROOT:?Use scripts/lumi/submit.sh}"
cd "$ONESHUI_ROOT"
source scripts/lumi/env.sh
if [[ "${SLURM_NTASKS:-1}" != 1 || "${SLURM_JOB_NUM_NODES:-1}" != 1 ]]; then
    echo 'This launcher supports one process on one GPU node only.' >&2
    exit 2
fi
printf 'Job=%s node=%s account=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$SLURM_JOB_ACCOUNT"
git rev-parse HEAD 2>/dev/null || true
printf 'Python arguments: '; printf '%q ' "$@"; printf '\n'
srun --ntasks=1 --ntasks-per-node=1 --cpu-bind=cores bash scripts/lumi/python.sh "$@"
