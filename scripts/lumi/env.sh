#!/usr/bin/env bash
ONESHUI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ONESHUI_ROOT
: "${LUMI_ACCOUNT:?Set LUMI_ACCOUNT before sourcing env.sh}"
export LUMI_ACCOUNT
export ONESHUI_DATA="${ONESHUI_DATA:-$(dirname "$ONESHUI_ROOT")/one_shui/dataset}"
export ONESHUI_PYTHON="${ONESHUI_PYTHON:-}"
if [[ -z "$ONESHUI_PYTHON" ]]; then
    export ONESHUI_SIF="${ONESHUI_SIF:-/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif}"
fi
export ONESHUI_CACHE="${ONESHUI_CACHE:-$(dirname "$ONESHUI_ROOT")/.cache/one-shui}"
export HF_HOME="${HF_HOME:-$ONESHUI_CACHE/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ONESHUI_CACHE/torch}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$ONESHUI_CACHE/singularity}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE="${WANDB_MODE:-offline}"
