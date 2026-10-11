#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
slurm_options=()
while (($#)) && [[ "$1" != -- ]]; do
    slurm_options+=("$1")
    shift
done
if (($# == 0)); then
    echo 'Usage: bash scripts/lumi/submit.sh [sbatch options] -- script.py [Python arguments]' >&2
    exit 2
fi
shift
(($#)) || { echo 'Missing Python script/arguments after --' >&2; exit 2; }
mkdir -p "$ONESHUI_ROOT/logs"
cd "$ONESHUI_ROOT"
exec sbatch --account="$LUMI_ACCOUNT" --chdir="$ONESHUI_ROOT" \
    "${slurm_options[@]}" scripts/lumi/gpu.sbatch "$@"
