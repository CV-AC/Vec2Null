#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$SINGULARITY_CACHEDIR"
if [[ -n "$ONESHUI_PYTHON" ]]; then
    [[ -x "$ONESHUI_PYTHON" ]] || { echo "Cannot execute $ONESHUI_PYTHON" >&2; exit 2; }
    exec "$ONESHUI_PYTHON" "$@"
fi
[[ -r "$ONESHUI_SIF" ]] || { echo "Cannot read ONESHUI_SIF=$ONESHUI_SIF" >&2; exit 2; }
rocm_options=()
bind_paths=/pfs,/users,/projappl,/scratch,/project,/flash
if [[ "$ONESHUI_SIF" == /appl/local/containers/* ]]; then
    if [[ -z "${EBROOTPYTORCH:-}" && -z "${EBROOTSINGULARITYMINAIMINBINDINGS:-}" ]]; then
        module use /appl/local/containers/ai-modules
        module load singularity-AI-bindings/24.03
    fi
    bind_paths=/users
else
    rocm_options+=(--rocm)
fi
case "${LUMI_BIND_ROCM:-auto}" in
    1) rocm_options=(--rocm) ;;
    0) rocm_options=() ;;
    auto) ;;
    *) echo 'LUMI_BIND_ROCM must be auto, 0, or 1' >&2; exit 2 ;;
esac
if [[ "${LUMI_PRIVATE_SHM:-0}" == 1 ]]; then
    : "${SLURM_JOB_ID:?Private shared memory is only supported inside Slurm}"
    export ONESHUI_HOST_SHM_DEVICE=$(stat -c %d /dev/shm)
    exec singularity exec --contain --home "$HOME" --pwd "$PWD" "${rocm_options[@]}" \
        --bind "${SINGULARITY_BIND:+$SINGULARITY_BIND,}$bind_paths,/dev/kfd,/dev/dri" \
        "$ONESHUI_SIF" python "$@"
else
    exec singularity exec "${rocm_options[@]}" \
        --bind "$bind_paths" \
        "$ONESHUI_SIF" python "$@"
fi
