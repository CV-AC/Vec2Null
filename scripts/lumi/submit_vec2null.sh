#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export LUMI_PRIVATE_SHM=1
exec bash scripts/lumi/submit.sh "$@"
