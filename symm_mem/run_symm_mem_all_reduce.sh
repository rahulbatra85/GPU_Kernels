#!/usr/bin/env bash
# One-shot Triton all-reduce over PyTorch symmetric memory, on GPUs 4-7.
set -euo pipefail
export HIP_VISIBLE_DEVICES=4,5,6,7   # local ranks 0-3 -> physical GPUs 4-7
exec torchrun --nproc-per-node 4 --master-port "${MASTER_PORT:-29577}" \
     "$(dirname "$0")/symm_mem_all_reduce.py" "$@"
