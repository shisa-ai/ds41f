#!/usr/bin/env bash
set -uo pipefail
cd /root/glm-testing/ds41f/inference
export CUDA_VISIBLE_DEVICES=0,1,2,3 DSV41F_ENGRAM_OFFLOAD=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/root/miniforge3/envs/ds41f/bin/torchrun
echo "===== BENCH ====="
$PY --nproc-per-node 4 benchmark_ds41f.py --prompt-lens 512 2048 8192 --no-parity-abort --output /root/ds41f/results/baseline-gpu0123.json 2>&1 | grep -vE '^\[rank[1-3]\]'
echo "===== PROFILE DECODE ====="
$PY --nproc-per-node 4 profile_decode.py 2>&1 | grep -vE '^\[rank[1-3]\]'
echo "===== PROFILE PREFILL ====="
$PY --nproc-per-node 4 profile_prefill.py 2>&1 | grep -vE '^\[rank[1-3]\]'
echo "===== DONE ====="
