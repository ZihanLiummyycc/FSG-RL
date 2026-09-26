#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/jjm/math
RUN_KIND="${1:-smoke}"
GPU_CSV="${2:-0,1}"

case "$RUN_KIND" in
  smoke)
    CONFIG="$ROOT/code/config/grpo_teacher_hard4_smoke.json"
    OUTPUT_DIR="$ROOT/outputs/grpo_teacher_hard4_smoke"
    ;;
  full)
    CONFIG="$ROOT/code/config/grpo_teacher_hard60.json"
    OUTPUT_DIR="$ROOT/outputs/grpo_teacher_hard60"
    ;;
  *)
    echo "Usage: $0 {smoke|full} GPU0,GPU1" >&2
    exit 2
    ;;
esac

if [[ ! "$GPU_CSV" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "Expected exactly two comma-separated GPU indices, got: $GPU_CSV" >&2
  exit 2
fi
if [[ -z "${TEACHER_API_KEY:-}" ]]; then
  echo "TEACHER_API_KEY is not set" >&2
  exit 1
fi
if [[ ! -s "$ROOT/data/grpo/teacher_hard60/hard60.jsonl" ]]; then
  echo "Hard-set data is missing; run select_teacher_grpo_hardset.py first" >&2
  exit 1
fi
if [[ -s "$OUTPUT_DIR/trajectories.jsonl" || -s "$OUTPUT_DIR/run_summary.json" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
source "$ROOT/math/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPU_CSV"
export HF_HOME="$ROOT/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/code"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export FSG_DISTRIBUTED_TIMEOUT_SECONDS="${FSG_DISTRIBUTED_TIMEOUT_SECONDS:-3600}"

echo "Starting teacher-guided hard-set GRPO: kind=$RUN_KIND GPUs=$GPU_CSV"
echo "Teacher calls are capped at one per problem; API key value is not logged."
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  "$ROOT/code/main.py" \
  --config "$CONFIG" \
  --mode train
