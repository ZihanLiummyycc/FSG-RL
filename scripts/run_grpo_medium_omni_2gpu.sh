#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/jjm/math
OUTPUT_DIR="$ROOT/outputs/grpo_medium_omni_v1_2gpu"
CONFIG="$ROOT/code/config/grpo_medium_omni_v1_2gpu.json"
DATA="$ROOT/data/grpo/medium_omni_v1/combined_grpo_train.jsonl"
CANDIDATE_GPUS=(0 1 2 3 4 5)
GPU_COUNT=2
STABLE_REQUIRED=3
SLEEP_SECONDS=60
MAX_MEMORY_MIB=1024
MAX_UTILIZATION=5

mkdir -p "$OUTPUT_DIR"
if [[ ! -s "$DATA" ]]; then
  echo "Combined GRPO data is missing: $DATA" >&2
  exit 1
fi
if pgrep -af "[m]ain.py.*grpo_medium_omni_v1_2gpu" >/dev/null; then
  echo "Two-GPU combined GRPO is already running" >&2
  exit 1
fi
if [[ -s "$OUTPUT_DIR/trajectories.jsonl" || -s "$OUTPUT_DIR/run_summary.json" ]]; then
  echo "Refusing to overwrite an existing run in $OUTPUT_DIR" >&2
  exit 1
fi
for rank_path in "$OUTPUT_DIR"/trajectories.rank-*.jsonl; do
  if [[ -s "$rank_path" ]]; then
    echo "Refusing to overwrite partial distributed output: $rank_path" >&2
    exit 1
  fi
done

declare -A stable_counts
for gpu in "${CANDIDATE_GPUS[@]}"; do
  stable_counts[$gpu]=0
done

selected_gpus=()
echo "[$(date '+%F %T')] Two-GPU Medium+Omni GRPO queue started"
echo "Candidate GPUs: ${CANDIDATE_GPUS[*]}"
echo "Need $GPU_COUNT GPUs stable for $STABLE_REQUIRED checks"
while (( ${#selected_gpus[@]} < GPU_COUNT )); do
  while IFS=',' read -r raw_gpu raw_memory raw_utilization; do
    gpu="${raw_gpu//[[:space:]]/}"
    memory="${raw_memory//[[:space:]]/}"
    utilization="${raw_utilization//[[:space:]]/}"
    if [[ ! " ${CANDIDATE_GPUS[*]} " =~ " $gpu " ]]; then
      continue
    fi
    echo "[$(date '+%F %T')] GPU=$gpu memory=${memory}MiB utilization=${utilization}%"
    if (( memory <= MAX_MEMORY_MIB && utilization <= MAX_UTILIZATION )); then
      stable_counts[$gpu]=$((stable_counts[$gpu] + 1))
    else
      stable_counts[$gpu]=0
    fi
  done < <(
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )

  selected_gpus=()
  for gpu in "${CANDIDATE_GPUS[@]}"; do
    if (( stable_counts[$gpu] >= STABLE_REQUIRED )); then
      selected_gpus+=("$gpu")
    fi
    if (( ${#selected_gpus[@]} == GPU_COUNT )); then
      break
    fi
  done

  if (( ${#selected_gpus[@]} < GPU_COUNT )); then
    echo "[$(date '+%F %T')] Fewer than two stable free GPUs; retry in ${SLEEP_SECONDS}s"
    sleep "$SLEEP_SECONDS"
  fi
done

gpu_csv=$(IFS=,; echo "${selected_gpus[*]}")
echo "[$(date '+%F %T')] Selected physical GPUs: ${selected_gpus[*]}"

cd "$ROOT"
source "$ROOT/math/bin/activate"
export CUDA_VISIBLE_DEVICES="$gpu_csv"
export HF_HOME="$ROOT/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="$ROOT/code"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export FSG_DISTRIBUTED_TIMEOUT_SECONDS="${FSG_DISTRIBUTED_TIMEOUT_SECONDS:-180}"
if [[ "$NCCL_SHM_DISABLE" == "1" ]]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
fi

echo "[$(date '+%F %T')] Starting synchronous two-GPU GRPO"
echo "NCCL settings: P2P=$NCCL_P2P_DISABLE IB=$NCCL_IB_DISABLE SHM=$NCCL_SHM_DISABLE"
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  "$ROOT/code/main.py" \
  --config "$CONFIG" \
  --mode train
echo "[$(date '+%F %T')] Two-GPU Medium+Omni GRPO completed"
