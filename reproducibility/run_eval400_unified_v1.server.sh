#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/jjm/math
FREEZE_ROOT="$ROOT/protocols/eval400_unified_v1"
CODE="$FREEZE_ROOT/code"

DATA="$ROOT/data/eval400_locked/eval400.jsonl"
LOCK_MANIFEST="$ROOT/data/eval400_locked/lock_manifest.json"

OUTPUT="$ROOT/outputs/eval400_unified_v1"
LOG="$OUTPUT/eval.log"
PROTOCOL_MANIFEST="$OUTPUT/protocol_manifest.json"

STAGE2="$ROOT/outputs/sft_stage2_code_e2/epoch-2"
MAIN_GRPO="$ROOT/outputs/grpo_medium_omni_v1_2gpu/final"
FEEDBACK_GRPO="$ROOT/outputs/grpo_teacher_hard60/final"
TEACHER_JUDGE="$ROOT/outputs/grpo_teacher_judge_hard60_v2/final"

CANDIDATE_GPUS=(0 1 2 3 4 5)
MAX_MEMORY_MIB=1024
MAX_UTILIZATION=5
STABLE_CHECKS=3
SLEEP_SECONDS=60

mkdir -p "$OUTPUT"
touch "$LOG"

exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] Frozen unified Eval400 started"

echo "Frozen code: $CODE"
echo "Output: $OUTPUT"
echo "Teacher-free evaluation"
echo "Greedy decoding"
echo "Qwen thinking disabled"

for path in \
  "$DATA" \
  "$LOCK_MANIFEST" \
  "$CODE/fsg_rl/rollout.py" \
  "$CODE/scripts/evaluate_executable_sft.py" \
  "$CODE/scripts/compare_locked_eval.py"
do
  if [[ ! -s "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

for adapter in \
  "$STAGE2" \
  "$MAIN_GRPO" \
  "$FEEDBACK_GRPO" \
  "$TEACHER_JUDGE"
do
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    echo "Missing adapter config: $adapter" >&2
    exit 1
  fi

  if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
    echo "Missing adapter weights: $adapter" >&2
    exit 1
  fi
done

if ! grep -q \
  '"enable_thinking": False' \
  "$CODE/scripts/evaluate_executable_sft.py"
then
  echo "Frozen evaluator does not explicitly disable thinking" >&2
  exit 1
fi

python - "$DATA" "$LOCK_MANIFEST" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

data_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])

lock = json.loads(
    lock_path.read_text(encoding="utf-8")
)

data_hash = hashlib.sha256(
    data_path.read_bytes()
).hexdigest()

rows = sum(
    1
    for line in data_path.open(encoding="utf-8")
    if line.strip()
)

assert lock["locked"] is True
assert lock["do_not_train"] is True
assert rows == 400
assert rows == lock["actual_rows"]
assert data_hash == lock["locked_jsonl_sha256"]
assert lock["train_id_overlap"] == 0
assert lock["train_text_overlap"] == 0
assert lock["prior_eval_id_overlap"] == 0
assert lock["prior_eval_text_overlap"] == 0

print("Locked Eval400 data and leakage audit passed")
print("Rows:", rows)
print("SHA256:", data_hash)
PY

python - \
  "$CODE" \
  "$DATA" \
  "$LOCK_MANIFEST" \
  "$STAGE2" \
  "$MAIN_GRPO" \
  "$FEEDBACK_GRPO" \
  "$TEACHER_JUDGE" \
  "$PROTOCOL_MANIFEST" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    code_string,
    data_string,
    lock_string,
    stage2_string,
    main_string,
    feedback_string,
    judge_string,
    output_string,
) = sys.argv[1:]

code = Path(code_string)
data = Path(data_string)
lock = Path(lock_string)
output = Path(output_string)

adapters = {
    "stage2": Path(stage2_string),
    "main_grpo": Path(main_string),
    "feedback_grpo": Path(feedback_string),
    "teacher_judge": Path(judge_string),
}

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()

def sha256_tree(root):
    digest = hashlib.sha256()

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )

    for path in paths:
        relative = path.relative_to(root).as_posix()

        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()

manifest = {
    "protocol_name": "eval400_unified_v1",
    "frozen": True,
    "teacher_free": True,
    "data": {
        "path": str(data),
        "sha256": sha256_file(data),
        "rows": sum(
            1
            for line in data.open(encoding="utf-8")
            if line.strip()
        ),
        "lock_manifest_path": str(lock),
        "lock_manifest_sha256": sha256_file(lock),
    },
    "code": {
        "root": str(code),
        "tree_sha256": sha256_tree(code),
        "rollout_sha256": sha256_file(
            code / "fsg_rl/rollout.py"
        ),
        "evaluator_sha256": sha256_file(
            code / "scripts/evaluate_executable_sft.py"
        ),
        "comparison_sha256": sha256_file(
            code / "scripts/compare_locked_eval.py"
        ),
    },
    "base_model": "Qwen/Qwen3.5-9B-Base",
    "adapters": {
        label: {
            "path": str(path),
            "adapter_config_sha256": sha256_file(
                path / "adapter_config.json"
            ),
            "adapter_model_sha256": sha256_file(
                path / "adapter_model.safetensors"
            ),
        }
        for label, path in adapters.items()
    },
    "decoding": {
        "mode": "greedy",
        "temperature": 0.0,
        "do_sample": False,
        "enable_thinking": False,
        "max_new_tokens": 2048,
        "seed": 42,
    },
    "execution": {
        "backend": "local_limited",
        "timeout_seconds": 5,
        "memory_fraction": 0.40,
        "local_memory_max": "8g",
    },
}

rendered = (
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

if output.is_file():
    previous = json.loads(
        output.read_text(encoding="utf-8")
    )

    if previous != manifest:
        raise RuntimeError(
            "Existing protocol manifest differs from "
            "the current frozen protocol. Refusing to mix results."
        )

    print("Existing frozen protocol manifest matches")
else:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        rendered,
        encoding="utf-8",
    )
    print("Frozen protocol manifest created")

print(rendered)
PY

declare -A STABLE_COUNTS

for gpu in "${CANDIDATE_GPUS[@]}"; do
  STABLE_COUNTS[$gpu]=0
done

SELECTED_GPU=""

echo "[$(date '+%F %T')] Waiting for one stable free GPU"
echo "Candidate GPUs: ${CANDIDATE_GPUS[*]}"
echo "Memory <= ${MAX_MEMORY_MIB} MiB"
echo "Utilization <= ${MAX_UTILIZATION}%"
echo "Stable checks required: $STABLE_CHECKS"

while [[ -z "$SELECTED_GPU" ]]; do
  for gpu in "${CANDIDATE_GPUS[@]}"; do
    values="$(
      nvidia-smi \
        -i "$gpu" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
      | tr -d ' ' \
      | tr ',' ' '
    )"

    read -r memory utilization <<<"$values"

    if [[ ! "$memory" =~ ^[0-9]+$ ]] \
      || [[ ! "$utilization" =~ ^[0-9]+$ ]]
    then
      echo "GPU=$gpu returned invalid metrics: $values"
      STABLE_COUNTS[$gpu]=0
      continue
    fi

    echo \
      "[$(date '+%F %T')] GPU=$gpu " \
      "memory=${memory}MiB utilization=${utilization}%"

    if (( memory <= MAX_MEMORY_MIB \
          && utilization <= MAX_UTILIZATION ))
    then
      STABLE_COUNTS[$gpu]=$(( ${STABLE_COUNTS[$gpu]:-0} + 1 ))
    else
      STABLE_COUNTS[$gpu]=0
    fi

    if (( STABLE_COUNTS[$gpu] >= STABLE_CHECKS ))
    then
      SELECTED_GPU="$gpu"
      break
    fi
  done

  if [[ -z "$SELECTED_GPU" ]]; then
    echo \
      "[$(date '+%F %T')] No stable free GPU; " \
      "retry in ${SLEEP_SECONDS}s"

    sleep "$SLEEP_SECONDS"
  fi
done

echo \
  "[$(date '+%F %T')] Starting unified Eval400 " \
  "on physical GPU $SELECTED_GPU"

cd "$ROOT"

source "$ROOT/math/bin/activate"

export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
export PYTHONPATH="$CODE"
export PYTHONDONTWRITEBYTECODE=1

export HF_HOME="$ROOT/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# 确保评估阶段不可能调用老师 API。
unset TEACHER_API_KEY || true
unset TEACHER_BASE_URL || true
unset TEACHER_MODEL || true

python -u \
  "$CODE/scripts/evaluate_executable_sft.py" \
  --data "$DATA" \
  --base-model Qwen/Qwen3.5-9B-Base \
  --adapter "stage2=$STAGE2" \
  --adapter "main_grpo=$MAIN_GRPO" \
  --adapter "feedback_grpo=$FEEDBACK_GRPO" \
  --adapter "teacher_judge=$TEACHER_JUDGE" \
  --output-dir "$OUTPUT" \
  --max-new-tokens 2048 \
  --timeout-seconds 5 \
  --memory-fraction 0.40 \
  --local-memory-max 8g \
  --seed 42 \
  --resume \
  --retry-errors

python -u \
  "$CODE/scripts/compare_locked_eval.py" \
  --data "$DATA" \
  --result "stage2=$OUTPUT/stage2_results.jsonl" \
  --result "main_grpo=$OUTPUT/main_grpo_results.jsonl" \
  --result "feedback_grpo=$OUTPUT/feedback_grpo_results.jsonl" \
  --result "teacher_judge=$OUTPUT/teacher_judge_results.jsonl" \
  --baseline-label stage2 \
  --output-json "$OUTPUT/statistical_report.json" \
  --output-markdown "$OUTPUT/statistical_report.md"

echo \
  "[$(date '+%F %T')] " \
  "FROZEN UNIFIED EVAL400 COMPLETED"
