# FSG-RL

Code and reproducibility records for **Function-Structured Reinforcement Learning with Executable Verifiers for Mathematical Reasoning**.

FSG-RL represents a math problem as a directed graph of subquestions. Executable nodes have public Python interfaces. A policy writes one tagged implementation per node and a final answer. Execution, unit and property tests, dependency checks, program–answer agreement, and final-answer equivalence provide distinct training and evaluation signals.

## What the paper actually evaluates

The reported experiments use **offline, dataset-annotated function graphs** (`decomposition.backend=dataset`). The policy sees the problem and public graph; gold answers, target calls, and private tests are not included in its generation prompt. Main GRPO starts from executable SFT. Two separate 60-problem continuations use feedback-conditioned teacher guidance and reward-only teacher judging. Evaluation is teacher-free and does not retrieve memory. Online policy decomposition, memory retrieval, and other repository features are optional framework components, not results claimed for the reported Eval400 comparison.

The evaluation set is a custom graph-conditioned executable benchmark, **not** an official GSM8K, MathQA, MATH, or Omni-MATH test score. Its 400 problems comprise 97 GSM8K, 95 MathQA, 128 MATH, and 80 Omni-MATH items. See [`DATA.md`](DATA.md) for provenance and the public/private boundary.

## Repository map

| Paper stage | Code | Historical config | Released adapter |
| --- | --- | --- | --- |
| Graph-construction SFT | `scripts/train_sft.py`, `fsg_rl/sft_data.py` | `config/sft_terra_bs4_e3_config.json` | Stage-1 weights are not in this backup |
| Executable-solving SFT | `scripts/train_sft.py`, `fsg_rl/code_sft.py` | `config/stage2_code_sft.json` | `stage2_sft` |
| Main FSG-RL | `main.py`, `fsg_rl/grpo.py`, `fsg_rl/reward.py` | `config/grpo_medium_omni_v1_2gpu.json` | `main_grpo` |
| Feedback-conditioned GRPO | `fsg_rl/repair.py`, `fsg_rl/teacher.py` | `config/grpo_teacher_hard60.json` | `feedback_grpo` |
| Teacher-scored GRPO | `fsg_rl/teacher.py`, `fsg_rl/reward.py` | `config/grpo_teacher_judge_hard60.json` | `teacher_judge` |
| Locked evaluation | `scripts/evaluate_executable_sft.py`, `scripts/compare_locked_eval.py` | `reproducibility/protocol_manifest.json` | — |

Data preparation is in `scripts/prepare_*.py`; hidden-verifier generation and validation are in `scripts/generate_hidden_verifiers.py` and `scripts/validate_hidden_verifiers.py`. The historical server launcher and per-model summaries are retained in `reproducibility/`. The launcher contains original `/root/jjm/math` paths and must be adapted to a new environment. Historical JSON configs likewise contain server paths and are not portable without editing.

## Setup and checks

The model base is [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base), which is **not** included in the LoRA releases. Use a Python environment with compatible PyTorch, Transformers 5, PEFT, and the dependencies in `requirements.txt`. Training and local evaluation were run on Linux A100 GPUs; local code execution is resource-limited and should only be used with appropriately isolated infrastructure.

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py'
```

The original frozen evaluation required the full private scoring records, which are not in this GitHub repository. The public evaluation input is deliberately separated from answers and verifier tests. `scripts/export_public_eval.py` documents and checks that split. No API key is stored here; teacher training reads `TEACHER_API_KEY` from the environment.

## Reproducibility notes

The original protocol manifest records hashes for the 400-item scoring file, four adapters, evaluator, rollout implementation, and decoding settings. The historical main-GRPO config sets `beta_kl=0.03`; the two teacher continuations set `beta_kl=0.05`. The paper's latest draft should be checked against these recorded values before publication. The official reported stage comparison is in `reproducibility/statistical_report.md`.

The public benchmark and four adapter repositories will be linked here after their Hugging Face uploads complete.
