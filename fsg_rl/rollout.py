"""Qwen3.5 policy rollout backends for training and inference."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set

from .schemas import FunctionGraph, MemoryContext, PolicyRollout, Problem


@dataclass
class PolicyGeneration:
    """Backend-neutral result for non-rollout generations such as decomposition."""

    text: str
    prompt_text: str
    backend: str
    generation_seconds: float
    prompt_token_ids: List[int]
    completion_token_ids: List[int]


class PolicyBackend(Protocol):
    trainable: bool

    def generate_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float,
        max_new_tokens: int,
        seed_offset: int = 0,
        top_p: Optional[float] = None,
    ) -> PolicyGeneration: ...

    def rollout(
        self,
        problem: Problem,
        graph: FunctionGraph,
        memory_context: MemoryContext,
        rollout_index: int = 0,
        repair_feedback: Optional[Dict[str, Any]] = None,
    ) -> PolicyRollout: ...


class TransformersPolicy:
    """Loads Qwen3.5 locally through Hugging Face Transformers."""

    def __init__(self, config: Dict[str, Any], *, trainable: bool = False):
        self.config = config
        self.policy_config = config.get("policy", {})
        self.rollout_config = config.get("rollout", {})
        self.trainable = trainable
        self.model_name_or_path = str(self.policy_config["model_name_or_path"])
        self.policy_adapter_name: Optional[str] = None
        self.reference_adapter_name: Optional[str] = None
        self.processor, self.model = self._load_model()
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    def _load_model(self) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Install torch, transformers, and accelerate for the Transformers policy"
            ) from exc

        dtype_name = str(self.policy_config.get("dtype", "bfloat16"))
        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported policy dtype: {dtype_name!r}")

        load_kwargs: Dict[str, Any] = {
            "dtype": dtype_map[dtype_name],
            "device_map": self.policy_config.get("device_map", "auto"),
            "low_cpu_mem_usage": True,
            "trust_remote_code": bool(self.policy_config.get("trust_remote_code", False)),
        }
        quantization = str(self.policy_config.get("quantization", "none"))
        if quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("4-bit QLoRA requires bitsandbytes") from exc
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization != "none":
            raise ValueError("policy.quantization must be 'none' or '4bit'")

        processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=load_kwargs["trust_remote_code"],
        )
        try:
            model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name_or_path,
                **load_kwargs,
            )
        except ValueError as exc:
            if "qwen3_5" in str(exc).lower() or "configuration" in str(exc).lower():
                raise RuntimeError(
                    "Installed Transformers does not support Qwen3.5; upgrade transformers"
                ) from exc
            raise

        adapter_path = str(self.policy_config.get("adapter_name_or_path", "")).strip()
        if adapter_path:
            model = self._load_existing_adapter(
                model,
                adapter_path=adapter_path,
                quantization=quantization,
            )
        elif self.trainable:
            model = self._enable_parameter_efficient_training(model, quantization)
        else:
            model.eval()
        return processor, model

    def _load_existing_adapter(
        self, model: Any, *, adapter_path: str, quantization: str
    ) -> Any:
        try:
            from peft import PeftModel, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("Loading a LoRA adapter requires the peft package") from exc

        training = self.config.get("training", {})
        if self.trainable:
            if quantization == "4bit":
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=bool(
                        training.get("gradient_checkpointing", True)
                    ),
                )
            elif training.get("gradient_checkpointing", True):
                model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=self.trainable,
        )
        self.policy_adapter_name = "default"
        if self.trainable and float(self.config.get("grpo", {}).get("beta_kl", 0.0)) > 0:
            reference_path = str(
                self.policy_config.get("reference_adapter_name_or_path", adapter_path)
            ).strip()
            if not reference_path:
                raise ValueError("GRPO KL reference adapter path cannot be empty")
            self.reference_adapter_name = "fsg_reference"
            model.load_adapter(
                reference_path,
                adapter_name=self.reference_adapter_name,
                is_trainable=False,
            )
            model.set_adapter(self.policy_adapter_name)
            _set_adapter_requires_grad(model, self.reference_adapter_name, False)
        if self.trainable and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        return model.train() if self.trainable else model.eval()

    def _enable_parameter_efficient_training(self, model: Any, quantization: str) -> Any:
        training = self.config.get("training", {})
        if not training.get("use_lora", True):
            if quantization == "4bit":
                raise ValueError("4-bit training requires training.use_lora=true")
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            return model.train()

        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("Training with LoRA/QLoRA requires the peft package") from exc

        if quantization == "4bit":
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
            )
        elif training.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        lora = LoraConfig(
            r=int(training.get("lora_rank", 32)),
            lora_alpha=int(training.get("lora_alpha", 64)),
            lora_dropout=float(training.get("lora_dropout", 0.0)),
            bias="none",
            target_modules=training.get("lora_target_modules", "all-linear"),
        )
        model = get_peft_model(model, lora)
        if training.get("gradient_checkpointing", True) and hasattr(
            model, "enable_input_require_grads"
        ):
            model.enable_input_require_grads()
        return model.train()

    def rollout(
        self,
        problem: Problem,
        graph: FunctionGraph,
        memory_context: MemoryContext,
        rollout_index: int = 0,
        repair_feedback: Optional[Dict[str, Any]] = None,
    ) -> PolicyRollout:
        messages = build_policy_messages(problem, graph, memory_context, repair_feedback)
        generation = self.generate_messages(
            messages,
            temperature=float(self.rollout_config.get("temperature", 0.8)),
            max_new_tokens=int(self.rollout_config.get("max_new_tokens", 1536)),
            seed_offset=rollout_index,
        )
        return PolicyRollout(
            problem_id=problem.id,
            raw_text=generation.text,
            prompt_text=generation.prompt_text,
            backend=generation.backend,
            generation_seconds=generation.generation_seconds,
            repaired=repair_feedback is not None,
            prompt_token_ids=generation.prompt_token_ids,
            completion_token_ids=generation.completion_token_ids,
            span_token_ranges=find_tagged_token_ranges(
                generation.completion_token_ids,
                [node.id for node in graph.nodes],
                self.tokenizer,
            ),
        )

    def generate_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float,
        max_new_tokens: int,
        seed_offset: int = 0,
        top_p: Optional[float] = None,
    ) -> PolicyGeneration:
        import torch

        prompt_text, inputs = _render_chat_messages(
            self.processor,
            self.tokenizer,
            messages,
            enable_thinking=self.rollout_config.get("enable_thinking"),
        )
        device = _input_device(self.model)
        inputs = inputs.to(device)
        prompt_ids = inputs["input_ids"][0].tolist()

        do_sample = bool(self.rollout_config.get("do_sample", True)) and temperature > 0
        seed = self.rollout_config.get("seed")

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": float(self.rollout_config.get("repetition_penalty", 1.0)),
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        }
        generation_kwargs.update(
            _sampling_generation_kwargs(
                do_sample=do_sample,
                temperature=temperature,
                top_p=(
                    float(top_p)
                    if top_p is not None
                    else float(self.rollout_config.get("top_p", 0.95))
                ),
            )
        )

        started = time.monotonic()
        was_training = self.model.training
        self.model.eval()
        generation_context = torch.no_grad() if self.trainable else torch.inference_mode()
        seeded_sampling = do_sample and seed is not None
        rng_devices = []
        if seeded_sampling and getattr(device, "type", None) == "cuda":
            rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        rng_context = torch.random.fork_rng(devices=rng_devices) if seeded_sampling else nullcontext()
        try:
            with rng_context:
                if seeded_sampling:
                    torch.manual_seed(int(seed) + seed_offset)
                with generation_context:
                    generated = self.model.generate(**inputs, **generation_kwargs)
        finally:
            if was_training:
                self.model.train()
        elapsed = time.monotonic() - started

        completion_ids = generated[0, len(prompt_ids) :].tolist()
        raw_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return PolicyGeneration(
            text=raw_text,
            prompt_text=prompt_text,
            backend="transformers",
            generation_seconds=elapsed,
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
        )

    @property
    def reference_policy_description(self) -> str:
        if self.reference_adapter_name:
            return "frozen_initial_adapter"
        if callable(getattr(self.model, "disable_adapter", None)):
            return "base_checkpoint"
        return "unavailable"

    @contextmanager
    def reference_adapter_context(self) -> Any:
        """Activate the fixed GRPO reference policy without training it."""

        if self.reference_adapter_name:
            if not self.policy_adapter_name:
                raise RuntimeError("Reference adapter exists without a policy adapter")
            was_training = bool(self.model.training)
            self.model.set_adapter(self.reference_adapter_name)
            _set_adapter_requires_grad(self.model, self.reference_adapter_name, False)
            self.model.eval()
            try:
                yield
            finally:
                self.model.set_adapter(self.policy_adapter_name)
                if was_training:
                    self.model.train()
            return

        disable = getattr(self.model, "disable_adapter", None)
        with disable() if callable(disable) else nullcontext():
            yield


class VLLMPolicy:
    """Uses a running vLLM OpenAI-compatible server for inference/collection."""

    trainable = False

    def __init__(self, config: Dict[str, Any]):
        section = config.get("policy", {})
        self.rollout_config = config.get("rollout", {})
        self.model_name_or_path = str(section["model_name_or_path"])
        self.base_url = str(section.get("vllm_base_url", "http://127.0.0.1:8000/v1"))
        self.api_key_env = str(section.get("vllm_api_key_env", "VLLM_API_KEY"))
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package for the vLLM backend") from exc
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.environ.get(self.api_key_env, "EMPTY"),
        )

    def rollout(
        self,
        problem: Problem,
        graph: FunctionGraph,
        memory_context: MemoryContext,
        rollout_index: int = 0,
        repair_feedback: Optional[Dict[str, Any]] = None,
    ) -> PolicyRollout:
        messages = build_policy_messages(problem, graph, memory_context, repair_feedback)
        generation = self.generate_messages(
            messages,
            temperature=float(self.rollout_config.get("temperature", 0.8)),
            max_new_tokens=int(self.rollout_config.get("max_new_tokens", 1536)),
            seed_offset=rollout_index,
        )
        return PolicyRollout(
            problem_id=problem.id,
            raw_text=generation.text,
            prompt_text=generation.prompt_text,
            backend=generation.backend,
            generation_seconds=generation.generation_seconds,
            repaired=repair_feedback is not None,
        )

    def generate_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float,
        max_new_tokens: int,
        seed_offset: int = 0,
        top_p: Optional[float] = None,
    ) -> PolicyGeneration:
        started = time.monotonic()
        request: Dict[str, Any] = {
            "model": self.model_name_or_path,
            "messages": _flatten_messages(messages),
            "temperature": temperature,
            "top_p": (
                float(top_p)
                if top_p is not None
                else float(self.rollout_config.get("top_p", 0.95))
            ),
            "max_tokens": max_new_tokens,
        }
        if self.rollout_config.get("seed") is not None:
            request["seed"] = int(self.rollout_config["seed"]) + seed_offset
        response = self.client.chat.completions.create(**request)
        raw_text = response.choices[0].message.content or ""
        return PolicyGeneration(
            text=raw_text,
            prompt_text=json.dumps(messages, ensure_ascii=False),
            backend="vllm",
            generation_seconds=time.monotonic() - started,
            prompt_token_ids=[],
            completion_token_ids=[],
        )


def build_policy(config: Dict[str, Any], *, trainable: bool = False) -> PolicyBackend:
    backend = config.get("policy", {}).get("backend", "transformers")
    if backend == "transformers":
        return TransformersPolicy(config, trainable=trainable)
    if backend == "vllm":
        if trainable:
            raise ValueError("The vLLM backend cannot be updated in this process")
        return VLLMPolicy(config)
    raise ValueError(f"Unsupported policy backend: {backend!r}")


def build_policy_messages(
    problem: Problem,
    graph: FunctionGraph,
    memory_context: MemoryContext,
    repair_feedback: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    node_instructions = []
    for node in graph.nodes:
        node_instructions.append(
            {
                "tag": f"a_{node.id}",
                "name": node.name,
                "subquestion": node.question,
                "signature": node.signature,
                "expected_output_type": node.expected_output_type,
                "verification_spec": node.verification_spec,
            }
        )

    payload: Dict[str, Any] = {
        "problem": problem.text,
        "function_graph": graph.to_dict(),
        "retrieved_memory": memory_context.to_dict(),
        "required_output_spans": node_instructions,
    }
    if repair_feedback:
        payload["repair_feedback"] = repair_feedback

    system = """You are the trainable FSG-RL policy for algorithmic mathematical reasoning.
Solve every function-graph node and respect all dependency edges. For each node with id X,
output exactly one <a_X>...</a_X> span. Put executable Python in fenced python blocks when
requested. For every python_function node, include exactly one fenced python block defining
exactly the function named by that node's signature. Each Python block is executed alone in a
fresh Python 3 interpreter. It must be self-contained: include every required standard-library
import, constant, and helper function, and do not rely on imports, variables, functions, or code
from any other answer span. Reasoning may follow graph edges, but Python blocks do not share
runtime state. Only these Python standard-library modules may be imported: collections,
fractions, functools, heapq, itertools, math, operator, and statistics. Third-party packages,
including sympy, are forbidden. Before finishing each block, check that every referenced name is
defined or imported, that calls match their argument signatures, that dictionary keys exist, and
that representative and boundary calls return the intended type and value. Begin immediately
with the first required <a_X> tag. Do not emit a plan, scratchpad, meta-analysis, repeated prompt,
or commentary outside the required spans. Keep each span concise and solve the requested node
directly. The main span must end with the final answer in \\boxed{...}. Do not omit tags, invent
extra node tags, call external APIs, access files, or use the network. Use retrieved memory only
when applicable and verify boundary cases before answering."""
    if repair_feedback:
        system += (
            " This is a repair rollout. Correct the reported failure while regenerating the "
            "complete tagged solution; do not merely discuss the error."
        )
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        },
    ]


def find_tagged_token_ranges(
    completion_ids: Sequence[int],
    node_ids: Sequence[str],
    tokenizer: Any,
) -> Dict[str, List[int]]:
    ranges: Dict[str, List[int]] = {}
    for node_id in node_ids:
        open_ids = tokenizer.encode(f"<a_{node_id}>", add_special_tokens=False)
        close_ids = tokenizer.encode(f"</a_{node_id}>", add_special_tokens=False)
        open_at = _find_subsequence(completion_ids, open_ids, 0)
        if open_at < 0:
            continue
        content_start = open_at + len(open_ids)
        close_at = _find_subsequence(completion_ids, close_ids, content_start)
        if close_at < content_start:
            continue
        ranges[node_id] = [content_start, close_at]
    return ranges


def _find_subsequence(values: Sequence[int], query: Sequence[int], start: int) -> int:
    if not query:
        return -1
    last = len(values) - len(query) + 1
    for index in range(start, max(start, last)):
        if list(values[index : index + len(query)]) == list(query):
            return index
    return -1


def _input_device(model: Any) -> Any:
    try:
        return next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")
    except StopIteration as exc:
        raise RuntimeError("Policy model has no materialized parameters") from exc


def _render_chat_messages(
    processor: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    enable_thinking: Optional[bool] = None,
) -> tuple[str, Any]:
    """Render text messages even when a Base checkpoint's processor lacks a template."""

    template_kwargs: Dict[str, Any] = {
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = bool(enable_thinking)

    attempted: Set[int] = set()
    for template_owner in (processor, tokenizer):
        if id(template_owner) in attempted:
            continue
        attempted.add(id(template_owner))
        apply_template = getattr(template_owner, "apply_chat_template", None)
        if not callable(apply_template):
            continue
        try:
            prompt_text = apply_template(
                messages,
                tokenize=False,
                **template_kwargs,
            )
            inputs = apply_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **template_kwargs,
            )
            return str(prompt_text), inputs
        except ValueError as exc:
            if "chat template" not in str(exc).lower():
                raise

    prompt_text = _qwen_chatml_prompt(messages)
    inputs = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return prompt_text, inputs


def _sampling_generation_kwargs(
    *, do_sample: bool, temperature: float, top_p: float
) -> Dict[str, Any]:
    if not do_sample:
        return {}
    return {"temperature": temperature, "top_p": top_p}


def _set_adapter_requires_grad(model: Any, adapter_name: str, enabled: bool) -> None:
    marker = f".{adapter_name}."
    for name, parameter in model.named_parameters():
        if marker in name:
            parameter.requires_grad_(enabled)


def _qwen_chatml_prompt(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for message in _flatten_messages(messages):
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported Qwen ChatML role: {role!r}")
        parts.append(
            f"<|im_start|>{role}\n{message['content']}<|im_end|>\n"
        )
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _flatten_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    flattened = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", "")) for part in content if part.get("type") == "text"
            )
        flattened.append({"role": str(message["role"]), "content": str(content)})
    return flattened
