"""Chat-template encoding and batching utilities for supervised fine-tuning."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def encode_sft_messages(
    messages: Sequence[Dict[str, Any]],
    tokenizer: Any,
    *,
    template_owner: Any | None = None,
    max_seq_length: int,
) -> Dict[str, List[int]]:
    """Encode one conversation and mask every token except the final assistant turn."""

    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("SFT messages must end with one assistant response")
    if not str(messages[-1].get("content", "")).strip():
        raise ValueError("SFT assistant response cannot be empty")
    if max_seq_length < 32:
        raise ValueError("max_seq_length must be at least 32")

    prompt_messages = list(messages[:-1])
    full_messages = list(messages)
    prompt_text = render_chat_text(
        prompt_messages,
        tokenizer,
        template_owner=template_owner,
        add_generation_prompt=True,
    )
    full_text = render_chat_text(
        full_messages,
        tokenizer,
        template_owner=template_owner,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prefix_length = _common_prefix_length(prompt_ids, full_ids)
    if prefix_length == 0:
        raise ValueError("Chat template produced no common prompt prefix")

    completion_ids = full_ids[prefix_length:]
    if not completion_ids:
        raise ValueError("Chat template produced no assistant tokens")
    if len(completion_ids) >= max_seq_length:
        raise ValueError(
            "Assistant response alone exceeds max_seq_length; increase the limit or "
            "regenerate a shorter label"
        )

    # If the prompt is too long, retain its suffix, including the generation marker.
    kept_prompt = prompt_ids[-(max_seq_length - len(completion_ids)) :]
    input_ids = kept_prompt + completion_ids
    labels = [-100] * len(kept_prompt) + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def render_chat_text(
    messages: Sequence[Dict[str, Any]],
    tokenizer: Any,
    *,
    template_owner: Any | None = None,
    add_generation_prompt: bool,
) -> str:
    """Render with the processor/tokenizer template, then fall back to Qwen ChatML."""

    flattened = [_flatten_message(message) for message in messages]
    attempted = set()
    for owner in (template_owner, tokenizer):
        if owner is None or id(owner) in attempted:
            continue
        attempted.add(id(owner))
        apply_template = getattr(owner, "apply_chat_template", None)
        if not callable(apply_template):
            continue
        try:
            return str(
                apply_template(
                    flattened,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        except ValueError as exc:
            if "chat template" not in str(exc).lower():
                raise

    parts = []
    for message in flattened:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported Qwen ChatML role: {role!r}")
        parts.append(f"<|im_start|>{role}\n{message['content']}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def make_sft_collator(tokenizer: Any):
    """Create a right-padding collator without depending on TRL."""

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

    def collate(features: Sequence[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def _flatten_message(message: Dict[str, Any]) -> Dict[str, str]:
    content = message.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return {"role": str(message.get("role", "")), "content": str(content)}


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    index = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        index += 1
    return index
