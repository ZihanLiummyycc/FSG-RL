#!/usr/bin/env python3
"""Joint QLoRA SFT for graph construction and graph-conditioned solving."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fsg_rl.rollout import TransformersPolicy, _input_device
from fsg_rl.sft_training import encode_sft_messages, make_sft_collator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = config.get("training", {})
    data = config.get("sft_data", {})
    train_paths = _resolve_paths(data.get("train_files", []), config_path.parent)
    validation_paths = _resolve_paths(
        data.get("validation_files", []), config_path.parent
    )
    if not train_paths:
        parser.error("sft_data.train_files cannot be empty")

    train_records = _load_records(train_paths)
    validation_records = _load_records(validation_paths)
    seed = int(training.get("seed", 42))
    random.Random(seed).shuffle(train_records)
    if args.limit is not None:
        train_records = train_records[: args.limit]
        validation_records = validation_records[: max(1, args.limit // 10)]

    print(f"train_records={len(train_records)} validation_records={len(validation_records)}")
    print("Loading trainable Qwen policy...")
    policy = TransformersPolicy(config, trainable=True)
    tokenizer = policy.tokenizer
    max_seq_length = int(training.get("max_seq_length", 4096))
    train_features, train_rejected = _encode_records(
        train_records, tokenizer, policy.processor, max_seq_length
    )
    validation_features, validation_rejected = _encode_records(
        validation_records, tokenizer, policy.processor, max_seq_length
    )
    print(
        f"encoded_train={len(train_features)} rejected_train={train_rejected} "
        f"encoded_validation={len(validation_features)} "
        f"rejected_validation={validation_rejected}"
    )
    if not train_features:
        raise SystemExit("No trainable records remain after tokenization")
    if args.dry_run:
        print("Dry run passed: model loaded and SFT records tokenized; no update was run.")
        return

    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    batch_size = int(training.get("batch_size", 1))
    accumulation = int(training.get("gradient_accumulation_steps", 16))
    epochs = int(training.get("epochs", 2))
    collator = make_sft_collator(tokenizer)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_features,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_features,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    parameters = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training.get("learning_rate", 2e-5)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_steps = int(total_updates * float(training.get("warmup_ratio", 0.03)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    output_dir = _resolve_path(
        str(args.output_dir or training.get("output_dir", "../outputs/sft_qwen35_9b")),
        config_path.parent,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _input_device(policy.model)
    log_every = int(training.get("log_every_updates", 5))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    global_update = 0
    history = []
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()

    for epoch in range(1, epochs + 1):
        policy.model.train()
        running_loss = 0.0
        running_batches = 0
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = policy.model(**batch)
            loss = output.loss
            window_start = (batch_index // accumulation) * accumulation
            window_size = min(accumulation, len(train_loader) - window_start)
            (loss / window_size).backward()
            running_loss += float(loss.detach())
            running_batches += 1
            final_batch = batch_index + 1 == len(train_loader)
            if (batch_index + 1) % accumulation != 0 and not final_batch:
                continue
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            if global_update % log_every == 0 or final_batch:
                average = running_loss / max(1, running_batches)
                print(
                    f"epoch={epoch}/{epochs} update={global_update}/{total_updates} "
                    f"train_loss={average:.6f} lr={scheduler.get_last_lr()[0]:.3e}"
                )
                running_loss = 0.0
                running_batches = 0

        validation_loss = _evaluate(policy.model, validation_loader, device)
        history.append(
            {"epoch": epoch, "global_update": global_update, "validation_loss": validation_loss}
        )
        print(f"epoch={epoch} validation_loss={validation_loss}")
        if bool(training.get("save_each_epoch", True)):
            _save_checkpoint(policy, output_dir / f"epoch-{epoch}")

    _save_checkpoint(policy, output_dir / "final")
    state = {
        "base_model": policy.model_name_or_path,
        "train_records": len(train_features),
        "validation_records": len(validation_features),
        "rejected_train_records": train_rejected,
        "rejected_validation_records": validation_rejected,
        "global_updates": global_update,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "config": str(config_path),
    }
    (output_dir / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"SFT complete. Final adapter: {output_dir / 'final'}")


def _encode_records(
    records: Iterable[Dict[str, Any]],
    tokenizer: Any,
    processor: Any,
    max_seq_length: int,
) -> tuple[List[Dict[str, List[int]]], int]:
    features = []
    rejected = 0
    for record in records:
        messages = record.get("messages")
        if not isinstance(messages, list):
            rejected += 1
            continue
        try:
            features.append(
                encode_sft_messages(
                    messages,
                    tokenizer,
                    template_owner=processor,
                    max_seq_length=max_seq_length,
                )
            )
        except ValueError as exc:
            rejected += 1
            print(f"reject id={record.get('id')} reason={exc}")
    return features, rejected


def _evaluate(model: Any, loader: Any, device: Any) -> float | None:
    if len(loader) == 0:
        return None
    import torch

    was_training = model.training
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _save_checkpoint(policy: TransformersPolicy, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    policy.model.save_pretrained(path, safe_serialization=True)
    policy.processor.save_pretrained(path)


def _load_records(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"SFT file not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


def _resolve_paths(values: Iterable[str], config_dir: Path) -> List[Path]:
    return [_resolve_path(str(value), config_dir) for value in values]


def _resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


if __name__ == "__main__":
    main()
