#!/usr/bin/env python3
"""Fail fast on teacher API configuration/authentication before loading the policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsg_rl.api_client import ChatAPIConfig, OpenAICompatibleChatClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as stream:
        config = json.load(stream)
    teacher = config.get("teacher", {})
    if not teacher.get("enabled", False):
        raise ValueError("teacher.enabled must be true")

    client = OpenAICompatibleChatClient(
        ChatAPIConfig.from_dict(teacher.get("api", {}), "teacher.api")
    )
    result = client.complete_json(
        [
            {
                "role": "system",
                "content": "Return exactly one JSON object with status set to ok.",
            },
            {"role": "user", "content": '{"health_check":true}'},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    if str(result.get("status", "")).strip().lower() != "ok":
        raise RuntimeError("Teacher API health check returned an unexpected payload")
    print("Teacher API authentication: PASS")


if __name__ == "__main__":
    main()
