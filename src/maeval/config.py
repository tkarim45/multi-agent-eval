"""Model, pricing, paths."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("MAEVAL_ROOT", Path(__file__).resolve().parents[2]))
TASKS_PATH = ROOT / "tasks" / "tasks.yaml"
REPORTS = ROOT / "reports"

MODEL = os.getenv("MAEVAL_MODEL", "claude-opus-4-8")
PRICE = {"input": 1.0, "output": 5.0}  # USD / 1M tokens (Claude Haiku 4.5 — the default live/Bedrock model)


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 1e6 * PRICE["input"] + output_tokens / 1e6 * PRICE["output"]
