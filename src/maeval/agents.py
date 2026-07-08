"""Single-agent baseline vs. a planner→workers→critic multi-agent system.

Both return a Result (answer + calls + tokens + latency) the harness scores identically.
Mock agents are deterministic and key-free so the comparison runs offline; Claude agents
make real calls. The mock is calibrated to the realistic pattern: multi-agent covers more
of a decomposable task but costs several× the tokens/latency, and adds nothing on simple tasks.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field


@dataclass
class Result:
    answer: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    subtasks: list = field(default_factory=list)


def _mock_answer(key_points, covered_idx) -> str:
    return "Answer addressing: " + "; ".join(key_points[i] for i in covered_idx) + "."


class MockSingleAgent:
    name = "single"

    def run(self, task: dict) -> Result:
        kp = task["key_points"]
        rng = random.Random(hash(("single", task["id"])) & 0xFFFFFFFF)
        n_cov = max(1, math.floor(0.65 * len(kp)))                 # one pass: covers ~65%
        idx = sorted(rng.sample(range(len(kp)), n_cov))
        ans = _mock_answer(kp, idx)
        return Result(ans, calls=1, input_tokens=300 + 25 * len(kp),
                      output_tokens=40 * len(kp), latency_ms=850)


class MockMultiAgent:
    name = "multi"

    def run(self, task: dict) -> Result:
        kp = task["key_points"]
        rng = random.Random(hash(("multi", task["id"])) & 0xFFFFFFFF)
        decomposable = task.get("type") == "decomposable"
        # planner decomposes; simple tasks get over-decomposed (overhead, no quality gain)
        n_workers = max(2, math.ceil(len(kp) / 2)) if decomposable else len(kp)
        cov_frac = 0.92 if decomposable else 0.66                  # gain only when decomposable
        n_cov = max(1, math.floor(cov_frac * len(kp)))
        idx = sorted(rng.sample(range(len(kp)), n_cov))
        ans = _mock_answer(kp, idx)

        planner = (300, 60, 700)
        workers = [(260, 90, 900) for _ in range(n_workers)]       # run in parallel
        critic = (250 + 40 * n_workers, 120, 800)
        all_calls = [planner, *workers, critic]
        in_tok = sum(c[0] for c in all_calls)
        out_tok = sum(c[1] for c in all_calls)
        latency = planner[2] + max(w[2] for w in workers) + critic[2]  # parallel workers
        return Result(ans, calls=len(all_calls), input_tokens=in_tok, output_tokens=out_tok,
                      latency_ms=latency, subtasks=[f"subtask {i+1}" for i in range(n_workers)])


# -----------------------------------------------------------------------------
def _use_bedrock() -> bool:
    if os.getenv("MAEVAL_PROVIDER", "").lower() == "bedrock":
        return True
    if os.getenv("MAEVAL_PROVIDER", "").lower() == "anthropic":
        return False
    # auto: prefer Bedrock when AWS creds are present but no direct key
    return bool(os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE")) and not os.getenv("ANTHROPIC_API_KEY")


class _Claude:
    def __init__(self, model):
        if _use_bedrock():
            from anthropic import AnthropicBedrock

            self.client = AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))
            self.model = os.getenv("MAEVAL_BEDROCK_MODEL",
                                   "global.anthropic.claude-haiku-4-5-20251001-v1:0")
        else:
            import anthropic

            self.client = anthropic.Anthropic()
            self.model = model

    def call(self, prompt: str, system: str = "", max_tokens: int = 700):
        import time

        t0 = time.perf_counter()
        resp = self.client.messages.create(model=self.model, max_tokens=max_tokens,
                                           system=system or None,
                                           messages=[{"role": "user", "content": prompt}])
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return text, resp.usage.input_tokens, resp.usage.output_tokens, int((time.perf_counter() - t0) * 1000)


class ClaudeSingleAgent:
    name = "single"

    def __init__(self, model):
        self.llm = _Claude(model)

    def run(self, task: dict) -> Result:
        text, it, ot, ms = self.llm.call(task["question"],
                                         system="Answer thoroughly and concisely.")
        return Result(text, calls=1, input_tokens=it, output_tokens=ot, latency_ms=ms)


class ClaudeMultiAgent:
    name = "multi"

    def __init__(self, model):
        self.llm = _Claude(model)

    def run(self, task: dict) -> Result:
        # 1) planner: break the question into 2-4 subtasks
        plan, i1, o1, m1 = self.llm.call(
            f"Break this into 2-4 independent subtasks, one per line, no numbering:\n{task['question']}",
            system="You are a planner.", max_tokens=200)
        subtasks = [s.strip("-• ").strip() for s in plan.splitlines() if s.strip()][:4] or [task["question"]]
        in_tok, out_tok, lat = i1, o1, m1
        # 2) workers (sequential here; latency summed)
        partials = []
        worker_lat = []
        for st in subtasks:
            t, it, ot, ms = self.llm.call(f"Subtask: {st}\n(Context: {task['question']})",
                                          system="You are a focused worker.", max_tokens=400)
            partials.append(f"## {st}\n{t}")
            in_tok += it; out_tok += ot; worker_lat.append(ms)
        # 3) critic synthesizes
        synth, i3, o3, m3 = self.llm.call(
            "Synthesize these into one coherent answer:\n\n" + "\n\n".join(partials),
            system="You are an editor.", max_tokens=700)
        in_tok += i3; out_tok += o3
        lat += max(worker_lat) + m3  # workers parallelizable
        return Result(synth, calls=2 + len(subtasks), input_tokens=in_tok, output_tokens=out_tok,
                      latency_ms=lat, subtasks=subtasks)


def get_agents(model):
    """Real Claude agents when any credential is available (direct API or AWS Bedrock), else the
    calibrated mock. Force with MAEVAL_PROVIDER=bedrock|anthropic|mock."""
    forced = os.getenv("MAEVAL_PROVIDER", "").lower()
    if forced == "mock":
        return MockSingleAgent(), MockMultiAgent()
    have_creds = os.getenv("ANTHROPIC_API_KEY") or os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE")
    if forced in ("bedrock", "anthropic") or have_creds:
        return ClaudeSingleAgent(model), ClaudeMultiAgent(model)
    return MockSingleAgent(), MockMultiAgent()
