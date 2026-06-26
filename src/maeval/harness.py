"""Benchmark single-agent vs. multi-agent across the task set and quantify the trade-off:
quality (key-point coverage), cost (USD), latency, and call count — overall and split by
whether the task is decomposable.

    python -m maeval.harness               # offline mock (no key)
    python -m maeval.harness --live        # real Claude (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import yaml

from .agents import get_agents
from .config import MODEL, REPORTS, TASKS_PATH, cost_usd
from .quality import coverage


def run(tasks, single, multi) -> dict:
    rows = []
    for t in tasks:
        for agent in (single, multi):
            r = agent.run(t)
            rows.append({
                "task": t["id"], "type": t.get("type"), "system": agent.name,
                "quality": round(coverage(r.answer, t["key_points"]), 3),
                "cost_usd": round(cost_usd(r.input_tokens, r.output_tokens), 6),
                "latency_ms": r.latency_ms, "calls": r.calls,
            })
    return {"rows": rows, "summary": _aggregate(rows)}


def _aggregate(rows) -> dict:
    by_sys = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    out = {}
    for sys, rs in by_sys.items():
        n = len(rs)
        out[sys] = {
            "quality": round(sum(r["quality"] for r in rs) / n, 3),
            "cost_usd": round(sum(r["cost_usd"] for r in rs) / n, 6),
            "latency_ms": round(sum(r["latency_ms"] for r in rs) / n),
            "calls": round(sum(r["calls"] for r in rs) / n, 1),
        }
    # split quality by decomposable vs simple
    for kind in ("decomposable", "simple"):
        for sys, rs in by_sys.items():
            sel = [r for r in rs if r["type"] == kind]
            if sel:
                out.setdefault(f"{sys}_{kind}_quality", round(sum(r["quality"] for r in sel) / len(sel), 3))
    return out


def verdict(summary) -> str:
    s, m = summary["single"], summary["multi"]
    dq = (m["quality"] - s["quality"]) * 100
    cx = m["cost_usd"] / s["cost_usd"] if s["cost_usd"] else float("inf")
    lx = m["latency_ms"] / s["latency_ms"] if s["latency_ms"] else float("inf")
    return (f"multi-agent: {dq:+.1f} quality pts for {cx:.1f}× cost and {lx:.1f}× latency. "
            + ("Worth it only where the quality gain matters."
               if dq > 0 else "Not worth it — single-agent wins."))


def main() -> None:
    ap = argparse.ArgumentParser(description="single vs multi-agent trade-off benchmark")
    ap.add_argument("--live", action="store_true", help="use real Claude (needs key)")
    args = ap.parse_args()

    tasks = yaml.safe_load(open(TASKS_PATH))["tasks"]
    single, multi = get_agents(MODEL)
    res = run(tasks, single, multi)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "report.json").write_text(json.dumps(res, indent=2))

    s = res["summary"]
    print(f"\n{'system':8} {'quality':>8} {'cost$':>10} {'latency_ms':>11} {'calls':>6}")
    print("-" * 48)
    for sys in ("single", "multi"):
        m = s[sys]
        print(f"{sys:8} {m['quality']:>8.3f} {m['cost_usd']:>10.5f} {m['latency_ms']:>11} {m['calls']:>6}")
    print("-" * 48)
    print(f"decomposable tasks — single {s.get('single_decomposable_quality')} vs "
          f"multi {s.get('multi_decomposable_quality')}")
    print(f"simple tasks       — single {s.get('single_simple_quality')} vs "
          f"multi {s.get('multi_simple_quality')}")
    print("\n" + verdict(s))


if __name__ == "__main__":
    main()
