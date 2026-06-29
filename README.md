# 💬 Multi-Agent Eval — does multi-agent actually beat single-agent?

> Multi-agent orchestration is the most over-hyped, under-measured space in applied AI. This
> runs a **planner→workers→critic** multi-agent system and a **single-agent baseline** over
> the same tasks and quantifies the trade-off: **answer quality, token cost, latency, and
> call count**. The senior insight isn't "I used multiple agents" — it's *proving when the
> coordination overhead is (and isn't) worth it*. Runs offline (calibrated mock); `--live` for Claude.

Spinning up five agents feels sophisticated. But each adds calls, tokens, and latency — and
on many tasks the quality gain is marginal or zero. This measures the gain against the cost,
split by whether the task actually decomposes.

---

## The two systems

| System | How it works | Calls |
|---|---|---|
| **single** | one agent answers the whole question | 1 |
| **multi** | **planner** splits → **workers** solve subtasks in parallel → **critic** synthesizes | 2 + #subtasks |

---

## Measured (`maeval`)

```
$ maeval
system    quality      cost$  latency_ms  calls
------------------------------------------------
single      0.500    0.00619         850    1.0
multi       0.639    0.01725        2400    4.7
------------------------------------------------
decomposable tasks — single 0.556 vs multi 0.833
simple tasks       — single 0.444 vs multi 0.444

multi-agent: +13.9 quality pts for 2.8× cost and 2.8× latency.
```

> *These numbers come from a **calibrated key-free mock** scored by a **lexical key-point overlap** metric — not live API runs or a semantic judge; `--live` runs the real pipeline to validate the pattern.*

Read the split, not the average: on **decomposable** tasks multi-agent jumps **0.556 → 0.833**
(+28 pts); on **simple** tasks it's **0.444 → 0.444** — *zero gain* for **2.8× the cost and
latency**. The orchestration earns its keep only when the task genuinely decomposes.

The finding: **multi-agent's quality edge is concentrated on genuinely decomposable tasks**
— on simple questions it spends several× the cost and latency for **no quality gain**. That
nuance (not "multi-agent is better/worse") is the hireable conclusion.

---

## Quickstart

> Uses the conda **`personal`** env (per environment conventions — never `base`).

```bash
PY=~/miniconda3/envs/personal/bin/python
$PY -m pip install -e ".[all]"

maeval                          # offline trade-off benchmark (mock agents)

export ANTHROPIC_API_KEY=sk-ant-...
maeval --live                   # real planner/workers/critic vs single agent on Claude
```

Writes `reports/report.json` (per-task quality/cost/latency + the aggregate verdict).

---

## What's measured

| Metric | Method |
|---|---|
| **quality** | fraction of a task's gold **key points** the answer covers (LLM-judge-swappable) |
| **cost_usd** | total tokens × per-model pricing, summed across every agent call |
| **latency_ms** | single = one call; multi = planner + max(worker) + critic (workers parallelized) |
| **calls** | number of model invocations |

---

## Repo layout

```
multi-agent-eval/
├── src/maeval/
│   ├── agents.py    single-agent + planner/workers/critic multi-agent (mock + Claude)
│   ├── quality.py   key-point coverage scorer
│   ├── harness.py   run both, aggregate, verdict  (CLI: maeval)
│   └── config.py    model, pricing, paths
├── tasks/tasks.yaml decomposable + simple tasks with gold key points
├── tests/           quality scorer + cost/latency trade-off + verdict — 5 cases
└── pyproject.toml · Dockerfile · Makefile · .github/workflows/ci.yml
```

---

## Résumé framing

> *Built a multi-agent evaluation harness benchmarking a planner→workers→critic system vs a
> single-agent baseline on quality (key-point coverage), token cost, latency, and call count;
> quantified the trade-off — multi-agent's quality gain concentrates on decomposable tasks and
> vanishes on simple ones at several× the cost — the judgment most "I used CrewAI" projects skip.*

## License
MIT (`LICENSE`).
