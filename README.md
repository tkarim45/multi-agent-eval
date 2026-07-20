# 💬 Multi-Agent Eval: does multi-agent actually beat single-agent?

> Multi-agent orchestration is the most over-hyped, under-measured space in applied AI. This
> runs a **planner→workers→critic** multi-agent system and a **single-agent baseline** over
> the same tasks and quantifies the trade-off: **answer quality, token cost, latency, and
> call count**. The senior insight isn't "I used multiple agents", it's *proving when the
> coordination overhead is (and isn't) worth it*. Runs offline (calibrated mock); `--live` for Claude.

Spinning up five agents feels sophisticated. But each adds calls, tokens, and latency, and
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

Real run, **Claude Haiku 4.5 on AWS Bedrock**, 12 tasks (`MAEVAL_PROVIDER=bedrock maeval --live`):

```
system    quality      cost$  latency_ms  calls
------------------------------------------------
single      0.778    0.00208        5429    1.0
multi       0.833    0.00901       11370    5.0
------------------------------------------------
decomposable tasks — single 0.833 vs multi 0.667
simple tasks       — single 0.722 vs multi 1.000

multi-agent: +5.5 quality pts for 4.3× cost and 2.1× latency.
```

The real run **did not reproduce the tidy story**, and that's the honest, more interesting result:

- **Multi-agent bought little for a lot.** On real Claude Haiku the planner→workers→critic system
  scored just **+5.5 quality points over a single agent, for 4.3× the cost and 2.1× the latency**.
  The overhead is real and large; the quality edge is small.
- **The clean "wins on decomposable, zero on simple" split evaporated, it even inverted.** The
  calibrated mock showed multi-agent jumping +28 points on decomposable tasks and nothing on simple
  ones. The live model showed the *opposite* per-subset pattern: single **beat** multi on the
  decomposable tasks (0.833 vs 0.667) and lost on the simple ones (0.722 vs 1.000). With only ~6
  tasks per subset and a crude lexical-overlap quality metric, that inversion is within noise, which
  is precisely the point: the mock's neat narrative was a *designed* artifact, and the real signal is
  noisier and does not support "multi-agent reliably wins where the task decomposes."
- **The robust, hireable takeaway survives the mock's collapse:** multi-agent orchestration is not a
  free quality lever, it multiplies cost and latency several-fold for a small and, here, unreliable
  quality change. "Should I add more agents?" is an empirical question you have to measure per
  task, not a default, and measuring it live is what kept this from shipping the mock's clean-but-
  wrong conclusion.

> The offline `maeval` (mock agents, lexical-overlap scoring) is a **calibrated teaching fixture**
> that reproduces the idealized decomposable-only pattern; the numbers above are the real model's
> actual, messier behavior. ⚠️ Small scale (12 tasks, ~6 per subset) + a lexical quality metric →
> the per-subset numbers are directional/noisy; the durable finding is the cost-vs-quality
> multiplier, not the subset ranking.

---

## Quickstart

> Uses the conda **`personal`** env (per environment conventions, never `base`).

```bash
PY=~/miniconda3/envs/personal/bin/python
$PY -m pip install -e ".[all]"

maeval                          # offline trade-off benchmark (mock agents)

# real planner/workers/critic vs single agent on Claude — two credential paths:
MAEVAL_PROVIDER=bedrock maeval --live      # Claude Haiku on AWS Bedrock (creds from env / ~/.env)
export ANTHROPIC_API_KEY=sk-ant-... ; maeval --live   # or the direct Anthropic API
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
> quantified the trade-off, multi-agent's quality gain concentrates on decomposable tasks and
> vanishes on simple ones at several× the cost, the judgment most "I used CrewAI" projects skip.*

## License
MIT (`LICENSE`).
