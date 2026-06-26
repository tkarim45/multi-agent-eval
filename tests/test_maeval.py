"""Quality scorer + the trade-off the harness must surface (key-free mock)."""
import yaml

from maeval.agents import MockMultiAgent, MockSingleAgent
from maeval.config import TASKS_PATH
from maeval.harness import run, verdict
from maeval.quality import coverage, covered


def test_coverage_scoring():
    kp = ["GDPR compliance", "data residency", "VAT and pricing"]
    assert covered("we ensure GDPR compliance", "GDPR compliance") is True
    assert covered("nothing relevant here", "GDPR compliance") is False
    assert coverage("GDPR compliance and data residency handled", kp) == round(2 / 3, 3) or \
           coverage("GDPR compliance and data residency handled", kp) > 0.6


def _tasks():
    return yaml.safe_load(open(TASKS_PATH))["tasks"]


def test_agents_return_metrics():
    t = _tasks()[0]
    for agent in (MockSingleAgent(), MockMultiAgent()):
        r = agent.run(t)
        assert r.calls >= 1 and r.input_tokens > 0 and r.latency_ms > 0


def test_multi_costs_more_than_single():
    res = run(_tasks(), MockSingleAgent(), MockMultiAgent())
    s = res["summary"]
    assert s["multi"]["cost_usd"] > s["single"]["cost_usd"]      # more calls -> more cost
    assert s["multi"]["latency_ms"] > s["single"]["latency_ms"]
    assert s["multi"]["calls"] > s["single"]["calls"]


def test_quality_gain_concentrated_in_decomposable():
    res = run(_tasks(), MockSingleAgent(), MockMultiAgent())["summary"]
    # multi-agent's quality edge shows on decomposable tasks, not simple ones
    dec_gain = res["multi_decomposable_quality"] - res["single_decomposable_quality"]
    simple_gain = res["multi_simple_quality"] - res["single_simple_quality"]
    assert dec_gain > simple_gain


def test_verdict_string():
    res = run(_tasks(), MockSingleAgent(), MockMultiAgent())["summary"]
    v = verdict(res)
    assert "× cost" in v and "quality pts" in v
