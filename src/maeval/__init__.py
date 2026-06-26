"""maeval — does multi-agent actually beat single-agent?

Runs a planner→workers→critic multi-agent system and a single-agent baseline over the same
tasks, and measures the trade-off: answer quality, token cost, latency, and call count — so
the (often surprising) verdict is quantified, not assumed.
"""

__version__ = "0.1.0"
