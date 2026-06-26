"""Answer quality = fraction of a task's gold key points the answer covers.

Heuristic + transparent (a content-word overlap test), so it runs key-free; an LLM-judge
can replace `covered` with the same (answer, key_points) -> bool signature for production.
"""
from __future__ import annotations

import re

_STOP = set("the a an of to and or in on for is are with per no not one and/or vs".split())


def _words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2}


def covered(answer: str, key_point: str) -> bool:
    kp = _words(key_point)
    if not kp:
        return False
    overlap = len(kp & _words(answer)) / len(kp)
    return overlap >= 0.5


def coverage(answer: str, key_points: list[str]) -> float:
    if not key_points:
        return 1.0
    return sum(covered(answer, kp) for kp in key_points) / len(key_points)
