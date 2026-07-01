# SPDX-License-Identifier: Apache-2.0
# (c) 2024-2026 GATERAGE — aGLM: Autonomous General Learning Model
"""
aGLM — Autonomous General Learning Model

The modern agnostic distillation of the autonomous-learning-loop pattern
that runs inside mindX (`agents/core/mindXagent.py`, `agents/core/agint.py`).
This package extracts the reusable primitives — Perceive-Orient-Decide-Act
cycles, belief revision, and the autonomous improvement runner — and ships
them as an agnostic Apache-2.0 Python library.

Three primitives:
  - `AGLMCore`     — the PODA decision cycle (Perceive → Orient → Decide → Act)
  - `BeliefSystem` — claim + confidence + source attribution, with revision rules
  - `AutonomousLoop` — wraps AGLMCore in a periodic runner with backoff + recovery

Companion repos:
  - github.com/GATERAGE/RAGE       — retrieval substrate (memory + grounding)
  - github.com/GATERAGE/mastermind — strategic orchestrator (directive layer)

Together: RAGE remembers, aGLM decides, MASTERMIND orchestrates.

This package preserves the historical research code at the repo root (older
easyAGI / Professor Codephreak references). The `aglm/` subdirectory is the
modern reusable distribution.

mindX (github.com/agenticplace) is one consumer of these patterns; this repo
is the canonical agnostic home.
"""

from .beliefs import Belief, BeliefSystem
from .core import AGLMCore, Decision, PerceptionContext
from .cycle import AutonomousLoop

__version__ = "0.1.0"

__all__ = [
    "AGLMCore",
    "Decision",
    "PerceptionContext",
    "Belief",
    "BeliefSystem",
    "AutonomousLoop",
    "__version__",
]
