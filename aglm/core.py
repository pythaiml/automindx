# SPDX-License-Identifier: Apache-2.0
# (c) 2024-2026 GATERAGE — aGLM
"""
AGLMCore — the Perceive-Orient-Decide-Act (PODA) cognitive cycle.

Distilled from mindX `agents/core/agint.py`. Each cycle:
  1. PERCEIVE — gather raw observations from external sources
  2. ORIENT   — update beliefs against the percept (BeliefSystem.revise)
  3. DECIDE   — pick an action; the picker is pluggable (LLM, rule, custom)
  4. ACT      — execute the picked action; record outcome as a new belief

Pluggable interfaces:
  - `Perceiver`   — async callable returning a PerceptionContext
  - `Decider`     — async callable returning a Decision given context + beliefs
  - `Actor`       — async callable executing a Decision, returning an outcome dict
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

from .beliefs import Belief, BeliefSystem

logger = logging.getLogger("aglm.core")


@dataclass
class PerceptionContext:
    """What the Perceiver observed this tick."""

    timestamp: float = field(default_factory=time.time)
    facts: Dict[str, Any] = field(default_factory=dict)
    source: str = "external"


@dataclass
class Decision:
    """What the Decider chose. Free-form; Actor consumes it."""

    action: str
    args: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.5


# Pluggable callable types.
Perceiver = Callable[[], Awaitable[PerceptionContext]]
Decider = Callable[[PerceptionContext, BeliefSystem], Awaitable[Decision]]
Actor = Callable[[Decision], Awaitable[Dict[str, Any]]]


class AGLMCore:
    """
    The PODA cycle. One call to `cycle()` runs one full Perceive →
    Orient → Decide → Act loop and returns the outcome dict.

    Wrap in `AutonomousLoop` to run it periodically.

    Example:
        async def perceive() -> PerceptionContext:
            return PerceptionContext(facts={"hour": datetime.now().hour})

        async def decide(ctx, beliefs) -> Decision:
            return Decision(action="log", args={"hour": ctx.facts["hour"]})

        async def act(d: Decision) -> dict:
            print("acting:", d.action, d.args)
            return {"success": True}

        core = AGLMCore(perceive=perceive, decide=decide, act=act)
        outcome = await core.cycle()
    """

    def __init__(
        self,
        perceive: Perceiver,
        decide: Decider,
        act: Actor,
        beliefs: Optional[BeliefSystem] = None,
        agent_id: str = "aglm.core",
    ):
        self.perceive_fn = perceive
        self.decide_fn = decide
        self.act_fn = act
        self.beliefs = beliefs or BeliefSystem()
        self.agent_id = agent_id

        self.cycle_count = 0
        self.last_cycle_started_at: Optional[float] = None
        self.last_outcome: Optional[Dict[str, Any]] = None

    async def cycle(self) -> Dict[str, Any]:
        """Run one PODA cycle. Returns the actor's outcome dict."""
        self.cycle_count += 1
        self.last_cycle_started_at = time.time()
        logger.info(f"{self.agent_id}: cycle {self.cycle_count} starting")

        # 1. Perceive
        try:
            ctx = await self.perceive_fn()
        except Exception as e:
            logger.warning(f"{self.agent_id}: perceive failed: {e}")
            self.last_outcome = {"success": False, "stage": "perceive", "error": str(e)}
            return self.last_outcome

        # 2. Orient — turn the percept into beliefs
        for claim, value in ctx.facts.items():
            self.beliefs.revise(
                claim=str(claim),
                new_confidence=0.7,  # observations come in with reasonable default confidence
                source=ctx.source,
                metadata={"value": value, "cycle": self.cycle_count},
            )

        # 3. Decide
        try:
            decision = await self.decide_fn(ctx, self.beliefs)
        except Exception as e:
            logger.warning(f"{self.agent_id}: decide failed: {e}")
            self.last_outcome = {"success": False, "stage": "decide", "error": str(e)}
            return self.last_outcome

        # 4. Act
        try:
            outcome = await self.act_fn(decision)
        except Exception as e:
            logger.warning(f"{self.agent_id}: act failed: {e}")
            self.last_outcome = {
                "success": False,
                "stage": "act",
                "decision": decision.action,
                "error": str(e),
            }
            return self.last_outcome

        # Update beliefs with the outcome.
        self.beliefs.add(Belief(
            claim=f"outcome:{decision.action}",
            confidence=1.0 if outcome.get("success") else 0.2,
            source=self.agent_id,
            metadata={"cycle": self.cycle_count, "decision": decision.action, **outcome},
        ))

        self.last_outcome = {
            "success": bool(outcome.get("success", True)),
            "stage": "complete",
            "cycle": self.cycle_count,
            "decision": decision.action,
            "rationale": decision.rationale,
            "outcome": outcome,
        }
        logger.info(f"{self.agent_id}: cycle {self.cycle_count} complete")
        return self.last_outcome

    def status(self) -> Dict[str, Any]:
        """Quick snapshot of where the loop is."""
        return {
            "agent_id": self.agent_id,
            "cycle_count": self.cycle_count,
            "last_cycle_started_at": self.last_cycle_started_at,
            "belief_count": len(self.beliefs),
            "last_outcome": self.last_outcome,
        }
