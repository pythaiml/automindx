# automind_aglm.py
# automindX × aGLM — wires the migrated Autonomous General Learning Model package
# (aglm/) to codephreak's realtime feedback (codephreak.py) so automindX can
# autonomously refine its persona.
#
# The aGLM PODA cycle (Perceive → Orient → Decide → Act):
#   PERCEIVE  read feedback stats from the self-improving engine
#   ORIENT    turn them into beliefs (AGLMCore does this automatically)
#   DECIDE    if the dislike ratio is high enough, decide to refine the persona
#   ACT       fold codephreak.py's learned directives into the persona
#
#   python3 automind_aglm.py            # run one cycle (demo)
#   python3 automind_aglm.py --loop     # run the autonomous loop (Ctrl-C to stop)

from __future__ import annotations

import asyncio
import sys

from aglm import AGLMCore, AutonomousLoop, Decision, PerceptionContext
from codephreak import SelfImprovingPersona

# Refine the persona once dislikes exceed this share of rated turns.
DISLIKE_THRESHOLD = 0.3
MIN_SAMPLES = 3


def build_agent(persona_id: str = "codephreak",
                engine: SelfImprovingPersona | None = None) -> AGLMCore:
    """An AGLMCore whose PODA cycle self-refines `persona_id` from feedback."""
    engine = engine or SelfImprovingPersona()

    async def perceive() -> PerceptionContext:
        by = engine.stats().get("by_persona", {}).get(persona_id, {"up": 0, "down": 0})
        total = by["up"] + by["down"]
        ratio = (by["down"] / total) if total else 0.0
        return PerceptionContext(
            facts={"rated": total, "dislike_ratio": round(ratio, 3), "persona": persona_id},
            source="codephreak.py",
        )

    async def decide(ctx: PerceptionContext, beliefs) -> Decision:
        if ctx.facts["rated"] >= MIN_SAMPLES and ctx.facts["dislike_ratio"] >= DISLIKE_THRESHOLD:
            return Decision(action="refine_persona",
                            args={"persona": persona_id},
                            rationale=f"dislike ratio {ctx.facts['dislike_ratio']} over "
                                      f"{ctx.facts['rated']} rated turns",
                            confidence=min(1.0, ctx.facts["dislike_ratio"] + 0.4))
        return Decision(action="hold", rationale="feedback within tolerance", confidence=0.6)

    async def act(decision: Decision) -> dict:
        if decision.action != "refine_persona":
            return {"success": True, "action": "hold"}
        directives = engine.directives(persona_id)
        improved = engine.improved_prompt(persona_id)
        return {"success": True, "action": "refine_persona",
                "learned": directives, "improved_len": len(improved)}

    return AGLMCore(perceive=perceive, decide=decide, act=act,
                    agent_id=f"automindx.aglm[{persona_id}]")


async def _demo(loop: bool) -> None:
    agent = build_agent("codephreak")
    if not loop:
        outcome = await agent.cycle()
        print("one PODA cycle →", outcome)
        print("beliefs:", agent.status()["belief_count"])
        return
    runner = AutonomousLoop(agent, interval_seconds=30.0)
    await runner.start()
    try:
        await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(_demo(loop="--loop" in sys.argv))
