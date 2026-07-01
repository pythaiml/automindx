# SPDX-License-Identifier: Apache-2.0
# (c) 2024-2026 GATERAGE — aGLM
"""
AutonomousLoop — periodic runner for AGLMCore.

Distilled from mindX `agents/orchestration/startup_agent.py` improvement
loop pattern. Features:
  - configurable cycle interval (default 300s like mindX)
  - circuit breaker: backoff after N consecutive failures
  - graceful start / stop via asyncio.Event
  - exception isolation: a failing cycle never kills the loop
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .core import AGLMCore

logger = logging.getLogger("aglm.cycle")


class AutonomousLoop:
    """Wrap an AGLMCore in a periodic runner."""

    def __init__(
        self,
        core: AGLMCore,
        interval_seconds: float = 300.0,
        max_consecutive_failures: int = 5,
        backoff_seconds: float = 120.0,
    ):
        self.core = core
        self.interval = interval_seconds
        self.max_failures = max_consecutive_failures
        self.backoff = backoff_seconds

        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._consecutive_failures = 0
        self._started_at: Optional[float] = None

    async def start(self) -> None:
        """Begin the periodic loop. Returns immediately; the loop runs as a task."""
        if self._task is not None and not self._task.done():
            logger.warning(f"{self.core.agent_id}: loop already running")
            return
        self._stop_event.clear()
        self._consecutive_failures = 0
        self._started_at = time.time()
        self._task = asyncio.create_task(self._run())
        logger.info(
            f"{self.core.agent_id}: autonomous loop started (interval={self.interval}s)"
        )

    async def stop(self) -> None:
        """Signal the loop to stop after the current cycle."""
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=self.interval + 10)
            except asyncio.TimeoutError:
                self._task.cancel()
                logger.warning(f"{self.core.agent_id}: loop did not stop cleanly")
        logger.info(f"{self.core.agent_id}: autonomous loop stopped")

    async def _run(self) -> None:
        """The actual loop body. Exception-isolated; never kills itself."""
        while not self._stop_event.is_set():
            try:
                outcome = await self.core.cycle()
                if outcome.get("success"):
                    self._consecutive_failures = 0
                else:
                    self._consecutive_failures += 1
                    logger.warning(
                        f"{self.core.agent_id}: cycle failed "
                        f"({self._consecutive_failures}/{self.max_failures}): "
                        f"{outcome.get('error', 'unknown')}"
                    )
            except Exception as e:
                # Defensive: AGLMCore.cycle should not raise, but guard anyway.
                self._consecutive_failures += 1
                logger.error(
                    f"{self.core.agent_id}: cycle raised {type(e).__name__}: {e} "
                    f"({self._consecutive_failures}/{self.max_failures})"
                )

            wait = self.interval
            if self._consecutive_failures >= self.max_failures:
                wait = self.backoff
                logger.warning(
                    f"{self.core.agent_id}: circuit-breaker OPEN — backing off {wait}s"
                )
                # Reset counter after backoff so we get a fresh window.
                self._consecutive_failures = 0

            # Wait for either the interval to elapse or stop signal.
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait)
            except asyncio.TimeoutError:
                pass  # interval elapsed; loop continues

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict:
        return {
            "is_running": self.is_running,
            "started_at": self._started_at,
            "interval_seconds": self.interval,
            "consecutive_failures": self._consecutive_failures,
            "max_failures": self.max_failures,
            "core": self.core.status(),
        }
