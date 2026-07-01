# SPDX-License-Identifier: Apache-2.0
# (c) 2024-2026 GATERAGE — aGLM
"""
Belief system — claim, confidence, source attribution.

Belief revision is monotonic by default (higher confidence wins). For
non-monotonic reasoning (revising in light of contradicting evidence),
extend `BeliefSystem` with `revise()`-style overrides.

Distilled from mindX `agents/core/belief_system.py`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("aglm.beliefs")


@dataclass
class Belief:
    """A single belief: a claim, who said it, when, and how confident."""

    claim: str
    confidence: float  # 0.0 .. 1.0
    source: str  # agent_id or external system id
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


class BeliefSystem:
    """
    In-memory belief store. Claims are keyed by a stable hash of their text;
    multiple beliefs about the same claim can co-exist (each with its own
    source + confidence).

    `top(claim)` returns the highest-confidence belief about a claim.
    `revise(claim, new)` adds a new belief and lets the caller decide
    whether to keep both or supersede.
    """

    def __init__(self) -> None:
        self._beliefs: Dict[str, List[Belief]] = {}

    @staticmethod
    def _key(claim: str) -> str:
        return claim.strip().lower()

    def add(self, belief: Belief) -> None:
        """Add a belief. Multiple beliefs about the same claim are allowed."""
        k = self._key(belief.claim)
        self._beliefs.setdefault(k, []).append(belief)

    def all(self, claim: str) -> List[Belief]:
        """All recorded beliefs about a claim, oldest first."""
        return list(self._beliefs.get(self._key(claim), []))

    def top(self, claim: str) -> Optional[Belief]:
        """Highest-confidence belief about a claim, or None."""
        entries = self._beliefs.get(self._key(claim), [])
        if not entries:
            return None
        return max(entries, key=lambda b: b.confidence)

    def revise(
        self,
        claim: str,
        new_confidence: float,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Belief:
        """
        Add a revised belief about a claim. The new belief gets recorded as a
        peer of any previous ones; callers query via `top()` to get the
        highest-confidence version.
        """
        b = Belief(
            claim=claim,
            confidence=new_confidence,
            source=source,
            metadata=metadata or {},
        )
        self.add(b)
        return b

    def claims(self) -> Iterable[str]:
        """All unique claims known to the system (lowercased keys)."""
        return list(self._beliefs.keys())

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Serialize the entire belief store (for persistence)."""
        return {k: [asdict(b) for b in v] for k, v in self._beliefs.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, List[Dict[str, Any]]]) -> "BeliefSystem":
        bs = cls()
        for _key, entries in data.items():
            for entry in entries:
                bs.add(Belief(**entry))
        return bs

    def __len__(self) -> int:
        return sum(len(v) for v in self._beliefs.values())

    def __contains__(self, claim: str) -> bool:
        return self._key(claim) in self._beliefs
