"""
Monitorimi i ekzekutimit të algoritmeve të kërkimit.

`SearchStats` mban numëruesit që rriten VETËM nga algoritmi gjatë
ekzekutimit real. Asnjë vlerë nuk llogaritet apo simulohet pas faktit.

Kuptimi i numëruesve (i njëjtë për të gjitha detyrat):
    generated  – gjendje/kandidatë të gjeneruar nga algoritmi
    explored   – gjendje të pranuara dhe të zgjeruara (nyje të vizituara)
    rejected   – kandidatë të refuzuar sepse shkelin kufizimet ose u krasitën
    valid      – gjendje të plota të vlefshme (p.sh. javë të plota)
    backtracks – sa herë algoritmi zhbëri një caktim
    max_depth  – thellësia maksimale e arritur në pemën e kërkimit
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Arsyet e përfundimit të kërkimit
TERMINATION_GOAL_REACHED = "goal_reached"
TERMINATION_EXHAUSTED = "exhausted"
TERMINATION_NODE_LIMIT = "node_limit"
TERMINATION_TIMEOUT = "timeout"
TERMINATION_GENERATED_LIMIT = "generated_limit"

TERMINATION_MESSAGES = {
    TERMINATION_GOAL_REACHED: "Qëllimi u arrit (Goal State u gjet).",
    TERMINATION_EXHAUSTED: "Hapësira e kërkimit u eksplorua plotësisht.",
    TERMINATION_NODE_LIMIT: "Kërkimi u ndërpre: u arrit kufiri i nyjeve.",
    TERMINATION_GENERATED_LIMIT: "Kërkimi u ndërpre: u arrit kufiri i gjendjeve të gjeneruara.",
    TERMINATION_TIMEOUT: "Kërkimi u ndërpre: u arrit kufiri i kohës (timeout).",
}

# Sa shpesh kontrollohet ora (çdo N nyje), për të mos ngadalësuar kërkimin
_TIME_CHECK_INTERVAL = 256


class SearchLimitReached(Exception):
    """Ngrihet brenda kërkimit kur arrihet kufiri i nyjeve ose i kohës."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class SearchStats:
    """Numëruesit dhe kufijtë e një ekzekutimi të vetëm."""

    max_nodes: int | None = None
    timeout_seconds: float | None = None

    generated: int = 0
    explored: int = 0
    rejected: int = 0
    valid: int = 0
    backtracks: int = 0
    max_depth: int = 0

    termination_reason: str | None = None
    elapsed_seconds: float = 0.0

    _start: float = field(default=0.0, repr=False)

    def start(self) -> None:
        self._start = time.perf_counter()

    def elapsed_now(self) -> float:
        """Koha e kaluar që nga fillimi (pa e ndalur kërkimin)."""
        return time.perf_counter() - self._start

    def stop(self, reason: str) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start
        self.termination_reason = reason

    def record_explored(self, depth: int) -> None:
        """Regjistron një nyje të zgjeruar dhe kontrollon kufijtë."""
        self.explored += 1
        if depth > self.max_depth:
            self.max_depth = depth
        self.check_limits()

    def check_limits(self) -> None:
        if self.max_nodes is not None and self.explored >= self.max_nodes:
            raise SearchLimitReached(TERMINATION_NODE_LIMIT)
        if (
            self.timeout_seconds is not None
            and self.explored % _TIME_CHECK_INTERVAL == 0
            and time.perf_counter() - self._start >= self.timeout_seconds
        ):
            raise SearchLimitReached(TERMINATION_TIMEOUT)

    @property
    def interrupted(self) -> bool:
        return self.termination_reason in (
            TERMINATION_NODE_LIMIT,
            TERMINATION_TIMEOUT,
            TERMINATION_GENERATED_LIMIT,
        )

    @property
    def termination_message(self) -> str:
        return TERMINATION_MESSAGES.get(self.termination_reason, "I panjohur")

    def as_dict(self) -> dict:
        return {
            "generated": self.generated,
            "explored": self.explored,
            "rejected": self.rejected,
            "valid": self.valid,
            "backtracks": self.backtracks,
            "max_depth": self.max_depth,
            "elapsed_seconds": self.elapsed_seconds,
            "termination_reason": self.termination_reason,
            "max_nodes": self.max_nodes,
            "timeout_seconds": self.timeout_seconds,
        }
