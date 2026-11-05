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


# ---------------------------------------------------------------------------
# Matja e kohës
# ---------------------------------------------------------------------------

class Timer:
    """
    Matës i thjeshtë kohe për përdorim me `with`:

        with Timer() as t:
            ...
        t.elapsed  # sekonda
    """

    def __init__(self):
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False


def measure_time(function, *args, **kwargs):
    """Ekzekuton funksionin dhe kthen (rezultati, sekondat)."""
    with Timer() as timer:
        result = function(*args, **kwargs)
    return result, timer.elapsed


# ---------------------------------------------------------------------------
# Formatimi
# ---------------------------------------------------------------------------

def format_count(value: int | None) -> str:
    """Numër me ndarës të mijëshëve (pikë, sipas përdorimit në shqip)."""
    if value is None:
        return "–"
    return f"{value:,}".replace(",", ".")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def format_big_number(value: int) -> str:
    """
    Numër shumë i madh në formë shkencore, p.sh. 9^51 → '≈ 4,64 × 10^48'.
    Numrat e vegjël (< 10^9) shfaqen të plotë.
    """
    if value < 10**9:
        return format_count(value)
    exponent = len(str(value)) - 1
    # rrumbullakim i saktë me numra të plotë në 3 shifra domethënëse
    scaled = (value * 1000 // 10**exponent + 5) // 10        # p.sh. 464
    if scaled >= 1000:                                        # 9,995 → 10,0
        scaled //= 10
        exponent += 1
    digits = str(scaled)
    return f"≈ {digits[0]},{digits[1:]} × 10^{exponent}"


# Etiketat dhe shpjegimet në shqip për çdo numërues të SearchStats
STAT_DESCRIPTIONS = {
    "generated": ("Gjendje të gjeneruara", "Kandidatë/gjendje të krijuar nga algoritmi."),
    "explored": ("Gjendje të eksploruara", "Nyje të zgjeruara (të vizituara) gjatë kërkimit."),
    "rejected": ("Gjendje të refuzuara", "Kandidatë që shkelën kufizimet ose u krasitën."),
    "valid": ("Gjendje të vlefshme", "Gjendje të pranuara (kuptimi sipas detyrës)."),
    "backtracks": ("Backtracks", "Sa herë u zhbë një caktim / u braktis një degë."),
    "max_depth": ("Thellësia maksimale", "Thellësia më e madhe e arritur në pemë."),
}


def stats_rows(stats: SearchStats, extra: dict | None = None) -> list[dict]:
    """
    Tabelë e statistikave reale të një ekzekutimi:
    [{"Metrika": ..., "Vlera": ..., "Shpjegimi": ...}, ...].
    `extra` = {etiketa: (vlera, shpjegimi)} për metrika specifike të detyrës
    (p.sh. madhësia maksimale e radhës, thellësia e zgjidhjes).
    """
    rows = []
    for key, (label, description) in STAT_DESCRIPTIONS.items():
        rows.append({"Metrika": label, "Vlera": format_count(getattr(stats, key)),
                     "Shpjegimi": description})
    for label, (value, description) in (extra or {}).items():
        shown = format_count(value) if isinstance(value, int) and not isinstance(value, bool) \
            else ("–" if value is None else str(value))
        rows.append({"Metrika": label, "Vlera": shown, "Shpjegimi": description})
    rows.append({"Metrika": "Koha e ekzekutimit", "Vlera": format_duration(stats.elapsed_seconds),
                 "Shpjegimi": "Koha reale e matur me perf_counter."})
    rows.append({"Metrika": "Kufiri i nyjeve",
                 "Vlera": format_count(stats.max_nodes) if stats.max_nodes else "pa kufi",
                 "Shpjegimi": "Kufiri i konfiguruar i nyjeve të eksploruara."})
    rows.append({"Metrika": "Timeout",
                 "Vlera": f"{stats.timeout_seconds:g} s" if stats.timeout_seconds else "pa kufi",
                 "Shpjegimi": "Kufiri i konfiguruar i kohës."})
    return rows


def termination_summary(stats: SearchStats) -> dict:
    """
    Përmbledhje e përfundimit për ndërfaqen:
      level = "success" (qëllimi u arrit), "info" (hapësira u shterua),
              "warning" (kërkimi u ndërpre nga kufiri i nyjeve/kohës).
    """
    if stats.termination_reason == TERMINATION_GOAL_REACHED:
        level = "success"
    elif stats.interrupted:
        level = "warning"
    else:
        level = "info"
    return {
        "reason": stats.termination_reason,
        "message": stats.termination_message,
        "interrupted": stats.interrupted,
        "level": level,
    }
