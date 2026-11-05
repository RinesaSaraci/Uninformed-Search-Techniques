"""
Detyra 3 – Latin Square me IDDFS dhe Backtracking.

Modelimi
--------
Për një numër të plotë n ndërtohet matrica n×n me vlera 1..n.
Variables:     çdo qelizë X[i][j] e matricës.
Domain:        {1, 2, ..., n}.
Constraints:   çdo rresht dhe çdo kolonë përmban secilën vlerë saktësisht një herë.
Initial State: matrica bosh (asnjë qelizë e caktuar).
Goal Test:     të n² qelizat janë caktuar dhe rreshtat/kolonat janë të vlefshme.
Objective Function:
    f(s) = numri i shkeljeve: çifte qelizash me të njëjtën vlerë (≠ 0)
           në të njëjtin rresht ose në të njëjtën kolonë.
    F(s) = numri i qelizave të caktuara.
    Qëllimi: gjendja s* me f(s*) = 0 dhe F(s*) = n².
    Kërkimi krijon vetëm gjendje me f(s) = 0; çdo nivel e rrit F(s) me 1.

Thellësia
---------
Çdo qelizë e caktuar e rrit thellësinë me 1. Qelizat caktohen në rend
rresht pas rreshti, prandaj gjendja në thellësinë d ka caktuar qelizat 0..d−1.
Goal State ndodhet gjithmonë në thellësinë n².

IDDFS
-----
Për limit = 0, 1, 2, ...:
    ekzekutohet Depth-Limited DFS (DLS) nga matrica bosh;
    DLS cakton qeliza vetëm derisa thellësia të arrijë `limit`;
    nyjet në thellësinë `limit` nuk zgjerohen (cutoff);
    nëse DLS gjen Goal State → ndalim; përndryshe limit += 1.
Çdo iteracion fillon nga e para, prandaj nyjet e cekëta rivizitohen –
kjo është sjellja normale e IDDFS dhe duket në statistikat për iteracion.

Backtracking
------------
Brenda DLS, një vlerë që përsëritet në rresht ose kolonë refuzohet menjëherë
(nuk krijohet gjendje fëmijë). Pas kthimit nga një degë, caktimi zhbëhet
(fshihen bitet në maskat e rreshtit/kolonës) dhe provohet vlera e radhës.

Thyerja e simetrisë (opsionale, aktive si parazgjedhje)
-------------------------------------------------------
Çdo Latin Square mund të shndërrohet në formë të reduktuar (rreshti i parë
dhe kolona e parë = 1, 2, ..., n) duke ndërruar vetëm kolonat dhe rreshtat,
gjë që ruan vetinë Latin. Prandaj kufizimi i domenit X[0][j] = j+1 dhe
X[i][0] = i+1 nuk e humb ekzistencën e zgjidhjes. Këto qeliza caktohen
gjithsesi nga kërkimi (një vlerë e vetme në domen), jo nga një formulë.
Pa këtë thyerje, kërkimi bëhet shumë më i shtrenjtë edhe për n të vogël.

Matrica përfundimtare gjendet gjithmonë nga kërkimi; nuk përdoret asnjë
ndërtim ciklik apo formulë matematikore për përgjigjen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import factorial

from utils.statistics import (
    TERMINATION_EXHAUSTED,
    TERMINATION_GOAL_REACHED,
    SearchLimitReached,
    SearchStats,
)
from utils.validation import validate_search_limits

MIN_N = 1
MAX_N = 7          # n = 7 zakonisht arrin kufijtë (shih README); n ≤ 6 zgjidhet
DEFAULT_N = 5
DEFAULT_MAX_NODES = 2_000_000
DEFAULT_TIMEOUT_SECONDS = 20.0

# Numri i saktë i Latin Squares (OEIS A002860) dhe i atyre të reduktuar
# (OEIS A000315). Për n ≤ 4 rikontrollohen me numërim të plotë në teste.
KNOWN_LATIN_SQUARE_COUNTS = {
    1: 1, 2: 2, 3: 12, 4: 576, 5: 161_280, 6: 812_851_200,
    7: 61_479_419_904_000,
}
KNOWN_REDUCED_COUNTS = {
    1: 1, 2: 1, 3: 1, 4: 4, 5: 56, 6: 9_408, 7: 16_942_080,
}


class InvalidLatinSquareConfigError(ValueError):
    """Konfigurim i pavlefshëm; `errors` përmban mesazhet në shqip."""

    def __init__(self, errors: list[str]):
        super().__init__(" ".join(errors))
        self.errors = errors


# ---------------------------------------------------------------------------
# Validimi i hyrjes dhe landscape
# ---------------------------------------------------------------------------

def validate_n(n) -> list[str]:
    if not isinstance(n, int) or isinstance(n, bool):
        return ["n duhet të jetë numër i plotë."]
    if not MIN_N <= n <= MAX_N:
        return [f"n duhet të jetë ndërmjet {MIN_N} dhe {MAX_N}."]
    return []


def compute_landscape(n: int) -> dict:
    """
    Madhësitë teorike:
      unrestricted_assignments = n^(n²) – të gjitha mbushjet pa kufizime;
      row_permutation_bound    = (n!)^n – çdo rresht një permutacion
                                 (vetëm kufizimi i rreshtave);
      latin_squares / reduced  – numri i saktë i Latin Squares, vetëm për n ≤ 7.
    """
    errors = validate_n(n)
    if errors:
        raise InvalidLatinSquareConfigError(errors)
    return {
        "n": n,
        "dimensions": f"{n}×{n}",
        "cells": n * n,
        "domain_size": n,
        "goal_depth": n * n,
        "unrestricted_assignments": n ** (n * n),
        "row_permutation_bound": factorial(n) ** n,
        "latin_squares": KNOWN_LATIN_SQUARE_COUNTS.get(n),
        "reduced_latin_squares": KNOWN_REDUCED_COUNTS.get(n),
    }


# ---------------------------------------------------------------------------
# Validimi i pavarur i zgjidhjes
# ---------------------------------------------------------------------------

def verify_latin_square(matrix, n: int) -> tuple[bool, list[str]]:
    """
    Verifikon një matricë në mënyrë të pavarur nga kërkimi:
    dimensionet n×n, vlerat në 1..n, çdo rresht dhe kolonë = {1..n}.
    """
    errors: list[str] = []
    if not isinstance(matrix, (list, tuple)) or len(matrix) != n:
        return False, [f"Matrica duhet të ketë {n} rreshta."]
    for i, row in enumerate(matrix, start=1):
        if not isinstance(row, (list, tuple)) or len(row) != n:
            errors.append(f"Rreshti {i} duhet të ketë {n} qeliza.")
    if errors:
        return False, errors

    expected = set(range(1, n + 1))
    for i, row in enumerate(matrix, start=1):
        for j, value in enumerate(row, start=1):
            if not isinstance(value, int) or isinstance(value, bool) or value not in expected:
                errors.append(f"Qeliza ({i}, {j}) ka vlerë të pavlefshme: {value!r}.")
    if errors:
        return False, errors

    for i, row in enumerate(matrix, start=1):
        if set(row) != expected:
            errors.append(f"Rreshti {i} nuk përmban secilën vlerë 1..{n} saktësisht një herë.")
    for j in range(n):
        column = [matrix[i][j] for i in range(n)]
        if set(column) != expected:
            errors.append(f"Kolona {j + 1} nuk përmban secilën vlerë 1..{n} saktësisht një herë.")
    return (not errors, errors)


def objective_function(matrix, n: int) -> dict:
    """
    f(s) = çifte me vlerë të njëjtë (≠ 0) në të njëjtin rresht ose kolonë;
    F(s) = qeliza të caktuara (≠ 0); is_goal = f(s) = 0 dhe F(s) = n².
    """
    if not isinstance(matrix, (list, tuple)) or len(matrix) != n or any(
        not isinstance(row, (list, tuple)) or len(row) != n for row in matrix
    ):
        raise InvalidLatinSquareConfigError([f"Matrica duhet të jetë {n}×{n}."])
    lines = [list(row) for row in matrix] + [[matrix[i][j] for i in range(n)] for j in range(n)]
    violations = 0
    for line in lines:
        for v in set(line) - {0}:
            k = line.count(v)
            violations += k * (k - 1) // 2
    filled = sum(1 for row in matrix for v in row if v != 0)
    return {
        "violations": violations,
        "filled": filled,
        "empty": n * n - filled,
        "is_goal": violations == 0 and filled == n * n,
    }


def format_matrix(matrix) -> str:
    """Matrica si tekst, një rresht për rresht (dalja e kërkuar në detyrë)."""
    return "\n".join(" ".join(str(v) for v in row) for row in matrix)


# ---------------------------------------------------------------------------
# Rezultati
# ---------------------------------------------------------------------------

@dataclass
class IterationStats:
    """Statistikat e një iteracioni (një Depth-Limited DFS)."""

    iteration: int
    depth_limit: int
    explored: int = 0        # nyje të vizituara (përfshirë rrënjën)
    generated: int = 0       # vlera kandidate të provuara
    rejected: int = 0        # vlera që shkelin rreshtin/kolonën
    cutoffs: int = 0         # nyje në thellësinë limit (të pazgjeruara)
    backtracks: int = 0      # caktime të zhbëra
    max_depth: int = 0
    found: bool = False
    elapsed_seconds: float = 0.0


@dataclass
class LatinSquareResult:
    n: int
    solution: list[list[int]] | None
    solved: bool
    message: str
    landscape: dict
    stats: SearchStats                     # totalet e të gjitha iteracioneve
    iterations: list[IterationStats]
    final_depth_limit: int | None
    symmetry_breaking: bool
    is_valid: bool
    validation_errors: list[str]
    solution_path: list[tuple[int, int, int]] = field(default_factory=list)  # (rresht, kolonë, vlerë)
    trace: list[dict] = field(default_factory=list)

    @property
    def total_explored(self) -> int:
        return self.stats.explored


# ---------------------------------------------------------------------------
# IDDFS me Backtracking
# ---------------------------------------------------------------------------

def solve_latin_square(
    n: int = DEFAULT_N,
    max_nodes: int | None = DEFAULT_MAX_NODES,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    symmetry_breaking: bool = True,
    max_depth_limit: int | None = None,
    trace_limit: int = 0,
) -> LatinSquareResult:
    """
    Ndërton një Latin Square n×n me IDDFS.

    max_nodes        – kufiri i nyjeve të eksploruara gjithsej (të gjitha iteracionet).
    timeout_seconds  – kufiri i kohës.
    symmetry_breaking– True: forma e reduktuar (shih docstring-un e modulit).
    max_depth_limit  – limiti më i madh i thellësisë që provohet (parazgjedhje n²).
    trace_limit      – sa ngjarje të para të ruhen për vizualizim.
    """
    errors = validate_n(n)
    errors += validate_search_limits(max_nodes, timeout_seconds)
    if max_depth_limit is not None and (
        not isinstance(max_depth_limit, int) or isinstance(max_depth_limit, bool)
        or max_depth_limit < 0
    ):
        errors.append("Limiti maksimal i thellësisë duhet të jetë numër jo-negativ.")
    if not isinstance(trace_limit, int) or isinstance(trace_limit, bool) or trace_limit < 0:
        errors.append("Kufiri i gjurmës duhet të jetë numër jo-negativ.")
    if errors:
        raise InvalidLatinSquareConfigError(errors)

    landscape = compute_landscape(n)
    goal_depth = n * n
    if max_depth_limit is None:
        max_depth_limit = goal_depth

    full = (1 << (n + 1)) - 2                       # bitet 1..n
    # domeni i secilës qelizë si bitmask (me ose pa thyerje simetrie)
    domain = [full] * goal_depth
    if symmetry_breaking:
        for j in range(n):
            domain[j] = 1 << (j + 1)                # rreshti i parë: 1, 2, ..., n
        for i in range(1, n):
            domain[i * n] = 1 << (i + 1)            # kolona e parë: 1, 2, ..., n

    matrix = [[0] * n for _ in range(n)]
    row_used = [0] * n
    col_used = [0] * n
    stats = SearchStats(max_nodes=max_nodes, timeout_seconds=timeout_seconds)
    iterations: list[IterationStats] = []
    trace: list[dict] = []
    path: list[tuple[int, int, int]] = []

    def log(event: str, it: IterationStats, depth: int, cell=None, value=None) -> None:
        if len(trace) < trace_limit:
            trace.append({
                "event": event,
                "iteration": it.iteration,
                "depth_limit": it.depth_limit,
                "depth": depth,
                "cell": cell,
                "value": value,
                "matrix": [row[:] for row in matrix] if event in ("assign", "goal") else None,
            })

    def dls(depth: int, limit: int, it: IterationStats) -> bool:
        """Depth-Limited DFS. Kthen True nëse u gjet Goal State."""
        it.explored += 1
        if depth > it.max_depth:
            it.max_depth = depth
        stats.record_explored(depth)

        if depth == goal_depth:                     # Goal Test
            log("goal", it, depth)
            return True
        if depth == limit:                          # cutoff: mos zgjero
            it.cutoffs += 1
            log("cutoff", it, depth)
            return False

        i, j = divmod(depth, n)                     # rend i qëndrueshëm: rresht pas rreshti
        allowed = domain[depth]
        for value in range(1, n + 1):               # renditje rritëse e vlerave
            bit = 1 << value
            if not allowed & bit:
                continue                            # jashtë domenit (thyerja e simetrisë)
            it.generated += 1
            stats.generated += 1
            if (row_used[i] | col_used[j]) & bit:   # shkelje rreshti/kolone
                it.rejected += 1
                stats.rejected += 1
                log("reject", it, depth, (i, j), value)
                continue
            # caktimi
            matrix[i][j] = value
            row_used[i] |= bit
            col_used[j] |= bit
            stats.valid += 1
            path.append((i, j, value))
            log("assign", it, depth + 1, (i, j), value)
            if dls(depth + 1, limit, it):
                return True
            # Backtracking: zhbëjmë caktimin dhe provojmë vlerën e radhës
            path.pop()
            matrix[i][j] = 0
            row_used[i] &= ~bit
            col_used[j] &= ~bit
            it.backtracks += 1
            stats.backtracks += 1
            log("backtrack", it, depth, (i, j), value)
        return False

    stats.start()
    reason = TERMINATION_EXHAUSTED
    final_limit = None
    try:
        for limit in range(0, max_depth_limit + 1):
            it = IterationStats(iteration=len(iterations) + 1, depth_limit=limit)
            iterations.append(it)
            final_limit = limit
            started = time.perf_counter()
            try:
                found = dls(0, limit, it)
            finally:
                it.elapsed_seconds = time.perf_counter() - started
            if found:
                it.found = True
                reason = TERMINATION_GOAL_REACHED
                break
            if it.cutoffs == 0:
                # Asnjë nyje nuk u ndërpre nga limiti: rritja e limitit
                # nuk sjell gjendje të reja – hapësira është shteruar.
                break
    except SearchLimitReached as limit_reached:
        reason = limit_reached.reason
    stats.stop(reason)

    if reason == TERMINATION_GOAL_REACHED:
        solution = [row[:] for row in matrix]
        is_valid, validation_errors = verify_latin_square(solution, n)
        solved = is_valid
        message = (
            f"IDDFS gjeti Latin Square {n}×{n} në iteracionin {len(iterations)} "
            f"(limiti i thellësisë {final_limit}); zgjidhja u verifikua në mënyrë të pavarur."
            if is_valid else "Gabim: matrica e gjetur nuk kaloi verifikimin e pavarur."
        )
        solution_path = list(path)
    else:
        solution, solved, solution_path = None, False, []
        is_valid, validation_errors = False, []
        if reason == TERMINATION_EXHAUSTED:
            if final_limit is not None and final_limit < goal_depth and iterations[-1].cutoffs > 0:
                message = (f"U arrit limiti maksimal i thellësisë ({final_limit}) para thellësisë "
                           f"së qëllimit ({goal_depth}); zgjidhja nuk u gjet.")
            else:
                message = "Hapësira e kërkimit u shterua pa gjetur Latin Square."
        else:
            message = (f"{stats.termination_message} Latin Square nuk u gjet brenda kufijve; "
                       f"kjo NUK do të thotë që nuk ekziston.")

    return LatinSquareResult(
        n=n,
        solution=solution,
        solved=solved,
        message=message,
        landscape=landscape,
        stats=stats,
        iterations=iterations,
        final_depth_limit=final_limit,
        symmetry_breaking=symmetry_breaking,
        is_valid=is_valid,
        validation_errors=validation_errors,
        solution_path=solution_path,
        trace=trace,
    )


def solution_states(n: int, solution_path) -> list[list[list[int]]]:
    """Gjendjet nga matrica bosh deri te zgjidhja (një qelizë për hap)."""
    current = [[0] * n for _ in range(n)]
    states = [[row[:] for row in current]]
    for i, j, value in solution_path:
        current[i][j] = value
        states.append([row[:] for row in current])
    return states


def iterations_to_rows(iterations: list[IterationStats]) -> list[dict]:
    """Tabela e statistikave për çdo iteracion (për ndërfaqen)."""
    return [
        {
            "Iteracioni": it.iteration,
            "Limiti i thellësisë": it.depth_limit,
            "Explored": it.explored,
            "Generated": it.generated,
            "Rejected": it.rejected,
            "Cutoffs": it.cutoffs,
            "Backtracks": it.backtracks,
            "U gjet": "Po" if it.found else "Jo",
            "Koha (s)": round(it.elapsed_seconds, 4),
        }
        for it in iterations
    ]


# ---------------------------------------------------------------------------
# Ekzekutim i pavarur:  python -m tasks.latin_square
# ---------------------------------------------------------------------------

def _main() -> None:
    result = solve_latin_square()
    land = result.landscape
    print(f"Latin Square {land['dimensions']} – IDDFS me Backtracking")
    print(f"Qeliza: {land['cells']}, domeni: {land['domain_size']}, "
          f"n^(n²) = {land['unrestricted_assignments']:.3e}, "
          f"Latin Squares të vlefshme: {land['latin_squares']}")
    print(result.message)
    print(f"Explored gjithsej: {result.stats.explored}, Koha: {result.stats.elapsed_seconds:.3f}s")
    for row in iterations_to_rows(result.iterations)[-5:]:
        print(row)
    if result.solution:
        print(format_matrix(result.solution))
        obj = objective_function(result.solution, result.n)
        print(f"Objective: f(s) = {obj['violations']}, F(s) = {obj['filled']}")


if __name__ == "__main__":
    _main()
