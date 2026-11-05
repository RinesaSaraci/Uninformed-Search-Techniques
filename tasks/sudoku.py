"""
Detyra 2 – Sudoku Solver me BFS dhe Backtracking.

Modelimi
--------
Variables:   qelizat bosh të tabelës 9×9 (0 = bosh).
Domain:      {1, ..., 9}.
Constraints: asnjë vlerë e përsëritur në rresht, kolonë apo nënrrjetë 3×3.
Initial State: enigma origjinale.
Goal Test:   të 81 qelizat janë plotësuar dhe të gjitha kufizimet plotësohen.
Objective Function:
    f(s) = V(s) = numri i shkeljeve të kufizimeve në tabelën s, ku një shkelje
           është një çift qelizash me të njëjtën vlerë (≠ 0) në të njëjtin
           rresht, kolonë ose nënrrjetë 3×3 (numërohet për secilën njësi).
    F(s) = numri i qelizave të plotësuara (progresi drejt qëllimit).
    Qëllimi: min f(s) me kusht F(s) = 81, pra gjendja s* me
             f(s*) = 0 dhe F(s*) = 81.
    BFS krijon vetëm gjendje me f(s) = 0; çdo nivel e rrit F(s) me 1.

Kërkimi (BFS)
-------------
Çdo gjendje është një tabelë e pjesshme. Përdoret radha FIFO
(`collections.deque`):
  1. radha inicializohet me enigmën origjinale;
  2. merret gjendja më e vjetër (popleft);
  3. kontrollohet Goal Test;
  4. zgjidhet një qelizë bosh (MRV – qeliza me më pak vlera të mundshme);
  5. llogariten vlerat e vlefshme të saj;
  6. për çdo vlerë të vlefshme krijohet një gjendje fëmijë;
  7. fëmijët shtohen në FUND të radhës (append).
Çdo kalim cakton saktësisht një qelizë, prandaj niveli i BFS = numri i
caktimeve. MRV vetëm zgjedh cilën qelizë të zgjerojmë; radha mbetet FIFO,
kështu gjendjet zgjerohen nivel pas niveli (jo në thellësi).

Backtracking brenda BFS
-----------------------
  * Caktimet që shkelin kufizimet refuzohen para se të krijohet gjendja
    (numëruesi `rejected`).
  * Nëse në një gjendje ndonjë qelizë bosh nuk ka asnjë vlerë të mundshme
    (dead end), e gjithë dega braktiset; kërkimi vazhdon me alternativat
    e tjera që janë tashmë në radhë (vëllezërit dhe kushërinjtë e saj).
    Ky është braktisja e degëve të pavlefshme (numëruesi `dead_ends`).
  * Meqë çdo gjendje ruhet si kopje e pavarur, nuk nevojitet "zhbërje"
    eksplicite – alternativat ekzistojnë si gjendje të veçanta në radhë.

Reprezentimi kompakt: tabela = `bytes` me 81 bajte; rruga e caktimeve
= `bytes` me indekset e qelizave të caktuara (për rindërtimin e hapave).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import log10, prod

from data.sudoku_puzzles import DIFFICULTIES, PUZZLES
from utils.statistics import (
    TERMINATION_EXHAUSTED,
    TERMINATION_GENERATED_LIMIT,
    TERMINATION_GOAL_REACHED,
    SearchLimitReached,
    SearchStats,
)
from utils.validation import validate_search_limits

SIZE = 9
CELLS = 81
DEFAULT_MAX_EXPLORED = 200_000
DEFAULT_MAX_GENERATED = 500_000
DEFAULT_TIMEOUT_SECONDS = 20.0

ALL_VALUES_MASK = 0b1111111110        # bitet 1..9

# Indekset (rreshti, kolona, kutia) për secilën qelizë 0..80
ROW_OF = tuple(i // 9 for i in range(CELLS))
COL_OF = tuple(i % 9 for i in range(CELLS))
BOX_OF = tuple((i // 27) * 3 + (i % 9) // 3 for i in range(CELLS))


class InvalidSudokuError(ValueError):
    """Tabelë e pavlefshme; `errors` përmban mesazhet në shqip."""

    def __init__(self, errors: list[str]):
        super().__init__(" ".join(errors))
        self.errors = errors


# ---------------------------------------------------------------------------
# Konvertimet dhe enigmat
# ---------------------------------------------------------------------------

def list_difficulties() -> list[tuple[str, str]]:
    """Kthen [(çelësi, emri në shqip), ...]."""
    return [(key, PUZZLES[key]["label"]) for key in DIFFICULTIES]


def get_puzzle(difficulty: str) -> list[list[int]]:
    """Kthen një kopje të re të enigmës (përdoret edhe për Rivendos)."""
    if difficulty not in PUZZLES:
        raise InvalidSudokuError([f"Vështirësi e panjohur: {difficulty}."])
    return string_to_grid(PUZZLES[difficulty]["puzzle"])


def get_reference_solution(difficulty: str) -> list[list[int]]:
    if difficulty not in PUZZLES:
        raise InvalidSudokuError([f"Vështirësi e panjohur: {difficulty}."])
    return string_to_grid(PUZZLES[difficulty]["solution"])


def string_to_grid(text: str) -> list[list[int]]:
    cleaned = text.replace(".", "0")
    return [[int(cleaned[r * 9 + c]) for c in range(9)] for r in range(9)]


def grid_to_bytes(grid: list[list[int]]) -> bytes:
    return bytes(value for row in grid for value in row)


def bytes_to_grid(board: bytes) -> list[list[int]]:
    return [list(board[r * 9:(r + 1) * 9]) for r in range(9)]


def copy_grid(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in grid]


# ---------------------------------------------------------------------------
# Validimi
# ---------------------------------------------------------------------------

def _find_duplicates(values: list[int]) -> list[int]:
    seen, dups = set(), set()
    for v in values:
        if v == 0:
            continue
        if v in seen:
            dups.add(v)
        seen.add(v)
    return sorted(dups)


def validate_grid(grid) -> list[str]:
    """
    Validon një tabelë (enigmë ose zgjidhje). Kontrollon formën 9×9,
    vlerat 0..9 dhe përsëritjet në rreshta, kolona dhe nënrrjeta 3×3.
    """
    if not isinstance(grid, (list, tuple)) or len(grid) != 9:
        return ["Tabela duhet të ketë saktësisht 9 rreshta."]
    errors: list[str] = []
    for r, row in enumerate(grid, start=1):
        if not isinstance(row, (list, tuple)) or len(row) != 9:
            errors.append(f"Rreshti {r} duhet të ketë saktësisht 9 qeliza.")
            continue
        for c, value in enumerate(row, start=1):
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"Qeliza ({r}, {c}) duhet të jetë numër i plotë.")
            elif not 0 <= value <= 9:
                errors.append(f"Qeliza ({r}, {c}) ka vlerë {value}; lejohen 0–9 (0 = bosh).")
    if errors:
        return errors

    for r in range(9):
        for v in _find_duplicates(list(grid[r])):
            errors.append(f"Rreshti {r + 1}: vlera {v} përsëritet.")
    for c in range(9):
        for v in _find_duplicates([grid[r][c] for r in range(9)]):
            errors.append(f"Kolona {c + 1}: vlera {v} përsëritet.")
    for b in range(9):
        br, bc = (b // 3) * 3, (b % 3) * 3
        cells = [grid[br + i][bc + j] for i in range(3) for j in range(3)]
        for v in _find_duplicates(cells):
            errors.append(f"Nënrrjeta 3×3 nr. {b + 1}: vlera {v} përsëritet.")
    return errors


def count_violations(grid) -> int:
    """
    f(s): numri i çifteve të qelizave me të njëjtën vlerë (≠ 0) brenda
    të njëjtës njësi (rresht, kolonë, nënrrjetë 3×3). Një çift që ndan
    dy njësi (p.sh. rresht dhe kuti) numërohet në secilën.
    """
    units = [list(grid[r]) for r in range(9)]
    units += [[grid[r][c] for r in range(9)] for c in range(9)]
    units += [
        [grid[br + i][bc + j] for i in range(3) for j in range(3)]
        for br in range(0, 9, 3)
        for bc in range(0, 9, 3)
    ]
    total = 0
    for unit in units:
        for v in range(1, 10):
            k = unit.count(v)
            total += k * (k - 1) // 2
    return total


def objective_function(grid) -> dict:
    """
    Vlerëson objective function për një tabelë 9×9 me vlera 0..9:
      violations = f(s), filled = F(s), empty = 81 − F(s),
      is_goal    = (f(s) = 0 dhe F(s) = 81).
    """
    if not isinstance(grid, (list, tuple)) or len(grid) != 9 or any(
        not isinstance(row, (list, tuple)) or len(row) != 9 for row in grid
    ):
        raise InvalidSudokuError(["Tabela duhet të jetë 9×9."])
    violations = count_violations(grid)
    filled = sum(1 for row in grid for v in row if v != 0)
    return {
        "violations": violations,
        "filled": filled,
        "empty": CELLS - filled,
        "is_goal": violations == 0 and filled == CELLS,
    }


def is_complete_solution(grid) -> bool:
    """Goal Test i pavarur: 81 qeliza të plota dhe pa shkelje kufizimesh."""
    return not validate_grid(grid) and all(v != 0 for row in grid for v in row)


def preserves_clues(puzzle, solution) -> bool:
    """A i ruan zgjidhja të gjitha shifrat fillestare të enigmës?"""
    return all(
        puzzle[r][c] == 0 or puzzle[r][c] == solution[r][c]
        for r in range(9)
        for c in range(9)
    )


# ---------------------------------------------------------------------------
# Landscape
# ---------------------------------------------------------------------------

def _masks(board: bytes) -> tuple[list[int], list[int], list[int]]:
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(board):
        if v:
            bit = 1 << v
            rows[ROW_OF[i]] |= bit
            cols[COL_OF[i]] |= bit
            boxes[BOX_OF[i]] |= bit
    return rows, cols, boxes


def _candidate_mask(i: int, rows, cols, boxes) -> int:
    return ALL_VALUES_MASK & ~(rows[ROW_OF[i]] | cols[COL_OF[i]] | boxes[BOX_OF[i]])


def compute_landscape(grid) -> dict:
    """
    Madhësitë teorike të problemit:
      unrestricted_upper_bound = 9^k (k = qeliza bosh), pa asnjë kufizim;
      initial_candidate_product = prodhimi i numrit të vlerave të mundshme
        të çdo qelize bosh në gjendjen fillestare (kufi më i ngushtë).
    Këto janë kufij teorikë, jo numri i gjendjeve që BFS gjeneron realisht.
    """
    errors = validate_grid(grid)
    if errors:
        raise InvalidSudokuError(errors)
    board = grid_to_bytes(grid)
    empty = board.count(0)
    rows, cols, boxes = _masks(board)
    counts = [_candidate_mask(i, rows, cols, boxes).bit_count()
              for i in range(CELLS) if board[i] == 0]
    product = prod(counts) if counts else 1
    return {
        "grid_size": "9×9",
        "total_cells": CELLS,
        "fixed_cells": CELLS - empty,
        "empty_cells": empty,
        "domain_size": 9,
        "unrestricted_upper_bound": 9 ** empty,
        "unrestricted_upper_bound_log10": empty * log10(9),
        "initial_candidate_product": product,
        "initial_candidate_product_log10": log10(product) if product > 0 else float("-inf"),
        "cells_without_candidates": counts.count(0),
    }


# ---------------------------------------------------------------------------
# Rezultati
# ---------------------------------------------------------------------------

@dataclass
class SudokuResult:
    puzzle: list[list[int]]
    solution: list[list[int]] | None
    solved: bool
    proven_unsolvable: bool
    message: str
    landscape: dict
    stats: SearchStats
    max_queue_size: int
    solution_depth: int | None
    initial_empty: int
    remaining_empty: int              # në gjendjen më të thellë të arritur
    dead_ends: int
    deepest_board: list[list[int]]    # gjendja më e thellë e zgjeruar
    initial_objective: dict           # f(s) dhe F(s) për enigmën
    final_objective: dict             # për zgjidhjen ose gjendjen më të thellë
    solution_path: list[tuple[int, int, int]] = field(default_factory=list)  # (rresht, kolonë, vlerë)
    trace: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------

def solve_sudoku(
    grid,
    max_explored: int | None = DEFAULT_MAX_EXPLORED,
    max_generated: int | None = DEFAULT_MAX_GENERATED,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    use_mrv: bool = True,
    trace_limit: int = 0,
) -> SudokuResult:
    """
    Zgjidh Sudoku me BFS (radhë FIFO).

    max_explored  – kufiri i gjendjeve të zgjeruara (hequra nga radha).
    max_generated – kufiri i gjendjeve të gjeneruara (kontrollon kujtesën).
    timeout_seconds – kufiri i kohës.
    use_mrv       – True: zgjidhet qeliza me më pak vlera (MRV);
                    False: qeliza e parë bosh (renditje rresht pas rreshti).
    trace_limit   – sa gjendje të para të zgjeruara të ruhen për vizualizim.
    """
    errors = validate_grid(grid)
    errors += validate_search_limits(max_explored, timeout_seconds)
    if max_generated is not None and (
        not isinstance(max_generated, int) or isinstance(max_generated, bool) or max_generated < 1
    ):
        errors.append("Kufiri i gjendjeve të gjeneruara duhet të jetë numër i plotë pozitiv.")
    if not isinstance(trace_limit, int) or isinstance(trace_limit, bool) or trace_limit < 0:
        errors.append("Kufiri i gjurmës duhet të jetë numër jo-negativ.")
    if errors:
        raise InvalidSudokuError(errors)

    puzzle = copy_grid(grid)
    landscape = compute_landscape(puzzle)
    initial = grid_to_bytes(puzzle)
    initial_empty = initial.count(0)

    stats = SearchStats(max_nodes=max_explored, timeout_seconds=timeout_seconds)
    queue: deque[tuple[bytes, bytes]] = deque([(initial, b"")])
    stats.generated = 1                 # gjendja fillestare
    max_queue_size = 1
    dead_ends = 0
    trace: list[dict] = []
    deepest = (initial, b"")
    found: tuple[bytes, bytes] | None = None

    stats.start()
    reason = TERMINATION_EXHAUSTED
    try:
        while queue:
            board, path = queue.popleft()            # FIFO: gjendja më e vjetër
            depth = len(path)
            stats.record_explored(depth)
            if depth > len(deepest[1]):
                deepest = (board, path)

            # Goal Test: tabelat në radhë janë gjithmonë të vlefshme
            # (krijohen vetëm me caktime të vlefshme), prandaj mjafton
            # të mos ketë qeliza bosh.
            if 0 not in board:
                found = (board, path)
                stats.valid += 1
                reason = TERMINATION_GOAL_REACHED
                if len(trace) < trace_limit:
                    trace.append(_trace_entry(stats.explored, board, depth, None, [], len(queue), False))
                break

            rows, cols, boxes = _masks(board)
            chosen, chosen_mask, chosen_count = -1, 0, 10
            for i in range(CELLS):
                if board[i] == 0:
                    mask = _candidate_mask(i, rows, cols, boxes)
                    count = mask.bit_count()
                    if count < chosen_count:
                        chosen, chosen_mask, chosen_count = i, mask, count
                        if count == 0 or not use_mrv:
                            break

            candidates = [v for v in range(1, 10) if chosen_mask >> v & 1]
            if len(trace) < trace_limit:
                trace.append(_trace_entry(
                    stats.explored, board, depth, chosen, candidates,
                    len(queue), chosen_count == 0,
                ))

            # Vlerat që shkelin kufizimet nuk krijojnë gjendje fëmijë
            stats.rejected += 9 - chosen_count
            if chosen_count == 0:
                # Dead end: dega braktiset (Backtracking), vazhdohet me radhën
                dead_ends += 1
                stats.backtracks += 1
                continue

            new_path = path + bytes((chosen,))
            for v in candidates:
                child = board[:chosen] + bytes((v,)) + board[chosen + 1:]
                queue.append((child, new_path))      # në fund të radhës
                stats.generated += 1
                if max_generated is not None and stats.generated >= max_generated:
                    raise SearchLimitReached(TERMINATION_GENERATED_LIMIT)
            if len(queue) > max_queue_size:
                max_queue_size = len(queue)
    except SearchLimitReached as limit:
        reason = limit.reason
    stats.stop(reason)

    if found is not None:
        board, path = found
        solution = bytes_to_grid(board)
        verified = is_complete_solution(solution) and preserves_clues(puzzle, solution)
        solution_path = [(i // 9, i % 9, board[i]) for i in path]
        message = (
            f"Zgjidhja u gjet në nivelin {len(path)} të BFS dhe u verifikua në mënyrë të pavarur."
            if verified else
            "Gabim: gjendja e gjetur nuk kaloi verifikimin e pavarur."
        )
        solved = verified
        proven_unsolvable = False
        solution_depth = len(path)
        remaining_empty = 0
    else:
        solution = None
        solved = False
        solution_path = []
        solution_depth = None
        remaining_empty = deepest[0].count(0)
        if reason == TERMINATION_EXHAUSTED:
            proven_unsolvable = True
            message = ("Radha u zbraz pa gjetur zgjidhje: BFS eksploroi të gjithë hapësirën "
                       "e vlefshme, prandaj enigma NUK ka zgjidhje.")
        else:
            proven_unsolvable = False
            message = (f"{stats.termination_message} Zgjidhja nuk u gjet brenda kufijve. "
                       f"Kjo NUK do të thotë që enigma nuk ka zgjidhje.")

    return SudokuResult(
        puzzle=puzzle,
        solution=solution,
        solved=solved,
        proven_unsolvable=proven_unsolvable,
        message=message,
        landscape=landscape,
        stats=stats,
        max_queue_size=max_queue_size,
        solution_depth=solution_depth,
        initial_empty=initial_empty,
        remaining_empty=remaining_empty,
        dead_ends=dead_ends,
        deepest_board=bytes_to_grid(deepest[0]),
        initial_objective=objective_function(puzzle),
        final_objective=objective_function(solution if solution is not None
                                           else bytes_to_grid(deepest[0])),
        solution_path=solution_path,
        trace=trace,
    )


def _trace_entry(order, board, depth, cell, candidates, queue_size, dead_end) -> dict:
    return {
        "order": order,                     # radha e zgjerimit (1, 2, 3, ...)
        "depth": depth,                     # niveli i BFS
        "board": bytes_to_grid(board),
        "cell": (cell // 9, cell % 9) if cell is not None else None,
        "candidates": candidates,
        "queue_size": queue_size,           # madhësia e radhës pas popleft
        "dead_end": dead_end,
    }


def solution_states(puzzle, solution_path) -> list[list[list[int]]]:
    """
    Rindërton gjendjet e ndërmjetme nga enigma te zgjidhja:
    gjendja 0 = enigma, gjendja k = pas k caktimeve të rrugës së BFS.
    """
    states = [copy_grid(puzzle)]
    current = copy_grid(puzzle)
    for r, c, v in solution_path:
        current[r][c] = v
        states.append(copy_grid(current))
    return states


# ---------------------------------------------------------------------------
# Ekzekutim i pavarur:  python -m tasks.sudoku
# ---------------------------------------------------------------------------

def _main() -> None:
    for key, label in list_difficulties():
        result = solve_sudoku(get_puzzle(key))
        s = result.stats
        land = result.landscape
        print(f"\n=== {label} ===")
        print(f"Qeliza bosh k = {land['empty_cells']}, 9^k ≈ 10^{land['unrestricted_upper_bound_log10']:.1f}")
        print(f"Përfundimi: {s.termination_message}")
        io, fo = result.initial_objective, result.final_objective
        print(f"Objective: fillimi f(s)={io['violations']}, F(s)={io['filled']}  →  "
              f"fundi f(s)={fo['violations']}, F(s)={fo['filled']}")
        print(f"Generated={s.generated}, Explored={s.explored}, Rejected={s.rejected}, "
              f"Dead ends={result.dead_ends}, Max queue={result.max_queue_size}, "
              f"Koha={s.elapsed_seconds:.3f}s")
        print(result.message)


if __name__ == "__main__":
    _main()
