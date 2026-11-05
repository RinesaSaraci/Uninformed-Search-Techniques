"""
Detyra 4b – Missionaries and Cannibals me BFS.

Problemi
--------
3 misionarë dhe 3 kanibalë janë në bregun e majtë bashkë me një varkë
që mban 1 ose 2 persona. Të gjithë duhet të kalojnë në bregun e djathtë.
Në asnjë breg, misionarët (nëse ka) nuk duhet të jenë më pak se kanibalët.

Reprezentimi
------------
Gjendja: (M_left, C_left, Boat) ku
    M_left, C_left = misionarët dhe kanibalët në bregun e majtë,
    Boat = 1 nëse varka është në bregun e majtë, 0 nëse në të djathtin.
Bregu i djathtë llogaritet: (M − M_left, C − C_left).
Initial State: (3, 3, 1)      Goal State: (0, 0, 0)

Veprimet (kalimet)
------------------
(m, c) = sa misionarë dhe kanibalë hipin në varkë, me 1 ≤ m + c ≤ kapaciteti.
Për kapacitet 2: (1,0), (2,0), (0,1), (0,2), (1,1).
Varka lëviz gjithmonë nga bregu ku ndodhet te bregu tjetër.

Vlefshmëria (kontrollohet për TË DY brigjet pas çdo kalimi)
-----------------------------------------------------------
  * 0 ≤ M_left ≤ M dhe 0 ≤ C_left ≤ C (nuk mund të hipin më shumë se ç'ka);
  * në secilin breg: misionarët = 0 ose misionarët ≥ kanibalët.
Kalimet me 0 persona nuk gjenerohen kurrë. (Gjendja brenda varkës gjatë
kalimit nuk kontrollohet – sipas formulimit standard të problemit.)

BFS me radhë FIFO, bashkësi vizitimi dhe prindër; Goal Test kur gjendja
gjenerohet. Kalimet e pavlefshme regjistrohen në pemë si gjendje të
refuzuara, që dallimi valid/invalid të jetë i dukshëm.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from utils.search_tree import SearchTreeRecorder
from utils.statistics import (
    TERMINATION_EXHAUSTED,
    TERMINATION_GOAL_REACHED,
    SearchLimitReached,
    SearchStats,
)
from utils.validation import validate_search_limits

State = tuple[int, int, int]

DEFAULT_MISSIONARIES = 3
DEFAULT_CANNIBALS = 3
DEFAULT_CAPACITY = 2
BOAT_LEFT = 1
BOAT_RIGHT = 0
DEFAULT_TREE_NODES = 80


class InvalidRiverConfigError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__(" ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class RiverProblem:
    missionaries: int = DEFAULT_MISSIONARIES
    cannibals: int = DEFAULT_CANNIBALS
    capacity: int = DEFAULT_CAPACITY

    def validate(self) -> list[str]:
        errors = []
        for value, label in ((self.missionaries, "Numri i misionarëve"),
                             (self.cannibals, "Numri i kanibalëve"),
                             (self.capacity, "Kapaciteti i varkës")):
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{label} duhet të jetë numër i plotë.")
        if errors:
            return errors
        if not 0 <= self.missionaries <= 10 or not 0 <= self.cannibals <= 10:
            errors.append("Numri i misionarëve dhe kanibalëve duhet të jetë 0–10.")
        if self.missionaries + self.cannibals == 0:
            errors.append("Duhet të ketë të paktën një person.")
        if not 1 <= self.capacity <= 6:
            errors.append("Kapaciteti i varkës duhet të jetë 1–6.")
        return errors

    # -- gjendjet -----------------------------------------------------------
    def initial_state(self) -> State:
        return (self.missionaries, self.cannibals, BOAT_LEFT)

    def goal_state(self) -> State:
        return (0, 0, BOAT_RIGHT)

    def is_goal(self, state: State) -> bool:
        return tuple(state) == self.goal_state()

    def right_bank(self, state: State) -> tuple[int, int]:
        return self.missionaries - state[0], self.cannibals - state[1]

    def within_bounds(self, state: State) -> bool:
        m, c, boat = state
        return 0 <= m <= self.missionaries and 0 <= c <= self.cannibals and boat in (0, 1)

    def is_safe(self, state: State) -> bool:
        """Kontrollon të dy brigjet."""
        m_left, c_left, _ = state
        m_right, c_right = self.right_bank(state)
        left_ok = m_left == 0 or m_left >= c_left
        right_ok = m_right == 0 or m_right >= c_right
        return left_ok and right_ok

    def is_valid_state(self, state) -> bool:
        return (
            isinstance(state, (tuple, list)) and len(state) == 3
            and all(isinstance(v, int) and not isinstance(v, bool) for v in state)
            and self.within_bounds(tuple(state)) and self.is_safe(tuple(state))
        )

    def crossings(self) -> list[tuple[int, int]]:
        """Të gjitha ngarkesat e mundshme të varkës (1 deri në kapacitet persona)."""
        return [(m, c) for m in range(self.capacity + 1) for c in range(self.capacity + 1)
                if 1 <= m + c <= self.capacity]

    def apply(self, state: State, crossing: tuple[int, int]) -> State:
        m, c = crossing
        m_left, c_left, boat = state
        if boat == BOAT_LEFT:
            return (m_left - m, c_left - c, BOAT_RIGHT)
        return (m_left + m, c_left + c, BOAT_LEFT)

    def all_transitions(self, state: State) -> list[tuple[tuple[int, int], State, bool]]:
        """[(kalimi, gjendja_e_re, e_vlefshme)] për çdo kalim të mundshëm."""
        result = []
        for crossing in self.crossings():
            child = self.apply(state, crossing)
            valid = self.within_bounds(child) and self.is_safe(child)
            result.append((crossing, child, valid))
        return result

    def successors(self, state: State) -> list[tuple[tuple[int, int], State]]:
        """Vetëm kalimet që çojnë në gjendje të vlefshme."""
        return [(cr, child) for cr, child, valid in self.all_transitions(state) if valid]

    def is_legal_transition(self, state_from: State, state_to: State) -> bool:
        return any(child == tuple(state_to) for _, child in self.successors(tuple(state_from)))

    def theoretical_states(self) -> int:
        """Numri i gjendjeve të mundshme pa kufizimin e sigurisë: (M+1)(C+1)·2."""
        return (self.missionaries + 1) * (self.cannibals + 1) * 2

    def count_safe_states(self) -> int:
        return sum(
            1 for m in range(self.missionaries + 1) for c in range(self.cannibals + 1)
            for b in (0, 1) if self.is_safe((m, c, b))
        )


def crossing_label(crossing: tuple[int, int], boat_before: int) -> str:
    m, c = crossing
    direction = "→" if boat_before == BOAT_LEFT else "←"
    parts = []
    if m:
        parts.append(f"{m}M")
    if c:
        parts.append(f"{c}K")
    return f"{'+'.join(parts)} {direction}"


def state_label(state: State) -> str:
    m, c, boat = state
    return f"({m},{c},{'M' if boat == BOAT_LEFT else 'Dj'})"


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------

@dataclass
class RiverResult:
    problem: RiverProblem
    solved: bool
    message: str
    stats: SearchStats
    rejected_invalid: int
    rejected_duplicate: int
    max_queue_size: int
    solution_depth: int | None
    path: list[State] = field(default_factory=list)
    moves: list[dict] = field(default_factory=list)   # {crossing, direction, from, to}
    tree: SearchTreeRecorder | None = None


def reconstruct_path(parents: dict, goal: State) -> tuple[list[State], list[tuple[int, int]]]:
    path, crossings = [goal], []
    current = goal
    while parents[current] is not None:
        parent, crossing = parents[current]
        path.append(parent)
        crossings.append(crossing)
        current = parent
    path.reverse()
    crossings.reverse()
    return path, crossings


def solve_missionaries_cannibals(
    missionaries: int = DEFAULT_MISSIONARIES,
    cannibals: int = DEFAULT_CANNIBALS,
    capacity: int = DEFAULT_CAPACITY,
    max_explored: int | None = 100_000,
    timeout_seconds: float | None = 10.0,
    tree_nodes: int = DEFAULT_TREE_NODES,
) -> RiverResult:
    problem = RiverProblem(missionaries, cannibals, capacity)
    errors = problem.validate() + validate_search_limits(max_explored, timeout_seconds)
    if not isinstance(tree_nodes, int) or isinstance(tree_nodes, bool) or tree_nodes < 0:
        errors.append("Numri i nyjeve të pemës duhet të jetë numër jo-negativ.")
    if errors:
        raise InvalidRiverConfigError(errors)

    start = problem.initial_state()
    stats = SearchStats(max_nodes=max_explored, timeout_seconds=timeout_seconds)
    tree = SearchTreeRecorder(tree_nodes)
    stats.start()

    parents: dict[State, tuple[State, tuple[int, int]] | None] = {start: None}
    depth_of = {start: 0}
    tree_id = {start: tree.add(None, state_label(start), "generated", 0)}
    queue: deque[State] = deque([start])
    stats.generated = 1
    rejected_invalid = rejected_duplicate = 0
    max_queue = 1
    goal_found = start if problem.is_goal(start) else None
    reason = TERMINATION_GOAL_REACHED if goal_found else TERMINATION_EXHAUSTED

    try:
        while queue and goal_found is None:
            state = queue.popleft()
            depth = depth_of[state]
            stats.record_explored(depth)
            tree.set_status(tree_id.get(state), "explored")
            for crossing, child, valid in problem.all_transitions(state):
                label = crossing_label(crossing, state[2])
                if not valid:
                    rejected_invalid += 1
                    stats.rejected += 1
                    tree.add(tree_id.get(state), state_label(child), "invalid", depth + 1, label)
                    continue
                if child in parents:
                    rejected_duplicate += 1
                    stats.rejected += 1
                    tree.add(tree_id.get(state), state_label(child), "duplicate", depth + 1, label)
                    continue
                parents[child] = (state, crossing)
                depth_of[child] = depth + 1
                stats.generated += 1
                tree_id[child] = tree.add(tree_id.get(state), state_label(child), "generated",
                                          depth + 1, label)
                if problem.is_goal(child):
                    goal_found = child
                    reason = TERMINATION_GOAL_REACHED
                    break
                queue.append(child)
            max_queue = max(max_queue, len(queue))
    except SearchLimitReached as limit:
        reason = limit.reason
    stats.stop(reason)

    if goal_found is None:
        if reason == TERMINATION_EXHAUSTED:
            message = ("BFS eksploroi të gjitha gjendjet e arritshme dhe të vlefshme: "
                       "problemi NUK ka zgjidhje për këtë konfigurim.")
        else:
            message = f"{stats.termination_message} Zgjidhja nuk u gjet brenda kufijve."
        return RiverResult(problem, False, message, stats, rejected_invalid, rejected_duplicate,
                           max_queue, None, tree=tree)

    path, crossings = reconstruct_path(parents, goal_found)
    stats.valid = 1
    stats.max_depth = max(stats.max_depth, len(crossings))
    tree.mark_path([tree_id.get(s) for s in path[:-1]], tree_id.get(goal_found))
    moves = [
        {
            "step": k + 1,
            "missionaries": crossings[k][0],
            "cannibals": crossings[k][1],
            "direction": "majtas → djathtas" if path[k][2] == BOAT_LEFT else "djathtas → majtas",
            "from": path[k],
            "to": path[k + 1],
        }
        for k in range(len(crossings))
    ]
    return RiverResult(
        problem, True, f"Zgjidhja më e shkurtër u gjet: {len(crossings)} kalime.", stats,
        rejected_invalid, rejected_duplicate, max_queue, len(crossings), path, moves, tree,
    )


def path_to_rows(result: RiverResult) -> list[dict]:
    """Tabelë hap pas hapi për ndërfaqen."""
    rows = []
    problem = result.problem
    for k, state in enumerate(result.path):
        m_r, c_r = problem.right_bank(state)
        move = result.moves[k - 1] if k > 0 else None
        rows.append({
            "Hapi": k,
            "Kalimi": (f"{move['missionaries']}M + {move['cannibals']}K, {move['direction']}"
                       if move else "Gjendja fillestare"),
            "Majtas (M, K)": f"{state[0]}, {state[1]}",
            "Djathtas (M, K)": f"{m_r}, {c_r}",
            "Varka": "Majtas" if state[2] == BOAT_LEFT else "Djathtas",
        })
    return rows
