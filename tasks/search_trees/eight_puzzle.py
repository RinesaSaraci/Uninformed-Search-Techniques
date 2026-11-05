"""
Detyra 4a – 8-Puzzle me BFS.

Problemi
--------
Tabelë 3×3 me pllakat 1..8 dhe një pozicion bosh (0). Një veprim e lëviz
pozicionin bosh lart, poshtë, majtas ose djathtas (pllaka fqinje rrëshqet
në vendin bosh).

Reprezentimi
------------
Gjendja: tuple i pandryshueshëm me 9 numra (rreshtat njëri pas tjetrit).
Initial State (parazgjedhje): (1,2,3, 4,0,6, 7,5,8)
Goal State:                   (1,2,3, 4,5,6, 7,8,0)
Kostoja e çdo veprimi = 1, prandaj BFS gjen rrugën më të shkurtër
(numrin minimal të lëvizjeve).

Zgjidhshmëria
-------------
Për tabelë me gjerësi tek (3), një lëvizje nuk e ndryshon paritetin e numrit
të inversioneve të pllakave (pa 0). Goal State ka 0 inversione, prandaj
gjendja është e zgjidhshme vetëm kur numri i inversioneve është çift.
Kjo kontrollohet para kërkimit.

BFS (Graph Search)
------------------
Radhë FIFO, bashkësi e gjendjeve të vizituara dhe fjalor prindërish
(gjendje → (prindi, veprimi)). Goal Test bëhet kur gjendja gjenerohet
(early goal test); meqë të gjitha veprimet kanë kosto 1, rruga e parë e
gjetur është më e shkurtra.

Numëruesit:
  generated – gjendje të reja të shtuara në radhë (përfshirë fillestaren);
  explored  – gjendje të nxjerra nga radha dhe të zgjeruara;
  rejected  – pasardhës të refuzuar sepse ishin vizituar më parë;
  max_queue – madhësia maksimale e radhës.
Hapësira e plotë e gjendjeve të arritshme ka 9!/2 = 181 440 gjendje.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import factorial

from utils.search_tree import SearchTreeRecorder
from utils.statistics import (
    TERMINATION_EXHAUSTED,
    TERMINATION_GOAL_REACHED,
    SearchLimitReached,
    SearchStats,
)
from utils.validation import is_int, validate_non_negative_int, validate_search_limits

State = tuple[int, ...]

GOAL_STATE: State = (1, 2, 3, 4, 5, 6, 7, 8, 0)
DEFAULT_INITIAL_STATE: State = (1, 2, 3, 4, 0, 6, 7, 5, 8)
DEFAULT_MAX_EXPLORED = 200_000
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_TREE_NODES = 60

# Veprimet: lëvizja e pozicionit bosh → ndryshimi i indeksit
ACTIONS = ("up", "down", "left", "right")
ACTION_LABELS = {"up": "Lart", "down": "Poshtë", "left": "Majtas", "right": "Djathtas"}
_DELTAS = {"up": -3, "down": 3, "left": -1, "right": 1}

PREDEFINED_STATES = {
    "default": {"label": "Parazgjedhja (2 lëvizje)", "state": DEFAULT_INITIAL_STATE},
    "medium": {"label": "Mesatare (8 lëvizje)", "state": (0, 1, 2, 4, 8, 3, 7, 6, 5)},
    "harder": {"label": "Më e vështirë (20 lëvizje)", "state": (0, 1, 2, 3, 4, 7, 8, 5, 6)},
    "hardest": {"label": "Më e vështira (31 lëvizje)", "state": (8, 6, 7, 2, 5, 4, 3, 0, 1)},
}


class InvalidPuzzleError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__(" ".join(errors))
        self.errors = errors


# ---------------------------------------------------------------------------
# Gjendjet, validimi, kalimet
# ---------------------------------------------------------------------------

def initial_state() -> State:
    return DEFAULT_INITIAL_STATE


def goal_state() -> State:
    return GOAL_STATE


def validate_state(state) -> list[str]:
    """Gjendja duhet të jetë një permutacion i numrave 0..8."""
    if not isinstance(state, (tuple, list)) or len(state) != 9:
        return ["Gjendja duhet të ketë saktësisht 9 numra (3×3)."]
    if not all(is_int(v) for v in state):
        return ["Të gjitha vlerat duhet të jenë numra të plotë."]
    if sorted(state) != list(range(9)):
        return ["Gjendja duhet të përmbajë secilin numër 0–8 saktësisht një herë (0 = bosh)."]
    return []


def parse_state(text: str) -> State:
    """Lexon një gjendje nga teksti, p.sh. '1 2 3 4 0 6 7 5 8' ose '123406758'."""
    cleaned = text.replace(",", " ").split()
    if len(cleaned) == 1 and len(cleaned[0]) == 9:
        cleaned = list(cleaned[0])
    try:
        state = tuple(int(x) for x in cleaned)
    except ValueError:
        raise InvalidPuzzleError(["Gjendja duhet të përmbajë vetëm numra."]) from None
    errors = validate_state(state)
    if errors:
        raise InvalidPuzzleError(errors)
    return state


def count_inversions(state: State) -> int:
    tiles = [v for v in state if v != 0]
    return sum(1 for i in range(len(tiles)) for j in range(i + 1, len(tiles)) if tiles[i] > tiles[j])


def is_solvable(state: State) -> bool:
    """E zgjidhshme ndaj GOAL_STATE ⇔ numri i inversioneve është çift."""
    return count_inversions(state) % 2 == 0


def is_goal(state: State) -> bool:
    return tuple(state) == GOAL_STATE


def successors(state: State) -> list[tuple[str, State]]:
    """Kthen [(veprimi, gjendja_e_re), ...] vetëm për lëvizjet e ligjshme."""
    blank = state.index(0)
    row, col = divmod(blank, 3)
    result = []
    for action in ACTIONS:
        if action == "up" and row == 0:
            continue
        if action == "down" and row == 2:
            continue
        if action == "left" and col == 0:
            continue
        if action == "right" and col == 2:
            continue
        target = blank + _DELTAS[action]
        board = list(state)
        board[blank], board[target] = board[target], board[blank]
        result.append((action, tuple(board)))
    return result


def is_legal_transition(state_from: State, state_to: State) -> bool:
    return any(child == tuple(state_to) for _, child in successors(tuple(state_from)))


def state_to_grid(state: State) -> list[list[int]]:
    return [list(state[r * 3:(r + 1) * 3]) for r in range(3)]


def state_label(state: State) -> str:
    """Etiketë kompakte për pemën: tre rreshta, 0 shfaqet si '_'."""
    return "\n".join("".join("_" if v == 0 else str(v) for v in state[r * 3:(r + 1) * 3])
                     for r in range(3))


def landscape() -> dict:
    return {
        "board": "3×3",
        "tiles": 8,
        "all_permutations": factorial(9),          # 362 880
        "reachable_states": factorial(9) // 2,     # 181 440
        "max_branching_factor": 4,
        "max_optimal_depth": 31,
    }


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------

@dataclass
class EightPuzzleResult:
    initial: State
    goal: State
    solvable: bool
    solved: bool
    message: str
    stats: SearchStats
    max_queue_size: int
    solution_depth: int | None
    path: list[State] = field(default_factory=list)      # nga fillestarja te qëllimi
    actions: list[str] = field(default_factory=list)
    tree: SearchTreeRecorder | None = None


def reconstruct_path(parents: dict, goal: State) -> tuple[list[State], list[str]]:
    """Ndjek prindërit nga qëllimi te rrënja dhe e kthen rrugën në rend."""
    path, actions = [goal], []
    current = goal
    while parents[current] is not None:
        parent, action = parents[current]
        path.append(parent)
        actions.append(action)
        current = parent
    path.reverse()
    actions.reverse()
    return path, actions


def solve_eight_puzzle(
    start=DEFAULT_INITIAL_STATE,
    max_explored: int | None = DEFAULT_MAX_EXPLORED,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    tree_nodes: int = DEFAULT_TREE_NODES,
) -> EightPuzzleResult:
    errors = validate_state(start)
    errors += validate_search_limits(max_explored, timeout_seconds)
    errors += validate_non_negative_int(tree_nodes, "Numri i nyjeve të pemës")
    if errors:
        raise InvalidPuzzleError(errors)

    start = tuple(start)
    stats = SearchStats(max_nodes=max_explored, timeout_seconds=timeout_seconds)
    tree = SearchTreeRecorder(tree_nodes)

    if not is_solvable(start):
        stats.start()
        stats.stop(TERMINATION_EXHAUSTED)
        return EightPuzzleResult(
            initial=start, goal=GOAL_STATE, solvable=False, solved=False,
            message=(f"Gjendja nuk është e zgjidhshme: ka {count_inversions(start)} inversione "
                     f"(numër tek). Kërkimi nuk u nis."),
            stats=stats, max_queue_size=0, solution_depth=None, tree=tree,
        )

    stats.start()
    parents: dict[State, tuple[State, str] | None] = {start: None}   # edhe visited
    tree_id: dict[State, int | None] = {start: tree.add(None, state_label(start), "generated", 0)}
    depth_of = {start: 0}
    queue: deque[State] = deque([start])
    stats.generated = 1
    max_queue = 1
    goal_found: State | None = start if is_goal(start) else None
    reason = TERMINATION_GOAL_REACHED if goal_found else TERMINATION_EXHAUSTED

    try:
        while queue and goal_found is None:
            state = queue.popleft()                         # FIFO
            depth = depth_of[state]
            stats.record_explored(depth)
            tree.set_status(tree_id.get(state), "explored")
            for action, child in successors(state):
                if child in parents:                        # vizituar më parë
                    stats.rejected += 1
                    tree.add(tree_id.get(state), state_label(child), "duplicate",
                             depth + 1, ACTION_LABELS[action])
                    continue
                parents[child] = (state, action)
                depth_of[child] = depth + 1
                stats.generated += 1
                tree_id[child] = tree.add(tree_id.get(state), state_label(child), "generated",
                                          depth + 1, ACTION_LABELS[action])
                if is_goal(child):                          # early goal test
                    goal_found = child
                    reason = TERMINATION_GOAL_REACHED
                    break
                queue.append(child)
            max_queue = max(max_queue, len(queue))
    except SearchLimitReached as limit:
        reason = limit.reason
    stats.stop(reason)

    if goal_found is not None:
        path, actions = reconstruct_path(parents, goal_found)
        stats.valid = 1
        stats.max_depth = max(stats.max_depth, len(actions))
        tree.mark_path([tree_id.get(s) for s in path[:-1]], tree_id.get(goal_found))
        return EightPuzzleResult(
            initial=start, goal=GOAL_STATE, solvable=True, solved=True,
            message=f"Rruga më e shkurtër u gjet: {len(actions)} lëvizje.",
            stats=stats, max_queue_size=max_queue, solution_depth=len(actions),
            path=path, actions=[ACTION_LABELS[a] for a in actions], tree=tree,
        )
    return EightPuzzleResult(
        initial=start, goal=GOAL_STATE, solvable=True, solved=False,
        message=(f"{stats.termination_message} Zgjidhja nuk u gjet brenda kufijve; "
                 f"gjendja është e zgjidhshme (paritet çift)."),
        stats=stats, max_queue_size=max_queue, solution_depth=None, tree=tree,
    )
