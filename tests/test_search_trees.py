"""
Testet për Detyrën 4 – Search Trees
(8-Puzzle BFS, Missionaries and Cannibals BFS, Workforce Scheduling DFS).
"""

from collections import deque

import matplotlib.pyplot as plt
import networkx as nx
import pytest
from matplotlib.figure import Figure

from tasks.search_trees import eight_puzzle as ep
from tasks.search_trees import missionaries_cannibals as mc
from tasks.search_trees import workforce_scheduling as wf
from utils.search_tree import SearchTreeRecorder
from utils.statistics import (
    TERMINATION_EXHAUSTED,
    TERMINATION_GOAL_REACHED,
    TERMINATION_NODE_LIMIT,
)
from utils.visualization import plot_river_state, plot_search_tree, plot_workload


# ===========================================================================
# Pema e kërkimit (e përbashkët)
# ===========================================================================

def assert_tree_consistent(recorder: SearchTreeRecorder):
    graph = recorder.to_networkx()
    assert graph.number_of_nodes() == len(recorder.nodes) <= recorder.max_nodes
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]
    assert len(roots) == 1
    assert nx.is_tree(graph.to_undirected()) or graph.number_of_nodes() == 1
    for node, data in recorder.nodes.items():
        if data["parent"] is not None:
            assert data["parent"] in recorder.nodes
            assert recorder.nodes[data["parent"]]["depth"] == data["depth"] - 1


def test_recorder_limits_and_parent_rule():
    rec = SearchTreeRecorder(max_nodes=3)
    root = rec.add(None, "r", "explored", 0)
    a = rec.add(root, "a", "generated", 1)
    rec.add(root, "b", "generated", 1)
    assert rec.add(a, "c", "generated", 2) is None        # kufiri u arrit
    assert rec.truncated
    assert rec.add(99, "x", "generated", 1) is None       # prindi nuk ekziston
    assert rec.to_networkx().number_of_edges() == 2


def test_disabled_recorder_records_nothing():
    rec = SearchTreeRecorder(0)
    assert rec.add(None, "r", "explored", 0) is None
    assert rec.nodes == {}


# ===========================================================================
# 8-Puzzle
# ===========================================================================

@pytest.fixture(scope="module")
def true_distances():
    """BFS i plotë nga qëllimi (graf i padrejtuar) – distancat e sakta për testet."""
    dist = {ep.GOAL_STATE: 0}
    queue = deque([ep.GOAL_STATE])
    while queue:
        state = queue.popleft()
        for _, child in ep.successors(state):
            if child not in dist:
                dist[child] = dist[state] + 1
                queue.append(child)
    return dist


def test_initial_and_goal_states():
    assert ep.initial_state() == (1, 2, 3, 4, 0, 6, 7, 5, 8)
    assert ep.goal_state() == (1, 2, 3, 4, 5, 6, 7, 8, 0)
    assert ep.is_goal(ep.goal_state())
    assert not ep.is_goal(ep.initial_state())


@pytest.mark.parametrize(
    "state,count",
    [((1, 2, 3, 4, 0, 6, 7, 5, 8), 4),       # qendër
     ((0, 1, 2, 3, 4, 5, 6, 7, 8), 2),       # qoshe
     ((1, 0, 2, 3, 4, 5, 6, 7, 8), 3)],      # anë
)
def test_successor_count_depends_on_blank_position(state, count):
    assert len(ep.successors(state)) == count


def test_successors_are_legal_single_swaps():
    state = ep.initial_state()
    for action, child in ep.successors(state):
        diff = [i for i in range(9) if state[i] != child[i]]
        assert len(diff) == 2
        assert 0 in (state[diff[0]], state[diff[1]])
        assert ep.is_legal_transition(state, child)
        assert ep.is_legal_transition(child, state)            # i kthyeshëm
    # dy pllaka që nuk janë fqinje me 0 nuk mund të ndërrohen
    assert not ep.is_legal_transition(state, (2, 1, 3, 4, 0, 6, 7, 5, 8))
    # nuk ka lëvizje "jashtë" tabelës nga qoshja
    corner = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert {a for a, _ in ep.successors(corner)} == {"down", "right"}


def test_state_validation_and_parsing():
    assert ep.validate_state(ep.initial_state()) == []
    assert ep.validate_state((1, 2, 3)) != []
    assert ep.validate_state((1, 1, 3, 4, 0, 6, 7, 5, 8)) != []
    assert ep.validate_state((1, 2, 3, 4, 9, 6, 7, 5, 8)) != []
    assert ep.parse_state("1 2 3 4 0 6 7 5 8") == ep.initial_state()
    assert ep.parse_state("123406758") == ep.initial_state()
    with pytest.raises(ep.InvalidPuzzleError):
        ep.parse_state("1 2 a 4 0 6 7 5 8")
    with pytest.raises(ep.InvalidPuzzleError):
        ep.solve_eight_puzzle((1, 2, 3))


def test_solvability_by_inversion_parity(true_distances):
    unsolvable = (1, 2, 3, 4, 5, 6, 8, 7, 0)
    assert ep.count_inversions(unsolvable) == 1
    assert not ep.is_solvable(unsolvable)
    assert unsolvable not in true_distances                    # BFS i plotë e konfirmon
    assert len(true_distances) == 181_440
    assert all(ep.is_solvable(s) for s in list(true_distances)[:2000])


def test_unsolvable_state_is_detected_before_search():
    result = ep.solve_eight_puzzle((1, 2, 3, 4, 5, 6, 8, 7, 0))
    assert not result.solvable and not result.solved
    assert result.stats.explored == 0
    assert "inversione" in result.message


def test_default_shortest_path():
    result = ep.solve_eight_puzzle()
    assert result.solved
    assert result.solution_depth == 2
    assert result.actions == ["Poshtë", "Djathtas"]
    assert result.path[0] == ep.initial_state()
    assert result.path[-1] == ep.GOAL_STATE


@pytest.mark.parametrize("key", list(ep.PREDEFINED_STATES))
def test_bfs_finds_shortest_path_for_predefined_states(key, true_distances):
    start = ep.PREDEFINED_STATES[key]["state"]
    result = ep.solve_eight_puzzle(start, max_explored=200_000, timeout_seconds=30)
    assert result.solved
    assert result.solution_depth == true_distances[start]      # optimal
    assert str(result.solution_depth) in ep.PREDEFINED_STATES[key]["label"]
    assert len(result.path) == result.solution_depth + 1
    for a, b in zip(result.path, result.path[1:]):
        assert ep.is_legal_transition(a, b)
    assert result.path[-1] == ep.GOAL_STATE


def test_goal_as_initial_state():
    result = ep.solve_eight_puzzle(ep.GOAL_STATE)
    assert result.solved and result.solution_depth == 0 and result.path == [ep.GOAL_STATE]


def test_eight_puzzle_statistics_and_tree():
    result = ep.solve_eight_puzzle(ep.PREDEFINED_STATES["medium"]["state"], tree_nodes=60)
    s = result.stats
    assert s.explored <= s.generated
    assert s.rejected > 0
    assert result.max_queue_size < s.generated
    assert_tree_consistent(result.tree)
    assert len(result.tree.nodes) == 60 and result.tree.truncated
    statuses = {d["status"] for d in result.tree.nodes.values()}
    assert "duplicate" in statuses and "explored" in statuses


def test_eight_puzzle_tree_marks_solution_path():
    result = ep.solve_eight_puzzle(tree_nodes=200)
    marked = [d for d in result.tree.nodes.values() if d["status"] in ("solution", "goal")]
    assert len(marked) == result.solution_depth + 1
    goal_nodes = [d for d in marked if d["status"] == "goal"]
    assert goal_nodes[0]["label"] == ep.state_label(ep.GOAL_STATE)


def test_eight_puzzle_node_limit():
    result = ep.solve_eight_puzzle(ep.PREDEFINED_STATES["hardest"]["state"], max_explored=1000)
    assert result.stats.termination_reason == TERMINATION_NODE_LIMIT
    assert not result.solved and result.path == []
    assert result.solvable


# ===========================================================================
# Missionaries and Cannibals
# ===========================================================================

PROBLEM = mc.RiverProblem()


def test_river_initial_and_goal():
    assert PROBLEM.initial_state() == (3, 3, 1)
    assert PROBLEM.goal_state() == (0, 0, 0)
    assert PROBLEM.is_goal((0, 0, 0)) and not PROBLEM.is_goal((0, 0, 1))


@pytest.mark.parametrize(
    "state,valid",
    [((3, 3, 1), True), ((0, 0, 0), True), ((3, 1, 0), True), ((1, 1, 0), True),
     ((1, 2, 1), False),      # majtas: 1M < 2K
     ((2, 1, 0), False),      # djathtas: 1M < 2K
     ((4, 3, 1), False),      # jashtë kufijve
     ((-1, 3, 1), False),
     ((3, 3, 2), False)],
)
def test_valid_riverbank_states(state, valid):
    assert PROBLEM.is_valid_state(state) is valid


def test_boat_capacity_and_no_empty_crossings():
    crossings = PROBLEM.crossings()
    assert sorted(crossings) == [(0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
    assert all(1 <= m + c <= 2 for m, c in crossings)
    assert (0, 0) not in crossings
    assert all(1 <= m + c <= 3 for m, c in mc.RiverProblem(3, 3, 3).crossings())


def test_legal_crossings_from_start():
    children = {child for _, child in PROBLEM.successors(PROBLEM.initial_state())}
    assert children == {(3, 1, 0), (2, 2, 0), (3, 2, 0)}
    assert PROBLEM.is_legal_transition((3, 3, 1), (3, 1, 0))
    assert not PROBLEM.is_legal_transition((3, 3, 1), (1, 3, 0))     # 2M kalojnë → majtas 1M < 3K
    assert not PROBLEM.is_legal_transition((3, 3, 1), (3, 3, 0))     # kalim me 0 persona
    assert not PROBLEM.is_legal_transition((3, 3, 1), (3, 0, 0))     # 3 persona > kapaciteti


def test_river_bfs_solution():
    result = mc.solve_missionaries_cannibals()
    assert result.solved
    assert result.solution_depth == 11
    assert result.path[0] == (3, 3, 1) and result.path[-1] == (0, 0, 0)
    for k, (a, b) in enumerate(zip(result.path, result.path[1:])):
        assert PROBLEM.is_legal_transition(a, b)
        assert a[2] != b[2]                                         # varka kalon çdo herë
        move = result.moves[k]
        assert 1 <= move["missionaries"] + move["cannibals"] <= 2
    assert all(PROBLEM.is_valid_state(s) for s in result.path)      # asnjë gjendje e pavlefshme
    assert len(mc.path_to_rows(result)) == 12


def test_river_statistics_and_tree():
    result = mc.solve_missionaries_cannibals(tree_nodes=200)
    assert result.rejected_invalid > 0 and result.rejected_duplicate > 0
    assert result.stats.rejected == result.rejected_invalid + result.rejected_duplicate
    assert_tree_consistent(result.tree)
    assert not result.tree.truncated
    invalid_labels = [d["label"] for d in result.tree.nodes.values() if d["status"] == "invalid"]
    assert invalid_labels
    for label in invalid_labels:
        m, c, b = label.strip("()").split(",")
        state = (int(m), int(c), 1 if b == "M" else 0)
        assert not PROBLEM.is_valid_state(state)


def test_river_unsolvable_configuration_is_proven():
    result = mc.solve_missionaries_cannibals(4, 4, 2)
    assert not result.solved
    assert result.stats.termination_reason == TERMINATION_EXHAUSTED
    assert "NUK" in result.message


def test_river_landscape_and_invalid_config():
    assert PROBLEM.theoretical_states() == 32
    assert PROBLEM.count_safe_states() < 32
    with pytest.raises(mc.InvalidRiverConfigError):
        mc.solve_missionaries_cannibals(3, 3, 0)
    with pytest.raises(mc.InvalidRiverConfigError):
        mc.solve_missionaries_cannibals("3", 3, 2)


# ===========================================================================
# Rotating Workforce Scheduling
# ===========================================================================

def independent_check(schedule, cfg):
    """Kontroll i pavarur i shkruar në test (punëtorët 1..W)."""
    W, D, S, R = cfg.num_workers, cfg.num_days, cfg.num_shifts, cfg.workers_per_shift
    assert len(schedule) == D
    for day in schedule:
        assert len(day) == S
        assert all(len(ws) == R for ws in day)                     # staffing
        flat = [w for ws in day for w in ws]
        assert len(flat) == len(set(flat))                         # pa dy turne në ditë
    for w in range(1, W + 1):
        shifts = [next((s for s in range(S) if w in schedule[d][s]), None) for d in range(D)]
        run = 0
        for d in range(D):
            run = run + 1 if shifts[d] is not None else 0
            assert run <= cfg.max_consecutive_days
            if d > 0 and shifts[d - 1] == 2 and shifts[d] == 0:
                pytest.fail("Natë → Mëngjes")


@pytest.mark.parametrize("preset", ["default", "rest_days", "backtracking"])
def test_presets_produce_valid_fair_schedules(preset):
    config = wf.PRESETS[preset]["config"]
    result = wf.solve_workforce_schedule(**config)
    cfg = result.config
    assert result.found
    independent_check(result.schedule, cfg)
    assert result.verification["valid"]
    assert sum(result.loads) == cfg.total_assignments
    assert result.imbalance == cfg.imbalance_lower_bound
    assert result.optimal_proven


def test_default_configuration_values():
    result = wf.solve_workforce_schedule()
    cfg = result.config
    assert (cfg.num_workers, cfg.num_days, cfg.num_shifts, cfg.workers_per_shift) == (6, 7, 3, 2)
    assert cfg.total_assignments == 42
    assert result.loads == [7] * 6
    assert result.landscape["unrestricted_assignments"] == 6 ** 42


def test_rest_period_definition():
    cfg = wf.WorkforceConfig()
    assert cfg.rest_hours(2, 0) == 0          # Natë → Mëngjes
    assert cfg.rest_hours(1, 0) == 8          # Pasdite → Mëngjes
    assert cfg.rest_hours(2, 1) == 8          # Natë → Pasdite
    assert cfg.rest_hours(0, 2) == 32
    assert cfg.incompatible(2, 0)
    assert not cfg.incompatible(1, 0)
    strict = wf.WorkforceConfig(num_workers=8, min_rest_hours=11)
    assert strict.incompatible(1, 0) and strict.incompatible(2, 1)


def test_stricter_rest_rule_is_respected():
    result = wf.solve_workforce_schedule(num_workers=8, max_consecutive_days=6, min_rest_hours=11)
    assert result.found
    sched = result.schedule
    for w in range(1, 9):
        shifts = [next((s for s in range(3) if w in sched[d][s]), None) for d in range(7)]
        for d in range(1, 7):
            if shifts[d - 1] is not None and shifts[d] is not None:
                assert result.config.rest_hours(shifts[d - 1], shifts[d]) >= 11


def test_check_assignment_reasons():
    cfg = wf.WorkforceConfig(num_workers=6, max_consecutive_days=2)
    sched = wf.initial_state(cfg)
    sched[0][2] = [0, 1]                       # P1, P2 natën e ditës 1
    sched[0][0] = [2, 3]
    sched[1][1] = [2]
    assert "kufizimi 4" in wf.check_assignment(cfg, sched, 1, 0, 0)
    assert "kufizimi 2" in wf.check_assignment(cfg, sched, 1, 0, 2)
    assert wf.check_assignment(cfg, sched, 1, 0, 4) is None
    sched[1][0] = [4]
    sched[0][1] = [4]
    assert "kufizimi 5" in wf.check_assignment(cfg, sched, 2, 0, 4)


def test_successors_and_next_slot():
    cfg = wf.WorkforceConfig()
    sched = wf.initial_state(cfg)
    assert wf.next_slot(cfg, sched) == (0, 0)
    assert wf.successors(cfg, sched) == list(range(6))
    sched[0][0] = [0, 1]
    assert wf.next_slot(cfg, sched) == (0, 1)
    assert 0 not in wf.successors(cfg, sched)


# --- validatori i pavarur --------------------------------------------------

SMALL = wf.WorkforceConfig(num_workers=4, num_days=2, num_shifts=3, workers_per_shift=1,
                           max_consecutive_days=2, min_rest_hours=8)


def checks_by_number(report):
    return {c["constraint"][0]: c["ok"] for c in report["checks"]}


def test_verifier_accepts_valid_small_schedule():
    report = wf.verify_schedule([[[1], [2], [3]], [[4], [3], [1]]], SMALL)
    assert report["valid"]
    assert report["loads"] == [2, 1, 2, 1]
    assert report["imbalance"] == 1


def test_verifier_detects_night_to_morning_and_rest():
    report = wf.verify_schedule([[[1], [2], [3]], [[3], [1], [2]]], SMALL)
    ok = checks_by_number(report)
    assert not report["valid"]
    assert ok["4"] is False and ok["3"] is False


def test_verifier_detects_overlap_and_staffing():
    report = wf.verify_schedule([[[1], [1], [3]], [[4], [2, 3], [1]]], SMALL)
    ok = checks_by_number(report)
    assert ok["2"] is False and ok["1"] is False


def test_verifier_detects_consecutive_days():
    cfg = wf.WorkforceConfig(num_workers=4, num_days=3, num_shifts=3, workers_per_shift=1,
                             max_consecutive_days=2, min_rest_hours=8)
    report = wf.verify_schedule([[[1], [2], [3]], [[1], [2], [4]], [[1], [3], [4]]], cfg)
    assert checks_by_number(report)["5"] is False


def test_workload_calculations():
    sched = [[[1, 2], [3, 4], [5, 6]], [[1, 3], [2, 4], [5, 6]]]
    loads = wf.workloads(sched, 6)
    assert loads == [2, 2, 2, 2, 2, 2]
    assert wf.imbalance([3, 1, 2]) == 2
    rows = wf.workload_rows(sched, wf.WorkforceConfig())
    assert rows[0] == {"Punëtori": "P1", "Mëngjes": 2, "Pasdite": 0, "Natë": 0, "Gjithsej": 2}


# --- pamundësia, kufijtë ---------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [dict(max_consecutive_days=5),                  # 6 × 6 = 36 < 42
     dict(num_workers=5),                           # nevojiten 6 në ditë
     dict(num_shifts=4), dict(num_workers=0), dict(min_rest_hours=30),
     dict(workers_per_shift="2"), dict(max_nodes=0)],
)
def test_infeasible_or_invalid_configurations_rejected_before_search(kwargs):
    with pytest.raises(wf.WorkforceConfigError):
        wf.solve_workforce_schedule(**kwargs)


def test_infeasibility_proven_by_exhaustive_search():
    result = wf.solve_workforce_schedule(**wf.PRESETS["infeasible"]["config"])
    assert not result.found and result.schedule is None
    assert result.stats.termination_reason == TERMINATION_EXHAUSTED
    assert result.stats.backtracks > 0
    assert "NUK ka" in result.message


def test_node_limit_reports_interruption():
    result = wf.solve_workforce_schedule(max_nodes=10)
    assert result.stats.termination_reason == TERMINATION_NODE_LIMIT
    assert not result.found
    assert "NUK" in result.message


def test_backtracking_happens_and_is_recorded():
    result = wf.solve_workforce_schedule(**wf.PRESETS["backtracking"]["config"], tree_nodes=200)
    assert result.stats.backtracks > 0
    assert result.stats.explored > result.config.total_assignments
    statuses = [d["status"] for d in result.tree.nodes.values()]
    assert "backtrack" in statuses and "goal" in statuses
    assert_tree_consistent(result.tree)


def test_first_solution_mode_does_not_overclaim():
    result = wf.solve_workforce_schedule(optimize=False)
    assert result.solutions_found == 1
    # optimaliteti pohohet vetëm sepse imbalance = kufiri teorik i poshtëm
    assert result.optimal_proven == (result.imbalance <= result.config.imbalance_lower_bound)


# ===========================================================================
# Vizualizimet
# ===========================================================================

def test_visualizations_return_figures():
    tree = ep.solve_eight_puzzle(tree_nodes=40).tree.to_networkx()
    figures = [
        plot_search_tree(tree, title="8-Puzzle"),
        plot_river_state((3, 1, 0), 3, 3, step=1, crossing=(0, 2), direction="majtas → djathtas"),
        plot_river_state((1, 2, 1), 3, 3, valid=False),
    ]
    result = wf.solve_workforce_schedule()
    rows = wf.workload_rows(result.schedule, result.config)
    figures.append(plot_workload(rows, [s[1] for s in result.config.shifts]))
    for fig in figures:
        assert isinstance(fig, Figure)
        plt.close(fig)
