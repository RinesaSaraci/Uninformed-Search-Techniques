"""
Testet për modulet e përbashkëta: utils/validation.py, utils/statistics.py,
utils/visualization.py.
"""

import time

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from tasks import latin_square, social_golfers, sudoku
from tasks.search_trees import eight_puzzle as ep
from tasks.search_trees import missionaries_cannibals as mc
from tasks.search_trees import workforce_scheduling as wf
from utils import validation as v
from utils import visualization as viz
from utils.statistics import (
    TERMINATION_GOAL_REACHED,
    SearchStats,
    Timer,
    format_big_number,
    format_count,
    format_duration,
    measure_time,
    stats_rows,
    termination_summary,
)


# ===========================================================================
# validation.py
# ===========================================================================

@pytest.mark.parametrize("value,ok", [(3, True), (0, True), (True, False), (2.0, False),
                                      ("3", False), (None, False)])
def test_is_int(value, ok):
    assert v.is_int(value) is ok


def test_validate_int_ranges():
    assert v.validate_int(5, "n", 1, 7) == []
    assert "ndërmjet 1 dhe 7" in v.validate_int(8, "n", 1, 7)[0]
    assert "të paktën 2" in v.validate_int(1, "p", minimum=2)[0]
    assert "më i madh se 64" in v.validate_int(65, "n", maximum=64)[0]
    assert "numër i plotë" in v.validate_int(2.5, "n")[0]


def test_positive_and_non_negative_helpers():
    assert v.validate_positive_int(1, "x") == []
    assert v.validate_positive_int(0, "x")
    assert v.validate_non_negative_int(0, "x") == []
    assert v.validate_non_negative_int(-1, "x")
    assert v.validate_optional_positive_int(None, "x") == []
    assert v.validate_optional_positive_int(0, "x")


@pytest.mark.parametrize(
    "max_nodes,timeout,ok",
    [(None, None, True), (10, 1.5, True), (0, 1, False), (10, 0, False),
     (10, "5", False), (True, 1, False), (1.5, 1, False)],
)
def test_validate_search_limits(max_nodes, timeout, ok):
    assert (v.validate_search_limits(max_nodes, timeout) == []) is ok


def test_matrix_shape_and_values():
    good = [[1, 2], [2, 1]]
    assert v.validate_matrix_shape(good, 2, 2) == []
    assert v.validate_matrix_shape([[1, 2]], 2, 2)
    assert v.validate_matrix_shape([[1, 2], [1]], 2, 2) == ["Rreshti 2 duhet të ketë saktësisht 2 qeliza."]
    assert v.validate_matrix_shape("12", 2, 2)
    assert v.validate_matrix_values(good, 1, 2) == []
    assert "lejohen 0–9 (0 = bosh)" in v.validate_matrix_values([[10]], 0, 9, empty_value=0)[0]
    assert "numër i plotë" in v.validate_matrix_values([["1"]], 0, 9)[0]


def test_duplicates_permutation_and_columns():
    assert v.find_duplicates([1, 0, 2, 0, 1, 3, 3], ignore=0) == [1, 3]
    assert v.find_duplicates([]) == []
    assert v.is_permutation([3, 1, 2], range(1, 4))
    assert not v.is_permutation([1, 1, 2], range(1, 4))
    assert not v.is_permutation([1, 2], range(1, 4))
    assert v.matrix_columns([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]


def test_problem_specific_validation_stays_in_task_files():
    # validimi i Sudoku-së dhe i Latin Square përdor ndihmësit, por kufizimet
    # specifike (rreshta, kolona, nënrrjeta) mbeten në skedarët e detyrave
    assert sudoku.validate_grid([[0] * 9 for _ in range(9)]) == []
    assert latin_square.verify_latin_square([[1, 2], [2, 1]], 2) == (True, [])
    assert not hasattr(v, "validate_grid")


# ===========================================================================
# statistics.py
# ===========================================================================

def test_timer_and_measure_time():
    with Timer() as timer:
        time.sleep(0.01)
    assert timer.elapsed >= 0.009
    result, seconds = measure_time(sum, [1, 2, 3])
    assert result == 6 and seconds >= 0


def test_formatting():
    assert format_count(181440) == "181.440"
    assert format_count(None) == "–"
    assert format_duration(0.0000005).endswith("µs")
    assert format_duration(0.0123) == "12.3 ms"
    assert format_duration(3.2) == "3.20 s"
    assert format_big_number(12345) == "12.345"
    assert format_big_number(9 ** 51) == "≈ 4,64 × 10^48"
    assert format_big_number(9995 * 10**20) == "≈ 1,00 × 10^24"


def test_stats_rows_use_actual_algorithm_statistics():
    result = sudoku.solve_sudoku(sudoku.get_puzzle("hard"))
    rows = stats_rows(result.stats, extra={
        "Madhësia maksimale e radhës": (result.max_queue_size, "Kulmi i radhës FIFO."),
        "Thellësia e zgjidhjes": (result.solution_depth, "Numri i caktimeve."),
    })
    values = {row["Metrika"]: row["Vlera"] for row in rows}
    assert values["Gjendje të eksploruara"] == format_count(result.stats.explored)
    assert values["Gjendje të gjeneruara"] == format_count(result.stats.generated)
    assert values["Gjendje të refuzuara"] == format_count(result.stats.rejected)
    assert values["Madhësia maksimale e radhës"] == format_count(result.max_queue_size)
    assert values["Thellësia e zgjidhjes"] == "60"
    assert values["Kufiri i nyjeve"] == format_count(result.stats.max_nodes)


def test_termination_summary_levels():
    goal = sudoku.solve_sudoku(sudoku.get_puzzle("easy")).stats
    assert termination_summary(goal)["level"] == "success"
    limited = sudoku.solve_sudoku(sudoku.get_puzzle("hard"), max_explored=10).stats
    summary = termination_summary(limited)
    assert summary["level"] == "warning" and summary["interrupted"]
    assert "kufiri i nyjeve" in summary["message"]
    timed = latin_square.solve_latin_square(7, max_nodes=None, timeout_seconds=0.1).stats
    assert termination_summary(timed)["level"] == "warning"
    assert "timeout" in termination_summary(timed)["message"]
    exhausted = mc.solve_missionaries_cannibals(4, 4, 2).stats
    assert termination_summary(exhausted)["level"] == "info"


def test_search_stats_limit_and_dict():
    stats = SearchStats(max_nodes=2)
    stats.start()
    stats.record_explored(1)
    with pytest.raises(Exception):
        stats.record_explored(2)
    stats.stop(TERMINATION_GOAL_REACHED)
    data = stats.as_dict()
    assert data["explored"] == 2 and data["termination_reason"] == TERMINATION_GOAL_REACHED


# ===========================================================================
# visualization.py – me rezultatet reale të algoritmeve
# ===========================================================================

def _check(fig):
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_sudoku_with_bfs_output():
    puzzle = sudoku.get_puzzle("medium")
    result = sudoku.solve_sudoku(puzzle)
    _check(viz.plot_sudoku(puzzle, title="Enigma"))
    _check(viz.plot_sudoku(result.solution, original=puzzle, title="Zgjidhja"))
    states = sudoku.solution_states(puzzle, result.solution_path)
    r, c, _ = result.solution_path[4]
    _check(viz.plot_sudoku(states[5], original=puzzle, highlight=(r, c)))


def test_plot_latin_square_with_iddfs_output():
    result = latin_square.solve_latin_square(5)
    _check(viz.plot_latin_square(result.solution))
    partial = latin_square.solution_states(5, result.solution_path)[12]
    _check(viz.plot_latin_square(partial, highlight=result.solution_path[11][:2]))


def test_plot_golfers_with_dfs_output():
    result = social_golfers.solve_social_golfers(seed=1, max_nodes=2_000)
    _check(viz.plot_golfers_schedule(result.schedule))
    _check(viz.plot_golfers_schedule(result.schedule, max_weeks=2))
    matrix = social_golfers.pair_week_matrix(result.schedule, 32)
    _check(viz.plot_pair_matrix(matrix))


def test_pair_week_matrix_marks_weeks_and_repeats():
    valid = [[(1, 2), (3, 4)], [(1, 3), (2, 4)]]
    m = social_golfers.pair_week_matrix(valid, 4)
    assert m[0][1] == m[1][0] == 1 and m[0][2] == 2 and m[0][3] == 0
    repeated = [[(1, 2), (3, 4)], [(1, 2), (3, 4)]]
    assert social_golfers.pair_week_matrix(repeated, 4)[0][1] == -1


def test_plot_eight_puzzle_with_bfs_output():
    result = ep.solve_eight_puzzle(ep.PREDEFINED_STATES["medium"]["state"])
    _check(viz.plot_eight_puzzle(result.initial, goal=ep.GOAL_STATE, title="Gjendja fillestare"))
    _check(viz.plot_eight_puzzle_path(result.path, result.actions, goal=ep.GOAL_STATE))
    long = ep.solve_eight_puzzle(ep.PREDEFINED_STATES["harder"]["state"])
    _check(viz.plot_eight_puzzle_path(long.path, long.actions, max_boards=8))


def test_plot_river_with_bfs_output():
    result = mc.solve_missionaries_cannibals()
    for move in result.moves[:3]:
        _check(viz.plot_river_state(move["to"], 3, 3, step=move["step"],
                                    crossing=(move["missionaries"], move["cannibals"]),
                                    direction=move["direction"]))
    _check(viz.plot_river_state((1, 2, 1), 3, 3, valid=mc.RiverProblem().is_valid_state((1, 2, 1))))


def test_plot_workforce_with_dfs_output():
    result = wf.solve_workforce_schedule(**wf.PRESETS["rest_days"]["config"])
    names = [s[1] for s in result.config.shifts]
    _check(viz.plot_workforce_schedule(result.schedule, names))
    _check(viz.plot_workload(wf.workload_rows(result.schedule, result.config), names))


@pytest.mark.parametrize(
    "tree_source",
    [
        lambda: ep.solve_eight_puzzle(tree_nodes=60).tree,
        lambda: mc.solve_missionaries_cannibals(tree_nodes=80).tree,
        lambda: wf.solve_workforce_schedule(**wf.PRESETS["backtracking"]["config"],
                                            tree_nodes=80).tree,
    ],
)
def test_plot_search_tree_with_real_trees(tree_source):
    _check(viz.plot_search_tree(tree_source().to_networkx()))


def test_large_tree_is_reduced_but_keeps_real_edges():
    recorder = ep.solve_eight_puzzle(ep.PREDEFINED_STATES["harder"]["state"],
                                     tree_nodes=400).tree
    graph = recorder.to_networkx()
    assert graph.number_of_nodes() == 400
    fig = viz.plot_search_tree(graph, max_display_nodes=50)
    assert "shfaqen 50 nga 400" in fig.axes[0].get_title()
    plt.close(fig)
    # nën-grafi i parë 50 nyjeve është i lidhur (çdo prind ka id më të vogël)
    kept = graph.subgraph(sorted(graph.nodes)[:50])
    assert all(kept.in_degree(n) == 1 for n in kept.nodes if n != 0)
