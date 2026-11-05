"""
Testet për Detyrën 3 – Latin Square (IDDFS + Backtracking).
"""

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from tasks.latin_square import (
    KNOWN_LATIN_SQUARE_COUNTS,
    KNOWN_REDUCED_COUNTS,
    InvalidLatinSquareConfigError,
    compute_landscape,
    format_matrix,
    iterations_to_rows,
    objective_function,
    solution_states,
    solve_latin_square,
    verify_latin_square,
)
from utils.statistics import (
    TERMINATION_EXHAUSTED,
    TERMINATION_GOAL_REACHED,
    TERMINATION_NODE_LIMIT,
    TERMINATION_TIMEOUT,
)
from utils.visualization import plot_latin_square


# ---------------------------------------------------------------------------
# Kontrolle të pavarura
# ---------------------------------------------------------------------------

def assert_latin(matrix, n):
    assert len(matrix) == n
    for row in matrix:
        assert sorted(row) == list(range(1, n + 1))
    for j in range(n):
        assert sorted(matrix[i][j] for i in range(n)) == list(range(1, n + 1))


def brute_force_count(n, reduced=False):
    """Numërues i pavarur (rekursiv) i Latin Squares, vetëm për n të vogël."""
    grid = [[0] * n for _ in range(n)]
    count = 0

    def rec(k):
        nonlocal count
        if k == n * n:
            count += 1
            return
        i, j = divmod(k, n)
        values = range(1, n + 1)
        if reduced and i == 0:
            values = [j + 1]
        elif reduced and j == 0:
            values = [i + 1]
        for v in values:
            if v in grid[i] or any(grid[r][j] == v for r in range(i)):
                continue
            grid[i][j] = v
            rec(k + 1)
            grid[i][j] = 0

    rec(0)
    return count


# ---------------------------------------------------------------------------
# Zgjidhja për n = 1, 2, 3, 5 (dhe 4, 6)
# ---------------------------------------------------------------------------

# Pa thyerje simetrie, n=5 tejkalon kufijtë e arsyeshëm (shih testin e kufijve)
@pytest.mark.parametrize(
    "n,symmetry_breaking",
    [(n, True) for n in (1, 2, 3, 4, 5)] + [(n, False) for n in (1, 2, 3, 4)],
)
def test_iddfs_finds_valid_latin_square(n, symmetry_breaking):
    result = solve_latin_square(n, symmetry_breaking=symmetry_breaking)
    assert result.solved
    assert result.stats.termination_reason == TERMINATION_GOAL_REACHED
    assert_latin(result.solution, n)
    assert result.is_valid and result.validation_errors == []
    assert result.final_depth_limit == n * n


def test_n_equals_1():
    result = solve_latin_square(1)
    assert result.solution == [[1]]
    assert [it.depth_limit for it in result.iterations] == [0, 1]


def test_n_equals_2():
    result = solve_latin_square(2)
    assert result.solution in ([[1, 2], [2, 1]], [[2, 1], [1, 2]])


def test_default_n_is_5_and_reduced_form_with_symmetry_breaking():
    result = solve_latin_square()
    assert result.n == 5
    assert result.solution[0] == [1, 2, 3, 4, 5]
    assert [row[0] for row in result.solution] == [1, 2, 3, 4, 5]


def test_n_equals_6_within_default_limits():
    result = solve_latin_square(6)
    assert result.solved
    assert_latin(result.solution, 6)


def test_solution_is_not_the_cyclic_construction():
    # Ndërtimi ciklik do të jepte rreshtin e dytë [2,3,4,5,1];
    # IDDFS me renditje rritëse gjen [2,1,4,5,3].
    result = solve_latin_square(5)
    cyclic = [[(i + j) % 5 + 1 for j in range(5)] for i in range(5)]
    assert result.solution != cyclic
    assert result.solution[1] == [2, 1, 4, 5, 3]


# ---------------------------------------------------------------------------
# Progresioni i limitit të thellësisë (IDDFS)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_depth_limit_progression(n):
    result = solve_latin_square(n)
    limits = [it.depth_limit for it in result.iterations]
    assert limits == list(range(n * n + 1))          # 0, 1, 2, ..., n²
    assert [it.iteration for it in result.iterations] == list(range(1, n * n + 2))
    for it in result.iterations[:-1]:
        assert not it.found
        assert it.cutoffs > 0                        # limiti e ndali kërkimin
        assert it.max_depth <= it.depth_limit        # limiti kontrollon thellësinë
    assert result.iterations[-1].found


def test_depth_limit_actually_restricts_search():
    result = solve_latin_square(4, max_depth_limit=6)
    assert not result.solved
    assert result.stats.termination_reason == TERMINATION_EXHAUSTED
    assert [it.depth_limit for it in result.iterations] == list(range(7))
    assert all(it.max_depth <= it.depth_limit for it in result.iterations)
    assert result.stats.max_depth == 6
    assert "limiti maksimal" in result.message


def test_iterations_revisit_shallow_nodes():
    result = solve_latin_square(4)
    explored = [it.explored for it in result.iterations[:-1]]
    assert explored == sorted(explored)              # çdo iteracion rifillon nga rrënja
    assert sum(it.explored for it in result.iterations) == result.stats.explored


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_cutoffs_at_last_level_equal_number_of_latin_squares(n):
    """
    Në limitin n²−1, DLS pa thyerje simetrie arrin çdo matricë me n²−1
    qeliza të caktuara në mënyrë të vlefshme. Secila plotësohet në mënyrë
    unike, prandaj numri i cutoff-eve = numri i Latin Squares L(n).
    """
    result = solve_latin_square(n, symmetry_breaking=False)
    last_before_goal = result.iterations[n * n - 1]
    assert last_before_goal.depth_limit == n * n - 1
    assert last_before_goal.cutoffs == KNOWN_LATIN_SQUARE_COUNTS[n]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_cutoffs_with_symmetry_breaking_equal_reduced_count(n):
    result = solve_latin_square(n)
    assert result.iterations[n * n - 1].cutoffs == KNOWN_REDUCED_COUNTS[n]


# ---------------------------------------------------------------------------
# Backtracking
# ---------------------------------------------------------------------------

def test_backtracking_undoes_assignment_and_tries_next_value():
    """
    n=3 me formë të reduktuar: (2,2)=1 çon në qorrsokak te (2,3),
    prandaj algoritmi zhbën 1 dhe provon 3 në të njëjtën qelizë.
    """
    result = solve_latin_square(3, trace_limit=10_000)
    final_iteration = result.iterations[-1].iteration
    events = [e for e in result.trace if e["iteration"] == final_iteration]
    sequence = [(e["event"], e["cell"], e["value"]) for e in events]
    backtrack = ("backtrack", (1, 1), 1)
    assert backtrack in sequence
    k = sequence.index(backtrack)
    assert ("assign", (1, 1), 3) in sequence[k + 1:]
    assert any(e[0] == "reject" for e in sequence[:k])


def test_statistics_are_consistent():
    result = solve_latin_square(5)
    s = result.stats
    assert s.generated == s.rejected + s.valid
    # çdo nyje e vizituar (përveç rrënjëve) është një caktim i vlefshëm
    assert s.explored == s.valid + len(result.iterations)
    assert sum(it.generated for it in result.iterations) == s.generated
    assert sum(it.backtracks for it in result.iterations) == s.backtracks
    assert s.max_depth == 25


def test_rejected_values_violate_row_or_column():
    result = solve_latin_square(4, trace_limit=5_000)
    for e in result.trace:
        if e["event"] == "assign":
            matrix = e["matrix"]
            i, j = e["cell"]
            v = e["value"]
            assert matrix[i].count(v) == 1
            assert [matrix[r][j] for r in range(4)].count(v) == 1


# ---------------------------------------------------------------------------
# Përfundimi i kërkimit
# ---------------------------------------------------------------------------

def test_node_limit_interrupts_honestly():
    result = solve_latin_square(5, symmetry_breaking=False, max_nodes=10_000)
    assert result.stats.termination_reason == TERMINATION_NODE_LIMIT
    assert result.stats.explored == 10_000
    assert not result.solved and result.solution is None
    assert "NUK" in result.message


def test_timeout_interrupts_honestly():
    result = solve_latin_square(7, max_nodes=None, timeout_seconds=0.2)
    assert result.stats.termination_reason == TERMINATION_TIMEOUT
    assert result.stats.elapsed_seconds < 3
    assert not result.solved


# ---------------------------------------------------------------------------
# Hyrje të pavlefshme
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, -1, 8, 2.5, "5", True, None])
def test_invalid_n(n):
    with pytest.raises(InvalidLatinSquareConfigError):
        solve_latin_square(n)


@pytest.mark.parametrize(
    "kwargs",
    [dict(max_nodes=0), dict(timeout_seconds=-2), dict(max_depth_limit=-1),
     dict(trace_limit=-1)],
)
def test_invalid_limits(kwargs):
    with pytest.raises(InvalidLatinSquareConfigError):
        solve_latin_square(3, **kwargs)


# ---------------------------------------------------------------------------
# Validatori i pavarur
# ---------------------------------------------------------------------------

def test_validator_accepts_valid_square():
    assert verify_latin_square([[1, 2, 3], [2, 3, 1], [3, 1, 2]], 3) == (True, [])


def test_validator_detects_row_violation():
    ok, errors = verify_latin_square([[1, 1, 3], [2, 3, 1], [3, 2, 2]], 3)
    assert not ok and any("Rreshti 1" in e for e in errors)


def test_validator_detects_column_violation():
    ok, errors = verify_latin_square([[1, 2, 3], [1, 3, 2], [3, 1, 2]], 3)
    assert not ok
    assert any("Kolona 1" in e for e in errors)
    assert not any("Rreshti" in e for e in errors)


def test_validator_detects_out_of_range_and_dimensions():
    assert not verify_latin_square([[1, 2], [2, 4]], 2)[0]
    assert not verify_latin_square([[1, 2], [2, 0]], 2)[0]
    assert not verify_latin_square([[1, 2]], 2)[0]
    assert not verify_latin_square([[1, 2, 3], [2, 1]], 2)[0]


# ---------------------------------------------------------------------------
# Landscape, gjendjet e ndërmjetme dhe vizualizimi
# ---------------------------------------------------------------------------

def test_landscape_default():
    land = compute_landscape(5)
    assert land["cells"] == 25
    assert land["domain_size"] == 5
    assert land["unrestricted_assignments"] == 5 ** 25
    assert land["latin_squares"] == 161_280
    assert land["reduced_latin_squares"] == 56
    assert land["latin_squares"] < land["row_permutation_bound"] < land["unrestricted_assignments"]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_known_counts_match_independent_enumeration(n):
    assert brute_force_count(n) == KNOWN_LATIN_SQUARE_COUNTS[n]
    assert brute_force_count(n, reduced=True) == KNOWN_REDUCED_COUNTS[n]


def test_solution_states_and_iteration_rows():
    result = solve_latin_square(4)
    states = solution_states(4, result.solution_path)
    assert len(states) == 17
    assert all(v == 0 for row in states[0] for v in row)
    assert states[-1] == result.solution
    rows = iterations_to_rows(result.iterations)
    assert len(rows) == len(result.iterations)
    assert rows[-1]["U gjet"] == "Po"


def test_matplotlib_visualization_returns_figure():
    result = solve_latin_square(5)
    fig = plot_latin_square(result.solution, highlight=(0, 0))
    assert isinstance(fig, Figure)
    plt.close(fig)
    partial = solution_states(5, result.solution_path)[7]
    fig = plot_latin_square(partial)
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Objective Function dhe dalja tekstuale
# ---------------------------------------------------------------------------

def test_objective_of_solution_and_empty_matrix():
    result = solve_latin_square(5)
    assert objective_function(result.solution, 5) == {
        "violations": 0, "filled": 25, "empty": 0, "is_goal": True}
    empty = objective_function([[0] * 5 for _ in range(5)], 5)
    assert empty["violations"] == 0 and empty["filled"] == 0 and not empty["is_goal"]


def test_objective_counts_row_and_column_violations():
    assert objective_function([[1, 1], [2, 2]], 2)["violations"] == 2     # dy rreshta
    assert objective_function([[1, 2], [1, 2]], 2)["violations"] == 2     # dy kolona
    assert objective_function([[1, 1, 1], [0, 0, 0], [0, 0, 0]], 3)["violations"] == 3


def test_objective_along_iddfs_path():
    result = solve_latin_square(4)
    values = [objective_function(s, 4) for s in solution_states(4, result.solution_path)]
    assert all(v["violations"] == 0 for v in values)
    assert [v["filled"] for v in values] == list(range(17))


def test_format_matrix():
    assert format_matrix([[1, 2], [2, 1]]) == "1 2\n2 1"
