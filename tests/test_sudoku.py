"""
Testet për Detyrën 2 – Sudoku (BFS + Backtracking).
"""

import pytest

from data.sudoku_puzzles import DIFFICULTIES, PUZZLES
from tasks.sudoku import (
    InvalidSudokuError,
    compute_landscape,
    copy_grid,
    count_violations,
    get_puzzle,
    get_reference_solution,
    is_complete_solution,
    list_difficulties,
    objective_function,
    preserves_clues,
    solution_states,
    solve_sudoku,
    string_to_grid,
    validate_grid,
)
from utils.statistics import (
    TERMINATION_EXHAUSTED,
    TERMINATION_GENERATED_LIMIT,
    TERMINATION_GOAL_REACHED,
    TERMINATION_NODE_LIMIT,
    TERMINATION_TIMEOUT,
)

EASY_SOLUTION = string_to_grid(PUZZLES["easy"]["solution"])


# ---------------------------------------------------------------------------
# Kontrolle të pavarura, të shkruara drejtpërdrejt në test
# ---------------------------------------------------------------------------

def rows_ok(grid):
    return all(sorted(row) == list(range(1, 10)) for row in grid)


def cols_ok(grid):
    return all(sorted(grid[r][c] for r in range(9)) == list(range(1, 10)) for c in range(9))


def boxes_ok(grid):
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = [grid[br + i][bc + j] for i in range(3) for j in range(3)]
            if sorted(box) != list(range(1, 10)):
                return False
    return True


def board_with_two_solutions():
    """Zgjidhja e lehtë me 4 qeliza të fshira në formë drejtkëndëshi (vlerat 1 dhe 3)."""
    grid = copy_grid(EASY_SOLUTION)
    for r, c in ((3, 5), (3, 8), (4, 5), (4, 8)):
        grid[r][c] = 0
    return grid


# Tabela të vlefshme (pa përsëritje) por pa zgjidhje
UNSOLVABLE_IMMEDIATE = string_to_grid(
    "123456780" "000000009" + "0" * 63
)  # qeliza (1,9): rreshti ka 1–8, kolona ka 9 → asnjë vlerë

UNSOLVABLE_DEEPER = string_to_grid(
    "123456700"
    "000000000"
    "000000000"
    "000000090"
    "000000000"
    "000000000"
    "000000009"
    "000000000"
    "000000000"
)  # qelizat (1,8) dhe (1,9) mund të marrin vetëm 8 → konflikt pas një caktimi


# ---------------------------------------------------------------------------
# Enigmat e paracaktuara
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", DIFFICULTIES)
def test_predefined_puzzle_and_solution_are_consistent(key):
    puzzle = get_puzzle(key)
    solution = get_reference_solution(key)
    assert validate_grid(puzzle) == []
    assert rows_ok(solution) and cols_ok(solution) and boxes_ok(solution)
    assert preserves_clues(puzzle, solution)


@pytest.mark.parametrize("key", DIFFICULTIES)
def test_bfs_solves_every_predefined_puzzle(key):
    puzzle = get_puzzle(key)
    result = solve_sudoku(puzzle, max_explored=200_000, max_generated=500_000,
                          timeout_seconds=30)
    assert result.solved
    assert result.stats.termination_reason == TERMINATION_GOAL_REACHED
    solution = result.solution
    assert rows_ok(solution) and cols_ok(solution) and boxes_ok(solution)
    assert preserves_clues(puzzle, solution)
    # Enigmat kanë zgjidhje unike, pra BFS duhet të gjejë pikërisht atë
    assert solution == get_reference_solution(key)
    assert result.solution_depth == result.initial_empty
    assert result.remaining_empty == 0


def test_list_difficulties_and_reset_returns_fresh_copy():
    assert [k for k, _ in list_difficulties()] == ["easy", "medium", "hard"]
    first = get_puzzle("easy")
    first[0][2] = 9
    assert get_puzzle("easy")[0][2] == 0       # "Rivendos" kthen origjinalin


def test_unknown_difficulty_is_rejected():
    with pytest.raises(InvalidSudokuError):
        get_puzzle("expert")


def test_solver_does_not_modify_input():
    puzzle = get_puzzle("medium")
    before = copy_grid(puzzle)
    solve_sudoku(puzzle)
    assert puzzle == before


# ---------------------------------------------------------------------------
# Validimi i kufizimeve
# ---------------------------------------------------------------------------

def test_validator_accepts_complete_solution():
    assert validate_grid(EASY_SOLUTION) == []
    assert is_complete_solution(EASY_SOLUTION)


def test_validator_detects_row_duplicate():
    grid = copy_grid(get_puzzle("easy"))
    grid[0][2] = 5                                  # 5 ekziston në rreshtin 1
    assert any("Rreshti 1" in e and "5" in e for e in validate_grid(grid))


def test_validator_detects_column_duplicate():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][4] = 7
    grid[8][4] = 7
    errors = validate_grid(grid)
    assert any("Kolona 5" in e for e in errors)
    assert not any("Rreshti" in e for e in errors)


def test_validator_detects_subgrid_duplicate():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 4
    grid[2][2] = 4                                  # rresht e kolonë të ndryshme, e njëjta kuti
    errors = validate_grid(grid)
    assert errors == ["Nënrrjeta 3×3 nr. 1: vlera 4 përsëritet."]


@pytest.mark.parametrize(
    "grid",
    [
        [[0] * 9 for _ in range(8)],                          # 8 rreshta
        [[0] * 8] + [[0] * 9 for _ in range(8)],              # rresht i shkurtër
        [[10] + [0] * 8] + [[0] * 9 for _ in range(8)],       # vlerë jashtë kufirit
        [[-1] + [0] * 8] + [[0] * 9 for _ in range(8)],
        [["5"] + [0] * 8] + [[0] * 9 for _ in range(8)],      # jo numër
        "530070000",
    ],
)
def test_invalid_shapes_and_values(grid):
    assert validate_grid(grid)
    with pytest.raises(InvalidSudokuError):
        solve_sudoku(grid)


def test_contradictory_starting_board_is_rejected_before_search():
    grid = get_puzzle("easy")
    grid[1][1] = 6                                  # 6 ekziston në rreshtin 2
    with pytest.raises(InvalidSudokuError) as info:
        solve_sudoku(grid)
    assert info.value.errors


def test_incomplete_board_is_not_a_solution():
    grid = copy_grid(EASY_SOLUTION)
    grid[8][8] = 0
    assert validate_grid(grid) == []
    assert not is_complete_solution(grid)


def test_preserves_clues_detects_changed_clue():
    puzzle = get_puzzle("easy")
    wrong = copy_grid(EASY_SOLUTION)
    wrong[0][0], wrong[0][1] = wrong[0][1], wrong[0][0]
    assert not preserves_clues(puzzle, wrong)


# ---------------------------------------------------------------------------
# Tabelat pa zgjidhje
# ---------------------------------------------------------------------------

def test_unsolvable_board_immediate_dead_end():
    assert validate_grid(UNSOLVABLE_IMMEDIATE) == []
    result = solve_sudoku(UNSOLVABLE_IMMEDIATE)
    assert not result.solved
    assert result.solution is None
    assert result.stats.termination_reason == TERMINATION_EXHAUSTED
    assert result.proven_unsolvable
    assert result.dead_ends == 1
    assert result.landscape["cells_without_candidates"] == 1


def test_unsolvable_board_found_after_search():
    assert validate_grid(UNSOLVABLE_DEEPER) == []
    result = solve_sudoku(UNSOLVABLE_DEEPER)
    assert result.proven_unsolvable
    assert result.stats.explored == 2              # fillestare + një fëmijë
    assert result.dead_ends == 1
    assert result.stats.backtracks == 1


# ---------------------------------------------------------------------------
# Sjellja FIFO e BFS
# ---------------------------------------------------------------------------

def test_bfs_explores_level_by_level_fifo():
    """
    Tabela me 4 qeliza bosh dhe 2 zgjidhje: në nivelin 1 ka 2 gjendje.
    BFS duhet të zgjerojë të dyja para çdo gjendjeje të nivelit 2:
    renditja e niveleve = [0, 1, 1, 2, 2, 3, 3, 4].
    (DFS do të jepte [0, 1, 2, 3, 4].)
    """
    result = solve_sudoku(board_with_two_solutions(), trace_limit=100)
    assert result.solved
    depths = [entry["depth"] for entry in result.trace]
    assert depths == [0, 1, 1, 2, 2, 3, 3, 4]
    assert result.stats.explored == 8
    assert result.max_queue_size == 2


@pytest.mark.parametrize("use_mrv", [True, False])
def test_explored_levels_never_decrease(use_mrv):
    result = solve_sudoku(get_puzzle("medium"), use_mrv=use_mrv, trace_limit=5_000,
                          max_explored=50_000)
    depths = [entry["depth"] for entry in result.trace]
    assert depths == sorted(depths)
    assert len(depths) > 10


def test_queue_size_is_not_explored_count():
    result = solve_sudoku(get_puzzle("hard"))
    assert result.max_queue_size < result.stats.explored
    assert result.stats.generated >= result.stats.explored


# ---------------------------------------------------------------------------
# Kufijtë e kërkimit
# ---------------------------------------------------------------------------

def test_explored_limit_returns_honest_incomplete_result():
    result = solve_sudoku(get_puzzle("hard"), max_explored=100)
    assert result.stats.termination_reason == TERMINATION_NODE_LIMIT
    assert result.stats.explored == 100
    assert not result.solved
    assert result.solution is None
    assert not result.proven_unsolvable
    assert "NUK" in result.message
    assert 0 < result.remaining_empty < result.initial_empty


def test_generated_limit():
    result = solve_sudoku(get_puzzle("hard"), max_generated=500)
    assert result.stats.termination_reason == TERMINATION_GENERATED_LIMIT
    assert result.stats.generated == 500
    assert not result.solved and not result.proven_unsolvable


def test_timeout():
    # Pa MRV, enigma e vështirë nuk zgjidhet shpejt me BFS
    result = solve_sudoku(get_puzzle("hard"), use_mrv=False, max_explored=None,
                          max_generated=None, timeout_seconds=0.2)
    assert result.stats.termination_reason == TERMINATION_TIMEOUT
    assert result.stats.elapsed_seconds < 3
    assert not result.solved and not result.proven_unsolvable


@pytest.mark.parametrize(
    "kwargs",
    [dict(max_explored=0), dict(max_generated=0), dict(timeout_seconds=0),
     dict(trace_limit=-1), dict(max_generated=1.5)],
)
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(InvalidSudokuError):
        solve_sudoku(get_puzzle("easy"), **kwargs)


# ---------------------------------------------------------------------------
# Landscape, statistikat dhe gjendjet e ndërmjetme
# ---------------------------------------------------------------------------

def test_landscape_for_easy_puzzle():
    land = compute_landscape(get_puzzle("easy"))
    assert land["total_cells"] == 81
    assert land["fixed_cells"] == 30
    assert land["empty_cells"] == 51
    assert land["domain_size"] == 9
    assert land["unrestricted_upper_bound"] == 9 ** 51
    assert land["initial_candidate_product"] < land["unrestricted_upper_bound"]


def test_statistics_are_real_and_consistent():
    result = solve_sudoku(get_puzzle("hard"))
    s = result.stats
    assert s.explored <= s.generated
    assert s.rejected > 0
    assert s.max_depth == result.solution_depth == 60
    assert result.dead_ends == s.backtracks
    assert s.elapsed_seconds > 0


def test_solution_states_reconstruct_path():
    puzzle = get_puzzle("easy")
    result = solve_sudoku(puzzle)
    states = solution_states(puzzle, result.solution_path)
    assert len(states) == result.initial_empty + 1
    assert states[0] == puzzle
    assert states[-1] == result.solution
    for before, after in zip(states, states[1:]):
        changed = [(r, c) for r in range(9) for c in range(9) if before[r][c] != after[r][c]]
        assert len(changed) == 1                      # çdo hap cakton një qelizë
        r, c = changed[0]
        assert before[r][c] == 0
        assert validate_grid(after) == []


# ---------------------------------------------------------------------------
# Objective Function
# ---------------------------------------------------------------------------

def test_objective_of_solution_is_goal():
    obj = objective_function(EASY_SOLUTION)
    assert obj == {"violations": 0, "filled": 81, "empty": 0, "is_goal": True}


def test_objective_of_puzzle():
    obj = objective_function(get_puzzle("easy"))
    assert obj["violations"] == 0
    assert obj["filled"] == 30 and obj["empty"] == 51
    assert not obj["is_goal"]


def test_objective_of_empty_board():
    obj = objective_function([[0] * 9 for _ in range(9)])
    assert obj["violations"] == 0 and obj["filled"] == 0


def test_violation_in_row_only():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 5
    grid[0][8] = 5                               # rresht i njëjtë, kuti të ndryshme
    assert count_violations(grid) == 1


def test_violation_in_row_and_box_counts_twice():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 5
    grid[0][1] = 5                               # rresht dhe kuti e njëjtë
    assert count_violations(grid) == 2


def test_three_equal_values_in_row():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = grid[0][3] = grid[0][6] = 7     # 3 kuti të ndryshme → C(3,2) = 3 çifte
    assert count_violations(grid) == 3


def test_full_board_with_swap_is_not_goal():
    grid = copy_grid(EASY_SOLUTION)
    grid[0][0], grid[0][1] = grid[0][1], grid[0][0]   # rreshti mbetet në rregull
    obj = objective_function(grid)
    assert obj["filled"] == 81
    assert obj["violations"] > 0
    assert not obj["is_goal"]


def test_objective_along_bfs_solution_path():
    puzzle = get_puzzle("medium")
    result = solve_sudoku(puzzle)
    states = solution_states(puzzle, result.solution_path)
    values = [objective_function(s) for s in states]
    assert all(v["violations"] == 0 for v in values)                 # BFS: vetëm gjendje të vlefshme
    assert [v["filled"] for v in values] == list(range(30, 82))     # +1 për nivel
    assert values[-1]["is_goal"]
    assert result.initial_objective == values[0]
    assert result.final_objective == values[-1]


def test_final_objective_when_interrupted():
    result = solve_sudoku(get_puzzle("hard"), max_explored=50)
    fo = result.final_objective
    assert fo["violations"] == 0
    assert result.initial_objective["filled"] < fo["filled"] < 81
    assert not fo["is_goal"]


def test_objective_rejects_wrong_shape():
    with pytest.raises(InvalidSudokuError):
        objective_function([[0] * 9])
