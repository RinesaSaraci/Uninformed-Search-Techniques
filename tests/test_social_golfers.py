"""
Testet për Detyrën 1 – Social Golfers (DFS + Backtracking).
"""

from collections import Counter
from itertools import combinations
from math import comb

import pytest

from tasks.social_golfers import (
    InvalidConfigurationError,
    compute_landscape,
    schedule_to_matrix,
    schedule_to_rows,
    solve_social_golfers,
    verify_schedule,
)
from utils.statistics import (
    TERMINATION_EXHAUSTED,
    TERMINATION_GOAL_REACHED,
    TERMINATION_NODE_LIMIT,
    TERMINATION_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Ndihmës: kontroll i pavarur, i shkruar drejtpërdrejt në test
# ---------------------------------------------------------------------------

def assert_schedule_properties(schedule, n, g, p):
    pair_counter = Counter()
    for week in schedule:
        assert len(week) == g
        golfers = [x for group in week for x in group]
        assert sorted(golfers) == list(range(1, n + 1))   # çdo golfer saktësisht 1 herë
        for group in week:
            assert len(group) == p
            pair_counter.update(combinations(sorted(group), 2))
    assert all(count == 1 for count in pair_counter.values())


# ---------------------------------------------------------------------------
# Landscape
# ---------------------------------------------------------------------------

def test_landscape_default_configuration():
    land = compute_landscape(32, 8, 4)
    assert land["total_pairs"] == 496
    assert land["pairs_per_group"] == 6
    assert land["pairs_per_week"] == 48
    assert land["pair_bound"] == 10
    assert land["golfer_bound"] == 10
    assert land["upper_bound"] == 10
    assert land["possible_groups"] == comb(32, 4) == 35960


@pytest.mark.parametrize(
    "n,g,p,expected",
    [(9, 3, 3, 4), (6, 2, 3, 2), (12, 4, 3, 5), (16, 4, 4, 5), (4, 2, 2, 3)],
)
def test_upper_bound_formula(n, g, p, expected):
    land = compute_landscape(n, g, p)
    assert land["pair_bound"] == comb(n, 2) // (g * comb(p, 2))
    assert land["upper_bound"] == expected


def test_weekly_partitions_small_case():
    # 4 golferë në 2 çifte: {12|34}, {13|24}, {14|23}
    assert compute_landscape(4, 2, 2)["weekly_partitions"] == 3


# ---------------------------------------------------------------------------
# Korrektësia e kërkimit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pivot_rule", ["mrv", "lowest"])
@pytest.mark.parametrize("n,g,p", [(4, 2, 2), (6, 3, 2), (9, 3, 3), (8, 4, 2), (16, 4, 4)])
def test_reaches_upper_bound_on_small_instances(n, g, p, pivot_rule):
    result = solve_social_golfers(n, g, p, max_nodes=100_000, timeout_seconds=10,
                                  pivot_rule=pivot_rule)
    assert result.stats.termination_reason == TERMINATION_GOAL_REACHED
    assert result.weeks_constructed == result.upper_bound
    assert result.target_reached
    assert result.optimality_proven
    assert result.is_valid and result.validation_errors == []
    assert_schedule_properties(result.schedule, n, g, p)


def test_exhaustive_search_proves_maximum_below_bound():
    # 6 golferë, 2 grupe me 3: java 2 është e pamundur (çdo treshe ka 2 nga i njëjti grup)
    result = solve_social_golfers(6, 2, 3, max_nodes=10_000, timeout_seconds=5)
    assert result.upper_bound == 2
    assert result.weeks_constructed == 1
    assert result.stats.termination_reason == TERMINATION_EXHAUSTED
    assert result.optimality_proven
    assert not result.target_reached


def test_requested_weeks_below_bound_does_not_claim_optimality():
    result = solve_social_golfers(16, 4, 4, requested_weeks=3, max_nodes=10_000)
    assert result.weeks_constructed == 3
    assert result.stats.termination_reason == TERMINATION_GOAL_REACHED
    assert result.target_reached
    assert not result.optimality_proven


def test_requested_one_week_uses_canonical_first_week():
    result = solve_social_golfers(8, 2, 4, requested_weeks=1)
    assert result.schedule == [[(1, 2, 3, 4), (5, 6, 7, 8)]]
    assert result.stats.explored == 0


@pytest.mark.parametrize("seed", [None, 1, 2])
def test_default_configuration_returns_valid_partial_schedule(seed):
    result = solve_social_golfers(max_nodes=3_000, timeout_seconds=10, seed=seed)
    assert result.num_golfers == 32 and result.num_groups == 8 and result.group_size == 4
    assert result.upper_bound == 10
    assert 1 <= result.weeks_constructed <= 10
    assert result.is_valid
    assert_schedule_properties(result.schedule, 32, 8, 4)
    if result.weeks_constructed < 10:
        assert not result.optimality_proven


def test_seed_makes_search_reproducible():
    a = solve_social_golfers(max_nodes=2_000, seed=5)
    b = solve_social_golfers(max_nodes=2_000, seed=5)
    assert a.schedule == b.schedule
    assert a.stats.explored == b.stats.explored
    assert a.stats.generated == b.stats.generated


# ---------------------------------------------------------------------------
# Kufijtë e kërkimit
# ---------------------------------------------------------------------------

def test_node_limit_interrupts_search():
    result = solve_social_golfers(max_nodes=500, timeout_seconds=30)
    assert result.stats.termination_reason == TERMINATION_NODE_LIMIT
    assert result.stats.explored == 500
    assert result.stats.interrupted
    assert not result.optimality_proven
    assert "NUK" in result.optimality_message
    assert result.is_valid


def test_timeout_interrupts_search():
    result = solve_social_golfers(max_nodes=None, timeout_seconds=0.2)
    assert result.stats.termination_reason == TERMINATION_TIMEOUT
    assert result.stats.elapsed_seconds < 3.0
    assert not result.optimality_proven
    assert result.is_valid


def test_statistics_are_consistent():
    result = solve_social_golfers(max_nodes=2_000, seed=3)
    s = result.stats
    assert s.generated >= s.rejected >= 0
    assert s.explored <= 2_000
    assert s.backtracks <= s.explored
    assert s.valid >= result.weeks_constructed   # java 1 + javët e ndërtuara
    assert result.best_found_at_explored <= s.explored
    assert s.elapsed_seconds > 0


def test_trace_respects_limit_and_records_real_events():
    result = solve_social_golfers(max_nodes=1_000, trace_limit=20)
    assert len(result.trace) == 20
    assert result.trace[0]["event"] == "place"
    assert result.trace[0]["week"] == 2
    assert {e["event"] for e in result.trace} <= {"place", "prune", "backtrack"}


# ---------------------------------------------------------------------------
# Konfigurime të pavlefshme
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_golfers=30, num_groups=8, group_size=4),     # n ≠ g·p
        dict(num_golfers=8, num_groups=8, group_size=1),      # p < 2
        dict(num_golfers=0, num_groups=0, group_size=4),      # g < 1
        dict(num_golfers=68, num_groups=17, group_size=4),    # n > 64
        dict(num_golfers="32", num_groups=8, group_size=4),   # jo numër
        dict(num_golfers=32.0, num_groups=8, group_size=4),
        dict(requested_weeks=11),                             # > kufiri teorik
        dict(requested_weeks=0),
        dict(max_nodes=0),
        dict(timeout_seconds=-1),
        dict(pivot_rule="random"),
        dict(trace_limit=-5),
    ],
)
def test_invalid_configurations_are_rejected_before_search(kwargs):
    with pytest.raises(InvalidConfigurationError) as info:
        solve_social_golfers(**kwargs)
    assert info.value.errors


def test_invalid_configuration_message_is_albanian():
    with pytest.raises(InvalidConfigurationError) as info:
        compute_landscape(30, 8, 4)
    assert "n = g × p" in info.value.errors[0]


# ---------------------------------------------------------------------------
# Validimi i pavarur
# ---------------------------------------------------------------------------

VALID_9_3_3 = [
    [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
    [(1, 4, 7), (2, 5, 8), (3, 6, 9)],
    [(1, 5, 9), (2, 6, 7), (3, 4, 8)],
    [(1, 6, 8), (2, 4, 9), (3, 5, 7)],
]


def test_verifier_accepts_known_valid_schedule():
    assert verify_schedule(VALID_9_3_3, 9, 3, 3) == (True, [])


def test_verifier_detects_repeated_pair():
    bad = VALID_9_3_3[:2] + [[(1, 2, 9), (3, 6, 7), (4, 5, 8)]]
    ok, errors = verify_schedule(bad, 9, 3, 3)
    assert not ok
    assert any("(1, 2)" in e for e in errors)


def test_verifier_detects_missing_and_duplicate_golfers():
    bad = [[(1, 2, 3), (3, 5, 6), (7, 8, 9)]]
    ok, errors = verify_schedule(bad, 9, 3, 3)
    assert not ok
    assert any("mungojnë" in e and "4" in e for e in errors)
    assert any("më shumë se një herë" in e for e in errors)


def test_verifier_detects_wrong_group_size_and_count():
    ok, errors = verify_schedule([[(1, 2, 3, 4), (5, 6, 7, 8, 9)]], 9, 3, 3)
    assert not ok
    assert any("grupe në vend të 3" in e for e in errors)
    assert any("golferë në vend të 3" in e for e in errors)


def test_verifier_detects_invalid_golfer_id():
    ok, errors = verify_schedule([[(1, 2, 3), (4, 5, 6), (7, 8, 10)]], 9, 3, 3)
    assert not ok
    assert any("i pavlefshëm 10" in e for e in errors)


def test_verifier_agrees_with_search_result():
    result = solve_social_golfers(16, 4, 4, max_nodes=10_000)
    assert verify_schedule(result.schedule, 16, 4, 4) == (True, [])


# ---------------------------------------------------------------------------
# Përgatitja për ndërfaqen
# ---------------------------------------------------------------------------

def test_schedule_formatting():
    rows = schedule_to_rows(VALID_9_3_3)
    assert len(rows) == 4 * 3
    assert rows[0] == {"Java": 1, "Grupi": 1, "Golferët": "1, 2, 3"}
    matrix = schedule_to_matrix(VALID_9_3_3)
    assert len(matrix) == 4 and all(len(row) == 3 for row in matrix)
    assert matrix[1][0] == "1-4-7"
