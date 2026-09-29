"""Unit tests for the shared test bed in utils/ (run with `pytest`)."""

import sys
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import (  # noqa: E402
    BENCHMARKS,
    BudgetExhausted,
    BudgetedObjective,
    best_so_far_on_grid,
    bhh_constant_estimate,
    check_close,
    check_true,
    count_satisfied,
    held_karp,
    knapsack_dp,
    qap_brute_force,
    qap_cost,
    random_euclidean_tsp,
    random_knapsack,
    random_max3sat,
    random_qap,
    random_rotation,
    rastrigin,
    shifted_rotated,
    sphere,
    tour_length,
)


# ------------------------------------------------------------ benchmarks
@pytest.mark.parametrize("name", sorted(BENCHMARKS))
@pytest.mark.parametrize("d", [2, 10])
def test_benchmark_optimum_value(name, d):
    b = BENCHMARKS[name]
    assert b.f(b.x_opt(d)) == pytest.approx(b.f_opt, abs=1e-6)


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_benchmark_optimum_is_lower_bound(name):
    b = BENCHMARKS[name]
    X = np.random.default_rng(0).uniform(b.lower, b.upper, (2000, 5))
    assert np.all(b.f(X) >= b.f_opt - 1e-9)


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_benchmark_batch_matches_single(name):
    f = BENCHMARKS[name].f
    X = np.random.default_rng(1).uniform(-2, 2, (7, 4))
    np.testing.assert_allclose(f(X), [f(x) for x in X])


def test_random_rotation_is_orthogonal():
    Q = random_rotation(6, np.random.default_rng(2))
    np.testing.assert_allclose(Q @ Q.T, np.eye(6), atol=1e-12)


def test_shifted_rotated_moves_optimum():
    rng = np.random.default_rng(3)
    shift = rng.uniform(-2, 2, 5)
    g = shifted_rotated(rastrigin, shift, random_rotation(5, rng))
    assert g(shift) == pytest.approx(0.0, abs=1e-9)
    assert g(np.zeros(5)) > 1.0


# ------------------------------------------------------------------ TSP
@pytest.mark.parametrize("n", [4, 6, 8])
def test_held_karp_matches_brute_force(n):
    _, D = random_euclidean_tsp(n, np.random.default_rng(n))
    best, tour = held_karp(D)
    brute = min(tour_length((0,) + p, D) for p in permutations(range(1, n)))
    assert best == pytest.approx(brute)
    assert sorted(tour) == list(range(n))
    assert tour_length(tour, D) == pytest.approx(best)


def test_tour_length_is_rotation_invariant():
    _, D = random_euclidean_tsp(10, np.random.default_rng(4))
    t = np.random.default_rng(5).permutation(10)
    assert tour_length(t, D) == pytest.approx(tour_length(np.roll(t, 3), D))
    assert tour_length(t, D) == pytest.approx(tour_length(t[::-1], D))


def test_bhh_estimate_scale():
    assert bhh_constant_estimate(100) == pytest.approx(7.124)


# ------------------------------------------------------------- knapsack
@pytest.mark.parametrize("corr", ["uncorrelated", "weak", "strong"])
def test_knapsack_dp_matches_brute_force(corr):
    rng = np.random.default_rng(6)
    v, w, C = random_knapsack(12, rng, corr)
    best, x = knapsack_dp(v, w, C)
    brute = max(v @ np.array(s) for s in product([0, 1], repeat=12) if w @ np.array(s) <= C)
    assert best == pytest.approx(brute)
    assert w @ x <= C and v @ x == pytest.approx(best)


# ------------------------------------------------------------------ QAP
def test_qap_brute_force_optimum():
    F, Dm = random_qap(6, np.random.default_rng(7))
    best, p = qap_brute_force(F, Dm)
    assert qap_cost(p, F, Dm) == pytest.approx(best)
    costs = [qap_cost(q, F, Dm) for q in permutations(range(6))]
    assert best == pytest.approx(min(costs))


# -------------------------------------------------------------- MAX-SAT
def test_count_satisfied_hand_example():
    clauses = np.array([[1, -2, 3], [-1, 2, -3]])
    assert count_satisfied(np.array([True, True, True]), clauses) == 2
    assert count_satisfied(np.array([False, True, False]), clauses) == 1


def test_random_max3sat_shape_and_distinct_vars():
    cl = random_max3sat(10, 40, np.random.default_rng(8))
    assert cl.shape == (40, 3)
    assert all(len(set(np.abs(row))) == 3 for row in cl)


# -------------------------------------------------------------- budgets
def test_budgeted_objective_counts_and_stops():
    obj = BudgetedObjective(sphere, budget=10)
    obj(np.ones((4, 3)))
    obj(np.zeros(3))
    assert obj.evals == 5 and obj.best_f == 0.0
    with pytest.raises(BudgetExhausted):
        obj(np.ones((6, 3)))
    assert obj.evals == 5  # a refused call does not consume budget


def test_best_so_far_grid():
    obj = BudgetedObjective(sphere, budget=10)
    for x in ([2.0], [1.0], [3.0], [0.5]):
        obj(np.array(x))
    e, b = obj.trace()
    np.testing.assert_allclose(best_so_far_on_grid(e, b, np.array([1, 2, 3, 4, 10])), [4, 1, 1, 0.25, 0.25])
    assert np.isnan(best_so_far_on_grid(e, b, np.array([0]))[0])


def test_maximisation_mode():
    obj = BudgetedObjective(lambda x: -sphere(x), budget=5, minimize=False)
    obj(np.array([[1.0], [0.0]]))
    assert obj.best_f == 0.0


def test_check_helpers(capsys):
    check_close(1.0, 1.0 + 1e-9, "tiny difference")
    check_true(True, "true condition")
    assert "[PASS]" in capsys.readouterr().out
    with pytest.raises(AssertionError):
        check_close(1.0, 2.0, "should fail")
