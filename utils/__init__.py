"""Shared helpers for the metaheuristic_optimization curriculum.

The algorithms themselves are built inside the notebooks. This package holds only
the shared test bed: benchmark functions, combinatorial problems with exact
reference solvers, evaluation-budget accounting, plot styling and validation checks.
"""

from .plotting import set_style, savefig, PALETTE
from .checks import check_close, check_true
from .benchmarks import (
    BENCHMARKS,
    Benchmark,
    sphere,
    ellipsoid,
    rosenbrock,
    rastrigin,
    ackley,
    griewank,
    schwefel,
    levy,
    shifted_rotated,
    random_rotation,
)
from .problems import (
    random_euclidean_tsp,
    tour_length,
    held_karp,
    bhh_constant_estimate,
    random_knapsack,
    knapsack_dp,
    random_qap,
    qap_cost,
    qap_brute_force,
    random_max3sat,
    count_satisfied,
)
from .tracking import BudgetedObjective, BudgetExhausted, best_so_far_on_grid

__all__ = [name for name in dir() if not name.startswith("_")]
