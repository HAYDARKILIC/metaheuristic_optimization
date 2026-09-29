"""Evaluation-budget accounting so that every algorithm is compared fairly.

Metaheuristics are compared by *function evaluations*, not iterations or wall time.
``BudgetedObjective`` wraps an objective, counts evaluations (batched calls count
one per row), records the best-so-far trace and raises ``BudgetExhausted`` when
the budget runs out.
"""

import numpy as np


class BudgetExhausted(Exception):
    """Raised when an algorithm asks for more evaluations than its budget."""


class BudgetedObjective:
    def __init__(self, f, budget, minimize=True):
        self.f = f
        self.budget = int(budget)
        self.minimize = minimize
        self.evals = 0
        self.best_f = np.inf if minimize else -np.inf
        self.best_x = None
        self.trace_evals = []
        self.trace_best = []

    @property
    def remaining(self):
        return self.budget - self.evals

    def _better(self, a, b):
        return a < b if self.minimize else a > b

    def __call__(self, x):
        x = np.asarray(x)
        batch = x.ndim == 2
        n = x.shape[0] if batch else 1
        if self.evals + n > self.budget:
            raise BudgetExhausted
        vals = np.atleast_1d(self.f(x))
        xs = x if batch else x[None]
        for xi, fi in zip(xs, vals):
            self.evals += 1
            if self._better(fi, self.best_f):
                self.best_f, self.best_x = float(fi), np.array(xi, copy=True)
            self.trace_evals.append(self.evals)
            self.trace_best.append(self.best_f)
        return vals if batch else float(vals[0])

    def trace(self):
        """(evaluations, best-so-far) arrays."""
        return np.asarray(self.trace_evals), np.asarray(self.trace_best)


def best_so_far_on_grid(trace_evals, trace_best, grid):
    """Interpolate a best-so-far trace onto a common evaluation grid (step function)."""
    idx = np.searchsorted(trace_evals, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    ok = idx >= 0
    out[ok] = np.asarray(trace_best)[idx[ok]]
    return out
