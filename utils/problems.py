"""Combinatorial test problems with exact reference solvers for small sizes."""

from itertools import permutations

import numpy as np


# ----------------------------------------------------------------- TSP
def random_euclidean_tsp(n, rng, clustered=False):
    """Cities in the unit square (optionally Gaussian clusters) and distance matrix."""
    if clustered:
        k = max(2, n // 15)
        centres = rng.random((k, 2))
        pts = centres[rng.integers(0, k, n)] + 0.05 * rng.standard_normal((n, 2))
    else:
        pts = rng.random((n, 2))
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return pts, D


def tour_length(tour, D):
    tour = np.asarray(tour)
    return float(D[tour, np.roll(tour, -1)].sum())


def held_karp(D):
    """Exact TSP by dynamic programming, O(n^2 2^n). Use for n <= 13.

    Returns (optimal length, optimal tour starting at city 0).
    """
    n = len(D)
    full = 1 << (n - 1)
    C = np.full((full, n), np.inf)
    P = np.full((full, n), -1, dtype=int)
    for k in range(1, n):
        C[1 << (k - 1), k] = D[0, k]
    for S in range(1, full):
        for k in range(1, n):
            bit = 1 << (k - 1)
            if not S & bit or C[S, k] == np.inf:
                continue
            for j in range(1, n):
                jb = 1 << (j - 1)
                if S & jb:
                    continue
                v = C[S, k] + D[k, j]
                if v < C[S | jb, j]:
                    C[S | jb, j] = v
                    P[S | jb, j] = k
    S = full - 1
    last = int(np.argmin(C[S, 1:] + D[1:, 0])) + 1
    best = float(C[S, last] + D[last, 0])
    tour = [last]
    while S:
        k = P[S, tour[-1]]
        S ^= 1 << (tour[-1] - 1)
        if k == -1:
            break
        tour.append(int(k))
    return best, [0] + tour[::-1]


def bhh_constant_estimate(n, area=1.0):
    """Beardwood-Halton-Hammersley asymptotic tour length ~ 0.7124 sqrt(n A)."""
    return 0.7124 * np.sqrt(n * area)


# ----------------------------------------------------------- knapsack
def random_knapsack(n, rng, correlation="weak", capacity_ratio=0.5):
    """Pisinger-style instances: 'uncorrelated', 'weak' or 'strong' correlation."""
    w = rng.integers(1, 101, n).astype(float)
    if correlation == "uncorrelated":
        v = rng.integers(1, 101, n).astype(float)
    elif correlation == "weak":
        v = np.maximum(1.0, w + rng.integers(-10, 11, n))
    elif correlation == "strong":
        v = w + 10.0
    else:
        raise ValueError(correlation)
    return v, w, float(np.floor(capacity_ratio * w.sum()))


def knapsack_dp(v, w, C):
    """Exact 0/1 knapsack by DP over integer capacity. Returns (value, selection)."""
    n, C = len(v), int(C)
    w = np.asarray(w, dtype=int)
    T = np.zeros((n + 1, C + 1))
    for i in range(1, n + 1):
        T[i] = T[i - 1]
        wi = w[i - 1]
        if wi <= C:
            T[i, wi:] = np.maximum(T[i - 1, wi:], T[i - 1, : C + 1 - wi] + v[i - 1])
    x, c = np.zeros(n, dtype=int), C
    for i in range(n, 0, -1):
        if T[i, c] != T[i - 1, c]:
            x[i - 1], c = 1, c - w[i - 1]
    return float(T[n, C]), x


# ----------------------------------------------------------------- QAP
def random_qap(n, rng):
    """Random symmetric flow and distance matrices (zero diagonal)."""
    F = rng.integers(0, 10, (n, n)).astype(float)
    F = np.triu(F, 1) + np.triu(F, 1).T
    pts = rng.random((n, 2))
    Dm = np.round(10 * np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)), 2)
    return F, Dm


def qap_cost(perm, F, Dm):
    """sum_ij F[i,j] * D[perm[i], perm[j]] (facility i placed at location perm[i])."""
    p = np.asarray(perm)
    return float((F * Dm[np.ix_(p, p)]).sum())


def qap_brute_force(F, Dm):
    """Exact QAP by enumeration (n <= 9)."""
    best, arg = np.inf, None
    for p in permutations(range(len(F))):
        c = qap_cost(p, F, Dm)
        if c < best:
            best, arg = c, p
    return best, np.array(arg)


# ------------------------------------------------------------- MAX-SAT
def random_max3sat(n_vars, n_clauses, rng):
    """Clauses as (n_clauses, 3) signed literals: +i means x_i, -i means not x_i (1-based)."""
    vars_ = np.array([rng.choice(n_vars, 3, replace=False) for _ in range(n_clauses)]) + 1
    signs = rng.choice([-1, 1], (n_clauses, 3))
    return vars_ * signs


def count_satisfied(assign, clauses):
    """assign: boolean array of length n_vars."""
    lit = np.asarray(assign)[np.abs(clauses) - 1]
    return int(np.any(np.where(clauses > 0, lit, ~lit), axis=1).sum())
