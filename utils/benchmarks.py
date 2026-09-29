"""Continuous benchmark functions (minimisation), vectorised over a batch.

Every function accepts ``x`` of shape ``(d,)`` or ``(n, d)`` and returns a scalar
or an array of shape ``(n,)``. ``BENCHMARKS`` records the search domain and the
global optimum so notebooks can validate what they find.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _as2d(x):
    x = np.asarray(x, dtype=float)
    return x[None, :] if x.ndim == 1 else x, x.ndim == 1


def _ret(v, single):
    return float(v[0]) if single else v


def sphere(x):
    X, s = _as2d(x)
    return _ret(np.sum(X**2, axis=1), s)


def ellipsoid(x, cond=1e6):
    """Ill-conditioned separable quadratic, condition number ``cond``."""
    X, s = _as2d(x)
    d = X.shape[1]
    w = cond ** (np.arange(d) / max(d - 1, 1))
    return _ret(X**2 @ w, s)


def rosenbrock(x):
    X, s = _as2d(x)
    return _ret(np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1), s)


def rastrigin(x):
    X, s = _as2d(x)
    d = X.shape[1]
    return _ret(10.0 * d + np.sum(X**2 - 10.0 * np.cos(2 * np.pi * X), axis=1), s)


def ackley(x):
    X, s = _as2d(x)
    d = X.shape[1]
    a = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(X**2, axis=1) / d))
    b = -np.exp(np.sum(np.cos(2 * np.pi * X), axis=1) / d)
    return _ret(a + b + 20.0 + np.e, s)


def griewank(x):
    X, s = _as2d(x)
    i = np.arange(1, X.shape[1] + 1)
    return _ret(1.0 + np.sum(X**2, axis=1) / 4000.0 - np.prod(np.cos(X / np.sqrt(i)), axis=1), s)


def schwefel(x):
    """Schwefel 2.26, deceptive: optimum far from the second-best basin."""
    X, s = _as2d(x)
    d = X.shape[1]
    return _ret(418.9828872724338 * d - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1), s)


def levy(x):
    X, s = _as2d(x)
    w = 1 + (X - 1) / 4
    t1 = np.sin(np.pi * w[:, 0]) ** 2
    t2 = np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=1)
    t3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    return _ret(t1 + t2 + t3, s)


@dataclass(frozen=True)
class Benchmark:
    name: str
    f: Callable
    lower: float
    upper: float
    x_opt: Callable  # d -> optimum location
    f_opt: float = 0.0
    modality: str = "unimodal"


BENCHMARKS = {
    "sphere": Benchmark("sphere", sphere, -5.12, 5.12, lambda d: np.zeros(d)),
    "ellipsoid": Benchmark("ellipsoid", ellipsoid, -5.0, 5.0, lambda d: np.zeros(d)),
    "rosenbrock": Benchmark("rosenbrock", rosenbrock, -5.0, 10.0, lambda d: np.ones(d)),
    "rastrigin": Benchmark("rastrigin", rastrigin, -5.12, 5.12, lambda d: np.zeros(d), modality="multimodal"),
    "ackley": Benchmark("ackley", ackley, -32.768, 32.768, lambda d: np.zeros(d), modality="multimodal"),
    "griewank": Benchmark("griewank", griewank, -600.0, 600.0, lambda d: np.zeros(d), modality="multimodal"),
    "schwefel": Benchmark("schwefel", schwefel, -500.0, 500.0, lambda d: np.full(d, 420.9687462275036), modality="deceptive"),
    "levy": Benchmark("levy", levy, -10.0, 10.0, lambda d: np.ones(d), modality="multimodal"),
}


def shifted_rotated(f, shift, rotation):
    """Return g(x) = f(R (x - shift)); breaks separability and centre bias."""
    shift = np.asarray(shift, dtype=float)
    rotation = np.asarray(rotation, dtype=float)

    def g(x):
        X, s = _as2d(x)
        return _ret(np.atleast_1d(f((X - shift) @ rotation.T)), s)

    return g


def random_rotation(d, rng):
    """Haar-uniform orthogonal matrix via QR with sign correction."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))
