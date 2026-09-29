# Solutions — Week 5: Swarm Intelligence: Particle Swarm and Ant Colony Optimization

Try every exercise yourself before you read these solutions: the struggle is where the learning happens.

Every code block below is self-contained: it runs on its own from inside `week5_swarm_intelligence/` (it imports
`utils` through `sys.path.insert(0, "..")` and defines every helper it needs, copied in compact form from the
notebooks). All randomness is seeded, and every number quoted in the text is the one the block prints. Only NumPy,
Matplotlib and SciPy are used (SciPy for statistics, quasi-Monte Carlo designs and reference optimisers). Where a
block compares with a "best-known" tour length from a notebook, that constant is written into the block and its
source is named.

---

## Notebook 1 — Particle Swarm Optimization

### Exercise 1 (★) — gbest as a limit of lbest; graph diameters

**Limit.** In lbest with radius $r$ the neighbourhood of particle $i$ is $N_r(i)=\lbrace i-r,\dots,i+r\rbrace$ taken
modulo $n$. These are $2r+1$ consecutive residues, so $N_r(i)$ is the whole index set as soon as $2r+1\ge n$, which
holds for every $r\ge n/2$ (in fact for every $r\ge(n-1)/2$). Then $l_i=\arg\min_{j}f(p_j)$ is the same point for every
$i$, which is exactly the gbest rule. So gbest is lbest with radius at least $n/2$.

**Diameters.** The diameter is the largest shortest-path distance in the information graph; it is the number of
iterations that the best memory may need to reach every particle.

* gbest: the complete graph $K_n$, diameter $1$.
* Ring (radius 1): the cycle $C_n$. The distance between $i$ and $j$ is $\min(|i-j|,n-|i-j|)$, which is at most
  $\lfloor n/2\rfloor$ and attains it for $|i-j|=\lfloor n/2\rfloor$. Diameter $\lfloor n/2\rfloor$ (20 for $n=40$).
* Von Neumann on an $r\times c$ torus: the Cartesian product $C_r\square C_c$. A shortest path moves independently
  in the row and column directions, so the distance is the sum of the two cyclic distances and the diameter is
  $\lfloor r/2\rfloor+\lfloor c/2\rfloor$. For a square torus $r=c=\sqrt n$ this is about $\sqrt n$. For the
  $5\times8$ torus that `make_neighbourhood` builds for $n=40$ it is $2+4=6$.

### Exercise 2 (★) — $0\lt\chi\lt1$ and $\chi$ decreasing for $\varphi\gt4$

Put $R=\sqrt{\varphi^2-4\varphi}\gt0$ and $s=\varphi-2+R\gt0$. Then $\chi=2/s$, and rationalising gives

$$
\chi=\frac{2}{\varphi-2+R}=\frac{2(\varphi-2-R)}{(\varphi-2)^2-R^2}=\frac{2(\varphi-2-R)}{4}=\frac{\varphi-2-R}{2}.
$$

*Positivity.* $s\gt0$, so $\chi\gt0$.

*Monotonicity.* $\chi'(\varphi)=\tfrac12\bigl(1-\frac{\varphi-2}{R}\bigr)$. Since $(\varphi-2)^2=R^2+4\gt R^2$, we have
$(\varphi-2)/R\gt1$, so $\chi'\lt0$.

*Bound.* $\chi$ is continuous on $[4,\infty)$ with $\chi(4)=(4-2-0)/2=1$ and strictly decreasing, so $\chi(\varphi)\lt1$ for
$\varphi\gt4$.

The acceleration coefficient $c=\chi\varphi/2$ also decreases. With the formula above,
$\frac{d}{d\varphi}(\varphi\chi)=\frac{(\varphi-1)R-\varphi(\varphi-3)}{R}$. Both terms are positive for $\varphi\gt4$, and
squaring and dividing by $\varphi$ compares $(\varphi-1)^2(\varphi-4)=\varphi^3-6\varphi^2+9\varphi-4$ with
$\varphi(\varphi-3)^2=\varphi^3-6\varphi^2+9\varphi$. The first is smaller by $4$, so the derivative is negative.

Numerical check on $\varphi\in(4,8]$ (4001 points): $\chi$ decreases from $0.99997$ to $0.17157$ and $c$ from $1.99994$
to $0.68629$, both strictly monotone. $\chi(4.1)=0.729844$ and $c(4.1)=1.496180$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt

phi = np.linspace(4 + 1e-9, 8, 4001)
chi = 2 / np.abs(2 - phi - np.sqrt(phi**2 - 4 * phi)); c = chi * phi / 2
plt.plot(phi, chi, label=r"$\chi$"); plt.plot(phi, c, label=r"$c=\chi\varphi/2$")
plt.xlabel(r"$\varphi$"); plt.legend(); plt.title("Constriction coefficient")
print("strictly decreasing:", np.all(np.diff(chi) < 0), np.all(np.diff(c) < 0))
print(f"chi: {chi[0]:.5f} -> {chi[-1]:.5f};  c: {c[0]:.5f} -> {c[-1]:.5f}")
chi41 = 2 / abs(2 - 4.1 - np.sqrt(4.1**2 - 16.4))
print(f"chi(4.1) = {chi41:.6f}, c(4.1) = {chi41 * 2.05:.6f}")
```

### Exercise 3 (★★) — Fully informed particle swarm (FIPS)

In FIPS (Mendes, Kennedy & Neves 2004) every neighbour $k\in\mathcal N(i)$ attracts the particle:
$v\leftarrow\chi\bigl[v+\sum_{k\in\mathcal N(i)}U_k\odot(p_k-x)\bigr]$ with $U_k\sim U(0,\varphi/|\mathcal N(i)|)^d$ and
$\varphi=4.1$. Following Mendes et al., the neighbourhoods exclude the particle itself (ring: 2 neighbours;
von Neumann: 4, the "USquare" variant). Same protocol as Section 5 ($d=10$, $n=40$, 20 000 evaluations, 25 runs,
absorbing bounds); the lbest columns use the Section 5 seeds and reproduce its table.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import BENCHMARKS

def make_neighbourhood(n, topology):                 # (n, k) neighbour indices, self included (Notebook 1)
    idx = np.arange(n)
    if topology == "ring":
        return np.stack([(idx - 1) % n, idx, (idx + 1) % n], axis=1)
    rows = int(np.sqrt(n))
    while n % rows: rows -= 1
    cols = n // rows; r, c = idx // cols, idx % cols
    return np.stack([idx, ((r - 1) % rows) * cols + c, ((r + 1) % rows) * cols + c,
                     r * cols + (c - 1) % cols, r * cols + (c + 1) % cols], axis=1)

def handle_bounds(X, Xold, V, lo, hi, mode, rng):   # Notebook 1, Section 3
    above, below = X > hi, X < lo; out = above | below
    if not out.any(): return X, V
    if mode == "absorb":
        return np.clip(X, lo, hi), np.where(out, 0.0, V)
    if mode == "reflect":
        X = np.where(above, 2 * hi - X, np.where(below, 2 * lo - X, X))
        return np.clip(X, lo, hi), np.where(out, -V, V)
    X = np.where(out, rng.uniform(lo, hi, X.shape), X)          # "random"
    return X, np.where(out, X - Xold, V)

def pso(f, d, lo, hi, budget, runs, rng, n=40, w=0.7298, c=1.49618, topology="ring", boundary="absorb"):
    """Standard lbest PSO vectorised over runs (Notebook 1); returns the best value of every run."""
    X = rng.uniform(lo, hi, (runs, n, d)); V = (rng.uniform(lo, hi, (runs, n, d)) - X) / 2
    Pf = f(X.reshape(-1, d)).reshape(runs, n); P = X.copy()
    nbr = make_neighbourhood(n, topology); ar = np.arange(runs)[:, None]
    for t in range(1, budget // n):
        L = P[ar, nbr[np.arange(n), Pf[:, nbr].argmin(2)]]
        r = rng.random((2, runs, n, d)); V = w * V + c * r[0] * (P - X) + c * r[1] * (L - X)
        X, V = handle_bounds(X + V, X, V, lo, hi, boundary, rng)
        F = f(X.reshape(-1, d)).reshape(runs, n); imp = F < Pf; P[imp], Pf[imp] = X[imp], F[imp]
    return Pf.min(1)

def fips(f, d, lo, hi, budget, runs, nbr, rng, phi=4.1, n=40):
    """Fully informed PSO in constriction form; nbr (n, K) lists the informants (self excluded)."""
    chi = 2 / abs(2 - phi - np.sqrt(phi**2 - 4 * phi)); K = nbr.shape[1]
    X = rng.uniform(lo, hi, (runs, n, d)); V = (rng.uniform(lo, hi, (runs, n, d)) - X) / 2
    Pf = f(X.reshape(-1, d)).reshape(runs, n); P = X.copy()
    for t in range(1, budget // n):
        U = rng.uniform(0, phi / K, (runs, n, K, d))
        V = chi * (V + (U * (P[:, nbr] - X[:, :, None])).sum(2))
        X, V = handle_bounds(X + V, X, V, lo, hi, "absorb", rng)
        F = f(X.reshape(-1, d)).reshape(runs, n); imp = F < Pf; P[imp], Pf[imp] = X[imp], F[imp]
    return Pf.min(1)

ring_ns = make_neighbourhood(40, "ring")[:, [0, 2]]; vn_ns = make_neighbourhood(40, "vonneumann")[:, 1:]
print(f"{'function':11s}{'lbest ring':>11s}{'lbest vN':>11s}{'FIPS ring':>11s}{'FIPS vN':>11s}   MWU p (FIPS-vN vs lbest-vN)")
for fname in ["sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "schwefel"]:
    bm = BENCHMARKS[fname]; args = (bm.f, 10, bm.lower, bm.upper, 20000, 25)
    e = [pso(*args, np.random.default_rng(102), topology="ring"),          # Section 5 seeds
         pso(*args, np.random.default_rng(101), topology="vonneumann"),
         fips(*args, ring_ns, np.random.default_rng(500)), fips(*args, vn_ns, np.random.default_rng(501))]
    e = [x - bm.f_opt for x in e]
    print(f"{fname:11s}" + "".join(f"{np.median(x):11.3g}" for x in e) + f"   {mannwhitneyu(e[3], e[1]).pvalue:.1e}")
```

```text
function    lbest ring   lbest vN  FIPS ring    FIPS vN   MWU p (FIPS-vN vs lbest-vN)
sphere        1.05e-13   1.67e-17   3.03e-05   1.63e-14   1.4e-09
rosenbrock         1.9       4.78       6.95       5.76   2.6e-08
rastrigin         6.96       3.98       13.6       1.05   2.1e-04
ackley        3.08e-06   3.16e-08     0.0475   1.13e-06   1.4e-09
griewank        0.0467     0.0591      0.424    0.00886   1.7e-04
schwefel           595        355        608   1.79e+03   2.9e-09
```

FIPS on the ring (only two informants, averaged) is the worst variant nearly everywhere: averaging two attractors
removes most of the pull towards the better one. FIPS on the von Neumann graph is clearly best on Griewank and
Rastrigin. The Rastrigin advantage is smaller than this seed suggests: with seeds 601–603 the FIPS-vN median is
2.9–3.4, still below lbest-vN's 3.98 but not by a factor of 4. FIPS-vN converges more slowly on the unimodal
functions and fails badly on the deceptive Schwefel function, where averaging many attractors pulls the swarm towards
the centre of mass of the memories rather than towards the far-away best basin. As in Section 5, no variant dominates.

### Exercise 4 (★★) — The search distribution of the standard update

Fix $x,v,p,l$ and write $a=p-x$, $b=l-x$. Coordinate $j$ of the new position is

$$
x'_j=x_j+wv_j+c_1r_{1j}a_j+c_2r_{2j}b_j,\qquad r_{1j},r_{2j}\sim U(0,1)\ \text{i.i.d.}
$$

*Support.* $c_1r_{1j}a_j$ ranges over the interval between $0$ and $c_1a_j$, and similarly for $b$. The coordinates
use independent random numbers, so $x'$ ranges over the axis-aligned box
$\prod_j\bigl[x_j+wv_j+\min(0,c_1a_j)+\min(0,c_2b_j),\ x_j+wv_j+\max(0,c_1a_j)+\max(0,c_2b_j)\bigr]$.

*Moments.* $\mathbb E x'=x+wv+\tfrac{c_1}{2}a+\tfrac{c_2}{2}b$. The coordinates are independent and
$\mathrm{Var}(r)=1/12$, so

$$
\mathrm{Cov}(x')=\Sigma(a,b)=\mathrm{diag}\Bigl(\tfrac{c_1^2a_j^2+c_2^2b_j^2}{12}\Bigr)_{j=1}^d .
$$

*Not rotation-equivariant.* Equivariance would mean that running the algorithm on rotated data $\tilde x=Qx$ (and
likewise for $v,p,l$) produces $Qx'$ in distribution, so the covariances would satisfy $\Sigma(Qa,Qb)=Q\Sigma(a,b)Q^\top$.
But the left side is always diagonal, while the right side is diagonal for all $a,b$ only when $Q$ is a signed
permutation. Counterexample in $d=2$: $p=l$, $a=b=(1,0)$ and $Q$ a rotation by $45^\circ$. Then
$\Sigma(a,b)=\mathrm{diag}(c^2/6,0)$ and $Q\Sigma Q^\top=\tfrac{c^2}{12}\begin{pmatrix}1&1\\1&1\end{pmatrix}$, but
$\Sigma(Qa,Qb)=\tfrac{c^2}{12}I$. With $c=1.49618$ a Monte Carlo check ($4\times10^5$ samples) gives
$\mathrm{Cov}=\mathrm{diag}(0.3729,0)$ against the formula $0.3731$, $Q\Sigma Q^\top$ has all entries $0.1864$, and the
rotated-frame covariance is $\approx0.186\,I$ (formula $0.1865$). In words: the original algorithm samples on a
segment, while the rotated algorithm samples on a full square. The search distribution depends on the coordinate
system.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

# d = 2, x = v = 0, p = l = (1, 0): sample x' = c r1 * a + c r2 * b in the original and in the rotated frame
c, N = 1.49618, 400_000; rng = np.random.default_rng(0)
th = np.pi / 4; Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
a = b = np.array([1.0, 0.0])
step = lambda a, b: c * rng.random((N, 2)) * a + c * rng.random((N, 2)) * b
C_orig = np.cov(step(a, b).T); C_rot = np.cov(step(Q @ a, Q @ b).T)
print("Cov original frame:", np.round(C_orig, 4), " formula c^2/6 =", round(c**2 / 6, 4))
print("Q Sigma Q^T:", np.round(Q @ C_orig @ Q.T, 4))
print("Cov of the algorithm run in the rotated frame:", np.round(C_rot, 4), " formula c^2/12 =", round(c**2 / 12, 4))
```

### Exercise 5 (★★) — "Let them fly" (infinity) boundary handling

Infeasible particles are not evaluated and cannot become personal bests; they keep their velocity and are pulled back
by the attractors, which are always feasible.

*Budget accounting.* The honest convention is to count only the evaluations actually performed ("free"), because an
unevaluated point costs nothing in a black-box setting. Then runs spend their budget at different iteration counts, so
each run must be stopped individually when its own counter reaches the budget (and the last iteration may evaluate
only part of the swarm). If a feasibility check is itself expensive, or if the comparison is per iteration, one
charges every generated point ("charged"). Always report which convention you used.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import BENCHMARKS

def make_neighbourhood(n, topology):                 # (n, k) neighbour indices, self included (Notebook 1)
    idx = np.arange(n)
    if topology == "ring":
        return np.stack([(idx - 1) % n, idx, (idx + 1) % n], axis=1)
    rows = int(np.sqrt(n))
    while n % rows: rows -= 1
    cols = n // rows; r, c = idx // cols, idx % cols
    return np.stack([idx, ((r - 1) % rows) * cols + c, ((r + 1) % rows) * cols + c,
                     r * cols + (c - 1) % cols, r * cols + (c + 1) % cols], axis=1)

def handle_bounds(X, Xold, V, lo, hi, mode, rng):   # Notebook 1, Section 3
    above, below = X > hi, X < lo; out = above | below
    if not out.any(): return X, V
    if mode == "absorb":
        return np.clip(X, lo, hi), np.where(out, 0.0, V)
    if mode == "reflect":
        X = np.where(above, 2 * hi - X, np.where(below, 2 * lo - X, X))
        return np.clip(X, lo, hi), np.where(out, -V, V)
    X = np.where(out, rng.uniform(lo, hi, X.shape), X)          # "random"
    return X, np.where(out, X - Xold, V)

def pso(f, d, lo, hi, budget, runs, rng, n=40, w=0.7298, c=1.49618, topology="ring", boundary="absorb"):
    """Standard lbest PSO vectorised over runs (Notebook 1); returns the best value of every run."""
    X = rng.uniform(lo, hi, (runs, n, d)); V = (rng.uniform(lo, hi, (runs, n, d)) - X) / 2
    Pf = f(X.reshape(-1, d)).reshape(runs, n); P = X.copy()
    nbr = make_neighbourhood(n, topology); ar = np.arange(runs)[:, None]
    for t in range(1, budget // n):
        L = P[ar, nbr[np.arange(n), Pf[:, nbr].argmin(2)]]
        r = rng.random((2, runs, n, d)); V = w * V + c * r[0] * (P - X) + c * r[1] * (L - X)
        X, V = handle_bounds(X + V, X, V, lo, hi, boundary, rng)
        F = f(X.reshape(-1, d)).reshape(runs, n); imp = F < Pf; P[imp], Pf[imp] = X[imp], F[imp]
    return Pf.min(1)

def pso_fly(f, d, lo, hi, budget, runs, rng, n=40, w=0.7298, c=1.49618, charge_infeasible=False):
    """von Neumann PSO with the let-them-fly rule; each run stops when its own counter reaches the budget."""
    nbr = make_neighbourhood(n, "vonneumann")
    X = rng.uniform(lo, hi, (runs, n, d)); V = (rng.uniform(lo, hi, (runs, n, d)) - X) / 2
    Pf = f(X.reshape(-1, d)).reshape(runs, n); P = X.copy()
    spent = np.full(runs, n); ar = np.arange(runs)[:, None]; gen = infeas = 0
    while (spent + (n if charge_infeasible else 0) <= budget).any() and (spent < budget).any():
        active = spent + (n if charge_infeasible else 1) <= budget
        L = P[ar, nbr[np.arange(n), Pf[:, nbr].argmin(2)]]
        r = rng.random((2, runs, n, d))
        V = np.where(active[:, None, None], w * V + c * r[0] * (P - X) + c * r[1] * (L - X), V)
        X = np.where(active[:, None, None], X + V, X)
        feas = np.all((X >= lo) & (X <= hi), axis=2)
        todo = feas & active[:, None]
        if not charge_infeasible:            # never exceed the budget: evaluate only what is left
            room = budget - spent
            todo &= np.cumsum(todo, axis=1) <= room[:, None]
        F = np.full((runs, n), np.inf); F[todo] = f(X[todo])          # infeasible: not evaluated
        spent += n * active if charge_infeasible else todo.sum(1)
        gen += n * active.sum(); infeas += (~feas & active[:, None]).sum()
        imp = F < Pf; P[imp], Pf[imp] = X[imp], F[imp]
    return Pf.min(1), infeas / gen

print(f"{'function':10s}{'absorb':>10s}{'reflect':>10s}{'random':>10s}{'fly free':>10s}{'fly chg':>10s}  infeasible (free, charged)")
for fname in ["schwefel", "rastrigin", "ackley"]:
    bm = BENCHMARKS[fname]; args = (bm.f, 10, bm.lower, bm.upper, 20000, 25)
    e = [pso(*args, np.random.default_rng(300 + k), topology="vonneumann", boundary=m)       # Section 7 seeds
         for k, m in enumerate(["absorb", "reflect", "random"])]
    (ef, qf), (ec, qc) = [pso_fly(*args, np.random.default_rng(310), charge_infeasible=ch) for ch in (False, True)]
    print(f"{fname:10s}" + "".join(f"{np.median(x - bm.f_opt):10.3g}" for x in e + [ef, ec]) + f"  {qf:.1%}, {qc:.1%}")
```

Von Neumann topology, 25 runs, 20 000 evaluations, median final error (the first three columns reproduce Section 7):

| function | absorb | reflect | random | fly (free) | fly (charged) | infeasible fraction (free, charged) |
|---|---|---|---|---|---|---|
| schwefel | 357 | 118 | 474 | 355 | 533 | 35.0%, 38.6% |
| rastrigin | 5.97 | 4.97 | 4.35 | 4.97 | 4.97 | 9.5%, 10.0% |
| ackley | 3.5e-08 | 6.4e-08 | 3.2e-08 | 4.0e-08 | 7.1e-08 | 2.5%, 2.6% |

On Schwefel more than a third of all generated points are infeasible: the particles overshoot the boundary near
which the optimum lies. "Let them fly" is then as good as absorbing when infeasible points are free and worse than
every other rule when they are charged. Reflection remains clearly best. On the centred functions the rule hardly
matters.

### Exercise 6 (★★★) — SPSO-2011 adaptive random topology on rotated Rastrigin

Each particle informs itself and $K=3$ particles drawn uniformly (with repetition). The informant graph of a run is
redrawn after every iteration in which that run's global best did not improve. Shift and rotation are those of
Section 8 (seeds 11 and 12), the box is $[-5.12,5.12]^{10}$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import BENCHMARKS

def make_neighbourhood(n, topology):                 # (n, k) neighbour indices, self included (Notebook 1)
    idx = np.arange(n)
    if topology == "ring":
        return np.stack([(idx - 1) % n, idx, (idx + 1) % n], axis=1)
    rows = int(np.sqrt(n))
    while n % rows: rows -= 1
    cols = n // rows; r, c = idx // cols, idx % cols
    return np.stack([idx, ((r - 1) % rows) * cols + c, ((r + 1) % rows) * cols + c,
                     r * cols + (c - 1) % cols, r * cols + (c + 1) % cols], axis=1)

def handle_bounds(X, Xold, V, lo, hi, mode, rng):   # Notebook 1, Section 3
    above, below = X > hi, X < lo; out = above | below
    if not out.any(): return X, V
    if mode == "absorb":
        return np.clip(X, lo, hi), np.where(out, 0.0, V)
    if mode == "reflect":
        X = np.where(above, 2 * hi - X, np.where(below, 2 * lo - X, X))
        return np.clip(X, lo, hi), np.where(out, -V, V)
    X = np.where(out, rng.uniform(lo, hi, X.shape), X)          # "random"
    return X, np.where(out, X - Xold, V)

def pso(f, d, lo, hi, budget, runs, rng, n=40, w=0.7298, c=1.49618, topology="ring", boundary="absorb"):
    """Standard lbest PSO vectorised over runs (Notebook 1); returns the best value of every run."""
    X = rng.uniform(lo, hi, (runs, n, d)); V = (rng.uniform(lo, hi, (runs, n, d)) - X) / 2
    Pf = f(X.reshape(-1, d)).reshape(runs, n); P = X.copy()
    nbr = make_neighbourhood(n, topology); ar = np.arange(runs)[:, None]
    for t in range(1, budget // n):
        L = P[ar, nbr[np.arange(n), Pf[:, nbr].argmin(2)]]
        r = rng.random((2, runs, n, d)); V = w * V + c * r[0] * (P - X) + c * r[1] * (L - X)
        X, V = handle_bounds(X + V, X, V, lo, hi, boundary, rng)
        F = f(X.reshape(-1, d)).reshape(runs, n); imp = F < Pf; P[imp], Pf[imp] = X[imp], F[imp]
    return Pf.min(1)
from utils import shifted_rotated, random_rotation, rastrigin

def pso_adaptive(f, d, lo, hi, budget, runs, rng, n=40, K=3, update="spso2011"):
    """SPSO-2011 adaptive random topology: each particle informs itself and K random others;
    a run's graph is redrawn after every iteration without improvement of its global best."""
    w, c = (1 / (2 * np.log(2)), 0.5 + np.log(2)) if update == "spso2011" else (0.7298, 1.49618)
    X = rng.uniform(lo, hi, (runs, n, d)); V = (rng.uniform(lo, hi, (runs, n, d)) - X) / 2
    Pf = f(X.reshape(-1, d)).reshape(runs, n); P = X.copy()
    ar, own, eye = np.arange(runs)[:, None], np.arange(n)[None, :], np.eye(n, dtype=bool)
    def new_graph():                                   # A[r, j, i] = True: j informs i
        A = np.zeros((runs, n, n), bool)
        A[np.arange(runs)[:, None, None], np.arange(n)[None, :, None], rng.integers(0, n, (runs, n, K))] = True
        return A | eye
    A, gbest = new_graph(), Pf.min(1)
    for t in range(1, budget // n):
        gi = np.where(A, Pf[:, :, None], np.inf).argmin(1); L = P[ar, gi]
        if update == "spso2011":                       # uniform sample in the ball B(G, |G - x|)
            G = np.where((gi == own)[..., None], X + c * (P - X) / 2, X + c * (P + L - 2 * X) / 3)
            rad = np.linalg.norm(G - X, axis=2, keepdims=True)
            u = rng.standard_normal((runs, n, d)); u /= np.linalg.norm(u, axis=2, keepdims=True)
            V = w * V + G + rad * rng.random((runs, n, 1)) ** (1 / d) * u - X
        else:
            r = rng.random((2, runs, n, d)); V = w * V + c * r[0] * (P - X) + c * r[1] * (L - X)
        X, V = handle_bounds(X + V, X, V, lo, hi, "absorb", rng)
        F = f(X.reshape(-1, d)).reshape(runs, n); imp = F < Pf; P[imp], Pf[imp] = X[imp], F[imp]
        stale = Pf.min(1) >= gbest; gbest = Pf.min(1)
        if stale.any(): A[stale] = new_graph()[stale]
    return Pf.min(1)

d = 10; shift = np.random.default_rng(11).uniform(-2, 2, d); R = random_rotation(d, np.random.default_rng(12))
fS, fSR = shifted_rotated(rastrigin, shift, np.eye(d)), shifted_rotated(rastrigin, shift, R)
def mi(a): q1, m, q3 = np.percentile(a, [25, 50, 75]); return f"{m:6.2f} [{q1:5.1f}, {q3:5.1f}]"
variants = [("standard update, ring", lambda f, s: pso(f, d, -5.12, 5.12, 30000, 25, np.random.default_rng(s))),
            ("standard update, adaptive random", lambda f, s: pso_adaptive(f, d, -5.12, 5.12, 30000, 25,
                                                                           np.random.default_rng(s), update="standard")),
            ("SPSO-2011 update, adaptive random", lambda f, s: pso_adaptive(f, d, -5.12, 5.12, 30000, 25,
                                                                            np.random.default_rng(s)))]
for k, (name, run) in enumerate(variants):
    a, b = run(fS, 600 + k), run(fSR, 600 + k)
    print(f"{name:34s} S {mi(a)}   SR {mi(b)}   MWU p = {mannwhitneyu(a, b).pvalue:.1e}")
```

Shifted (S) and shifted-rotated (SR) Rastrigin, $d=10$, 30 000 evaluations, 25 seeds, median [IQR]:

| variant | S | SR | MWU p |
|---|---|---|---|
| standard update, ring | 7.39 [5.0, 9.0] | 12.94 [10.1, 15.1] | 1.4e-06 |
| standard update, adaptive random | 6.58 [5.1, 10.3] | 17.17 [13.3, 23.4] | 1.1e-05 |
| SPSO-2011 update, adaptive random | 31.64 [27.9, 39.9] | 36.26 [29.4, 41.3] | 0.68 |

The topology does not change the rotation picture. With the coordinate-wise update, rotation multiplies the median
error by 1.8 (ring) and 2.6 (adaptive random topology). With the hypersphere update, S and SR are statistically
indistinguishable (rotation invariance), but the level is much worse. This is the dimension-dependent instability of
uniform hypersphere sampling shown in Section 8: in $d=10$ the swarm does not contract, and the adaptive topology
cannot repair that.

---

## Notebook 2 — PSO Dynamics and Stability

### Exercise 1 (★) — The deterministic region is a triangle; complex vs real roots

The region $\lbrace |w|\lt1,\ 0\lt\varphi\lt2(1+w)\rbrace$ is the intersection of three half-planes: $w\lt1$, $\varphi\gt0$
and $\varphi\lt2+2w$ (the last one already implies $w\gt-1$ when $\varphi\gt0$). Its vertices are the pairwise
intersections of the boundary lines: $(w,\varphi)=(-1,0)$, $(1,0)$ and $(1,4)$. So the region is the open triangle with
these vertices.

The roots of $\lambda^2-b\lambda+w$ with $b=1+w-\varphi$ are complex iff $b^2\lt4w$. This requires $w\gt0$ and
$|1+w-\varphi|\lt2\sqrt w$, i.e.

$$
(1-\sqrt w)^2\lt\varphi\lt(1+\sqrt w)^2 .
$$

Because $(1+\sqrt w)^2=1+2\sqrt w+w\le2(1+w)$ (by $2\sqrt w\le1+w$), this lens lies inside the triangle. There
$|\lambda|=\sqrt w$, and the convergence is an oscillating spiral. Outside the lens the roots are real:

* $w\gt0$ and $\varphi\lt(1-\sqrt w)^2$: both roots are positive (product $w\gt0$, sum $b\gt0$), giving monotone convergence;
* $w\gt0$ and $\varphi\gt(1+\sqrt w)^2$: both roots are negative, giving zig-zag convergence (sign alternation);
* $w\lt0$: the roots have opposite signs, giving a mixture dominated by the larger modulus.

### Exercise 2 (★) — $\mathbb E[a^2]$ for $c_1\ne c_2$; order-2 region for $c_2=2c_1$

With $a=1+w-\varphi$ and $\varphi=c_1r_1+c_2r_2$ ($r_k$ i.i.d. $U(0,1)$, mean $1/2$, variance $1/12$):
$\mathbb E a=1+w-\tfrac{c_1+c_2}{2}$ and $\mathrm{Var}(a)=\mathrm{Var}(\varphi)=\tfrac{c_1^2+c_2^2}{12}$, so

$$
\mathbb E[a^2]=\Bigl(1+w-\frac{c_1+c_2}{2}\Bigr)^2+\frac{c_1^2+c_2^2}{12}.
$$

The notebook's `moment_matrix` already uses this general form. The boundary where an eigenvalue of $M$ crosses $+1$ is
$\det(I-M)=0$. With $\mu=\mathbb Ea$, $s=\mathbb E[a^2]$ and $M$ from the notebook, expansion along the first row gives
$\det(I-M)=(1+w)(1-s-w^2)+2w\mu^2$. Substituting $\mu=(1+w)-C/2$ with $C=c_1+c_2$ and $s=\mu^2+q/12$ with
$q=c_1^2+c_2^2$, the terms without $C$ or $q$ cancel, and $\det(I-M)=(1-w^2)C-\tfrac{1-w}{4}C^2-\tfrac{1+w}{12}q$.
Using $C^2=q+2c_1c_2$,

$$
\det(I-M)=-\tfrac16\Bigl[(2-w)(c_1^2+c_2^2)+3(1-w)c_1c_2-6(1-w^2)(c_1+c_2)\Bigr].
$$

For $|w|\lt1$ the order-2 region is therefore $(2-w)(c_1^2+c_2^2)+3(1-w)c_1c_2\lt6(1-w^2)(c_1+c_2)$, provided that the
eigenvalue $-1$ crossing never comes first (checked below). Checks:

* For $c_1=c_2=c/2$ the left side is $c^2(7-5w)/4$, which gives Poli's $c\lt24(1-w^2)/(7-5w)$.
* For $c_2=2c_1$ put $c=3c_1$. The left side is $c_1^2(16-11w)$, which gives

$$
c_1+c_2\lt\frac{54(1-w^2)}{16-11w}.
$$

The closed form of $\det(I-M)$ agrees with the numerical determinant to $1.2\times10^{-14}$ at 1000 random points, and
bisection on $\rho(M)=1$ reproduces the $c_2=2c_1$ curve to $8\times10^{-15}$ on 39 values of $w\in[-0.95,0.95]$, so
the $\lambda=-1$ crossing never comes first. At $w=0.7298$ the limit is $3.1659$, below the $3.3475$ allowed for
$c_1=c_2$: unequal coefficients increase $\mathrm{Var}(\varphi)$ for the same sum.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt

def moment_matrix(w, c1, c2):                        # Notebook 2 (general c1, c2)
    Ea = 1 + w - (c1 + c2) / 2; Ea2 = Ea**2 + (c1**2 + c2**2) / 12
    return np.array([[Ea2, w**2, -2 * w * Ea], [1.0, 0.0, 0.0], [Ea, 0.0, -w]])
rho = lambda w, c1, c2: np.abs(np.linalg.eigvals(moment_matrix(w, c1, c2))).max()

# closed form of det(I - M) against the numerical determinant at random parameters
g = np.random.default_rng(0); err = 0.0
for w, c1, c2 in g.uniform([-1, 0, 0], [1, 4, 4], (1000, 3)):
    closed = -((2 - w) * (c1**2 + c2**2) + 3 * (1 - w) * c1 * c2 - 6 * (1 - w**2) * (c1 + c2)) / 6
    err = max(err, abs(np.linalg.det(np.eye(3) - moment_matrix(w, c1, c2)) - closed))
print(f"max |det(I-M) - closed form| over 1000 random points: {err:.1e}")

# c2 = 2 c1: bisection on rho(M) = 1 along c = c1 + c2 against 54(1 - w^2)/(16 - 11w)
ws, dev = np.linspace(-0.95, 0.95, 39), 0.0
for w in ws:
    a, b = 1e-6, 12.0                                # rho < 1 at a, > 1 at b
    for _ in range(100):
        m = (a + b) / 2; a, b = (m, b) if rho(w, m / 3, 2 * m / 3) < 1 else (a, m)
    dev = max(dev, abs(a - 54 * (1 - w**2) / (16 - 11 * w)))
print(f"max deviation bisection vs closed form (39 values of w): {dev:.1e}")
w0 = 0.7298
print(f"w = {w0}: c1+c2 limit {54 * (1 - w0**2) / (16 - 11 * w0):.4f} (c2 = 2c1) vs "
      f"{24 * (1 - w0**2) / (7 - 5 * w0):.4f} (c1 = c2)")

wg = np.linspace(-0.999, 0.999, 400)
plt.fill_between(wg, 0, 54 * (1 - wg**2) / (16 - 11 * wg), alpha=0.4, label="order-2 region, $c_2=2c_1$")
plt.plot(wg, 24 * (1 - wg**2) / (7 - 5 * wg), "k--", label="Poli's boundary, $c_1=c_2$")
plt.xlabel("$w$"); plt.ylabel("$c_1+c_2$"); plt.title("Order-2 stability region"); plt.legend()
```

### Exercise 3 (★★) — Stationary mean and variance when $p\ne l$

$$
x_{t+1}=(1+w-\varphi_t)x_t-wx_{t-1}+c_1r_1p+c_2r_2l .
$$

*Mean.* Take expectations. The $r$'s are independent of $x_t$, so $m=\mathbb E x$ at stationarity satisfies
$m=(1+w-\tfrac{c_1+c_2}{2})m-wm+\tfrac{c_1p+c_2l}{2}$, i.e.

$$
m=\frac{c_1p+c_2l}{c_1+c_2}.
$$

*Variance.* Put $y_t=x_t-m$, $u_1=p-m$, $u_2=l-m$. Then $y_{t+1}=a_ty_t-wy_{t-1}+\xi_t$ with $a_t=1+w-\varphi_t$ and
$\xi_t=c_1r_1u_1+c_2r_2u_2$. The pair $(a_t,\xi_t)$ is independent of $(y_t,y_{t-1})$. Also $c_1u_1+c_2u_2=0$, so
$\mathbb E\xi=0$ and $c_1c_2u_1u_2=-(c_1u_1)^2$. With $\mathbb E r^2=1/3$ and $\mathbb E r_1r_2=1/4$,

$$
\mathbb E\xi^2=\frac{c_1^2u_1^2+c_2^2u_2^2}{3}+\frac{c_1c_2u_1u_2}{2}=\frac{(c_1u_1)^2}{6}=\frac{c_1^2c_2^2(p-l)^2}{6(c_1+c_2)^2}.
$$

At stationarity $\mathbb E y=0$, so the cross terms $\mathbb E[a\xi]\,\mathbb Ey_t$ and $\mathbb E\xi\,\mathbb E y_{t-1}$
vanish. With $V=\mathbb E y^2$, $q=\mathbb E y_ty_{t-1}$, $\mu=\mathbb E a$ and $s=\mathbb E a^2$:

$$
V=sV+w^2V-2w\mu q+\mathbb E\xi^2,\qquad q=\mu V-wq\ \Rightarrow\ q=\frac{\mu V}{1+w},
$$

$$
\mathrm{Var}(x_\infty)=V=\frac{c_1^2c_2^2(p-l)^2}{6(c_1+c_2)^2\bigl(1-s-w^2+\frac{2w\mu^2}{1+w}\bigr)} .
$$

The denominator equals $\det(I-M)/(1+w)$ (Exercise 2), so it vanishes exactly on the order-2 boundary (checked
numerically on Poli's curve, to $5\times10^{-16}$): the variance is finite iff the particle is mean-square stable, and
it is proportional to $(p-l)^2$. For $c_1=c_2=c$ the numerator is $c^2(p-l)^2/24$.

Simulation ($p=1$, $l=3$, $2\times10^5$ particles, 300 steps from $x_0=x_{-1}=2$):

| $(w,c_1,c_2)$ | mean: theory / sim | variance: theory / sim |
|---|---|---|
| (0.7298, 1.49618, 1.49618) | 2.0000 / 1.9963 | 4.3497 / 4.18 ± 0.05 (8 replicates, range 4.00–4.45) |
| (0.6, 1, 2) | 2.3333 / 2.3283 | 1.3417 / 1.3484 |

The second setting agrees to 0.5%. The default parameters are close to the order-2 boundary ($\rho(M)=0.944$), so
the stationary law is heavy-tailed: its fourth moment is (at best) barely finite, and the sample variance is a
right-skewed estimator that is usually *below* the true variance and occasionally far above it. That is why the eight
replicates scatter from 4.00 to 4.45 and why their mean sits about 4% below the theory; the "± SE" is only indicative
for such a heavy-tailed estimator.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

def stationary(w, c1, c2, p, l):
    """Stationary mean and variance of x_t for fixed p != l (derived above)."""
    m = (c1 * p + c2 * l) / (c1 + c2); mu = 1 + w - (c1 + c2) / 2; s = mu**2 + (c1**2 + c2**2) / 12
    return m, (c1 * c2 * (p - l) / (c1 + c2))**2 / 6 / (1 - s - w**2 + 2 * w * mu**2 / (1 + w))

def simulate(w, c1, c2, p, l, rng, N=200_000, T=300):
    x = np.full(N, 2.0); xp = x.copy()
    for t in range(T):
        x, xp = x + w * (x - xp) + c1 * rng.random(N) * (p - x) + c2 * rng.random(N) * (l - x), x
    return x.mean(), x.var()

# the denominator vanishes on Poli's boundary c1 = c2 = c/2, c = 24(1 - w^2)/(7 - 5w)
den = lambda w, c1, c2: (1 - (1 + w - (c1 + c2) / 2)**2 - (c1**2 + c2**2) / 12 - w**2
                         + 2 * w * (1 + w - (c1 + c2) / 2)**2 / (1 + w))
wb = np.linspace(-0.9, 0.9, 19); cb = 24 * (1 - wb**2) / (7 - 5 * wb)
print(f"max |denominator| on Poli's curve: {np.abs(den(wb, cb / 2, cb / 2)).max():.1e}")

p, l = 1.0, 3.0
for (w, c1, c2), reps in [((0.7298, 1.49618, 1.49618), 8), ((0.6, 1.0, 2.0), 1)]:
    m, v = stationary(w, c1, c2, p, l)
    sims = np.array([simulate(w, c1, c2, p, l, np.random.default_rng(10 + k)) for k in range(reps)])
    se = f" +- {sims[:, 1].std(ddof=1) / np.sqrt(reps):.2f} (range {sims[:, 1].min():.2f}-{sims[:, 1].max():.2f})" if reps > 1 else ""
    print(f"(w, c1, c2) = ({w}, {c1}, {c2}): mean {m:.4f} / {sims[0, 0]:.4f}   "
          f"variance {v:.4f} / {sims[:, 1].mean():.4f}{se}")
```

### Exercise 4 (★★) — Momentum gradient descent is the deterministic PSO model

Heavy-ball momentum on $f(x)=\tfrac12x^\top Hx$ ($H$ symmetric, eigenvalues $0\lt\lambda_1\le\dots\le\lambda_d=L$) is
$x_{t+1}=x_t-\eta Hx_t+\beta(x_t-x_{t-1})$. In the eigenbasis of $H$ each coordinate $z$ evolves independently:

$$
z_{t+1}=(1+\beta-\eta\lambda)z_t-\beta z_{t-1},
$$

with characteristic polynomial $\mu^2-(1+\beta-\eta\lambda)\mu+\beta$. This is the deterministic PSO polynomial with
$w=\beta$ and $\varphi=\eta\lambda$, with the attractor at the minimiser. By the Jury conditions every coordinate
converges iff $|\beta|\lt1$ and $0\lt\eta\lambda_i\lt2(1+\beta)$ for all $i$. Since $\eta\lambda_i\gt0$ automatically,
the binding constraint is the largest eigenvalue:

$$
|\beta|\lt1,\qquad 0\lt\eta\lt\frac{2(1+\beta)}{L}.
$$

For $\beta=0$ this is the classical gradient-descent bound $\eta\lt2/L$. Momentum widens the admissible step up to
nearly $4/L$ as $\beta\to1$.

### Exercise 5 (★★) — Almost-sure (Lyapunov) region vs order-2 region

The top Lyapunov exponent $\gamma=\lim_t\frac1t\log\Vert(y_t,y_{t-1})\Vert$ is estimated with renormalisation
(3000 particles, 300 steps, $c_1=c_2$). $\gamma\lt0$ means that almost every trajectory converges.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt

def lyapunov(w, c, rng, N=3000, T=300):
    """Top Lyapunov exponent of y_{t+1} = (1 + w - phi_t) y_t - w y_{t-1}, phi_t = (c/2)(r1 + r2);
    w and c are arrays of grid points (vectorised), N particles per point, renormalised every step."""
    w, c = w[:, None], c[:, None]
    y, yp = rng.standard_normal((2, len(w), N)); s = np.zeros(y.shape)
    for t in range(T):
        phi = c / 2 * (rng.random(y.shape) + rng.random(y.shape))
        y, yp = (1 + w - phi) * y - w * yp, y
        nr = np.sqrt(y**2 + yp**2); s += np.log(nr); y /= nr; yp /= nr
    return s.mean(1) / T

def rho2(w, c):                                          # order-2 spectral radius, c1 = c2 = c/2
    Ea = 1 + w - c / 2; M = np.array([[Ea**2 + c**2 / 24, w**2, -2 * w * Ea], [1, 0, 0], [Ea, 0, -w]])
    return np.abs(np.linalg.eigvals(M)).max()

wv, cv = np.linspace(-0.9, 0.95, 20), np.linspace(0.1, 7.5, 38)
W, C = np.meshgrid(wv, cv, indexing="ij")
gam = lyapunov(W.ravel(), C.ravel(), np.random.default_rng(0)).reshape(W.shape)
R2 = np.vectorize(rho2)(W, C)
a_s, m_s = gam < 0, R2 < 1
print(f"a.s. stable {a_s.mean():.0%}, mean-square stable {m_s.mean():.0%}, "
      f"m.s. but gamma >= 0: {(m_s & ~a_s).sum()}, a.s. but not m.s.: {(a_s & ~m_s).sum()}")
for i in [9, 13, 17, 18]:
    print(f"w = {wv[i]:6.3f}: largest stable c1+c2 on the grid  gamma<0: {cv[a_s[i]].max():.2f}   "
          f"rho(M)<1: {cv[m_s[i]].max():.2f}   order-1 limit 4(1+w) = {4 * (1 + wv[i]):.2f}")

plt.contourf(W, C, gam, levels=[-10, 0], colors=["C0"], alpha=0.35)
plt.contour(W, C, R2, levels=[1], colors="k")
plt.plot(wv, 4 * (1 + wv), "k:", label="order-1 boundary")
plt.xlabel("$w$"); plt.ylabel("$c_1+c_2$"); plt.legend()
plt.title("shaded: Lyapunov exponent < 0 (a.s.); solid: rho(M) = 1 (order 2)")
```

On a $20\times38$ grid ($w\in[-0.9,0.95]$, $c_1+c_2\in[0.1,7.5]$), 46% of the points are almost-surely stable
and 35% mean-square stable. No point is mean-square stable with $\gamma\ge0$, while 79 points are almost-surely stable
but not mean-square stable. Largest stable $c_1+c_2$ on the grid:

| $w$ | $\gamma\lt0$ | $\rho(M)\lt1$ | order-1 limit $4(1+w)$ |
|---|---|---|---|
| -0.024 | 4.50 | 3.30 | 3.91 |
| 0.366 | 5.10 | 3.90 | 5.46 |
| 0.755 | 4.50 | 3.10 | 7.02 |
| 0.853 | 3.70 | 2.30 | 7.41 |

**The almost-sure region is larger.** By Jensen's inequality,
$\mathbb E\log\Vert Y_t\Vert\le\tfrac12\log\mathbb E\Vert Y_t\Vert^2$ for $Y_t=(y_t,y_{t-1})$, and
$\mathbb E\Vert Y_t\Vert^2$ grows like $\rho(M)^t$, so $\gamma\le\tfrac12\log\rho(M)$. Mean-square stability therefore
implies almost-sure stability, but not conversely. In between, the typical trajectory converges while rare, huge
excursions keep the second moment growing (heavy tails). The a.s. region is not contained in the order-1 region
either: for $w\approx0$ it extends beyond $4(1+w)$. Order-1 and a.s. stability are different notions, and neither
implies the other.

### Exercise 6 (★★★) — Improvements vs the stagnation prediction on the sphere

*Design.* Run gbest PSO on the 10-D sphere ($n=40$, 20 runs, 300 iterations, no bound handling) for several $(w,c)$
with different $\rho(M)$. Record (i) the fitted slope of the median log swarm radius over iterations 50–300, (ii) the
fraction of particle updates that improve $p_i$, and (iii) the two stagnation predictions: the RMS rate
$\tfrac12\log\rho(M)$ and the typical rate $\gamma$ (Lyapunov exponent, simulated).

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import sphere

def rho2(w, c):                                   # order-2 spectral radius, c1 = c2 = c (Notebook 2)
    Ea = 1 + w - c; M = np.array([[Ea**2 + c**2 / 6, w**2, -2 * w * Ea], [1, 0, 0], [Ea, 0, -w]])
    return np.abs(np.linalg.eigvals(M)).max()

def lyapunov(w, c, rng, N=20000, T=400):          # typical contraction rate under stagnation
    y, yp, s = rng.standard_normal(N), rng.standard_normal(N), 0.0
    for t in range(T):
        y, yp = (1 + w - c * (rng.random(N) + rng.random(N))) * y - w * yp, y
        nr = np.sqrt(y**2 + yp**2); s += np.log(nr).mean(); y /= nr; yp /= nr
    return s / T

def gbest_sphere(w, c, rng, d=10, n=40, runs=20, T=300):
    """gbest PSO on the sphere without bound handling; median swarm radius and improvement rate per iteration."""
    X = rng.uniform(-5.12, 5.12, (runs, n, d)); V = (rng.uniform(-5.12, 5.12, (runs, n, d)) - X) / 2
    Pf = sphere(X.reshape(-1, d)).reshape(runs, n); P = X.copy(); rad, imp_rate = [], []
    for t in range(T):
        L = P[np.arange(runs), Pf.argmin(1)][:, None]
        r = rng.random((2, runs, n, d)); V = w * V + c * r[0] * (P - X) + c * r[1] * (L - X); X = X + V
        F = sphere(X.reshape(-1, d)).reshape(runs, n); imp = F < Pf; P[imp], Pf[imp] = X[imp], F[imp]
        rad.append(np.median(np.linalg.norm(X - X.mean(1, keepdims=True), axis=2).mean(1))); imp_rate.append(imp.mean())
    slope = np.polyfit(np.arange(50, T), np.log(rad[50:]), 1)[0]
    return slope, np.mean(imp_rate)

print("    w      c   rho(M)  0.5 log rho   gamma   radius slope   improvement rate")
for k, (w, c) in enumerate([(0.7298, 0.9), (0.7298, 1.2), (0.7298, 1.49618), (0.7298, 1.6), (0.5, 1.5), (0.4, 1.9)]):
    rho = rho2(w, c); slope, ir = gbest_sphere(w, c, np.random.default_rng(k))
    print(f"{w:6.3f} {c:6.3f}  {rho:6.4f}  {0.5 * np.log(rho):10.3f}  {lyapunov(w, c, np.random.default_rng(50 + k)):7.3f}"
          f"  {slope:12.3f}  {ir:16.3f}")
```

| $w$ | $c$ | $\rho(M)$ | $\tfrac12\log\rho$ | $\gamma$ | radius slope | improvement rate |
|---|---|---|---|---|---|---|
| 0.730 | 0.900 | 0.8234 | -0.097 | -0.127 | -0.115 | 0.447 |
| 0.730 | 1.200 | 0.8744 | -0.067 | -0.112 | -0.087 | 0.345 |
| 0.730 | 1.496 | 0.9442 | -0.029 | -0.097 | -0.057 | 0.230 |
| 0.730 | 1.600 | 0.9752 | -0.013 | -0.090 | -0.044 | 0.190 |
| 0.500 | 1.500 | 0.7215 | -0.163 | -0.261 | -0.146 | 0.589 |
| 0.400 | 1.900 | 0.8775 | -0.065 | -0.266 | -0.108 | 0.441 |

*Findings.*

* Along $w=0.7298$, the closer $\rho(M)$ is to 1, the slower the swarm contracts and the fewer updates improve a
  memory. The ordering of the observed rates follows $\rho(M)$ exactly.
* The observed contraction is always slower than the typical stagnation rate $\gamma$: every improvement moves an
  attractor and re-injects spread. It is usually faster than the RMS rate, because the median radius is a typical, not
  a mean-square, statistic.
* When improvements are very frequent ($w=0.5$, 59% improving updates), the swarm contracts even more slowly than the
  RMS prediction. The stagnation model then fails, because the attractors move almost every iteration.

In short, the stagnation analysis predicts the regime and the ranking of parameter settings, and it gives a bound on
the contraction speed. Improvement frequency measures how far a real swarm is from stagnation.

---

## Notebook 3 — Ant Colony Optimization for the TSP

### Exercise 1 (★) — Decay of an unused edge; time to reach $\tau_{\min}$

If edge $(i,j)$ receives no deposit after iteration $t_0$, the update reduces to $\tau_{t+1}=(1-\rho)\tau_t$. By
induction, $\tau_t=(1-\rho)^{t-t_0}\tau_{t_0}$ for $t\ge t_0$, which is geometric decay. It falls below a level
$\tau_{\min}$ at the first $t$ with $(1-\rho)^{t-t_0}\tau_{t_0}\le\tau_{\min}$:

$$
t-t_0\ \ge\ \Bigl\lceil\frac{\ln(\tau_{\min}/\tau_{t_0})}{\ln(1-\rho)}\Bigr\rceil .
$$

For an edge that starts at $\tau_{\max}$, the MMAS ratio is
$\tau_{\min}/\tau_{\max}=(1-p_{\text{dec}})/((n/2-1)p_{\text{dec}})$ with $p_{\text{dec}}=p_{\text{best}}^{1/n}$.
With $p_{\text{best}}=0.05$:

| $n$ | $\rho$ | $\tau_{\min}/\tau_{\max}$ | iterations |
|---|---|---|---|
| 100 | 0.02 | 6.21e-4 | 366 |
| 100 | 0.5 | 6.21e-4 | 11 |
| 30 | 0.1 | 7.50e-3 | 47 |
| 10 | 0.3 | 8.73e-2 | 7 (the $t_{\text{hit}}$ of Notebook 4, Section 4) |

```python
import sys; sys.path.insert(0, "..")
import numpy as np

def iterations_to_tau_min(n, rho, p_best=0.05):
    """Iterations for an unused edge to decay from tau_max to the MMAS tau_min."""
    pdec = p_best ** (1 / n); ratio = (1 - pdec) / ((n / 2 - 1) * pdec)
    return ratio, int(np.ceil(np.log(ratio) / np.log(1 - rho)))

for n, rho in [(100, 0.02), (100, 0.5), (30, 0.1), (10, 0.3)]:
    ratio, t = iterations_to_tau_min(n, rho)
    tau = 1.0                                       # direct check: iterate tau <- (1 - rho) tau
    for k in range(t): tau *= 1 - rho
    print(f"n = {n:3d}, rho = {rho:4.2f}: tau_min/tau_max = {ratio:.3g}, iterations = {t}, "
          f"(1-rho)^t = {tau:.3g} <= ratio: {tau <= ratio}")
```

### Exercise 2 (★) — ACS keeps pheromone in $[\tau_0,1/L_{bs}]$

*Claim.* If $\tau_0\le1/L_{bs}(t)$, then every trail satisfies $\tau_0\le\tau_{ij}\le1/L_{bs}(t)$ at every time $t$.

*Proof by induction over update events.* Initially $\tau_{ij}=\tau_0$. The best-so-far length $L_{bs}$ never increases,
so the upper end $1/L_{bs}$ never decreases. The hypothesis at the first iteration therefore implies it for all later
times, and an interval that contains $\tau_{ij}$ keeps containing it when its upper end grows. There are two kinds of
updates.

* Local update: $\tau\leftarrow(1-\xi)\tau+\xi\tau_0$ with $\xi\in(0,1]$. This is a convex combination of $\tau$ and
  $\tau_0$, both in $[\tau_0,1/L_{bs}]$, so the result is too.
* Global update on the best-so-far edges: $\tau\leftarrow(1-\rho)\tau+\rho/L_{bs}$, a convex combination of $\tau$
  and $1/L_{bs}$, both in the interval.

Edges not touched by either update keep their value. $\square$

Consequence: ACS has implicit MAX–MIN bounds with $\tau_{\min}=\tau_0=1/(nC^{nn})$. That is why its pheromone
entropy stays high in Notebook 4 (median 3.09 at $t=10$ and 3.09 at $t=399$, against 1.24 for MMAS).

### Exercise 3 (★★) — Candidate lists

Restrict the choice to the $K=15$ nearest cities that are still unvisited, and fall back to all unvisited cities when
none of them is left. MMAS on the 100-city instance of Section 7 ($m=25$, $\rho=0.1$, 6000 tours, 8 runs, the same
seed as the notebook, so the first row reproduces the notebook's "MMAS (no LS)" result):

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import random_euclidean_tsp, tour_length

def sample_next(w, rng):                             # row-wise roulette wheel (Notebook 3)
    cum = np.cumsum(w, axis=-1); u = rng.random(w.shape[:-1]) * cum[..., -1]
    return (cum <= u[..., None]).sum(-1)

def construct_tours(W, m, rng, cand=None):
    """m tours per run from weights W (runs, n, n); cand (n, n) bool = candidate lists (fallback: all unvisited).
    Also returns the number of weights examined per tour."""
    R, n, _ = W.shape; rr, mm = np.arange(R)[:, None], np.arange(m)[None, :]
    tours = np.empty((R, m, n), dtype=int); cur = rng.integers(0, n, (R, m)); tours[:, :, 0] = cur
    unvisited = np.ones((R, m, n), bool); unvisited[rr, mm, cur] = False; examined = 0
    for s in range(1, n):
        w = W[rr, cur] * unvisited
        dead = w.sum(-1) <= 0
        if dead.any(): w[dead] = unvisited[dead]          # numerical underflow guard
        if cand is not None:
            wc = w * cand[cur]; has = (wc > 0).any(-1)
            examined += np.where(has, cand[cur].sum(-1), unvisited.sum(-1)).mean()   # scan K candidates
            w = np.where(has[..., None], wc, w)
        else:
            examined += unvisited.sum(-1).mean()
        nxt = sample_next(w, rng); tours[:, :, s] = nxt; unvisited[rr, mm, nxt] = False; cur = nxt
    return tours, examined

def tour_lengths(tours, D):
    return D[tours, np.roll(tours, -1, axis=-1)].sum(-1)

def deposit(tau, tours, amount, symmetric=True):     # amount (runs, k) on the edges of tours (runs, k, n)
    a, b = tours, np.roll(tours, -1, axis=-1)
    ri = np.broadcast_to(np.arange(tours.shape[0])[:, None, None], a.shape); amt = np.broadcast_to(amount[..., None], a.shape)
    np.add.at(tau, (ri, a, b), amt)
    if symmetric: np.add.at(tau, (ri, b, a), amt)

def tau_bounds(L_bs, rho, n, p_best=0.05):           # MMAS bounds (Stuetzle & Hoos 2000)
    tmax = 1.0 / (rho * L_bs); pdec = p_best ** (1.0 / n)
    return tmax, tmax * (1 - pdec) / ((n / 2 - 1) * pdec)

def nn_length(D):                                    # nearest-neighbour tour from city 0
    n = len(D); t = [0]; left = set(range(1, n))
    while left:
        j = min(left, key=lambda c: D[t[-1], c]); t.append(j); left.remove(j)
    return tour_length(t, D)
import time

def mmas_tsp(D, runs, iters, m, rho, rng, beta=2.0, cand=None):
    """MMAS (iteration-best deposit, no restarts) vectorised over runs."""
    n = len(D); eta_b = np.where(np.eye(n, dtype=bool), 0.0, 1.0 / (D + np.eye(n))) ** beta
    tau = np.full((runs, n, n), 1.0 / (rho * nn_length(D))); best = np.full(runs, np.inf); rr = np.arange(runs); ex = 0.0
    for it in range(iters):
        tours, e = construct_tours(tau * eta_b, m, rng, cand); ex += e
        L = tour_lengths(tours, D); ib = L.argmin(1); best = np.minimum(best, L[rr, ib])
        tau *= 1 - rho; deposit(tau, tours[rr, ib][:, None], 1.0 / L[rr, ib][:, None])
        tmax, tmin = tau_bounds(best, rho, n); tau = np.clip(tau, tmin[:, None, None], tmax[:, None, None])
    return best, ex / iters

_, D100 = random_euclidean_tsp(100, np.random.default_rng(60))     # the 100-city instance of Notebook 3, Section 7
L_BEST = 8.3476                                                     # best-known length printed by Notebook 3
K = 15
order = np.argsort(D100 + np.diag(np.full(100, np.inf)), axis=1)[:, :K]
CM = np.zeros((100, 100), bool); CM[np.arange(100)[:, None], order] = True
res = {}
for name, cand in [("no candidate list", None), (f"K = {K}", CM)]:
    t0 = time.time(); b, ex = mmas_tsp(D100, 8, 240, 25, 0.1, np.random.default_rng(62), cand=cand)
    res[name] = b
    print(f"{name:18s} median best {np.median(b):.4f}  gap {100 * (np.median(b) / L_BEST - 1):5.2f}%   "
          f"weights examined per tour {ex:6.0f}   time {time.time() - t0:4.1f} s")
print(f"Mann-Whitney p = {mannwhitneyu(*res.values()).pvalue:.3f}")
```

| | median best | gap to 8.3476 | weights examined per tour |
|---|---|---|---|
| no candidate list | 8.5840 | 2.83% | 4950 |
| $K=15$ | 8.3935 | 0.55% | 1488 |

Mann–Whitney p = 0.007. **Speed.** The work per construction step drops from the $\approx n/2$ unvisited cities on
average to the $K$ list entries, 3.3 times less here: $O(nK)$ instead of $O(n^2)$ per tour in a scalar
implementation. In our vectorised NumPy code the full row is still formed and masked, and the extra masking makes the
candidate version *slower* in wall-clock time. This is a reminder to measure speed-ups by operation counts, or with an
implementation that actually exploits them. **Quality** improves markedly at an equal tour budget: the sampler no
longer wastes probability on long edges, which acts like a much stronger heuristic.

### Exercise 4 (★★) — Iteration-best / best-so-far deposit schedule

Schedule (in the spirit of Stützle & Hoos 2000): the best-so-far tour deposits every $k$-th iteration, with $k=5$ for
iterations below 25, $k=3$ below 75, $k=2$ below 125 and $k=1$ afterwards. Otherwise the iteration-best tour deposits.
MMAS + 2-opt on the 100-city instance of Section 7 ($m=10$, $\rho=0.2$, 6000 cost units, 8 seeds, the cost
accounting of the notebook). Gap to the notebook's best-known 8.3476:

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import random_euclidean_tsp, tour_length

def sample_next(w, rng):                             # row-wise roulette wheel (Notebook 3)
    cum = np.cumsum(w, axis=-1); u = rng.random(w.shape[:-1]) * cum[..., -1]
    return (cum <= u[..., None]).sum(-1)

def construct_tours(W, m, rng, cand=None):
    """m tours per run from weights W (runs, n, n); cand (n, n) bool = candidate lists (fallback: all unvisited).
    Also returns the number of weights examined per tour."""
    R, n, _ = W.shape; rr, mm = np.arange(R)[:, None], np.arange(m)[None, :]
    tours = np.empty((R, m, n), dtype=int); cur = rng.integers(0, n, (R, m)); tours[:, :, 0] = cur
    unvisited = np.ones((R, m, n), bool); unvisited[rr, mm, cur] = False; examined = 0
    for s in range(1, n):
        w = W[rr, cur] * unvisited
        dead = w.sum(-1) <= 0
        if dead.any(): w[dead] = unvisited[dead]          # numerical underflow guard
        if cand is not None:
            wc = w * cand[cur]; has = (wc > 0).any(-1)
            examined += np.where(has, cand[cur].sum(-1), unvisited.sum(-1)).mean()   # scan K candidates
            w = np.where(has[..., None], wc, w)
        else:
            examined += unvisited.sum(-1).mean()
        nxt = sample_next(w, rng); tours[:, :, s] = nxt; unvisited[rr, mm, nxt] = False; cur = nxt
    return tours, examined

def tour_lengths(tours, D):
    return D[tours, np.roll(tours, -1, axis=-1)].sum(-1)

def deposit(tau, tours, amount, symmetric=True):     # amount (runs, k) on the edges of tours (runs, k, n)
    a, b = tours, np.roll(tours, -1, axis=-1)
    ri = np.broadcast_to(np.arange(tours.shape[0])[:, None, None], a.shape); amt = np.broadcast_to(amount[..., None], a.shape)
    np.add.at(tau, (ri, a, b), amt)
    if symmetric: np.add.at(tau, (ri, b, a), amt)

def tau_bounds(L_bs, rho, n, p_best=0.05):           # MMAS bounds (Stuetzle & Hoos 2000)
    tmax = 1.0 / (rho * L_bs); pdec = p_best ** (1.0 / n)
    return tmax, tmax * (1 - pdec) / ((n / 2 - 1) * pdec)

def nn_length(D):                                    # nearest-neighbour tour from city 0
    n = len(D); t = [0]; left = set(range(1, n))
    while left:
        j = min(left, key=lambda c: D[t[-1], c]); t.append(j); left.remove(j)
    return tour_length(t, D)

def neighbour_lists(D, K):
    order = np.argsort(D, axis=1)
    return [list(map(int, row[row != i][:K])) for i, row in enumerate(order)]

def two_opt_nl(tour, D, nbrs):
    """First-improvement 2-opt with neighbour lists and don't-look bits (Notebook 3); returns (tour, move evaluations)."""
    t = [int(c) for c in tour]; n = len(t); pos = [0] * n
    for i, c in enumerate(t): pos[c] = i
    Dl = D.tolist()
    def reverse(i, j):
        for k in range(((j - i) % n + 1) // 2):
            a, b = (i + k) % n, (j - k) % n; t[a], t[b] = t[b], t[a]; pos[t[a]], pos[t[b]] = a, b
    active = list(range(n)); dlb = [False] * n; evals = 0
    while active:
        a = active.pop()
        if dlb[a]: continue
        improved = False
        for succ in (True, False):
            ia = pos[a]; b = t[(ia + 1) % n] if succ else t[(ia - 1) % n]; d_ab = Dl[a][b]
            for c in nbrs[a]:
                d_ac = Dl[a][c]
                if d_ac >= d_ab: break
                ic = pos[c]; d = t[(ic + 1) % n] if succ else t[(ic - 1) % n]
                if c == b or d == a: continue
                evals += 1
                if d_ac + Dl[b][d] - d_ab - Dl[c][d] < -1e-12:
                    reverse(ia + 1, ic) if succ else reverse(ic, ia - 1)
                    for x in (a, b, c, d):
                        if dlb[x] or x not in active: dlb[x] = False; active.append(x)
                    improved = True; break
            if improved: break
        if not improved: dlb[a] = True
    return np.array(t), evals

def schedule_k(it):                                  # best-so-far deposits every k-th iteration
    return 5 if it < 25 else 3 if it < 75 else 2 if it < 125 else 1

def mmas_2opt(D, runs, m, rho, rng, units, rule="ib"):
    """MMAS + 2-opt; cost = tours + 2-opt move evaluations * 4/n (Notebook 3). Best length within `units`."""
    n = len(D); nbrs = neighbour_lists(D, 10)
    eta_b = np.where(np.eye(n, dtype=bool), 0.0, 1.0 / (D + np.eye(n))) ** 2
    tau = np.full((runs, n, n), 1.0 / (rho * nn_length(D))); rr = np.arange(runs)
    best, best_tour = np.full(runs, np.inf), np.zeros((runs, n), int); cost = np.zeros(runs); out = np.full(runs, np.inf)
    it = 0
    while (cost < units).any():
        tours, _ = construct_tours(tau * eta_b, m, rng); cost += m
        for r in range(runs):
            for k in range(m):
                tours[r, k], ev = two_opt_nl(tours[r, k], D, nbrs); cost[r] += ev * 4.0 / n
        L = tour_lengths(tours, D); ib = L.argmin(1)
        imp = L[rr, ib] < best - 1e-12; best[imp], best_tour[imp] = L[rr, ib][imp], tours[rr, ib][imp]
        ok = cost <= units; out[ok] = best[ok]                   # only iterations completed within the budget count
        use_bs = rule == "bs" or (rule == "schedule" and it % schedule_k(it) == schedule_k(it) - 1)
        dt, dl = (best_tour, best) if use_bs else (tours[rr, ib], L[rr, ib])
        tau *= 1 - rho; deposit(tau, dt[:, None], 1.0 / dl[:, None])
        tmax, tmin = tau_bounds(best, rho, n); tau = np.clip(tau, tmin[:, None, None], tmax[:, None, None]); it += 1
    return out

_, D100 = random_euclidean_tsp(100, np.random.default_rng(60)); L_BEST = 8.3476    # Notebook 3, Section 7
res = {rule: 100 * (mmas_2opt(D100, 8, 10, 0.2, np.random.default_rng(61), 6000, rule) / L_BEST - 1)
       for rule in ["ib", "schedule", "bs"]}
for rule, g in res.items():
    print(f"{rule:9s} gap median {np.median(g):5.2f}%  IQR [{np.percentile(g, 25):.2f}, {np.percentile(g, 75):.2f}]  best {g.min():.2f}%")
print(f"MWU p: schedule vs ib {mannwhitneyu(res['schedule'], res['ib']).pvalue:.2f},  "
      f"best-so-far vs ib {mannwhitneyu(res['bs'], res['ib']).pvalue:.3f}")
```

| deposit rule | median | IQR | best |
|---|---|---|---|
| iteration-best (reproduces Section 7) | 0.29% | [0.03, 0.42] | 0.00% |
| schedule | 0.33% | [-0.07, 0.64] | -0.44% |
| best-so-far only | -0.30% | [-0.46, -0.02] | -0.71% |

The schedule is not distinguishable from iteration-best (p = 0.96). Pure best-so-far deposit is significantly better
(p = 0.007) and even beats the notebook's best-known tour: with 2-opt the colony can afford strong exploitation at
this short budget (only about 24 iterations fit into 6000 cost units, so the schedule never leaves its exploratory
$k=5$ phase). The schedule was designed for long runs, where early exploration pays off. This is again a
budget-dependent conclusion.

### Exercise 5 (★★) — $\lambda$-branching factor and restarts

*Definition* (Gambardella & Dorigo 1995). For city $i$ let $\tau^i_{\min}$ and $\tau^i_{\max}$ be the smallest and
largest trail on its row. The $\lambda$-branching factor of $i$ counts the edges with
$\tau_{ij}\ge\tau^i_{\min}+\lambda(\tau^i_{\max}-\tau^i_{\min})$. The average over cities (with $\lambda=0.05$) is
$n-1$ for uniform pheromone and $2$ for a fully converged symmetric MMAS (two tour neighbours at $\tau_{\max}$).
Restart rule: reset all trails to $\tau_{\max}$ when the branching factor is below $2.05$ and the best-so-far tour has
not improved for 25 iterations.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import random_euclidean_tsp, tour_length

def sample_next(w, rng):                             # row-wise roulette wheel (Notebook 3)
    cum = np.cumsum(w, axis=-1); u = rng.random(w.shape[:-1]) * cum[..., -1]
    return (cum <= u[..., None]).sum(-1)

def construct_tours(W, m, rng, cand=None):
    """m tours per run from weights W (runs, n, n); cand (n, n) bool = candidate lists (fallback: all unvisited).
    Also returns the number of weights examined per tour."""
    R, n, _ = W.shape; rr, mm = np.arange(R)[:, None], np.arange(m)[None, :]
    tours = np.empty((R, m, n), dtype=int); cur = rng.integers(0, n, (R, m)); tours[:, :, 0] = cur
    unvisited = np.ones((R, m, n), bool); unvisited[rr, mm, cur] = False; examined = 0
    for s in range(1, n):
        w = W[rr, cur] * unvisited
        dead = w.sum(-1) <= 0
        if dead.any(): w[dead] = unvisited[dead]          # numerical underflow guard
        if cand is not None:
            wc = w * cand[cur]; has = (wc > 0).any(-1)
            examined += np.where(has, cand[cur].sum(-1), unvisited.sum(-1)).mean()   # scan K candidates
            w = np.where(has[..., None], wc, w)
        else:
            examined += unvisited.sum(-1).mean()
        nxt = sample_next(w, rng); tours[:, :, s] = nxt; unvisited[rr, mm, nxt] = False; cur = nxt
    return tours, examined

def tour_lengths(tours, D):
    return D[tours, np.roll(tours, -1, axis=-1)].sum(-1)

def deposit(tau, tours, amount, symmetric=True):     # amount (runs, k) on the edges of tours (runs, k, n)
    a, b = tours, np.roll(tours, -1, axis=-1)
    ri = np.broadcast_to(np.arange(tours.shape[0])[:, None, None], a.shape); amt = np.broadcast_to(amount[..., None], a.shape)
    np.add.at(tau, (ri, a, b), amt)
    if symmetric: np.add.at(tau, (ri, b, a), amt)

def tau_bounds(L_bs, rho, n, p_best=0.05):           # MMAS bounds (Stuetzle & Hoos 2000)
    tmax = 1.0 / (rho * L_bs); pdec = p_best ** (1.0 / n)
    return tmax, tmax * (1 - pdec) / ((n / 2 - 1) * pdec)

def nn_length(D):                                    # nearest-neighbour tour from city 0
    n = len(D); t = [0]; left = set(range(1, n))
    while left:
        j = min(left, key=lambda c: D[t[-1], c]); t.append(j); left.remove(j)
    return tour_length(t, D)
import matplotlib.pyplot as plt

def branching_factor(tau, lam=0.05):
    """Average lambda-branching factor of pheromone matrices tau (runs, n, n)."""
    n = tau.shape[-1]; T = tau[:, ~np.eye(n, dtype=bool)].reshape(tau.shape[0], n, n - 1)
    lo, hi = T.min(-1, keepdims=True), T.max(-1, keepdims=True)
    return (T >= lo + lam * (hi - lo)).sum(-1).mean(-1)

def row_entropy(tau):
    n = tau.shape[-1]; T = tau[:, ~np.eye(n, dtype=bool)].reshape(tau.shape[0], n, n - 1)
    P = T / T.sum(-1, keepdims=True); return -(P * np.log(P)).sum(-1).mean(-1)

def mmas_restart(D, runs, iters, m, rho, rng, restart="none"):
    n = len(D); eta_b = np.where(np.eye(n, dtype=bool), 0.0, 1.0 / (D + np.eye(n))) ** 2
    tau = np.full((runs, n, n), 1.0 / (rho * nn_length(D))); rr = np.arange(runs)
    best = np.full(runs, np.inf); last = np.zeros(runs, int); n_restart = np.zeros(runs, int); bf, ent = [], []
    for it in range(iters):
        tours, _ = construct_tours(tau * eta_b, m, rng); L = tour_lengths(tours, D); ib = L.argmin(1)
        imp = L[rr, ib] < best - 1e-12; best[imp], last[imp] = L[rr, ib][imp], it
        tau *= 1 - rho; deposit(tau, tours[rr, ib][:, None], 1.0 / L[rr, ib][:, None])
        tmax, tmin = tau_bounds(best, rho, n); tau = np.clip(tau, tmin[:, None, None], tmax[:, None, None])
        b = branching_factor(tau); bf.append(b); ent.append(row_entropy(tau))
        stale = {"none": np.zeros(runs, bool), "stale100": it - last >= 100,
                 "bf": (b < 2.05) & (it - last >= 25)}[restart]
        tau[stale] = tmax[stale, None, None]; last[stale] = it; n_restart += stale
    return best, n_restart, np.array(bf).T, np.array(ent).T

_, D40 = random_euclidean_tsp(40, np.random.default_rng(40)); L_BEST = 5.2164      # Notebook 3, Section 6
gaps = {}
for k, rule in enumerate(["none", "stale100", "bf"]):
    best, nr, bf, ent = mmas_restart(D40, 10, 500, 40, 0.1, np.random.default_rng(80), rule)
    g = 100 * (best / L_BEST - 1); gaps[rule] = g
    print(f"{rule:9s} gap median {np.median(g):.2f}%  IQR [{np.percentile(g, 25):.2f}, {np.percentile(g, 75):.2f}]"
          f"  restarts/run {nr.mean():.1f}  runs at best-known {np.sum(g < 1e-3)}")
    if rule == "none": bf0, ent0 = bf, ent
from scipy.stats import kruskal
print(f"Kruskal-Wallis p = {kruskal(*gaps.values()).pvalue:.2f}")
mb, me = np.median(bf0, 0), np.median(ent0, 0)
print("median branching factor at t = 0, 10, 50, 100, 200, 499:", np.round(mb[[0, 10, 50, 100, 200, 499]], 2))
print("median row entropy at t = 0, 50, 499:", np.round(me[[0, 50, 499]], 2),
      f"(uniform: log 39 = {np.log(39):.2f});  corr(bf, entropy) = {np.corrcoef(bf0.ravel(), ent0.ravel())[0, 1]:.2f}")
fig, ax = plt.subplots(); ax.plot(mb, label="branching factor"); ax.set_xlabel("iteration")
ax.set_ylabel("lambda-branching factor (lambda = 0.05)"); ax2 = ax.twinx(); ax2.plot(me, "C1", label="row entropy")
ax2.set_ylabel("mean row entropy (nats)"); ax.set_title("MMAS without restarts, 40 cities (median of 10 runs)")
```

MMAS on the 40-city instance of Notebook 3, Section 5 ($m=40$, $\rho=0.1$, 500 iterations, 10 runs; gap to the
notebook's best-known 5.2164):

| restart rule | median gap | IQR | restarts per run | runs reaching best-known |
|---|---|---|---|---|
| none | 0.37% | [0.37, 0.41] | 0 | 1 |
| 100 stale iterations | 0.06% | [0.01, 0.37] | 2.9 | 3 |
| branching factor < 2.05 | 0.37% | [0.06, 0.37] | 3.8 | 2 |

Without restarts most runs stay in the same near-optimal tour (0.37% above the best-known). Both restart rules free
some runs from it; the stale-iteration rule does so most often here, but the three rules are only borderline
different (Kruskal–Wallis p = 0.05 with 10 runs). Over time the median branching factor falls from 39 to 9.8 ($t=10$),
4.65 ($t=50$), 2.1 ($t=100$) and exactly 2.00 from $t=200$ on. The median row entropy falls from $\log39=3.66$ to 1.73
($t=50$) and then settles at 1.15, which is the MMAS limit $H_{\lim}$ of Notebook 4 evaluated for $n=40$. The two
measures have correlation 0.80. The branching factor saturates at exactly 2, which makes it a crisper convergence
detector than entropy (whose limit depends on $n$ and $p_{\text{best}}$). The block plots both on twin axes.

### Exercise 6 (★★★) — MMAS for the asymmetric TSP

*Where the implementation assumes symmetry.*

1. `deposit` adds to both $\tau_{ab}$ and $\tau_{ba}$.
2. The ACS local and global updates write both directions.
3. `tour_lengths` is fine, since it reads $D[a,b]$ along the tour direction.
4. 2-opt reverses a segment, which changes the cost of every reversed edge in an ATSP. Its delta formula is wrong, so
   it must be replaced by orientation-preserving moves, for example or-opt (segment insertion without reversal) or
   the orientation-preserving 3-opt move.
5. $\eta=1/D$ is directional and fine, and so are the $\tau_{\min}$ formula and the construction.

Held–Karp is already directional: it uses $D[k,j]$ and closes with $D[\text{last},0]$.

*Implementation.* Deposit only on the directed edges $(a,b)$ of the depositing tour, with no 2-opt. Instances:
Euclidean distances multiplied by i.i.d. $U(0.5,1.5)$ factors, so $D_{ij}\ne D_{ji}$. MMAS with $m=n$, $\rho=0.1$,
100 iterations, 10 runs per instance:

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import random_euclidean_tsp, tour_length

def sample_next(w, rng):                             # row-wise roulette wheel (Notebook 3)
    cum = np.cumsum(w, axis=-1); u = rng.random(w.shape[:-1]) * cum[..., -1]
    return (cum <= u[..., None]).sum(-1)

def construct_tours(W, m, rng, cand=None):
    """m tours per run from weights W (runs, n, n); cand (n, n) bool = candidate lists (fallback: all unvisited).
    Also returns the number of weights examined per tour."""
    R, n, _ = W.shape; rr, mm = np.arange(R)[:, None], np.arange(m)[None, :]
    tours = np.empty((R, m, n), dtype=int); cur = rng.integers(0, n, (R, m)); tours[:, :, 0] = cur
    unvisited = np.ones((R, m, n), bool); unvisited[rr, mm, cur] = False; examined = 0
    for s in range(1, n):
        w = W[rr, cur] * unvisited
        dead = w.sum(-1) <= 0
        if dead.any(): w[dead] = unvisited[dead]          # numerical underflow guard
        if cand is not None:
            wc = w * cand[cur]; has = (wc > 0).any(-1)
            examined += np.where(has, cand[cur].sum(-1), unvisited.sum(-1)).mean()   # scan K candidates
            w = np.where(has[..., None], wc, w)
        else:
            examined += unvisited.sum(-1).mean()
        nxt = sample_next(w, rng); tours[:, :, s] = nxt; unvisited[rr, mm, nxt] = False; cur = nxt
    return tours, examined

def tour_lengths(tours, D):
    return D[tours, np.roll(tours, -1, axis=-1)].sum(-1)

def deposit(tau, tours, amount, symmetric=True):     # amount (runs, k) on the edges of tours (runs, k, n)
    a, b = tours, np.roll(tours, -1, axis=-1)
    ri = np.broadcast_to(np.arange(tours.shape[0])[:, None, None], a.shape); amt = np.broadcast_to(amount[..., None], a.shape)
    np.add.at(tau, (ri, a, b), amt)
    if symmetric: np.add.at(tau, (ri, b, a), amt)

def tau_bounds(L_bs, rho, n, p_best=0.05):           # MMAS bounds (Stuetzle & Hoos 2000)
    tmax = 1.0 / (rho * L_bs); pdec = p_best ** (1.0 / n)
    return tmax, tmax * (1 - pdec) / ((n / 2 - 1) * pdec)

def nn_length(D):                                    # nearest-neighbour tour from city 0
    n = len(D); t = [0]; left = set(range(1, n))
    while left:
        j = min(left, key=lambda c: D[t[-1], c]); t.append(j); left.remove(j)
    return tour_length(t, D)
from utils import held_karp

def mmas_atsp(D, runs, iters, m, rho, rng, symmetric=False):
    """MMAS with directed pheromone (deposit on (a, b) only) or the symmetric deposit of Notebook 3."""
    n = len(D); eta_b = np.where(np.eye(n, dtype=bool), 0.0, 1.0 / (D + np.eye(n))) ** 2
    tau = np.full((runs, n, n), 1.0 / (rho * nn_length(D))); rr = np.arange(runs); best = np.full(runs, np.inf)
    for it in range(iters):
        tours, _ = construct_tours(tau * eta_b, m, rng); L = tour_lengths(tours, D); ib = L.argmin(1)
        best = np.minimum(best, L[rr, ib])
        tau *= 1 - rho; deposit(tau, tours[rr, ib][:, None], 1.0 / L[rr, ib][:, None], symmetric)
        tmax, tmin = tau_bounds(best, rho, n); tau = np.clip(tau, tmin[:, None, None], tmax[:, None, None])
    return best

print("  n  inst  hit(directed)  hit(symmetric)  max gap(directed)  reversed optimum longer (median)  min gap")
for n, n_inst in [(8, 20), (10, 20), (12, 10)]:
    hd, hs, gmax, rev, gmin = [], [], 0.0, [], np.inf
    for s in range(n_inst):
        g = np.random.default_rng(1000 * n + s); _, D = random_euclidean_tsp(n, g)
        D = D * g.uniform(0.5, 1.5, (n, n)); np.fill_diagonal(D, 0.0)          # D[i, j] != D[j, i]
        opt, tour = held_karp(D)                                              # the DP is directional
        rev.append(tour_length(tour[::-1], D) / opt - 1)
        bd = mmas_atsp(D, 10, 100, n, 0.1, np.random.default_rng(s))
        bs = mmas_atsp(D, 10, 100, n, 0.1, np.random.default_rng(s), symmetric=True)
        hd.append(np.mean(bd < opt + 1e-9)); hs.append(np.mean(bs < opt + 1e-9))
        gmax = max(gmax, (bd / opt - 1).max()); gmin = min(gmin, (bd / opt - 1).min(), (bs / opt - 1).min())
    print(f"{n:3d} {n_inst:5d} {np.mean(hd):13.1%} {np.mean(hs):15.1%} {gmax:18.1%} {np.median(rev):33.1%} {gmin:8.1e}")
```

| $n$ | instances | hit rate (directed) | hit rate (symmetric deposit) | max gap (directed) | reversed optimum longer by (median) |
|---|---|---|---|---|---|
| 8 | 20 | 100% | 99.5% | 0.00% | 36.7% |
| 10 | 20 | 90% | 89% | 11.2% | 37.0% |
| 12 | 10 | 78% | 85% | 4.1% | 38.9% |

No run is shorter than the Held–Karp optimum (the smallest gap is $-3\times10^{-16}$, rounding), which validates both
the DP and the ACO. The optimal tour traversed backwards is about 37% longer, so the problem is genuinely asymmetric.
Surprisingly, the symmetric deposit is not worse at these tiny sizes: reinforcing $(b,a)$ as well only raises the
probability of reversed edges, and the heuristic $\eta_{ij}=1/D_{ij}$ is directional and corrects most of those
choices. For larger ATSPs directed pheromone is the correct model.

---

## Notebook 4 — ACO Theory and Extensions

### Exercise 1 (★) — Score-function gradient for the TSP construction model

An ant starting at $s_0$ builds $s=(s_0,\dots,s_{n-1})$. At step $t$ it is at $i=s_t$ with unvisited set $\mathcal N_t$
and chooses $j$ with $p^{(t)}_{ij}=\tau_{ij}\eta^\beta_{ij}/\sum_{l\in\mathcal N_t}\tau_{il}\eta^\beta_{il}$ ($\alpha=1$).
Put $\theta_{ij}=\log\tau_{ij}$, so $\tau_{ij}=e^{\theta_{ij}}$. Then $\log P_\theta(s)=\text{const}+\sum_t\log p^{(t)}_{s_ts_{t+1}}$, and

$$
\frac{\partial\log p^{(t)}_{s_ts_{t+1}}}{\partial\theta_{ab}}
=\mathbb 1[a=s_t,\ b=s_{t+1}]-\mathbb 1[a=s_t]\,\mathbb 1[b\in\mathcal N_t]\,p^{(t)}_{ab}.
$$

Summing over the construction steps:

$$
\frac{\partial\log P_\theta(s)}{\partial\theta_{ab}}=\underbrace{\mathbb 1[(a,b)\in s]}_{\text{edge used}}-\underbrace{\sum_t\mathbb 1[s_t=a]\,p^{(t)}_{ab}}_{\text{expected use of }(a,b)\text{ given the states visited}} .
$$

The stochastic gradient of $J(\theta)=\mathbb E[F(s)]$ from $m$ ants is
$\frac1m\sum_kF(s_k)\bigl(\mathbb 1[(a,b)\in s_k]-\sum_t\mathbb 1[s_{k,t}=a]p^{(t)}_{ab}\bigr)$. A gradient-ascent step in $\theta$
therefore **deposits** $\propto F(s_k)$ on the used edges and **subtracts** the quality-weighted expected usage. The
subtraction plays the role of evaporation, but it is targeted at the edges the ant could have chosen. With
symmetric pheromone ($\theta_{ab}=\theta_{ba}$) add the $(a,b)$ and $(b,a)$ terms. Meuleau & Dorigo (2002) derive
AS-like updates from exactly this expression.

### Exercise 2 (★) — CE with $a=1$ and a single elite sample

With one elite sample $s^{e}\in\lbrace0,1\rbrace^n$, the maximum-likelihood fit is $\hat p=s^{e}$. With $a=1$ the new
model is $p=s^{e}$, which puts probability $\prod_ip_i^{s_i}(1-p_i)^{1-s_i}=\mathbb 1[s=s^{e}]$ on a single point. All
subsequent samples equal $s^{e}$, the elite is again $s^{e}$, and the fit reproduces $p=s^{e}$. The model is an
absorbing, deterministic state after one iteration, and the search has stopped.

With $a\lt1$ and $p_0\in(0,1)^n$, the update $p_{t+1}=(1-a)p_t+a\hat p_t$ gives
$p_{t+1,i}\ge(1-a)p_{t,i}\ge(1-a)^{t+1}p_{0,i}\gt0$, and symmetrically $1-p_{t+1,i}\gt0$. Every solution keeps a positive
probability at every finite time. This is the CE analogue of $\tau_{\min}\gt0$, although the lower bound decays
geometrically, whereas MMAS keeps a *constant* lower bound. Smoothing is also a learning rate. It averages the noisy
elite fits over iterations, so one lucky sample cannot erase what the model has learned. Without smoothing, even a
larger elite can freeze a coordinate at 0 or 1 whenever all elite samples agree on it.

### Exercise 3 (★★) — Proof of $P^{\ast}(t)\ge1-(1-\hat p)^t$; numbers for the 10-city instance

Let $\mathcal F_{k}$ be the history up to iteration $k$, and let $A_k$ be the event that some ant builds $s^{\ast}$ in
iteration $k$. Every trail lies in $[\tau_{\min},\tau_{\max}]$ and $\eta$ lies in $[\eta_{\min},\eta_{\max}]$, so every
construction step picks the next component of $s^{\ast}$ with probability at least
$p_{\min}=\tau_{\min}\eta^\beta_{\min}/\bigl((n-2)\tau_{\max}\eta^\beta_{\max}+\tau_{\min}\eta^\beta_{\min}\bigr)$,
*whatever the history*. Whatever the start city, one ant follows the cycle $s^{\ast}$ in $n-1$ decisions, so it builds
$s^{\ast}$ with probability at least $p_{\min}^{n-1}$, and the $m$ ants construct independently given
$\mathcal F_{k-1}$. Hence $P(A_k\mid\mathcal F_{k-1})\ge\hat p:=1-(1-p_{\min}^{n-1})^m$ almost surely, and

$$
P\Bigl(\bigcap_{k\le t}A_k^c\Bigr)=\mathbb E\Bigl[\mathbb 1\Bigl[\bigcap_{k\le t-1}A_k^c\Bigr]\,P(A_t^c\mid\mathcal F_{t-1})\Bigr]\le(1-\hat p)\,P\Bigl(\bigcap_{k\le t-1}A_k^c\Bigr)\le\dots\le(1-\hat p)^t ,
$$

so $P^{\ast}(t)=1-P(\bigcap_{k\le t}A_k^c)\ge1-(1-\hat p)^t\to1$. $\square$

For the notebook's instance ($n=10$, $m=2$, $\rho=0.3$), with the extreme trail and heuristic values of Notebook 4,
$p_{\min}=4.69\times10^{-5}$ and $\hat p=2.19\times10^{-39}$. The bound exceeds $0.5$ once
$t\ge\ln0.5/\ln(1-\hat p)\approx\ln2/\hat p=3.2\times10^{38}$ iterations. In the notebook's 100 runs (the block
re-runs them with the same seed) the median iteration at which $s^{\ast}$ is first found is **14** (90% quantile 70,
maximum 184). The bound is 37 orders of magnitude pessimistic, because it assumes that the pheromone always works
*against* $s^{\ast}$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import random_euclidean_tsp, tour_length

def sample_next(w, rng):                             # row-wise roulette wheel (Notebook 3)
    cum = np.cumsum(w, axis=-1); u = rng.random(w.shape[:-1]) * cum[..., -1]
    return (cum <= u[..., None]).sum(-1)

def construct_tours(W, m, rng, cand=None):
    """m tours per run from weights W (runs, n, n); cand (n, n) bool = candidate lists (fallback: all unvisited).
    Also returns the number of weights examined per tour."""
    R, n, _ = W.shape; rr, mm = np.arange(R)[:, None], np.arange(m)[None, :]
    tours = np.empty((R, m, n), dtype=int); cur = rng.integers(0, n, (R, m)); tours[:, :, 0] = cur
    unvisited = np.ones((R, m, n), bool); unvisited[rr, mm, cur] = False; examined = 0
    for s in range(1, n):
        w = W[rr, cur] * unvisited
        dead = w.sum(-1) <= 0
        if dead.any(): w[dead] = unvisited[dead]          # numerical underflow guard
        if cand is not None:
            wc = w * cand[cur]; has = (wc > 0).any(-1)
            examined += np.where(has, cand[cur].sum(-1), unvisited.sum(-1)).mean()   # scan K candidates
            w = np.where(has[..., None], wc, w)
        else:
            examined += unvisited.sum(-1).mean()
        nxt = sample_next(w, rng); tours[:, :, s] = nxt; unvisited[rr, mm, nxt] = False; cur = nxt
    return tours, examined

def tour_lengths(tours, D):
    return D[tours, np.roll(tours, -1, axis=-1)].sum(-1)

def deposit(tau, tours, amount, symmetric=True):     # amount (runs, k) on the edges of tours (runs, k, n)
    a, b = tours, np.roll(tours, -1, axis=-1)
    ri = np.broadcast_to(np.arange(tours.shape[0])[:, None, None], a.shape); amt = np.broadcast_to(amount[..., None], a.shape)
    np.add.at(tau, (ri, a, b), amt)
    if symmetric: np.add.at(tau, (ri, b, a), amt)

def tau_bounds(L_bs, rho, n, p_best=0.05):           # MMAS bounds (Stuetzle & Hoos 2000)
    tmax = 1.0 / (rho * L_bs); pdec = p_best ** (1.0 / n)
    return tmax, tmax * (1 - pdec) / ((n / 2 - 1) * pdec)

def nn_length(D):                                    # nearest-neighbour tour from city 0
    n = len(D); t = [0]; left = set(range(1, n))
    while left:
        j = min(left, key=lambda c: D[t[-1], c]); t.append(j); left.remove(j)
    return tour_length(t, D)
from utils import held_karp

def aco_bs(D, runs, iters, m, rho, rng, p_best=0.05):
    """ACO_bs,tau_min (best-so-far deposit, MMAS bounds) as in Notebook 4, Section 4."""
    n = len(D); eta_b = np.where(np.eye(n, dtype=bool), 0.0, 1.0 / (D + np.eye(n))) ** 2
    tau = np.full((runs, n, n), 1.0 / (rho * nn_length(D))); rr = np.arange(runs)
    best, best_tour, hist = np.full(runs, np.inf), np.zeros((runs, n), int), []
    for it in range(iters):
        tours, _ = construct_tours(tau * eta_b, m, rng); L = tour_lengths(tours, D); ib = L.argmin(1)
        imp = L[rr, ib] < best - 1e-12; best[imp], best_tour[imp] = L[rr, ib][imp], tours[rr, ib][imp]
        tau *= 1 - rho; deposit(tau, best_tour[:, None], 1.0 / best[:, None])
        tmax, tmin = tau_bounds(best, rho, n, p_best); tau = np.clip(tau, tmin[:, None, None], tmax[:, None, None])
        hist.append(best.copy())
    return np.array(hist).T

_, D10 = random_euclidean_tsp(10, np.random.default_rng(21)); n, rho, m = 10, 0.3, 2      # Notebook 4 instance
opt10, _ = held_karp(D10)
# worst-case per-step probability of following s*, from the extreme trail and heuristic values (Notebook 4)
off = ~np.eye(n, dtype=bool); eta2 = (1 / (D10 + np.eye(n)))[off] ** 2
tmax_b = 1 / (rho * opt10); pd = 0.05 ** (1 / n)
tmin_b = 1 / (rho * D10[off].max() * n) * (1 - pd) / ((n / 2 - 1) * pd)
p_min = tmin_b * eta2.min() / ((n - 2) * tmax_b * eta2.max() + tmin_b * eta2.min())
p_hat = -np.expm1(m * np.log1p(-p_min ** (n - 1)))                  # 1 - (1 - p_min^(n-1))^m, computed stably
t_half = np.log(0.5) / np.log1p(-p_hat)
print(f"p_min = {p_min:.3e}, p_hat = {p_hat:.3e}; the bound exceeds 0.5 after t = {t_half:.2e} iterations")
hist = aco_bs(D10, 100, 300, m, rho, np.random.default_rng(22))
found = hist <= opt10 + 1e-9; first = np.where(found.any(1), found.argmax(1) + 1, np.inf)
print(f"runs that found s*: {found[:, -1].mean():.0%}; first-hit iteration median {np.median(first):.0f}, "
      f"90% quantile {np.percentile(first, 90):.0f}, max {first.max():.0f}")
```

### Exercise 4 (★★) — CE with a transition-matrix model vs MMAS on the TSP

Model: a row-stochastic matrix $P$. Tours are sampled city by city from $P$, renormalised over the unvisited cities
(optionally multiplied by $\eta^\beta$). The maximum-likelihood fit to the elite is the matrix of elite transition
frequencies, whose rows sum to 1 because every city has exactly one successor. Then smooth.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import random_euclidean_tsp, tour_length

def sample_next(w, rng):                             # row-wise roulette wheel (Notebook 3)
    cum = np.cumsum(w, axis=-1); u = rng.random(w.shape[:-1]) * cum[..., -1]
    return (cum <= u[..., None]).sum(-1)

def construct_tours(W, m, rng, cand=None):
    """m tours per run from weights W (runs, n, n); cand (n, n) bool = candidate lists (fallback: all unvisited).
    Also returns the number of weights examined per tour."""
    R, n, _ = W.shape; rr, mm = np.arange(R)[:, None], np.arange(m)[None, :]
    tours = np.empty((R, m, n), dtype=int); cur = rng.integers(0, n, (R, m)); tours[:, :, 0] = cur
    unvisited = np.ones((R, m, n), bool); unvisited[rr, mm, cur] = False; examined = 0
    for s in range(1, n):
        w = W[rr, cur] * unvisited
        dead = w.sum(-1) <= 0
        if dead.any(): w[dead] = unvisited[dead]          # numerical underflow guard
        if cand is not None:
            wc = w * cand[cur]; has = (wc > 0).any(-1)
            examined += np.where(has, cand[cur].sum(-1), unvisited.sum(-1)).mean()   # scan K candidates
            w = np.where(has[..., None], wc, w)
        else:
            examined += unvisited.sum(-1).mean()
        nxt = sample_next(w, rng); tours[:, :, s] = nxt; unvisited[rr, mm, nxt] = False; cur = nxt
    return tours, examined

def tour_lengths(tours, D):
    return D[tours, np.roll(tours, -1, axis=-1)].sum(-1)

def deposit(tau, tours, amount, symmetric=True):     # amount (runs, k) on the edges of tours (runs, k, n)
    a, b = tours, np.roll(tours, -1, axis=-1)
    ri = np.broadcast_to(np.arange(tours.shape[0])[:, None, None], a.shape); amt = np.broadcast_to(amount[..., None], a.shape)
    np.add.at(tau, (ri, a, b), amt)
    if symmetric: np.add.at(tau, (ri, b, a), amt)

def tau_bounds(L_bs, rho, n, p_best=0.05):           # MMAS bounds (Stuetzle & Hoos 2000)
    tmax = 1.0 / (rho * L_bs); pdec = p_best ** (1.0 / n)
    return tmax, tmax * (1 - pdec) / ((n / 2 - 1) * pdec)

def nn_length(D):                                    # nearest-neighbour tour from city 0
    n = len(D); t = [0]; left = set(range(1, n))
    while left:
        j = min(left, key=lambda c: D[t[-1], c]); t.append(j); left.remove(j)
    return tour_length(t, D)

def ce_tsp(D, runs, budget, rng, N=300, elite_frac=0.05, a=0.7, beta=0.0):
    """Cross-entropy method with a row-stochastic transition-matrix model (optionally times eta^beta)."""
    n = len(D); Ne = int(elite_frac * N)
    P = np.full((runs, n, n), 1.0 / (n - 1)); P[:, np.arange(n), np.arange(n)] = 0.0
    eta_b = np.where(np.eye(n, dtype=bool), 0.0, 1.0 / (D + np.eye(n))) ** beta
    rr = np.arange(runs)[:, None, None]; best = np.full(runs, np.inf)
    for it in range(budget // N):
        tours, _ = construct_tours(P * eta_b, N, rng)                         # sample, renormalised over unvisited
        L = tour_lengths(tours, D)
        E = np.take_along_axis(tours, np.argsort(L, 1)[:, :Ne, None], 1)       # elite tours
        Phat = np.zeros_like(P)                                               # ML fit: elite transition frequencies
        np.add.at(Phat, (np.broadcast_to(rr, E.shape), E, np.roll(E, -1, axis=-1)), 1.0)
        P = (1 - a) * P + a * Phat / Ne                                       # rows still sum to 1
        best = np.minimum(best, L.min(1))
    return best

def mmas_tsp(D, runs, iters, m, rho, rng):
    """MMAS, iteration-best deposit (Notebook 3/4)."""
    n = len(D); eta_b = np.where(np.eye(n, dtype=bool), 0.0, 1.0 / (D + np.eye(n))) ** 2
    tau = np.full((runs, n, n), 1.0 / (rho * nn_length(D))); best = np.full(runs, np.inf); rr = np.arange(runs)
    for it in range(iters):
        tours, _ = construct_tours(tau * eta_b, m, rng); L = tour_lengths(tours, D); ib = L.argmin(1)
        best = np.minimum(best, L[rr, ib]); tau *= 1 - rho; deposit(tau, tours[rr, ib][:, None], 1.0 / L[rr, ib][:, None])
        tmax, tmin = tau_bounds(best, rho, n); tau = np.clip(tau, tmin[:, None, None], tmax[:, None, None])
    return best

_, D30 = random_euclidean_tsp(30, np.random.default_rng(30))           # Notebook 4, Section 5 instance
res = {"MMAS (ib, rho=0.1, m=30)": mmas_tsp(D30, 8, 400, 30, 0.1, np.random.default_rng(1)),
       "CE, no heuristic": ce_tsp(D30, 8, 12000, np.random.default_rng(2)),
       "CE, eta^2 in the sampler": ce_tsp(D30, 8, 12000, np.random.default_rng(3), beta=2.0)}
L_best = min(v.min() for v in res.values())
for name, b in res.items():
    g = 100 * (b / L_best - 1)
    print(f"{name:26s} median {np.median(b):.4f}  gap {np.median(g):5.2f}%  IQR [{np.percentile(g, 25):.2f}, {np.percentile(g, 75):.2f}]")
v = list(res.values())
print(f"best length {L_best:.4f}; MWU p: MMAS vs CE {mannwhitneyu(v[0], v[1]).pvalue:.1e}, "
      f"MMAS vs CE+eta {mannwhitneyu(v[0], v[2]).pvalue:.2f}")
```

30 cities (the Section 5 instance), 12 000 tours, 8 runs. Gap to the best tour found by any method (4.5664):

| method | median length | median gap | IQR |
|---|---|---|---|
| MMAS (iteration-best, $\rho=0.1$, $m=30$) | 4.5760 | 0.21% | [0.00, 0.58] |
| CE, no heuristic ($N=300$, 5% elite, $a=0.7$) | 5.0318 | 10.19% | [7.54, 12.57] |
| CE, $\eta^2$ in the sampler | 4.5991 | 0.72% | [0.16, 1.09] |

MMAS beats plain CE clearly (p = 0.0009). Once CE is given the same heuristic, the difference is no longer
significant (p = 0.16). The update rules are close relatives (Section 2 of the notebook), and most of the gap comes
from the heuristic information, not from pheromone vs ML fitting. CE also collapses its model faster ($a=0.7$), which
costs it some final quality.

### Exercise 5 (★★) — ACO$_\mathbb{R}$ with variable-correlation handling

Socha & Dorigo (2008) let each ant sample in a coordinate system adapted to the archive. The ant picks its guide
$s_l$, draws archive members with probability proportional to their squared distance from $s_l$, and
orthonormalises the difference vectors (Gram–Schmidt, here via QR, padded with random directions when fewer than $d$
distinct members exist). It then expresses the archive in that basis, computes $\sigma$ per new coordinate, samples,
and rotates back. The block vectorises this over runs and ants (the draw without replacement uses the Gumbel-top-$k$
trick, which is equivalent to successive weighted sampling).

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import ellipsoid, shifted_rotated, random_rotation

def aco_r(f, d, lo, hi, budget, runs, rng, k=50, m=2, q=1e-4, xi=0.85, rotate=False):
    """ACO_R (Socha & Dorigo 2008), vectorised over runs. rotate=True adds the variable-correlation handling:
    every ant samples in an orthonormal basis built from archive difference vectors around its guide."""
    A = rng.uniform(lo, hi, (runs, k, d)); fA = f(A.reshape(-1, d)).reshape(runs, k)
    o = np.argsort(fA, 1); A, fA = np.take_along_axis(A, o[..., None], 1), np.take_along_axis(fA, o, 1)
    wts = np.exp(-np.arange(k)**2 / (2 * (q * k)**2)); cdf = np.cumsum(wts / wts.sum()); cdf[-1] = 1.0
    rr = np.arange(runs)[:, None]; evals = k
    while evals + m <= budget:
        l = np.searchsorted(cdf, rng.random((runs, m)), side="right"); G = A[rr, l]         # guides (runs, m, d)
        diff = A[:, None] - G[:, :, None]                                                    # (runs, m, k, d)
        if rotate:
            d2 = (diff**2).sum(-1)                                                           # (runs, m, k)
            # d archive members drawn without replacement with prob ~ d2 (Gumbel-top-k); zero-distance ones last
            keys = np.where(d2 > 0, np.log(np.where(d2 > 0, d2, 1.0)) + rng.gumbel(size=d2.shape), -np.inf)
            e = np.argsort(-keys, -1)[..., :d]
            B = np.take_along_axis(diff, e[..., None], 2).swapaxes(-1, -2)                  # (runs, m, d, d) columns
            ok = np.take_along_axis(keys, e, -1) > -np.inf                                   # pad with random directions
            B = np.where(ok[..., None, :], B, rng.standard_normal(B.shape))
            Q, _ = np.linalg.qr(B)                                                           # Gram-Schmidt
        else:
            Q = np.broadcast_to(np.eye(d), (runs, m, d, d))
        sigma = xi * np.abs(diff @ Q).sum(2) / (k - 1)                                       # spread along the new axes
        z = sigma * rng.standard_normal((runs, m, d))
        X = np.clip(G + (Q @ z[..., None])[..., 0], lo, hi)
        fX = f(X.reshape(-1, d)).reshape(runs, m); evals += m
        A2, f2 = np.concatenate([A, X], 1), np.concatenate([fA, fX], 1)
        o = np.argsort(f2, 1, kind="stable")[:, :k]; A, fA = A2[rr, o], f2[rr, o]
    return fA[:, 0]

d, RUNS, BUDGET = 10, 15, 10000
shift = np.random.default_rng(11).uniform(-2, 2, d); R = random_rotation(d, np.random.default_rng(12))
fS, fSR = shifted_rotated(ellipsoid, shift, np.eye(d)), shifted_rotated(ellipsoid, shift, R)
for k, rot in enumerate((False, True)):
    a = aco_r(fS, d, -5, 5, BUDGET, RUNS, np.random.default_rng(70 + k), rotate=rot)
    b = aco_r(fSR, d, -5, 5, BUDGET, RUNS, np.random.default_rng(70 + k), rotate=rot)
    print(f"{'with correlation handling' if rot else 'axis-aligned (notebook)':26s} S median {np.median(a):.2g}  "
          f"SR median {np.median(b):.3g}  ratio SR/S {np.median(b) / np.median(a):.2g}  MWU p {mannwhitneyu(a, b).pvalue:.1g}")
```

Shifted (S) vs shifted-rotated (SR) ellipsoid (condition $10^6$, shift and rotation of Notebook 1, Section 8),
$d=10$, 10 000 evaluations, 15 runs:

| variant | S median | SR median | ratio SR/S | MWU p |
|---|---|---|---|---|
| axis-aligned ACO$_\mathbb{R}$ (notebook) | 3.1e-27 | 951 | 3.1e+29 | 3e-06 |
| with correlation handling | 2.5 | 4.44 | 1.8 | 0.4 |

Correlation handling makes ACO$_\mathbb{R}$ (statistically) rotation invariant, and it improves the rotated case by
a factor of about 200. It pays with the loss of the huge separable advantage: the random basis ignores the fact that
the coordinate axes happened to be optimal. The basis is built from the archive's geometry, not from a learned
covariance, so it does not solve the ill-conditioned rotated ellipsoid either.

### Exercise 6 (★★★) — MMAS for the QAP, validated against brute force

$\tau_{ij}$ is the desirability of putting facility $i$ at location $j$. An ant assigns the facilities in its own
random order, each to a free location with probability $\propto\tau_{ij}$ (no heuristic). The iteration-best
assignment deposits $1/\text{cost}$. Bounds follow MMAS with $\tau_{\max}=1/(\rho\,C_{bs})$, and the trails are
initialised at $\tau_{\max}$ after the first iteration.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import random_qap, qap_cost, qap_brute_force

def qap_costs(perms, F, Dm):                          # vectorised qap_cost: facility i at location perm[i]
    return (F * Dm[perms[..., :, None], perms[..., None, :]]).sum((-1, -2))

def mmas_qap(F, Dm, budget, rng, m=8, rho=0.1, p_best=0.05):
    """MMAS for the QAP: tau[i, j] = desirability of facility i at location j, iteration-best deposit."""
    n = len(F); tau = np.ones((n, n)); best, evals = np.inf, 0
    while evals + m <= budget:
        perms = np.empty((m, n), int); free = np.ones((m, n), bool); am = np.arange(m)
        order = np.argsort(rng.random((m, n)), 1)       # every ant assigns the facilities in its own random order
        for s in range(n):
            i = order[:, s]; cum = np.cumsum(tau[i] * free, 1)
            j = (cum <= rng.random(m)[:, None] * cum[:, -1:]).sum(1)   # location with prob ~ tau[i, j] (free only)
            perms[am, i] = j; free[am, j] = False
        c = qap_costs(perms, F, Dm); evals += m; ib = c.argmin(); best = min(best, c[ib])
        tmax = 1 / (rho * best); pdec = p_best ** (1 / n); tmin = tmax * (1 - pdec) / ((n / 2 - 1) * pdec)
        if evals == m: tau[:] = tmax                   # initialise at tau_max after the first iteration
        tau *= 1 - rho; tau[np.arange(n), perms[ib]] += 1 / c[ib]; tau = np.clip(tau, tmin, tmax)
    return best

def random_search(F, Dm, budget, rng):
    return qap_costs(np.argsort(rng.random((budget, len(F))), 1), F, Dm).min()

opts, insts = [], []
for s in range(20):
    F, Dm = random_qap(8, np.random.default_rng(s)); opt, perm = qap_brute_force(F, Dm)
    assert abs(qap_costs(perm[None], F, Dm)[0] - qap_cost(perm, F, Dm)) < 1e-9   # vectorised cost = qap_cost
    opts.append(opt); insts.append((F, Dm))
opts = np.array(opts)
for budget, label, run in [(800, "MMAS rho=0.1", lambda F, Dm, g: mmas_qap(F, Dm, 800, g)),
                           (800, "random search", lambda F, Dm, g: random_search(F, Dm, 800, g)),
                           (2000, "MMAS rho=0.1", lambda F, Dm, g: mmas_qap(F, Dm, 2000, g)),
                           (2000, "MMAS rho=0.3", lambda F, Dm, g: mmas_qap(F, Dm, 2000, g, rho=0.3)),
                           (2000, "random search", lambda F, Dm, g: random_search(F, Dm, 2000, g))]:
    gaps = np.array([[run(F, Dm, np.random.default_rng(100 * s + r)) / opts[s] - 1 for r in range(5)]
                     for s, (F, Dm) in enumerate(insts)])
    assert gaps.min() > -1e-12                          # never below the brute-force optimum
    print(f"budget {budget:5d}  {label:14s} hit rate {np.mean(gaps < 1e-9):4.0%}   mean gap {gaps.mean():6.2%}"
          f"   max gap {gaps.max():6.2%}")
```

20 random instances with $n=8$ (`random_qap`, seeds 0–19), exact optimum by `qap_brute_force`, 5 runs per instance,
$m=8$ ants, and random search at the same budget as the control:

| budget | method | hit rate | mean gap | max gap |
|---|---|---|---|---|
| 800 (2% of $8!$) | MMAS, $\rho=0.1$ | 5% | 1.94% | 7.15% |
| 800 | random search | 1% | 2.42% | 6.12% |
| 2000 (5%) | MMAS, $\rho=0.1$ | 12% | 1.02% | 3.40% |
| 2000 | MMAS, $\rho=0.3$ | 21% | 0.77% | 3.34% |
| 2000 | random search | 2% | 1.66% | 4.23% |

No run is below the brute-force optimum, and the vectorised cost agrees with `qap_cost` on every optimal permutation
(both are asserted in the block). ACO learns: at 2000 evaluations it reaches the optimum 6–10 times more often than
random search, and a faster evaporation helps at this short budget. Without a heuristic or local search, however,
pure MMAS is a weak QAP solver. Practical MMAS-QAP (Stützle & Hoos 2000) applies 2-exchange local search to the ants'
assignments.

---

## Lab — Swarm Showdown

### Exercise 1 (★) — Rotation invariance of the (1+1)-ES

Let $g=f\circ R^\top$ with $R$ orthogonal. Run A works on $f$ from $x_0$, and run B works on $g$ from $y_0=Rx_0$, both
with $\sigma_0$ equal. Couple the runs through their mutations: if A uses $z_t\sim\mathcal N(0,\sigma_t^2I)$, let B use
$Rz_t$. This has the same law, because $\mathcal N(0,\sigma^2I)$ is invariant under orthogonal maps: the covariance
is $R\sigma^2IR^\top=\sigma^2I$.

*Induction.* Suppose $y_t=Rx_t$ and both runs have the same $\sigma_t$. B's offspring is $y_t+Rz_t=R(x_t+z_t)$ and
$g(R(x_t+z_t))=f(x_t+z_t)$, so its fitness equals A's offspring fitness, and so does the parent's:
$g(y_t)=f(x_t)$. The acceptance decision is therefore identical. B keeps $y_{t+1}=Ry'$ exactly when A keeps
$x_{t+1}=x'$, and the 1/5-rule update of $\sigma$ depends only on that success indicator, so $\sigma_{t+1}$ agrees. Hence
$(y_t,\sigma_t)_{t\ge0}$ equals $(Rx_t,\sigma_t)_{t\ge0}$ along the coupling, and in distribution for the uncoupled
algorithm. All $f$-values coincide, so the best-so-far traces have the same law. $\square$ (Clipping to an axis-aligned
box breaks the argument; that is why the box is ignored.)

### Exercise 2 (★) — Closed-form LOO residual via Sherman–Morrison

Write kernel ridge regression in feature space: $\hat f(x)=\phi(x)^\top w$ with $w=G^{-1}\Phi^\top y$,
$G=\Phi^\top\Phi+\lambda I$. The hat matrix is $H=\Phi G^{-1}\Phi^\top=K(K+\lambda I)^{-1}$ (push-through identity), and
$h_i=H_{ii}=\phi_i^\top G^{-1}\phi_i$. Leaving out point $i$ gives $G_{-i}=G-\phi_i\phi_i^\top$ and
$b_{-i}=\Phi^\top y-y_i\phi_i$. Sherman–Morrison gives

$$
G_{-i}^{-1}=G^{-1}+\frac{G^{-1}\phi_i\phi_i^\top G^{-1}}{1-h_i}\quad\Rightarrow\quad
\phi_i^\top G_{-i}^{-1}=\phi_i^\top G^{-1}\Bigl(1+\frac{h_i}{1-h_i}\Bigr)=\frac{\phi_i^\top G^{-1}}{1-h_i}.
$$

Hence $\hat f^{(-i)}(x_i)=\phi_i^\top G_{-i}^{-1}b_{-i}=\frac{\hat y_i-h_iy_i}{1-h_i}$ and

$$
e_i^{\text{LOO}}=y_i-\hat f^{(-i)}(x_i)=\frac{y_i-\hat y_i}{1-h_i}.
$$

Finally, with $A=K+\lambda I$: $I-H=(A-K)A^{-1}=\lambda A^{-1}$. This gives $y-\hat y=\lambda A^{-1}y=\lambda\alpha$ and
$1-h_i=\lambda(A^{-1})_{ii}$, so $e_i^{\text{LOO}}=\alpha_i/(A^{-1})_{ii}$. $\square$ For an infinite-dimensional feature
map (Gaussian kernel) the same algebra holds in the RKHS, or can be done with block inversion on $A$ directly. The
notebook checks the formula against 90 explicit refits.

### Exercise 3 (★★) — "Linear" PSO with scalar $r_1,r_2$

Update: $v\leftarrow wv+c_1r_1(p-x)+c_2r_2(l-x)$ with **scalar** $r_1,r_2\sim U(0,1)$ per particle and iteration.

*Invariance.* Let $Q$ be orthogonal and run the algorithm on $g=f\circ Q^\top$ from $\tilde x_0=Qx_0$, $\tilde v_0=Qv_0$.
If $\tilde x_t=Qx_t$, $\tilde v_t=Qv_t$ and $\tilde p=Qp$, $\tilde l=Ql$ (the memories agree because
$g(Qx)=f(x)$, so all comparisons coincide), then with the *same* scalars $r_1,r_2$

$$
\tilde v_{t+1}=Q\bigl(wv_t+c_1r_1(p-x_t)+c_2r_2(l-x_t)\bigr)=Qv_{t+1},\qquad\tilde x_{t+1}=Qx_{t+1}.
$$

By induction the whole run is the rotated image of the original (ignoring the box). With vector $r$'s the step fails
because $\mathrm{diag}(r)Q\ne Q\,\mathrm{diag}(r)$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import BENCHMARKS, shifted_rotated, random_rotation

def make_suite(d, names, seed):
    """Shifted (S) and shifted-rotated (SR) problems on the unit cube (lab, Section 3)."""
    g = np.random.default_rng(seed); R = random_rotation(d, g); suite = {}
    for nm in names:
        bm = BENCHMARKS[nm]
        s = g.uniform(-2, 2, d) if nm == "rosenbrock" else g.uniform(0.4 * bm.lower, 0.4 * bm.upper, d)
        for tag, rot in [("S", np.eye(d)), ("SR", R)]:
            f = shifted_rotated(bm.f, s, rot)
            suite[f"{nm} {tag}"] = (lambda U, f=f, lo=bm.lower, hi=bm.upper: f(lo + U * (hi - lo)))
    return suite

def stacked(fs):
    """P problems stacked along the run axis: the batch is split into P contiguous chunks."""
    return lambda U: np.concatenate([f(c) for f, c in zip(fs, np.split(np.atleast_2d(U), len(fs)))])

def pso_gbest(f, d, budget, runs, rng, n=40, w=0.7298, c=1.49618, linear=False):
    """gbest PSO on [0,1]^d with absorbing bounds (lab); linear=True uses one scalar r per particle."""
    X = rng.uniform(0, 1, (runs, n, d)); V = (rng.uniform(0, 1, (runs, n, d)) - X) / 2
    Pf = f(X.reshape(-1, d)).reshape(runs, n); P = X.copy(); ar = np.arange(runs)[:, None]
    for t in range(1, budget // n):
        L = P[ar, Pf.argmin(1)[:, None]]
        r = rng.random((2, runs, n, 1 if linear else d))
        V = w * V + c * r[0] * (P - X) + c * r[1] * (L - X)
        Xn = X + V; out = (Xn < 0) | (Xn > 1); X, V = np.clip(Xn, 0, 1), np.where(out, 0.0, V)
        F = f(X.reshape(-1, d)).reshape(runs, n); imp = F < Pf; P[imp], Pf[imp] = X[imp], F[imp]
    return Pf.min(1)

suite = make_suite(10, ["ellipsoid", "rosenbrock", "rastrigin", "ackley"], 10); R_ = 12
names, P = list(suite), len(suite)
res = {}
for k, lin in enumerate([False, True]):
    b = pso_gbest(stacked(list(suite.values())), 10, 10000, P * R_, np.random.default_rng(10000 + 7 * k), linear=lin)
    for p, nm in enumerate(names): res[lin, nm] = b[p * R_:(p + 1) * R_]
print(f"{'function':11s}{'standard: S / SR (ratio, p)':>42s}{'linear: S / SR (ratio, p)':>42s}")
for fn in ["ellipsoid", "rosenbrock", "rastrigin", "ackley"]:
    row = ""
    for lin in [False, True]:
        a, b = res[lin, f"{fn} S"], res[lin, f"{fn} SR"]
        row += f"{np.median(a):>12.2g} / {np.median(b):<8.2g} ({np.median(b) / np.median(a):.2g}, {mannwhitneyu(a, b).pvalue:.1g})"
    print(f"{fn:11s}{row}")
```

Suite of Section 3 ($d=10$, 10 000 evaluations, 12 runs, gbest, absorbing bounds on the unit cube; the standard
columns reproduce the lab's "PSO-constr gbest" results), median final error, ratio SR/S and MWU p:

| function | standard: S / SR (ratio, p) | linear: S / SR (ratio, p) |
|---|---|---|
| ellipsoid | 2.1e-07 / 1.3e+03 (6.3e+09, 4e-05) | 99 / 280 (2.8, 0.3) |
| rosenbrock | 4.7 / 7.5 (1.6, 0.1) | 31 / 9.5 (0.31, 0.9) |
| rastrigin | 5.8 / 19 (3.3, 0.002) | 16 / 18 (1.1, 0.5) |
| ackley | 1.5e-05 / 0.58 (4e+04, 0.03) | 3.4 / 2.6 (0.76, 0.2) |

Linear PSO is rotation invariant (no significant S/SR difference). On the rotated ellipsoid it beats the standard
update (280 vs 1300); on rotated Rastrigin the two are level (18 vs 19). **What is lost.** With a scalar $r$ the new
velocity lies in the span of $v$, $p-x$ and $l-x$. The swarm easily collapses onto a low-dimensional affine subspace
(a line-search-like behaviour) and loses the coordinate-wise diversity that makes standard PSO so strong on separable
problems. It performs much worse on the separable functions and on Ackley.

### Exercise 4 (★★) — IPOP-style restarts

The restart version is written per run (not vectorised), because restarts change the swarm size. It is compared with
the same code without restarts (tolerance 0, same seeds), on the $d=10$ suite with 10 000 evaluations and 12 runs per
problem, and the ECDF uses the lab's 51 targets and the evaluations/$d$ axis of the lab's ECDF figure.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import mannwhitneyu
from utils import BENCHMARKS, shifted_rotated, random_rotation

def make_suite(d, names, seed):
    """Shifted (S) and shifted-rotated (SR) problems on the unit cube (lab, Section 3)."""
    g = np.random.default_rng(seed); R = random_rotation(d, g); suite = {}
    for nm in names:
        bm = BENCHMARKS[nm]
        s = g.uniform(-2, 2, d) if nm == "rosenbrock" else g.uniform(0.4 * bm.lower, 0.4 * bm.upper, d)
        for tag, rot in [("S", np.eye(d)), ("SR", R)]:
            f = shifted_rotated(bm.f, s, rot)
            suite[f"{nm} {tag}"] = (lambda U, f=f, lo=bm.lower, hi=bm.upper: f(lo + U * (hi - lo)))
    return suite

def stacked(fs):
    """P problems stacked along the run axis: the batch is split into P contiguous chunks."""
    return lambda U: np.concatenate([f(c) for f, c in zip(fs, np.split(np.atleast_2d(U), len(fs)))])

def pso_ipop(f, d, budget, rng, n0=40, w=0.7298, c=1.49618, tol=1e-8):
    """gbest PSO on [0,1]^d with IPOP-style restarts (swarm radius < tol -> restart with a doubled swarm)."""
    evals, best, grid, trace, n, restarts = 0, np.inf, [], [], n0, -1
    while evals + n <= budget:
        restarts += 1
        X = rng.random((n, d)); V = (rng.random((n, d)) - X) / 2; Pf = f(X); P = X.copy(); evals += n
        best = min(best, Pf.min()); grid.append(evals); trace.append(best)
        while evals + n <= budget:
            g = P[Pf.argmin()]; r = rng.random((2, n, d))
            V = w * V + c * r[0] * (P - X) + c * r[1] * (g - X)
            Xn = X + V; out = (Xn < 0) | (Xn > 1); X, V = np.clip(Xn, 0, 1), np.where(out, 0.0, V)
            F = f(X); evals += n; imp = F < Pf; P[imp], Pf[imp] = X[imp], F[imp]
            best = min(best, Pf.min()); grid.append(evals); trace.append(best)
            if np.linalg.norm(X - X.mean(0), axis=1).mean() < tol: break
        n *= 2
    return np.array(grid), np.array(trace), restarts

targets = 10.0 ** np.linspace(2, -8, 51)                       # the lab's 51 targets
suite = make_suite(10, ["ellipsoid", "rosenbrock", "rastrigin", "ackley"], 10); B, R_ = 10000, 12
xs = np.logspace(0, np.log10(B / 10), 200)                     # evaluations / d, as in the lab's ECDF
for label, tol in [("plain gbest (no restart)", 0.0), ("IPOP, tol 1e-8", 1e-8), ("IPOP, tol 1e-3", 1e-3)]:
    rts, nres = [], []
    for p, f in enumerate(suite.values()):
        for r in range(R_):
            grid, tr, nr = pso_ipop(f, 10, B, np.random.default_rng(100 * p + r), tol=tol); nres.append(nr)
            hit = tr[:, None] < targets[None, :]
            rts.append(np.where(hit.any(0), grid[hit.argmax(0)], np.inf) / 10)
    rt = np.concatenate(rts); ecdf = (rt[None, :] <= xs[:, None]).mean(1)
    print(f"{label:26s} restarts/run {np.mean(nres):.2f}   solved at full budget {ecdf[-1]:.3f}   "
          f"mean ECDF over the log-budget grid {ecdf.mean():.3f}")
```

| variant | restarts per run | fraction of (problem, target, run) solved | mean ECDF over the log-budget grid |
|---|---|---|---|
| plain gbest (no restart) | – | 0.329 | 0.057 |
| IPOP, tol $10^{-8}$ | 0.00 | 0.329 | 0.057 |
| IPOP, tol $10^{-3}$ | 0.43 | 0.229 | 0.054 |

With the prescribed tolerance $10^{-8}$ **no restart is ever triggered** within $1000d$ evaluations: the swarm radius
(in unit-cube coordinates) does not get that small before the budget ends, so the run is identical to the plain
swarm. (The plain value 0.329 differs from the lab's 0.29 for PSO-constr gbest only through the random streams of a
per-run instead of a vectorised implementation.) With a looser tolerance restarts do happen, but they discard progress
on the unimodal problems and reduce the solved fraction. Restarts help when budgets are long compared with the time
to convergence, as in BBOB runs of $10^5d$ or more. The restart trigger must be matched to the budget.

### Exercise 5 (★★) — Space-filling designs instead of random search

Same protocol as the lab (kernel ridge regression, closed-form LOO-MSE, 60 evaluations, 20 runs); the PSO and
random-search rows use the lab's seeds and reproduce its numbers.

```python
import sys; sys.path.insert(0, "..")
import warnings
import numpy as np
from scipy.stats import mannwhitneyu, qmc

# ---- the lab's kernel-ridge HPO problem (Section 6): closed-form LOO-MSE on [0,1]^4
def make_data(n, rng):
    X = rng.uniform(-2, 2, (n, 6))
    return X, np.sin(2 * X[:, 0]) * np.cos(X[:, 1]) + 0.15 * X[:, 2]**2 - 0.1 * X[:, 3] + 0.3 * rng.standard_normal(n)
Xtr, ytr = make_data(90, np.random.default_rng(7))
DTR = np.stack([((Xtr[:, None, g] - Xtr[None, :, g])**2).sum(-1) for g in [slice(0, 2), slice(2, 4), slice(4, 6)]])
HP_LO, HP_HI = np.array([-6.0, -3, -3, -3]), np.array([1.0, 2, 2, 2])
def loo_mse(theta):
    lam, gam = 10.0 ** theta[0], 10.0 ** theta[1:]
    Ainv = np.linalg.inv(np.exp(-np.tensordot(gam, DTR, axes=1)) + lam * np.eye(len(ytr)))
    return float(np.mean(((Ainv @ ytr) / np.diag(Ainv))**2))
hp_objective = lambda U: np.array([loo_mse(HP_LO + u * (HP_HI - HP_LO)) for u in np.atleast_2d(U)])

def pso(f, d, budget, runs, rng, n=10, w=0.7298, c=1.49618):     # the lab's gbest PSO on [0,1]^d
    X = rng.uniform(0, 1, (runs, n, d)); V = (rng.uniform(0, 1, (runs, n, d)) - X) / 2
    Pf = f(X.reshape(-1, d)).reshape(runs, n); P = X.copy(); ar = np.arange(runs)[:, None]
    for t in range(1, budget // n):
        r = rng.random((2, runs, n, d)); V = w * V + c * r[0] * (P - X) + c * r[1] * (P[ar, Pf.argmin(1)[:, None]] - X)
        Xn = X + V; out = (Xn < 0) | (Xn > 1); X, V = np.clip(Xn, 0, 1), np.where(out, 0.0, V)
        F = f(X.reshape(-1, d)).reshape(runs, n); imp = F < Pf; P[imp], Pf[imp] = X[imp], F[imp]
    return Pf.min(1)

HB, HR = 60, 20
res = {"PSO (10 x 6)": pso(hp_objective, 4, HB, HR, np.random.default_rng(70)),
       "random search": hp_objective(np.random.default_rng(71).random((HR * HB, 4))).reshape(HR, HB).min(1),
       "Latin hypercube": np.array([hp_objective(qmc.LatinHypercube(d=4, seed=100 + r).random(HB)).min() for r in range(HR)])}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                  # 60 is not a power of 2 (SciPy's balance warning)
    res["scrambled Sobol"] = np.array([hp_objective(qmc.Sobol(d=4, scramble=True, seed=200 + r).random(HB)).min()
                                       for r in range(HR)])
for name, b in res.items():
    print(f"{name:16s} LOO-MSE median {np.median(b):.4f}  IQR [{np.percentile(b, 25):.4f}, {np.percentile(b, 75):.4f}]")
p = lambda a, b: mannwhitneyu(res[a], res[b]).pvalue
print(f"MWU p: LHS vs random {p('Latin hypercube', 'random search'):.2f}, Sobol vs random {p('scrambled Sobol', 'random search'):.2f}, "
      f"PSO vs LHS {p('PSO (10 x 6)', 'Latin hypercube'):.1g}, PSO vs Sobol {p('PSO (10 x 6)', 'scrambled Sobol'):.1g}")
```

| tuner | LOO-MSE median | IQR |
|---|---|---|
| PSO (10 × 6) | 0.1486 | [0.1306, 0.1550] |
| random search | 0.1661 | [0.1530, 0.1774] |
| Latin hypercube | 0.1639 | [0.1539, 0.1802] |
| scrambled Sobol (60 points) | 0.1731 | [0.1630, 0.1811] |

Space-filling designs do **not** close the gap. LHS vs random p = 0.76 and Sobol vs random p = 0.11, while PSO vs
LHS p = 0.003 and PSO vs Sobol p = 4e-05. In 4 dimensions with 60 points, better coverage barely changes the best
value found. The good region (large $\gamma_1$ and tiny $\gamma_3$ in the lab's selected hyperparameters) is a small
corner that only an adaptive method concentrates on. (60 is not a power of 2, so the Sobol sequence loses its balance
properties; SciPy warns about this, and the block silences the warning.)

### Exercise 6 (★★★) — Logistic regression HPO: PSO vs ACO$_\mathbb{R}$ vs random search, Friedman test

*Task.* 240 points with 6 standard Gaussian features in 3 groups. The true logit uses group 1 strongly (including an
interaction), group 2 weakly and group 3 not at all. The group-scaled features are expanded to all monomials of
degree 1 and 2 (6 + 21 = 27 features). The hyperparameters are
$\theta=(\log_{10}\lambda,\log_{10}s_1,\log_{10}s_2,\log_{10}s_3)\in[-4,3]\times[-2,1]^3$: the ridge strength and a
scale per feature group, which amounts to group-specific regularisation. The objective is the 5-fold cross-validated
log-loss. The model is L2-regularised logistic regression fitted by Newton's method (intercept not penalised),
validated against SciPy's BFGS on the penalised likelihood (max weight difference $4.4\times10^{-8}$). Budgets: 40
evaluations each (PSO: 8 particles × 5 iterations; ACO$_\mathbb{R}$: $k=10$, $m=2$, $q=0.3$, i.e. $10+15\times2$;
random search: 40 points), 20 seeds per method.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import minimize
from scipy.stats import friedmanchisquare, wilcoxon

# ---- synthetic task: 6 Gaussian features in 3 groups; group 1 strong (with interaction), group 2 weak, group 3 noise
g = np.random.default_rng(0); N = 240
Z = g.standard_normal((N, 6)); group = np.array([0, 0, 1, 1, 2, 2])
logit = 1.5 * Z[:, 0] - 1.0 * Z[:, 1] + 1.2 * Z[:, 0] * Z[:, 1] + 0.3 * Z[:, 2] - 0.2 * Z[:, 3]
y = (g.random(N) < 1 / (1 + np.exp(-logit))).astype(float)
pairs = list(combinations_with_replacement(range(6), 2))               # 21 quadratic monomials
def expand(Zs):
    return np.hstack([Zs, np.stack([Zs[:, i] * Zs[:, j] for i, j in pairs], 1)])   # 6 + 21 = 27 features
folds = np.array_split(g.permutation(N), 5)

def fit_logreg(X, y, lam, iters=12):
    """L2-regularised logistic regression by Newton's method (intercept not penalised)."""
    X1 = np.hstack([np.ones((len(X), 1)), X]); w = np.zeros(X1.shape[1])
    Reg = lam * np.eye(X1.shape[1]); Reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        w -= np.linalg.solve(X1.T @ (X1 * (p * (1 - p))[:, None]) + Reg, X1.T @ (p - y) + Reg @ w)
    return w

# validation of the Newton solver against SciPy's BFGS on the penalised negative log-likelihood
Xv = expand(Z); lam_v = 0.5
def nll(w):
    X1 = np.hstack([np.ones((N, 1)), Xv]); s = X1 @ w
    return np.sum(np.logaddexp(0, s) - y * s) + 0.5 * lam_v * w[1:] @ w[1:]
w_bfgs = minimize(nll, np.zeros(28), method="BFGS", options=dict(gtol=1e-9)).x
print(f"max |w_Newton - w_BFGS| = {np.abs(fit_logreg(Xv, y, lam_v) - w_bfgs).max():.1e}")

LO, HI = np.array([-4.0, -2, -2, -2]), np.array([3.0, 1, 1, 1])     # log10 lambda, log10 s_1..s_3
def cv_logloss(u):
    """5-fold CV log-loss at u in [0,1]^4; the group scales multiply the raw features before expansion."""
    th = LO + u * (HI - LO); lam, s = 10.0 ** th[0], 10.0 ** th[1:]; X = expand(Z * s[group]); ll = 0.0
    for k in range(5):
        te = folds[k]; tr = np.hstack([folds[j] for j in range(5) if j != k]); w = fit_logreg(X[tr], y[tr], lam)
        p = np.clip(1 / (1 + np.exp(-(w[0] + X[te] @ w[1:]))), 1e-12, 1 - 1e-12)
        ll -= np.sum(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
    return ll / N
F = lambda U: np.array([cv_logloss(u) for u in np.atleast_2d(U)])

BUDGET = 40
def pso(rng, n=8, w=0.7298, c=1.49618):             # 8 particles x 5 iterations = 40 evaluations
    X = rng.random((n, 4)); V = (rng.random((n, 4)) - X) / 2; Pf = F(X); P = X.copy()
    for t in range(1, BUDGET // n):
        r = rng.random((2, n, 4)); V = w * V + c * r[0] * (P - X) + c * r[1] * (P[Pf.argmin()] - X)
        Xn = X + V; out = (Xn < 0) | (Xn > 1); X, V = np.clip(Xn, 0, 1), np.where(out, 0.0, V)
        Fx = F(X); imp = Fx < Pf; P[imp], Pf[imp] = X[imp], Fx[imp]
    return Pf.min()
def aco_r(rng, k=10, m=2, q=0.3, xi=0.85):          # 10 + 15 x 2 = 40 evaluations
    A = rng.random((k, 4)); fA = F(A); o = np.argsort(fA); A, fA = A[o], fA[o]
    wts = np.exp(-np.arange(k)**2 / (2 * (q * k)**2)); p = wts / wts.sum(); ev = k
    while ev + m <= BUDGET:
        G = A[rng.choice(k, m, p=p)]; sigma = xi * np.abs(A[None] - G[:, None]).sum(1) / (k - 1)
        X = np.clip(G + sigma * rng.standard_normal((m, 4)), 0, 1); fX = F(X); ev += m
        A2, f2 = np.vstack([A, X]), np.concatenate([fA, fX]); o = np.argsort(f2, kind="stable")[:k]; A, fA = A2[o], f2[o]
    return fA[0]
def random_search(rng):
    return F(rng.random((BUDGET, 4))).min()

res = {name: np.array([alg(np.random.default_rng(1000 * i + s)) for s in range(20)])
       for i, (name, alg) in enumerate([("PSO", pso), ("ACO_R", aco_r), ("random search", random_search)])}
ranks = np.argsort(np.argsort(np.column_stack(list(res.values())), 1), 1) + 1
for i, (name, b) in enumerate(res.items()):
    print(f"{name:14s} median {np.median(b):.4f}  IQR [{np.percentile(b, 25):.4f}, {np.percentile(b, 75):.4f}]"
          f"  mean rank {ranks[:, i].mean():.2f}")
fr = friedmanchisquare(*res.values())
print(f"Friedman chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.1e}")
cmp = [("PSO", "random search"), ("ACO_R", "random search"), ("PSO", "ACO_R")]
pv = np.array([wilcoxon(res[a], res[b]).pvalue for a, b in cmp]); o = np.argsort(pv)
holm = np.empty(3); holm[o] = np.minimum(1, np.maximum.accumulate(pv[o] * (3 - np.arange(3))))   # Holm step-down
for (a, b), p_, h in zip(cmp, pv, holm):
    print(f"  Wilcoxon {a} vs {b}: p = {p_:.1e}, Holm-adjusted {h:.1e}")
```

CV log-loss of the best configuration:

| method | median | IQR | mean rank |
|---|---|---|---|
| PSO | 0.4698 | [0.4684, 0.4733] | 1.45 |
| ACO$_\mathbb{R}$ | 0.4724 | [0.4692, 0.4744] | 2.05 |
| random search | 0.4743 | [0.4736, 0.4773] | 2.50 |

Friedman $\chi^2=11.10$, p = 0.004. Post-hoc Wilcoxon signed-rank tests with Holm's step-down correction: PSO vs
random p = 0.0005, ACO$_\mathbb{R}$ vs random p = 0.021, PSO vs ACO$_\mathbb{R}$ p = 0.29.

*Conclusion.* The omnibus test rejects equality of the three tuners, and after the multiplicity correction both
swarm methods beat random search at this budget, while PSO and ACO$_\mathbb{R}$ cannot be separated. The effect is
small in absolute terms (about 0.005 in log-loss between PSO and random search), so whether it matters depends on the
application. The order of the analysis is the point of the exercise: the Friedman test comes before any pairwise
claim, and pairwise tests after it need a multiplicity correction (Holm).
