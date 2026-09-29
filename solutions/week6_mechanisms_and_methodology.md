# Solutions — Week 6: Unified Mechanisms, Parameter Control, Multi-Objective Search and Rigorous Benchmarking

> Try each exercise yourself before you read its solution. The algorithms get easier to see once you have fought with them.

Every ```python block below is **standalone**: run it from inside `week6_mechanisms_and_methodology/` (it adds `..` to
the path for `utils`). Each block defines, in compact form, the notebook classes it needs (`AskTell`, `optimize`, `SA`,
`TS`, `GA`, `PSO`, `ACOR`, `DE`, `CMAES`, `JADE`, …); these copies reproduce the notebooks' runs exactly. All numbers
quoted in the text are the ones the blocks print (shown in the ```text blocks). Randomness is seeded. Libraries: NumPy,
SciPy (statistics only), Matplotlib. Running times are between a few seconds and about two minutes per block.

---

## Notebook 1 — A unified view of metaheuristic mechanisms

### Exercise 1 (★) — the $(1+1)$-EA as an ask/tell plug-in

**Plug-in.** The memory is the current bit string $x$. The generation step flips each bit independently with
probability $1/n$ (standard bit mutation). Acceptance is elitist: the offspring replaces $x$ if $f(y)\le f(x)$.

**Row of the table.** It is a trajectory method with a single-state memory. It is the SA row with $T=0$, or equally
the $(1+1)$-ES row with a discrete mutation distribution. Because acceptance never keeps a worse state, the
worsening-acceptance rate is $\omega_t\equiv0$. It is elitist, like DE and ACO<sub>ℝ</sub>.

**Check.** Over 200 runs with $n=50$ the mean time to the optimum is $447\pm11$ evaluations. The precise expansion
$e\,n\ln n-1.8925\,n+\tfrac{e}{2}\ln n+O(1)$ (Hwang, Panholzer, Rolin, Tsai & Chen 2018) gives $442$; the leading term
$e\,n\ln n=532$ alone is too large at this $n$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import BudgetedObjective, BENCHMARKS, shifted_rotated
# --- ask-tell core of Notebook 1 (compact copy) ---
class AskTell:
    name = "base"
    def __init__(self, d, lo, hi, budget, rng):
        self.d, self.lo, self.hi, self.budget, self.rng = d, lo, hi, budget, rng
        self.span = (hi - lo) if lo is not None else 1.0
        self.n_told = 0
    def clip(self, X): return X if self.lo is None else np.clip(X, self.lo, self.hi)
    def uniform(self, n): return self.rng.uniform(self.lo, self.hi, (n, self.d))
    def progress(self): return min(1.0, self.n_told / self.budget)

def distinct_indices(rng, n, k, n_pool=None, exclude=None):
    n_pool = n if n_pool is None else n_pool
    keys = rng.random((n, n_pool)); keys[np.arange(n), np.arange(n)] = np.inf
    if exclude is not None: keys[np.arange(n), exclude] = np.inf
    return np.argsort(keys, axis=1)[:, :k]

def optimize(opt, f, budget, callback=None):
    obj = BudgetedObjective(f, budget)
    while obj.remaining > 0:
        X = opt.ask()
        if len(X) > obj.remaining: obj(X[: obj.remaining]); break
        y = obj(X); opt.tell(X, y); opt.n_told += len(y)
        if callback is not None: callback(opt, X, y, obj)
    return obj

class OnePlusOneEA(AskTell):
    """(1+1)-EA on bit strings: memory = one string, standard bit mutation (rate 1/n), elitist acceptance."""
    name = "(1+1)-EA"
    def __init__(self, n, budget, rng):
        super().__init__(n, None, None, budget, rng)
        self.x = rng.integers(0, 2, n); self.fx = None
    def ask(self):
        if self.fx is None: return self.x[None].astype(float)
        flip = self.rng.random(self.d) < 1.0 / self.d
        return np.where(flip, 1 - self.x, self.x)[None].astype(float)
    def tell(self, X, y):
        if self.fx is None or y[0] <= self.fx:
            self.x, self.fx = X[0].astype(int), y[0]

onemax = lambda X: -np.atleast_2d(X).sum(1)            # minimise -OneMax
n, runs, B = 50, 200, 3000
hits = []
for s in range(runs):
    ev, fb = optimize(OnePlusOneEA(n, B, np.random.default_rng(s)), onemax, B).trace()
    assert fb[-1] == -n, "budget too small"
    hits.append(ev[np.argmax(fb <= -n)])
print(f"OneMax n={n}: mean evaluations to the optimum {np.mean(hits):.0f} +- {np.std(hits, ddof=1) / np.sqrt(runs):.0f}")
print(f"e n ln n = {np.e * n * np.log(n):.0f};  e n ln n - 1.8925 n + (e/2) ln n = {np.e * n * np.log(n) - 1.8925 * n + np.e / 2 * np.log(n):.0f}")
```

```text
OneMax n=50: mean evaluations to the optimum 447 +- 11
e n ln n = 532;  e n ln n - 1.8925 n + (e/2) ln n = 442
```

### Exercise 2 (★★) — SBX preserves the midpoint; $\lvert c_1-c_2\rvert=\beta\lvert p_1-p_2\rvert$

SBX sets $c_1=\tfrac12[(1+\beta)p_1+(1-\beta)p_2]$ and $c_2=\tfrac12[(1-\beta)p_1+(1+\beta)p_2]$. Then

$$
c_1+c_2=\tfrac12\big[(1+\beta+1-\beta)p_1+(1-\beta+1+\beta)p_2\big]=p_1+p_2,
\qquad
c_1-c_2=\tfrac12\big[2\beta p_1-2\beta p_2\big]=\beta(p_1-p_2).
$$

So the midpoint is preserved exactly, and because $\beta\ge0$ the spread satisfies
$\lvert c_1-c_2\rvert=\beta\lvert p_1-p_2\rvert$.

**Consequence for diversity.** Crossover alone never moves the population mean: each pair keeps its mean. It rescales
each pair's spread by $\beta$. With $u\sim U(0,1)$, the contracting branch is $\beta=(2u)^{1/(\eta+1)}$ and the
expanding branch is $\beta=(2(1-u))^{-1/(\eta+1)}$. Each branch has probability $1/2$, and
$\mathbb{E}[v^{2/(\eta+1)}]=\frac{\eta+1}{\eta+3}$, $\mathbb{E}[v^{-2/(\eta+1)}]=\frac{\eta+1}{\eta-1}$ for $v\sim U(0,1)$ ($\eta\gt1$), so

$$
\mathbb{E}[\beta^2]=\tfrac12\cdot\frac{\eta+1}{\eta+3}+\tfrac12\cdot\frac{\eta+1}{\eta-1}=\frac{(\eta+1)^2}{(\eta+1)^2-4}\gt 1 .
$$

For $\eta=15$ this is $1.0159$; a Monte Carlo estimate gives $1.0160$. On average SBX therefore very slightly
*inflates* the within-pair variance, and its step length is always proportional to the current parent distance. Its
exploration shrinks exactly as fast as the population converges. This is the "self-adaptive" property of SBX
(Deb & Beyer 2001). SBX never creates diversity from nothing: when $p_1=p_2$, then $c_1=c_2=p_1$.

```python
import numpy as np
eta = 15.0
exact = (eta + 1) ** 2 / ((eta + 1) ** 2 - 4)
u = np.random.default_rng(0).random(2_000_000)
beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (2 * (1 - u)) ** (-1 / (eta + 1)))
print(f"E[beta^2]: formula {exact:.4f}, Monte Carlo {np.mean(beta**2):.4f} +- {np.std(beta**2) / np.sqrt(len(u)):.4f}")
p1, p2 = np.array([0.3, -1.0]), np.array([2.0, 0.5]); b = beta[:5, None]
c1, c2 = 0.5 * ((1 + b) * p1 + (1 - b) * p2), 0.5 * ((1 - b) * p1 + (1 + b) * p2)
print("midpoint preserved:", np.allclose(c1 + c2, p1 + p2), "| |c1-c2| = beta |p1-p2|:", np.allclose(np.abs(c1 - c2), b * np.abs(p1 - p2)))
```

```text
E[beta^2]: formula 1.0159, Monte Carlo 1.0160 +- 0.0001
midpoint preserved: True | |c1-c2| = beta |p1-p2|: True
```

### Exercise 3 (★★) — second-moment matrix for $c_1\ne c_2$

Take a stagnant particle ($p=g=0$) in one coordinate:
$x_{t+1}=(1+w-\varphi_t)x_t-w\,x_{t-1}$ with $\varphi_t=c_1r_{1}+c_2r_{2}$ and $r_{1},r_{2}\sim U(0,1)$
independent. Write $a_t=1+w-\varphi_t$. It is independent of $(x_t,x_{t-1})$, and

$$
\mathbb{E}[\varphi]=\frac{c_1+c_2}{2},\qquad \mathrm{Var}[\varphi]=\frac{c_1^2+c_2^2}{12},\qquad
\mathbb{E}[a]=1+w-\mathbb{E}[\varphi],\qquad \mathbb{E}[a^2]=\mathbb{E}[a]^2+\mathrm{Var}[\varphi].
$$

Square the recursion, $x_{t+1}^2=a_t^2x_t^2-2wa_tx_tx_{t-1}+w^2x_{t-1}^2$, and multiply it by $x_t$,
$x_{t+1}x_t=a_tx_t^2-wx_tx_{t-1}$; then take expectations using independence. The vector
$s_t=(\mathbb{E}x_t^2,\ \mathbb{E}x_tx_{t-1},\ \mathbb{E}x_{t-1}^2)^{\top}$ satisfies $s_{t+1}=Ms_t$ with

$$
M=\begin{pmatrix}\mathbb{E}[a^2] & -2w\,\mathbb{E}[a] & w^2\\ \mathbb{E}[a] & -w & 0\\ 1 & 0 & 0\end{pmatrix}.
$$

The entries involve $c_1,c_2$ only through $\mathbb{E}[a]$ and $\mathbb{E}[a^2]$, that is, through $\mathbb{E}[\varphi]$
and $\mathrm{Var}[\varphi]$. Order-2 stability ($\rho(M)\lt1$) therefore depends only on these two moments. For
$c_1=c_2=c$ you get back the notebook's $\mathbb{E}\varphi=c$ and $\mathrm{Var}\varphi=c^2/6$. If $p\ne g$, the
recursion gains inhomogeneous forcing terms (built from $p$, $g$ and the first moments). They move the fixed point but
leave the homogeneous part $M$, and hence the stability region, unchanged.

### Exercise 4 (★★) — SA with $T\to0$ and TS with $k=1$

**SA.** When $\Delta\le0$ the Metropolis rule always accepts. When $\Delta\gt0$ it accepts with probability
$e^{-\Delta/T}\to0$. In the limit, SA accepts exactly when $f(y)\le f(x)$, which is the elitist $(1+1)$-ES rule. Its
proposal $x+\sigma_t\mathcal{N}(0,I)$ keeps the plug-in's deterministic schedule
$\sigma_t=\text{span}\cdot0.1\,(10^{-4})^{t/B}/\sqrt d$. So SA with $T\to0$ is a $(1+1)$-ES with a scheduled step size
instead of a success rule.

**TS.** With $k=1$ there is one neighbour. If it is admissible it is the best admissible one. If it is not, the
fallback takes it anyway. The move $x\leftarrow\text{clip}(x+\sigma_t z)$ therefore happens at every step whatever $L$
is: a Gaussian random walk with a shrinking step and no selection at all.

**Numerical check (common random numbers).** The hand-written loops consume the random numbers in the same order as
the plug-ins (SA still draws its Metropolis uniform after a worsening proposal, although it can never accept at
$T_0=10^{-300}$).

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import BudgetedObjective, BENCHMARKS, shifted_rotated
# --- ask-tell core of Notebook 1 (compact copy) ---
class AskTell:
    name = "base"
    def __init__(self, d, lo, hi, budget, rng):
        self.d, self.lo, self.hi, self.budget, self.rng = d, lo, hi, budget, rng
        self.span = (hi - lo) if lo is not None else 1.0
        self.n_told = 0
    def clip(self, X): return X if self.lo is None else np.clip(X, self.lo, self.hi)
    def uniform(self, n): return self.rng.uniform(self.lo, self.hi, (n, self.d))
    def progress(self): return min(1.0, self.n_told / self.budget)

def distinct_indices(rng, n, k, n_pool=None, exclude=None):
    n_pool = n if n_pool is None else n_pool
    keys = rng.random((n, n_pool)); keys[np.arange(n), np.arange(n)] = np.inf
    if exclude is not None: keys[np.arange(n), exclude] = np.inf
    return np.argsort(keys, axis=1)[:, :k]

def optimize(opt, f, budget, callback=None):
    obj = BudgetedObjective(f, budget)
    while obj.remaining > 0:
        X = opt.ask()
        if len(X) > obj.remaining: obj(X[: obj.remaining]); break
        y = obj(X); opt.tell(X, y); opt.n_told += len(y)
        if callback is not None: callback(opt, X, y, obj)
    return obj
class SA(AskTell):
    name = "SA"
    def __init__(self, d, lo, hi, budget, rng, step0=0.1, step_end=1e-5, T_ratio=1e-8, n_cal=50, T0=None):
        super().__init__(d, lo, hi, budget, rng)
        self.x = self.uniform(1)[0] if lo is not None else rng.standard_normal(d)
        self.fx, self.step0, self.step_end, self.T_ratio, self.n_cal, self.T0, self.cal = None, step0, step_end, T_ratio, n_cal, T0, []
    def sched(self):
        p = self.progress()
        return (self.span * self.step0 * (self.step_end / self.step0) ** p / np.sqrt(self.d),
                self.T0 * self.T_ratio ** p if self.T0 is not None else np.inf)
    def ask(self):
        if self.fx is None: return self.x[None]
        return self.clip(self.x + self.sched()[0] * self.rng.standard_normal(self.d))[None]
    def tell(self, X, y):
        if self.fx is None: self.fx = y[0]; return
        delta = y[0] - self.fx
        if self.T0 is None:                                   # calibration random walk
            if delta > 0: self.cal.append(delta)
            self.x, self.fx = X[0], y[0]
            if len(self.cal) >= self.n_cal: self.T0 = np.mean(self.cal) / np.log(2.0)
            return
        if delta <= 0 or self.rng.random() < np.exp(-delta / self.sched()[1]): self.x, self.fx = X[0], y[0]
class TS(AskTell):
    name = "TS"
    def __init__(self, d, lo, hi, budget, rng, k=8, L=20, step0=0.1, step_end=1e-5, rho_factor=0.5):
        super().__init__(d, lo, hi, budget, rng)
        self.x = self.uniform(1)[0] if lo is not None else rng.standard_normal(d)
        self.fx, self.k, self.L, self.step0, self.step_end, self.rho_factor = None, k, L, step0, step_end, rho_factor
        self.tabu, self.best = [], np.inf
    def sigma(self): return self.span * self.step0 * (self.step_end / self.step0) ** self.progress() / np.sqrt(self.d)
    def ask(self):
        if self.fx is None: return self.x[None]
        return self.clip(self.x + self.sigma() * self.rng.standard_normal((self.k, self.d)))
    def tell(self, X, y):
        if self.fx is None: self.fx = self.best = y[0]; self.tabu.append(self.x.copy()); return
        rho = self.rho_factor * self.sigma() * np.sqrt(self.d)
        dist = np.sqrt(((X[:, None, :] - np.array(self.tabu)[None]) ** 2).sum(-1)).min(1)
        adm = (dist >= rho) | (y < self.best)
        if not adm.any(): adm[:] = True                        # all tabu, none aspirated: take the best anyway
        idx = np.flatnonzero(adm); j = idx[np.argmin(y[idx])]
        self.x, self.fx = X[j].copy(), y[j]; self.best = min(self.best, y[j])
        self.tabu = (self.tabu + [self.x.copy()])[-self.L:]

f = BENCHMARKS["rastrigin"].f; d, lo, hi, B = 10, -5.12, 5.12, 3000
sched = lambda told: (hi - lo) * 0.1 * (1e-5 / 0.1) ** min(1.0, told / B) / np.sqrt(d)   # the plug-ins' step schedule

# SA with T0 = 1e-300 (given, so no calibration walk) vs a hand-written elitist (1+1)-ES, same random numbers
x_sa = []; optimize(SA(d, lo, hi, B, np.random.default_rng(5), T0=1e-300), f, B, callback=lambda o, X, y, ob: x_sa.append(o.x.copy()))
r = np.random.default_rng(5); x = r.uniform(lo, hi, (1, d))[0]; fx = f(x); x_es = [x.copy()]; told = 1
while told < B:
    y = np.clip(x + sched(told) * r.standard_normal(d), lo, hi); fy = f(y); told += 1
    if fy > fx: r.random()          # SA still draws its (useless) Metropolis uniform
    else: x, fx = y, fy
    x_es.append(x.copy())
print("SA(T -> 0) == elitist (1+1)-ES:", np.array_equal(np.array(x_sa), np.array(x_es)))

# TS with k = 1 vs a clipped Gaussian random walk, for two tabu tenures
for L in [1, 20]:
    xs = []; optimize(TS(d, lo, hi, B, np.random.default_rng(9), k=1, L=L), f, B, callback=lambda o, X, y, ob: xs.append(o.x.copy()))
    r = np.random.default_rng(9); x = r.uniform(lo, hi, (1, d))[0]; walk = [x.copy()]; told = 1
    while told < B:
        x = np.clip(x + sched(told) * r.standard_normal((1, d)), lo, hi)[0]; told += 1; walk.append(x.copy())
    print(f"TS(k=1, L={L}) == Gaussian random walk:", np.array_equal(np.array(xs), np.array(walk)))
```

```text
SA(T -> 0) == elitist (1+1)-ES: True
TS(k=1, L=1) == Gaussian random walk: True
TS(k=1, L=20) == Gaussian random walk: True
```

### Exercise 5 (★★★) — the grey wolf optimiser, rewritten and tested for centre bias

**Mechanisms** (Mirjalili, Mirjalili & Lewis 2014). "Wolves" form a population $X$. "Alpha, beta, delta" is an elitist
archive of the three best points found so far. Each new point is

$$
x^{\text{new}}=\frac13\sum_{l\in\lbrace\alpha,\beta,\delta\rbrace}\Big(x_l-A_l\odot\lvert C_l\odot x_l-x\rvert\Big),\qquad
A_l\sim U(-a,a)^d,\ C_l\sim U(0,2)^d,
$$

where $a$ decreases linearly from 2 to 0, a deterministic parameter-control schedule. The population is replaced
unconditionally (non-elitist), while the archive is updated elitistically. In the vocabulary of §1:

- memory: the population plus an archive of the 3 best;
- generation: a product of per-coordinate distributions centred at the three archive points;
- acceptance: plus-selection for the archive, comma-selection for the population.

As $a\to0$ every candidate collapses onto the centroid of the three leaders (checked in the block).

**Where the centre bias comes from.** The perturbation scale is $\lvert C\odot x_l-x\rvert$, not
$\lvert x_l-x\rvert$. Once the pack has converged ($x\approx x_l$), this becomes $\lvert(C-1)\odot x_l\rvert$, which
is proportional to the *absolute coordinates* of the leader. Coordinates near $0$ are therefore sampled with tiny
steps and coordinates far from $0$ with large ones. The origin is a privileged point.

**Experiment** (same protocol as §5: $d=10$, 5000 evaluations, 8 seeds, medians; $p$ from a Mann–Whitney test of origin
vs shifted):

| function | GWO origin → shifted | PSO origin → shifted | DE origin → shifted |
|---|---|---|---|
| Rastrigin | 3.7 → 11 ($p=0.007$) | 7.5 → 9.0 ($p=0.51$) | 39 → 41 ($p=0.65$) |
| Ackley | $8\cdot10^{-14}$ → 2.3 ($p=9\cdot10^{-4}$) | $9.3\cdot10^{-3}$ → $7.2\cdot10^{-3}$ ($p=0.72$) | 1.0 → 0.94 ($p=0.44$) |
| Griewank | 0.068 → 1.0 ($p=1.6\cdot10^{-4}$) | 0.12 → 0.14 ($p=0.72$) | 0.74 → 0.80 ($p=0.28$) |

GWO is the best of the three on every function when the optimum sits at the origin. After the shift it loses that
advantage: on Ackley it falls more than two orders of magnitude behind PSO, on Griewank almost one, and on Rastrigin it
drops behind PSO. Its origin-vs-shift change is significant on all three functions; PSO's and DE's changes are
significant on none. This is the qualitative finding of Camacho-Villalón, Dorigo & Stützle.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import BudgetedObjective, BENCHMARKS, shifted_rotated
# --- ask-tell core of Notebook 1 (compact copy) ---
class AskTell:
    name = "base"
    def __init__(self, d, lo, hi, budget, rng):
        self.d, self.lo, self.hi, self.budget, self.rng = d, lo, hi, budget, rng
        self.span = (hi - lo) if lo is not None else 1.0
        self.n_told = 0
    def clip(self, X): return X if self.lo is None else np.clip(X, self.lo, self.hi)
    def uniform(self, n): return self.rng.uniform(self.lo, self.hi, (n, self.d))
    def progress(self): return min(1.0, self.n_told / self.budget)

def distinct_indices(rng, n, k, n_pool=None, exclude=None):
    n_pool = n if n_pool is None else n_pool
    keys = rng.random((n, n_pool)); keys[np.arange(n), np.arange(n)] = np.inf
    if exclude is not None: keys[np.arange(n), exclude] = np.inf
    return np.argsort(keys, axis=1)[:, :k]

def optimize(opt, f, budget, callback=None):
    obj = BudgetedObjective(f, budget)
    while obj.remaining > 0:
        X = opt.ask()
        if len(X) > obj.remaining: obj(X[: obj.remaining]); break
        y = obj(X); opt.tell(X, y); opt.n_told += len(y)
        if callback is not None: callback(opt, X, y, obj)
    return obj
class PSO(AskTell):
    name = "PSO"
    def __init__(self, d, lo, hi, budget, rng, n=40, w=0.7298, c1=1.49618, c2=1.49618, vmax=0.2):
        super().__init__(d, lo, hi, budget, rng)
        self.n, self.w, self.c1, self.c2, self.vmax = n, w, c1, c2, vmax * self.span
        self.X = self.uniform(n); self.V = rng.uniform(-1, 1, (n, d)) * 0.1 * self.span; self.Pb = self.fPb = None
    def ask(self):
        if self.Pb is None: return self.X
        g = self.Pb[np.argmin(self.fPb)]; r1, r2 = self.rng.random((2, self.n, self.d))
        self.V = np.clip(self.w * self.V + self.c1 * r1 * (self.Pb - self.X) + self.c2 * r2 * (g - self.X), -self.vmax, self.vmax)
        self.X = self.clip(self.X + self.V); return self.X
    def tell(self, X, y):
        if self.Pb is None: self.Pb, self.fPb = X.copy(), y.copy(); return
        imp = y < self.fPb; self.Pb[imp], self.fPb[imp] = X[imp], y[imp]
class DE(AskTell):
    name = "DE"
    def __init__(self, d, lo, hi, budget, rng, n=50, F=0.5, CR=0.9):
        super().__init__(d, lo, hi, budget, rng); self.n, self.F, self.CR, self.P, self.fP = n, F, CR, None, None
    def ask(self):
        if self.P is None: return self.uniform(self.n)
        n, d = self.n, self.d; idx = distinct_indices(self.rng, n, 3)
        V = self.P[idx[:, 0]] + self.F * (self.P[idx[:, 1]] - self.P[idx[:, 2]])
        mask = self.rng.random((n, d)) < self.CR; mask[np.arange(n), self.rng.integers(0, d, n)] = True
        return self.clip(np.where(mask, V, self.P))
    def tell(self, X, y):
        if self.P is None: self.P, self.fP = X.copy(), y.copy(); return
        imp = y <= self.fP; self.P[imp], self.fP[imp] = X[imp], y[imp]

class GWO(AskTell):
    """Grey wolf optimiser in mechanism form: population X + elitist archive of the 3 best points."""
    name = "GWO"
    def __init__(self, d, lo, hi, budget, rng, n=20):
        super().__init__(d, lo, hi, budget, rng); self.n, self.X, self.L, self.fL = n, None, None, None
    def ask(self):
        if self.X is None: return self.uniform(self.n)
        a, r, new = 2 * (1 - self.progress()), self.rng, 0.0
        for l in range(3):
            A = a * (2 * r.random(self.X.shape) - 1); C = 2 * r.random(self.X.shape)
            new = new + self.L[l] - A * np.abs(C * self.L[l] - self.X)
        return self.clip(new / 3)
    def tell(self, X, y):
        self.X = X.copy()
        P = X if self.L is None else np.vstack([self.L, X]); fP = y if self.L is None else np.concatenate([self.fL, y])
        o = np.argsort(fP, kind="stable")[:3]; self.L, self.fL = P[o], fP[o]

# a -> 0: every candidate is the centroid of the three leaders
g = GWO(2, -1, 1, 100, np.random.default_rng(0)); g.X = np.array([[0.3, -0.2]]); g.L = np.array([[0.5, 0.5], [0.1, 0.3], [-0.2, 0.4]])
g.n_told = 100
print("a = 0: candidate", g.ask()[0], "= centroid of leaders", g.L.mean(0))

# centre-bias test, same protocol as Notebook 1 section 5
B_c, seeds = 5000, 8
for fn in ["rastrigin", "ackley", "griewank"]:
    b = BENCHMARKS[fn]; o = np.random.default_rng(3).uniform(0.4 * b.lower, 0.4 * b.upper, 10)
    row = []
    for A in [GWO, PSO, DE]:
        res = {tag: np.array([optimize(A(10, b.lower, b.upper, B_c, np.random.default_rng(s)), ff, B_c).best_f for s in range(seeds)])
               for tag, ff in [("origin", b.f), ("shifted", shifted_rotated(b.f, o, np.eye(10)))]}
        p = stats.mannwhitneyu(res["origin"], res["shifted"]).pvalue
        row.append(f"{A.name}: {np.median(res['origin']):.2g} -> {np.median(res['shifted']):.2g} (p={p:.2g})")
    print(f"{fn:9s}", " | ".join(row))
```

```text
a = 0: candidate [0.13333333 0.4       ] = centroid of leaders [0.13333333 0.4       ]
rastrigin GWO: 3.7 -> 11 (p=0.007) | PSO: 7.5 -> 9 (p=0.51) | DE: 39 -> 41 (p=0.65)
ackley    GWO: 8.2e-14 -> 2.3 (p=0.00092) | PSO: 0.0093 -> 0.0072 (p=0.72) | DE: 1 -> 0.94 (p=0.44)
griewank  GWO: 0.068 -> 1 (p=0.00016) | PSO: 0.12 -> 0.14 (p=0.72) | DE: 0.74 -> 0.8 (p=0.28)
```

### Exercise 6 (★★★) — translation-invariance test

**Test.** Run algorithm $\mathcal{A}$ on $f$ over the box $[\ell,u]$, and on $g(x)=f(x-o)$ over $[\ell+o,u+o]$. Use the
same seed, so both runs draw identical random numbers. Compare the best-so-far traces.

**Which plug-ins should be invariant.** A plug-in is translation-equivariant if every candidate it produces is
$x_0$, or a point of the memory plus a combination of *differences* of memory points and random vectors, and if every
decision it makes depends only on $f$-values or on distances. SA, TS, GA (SBX and polynomial mutation are affine and
translation-equivariant), PSO, ACO<sub>ℝ</sub>, DE and CMA-ES all qualify in exact arithmetic. The uniform
initialisation over the shifted box is the shifted initialisation. OPO is the counter-example: $r\odot x$ is not
equivariant.

**Numerical result** (Rastrigin, $d=10$, 5000 evaluations; "agree" means relative difference $\le10^{-6}$):

| plug-in | traces agree for | final best $f$ (original / shifted) |
|---|---|---|
| SA, TS, GA, PSO, ACO<sub>ℝ</sub>, DE | all 5000 evaluations | identical |
| CMA-ES (as implemented) | 20 evaluations only | 6.965 / 7.960 |
| CMA-ES sampling with the symmetric root $BDB^{\top}$ | all 5000 evaluations | identical |

The CMA-ES result is instructive. The algorithm is invariant *in distribution*, but it samples
$y=BDz$ with eigenvectors $B$ from `eigh`. The sign (and, for equal eigenvalues, the basis) of each eigenvector is
arbitrary. Two covariance matrices that differ by rounding errors can therefore get different $B$, so identical $z$
map to different $y$, and the paths separate after the first update. The symmetric root $C^{1/2}=BDB^{\top}$ is unique.
With it, the sample is a function of $C$ alone and the paths coincide (`CMAESsym` in the block).

**Why box constraints break invariance.** Suppose the box is *not* moved with $o$. The operator
$\text{clip}(\cdot,\ell,u)$ is then applied at different positions relative to the landscape. A sample that is clipped
in one run is left alone in the other, and from the moment such a sample is accepted the runs follow different paths.
Every plug-in clips, and GA clips inside `sbx_pm`, so none of them is translation invariant on a fixed box once the
search reaches the boundary. The block tests this with SA and TS started inside the box and the optimum shifted by
$o/4$. Both searches reach the boundary early: the current states of the two SA runs separate after 26 evaluations
(during the calibration random walk), those of the two TS runs after 56, and TS's best-so-far traces differ from
evaluation 126 on. SA's best-so-far traces happen to coincide for all 5000 evaluations only because the best point of
both runs was found before the separation and neither run ever improved on it: equal best-so-far traces do not prove
equal trajectories.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import BudgetedObjective, BENCHMARKS, shifted_rotated
# --- ask-tell core of Notebook 1 (compact copy) ---
class AskTell:
    name = "base"
    def __init__(self, d, lo, hi, budget, rng):
        self.d, self.lo, self.hi, self.budget, self.rng = d, lo, hi, budget, rng
        self.span = (hi - lo) if lo is not None else 1.0
        self.n_told = 0
    def clip(self, X): return X if self.lo is None else np.clip(X, self.lo, self.hi)
    def uniform(self, n): return self.rng.uniform(self.lo, self.hi, (n, self.d))
    def progress(self): return min(1.0, self.n_told / self.budget)

def distinct_indices(rng, n, k, n_pool=None, exclude=None):
    n_pool = n if n_pool is None else n_pool
    keys = rng.random((n, n_pool)); keys[np.arange(n), np.arange(n)] = np.inf
    if exclude is not None: keys[np.arange(n), exclude] = np.inf
    return np.argsort(keys, axis=1)[:, :k]

def optimize(opt, f, budget, callback=None):
    obj = BudgetedObjective(f, budget)
    while obj.remaining > 0:
        X = opt.ask()
        if len(X) > obj.remaining: obj(X[: obj.remaining]); break
        y = obj(X); opt.tell(X, y); opt.n_told += len(y)
        if callback is not None: callback(opt, X, y, obj)
    return obj
class SA(AskTell):
    name = "SA"
    def __init__(self, d, lo, hi, budget, rng, step0=0.1, step_end=1e-5, T_ratio=1e-8, n_cal=50, T0=None):
        super().__init__(d, lo, hi, budget, rng)
        self.x = self.uniform(1)[0] if lo is not None else rng.standard_normal(d)
        self.fx, self.step0, self.step_end, self.T_ratio, self.n_cal, self.T0, self.cal = None, step0, step_end, T_ratio, n_cal, T0, []
    def sched(self):
        p = self.progress()
        return (self.span * self.step0 * (self.step_end / self.step0) ** p / np.sqrt(self.d),
                self.T0 * self.T_ratio ** p if self.T0 is not None else np.inf)
    def ask(self):
        if self.fx is None: return self.x[None]
        return self.clip(self.x + self.sched()[0] * self.rng.standard_normal(self.d))[None]
    def tell(self, X, y):
        if self.fx is None: self.fx = y[0]; return
        delta = y[0] - self.fx
        if self.T0 is None:                                   # calibration random walk
            if delta > 0: self.cal.append(delta)
            self.x, self.fx = X[0], y[0]
            if len(self.cal) >= self.n_cal: self.T0 = np.mean(self.cal) / np.log(2.0)
            return
        if delta <= 0 or self.rng.random() < np.exp(-delta / self.sched()[1]): self.x, self.fx = X[0], y[0]
class TS(AskTell):
    name = "TS"
    def __init__(self, d, lo, hi, budget, rng, k=8, L=20, step0=0.1, step_end=1e-5, rho_factor=0.5):
        super().__init__(d, lo, hi, budget, rng)
        self.x = self.uniform(1)[0] if lo is not None else rng.standard_normal(d)
        self.fx, self.k, self.L, self.step0, self.step_end, self.rho_factor = None, k, L, step0, step_end, rho_factor
        self.tabu, self.best = [], np.inf
    def sigma(self): return self.span * self.step0 * (self.step_end / self.step0) ** self.progress() / np.sqrt(self.d)
    def ask(self):
        if self.fx is None: return self.x[None]
        return self.clip(self.x + self.sigma() * self.rng.standard_normal((self.k, self.d)))
    def tell(self, X, y):
        if self.fx is None: self.fx = self.best = y[0]; self.tabu.append(self.x.copy()); return
        rho = self.rho_factor * self.sigma() * np.sqrt(self.d)
        dist = np.sqrt(((X[:, None, :] - np.array(self.tabu)[None]) ** 2).sum(-1)).min(1)
        adm = (dist >= rho) | (y < self.best)
        if not adm.any(): adm[:] = True                        # all tabu, none aspirated: take the best anyway
        idx = np.flatnonzero(adm); j = idx[np.argmin(y[idx])]
        self.x, self.fx = X[j].copy(), y[j]; self.best = min(self.best, y[j])
        self.tabu = (self.tabu + [self.x.copy()])[-self.L:]
def sbx_pm(P1, P2, lo, hi, rng, pc=0.9, eta_c=15.0, pm=None, eta_m=20.0):
    n, d = P1.shape; pm = 1.0 / d if pm is None else pm
    u = rng.random((n, d))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
    cross = (rng.random((n, 1)) < pc) & (rng.random((n, d)) < 0.5); beta = np.where(cross, beta, 1.0)
    C1, C2 = 0.5 * ((1 + beta) * P1 + (1 - beta) * P2), 0.5 * ((1 - beta) * P1 + (1 + beta) * P2)
    swap = cross & (rng.random((n, d)) < 0.5)
    C = np.vstack([np.where(swap, C2, C1), np.where(swap, C1, C2)])
    u = rng.random(C.shape)
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
    return np.clip(C + (rng.random(C.shape) < pm) * delta * (hi - lo), lo, hi)

class GA(AskTell):
    name = "GA"
    def __init__(self, d, lo, hi, budget, rng, n=50, pc=0.9, pm=None, tour=2):
        super().__init__(d, lo, hi, budget, rng); self.n, self.pc, self.pm, self.tour, self.P, self.fP = n, pc, pm, tour, None, None
    def select(self, k):
        c = self.rng.integers(0, self.n, (k, self.tour)); return c[np.arange(k), np.argmin(self.fP[c], axis=1)]
    def ask(self):
        if self.P is None: return self.uniform(self.n)
        m = self.n // 2 + 1; a, b = self.select(m), self.select(m)
        return sbx_pm(self.P[a], self.P[b], self.lo, self.hi, self.rng, pc=self.pc, pm=self.pm)[: self.n - 1]
    def tell(self, X, y):
        if self.P is None: self.P, self.fP = X.copy(), y.copy(); return
        e = np.argmin(self.fP); self.P, self.fP = np.vstack([self.P[e], X]), np.concatenate([[self.fP[e]], y])
class PSO(AskTell):
    name = "PSO"
    def __init__(self, d, lo, hi, budget, rng, n=40, w=0.7298, c1=1.49618, c2=1.49618, vmax=0.2):
        super().__init__(d, lo, hi, budget, rng)
        self.n, self.w, self.c1, self.c2, self.vmax = n, w, c1, c2, vmax * self.span
        self.X = self.uniform(n); self.V = rng.uniform(-1, 1, (n, d)) * 0.1 * self.span; self.Pb = self.fPb = None
    def ask(self):
        if self.Pb is None: return self.X
        g = self.Pb[np.argmin(self.fPb)]; r1, r2 = self.rng.random((2, self.n, self.d))
        self.V = np.clip(self.w * self.V + self.c1 * r1 * (self.Pb - self.X) + self.c2 * r2 * (g - self.X), -self.vmax, self.vmax)
        self.X = self.clip(self.X + self.V); return self.X
    def tell(self, X, y):
        if self.Pb is None: self.Pb, self.fPb = X.copy(), y.copy(); return
        imp = y < self.fPb; self.Pb[imp], self.fPb[imp] = X[imp], y[imp]
class ACOR(AskTell):
    name = "ACO_R"
    def __init__(self, d, lo, hi, budget, rng, k=50, m=2, q=0.1, xi=0.85):
        super().__init__(d, lo, hi, budget, rng); self.k, self.m, self.xi = k, m, xi
        r = np.arange(1, k + 1); w = np.exp(-((r - 1) ** 2) / (2 * (q * k) ** 2)) / (q * k * np.sqrt(2 * np.pi))
        self.cdf = np.cumsum(w / w.sum()); self.S = self.fS = None
    def ask(self):
        if self.S is None: return self.uniform(self.k)
        l = np.minimum(np.searchsorted(self.cdf, self.rng.random(self.m)), self.k - 1)
        sig = self.xi * np.abs(self.S[None, :, :] - self.S[l][:, None, :]).sum(1) / (self.k - 1)
        return self.clip(self.S[l] + sig * self.rng.standard_normal((self.m, self.d)))
    def tell(self, X, y):
        S = X if self.S is None else np.vstack([self.S, X]); fS = y if self.fS is None else np.concatenate([self.fS, y])
        o = np.argsort(fS, kind="stable")[: self.k]; self.S, self.fS = S[o], fS[o]
class DE(AskTell):
    name = "DE"
    def __init__(self, d, lo, hi, budget, rng, n=50, F=0.5, CR=0.9):
        super().__init__(d, lo, hi, budget, rng); self.n, self.F, self.CR, self.P, self.fP = n, F, CR, None, None
    def ask(self):
        if self.P is None: return self.uniform(self.n)
        n, d = self.n, self.d; idx = distinct_indices(self.rng, n, 3)
        V = self.P[idx[:, 0]] + self.F * (self.P[idx[:, 1]] - self.P[idx[:, 2]])
        mask = self.rng.random((n, d)) < self.CR; mask[np.arange(n), self.rng.integers(0, d, n)] = True
        return self.clip(np.where(mask, V, self.P))
    def tell(self, X, y):
        if self.P is None: self.P, self.fP = X.copy(), y.copy(); return
        imp = y <= self.fP; self.P[imp], self.fP[imp] = X[imp], y[imp]
class CMAES(AskTell):
    """(mu/mu_w, lambda)-CMA-ES, defaults of Hansen's 2016 tutorial, positive weights only (as in Notebook 1)."""
    name = "CMA-ES"
    def __init__(self, d, lo, hi, budget, rng, m0=None, sigma0=0.3, lam=None):
        super().__init__(d, lo, hi, budget, rng); n = d
        self.m = (self.uniform(1)[0] if lo is not None else np.zeros(n)) if m0 is None else np.array(m0, float)
        self.sigma = sigma0 * self.span; self.lam = lam or 4 + int(3 * np.log(n)); self.mu = self.lam // 2
        wp = np.log((self.lam + 1) / 2) - np.log(np.arange(1, self.mu + 1)); self.w = wp / wp.sum(); self.mueff = 1 / np.sum(self.w**2)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5); self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n); self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        self.ps, self.pc, self.C, self.B, self.D, self.g = np.zeros(n), np.zeros(n), np.eye(n), np.eye(n), np.ones(n), 0
    def ask(self):
        Z = self.rng.standard_normal((self.lam, self.d)); return self.clip(self.m + self.sigma * (Z * self.D) @ self.B.T)
    def tell(self, X, y):
        n = self.d; Y = (X[np.argsort(y)[: self.mu]] - self.m) / self.sigma; yw = self.w @ Y
        self.m = self.m + self.sigma * yw; Cis = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cis @ yw; self.g += 1
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.g)) < (1.4 + 2 / (n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        dh = (1 - hs) * self.cc * (2 - self.cc)
        self.C = ((1 - self.c1 - self.cmu + self.c1 * dh) * self.C + self.c1 * np.outer(self.pc, self.pc) + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= np.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.C = (self.C + self.C.T) / 2; ev, self.B = np.linalg.eigh(self.C); self.D = np.sqrt(np.maximum(ev, 1e-30))

class CMAESsym(CMAES):
    """Same algorithm, but y = C^{1/2} z with the SYMMETRIC root B D B^T (unique, hence a function of C alone)."""
    name = "CMA-ES (sym. root)"
    def ask(self):
        Z = self.rng.standard_normal((self.lam, self.d))
        return self.clip(self.m + self.sigma * Z @ ((self.B * self.D) @ self.B.T))

f0 = BENCHMARKS["rastrigin"].f
o = np.array([1.5, -2.0, 0.75, 3.0, -1.25, 2.5, -0.5, 1.0, -3.0, 0.25])
lo0, hi0 = -5.12 * np.ones(10), 5.12 * np.ones(10)

def agree_len(t1, t2):
    rel = np.abs(t1 - t2) / np.maximum(np.abs(t1), 1e-300)
    return int(np.argmax(rel > 1e-6)) if np.any(rel > 1e-6) else len(rel)

# (1) box moved with the shift: run on f over [lo, hi] and on f(. - o) over [lo + o, hi + o], same seed
for A in [SA, TS, GA, PSO, ACOR, DE, CMAES, CMAESsym]:
    runs = []
    for lo, hi, ff in [(lo0, hi0, f0), (lo0 + o, hi0 + o, lambda X: f0(np.atleast_2d(X) - o))]:
        a = A(10, lo, hi, 5000, np.random.default_rng(1))
        a.span = 10.24                                   # scalar span, identical in both runs
        if isinstance(a, CMAES): a.sigma = 0.3 * 10.24
        runs.append(optimize(a, ff, 5000).trace()[1])
    print(f"box moved:  {A.name:18s} traces agree for {agree_len(*runs):4d} of 5000 evals; final {runs[0][-1]:.4g} / {runs[1][-1]:.4g}")

# (2) box NOT moved: single-state methods started inside the box, optimum shifted by o/4
o4 = o / 4
for A in [SA, TS]:
    a1, a2 = A(10, -5.12, 5.12, 5000, np.random.default_rng(1)), A(10, -5.12, 5.12, 5000, np.random.default_rng(1))
    a1.x = 0.5 * a1.x; a2.x = a1.x + o4
    s1, s2 = [], []                                      # current states after every tell
    ob1 = optimize(a1, f0, 5000, callback=lambda a, X, y, ob: s1.append((ob.evals, a.x + o4)))
    ob2 = optimize(a2, lambda X: f0(np.atleast_2d(X) - o4), 5000, callback=lambda a, X, y, ob: s2.append((ob.evals, a.x.copy())))
    split = [e for (e, x1), (_, x2) in zip(s1, s2) if not np.allclose(x1, x2, rtol=0, atol=1e-9)]
    print(f"fixed box:  {A.name}: states identical up to eval {split[0] - 1 if split else 5000}, best-so-far traces agree for "
          f"{agree_len(ob1.trace()[1], ob2.trace()[1])} evals; best f {ob1.best_f:.4g} / {ob2.best_f:.4g} "
          f"(first value {ob1.trace()[1][0]:.4g})")
```

```text
box moved:  SA                 traces agree for 5000 of 5000 evals; final 77.61 / 77.61
box moved:  TS                 traces agree for 5000 of 5000 evals; final 53.73 / 53.73
box moved:  GA                 traces agree for 5000 of 5000 evals; final 5.563 / 5.563
box moved:  PSO                traces agree for 5000 of 5000 evals; final 10.97 / 10.97
box moved:  ACO_R              traces agree for 5000 of 5000 evals; final 38.28 / 38.28
box moved:  DE                 traces agree for 5000 of 5000 evals; final 38.83 / 38.83
box moved:  CMA-ES             traces agree for   20 of 5000 evals; final 6.965 / 7.96
box moved:  CMA-ES (sym. root) traces agree for 5000 of 5000 evals; final 6.965 / 6.965
fixed box:  SA: states identical up to eval 26, best-so-far traces agree for 5000 evals; best f 90.58 / 90.58 (first value 137.9)
fixed box:  TS: states identical up to eval 56, best-so-far traces agree for 125 evals; best f 30.84 / 30.84 (first value 137.9)
```

---

## Notebook 2 — Differential evolution and CMA-ES

### Exercise 1 (★) — covariance of a difference vector; the finite-population term

If $x_{r_2},x_{r_3}$ are independent with covariance $\Sigma$, then

$$
\mathrm{Cov}(x_{r_2}-x_{r_3})=\mathrm{Cov}(x_{r_2})+\mathrm{Cov}(x_{r_3})-\mathrm{Cov}(x_{r_2},x_{r_3})-\mathrm{Cov}(x_{r_3},x_{r_2})=2\Sigma .
$$

In DE the two members are drawn *without replacement* from a finite population with (biased) sample covariance $S$.
They are then negatively correlated, $\mathrm{Cov}(x_{r_2},x_{r_3})=-S/(N-1)$, so the difference has covariance
$2S\,N/(N-1)$.

The $-2p/N$ (and $+p^2/N$) terms have a different origin. The new population's sample variance is
$\overline{z^2}-\bar z^2$. Because the $N$ trial vectors are drawn independently, the new sample mean $\bar z$
fluctuates around the old one, with variance of order $1/N$. That fluctuation is subtracted from the variance. It is
random drift of the population centre, the finite-population effect known from genetic drift.

### Exercise 2 (★★) — Zaharie's law, and the analogue for DE/best/1

Work one coordinate at a time, conditional on the population $x_1,\dots,x_N$. Without loss of generality its mean is
$0$. Write $s^2=\frac1N\sum_jx_j^2$ and $S=Ns^2$. The trial is $z_i=v_i$ with probability $p$ and $z_i=x_i$ otherwise.
Crossover decisions and indices are independent across $i$, so the $z_i$ are conditionally independent and

$$
\mathbb{E}[\mathrm{Var}(Z)]=\frac1N\sum_i\mathbb{E}z_i^2-\Big(\mathbb{E}\bar z\Big)^2-\frac1{N^2}\sum_i\mathrm{Var}(z_i).
$$

**(a) $r_1$ uniform over all $N$ members, $r_2\ne r_3$ a uniform distinct pair.** In this case
$\mathbb{E}v_i=0$ and $\mathbb{E}v_i^2=s^2+F^2\cdot2s^2\frac{N}{N-1}$. The cross term vanishes by the $r_2\leftrightarrow r_3$
symmetry. Write $\tilde s^2=s^2N/(N-1)$. Then

$$
\tfrac1N\textstyle\sum_i\mathbb{E}z_i^2=s^2(1-p)+p(s^2+2F^2\tilde s^2),\qquad \mathbb{E}z_i=(1-p)x_i,\qquad \mathbb{E}\bar z=0,
$$

$$
\tfrac1{N^2}\textstyle\sum_i\mathrm{Var}(z_i)=\tfrac{s^2}{N}\big[(1-p)p+p+2pF^2\tfrac{N}{N-1}\big].
$$

Collecting terms, and using $\tilde s^2(1-1/N)=s^2$,

$$
\mathbb{E}[\mathrm{Var}(Z)]=s^2\Big(1+2F^2p-\frac{2p}{N}+\frac{p^2}{N}\Big),
$$

which is Zaharie's formula.

**(b) $r_1,r_2,r_3$ distinct and all different from $i$ (the classical scheme, used in the notebook).** Now
$\mathbb{E}x_{r_1}=-x_i/(N-1)$ and $\mathbb{E}x_{r_1}^2=(S-x_i^2)/(N-1)$. Also
$\mathbb{E}x_{r_2}x_{r_3}=\big(x_i^2-(S-x_i^2)\big)/((N-1)(N-2))$. The same bookkeeping gives

$$
\mathbb{E}[\mathrm{Var}(Z)]=s^2\Big(1+2F^2p-\frac{2p}{N-1}+\frac{p^2N}{(N-1)^2}\Big),
$$

which differs from (a) by $O(p/N^2)$. For $N=5$, $p=1$, $F=0.5$, formula (a) gives $1.3000$ and formula (b) gives
$1.3125$; a simulation of the notebook's operator gives $1.3118\pm0.0010$ in the block below (the notebook's own
discriminating test reports $1.3135\pm0.0014$), which rejects (a) and agrees with (b).

**DE/best/1.** Here $v_i=b+F(x_{r_1}-x_{r_2})$ with $b=x_{\text{best}}-\bar x$ fixed, and $r_1\ne r_2$ uniform over the
population. Now $\mathbb{E}z_i=(1-p)x_i+pb$, so $\mathbb{E}\bar z=pb\ne0$, and

$$
\mathbb{E}[\mathrm{Var}(Z)]=s^2\Big(1-p+2pF^2-\frac{p(1-p)}{N}\Big)+p(1-p)\Big(1-\frac1N\Big)b^2 .
$$

The block checks it by simulation with $N=12$, $F=0.6$, $p=0.7$ on a fixed population: $0.7846\pm0.0006$ empirical vs
$0.7837$ predicted. The structural difference from DE/rand/1 is the term $-p\,s^2$. The mutant no longer carries the
base vector's own spread, so the population keeps its variance only if $2F^2\gtrsim1$, that is $F\gtrsim0.71$,
instead of $F\gtrsim\sqrt{(1-p/2)/N}$. The $b^2$ term is a transient spread that disappears once the population
gathers around the best point. This is why DE/best/1 converges (and converges prematurely) much faster.

```python
import numpy as np
r = np.random.default_rng(1)

def zaharie_a(F, p, N): return 1 + 2 * F**2 * p - 2 * p / N + p**2 / N                       # r1 uniform over all N
def zaharie_b(F, p, N): return 1 + 2 * F**2 * p - 2 * p / (N - 1) + p**2 * N / (N - 1) ** 2   # r1, r2, r3 distinct, != i

# DE/rand/1 with the notebook's index scheme (distinct, != i), N = 5, p = 1: many independent Gaussian populations
N, F, reps, d = 5, 0.5, 200000, 4
P = r.standard_normal((reps, N, d))
keys = r.random((reps, N, N)); keys[:, np.arange(N), np.arange(N)] = np.inf
idx = np.argsort(keys, axis=2)[:, :, :3]
take = lambda k: np.take_along_axis(P, idx[:, :, k:k + 1].repeat(d, axis=2), axis=1)
V = take(0) + F * (take(1) - take(2))
num, den = V.var(1).sum(1), P.var(1).sum(1); ratio = num.sum() / den.sum()
se = np.sqrt(reps * np.var(num - ratio * den)) / den.sum()
print(f"DE/rand/1, N=5, p=1, F=0.5: formula (a) {zaharie_a(F, 1, N):.4f}, formula (b) {zaharie_b(F, 1, N):.4f}, simulation {ratio:.4f} +- {se:.4f}")

# DE/best/1 on a FIXED population: v_i = x_best + F (x_r1 - x_r2), r1 != r2 uniform over the population, binomial crossover w.p. p
N, F, p, reps = 12, 0.6, 0.7, 400000
x = r.standard_normal(N); x -= x.mean(); s2 = np.mean(x**2); b = x[np.argmin(x**2 + (x - 1.5) ** 2)]   # any fixed "best" member
keys = r.random((reps, N, N)); r12 = np.argsort(keys, axis=2)[:, :, :2]                # distinct pair per target
V = b + F * (x[r12[:, :, 0]] - x[r12[:, :, 1]])
Z = np.where(r.random((reps, N)) < p, V, x)
emp = Z.var(1); pred = s2 * (1 - p + 2 * p * F**2 - p * (1 - p) / N) + p * (1 - p) * (1 - 1 / N) * b**2
print(f"DE/best/1, N={N}, F={F}, p={p}: E[Var] empirical {emp.mean():.4f} +- {emp.std() / np.sqrt(reps):.4f}, predicted {pred:.4f}")
```

```text
DE/rand/1, N=5, p=1, F=0.5: formula (a) 1.3000, formula (b) 1.3125, simulation 1.3118 +- 0.0010
DE/best/1, N=12, F=0.6, p=0.7: E[Var] empirical 0.7846 +- 0.0006, predicted 0.7837
```

### Exercise 3 (★★) — $(1+1)$-ES on the sphere as $d\to\infty$

Let $x=Re_1$, $y=x+\sigma z$ with $z\sim\mathcal{N}(0,I)$, and $\sigma=\sigma^{\ast}R/d$. Then

$$
\frac{\Vert y\Vert^2}{R^2}=1+\frac{2\sigma^{\ast}z_1}{d}+\frac{\sigma^{\ast2}}{d}\cdot\frac{\Vert z\Vert^2}{d}
=1+\frac{2\sigma^{\ast}z_1+\sigma^{\ast2}}{d}+o\Big(\frac1d\Big),
$$

because $\Vert z\Vert^2/d\to1$ by the law of large numbers. Hence $R'/R=1+(\sigma^{\ast}z_1+\sigma^{\ast2}/2)/d+o(1/d)$,
and the normalised gain $G=d(R-R')/R\to-\sigma^{\ast}z_1-\sigma^{\ast2}/2$.

**Success probability.** Success means $G\gt0$, that is $z_1\lt-\sigma^{\ast}/2$, so $P_s=\Phi(-\sigma^{\ast}/2)=1-\Phi(\sigma^{\ast}/2)$.

**Progress rate.** Elitist selection keeps $\max(G,0)$. Using $\int_{-\infty}^{a}(-t)\phi(t)\,dt=\phi(a)$,

$$
\varphi^{\ast}=\int_{-\infty}^{-\sigma^{\ast}/2}\Big(-\sigma^{\ast}t-\frac{\sigma^{\ast2}}{2}\Big)\phi(t)\,dt
=\sigma^{\ast}\phi\Big(\frac{\sigma^{\ast}}2\Big)-\frac{\sigma^{\ast2}}2\Big(1-\Phi\Big(\frac{\sigma^{\ast}}2\Big)\Big),
$$

with $\phi(\sigma^{\ast}/2)=e^{-\sigma^{\ast2}/8}/\sqrt{2\pi}$. Setting the derivative to zero (the two terms
$\mp\frac{\sigma^{\ast2}}{4}\phi(\sigma^{\ast}/2)$ cancel) gives $\phi(\sigma^{\ast}/2)=\sigma^{\ast}\big(1-\Phi(\sigma^{\ast}/2)\big)$.
Its root is $\sigma^{\ast}=1.224$, where $\varphi^{\ast}=0.2025$ and $P_s=0.270$, matching the notebook; a Monte Carlo
at $d=5000$ gives $0.2032$ and $0.2706$.

```python
import numpy as np
from scipy import stats, optimize as so
Phi, phi = stats.norm.cdf, stats.norm.pdf
prog = lambda s: s * phi(s / 2) - s**2 / 2 * (1 - Phi(s / 2))
root = so.brentq(lambda s: phi(s / 2) - s * (1 - Phi(s / 2)), 0.5, 3)          # stationarity condition
print(f"sigma* = {root:.4f}, phi* = {prog(root):.4f}, P_s = {1 - Phi(root / 2):.4f}")
# Monte Carlo of the normalised elitist gain at d = 5000 and sigma* = 1.224
d, n = 5000, 400000; r = np.random.default_rng(0); z1, rest = r.standard_normal(n), r.chisquare(d - 1, n); s = root / d
Rn = np.sqrt((1 + s * z1) ** 2 + s**2 * rest)
print(f"Monte Carlo d={d}: progress {np.mean(np.maximum(0, 1 - Rn)) * d:.4f}, success rate {np.mean(Rn < 1):.4f}")
```

```text
sigma* = 1.2240, phi* = 0.2025, P_s = 0.2703
Monte Carlo d=5000: progress 0.2032, success rate 0.2706
```

### Exercise 4 (★★) — invariance of CMA-ES

**Monotone transformations.** $f$ enters the update only through the permutation that sorts the $\lambda$ values
$f(x_k)$. A strictly increasing $g$ preserves that permutation, ties included. By induction over generations, with the
same random numbers, the runs on $f$ and on $g\circ f$ produce identical states. The notebook checks this bit for bit
with $g(t)=t^3$.

**Rotations.** Let $h(x)=f(R^{\top}x)$ with $R$ orthogonal. Suppose that at generation $g$ the state on $h$ is the image
of the state on $f$:

$$
(m',\sigma',C',p'_\sigma,p'_c)=(Rm,\ \sigma,\ RCR^{\top},\ Rp_\sigma,\ Rp_c),
$$

which holds at $g=0$ if $m_0'=Rm_0$ and $C_0'=RC_0R^{\top}$. Couple the samples by $y'_k=Ry_k$; this is legitimate
because $Ry_k\sim\mathcal{N}(0,RCR^{\top})$. Then $x'_k=Rx_k$ and $h(x'_k)=f(x_k)$, so the ranking is the same and
$y'_w=Ry_w$, hence $m'\leftarrow Rm$. The symmetric root is unique, so $(C')^{-1/2}=RC^{-1/2}R^{\top}$. This gives
$p'_\sigma\leftarrow Rp_\sigma$ and $\Vert p'_\sigma\Vert=\Vert p_\sigma\Vert$, so $\sigma'=\sigma$ and $h_\sigma$ is
the same. Next $p'_c\leftarrow Rp_c$, and each covariance term transforms as
$R(\cdot)R^{\top}$ ($Rp_cp_c^{\top}R^{\top}$ and $Ry_ky_k^{\top}R^{\top}$). The hypothesis therefore holds at $g+1$.
The sequences of $f$-values have the same law, so every performance measure is identical.

The notebook checks the equivariance of `tell` directly: a rotated copy stays the rotated image to $10^{-14}$ over 80
generations. Exercise 6 of Notebook 1 shows why the *path* also depends on how $C^{1/2}$ is computed.

### Exercise 5 (★★★) — active CMA-ES

Use the 2016 tutorial defaults with negative weights. The raw weights are
$w'_i=\ln\frac{\lambda+1}2-\ln i$ for all $i\le\lambda$. The positive ones are normalised to sum to 1. The negative
ones are scaled by $\min(\alpha_\mu^-,\alpha_{\mu_{\text{eff}}}^-,\alpha_{\text{posdef}}^-)/\sum\lvert w'^-\rvert$, where

$$
\alpha_\mu^-=1+\frac{c_1}{c_\mu},\qquad \alpha_{\mu_{\text{eff}}}^-=1+\frac{2\mu_{\text{eff}}^-}{\mu_{\text{eff}}+2},\qquad
\alpha_{\text{posdef}}^-=\frac{1-c_1-c_\mu}{n\,c_\mu}.
$$

In the rank-$\mu$ sum, a negative weight is used as $w_i^{\circ}=w_i\,n/\Vert C^{-1/2}y_{i:\lambda}\Vert^2$. The decay
factor becomes $1+c_1\delta(h_\sigma)-c_1-c_\mu\sum_jw_j$. The mean update still uses the positive weights only. This
follows Jastrebski & Arnold (2006) in the form of Hansen's 2016 tutorial.

**Result.** Rotated ellipsoid with condition $10^6$, $d=10$, $m_0=R^{\top}(3,\dots,3)$, $\sigma_0=1$, 10 seeds, evaluations
to $f\le10^{-8}$ (counted at the end of the generation that reaches the target):

| variant | median | IQR |
|---|---|---|
| CMA-ES | 5885 | [5748, 5990] |
| active CMA-ES | 4165 | [4085, 4298] |

The speed-up is $1.41\times$ (paired Wilcoxon $p=2\cdot10^{-3}$). On the sphere the two are
indistinguishable (medians 1495 vs 1510). The negative update removes variance in directions that were bad, and that
speeds up learning of the small eigenvalues of $C$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import BudgetedObjective, BENCHMARKS, shifted_rotated
from utils import random_rotation, ellipsoid, sphere
# --- ask-tell core of Notebook 1 (compact copy) ---
class AskTell:
    name = "base"
    def __init__(self, d, lo, hi, budget, rng):
        self.d, self.lo, self.hi, self.budget, self.rng = d, lo, hi, budget, rng
        self.span = (hi - lo) if lo is not None else 1.0
        self.n_told = 0
    def clip(self, X): return X if self.lo is None else np.clip(X, self.lo, self.hi)
    def uniform(self, n): return self.rng.uniform(self.lo, self.hi, (n, self.d))
    def progress(self): return min(1.0, self.n_told / self.budget)

def distinct_indices(rng, n, k, n_pool=None, exclude=None):
    n_pool = n if n_pool is None else n_pool
    keys = rng.random((n, n_pool)); keys[np.arange(n), np.arange(n)] = np.inf
    if exclude is not None: keys[np.arange(n), exclude] = np.inf
    return np.argsort(keys, axis=1)[:, :k]

def optimize(opt, f, budget, callback=None):
    obj = BudgetedObjective(f, budget)
    while obj.remaining > 0:
        X = opt.ask()
        if len(X) > obj.remaining: obj(X[: obj.remaining]); break
        y = obj(X); opt.tell(X, y); opt.n_told += len(y)
        if callback is not None: callback(opt, X, y, obj)
    return obj
class CMAES(AskTell):
    """(mu/mu_w, lambda)-CMA-ES, defaults of Hansen's 2016 tutorial, positive weights only (as in Notebook 1)."""
    name = "CMA-ES"
    def __init__(self, d, lo, hi, budget, rng, m0=None, sigma0=0.3, lam=None):
        super().__init__(d, lo, hi, budget, rng); n = d
        self.m = (self.uniform(1)[0] if lo is not None else np.zeros(n)) if m0 is None else np.array(m0, float)
        self.sigma = sigma0 * self.span; self.lam = lam or 4 + int(3 * np.log(n)); self.mu = self.lam // 2
        wp = np.log((self.lam + 1) / 2) - np.log(np.arange(1, self.mu + 1)); self.w = wp / wp.sum(); self.mueff = 1 / np.sum(self.w**2)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5); self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n); self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        self.ps, self.pc, self.C, self.B, self.D, self.g = np.zeros(n), np.zeros(n), np.eye(n), np.eye(n), np.ones(n), 0
    def ask(self):
        Z = self.rng.standard_normal((self.lam, self.d)); return self.clip(self.m + self.sigma * (Z * self.D) @ self.B.T)
    def tell(self, X, y):
        n = self.d; Y = (X[np.argsort(y)[: self.mu]] - self.m) / self.sigma; yw = self.w @ Y
        self.m = self.m + self.sigma * yw; Cis = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cis @ yw; self.g += 1
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.g)) < (1.4 + 2 / (n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        dh = (1 - hs) * self.cc * (2 - self.cc)
        self.C = ((1 - self.c1 - self.cmu + self.c1 * dh) * self.C + self.c1 * np.outer(self.pc, self.pc) + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= np.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.C = (self.C + self.C.T) / 2; ev, self.B = np.linalg.eigh(self.C); self.D = np.sqrt(np.maximum(ev, 1e-30))

class ActiveCMAES(CMAES):
    """CMA-ES with negative weights for the lambda - mu worst samples (Hansen 2016 defaults, Jastrebski & Arnold 2006)."""
    name = "active CMA-ES"
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        wp = np.log((self.lam + 1) / 2) - np.log(np.arange(1, self.lam + 1)); pos, neg = wp[:self.mu], wp[self.mu:]
        mueff_neg = neg.sum() ** 2 / np.sum(neg ** 2)
        scale = min(1 + self.c1 / self.cmu, 1 + 2 * mueff_neg / (self.mueff + 2), (1 - self.c1 - self.cmu) / (self.d * self.cmu))
        self.wall = np.concatenate([pos / pos.sum(), scale * neg / np.abs(neg).sum()])
    def tell(self, X, y):
        n = self.d; Yall = (X[np.argsort(y)] - self.m) / self.sigma; yw = self.w @ Yall[:self.mu]
        self.m = self.m + self.sigma * yw; Ci = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Ci @ yw; self.g += 1
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.g)) < (1.4 + 2 / (n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        dh = (1 - hs) * self.cc * (2 - self.cc); wo = self.wall.copy(); neg = wo < 0
        wo[neg] *= n / np.sum((Yall[neg] @ Ci) ** 2, axis=1)            # w_i * n / ||C^{-1/2} y_i||^2
        self.C = ((1 + self.c1 * dh - self.c1 - self.cmu * self.wall.sum()) * self.C
                  + self.c1 * np.outer(self.pc, self.pc) + self.cmu * (Yall.T * wo) @ Yall)
        self.sigma *= np.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.C = (self.C + self.C.T) / 2; ev, self.B = np.linalg.eigh(self.C); self.D = np.sqrt(np.maximum(ev, 1e-30))

def evals_to(A, f, m0, seed, target=1e-8, budget=8000):
    hit = []
    optimize(A(10, None, None, budget, np.random.default_rng(seed), m0=m0, sigma0=1.0), f, budget,
             callback=lambda o, X, y, ob: hit.append(ob.evals) if ob.best_f <= target and not hit else None)
    return hit[0] if hit else np.inf

d = 10; Rot = random_rotation(d, np.random.default_rng(20))
f_rot = shifted_rotated(lambda X: ellipsoid(X, 1e6), np.zeros(d), Rot)
res = {A.name: np.array([evals_to(A, f_rot, Rot.T @ np.full(d, 3.0), s) for s in range(10)]) for A in [CMAES, ActiveCMAES]}
for k, v in res.items():
    print(f"rotated ellipsoid (cond 1e6): {k:14s} median {np.median(v):.0f}  IQR [{np.percentile(v, 25):.0f}, {np.percentile(v, 75):.0f}]")
a, b = res["CMA-ES"], res["active CMA-ES"]
print(f"speed-up {np.median(a) / np.median(b):.2f}x, paired Wilcoxon p = {stats.wilcoxon(a, b).pvalue:.1e}")
sph = [np.median([evals_to(A, sphere, np.full(d, 3.0), s) for s in range(10)]) for A in [CMAES, ActiveCMAES]]
print(f"sphere: medians {sph[0]:.0f} vs {sph[1]:.0f}")
```

```text
rotated ellipsoid (cond 1e6): CMA-ES         median 5885  IQR [5748, 5990]
rotated ellipsoid (cond 1e6): active CMA-ES  median 4165  IQR [4085, 4298]
speed-up 1.41x, paired Wilcoxon p = 2.0e-03
sphere: medians 1495 vs 1510
```

### Exercise 6 (★★★) — IPOP-CMA-ES vs JADE on the shifted-rotated Rastrigin

IPOP-CMA-ES (Auger & Hansen 2005) restarts CMA-ES with $\lambda$ doubled. The new mean is uniform in the box and
$\sigma_0=0.2\cdot$ span. A restart happens when any of three stopping rules fires:

- $\sigma\max_iD_i\lt10^{-12}\sigma_0$;
- the best $f$ per generation stays flat within $10^{-12}$ over $10+\lceil30n/\lambda\rceil$ generations;
- $\mathrm{cond}(C)\gt10^{14}$ (that is, $\max D/\min D\gt10^7$).

**Result** (Notebook 2's shifted-rotated Rastrigin, $d=10$, $10^4d=10^5$ evaluations, 10 seeds):

| algorithm | final error median | IQR | runs with error $\lt10^{-8}$ |
|---|---|---|---|
| IPOP-CMA-ES | 0 | [0, $2\cdot10^{-14}$] | 80 % |
| JADE ($N=50$) | 3.50 | [2.99, 3.98] | 0 % |
| CMA-ES, no restart | 11.9 | [7.2, 18.7] | 0 % |

Mann–Whitney IPOP vs JADE: $p=1.3\cdot10^{-4}$. The median number of restarts was 6, so the final $\lambda$ was 640.
A large population smooths Rastrigin's global quadratic trend. This is the reason IPOP was designed, and it reverses
the Rastrigin picture of the notebook's small-budget, single-start comparison.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import BudgetedObjective, BENCHMARKS, shifted_rotated
from utils import random_rotation
# --- ask-tell core of Notebook 1 (compact copy) ---
class AskTell:
    name = "base"
    def __init__(self, d, lo, hi, budget, rng):
        self.d, self.lo, self.hi, self.budget, self.rng = d, lo, hi, budget, rng
        self.span = (hi - lo) if lo is not None else 1.0
        self.n_told = 0
    def clip(self, X): return X if self.lo is None else np.clip(X, self.lo, self.hi)
    def uniform(self, n): return self.rng.uniform(self.lo, self.hi, (n, self.d))
    def progress(self): return min(1.0, self.n_told / self.budget)

def distinct_indices(rng, n, k, n_pool=None, exclude=None):
    n_pool = n if n_pool is None else n_pool
    keys = rng.random((n, n_pool)); keys[np.arange(n), np.arange(n)] = np.inf
    if exclude is not None: keys[np.arange(n), exclude] = np.inf
    return np.argsort(keys, axis=1)[:, :k]

def optimize(opt, f, budget, callback=None):
    obj = BudgetedObjective(f, budget)
    while obj.remaining > 0:
        X = opt.ask()
        if len(X) > obj.remaining: obj(X[: obj.remaining]); break
        y = obj(X); opt.tell(X, y); opt.n_told += len(y)
        if callback is not None: callback(opt, X, y, obj)
    return obj
class CMAES(AskTell):
    """(mu/mu_w, lambda)-CMA-ES, defaults of Hansen's 2016 tutorial, positive weights only (as in Notebook 1)."""
    name = "CMA-ES"
    def __init__(self, d, lo, hi, budget, rng, m0=None, sigma0=0.3, lam=None):
        super().__init__(d, lo, hi, budget, rng); n = d
        self.m = (self.uniform(1)[0] if lo is not None else np.zeros(n)) if m0 is None else np.array(m0, float)
        self.sigma = sigma0 * self.span; self.lam = lam or 4 + int(3 * np.log(n)); self.mu = self.lam // 2
        wp = np.log((self.lam + 1) / 2) - np.log(np.arange(1, self.mu + 1)); self.w = wp / wp.sum(); self.mueff = 1 / np.sum(self.w**2)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5); self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n); self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        self.ps, self.pc, self.C, self.B, self.D, self.g = np.zeros(n), np.zeros(n), np.eye(n), np.eye(n), np.ones(n), 0
    def ask(self):
        Z = self.rng.standard_normal((self.lam, self.d)); return self.clip(self.m + self.sigma * (Z * self.D) @ self.B.T)
    def tell(self, X, y):
        n = self.d; Y = (X[np.argsort(y)[: self.mu]] - self.m) / self.sigma; yw = self.w @ Y
        self.m = self.m + self.sigma * yw; Cis = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cis @ yw; self.g += 1
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.g)) < (1.4 + 2 / (n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        dh = (1 - hs) * self.cc * (2 - self.cc)
        self.C = ((1 - self.c1 - self.cmu + self.c1 * dh) * self.C + self.c1 * np.outer(self.pc, self.pc) + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= np.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.C = (self.C + self.C.T) / 2; ev, self.B = np.linalg.eigh(self.C); self.D = np.sqrt(np.maximum(ev, 1e-30))
class JADE(AskTell):
    """DE/current-to-pbest/1/bin with archive and adaptive mu_F, mu_CR (Zhang & Sanderson 2009), as in Notebook 2."""
    name = "JADE"
    def __init__(self, d, lo, hi, budget, rng, n=50, p=0.05, c=0.1):
        super().__init__(d, lo, hi, budget, rng); self.n, self.p, self.c = n, p, c
        self.muF, self.muCR, self.P, self.fP, self.A = 0.5, 0.5, None, None, np.empty((0, d))
    def ask(self):
        if self.P is None: return self.uniform(self.n)
        n, d, r = self.n, self.d, self.rng
        self.CRi = np.clip(r.normal(self.muCR, 0.1, n), 0, 1); F = np.empty(n); todo = np.arange(n)
        while len(todo):
            F[todo] = self.muF + 0.1 * np.tan(np.pi * (r.random(len(todo)) - 0.5)); todo = todo[F[todo] <= 0]
        self.Fi = np.minimum(F, 1.0)
        pbest = self.P[r.choice(np.argsort(self.fP)[: max(1, int(np.ceil(self.p * n)))], n)]
        pool = np.vstack([self.P, self.A]); r1 = distinct_indices(r, n, 1)[:, 0]
        r2 = distinct_indices(r, n, 1, n_pool=len(pool), exclude=r1)[:, 0]
        V = self.P + self.Fi[:, None] * (pbest - self.P) + self.Fi[:, None] * (self.P[r1] - pool[r2])
        mask = r.random((n, d)) < self.CRi[:, None]; mask[np.arange(n), r.integers(0, d, n)] = True
        return self.clip(np.where(mask, V, self.P))
    def tell(self, X, y):
        if self.P is None: self.P, self.fP = X.copy(), y.copy(); return
        s = y < self.fP
        if s.any():
            self.A = np.vstack([self.A, self.P[s]])
            if len(self.A) > self.n: self.A = self.A[self.rng.choice(len(self.A), self.n, replace=False)]
        self.P[s], self.fP[s] = X[s], y[s]
        if s.any():
            SF = self.Fi[s]; self.muCR = (1 - self.c) * self.muCR + self.c * self.CRi[s].mean()
            self.muF = (1 - self.c) * self.muF + self.c * (SF**2).sum() / SF.sum()

class IPOPCMAES(AskTell):
    """IPOP-CMA-ES (Auger & Hansen 2005): restart with doubled lambda, uniform mean, sigma0 = 0.2 * span."""
    name = "IPOP-CMA-ES"
    def __init__(self, d, lo, hi, budget, rng):
        super().__init__(d, lo, hi, budget, rng); self.lam = 4 + int(3 * np.log(d)); self.restarts = 0; self.new_run()
    def new_run(self):
        self.cma = CMAES(self.d, self.lo, self.hi, self.budget, self.rng, sigma0=0.2, lam=self.lam)
        self.hist, self.sigma0 = [], self.cma.sigma
    def ask(self): return self.cma.ask()
    def tell(self, X, y):
        c = self.cma; c.tell(X, y); self.hist.append(y.min()); k = 10 + int(np.ceil(30 * self.d / c.lam))
        flat = len(self.hist) > k and max(self.hist[-k:]) - min(self.hist[-k:]) < 1e-12
        if c.sigma * c.D.max() < 1e-12 * self.sigma0 or flat or c.D.max() / c.D.min() > 1e7:   # cond(C) > 1e14
            self.lam *= 2; self.restarts += 1; self.new_run()

d = 10; b = BENCHMARKS["rastrigin"]; rr = np.random.default_rng(100 + len("rastrigin"))       # Notebook 2's instance
f = shifted_rotated(b.f, rr.uniform(0.6 * b.lower, 0.6 * b.upper, d), random_rotation(d, rr))
B, seeds = 10**4 * d, 10
res, restarts = {}, []
for A in [IPOPCMAES, JADE, CMAES]:
    errs = []
    for s in range(seeds):
        a = A(d, b.lower, b.upper, B, np.random.default_rng(s)); errs.append(optimize(a, f, B).best_f)
        if A is IPOPCMAES: restarts.append(a.restarts)
    res[A.name] = np.array(errs)
    q = np.percentile(errs, [25, 50, 75])
    print(f"{A.name:12s} final error median {q[1]:.3g}  IQR [{q[0]:.2g}, {q[2]:.3g}]  runs < 1e-8: {np.mean(res[A.name] < 1e-8):.0%}")
print(f"IPOP restarts: median {np.median(restarts):.0f} (final lambda {(4 + int(3 * np.log(d))) * 2 ** int(np.median(restarts))}); "
      f"Mann-Whitney IPOP vs JADE p = {stats.mannwhitneyu(res['IPOP-CMA-ES'], res['JADE']).pvalue:.1e}")
```

```text
IPOP-CMA-ES  final error median 0  IQR [0, 2.13e-14]  runs < 1e-8: 80%
JADE         final error median 3.5  IQR [3, 3.98]  runs < 1e-8: 0%
CMA-ES       final error median 11.9  IQR [7.2, 18.7]  runs < 1e-8: 0%
IPOP restarts: median 6 (final lambda 640); Mann-Whitney IPOP vs JADE p = 1.3e-04
```

---

## Notebook 3 — Parameter tuning, parameter control and hyper-heuristics

### Exercise 1 (★) — the log-normal rule

$\ln\sigma'=\ln\sigma+\tau\xi$ with $\xi\sim\mathcal{N}(0,1)$, so $\mathbb{E}[\ln\sigma']=\ln\sigma$. By the Gaussian
moment generating function, $\mathbb{E}[\sigma']=\sigma\,\mathbb{E}[e^{\tau\xi}]=\sigma e^{\tau^2/2}$. The density of
$\ln\sigma'$ is symmetric about $\ln\sigma$, so multiplying by $c$ and dividing by $c$ are equally likely: the rule is
unbiased on the scale that matters for step sizes.

**Additive mutation** $\sigma'=\sigma+\tau'\xi$ is a poor alternative for three reasons.

1. It can produce $\sigma'\le0$, which then needs an ad-hoc repair.
2. It is not scale invariant. The relative change $\tau'\xi/\sigma$ is huge when $\sigma\ll\tau'$ and negligible when
   $\sigma\gg\tau'$. The linear convergence seen in the notebook requires $\sigma$ to shrink geometrically over many
   orders of magnitude, which needs multiplicative steps.
3. Its selection pressure is asymmetric. Near the optimum the small $\sigma'$ are selected, but an additive walk cannot
   keep reducing $\sigma$ by a constant *factor* per generation.

### Exercise 2 (★★) — $c_{1,\lambda}$ and the stationary distance

Let $\xi_{1:\lambda}=\max_k\xi_k$ for i.i.d. standard normals. Its CDF is $\Phi(x)^{\lambda}$, so its density is
$\lambda\phi(x)\Phi(x)^{\lambda-1}$ and

$$
c_{1,\lambda}=\mathbb{E}[\xi_{1:\lambda}]=\int_{-\infty}^{\infty}x\,\lambda\,\phi(x)\,\Phi(x)^{\lambda-1}\,dx
\qquad(c_{1,10}=1.5388).
$$

For the $(1,\lambda)$-ES on the sphere, the calculation of Exercise 3 of Notebook 2 gives each offspring the
normalised gain $-\sigma^{\ast}z_{1,k}-\sigma^{\ast2}/2$. The term $\sigma^{\ast2}/2$ is common to all offspring
(concentration of $\Vert z\Vert^2/d$), so comma selection picks the largest $-z_{1,k}$. That largest value is
distributed as $\xi_{1:\lambda}$, which gives $\varphi^{\ast}=c_{1,\lambda}\sigma^{\ast}-\sigma^{\ast2}/2$.

With constant $\sigma$, $\sigma^{\ast}=\sigma d/R$ grows as $R$ shrinks. Progress stops where $\varphi^{\ast}=0$,
at $\sigma^{\ast}=2c_{1,\lambda}$, so the stationary distance is

$$
R_\infty=\frac{\sigma d}{2c_{1,\lambda}} .
$$

The point is stable: for $R\gt R_\infty$ progress is positive, and for $R\lt R_\infty$ it is negative, pushing $R$
back up. For $d=100$ and $\sigma=0.01$ this gives $0.3249$; the notebook measures $0.3240$.

### Exercise 3 (★★) — convergence of adaptive pursuit

Let $K$ be the number of arms and $p_{\max}=1-(K-1)p_{\min}$. After each update, AP moves $p$ towards the target
$t^{(i^{\ast})}$, which has $p_{\max}$ at $i^{\ast}=\arg\max_iq_i$ and $p_{\min}$ elsewhere:
$p\leftarrow p+\beta(t-p)$. Suppose the rewards are stationary with a unique best mean $\mu_{i^{\ast}}$, and the
quality estimates converge to their means, $q_i\to\mu_i$, once every arm is played infinitely often. Two cases give
this:

- deterministic rewards with any $\alpha\in(0,1]$;
- stochastic rewards with Robbins–Monro step sizes $\alpha_t\to0$, $\sum\alpha_t=\infty$ and $\sum\alpha_t^2\lt\infty$.

Every arm has probability at least $p_{\min}\gt0$ at every step, whatever happened before. By the conditional
(Lévy) form of the second Borel–Cantelli lemma, $\sum_t\Pr[\text{arm }i\text{ at }t\mid\text{past}]=\infty$ implies
that every arm is played infinitely often a.s., so the estimates do converge. There is then an a.s. finite time $T_0$
after which $\arg\max_iq_i=i^{\ast}$ for ever. From then on the target $t$ is fixed and after $k$ further updates

$$
p_{T_0+k}-t=(1-\beta)^{k}\,(p_{T_0}-t),
$$

so $p_{i^{\ast}}\to p_{\max}$ geometrically with rate $1-\beta$ per update. For $\beta=0.3$ the error shrinks by a
factor $10^{3}$ in about 20 updates ($0.7^{20}=8\cdot10^{-4}$).

With a constant $\alpha$ and noisy rewards, $q$ does not converge. It fluctuates around $\mu$ with variance of order
$\alpha$. The argmax can still switch occasionally, so $p_{i^{\ast}}$ spends most of its time near $p_{\max}$ without
converging. The notebook sees $0.919991\approx p_{\max}=0.92$ because, with $\alpha=0.3$ and these arm gaps,
switches are rare in 20 000 steps.

### Exercise 4 (★★) — racing with Wilcoxon + Holm

After each instance (once $n\ge n_{\min}=5$), take the candidate with the best mean rank as control. Test every other
alive candidate against it with a one-sided Wilcoxon signed-rank test (alternative: the other has larger costs).
Adjust the $p$-values by Holm and drop the candidates with adjusted $p\lt\alpha$.

**Result** on the notebook's DE tuning task (same 60 instances, same order, same seeds):

| race | runs | survivors | chosen configuration |
|---|---|---|---|
| Friedman–Conover (notebook) | 303 | $(0.5,0.7)$, $(0.5,1.0)$, $(0.2,0.3)$ | $(F,CR)=(0.5,0.7)$ |
| Wilcoxon–Holm | 357 | $(0.5,0.7)$, $(0.5,1.0)$ | $(F,CR)=(0.5,0.7)$ |

The chosen configuration is the brute-force best (full-data mean rank 3.58). Pairwise tests against the control spend
more runs early, because nothing can be eliminated before the paired test has enough non-zero differences. They end
more decisively (two survivors instead of three).

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import BudgetedObjective, BENCHMARKS, shifted_rotated
from utils import random_rotation, ellipsoid
# --- ask-tell core of Notebook 1 (compact copy) ---
class AskTell:
    name = "base"
    def __init__(self, d, lo, hi, budget, rng):
        self.d, self.lo, self.hi, self.budget, self.rng = d, lo, hi, budget, rng
        self.span = (hi - lo) if lo is not None else 1.0
        self.n_told = 0
    def clip(self, X): return X if self.lo is None else np.clip(X, self.lo, self.hi)
    def uniform(self, n): return self.rng.uniform(self.lo, self.hi, (n, self.d))
    def progress(self): return min(1.0, self.n_told / self.budget)

def distinct_indices(rng, n, k, n_pool=None, exclude=None):
    n_pool = n if n_pool is None else n_pool
    keys = rng.random((n, n_pool)); keys[np.arange(n), np.arange(n)] = np.inf
    if exclude is not None: keys[np.arange(n), exclude] = np.inf
    return np.argsort(keys, axis=1)[:, :k]

def optimize(opt, f, budget, callback=None):
    obj = BudgetedObjective(f, budget)
    while obj.remaining > 0:
        X = opt.ask()
        if len(X) > obj.remaining: obj(X[: obj.remaining]); break
        y = obj(X); opt.tell(X, y); opt.n_told += len(y)
        if callback is not None: callback(opt, X, y, obj)
    return obj
class DE(AskTell):
    name = "DE"
    def __init__(self, d, lo, hi, budget, rng, n=50, F=0.5, CR=0.9):
        super().__init__(d, lo, hi, budget, rng); self.n, self.F, self.CR, self.P, self.fP = n, F, CR, None, None
    def ask(self):
        if self.P is None: return self.uniform(self.n)
        n, d = self.n, self.d; idx = distinct_indices(self.rng, n, 3)
        V = self.P[idx[:, 0]] + self.F * (self.P[idx[:, 1]] - self.P[idx[:, 2]])
        mask = self.rng.random((n, d)) < self.CR; mask[np.arange(n), self.rng.integers(0, d, n)] = True
        return self.clip(np.where(mask, V, self.P))
    def tell(self, X, y):
        if self.P is None: self.P, self.fP = X.copy(), y.copy(); return
        imp = y <= self.fP; self.P[imp], self.fP[imp] = X[imp], y[imp]
# --- Notebook 3's DE tuning task: 16 (F, CR) candidates, 60 heterogeneous instances in d = 5, 1500 evaluations ---
d_t, n_inst, B_t = 5, 60, 1500
cands = [(F, CR) for F in [0.2, 0.5, 0.8, 1.0] for CR in [0.0, 0.3, 0.7, 1.0]]
ir = np.random.default_rng(2024); instances = []
for i in range(n_inst):
    kind = ["ellipsoid", "rastrigin", "ackley"][i % 3]; b = BENCHMARKS[kind]
    base = (lambda X: ellipsoid(X, 1e3)) if kind == "ellipsoid" else b.f
    instances.append((shifted_rotated(base, ir.uniform(0.5 * b.lower, 0.5 * b.upper, d_t), random_rotation(d_t, ir)), b.lower, b.upper))
order = ir.permutation(n_inst); instances = [instances[j] for j in order]

cache = {}
def run_FC(F, CR, i):
    """Final error of DE(F, CR) on instance i (fixed seed per instance, cached)."""
    key = (round(F, 6), round(CR, 6), i)
    if key not in cache:
        f, lo, hi = instances[i]
        cache[key] = optimize(DE(d_t, lo, hi, B_t, np.random.default_rng(1000 * i + 7), n=20, F=F, CR=CR), f, B_t).best_f
    return cache[key]

def average_ranks(row):
    order = np.argsort(row, kind="mergesort"); ranks = np.empty(len(row)); sr = row[order]; i = 0
    while i < len(row):
        j = i
        while j + 1 < len(row) and sr[j + 1] == sr[i]: j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return ranks

def friedman_test(Y):
    n, k = Y.shape; Rm = np.array([average_ranks(r) for r in Y]); Rj = Rm.sum(0)
    chi2 = 12.0 / (n * k * (k + 1)) * np.sum(Rj**2) - 3.0 * n * (k + 1)
    chi2 /= 1.0 - sum(np.sum(c**3 - c) for c in (np.unique(r, return_counts=True)[1] for r in Y)) / (n * k * (k**2 - 1))
    return chi2, stats.chi2.sf(chi2, k - 1), Rm

def conover_critical(Rm, alpha):
    n, k = Rm.shape
    return stats.t.ppf(1 - alpha / 2, (n - 1) * (k - 1)) * np.sqrt(2 * (n * np.sum(Rm**2) - np.sum(Rm.sum(0)**2)) / ((n - 1) * (k - 1)))

def f_race(configs, inst_ids, max_runs=np.inf, n_min=5, alpha=0.05):
    """Notebook 3's F-Race (Friedman + Conover) on arbitrary (F, CR) configurations; returns alive configs ranked, runs used."""
    alive, Y, used = list(range(len(configs))), [], 0
    for i in inst_ids:
        if used + len(alive) > max_runs: break
        row = np.full(len(configs), np.nan)
        for c in alive: row[c] = run_FC(*configs[c], i); used += 1
        Y.append(row)
        if len(Y) >= n_min and len(alive) > 1:
            _, p, Rm = friedman_test(np.array(Y)[:, alive])
            if p < alpha:
                Rj = Rm.sum(0); crit = conover_critical(Rm, alpha); alive = [c for c, r in zip(alive, Rj) if r - Rj.min() <= crit]
        if len(alive) == 1: break
    mr = np.array([average_ranks(r) for r in np.array(Y)[:, alive]]).mean(0)
    return [alive[j] for j in np.argsort(mr, kind="stable")], used

def holm(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(np.argsort(p)): run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj

def wilcoxon_race(configs, inst_ids, n_min=5, alpha=0.05):
    """Race: control = best mean rank; one-sided Wilcoxon of every other survivor vs the control, Holm-adjusted."""
    alive, Y, used = list(range(len(configs))), [], 0
    for i in inst_ids:
        row = np.full(len(configs), np.nan)
        for c in alive: row[c] = run_FC(*configs[c], i); used += 1
        Y.append(row)
        if len(Y) >= n_min and len(alive) > 1:
            A = np.array(Y); mr = np.array([average_ranks(r) for r in A[:, alive]]).mean(0)
            b = alive[int(np.argmin(mr))]; others = [c for c in alive if c != b]
            p = [stats.wilcoxon(A[:, c], A[:, b], alternative="greater").pvalue for c in others]
            alive = [b] + [c for c, q in zip(others, holm(p)) if q >= alpha]
        if len(alive) == 1: break
    mr = np.array([average_ranks(r) for r in np.array(Y)[:, alive]]).mean(0)
    return [alive[j] for j in np.argsort(mr, kind="stable")], used

full = np.array([[run_FC(*c, i) for c in cands] for i in range(n_inst)])        # brute force, for the ground truth
mean_rank = np.array([average_ranks(r) for r in full]).mean(0)
for name, race in [("Friedman-Conover", f_race), ("Wilcoxon-Holm", wilcoxon_race)]:
    ranked, used = race(cands, range(n_inst))
    print(f"{name:17s}: {used} runs, survivors {[cands[j] for j in ranked]}, chosen {cands[ranked[0]]} "
          f"(full-data mean rank {mean_rank[ranked[0]]:.2f}; brute-force best {cands[int(np.argmin(mean_rank))]})")
```

```text
Friedman-Conover : 303 runs, survivors [(0.5, 0.7), (0.5, 1.0), (0.2, 0.3)], chosen (0.5, 0.7) (full-data mean rank 3.58; brute-force best (0.5, 0.7))
Wilcoxon-Holm    : 357 runs, survivors [(0.5, 0.7), (0.5, 1.0)], chosen (0.5, 0.7) (full-data mean rank 3.58; brute-force best (0.5, 0.7))
```

### Exercise 5 (★★★) — a miniature irace

Sample 16 configurations from truncated normals, initially centred at $(0.55,0.5)$ with standard deviations
$(0.45,0.5)$ on $[0.05,1.2]\times[0,1]$. Race them with the notebook's Friedman–Conover race on a fresh random order of
the 60 instances, with a run budget of (remaining runs)/(remaining iterations). Keep the 3 best as elites. Resample 13
new configurations around randomly chosen elites and shrink the standard deviations by $0.6$ per iteration. Repeat for
4 iterations. In the block, `f_race` is the notebook's race generalised to arbitrary real configurations and a run
budget; runs are cached per $(F,CR,\text{instance})$ with the notebook's seeds.

**Result.** With a budget equal to the grid race's 303 runs, the mini-irace used 288 runs and chose
$(F,CR)=(0.45,0.69)$. Evaluated on all 60 instances against the grid race's choice $(0.5,0.7)$, it wins on 36 of 60
(paired Wilcoxon $p=0.16$; median $\log_{10}$ cost ratio $-0.10$). It lands in the same region, slightly better but not
significantly so, and it did not need a hand-made grid.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import BudgetedObjective, BENCHMARKS, shifted_rotated
from utils import random_rotation, ellipsoid
# --- ask-tell core of Notebook 1 (compact copy) ---
class AskTell:
    name = "base"
    def __init__(self, d, lo, hi, budget, rng):
        self.d, self.lo, self.hi, self.budget, self.rng = d, lo, hi, budget, rng
        self.span = (hi - lo) if lo is not None else 1.0
        self.n_told = 0
    def clip(self, X): return X if self.lo is None else np.clip(X, self.lo, self.hi)
    def uniform(self, n): return self.rng.uniform(self.lo, self.hi, (n, self.d))
    def progress(self): return min(1.0, self.n_told / self.budget)

def distinct_indices(rng, n, k, n_pool=None, exclude=None):
    n_pool = n if n_pool is None else n_pool
    keys = rng.random((n, n_pool)); keys[np.arange(n), np.arange(n)] = np.inf
    if exclude is not None: keys[np.arange(n), exclude] = np.inf
    return np.argsort(keys, axis=1)[:, :k]

def optimize(opt, f, budget, callback=None):
    obj = BudgetedObjective(f, budget)
    while obj.remaining > 0:
        X = opt.ask()
        if len(X) > obj.remaining: obj(X[: obj.remaining]); break
        y = obj(X); opt.tell(X, y); opt.n_told += len(y)
        if callback is not None: callback(opt, X, y, obj)
    return obj
class DE(AskTell):
    name = "DE"
    def __init__(self, d, lo, hi, budget, rng, n=50, F=0.5, CR=0.9):
        super().__init__(d, lo, hi, budget, rng); self.n, self.F, self.CR, self.P, self.fP = n, F, CR, None, None
    def ask(self):
        if self.P is None: return self.uniform(self.n)
        n, d = self.n, self.d; idx = distinct_indices(self.rng, n, 3)
        V = self.P[idx[:, 0]] + self.F * (self.P[idx[:, 1]] - self.P[idx[:, 2]])
        mask = self.rng.random((n, d)) < self.CR; mask[np.arange(n), self.rng.integers(0, d, n)] = True
        return self.clip(np.where(mask, V, self.P))
    def tell(self, X, y):
        if self.P is None: self.P, self.fP = X.copy(), y.copy(); return
        imp = y <= self.fP; self.P[imp], self.fP[imp] = X[imp], y[imp]
# --- Notebook 3's DE tuning task: 16 (F, CR) candidates, 60 heterogeneous instances in d = 5, 1500 evaluations ---
d_t, n_inst, B_t = 5, 60, 1500
cands = [(F, CR) for F in [0.2, 0.5, 0.8, 1.0] for CR in [0.0, 0.3, 0.7, 1.0]]
ir = np.random.default_rng(2024); instances = []
for i in range(n_inst):
    kind = ["ellipsoid", "rastrigin", "ackley"][i % 3]; b = BENCHMARKS[kind]
    base = (lambda X: ellipsoid(X, 1e3)) if kind == "ellipsoid" else b.f
    instances.append((shifted_rotated(base, ir.uniform(0.5 * b.lower, 0.5 * b.upper, d_t), random_rotation(d_t, ir)), b.lower, b.upper))
order = ir.permutation(n_inst); instances = [instances[j] for j in order]

cache = {}
def run_FC(F, CR, i):
    """Final error of DE(F, CR) on instance i (fixed seed per instance, cached)."""
    key = (round(F, 6), round(CR, 6), i)
    if key not in cache:
        f, lo, hi = instances[i]
        cache[key] = optimize(DE(d_t, lo, hi, B_t, np.random.default_rng(1000 * i + 7), n=20, F=F, CR=CR), f, B_t).best_f
    return cache[key]

def average_ranks(row):
    order = np.argsort(row, kind="mergesort"); ranks = np.empty(len(row)); sr = row[order]; i = 0
    while i < len(row):
        j = i
        while j + 1 < len(row) and sr[j + 1] == sr[i]: j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return ranks

def friedman_test(Y):
    n, k = Y.shape; Rm = np.array([average_ranks(r) for r in Y]); Rj = Rm.sum(0)
    chi2 = 12.0 / (n * k * (k + 1)) * np.sum(Rj**2) - 3.0 * n * (k + 1)
    chi2 /= 1.0 - sum(np.sum(c**3 - c) for c in (np.unique(r, return_counts=True)[1] for r in Y)) / (n * k * (k**2 - 1))
    return chi2, stats.chi2.sf(chi2, k - 1), Rm

def conover_critical(Rm, alpha):
    n, k = Rm.shape
    return stats.t.ppf(1 - alpha / 2, (n - 1) * (k - 1)) * np.sqrt(2 * (n * np.sum(Rm**2) - np.sum(Rm.sum(0)**2)) / ((n - 1) * (k - 1)))

def f_race(configs, inst_ids, max_runs=np.inf, n_min=5, alpha=0.05):
    """Notebook 3's F-Race (Friedman + Conover) on arbitrary (F, CR) configurations; returns alive configs ranked, runs used."""
    alive, Y, used = list(range(len(configs))), [], 0
    for i in inst_ids:
        if used + len(alive) > max_runs: break
        row = np.full(len(configs), np.nan)
        for c in alive: row[c] = run_FC(*configs[c], i); used += 1
        Y.append(row)
        if len(Y) >= n_min and len(alive) > 1:
            _, p, Rm = friedman_test(np.array(Y)[:, alive])
            if p < alpha:
                Rj = Rm.sum(0); crit = conover_critical(Rm, alpha); alive = [c for c, r in zip(alive, Rj) if r - Rj.min() <= crit]
        if len(alive) == 1: break
    mr = np.array([average_ranks(r) for r in np.array(Y)[:, alive]]).mean(0)
    return [alive[j] for j in np.argsort(mr, kind="stable")], used

def mini_irace(total_runs, iters=4, n_conf=16, n_elite=3, seed=0):
    r = np.random.default_rng(seed); mu, sd = np.array([0.55, 0.5]), np.array([0.45, 0.5])
    lo, hi = np.array([0.05, 0.0]), np.array([1.2, 1.0]); elites, used = [], 0
    for it in range(iters):
        new = []
        while len(new) < n_conf - len(elites):                   # truncated normal by rejection
            c = r.normal(mu if not elites else np.array(elites[r.integers(len(elites))]), sd)
            if np.all((c >= lo) & (c <= hi)): new.append(tuple(c))
        confs = elites + new
        ranked, u = f_race(confs, r.permutation(n_inst), (total_runs - used) // (iters - it))
        used += u; elites = [confs[j] for j in ranked[:n_elite]]; sd = sd * 0.6
    return elites[0], used

grid_ranked, grid_runs = f_race(cands, range(n_inst))                 # the notebook's grid race
best_grid = cands[grid_ranked[0]]
best_ir, used_ir = mini_irace(grid_runs)
c_ir = np.array([run_FC(*best_ir, i) for i in range(n_inst)]); c_gr = np.array([run_FC(*best_grid, i) for i in range(n_inst)])
print(f"grid race: {grid_runs} runs, chosen {best_grid};  mini-irace: {used_ir} runs, chosen (F, CR) = ({best_ir[0]:.2f}, {best_ir[1]:.2f})")
print(f"on all 60 instances: mini-irace better on {np.sum(c_ir < c_gr)}/60, paired Wilcoxon p = {stats.wilcoxon(c_ir, c_gr).pvalue:.2f}, "
      f"median log10(cost ratio) = {np.median(np.log10(c_ir / c_gr)):.2f}")
```

```text
grid race: 303 runs, chosen (0.5, 0.7);  mini-irace: 288 runs, chosen (F, CR) = (0.45, 0.69)
on all 60 instances: mini-irace better on 36/60, paired Wilcoxon p = 0.16, median log10(cost ratio) = -0.10
```

### Exercise 6 (★★★) — move acceptance in the bin-packing hyper-heuristic

The acceptance rules compared:

- IE: improving-or-equal.
- SA: accept a worse solution with probability $e^{-\Delta/T}$, with $T$ decreasing linearly from $T_0$ to 0.
- LA: late acceptance (Burke & Bykov 2017). Accept if $f(y)\le f(x)$ or $f(y)\le h[t\bmod L]$, where $h$ stores the
  current objective of $L$ iterations ago.

**Result** (the notebook's 8 triplet instances with $K=20$, 5000 evaluations): *no* combination packs any instance
into $L_1$ bins.

| acceptance | selection | instances solved | median best − $L_1$ |
|---|---|---|---|
| IE | uniform / AP / SW-UCB | 0/8 each | 1.067 / 1.067 / 1.068 |
| SA ($T_0=0.02$) | uniform / AP / SW-UCB | 0/8 each | 1.084 / 1.085 / 1.086 |
| LA ($L=50$) | uniform / AP / SW-UCB | 0/8 each | 1.070 / 1.068 / 1.069 |

A sensitivity scan with AP (second block) also solved nothing: $T_0\in\lbrace0.005,0.1,0.5\rbrace$ and
$L\in\lbrace10,200,1000\rbrace$ gave 0/8 each, with a best gap of $1.067$. The honest answer is therefore "none within
5000 evaluations". The bottleneck is the representation, not the acceptance rule. A permutation decoded by first fit
reaches the 20-bin packings, where every bin is filled exactly by one triplet, only through a vanishingly small set of
orders, and the four LLHs make small random moves in order space. Falkenauer's grouping GA works directly on *bins*
(groups) for this reason. A decisive LLH here would be "empty the least-filled bin and re-insert its items with best
fit". This is an example of the domain knowledge that the hyper-heuristic's domain barrier deliberately hides.

```python
import numpy as np, random
Cap = 1000
def triplet_instance(K, r):
    items = []
    for _ in range(K):
        a = int(r.integers(380, 491)); b = int(r.integers(250, Cap - a - 250 + 1)); items += [a, b, Cap - a - b]
    r.shuffle(items); return items

def first_fit(order, w):
    res, bins = [], []
    for i in order:
        wi = w[i]
        for b in range(len(res)):
            if res[b] >= wi: res[b] -= wi; bins[b].append(i); break
        else: res.append(Cap - wi); bins.append([i])
    return res, bins

def bp_cost(order, w):
    res, bins = first_fit(order, w); N = len(res)
    return N + 1 - sum(((Cap - r) / Cap) ** 2 for r in res) / N, bins, res

def apply_llh(k, order, bins, res, R):          # swap, to-front, empty-bin, reverse (Notebook 3)
    o = order[:]; n = len(o)
    if k == 0: i, j = R.sample(range(n), 2); o[i], o[j] = o[j], o[i]
    elif k == 1: i = R.randrange(n); o.insert(0, o.pop(i))
    elif k == 2:
        b = max(range(len(res)), key=res.__getitem__); front = bins[b][:]; R.shuffle(front); s = set(front)
        o = front + [x for x in o if x not in s]
    else: i, j = sorted(R.sample(range(n + 1), 2)); o[i:j] = o[i:j][::-1]
    return o

class UniformOp:
    def __init__(self, K, rng): self.K, self.rng = K, rng
    def select(self): return int(self.rng.integers(self.K))
    def update(self, i, r): pass

class AdaptivePursuit:
    def __init__(self, K, rng, p_min=0.05, alpha=0.3, beta=0.3):
        self.K, self.rng, self.p_min, self.alpha, self.beta, self.q, self.p = K, rng, p_min, alpha, beta, np.ones(K), np.full(K, 1 / K)
    def select(self): return int(self.rng.choice(self.K, p=self.p))
    def update(self, i, r):
        self.q[i] += self.alpha * (r - self.q[i])
        t = np.full(self.K, self.p_min); t[np.argmax(self.q)] = 1 - (self.K - 1) * self.p_min; self.p += self.beta * (t - self.p)

class SlidingWindowUCB:
    def __init__(self, K, rng, W=100, C=0.5):
        self.K, self.rng, self.W, self.C, self.ops, self.rs, self.t = K, rng, W, C, np.full(W, -1), np.zeros(W), 0
    def select(self):
        m = self.ops >= 0; n = np.bincount(self.ops[m], minlength=self.K).astype(float)
        s = np.bincount(self.ops[m], weights=self.rs[m], minlength=self.K)
        if (n == 0).any(): z = np.flatnonzero(n == 0); return int(z[self.rng.integers(len(z))])
        return int(np.argmax(s / n / max(1e-12, self.rs[m].max()) + self.C * np.sqrt(2 * np.log(n.sum()) / n)))
    def update(self, i, r): self.ops[self.t % self.W], self.rs[self.t % self.W] = i, r; self.t += 1

def hh_accept(w, selector, budget, seed, accept="IE", L=50, T0=0.02):
    R = random.Random(seed); order = list(range(len(w))); R.shuffle(order)
    fx, bins, res = bp_cost(order, w); hist = [fx] * L; best = fx
    for t in range(budget - 1):
        k = selector.select(); o = apply_llh(k, order, bins, res, R); fy, b2, r2 = bp_cost(o, w)
        selector.update(k, max(0.0, fx - fy))
        if accept == "IE": ok = fy <= fx
        elif accept == "SA": ok = fy <= fx or R.random() < np.exp(-(fy - fx) / (T0 * (1 - t / budget) + 1e-9))
        else: ok = fy <= fx or fy <= hist[t % L]                     # late acceptance
        if ok: order, fx, bins, res = o, fy, b2, r2
        if accept == "LA": hist[t % L] = fx
        best = min(best, fx)
    return best

insts = [triplet_instance(20, np.random.default_rng(100 + s)) for s in range(8)]    # OPT = L1 = 20 bins
sels = {"uniform": UniformOp, "AP": AdaptivePursuit, "SW-UCB": SlidingWindowUCB}
def study(accept, make, **kw):
    gaps = np.array([hh_accept(w, make(4, np.random.default_rng(s)), 5000, s, accept=accept, **kw) - 20 for s, w in enumerate(insts)])
    return int(np.sum(gaps < 1)), gaps
for acc in ["IE", "SA", "LA"]:
    out = [(n, *study(acc, mk)) for n, mk in sels.items()]
    print(f"{acc}: " + "; ".join(f"{n} solved {k}/8, median gap {np.median(g):.3f}" for n, k, g in out))
```

```text
IE: uniform solved 0/8, median gap 1.067; AP solved 0/8, median gap 1.067; SW-UCB solved 0/8, median gap 1.068
SA: uniform solved 0/8, median gap 1.084; AP solved 0/8, median gap 1.085; SW-UCB solved 0/8, median gap 1.086
LA: uniform solved 0/8, median gap 1.070; AP solved 0/8, median gap 1.068; SW-UCB solved 0/8, median gap 1.069
```

```python
import numpy as np, random
Cap = 1000
def triplet_instance(K, r):
    items = []
    for _ in range(K):
        a = int(r.integers(380, 491)); b = int(r.integers(250, Cap - a - 250 + 1)); items += [a, b, Cap - a - b]
    r.shuffle(items); return items

def first_fit(order, w):
    res, bins = [], []
    for i in order:
        wi = w[i]
        for b in range(len(res)):
            if res[b] >= wi: res[b] -= wi; bins[b].append(i); break
        else: res.append(Cap - wi); bins.append([i])
    return res, bins

def bp_cost(order, w):
    res, bins = first_fit(order, w); N = len(res)
    return N + 1 - sum(((Cap - r) / Cap) ** 2 for r in res) / N, bins, res

def apply_llh(k, order, bins, res, R):          # swap, to-front, empty-bin, reverse (Notebook 3)
    o = order[:]; n = len(o)
    if k == 0: i, j = R.sample(range(n), 2); o[i], o[j] = o[j], o[i]
    elif k == 1: i = R.randrange(n); o.insert(0, o.pop(i))
    elif k == 2:
        b = max(range(len(res)), key=res.__getitem__); front = bins[b][:]; R.shuffle(front); s = set(front)
        o = front + [x for x in o if x not in s]
    else: i, j = sorted(R.sample(range(n + 1), 2)); o[i:j] = o[i:j][::-1]
    return o

class UniformOp:
    def __init__(self, K, rng): self.K, self.rng = K, rng
    def select(self): return int(self.rng.integers(self.K))
    def update(self, i, r): pass

class AdaptivePursuit:
    def __init__(self, K, rng, p_min=0.05, alpha=0.3, beta=0.3):
        self.K, self.rng, self.p_min, self.alpha, self.beta, self.q, self.p = K, rng, p_min, alpha, beta, np.ones(K), np.full(K, 1 / K)
    def select(self): return int(self.rng.choice(self.K, p=self.p))
    def update(self, i, r):
        self.q[i] += self.alpha * (r - self.q[i])
        t = np.full(self.K, self.p_min); t[np.argmax(self.q)] = 1 - (self.K - 1) * self.p_min; self.p += self.beta * (t - self.p)

class SlidingWindowUCB:
    def __init__(self, K, rng, W=100, C=0.5):
        self.K, self.rng, self.W, self.C, self.ops, self.rs, self.t = K, rng, W, C, np.full(W, -1), np.zeros(W), 0
    def select(self):
        m = self.ops >= 0; n = np.bincount(self.ops[m], minlength=self.K).astype(float)
        s = np.bincount(self.ops[m], weights=self.rs[m], minlength=self.K)
        if (n == 0).any(): z = np.flatnonzero(n == 0); return int(z[self.rng.integers(len(z))])
        return int(np.argmax(s / n / max(1e-12, self.rs[m].max()) + self.C * np.sqrt(2 * np.log(n.sum()) / n)))
    def update(self, i, r): self.ops[self.t % self.W], self.rs[self.t % self.W] = i, r; self.t += 1

def hh_accept(w, selector, budget, seed, accept="IE", L=50, T0=0.02):
    R = random.Random(seed); order = list(range(len(w))); R.shuffle(order)
    fx, bins, res = bp_cost(order, w); hist = [fx] * L; best = fx
    for t in range(budget - 1):
        k = selector.select(); o = apply_llh(k, order, bins, res, R); fy, b2, r2 = bp_cost(o, w)
        selector.update(k, max(0.0, fx - fy))
        if accept == "IE": ok = fy <= fx
        elif accept == "SA": ok = fy <= fx or R.random() < np.exp(-(fy - fx) / (T0 * (1 - t / budget) + 1e-9))
        else: ok = fy <= fx or fy <= hist[t % L]                     # late acceptance
        if ok: order, fx, bins, res = o, fy, b2, r2
        if accept == "LA": hist[t % L] = fx
        best = min(best, fx)
    return best

insts = [triplet_instance(20, np.random.default_rng(100 + s)) for s in range(8)]    # OPT = L1 = 20 bins
sels = {"uniform": UniformOp, "AP": AdaptivePursuit, "SW-UCB": SlidingWindowUCB}
def study(accept, make, **kw):
    gaps = np.array([hh_accept(w, make(4, np.random.default_rng(s)), 5000, s, accept=accept, **kw) - 20 for s, w in enumerate(insts)])
    return int(np.sum(gaps < 1)), gaps
for acc, kw in [("SA", dict(T0=0.005)), ("SA", dict(T0=0.1)), ("SA", dict(T0=0.5)), ("LA", dict(L=10)), ("LA", dict(L=200)), ("LA", dict(L=1000))]:
    k, g = study(acc, AdaptivePursuit, **kw)
    print(f"AP + {acc} {kw}: solved {k}/8, median gap {np.median(g):.3f}, best gap {g.min():.3f}")
```

```text
AP + SA {'T0': 0.005}: solved 0/8, median gap 1.078, best gap 1.068
AP + SA {'T0': 0.1}: solved 0/8, median gap 1.085, best gap 1.071
AP + SA {'T0': 0.5}: solved 0/8, median gap 1.087, best gap 1.075
AP + LA {'L': 10}: solved 0/8, median gap 1.070, best gap 1.067
AP + LA {'L': 200}: solved 0/8, median gap 1.081, best gap 1.068
AP + LA {'L': 1000}: solved 0/8, median gap 1.088, best gap 1.085
```

---

## Notebook 4 — Multi-objective metaheuristics

### Exercise 1 (★) — properties of Pareto dominance

- **Irreflexive:** $a\prec a$ would need $F_m(a)\lt F_m(a)$ for some $m$, which is impossible.
- **Transitive:** from $a\prec b$ and $b\prec c$ we get $F_m(a)\le F_m(b)\le F_m(c)$ for all $m$. For the index $m_0$
  with $F_{m_0}(a)\lt F_{m_0}(b)$, we get $F_{m_0}(a)\lt F_{m_0}(c)$. Hence $a\prec c$.
- **Asymmetric:** $a\prec b\prec a$ would give $a\prec a$ by transitivity, contradicting irreflexivity. So dominance is
  a strict partial order.
- **$\mathcal F_1\ne\emptyset$:** let $a$ minimise $\sum_mF_m$ over the finite non-empty set. If some $b\prec a$, then
  $\sum_mF_m(b)\lt\sum_mF_m(a)$, which contradicts minimality. So $a$ is non-dominated.

### Exercise 2 (★★) — Tchebycheff reaches every Pareto-optimal point; the weighted sum does not

Let $x^{\ast}$ be Pareto optimal and $z^{\ast}$ strictly below the ideal point, so $F_m(x)-z^{\ast}_m\gt0$ for all $x$
and $m$. Choose $\lambda_m=1/(F_m(x^{\ast})-z^{\ast}_m)$ (normalise if you like). Then
$g^{\text{te}}(x^{\ast}\mid\lambda)=1$. Suppose some $x$ had $g^{\text{te}}(x\mid\lambda)\lt1$. Then
$\lambda_m(F_m(x)-z^{\ast}_m)\lt1$ for every $m$, that is $F_m(x)\lt F_m(x^{\ast})$ for all $m$, so $x\prec x^{\ast}$.
That is a contradiction, so $x^{\ast}$ minimises $g^{\text{te}}(\cdot\mid\lambda)$. It need not be the unique
minimiser, but any other minimiser has the same $g^{\text{te}}$ value.

**Weighted sum.** Let the front be $\lbrace(t,h(t)):t\in[a,b]\rbrace$ with $h$ strictly concave (and decreasing), and
take weights $w\ge0$, $w\ne0$.

- If $w_1,w_2\gt0$, a minimiser of $w_1F_1+w_2F_2$ over $\mathcal X$ is Pareto optimal. So it suffices to minimise over
  the front: $\psi(t)=w_1t+w_2h(t)$ is strictly concave on $[a,b]$, and a strictly concave function on an interval
  attains its minimum only at an endpoint.
- If $w_2=0$ (or $w_1=0$), the minimisers over the front are $t=a$ (or $t=b$) only.

In no case is an interior point of the front a minimiser. On the ZDT2 front $f_2=1-f_1^2$ (block at the end of Exercise 3, as in
the notebook), 50 weight vectors reach only the two extreme points, while Tchebycheff reaches 50 distinct points.

### Exercise 3 (★★) — HV is Pareto compliant, IGD is not

Write $D(A)=\bigcup_{a\in A}[a,r]$ for the region dominated by $A$ and bounded by $r$, so that
$\mathrm{HV}(A)=\mathrm{vol}\,D(A)$. If $A$ weakly dominates $B$, every $b\in B$ has some $a\in A$ with $a\le b$
componentwise, hence $[b,r]\subseteq[a,r]\subseteq D(A)$. So $D(B)\subseteq D(A)$, and by monotonicity of the
Lebesgue measure $\mathrm{HV}(A)\ge\mathrm{HV}(B)$.

**IGD counterexample.** Take the front $f_1+f_2=1$ and the reference set $\mathcal R=\lbrace(0,1),(0.5,0.5),(1,0)\rbrace$.
Let $B=\lbrace(0.5,0.55)\rbrace$, which is feasible, and $A=\lbrace(0.45,0.55)\rbrace$, a point on the front that
dominates $B$. Then

$$
\mathrm{IGD}(B)=\tfrac13(0.6727+0.0500+0.7433)=0.4887\ \lt\ \mathrm{IGD}(A)=\tfrac13(0.6364+0.0707+0.7778)=0.4950 .
$$

The dominating set gets the worse IGD. The reason is that IGD measures distance to a finite set of reference points,
not dominance.

```python
import numpy as np
from itertools import product
# Exercise 2: Tchebycheff vs weighted sum on the strictly concave front f2 = 1 - f1^2 (z* = (0, 0))
f1 = np.linspace(0, 1, 2001); front = np.c_[f1, 1 - f1**2]; lams = np.linspace(0.01, 0.99, 50)
ws = {int(np.argmin(front @ np.array([l, 1 - l]))) for l in lams}
te = {int(np.argmin(np.max(np.array([l, 1 - l]) * front, 1))) for l in lams}
print(f"weighted sum reaches {sorted(ws)} (the extremes only); Tchebycheff reaches {len(te)} distinct points")
# Exercise 3: IGD is not Pareto compliant
R = np.array([[0, 1], [0.5, 0.5], [1, 0]]); A, B = np.array([[0.45, 0.55]]), np.array([[0.5, 0.55]])
igd = lambda S: np.mean(np.min(np.linalg.norm(R[:, None] - S[None], axis=2), 1))
print(f"A dominates B: {bool(np.all(A <= B) and np.any(A < B))};  IGD(A) = {igd(A):.4f} > IGD(B) = {igd(B):.4f}")
```

```text
weighted sum reaches [0, 2000] (the extremes only); Tchebycheff reaches 50 distinct points
A dominates B: True;  IGD(A) = 0.4950 > IGD(B) = 0.4887
```

### Exercise 4 (★★) — exact 3-D hypervolume by slicing

Sort the points by $f_3$, giving $z_1\le\dots\le z_n$, and set $z_{n+1}=r_3$. In the slab $z_i\le f_3\lt z_{i+1}$, the
dominated cross-section is the 2-D dominated region of the first $i$ points' $(f_1,f_2)$. Hence

$$
\mathrm{HV}_3=\sum_{i=1}^{n}(z_{i+1}-z_i)\,\mathrm{HV}_2\big(\lbrace(f_1,f_2)(p_1),\dots,(f_1,f_2)(p_i)\rbrace\big).
$$

This costs $O(n^2\log n)$ with a sort-and-sweep 2-D hypervolume (the block vectorises the notebook's sweep with a
running minimum).

**Validation.** Points on the DTLZ2 front ($M=3$) with $r=(1.1,1.1,1.1)$, compared with $10^6$-sample Monte Carlo:

| front points | sliced HV | Monte Carlo | $\lvert\text{diff}\rvert$/SE |
|---|---|---|---|
| 10 | 0.47897 | 0.47885 ± 0.00050 | 0.24 |
| 50 | 0.62873 | 0.62917 ± 0.00062 | 0.70 |
| 200 | 0.72492 | 0.72480 ± 0.00066 | 0.19 |

For 6 points, inclusion–exclusion gives the identical value $0.326689870483$. The whole front is the unit-sphere
octant, and every point of the box $[0,1.1]^3$ outside the unit ball is dominated by its radial projection onto the
front, so its HV is $1.1^3-\pi/6=0.80740$. A $120\times120$ grid of front points gives $0.80176$, approaching this
value from below as it should.

```python
import numpy as np
from itertools import combinations

def hv2d(F, ref):
    """Exact 2-D hypervolume (minimisation): sort by f1, sum the slabs below the running minimum of f2."""
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0: return 0.0
    F = F[np.lexsort((F[:, 1], F[:, 0]))]
    prev = np.r_[ref[1], np.minimum.accumulate(F[:, 1])[:-1]]
    return float(np.sum((ref[0] - F[:, 0]) * np.maximum(prev - F[:, 1], 0)))

def hv3d(F, ref):
    """Slice along f3: in the slab z_i <= f3 < z_{i+1} the dominated cross-section is the 2-D HV of points 1..i."""
    F = F[np.all(F < ref, axis=1)]; F = F[np.argsort(F[:, 2])]; z = np.append(F[:, 2], ref[2]); hv = 0.0
    for i in range(len(F)):
        if z[i + 1] > z[i]: hv += (z[i + 1] - z[i]) * hv2d(F[: i + 1, :2], ref[:2])
    return hv

def hv_mc(F, ref, n=1_000_000, seed=0):
    r = np.random.default_rng(seed); lo = F.min(0); U = lo + r.random((n, 3)) * (ref - lo); dom = np.zeros(n, bool)
    for p in F: dom |= np.all(U >= p, axis=1)
    return dom.mean() * np.prod(ref - lo), np.sqrt(dom.mean() * (1 - dom.mean()) / n) * np.prod(ref - lo)

def dtlz2_front(U):                               # DTLZ2 (M = 3) at g = 0: a point of the unit-sphere octant
    a, b = U[:, 0] * np.pi / 2, U[:, 1] * np.pi / 2
    return np.c_[np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)]

ref = np.array([1.1, 1.1, 1.1]); r = np.random.default_rng(1)
for n_pts in [10, 50, 200]:
    Fp = dtlz2_front(r.random((n_pts, 2))); exact = hv3d(Fp, ref); mc, se = hv_mc(Fp, ref)
    print(f"{n_pts:3d} front points: sliced HV {exact:.5f}, Monte Carlo {mc:.5f} +- {se:.5f}, |diff|/SE = {abs(exact - mc) / se:.2f}")
P6 = dtlz2_front(r.random((6, 2)))
ie = sum((-1) ** (k + 1) * sum(np.prod(ref - np.max(P6[list(c)], 0)) for c in combinations(range(6), k)) for k in range(1, 7))
print(f"6 points: sliced {hv3d(P6, ref):.12f}, inclusion-exclusion {ie:.12f}")
u = np.linspace(0, 1, 120); G = dtlz2_front(np.array(np.meshgrid(u, u)).reshape(2, -1).T)
print(f"whole front: 1.1^3 - pi/6 = {1.331 - np.pi / 6:.5f};  {len(G)} grid points: {hv3d(G, ref):.5f}")
```

```text
 10 front points: sliced HV 0.47897, Monte Carlo 0.47885 +- 0.00050, |diff|/SE = 0.24
 50 front points: sliced HV 0.62873, Monte Carlo 0.62917 +- 0.00062, |diff|/SE = 0.70
200 front points: sliced HV 0.72492, Monte Carlo 0.72480 +- 0.00066, |diff|/SE = 0.19
6 points: sliced 0.326689870483, inclusion-exclusion 0.326689870483
whole front: 1.1^3 - pi/6 = 0.80740;  14400 grid points: 0.80176
```

### Exercise 5 (★★★) — NSGA-III vs NSGA-II vs MOEA/D on DTLZ2

**NSGA-III** (Deb & Jain 2014) keeps NSGA-II's elitist $(\mu+\mu)$ loop and non-dominated sorting. It replaces crowding
in the last front by reference-point niching, in four steps:

1. Normalise with the ideal point and the hyperplane through the extreme points found by the ASF (achievement scalarising function).
2. Associate each member with the nearest reference line from Das–Dennis points.
3. Count how many selected members each niche holds.
4. Repeatedly fill the least-crowded niche, with the member closest to the line if the niche is empty and a random
   member otherwise.

Mating is random, with SBX $\eta_c=30$ and PM $\eta_m=20$. MOEA/D and NSGA-II are the notebook's algorithms,
generalised to $M$ objectives with the same Das–Dennis weight vectors.

DTLZ2 (Deb, Thiele, Laumanns & Zitzler) with $g=\sum_{i\ge M}(x_i-\tfrac12)^2$:
$f_1=(1+g)\prod_{i=1}^{M-1}\cos(\tfrac{\pi}{2}x_i)$ and
$f_m=(1+g)\prod_{i=1}^{M-m}\cos(\tfrac{\pi}{2}x_i)\,\sin(\tfrac{\pi}{2}x_{M-m+1})$ for $m\ge2$. Its front is the
unit-sphere orthant ($g=0$).

**Protocol.** DTLZ2 with $n=M+9$ variables. For $M=3$: $H=12$, 91 reference points, $N=92$, 25 000 evaluations,
5 seeds. For $M=5$: $H=6$, 210 reference points, $N=212$, 50 000 evaluations, 4 seeds (to keep the block under about
two minutes). IGD is computed to normalised Das–Dennis points on the sphere (496 for $M=3$, 1001 for $M=5$). Median
and IQR of IGD:

| $M$ | NSGA-III | NSGA-II | MOEA/D-Tchebycheff |
|---|---|---|---|
| 3 | 0.0527 [0.0526, 0.0527] | 0.0714 [0.0711, 0.0728] | 0.0777 [0.0774, 0.0777] |
| 5 | 0.1597 [0.1595, 0.1598] | 0.2230 [0.2212, 0.2238] | 0.3066 [0.3063, 0.3068] |

Every pairwise Mann–Whitney test against NSGA-III gives the smallest $p$ possible for the sample sizes: $0.008$ (5 vs
5) and $0.029$ (4 vs 4).

NSGA-III is best and its spread across seeds is tiny: reference lines impose a uniform distribution on the front.
NSGA-II's crowding distance loses resolution as $M$ grows. MOEA/D-Tchebycheff ranks last here. With Tchebycheff, the
optimum for weight $\lambda$ lies on the ray with direction $\propto1/\lambda$, not $\lambda$. Uniform $\lambda$
therefore give a non-uniform, boundary-heavy set on the sphere. The usual remedies are the inverse-weight transform or
PBI decomposition, as in NSGA-III's own comparison.

```python
import numpy as np
from scipy import stats
def fast_non_dominated_sort(F):
    le, lt = np.ones((len(F), len(F)), bool), np.zeros((len(F), len(F)), bool)
    for f in F.T: le &= f[:, None] <= f[None, :]; lt |= f[:, None] < f[None, :]
    D = le & lt                                           # D[i, j]: i dominates j
    n = D.sum(0).astype(int); rank = np.full(len(F), -1); cur = np.flatnonzero(n == 0); i = 0
    while len(cur):
        rank[cur] = i; n = n - D[cur].sum(0); n[rank >= 0] = -1; cur = np.flatnonzero(n == 0); i += 1
    return rank

def crowding_distance(F):
    n, M = F.shape; cd = np.zeros(n)
    if n <= 2: return np.full(n, np.inf)
    for m in range(M):
        o = np.argsort(F[:, m], kind="stable"); span = F[o[-1], m] - F[o[0], m]; cd[o[0]] = cd[o[-1]] = np.inf
        if span > 0: cd[o[1:-1]] += (F[o[2:], m] - F[o[:-2], m]) / span
    return cd

def rank_and_crowding(F):
    rank = fast_non_dominated_sort(F); cd = np.zeros(len(F))
    for k in range(rank.max() + 1): idx = np.flatnonzero(rank == k); cd[idx] = crowding_distance(F[idx])
    return rank, cd
def sbx_pm(P1, P2, rng, pc=0.9, eta_c=20.0, eta_m=20.0):
    """SBX + polynomial mutation on [0, 1]^n (Notebook 4)."""
    n, d = P1.shape; u = rng.random((n, d))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
    cross = (rng.random((n, 1)) < pc) & (rng.random((n, d)) < 0.5); beta = np.where(cross, beta, 1.0)
    C1, C2 = 0.5 * ((1 + beta) * P1 + (1 - beta) * P2), 0.5 * ((1 - beta) * P1 + (1 + beta) * P2)
    swap = cross & (rng.random((n, d)) < 0.5); C = np.vstack([np.where(swap, C2, C1), np.where(swap, C1, C2)])
    u = rng.random(C.shape); delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
    return np.clip(C + (rng.random(C.shape) < 1 / d) * delta, 0, 1)
from itertools import combinations
def dtlz2(M, k=10):
    """DTLZ2: n = M - 1 + k variables in [0, 1]; the Pareto front is the unit-sphere orthant."""
    def F(X):
        X = np.atleast_2d(X); g = np.sum((X[:, M - 1:] - 0.5) ** 2, 1)
        c, s = np.cos(X[:, : M - 1] * np.pi / 2), np.sin(X[:, : M - 1] * np.pi / 2); out = np.empty((len(X), M))
        for m in range(M):                     # f_{m+1} = (1+g) cos(x_1)...cos(x_{M-1-m}) sin(x_{M-m})
            v = np.prod(c[:, : M - 1 - m], 1); out[:, m] = (1 + g) * (v * s[:, M - 1 - m] if m > 0 else v)
        return out
    return F

def das_dennis(M, H):
    return np.array([(np.diff(np.r_[-1, c, H + M - 1]) - 1) / H for c in combinations(range(H + M - 1), M - 1)])

def environmental_nsga3(F, N, refs, r):
    rank = fast_non_dominated_sort(F); cnt = np.cumsum(np.bincount(rank)); l = np.searchsorted(cnt, N)
    St = np.flatnonzero(rank <= l)
    if cnt[l] == N: return St
    M = F.shape[1]; Fn = F[St] - F[St].min(0)
    E = np.array([Fn[np.argmin(np.max(Fn / w, 1))] for w in np.eye(M) + 1e-6])   # extreme points (ASF)
    try:
        a = np.linalg.solve(E, np.ones(M)); icpt = 1 / a
        if np.any(a <= 0) or np.any(icpt < 1e-6): raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        icpt = Fn.max(0)                                                        # fallback: nadir of S_t
    Fn = Fn / np.maximum(icpt, 1e-10); U = refs / np.linalg.norm(refs, axis=1, keepdims=True)
    proj = Fn @ U.T; dist = np.sqrt(np.maximum((Fn ** 2).sum(1)[:, None] - proj ** 2, 0))
    pi, dmin = dist.argmin(1), dist.min(1)
    chosen = list(St[rank[St] < l]); last = rank[St] == l                       # last: members of the split front
    niche = np.bincount(pi[rank[St] < l], minlength=len(refs)).astype(float); active = np.ones(len(refs), bool)
    while len(chosen) < N:
        cand = np.flatnonzero(active); j = r.choice(cand[niche[cand] == niche[cand].min()])
        members = np.flatnonzero(last & (pi == j))
        if not len(members): active[j] = False; continue
        pick = members[np.argmin(dmin[members])] if niche[j] == 0 else members[r.integers(len(members))]
        chosen.append(St[pick]); last[pick] = False; niche[j] += 1
    return np.array(chosen)

def nsga3(problem, budget, seed, refs, d):
    r = np.random.default_rng(seed); N = len(refs) + (-len(refs)) % 4
    P = r.random((N, d)); F = problem(P); ev = N
    while ev + N <= budget:
        par = r.integers(0, N, N)                                                 # random mating
        Q = sbx_pm(P[par[: N // 2]], P[par[N // 2:]], r, eta_c=30.0); FQ = problem(Q); ev += N
        RP, RF = np.vstack([P, Q]), np.vstack([F, FQ]); keep = environmental_nsga3(RF, N, refs, r)
        P, F = RP[keep], RF[keep]
    return P, F, ev

def nsga2(problem, budget, seed, N, d):
    r = np.random.default_rng(seed); P = r.random((N, d)); F = problem(P); ev = N; rank, cd = rank_and_crowding(F)
    while ev + N <= budget:
        a, b = r.integers(0, N, (2, N)); par = np.where((rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] > cd[b])), a, b)
        Q = sbx_pm(P[par[: N // 2]], P[par[N // 2:]], r); FQ = problem(Q); ev += N
        RP, RF = np.vstack([P, Q]), np.vstack([F, FQ]); rk, cdi = rank_and_crowding(RF); keep = np.lexsort((-cdi, rk))[:N]
        P, F, rank, cd = RP[keep], RF[keep], rk[keep], cdi[keep]
    return P, F, ev

def moead(problem, budget, seed, W, d, T=20, delta=0.9, n_r=2):
    """Notebook 4's MOEA/D-Tchebycheff (delta, n_r safeguards), with M-objective Das-Dennis weights W."""
    r = np.random.default_rng(seed); N = len(W); W = np.maximum(W, 1e-6)
    B = np.argsort(((W[:, None] - W[None]) ** 2).sum(-1), 1)[:, :T]
    X = r.random((N, d)); F = problem(X); ev = N; z = F.min(0)
    while ev < budget:
        nc = min(N, budget - ev); sub = r.permutation(N)[:nc]; loc = r.random(nc) < delta
        k = np.where(loc, B[sub, r.integers(0, T, nc)], r.integers(0, N, nc)); l = np.where(loc, B[sub, r.integers(0, T, nc)], r.integers(0, N, nc))
        Y = sbx_pm(X[k], X[l], r)[r.integers(0, 2, nc) * nc + np.arange(nc)]; FY = problem(Y); ev += nc
        for c, i in enumerate(sub):
            z = np.minimum(z, FY[c]); o = r.permutation(B[i] if loc[c] else np.arange(N))
            upd = o[np.max(W[o] * np.abs(FY[c] - z), 1) <= np.max(W[o] * np.abs(F[o] - z), 1)][:n_r]; X[upd], F[upd] = Y[c], FY[c]
    return X, F, ev

igd = lambda A, R: np.mean(np.min(np.sqrt(((R[:, None] - A[None]) ** 2).sum(-1)), 1))

def compare(M, H, Href, budget, seeds=5):
    refs = das_dennis(M, H); d = M - 1 + 10; prob = dtlz2(M); N = len(refs) + (-len(refs)) % 4
    R = das_dennis(M, Href); R = R / np.linalg.norm(R, axis=1, keepdims=True)       # reference points on the sphere
    out = {}
    for name, run in [("NSGA-III", lambda s: nsga3(prob, budget, s, refs, d)), ("NSGA-II", lambda s: nsga2(prob, budget, s, N, d)),
                      ("MOEA/D", lambda s: moead(prob, budget, s, refs, d))]:
        v = []
        for s in range(seeds):
            _, F, ev = run(s); assert ev <= budget; v.append(igd(F[fast_non_dominated_sort(F) == 0], R))
        out[name] = np.array(v); q = np.percentile(v, [25, 50, 75])
        print(f"M={M} (|refs|={len(refs)}, N={N}, {budget} evals, {len(R)} IGD points): {name:8s} IGD {q[1]:.4f} [{q[0]:.4f}, {q[2]:.4f}]")
    for other in ["NSGA-II", "MOEA/D"]:
        print(f"   Mann-Whitney NSGA-III vs {other}: p = {stats.mannwhitneyu(out['NSGA-III'], out[other]).pvalue:.3f}")

print("DTLZ2 check: front points satisfy sum f^2 = 1:",
      np.allclose((dtlz2(5)(np.c_[np.random.default_rng(0).random((5, 4)), np.full((5, 10), 0.5)]) ** 2).sum(1), 1))
compare(M=3, H=12, Href=30, budget=25000)
```

```text
DTLZ2 check: front points satisfy sum f^2 = 1: True
M=3 (|refs|=91, N=92, 25000 evals, 496 IGD points): NSGA-III IGD 0.0527 [0.0526, 0.0527]
M=3 (|refs|=91, N=92, 25000 evals, 496 IGD points): NSGA-II  IGD 0.0714 [0.0711, 0.0728]
M=3 (|refs|=91, N=92, 25000 evals, 496 IGD points): MOEA/D   IGD 0.0777 [0.0774, 0.0777]
   Mann-Whitney NSGA-III vs NSGA-II: p = 0.008
   Mann-Whitney NSGA-III vs MOEA/D: p = 0.008
```

```python
import numpy as np
from scipy import stats
def fast_non_dominated_sort(F):
    le, lt = np.ones((len(F), len(F)), bool), np.zeros((len(F), len(F)), bool)
    for f in F.T: le &= f[:, None] <= f[None, :]; lt |= f[:, None] < f[None, :]
    D = le & lt                                           # D[i, j]: i dominates j
    n = D.sum(0).astype(int); rank = np.full(len(F), -1); cur = np.flatnonzero(n == 0); i = 0
    while len(cur):
        rank[cur] = i; n = n - D[cur].sum(0); n[rank >= 0] = -1; cur = np.flatnonzero(n == 0); i += 1
    return rank

def crowding_distance(F):
    n, M = F.shape; cd = np.zeros(n)
    if n <= 2: return np.full(n, np.inf)
    for m in range(M):
        o = np.argsort(F[:, m], kind="stable"); span = F[o[-1], m] - F[o[0], m]; cd[o[0]] = cd[o[-1]] = np.inf
        if span > 0: cd[o[1:-1]] += (F[o[2:], m] - F[o[:-2], m]) / span
    return cd

def rank_and_crowding(F):
    rank = fast_non_dominated_sort(F); cd = np.zeros(len(F))
    for k in range(rank.max() + 1): idx = np.flatnonzero(rank == k); cd[idx] = crowding_distance(F[idx])
    return rank, cd
def sbx_pm(P1, P2, rng, pc=0.9, eta_c=20.0, eta_m=20.0):
    """SBX + polynomial mutation on [0, 1]^n (Notebook 4)."""
    n, d = P1.shape; u = rng.random((n, d))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
    cross = (rng.random((n, 1)) < pc) & (rng.random((n, d)) < 0.5); beta = np.where(cross, beta, 1.0)
    C1, C2 = 0.5 * ((1 + beta) * P1 + (1 - beta) * P2), 0.5 * ((1 - beta) * P1 + (1 + beta) * P2)
    swap = cross & (rng.random((n, d)) < 0.5); C = np.vstack([np.where(swap, C2, C1), np.where(swap, C1, C2)])
    u = rng.random(C.shape); delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
    return np.clip(C + (rng.random(C.shape) < 1 / d) * delta, 0, 1)
from itertools import combinations
def dtlz2(M, k=10):
    """DTLZ2: n = M - 1 + k variables in [0, 1]; the Pareto front is the unit-sphere orthant."""
    def F(X):
        X = np.atleast_2d(X); g = np.sum((X[:, M - 1:] - 0.5) ** 2, 1)
        c, s = np.cos(X[:, : M - 1] * np.pi / 2), np.sin(X[:, : M - 1] * np.pi / 2); out = np.empty((len(X), M))
        for m in range(M):                     # f_{m+1} = (1+g) cos(x_1)...cos(x_{M-1-m}) sin(x_{M-m})
            v = np.prod(c[:, : M - 1 - m], 1); out[:, m] = (1 + g) * (v * s[:, M - 1 - m] if m > 0 else v)
        return out
    return F

def das_dennis(M, H):
    return np.array([(np.diff(np.r_[-1, c, H + M - 1]) - 1) / H for c in combinations(range(H + M - 1), M - 1)])

def environmental_nsga3(F, N, refs, r):
    rank = fast_non_dominated_sort(F); cnt = np.cumsum(np.bincount(rank)); l = np.searchsorted(cnt, N)
    St = np.flatnonzero(rank <= l)
    if cnt[l] == N: return St
    M = F.shape[1]; Fn = F[St] - F[St].min(0)
    E = np.array([Fn[np.argmin(np.max(Fn / w, 1))] for w in np.eye(M) + 1e-6])   # extreme points (ASF)
    try:
        a = np.linalg.solve(E, np.ones(M)); icpt = 1 / a
        if np.any(a <= 0) or np.any(icpt < 1e-6): raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        icpt = Fn.max(0)                                                        # fallback: nadir of S_t
    Fn = Fn / np.maximum(icpt, 1e-10); U = refs / np.linalg.norm(refs, axis=1, keepdims=True)
    proj = Fn @ U.T; dist = np.sqrt(np.maximum((Fn ** 2).sum(1)[:, None] - proj ** 2, 0))
    pi, dmin = dist.argmin(1), dist.min(1)
    chosen = list(St[rank[St] < l]); last = rank[St] == l                       # last: members of the split front
    niche = np.bincount(pi[rank[St] < l], minlength=len(refs)).astype(float); active = np.ones(len(refs), bool)
    while len(chosen) < N:
        cand = np.flatnonzero(active); j = r.choice(cand[niche[cand] == niche[cand].min()])
        members = np.flatnonzero(last & (pi == j))
        if not len(members): active[j] = False; continue
        pick = members[np.argmin(dmin[members])] if niche[j] == 0 else members[r.integers(len(members))]
        chosen.append(St[pick]); last[pick] = False; niche[j] += 1
    return np.array(chosen)

def nsga3(problem, budget, seed, refs, d):
    r = np.random.default_rng(seed); N = len(refs) + (-len(refs)) % 4
    P = r.random((N, d)); F = problem(P); ev = N
    while ev + N <= budget:
        par = r.integers(0, N, N)                                                 # random mating
        Q = sbx_pm(P[par[: N // 2]], P[par[N // 2:]], r, eta_c=30.0); FQ = problem(Q); ev += N
        RP, RF = np.vstack([P, Q]), np.vstack([F, FQ]); keep = environmental_nsga3(RF, N, refs, r)
        P, F = RP[keep], RF[keep]
    return P, F, ev

def nsga2(problem, budget, seed, N, d):
    r = np.random.default_rng(seed); P = r.random((N, d)); F = problem(P); ev = N; rank, cd = rank_and_crowding(F)
    while ev + N <= budget:
        a, b = r.integers(0, N, (2, N)); par = np.where((rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] > cd[b])), a, b)
        Q = sbx_pm(P[par[: N // 2]], P[par[N // 2:]], r); FQ = problem(Q); ev += N
        RP, RF = np.vstack([P, Q]), np.vstack([F, FQ]); rk, cdi = rank_and_crowding(RF); keep = np.lexsort((-cdi, rk))[:N]
        P, F, rank, cd = RP[keep], RF[keep], rk[keep], cdi[keep]
    return P, F, ev

def moead(problem, budget, seed, W, d, T=20, delta=0.9, n_r=2):
    """Notebook 4's MOEA/D-Tchebycheff (delta, n_r safeguards), with M-objective Das-Dennis weights W."""
    r = np.random.default_rng(seed); N = len(W); W = np.maximum(W, 1e-6)
    B = np.argsort(((W[:, None] - W[None]) ** 2).sum(-1), 1)[:, :T]
    X = r.random((N, d)); F = problem(X); ev = N; z = F.min(0)
    while ev < budget:
        nc = min(N, budget - ev); sub = r.permutation(N)[:nc]; loc = r.random(nc) < delta
        k = np.where(loc, B[sub, r.integers(0, T, nc)], r.integers(0, N, nc)); l = np.where(loc, B[sub, r.integers(0, T, nc)], r.integers(0, N, nc))
        Y = sbx_pm(X[k], X[l], r)[r.integers(0, 2, nc) * nc + np.arange(nc)]; FY = problem(Y); ev += nc
        for c, i in enumerate(sub):
            z = np.minimum(z, FY[c]); o = r.permutation(B[i] if loc[c] else np.arange(N))
            upd = o[np.max(W[o] * np.abs(FY[c] - z), 1) <= np.max(W[o] * np.abs(F[o] - z), 1)][:n_r]; X[upd], F[upd] = Y[c], FY[c]
    return X, F, ev

igd = lambda A, R: np.mean(np.min(np.sqrt(((R[:, None] - A[None]) ** 2).sum(-1)), 1))

def compare(M, H, Href, budget, seeds=5):
    refs = das_dennis(M, H); d = M - 1 + 10; prob = dtlz2(M); N = len(refs) + (-len(refs)) % 4
    R = das_dennis(M, Href); R = R / np.linalg.norm(R, axis=1, keepdims=True)       # reference points on the sphere
    out = {}
    for name, run in [("NSGA-III", lambda s: nsga3(prob, budget, s, refs, d)), ("NSGA-II", lambda s: nsga2(prob, budget, s, N, d)),
                      ("MOEA/D", lambda s: moead(prob, budget, s, refs, d))]:
        v = []
        for s in range(seeds):
            _, F, ev = run(s); assert ev <= budget; v.append(igd(F[fast_non_dominated_sort(F) == 0], R))
        out[name] = np.array(v); q = np.percentile(v, [25, 50, 75])
        print(f"M={M} (|refs|={len(refs)}, N={N}, {budget} evals, {len(R)} IGD points): {name:8s} IGD {q[1]:.4f} [{q[0]:.4f}, {q[2]:.4f}]")
    for other in ["NSGA-II", "MOEA/D"]:
        print(f"   Mann-Whitney NSGA-III vs {other}: p = {stats.mannwhitneyu(out['NSGA-III'], out[other]).pvalue:.3f}")

compare(M=5, H=6, Href=10, budget=50000, seeds=4)      # 4 seeds keep the block under ~2 minutes
```

```text
M=5 (|refs|=210, N=212, 50000 evals, 1001 IGD points): NSGA-III IGD 0.1597 [0.1595, 0.1598]
M=5 (|refs|=210, N=212, 50000 evals, 1001 IGD points): NSGA-II  IGD 0.2230 [0.2212, 0.2238]
M=5 (|refs|=210, N=212, 50000 evals, 1001 IGD points): MOEA/D   IGD 0.3066 [0.3063, 0.3068]
   Mann-Whitney NSGA-III vs NSGA-II: p = 0.029
   Mann-Whitney NSGA-III vs MOEA/D: p = 0.029
```

### Exercise 6 (★★★) — a toy NAS problem

**Problem.**

- Search space: an MLP $5\to h_1\to h_2\to h_3\to1$ with $h_j\in\lbrace4,8,\dots,64\rbrace$, i.e. genes $0..15$, $16^3=4096$ architectures.
- Surrogate training: the three hidden ReLU layers are fixed random features, seeded by the architecture. Only the
  output layer is fitted, in closed form by ridge regression ($\lambda=10^{-3}$), on 400 noisy samples of
  $\sin(3x_1)x_2+\cos(2x_3+x_4)-x_5^2$.
- Objective 1: validation MSE on 400 fresh points.
- Objective 2: parameter count (log scale).

The space is small enough to enumerate: exhaustive enumeration finds 24 Pareto-optimal architectures, which gives an
exact ground truth.

**NSGA-II with integer genes.** Population 24, uniform crossover, mutation by $\pm1$ or $\pm2$ steps with probability
$1/3$ per gene, and duplicate genotypes removed before truncation. 600 evaluations, about 15 % of the space:

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| true Pareto-optimal found (of 24) | 9 | 12 | 10 | 9 | 4 |
| returned but not optimal | 10 | 5 | 3 | 9 | 14 |

For comparison, 600 uniformly random distinct architectures contain on average $24\cdot600/4096\approx3.5$ of the
optimal ones. NSGA-II concentrates its sampling on the front, 2–3× better, but the front of this small, noisy-looking
landscape is hard to cover completely. The random-feature surrogate makes neighbouring widths weakly correlated, which
is the NAS situation where weight sharing or learning-curve surrogates earn their keep. The plot at the end of the
block (the 4096 architectures in grey, the true front and the NSGA-II set in (parameters, MSE) log–log axes) shows the
gaps directly.

```python
import numpy as np
import matplotlib.pyplot as plt
def fast_non_dominated_sort(F):
    le, lt = np.ones((len(F), len(F)), bool), np.zeros((len(F), len(F)), bool)
    for f in F.T: le &= f[:, None] <= f[None, :]; lt |= f[:, None] < f[None, :]
    D = le & lt                                           # D[i, j]: i dominates j
    n = D.sum(0).astype(int); rank = np.full(len(F), -1); cur = np.flatnonzero(n == 0); i = 0
    while len(cur):
        rank[cur] = i; n = n - D[cur].sum(0); n[rank >= 0] = -1; cur = np.flatnonzero(n == 0); i += 1
    return rank

def crowding_distance(F):
    n, M = F.shape; cd = np.zeros(n)
    if n <= 2: return np.full(n, np.inf)
    for m in range(M):
        o = np.argsort(F[:, m], kind="stable"); span = F[o[-1], m] - F[o[0], m]; cd[o[0]] = cd[o[-1]] = np.inf
        if span > 0: cd[o[1:-1]] += (F[o[2:], m] - F[o[:-2], m]) / span
    return cd

def rank_and_crowding(F):
    rank = fast_non_dominated_sort(F); cd = np.zeros(len(F))
    for k in range(rank.max() + 1): idx = np.flatnonzero(rank == k); cd[idx] = crowding_distance(F[idx])
    return rank, cd

# data: 5-D regression task
rd = np.random.default_rng(0)
Xtr, Xva = rd.uniform(-1, 1, (400, 5)), rd.uniform(-1, 1, (400, 5))
target = lambda X: np.sin(3 * X[:, 0]) * X[:, 1] + np.cos(2 * X[:, 2] + X[:, 3]) - X[:, 4] ** 2
ytr, yva = target(Xtr) + 0.05 * rd.standard_normal(400), target(Xva)
WIDTHS = 4 * np.arange(1, 17)                      # each layer width in {4, 8, ..., 64}: gene values 0..15

def surrogate(genes, lam=1e-3):
    """Validation MSE of a 3-hidden-layer ReLU MLP whose hidden layers are FIXED random features (seeded by the
    architecture) and whose output layer is fitted in closed form by ridge regression; and its parameter count."""
    h = WIDTHS[np.asarray(genes)]; r = np.random.default_rng(int(genes[0]) * 256 + int(genes[1]) * 16 + int(genes[2]))
    A, V, dims = Xtr, Xva, [5] + list(h)
    for a, b in zip(dims[:-1], dims[1:]):
        Wl = r.standard_normal((a, b)) * np.sqrt(2 / a); bl = 0.1 * r.standard_normal(b)
        A, V = np.maximum(A @ Wl + bl, 0), np.maximum(V @ Wl + bl, 0)
    A1, V1 = np.c_[A, np.ones(len(A))], np.c_[V, np.ones(len(V))]
    beta = np.linalg.solve(A1.T @ A1 + lam * np.eye(A1.shape[1]), A1.T @ ytr)
    params = sum(a * b + b for a, b in zip(dims[:-1], dims[1:])) + dims[-1] + 1
    return np.mean((V1 @ beta - yva) ** 2), params

# ground truth by exhaustive enumeration of all 16^3 = 4096 architectures
allg = np.array(np.meshgrid(*[np.arange(16)] * 3, indexing="ij")).reshape(3, -1).T
allF = np.array([surrogate(gg) for gg in allg]); allF[:, 1] = np.log10(allF[:, 1])
true_nd = fast_non_dominated_sort(allF) == 0
print(f"exhaustive: {len(allg)} architectures, {true_nd.sum()} Pareto-optimal")
lookup = {tuple(gg): f for gg, f in zip(map(tuple, allg), allF)}

def nsga2_int(budget, seed, N=24, pm=1 / 3):
    r = np.random.default_rng(seed); P = r.integers(0, 16, (N, 3)); F = np.array([lookup[tuple(p)] for p in P]); ev = N
    rank, cd = rank_and_crowding(F)
    while ev + N <= budget:
        a, b = r.integers(0, N, (2, N)); par = np.where((rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] > cd[b])), a, b)
        P1, P2 = P[par[: N // 2]], P[par[N // 2:]]
        mask = r.random(P1.shape) < 0.5; C = np.vstack([np.where(mask, P1, P2), np.where(mask, P2, P1)])   # uniform crossover
        step = r.choice([-2, -1, 1, 2], C.shape); C = np.where(r.random(C.shape) < pm, np.clip(C + step, 0, 15), C)  # integer mutation
        FC = np.array([lookup[tuple(c)] for c in C]); ev += N
        RP, RF = np.vstack([P, C]), np.vstack([F, FC]); rk, cdi = rank_and_crowding(RF)
        # remove duplicate genotypes before truncation (keeps the front diverse)
        _, uniq = np.unique(RP, axis=0, return_index=True); dup = np.ones(len(RP), bool); dup[uniq] = False
        keep = np.lexsort((-cdi, rk, dup))[:N]; P, F, rank, cd = RP[keep], RF[keep], rk[keep], cdi[keep]
    return P, F
true_set = {tuple(gg) for gg in allg[true_nd]}
for s in range(5):
    P, F = nsga2_int(600, s)
    found = {tuple(p) for p, k in zip(P, fast_non_dominated_sort(F)) if k == 0}
    print(f"NSGA-II seed {s}, 600 evaluations (15% of the space): {len(found & true_set)} of {len(true_set)} true Pareto-optimal "
          f"architectures found; {len(found - true_set)} returned non-optimal")
P, F = nsga2_int(600, 0)                          # the front plot described in the text
plt.scatter(10 ** allF[:, 1], allF[:, 0], s=3, color="lightgrey", label="all 4096 architectures")
plt.scatter(10 ** allF[true_nd, 1], allF[true_nd, 0], s=25, facecolor="none", edgecolor="k", label="true Pareto front")
plt.scatter(10 ** F[:, 1], F[:, 0], s=10, color="C1", label="NSGA-II, 600 evaluations")
plt.xscale("log"); plt.yscale("log"); plt.xlabel("parameters"); plt.ylabel("validation MSE"); plt.legend(); plt.show()
```

```text
exhaustive: 4096 architectures, 24 Pareto-optimal
NSGA-II seed 0, 600 evaluations (15% of the space): 9 of 24 true Pareto-optimal architectures found; 10 returned non-optimal
NSGA-II seed 1, 600 evaluations (15% of the space): 12 of 24 true Pareto-optimal architectures found; 5 returned non-optimal
NSGA-II seed 2, 600 evaluations (15% of the space): 10 of 24 true Pareto-optimal architectures found; 3 returned non-optimal
NSGA-II seed 3, 600 evaluations (15% of the space): 9 of 24 true Pareto-optimal architectures found; 9 returned non-optimal
NSGA-II seed 4, 600 evaluations (15% of the space): 4 of 24 true Pareto-optimal architectures found; 14 returned non-optimal
```

---

## Lab — Rigorous benchmarking: the grand comparison

### Exercise 1 (★) — ERT for geometric runtimes

Let $T_i$ be i.i.d. geometric with success probability $p$, write $q=1-p$, and let the budget be $B$. Then

$$
\mathbb{E}\min(T,B)=\sum_{k=0}^{B-1}\Pr[T\gt k]=\sum_{k=0}^{B-1}q^k=\frac{1-q^B}{p},\qquad \pi:=\Pr[T\le B]=1-q^B .
$$

$\mathrm{ERT}_n=\frac{\frac1n\sum_i\min(T_i,B)}{\frac1n\sum_i\mathbb 1[T_i\le B]}$. By the strong law of large numbers
the numerator converges to $(1-q^B)/p$ and the denominator to $1-q^B\gt0$. Hence $\mathrm{ERT}_n\to1/p$ a.s., for every
$B\ge1$.

**Finite $n$.** Condition on $n_s=k\ge1$ successes, where $n_s\sim\mathrm{Bin}(n,\pi)$. The successful runtimes are
i.i.d. with mean

$$
\mathbb{E}[T\mid T\le B]=\frac{(1-q^B)/p-Bq^B}{1-q^B},
$$

and each failure contributes $B$, so $\mathrm{ERT}_n=\overline{T}_{\text{succ}}+B\,(n-k)/k$. Hence

$$
\mathbb{E}\big[\mathrm{ERT}_n\mid n_s\ge1\big]=\mathbb{E}[T\mid T\le B]+B\sum_{k=1}^{n}\frac{n-k}{k}\cdot\frac{\binom nk\pi^k(1-\pi)^{n-k}}{1-(1-\pi)^n}.
$$

With probability $(1-\pi)^n$ there is no success and $\mathrm{ERT}=\infty$. Because $1/k$ is convex, Jensen's
inequality makes the estimator biased *upwards*. Example: $p=0.004$, $B=100$, $n=20$ gives $293.0$ from the formula
and $292.9\pm0.3$ by simulation, against $1/p=250$. The bias vanishes as $n\to\infty$. The notebook's needle test with
$n=3000$ runs sits well inside its 4-SE tolerance.

```python
import numpy as np
from scipy import stats
p, B, n = 0.004, 100, 20
q = 1 - p; pi = 1 - q**B
cond_mean = ((1 - q**B) / p - B * q**B) / (1 - q**B)                 # E[T | T <= B]
k = np.arange(1, n + 1); w = stats.binom.pmf(k, n, pi) / (1 - (1 - pi) ** n)
formula = cond_mean + B * np.sum((n - k) / k * w)
r = np.random.default_rng(0); T = r.geometric(p, (400000, n)); ns = (T <= B).sum(1); ok = ns > 0
ert = np.minimum(T, B).sum(1)[ok] / ns[ok]
print(f"E[ERT | n_s >= 1]: formula {formula:.1f}, simulation {ert.mean():.1f} +- {ert.std() / np.sqrt(ok.sum()):.1f}, 1/p = {1 / p:.0f}; "
      f"P(no success) = {(1 - pi) ** n:.1e}")
print(f"Holm/Bonferroni FWER under independence, m = 10: 1 - (1 - 0.05/10)^10 = {1 - (1 - 0.005) ** 10:.4f}")
```

```text
E[ERT | n_s >= 1]: formula 293.0, simulation 292.9 +- 0.3, 1/p = 250; P(no success) = 3.3e-04
Holm/Bonferroni FWER under independence, m = 10: 1 - (1 - 0.05/10)^10 = 0.0489
```

### Exercise 2 (★) — properties of $\rho_s(1)$

For a problem $p$ solved by at least one solver, at least one solver attains $r_{p,s}=1$. Therefore

$$
\sum_s\rho_s(1)=\frac1{\lvert P\rvert}\sum_{p}\#\lbrace s:r_{p,s}=1\rbrace\ \ge\ 1,
$$

with equality iff no problem has a tie for the best cost. Each tie counts all tied solvers as "fastest" and adds to
the sum. Ties are common with integer costs such as iteration counts.

**Dependence on the other solvers.** The ratios are normalised by the best solver *in the comparison*, so adding or
removing a solver changes everyone's profile. For example, take two problems with costs $s_1=(1,4)$ and $s_2=(2,2)$.
The profiles of $s_1$ and $s_2$ are identical: $1/2$ on $[1,2)$ and $1$ from $\tau=2$. Now add $s_3=(\infty,1)$. On
$[1,2)$, $s_1$ has $\rho=1/2$ and $s_2$ has $0$; on $[2,4)$, $s_1$ has $1/2$ and $s_2$ has $1$. A tie turned into a
crossing even though neither $s_1$ nor $s_2$ changed. Profiles compare solvers *within one set* and are not a
per-solver score (cf. Gould & Scott 2016).

### Exercise 3 (★★) — exact null distribution of $W^{+}$

Under $H_0$, the differences are symmetric about 0, continuous, with no ties or zeros. Then the signs are i.i.d. fair
coins independent of the ranks of $\lvert D_i\rvert$, which are a permutation of $1,\dots,n$. Hence

$$
W^{+}=\sum_{j=1}^{n}j\,\varepsilon_j,\qquad\varepsilon_j\stackrel{\text{iid}}{\sim}\mathrm{Bernoulli}(1/2).
$$

$\Pr[W^{+}=w]=c_n(w)/2^n$, where $c_n(w)$ counts the subsets of $\lbrace1,\dots,n\rbrace$ with sum $w$. Splitting on
whether $n$ belongs to the subset gives the recursion

$$
c_k(w)=c_{k-1}(w)+c_{k-1}(w-k),\qquad c_0(0)=1 ,
$$

which is the `counts[k:] += counts[:-k]` update in the lab. The moments follow from linearity and independence:

$$
\mathbb{E}W^{+}=\tfrac12\sum j=\frac{n(n+1)}4,\qquad \mathrm{Var}W^{+}=\tfrac14\sum j^2=\frac{n(n+1)(2n+1)}{24}.
$$

**Ties.** Conditional on the tie pattern, the ranks are replaced by midranks $a_j$. $W^{+}=\sum a_j\varepsilon_j$ keeps
its mean because $\sum a_j=\sum j$, and $\mathrm{Var}=\frac14\sum a_j^2$. Take a tie group of size $t$ occupying ranks
$r+1,\dots,r+t$. Replacing them by their mean reduces $\sum a^2$ by $\sum_{i=1}^{t}(i-\frac{t+1}2)^2=\frac{t^3-t}{12}$.
Hence

$$
\mathrm{Var}W^{+}=\frac{n(n+1)(2n+1)}{24}-\sum_{\text{groups}}\frac{t^3-t}{48}.
$$

### Exercise 4 (★★) — Holm controls the FWER under arbitrary dependence

Let $I_0$ be the set of true nulls, with $m_0=\lvert I_0\rvert\ge1$. Sort the $p$-values and let $j$ be the position of
the smallest true-null $p$-value, $p_{(j)}=\min_{i\in I_0}p_i$. At most $m-m_0$ hypotheses come before it, so
$j\le m-m_0+1$, i.e. $m-j+1\ge m_0$. Holm is step-down, so any false rejection requires rejecting $H_{(j)}$. That
requires $p_{(j)}\le\alpha/(m-j+1)\le\alpha/m_0$. Therefore

$$
\mathrm{FWER}\le\Pr\Big[\min_{i\in I_0}p_i\le\frac{\alpha}{m_0}\Big]\le\sum_{i\in I_0}\Pr\Big[p_i\le\frac{\alpha}{m_0}\Big]\le m_0\cdot\frac{\alpha}{m_0}=\alpha .
$$

Only the union bound and the validity of each null $p$-value ($\Pr[p_i\le u]\le u$) are used, so no assumption on the
dependence between tests is needed. Under independence and the global null the FWER is exactly
$1-(1-\alpha/m)^m=0.0489$ for $m=10$ (printed by the Exercise 1 block); the lab measures $0.051$.

### Exercise 5 (★★) — adding JADE and the $(1+1)$-ES, $B=1000d$, 10 instances

The lab's instance generator, protocol and statistics are used with 9 algorithms: 6 functions × 10 instances
$=60$ blocks, 540 runs. The block keeps only the final error of each run (a trace-free copy of the generic loop), which
makes it a few times faster than the lab's `optimize`; it still takes one to a few minutes.

- Friedman: $\chi^2=295.8$, $p=3\cdot10^{-59}$. Nemenyi CD ($k=9$, $N=60$) $=1.55$.
- Average ranks: CMA-ES 1.73, JADE 3.21, ACO<sub>ℝ</sub> 3.59, PSO 4.10, DE 4.73, GA 5.88, $(1+1)$-ES 5.95, SA 7.40, RS 8.40.
- Groups not separated by Nemenyi:

| group |
|---|
| CMA-ES, JADE |
| JADE, ACO<sub>ℝ</sub>, PSO, DE |
| DE, GA, $(1+1)$-ES |
| GA, $(1+1)$-ES, SA |
| SA, RS |

- Wilcoxon–Holm against CMA-ES (the lab's own signed-rank test): every comparison is significant; the largest adjusted
  $p$ is $3\cdot10^{-8}$ (vs JADE).

**What changed** compared with the lab ($B=500d$, 5 instances, 7 algorithms):

1. CMA-ES is still first, but with twice the budget and 60 blocks its Nemenyi clique now contains JADE. The lab showed
   ACO<sub>ℝ</sub> as the runner-up.
2. JADE, the adaptive DE, is clearly better than plain DE (3.21 vs 4.73). Parameter control pays off once the budget
   lets the adaptation act.
3. The $(1+1)$-ES, an isotropic single-point method with one step size, ranks with GA (5.95 vs 5.88): one isotropic
   step size cannot adapt to the ill-conditioned functions, and a single elitist point is easily trapped on the
   multimodal ones.
4. The larger $N$ makes the CD smaller (1.55 vs 1.64 despite the larger $k$), so more pairs are separated.

The CD diagram can be drawn with the lab's `cd_diagram(avg_rank, names, CD, ax, title)` from `avg`, `names` and `CD`;
the block prints the same cliques that the diagram's bars show.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import BudgetedObjective, BENCHMARKS, shifted_rotated
from utils import random_rotation
# --- ask-tell core of Notebook 1 (compact copy) ---
class AskTell:
    name = "base"
    def __init__(self, d, lo, hi, budget, rng):
        self.d, self.lo, self.hi, self.budget, self.rng = d, lo, hi, budget, rng
        self.span = (hi - lo) if lo is not None else 1.0
        self.n_told = 0
    def clip(self, X): return X if self.lo is None else np.clip(X, self.lo, self.hi)
    def uniform(self, n): return self.rng.uniform(self.lo, self.hi, (n, self.d))
    def progress(self): return min(1.0, self.n_told / self.budget)

def distinct_indices(rng, n, k, n_pool=None, exclude=None):
    n_pool = n if n_pool is None else n_pool
    keys = rng.random((n, n_pool)); keys[np.arange(n), np.arange(n)] = np.inf
    if exclude is not None: keys[np.arange(n), exclude] = np.inf
    return np.argsort(keys, axis=1)[:, :k]

def optimize(opt, f, budget, callback=None):
    obj = BudgetedObjective(f, budget)
    while obj.remaining > 0:
        X = opt.ask()
        if len(X) > obj.remaining: obj(X[: obj.remaining]); break
        y = obj(X); opt.tell(X, y); opt.n_told += len(y)
        if callback is not None: callback(opt, X, y, obj)
    return obj
class RandomSearch(AskTell):
    name = "RS"
    def ask(self): return self.uniform(20)
    def tell(self, X, y): pass
class SA(AskTell):
    name = "SA"
    def __init__(self, d, lo, hi, budget, rng, step0=0.1, step_end=1e-5, T_ratio=1e-8, n_cal=50, T0=None):
        super().__init__(d, lo, hi, budget, rng)
        self.x = self.uniform(1)[0] if lo is not None else rng.standard_normal(d)
        self.fx, self.step0, self.step_end, self.T_ratio, self.n_cal, self.T0, self.cal = None, step0, step_end, T_ratio, n_cal, T0, []
    def sched(self):
        p = self.progress()
        return (self.span * self.step0 * (self.step_end / self.step0) ** p / np.sqrt(self.d),
                self.T0 * self.T_ratio ** p if self.T0 is not None else np.inf)
    def ask(self):
        if self.fx is None: return self.x[None]
        return self.clip(self.x + self.sched()[0] * self.rng.standard_normal(self.d))[None]
    def tell(self, X, y):
        if self.fx is None: self.fx = y[0]; return
        delta = y[0] - self.fx
        if self.T0 is None:                                   # calibration random walk
            if delta > 0: self.cal.append(delta)
            self.x, self.fx = X[0], y[0]
            if len(self.cal) >= self.n_cal: self.T0 = np.mean(self.cal) / np.log(2.0)
            return
        if delta <= 0 or self.rng.random() < np.exp(-delta / self.sched()[1]): self.x, self.fx = X[0], y[0]
def sbx_pm(P1, P2, lo, hi, rng, pc=0.9, eta_c=15.0, pm=None, eta_m=20.0):
    n, d = P1.shape; pm = 1.0 / d if pm is None else pm
    u = rng.random((n, d))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
    cross = (rng.random((n, 1)) < pc) & (rng.random((n, d)) < 0.5); beta = np.where(cross, beta, 1.0)
    C1, C2 = 0.5 * ((1 + beta) * P1 + (1 - beta) * P2), 0.5 * ((1 - beta) * P1 + (1 + beta) * P2)
    swap = cross & (rng.random((n, d)) < 0.5)
    C = np.vstack([np.where(swap, C2, C1), np.where(swap, C1, C2)])
    u = rng.random(C.shape)
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
    return np.clip(C + (rng.random(C.shape) < pm) * delta * (hi - lo), lo, hi)

class GA(AskTell):
    name = "GA"
    def __init__(self, d, lo, hi, budget, rng, n=50, pc=0.9, pm=None, tour=2):
        super().__init__(d, lo, hi, budget, rng); self.n, self.pc, self.pm, self.tour, self.P, self.fP = n, pc, pm, tour, None, None
    def select(self, k):
        c = self.rng.integers(0, self.n, (k, self.tour)); return c[np.arange(k), np.argmin(self.fP[c], axis=1)]
    def ask(self):
        if self.P is None: return self.uniform(self.n)
        m = self.n // 2 + 1; a, b = self.select(m), self.select(m)
        return sbx_pm(self.P[a], self.P[b], self.lo, self.hi, self.rng, pc=self.pc, pm=self.pm)[: self.n - 1]
    def tell(self, X, y):
        if self.P is None: self.P, self.fP = X.copy(), y.copy(); return
        e = np.argmin(self.fP); self.P, self.fP = np.vstack([self.P[e], X]), np.concatenate([[self.fP[e]], y])
class PSO(AskTell):
    name = "PSO"
    def __init__(self, d, lo, hi, budget, rng, n=40, w=0.7298, c1=1.49618, c2=1.49618, vmax=0.2):
        super().__init__(d, lo, hi, budget, rng)
        self.n, self.w, self.c1, self.c2, self.vmax = n, w, c1, c2, vmax * self.span
        self.X = self.uniform(n); self.V = rng.uniform(-1, 1, (n, d)) * 0.1 * self.span; self.Pb = self.fPb = None
    def ask(self):
        if self.Pb is None: return self.X
        g = self.Pb[np.argmin(self.fPb)]; r1, r2 = self.rng.random((2, self.n, self.d))
        self.V = np.clip(self.w * self.V + self.c1 * r1 * (self.Pb - self.X) + self.c2 * r2 * (g - self.X), -self.vmax, self.vmax)
        self.X = self.clip(self.X + self.V); return self.X
    def tell(self, X, y):
        if self.Pb is None: self.Pb, self.fPb = X.copy(), y.copy(); return
        imp = y < self.fPb; self.Pb[imp], self.fPb[imp] = X[imp], y[imp]
class ACOR(AskTell):
    name = "ACO_R"
    def __init__(self, d, lo, hi, budget, rng, k=50, m=2, q=0.1, xi=0.85):
        super().__init__(d, lo, hi, budget, rng); self.k, self.m, self.xi = k, m, xi
        r = np.arange(1, k + 1); w = np.exp(-((r - 1) ** 2) / (2 * (q * k) ** 2)) / (q * k * np.sqrt(2 * np.pi))
        self.cdf = np.cumsum(w / w.sum()); self.S = self.fS = None
    def ask(self):
        if self.S is None: return self.uniform(self.k)
        l = np.minimum(np.searchsorted(self.cdf, self.rng.random(self.m)), self.k - 1)
        sig = self.xi * np.abs(self.S[None, :, :] - self.S[l][:, None, :]).sum(1) / (self.k - 1)
        return self.clip(self.S[l] + sig * self.rng.standard_normal((self.m, self.d)))
    def tell(self, X, y):
        S = X if self.S is None else np.vstack([self.S, X]); fS = y if self.fS is None else np.concatenate([self.fS, y])
        o = np.argsort(fS, kind="stable")[: self.k]; self.S, self.fS = S[o], fS[o]
class DE(AskTell):
    name = "DE"
    def __init__(self, d, lo, hi, budget, rng, n=50, F=0.5, CR=0.9):
        super().__init__(d, lo, hi, budget, rng); self.n, self.F, self.CR, self.P, self.fP = n, F, CR, None, None
    def ask(self):
        if self.P is None: return self.uniform(self.n)
        n, d = self.n, self.d; idx = distinct_indices(self.rng, n, 3)
        V = self.P[idx[:, 0]] + self.F * (self.P[idx[:, 1]] - self.P[idx[:, 2]])
        mask = self.rng.random((n, d)) < self.CR; mask[np.arange(n), self.rng.integers(0, d, n)] = True
        return self.clip(np.where(mask, V, self.P))
    def tell(self, X, y):
        if self.P is None: self.P, self.fP = X.copy(), y.copy(); return
        imp = y <= self.fP; self.P[imp], self.fP[imp] = X[imp], y[imp]
class JADE(AskTell):
    """DE/current-to-pbest/1/bin with archive and adaptive mu_F, mu_CR (Zhang & Sanderson 2009), as in Notebook 2."""
    name = "JADE"
    def __init__(self, d, lo, hi, budget, rng, n=50, p=0.05, c=0.1):
        super().__init__(d, lo, hi, budget, rng); self.n, self.p, self.c = n, p, c
        self.muF, self.muCR, self.P, self.fP, self.A = 0.5, 0.5, None, None, np.empty((0, d))
    def ask(self):
        if self.P is None: return self.uniform(self.n)
        n, d, r = self.n, self.d, self.rng
        self.CRi = np.clip(r.normal(self.muCR, 0.1, n), 0, 1); F = np.empty(n); todo = np.arange(n)
        while len(todo):
            F[todo] = self.muF + 0.1 * np.tan(np.pi * (r.random(len(todo)) - 0.5)); todo = todo[F[todo] <= 0]
        self.Fi = np.minimum(F, 1.0)
        pbest = self.P[r.choice(np.argsort(self.fP)[: max(1, int(np.ceil(self.p * n)))], n)]
        pool = np.vstack([self.P, self.A]); r1 = distinct_indices(r, n, 1)[:, 0]
        r2 = distinct_indices(r, n, 1, n_pool=len(pool), exclude=r1)[:, 0]
        V = self.P + self.Fi[:, None] * (pbest - self.P) + self.Fi[:, None] * (self.P[r1] - pool[r2])
        mask = r.random((n, d)) < self.CRi[:, None]; mask[np.arange(n), r.integers(0, d, n)] = True
        return self.clip(np.where(mask, V, self.P))
    def tell(self, X, y):
        if self.P is None: self.P, self.fP = X.copy(), y.copy(); return
        s = y < self.fP
        if s.any():
            self.A = np.vstack([self.A, self.P[s]])
            if len(self.A) > self.n: self.A = self.A[self.rng.choice(len(self.A), self.n, replace=False)]
        self.P[s], self.fP[s] = X[s], y[s]
        if s.any():
            SF = self.Fi[s]; self.muCR = (1 - self.c) * self.muCR + self.c * self.CRi[s].mean()
            self.muF = (1 - self.c) * self.muF + self.c * (SF**2).sum() / SF.sum()
class OnePlusOneES(AskTell):
    """(1+1)-ES with the smooth 1/5th success rule of Notebook 2."""
    name = "(1+1)-ES"
    def __init__(self, d, lo, hi, budget, rng, sigma0=0.3, x0=None):
        super().__init__(d, lo, hi, budget, rng)
        self.x = (self.uniform(1)[0] if lo is not None else rng.standard_normal(d)) if x0 is None else np.array(x0, float)
        self.fx, self.sigma = None, sigma0 * self.span
    def ask(self):
        if self.fx is None: return self.x[None]
        return self.clip(self.x + self.sigma * self.rng.standard_normal(self.d))[None]
    def tell(self, X, y):
        if self.fx is None: self.fx = y[0]; return
        s = y[0] <= self.fx
        if s: self.x, self.fx = X[0], y[0]
        self.sigma *= np.exp((s - 0.2) / (0.8 * np.sqrt(self.d + 1)))
class CMAES(AskTell):
    """(mu/mu_w, lambda)-CMA-ES, defaults of Hansen's 2016 tutorial, positive weights only (as in Notebook 1)."""
    name = "CMA-ES"
    def __init__(self, d, lo, hi, budget, rng, m0=None, sigma0=0.3, lam=None):
        super().__init__(d, lo, hi, budget, rng); n = d
        self.m = (self.uniform(1)[0] if lo is not None else np.zeros(n)) if m0 is None else np.array(m0, float)
        self.sigma = sigma0 * self.span; self.lam = lam or 4 + int(3 * np.log(n)); self.mu = self.lam // 2
        wp = np.log((self.lam + 1) / 2) - np.log(np.arange(1, self.mu + 1)); self.w = wp / wp.sum(); self.mueff = 1 / np.sum(self.w**2)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5); self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n); self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        self.ps, self.pc, self.C, self.B, self.D, self.g = np.zeros(n), np.zeros(n), np.eye(n), np.eye(n), np.ones(n), 0
    def ask(self):
        Z = self.rng.standard_normal((self.lam, self.d)); return self.clip(self.m + self.sigma * (Z * self.D) @ self.B.T)
    def tell(self, X, y):
        n = self.d; Y = (X[np.argsort(y)[: self.mu]] - self.m) / self.sigma; yw = self.w @ Y
        self.m = self.m + self.sigma * yw; Cis = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cis @ yw; self.g += 1
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.g)) < (1.4 + 2 / (n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        dh = (1 - hs) * self.cc * (2 - self.cc)
        self.C = ((1 - self.c1 - self.cmu + self.c1 * dh) * self.C + self.c1 * np.outer(self.pc, self.pc) + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= np.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.C = (self.C + self.C.T) / 2; ev, self.B = np.linalg.eigh(self.C); self.D = np.sqrt(np.maximum(ev, 1e-30))
d = 10
fnames = ["sphere", "ellipsoid", "rosenbrock", "rastrigin", "ackley", "levy"]
def make_instance(fn, s):                          # the lab's instance generator
    b = BENCHMARKS[fn]; r = np.random.default_rng(10_000 + 97 * s + len(fn)); R = random_rotation(d, r)
    if fn in ("rosenbrock", "levy"):
        tgt = r.uniform(0.5 * b.lower, 0.5 * b.upper, d) if fn == "levy" else r.uniform(-2, 2, d); o = tgt - R.T @ np.ones(d)
    else:
        o = r.uniform(0.6 * b.lower, 0.6 * b.upper, d)
    return shifted_rotated(b.f, o, R), b.lower, b.upper

def average_ranks(v):
    order = np.argsort(v, kind="mergesort"); ranks = np.empty(len(v)); sv = np.asarray(v)[order]; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]: j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return ranks

def friedman_test(Y):
    n, k = Y.shape; Rm = np.array([average_ranks(r) for r in Y]); Rj = Rm.sum(0)
    chi2 = 12.0 / (n * k * (k + 1)) * np.sum(Rj**2) - 3.0 * n * (k + 1)
    chi2 /= 1.0 - sum(np.sum(c**3 - c) for c in (np.unique(r, return_counts=True)[1] for r in Y)) / (n * k * (k**2 - 1))
    return chi2, stats.chi2.sf(chi2, k - 1), Rm.mean(0)

def wilcoxon_signed_rank(x, y):                      # the lab's test: exact null without ties, else normal approximation
    D = np.asarray(x, float) - np.asarray(y, float); D = D[D != 0]; n = len(D)
    r = average_ranks(np.abs(D)); T = min(r[D > 0].sum(), r[D < 0].sum())
    if n <= 50 and len(np.unique(np.abs(D))) == n:
        c = np.zeros(n * (n + 1) // 2 + 1); c[0] = 1
        for k in range(1, n + 1): c[k:] = c[k:] + c[:-k].copy()
        return min(1.0, 2 * np.cumsum(c)[int(np.floor(T))] / 2.0**n)
    _, t = np.unique(np.abs(D), return_counts=True)
    return min(1.0, 2 * stats.norm.cdf((T - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24 - np.sum(t**3 - t) / 48)))

def holm(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(np.argsort(p)): run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj

def best_after(opt, f, budget):
    """The generic loop of `optimize`, keeping only the best value (no per-evaluation trace: much faster)."""
    n, best = 0, np.inf
    while n < budget:
        X = opt.ask()
        if len(X) > budget - n: best = min(best, f(X[: budget - n]).min()); break
        y = np.atleast_1d(f(X)); best = min(best, y.min()); opt.tell(X, y); opt.n_told += len(y); n += len(y)
    return best

algs = [RandomSearch, SA, GA, PSO, ACOR, DE, JADE, OnePlusOneES, CMAES]; names = [A.name for A in algs]
B, n_inst = 1000 * d, 10
blocks = [(fn, s) for fn in fnames for s in range(n_inst)]
Y = np.empty((len(blocks), len(algs)))
for bi, (fn, s) in enumerate(blocks):
    f, lo, hi = make_instance(fn, s)
    for j, A in enumerate(algs):
        Y[bi, j] = max(best_after(A(d, lo, hi, B, np.random.default_rng(s)), f, B), 1e-8)   # floored at the final target
chi2, p, avg = friedman_test(Y); k, N = len(names), len(blocks)
CD = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
o = np.argsort(avg); r = avg[o]
print(f"Friedman chi2 = {chi2:.1f}, p = {p:.1e}; Nemenyi CD (k={k}, N={N}) = {CD:.2f}")
print("average ranks:", ", ".join(f"{names[i]} {avg[i]:.2f}" for i in o))
cliques = []
for i in range(k):                                    # maximal runs of consecutive algorithms with rank range < CD
    j = i
    while j + 1 < k and r[j + 1] - r[i] < CD: j += 1
    if j > i and not any(a <= i and j <= b for a, b in cliques): cliques.append((i, j))
print("not separated by Nemenyi:", [[names[o[t]] for t in range(i, j + 1)] for i, j in cliques])
best = names[o[0]]; others = [n for n in names if n != best]
adj = holm([wilcoxon_signed_rank(Y[:, names.index(best)], Y[:, names.index(n)]) for n in others])
print(f"Wilcoxon-Holm vs {best}:", ", ".join(f"{n} {a:.1e}" for n, a in zip(others, adj)))
```

```text
Friedman chi2 = 295.8, p = 3.3e-59; Nemenyi CD (k=9, N=60) = 1.55
average ranks: CMA-ES 1.73, JADE 3.21, ACO_R 3.59, PSO 4.10, DE 4.73, GA 5.88, (1+1)-ES 5.95, SA 7.40, RS 8.40
not separated by Nemenyi: [['CMA-ES', 'JADE'], ['JADE', 'ACO_R', 'PSO', 'DE'], ['DE', 'GA', '(1+1)-ES'], ['GA', '(1+1)-ES', 'SA'], ['SA', 'RS']]
Wilcoxon-Holm vs CMA-ES: RS 1.1e-10, SA 2.3e-10, GA 1.5e-08, PSO 1.5e-08, ACO_R 7.5e-09, DE 3.6e-10, JADE 3.0e-08, (1+1)-ES 1.4e-14
```

### Exercise 6 (★★★) — bootstrap band for the runtime ECDF

The unit of replication is the (function, instance) block, because runs on the same block are dependent through the
block. Resample the 60 blocks of Exercise 5 with replacement (5000 bootstrap samples), keep PSO and CMA-ES *paired* on
the same resampled blocks, and recompute the ECDF over all (block, target) pairs at $t=100d$ evaluations. The block
reruns only these two algorithms (same instances, seeds and budget as Exercise 5).

**Result** ($B=1000d$): at $t=100d$ the ECDF of PSO is $0.114$ (95 % CI $[0.088,0.141]$) and that of CMA-ES is $0.281$
($[0.217,0.349]$). The paired difference is $0.168$ with 95 % CI $[0.128,0.209]$, which excludes 0, so the two ECDFs
differ significantly at $t=100d$. Pairing by block removes the between-function variance the two curves share, which
is why the CI of the difference is narrower than either marginal CI. A pointwise band over all $t$ is obtained by
taking the 2.5 % and 97.5 % percentiles of the bootstrap ECDFs at each $t$. A simultaneous band would need, for
example, a max-statistic bootstrap.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import BudgetedObjective, BENCHMARKS, shifted_rotated
from utils import random_rotation
# --- ask-tell core of Notebook 1 (compact copy) ---
class AskTell:
    name = "base"
    def __init__(self, d, lo, hi, budget, rng):
        self.d, self.lo, self.hi, self.budget, self.rng = d, lo, hi, budget, rng
        self.span = (hi - lo) if lo is not None else 1.0
        self.n_told = 0
    def clip(self, X): return X if self.lo is None else np.clip(X, self.lo, self.hi)
    def uniform(self, n): return self.rng.uniform(self.lo, self.hi, (n, self.d))
    def progress(self): return min(1.0, self.n_told / self.budget)

def distinct_indices(rng, n, k, n_pool=None, exclude=None):
    n_pool = n if n_pool is None else n_pool
    keys = rng.random((n, n_pool)); keys[np.arange(n), np.arange(n)] = np.inf
    if exclude is not None: keys[np.arange(n), exclude] = np.inf
    return np.argsort(keys, axis=1)[:, :k]

def optimize(opt, f, budget, callback=None):
    obj = BudgetedObjective(f, budget)
    while obj.remaining > 0:
        X = opt.ask()
        if len(X) > obj.remaining: obj(X[: obj.remaining]); break
        y = obj(X); opt.tell(X, y); opt.n_told += len(y)
        if callback is not None: callback(opt, X, y, obj)
    return obj
class PSO(AskTell):
    name = "PSO"
    def __init__(self, d, lo, hi, budget, rng, n=40, w=0.7298, c1=1.49618, c2=1.49618, vmax=0.2):
        super().__init__(d, lo, hi, budget, rng)
        self.n, self.w, self.c1, self.c2, self.vmax = n, w, c1, c2, vmax * self.span
        self.X = self.uniform(n); self.V = rng.uniform(-1, 1, (n, d)) * 0.1 * self.span; self.Pb = self.fPb = None
    def ask(self):
        if self.Pb is None: return self.X
        g = self.Pb[np.argmin(self.fPb)]; r1, r2 = self.rng.random((2, self.n, self.d))
        self.V = np.clip(self.w * self.V + self.c1 * r1 * (self.Pb - self.X) + self.c2 * r2 * (g - self.X), -self.vmax, self.vmax)
        self.X = self.clip(self.X + self.V); return self.X
    def tell(self, X, y):
        if self.Pb is None: self.Pb, self.fPb = X.copy(), y.copy(); return
        imp = y < self.fPb; self.Pb[imp], self.fPb[imp] = X[imp], y[imp]
class CMAES(AskTell):
    """(mu/mu_w, lambda)-CMA-ES, defaults of Hansen's 2016 tutorial, positive weights only (as in Notebook 1)."""
    name = "CMA-ES"
    def __init__(self, d, lo, hi, budget, rng, m0=None, sigma0=0.3, lam=None):
        super().__init__(d, lo, hi, budget, rng); n = d
        self.m = (self.uniform(1)[0] if lo is not None else np.zeros(n)) if m0 is None else np.array(m0, float)
        self.sigma = sigma0 * self.span; self.lam = lam or 4 + int(3 * np.log(n)); self.mu = self.lam // 2
        wp = np.log((self.lam + 1) / 2) - np.log(np.arange(1, self.mu + 1)); self.w = wp / wp.sum(); self.mueff = 1 / np.sum(self.w**2)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5); self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n); self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        self.ps, self.pc, self.C, self.B, self.D, self.g = np.zeros(n), np.zeros(n), np.eye(n), np.eye(n), np.ones(n), 0
    def ask(self):
        Z = self.rng.standard_normal((self.lam, self.d)); return self.clip(self.m + self.sigma * (Z * self.D) @ self.B.T)
    def tell(self, X, y):
        n = self.d; Y = (X[np.argsort(y)[: self.mu]] - self.m) / self.sigma; yw = self.w @ Y
        self.m = self.m + self.sigma * yw; Cis = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cis @ yw; self.g += 1
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.g)) < (1.4 + 2 / (n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        dh = (1 - hs) * self.cc * (2 - self.cc)
        self.C = ((1 - self.c1 - self.cmu + self.c1 * dh) * self.C + self.c1 * np.outer(self.pc, self.pc) + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= np.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.C = (self.C + self.C.T) / 2; ev, self.B = np.linalg.eigh(self.C); self.D = np.sqrt(np.maximum(ev, 1e-30))
d = 10
fnames = ["sphere", "ellipsoid", "rosenbrock", "rastrigin", "ackley", "levy"]
def make_instance(fn, s):                          # the lab's instance generator
    b = BENCHMARKS[fn]; r = np.random.default_rng(10_000 + 97 * s + len(fn)); R = random_rotation(d, r)
    if fn in ("rosenbrock", "levy"):
        tgt = r.uniform(0.5 * b.lower, 0.5 * b.upper, d) if fn == "levy" else r.uniform(-2, 2, d); o = tgt - R.T @ np.ones(d)
    else:
        o = r.uniform(0.6 * b.lower, 0.6 * b.upper, d)
    return shifted_rotated(b.f, o, R), b.lower, b.upper

B, n_inst = 1000 * d, 10                              # Exercise 5's protocol; only the two algorithms compared here
blocks = [(fn, s) for fn in fnames for s in range(n_inst)]
targets = 10.0 ** np.arange(2, -8.01, -0.2)          # the lab's 51 targets

def runtimes(A):
    """(blocks, 51 targets): first evaluation reaching each target, np.inf if never."""
    RT = np.full((len(blocks), len(targets)), np.inf)
    for bi, (fn, s) in enumerate(blocks):
        f, lo, hi = make_instance(fn, s); ev, fb = optimize(A(d, lo, hi, B, np.random.default_rng(s)), f, B).trace()
        for j, t in enumerate(targets):
            h = np.flatnonzero(fb <= t)
            if len(h): RT[bi, j] = ev[h[0]]
    return RT

RTp, RTc = runtimes(PSO), runtimes(CMAES)
ecdf = lambda RT, t=100 * d: np.mean(RT <= t)
rb = np.random.default_rng(0); ep, ec = [], []
for _ in range(5000):
    idx = rb.integers(0, len(blocks), len(blocks))    # resample (function, instance) blocks, not runs; PSO and CMA-ES paired
    ep.append(ecdf(RTp[idx])); ec.append(ecdf(RTc[idx]))
ep, ec = np.array(ep), np.array(ec); ci = lambda v: np.percentile(v, [2.5, 97.5])
print(f"ECDF at t = 100 d: PSO {ecdf(RTp):.3f} {np.round(ci(ep), 3)}, CMA-ES {ecdf(RTc):.3f} {np.round(ci(ec), 3)}; "
      f"paired difference {ecdf(RTc) - ecdf(RTp):.3f} {np.round(ci(ec - ep), 3)}")
```

```text
ECDF at t = 100 d: PSO 0.114 [0.088 0.141], CMA-ES 0.281 [0.217 0.349]; paired difference 0.168 [0.128 0.209]
```
