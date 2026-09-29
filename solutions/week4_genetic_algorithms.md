# Solutions — Week 4: Genetic Algorithms and Evolutionary Mechanisms

> Attempt every exercise yourself before reading on: the solutions are most useful as a check, not as a substitute.

Conventions: fitness is maximised unless stated otherwise. Every code block is a standalone script meant to be run
from inside `week4_genetic_algorithms/` (it adds `..` to the path for `utils`, and re-defines the notebook helpers it
needs). Every number quoted below is printed by the block of the same exercise, and all randomness is seeded. The
runner `python3 tools/run_solution_blocks.py week4_genetic_algorithms` executes all blocks.

---

## Notebook 1 — Anatomy of a GA and the mathematics of selection

### Exercise 1 (★) — Selection intensity of linear ranking is $(s-1)/\sqrt{\pi}$

**Proof.** Let the fitnesses be i.i.d. $N(0,1)$ and let $n \to \infty$. An individual of rank $r$ (0 = worst) has
quantile $u = r/(n-1)$ and is drawn with probability $p_r = \frac1n\big((2-s) + (2s-2)u\big)$. So the rank quantile
$U$ of a selected parent has density $g(u) = (2-s) + 2(s-1)u$ on $[0,1]$ (it integrates to 1). The fitness at quantile
$u$ is $\Phi^{-1}(u)$, so with $Z = \Phi^{-1}(U_0)$ for a uniform $U_0$,

$$\bar f_{\text{sel}} = \int_0^1 \Phi^{-1}(u)\,g(u)\,du = E\big[Z\,g(\Phi(Z))\big] = (2-s)\,E[Z] + 2(s-1)\,E[Z\,\Phi(Z)] .$$

$E[Z] = 0$, and Stein's identity $E[Z h(Z)] = E[h'(Z)]$ with $h = \Phi$ gives

$$E[Z\,\Phi(Z)] = E[\varphi(Z)] = \int \varphi(z)^2\,dz = \frac{1}{2\pi}\int e^{-z^2}dz = \frac{1}{2\sqrt\pi}.$$

Since $\bar f = 0$ and $\sigma_f = 1$, $I = \bar f_{\text{sel}} = 2(s-1)/(2\sqrt\pi) = (s-1)/\sqrt\pi$. $\square$

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import integrate, stats

# E[Z Phi(Z)] by quadrature vs 1/(2 sqrt(pi)), and the resulting intensity (s-1)/sqrt(pi)
val = integrate.quad(lambda z: z * stats.norm.pdf(z) * stats.norm.cdf(z), -12, 12)[0]
print(f"E[Z Phi(Z)] = {val:.8f}   1/(2 sqrt(pi)) = {1 / (2 * np.sqrt(np.pi)):.8f}")
# direct check of the integral int_0^1 Phi^{-1}(u) g(u) du for s = 1.5 and s = 2
for s in (1.5, 2.0):
    I = integrate.quad(lambda u: stats.norm.ppf(u) * ((2 - s) + 2 * (s - 1) * u), 0, 1)[0]
    print(f"s = {s}: int Phi^-1(u) g(u) du = {I:.6f}   (s-1)/sqrt(pi) = {(s - 1) / np.sqrt(np.pi):.6f}")
```

The quadrature gives $E[Z\Phi(Z)] = 0.28209479 = 1/(2\sqrt\pi)$, and the integral $\int_0^1\Phi^{-1}(u)g(u)\,du$
equals $(s-1)/\sqrt\pi$ to six digits for $s = 1.5$ and $s = 2$. Notebook 1 measured $0.5638$ for $s = 2$ by
simulation (theory $0.5642$).

### Exercise 2 (★) — Roulette is unbiased; probability of losing the best

**Proof.** Roulette makes $m = n$ i.i.d. draws with $P(\text{draw} = i) = p_i = f_i/\sum_j f_j$. The copy count is
$c_i \sim \mathrm{Bin}(n, p_i)$, so $E[c_i] = np_i = e_i$: the sampler is unbiased. The best individual is lost iff
none of the $n$ draws selects it:

$$P(c_{\max} = 0) = (1 - p_{\max})^n = \Big(1 - \frac{e_{\max}}{n}\Big)^n \xrightarrow{n\to\infty} e^{-e_{\max}} .$$

For $e_{\max} = 2$ and $n = 100$: $(0.98)^{100} = 0.1326$ (the Poisson limit is $e^{-2} = 0.1353$). Under SUS the count
of an individual with $e_i = 2$ lies in $\lbrace\lfloor 2\rfloor, \lceil 2\rceil\rbrace = \lbrace 2\rbrace$ (Notebook 1,
Section 2.2), so the probability of losing it is exactly $0$; the same holds for any $e_{\max} \ge 1$, since then
$\lfloor e_{\max}\rfloor \ge 1$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

n, e_max = 100, 2.0
exact = (1 - e_max / n) ** n
print(f"(1 - e_max/n)^n = {exact:.4f}   Poisson limit e^-2 = {np.exp(-e_max):.4f}")
rng = np.random.default_rng(0)
c = rng.binomial(n, e_max / n, size=400_000)            # roulette copy count of the best individual
print(f"simulated P(c_max = 0) over 4e5 generations: {np.mean(c == 0):.4f}")

def sus(weights, m, rng):
    c = np.cumsum(weights, dtype=float); c = c / c[-1] * m; c[-1] = m
    return np.searchsorted(c, rng.random() + np.arange(m), side="right")

f = np.ones(n); f[0] = e_max * (n - 1) / (n - e_max)     # gives e_0 = n f_0 / sum f = 2 exactly
lost = sum(np.count_nonzero(sus(f, n, rng) == 0) == 0 for _ in range(20_000))
print(f"SUS: e_0 = {n * f[0] / f.sum():.3f}, best lost in {lost} of 20,000 generations")
```

The simulation of $4\times10^5$ binomial counts gives $0.1333$, and SUS never loses an individual with $e_0 = 2$ in
20,000 generations. This is the mechanism behind the takeover table of Notebook 1: roulette loses the best individual
in 12–38% of the runs, SUS in none.

### Exercise 3 (★★) — Exact takeover-time distribution of 2-tournament

Given $j$ best copies, each of the $n$ tournaments returns a best copy iff at least one of its two entrants (drawn
with replacement) is a best copy, independently across tournaments. Hence
$J_{t+1} \mid J_t = j \sim \mathrm{Bin}\big(n, 1 - (1 - j/n)^2\big)$, a chain on $\lbrace 0, \dots, n\rbrace$ with
absorbing states $0$ (best lost) and $n$ (takeover). Let $Q$ be the transient block ($j = 1..n-1$),
$N = (I - Q)^{-1} = \sum_{t \ge 0} Q^t$ and $R_n$ the one-step absorption vector into $n$. The fixation probabilities
are $h = N R_n$. Since $P(\tau \gt t, \text{fix} \mid J_0 = 1) = \sum_k (Q^t)_{1k} h_k$, summing over $t \ge 0$ gives
$E[\tau\,\mathbf 1_{\text{fix}}] = (N h)_1$ and

$$E[\tau \mid \text{fix}] = \frac{(N h)_1}{h_1}, \qquad P(\tau \le t \mid \text{fix}) = \frac{(e_1^\top P^t)_n}{h_1}.$$

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats

n = 64; j = np.arange(n + 1)
P = stats.binom.pmf(j[None, :], n, (1 - (1 - j / n) ** 2)[:, None])      # P[i, j] = P(J' = j | J = i)
T = np.arange(1, n); Q = P[np.ix_(T, T)]; N = np.linalg.inv(np.eye(n - 1) - Q)
h = N @ P[T, n]                                                          # fixation probabilities
print(f"P(fix) = {h[0]:.4f}   E[tau | fix] = {(N @ h)[0] / h[0]:.2f}")
v = np.zeros(n + 1); v[1] = 1; cdf = []
for t in range(200):
    v = v @ P; cdf.append(v[n] / h[0])
print("median takeover | fix:", np.argmax(np.array(cdf) >= 0.5) + 1)
print(f"deterministic approximation (ln n + ln ln n)/ln 2 = {(np.log(n) + np.log(np.log(n))) / np.log(2):.1f}")

# direct simulation of 2-tournament takeover with the actual operator
rng = np.random.default_rng(1)
times = []
for _ in range(3000):
    fit = np.zeros(n, bool); fit[0] = True
    t = 0
    while 0 < fit.sum() < n:
        idx = rng.integers(0, n, (n, 2)); fit = fit[idx].any(1); t += 1
    if fit.all():
        times.append(t)
times = np.array(times)
print(f"simulation (3000 runs): P(fix) = {len(times) / 3000:.4f}, mean {times.mean():.2f}, median {np.median(times):.0f}")
```

For $n = 64$: $P(\text{fixation}) = 0.7998$, $E[\tau \mid \text{fix}] = 8.83$, median $9$. A direct simulation of the
operator (3,000 runs) gives $P(\text{fix}) = 0.8067$, mean $8.87$ and median $9$. Notebook 1's 40-run median at
$n = 64$ was $9.0$. The deterministic approximation $(\ln n + \ln\ln n)/\ln 2 = 8.1$ is about one generation too
small, because it ignores the slow final stage in which the last few non-best copies must disappear.

### Exercise 4 (★★) — Exponential ranking matching 3-tournament for $n = 100$

For a finite population the intensity is $I = \sum_r p_r\,E[Z_{(r)}]$, where $E[Z_{(r)}]$ is the expected $r$-th
smallest of $n$ standard normals (whose density is $\varphi(z)\,\mathrm{Beta}(\Phi(z); r, n-r+1)$) and $p_r$ is the
selection probability of rank $r$. For 3-tournament $p_i = (i^3 - (i-1)^3)/n^3$ ($i = 1$ worst). For exponential
ranking $p_r \propto c^{\,n-1-r}$ ($r = 0$ worst), so the best gets the largest weight when $c \lt 1$, and the scheme
is sampled with SUS or roulette like linear ranking. $I(c)$ decreases continuously from about $2.5$ (as $c \to 0$ all
mass sits on the best) to $0$ ($c = 1$, uniform), so the matching $c$ is unique and a bracketing root finder applies.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import integrate, optimize, stats

nn = 100
eos = lambda r: integrate.quad(lambda z: z * stats.norm.pdf(z) *
                               stats.beta.pdf(stats.norm.cdf(z), r, nn - r + 1), -10, 10, limit=200)[0]
mu = np.array([eos(r) for r in range(1, nn + 1)])             # E[r-th smallest of n standard normals]
i = np.arange(1, nn + 1); p3 = (i**3 - (i - 1)**3) / nn**3
I3 = p3 @ mu

def p_exp(c):                                                 # exponential ranking, r = 0 is the worst
    w = c ** (nn - 1 - np.arange(nn)); return w / w.sum()

c_star = optimize.brentq(lambda c: p_exp(c) @ mu - I3, 0.5, 0.99999)
print(f"I_3(n=100) = {I3:.4f}  (infinite-n value {3 / (2 * np.sqrt(np.pi)):.4f});  c* = {c_star:.4f}")

# Monte Carlo check: sorted normal populations, expected fitness of a selected parent
rng = np.random.default_rng(0)
Z = np.sort(rng.standard_normal((20_000, nn)), axis=1)
print(f"Monte Carlo intensity: exponential ranking {(Z @ p_exp(c_star)).mean():.4f}, 3-tournament {(Z @ p3).mean():.4f}")
```

Result: $I_3(n = 100) = 0.8378$ (slightly below the infinite-population value $3/(2\sqrt\pi) = 0.8463$), and
$c^{\ast} = 0.9657$. A Monte Carlo check with $2\times10^4$ sorted normal populations gives the expected fitness of a
selected parent as $0.8385$ for both schemes.

### Exercise 5 (★★) — One-point instead of uniform crossover on planted MAX-3SAT

The block re-implements Notebook 1's generational GA with an `xover` switch; with `xover="uniform"` it makes exactly
the notebook's random draws, so the uniform column reproduces the notebook's table. One-point crossover swaps the
tail after a cut $c \sim U\lbrace 1..L-1\rbrace$. Same instance, budget (30,000 evaluations) and 15 seeds.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats

def roulette(f, m, rng):
    c = np.cumsum(f, dtype=float)
    return np.searchsorted(c, rng.random(m) * c[-1], side="right")

def sus(w, m, rng):
    c = np.cumsum(w, dtype=float); c = c / c[-1] * m; c[-1] = m
    return np.searchsorted(c, rng.random() + np.arange(m), side="right")

def linear_ranking(f, m, rng, s=2.0):
    n = len(f); r = np.empty(n); r[np.argsort(f, kind="stable")] = np.arange(n)
    return sus(((2 - s) + (2 * s - 2) * r / (n - 1)) / n, m, rng)

def tournament(f, m, rng, k=2):
    idx = rng.integers(0, len(f), size=(m, k))
    return idx[np.arange(m), np.argmax(f[idx], axis=1)]

def truncation(f, m, rng, tau=0.5):
    top = np.argsort(f)[::-1][: int(np.ceil(tau * len(f)))]
    return top[rng.integers(0, len(top), m)]

def planted_3sat(nv, m, rng):
    x_star = rng.random(nv) < 0.5; out = []
    while len(out) < m:
        v = rng.choice(nv, 3, replace=False); sg = rng.choice([-1, 1], 3)
        if np.any(np.where(sg > 0, x_star[v], ~x_star[v])):
            out.append((v + 1) * sg)
    return np.array(out), x_star

def n_satisfied(X, cl):
    return ((X[:, np.abs(cl) - 1] == (cl > 0)).any(axis=2)).sum(axis=1)

def ga(fitness, L, select, rng, N=100, budget=30_000, pc=0.9, f_target=None, xover="uniform"):
    """Notebook 1's generational GA (1-elitism); returns the best fitness found."""
    P = rng.random((N, L)) < 0.5
    f = fitness(P); evals = N; best = f.max()
    while evals + N <= budget and best < f_target:
        par = P[select(f.astype(float), N, rng)]
        a, b = par[0::2], par[1::2]
        if xover == "uniform":
            swap = (rng.random((N // 2, L)) < 0.5) & (rng.random((N // 2, 1)) < pc)
        else:                                                   # one-point: swap the tail after a random cut
            swap = (np.arange(L) >= rng.integers(1, L, (N // 2, 1))) & (rng.random((N // 2, 1)) < pc)
        C = np.concatenate([np.where(swap, b, a), np.where(swap, a, b)])
        C ^= rng.random((N, L)) < 1.0 / L
        fc = fitness(C); evals += N
        if f.max() > fc.max():
            iw = np.argmin(fc); C[iw] = P[np.argmax(f)]; fc[iw] = f.max()
        P, f = C, fc; best = max(best, f.max())
    return best

cl, _ = planted_3sat(100, 426, np.random.default_rng(2024))
fit = lambda X: n_satisfied(X, cl)
schemes = {"roulette": roulette, "SUS": sus, "linear ranking s=2": linear_ranking,
           "tournament k=2": lambda f, m, g: tournament(f, m, g, 2),
           "tournament k=4": lambda f, m, g: tournament(f, m, g, 4),
           "truncation tau=0.3": lambda f, m, g: truncation(f, m, g, 0.3)}
unsat = {}
for name, sel in schemes.items():
    for xo in ("uniform", "one-point"):
        unsat[name, xo] = np.array([426 - ga(fit, 100, sel, np.random.default_rng(100 + s), f_target=426, xover=xo)
                                    for s in range(15)])
    u, o = unsat[name, "uniform"], unsat[name, "one-point"]
    fmt = lambda x: f"{np.median(x):g} [{np.percentile(x, 25):g}, {np.percentile(x, 75):g}]"
    p = stats.wilcoxon(u, o).pvalue if np.any(u != o) else 1.0
    print(f"{name:20s} uniform {fmt(u):16s} one-point {fmt(o):16s} paired Wilcoxon p = {p:.3f}")
```

| selection | uniform: median [IQR] | one-point: median [IQR] | paired Wilcoxon $p$ |
|---|---|---|---|
| roulette | 16 [14, 17.5] | 13 [11.5, 13.5] | 0.001 |
| SUS | 15 [13, 16] | 13 [10.5, 13.5] | 0.082 |
| linear ranking $s=2$ | 2 [1, 2] | 1 [1, 2] | 0.427 |
| tournament $k=2$ | 2 [1, 2] | 2 [1, 2] | 0.675 |
| tournament $k=4$ | 2 [1, 2] | 2 [1, 3] | 0.273 |
| truncation $\tau=0.3$ | 2 [1.5, 2] | 2 [1, 3] | 0.260 |

(unsatisfied clauses of the best solution.) **The ranking of selection schemes does not change**: fitness-proportional
selection stays an order of magnitude behind all rank-based schemes, and among the rank-based schemes no difference
between the two crossovers is detectable. This is expected. The variables are numbered at random, so the three
variables of a clause sit at random loci. The defining length of three random positions among $L = 100$ is
$(L+1)(3-1)/(3+1) \approx L/2$ on average, so one-point crossover separates a clause with probability about $1/2$,
comparable to the uniform-crossover disruption $1 - 2^{1-3} = 3/4$. One-point crossover has linkage only when the
representation places interacting variables close together, which a random variable order does not do. The
improvement for proportional selection (significant for roulette, not for SUS) is consistent with one-point crossover
being less disruptive overall — it exchanges one contiguous block rather than every locus independently — in a regime
where selection is too weak to repair disruption.

### Exercise 6 (★★★) — Sigma scaling repairs proportional selection

**Theory.** With $f' = \max\big(0, 1 + (f - \bar f)/(2\sigma_f)\big)$ and a normal population, write
$Z = (f - \bar f)/\sigma_f \sim N(0,1)$, so $f' = (1 + Z/2)^+$ and $f' \gt 0 \iff Z \gt -2$. Proportional selection
on $f'$ picks a parent with density proportional to $f'$, so its intensity is $E[Z f']/E[f']$. Using
$E[\mathbf 1_{Z \gt a}] = 1 - \Phi(a)$, $E[Z\,\mathbf 1_{Z \gt a}] = \varphi(a)$ and
$E[Z^2 \mathbf 1_{Z \gt a}] = 1 - \Phi(a) + a\varphi(a)$ (integrate $z\cdot z\varphi(z)$ by parts) with $a = -2$:

$$E[f'] = \Phi(2) + \tfrac12\varphi(2), \qquad E[Z f'] = \varphi(2) + \tfrac12\big(\Phi(2) - 2\varphi(2)\big) = \tfrac12\Phi(2),$$

$$I_\sigma = \frac{\Phi(2)/2}{\Phi(2) + \varphi(2)/2} = 0.4866,$$

independent of $\bar f$ and $\sigma_f$. Raw proportional selection instead has intensity
$E[Z(\bar f + \sigma_f Z)]/\bar f = \sigma_f/\bar f$ (the coefficient of variation), which tends to 0 when fitness
values are compressed. On planted MAX-3SAT $\sigma_f/\bar f \approx 3/400 = 0.0075$.

**Experiment.**

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats

def roulette(f, m, rng):
    c = np.cumsum(f, dtype=float)
    return np.searchsorted(c, rng.random(m) * c[-1], side="right")

def sus(w, m, rng):
    c = np.cumsum(w, dtype=float); c = c / c[-1] * m; c[-1] = m
    return np.searchsorted(c, rng.random() + np.arange(m), side="right")

def linear_ranking(f, m, rng, s=2.0):
    n = len(f); r = np.empty(n); r[np.argsort(f, kind="stable")] = np.arange(n)
    return sus(((2 - s) + (2 * s - 2) * r / (n - 1)) / n, m, rng)

def tournament(f, m, rng, k=2):
    idx = rng.integers(0, len(f), size=(m, k))
    return idx[np.arange(m), np.argmax(f[idx], axis=1)]

def truncation(f, m, rng, tau=0.5):
    top = np.argsort(f)[::-1][: int(np.ceil(tau * len(f)))]
    return top[rng.integers(0, len(top), m)]

def planted_3sat(nv, m, rng):
    x_star = rng.random(nv) < 0.5; out = []
    while len(out) < m:
        v = rng.choice(nv, 3, replace=False); sg = rng.choice([-1, 1], 3)
        if np.any(np.where(sg > 0, x_star[v], ~x_star[v])):
            out.append((v + 1) * sg)
    return np.array(out), x_star

def n_satisfied(X, cl):
    return ((X[:, np.abs(cl) - 1] == (cl > 0)).any(axis=2)).sum(axis=1)

def ga(fitness, L, select, rng, N=100, budget=30_000, pc=0.9, f_target=None, xover="uniform"):
    """Notebook 1's generational GA (1-elitism); returns the best fitness found."""
    P = rng.random((N, L)) < 0.5
    f = fitness(P); evals = N; best = f.max()
    while evals + N <= budget and best < f_target:
        par = P[select(f.astype(float), N, rng)]
        a, b = par[0::2], par[1::2]
        if xover == "uniform":
            swap = (rng.random((N // 2, L)) < 0.5) & (rng.random((N // 2, 1)) < pc)
        else:                                                   # one-point: swap the tail after a random cut
            swap = (np.arange(L) >= rng.integers(1, L, (N // 2, 1))) & (rng.random((N // 2, 1)) < pc)
        C = np.concatenate([np.where(swap, b, a), np.where(swap, a, b)])
        C ^= rng.random((N, L)) < 1.0 / L
        fc = fitness(C); evals += N
        if f.max() > fc.max():
            iw = np.argmin(fc); C[iw] = P[np.argmax(f)]; fc[iw] = f.max()
        P, f = C, fc; best = max(best, f.max())
    return best

def sigma_scaled_roulette(f, m, rng):
    sd = f.std()
    w = np.ones_like(f) if sd == 0 else np.maximum(0.0, 1 + (f - f.mean()) / (2 * sd))
    return roulette(w, m, rng)

# theory: intensity of proportional selection on f' = (1 + Z/2)^+
Phi2, phi2 = stats.norm.cdf(2), stats.norm.pdf(2)
print(f"I_sigma theory = {0.5 * Phi2 / (Phi2 + 0.5 * phi2):.4f}")
def intensity(f, w):                                          # exact E[f of a selected parent], standardised
    return ((w / w.sum()) @ f - f.mean()) / f.std()

rng = np.random.default_rng(0)
for mu, sd in ((0, 1), (400, 3), (-5, 0.01)):
    f = mu + sd * rng.standard_normal(1_000_000)
    print(f"(mu, sigma) = ({mu}, {sd}): I_sigma on 1e6 normal fitnesses = "
          f"{intensity(f, np.maximum(0, 1 + (f - f.mean()) / (2 * f.std()))):.4f}")
f = 400 + 3 * rng.standard_normal(1_000_000)
print(f"raw proportional selection at (400, 3): I = {intensity(f, f):.4f}   (sigma/mu = {3 / 400:.4f})")

cl, _ = planted_3sat(100, 426, np.random.default_rng(2024))
fit = lambda X: n_satisfied(X, cl)
res = {}
for name, sel in (("roulette", roulette), ("sigma scaling + roulette", sigma_scaled_roulette),
                  ("tournament k=2", lambda f, m, g: tournament(f, m, g, 2))):
    res[name] = np.array([426 - ga(fit, 100, sel, np.random.default_rng(100 + s), f_target=426) for s in range(15)])
    u = res[name]
    print(f"{name:26s} unsatisfied: median {np.median(u):g} [{np.percentile(u, 25):g}, {np.percentile(u, 75):g}]")
p = stats.mannwhitneyu(res["sigma scaling + roulette"], res["roulette"], alternative="less").pvalue
print(f"one-sided Mann-Whitney (sigma scaling < roulette): p = {p:.1e}")
```

The exact selection intensity on $10^6$ normal fitnesses is $0.4865$, $0.4866$ and $0.4864$ for
$(\mu,\sigma) = (0,1), (400,3), (-5, 0.01)$, while raw proportional selection at $(400, 3)$ gives $0.0075$. On planted
MAX-3SAT (15 seeds, 30,000 evaluations): roulette median 16 [14, 17.5] unsatisfied clauses, sigma scaling + roulette
median 2 [1, 2.5], binary tournament 2 [1, 2]; one-sided Mann–Whitney (sigma < roulette) $p = 1.4\times10^{-6}$.
Sigma scaling puts proportional selection on the same footing as the rank-based schemes, with an intensity ($0.49$)
close to binary tournament ($0.56$).

---

## Notebook 2 — Variation operators

### Exercise 1 (★) — Separation probability of $k$-point crossover

The $k$ cut points are a uniformly random $k$-subset of the $L-1$ gaps. Two loci at distance $\delta$ have $\delta$
gaps between them and end up in different parents iff an odd number $j$ of cuts falls into those $\delta$ gaps
(hypergeometric count):

$$P_k(\delta) = \sum_{j\ \text{odd}} \frac{\binom{\delta}{j}\binom{L-1-\delta}{k-j}}{\binom{L-1}{k}} .$$

For $k = 1$ only $j = 1$ contributes: $\delta/(L-1)$. For $k = 2$ only $j = 1$ contributes:
$\delta(L-1-\delta)/\binom{L-1}{2} = 2\delta(L-1-\delta)/\big((L-1)(L-2)\big)$ — the notebook formulas. If the cuts
were drawn independently (with replacement), the count in the $\delta$ gaps would be $\mathrm{Bin}(k, q)$ with
$q = \delta/(L-1)$ and $P(\text{odd}) = \frac12\big(1 - (1-2q)^k\big)$; this is a good approximation for $k \ll L$,
and it tends to the uniform-crossover value $1/2$ as $k$ grows.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from math import comb

def P_sep(k, d, L):
    """P(k distinct uniform cuts among the L-1 gaps put an odd number between two loci at distance d)."""
    return sum(comb(d, j) * comb(L - 1 - d, k - j) for j in range(1, k + 1, 2)) / comb(L - 1, k)

L = 20
d = np.arange(1, L)
assert np.allclose([P_sep(1, x, L) for x in d], d / (L - 1))
assert np.allclose([P_sep(2, x, L) for x in d], 2 * d * (L - 1 - d) / ((L - 1) * (L - 2)))
print("k = 1 and k = 2 reduce to the one- and two-point formulas")

rng = np.random.default_rng(0); T = 100_000; zmax = 0
for k in (1, 2, 3, 5):
    cuts = np.argsort(rng.random((T, L - 1)), axis=1)[:, :k] + 1     # k distinct cut positions in 1..L-1
    for x in (1, 5, 10, 19):                                           # loci 0 and x: cuts in 1..x separate them
        emp = np.mean(((cuts <= x).sum(1) % 2) == 1); p = P_sep(k, x, L)
        if p in (0.0, 1.0):                                            # delta = L-1: every cut lies between them
            assert emp == p
        else:
            zmax = max(zmax, abs(emp - p) / np.sqrt(p * (1 - p) / T))
print(f"simulation, 1e5 crossovers per k, k in {{1,2,3,5}}, delta in {{1,5,10,19}}: max |z| = {zmax:.2f}")
```

The block checks the two special cases exactly and compares $P_k(\delta)$ with $10^5$ simulated crossovers for
$L = 20$, $k \in \lbrace 1,2,3,5\rbrace$ and $\delta \in \lbrace 1,5,10,19\rbrace$: max $|z| = 1.99$ (for
$\delta = L-1$ every cut lies between the loci, so the probability is exactly $k \bmod 2$ and matched exactly).

### Exercise 2 (★) — Distance of the child to $p_1$ after uniform crossover and mutation

Split the loci into the $L-d$ loci where the parents agree and the $d$ where they differ.

* Agreeing locus: the child copies the common value and differs from $p_1$ only if mutated: $\mathrm{Bernoulli}(p)$.
* Differing locus: before mutation the child equals $p_1$ with probability $1/2$. XOR with an independent flip keeps a
  fair coin fair, so after mutation it differs from $p_1$ with probability exactly $1/2$.

All loci are independent, so $d_H(\text{child}, p_1) \sim \mathrm{Bin}(L-d, p) + \mathrm{Bin}(d, 1/2)$ (independent
summands):

$$E = (L-d)\,p + \frac d2, \qquad \mathrm{Var} = (L-d)\,p(1-p) + \frac d4 .$$

```python
import sys; sys.path.insert(0, "..")
import numpy as np

L, d, p, n = 60, 20, 0.03, 200_000
rng = np.random.default_rng(0)
p1 = rng.random(L) < 0.5
p2 = p1.copy(); p2[rng.choice(L, d, replace=False)] ^= True             # Hamming distance d
mask = rng.random((n, L)) < 0.5
child = np.where(mask, p2, p1) ^ (rng.random((n, L)) < p)              # uniform crossover + bit-flip rate p
dist = (child != p1).sum(1)
print(f"theory: mean {(L - d) * p + d / 2:.3f}, var {(L - d) * p * (1 - p) + d / 4:.3f}")
print(f"simulation ({n} children): mean {dist.mean():.3f}, var {dist.var():.3f}")
```

For $L = 60$, $d = 20$, $p = 0.03$: theory $11.200$ and $6.164$; $2\times10^5$ simulated children give $11.197$ and
$6.170$.

### Exercise 3 (★★) — SBX preserves the mean and always expands the variance

With $c_{1,2} = \bar p \pm \beta\,(p_1 - p_2)/2$ and $\bar p = (p_1 + p_2)/2$ (the notebook's
$c_{1,2} = \tfrac12[(1\pm\beta)p_1 + (1\mp\beta)p_2]$): (i) $c_1 + c_2 = p_1 + p_2$ exactly, so the mean of the two
children equals the parents' mean for every $\beta$. (ii) A child chosen at random is $\bar p + S\beta(p_1-p_2)/2$
with a symmetric sign $S$ independent of $\beta$, so $E[\text{child} \mid p_1, p_2] = \bar p$ as well. For the second
moment,

$$E[\beta^2] = \int_0^1 \tfrac{\eta+1}{2}\beta^{\eta+2}d\beta + \int_1^\infty \tfrac{\eta+1}{2}\beta^{-\eta}d\beta = \frac{\eta+1}{2}\Big(\frac{1}{\eta+3} + \frac{1}{\eta-1}\Big) = \frac{(\eta+1)^2}{(\eta+3)(\eta-1)} \qquad (\eta \gt 1),$$

and for $\eta \le 1$ the second integral diverges. For i.i.d. parents with variance $\sigma^2$,
$\mathrm{Var}(\text{child}) = \mathrm{Var}(\bar p) + E[\beta^2]\,E[(p_1-p_2)^2]/4 = \sigma^2(1 + E[\beta^2])/2$
(Notebook 2, Section 3). Exact preservation needs $E[\beta^2] = 1$, i.e. $(\eta+1)^2 = (\eta+3)(\eta-1)$, i.e.
$\eta^2 + 2\eta + 1 = \eta^2 + 2\eta - 3$: impossible. In fact $(\eta+1)^2 - (\eta+3)(\eta-1) = 4 \gt 0$, so
$E[\beta^2] = 1 + 4/((\eta+3)(\eta-1)) \gt 1$ for every $\eta \gt 1$: SBX always expands the variance, by a factor
that tends to 1 as $\eta \to \infty$. $\square$

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import integrate

p = lambda b, eta: 0.5 * (eta + 1) * (b**eta if b <= 1 else b ** (-(eta + 2)))
for eta in (2.0, 5.0, 20.0):
    m2 = integrate.quad(lambda b: b * b * p(b, eta), 0, 1)[0] + integrate.quad(lambda b: b * b * p(b, eta), 1, np.inf)[0]
    closed = (eta + 1) ** 2 / ((eta + 3) * (eta - 1))
    print(f"eta = {eta:4}: E[beta^2] by quadrature {m2:.4f}, closed form {closed:.4f}, variance factor {(1 + closed) / 2:.4f}")
```

Values: $E[\beta^2] = 1.8$ ($\eta = 2$), $1.125$ ($\eta = 5$), $1.0092$ ($\eta = 20$), i.e. variance factors $1.4$,
$1.0625$ and $1.0046$; quadrature of the density agrees with the closed form.

### Exercise 4 (★★) — Bounded SBX by truncation

Both children lie in $[\ell, u]$ iff $\bar p \pm \beta|p_1 - p_2|/2 \in [\ell, u]$, i.e.

$$\beta \le \beta_{\max} = \frac{2\min(\bar p - \ell,\ u - \bar p)}{|p_1 - p_2|} .$$

The truncated density is $p(\beta)/F(\beta_{\max})$ on $[0, \beta_{\max}]$, with CDF $F(\beta)/F(\beta_{\max})$. To
sample it, draw $u \sim U[0, F(\beta_{\max}))$ and apply the usual inverse of $F$: then
$P(\beta \le b) = P(u \le F(b)) = F(b)/F(\beta_{\max})$ for $b \le \beta_{\max}$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats

F = lambda b, eta: np.where(b <= 1, 0.5 * b**(eta + 1), 1 - 0.5 * b**(-(eta + 1)))   # SBX spread-factor CDF

def sbx_bounded(p1, p2, lo, hi, eta, rng):
    m, I = (p1 + p2) / 2, np.abs(p1 - p2)
    bmax = 2 * np.minimum(m - lo, hi - m) / np.maximum(I, 1e-300)
    u = rng.random(p1.shape) * F(bmax, eta)                      # u ~ U[0, F(beta_max))
    beta = np.where(u <= 0.5, (2 * u)**(1 / (eta + 1)), (2 * (1 - u))**(-1 / (eta + 1)))
    return m + beta * (p1 - p2) / 2, m - beta * (p1 - p2) / 2, bmax

eta, lo, hi, n = 2.0, 0.0, 1.0, 100_000
rng = np.random.default_rng(0)
# (a) fixed parents 0.1, 0.3: beta_max = 2 * min(0.2, 0.8) / 0.2 = 2
c1, c2, bmax = sbx_bounded(np.full(n, 0.1), np.full(n, 0.3), lo, hi, eta, rng)
b = np.abs(c1 - c2) / 0.2
print(f"beta_max = {bmax[0]:.3f}; KS against F(min(beta, 2))/F(2): p = "
      f"{stats.kstest(b, lambda t: F(np.minimum(t, 2.0), eta) / F(2.0, eta)).pvalue:.2f}")
# (b) random parents: a different beta_max per pair -> probability-integral transform must be U[0,1]
p1, p2 = rng.random(n), rng.random(n)
c1, c2, bmax = sbx_bounded(p1, p2, lo, hi, eta, rng)
b = np.abs(c1 - c2) / np.abs(p1 - p2)
print(f"random parents: KS of F(beta)/F(beta_max) against U[0,1]: p = {stats.kstest(F(b, eta) / F(bmax, eta), 'uniform').pvalue:.2f}")
print("all children inside [0, 1]:", bool(np.all((c1 >= lo - 1e-12) & (c1 <= hi + 1e-12) & (c2 >= lo - 1e-12) & (c2 <= hi + 1e-12))))
```

Validation ($\eta = 2$, $[\ell, u] = [0, 1]$): with fixed parents $0.1, 0.3$ ($\beta_{\max} = 2$) the recovered
$\beta = |c_1 - c_2|/|p_1 - p_2|$ of $10^5$ crossovers passes a KS test against $F(\min(\beta, 2))/F(2)$ with
$p = 0.50$. With random parents (a different $\beta_{\max}$ for every pair) the probability-integral transform
$F(\beta)/F(\beta_{\max})$ passes a KS test for uniformity with $p = 0.26$. All children lie in the box.

### Exercise 5 (★★) — Enhanced edge recombination

Enhanced ERX (Starkweather et al. 1991) flags the edges present in both parents. When choosing the next city it
first restricts the candidates to neighbours joined to the current city by a flagged edge, and then applies the usual
"fewest remaining neighbours" rule.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

def edges(p):
    q = np.roll(p, -1)
    return set(zip(np.minimum(p, q).tolist(), np.maximum(p, q).tolist()))

def erx(p1, p2, rng, enhanced=False):
    """Edge recombination (Notebook 2); enhanced=True prefers edges common to both parents."""
    n = len(p1)
    adj = [set() for _ in range(n)]
    for p in (p1.tolist(), p2.tolist()):
        for i in range(n):
            adj[p[i]].add(p[i - 1]); adj[p[i - 1]].add(p[i])
    common = edges(p1) & edges(p2)
    c = int(p1[0] if rng.random() < 0.5 else p2[0])
    child, unvisited = [c], set(range(n)) - {c}
    u = rng.random(n)
    for step in range(1, n):
        for nb in adj[c]:
            adj[nb].discard(c)
        cand = list(adj[c])
        if cand:
            if enhanced:
                flagged = [x for x in cand if (min(c, x), max(c, x)) in common]
                if flagged:
                    cand = flagged
            dmin = min(len(adj[x]) for x in cand)
            ties = [x for x in cand if len(adj[x]) == dmin]
            c = ties[int(u[step] * len(ties))]
        else:
            rest = sorted(unvisited); c = rest[int(u[step] * len(rest))]
        child.append(c); unvisited.discard(c)
    return np.array(child)

def inversion(p, rng):
    i, j = np.sort(rng.choice(len(p), 2, replace=False))
    q = p.copy(); q[i:j + 1] = q[i:j + 1][::-1]; return q

rng = np.random.default_rng(0); n = 30
print(f"{'parents':10s} {'ERX':>6s} {'enhanced':>9s}   (fraction of common edges kept, 400 pairs)")
for k in ("random", 1, 4, 16):
    kept = {False: [0, 0], True: [0, 0]}
    for _ in range(400):
        p1 = rng.permutation(n)
        if k == "random":
            p2 = rng.permutation(n)
        else:
            p2 = p1.copy()
            for _ in range(k):
                p2 = inversion(p2, rng)
        common = edges(p1) & edges(p2)
        for enh in (False, True):
            c = erx(p1, p2, rng, enh)
            assert sorted(c) == list(range(n))
            kept[enh][0] += len(edges(c) & common); kept[enh][1] += len(common)
    print(f"{str(k):10s} {kept[False][0] / kept[False][1]:6.3f} {kept[True][0] / kept[True][1]:9.3f}")
```

Fraction of the parents' common edges that survive in the child ($n = 30$, 400 pairs per row; $k$ = number of random
inversions separating $p_2$ from $p_1$; "random" = independent random parents, which share only about two edges):

| parents | ERX | enhanced ERX |
|---|---|---|
| random | 0.831 | 1.000 |
| $k = 1$ | 0.995 | 1.000 |
| $k = 4$ | 0.957 | 0.988 |
| $k = 16$ | 0.873 | 0.993 |

Plain ERX loses a common edge when the "fewest neighbours" rule prefers another candidate, or when the walk enters a
chain of common edges in its middle and leaves it at the other side. Flagging recovers almost all of them. The
remaining losses of the enhanced version presumably happen when the walk reaches a city from a non-flagged edge in the middle of
a common chain: it can follow the chain in only one direction, and the other half is later entered from elsewhere.

### Exercise 6 (★★★) — When does crossover pay off on the TSP?

A $(\mu+\lambda)$ EA with inversion (2-opt move) mutation only: $\lambda$ parents drawn uniformly, one inversion each,
and truncation of parents plus offspring to the best $\mu$. It is compared with the ERX GA of Section 5 ($N = 50$,
binary tournament, $p_c = 0.9$, $p_{\text{mut}} = 0.3$, 1-elitism; the block reproduces the notebook's runs exactly)
on the same 30-city instance, with 6 seeds and 6,000 evaluations each.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import random_euclidean_tsp

def erx(p1, p2, rng):
    n = len(p1)
    adj = [set() for _ in range(n)]
    for p in (p1.tolist(), p2.tolist()):
        for i in range(n):
            adj[p[i]].add(p[i - 1]); adj[p[i - 1]].add(p[i])
    c = int(p1[0] if rng.random() < 0.5 else p2[0])
    child, unvisited = [c], set(range(n)) - {c}
    u = rng.random(n)
    for step in range(1, n):
        for nb in adj[c]:
            adj[nb].discard(c)
        if adj[c]:
            dmin = min(len(adj[x]) for x in adj[c])
            ties = [x for x in adj[c] if len(adj[x]) == dmin]
            c = ties[int(u[step] * len(ties))]
        else:
            rest = sorted(unvisited); c = rest[int(u[step] * len(rest))]
        child.append(c); unvisited.discard(c)
    return np.array(child)

def inversion(p, rng):
    i, j = np.sort(rng.choice(len(p), 2, replace=False))
    q = p.copy(); q[i:j + 1] = q[i:j + 1][::-1]; return q

def ga_tsp(D, rng, N=50, budget=6000, pc=0.9, pmut=0.3, k=2):
    """Notebook 2's generational ERX GA (k-tournament, inversion mutation, 1-elitism)."""
    n = len(D); cost = lambda T: D[T, np.roll(T, -1, axis=1)].sum(1)
    P = np.array([rng.permutation(n) for _ in range(N)]); f = cost(P); evals = N; best = f.min()
    while evals + N <= budget:
        T = rng.integers(0, N, (N, 2, k))
        win = np.take_along_axis(T, np.argmin(f[T], axis=2)[..., None], 2)[..., 0]
        do_x, do_m = rng.random(N) < pc, rng.random(N) < pmut
        C = np.empty_like(P)
        for i in range(N):
            a, b = P[win[i, 0]], P[win[i, 1]]
            c = erx(a, b, rng) if do_x[i] else a.copy()
            C[i] = inversion(c, rng) if do_m[i] else c
        fc = cost(C); evals += N
        if f.min() < fc.min():
            w = np.argmax(fc); C[w] = P[np.argmin(f)]; fc[w] = f.min()
        P, f = C, fc; best = min(best, f.min())
    return best

def mu_plus_lambda(D, rng, mu, lam, budget=6000):
    """(mu+lambda) EA: uniform parent choice, one inversion per offspring, truncation of P u C."""
    n = len(D); cost = lambda T: D[T, np.roll(T, -1, axis=1)].sum(1)
    P = np.array([rng.permutation(n) for _ in range(mu)]); f = cost(P); ev = mu
    while ev + lam <= budget:
        C = np.array([inversion(p, rng) for p in P[rng.integers(0, mu, lam)]]); fc = cost(C); ev += lam
        U, fu = np.concatenate([P, C]), np.concatenate([f, fc]); o = np.argsort(fu, kind="stable")[:mu]
        P, f = U[o], fu[o]
    return f.min()

_, D = random_euclidean_tsp(30, np.random.default_rng(42))       # the instance of Notebook 2, Section 5
fmt = lambda v: f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]"
res = {"ERX GA (N = 50)": np.array([ga_tsp(D, np.random.default_rng(10 + s)) for s in range(6)])}
for mu, lam in ((1, 1), (5, 20), (10, 50), (50, 50)):
    res[f"({mu}+{lam}) EA"] = np.array([mu_plus_lambda(D, np.random.default_rng(10 + s), mu, lam) for s in range(6)])
for name, v in res.items():
    print(f"{name:18s} {fmt(v)}")
for name in ("(5+20) EA", "(50+50) EA"):
    print(f"two-sided Wilcoxon, ERX GA vs {name}: p = {stats.wilcoxon(res['ERX GA (N = 50)'], res[name]).pvalue:.3f}")
for B in (1500, 24000):
    g = np.median([ga_tsp(D, np.random.default_rng(10 + s), budget=B) for s in range(6)])
    e = np.median([mu_plus_lambda(D, np.random.default_rng(10 + s), 5, 20, budget=B) for s in range(6)])
    print(f"budget {B:5d}: ERX GA median {g:.3f}   (5+20) EA median {e:.3f}")
```

| algorithm | final length, median [IQR] |
|---|---|
| ERX GA ($N = 50$) | 4.859 [4.841, 4.866] |
| $(1+1)$ EA, inversion | 4.629 [4.619, 4.753] |
| $(5+20)$ EA | 4.619 [4.606, 4.642] |
| $(10+50)$ EA | 4.702 [4.659, 4.717] |
| $(50+50)$ EA | 5.550 [5.522, 5.637] |

At this budget the elitist, small-population mutation-only EAs *beat* the ERX GA; the $(5+20)$ EA is better on every
seed (two-sided Wilcoxon $p = 0.031$, the smallest value possible with 6 pairs). Only the $(50+50)$ EA, whose
population is as large as the GA's, is worse (also on every seed, $p = 0.031$). With 1,500 evaluations the gap
widens (ERX GA median 6.679 vs $(5+20)$ 5.277); with 24,000 it nearly closes (4.634 vs 4.610). The notebook's
"mutation-only" baseline was a *generational* $N = 50$ GA without crossover, a weak strawman. So, on a 30-city
instance with a small budget, the inversion neighbourhood plus strong elitist selection is hard to beat. Crossover
pays off when (i) the parents are good and diverse enough that recombining their edges creates tours that inversion
would need many steps to reach, which requires a larger population and budget, and (ii) the operator has high
heritability (ERX, not CX; Section 4). This is exactly why the successful TSP GAs are memetic (Lab): local search
supplies good, diverse parents, and crossover recombines their common structure.

---

## Notebook 3 — GA theory

### Exercise 1 (★) — Schema theorem for uniform crossover

Take the setting of the notebook proof: $q = m(H,t)f(H,t)/(N\bar f)$ is the probability that a roulette-selected
parent is in $H$, and a child is built from i.i.d. parents $a, b$. If crossover happens, each of the $o(H)$ fixed loci
is taken from $a$ or from $b$ independently with probability $1/2$, independently of $a$ and $b$. With probability
$2^{-o(H)}$ all fixed loci come from $a$, and then the child is in $H$ iff $a$ is (probability $q$); likewise for $b$.
Ignoring the (non-negative) probability that a mixed choice still gives an instance — which is zero in the worst
case, e.g. when the other parent is the complement of $H$ on its fixed loci —

$$P(\text{child} \in H \text{ before mutation}) \ge (1 - p_c)\,q + p_c\,2^{1-o(H)}\,q = q\big[1 - p_c\,(1 - 2^{1-o(H)})\big].$$

Mutation keeps the fixed bits with probability $(1-p_m)^{o(H)}$; summing over the $N$ children,

$$E[m(H,t+1)\mid P_t] \ge m(H,t)\frac{f(H,t)}{\bar f(t)}\big[1 - p_c(1 - 2^{1-o(H)})\big]\,(1-p_m)^{o(H)} .$$

The disruption no longer depends on $\delta(H)$ but grows quickly with $o(H)$ (it is 0 for $o = 1$, where the
inequality is exact up to mutation).

```python
import sys; sys.path.insert(0, "..")
import numpy as np

def roulette_idx(f, size, rng):
    c = np.cumsum(f, dtype=float)
    return np.searchsorted(c, rng.random(size) * c[-1], side="right")

def step(P, f_fun, rng, pc, pm, reps, xover):
    """One generation of the canonical GA from P, replicated reps times -> (reps, N, L)."""
    N, L = P.shape
    par = P[roulette_idx(f_fun(P), (reps, N), rng)]
    a, b = par[:, 0::2], par[:, 1::2]
    do = rng.random((reps, N // 2, 1)) < pc
    if xover == "uniform":
        swap = (rng.random((reps, N // 2, L)) < 0.5) & do
    else:
        swap = (np.arange(L) >= rng.integers(1, L, (reps, N // 2, 1))) & do
    C = np.concatenate([np.where(swap, b, a), np.where(swap, a, b)], axis=1)
    return C ^ (rng.random(C.shape) < pm)

in_schema = lambda X, pos, val: np.all(X[..., pos] == val, axis=-1)
L, N, pc, pm = 16, 40, 0.8, 0.02                               # the setting of Notebook 3, Section 1
w = np.linspace(1, 2, L); f_sch = lambda X: 1.0 + (X * w).sum(-1)
rng = np.random.default_rng(0)
schemata = [(np.sort(rng.choice(L, o, replace=False)), rng.random(o) < 0.5) for o in (1, 2, 3, 4) for _ in range(16)]
for xover in ("uniform", "one-point"):
    P = rng.random((N, L)) < 0.5; ok, slack = True, []
    for t in range(12):
        f = f_sch(P); nxt = step(P, f_sch, rng, pc, pm, 3000, xover)
        for pos, val in schemata:
            inst = in_schema(P, pos, val); m = inst.sum()
            if m == 0:
                continue
            o, delta = len(pos), pos[-1] - pos[0]
            disrupt = 1 - 2.0 ** (1 - o) if xover == "uniform" else delta / (L - 1)
            bound = m * f[inst].mean() / f.mean() * (1 - pc * disrupt) * (1 - pm) ** o
            c = in_schema(nxt, pos, val).sum(1)
            ok &= c.mean() + 4 * c.std() / np.sqrt(len(c)) >= bound
            slack.append(c.mean() / bound)
        P = step(P, f_sch, rng, pc, pm, 1, xover)[0]
    print(f"{xover:9s}: bound holds on all {len(slack)} (schema, generation) pairs (4 SE): {bool(ok)}; "
          f"median slack E/bound = {np.median(slack):.2f}")
```

Monte Carlo check in the notebook's setting ($L = 16$, $N = 40$, $p_c = 0.8$, $p_m = 0.02$, weighted OneMax fitness,
3,000 replicated steps per generation, 64 random schemata of order 1–4, 12 generations): with uniform crossover the
bound holds on all 725 (schema, generation) pairs with a 4-SE margin, with median slack $E/\text{bound} = 1.95$. The
same code with one-point crossover and the $\delta(H)/(L-1)$ bound gives median slack $1.57$ on 666 pairs. The
uniform bound is looser because a converging population often has *both* parents in $H$, which the worst case
ignores.

### Exercise 2 (★) — Fitness-level bound $E[T] \le e\,nH_n$

**Proof.** Partition $\lbrace0,1\rbrace^n$ into levels $A_i = \lbrace x : \mathrm{OneMax}(x) = i\rbrace$. The (1+1) EA
never moves to a lower level. From any $x \in A_i$, flipping exactly one of the $n - i$ zero-bits and nothing else
produces a point of $A_{i+1}$, so the probability $s_i$ of leaving level $i$ upwards satisfies

$$s_i \ge (n-i)\cdot\frac1n\Big(1-\frac1n\Big)^{n-1} \ge \frac{n-i}{e\,n},$$

since $(1 - 1/n)^{n-1} \ge e^{-1}$. The time spent on level $i$ is stochastically dominated by a geometric variable
with mean $1/s_i$, and each level is visited at most once, so

$$E[T] \le \sum_{i=0}^{n-1}\frac{1}{s_i} \le e\,n\sum_{i=0}^{n-1}\frac{1}{n-i} = e\,n\,H_n \le e\,n(\ln n + 1). \qquad \square$$

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats

def onemax_exact(n, p=None, K=20):
    """Exact E[T] of the (1+1) EA with rate p (default 1/n) on OneMax from a uniform start (Notebook 3, 4.1)."""
    p = 1.0 / n if p is None else p
    E = np.zeros(n + 1); kk = np.arange(K + 1); gains = np.arange(-K, K + 1)
    for i in range(n - 1, -1, -1):
        diff = np.convolve(stats.binom.pmf(kk, n - i, p), stats.binom.pmf(kk, i, p)[::-1])   # law of B - A
        up = (gains > 0) & (i + gains <= n)
        E[i] = (1 + diff[up] @ E[i + gains[up]]) / diff[up].sum()
    return float(stats.binom.pmf(np.arange(n + 1), n, 0.5) @ E)

for n in (100, 1000):
    ex, bound = onemax_exact(n), np.e * n * np.sum(1.0 / np.arange(1, n + 1))
    print(f"n = {n:4d}: exact E[T] = {ex:8.1f}   e n H_n = {bound:8.1f}   ratio = {ex / bound:.3f}")
```

| $n$ | exact $E[T]$ | $e\,nH_n$ | ratio |
|---|---|---|---|
| 100 | 1069.5 | 1410.1 | 0.758 |
| 1000 | 16894.7 | 20347.6 | 0.830 |

The bound overestimates for three reasons: it starts every run at level 0 (a uniform start is at level $\approx n/2$,
which saves about $e\,n\sum_{i \lt n/2} 1/(n-i) \approx e\,n\ln 2$), it ignores jumps of more than one level, and
$(1-1/n)^{n-1} \ge e^{-1}$ is slightly loose. The ratio tends to 1 only slowly, since both quantities are
$e\,n\ln n\,(1 + O(1/\ln n))$.

### Exercise 3 (★★) — Optimal static mutation rate on OneMax for $n = 500$

The exact recursion of Section 4.1 holds for any rate $p$ (replace $1/n$ by $p = c/n$ in both binomials).

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

def onemax_exact(n, p=None, K=20):
    """Exact E[T] of the (1+1) EA with rate p (default 1/n) on OneMax from a uniform start (Notebook 3, 4.1)."""
    p = 1.0 / n if p is None else p
    E = np.zeros(n + 1); kk = np.arange(K + 1); gains = np.arange(-K, K + 1)
    for i in range(n - 1, -1, -1):
        diff = np.convolve(stats.binom.pmf(kk, n - i, p), stats.binom.pmf(kk, i, p)[::-1])   # law of B - A
        up = (gains > 0) & (i + gains <= n)
        E[i] = (1 + diff[up] @ E[i + gains[up]]) / diff[up].sum()
    return float(stats.binom.pmf(np.arange(n + 1), n, 0.5) @ E)

n = 500
for c in (0.8, 0.9, 1.0, 1.1, 1.2):
    print(f"c = {c}: E[T] = {onemax_exact(n, c / n):.1f}")
r = minimize_scalar(lambda c: onemax_exact(n, c / n), bounds=(0.6, 1.6), method="bounded", options={"xatol": 1e-4})
E1 = onemax_exact(n)
print(f"c* = {r.x:.3f}, E[T] = {r.fun:.1f}, {100 * (1 - r.fun / E1):.2f}% below c = 1")
```

| $c$ | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 |
|---|---|---|---|---|---|
| $E[T]$ | 7853.5 | 7632.5 | 7509.3 | 7461.9 | 7475.9 |

The minimiser is $c^{\ast} = 1.125$ with $E[T] = 7460.1$, only 0.66% below the value at $c = 1$. So $c = 1$ is **not**
exactly optimal at $n = 500$, but it is within a fraction of a percent. This agrees with Witt (2013): for linear
functions the (1+1) EA with rate $c/n$ needs $(1 \pm o(1))\,\frac{e^c}{c}\,n\ln n$ steps, which is minimised at
$c = 1$ *asymptotically*. The lower-order $\Theta(n)$ terms also depend on $c$; since $e^c/c$ is flat at $c = 1$
(second derivative $e$), they shift the finite-$n$ optimum by $O(1/\log n)$, and $E[T]$ is very flat in $c$ near
the optimum.

### Exercise 4 (★★) — Full deception of trap-$k$, and population sizing for trap-5

**Proof.** Let $H$ fix $o$ positions ($1 \le o \lt k$), $j$ of them to 1. The other $k-o$ bits of a uniform member
of $H$ contain $X \sim \mathrm{Bin}(k-o, 1/2)$ ones. Write $\mathrm{trap}_k(u) = (k-1-u) + (k+1)\mathbf 1[u = k]$. Then

$$\bar f(H) = k - 1 - j - \frac{k-o}{2} + (k+1)\,P(j + X = k).$$

The indicator can be 1 only if $j = o$ (all fixed bits are 1), with $P(X = k-o) = 2^{-(k-o)}$. So among the $2^o$
competitors, every schema with $0 \lt j \lt o$ is strictly beaten by $j = 0$ (smaller $j$ is better), and the all-zeros
schema beats the all-ones schema strictly iff

$$o \;\gt\; (k+1)\,2^{-(k-o)} \iff g(r) := (k-r)\,2^{r} \gt k+1, \qquad r = k - o \in \lbrace 1, \dots, k-1\rbrace .$$

$g(r+1)/g(r) = 2(k-r-1)/(k-r) \ge 1$ iff $k - r \ge 2$, so $g$ is non-decreasing on $1 \le r \le k-1$ (with
$g(k-2) = g(k-1) = 2^{k-1}$) and its minimum is $g(1) = 2(k-1)$. Since $2(k-1) \gt k+1 \iff k \gt 3$, trap-$k$ is
**strictly fully deceptive for every $k \ge 4$**. For $k = 3$, $g(1) = g(2) = 4 = k+1$: the order-2 *and* the order-1
competitions between the all-zeros and all-ones schemata end in an exact tie, e.g. $\bar f(00{\ast}) = \bar f(11{\ast}) = 1.5$
and $\bar f(0{\ast}{\ast}) = \bar f(1{\ast}{\ast}) = 2$ for $\mathrm{trap}_3 = (2,1,0,3)$. $\square$

**Population sizing.** The tight-linkage GA of Section 3 (binary tournament, one-point crossover with $p_c = 1$,
bit-flip $1/L$, 1-elitism), run on $m = 10$ blocks of trap-5 ($L = 50$) for 100 generations with 10 seeds:

```python
import sys; sys.path.insert(0, "..")
import numpy as np
import itertools

def trap(u, k):
    return np.where(u == k, k, k - 1 - u)

# (a) exhaustive check of deception: in every competition of order 1..k-1, who wins, strictly or tied?
for k in (3, 4, 5, 6):
    B = np.array(list(itertools.product([0, 1], repeat=k)), dtype=bool); f = trap(B.sum(1), k)
    strict, weak = True, True
    for o in range(1, k):
        for pos in itertools.combinations(range(k), o):
            avg = {val: f[np.all(B[:, list(pos)] == np.array(val, bool), 1)].mean()
                   for val in itertools.product([0, 1], repeat=o)}
            best_other = max(v for key, v in avg.items() if any(key))
            strict &= avg[(0,) * o] > best_other; weak &= avg[(0,) * o] >= best_other
    print(f"trap-{k}: zeros win every competition strictly: {strict}; at least weakly: {weak}")

# (b) population sizing for m = 10 blocks of trap-5 with tight linkage (the GA of Notebook 3, Section 3)
kb, mb = 5, 10; L = kb * mb
blocks = lambda X: X.reshape(*X.shape[:-1], mb, kb).sum(-1)
fitness = lambda X: trap(blocks(X), kb).sum(-1)

def ga_trap(rng, N, gens=100):
    P = rng.random((N, L)) < 0.5; f = fitness(P)
    best, fbest = P[np.argmax(f)].copy(), f.max()
    for _ in range(gens):
        T = rng.integers(0, N, (N, 2)); par = P[T[np.arange(N), np.argmax(f[T], 1)]]   # binary tournament
        a, b = par[0::2], par[1::2]
        msk = np.arange(L) >= rng.integers(1, L, (N // 2, 1))                           # one-point, p_c = 1
        C = np.concatenate([np.where(msk, b, a), np.where(msk, a, b)])
        C ^= rng.random(C.shape) < 1 / L
        fc = fitness(C)
        if fbest > fc.max():
            wi = np.argmin(fc); C[wi] = best; fc[wi] = fbest                            # 1-elitism
        P, f = C, fc
        if f.max() > fbest:
            best, fbest = P[np.argmax(f)].copy(), f.max()
    return int((blocks(best) == kb).sum())

print(f"{'N':>5s}  runs solving all 10 blocks   median blocks solved")
for N in (50, 100, 200, 400, 800, 1600, 3200, 6400):
    solved = [ga_trap(np.random.default_rng(s), N) for s in range(10)]
    print(f"{N:5d}  {sum(v == mb for v in solved):14d}   {np.median(solved):18.1f}")
```

| $N$ | 50 | 100 | 200 | 400 | 800 | 1600 | 3200 | 6400 |
|---|---|---|---|---|---|---|---|---|
| runs solving all 10 blocks | 0 | 0 | 0 | 1 | 1 | 6 | 9 | 10 |
| median blocks solved | 5 | 5.5 | 7 | 7 | 9 | 10 | 10 | 10 |

The exhaustive enumeration confirms the proof (strict for $k = 4, 5, 6$, only weak for $k = 3$). The GA solves trap-5
reliably (at least 9 of 10 runs) from $N \approx 3200$. That is more than ten times the population that solved all
blocks of trap-4 in the notebook ($N = 200$, with 300 generations). The growth is consistent with gambler's-ruin
population sizing (Harik et al. 1999): the required $N$ grows like $2^k$ times a signal-to-noise factor. Each block
has $2^k$ alternatives that must be sampled, and the deceptive signal shrinks relative to the collateral noise of the
other $m-1$ blocks.

### Exercise 5 (★★) — The Nix–Vose chain without mutation

With $p_m = 0$, $U = I$. A population is absorbing iff $Q_{uu} = 1$, i.e. $\mathcal G(u/N) = u/N$ is a point mass.
A uniform population $\lbrace z, z, z, z\rbrace$ produces only $z$ (crossover of identical parents is the identity),
so it is absorbing. Conversely, if a population contains a genotype $z$, every child is independently $z$ with
probability at least $P(\text{both parents are } z) \gt 0$, so the chain moves to the uniform population of $z$ in one
step with positive probability, and a non-uniform state is not absorbing. Hence the absorbing states are **exactly
the eight uniform populations**, every non-uniform state is transient, and the GA is absorbed with probability 1.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
import itertools
from math import factorial

ell, Npop, pc_v = 3, 4, 0.8
G2 = 2**ell
geno = np.array(list(itertools.product([0, 1], repeat=ell)), dtype=int)     # row z = bits of z, MSB first
f_v = np.array([3, 2, 2, 1, 2, 1, 1, 4], float)                           # 3-bit trap, optimum 111
M = np.zeros((G2, G2, G2))                                                # kept child of parents (i, j)
for i in range(G2):
    for j in range(G2):
        M[i, j, i] += (1 - pc_v) / 2; M[i, j, j] += (1 - pc_v) / 2
        for cut in range(1, ell):
            for c in (np.r_[geno[i][:cut], geno[j][cut:]], np.r_[geno[j][:cut], geno[i][cut:]]):
                M[i, j, int("".join(map(str, c)), 2)] += pc_v / (ell - 1) / 2
dH = (geno[:, None, :] != geno[None, :, :]).sum(-1)

def heuristic_G(p, U):
    s = p * f_v / (p @ f_v)
    return np.einsum("i,j,ijk->k", s, s, M) @ U

def transition_matrix(Npop, U):
    S = np.array([c for c in itertools.product(range(Npop + 1), repeat=G2) if sum(c) == Npop])
    logfact = np.array([np.log(factorial(k)) for k in range(Npop + 1)])
    Q = np.array([np.exp(np.log(factorial(Npop)) + (S * np.log(np.maximum(heuristic_G(u / Npop, U), 1e-300))).sum(1)
                         - logfact[S].sum(1)) for u in S])
    return S, Q

U0 = (dH == 0).astype(float)                                             # p_m = 0: U = I
S, Q = transition_matrix(Npop, U0)
absorbing = np.isclose(np.diag(Q), 1.0)
print("absorbing states = uniform populations of:", [f"{int(np.argmax(s)):03b}" for s in S[absorbing]],
      "| all uniform:", bool(np.all(S[absorbing].max(1) == Npop)))
Tr, Ab = np.flatnonzero(~absorbing), np.flatnonzero(absorbing)
Bmat = np.linalg.solve(np.eye(len(Tr)) - Q[np.ix_(Tr, Tr)], Q[np.ix_(Tr, Ab)])
start = tuple(np.bincount([0, 1, 2, 4], minlength=G2))                      # {000, 001, 010, 100}
row = Bmat[list(map(tuple, S[Tr])).index(start)]
target = [int(np.argmax(S[a])) for a in Ab]                               # genotype of each uniform population
exact = np.zeros(G2); exact[target] = row

# operational simulation: roulette, random cut, keep one child at random, no mutation
rng = np.random.default_rng(0); reps = 200_000
Pop = np.tile(geno[[0, 1, 2, 4]], (reps, 1, 1))
powers = 2 ** np.arange(ell)[::-1]
for gen in range(400):
    z = Pop @ powers
    if np.all(z == z[:, :1]):
        break
    cum = np.cumsum(f_v[z], 1)
    pick = lambda: (rng.random((reps, Npop, 1)) * cum[:, -1:, None] >= cum[:, None, :]).sum(-1)
    A = np.take_along_axis(Pop, pick()[..., None], 1); B = np.take_along_axis(Pop, pick()[..., None], 1)
    tail = (np.arange(ell) >= rng.integers(1, ell, (reps, Npop, 1))) & (rng.random((reps, Npop, 1)) < pc_v)
    keep_first = rng.random((reps, Npop, 1)) < 0.5
    Pop = np.where(keep_first, np.where(tail, B, A), np.where(tail, A, B))
z = Pop @ powers
print(f"all {reps} runs absorbed after {gen} generations: {bool(np.all(z == z[:, :1]))}")
sim = np.bincount(z[:, 0], minlength=G2) / reps
se = np.sqrt(exact * (1 - exact) / reps)
print("genotype   exact    simulated   |z|")
for g in range(G2):
    print(f"   {g:03b}    {exact[g]:.4f}   {sim[g]:.4f}    {abs(sim[g] - exact[g]) / se[g]:.2f}")
```

Absorption probabilities from $\lbrace 000, 001, 010, 100\rbrace$ ($B = (I - Q_{TT})^{-1}Q_{TA}$), and an operational
simulation of $2\times10^5$ runs (all absorbed within 39 generations):

| absorbed into | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|---|---|---|---|---|---|---|---|---|
| exact | 0.7768 | 0.0676 | 0.0783 | 0.0011 | 0.0676 | 0.0017 | 0.0011 | 0.0058 |
| simulated | 0.7751 | 0.0676 | 0.0790 | 0.0012 | 0.0685 | 0.0017 | 0.0011 | 0.0059 |

All differences are within 2 standard errors. The optimum $111$ is absent at the start. It can still be created,
because one-point crossover builds $011$ from $001 \times 010$ (cut after the second bit) and then $111$ from
$100 \times 011$ (cut after the first bit). Even so, the population fixes on it in only 0.58% of runs: without
mutation the GA converges to a uniform population, and selection plus drift make the deceptive attractor $000$ by far
the most likely.

### Exercise 6 (★★★) — The infinite-population map and its fixed points

Iterate $p \mapsto \mathcal G(p)$ (same $M$, $U$ with $p_m = 0.05$, $f = (3,2,2,1,2,1,1,4)$). Because $\mathcal G$ is
invariant under rescaling of $p$ and maps into the simplex, its Jacobian has the eigenvalue 0 in the direction $p$ and
its other eigenvalues are those on the tangent space $\lbrace x : \sum x = 0\rbrace$; the spectral radius therefore
decides local stability.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
import itertools
from math import factorial
from scipy import optimize

ell, Npop, pc_v = 3, 4, 0.8
G2 = 2**ell
geno = np.array(list(itertools.product([0, 1], repeat=ell)), dtype=int)     # row z = bits of z, MSB first
f_v = np.array([3, 2, 2, 1, 2, 1, 1, 4], float)                           # 3-bit trap, optimum 111
M = np.zeros((G2, G2, G2))                                                # kept child of parents (i, j)
for i in range(G2):
    for j in range(G2):
        M[i, j, i] += (1 - pc_v) / 2; M[i, j, j] += (1 - pc_v) / 2
        for cut in range(1, ell):
            for c in (np.r_[geno[i][:cut], geno[j][cut:]], np.r_[geno[j][:cut], geno[i][cut:]]):
                M[i, j, int("".join(map(str, c)), 2)] += pc_v / (ell - 1) / 2
dH = (geno[:, None, :] != geno[None, :, :]).sum(-1)

def heuristic_G(p, U):
    s = p * f_v / (p @ f_v)
    return np.einsum("i,j,ijk->k", s, s, M) @ U

def transition_matrix(Npop, U):
    S = np.array([c for c in itertools.product(range(Npop + 1), repeat=G2) if sum(c) == Npop])
    logfact = np.array([np.log(factorial(k)) for k in range(Npop + 1)])
    Q = np.array([np.exp(np.log(factorial(Npop)) + (S * np.log(np.maximum(heuristic_G(u / Npop, U), 1e-300))).sum(1)
                         - logfact[S].sum(1)) for u in S])
    return S, Q

pm_v = 0.05
U = pm_v**dH * (1 - pm_v) ** (ell - dH)
G = lambda p: heuristic_G(p, U)

def iterate(p, n=5000):
    for _ in range(n):
        p = G(p)
    return p

def spectral_radius(p, eps=1e-7):
    """Jacobian of G by central differences; G is 0-homogeneous and maps into the simplex, so the
    spectrum of the full Jacobian is {0} plus its spectrum on the tangent space sum(x) = 0."""
    J = np.column_stack([(G(p + eps * e) - G(p - eps * e)) / (2 * eps) for e in np.eye(G2)])
    return np.abs(np.linalg.eigvals(J)).max()

fmt = lambda p: "(" + ", ".join(f"{x:.3f}" for x in p) + ")"
p0 = iterate(np.eye(G2)[0]); p1 = iterate(np.full(G2, 1 / G2))
print("from e_000  :", fmt(p0)); print("from uniform:", fmt(p1))
rng = np.random.default_rng(0)
ends = np.array([iterate(rng.dirichlet(np.ones(G2)), 2000) for _ in range(300)])
print("distinct attractors from 300 Dirichlet starts:", len(np.unique(np.round(ends, 4), axis=0)))
fixed = []
for _ in range(400):
    x, info, ier, _ = optimize.fsolve(lambda p: G(np.abs(p) / np.abs(p).sum()) - np.abs(p) / np.abs(p).sum(),
                                      rng.dirichlet(np.ones(G2)), full_output=True)
    x = np.abs(x) / np.abs(x).sum()
    if ier == 1 and np.abs(G(x) - x).max() < 1e-10 and not any(np.abs(x - y).max() < 1e-6 for y in fixed):
        fixed.append(x)
for x in fixed:
    print(f"fixed point {fmt(x)}  spectral radius of DG = {spectral_radius(x):.3f}")

print(f"{'N':>2s}  E[p_111]  E[p_000]   (stationary distribution of the finite-N chain)")
for Npop_ in (2, 3, 4, 5):
    S, Q = transition_matrix(Npop_, U)
    w, V = np.linalg.eig(Q.T)
    pi = np.real(V[:, np.argmin(np.abs(w - 1))]); pi /= pi.sum()
    print(f"{Npop_:2d}  {pi @ S[:, 7] / Npop_:8.3f}  {pi @ S[:, 0] / Npop_:8.3f}")
```

* from $p_0 = e_{000}$: convergence to $p^{(0)} = (0.652, 0.100, 0.098, 0.016, 0.100, 0.015, 0.016, 0.003)$, a fixed
  point concentrated on $000$;
* from the uniform distribution: convergence to $p^{(1)} = (0.001, 0.006, 0.005, 0.060, 0.006, 0.059, 0.060, 0.801)$,
  concentrated on $111$.

300 random Dirichlet starts reach only these two attractors. Solving $\mathcal G(p) = p$ with `scipy.optimize.fsolve`
from 400 starts finds exactly one more fixed point, an **unstable** interior one,
$p^{(s)} = (0.258, 0.140, 0.076, 0.093, 0.140, 0.071, 0.093, 0.130)$. The spectral radius of the Jacobian is $1.246$
there, against $0.688$ at $p^{(0)}$ and $0.465$ at $p^{(1)}$. So the infinite-population GA is bistable: which optimum
it reaches depends on the initial distribution, and the deceptive fixed point is stable.

Finite-$N$ chains ($p_m = 0.05$) are irreducible, with a unique stationary distribution $\pi_N$. The stationary mean
genotype frequencies $E_{\pi_N}[p]$ are:

| $N$ | $E[p_{111}]$ | $E[p_{000}]$ |
|---|---|---|
| 2 | 0.234 | 0.212 |
| 3 | 0.348 | 0.228 |
| 4 | 0.460 | 0.198 |
| 5 | 0.555 | 0.154 |

As $N$ grows, the finite chain behaves like a noisy version of the deterministic map: it spends long metastable
periods near $p^{(0)}$ and near $p^{(1)}$ and switches between them by rare large fluctuations. The data show the
stationary mass moving towards the $111$ fixed point ($E[p_{111}]$ increases with $N$), which is what one expects if
escaping from $p^{(1)}$ (the more strongly contracting attractor, with the larger fitness) becomes rare faster than
escaping from $p^{(0)}$. The two models agree in the limit $N \to \infty$ only in this sense of concentration near the
attractors. For finite $N$ the stationary distribution mixes both, and no single fixed point of $\mathcal G$ describes
it.

---

## Notebook 4 — Diversity, niching, island models and constraints

### Exercise 1 (★) — Drift with uniform crossover and mutation

**Proof of the drift law with crossover.** For one locus let $c$ be the count of ones, $p = c/N$, and
$h = 2c(N-c)/(N(N-1))$, the probability that two distinct random individuals differ at that locus. Neutral selection
draws $N$ parents i.i.d., so $c' \sim \mathrm{Bin}(N, p)$ and

$$E[c'(N-c')] = N^2 p - \big(Np(1-p) + N^2p^2\big) = N(N-1)\,p(1-p) \quad\Rightarrow\quad E[h'] = 2p(1-p) = \Big(1-\frac1N\Big)h .$$

Uniform crossover of a pair exchanges alleles between its two children, so the per-locus count is unchanged. $h$
depends only on the counts, so crossover leaves $h$ unchanged and the law holds with crossover too (averaging over
loci gives it for the population measure $h_t$).

**Adding mutation.** Flip each bit independently with probability $\mu$ after crossover, and let $a = 2\mu(1-\mu)$.
Two individuals that agree at a locus disagree afterwards with probability $a$; two that disagree still disagree with
probability $\mu^2 + (1-\mu)^2 = 1 - a$. Since $h$ is the average of the disagreement indicator over all pairs,
$E[h'' \mid h'] = h'(1-a) + (1-h')a = (1-2a)h' + a$, and

$$E[h_{t+1}] = \Big(1-\frac1N\Big)(1-2a)\,E[h_t] + a, \qquad h^{\ast}(N,\mu) = \frac{a}{1 - (1-\frac1N)(1-2a)} = \frac{2\mu(1-\mu)}{1 - (1-\frac1N)(1-4\mu(1-\mu))}.$$

The recursion is a contraction (factor $\lt 1$), so $E[h_t] \to h^{\ast}$ geometrically. For small $\mu$ and large
$N$, $h^{\ast} \approx 2N\mu/(1 + 4N\mu)$: a mutation–drift balance, which stays far from 0 once $N\mu$ is of order 1.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

N, L, gens, R = 30, 60, 150, 300
def h(P):                                                     # mean pairwise Hamming distance per locus, per run
    c = P.sum(1); return (2 * c * (N - c)).sum(1) / (N * (N - 1)) / L

rng = np.random.default_rng(0)
for mu in (0.0, 0.01):
    a = 2 * mu * (1 - mu)
    P = rng.random((R, N, L)) < 0.5                           # R independent populations, simulated together
    H = [h(P)]
    for _ in range(gens):
        P = np.take_along_axis(P, rng.integers(0, N, (R, N, 1)), 1)          # neutral selection
        A, B = P[:, 0::2], P[:, 1::2]; m = rng.random(A.shape) < 0.5
        P = np.concatenate([np.where(m, B, A), np.where(m, A, B)], 1)       # uniform crossover
        P ^= rng.random(P.shape) < mu                                        # bit-flip mutation
        H.append(h(P))
    H = np.array(H)                                                          # (gens+1, R)
    rec = [H[0].mean()]
    for _ in range(gens):
        rec.append((1 - 1 / N) * (1 - 2 * a) * rec[-1] + a)                 # mu = 0: the drift law
    z = np.abs(H.mean(1) - rec) / (H.std(1) / np.sqrt(R) + 1e-12)
    print(f"mu = {mu}: max |simulated mean - recursion| / SE over {gens} generations = {z.max():.2f}")
    if mu > 0:
        h_star = a / (1 - (1 - 1 / N) * (1 - 2 * a))
        print(f"          h* = {h_star:.4f}; simulated mean over the last 50 generations = {H[-50:].mean():.4f}")
```

Check ($N = 30$, $L = 60$, 300 runs of 150 generations, uniform crossover): for $\mu = 0$ (the drift law) and for
$\mu = 0.01$ (the recursion) the simulated mean trajectory stays within $1.9$ standard errors of the theory at every
generation. For $\mu = 0.01$, $h^{\ast} = 0.2765$, and the simulated mean over the last 50 generations is $0.2761$.

### Exercise 2 (★) — Equilibrium of fitness sharing

Let the $K$ peaks be separated by more than $\sigma_{\text{share}}$ and suppose all $n_k$ individuals of niche $k$
sit at its peak ($d = 0$, $\mathrm{sh}(0) = 1$). Individuals in different niches do not interact
($\mathrm{sh}(d) = 0$ for $d \ge \sigma_{\text{share}}$), so the niche count of each member of niche $k$ is $n_k$ and its
shared fitness is $f_k/n_k$. Under proportional selection the expected number of offspring of niche $k$ is

$$E[n_k'] = N\,\frac{n_k \cdot f_k/n_k}{\sum_l n_l \cdot f_l/n_l} = N\,\frac{f_k}{\sum_l f_l},$$

independent of the current $n_k$ (as long as every niche is occupied). The expected dynamics therefore reach in one
generation the fixed point $n_k^{\ast} = N f_k / \sum_l f_l \propto f_k$. It is also the only allocation at which
every shared fitness $f_k/n_k$ equals the common value $\sum_l f_l / N$: no individual can gain by moving to another
peak (an ideal free distribution). $\square$ The same argument holds for any $\alpha$, because only
$\mathrm{sh}(0) = 1$ and $\mathrm{sh}(d) = 0$ for $d \ge \sigma_{\text{share}}$ are used. (With finite $N$ the counts
fluctuate around $n_k^{\ast}$, and a niche can be lost by drift — Notebook 4, Section 2.)

### Exercise 3 (★★) — Restricted tournament selection vs deterministic crowding

RTS (Harik 1995): for each child, draw a window of $w$ random population members, find the one closest to the child,
and replace it if the child is better. Both methods use the notebook's variation (SBX + polynomial mutation).

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import rastrigin

def vary(A, B, rng, lo, hi, eta_c=10.0, eta_m=20.0, pc=0.9):
    """Notebook 4: SBX (per pair, prob pc) + polynomial mutation (rate 1/d), clipped. Two children per pair."""
    u = rng.random(A.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (2 * (1 - u)) ** (-1 / (eta_c + 1)))
    do = rng.random((len(A), 1)) < pc
    C = np.concatenate([np.where(do, 0.5 * ((1 + beta) * A + (1 - beta) * B), A),
                        np.where(do, 0.5 * ((1 - beta) * A + (1 + beta) * B), B)])
    u = rng.random(C.shape)
    dl = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
    return np.clip(C + (rng.random(C.shape) < 1.0 / C.shape[1]) * dl * (hi - lo), lo, hi)

def ga_crowding(f, lo, hi, d, rng, N, gens):
    """Deterministic crowding (Mahfoud 1995), as in Notebook 4 (maximisation)."""
    P = rng.uniform(lo, hi, (N, d)); fx = f(P)
    for _ in range(gens):
        perm = rng.permutation(N); a, b = perm[0::2], perm[1::2]
        C = vary(P[a], P[b], rng, lo, hi); fc = f(C)
        c1, c2, f1, f2 = C[: N // 2], C[N // 2:], fc[: N // 2], fc[N // 2:]
        straight = (np.linalg.norm(P[a] - c1, axis=1) + np.linalg.norm(P[b] - c2, axis=1)
                    <= np.linalg.norm(P[a] - c2, axis=1) + np.linalg.norm(P[b] - c1, axis=1))
        ca, fa = np.where(straight[:, None], c1, c2), np.where(straight, f1, f2)
        cb, fb = np.where(straight[:, None], c2, c1), np.where(straight, f2, f1)
        ra, rb = fa > fx[a], fb > fx[b]
        P[a[ra]], fx[a[ra]] = ca[ra], fa[ra]
        P[b[rb]], fx[b[rb]] = cb[rb], fb[rb]
    return P

def ga_rts(f, lo, hi, d, rng, N, gens, w=None):
    """Restricted tournament selection (Harik 1995): each child competes with the closest of w random members."""
    w = w or max(2, N // 5)
    P = rng.uniform(lo, hi, (N, d)); fx = f(P)
    for _ in range(gens):
        perm = rng.permutation(N); C = vary(P[perm[0::2]], P[perm[1::2]], rng, lo, hi); fc = f(C)
        for c, fcv in zip(C, fc):
            W = rng.choice(N, w, replace=False); j = W[np.argmin(np.linalg.norm(P[W] - c, axis=1))]
            if fcv > fx[j]:
                P[j], fx[j] = c, fcv
    return P

himm = lambda X: (X[..., 0] ** 2 + X[..., 1] - 11) ** 2 + (X[..., 0] + X[..., 1] ** 2 - 7) ** 2
f_himm = lambda X: 1.0 / (1.0 + himm(X))
HIMM_MIN = np.array([[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]])
f_ras = lambda X: -rastrigin(X)
lattice_optima = lambda P: len({tuple(z) for z, x in zip(np.rint(P).astype(int), P) if np.linalg.norm(x - np.rint(x)) < 0.1})

for name, algo in (("deterministic crowding", ga_crowding), ("RTS (w = N/5)", ga_rts)):
    hf = [int((np.linalg.norm(HIMM_MIN[:, None] - algo(f_himm, -6.0, 6.0, 2, np.random.default_rng(s), 80, 100)[None],
                              axis=2).min(1) < 0.1).sum()) for s in range(20)]
    ro = [lattice_optima(algo(f_ras, -5.12, 5.12, 5, np.random.default_rng(s), 200, 200)) for s in range(10)]
    print(f"{name:24s} Himmelblau: all 4 optima in {np.mean(np.array(hf) == 4):.0%} of 20 runs;  "
          f"5-D Rastrigin lattice optima: median {np.median(ro):g} [{np.percentile(ro, 25):g}, {np.percentile(ro, 75):g}]")
```

* Himmelblau ($N = 80$, 100 generations, 20 seeds): both deterministic crowding and RTS ($w = 16$) find all four
  optima (within 0.1) in 100% of runs.
* 5-D Rastrigin on $[-5.12, 5.12]^5$ ($N = 200$, 200 generations, 10 seeds; distinct integer-lattice local optima
  occupied within radius 0.1): deterministic crowding median 81.5 [71.25, 84], RTS ($w = 40$) median 44.5 [42, 48.25].

Crowding keeps more niches because each child competes only with its own (closer) parent, a purely local
replacement. RTS lets a child replace the nearest of 40 random members, which is sometimes a member of a *different*
(but nearby) basin, so on a landscape with $11^5$ basins neighbouring niches merge. RTS's window size $w$ is the knob:
a larger $w$ makes replacement more local and preserves more niches, at a higher cost in distance computations.

### Exercise 4 (★★) — Deb's rules are a total preorder; stochastic ranking with $P_f = 0$

**Proof.** Define the key $\kappa(x) = (0, -f(x))$ if $\phi(x) = 0$ and $\kappa(x) = (1, \phi(x))$ otherwise, and
say $x \preceq y$ ("$x$ at least as good as $y$") iff $\kappa(x) \le \kappa(y)$ lexicographically. Deb's three rules
say exactly this. A feasible solution beats an infeasible one because the first component is smaller. Two feasible
solutions compare by $-f$, and two infeasible ones by $\phi$. Lexicographic order on $\mathbb R^2$ is a total order,
so its pullback $\preceq$ is reflexive, transitive and total — a total preorder (distinct solutions with equal keys
are tied, so it is not antisymmetric).

With $P_f = 0$, stochastic ranking compares two adjacent individuals by objective only if both are feasible, and
otherwise by violation. If exactly one is feasible, its $\phi = 0$ is smaller, so it moves up. So the comparator
swaps an adjacent pair exactly when the lower one is strictly better under $\preceq$. Stochastic ranking is then a
deterministic bubble sort with this comparator. Runarsson & Yao use $N$ sweeps, and the notebook's odd–even
transposition form uses $N$ sweeps of two phases each; $N$ phases of odd–even transposition sort any sequence
(Knuth, TAOCP vol. 3, §5.3.4), and a sweep without swaps means that no adjacent pair is out of order, i.e. the
sequence is sorted. Either way the output is sorted by $\preceq$: stochastic ranking with $P_f = 0$ reduces to Deb's
rules. $\square$

```python
import sys; sys.path.insert(0, "..")
import numpy as np

def stochastic_rank(obj, phi, rng, pf=0.45):
    """Notebook 4: stochastic ranking (Runarsson & Yao 2000), odd-even transposition form; indices best-first."""
    n = len(obj)
    idx, o, ph = np.arange(n), np.asarray(obj, float).copy(), np.asarray(phi, float).copy()
    U = rng.random((n, 2, n // 2))
    for sweep in range(n):
        swapped = False
        for start in (0, 1):
            npairs = (n - start) // 2
            si, sj = slice(start, start + 2 * npairs, 2), slice(start + 1, start + 2 * npairs, 2)
            by_obj = ((ph[si] == 0) & (ph[sj] == 0)) | (U[sweep, start, :npairs] < pf)
            swap = np.where(by_obj, o[si] < o[sj], ph[si] > ph[sj])
            if swap.any():
                swapped = True
                a = start + 2 * np.flatnonzero(swap)
                for arr in (idx, o, ph):
                    arr[a], arr[a + 1] = arr[a + 1].copy(), arr[a].copy()
        if not swapped:
            break
    return idx

def key(o, ph):                                                   # Deb's lexicographic key (smaller = better)
    return np.where(ph == 0, 0, 1), np.where(ph == 0, -o, ph)
rng = np.random.default_rng(0); ok = True
for _ in range(500):
    n = int(rng.integers(2, 60))
    obj = rng.integers(0, 5, n).astype(float)                         # few distinct values: many ties
    phi = np.where(rng.random(n) < 0.5, 0.0, rng.integers(1, 4, n).astype(float))
    r = stochastic_rank(obj, phi, rng, pf=0.0)
    k1, k2 = key(obj[r], phi[r])
    ok &= bool(np.all((k1[:-1] < k1[1:]) | ((k1[:-1] == k1[1:]) & (k2[:-1] <= k2[1:]))))   # sorted in Deb's order
print("stochastic ranking with P_f = 0 returns Deb's order on 500 random instances:", ok)
```

On 500 random instances ($n$ from 2 to 59, integer objectives and violations with many ties, about half feasible),
the notebook's `stochastic_rank(obj, phi, rng, pf=0.0)` returns Deb's order every time. (With $P_f = 0.45$ the same
check fails, as it should.)

### Exercise 5 (★★) — Sweeping the static penalty coefficient on the knapsack

Same instance ($n = 100$, strongly correlated, DP optimum 3214, capacity 2524) and GA as Section 4 (binary tournament
on the key $v^\top x - \rho\,\phi(x)$, uniform crossover, bit-flip $1/n$, 1-elitism under Deb's order), 10 seeds,
8,000 evaluations; $\rho_0 = \max_i v_i/w_i = 11$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import random_knapsack, knapsack_dp

def knap_ga_static(v, w, C, rho, rng, N=40, budget=8_000):
    """Notebook 4's knapsack GA with the static penalty key v.x - rho*phi (binary tournament, uniform crossover,
    bit-flip 1/n, 1-elitism under Deb's order). Returns (best feasible value, infeasible fraction of final P)."""
    n = len(v); Mbig = v.sum() + 1
    phi = lambda X: np.maximum(0.0, X @ w - C)
    deb_key = lambda X: np.where(phi(X) > 0, -phi(X) - Mbig, X @ v)
    feas_best = lambda X: np.max(np.where(X @ w <= C, X @ v, -np.inf))
    P = rng.random((N, n)) < 0.5; evals = N; best = feas_best(P)
    while evals + N <= budget:
        k = P @ v - rho * phi(P)
        T = rng.integers(0, N, (N, 2)); par = P[T[np.arange(N), np.argmax(k[T], 1)]]
        a, b = par[0::2], par[1::2]; msk = rng.random(a.shape) < 0.5
        Cc = np.concatenate([np.where(msk, b, a), np.where(msk, a, b)])
        Cc ^= rng.random(Cc.shape) < 1.0 / n
        Cc[rng.integers(N)] = P[np.argmax(deb_key(P))]
        P = Cc; evals += N; best = max(best, feas_best(P))
    return best, np.mean(phi(P) > 0)

v, w, C = random_knapsack(100, np.random.default_rng(3), correlation="strong", capacity_ratio=0.5)
opt, _ = knapsack_dp(v, w, C)
rho0 = np.max(v / w)
print(f"DP optimum {opt:.0f}, capacity {C:.0f}, rho_0 = max v/w = {rho0:.2f}")
print(f"{'rho/rho0':>8s}  gap to optimum (%): median   infeasible fraction of final population (mean)")
for r in (0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 5, 10):
    out = [knap_ga_static(v, w, C, r * rho0, np.random.default_rng(s)) for s in range(10)]
    gap = [100 * (opt - b) / opt for b, _ in out]
    print(f"{r:8g}  {np.median(gap):10.2f}   {np.mean([x for _, x in out]):10.2f}")

# threshold: V(c) = DP optimum at capacity c; the penalised optimum is infeasible iff rho < max_k (V(C+k) - V(C)) / k
Cmax = int(w.sum()); V = np.zeros(Cmax + 1)
for vi, wi in zip(v, w.astype(int)):
    V[wi:] = np.maximum(V[wi:], V[:Cmax + 1 - wi] + vi)
ks = np.arange(1, Cmax - int(C) + 1); ratios = (V[int(C) + ks] - V[int(C)]) / ks
print(f"V(C) = {V[int(C)]:.0f} (= DP optimum);  rho* = {ratios.max():.3f} = {ratios.max() / rho0:.3f} rho_0, attained at overload k = {ks[np.argmax(ratios)]}")
```

| $\rho/\rho_0$ | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 1 | 2 | 5 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| median gap to optimum (%) | 4.67 | 1.24 | 1.43 | 1.59 | 1.76 | 1.85 | 1.87 | 1.77 | 1.84 |
| infeasible fraction of final population (mean) | 0.97 | 0.51 | 0.42 | 0.32 | 0.29 | 0.28 | 0.28 | 0.21 | 0.23 |

(The $\rho = \rho_0$ column reproduces the notebook's static-penalty result, 1.851%; the table is the requested plot
in numbers.)

**Where is the penalised optimum infeasible?** Let $V(c)$ be the DP optimum with capacity $c$ (integer weights), so
the best feasible penalised value is $V(C)$. An infeasible $x$ with overload $k \ge 1$ has $v^\top x \le V(C+k)$, so
it can beat $V(C)$ only if $V(C+k) - \rho k \gt V(C)$ for some $k$. Conversely, if this holds for some $k$, the
maximiser $y$ of $V(C+k)$ has $v^\top y \gt V(C)$, hence it is infeasible, with overload $1 \le k' \le k$ and
penalised value $\ge V(C+k) - \rho k \gt V(C)$. So the penalised optimum is infeasible iff

$$\rho \lt \rho^{\ast} = \max_{k \ge 1}\frac{V(C+k) - V(C)}{k}.$$

One DP table gives $\rho^{\ast} = 2.0 = 0.18\,\rho_0$ (attained at overload $k = 10$). Here $v_i = w_i + 10$, so an
extra unit of capacity is worth up to $1 + 10/w_i$, far less than $\rho_0 = 11$ (the value density of the lightest
item). Hence for $\rho = 0.1\rho_0 = 1.1 \lt \rho^{\ast}$ the penalised problem rewards overfilling: almost the whole
final population is infeasible (only the Deb-elite is feasible), and the best feasible value found is poor. Just above
$\rho^{\ast}$ ($0.2\rho_0$) the search works best: the population straddles the boundary (half infeasible), where the
knapsack optimum lies. Larger $\rho$ pushes the population away from the boundary, and the gap grows to about 1.8%.
The common rule $\rho = \max_i v_i/w_i$ is safe but conservative; the minimal *exact* coefficient $\rho^{\ast}$ is what
a well-tuned static penalty should approach.

### Exercise 6 (★★★) — Migration topology and interval

Setting of Section 3 (10-D Rastrigin, 4 islands × 50, $m$ = 2 migrants per source replacing the worst, 200
generations = 40,000 evaluations), 8 seeds per cell. "Full" = every island receives the best 2 of each other island;
"random" = one random source island per island and migration.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import rastrigin

def vary(A, B, rng, lo, hi, eta_c=10.0, eta_m=20.0, pc=0.9):
    """Notebook 4: SBX (per pair, prob pc) + polynomial mutation (rate 1/d), clipped. Two children per pair."""
    u = rng.random(A.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (2 * (1 - u)) ** (-1 / (eta_c + 1)))
    do = rng.random((len(A), 1)) < pc
    C = np.concatenate([np.where(do, 0.5 * ((1 + beta) * A + (1 - beta) * B), A),
                        np.where(do, 0.5 * ((1 - beta) * A + (1 + beta) * B), B)])
    u = rng.random(C.shape)
    dl = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
    return np.clip(C + (rng.random(C.shape) < 1.0 / C.shape[1]) * dl * (hi - lo), lo, hi)

def island_ga(f, lo, hi, d, rng, K=4, n=50, gens=200, tau=10, m=2, topology="ring"):
    """Notebook 4's island GA (minimisation) with a migration topology. Returns (final best, final RMS diversity)."""
    P = rng.uniform(lo, hi, (K, n, d)); F = np.array([f(Pi) for Pi in P])
    for g in range(1, gens + 1):
        for k in range(K):
            T = rng.integers(0, n, (n, 2)); par = P[k][T[np.arange(n), np.argmin(F[k][T], 1)]]
            C = vary(par[0::2], par[1::2], rng, lo, hi); fc = f(C)
            if F[k].min() < fc.min():
                w = np.argmax(fc); C[w] = P[k][np.argmin(F[k])]; fc[w] = F[k].min()
            P[k], F[k] = C, fc
        if g % tau == 0:
            best = [np.argsort(F[k])[:m] for k in range(K)]
            out = [(P[k][best[k]].copy(), F[k][best[k]].copy()) for k in range(K)]
            for k in range(K):
                if topology == "ring":
                    src = [(k - 1) % K]
                elif topology == "full":
                    src = [j for j in range(K) if j != k]
                else:                                               # one random source island
                    src = [int(rng.choice([j for j in range(K) if j != k]))]
                X = np.concatenate([out[j][0] for j in src]); FX = np.concatenate([out[j][1] for j in src])
                worst = np.argsort(F[k])[-len(X):]
                P[k][worst], F[k][worst] = X, FX
    allP = P.reshape(-1, d)
    return F.min(), np.sqrt(2 * len(allP) / (len(allP) - 1) * allP.var(0).sum())

print(f"{'topology':8s} {'tau':>4s}   final best: median [IQR]    final RMS diversity (median)")
for topo in ("ring", "full", "random"):
    for tau in (1, 5, 20, 50):
        r = np.array([island_ga(rastrigin, -5.12, 5.12, 10, np.random.default_rng(s), tau=tau, topology=topo)
                      for s in range(8)])
        print(f"{topo:8s} {tau:4d}   {np.median(r[:, 0]):5.2f} [{np.percentile(r[:, 0], 25):.2f}, "
              f"{np.percentile(r[:, 0], 75):.2f}]        {np.median(r[:, 1]):.2f}")
r = np.array([island_ga(rastrigin, -5.12, 5.12, 10, np.random.default_rng(s), K=1, n=200, tau=10**9) for s in range(8)])
print(f"panmictic 1 x 200 (no migration): {np.median(r[:, 0]):.2f} [{np.percentile(r[:, 0], 25):.2f}, "
      f"{np.percentile(r[:, 0], 75):.2f}]   diversity {np.median(r[:, 1]):.2f}")
```

| topology | $\tau$ | final best, median [IQR] | final RMS diversity (median) |
|---|---|---|---|
| ring | 1 | 0.00 [0.00, 0.01] | 1.34 |
| ring | 5 | 1.53 [1.17, 1.97] | 1.67 |
| ring | 20 | 4.18 [3.13, 5.54] | 1.88 |
| ring | 50 | 3.25 [2.22, 3.92] | 2.74 |
| full | 1 | 0.00 [0.00, 0.00] | 0.94 |
| full | 5 | 0.06 [0.03, 0.16] | 1.36 |
| full | 20 | 2.78 [1.26, 3.73] | 1.73 |
| full | 50 | 3.25 [2.69, 4.42] | 1.86 |
| random | 1 | 0.00 [0.00, 0.01] | 1.32 |
| random | 5 | 1.92 [1.15, 2.06] | 1.61 |
| random | 20 | 3.32 [2.59, 3.58] | 1.75 |
| random | 50 | 2.90 [2.28, 4.07] | 2.30 |
| panmictic $1\times200$ | — | 5.00 [4.07, 6.79] | 2.11 |

**Diversity and takeover.** Treat the best migrant as a single copy of a superior individual arriving in an island.
Within an island of 50, binary-tournament takeover needs about $(\ln 50 + \ln\ln 50)/\ln 2 \approx 7.6$ generations
(Notebook 1). If $\tau$ is much shorter than this, migrants arrive faster than islands can digest them, and the
archipelago behaves like one well-mixed population. Diversity is then lowest, and lowest of all for the fully
connected topology, whose migrants reach every island in one hop. If $\tau \gg 8$, each island converges on its own
before the next migration, and diversity grows with $\tau$. A ring needs $K-1 = 3$ hops, i.e. $3\tau$ generations, to
spread a migrant everywhere, so at equal $\tau$ it preserves more diversity than the fully connected topology, with
the random topology in between.

**Surprise at $\tau = 1$.** Frequent migration nearly solves 10-D Rastrigin at this budget, whereas the panmictic
GA (median 5.00 here, 4.60 over the notebook's 10 seeds) does not — even though the archipelago's final diversity is
*lower* (1.34 vs 2.11 for the ring). The reason is that migration of the two best individuals every generation is an
extra, very strong elitism. The best points of each island are copied into its neighbours each generation, which
protects and propagates good coordinate values (a separable function rewards this). So the gain is due to preserved
elites, not to preserved diversity. An honest comparison of topologies must separate the two effects, for example by
adding matching elitism to the panmictic GA.

---

## Lab — Memetic algorithms for the TSP and a toy evolutionary NAS

### Exercise 1 (★) — Termination of don't-look-bit 2-opt, its failure mode, and a fix

**Termination.** Every iteration pops a city $x$. Either an improving move is applied, which decreases the tour length
by more than $10^{-10}$, or nothing is applied and the queue shrinks by one (nothing is appended). There are finitely
many tours, so only finitely many moves are applied; between two consecutive moves the queue strictly shrinks, so it
empties after finitely many iterations. (Equivalently, the pair (tour length, queue length) decreases
lexicographically at every iteration and takes finitely many values.)

**Why the result need not be 2-opt optimal.** For two tour edges $e = (u_1, u_2)$ and $f = (v_1, v_2)$, the 2-opt
move is the unique other way of joining the two paths obtained by deleting $e$ and $f$. If the tour reads
$u_1 u_2 \dots v_1 v_2 \dots$, the move adds $(u_1, v_1), (u_2, v_2)$. Now let a later move act on edges $g, h$ with
$g$ on the path $u_2 \to v_1$ and $h$ on the path $v_2 \to u_1$. It reverses the path from the second endpoint of $g$
to the first endpoint of $h$, which contains $f$ but not $e$, so the tour now reads $u_1 u_2 \dots v_2 v_1 \dots$:
the move for the *untouched* pair $(e, f)$ now adds $(u_1, v_2), (u_2, v_1)$, and its delta has changed. The move on
$g, h$ queues only the endpoints of $g$ and $h$, so the pair $(e, f)$ need not be re-examined, and the final tour
can admit an improving move. Section A.2 exhibits this on a non-metric symmetric instance ($n = 11$, seed 4023): the
don't-look-bit result has length 6.3159, and a single reversal reaches 5.8286. Such failures are rare: none occurred
in the 3,000 random non-metric and 3,000 random Euclidean instances ($n = 6..13$) of the block below, and nothing in
the algorithm excludes them.

**A provably correct modification.** When the queue empties, run one full scan of all $n(n-3)/2$ pairs of
non-adjacent edges. If an improving move exists, apply the best one, queue its four endpoints and continue the
don't-look-bit search; otherwise stop. The algorithm stops only after a full scan finds no improving move, so the
output is a 2-opt local optimum by definition. It terminates because every scan either stops the algorithm or applies
an improving move. In the typical case (the don't-look-bit descent was already optimal) the price is exactly one scan.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from collections import deque
from utils import random_euclidean_tsp, tour_length

def two_opt_ls(t, D, queue=None):
    """Lab A.1: 2-opt with don't-look bits. Returns (tour, number of delta evaluations)."""
    t = np.array(t); n = len(t)
    pos = np.empty(n, int); pos[t] = np.arange(n)
    q = deque(range(n) if queue is None else queue)
    inq = np.zeros(n, bool); inq[list(q)] = True
    deltas, ar = 0, np.arange(n)
    while q:
        x = q.popleft(); inq[x] = False
        for p in (pos[x], (pos[x] - 1) % n):
            nxt = np.roll(t, -1); a, b = t[p], nxt[p]
            delta = D[a, t] + D[b, nxt] - D[a, b] - D[t, nxt]
            delta[[(p - 1) % n, p, (p + 1) % n]] = np.inf
            deltas += n - 3
            k = int(np.argmin(delta))
            if delta[k] < -1e-10:
                c, d = t[k], nxt[k]
                lo, hi = (p + 1, k) if p < k else (k + 1, p)
                t[lo:hi + 1] = t[lo:hi + 1][::-1]; pos[t[lo:hi + 1]] = ar[lo:hi + 1]
                for y in (a, b, c, d):
                    if not inq[y]:
                        q.append(y); inq[y] = True
                break
    return t, deltas

def full_scan_best(t, D):
    """Best 2-opt move over all n(n-3)/2 pairs of non-adjacent edges: (delta, (p, k)) or (0, None)."""
    n = len(t); nxt = np.roll(t, -1)
    delta = D[t][:, t] + D[nxt][:, nxt] - D[t, nxt][:, None] - D[t, nxt][None, :]   # edges p and k
    for s in (-1, 0, 1):
        delta[np.arange(n), (np.arange(n) + s) % n] = np.inf
    p, k = np.unravel_index(np.argmin(delta), delta.shape)
    return (delta[p, k], (int(p), int(k))) if delta[p, k] < -1e-10 else (0.0, None)

def two_opt_exact(t, D):
    """Don't-look-bit descent, then a full scan; repeat until a full scan finds no improving move."""
    t, d = two_opt_ls(t, D); n = len(t); scans = 0
    while True:
        d += n * (n - 3) // 2; scans += 1
        delta, mv = full_scan_best(t, D)
        if mv is None:
            return t, d, scans
        p, k = mv; lo, hi = (p + 1, k) if p < k else (k + 1, p)
        ends = [t[p], t[(p + 1) % n], t[k], t[(k + 1) % n]]
        t = t.copy(); t[lo:hi + 1] = t[lo:hi + 1][::-1]
        t, d2 = two_opt_ls(t, D, ends); d += d2

def is_2opt_optimal(t, D):
    L0, n = tour_length(t, D), len(t)
    return all(tour_length(np.r_[t[:i + 1], t[i + 1:j + 1][::-1], t[j + 1:]], D) >= L0 - 1e-9
               for i in range(n) for j in range(i + 1, n))

def nonmetric(n, rng):
    Dn = rng.random((n, n)); Dn = Dn + Dn.T; np.fill_diagonal(Dn, 0.0); return Dn

# the counterexample of Section A.2
g = np.random.default_rng(4023); Dn = nonmetric(11, g); t0 = g.permutation(11)
t_dlb, _ = two_opt_ls(t0, Dn); t_ex, _, scans = two_opt_exact(t0, Dn)
print(f"counterexample: don't-look bits {tour_length(t_dlb, Dn):.4f} (2-opt optimal: {is_2opt_optimal(t_dlb, Dn)}); "
      f"with final scans {tour_length(t_ex, Dn):.4f} after {scans} scans (2-opt optimal: {is_2opt_optimal(t_ex, Dn)})")

rng = np.random.default_rng(0)
fail_nm = fail_eu = fail_exact = 0
for _ in range(3000):
    n = int(rng.integers(6, 14)); Dn = nonmetric(n, rng); t0 = rng.permutation(n)
    fail_nm += not is_2opt_optimal(two_opt_ls(t0, Dn)[0], Dn)
    fail_exact += not is_2opt_optimal(two_opt_exact(t0, Dn)[0], Dn)
    _, De = random_euclidean_tsp(n, rng)
    fail_eu += not is_2opt_optimal(two_opt_ls(rng.permutation(n), De)[0], De)
print(f"3,000 random instances (n = 6..13): don't-look-bit failures non-metric {fail_nm}, Euclidean {fail_eu}; "
      f"failures of the modified search {fail_exact}")

_, D100 = random_euclidean_tsp(100, np.random.default_rng(2024))      # the Lab instance
dd = np.mean([two_opt_ls(rng.permutation(100), D100)[1] for _ in range(20)])
print(f"n = 100: a descent evaluates {dd:,.0f} deltas on average; one full scan adds {100 * 97 // 2:,} "
      f"({100 * 4850 / dd:.0f}%)")
```

On the counterexample the modified search reaches length 5.4872 after 2 scans and is certified 2-opt optimal by
brute force; on the 3,000 random non-metric instances it never leaves an improving move. On the Lab's Euclidean
$n = 100$ instance a don't-look-bit descent from a random tour evaluates about 38,400 deltas, so the confirming scan
(4,850 deltas) adds about 13% to the cost of a descent. That is the price of the guarantee.

### Exercise 2 (★) — Expected regret of random search with replacement

With $B$ i.i.d. uniform draws from $M$ configurations, the best rank found is $\ge r$ iff all draws have rank
$\ge r$: $P(R \ge r) = \big((M-r+1)/M\big)^B$. Hence

$$P(R = r) = \Big(\frac{M-r+1}{M}\Big)^B - \Big(\frac{M-r}{M}\Big)^B, \qquad E[\text{regret}] = \sum_{r=1}^{M} P(R = r)\,(\ell_{(r)} - \ell_{(1)}),$$

to be compared with $P(R = r) = \binom{M-r}{B-1}/\binom{M}{B}$ without replacement. The block rebuilds the Lab's table
($M = 216$ configurations, about 25 s):

```python
import sys; sys.path.insert(0, "..")
import numpy as np
import itertools, time
from math import comb

def spirals(n_per, K, rng, noise=0.5):
    X, y = [], []
    for k in range(K):
        r = np.linspace(0.05, 1, n_per)
        th = np.linspace(2 * np.pi * k / K, 2 * np.pi * k / K + 5.0, n_per) + noise * rng.standard_normal(n_per)
        X.append(np.c_[r * np.sin(th), r * np.cos(th)]); y.append(np.full(n_per, k))
    X, y = np.concatenate(X), np.concatenate(y); p = rng.permutation(len(y))
    return X[p], y[p]

drng = np.random.default_rng(0)
Xtr, ytr = spirals(100, 3, drng); Xva, yva = spirals(300, 3, drng)       # the Lab's data

def forward(X, W, B, act):
    H = [X]
    for i, (w, b) in enumerate(zip(W, B)):
        z = H[-1] @ w + b
        H.append(z if i == len(W) - 1 else (np.tanh(z) if act == "tanh" else np.maximum(z, 0)))
    return H

def train_mlp(widths, act, lr, l2, steps=100, seed=0):
    """The Lab's full-batch Adam trainer, with one width per hidden layer; returns the validation loss."""
    r = np.random.default_rng(seed)
    sizes = [2] + list(widths) + [3]
    W = [r.standard_normal((a, b)) * np.sqrt((2.0 if act == "relu" else 1.0) / a) for a, b in zip(sizes[:-1], sizes[1:])]
    B = [np.zeros(b) for b in sizes[1:]]; params = W + B
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]; Y = np.eye(3)[ytr]
    for t in range(1, steps + 1):
        H = forward(Xtr, W, B, act)
        z = H[-1] - H[-1].max(1, keepdims=True); P = np.exp(z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / len(ytr); gW, gB = [None] * len(W), [None] * len(W)
        for i in range(len(W) - 1, -1, -1):
            gW[i] = H[i].T @ G + l2 * W[i]; gB[i] = G.sum(0)
            if i > 0:
                G = G @ W[i].T; G = G * (1 - H[i] ** 2) if act == "tanh" else G * (H[i] > 0)
        for k, (p, g) in enumerate(zip(params, gW + gB)):
            m[k] = 0.9 * m[k] + 0.1 * g; v[k] = 0.999 * v[k] + 0.001 * g * g
            p -= lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v[k] / (1 - 0.999**t)) + 1e-8)
    h = forward(Xva, W, B, act)[-1]; z = h - h.max(1, keepdims=True)
    return float(-(z - np.log(np.exp(z).sum(1, keepdims=True)))[np.arange(len(yva)), yva].mean())

SPACE = {"depth": [1, 2, 3], "width": [8, 16, 32], "act": ["tanh", "relu"], "lr": [0.003, 0.01, 0.03, 0.1],
         "l2": [0.0, 1e-4, 1e-3]}
CARD = [len(s) for s in SPACE.values()]
KEYS = list(itertools.product(*[range(c) for c in CARD]))

def table(seed=0):
    """Validation loss of every configuration of the Lab's 216-point space (depth x width x act x lr x l2)."""
    return {k: train_mlp([SPACE["width"][k[1]]] * SPACE["depth"][k[0]], SPACE["act"][k[2]], SPACE["lr"][k[3]],
                         SPACE["l2"][k[4]], seed=seed) for k in KEYS}

t0 = time.perf_counter(); TABLE = table(); print(f"216 configurations trained in {time.perf_counter() - t0:.1f} s")
sl = np.sort(list(TABLE.values())); M = len(sl); r = np.arange(1, M + 1)
print(f"best validation loss {sl[0]:.4f} (the Lab reports 0.1969)")
print(f"{'B':>4s}  without replacement   with replacement   ratio")
for B in (5, 10, 20, 50, 100, 216):
    p_wo = np.array([comb(M - k, B - 1) / comb(M, B) for k in r])
    p_w = ((M - r + 1) / M) ** B - ((M - r) / M) ** B
    a, b = p_wo @ (sl - sl[0]), p_w @ (sl - sl[0])
    print(f"{B:4d}  {a:19.4f}   {b:16.4f}   {b / a if a > 0 else np.inf:5.2f}")
rng = np.random.default_rng(0)
draws = rng.integers(0, M, (200_000, 20))
print(f"simulation, 2e5 runs with replacement, B = 20: {(sl[draws].min(1) - sl[0]).mean():.5f}")
print(f"expected repeated draws at B = 20: {20 - M * (1 - (1 - 1 / M) ** 20):.2f};  P(miss the best | B = M) = {(1 - 1 / M) ** M:.3f}")
```

| $B$ | 5 | 10 | 20 | 50 | 100 | 216 |
|---|---|---|---|---|---|---|
| without replacement | 0.0674 | 0.0305 | 0.0139 | 0.0057 | 0.0031 | 0 |
| with replacement | 0.0683 | 0.0313 | 0.0146 | 0.0063 | 0.0039 | 0.0019 |

A simulation of $2\times10^5$ runs with replacement at $B = 20$ gives $0.01460$. At $B = 20$ the difference is 5%: the
expected number of repeated draws is $B - M\big(1 - (1-1/M)^B\big) = 0.86 \approx B^2/(2M)$, so fewer than one
evaluation is wasted. The difference matters once $B^2/(2M)$ is not small, i.e. once $B$ is of the order of
$\sqrt{M}$ or more: at $B = 100$ it is 26%, and at $B = M$ sampling without replacement is exhaustive (regret 0),
while sampling with replacement still misses the best with probability $(1-1/M)^M = 0.367 \approx e^{-1}$.

### Exercise 3 (★★) — Or-opt and 2-opt + Or-opt inside the memetic algorithm

A single don't-look-bit local search with two move types. For a popped city $x$ it evaluates the 2-opt moves of
both edges of $x$ and the Or-opt moves that take the segment of 1–3 cities starting at $x$ and insert it, in either
orientation, between any two consecutive cities of the rest of the tour. It applies the best improving move and
queues the endpoints of all changed edges. The cost model is the Lab's, expressed in distance reads: a full
evaluation costs 1 unit and every distance read of a local search $1/n$ unit (a 2-opt delta reads 4 distances, i.e.
$4/n$ units; an Or-opt insertion delta reads 6, plus 3 for the removal). The 2-opt variant uses the Lab's `two_opt_ls`
unchanged, so its row reproduces the Lab exactly.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from collections import deque
from scipy import stats
from utils import random_euclidean_tsp, tour_length

def two_opt_ls(t, D, queue=None):
    """Lab A.1: 2-opt with don't-look bits. Returns (tour, number of delta evaluations)."""
    t = np.array(t); n = len(t)
    pos = np.empty(n, int); pos[t] = np.arange(n)
    q = deque(range(n) if queue is None else queue)
    inq = np.zeros(n, bool); inq[list(q)] = True
    deltas, ar = 0, np.arange(n)
    while q:
        x = q.popleft(); inq[x] = False
        for p in (pos[x], (pos[x] - 1) % n):
            nxt = np.roll(t, -1); a, b = t[p], nxt[p]
            delta = D[a, t] + D[b, nxt] - D[a, b] - D[t, nxt]
            delta[[(p - 1) % n, p, (p + 1) % n]] = np.inf
            deltas += n - 3
            k = int(np.argmin(delta))
            if delta[k] < -1e-10:
                c, d = t[k], nxt[k]
                lo, hi = (p + 1, k) if p < k else (k + 1, p)
                t[lo:hi + 1] = t[lo:hi + 1][::-1]; pos[t[lo:hi + 1]] = ar[lo:hi + 1]
                for y in (a, b, c, d):
                    if not inq[y]:
                        q.append(y); inq[y] = True
                break
    return t, deltas

def ox(p1, p2, rng):
    """Order crossover (Davis 1985); same random cut draw as the Lab's vectorised ox_batch."""
    n = len(p1); a, b = np.sort(np.argsort(rng.random(n + 1))[:2])
    seg = set(p1[a:b].tolist()); child = p1.copy()
    fill = [c for c in np.r_[p2[b:], p2[:b]].tolist() if c not in seg]
    child[[(b + k) % n for k in range(n - (b - a))]] = fill
    return child

def edge_set(t):
    u = np.roll(t, -1)
    return set(zip(np.minimum(t, u).tolist(), np.maximum(t, u).tolist()))

def memetic(D, rng, budget, ls, N=10, xover=None):
    """The Lab's steady-state Lamarckian memetic GA. ls(t, D, queue) -> (tour, distance reads);
    a full evaluation costs 1 unit and every distance read of a local search 1/n unit."""
    n = len(D); units = 0.0
    def improve(t, queue=None):
        nonlocal units
        t2, reads = ls(t, D, queue); units += 1 + reads / n
        return t2
    geno = [improve(rng.permutation(n)) for _ in range(N)]
    fit = np.array([tour_length(g, D) for g in geno]); best = fit.min()
    while units < budget:
        i1, i2 = rng.choice(N, 2, replace=False), rng.choice(N, 2, replace=False)
        a, b = geno[i1[np.argmin(fit[i1])]], geno[i2[np.argmin(fit[i2])]]
        c = (xover or ox)(a, b, rng)
        foreign = edge_set(c) - (edge_set(a) | edge_set(b))
        imp = improve(c, sorted({x for e in foreign for x in e}))
        if units > budget:
            break
        fc = tour_length(imp, D); w = np.argmax(fit)
        if fc < fit[w] and not np.any(np.isclose(fit, fc)):
            geno[w], fit[w] = imp, fc
        best = min(best, fc)
    return best

def two_opt_reads(t, D, queue=None):
    t, d = two_opt_ls(t, D, queue); return t, 4 * d                     # a 2-opt delta reads 4 distances

def ls_moves(t, D, queue=None, use_2opt=True, use_or=True):
    """Don't-look-bit local search with 2-opt and/or Or-opt (segments of 1-3 cities starting at the popped
    city, reinserted in either orientation). Applies the best improving move. Returns (tour, distance reads)."""
    t = np.array(t); n = len(t)
    q = deque(range(n) if queue is None else queue); inq = np.zeros(n, bool); inq[list(q)] = True
    reads = 0
    while q:
        x = q.popleft(); inq[x] = False
        pos = np.empty(n, int); pos[t] = np.arange(n)
        best, move = -1e-10, None
        if use_2opt:
            nxt = np.roll(t, -1)
            for p in (pos[x], (pos[x] - 1) % n):
                a, b = t[p], nxt[p]
                delta = D[a, t] + D[b, nxt] - D[a, b] - D[t, nxt]
                delta[[(p - 1) % n, p, (p + 1) % n]] = np.inf; reads += 4 * (n - 3)
                k = int(np.argmin(delta))
                if delta[k] < best:
                    best, move = delta[k], ("2opt", p, k)
        if use_or:
            r = np.roll(t, -(pos[x] - 1))                                  # r[0] = predecessor, r[1] = x
            for Ls in (1, 2, 3):
                pz, seg, R = r[0], r[1:Ls + 1], np.r_[r[Ls + 1:], r[0]]    # R: path q ... pz without the segment
                rem = D[pz, R[0]] - D[pz, seg[0]] - D[seg[-1], R[0]]
                A, B = R[:-1], R[1:]
                ins = D[A, seg[0]] + D[seg[-1], B] - D[A, B]
                insr = D[A, seg[-1]] + D[seg[0], B] - D[A, B]
                reads += 3 + 6 * len(A)
                for rev, arr in ((False, ins), (True, insr)):
                    k = int(np.argmin(arr))
                    if rem + arr[k] < best:
                        best, move = rem + arr[k], ("or", R, seg, k, rev)
        if move is None:
            continue
        if move[0] == "2opt":
            _, p, k = move; nxt = np.roll(t, -1)
            ends = [t[p], nxt[p], t[k], nxt[k]]
            lo, hi = (p + 1, k) if p < k else (k + 1, p)
            t[lo:hi + 1] = t[lo:hi + 1][::-1]
        else:
            _, R, seg, k, rev = move
            ends = [R[0], R[-1], seg[0], seg[-1], R[k], R[k + 1]]
            t = np.r_[R[:k + 1], seg[::-1] if rev else seg, R[k + 1:]]
        for y in ends:
            if not inq[y]:
                q.append(y); inq[y] = True
    return t, reads

def or_opt(t, D, queue=None): return ls_moves(t, D, queue, use_2opt=False)
def two_or(t, D, queue=None): return ls_moves(t, D, queue)

def improving_move_left(t, D, use_2opt, use_or):
    """Brute force over the whole neighbourhood, lengths recomputed from scratch."""
    n, L0 = len(t), tour_length(t, D)
    if use_2opt and any(tour_length(np.r_[t[:i + 1], t[i + 1:j + 1][::-1], t[j + 1:]], D) < L0 - 1e-9
                        for i in range(n) for j in range(i + 1, n)):
        return True
    if use_or:
        for s in range(n):
            r = np.roll(t, -s)
            for Ls in (1, 2, 3):
                seg, R = r[:Ls], r[Ls:]
                for k in range(len(R) - 1):
                    for sg in (seg, seg[::-1]):
                        if tour_length(np.r_[R[:k + 1], sg, R[k + 1:]], D) < L0 - 1e-9:
                            return True
    return False

_, D30 = random_euclidean_tsp(30, np.random.default_rng(1)); g = np.random.default_rng(0)
for name, ls, u2, uo in (("2-opt", two_opt_reads, True, False), ("Or-opt", or_opt, False, True),
                         ("2-opt + Or-opt", two_or, True, True)):
    left = sum(improving_move_left(ls(g.permutation(30), D30)[0], D30, u2, uo) for _ in range(10))
    print(f"{name:15s}: an improving move of its own neighbourhood remains in {left} of 10 descents (n = 30)")

_, D100 = random_euclidean_tsp(100, np.random.default_rng(2024))        # the Lab instance
res = {}
for name, ls in (("2-opt", two_opt_reads), ("Or-opt", or_opt), ("2-opt + Or-opt", two_or)):
    res[name] = np.array([memetic(D100, np.random.default_rng(100 + s), 60_000, ls) for s in range(6)])
    v = res[name]
    print(f"memetic OX, {name:15s}: {np.median(v):.4f} [{np.percentile(v, 25):.4f}, {np.percentile(v, 75):.4f}]")
print(f"one-sided Wilcoxon, 2-opt + Or-opt < 2-opt: p = {stats.wilcoxon(res['2-opt + Or-opt'], res['2-opt'], alternative='less').pvalue:.2f}")
```

All three searches pass brute-force local-optimality checks of their own neighbourhood on 10 descents of a 30-city
instance. Memetic OX, Lamarckian, 60,000 units, 6 seeds, the Lab's 100-city instance:

| local search | final length, median [IQR] |
|---|---|
| 2-opt | 8.0315 [8.0271, 8.0442] |
| Or-opt | 8.5032 [8.4177, 8.5473] |
| 2-opt + Or-opt | 8.0691 [8.0610, 8.0849] |

**2-opt alone is best at this budget.** Or-opt alone is much weaker: it cannot remove crossings efficiently, and each
popped city costs about $36n$ distance reads (3 segment lengths × 2 orientations × $n$ insertion points × 6 reads)
against $8n$ for 2-opt. The combination pays both costs for every popped city, so for the same budget the algorithm
sees far fewer children, and it does not beat 2-opt (one-sided Wilcoxon $p = 0.98$). A richer neighbourhood pays off
only when its extra cost is recovered, e.g. with neighbour lists (as in Or-3opt or Lin–Kernighan implementations)
that make each move evaluation $O(1)$ instead of $O(n)$.

### Exercise 4 (★★) — Distance-preserving crossover (DPX)

DPX (Freisleben & Merz 1996) copies the edges common to both parents, which form a set of path fragments (single
cities included). It then reconnects the fragments greedily, from the current end to the nearest free fragment
endpoint, using only edges contained in neither parent whenever possible. As a result the child is (almost) exactly
as far from each parent as the parents are from each other.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from collections import deque
from scipy import stats
from utils import random_euclidean_tsp, tour_length

def two_opt_ls(t, D, queue=None):
    """Lab A.1: 2-opt with don't-look bits. Returns (tour, number of delta evaluations)."""
    t = np.array(t); n = len(t)
    pos = np.empty(n, int); pos[t] = np.arange(n)
    q = deque(range(n) if queue is None else queue)
    inq = np.zeros(n, bool); inq[list(q)] = True
    deltas, ar = 0, np.arange(n)
    while q:
        x = q.popleft(); inq[x] = False
        for p in (pos[x], (pos[x] - 1) % n):
            nxt = np.roll(t, -1); a, b = t[p], nxt[p]
            delta = D[a, t] + D[b, nxt] - D[a, b] - D[t, nxt]
            delta[[(p - 1) % n, p, (p + 1) % n]] = np.inf
            deltas += n - 3
            k = int(np.argmin(delta))
            if delta[k] < -1e-10:
                c, d = t[k], nxt[k]
                lo, hi = (p + 1, k) if p < k else (k + 1, p)
                t[lo:hi + 1] = t[lo:hi + 1][::-1]; pos[t[lo:hi + 1]] = ar[lo:hi + 1]
                for y in (a, b, c, d):
                    if not inq[y]:
                        q.append(y); inq[y] = True
                break
    return t, deltas

def ox(p1, p2, rng):
    """Order crossover (Davis 1985); same random cut draw as the Lab's vectorised ox_batch."""
    n = len(p1); a, b = np.sort(np.argsort(rng.random(n + 1))[:2])
    seg = set(p1[a:b].tolist()); child = p1.copy()
    fill = [c for c in np.r_[p2[b:], p2[:b]].tolist() if c not in seg]
    child[[(b + k) % n for k in range(n - (b - a))]] = fill
    return child

def edge_set(t):
    u = np.roll(t, -1)
    return set(zip(np.minimum(t, u).tolist(), np.maximum(t, u).tolist()))

def memetic(D, rng, budget, ls, N=10, xover=None):
    """The Lab's steady-state Lamarckian memetic GA. ls(t, D, queue) -> (tour, distance reads);
    a full evaluation costs 1 unit and every distance read of a local search 1/n unit."""
    n = len(D); units = 0.0
    def improve(t, queue=None):
        nonlocal units
        t2, reads = ls(t, D, queue); units += 1 + reads / n
        return t2
    geno = [improve(rng.permutation(n)) for _ in range(N)]
    fit = np.array([tour_length(g, D) for g in geno]); best = fit.min()
    while units < budget:
        i1, i2 = rng.choice(N, 2, replace=False), rng.choice(N, 2, replace=False)
        a, b = geno[i1[np.argmin(fit[i1])]], geno[i2[np.argmin(fit[i2])]]
        c = (xover or ox)(a, b, rng)
        foreign = edge_set(c) - (edge_set(a) | edge_set(b))
        imp = improve(c, sorted({x for e in foreign for x in e}))
        if units > budget:
            break
        fc = tour_length(imp, D); w = np.argmax(fit)
        if fc < fit[w] and not np.any(np.isclose(fit, fc)):
            geno[w], fit[w] = imp, fc
        best = min(best, fc)
    return best

def two_opt_reads(t, D, queue=None):
    t, d = two_opt_ls(t, D, queue); return t, 4 * d

def erx(p1, p2, rng):
    """The Lab's edge recombination."""
    n = len(p1); adj = [set() for _ in range(n)]
    for p in (p1.tolist(), p2.tolist()):
        for i in range(n):
            adj[p[i]].add(p[i - 1]); adj[p[i - 1]].add(p[i])
    c = int(p1[0] if rng.random() < 0.5 else p2[0]); child, unvisited = [c], set(range(n)) - {c}
    u = rng.random(n)
    for step in range(1, n):
        for nb in adj[c]:
            adj[nb].discard(c)
        if adj[c]:
            dmin = min(len(adj[x]) for x in adj[c]); ties = [x for x in adj[c] if len(adj[x]) == dmin]
            c = ties[int(u[step] * len(ties))]
        else:
            rest = sorted(unvisited); c = rest[int(u[step] * len(rest))]
        child.append(c); unvisited.discard(c)
    return np.array(child)

def dpx(a, b, D, rng):
    """Distance-preserving crossover (Freisleben & Merz 1996)."""
    n = len(a); Ea, Eb = edge_set(a), edge_set(b); common, parent = Ea & Eb, Ea | Eb
    adj = [[] for _ in range(n)]
    for u, v in common:
        adj[u].append(v); adj[v].append(u)
    seen = np.zeros(n, bool); frags = []
    for s in range(n):                                   # walk each common-edge fragment from an endpoint
        if seen[s] or len(adj[s]) == 2:
            continue
        path, prev, cur = [s], -1, s; seen[s] = True
        while True:
            nx = [y for y in adj[cur] if y != prev and not seen[y]]
            if not nx:
                break
            prev, cur = cur, nx[0]; path.append(cur); seen[cur] = True
        frags.append(path)
    if not frags:                                        # identical parents
        return a.copy()
    endp = {}
    for i, f in enumerate(frags):
        endp[f[0]] = i; endp[f[-1]] = i
    i0 = int(rng.integers(len(frags))); tour, used = list(frags[i0]), {i0}
    while len(used) < len(frags):                        # greedy reconnection, avoiding parental edges if possible
        e = tour[-1]; cands = [x for x, i in endp.items() if i not in used]
        good = [x for x in cands if (min(e, x), max(e, x)) not in parent]
        x = min(good or cands, key=lambda y: D[e, y]); f = frags[endp[x]]
        tour += f if f[0] == x else f[::-1]; used.add(endp[x])
    return np.array(tour)

_, D100 = random_euclidean_tsp(100, np.random.default_rng(2024))        # the Lab instance
rng = np.random.default_rng(0)
ops = {"OX": ox, "ERX": erx, "DPX": lambda a, b, g: dpx(a, b, D100, g)}
herit = {k: [] for k in ops}; valid = True
for _ in range(20):
    a = two_opt_ls(rng.permutation(100), D100)[0]; b = two_opt_ls(rng.permutation(100), D100)[0]
    for name, op in ops.items():
        c = op(a, b, rng)
        herit[name].append(len(edge_set(c) & (edge_set(a) | edge_set(b))) / 100)
        if name == "DPX":
            valid &= sorted(c.tolist()) == list(range(100)) and (edge_set(a) & edge_set(b)) <= edge_set(c)
print("every DPX child is a permutation containing all common edges:", valid)
print("heritability on 20 pairs of 2-opt local optima:", ", ".join(f"{k} {np.mean(v):.3f}" for k, v in herit.items()))
fin = {}
for name, op in ops.items():
    v = fin[name] = np.array([memetic(D100, np.random.default_rng(100 + s), 60_000, two_opt_reads, xover=op) for s in range(6)])
    print(f"Lamarckian memetic with {name}: {np.median(v):.4f} [{np.percentile(v, 25):.4f}, {np.percentile(v, 75):.4f}]")
print(f"Friedman test over the 6 paired seeds: p = {stats.friedmanchisquare(*fin.values()).pvalue:.2f}")
```

On 20 pairs of 2-opt local optima of the 100-city instance every DPX child is a valid tour containing every common
edge. Heritability (fraction of child edges found in a parent) is OX 0.955, ERX 0.948 and **DPX 0.597**: by design
DPX replaces every non-common edge by a new one. Inside the Lamarckian 2-opt memetic algorithm (60,000 units, 6
seeds) the three crossovers are statistically indistinguishable (Friedman test $p = 0.85$): OX 8.0315
[8.0271, 8.0442], ERX 8.0386 [8.0276, 8.0467], DPX 8.0655 [8.0319, 8.0892] (the OX and ERX rows reproduce the Lab).
DPX has low heritability but keeps *exactly* the structure the parents agree on, and the greedy reconnection uses
short edges, so 2-opt repairs the child quickly. DPX was designed for exactly this setting, recombining local optima in
a memetic algorithm, and its advantage is expected on larger instances, where the common-edge backbone of good local
optima is a stronger signal. At $n = 100$ the backbone is easy to find with any of the three operators.

### Exercise 5 (★★) — Aging vs non-aging evolution under evaluation noise

Every configuration is trained with 3 seeds (648 training runs, about 80 s; 3 rather than more seeds keeps the block
under two minutes). A query returns the validation loss of one randomly chosen seed, and *every* query costs one
evaluation, repeated ones included (a noisy query is a new training run). Each algorithm returns the configuration
with the best *observed* loss, and its regret is measured on the *mean* over seeds. Settings: 60 evaluations,
$P = 8$, $s = 3$, 400 runs; random search queries 60 distinct configurations once each.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
import itertools
from collections import deque
from scipy import stats

def spirals(n_per, K, rng, noise=0.5):
    X, y = [], []
    for k in range(K):
        r = np.linspace(0.05, 1, n_per)
        th = np.linspace(2 * np.pi * k / K, 2 * np.pi * k / K + 5.0, n_per) + noise * rng.standard_normal(n_per)
        X.append(np.c_[r * np.sin(th), r * np.cos(th)]); y.append(np.full(n_per, k))
    X, y = np.concatenate(X), np.concatenate(y); p = rng.permutation(len(y))
    return X[p], y[p]

drng = np.random.default_rng(0)
Xtr, ytr = spirals(100, 3, drng); Xva, yva = spirals(300, 3, drng)       # the Lab's data

def forward(X, W, B, act):
    H = [X]
    for i, (w, b) in enumerate(zip(W, B)):
        z = H[-1] @ w + b
        H.append(z if i == len(W) - 1 else (np.tanh(z) if act == "tanh" else np.maximum(z, 0)))
    return H

def train_mlp(widths, act, lr, l2, steps=100, seed=0):
    """The Lab's full-batch Adam trainer, with one width per hidden layer; returns the validation loss."""
    r = np.random.default_rng(seed)
    sizes = [2] + list(widths) + [3]
    W = [r.standard_normal((a, b)) * np.sqrt((2.0 if act == "relu" else 1.0) / a) for a, b in zip(sizes[:-1], sizes[1:])]
    B = [np.zeros(b) for b in sizes[1:]]; params = W + B
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]; Y = np.eye(3)[ytr]
    for t in range(1, steps + 1):
        H = forward(Xtr, W, B, act)
        z = H[-1] - H[-1].max(1, keepdims=True); P = np.exp(z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / len(ytr); gW, gB = [None] * len(W), [None] * len(W)
        for i in range(len(W) - 1, -1, -1):
            gW[i] = H[i].T @ G + l2 * W[i]; gB[i] = G.sum(0)
            if i > 0:
                G = G @ W[i].T; G = G * (1 - H[i] ** 2) if act == "tanh" else G * (H[i] > 0)
        for k, (p, g) in enumerate(zip(params, gW + gB)):
            m[k] = 0.9 * m[k] + 0.1 * g; v[k] = 0.999 * v[k] + 0.001 * g * g
            p -= lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v[k] / (1 - 0.999**t)) + 1e-8)
    h = forward(Xva, W, B, act)[-1]; z = h - h.max(1, keepdims=True)
    return float(-(z - np.log(np.exp(z).sum(1, keepdims=True)))[np.arange(len(yva)), yva].mean())

SPACE = {"depth": [1, 2, 3], "width": [8, 16, 32], "act": ["tanh", "relu"], "lr": [0.003, 0.01, 0.03, 0.1],
         "l2": [0.0, 1e-4, 1e-3]}
CARD = [len(s) for s in SPACE.values()]
KEYS = list(itertools.product(*[range(c) for c in CARD]))

def table(seed=0):
    """Validation loss of every configuration of the Lab's 216-point space (depth x width x act x lr x l2)."""
    return {k: train_mlp([SPACE["width"][k[1]]] * SPACE["depth"][k[0]], SPACE["act"][k[2]], SPACE["lr"][k[3]],
                         SPACE["l2"][k[4]], seed=seed) for k in KEYS}

SEEDS = 3                                        # training seeds per configuration (648 training runs, ~70 s)
L = np.array([[t[k] for k in KEYS] for t in (table(seed) for seed in range(SEEDS))])   # (SEEDS, 216)
mean_loss = L.mean(0); idx = {k: i for i, k in enumerate(KEYS)}
print(f"mean seed-to-seed std of the validation loss: {L.std(0, ddof=1).mean():.3f}; "
      f"best mean loss {mean_loss.min():.4f}; 10th-best {np.sort(mean_loss)[9]:.4f}")

class NoisyOracle:
    """Each query = one training run with a random seed; returns its validation loss and records the best observation."""
    def __init__(self, budget, rng):
        self.left, self.rng, self.best, self.seen = budget, rng, (np.inf, None), set()
    def __call__(self, key):
        if self.left == 0:
            raise StopIteration
        self.left -= 1; self.seen.add(key)
        y = L[self.rng.integers(SEEDS), idx[key]]
        self.best = min(self.best, (y, key))
        return y

def mutate_one(key, rng):
    g = int(rng.integers(len(CARD))); new = int(rng.integers(CARD[g] - 1)); new += new >= key[g]
    return key[:g] + (new,) + key[g + 1:]

def evolution(orc, rng, P=8, s=3, aging=True):
    pop = deque()
    while len(pop) < P:
        k = tuple(int(rng.integers(c)) for c in CARD); pop.append((k, orc(k)))
    while True:
        parent = min((pop[i] for i in rng.choice(len(pop), s, replace=False)), key=lambda e: e[1])[0]
        child = mutate_one(parent, rng); pop.append((child, orc(child)))
        if aging:
            pop.popleft()
        else:
            pop.remove(max(pop, key=lambda e: e[1]))

def random_search(orc, rng):
    for i in rng.permutation(len(KEYS)):
        orc(KEYS[i])

def true_regret(algo, seed, budget=60, **kw):
    rng = np.random.default_rng(seed); orc = NoisyOracle(budget, rng)
    try:
        algo(orc, rng, **kw)
    except StopIteration:
        pass
    return mean_loss[idx[orc.best[1]]] - mean_loss.min(), len(orc.seen)   # regret of the returned configuration

R = {name: np.array([true_regret(a, 5000 + s, **kw) for s in range(400)])
     for name, (a, kw) in {"regularised evolution (aging)": (evolution, {"aging": True}),
                           "evolution, remove worst": (evolution, {"aging": False}),
                           "random search": (random_search, {})}.items()}
for name, r in R.items():
    print(f"{name:30s} true regret {r[:, 0].mean():.4f} ± {r[:, 0].std() / np.sqrt(len(r)):.4f}   "
          f"distinct configurations queried: {r[:, 1].mean():.1f}")
    R[name] = r[:, 0]
p = stats.mannwhitneyu(R["regularised evolution (aging)"], R["evolution, remove worst"], alternative="less").pvalue
print(f"one-sided Mann-Whitney, aging < remove worst: p = {p:.2f}")
p = stats.mannwhitneyu(R["random search"], R["regularised evolution (aging)"]).pvalue
print(f"two-sided Mann-Whitney, random search vs aging: p = {p:.3f}")
```

The mean seed-to-seed standard deviation of the loss is 0.043, comparable to the spread between good configurations
(the best mean loss is 0.1922, the 10th best 0.2135).

| algorithm | true regret of the returned configuration, mean ± SE | distinct configurations queried |
|---|---|---|
| regularised evolution (aging) | 0.0232 ± 0.0012 | 41.4 |
| evolution, remove worst | 0.0236 ± 0.0012 | 34.4 |
| random search | 0.0194 ± 0.0011 | 60.0 |

We do **not** observe an advantage of aging at this scale (one-sided Mann–Whitney $p = 0.26$), and under noise both
evolutionary variants now lose to random search (random search vs aging, two-sided $p = 0.030$). The last column
explains most of it: with costly re-evaluations, evolution spends a third of its budget re-training configurations it
has already seen (non-aging evolution even more, because it keeps its members and mutates them repeatedly), and it
returns the configuration with the best of several noisy observations — a "winner's curse" that favours
high-variance configurations. The ordering is also fragile: setting `SEEDS = 5` (about 150 s) gives 0.0223, 0.0190
and 0.0223 for aging, non-aging and random search. Real et al. (2019) argued that aging helps because a lucky noisy
evaluation cannot keep a mediocre architecture in the population forever. With $P = 8$ and 60 evaluations, however,
the whole population is renewed within about 8 steps anyway, so that mechanism has no time to matter. The regularising
effect needs long runs relative to $P$ (thousands of evaluations with $P = 100$ in the paper), where a non-aging
population can stagnate around a few over-estimated members. A fair small-budget protocol under noise would also
re-evaluate or average the final candidates instead of trusting a single best observation.

### Exercise 6 (★★★) — Variable-length architectures

The genotype is a tuple of 1–3 layer widths from $\lbrace 8, 16, 32\rbrace$ plus the activation, learning-rate and
$L_2$ genes. To keep the block under two minutes, $L_2 \in \lbrace 0, 10^{-4}\rbrace$ (the value $10^{-3}$ is
dropped), giving $(3 + 9 + 27)\times 2\times 4\times 2 = 624$ configurations, all trained once (seed 0) with the Lab's
trainer generalised to per-layer widths. The fixed-width configurations (all layers equally wide) form a 144-point
subspace of the same table, so the scaling study compares two nested spaces built from identical training runs.
Mutation picks one of: change one layer's width, flip the activation, change the learning rate, change $L_2$, insert a
random layer (if depth $\lt 3$), or delete a layer (if depth $\gt 1$). In the fixed-width space it changes one of the
five genes (depth, width, activation, learning rate, $L_2$), as in the Lab.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
import itertools
from collections import deque
from math import comb

def spirals(n_per, K, rng, noise=0.5):
    X, y = [], []
    for k in range(K):
        r = np.linspace(0.05, 1, n_per)
        th = np.linspace(2 * np.pi * k / K, 2 * np.pi * k / K + 5.0, n_per) + noise * rng.standard_normal(n_per)
        X.append(np.c_[r * np.sin(th), r * np.cos(th)]); y.append(np.full(n_per, k))
    X, y = np.concatenate(X), np.concatenate(y); p = rng.permutation(len(y))
    return X[p], y[p]

drng = np.random.default_rng(0)
Xtr, ytr = spirals(100, 3, drng); Xva, yva = spirals(300, 3, drng)       # the Lab's data

def forward(X, W, B, act):
    H = [X]
    for i, (w, b) in enumerate(zip(W, B)):
        z = H[-1] @ w + b
        H.append(z if i == len(W) - 1 else (np.tanh(z) if act == "tanh" else np.maximum(z, 0)))
    return H

def train_mlp(widths, act, lr, l2, steps=100, seed=0):
    """The Lab's full-batch Adam trainer, with one width per hidden layer; returns the validation loss."""
    r = np.random.default_rng(seed)
    sizes = [2] + list(widths) + [3]
    W = [r.standard_normal((a, b)) * np.sqrt((2.0 if act == "relu" else 1.0) / a) for a, b in zip(sizes[:-1], sizes[1:])]
    B = [np.zeros(b) for b in sizes[1:]]; params = W + B
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]; Y = np.eye(3)[ytr]
    for t in range(1, steps + 1):
        H = forward(Xtr, W, B, act)
        z = H[-1] - H[-1].max(1, keepdims=True); P = np.exp(z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / len(ytr); gW, gB = [None] * len(W), [None] * len(W)
        for i in range(len(W) - 1, -1, -1):
            gW[i] = H[i].T @ G + l2 * W[i]; gB[i] = G.sum(0)
            if i > 0:
                G = G @ W[i].T; G = G * (1 - H[i] ** 2) if act == "tanh" else G * (H[i] > 0)
        for k, (p, g) in enumerate(zip(params, gW + gB)):
            m[k] = 0.9 * m[k] + 0.1 * g; v[k] = 0.999 * v[k] + 0.001 * g * g
            p -= lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v[k] / (1 - 0.999**t)) + 1e-8)
    h = forward(Xva, W, B, act)[-1]; z = h - h.max(1, keepdims=True)
    return float(-(z - np.log(np.exp(z).sum(1, keepdims=True)))[np.arange(len(yva)), yva].mean())

SPACE = {"depth": [1, 2, 3], "width": [8, 16, 32], "act": ["tanh", "relu"], "lr": [0.003, 0.01, 0.03, 0.1],
         "l2": [0.0, 1e-4, 1e-3]}
CARD = [len(s) for s in SPACE.values()]
KEYS = list(itertools.product(*[range(c) for c in CARD]))

def table(seed=0):
    """Validation loss of every configuration of the Lab's 216-point space (depth x width x act x lr x l2)."""
    return {k: train_mlp([SPACE["width"][k[1]]] * SPACE["depth"][k[0]], SPACE["act"][k[2]], SPACE["lr"][k[3]],
                         SPACE["l2"][k[4]], seed=seed) for k in KEYS}

# Variable-length space: 1-3 hidden layers, each of width 8/16/32; activation, learning rate; L2 in {0, 1e-4}
# (L2 = 1e-3 dropped to keep the block under two minutes: 39 * 2 * 4 * 2 = 624 training runs).
L2S = [0.0, 1e-4]
ARCHS = [a for d in (1, 2, 3) for a in itertools.product(range(3), repeat=d)]
VAR = {(a, ai, li, ri): train_mlp([SPACE["width"][j] for j in a], SPACE["act"][ai], SPACE["lr"][li], L2S[ri])
       for a in ARCHS for ai in range(2) for li in range(4) for ri in range(2)}
# the fixed-width subspace (all layers equally wide) is contained in it: 9 * 2 * 4 * 2 = 144 configurations
FIX = {k: v for k, v in VAR.items() if len(set(k[0])) == 1}
for name, T in (("variable-length", VAR), ("fixed-width", FIX)):
    print(f"{name:15s} space: {len(T)} configurations, best validation loss {min(T.values()):.4f}")

def mutate_var(k, rng):
    a, ai, li, ri = k; a = list(a)
    ops = ["width", "act", "lr", "l2"] + (["insert"] if len(a) < 3 else []) + (["delete"] if len(a) > 1 else [])
    op = ops[int(rng.integers(len(ops)))]
    if op == "width":
        j = int(rng.integers(len(a))); a[j] = (a[j] + 1 + int(rng.integers(2))) % 3
    elif op == "act":
        ai = 1 - ai
    elif op == "lr":
        li = (li + 1 + int(rng.integers(3))) % 4
    elif op == "l2":
        ri = 1 - ri
    elif op == "insert":
        a.insert(int(rng.integers(len(a) + 1)), int(rng.integers(3)))
    else:
        a.pop(int(rng.integers(len(a))))
    return (tuple(a), ai, li, ri)

def mutate_fix(k, rng):
    """One gene of (depth, width, act, lr, l2) changed to a different value, as in the Lab."""
    a, ai, li, ri = k; d, w = len(a), a[0]
    g = int(rng.integers(5))
    if g == 0:
        d = (d - 1 + 1 + int(rng.integers(2))) % 3 + 1
    elif g == 1:
        w = (w + 1 + int(rng.integers(2))) % 3
    elif g == 2:
        ai = 1 - ai
    elif g == 3:
        li = (li + 1 + int(rng.integers(3))) % 4
    else:
        ri = 1 - ri
    return ((w,) * d, ai, li, ri)

def regularised_evolution(T, mutate, budget, rng, P=8, s=3):
    """Aging evolution; budget counts distinct configurations (repeats hit the cache). Returns the regret."""
    keys = list(T); seen = set(); best = np.inf
    def ev(k):
        nonlocal best
        if k not in seen:
            if len(seen) == budget:
                raise StopIteration
            seen.add(k); best = min(best, T[k])
        return T[k]
    pop = deque()
    try:
        while len(pop) < P:
            k = keys[int(rng.integers(len(keys)))]; pop.append((k, ev(k)))
        while True:
            parent = min((pop[i] for i in rng.choice(P, s, replace=False)), key=lambda e: e[1])[0]
            child = mutate(parent, rng); pop.append((child, ev(child))); pop.popleft()
    except StopIteration:
        pass
    return best - min(T.values())

def random_search_exact(T, B):
    sl = np.sort(list(T.values())); M = len(sl)
    return np.array([comb(M - r, B - 1) / comb(M, B) for r in range(1, M + 1)]) @ (sl - sl[0])

print(f"{'space':15s} {'B':>3s} {'% of space':>10s}   RE regret (400 runs)   random search (exact)   ratio")
for name, T, mut in (("fixed-width", FIX, mutate_fix), ("variable-length", VAR, mutate_var)):
    for B in (10, 20, 40, 80):
        r = np.array([regularised_evolution(T, mut, B, np.random.default_rng(s)) for s in range(400)])
        rs = random_search_exact(T, B)
        print(f"{name:15s} {B:3d} {100 * B / len(T):9.1f}%   {r.mean():.4f} ± {r.std() / 20:.4f}        "
              f"{rs:.4f}             {r.mean() / rs:.2f}")
```

The best validation loss is $0.1761$ in the variable-length space and $0.1969$ in the fixed-width subspace (the Lab's
best, which uses $L_2 = 10^{-4}$). Regularised evolution ($P = 8$, $s = 3$, distinct evaluations counted, 400 runs)
against the exact random-search regret:

| space | budget $B$ | % of space | RE regret | random search (exact) | ratio |
|---|---|---|---|---|---|
| fixed-width (144) | 10 | 6.9% | 0.0336 ± 0.0018 | 0.0362 | 0.93 |
| fixed-width (144) | 20 | 13.9% | 0.0132 ± 0.0009 | 0.0156 | 0.85 |
| fixed-width (144) | 40 | 27.8% | 0.0047 ± 0.0002 | 0.0068 | 0.69 |
| fixed-width (144) | 80 | 55.6% | 0.0019 ± 0.0001 | 0.0031 | 0.60 |
| variable-length (624) | 10 | 1.6% | 0.0389 ± 0.0013 | 0.0385 | 1.01 |
| variable-length (624) | 20 | 3.2% | 0.0233 ± 0.0008 | 0.0243 | 0.96 |
| variable-length (624) | 40 | 6.4% | 0.0144 ± 0.0005 | 0.0155 | 0.93 |
| variable-length (624) | 80 | 12.8% | 0.0073 ± 0.0004 | 0.0096 | 0.76 |

In both spaces the advantage of regularised evolution grows with the budget: it needs a population of reasonable
configurations before local mutation beats uniform sampling, and with $B = 10$ (little more than the $P = 8$ random
initial samples) there is no advantage at all. At a fixed *absolute* budget the advantage is smaller in the larger
space (ratio 0.76 vs 0.60 at $B = 80$), because the same number of evaluations is a smaller fraction of it. At a
fixed *fraction* of the space the two spaces behave alike (0.93 at 6.4–6.9%) or the larger one does better (0.76 at
12.8% vs 0.85 at 13.9%), presumably because the larger space has more good configurations connected by single mutations. The
textbook claim that evolution's edge over random search grows with the size of the space therefore holds only when
the budget grows with the space and the space has exploitable locality. Measuring the advantage as a function of the
*absolute* budget (the regime of NAS practice) is the honest way to report it.
