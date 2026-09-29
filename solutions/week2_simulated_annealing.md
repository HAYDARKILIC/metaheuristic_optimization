# Solutions — Week 2: Simulated Annealing

> Try every exercise yourself before reading on: the proofs are short once you have found the right quantity to look at, and the coding exercises are where the week's lessons really sink in.

Conventions. Every code block is a **standalone script**: run it from inside `week2_simulated_annealing/` (so that
`sys.path.insert(0, "..")` finds `utils/`). The helpers it needs are copied from the notebook, in compact form, into
the block itself. `python tools/run_solution_blocks.py week2_simulated_annealing` runs all of them. All numbers quoted
below are printed by these blocks with the seeds shown. Stochastic comparisons are reported with the number of seeds
and a paired or rank test, because at these budgets many small differences are not significant.

---

## Notebook 1 — The Metropolis Rule and Simulated Annealing

### Exercise 1 (★) — Barker's rule: detailed balance, stationarity, spectral gap versus Metropolis

**Proof.** Take a symmetric proposal $q$ and Barker's acceptance $\alpha_B(x,y)=\pi(y)/(\pi(x)+\pi(y))$. For $x\ne y$,

$$
\pi(x)P_B(x,y)=q(x,y)\,\frac{\pi(x)\pi(y)}{\pi(x)+\pi(y)} ,
$$

which is symmetric in $(x,y)$ because $q$ is. So detailed balance holds, and summing it over $x$ gives $\pi P_B=\pi$.
For the Boltzmann law, $\alpha_B = 1/(1+e^{\Delta/T})$ with $\Delta=E(y)-E(x)$.

**Peskun ordering.** With $r=\pi(y)/\pi(x)$, Metropolis accepts with $\min(1,r)$ and Barker with $r/(1+r)$. Since
$r/(1+r)\le\min(1,r)$, we get $P_M(x,y)\ge P_B(x,y)$ for all $x\ne y$. The Dirichlet form of a reversible kernel is

$$
\mathcal E_P(f)=\tfrac12\sum_{x,y}\pi(x)P(x,y)\big(f(x)-f(y)\big)^2 .
$$

It involves only off-diagonal entries, so $\mathcal E_{P_M}(f)\ge\mathcal E_{P_B}(f)$ for every $f$. For a reversible
kernel the spectral gap is the Rayleigh quotient $1-\lambda_2=\inf_f \mathcal E_P(f)/\mathrm{Var}_\pi(f)$ over
non-constant $f$, hence $\mathrm{gap}(P_M)\ge\mathrm{gap}(P_B)$. Peskun (1973) proved the corresponding statement for
asymptotic variances: Metropolis is never worse than Barker for any estimator.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

E = np.array([[4, 3, 5, 6, 2], [3, 1, 4, 5, 3], [5, 4, 6, 4, 5], [6, 5, 3, 2, 4], [0, 4, 5, 3, 0]], float).ravel()
nbrs = [[r2 * 5 + c2 for r2, c2 in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)) if 0 <= r2 < 5 and 0 <= c2 < 5]
        for r, c in (divmod(i, 5) for i in range(25))]
boltzmann = lambda E, T: np.exp(-(E - E.min()) / T) / np.exp(-(E - E.min()) / T).sum()

def kernel(E, nbrs, T, rule):
    """Lazy symmetric proposal (1/4 per grid direction) with Metropolis or Barker acceptance."""
    P = np.zeros((len(E), len(E)))
    for i in range(len(E)):
        for j in nbrs[i]:
            d = (E[j] - E[i]) / T
            P[i, j] = 0.25 * (np.exp(min(0.0, -d)) if rule == "metropolis" else 1.0 / (1.0 + np.exp(d)))
        P[i, i] = 1.0 - P[i].sum()
    return P

def stationary_by_eigen(P):
    w, V = np.linalg.eig(P.T); v = np.real(V[:, np.argmin(np.abs(w - 1))]); return v / v.sum()

def gap(P, pi):
    s = np.sqrt(pi); S = s[:, None] * P / s[None, :]; S = 0.5 * (S + S.T)
    return 1 - np.sort(np.linalg.eigvalsh(S))[-2]

for T in (0.5, 1.5):
    pi = boltzmann(E, T); PB, PM = kernel(E, nbrs, T, "barker"), kernel(E, nbrs, T, "metropolis")
    flow = pi[:, None] * PB
    gM, gB = gap(PM, pi), gap(PB, pi)
    print(f"T={T}: detailed balance err {np.abs(flow - flow.T).max():.1e}, |eigvec - pi| {np.abs(stationary_by_eigen(PB) - pi).max():.1e}, "
          f"gap Metropolis {gM:.4g}, gap Barker {gB:.4g}, ratio {gM / gB:.3f}")
```

| $T$ | $\max\vert\text{eigvec}-\pi_T\vert$ (Barker) | gap Metropolis | gap Barker | ratio |
|---|---|---|---|---|
| 0.5 | $4.8\times10^{-12}$ | $2.357\times10^{-5}$ | $2.182\times10^{-5}$ | 1.08 |
| 1.5 | $6.5\times10^{-15}$ | $1.078\times10^{-2}$ | $0.841\times10^{-2}$ | 1.28 |

Barker's chain has the right stationary law, and Metropolis mixes faster, as Peskun's ordering predicts. The advantage
shrinks at low $T$. There, the moves that limit mixing are uphill with $r\ll1$, where $r/(1+r)\approx r=\min(1,r)$,
so the two kernels nearly coincide on exactly those moves.

### Exercise 2 (★) — $D^{1/2}PD^{-1/2}$ is symmetric iff $P$ is reversible; real spectrum

Let $A=D^{1/2}PD^{-1/2}$ with $D=\mathrm{diag}(\pi)$ and $\pi\gt0$. Then

$$
A_{xy}=\sqrt{\pi(x)}\,P(x,y)/\sqrt{\pi(y)} .
$$

$A_{xy}=A_{yx}$ holds iff $\pi(x)P(x,y)=\pi(y)P(y,x)$: multiply both sides by $\sqrt{\pi(x)\pi(y)}\gt0$. So $A$ is
symmetric iff $P$ is reversible. $A$ and $P$ are similar matrices, so they have the same eigenvalues. A real symmetric
matrix has real eigenvalues and an orthonormal eigenbasis. Hence a reversible $P$ has a real spectrum, contained in
$[-1,1]$ because $P$ is stochastic. $\square$

### Exercise 3 (★★) — The spectral bound $4\Vert\delta_xP^k-\pi\Vert_{TV}^2\le\frac{1-\pi(x)}{\pi(x)}\lambda_\ast^{2k}$

Let $A=D^{1/2}PD^{-1/2}=\sum_j\lambda_j\varphi_j\varphi_j^\top$ with orthonormal $\varphi_j$ (Exercise 2), where
$\lambda_1=1$ and $\varphi_1=\sqrt\pi$. Put $f_j=D^{-1/2}\varphi_j$. These functions are orthonormal in
$\ell^2(\pi)$, satisfy $Pf_j=\lambda_jf_j$, and $f_1\equiv1$. From $P^k=D^{-1/2}A^kD^{1/2}$,

$$
\frac{P^k(x,y)}{\pi(y)}=\sum_j\lambda_j^k f_j(x)f_j(y),\qquad\text{so}\qquad h_k(y):=\frac{P^k(x,y)}{\pi(y)}-1=\sum_{j\ge2}\lambda_j^kf_j(x)f_j(y).
$$

*Step 1 (Cauchy–Schwarz).* Since $\sum_y\pi(y)=1$,

$$
2\Vert\delta_xP^k-\pi\Vert_{TV}=\sum_y\pi(y)\vert h_k(y)\vert\le\Big(\sum_y\pi(y)h_k(y)^2\Big)^{1/2}=\Vert h_k\Vert_{2,\pi}.
$$

*Step 2 (orthonormality).* With $\lambda_\ast=\max_{j\ge2}\vert\lambda_j\vert$,

$$
\Vert h_k\Vert_{2,\pi}^2=\sum_{j\ge2}\lambda_j^{2k}f_j(x)^2\le\lambda_\ast^{2k}\sum_{j\ge2}f_j(x)^2 .
$$

*Step 3 (the constant).* Take $k=0$ in the expansion: $\delta_x(y)/\pi(y)=\sum_jf_j(x)f_j(y)$. At $y=x$ this reads
$1/\pi(x)=\sum_jf_j(x)^2$. Removing $j=1$ gives $\sum_{j\ge2}f_j(x)^2=1/\pi(x)-1$.

Combining the three steps, $4\Vert\delta_xP^k-\pi\Vert_{TV}^2\le\frac{1-\pi(x)}{\pi(x)}\lambda_\ast^{2k}$. $\square$

### Exercise 4 (★★) — Unique global minimum: then $m=d^{\ast}$ always

**Numerics.** Raise one of the two tied optima, `E[24] = 0.5`. The depths become $\lbrace 4:3,\ 6:4,\ 18:1,\ 24:4.5\rbrace$,
so $d^{\ast}=4.5$ (state 24 is now a non-global minimum), and the critical height is $m=4.5$. The fitted Arrhenius
slope over $T\in\lbrace0.25,0.22,0.2,0.18\rbrace$ is $4.510$, so $m=d^{\ast}$ here. The original grid (slope 4.990)
is the example with $m=5\gt d^{\ast}=4$: its two tied global minima are separated by a barrier of height 5.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

E = np.array([[4, 3, 5, 6, 2], [3, 1, 4, 5, 3], [5, 4, 6, 4, 5], [6, 5, 3, 2, 4], [0, 4, 5, 3, 0]], float).ravel()
nbrs = [[r2 * 5 + c2 for r2, c2 in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)) if 0 <= r2 < 5 and 0 <= c2 < 5]
        for r, c in (divmod(i, 5) for i in range(25))]

def bottleneck_heights(E, nbrs):
    """H[x,y] = min over paths of the max energy on the path (minimax Floyd-Warshall)."""
    H = np.full((len(E), len(E)), np.inf)
    for i in range(len(E)):
        H[i, i] = E[i]
        for j in nbrs[i]:
            H[i, j] = max(E[i], E[j])
    for k in range(len(E)):
        H = np.minimum(H, np.maximum(H[:, k, None], H[None, k, :]))
    return H

def depths_and_m(E, nbrs):
    H = bottleneck_heights(E, nbrs)
    lm = [i for i in range(len(E)) if all(E[j] > E[i] for j in nbrs[i]) and E[i] > E.min()]
    dep = {i: float(H[i][E < E[i]].min() - E[i]) for i in lm}
    return dep, float((H - E[:, None] - E[None, :]).max() + E.min())

def spectral_gap(E, nbrs, T):
    P = np.zeros((len(E), len(E)))
    for i in range(len(E)):
        for j in nbrs[i]:
            P[i, j] = 0.25 * np.exp(min(0.0, -(E[j] - E[i]) / T))
        P[i, i] = 1 - P[i].sum()
    s = np.sqrt(np.exp(-(E - E.min()) / T)); S = s[:, None] * P / s[None, :]; S = 0.5 * (S + S.T)
    return 1 - np.sort(np.linalg.eigvalsh(S))[-2]

Tg = np.array([0.25, 0.22, 0.2, 0.18])
for name, E_ in [("original (two tied optima)", E), ("E[24] = 0.5 (unique optimum)", np.where(np.arange(25) == 24, 0.5, E))]:
    dep, m = depths_and_m(E_, nbrs)
    slope = -np.polyfit(1 / Tg, np.log([spectral_gap(E_, nbrs, t) for t in Tg]), 1)[0]
    print(f"{name}: depths {dep}, d* = {max(dep.values())}, m = {m}, Arrhenius slope = {slope:.3f}")

# m = d* on random 5x5 landscapes with (a.s.) unique global minima
rng = np.random.default_rng(0); worst = 0.0
for _ in range(300):
    E_ = rng.random(25) * 5
    dep, m = depths_and_m(E_, nbrs)
    H = bottleneck_heights(E_, nbrs); x_star = int(np.argmin(E_))
    d_all = max(H[x][E_ < E_[x]].min() - E_[x] for x in range(25) if x != x_star)   # d* over all x != x*
    worst = max(worst, abs(m - d_all), abs(m - max(dep.values(), default=0.0)))
print(f"max |m - d*| over 300 random landscapes: {worst:.1e}")
```

**Theorem.** If the global minimiser $x^{\ast}$ is unique, then $m=d^{\ast}$. So $m\gt d^{\ast}$ is impossible with a
unique global minimum.

*Definitions.* $H(x,y)$ is the minimax path height. The depth of $x\ne x^{\ast}$ is
$d(x)=\min_{z:E(z)\lt E(x)}H(x,z)-E(x)$. The set of lower states is nonempty because $E(x^{\ast})\lt E(x)$, and
$d(x)=0$ whenever $x$ has a strictly lower neighbour. Put $d^{\ast}=\max_{x\ne x^{\ast}}d(x)$; this is the largest depth
of a non-global local minimum (for energies with ties, "local minimum" must include non-strict ones, i.e. plateaus).
Two facts about $H$ are used: $H(x,y)\ge\max(E(x),E(y))$, and the ultrametric inequality
$H(x,y)\le\max(H(x,z),H(z,y))$ (concatenate paths).

*$m\ge d^{\ast}$ (this direction holds even with ties).* Take $y=x^{\ast}$ in the definition of $m$. For $x\ne x^{\ast}$,

$$
H(x,x^{\ast})-E(x)-E^{\ast}+E^{\ast}\ge\min_{z:E(z)\lt E(x)}H(x,z)-E(x)=d(x).
$$

*$m\le d^{\ast}$.* First, $H(x,x^{\ast})\le E(x)+d^{\ast}$ for every $x$. We prove this by induction along increasing
energy. The case $x=x^{\ast}$ is trivial. Otherwise choose $z$ with $E(z)\lt E(x)$ and $H(x,z)=E(x)+d(x)\le E(x)+d^{\ast}$.
By induction $H(z,x^{\ast})\le E(z)+d^{\ast}\lt E(x)+d^{\ast}$, and the ultrametric inequality gives the claim. Then for
any $x,y$,

$$
H(x,y)\le\max\big(H(x,x^{\ast}),H(x^{\ast},y)\big)\le\max(E(x),E(y))+d^{\ast},
$$

so $H(x,y)-E(x)-E(y)+E^{\ast}\le d^{\ast}-\big(\min(E(x),E(y))-E^{\ast}\big)\le d^{\ast}$. $\square$

The block also checks $m=d^{\ast}$ on 300 random $5\times5$ landscapes (continuous energies, so the minimum is a.s.
unique); the largest discrepancy is $8.9\times10^{-16}$.

### Exercise 5 (★★) — Insertion neighbourhood for the QAP

*Symmetry.* An insertion move $(i\to j)$ removes the element at position $i$ and re-inserts it so that it ends at
position $j$. The move $(j\to i)$ undoes it. The map $(i,j)\mapsto(j,i)$ is therefore a bijection between the moves
taking $x$ to $y$ and the moves taking $y$ to $x$. With $(i,j)$ uniform over ordered pairs $i\ne j$, it follows that
$q(x,y)=q(y,x)$, even though adjacent insertions coincide ($(i,i+1)$ and $(i+1,i)$ both swap two neighbours). An
exhaustive count for $n=5$ confirms this: $16=(n-1)^2$ distinct neighbours, and $\max\vert\#(x\to y)-\#(y\to x)\vert=0$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from itertools import permutations
from collections import Counter
from scipy.stats import binomtest
from utils import BudgetedObjective, BudgetExhausted, random_qap, qap_cost, qap_brute_force

def swap_neighbour(x, rng):
    i, j = rng.choice(len(x), 2, replace=False); y = x.copy(); y[i], y[j] = y[j], y[i]; return y

def insert_move(x, i, j):
    y = list(np.delete(x, i)); y.insert(j, x[i]); return np.array(y)

def insertion_neighbour(x, rng):
    i, j = rng.choice(len(x), 2, replace=False); return insert_move(x, i, j)

# symmetry check for n = 5: #moves x->y equals #moves y->x for every pair
n = 5; cnt = Counter()
for p in permutations(range(n)):
    for i in range(n):
        for j in range(n):
            if i != j:
                cnt[p, tuple(insert_move(np.array(p), i, j))] += 1
x0 = tuple(range(n))
print("distinct insertion neighbours of a permutation (n=5):", sum(1 for (a, b) in cnt if a == x0 and b != x0),
      "; max |#(x->y) - #(y->x)| =", max(abs(c - cnt.get((b, a), 0)) for (a, b), c in cnt.items()))

def simulated_annealing(x0, energy, propose, schedule, budget, rng):
    obj = BudgetedObjective(energy, budget); x, Ex = x0, obj(x0); k = 0
    try:
        while True:
            y = propose(x, rng); Ey = obj(y); d = Ey - Ex
            if d <= 0 or rng.random() < np.exp(-d / schedule(k)):
                x, Ex = y, Ey
            k += 1
    except BudgetExhausted:
        pass
    return obj.best_f

def sa_qap(F, Dm, budget, rng, propose, n_probe=50):
    """As in the notebook: T0 from n_probe probe moves, geometric cooling to T0/200, probes charged to the budget."""
    energy = lambda p: qap_cost(p, F, Dm)
    x = rng.permutation(len(F)); Ex = energy(x)
    T0 = np.mean([abs(energy(propose(x, rng)) - Ex) for _ in range(n_probe)])
    rest = budget - n_probe - 1; alpha = (1 / 200) ** (1 / rest)
    return min(simulated_annealing(x, energy, propose, lambda k: T0 * alpha**k, rest, rng), Ex)

inst_rng = np.random.default_rng(2024); instances = []
for _ in range(5):
    F, Dm = random_qap(8, inst_rng); instances.append((F, Dm, qap_brute_force(F, Dm)[0]))
succ = {}
for name, prop in [("swap", swap_neighbour), ("insertion", insertion_neighbour)]:
    succ[name] = np.array([sa_qap(F, Dm, 1500, np.random.default_rng(1000 * a + s), prop) - opt < 1e-9
                           for a, (F, Dm, opt) in enumerate(instances) for s in range(12)])
    print(f"{name:9s}: exact optimum in {succ[name].mean():.1%} of 60 runs")
n10 = int(np.sum(succ["swap"] & ~succ["insertion"])); n01 = int(np.sum(~succ["swap"] & succ["insertion"]))
print(f"swap-only {n10}, insertion-only {n01}, exact McNemar p = {binomtest(n10, n10 + n01, 0.5).pvalue:.2g}")
```

On the notebook's 5 instances $\times$ 12 seeds at 1500 evaluations, swap reaches the exact optimum in **48.3%** of
runs (this reproduces the notebook) and insertion in **13.3%**. Paired by (instance, seed), 25 runs succeed only with
swap and 4 only with insertion; the exact McNemar test gives $p=1.0\times10^{-4}$.

*Why.* The QAP cost depends on the assignment facility $\to$ location. A swap changes exactly two assignments, so
$\Delta$ is local and small. An insertion from $i$ to $j$ shifts every entry between $i$ and $j$ by one position,
which changes up to $\vert i-j\vert+1$ assignments. The resulting landscape is much less correlated: neighbours of a
good permutation are rarely good. The rule of thumb is to choose the neighbourhood whose moves change the objective
the least, not the one with the most neighbours.

### Exercise 6 (★★★) — Exact law of SA, the lag behind quasi-equilibrium, and the fastest safe schedule

The law of the inhomogeneous chain is propagated exactly, $\mu_{k+1}=\mu_kP_{T_k}$, with an $O(\vert\text{edges}\vert)$
update instead of dense matrices (checked against a dense $P_T$: difference $10^{-17}$). Many geometric schedules
$T_0=5\to T_{\text{end}}$ in $K$ steps are propagated at once. The start is the deepest non-global minimum (state 6)
and the lazy symmetric proposal of Section 3 is used. The criterion $\mu_K(S^{\ast})$ counts the *final state*.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt

E = np.array([[4, 3, 5, 6, 2], [3, 1, 4, 5, 3], [5, 4, 6, 4, 5], [6, 5, 3, 2, 4], [0, 4, 5, 3, 0]], float).ravel()
nbrs = [[r2 * 5 + c2 for r2, c2 in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)) if 0 <= r2 < 5 and 0 <= c2 < 5]
        for r, c in (divmod(i, 5) for i in range(25))]
S_star, start, T0 = [20, 24], 6, 5.0                      # global minima; deepest non-global minimum (depth 4)
boltzmann = lambda T: np.exp(-(E - E.min()) / T) / np.exp(-(E - E.min()) / T).sum()
src, dst = map(np.array, zip(*[(i, j) for i in range(25) for j in nbrs[i]]))
up = np.maximum(E[dst] - E[src], 0.0)
M_in = np.zeros((len(src), 25)); M_in[np.arange(len(src)), dst] = 1       # edge -> target state
M_out = np.zeros((len(src), 25)); M_out[np.arange(len(src)), src] = 1     # edge -> source state

def step(mu, T):
    """mu P_T for a batch of laws (rows) and temperatures; lazy symmetric proposal 1/4, Metropolis acceptance."""
    flow = mu[:, src] * 0.25 * np.exp(-up[None, :] / T[:, None])
    return mu + flow @ M_in - flow @ M_out

# check the edge update against a dense transition matrix
P = np.zeros((25, 25))
for i in range(25):
    for j in nbrs[i]:
        P[i, j] = 0.25 * np.exp(-max(E[j] - E[i], 0) / 1.3)
    P[i, i] = 1 - P[i].sum()
mu0 = np.random.default_rng(0).dirichlet(np.ones(25))[None]
print("edge update vs dense matrix, max diff:", np.abs(step(mu0, np.array([1.3])) - mu0 @ P).max())

def run(Ks, Tends, track=None):
    """Exact laws of SA for geometric schedules T0 -> Tend in K steps, all (K, Tend) pairs at once.
    Returns final mu_K(S*) per pair; for pair index `track`, also (T_k, lag_k, mu_k(S*)) along the run."""
    Ks, Tends = np.asarray(Ks), np.asarray(Tends, float)
    a = (Tends / T0) ** (1 / Ks); mu = np.tile(np.eye(25)[start], (len(Ks), 1)); out = np.zeros(len(Ks)); log = []
    for k in range(Ks.max()):
        T = T0 * a ** np.minimum(k, Ks - 1)
        if track is not None and k % max(1, Ks[track] // 400) == 0:
            log.append((T[track], 0.5 * np.abs(mu[track] - boltzmann(T[track])).sum(), mu[track, S_star].sum()))
        mu = step(mu, T)
        out[Ks == k + 1] = mu[Ks == k + 1][:, S_star].sum(1)
    return out, np.array(log)

# (1) T_end = 0.01 and increasing K; lag along the K = 3200 run
Ks = [100, 400, 1600, 3200]
res, lag = run(Ks, [0.01] * 4, track=3)
print("T_end = 0.01: mu_K(S*) =", {K: round(float(v), 2) for K, v in zip(Ks, res)})
print(f"K = 3200: lag after 10% of the run {lag[40, 1]:.2f}, max lag after that {lag[40:, 1].max():.2f}, at the end {lag[-1, 1]:.2f}")

# (2) grid over K and T_end (one batched run)
Kg, Tg = [10**3, 10**4, 10**5, 5 * 10**5, 7 * 10**5], [0.05, 0.1, 0.2, 0.25, 0.3, 0.4]
pairs = [(K, T) for K in Kg for T in Tg] + [(K, 0.25) for K in (6.8e5, 6.9e5, 7.1e5, 7.2e5)]
pairs = [(int(K), T) for K, T in pairs]
tr = pairs.index((7 * 10**5, 0.25))
res, log = run([p[0] for p in pairs], [p[1] for p in pairs], track=tr)
tab = dict(zip(pairs, res))
print("K \\ T_end " + " ".join(f"{t:>7}" for t in Tg))
for K in Kg:
    print(f"{K:9.0e} " + " ".join(f"{tab[K, t]:7.4f}" for t in Tg))
print("T_end = 0.25:", {K: round(float(tab[K, 0.25]), 4) for K in sorted({p[0] for p in pairs if p[1] == 0.25})})
Kmin = min(K for (K, t), v in tab.items() if v >= 0.95)
print(f"smallest K on the grid with mu_K(S*) >= 0.95: {Kmin:.2e}")
b = np.polyfit(np.log(Kg), np.log([1 - max(tab[K, t] for t in Tg) for K in Kg]), 1)[0]
print(f"1 - best mu_K(S*) ~ K^{b:.2f}")
for T_q in (4.0, 3.2, 2.0, 1.0, 0.5, 0.35, 0.25):
    r = log[np.argmin(np.abs(log[:, 0] - T_q))]
    print(f"K = 7e5, T_end = 0.25: at T = {r[0]:.2f} lag = {r[1]:.1e}, mu_k(S*) = {r[2]:.3f}, pi_T(S*) = {boltzmann(r[0])[S_star].sum():.3f}")
print(f"final mu_K(S*) = {tab[7 * 10**5, 0.25]:.3f} vs pi_0.25(S*) = {boltzmann(0.25)[S_star].sum():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].semilogx(log[:, 0], log[:, 1]); ax[0].invert_xaxis(); ax[0].set_xlabel("temperature T_k (time runs to the right)")
ax[0].set_ylabel(r"lag $\Vert\mu_k-\pi_{T_k}\Vert_{TV}$"); ax[0].set_title("K = 7e5, T_end = 0.25")
ax[1].semilogx(log[:, 0], log[:, 2], label=r"$\mu_k(S^\ast)$")
ax[1].semilogx(log[:, 0], [boltzmann(t)[S_star].sum() for t in log[:, 0]], ":", label=r"$\pi_{T_k}(S^\ast)$")
ax[1].invert_xaxis(); ax[1].set_xlabel("temperature T_k"); ax[1].set_ylabel("probability"); ax[1].legend()
plt.tight_layout(); plt.show()
```

**Results.** With $T_{\text{end}}=0.01$, the final mass on $S^{\ast}$ is 0.25, 0.54, 0.69 and 0.74 for $K=100$, 400,
1600 and 3200. Along the $K=3200$ run the lag $\Vert\mu_k-\pi_{T_k}\Vert_{TV}$ is 0.01 after the first 10% of the run
(the chain has forgotten its start while hot), then grows as the chain freezes and ends at 0.26: the missing mass is
trapped in state 6 (energy 1, depth 4).

To find the fastest schedule we let both $K$ and $T_{\text{end}}$ vary:

| $K$ | $T_{\text{end}}=0.05$ | 0.1 | 0.2 | 0.25 | 0.3 | 0.4 |
|---|---|---|---|---|---|---|
| $10^3$ | 0.6716 | 0.6857 | 0.7019 | 0.7071 | 0.7110 | 0.7143 |
| $10^4$ | 0.8296 | 0.8375 | 0.8463 | 0.8492 | 0.8511 | 0.8509 |
| $10^5$ | 0.9098 | 0.9136 | 0.9178 | 0.9191 | 0.9197 | 0.9151 |
| $5\times10^5$ | 0.9400 | 0.9423 | 0.9450 | **0.9457** | 0.9456 | 0.9369 |
| $7\times10^5$ | 0.9447 | 0.9468 | 0.9493 | **0.9499** | 0.9497 | 0.9399 |

At $T_{\text{end}}=0.25$ a finer scan gives 0.9496, 0.9498, 0.9499, **0.9501**, 0.9503 for
$K=6.8,6.9,7.0,7.1,7.2\times10^5$. The fastest geometric schedule from $T_0=5$ with $\mu_K(S^{\ast})\ge0.95$ therefore
has $K\approx7.1\times10^5$ steps and $T_{\text{end}}\approx0.25$. (The best-so-far criterion of Section 6 of the
notebook is far easier to meet: it only needs one *visit* to $S^{\ast}$.) Two lessons follow.

1. Cooling to a very low $T_{\text{end}}$ is useless. Below $T\approx0.25$ the chain is frozen, while stopping at
   $T=0.25$ still gives $\pi_T(S^{\ast})=0.991$.
2. The required length grows like a power of the target failure probability: $1-\max_{T_{\text{end}}}\mu_K(S^{\ast})\approx
   K^{-0.27}$ between $K=10^3$ and $7\times10^5$. This is the Arrhenius cost $e^{d^{\ast}/T}$ of escaping the depth-4
   trap near the freezing temperature, the finite-time face of Hajek's theorem. Along the $K=7\times10^5$ run the chain
   is in equilibrium while it is hot (lag $6\times10^{-6}$ at $T=4$, $1.0\times10^{-5}$ at $T=3.2$,
   $3.5\times10^{-4}$ at $T=1$) and lags only at the end: $8.8\times10^{-3}$ at $T=0.5$, $0.025$ at $T=0.35$ and
   $0.041$ at $T=0.25$, where $\mu_K(S^{\ast})=0.950$ against $\pi_T(S^{\ast})=0.991$.

---

## Notebook 2 — Cooling Schedules, Convergence Theory and Alternative Acceptance Rules

### Exercise 1 (★) — Acceptance $(k+1)^{-h/c}$ and the Borel–Cantelli heuristic

With $T_k=c/\log(k+1)$, an uphill move of height $h$ is accepted with probability
$e^{-h/T_k}=e^{-h\log(k+1)/c}=(k+1)^{-h/c}$.

To leave a trap of depth $d$, the chain must climb at least $d$ above the trap's energy along some path of bounded
length $\ell$. While the path is traversed, $T$ is essentially constant ($T_{k+\ell}/T_k\to1$). The probability that
an escape starts at step $k$ is therefore of order $C\,(k+1)^{-d/c}$ with $C$ independent of $k$. The expected number
of escapes is $\sum_k C(k+1)^{-d/c}$, which is infinite iff $d/c\le1$, i.e. iff $c\ge d$.

- If $c\lt d^{\ast}$, the series converges for the deepest trap. By the first Borel–Cantelli lemma only finitely many
  escapes occur a.s. A chain started in the trap therefore stays there forever after some time with positive
  probability, and $\Pr(X_k\in S^{\ast})\not\to1$.
- If $c\ge d^{\ast}$, the series diverges for every non-global trap. Escapes from every trap happen infinitely often
  (the second Borel–Cantelli lemma would need independence; Hajek's proof handles the dependence). Leaving $S^{\ast}$
  costs an ever larger climb, so those excursions become ever rarer relative to the time spent in $S^{\ast}$.

The argument is heuristic because of the constant $C$ and the dependence between attempts, which is exactly what
Hajek (1988) makes rigorous.

### Exercise 2 (★) — The empirical acceptance equation has a unique root

Let $g(T)=\frac1M\sum_me^{-\Delta_m/T}$ with all $\Delta_m\gt0$. Then $g$ is continuous on $(0,\infty)$ and strictly
increasing, because

$$
g'(T)=\frac1M\sum_m\frac{\Delta_m}{T^2}e^{-\Delta_m/T}\gt0 .
$$

Moreover $g(T)\to0$ as $T\to0^+$ and $g(T)\to1$ as $T\to\infty$. For $\chi_0\in(0,1)$ the intermediate value theorem
gives a root, and strict monotonicity makes it unique. If all $\Delta_m=\Delta$, then $g(T)=e^{-\Delta/T}=\chi_0$ gives
$T=-\Delta/\ln\chi_0=-\overline{\Delta^+}/\ln\chi_0=T_0^{\text{K}}$, so the two estimators coincide. The root depends
continuously on the $\Delta_m$ (implicit function theorem, $g'\gt0$), so the estimators stay close when the deltas are
nearly equal; Jensen's gap, and with it $T_0^{\text{K}}-T_0^{\text{exact}}\ge0$, grows with the spread of the
$\Delta^+_m$. $\square$

### Exercise 3 (★★) — Two traps of depths 1 and 2

The landscape `E3 = [2.5, 1.0, 2.0, 0.5, 1.5, 2.5, 1.0, 0.0, 1.0, 2.0]` has a shallow trap A (state 1, depth 1), a
deep trap B (state 3, depth 2) and the global minimum at state 7; `depths()` returns `{1: 1.0, 3: 2.0}`, so
$d^{\ast}=2$. We start in the **shallow** trap and propagate the exact law for $K=10^6$ steps, recording the first step
at which $\Pr(X_k\in S^{\ast})\ge0.5$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt

E3 = np.array([2.5, 1.0, 2.0, 0.5, 1.5, 2.5, 1.0, 0.0, 1.0, 2.0])
nb3 = [[j for j in (i - 1, i + 1) if 0 <= j < len(E3)] for i in range(len(E3))]

def depths(E, nbrs):
    H = np.full((len(E), len(E)), np.inf)
    for i in range(len(E)):
        H[i, i] = E[i]
        for j in nbrs[i]:
            H[i, j] = max(E[i], E[j])
    for k in range(len(E)):
        H = np.minimum(H, np.maximum(H[:, k, None], H[None, k, :]))
    return {i: float(H[i][E < E[i]].min() - E[i]) for i in range(len(E))
            if all(E[j] > E[i] for j in nbrs[i]) and E[i] > E.min()}

dep = depths(E3, nb3); A, B, glob = 1, 3, int(np.argmin(E3))
print("depths:", dep, " d* =", max(dep.values()))

def exact_sa_law(E, c_values, K, start, glob):
    """Exact law of SA on a line (left/right proposal 1/2 each), T_k = c/log(k+1), several c at once.
    Returns the final law and the first k with P(X_k = glob) >= 0.5 (np.inf if never)."""
    dR = np.maximum(np.append(E[1:] - E[:-1], np.inf), 0.0); dL = np.maximum(np.insert(E[:-1] - E[1:], 0, np.inf), 0.0)
    c = np.asarray(c_values, float)[:, None]; p = np.zeros((len(c), len(E))); p[:, start] = 1.0
    first = np.full(len(c), np.inf)
    for k in range(1, K + 1):
        inv_T = np.log(k + 1) / c
        fR = p * 0.5 * np.exp(-inv_T * dR); fL = p * 0.5 * np.exp(-inv_T * dL)
        p = p - fR - fL; p[:, 1:] += fR[:, :-1]; p[:, :-1] += fL[:, 1:]
        first[(p[:, glob] >= 0.5) & np.isinf(first)] = k
    return p, first

cs = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]); K = 10**6
p, first = exact_sa_law(E3, cs, K, start=A, glob=glob)
print("c              ", cs)
print("P(S*) at K     ", np.round(p[:, glob], 3))
print("mass in A      ", np.round(p[:, A], 3))
print("mass in B      ", np.round(p[:, B], 3))
print("first k >= 0.5 ", first)
plt.loglog(cs[np.isfinite(first)], first[np.isfinite(first)], "o-"); plt.axvline(2.0, ls=":", c="grey", label="d* = 2")
plt.xlabel("c"); plt.ylabel(r"first k with $\Pr(X_k\in S^\ast)\geq 0.5$"); plt.legend(); plt.show()
```

| $c$ | 0.5 | 0.8 | 1.0 | 1.2 | 1.5 | 1.8 | 2.0 | 2.5 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| $\Pr(S^{\ast})$ at $K$ | 0.000 | 0.003 | 0.015 | 0.048 | 0.205 | 0.588 | 0.847 | 0.926 | 0.883 | 0.781 |
| mass in A | 0.855 | 0.446 | 0.049 | 0.004 | 0.008 | 0.009 | 0.005 | 0.004 | 0.009 | 0.025 |
| mass in B | 0.145 | 0.551 | 0.937 | 0.948 | 0.787 | 0.402 | 0.146 | 0.062 | 0.089 | 0.139 |
| first $k$ with $\Pr(S^{\ast})\ge0.5$ | – | – | – | – | – | 40 965 | 2 082 | 516 | 476 | 1 720 |

Escaping the shallow trap needs only $c\gtrsim1$: the mass in A collapses between $c=0.8$ and $c=1.2$. The mass then
piles up in the deep trap B, and $\Pr(S^{\ast})$ becomes large only from $c\approx d^{\ast}=2$. The threshold is set by
the deeper trap. The time to reach $\Pr(S^{\ast})\ge0.5$ (plotted by the block) is U-shaped in $c$. It explodes as $c$
decreases towards $d^{\ast}$; $c=1.8$ still reaches 0.5 at a finite time, but by Hajek's theorem $\Pr(S^{\ast})$ will
not tend to 1. It grows again for large $c$, because the chain stays hot. The fastest values here are
$c\approx2.5$–3, i.e. $1.25$–$1.5\,d^{\ast}$.

### Exercise 4 (★★) — Aarts & van Laarhoven's adaptive decrement on the SK benchmark

The rule runs homogeneous stages of $L=n=100$ proposals. At the end of each stage it measures the standard deviation
$\sigma_j$ of the current energy over the stage and sets $T_{j+1}=T_j/(1+T_j\ln(1+\delta)/(3\sigma_j))$. The distance
parameter $\delta$ is tuned on the notebook's pilot instance with the same protocol as the other rules, then the rule is
run on the test instance from the same 24 starts as geometric and logarithmic SA.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.optimize import brentq

def sk_instance(n, rng):
    J = np.triu(rng.standard_normal((n, n)), 1); return (J + J.T) / np.sqrt(n)

def sample_uphill(J, M, rng):
    S = rng.choice([-1.0, 1.0], (M, len(J))); i = rng.integers(0, len(J), M)
    d = 2 * S[np.arange(M), i] * (S @ J)[np.arange(M), i]; return d[d > 0]

def T0_of(J, seed, chi0=0.8):
    d = sample_uphill(J, 4000, np.random.default_rng(seed)); return brentq(lambda T: np.mean(np.exp(-d / T)) - chi0, 1e-6, 1e6)

def run_chains(J, S0, n_steps, rule, rng):
    """R single-flip chains in parallel (the notebook's engine); returns the best energy per chain."""
    S = S0.astype(float).copy(); R, n = S.shape; Hf = S @ J
    E_cur = -0.5 * np.sum(S * Hf, axis=1); E_best = E_cur.copy(); rows = np.arange(R)
    for k0 in range(0, n_steps, 4096):
        m = min(4096, n_steps - k0); I = rng.integers(0, n, (m, R)); U = rng.random((m, R))
        for t in range(m):
            i = I[t]; s_i = S[rows, i]; E_new = E_cur + 2.0 * s_i * Hf[rows, i]
            idx = np.flatnonzero(rule(k0 + t, E_cur, E_new, E_best, U[t]))
            if idx.size:
                Hf[idx] -= 2.0 * s_i[idx, None] * J[i[idx]]; S[idx, i[idx]] = -s_i[idx]; E_cur[idx] = E_new[idx]
                np.minimum(E_best, E_cur, out=E_best)
    return E_best

def metropolis(T_of_k, log=None):
    def rule(k, Ec, En, Eb, u):
        T = T_of_k(k)
        if log is not None: log.append(T)
        return (En <= Ec) | (u < np.exp(-np.maximum(En - Ec, 0) / T))
    return rule
from scipy.stats import wilcoxon

def make_aarts_rule(T0, delta, L, log):
    """Aarts & van Laarhoven (1985): stages of L proposals at fixed T, then T <- T / (1 + T ln(1+delta) / (3 sigma))."""
    st = {"T": None}
    def rule(k, Ec, En, Eb, u):
        if st["T"] is None:
            st["T"] = np.full(Ec.shape, T0); st["s1"] = np.zeros(Ec.shape); st["s2"] = np.zeros(Ec.shape)
        T = st["T"]
        acc = (En <= Ec) | (u < np.exp(-np.maximum(En - Ec, 0) / T))
        Enew = np.where(acc, En, Ec); st["s1"] += Enew; st["s2"] += Enew**2
        if (k + 1) % L == 0:                              # end of a stage: energy std over the stage
            sig = np.sqrt(np.maximum(st["s2"] / L - (st["s1"] / L) ** 2, 1e-12))
            st["T"] = T / (1 + T * np.log(1 + delta) / (3 * sig)); st["s1"][:] = 0; st["s2"][:] = 0
        log.append(st["T"][0]); return acc
    return rule

n = 100; K = 300 * n
# pilot instance of the notebook (seed 7, starts seed 8, T0 probe seed 9, run seed 10): tune delta
pJ = sk_instance(n, np.random.default_rng(7)); pS0 = np.random.default_rng(8).choice([-1.0, 1.0], (8, n)); pT0 = T0_of(pJ, 9)
scores = {}
for delta in (0.03, 0.1, 0.3, 1.0):
    log = []; Eb = run_chains(pJ, pS0, K, make_aarts_rule(pT0, delta, n, log), np.random.default_rng(10))
    scores[delta] = np.median(Eb) / n
    print(f"pilot delta = {delta}: median E/n = {scores[delta]:.4f}, final T = {log[-1]:.2g}")
delta = min(scores, key=scores.get)
# test instance of the notebook (seed 42, T0 probe seed 3), 24 common starts (seed 123), run seed 999
J = sk_instance(n, np.random.default_rng(42)); T0 = T0_of(J, 3); S0 = np.random.default_rng(123).choice([-1.0, 1.0], (24, n))
logs = {"Aarts-van Laarhoven": [], "geometric": [], "logarithmic": []}
a = (1 / 2000) ** (1 / K)
rules = {"Aarts-van Laarhoven": make_aarts_rule(T0, delta, n, logs["Aarts-van Laarhoven"]),
         "geometric": metropolis(lambda k: T0 * a**k, logs["geometric"]),
         "logarithmic": metropolis(lambda k: T0 * np.log(2) / np.log(k + 2), logs["logarithmic"])}
res = {nm: run_chains(J, S0, K, r, np.random.default_rng(999)) for nm, r in rules.items()}
E_ref = min(v.min() for v in res.values())      # = the notebook's best known (-0.7240 n), found by geometric/log SA
print(f"best known E/n = {E_ref / n:.4f}")
for nm, Eb in res.items():
    g = 100 * (Eb - E_ref) / abs(E_ref); Tl = np.array(logs[nm])
    p = "" if nm == "Aarts-van Laarhoven" else f", Wilcoxon p vs Aarts = {wilcoxon(Eb, res['Aarts-van Laarhoven']).pvalue:.2g}"
    print(f"{nm:20s} median gap {np.median(g):.2f}% [IQR {np.percentile(g, 25):.2f}, {np.percentile(g, 75):.2f}], "
          f"budget at 0.3<=T<=1: {np.mean((Tl >= 0.3) & (Tl <= 1)):.0%}, final T {Tl[-1]:.2g}{p}")
```

| $\delta$ | 0.03 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|
| pilot median $E/n$ | −0.6249 | **−0.7148** | −0.7113 | −0.7060 |
| final $T$ | 1.0 | $2.5\times10^{-7}$ | $4.9\times10^{-8}$ | $4.1\times10^{-8}$ |

With $\delta=0.1$ on the test instance (24 paired runs), the median gap to the best known is **1.41%** (IQR 0.45–2.64).
Geometric SA has 1.11% (Wilcoxon $p=0.86$) and the logarithmic schedule 0.03% ($p=0.002$). The rule spends 21% of the
budget in the window $0.3\le T\le1$, about as much as the geometric schedule (16%). This is consistent with the
notebook's diagnosis that time spent in the productive window drives quality at this budget. The rule adapts to
$\sigma_j$, but $\delta$ still fixes how many stages are spent per unit of $\ln T$. It is $\delta$, not the
adaptivity, that decides where the budget goes: $\delta=0.03$ never cools below $T\approx1$.

### Exercise 5 (★★) — Ten test instances and a Friedman test

The parameters tuned on the notebook's pilot instance are kept ($T_{\text{end}}=T_0/2000$, $\tau_0=0.5T_0$, GD target
$-0.74n$, RRT $D=1$, LAHC $L=100$). Ten new SK instances (seeds 1000–1009) are drawn, $T_0$ is recomputed on each, and
every rule makes 24 runs from common starts. The score of a rule on an instance is its median $E/n$. To keep the run
time down, the ten rules run side by side in one vectorised engine call per instance (rule $j$ owns chains
$24j,\dots,24j+23$), which does not change what any rule does.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.optimize import brentq

def sk_instance(n, rng):
    J = np.triu(rng.standard_normal((n, n)), 1); return (J + J.T) / np.sqrt(n)

def sample_uphill(J, M, rng):
    S = rng.choice([-1.0, 1.0], (M, len(J))); i = rng.integers(0, len(J), M)
    d = 2 * S[np.arange(M), i] * (S @ J)[np.arange(M), i]; return d[d > 0]

def T0_of(J, seed, chi0=0.8):
    d = sample_uphill(J, 4000, np.random.default_rng(seed)); return brentq(lambda T: np.mean(np.exp(-d / T)) - chi0, 1e-6, 1e6)

def run_chains(J, S0, n_steps, rule, rng):
    """R single-flip chains in parallel (the notebook's engine); returns the best energy per chain."""
    S = S0.astype(float).copy(); R, n = S.shape; Hf = S @ J
    E_cur = -0.5 * np.sum(S * Hf, axis=1); E_best = E_cur.copy(); rows = np.arange(R)
    for k0 in range(0, n_steps, 4096):
        m = min(4096, n_steps - k0); I = rng.integers(0, n, (m, R)); U = rng.random((m, R))
        for t in range(m):
            i = I[t]; s_i = S[rows, i]; E_new = E_cur + 2.0 * s_i * Hf[rows, i]
            idx = np.flatnonzero(rule(k0 + t, E_cur, E_new, E_best, U[t]))
            if idx.size:
                Hf[idx] -= 2.0 * s_i[idx, None] * J[i[idx]]; S[idx, i[idx]] = -s_i[idx]; E_cur[idx] = E_new[idx]
                np.minimum(E_best, E_cur, out=E_best)
    return E_best

def metropolis(T_of_k, log=None):
    def rule(k, Ec, En, Eb, u):
        T = T_of_k(k)
        if log is not None: log.append(T)
        return (En <= Ec) | (u < np.exp(-np.maximum(En - Ec, 0) / T))
    return rule
from scipy.stats import friedmanchisquare, rankdata

def make_rule(name, K, T0, T_end, **p):
    """The notebook's acceptance rules (state kept in closures)."""
    if name in ("SA geometric", "SA linear", "SA logarithmic"):
        a = (T_end / T0) ** (1 / K)
        T = {"SA geometric": lambda k: T0 * a**k, "SA linear": lambda k: T0 * (1 - k / K) + T_end * k / K,
             "SA logarithmic": lambda k: T0 * np.log(2) / np.log(k + 2)}[name]
        return metropolis(T)
    st = {}
    if name == "SA adaptive (mod. Lam)":
        tgt = lambda t: 0.44 + 0.56 * 560 ** (-t / 0.15) if t < 0.15 else (0.44 if t < 0.65 else 0.44 * 440 ** (-(t - 0.65) / 0.35))
        def rule(k, Ec, En, Eb, u):
            if not st: st["T"], st["acc"] = np.full(Ec.shape, T0), np.full(Ec.shape, 0.5)
            acc = (En <= Ec) | (u < np.exp(-np.maximum(En - Ec, 0) / st["T"]))
            st["acc"] = (499 * st["acc"] + acc) / 500
            st["T"] = np.where(st["acc"] > tgt(k / K), st["T"] * 0.999, st["T"] / 0.999); return acc
    elif name == "SA geo + reheating":
        a = (T_end / T0) ** (4 / K)
        def rule(k, Ec, En, Eb, u):
            if not st: st["T"] = np.full(Ec.shape, T0)
            acc = (En <= Ec) | (u < np.exp(-np.maximum(En - Ec, 0) / st["T"]))
            st["T"] *= a; st["T"][st["T"] < T_end] = T0 / 5; return acc
    elif name == "Threshold accepting":
        rule = lambda k, Ec, En, Eb, u: (En - Ec) < p["tau0"] * (1 - k / K) + 1e-12
    elif name == "Great deluge":
        def rule(k, Ec, En, Eb, u):
            if not st: st["B0"] = Ec.copy()
            return (En <= st["B0"] + (p["target"] - st["B0"]) * (k / K)) | (En <= Ec)
    elif name == "Record-to-record":
        rule = lambda k, Ec, En, Eb, u: (En < Eb + p["D"]) | (En <= Ec)
    elif name == "Late acceptance HC":
        def rule(k, Ec, En, Eb, u):
            if not st: st["h"] = np.tile(Ec, (p["L"], 1))
            v = k % p["L"]; acc = (En <= Ec) | (En <= st["h"][v]); st["h"][v] = np.where(acc, En, Ec); return acc
    elif name == "Hill climbing":
        rule = lambda k, Ec, En, Eb, u: En <= Ec
    return rule

def composite(rules, R):
    """Run several rules side by side, rule j on chains j*R .. (j+1)*R-1 (one engine call for all rules)."""
    sl = [slice(j * R, (j + 1) * R) for j in range(len(rules))]
    return lambda k, Ec, En, Eb, u: np.concatenate([r(k, Ec[s], En[s], Eb[s], u[s]) for r, s in zip(rules, sl)])

n, R = 100, 24; K = 300 * n
names = ["SA geometric", "SA linear", "SA logarithmic", "SA adaptive (mod. Lam)", "SA geo + reheating",
         "Threshold accepting", "Great deluge", "Record-to-record", "Late acceptance HC", "Hill climbing"]
med = np.zeros((10, len(names)))
for t in range(10):
    J = sk_instance(n, np.random.default_rng(1000 + t)); T0 = T0_of(J, 2000 + t)
    cfg = {"Threshold accepting": {"tau0": 0.5 * T0}, "Great deluge": {"target": -0.74 * n},     # parameters tuned on the
           "Record-to-record": {"D": 1.0}, "Late acceptance HC": {"L": 100}}                     # notebook's pilot instance
    S0 = np.random.default_rng(3000 + t).choice([-1.0, 1.0], (R, n))                           # common starts for all rules
    rule = composite([make_rule(nm, K, T0, T0 / 2000, **cfg.get(nm, {})) for nm in names], R)
    Eb = run_chains(J, np.tile(S0, (len(names), 1)), K, rule, np.random.default_rng(4000 + t)).reshape(len(names), R)
    med[t] = np.median(Eb, axis=1) / n
ranks = np.array([rankdata(r) for r in med]); mr = ranks.mean(0)
stat = friedmanchisquare(*med.T)
CD = 3.164 * np.sqrt(len(names) * (len(names) + 1) / (6 * 10))      # Nemenyi, q_0.05 for k = 10 (Demsar 2006)
print(f"Friedman chi2 = {stat.statistic:.1f}, p = {stat.pvalue:.2g}; Nemenyi critical difference = {CD:.2f}")
order = np.argsort(mr)
for j in order:
    better = [names[i] for i in order if mr[i] - mr[j] > CD]
    print(f"{names[j]:24s} mean rank {mr[j]:4.1f}   significantly better than: {', '.join(better) or '-'}")
```

| rule | log | geo + reheat | geometric | TA | RRT | LAHC | mod. Lam | linear | GD | HC |
|---|---|---|---|---|---|---|---|---|---|---|
| mean rank (1 = best) | **1.4** | 2.6 | 3.5 | 3.6 | 5.8 | 5.8 | 6.4 | 7.6 | 8.1 | 10.0 |

The Friedman statistic is $\chi^2_F=71.1$ with 9 d.f., $p=9.3\times10^{-12}$: the rules are not equivalent. The Nemenyi
critical difference (Demšar 2006) for $k=10$ rules, $N=10$ instances and $\alpha=0.05$ is
$q_{0.05}\sqrt{k(k+1)/(6N)}=3.164\times1.354=4.28$; differences in mean rank larger than this survive.

- The logarithmic schedule is significantly better than RRT, LAHC, modified Lam, linear, GD and HC.
- Geometric + reheating is significantly better than linear, GD and HC.
- Geometric SA and threshold accepting are significantly better than GD and HC only.
- Among the top four (log, reheating, geometric, TA) and among the middle group (RRT, LAHC, modified Lam, linear), no
  difference survives.

The single-instance study of the notebook largely replicates: logarithmic first, HC last, GD near the bottom, and
geometric SA and TA indistinguishable. The one new finding is that great deluge is now significantly worse than
geometric SA and TA, which the single instance only hinted at.

### Exercise 6 (★★★) — Record-to-record travel and great deluge cannot cross barriers they are not allowed to see

**(a) RRT with $D\lt d^{\ast}$ started at the deepest trap $x_0$.** RRT accepts $y$ iff $E(y)\lt E_{\text{best}}+D$
or $E(y)\le E(x)$. Initially $E_{\text{best}}=E(x_0)$.

*Claim:* as long as $E_{\text{best}}=E(x_0)$, every visited state lies in the connected component $C$ of
$\lbrace z:E(z)\lt E(x_0)+D\rbrace$ that contains $x_0$. *Proof by induction:* an accepted move from $x\in C$ to a
neighbour $y$ has $E(y)\lt E(x_0)+D$ (first clause), or $E(y)\le E(x)\lt E(x_0)+D$ (second clause). Either way
$y\in C$.

Every state of $C$ has energy $\ge E(x_0)$: a state $z\in C$ with $E(z)\lt E(x_0)$ would be reachable from $x_0$ by a
path of maximal height $\lt E(x_0)+D\lt E(x_0)+d^{\ast}$, contradicting $d(x_0)=d^{\ast}$. So the record never
improves, the invariant holds forever, and since $S^{\ast}\cap C=\emptyset$ the global minimum is never reached. $\square$

**(b) Great deluge.** The level is $B_k=E(x_0)+(\text{target}-E(x_0))k/K\le E(x_0)$ for every target $\le E(x_0)$.
An accepted move has $E(y)\le B_k\le E(x_0)$ or $E(y)\le E(x)$. By induction every visited state has
$E\le E(x_0)$. $\square$

*Consequence:* GD can only move inside the component of $\lbrace E\le E(x_0)\rbrace$ that contains $x_0$. It escapes a
trap $x$ only if the barrier around $x$ is below the *starting* energy, $H(x,\text{lower})\le E(x_0)$, and the level
has not yet dropped below that barrier when GD gets there. In particular, GD started *at* a strict local minimum never
moves: every neighbour is strictly higher than $E(x_0)\ge B_k$ and than the current energy.

**(c) Simulation** on the line landscape of Section 2 (trap = state 2 with $E=0.8$, barrier 3.0, $d^{\ast}=2.2$),
200 runs of $10^4$ steps each.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

E = np.array([1.5, 2.5, 0.8, 2.0, 3.0, 1.6, 2.6, 0.0, 1.2, 2.2, 1.0])     # the line landscape of Section 2
n, trap, glob, d_star = len(E), 2, 7, 2.2

def rrt(D, steps, rng, x0=trap):
    """Record-to-record travel; True iff the global minimum is visited within `steps` proposals."""
    x, rec = x0, E[x0]
    for _ in range(steps):
        y = x + (1 if rng.random() < 0.5 else -1)
        if not 0 <= y < n:
            continue
        if E[y] < rec + D or E[y] <= E[x]:
            x = y; rec = min(rec, E[x])
            if x == glob:
                return True
    return False

def great_deluge(x0, target, steps, rng):
    """Great deluge (level lowered linearly from E(x0) to target, improving moves also accepted).
    Returns (max energy visited, final state)."""
    x, emax = x0, E[x0]
    for k in range(steps):
        y = x + (1 if rng.random() < 0.5 else -1)
        if not 0 <= y < n:
            continue
        if E[y] <= E[x0] + (target - E[x0]) * k / steps or E[y] <= E[x]:
            x = y; emax = max(emax, E[x])
    return emax, x

print(f"float check: 0.8 + 2.2 = {0.8 + 2.2!r} (> 3.0: {0.8 + 2.2 > 3.0})")
for D in (1.0, 2.0, 2.19, 2.2 - 1e-9, 2.2, 2.2 + 1e-9, 2.21, 2.3, 2.5):
    rate = np.mean([rrt(D, 10**4, np.random.default_rng(s)) for s in range(200)])
    print(f"RRT D = {D!r:22}: reaches S* in {rate:.2f} of 200 runs")
for x0 in (trap, 4, 6):
    out = [great_deluge(x0, -0.5, 10**4, np.random.default_rng(s)) for s in range(200)]
    print(f"GD from state {x0} (E = {E[x0]}): max energy visited {max(o[0] for o in out)}, "
          f"ends at the optimum in {np.mean([o[1] == glob for o in out]):.0%} of runs")
```

| $D$ | 1.0 | 2.0 | 2.19 | $2.2-10^{-9}$ | 2.2 | $2.2+10^{-9}$ | 2.21 | 2.3 | 2.5 |
|---|---|---|---|---|---|---|---|---|---|
| RRT success | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |

This confirms (a). Because the acceptance test is strict, the barrier state becomes admissible only when
$3.0\lt0.8+D$, i.e. for $D\gt d^{\ast}=2.2$ (in floating point $0.8+2.2$ evaluates to exactly 3.0, so $D=2.2$ fails as
the mathematics says). The smallest $D$ reaching 90% success is therefore "any $D\gt d^{\ast}$": just above the
threshold the walk reaches the optimum in all 200 runs.

Great deluge with target $-0.5$ was run from three starting states:

| start | max energy visited | runs ending at the optimum |
|---|---|---|
| trap (state 2, $E=0.8$) | 0.8 | 0% |
| barrier top (state 4, $E=3.0$) | 3.0 | 20% |
| state 6 ($E=2.6$) | 2.6 | 28% |

From the trap GD never moves. From higher starts it never exceeds $E(x_0)$, as proved in (b), and it ends at the
optimum only when the falling level catches it in the right basin.

---

## Notebook 3 — Continuous Simulated Annealing and Parallel Tempering

### Exercise 1 (★) — Moments of the Tsallis visiting law

By Proposition 1, $\xi=s\,z/\sqrt{w/\nu}$ with $z\sim\mathcal N(0,I_d)$ independent of $w\sim\chi^2_\nu$, where
$\nu=(3-q_v)/(q_v-1)$. For $r\gt0$,

$$
\mathbb E\Vert\xi\Vert^r=s^r\,\mathbb E\Vert z\Vert^r\;\mathbb E\big[(w/\nu)^{-r/2}\big].
$$

The first factor is finite for all $r$. For $w\sim\chi^2_\nu$, $\mathbb E[w^{-r/2}]\lt\infty$ iff $r\lt\nu$: the
density $\propto w^{\nu/2-1}e^{-w/2}$ is integrable against $w^{-r/2}$ near 0 iff $\nu/2-r/2\gt0$ (the tail at
$\infty$ is harmless). So moments of order $r$ exist iff $r\lt\nu$. The mean requires $\nu\gt1$, i.e.
$3-q_v\gt q_v-1$, i.e. $q_v\lt2$. The variance requires $\nu\gt2$, i.e. $3-q_v\gt2q_v-2$, i.e. $q_v\lt5/3$. $\square$

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats

q = 2.62; nu = (3 - q) / (q - 1)
print(f"q_v = {q}: nu = {nu:.4f}; moments of order r exist iff r < nu")
for c in (10, 100, 1e4):
    print(f"P(|xi_1| > {c:g} s) = {2 * stats.t.sf(c, nu):.2f}")
# Monte Carlo: sample moments of order 1/4 do not settle (infinite mean), those of order 0.1 do
rng = np.random.default_rng(0)
for r in (0.1, 0.25):
    m = [np.mean(np.abs(stats.t.rvs(nu, size=10**N, random_state=rng)) ** r) for N in (4, 5, 6)]
    print(f"E|xi_1/s|^{r}: sample means with 1e4, 1e5, 1e6 draws = {np.round(m, 2)}")
```

For $q_v=2.62$, $\nu=0.2346$: not even $\mathbb E\Vert\xi\Vert^{1/4}$ is finite (the block shows its sample mean
jumping from 11.6 to 94 between $10^5$ and $10^6$ draws, while the order-0.1 moment is stable at 1.51). One coordinate
satisfies $P(\vert\xi_1\vert\gt10s)=0.43$, $P(\vert\xi_1\vert\gt100s)=0.25$ and $P(\vert\xi_1\vert\gt10^4s)=0.08$. A large
fraction of proposals are therefore jumps of many box widths. After reflection these land almost uniformly in the box,
so GSA at $q_v=2.62$ behaves like a mixture of local moves and uniform restarts.

### Exercise 2 (★) — Reflection keeps the proposal symmetric, clipping does not

Take the box $[0,w]$ in one dimension (coordinates are independent for an isotropic Gaussian, so $d$ dimensions
follow). Reflection is the triangle-wave map $\rho$. The preimages of $y\in(0,w)$ are $y+2kw$ and $-y+2kw$ for
$k\in\mathbb Z$, so the density of $\rho(x+\sigma Z)$ is

$$
q(x,y)=\sum_{k\in\mathbb Z}\Big[\varphi_\sigma(y-x+2kw)+\varphi_\sigma(-y-x+2kw)\Big].
$$

In the first sum substitute $k\to-k$ and use the evenness of $\varphi_\sigma$: $\varphi_\sigma(y-x-2kw)=\varphi_\sigma(x-y+2kw)$,
so the first sum is unchanged under $x\leftrightarrow y$. The second sum depends on $x+y$ only. Hence $q(x,y)=q(y,x)$.

Clipping maps the whole event $\lbrace x+\sigma Z\ge w\rbrace$ onto the point $w$. The proposal law from any interior
$x$ then has an atom at $w$ of mass $1-\Phi((w-x)/\sigma)$, while from $w$ the proposal has only a density at interior
points. No symmetric $q$ exists, the plain Metropolis ratio is wrong, and the chain accumulates mass on the boundary.
$\square$

The numerical check uses $E\equiv0$ on $[0,1]$, $\sigma=0.3$, 2000 chains $\times$ 500 steps started uniformly.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats

def reflect(Y, lo, hi):
    w = hi - lo; z = np.mod(Y - lo, 2 * w); return lo + np.where(z > w, 2 * w - z, z)

sigma, R, steps = 0.3, 2000, 500
for mode in ("reflect", "clip"):
    rng = np.random.default_rng(1); x = rng.random(R)            # E = 0: target uniform, every proposal accepted
    for _ in range(steps):
        y = x + sigma * rng.standard_normal(R)
        x = reflect(y, 0.0, 1.0) if mode == "reflect" else np.clip(y, 0.0, 1.0)
    h = np.histogram(x, bins=10, range=(0, 1))[0]
    print(f"{mode:7s}: histogram {h}, chi2 vs uniform p = {stats.chisquare(h).pvalue:.2g}, "
          f"mass exactly on {{0,1}} = {np.mean((x == 0) | (x == 1)):.0%}")

# the image-sum density of the reflected proposal is symmetric
phi = lambda z: np.exp(-z**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
q = lambda x, y: sum(phi(y - x + 2 * k) + phi(-y - x + 2 * k) for k in range(-10, 11))
X, Y = np.random.default_rng(2).random((2, 10000))
print(f"max |q(x,y) - q(y,x)| = {np.abs(q(X, Y) - q(Y, X)).max():.1e}; "
      f"integral of q(0.1, .) over [0,1] = {np.trapezoid(q(0.1, np.linspace(0, 1, 20001)), dx=1 / 20000):.6f}")
```

| proposal | histogram (10 bins) | $\chi^2$ test vs uniform | mass exactly on $\lbrace0,1\rbrace$ |
|---|---|---|---|
| reflect | 212 188 199 207 180 212 213 195 192 202 | $p=0.77$ | 0 |
| clip | 407 127 168 152 149 149 153 129 132 434 | $p=5.5\times10^{-127}$ | 32% |

The image-sum formula for $q$ also gives $\max\vert q(x,y)-q(y,x)\vert=4.4\times10^{-16}$ on random pairs, and
$q(0.1,\cdot)$ integrates to 1 on $[0,1]$.

### Exercise 3 (★★) — Swap acceptance for Gaussian energies, and a constant-acceptance ladder for SK

**Derivation.** At stationarity the replicas are independent. Write $\Delta\beta=\beta_1-\beta_2\gt0$ and
$X=\Delta\beta\,(E_1-E_2)$, the log of the swap ratio. For Gaussian energies,

$$
X\sim\mathcal N(m,s^2),\qquad m=\Delta\beta(\mu_1-\mu_2),\qquad s^2=\Delta\beta^2(\sigma_1^2+\sigma_2^2).
$$

For $X\sim\mathcal N(m,s^2)$ one has $\mathbb E\min(1,e^X)=\Phi(m/s)+e^{m+s^2/2}\Phi(-m/s-s)$. For small $\Delta\beta$
use $d\mu/d\beta=-\sigma^2$, so that $\mu_1-\mu_2\approx-\sigma^2\Delta\beta$ and $\sigma_1\approx\sigma_2\approx\sigma$.
Then $s^2=2\Delta\beta^2\sigma^2$ and $m=-\Delta\beta^2\sigma^2=-s^2/2$, the exponential factor equals 1, and, with
$2\Phi(-a)=\mathrm{erfc}(a/\sqrt2)$,

$$
A=2\Phi(-s/2)=\mathrm{erfc}\Big(\frac{\Delta\beta\,\sigma}{2}\Big),\qquad(\Delta\beta\,\sigma)^2=\Big(\frac{\Delta\beta}{\beta}\Big)^2C(\beta),\quad C=\beta^2\sigma^2 .
$$

A constant acceptance $A^{\ast}$ therefore needs $\Delta\beta_i=2\,\mathrm{erfc}^{-1}(A^{\ast})/\sigma(\beta_i)$:
closely spaced temperatures where the heat capacity peaks (Kofke 2002). A Monte Carlo check with $10^6$ draws gives
0.7238 against 0.7237 for $\Delta\beta\sigma=0.5$, and 0.3959 against 0.3961 for $\Delta\beta\sigma=1.2$.

**SK ladder** ($n=100$, the test instance of Notebook 2). $\sigma(\beta)$ is measured on 12 geometric $\beta$ values by
fixed-$\beta$ single-flip Metropolis (1500 sweeps, 500 burn-in); the ladder is built by the recursion above with
$A^{\ast}=0.4$ and checked by a PT run of 2000 sweeps (Rao–Blackwellised acceptance, i.e. the mean of
$\min(1,e^X)$ over the attempts).

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.special import erfc, erfcinv

# (1) Monte Carlo check of A = E min(1, e^X) = erfc(dbeta*sigma/2) for Gaussian energies
rng = np.random.default_rng(0)
for x in (0.5, 1.2):                                     # x = dbeta * sigma (take sigma = 1, beta_1 - beta_2 = x)
    E1 = rng.normal(-x, 1, 10**6); E2 = rng.normal(0, 1, 10**6)   # mu_1 - mu_2 = -sigma^2 dbeta
    print(f"dbeta*sigma = {x}: Monte Carlo {np.mean(np.minimum(1, np.exp(x * (E1 - E2)))):.4f}, formula {erfc(x / 2):.4f}")

# (2) SK model of Notebook 2 (test instance, seed 42)
n = 100
J = np.triu(np.random.default_rng(42).standard_normal((n, n)), 1); J = (J + J.T) / np.sqrt(n)

def pt_sk(betas, sweeps, rng, swaps=True):
    """Single-flip Metropolis at each beta (one sweep = n proposals per replica), then even/odd neighbour swaps.
    Returns the energy trace (sweeps x M) and the Rao-Blackwellised swap acceptance of each neighbouring pair."""
    M = len(betas); S = rng.choice([-1.0, 1.0], (M, n)); Hf = S @ J; E = -0.5 * np.sum(S * Hf, 1)
    rows = np.arange(M); trace = np.empty((sweeps, M)); acc_sum = np.zeros(M - 1); acc_cnt = np.zeros(M - 1)
    for t in range(sweeps):
        I = rng.integers(0, n, (n, M)); U = rng.random((n, M))
        for s in range(n):
            i = I[s]; s_i = S[rows, i]; dE = 2 * s_i * Hf[rows, i]
            idx = np.flatnonzero((dE <= 0) | (U[s] < np.exp(-betas * np.maximum(dE, 0))))
            if idx.size:
                Hf[idx] -= 2 * s_i[idx, None] * J[i[idx]]; S[idx, i[idx]] = -s_i[idx]; E[idx] += dE[idx]
        if swaps:
            for m in range(t % 2, M - 1, 2):
                a = min(1.0, np.exp((betas[m] - betas[m + 1]) * (E[m] - E[m + 1])))
                acc_sum[m] += a; acc_cnt[m] += 1
                if rng.random() < a:
                    S[[m, m + 1]], Hf[[m, m + 1]], E[[m, m + 1]] = S[[m + 1, m]], Hf[[m + 1, m]], E[[m + 1, m]]
        trace[t] = E
    return trace, acc_sum / np.maximum(acc_cnt, 1)

bgrid = np.geomspace(0.3, 3.0, 12)
trace, _ = pt_sk(bgrid, 1500, np.random.default_rng(1), swaps=False)     # fixed-beta Metropolis chains
sig = trace[500:].std(0); C = bgrid**2 * sig**2
print("beta      ", np.round(bgrid, 2)); print("heat cap. ", np.round(C, 1))
ladder = [bgrid[0]]
while ladder[-1] < bgrid[-1]:                      # dbeta_i = 2 erfcinv(A*) / sigma(beta_i), A* = 0.4
    ladder.append(ladder[-1] + 2 * erfcinv(0.4) / np.interp(np.log(ladder[-1]), np.log(bgrid), sig))
ladder = np.array(ladder[:-1] + [bgrid[-1]])
print(f"designed ladder (M = {len(ladder)}):", np.round(ladder, 3))
for name, lad in [("designed", ladder), ("geometric", np.geomspace(0.3, 3.0, len(ladder)))]:
    _, acc = pt_sk(lad[::-1].copy(), 2000, np.random.default_rng(2))         # pt_sk expects beta_1 > beta_2 > ...
    print(f"{name:9s} pair acceptances (hot -> cold): {np.round(acc[::-1], 2)}")
```

The heat capacity $C=\beta^2\sigma^2$ rises from 4.4 at $\beta=0.3$ (the paramagnetic value is $\beta^2n/2=4.5$) to a
peak of about 44 at $\beta\approx1.05$, i.e. at the spin-glass transition $T_c=1$, and falls to 17–18 at
$\beta\approx2.4$–3. The designed ladder has $M=11$ temperatures: 0.300, 0.470, 0.639, 0.812, 0.994, 1.183, 1.399,
1.666, 2.010, 2.462, 3.000, tightest around the peak.

| ladder | pair acceptances, in order of increasing $\beta$ (hot $\to$ cold) | spread |
|---|---|---|
| designed | 0.38 0.44 0.40 0.40 0.43 0.44 0.47 0.45 0.49 0.51 | 0.38–0.51 |
| geometric, same $M$ | 0.70 0.65 0.54 0.46 0.37 0.32 0.30 0.34 0.39 0.48 | 0.30–0.70 |

The designed ladder is close to flat around the target; the residual drift towards the cold end comes from the
Gaussian and small-$\Delta\beta$ approximations and from the noisy $\sigma(\beta)$ estimates. The geometric ladder has a
bottleneck just below $T_c$.

### Exercise 4 (★★) — Ingber's very fast simulated re-annealing (VFSR)

Per coordinate, $y_i=x_i+(u_i-\ell_i)\,\mathrm{sgn}(v-\tfrac12)\,T_g[(1+1/T_g)^{\vert2v-1\vert}-1]$ with $v\sim U(0,1)$.
The generating and acceptance temperatures both follow $T(k)=T(0)e^{-ck^{1/d}}$ (Ingber 1989), with $c$ chosen so that
$T_g$ ends at `gen_end` and $T_{\text{acc}}$ at $10^{-5}T_0$. There is no re-annealing step. The sampler is checked
against the documented CDF $\tfrac12+\mathrm{sgn}(s)\ln(1+\vert s\vert/T)/(2\ln(1+1/T))$: at $T=0.01$ the empirical
values are 0.4246, 0.5106, 0.7596 against 0.4249, 0.5103, 0.7598.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from scipy.optimize import brentq
from utils import BENCHMARKS, shifted_rotated, random_rotation

def reflect(Y, lo, hi):
    w = hi - lo; z = np.mod(Y - lo, 2 * w); return lo + np.where(z > w, 2 * w - z, z)

def estimate_T0(f, lo, hi, d, sigma, chi0, rng, M=2000):
    X = rng.uniform(lo, hi, (M, d)); dl = f(reflect(X + sigma * rng.standard_normal((M, d)), lo, hi)) - f(X)
    up = dl[dl > 0]; return brentq(lambda T: np.mean(np.exp(-up / T)) - chi0, 1e-9, 1e9)

def gauss_adaptive_sa(f, lo, hi, d, budget, R, rng, T0, sigma0, X0, a_target=0.234, T_end_ratio=1e-5):
    """The notebook's 'Gauss-adaptive' variant: Robbins-Monro step size, geometric T0 -> 1e-5 T0."""
    X = X0.copy(); fX = f(X); best = fX.copy(); log_sig = np.full(R, np.log(sigma0)); alpha = T_end_ratio ** (1 / budget)
    for k in range(1, budget):
        Y = reflect(X + np.exp(log_sig)[:, None] * rng.standard_normal((R, d)), lo, hi); fY = f(Y)
        acc = (fY <= fX) | (rng.random(R) < np.exp(-np.maximum(fY - fX, 0) / (T0 * alpha**k)))
        X[acc], fX[acc] = Y[acc], fY[acc]; np.minimum(best, fX, out=best); log_sig += 0.02 * (acc - a_target)
    return best

def vfsr_step(u, Tg):
    """Ingber's per-coordinate generating law on [-1, 1] (multiply by the box width)."""
    return np.sign(u - 0.5) * Tg * ((1 + 1 / Tg) ** np.abs(2 * u - 1) - 1)

def vfsr(f, lo, hi, d, budget, R, rng, T0, X0, gen_end=1e-6, acc_end=1e-5):
    """Generating and acceptance temperatures T(k) = T(0) exp(-c k^(1/d)); no re-annealing."""
    X = X0.copy(); fX = f(X); best = fX.copy()
    c_g = -np.log(gen_end) / budget ** (1 / d); c_a = -np.log(acc_end) / budget ** (1 / d)
    for k in range(1, budget):
        Tg = np.exp(-c_g * k ** (1 / d)); Ta = T0 * np.exp(-c_a * k ** (1 / d))
        Y = reflect(X + (hi - lo) * vfsr_step(rng.random((R, d)), Tg), lo, hi); fY = f(Y)
        acc = (fY <= fX) | (rng.random(R) < np.exp(-np.maximum(fY - fX, 0) / Ta))
        X[acc], fX[acc] = Y[acc], fY[acc]; np.minimum(best, fX, out=best)
    return best

# the sampler against its documented CDF  1/2 + sgn(s) ln(1 + |s|/T) / (2 ln(1 + 1/T))
s = vfsr_step(np.random.default_rng(0).random(10**6), 0.01)
cdf = lambda x, T: 0.5 + np.sign(x) * np.log1p(np.abs(x) / T) / (2 * np.log1p(1 / T))
print("VFSR CDF at s = -0.01, 0.001, 0.1: empirical", [round(float(np.mean(s <= x)), 4) for x in (-0.01, 0.001, 0.1)],
      "formula", [round(float(cdf(x, 0.01)), 4) for x in (-0.01, 0.001, 0.1)])

d, budget, R = 10, 20000, 15
Rot = random_rotation(d, np.random.default_rng(5))
problems = {"rastrigin": BENCHMARKS["rastrigin"].f, "ackley": BENCHMARKS["ackley"].f,
            "rastrigin (rotated)": shifted_rotated(BENCHMARKS["rastrigin"].f, np.zeros(d), Rot)}
for name, f in problems.items():
    b = BENCHMARKS[name.split()[0]]; sig0 = 0.1 * (b.upper - b.lower)
    T0 = estimate_T0(f, b.lower, b.upper, d, sig0, 0.8, np.random.default_rng(8))
    X0 = np.random.default_rng(9).uniform(b.lower, b.upper, (R, d))            # the starts of Section 3
    ref = gauss_adaptive_sa(f, b.lower, b.upper, d, budget, R, np.random.default_rng(10), T0, sig0, X0)
    out = [f"Gauss-adaptive {np.median(ref):.3g}"]
    for ge in ((1e-3, 1e-6, 1e-9) if name != "rastrigin (rotated)" else (1e-9,)):
        v = vfsr(f, b.lower, b.upper, d, budget, R, np.random.default_rng(10), T0, X0, gen_end=ge)
        out.append(f"VFSR gen_end {ge:g}: {np.median(v):.3g} (p = {stats.mannwhitneyu(v, ref).pvalue:.2g})")
    print(f"{name:20s} median f_best - f*: " + " | ".join(out))
```

The setting is $d=10$, 20 000 evaluations, 15 seeds, and the $T_0$ and starting points of Section 3. The table gives
the median of $f_{\text{best}}-f^{\ast}$, with Mann–Whitney $p$-values against Gauss-adaptive SA.

| function | VFSR, gen_end $10^{-3}$ | $10^{-6}$ | $10^{-9}$ | Gauss-adaptive |
|---|---|---|---|---|
| Rastrigin | 26.9 ($p=0.03$) | 2.37 ($p=3\times10^{-6}$) | **0.0056** ($p=3\times10^{-6}$) | 29.9 |
| Ackley | 5.4 ($p=3\times10^{-6}$) | 0.37 ($p=0.03$) | 0.024 ($p=0.03$) | $\mathbf{9.3\times10^{-5}}$ |
| Rastrigin, randomly **rotated** | – | – | 37.8 ($p=0.06$) | **31.8** |

VFSR is spectacular on Rastrigin, but only because Rastrigin is **separable**. VFSR perturbs each coordinate
independently with a law that concentrates on tiny steps with heavy tails, so it effectively does coordinate-wise
search, and a separable function can be solved coordinate by coordinate. After a random rotation the advantage
disappears: VFSR is worse in median, though not significantly. On Ackley it stalls at the precision allowed by its
generating temperature. This is a textbook case for always testing rotated variants before crediting an algorithm.

### Exercise 5 (★★) — Round trips, not swap acceptance, track sampling quality

Each replica carries a label, and we count completed round trips: a walker that reaches the coldest slot after having
visited the hottest one completes a trip. The setting is the bimodal target of Section 4 with a geometric ladder from
$\beta=1$ to $1/32$, step sizes $0.4/\sqrt\beta$, and an equal budget of 240 000 evaluations per run
(sweeps $=240000/M$). The 48 runs per $M$ are vectorised. Quality is the RMSE, over the 48 runs, of the $\beta=1$
estimate of $\Pr(X\lt0)=0.6878$ (10% burn-in).

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.integrate import quad

U = lambda x: 16 * (x**2 - 1) ** 2 + 0.4 * x                 # the bimodal target of Section 4
exact = quad(lambda x: np.exp(-U(x)), -np.inf, 0)[0] / quad(lambda x: np.exp(-U(x)), -np.inf, np.inf)[0]

def pt_round_trips(M, budget, R, rng, burn=0.1):
    """R independent PT runs (vectorised) with a geometric ladder beta = 1 .. 1/32 and M replicas.
    Returns the cold-replica estimates of P(X<0), the mean swap acceptance and the round trips per run."""
    betas = 1.0 / np.geomspace(1, 32, M); step = 0.4 / np.sqrt(betas); sweeps = budget // M
    x = np.ones((R, M)); ux = U(x); label = np.tile(np.arange(M), (R, 1))    # label[r, slot] = walker in that slot
    last_end = np.full((R, M), -1)            # per walker: 1 = hot end touched since the last visit to the cold end
    trips = np.zeros(R); cold_left = np.zeros(R); n_acc = n_try = 0; rows = np.arange(R)
    for t in range(sweeps):
        y = x + step * rng.standard_normal((R, M)); uy = U(y)
        acc = np.log(rng.random((R, M))) < -betas * (uy - ux); x[acc], ux[acc] = y[acc], uy[acc]
        m = np.arange(t % 2, M - 1, 2)
        if m.size:
            ok = np.log(rng.random((R, m.size))) < (betas[m] - betas[m + 1]) * (ux[:, m] - ux[:, m + 1])
            n_acc += ok.sum(); n_try += ok.size
            r_, j_ = np.nonzero(ok); a_, b_ = m[j_], m[j_] + 1
            for arr in (x, ux, label):
                arr[r_, a_], arr[r_, b_] = arr[r_, b_].copy(), arr[r_, a_].copy()
        w0, wM = label[:, 0], label[:, -1]                      # walkers at the coldest / hottest slot
        trips += last_end[rows, w0] == 1                        # back at the cold end after touching the hot end
        last_end[rows, w0] = 0; last_end[rows, wM] = 1
        if t >= burn * sweeps:
            cold_left += x[:, 0] < 0
    return cold_left / (sweeps - int(np.ceil(burn * sweeps))), n_acc / max(n_try, 1), trips

print(f"exact P(X<0) at beta = 1: {exact:.4f}")
print(" M  swap acc.  round trips/run  RMSE")
for M in (2, 3, 4, 5, 7, 10, 14, 20):
    est, acc, trips = pt_round_trips(M, 240_000, 48, np.random.default_rng(M))
    print(f"{M:2d}  {acc:9.2f}  {trips.mean():15.0f}  {np.sqrt(np.mean((est - exact) ** 2)):.4f}")
```

| $M$ | 2 | 3 | 4 | 5 | 7 | 10 | 14 | 20 |
|---|---|---|---|---|---|---|---|---|
| swap acceptance | 0.20 | 0.51 | 0.65 | 0.73 | 0.82 | 0.88 | 0.92 | 0.94 |
| round trips per run | 12 284 | 10 967 | 9 234 | 7 925 | 6 119 | 4 583 | 3 439 | 2 499 |
| RMSE | 0.0077 | 0.0068 | 0.0077 | 0.0075 | 0.0068 | 0.0088 | 0.0090 | 0.0106 |

Swap acceptance increases monotonically with $M$, yet the error gets *worse* for large $M$. Round trips decrease with
$M$ (at a fixed evaluation budget each walker needs about $M^2$ swap attempts to diffuse across the ladder, and there
are fewer sweeps), and the error follows them once they drop below about 5000 per run: 0.0068–0.0077 for $M\le7$
against 0.0088–0.0106 for $M\ge10$. With 48 runs the relative standard error of an RMSE is about 10%, so the
differences among $M\le7$ are noise, while the increase from $M=7$ to $M=20$ (0.0068 to 0.0106) is not. In this
one-dimensional problem the hottest replica crosses the barrier easily, so even $M=2$ with 20% swap acceptance
transports enough; the acceptance rate alone would have recommended the worst ladder.

### Exercise 6 (★★★) — PT sweeps are $\Pi$-invariant but not reversible; ergodicity of the cold replica

Let $U=\bigotimes_mP_m$, where $P_m=P_{\beta_m}$ is the Metropolis kernel, and let $S$ be a swap kernel (one pair, or
several disjoint pairs).

*Invariance.* $(\Pi U)(y)=\prod_m(\pi_mP_m)(y_m)=\Pi(y)$ because each factor is $\pi_m$-invariant. $S$ is a
Metropolis–Hastings kernel for $\Pi$ with an involutive, hence symmetric, proposal, so it is $\Pi$-reversible and in
particular $\Pi$-invariant. Therefore $\Pi US=\Pi S=\Pi$, and the same holds for any composition of sweeps.

*Non-reversibility.* $U$ and $S$ are $\Pi$-reversible: $D_\Pi U=U^\top D_\Pi$ and $D_\Pi S=S^\top D_\Pi$. Then
$K=US$ is reversible iff $D_\Pi US=(D_\Pi US)^\top=S^\top U^\top D_\Pi=S^\top D_\Pi U=D_\Pi SU$. Since $D_\Pi$ is
invertible, this happens iff $US=SU$, which fails in general.

*Numerics* on the 6-state ring with $\beta=(2,1,0.4)$, 216 joint states:

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from itertools import product

E6 = np.array([0.0, 2.0, 1.0, 3.0, 0.5, 2.5]); n = len(E6); betas = [2.0, 1.0, 0.4]

def metro_ring(beta):
    P = np.zeros((n, n))
    for i in range(n):
        for j in ((i - 1) % n, (i + 1) % n):
            P[i, j] += 0.5 * min(1.0, np.exp(-beta * (E6[j] - E6[i])))
        P[i, i] = 1 - P[i].sum()
    return P

gibbs = lambda b: np.exp(-b * E6) / np.exp(-b * E6).sum()
states = list(product(range(n), repeat=3)); idx = {s: k for k, s in enumerate(states)}
Pi = np.array([np.prod([gibbs(b)[s[m]] for m, b in enumerate(betas)]) for s in states])
Ukern = np.kron(np.kron(metro_ring(betas[0]), metro_ring(betas[1])), metro_ring(betas[2]))

def swap_kernel(m):
    """Swap replicas m and m+1 with probability min(1, exp((b_m - b_{m+1})(E_m - E_{m+1})))."""
    S = np.zeros((len(states), len(states)))
    for s in states:
        a = min(1.0, np.exp((betas[m] - betas[m + 1]) * (E6[s[m]] - E6[s[m + 1]])))
        t = list(s); t[m], t[m + 1] = t[m + 1], t[m]
        S[idx[s], idx[tuple(t)]] += a; S[idx[s], idx[s]] += 1 - a
    return S

S12, S23 = swap_kernel(0), swap_kernel(1)
inv = lambda K: np.abs(Pi @ K - Pi).max()
rev = lambda K: np.abs(Pi[:, None] * K - (Pi[:, None] * K).T).max()
C = Ukern @ S12 @ Ukern @ S23
for name, K in [("U", Ukern), ("S12", S12), ("U S12 (even sweep)", Ukern @ S12), ("U S23 (odd sweep)", Ukern @ S23), ("full cycle", C)]:
    print(f"{name:20s} invariance error {inv(K):.1e}   detailed-balance error {rev(K):.1e}")
ev = np.sort(np.abs(np.linalg.eigvals(C)))[::-1]
C200 = np.linalg.matrix_power(C, 200)
print(f"cycle: second-largest eigenvalue modulus {ev[1]:.3f}; max_x TV(delta_x C^200, Pi) = {0.5 * np.abs(C200 - Pi).sum(1).max():.1e}")
print("C^2 entrywise positive:", bool((np.linalg.matrix_power(C, 2) > 0).all()))
```

| kernel | $\max\vert\Pi K-\Pi\vert$ | $\max\vert\Pi(x)K(x,y)-\Pi(y)K(y,x)\vert$ |
|---|---|---|
| $U$ alone, $S_{12}$ alone | $2\times10^{-17}$, $4\times10^{-18}$ | $4\times10^{-18}$ (reversible) |
| even sweep $US_{12}$ | $2\times10^{-17}$ | $1.0\times10^{-3}$ |
| odd sweep $US_{23}$ | $3\times10^{-17}$ | $7.8\times10^{-3}$ |
| full cycle $C=US_{12}US_{23}$ | $3\times10^{-17}$ | $1.3\times10^{-2}$ |

The cycle $C$ has second-largest eigenvalue modulus 0.804, $C^2$ is already entrywise positive, and
$\max_x\Vert\delta_xC^{200}-\Pi\Vert_{TV}=5.5\times10^{-15}$.

*Ergodicity.* Assume each $P_m$ is irreducible and aperiodic, i.e. primitive.

1. For a Metropolis kernel with a fixed proposal $q$, the **zero pattern** of $P_\beta$ does not depend on
   $\beta\in(0,\infty)$. Off-diagonal entries are $q(x,y)\min(1,e^{-\beta\Delta})$, positive iff $q(x,y)\gt0$. The
   diagonal entry is positive iff $q(x,x)\gt0$ or some proposed move from $x$ is uphill, which depends only on $q$ and
   the signs of the $\Delta$'s. The zero pattern of a product of nonnegative matrices depends only on the factors'
   patterns, and $P_\beta^t\gt0$ for $t\ge t_0$. Hence any product of $t\ge t_0$ Metropolis kernels with arbitrary
   temperatures is entrywise positive.
2. A single pair swap is accepted with probability $\min(1,e^{\Delta\beta\,\Delta E})\ge a_{\min}\gt0$ on the finite
   space. A swap kernel acting on $p$ disjoint pairs is the product of the pair kernels, so $S\ge a_{\min}^pQ$
   entrywise, where $Q$ is the permutation matrix that swaps all $p$ pairs.
3. Therefore $C^r\ge c_r\,UQ_eUQ_o\cdots UQ_eUQ_o$ with $c_r\gt0$. Moving the permutations to the right conjugates
   each $U$ into $\bigotimes_mP_{\beta_{\sigma(m)}}$ for some permutation $\sigma$, a tensor product of Metropolis
   kernels with permuted temperatures. By the mixed-product rule the product becomes
   $\big[\bigotimes_m(\text{product of }2r\text{ Metropolis kernels})\big]\cdot Q'$, with $Q'$ a permutation matrix.
   For $2r\ge t_0$ each tensor factor is positive by step 1, so $C^r\gt0$.

$C$ is therefore primitive with the unique invariant law $\Pi$. The ergodic theorem for finite chains gives
$\frac1N\sum_{t\lt N}g(X_t)\to\Pi(g)$ a.s. for every $g$; the same argument applies to the chain observed after the
first half-sweep, so averaging over all half-sweeps also converges to $\Pi(g)$. Taking $g(x)=\mathbf 1[x_1=z]$ shows
that the empirical measure of the coldest replica converges to its $\Pi$-marginal, which is $\pi_{\beta_1}$. $\square$

---

## Lab — Simulated Annealing for the TSP and MAX-3SAT

### Exercise 1 (★) — 2-opt inverse and neighbourhood size

A 2-opt move deletes the tour edges $(a,b),(c,d)$ and adds $(a,c),(b,d)$, so
$\Delta=D_{ac}+D_{bd}-D_{ab}-D_{cd}$. The inverse move deletes $(a,c),(b,d)$ and restores $(a,b),(c,d)$. Its delta is
$D_{ab}+D_{cd}-D_{ac}-D_{bd}=-\Delta$.

*Counting.* A 2-opt move is determined by an unordered pair of tour edges. There are $\binom n2$ pairs. The $n$ pairs
of adjacent edges, which share a city, give back the same tour, so $\binom n2-n=n(n-3)/2$ non-trivial moves remain.
They are distinct: for non-adjacent edges the new edges $(a,c),(b,d)$ are not tour edges, so the new tour differs from
the old one and contains exactly $n-2$ old edges. The removed pair is recovered as the set of old edges missing from
the new tour, so different pairs give different tours. For $n=4$ this gives 2 neighbours out of the $(n-1)!/2=3$
tours, as it must. $\square$

### Exercise 2 (★) — Segment insertion with $L$ up to $n/2$; should $p_{\text{or}}$ depend on $n$?

`sa_tsp` below takes the maximal segment length `L_max` as a parameter (the segment length is uniform on
$1..L_{\max}$; `L_max=3` is the notebook's move). The runs use $\chi_0=0.5$, $T_{\text{end}}=T_0/100$, $20n^2$ moves,
8 instances per $n$ and common random numbers across the six variants. The table shows the mean excess over the best of
the six variants on each instance.

```python
import sys; sys.path.insert(0, "..")
import math
import numpy as np
from scipy.optimize import brentq
from utils import random_euclidean_tsp, tour_length

def neighbour_lists(D, k):
    return np.argsort(D, axis=1)[:, 1:k + 1].tolist()

def nearest_neighbour_tour(D, start=0):
    n = len(D); left = set(range(n)); t = [start]; left.remove(start)
    while left:
        a = t[-1]; b = min(left, key=lambda j: D[a, j]); t.append(b); left.remove(b)
    return t

def or_opt_delta(t, pos, Dl, i, L, c):
    n = len(t)
    if i + L > n:
        return None
    s1, sL, p, x = t[i], t[i + L - 1], t[i - 1], t[(i + L) % n]; jc = pos[c]
    if i <= jc < i + L or c == p:
        return None
    e = t[(jc + 1) % n]
    if i <= pos[e] < i + L:
        return None
    fwd, rev = Dl[c][s1] + Dl[sL][e], Dl[c][sL] + Dl[s1][e]
    return Dl[p][x] - Dl[p][s1] - Dl[sL][x] + min(fwd, rev) - Dl[c][e], rev < fwd

def apply_or_opt(t, i, L, c, reverse):
    seg = t[i:i + L]
    if reverse:
        seg.reverse()
    rest = t[:i] + t[i + L:]; ic = rest.index(c); return rest[:ic + 1] + seg + rest[ic + 1:]

def t0_from_acceptance(D, tour, chi0, rng, M=3000, k_nb=10):
    """Root of mean(exp(-delta+/T)) = chi0 over uphill neighbour-list 2-opt deltas sampled around `tour`."""
    n = len(D); NB = neighbour_lists(D, min(k_nb, n - 1)); pos = [0] * n
    for q, c in enumerate(tour): pos[c] = q
    ups = []
    for _ in range(M):
        i = int(rng.integers(n)); a, b = tour[i], tour[(i + 1) % n]; c = NB[a][int(rng.integers(len(NB[0])))]
        d = tour[(pos[c] + 1) % n]
        if c != b and d != a and D[a, c] + D[b, d] - D[a, b] - D[c, d] > 0:
            ups.append(D[a, c] + D[b, d] - D[a, b] - D[c, d])
    ups = np.array(ups); return brentq(lambda T: np.mean(np.exp(-ups / T)) - chi0, 1e-9, 1e3)

def sa_tsp(D, start, n_moves, T0, T_end, rng, k_nb=10, p_or=0.3, L_max=3, schedule="geometric"):
    """The notebook's SA-TSP (2-opt + or-opt, O(1) deltas, neighbour lists). Options: L_max (or-opt segment length
    uniform on 1..L_max; 3 = notebook), schedule 'geometric' (notebook) or 'lam' (modified-Lam acceptance targeting)."""
    n = len(D); Dl = D.tolist(); NB = neighbour_lists(D, min(k_nb, n - 1)); kn = len(NB[0])
    t = list(start); pos = [0] * n
    for q, c in enumerate(t): pos[c] = q
    L = tour_length(t, D); best_L = L; alpha = (T_end / T0) ** (1 / n_moves); T = T0
    U = rng.random((n_moves, 4)).tolist(); exp = math.exp; rho = 0.5
    for k in range(n_moves):
        u0, u1, u2, u3 = U[k]; i = int(u0 * n); accepted = False
        if u3 >= p_or:
            a = t[i]; ip = i + 1 if i + 1 < n else 0; b = t[ip]
            c = NB[a][int(u1 * kn)]; j = pos[c]; jp = j + 1 if j + 1 < n else 0; d = t[jp]
            if c != b and d != a:
                delta = Dl[a][c] + Dl[b][d] - Dl[a][b] - Dl[c][d]
                if delta <= 0 or u2 < exp(-delta / T):
                    lo, hi = (i + 1, j) if i < j else (j + 1, i)
                    seg = t[lo:hi + 1]; seg.reverse(); t[lo:hi + 1] = seg
                    for q in range(lo, hi + 1): pos[t[q]] = q
                    L += delta; accepted = True
        else:
            Ls = 1 + int(u1 * L_max); c = NB[t[i]][int(u2 * kn)]
            r = or_opt_delta(t, pos, Dl, i, Ls, c)
            if r is not None and (r[0] <= 0 or rng.random() < exp(-r[0] / T)):
                t = apply_or_opt(t, i, Ls, c, r[1])
                for q in range(n): pos[t[q]] = q
                L += r[0]; accepted = True
        if L < best_L - 1e-12:
            best_L = L
        if schedule == "geometric":
            T *= alpha
        else:                                   # modified Lam: follow the target acceptance profile of Notebook 2
            rho = (499 * rho + accepted) / 500; x = k / n_moves
            target = 0.44 + 0.56 * 560 ** (-x / 0.15) if x < 0.15 else (0.44 if x < 0.65 else 0.44 * 440 ** (-(x - 0.65) / 0.35))
            T = T * 0.999 if rho > target else T / 0.999
    return best_L
from scipy.stats import wilcoxon

n_inst = 8
for n in (50, 100):
    variants = [(Lm, p) for Lm in (3, n // 2) for p in (0.1, 0.3, 0.6)]
    res = np.zeros((n_inst, len(variants)))
    for a in range(n_inst):
        _, D = random_euclidean_tsp(n, np.random.default_rng(100 * n + a)); s = nearest_neighbour_tour(D)
        T0 = t0_from_acceptance(D, s, 0.5, np.random.default_rng(a))
        for v, (Lm, p) in enumerate(variants):          # common random numbers across variants
            res[a, v] = sa_tsp(D, s, 20 * n * n, T0, T0 / 100, np.random.default_rng(a), p_or=p, L_max=Lm)
    exc = 100 * (res / res.min(1, keepdims=True) - 1)
    for v, (Lm, p) in enumerate(variants):
        print(f"n = {n:3d}, L_max = {Lm:2d}, p_or = {p}: mean excess over the best variant {exc[:, v].mean():.2f}%")
    short, long_ = res[:, :3].mean(1), res[:, 3:].mean(1)
    print(f"n = {n}: L_max = 3 vs n/2 (averaged over p_or), paired Wilcoxon p = {wilcoxon(short, long_).pvalue:.2g}; "
          f"p_or = 0.1 vs 0.6 at L_max = 3: p = {wilcoxon(res[:, 0], res[:, 2]).pvalue:.2g}")
```

| $n$ | $L_{\max}$ | $p_{\text{or}}=0.1$ | 0.3 | 0.6 |
|---|---|---|---|---|
| 50 | 3 | 0.62% | 1.01% | **0.59%** |
| 50 | 25 | 1.05% | 1.05% | 1.70% |
| 100 | 3 | **0.73%** | 0.94% | 1.03% |
| 100 | 50 | 0.79% | 0.78% | 1.20% |

Long segments are slightly worse on average at $n=50$, but not significantly (paired Wilcoxon over the 8 instances,
$L_{\max}=3$ vs $n/2$: $p=0.31$ at $n=50$, $0.84$ at $n=100$). A long insertion is a large, poorly correlated move, like
the QAP insertion of Notebook 1, Exercise 5, and it buys nothing here. There is no trend of the best $p_{\text{or}}$ with
$n$ either (at $L_{\max}=3$, $p_{\text{or}}=0.1$ vs 0.6: $p=0.64$ and $0.46$); all differences are below 1.1% and of the
order of the run-to-run noise. At this budget there is no evidence that $p_{\text{or}}$ should be adapted to $n$.

### Exercise 3 (★★) — Modified-Lam schedule for the TSP

`sa_tsp` is given a `schedule="lam"` option: after each move a running acceptance rate
$\rho\leftarrow(499\rho+\mathbf 1_{\text{acc}})/500$ is updated, and $T\leftarrow0.999T$ if $\rho$ exceeds the
modified-Lam target of Notebook 2, $T\leftarrow T/0.999$ otherwise. The comparison uses the three instances of A.4,
$2\times10^5$ moves, the same seeds, and the mean excess over the best tour found on each instance by any of the ten
cells.

```python
import sys; sys.path.insert(0, "..")
import math
import numpy as np
from scipy.optimize import brentq
from utils import random_euclidean_tsp, tour_length

def neighbour_lists(D, k):
    return np.argsort(D, axis=1)[:, 1:k + 1].tolist()

def nearest_neighbour_tour(D, start=0):
    n = len(D); left = set(range(n)); t = [start]; left.remove(start)
    while left:
        a = t[-1]; b = min(left, key=lambda j: D[a, j]); t.append(b); left.remove(b)
    return t

def or_opt_delta(t, pos, Dl, i, L, c):
    n = len(t)
    if i + L > n:
        return None
    s1, sL, p, x = t[i], t[i + L - 1], t[i - 1], t[(i + L) % n]; jc = pos[c]
    if i <= jc < i + L or c == p:
        return None
    e = t[(jc + 1) % n]
    if i <= pos[e] < i + L:
        return None
    fwd, rev = Dl[c][s1] + Dl[sL][e], Dl[c][sL] + Dl[s1][e]
    return Dl[p][x] - Dl[p][s1] - Dl[sL][x] + min(fwd, rev) - Dl[c][e], rev < fwd

def apply_or_opt(t, i, L, c, reverse):
    seg = t[i:i + L]
    if reverse:
        seg.reverse()
    rest = t[:i] + t[i + L:]; ic = rest.index(c); return rest[:ic + 1] + seg + rest[ic + 1:]

def t0_from_acceptance(D, tour, chi0, rng, M=3000, k_nb=10):
    """Root of mean(exp(-delta+/T)) = chi0 over uphill neighbour-list 2-opt deltas sampled around `tour`."""
    n = len(D); NB = neighbour_lists(D, min(k_nb, n - 1)); pos = [0] * n
    for q, c in enumerate(tour): pos[c] = q
    ups = []
    for _ in range(M):
        i = int(rng.integers(n)); a, b = tour[i], tour[(i + 1) % n]; c = NB[a][int(rng.integers(len(NB[0])))]
        d = tour[(pos[c] + 1) % n]
        if c != b and d != a and D[a, c] + D[b, d] - D[a, b] - D[c, d] > 0:
            ups.append(D[a, c] + D[b, d] - D[a, b] - D[c, d])
    ups = np.array(ups); return brentq(lambda T: np.mean(np.exp(-ups / T)) - chi0, 1e-9, 1e3)

def sa_tsp(D, start, n_moves, T0, T_end, rng, k_nb=10, p_or=0.3, L_max=3, schedule="geometric"):
    """The notebook's SA-TSP (2-opt + or-opt, O(1) deltas, neighbour lists). Options: L_max (or-opt segment length
    uniform on 1..L_max; 3 = notebook), schedule 'geometric' (notebook) or 'lam' (modified-Lam acceptance targeting)."""
    n = len(D); Dl = D.tolist(); NB = neighbour_lists(D, min(k_nb, n - 1)); kn = len(NB[0])
    t = list(start); pos = [0] * n
    for q, c in enumerate(t): pos[c] = q
    L = tour_length(t, D); best_L = L; alpha = (T_end / T0) ** (1 / n_moves); T = T0
    U = rng.random((n_moves, 4)).tolist(); exp = math.exp; rho = 0.5
    for k in range(n_moves):
        u0, u1, u2, u3 = U[k]; i = int(u0 * n); accepted = False
        if u3 >= p_or:
            a = t[i]; ip = i + 1 if i + 1 < n else 0; b = t[ip]
            c = NB[a][int(u1 * kn)]; j = pos[c]; jp = j + 1 if j + 1 < n else 0; d = t[jp]
            if c != b and d != a:
                delta = Dl[a][c] + Dl[b][d] - Dl[a][b] - Dl[c][d]
                if delta <= 0 or u2 < exp(-delta / T):
                    lo, hi = (i + 1, j) if i < j else (j + 1, i)
                    seg = t[lo:hi + 1]; seg.reverse(); t[lo:hi + 1] = seg
                    for q in range(lo, hi + 1): pos[t[q]] = q
                    L += delta; accepted = True
        else:
            Ls = 1 + int(u1 * L_max); c = NB[t[i]][int(u2 * kn)]
            r = or_opt_delta(t, pos, Dl, i, Ls, c)
            if r is not None and (r[0] <= 0 or rng.random() < exp(-r[0] / T)):
                t = apply_or_opt(t, i, Ls, c, r[1])
                for q in range(n): pos[t[q]] = q
                L += r[0]; accepted = True
        if L < best_L - 1e-12:
            best_L = L
        if schedule == "geometric":
            T *= alpha
        else:                                   # modified Lam: follow the target acceptance profile of Notebook 2
            rho = (499 * rho + accepted) / 500; x = k / n_moves
            target = 0.44 + 0.56 * 560 ** (-x / 0.15) if x < 0.15 else (0.44 if x < 0.65 else 0.44 * 440 ** (-(x - 0.65) / 0.35))
            T = T * 0.999 if rho > target else T / 0.999
    return best_L

chis = [0.01, 0.05, 0.2, 0.5, 0.8]; n, K = 100, 20 * 100**2
res = np.zeros((2, 3, len(chis)))
for a in range(3):                                           # the three instances of A.4, same seeds
    _, D = random_euclidean_tsp(n, np.random.default_rng(50 + a)); s = nearest_neighbour_tour(D)
    for ci, chi in enumerate(chis):
        T0 = t0_from_acceptance(D, s, chi, np.random.default_rng(a))
        for si, sch in enumerate(("geometric", "lam")):
            res[si, a, ci] = sa_tsp(D, s, K, T0, T0 / 100, np.random.default_rng(7 + a), schedule=sch)
exc = 100 * (res / res.min(axis=(0, 2), keepdims=True) - 1).mean(1)       # mean excess over the best of all 10 cells
for si, sch in enumerate(("geometric, T_end = T0/100", "modified Lam")):
    print(f"{sch:26s}" + " ".join(f"{e:6.2f}" for e in exc[si]) + f"   range {exc[si].max() - exc[si].min():.2f}")
```

| $\chi_0$ | 0.01 | 0.05 | 0.2 | 0.5 | 0.8 | range |
|---|---|---|---|---|---|---|
| geometric, $T_{\text{end}}=T_0/100$ | 4.02 | 1.24 | 0.84 | **0.66** | 2.30 | 3.36 |
| modified Lam | 1.56 | 1.95 | 2.90 | 1.30 | **1.03** | 1.87 |

Adaptation roughly halves the sensitivity to $\chi_0$ and removes the two failure corners (too cold, too hot): the
controller quickly moves $T$ to where the acceptance matches the profile. It does not remove the sensitivity, and the
best geometric cell (0.66%) beats every Lam cell. With one run per cell and instance the individual entries are noisy
(about ±1%), so only the reduced range is a robust conclusion.

### Exercise 4 (★★) — Don't-look bits

A city whose last 10 proposals (as the anchor of a move) were all rejected goes to sleep. Drawing a sleeping city costs
no delta evaluation (it is redrawn); any accepted move wakes the cities it touches, and all cities are woken if all are
asleep. The budget counts computed deltas, $20n^2$. Plain SA runs through the same function with the bits disabled. To
keep the block under two minutes it uses 4 instances at $n=100$ and 2 at $n=200$.

```python
import sys; sys.path.insert(0, "..")
import math
import numpy as np
from scipy.optimize import brentq
from utils import random_euclidean_tsp, tour_length

def neighbour_lists(D, k):
    return np.argsort(D, axis=1)[:, 1:k + 1].tolist()

def nearest_neighbour_tour(D, start=0):
    n = len(D); left = set(range(n)); t = [start]; left.remove(start)
    while left:
        a = t[-1]; b = min(left, key=lambda j: D[a, j]); t.append(b); left.remove(b)
    return t

def or_opt_delta(t, pos, Dl, i, L, c):
    n = len(t)
    if i + L > n:
        return None
    s1, sL, p, x = t[i], t[i + L - 1], t[i - 1], t[(i + L) % n]; jc = pos[c]
    if i <= jc < i + L or c == p:
        return None
    e = t[(jc + 1) % n]
    if i <= pos[e] < i + L:
        return None
    fwd, rev = Dl[c][s1] + Dl[sL][e], Dl[c][sL] + Dl[s1][e]
    return Dl[p][x] - Dl[p][s1] - Dl[sL][x] + min(fwd, rev) - Dl[c][e], rev < fwd

def apply_or_opt(t, i, L, c, reverse):
    seg = t[i:i + L]
    if reverse:
        seg.reverse()
    rest = t[:i] + t[i + L:]; ic = rest.index(c); return rest[:ic + 1] + seg + rest[ic + 1:]

def t0_from_acceptance(D, tour, chi0, rng, M=3000, k_nb=10):
    """Root of mean(exp(-delta+/T)) = chi0 over uphill neighbour-list 2-opt deltas sampled around `tour`."""
    n = len(D); NB = neighbour_lists(D, min(k_nb, n - 1)); pos = [0] * n
    for q, c in enumerate(tour): pos[c] = q
    ups = []
    for _ in range(M):
        i = int(rng.integers(n)); a, b = tour[i], tour[(i + 1) % n]; c = NB[a][int(rng.integers(len(NB[0])))]
        d = tour[(pos[c] + 1) % n]
        if c != b and d != a and D[a, c] + D[b, d] - D[a, b] - D[c, d] > 0:
            ups.append(D[a, c] + D[b, d] - D[a, b] - D[c, d])
    ups = np.array(ups); return brentq(lambda T: np.mean(np.exp(-ups / T)) - chi0, 1e-9, 1e3)
import time

def sa_tsp_dlb(D, start, n_evals, T0, T_end, rng, dlb=True, patience=10, k_nb=10, p_or=0.3):
    """The notebook's SA-TSP, optionally with don't-look bits: a city whose last `patience` proposals (as the anchor
    of the move) were all rejected goes to sleep; drawing a sleeping city costs no delta evaluation; any accepted
    move wakes the cities it touches; if everything sleeps, all cities are woken. Budget = computed deltas."""
    n = len(D); Dl = D.tolist(); NB = neighbour_lists(D, min(k_nb, n - 1)); kn = len(NB[0]); exp = math.exp
    t = list(start); pos = [0] * n
    for q, c in enumerate(t): pos[c] = q
    L = tour_length(t, D); best_L = L; alpha = (T_end / T0) ** (1 / n_evals); T = T0
    fails = [0] * n; n_awake = n; evals = draws = 0
    while evals < n_evals:
        u0, u1, u2, u3, u4 = rng.random(5).tolist(); i = int(u0 * n); draws += 1
        if dlb and fails[t[i]] >= patience:
            continue
        evals += 1; anchor = t[i]; touched = None
        if u3 >= p_or:
            a = t[i]; b = t[(i + 1) % n]; c = NB[a][int(u1 * kn)]; j = pos[c]; d = t[(j + 1) % n]
            if c != b and d != a:
                delta = Dl[a][c] + Dl[b][d] - Dl[a][b] - Dl[c][d]
                if delta <= 0 or u2 < exp(-delta / T):
                    lo, hi = (i + 1, j) if i < j else (j + 1, i)
                    seg = t[lo:hi + 1]; seg.reverse(); t[lo:hi + 1] = seg
                    for q in range(lo, hi + 1): pos[t[q]] = q
                    L += delta; touched = (a, b, c, d)
        else:
            Ls = 1 + int(u1 * 3); c = NB[t[i]][int(u2 * kn)]
            r = or_opt_delta(t, pos, Dl, i, Ls, c)
            if r is not None and (r[0] <= 0 or u4 < exp(-r[0] / T)):
                touched = [t[i - 1], t[(i + Ls) % n], c, t[(pos[c] + 1) % n]] + t[i:i + Ls]
                t = apply_or_opt(t, i, Ls, c, r[1])
                for q in range(n): pos[t[q]] = q
                L += r[0]
        if touched is None:
            fails[anchor] += 1
            if fails[anchor] == patience:
                n_awake -= 1
                if n_awake == 0:
                    fails = [0] * n; n_awake = n
        else:
            for c in touched:
                if fails[c] >= patience: n_awake += 1
                fails[c] = 0
        best_L = min(best_L, L); T *= alpha
    return best_L, draws / evals

for n, n_inst in ((100, 4), (200, 2)):
    out = {False: [], True: []}; cpu = {False: 0.0, True: 0.0}; dpe = []
    for a in range(n_inst):
        _, D = random_euclidean_tsp(n, np.random.default_rng(1000 * n + a)); s = nearest_neighbour_tour(D)
        T0 = t0_from_acceptance(D, s, 0.5, np.random.default_rng(a))
        for dlb in (False, True):
            c0 = time.process_time()
            L, r = sa_tsp_dlb(D, s, 20 * n * n, T0, T0 / 100, np.random.default_rng(a), dlb=dlb)
            cpu[dlb] += time.process_time() - c0; out[dlb].append(L)
            if dlb: dpe.append(r)
    pl, dl = np.mean(out[False]), np.mean(out[True])
    print(f"n = {n}, {n_inst} instances: plain {pl:.3f}, DLB {dl:.3f} ({100 * (dl / pl - 1):+.2f}%), "
          f"random draws per evaluation {np.mean(dpe):.1f}, CPU time DLB / plain = {cpu[True] / cpu[False]:.1f}")
```

| $n$ | plain SA | with DLB | rel. diff | random draws per evaluation | CPU time DLB / plain |
|---|---|---|---|---|---|
| 100 (4 inst.) | 7.744 | 7.787 | +0.56% | 2.9 | 1.9 |
| 200 (2 inst.) | 10.592 | 10.528 | −0.60% | 3.0 | 1.7 |

Per *evaluation*, don't-look bits change the tour length by less than 1%, in opposite directions at the two sizes:
there is no detectable quality effect with so few instances. About two thirds of the cities are asleep at a typical
draw (3 draws per evaluation), and with random city selection and cheap $O(1)$ deltas those wasted draws cost more
than the deltas they save, so DLB is 1.7–1.9 times slower in CPU. Don't-look bits pay off in Bentley's setting, where
the moves are expensive local-search scans and the awake cities are kept in a *queue* instead of being found by
rejection sampling. Implementing that queue is the natural next step.

### Exercise 5 (★★) — Focused Metropolis search at fixed temperature

A fixed temperature corresponds to $\eta=e^{-1/T}$, the acceptance probability of a move that breaks one extra clause.
Focused SA runs at fixed $T=-1/\ln\eta$ with $n=200$, a budget of $2\times10^5$ flips and 10 instances per ratio. The
table shows the median number of flips to a satisfying assignment ($\infty$ when fewer than half of the runs succeed)
and the fraction solved.

```python
import sys; sys.path.insert(0, "..")
import math
import numpy as np
from utils import random_max3sat

class MaxSat:
    """Incremental MAX-SAT state: true-literal counts per clause and the set of unsatisfied clauses (the notebook's)."""
    def __init__(self, clauses, n, assign):
        self.cl = [list(map(int, c)) for c in clauses]; self.a = [bool(x) for x in assign]
        self.occ = [[] for _ in range(n)]
        for ci, c in enumerate(self.cl):
            for lit in c: self.occ[abs(lit) - 1].append((ci, lit > 0))
        self.tc = [sum(self.a[abs(l) - 1] == (l > 0) for l in c) for c in self.cl]
        self.unsat, self.where = [], [-1] * len(self.cl)
        for ci, k in enumerate(self.tc):
            if k == 0: self._add(ci)
    def _add(self, ci):
        self.where[ci] = len(self.unsat); self.unsat.append(ci)
    def _remove(self, ci):
        k = self.where[ci]; last = self.unsat.pop()
        if last != ci: self.unsat[k] = last; self.where[last] = k
        self.where[ci] = -1
    def break_make(self, v):
        br = mk = 0; av = self.a[v]
        for ci, pos in self.occ[v]:
            k = self.tc[ci]
            if k == 0: mk += 1
            elif k == 1 and av == pos: br += 1
        return br, mk
    def flip(self, v):
        av = self.a[v]
        for ci, pos in self.occ[v]:
            if av == pos:
                self.tc[ci] -= 1
                if self.tc[ci] == 0: self._add(ci)
            else:
                self.tc[ci] += 1
                if self.tc[ci] == 1: self._remove(ci)
        self.a[v] = not av

def focused_metropolis(clauses, n, T, budget, rng):
    """Focused SA at fixed T (= Focused Metropolis Search): flips used until a satisfying assignment, or inf."""
    st = MaxSat(clauses, n, rng.random(n) < 0.5)
    for flips in range(budget):
        if not st.unsat:
            return flips
        c = st.cl[st.unsat[int(rng.integers(len(st.unsat)))]]; v = abs(c[int(rng.integers(3))]) - 1
        br, mk = st.break_make(v)
        if br - mk <= 0 or rng.random() < math.exp(-(br - mk) / T):
            st.flip(v)
    return math.inf if st.unsat else budget

n, n_inst, budget = 200, 10, 200_000
etas = [0.2, 0.3, 0.36, 0.45, 0.55]
for ratio in (3.8, 4.0, 4.2):
    row = []
    for eta in etas:
        T = -1 / math.log(eta)                            # eta = exp(-1/T): acceptance of a move breaking one more clause
        f = np.array([focused_metropolis(random_max3sat(n, int(round(ratio * n)), np.random.default_rng(700 + a)), n, T,
                                         budget, np.random.default_rng(a)) for a in range(n_inst)])
        row.append(f"{np.median(f):>9.0f} ({np.mean(np.isfinite(f)):.0%})")
    print(f"m/n = {ratio}: " + " | ".join(f"eta={e}: {r}" for e, r in zip(etas, row)))
```

| $m/n$ | $\eta=0.2$ | 0.3 | 0.36 | 0.45 | 0.55 |
|---|---|---|---|---|---|
| 3.8 | 4 406 (100%) | **2 753** (100%) | 4 274 (100%) | 8 252 (100%) | 11 518 (100%) |
| 4.0 | 8 647 (90%) | **5 744** (100%) | 8 952 (100%) | 7 802 (90%) | 8 855 (90%) |
| 4.2 | $\infty$ (50%) | 139 328 (60%) | **47 366** (60%) | 71 510 (60%) | $\infty$ (40%) |

Times grow steeply with $m/n$. At $m/n=4.2$ the 40% of instances that no $\eta$ solves are probably unsatisfiable at
$n=200$ (finite-size threshold effects are large). The greedy end ($\eta=0.2$) and the noisy end ($\eta=0.55$) are
worse at 4.2, and the noisy end also at 3.8; the best values lie in the band $\eta\approx0.3$–$0.45$. This agrees
with Seitz, Alava & Orponen (2005), who report $\eta_{\text{opt}}\approx0.36$ for $N=10^5$ and $\alpha$ up to about 4.2.
With $n=200$ and 10 instances our location of the optimum within the band is noisy.

### Exercise 6 (★★★) — Noisy hyperparameter search: the winner's curse and remedies

Each evaluation now returns the MSE on a random half of the validation set. Its standard deviation at the optimum is
0.051, while the best and second-best configurations differ by only $6.7\times10^{-4}$. Budget: 300 evaluations, 40
seeds. The quality measure is the true (full-validation) regret of the returned configuration. Three variants:

- **plain** — the current point keeps its first noisy value; the returned point is the one with the lowest value ever
  recorded (1 evaluation per step);
- **re-evaluate** — the current point is re-evaluated at every step and compared with the candidate by the two fresh
  values (2 evaluations per step); a configuration's value is the mean of all its measurements;
- **average of 3** — every candidate is evaluated 3 times and the mean is used (3 evaluations per step).

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import wilcoxon

# Part C of the lab: 12 candidate features x 5 ridge strengths
dr = np.random.default_rng(12); n_tr, n_va, p_f = 60, 200, 12
Z = dr.standard_normal((n_tr + n_va, 4))
Xf = np.hstack([Z, Z + 0.8 * dr.standard_normal(Z.shape), dr.standard_normal((n_tr + n_va, 4))])
yf = Z @ np.array([1.0, -0.8, 0.6, 0.4]) + 0.8 * dr.standard_normal(n_tr + n_va)
Xtr, Xva, ytr, yva = Xf[:n_tr], Xf[n_tr:], yf[:n_tr], yf[n_tr:]
lambdas = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])

def residuals(code, li):
    """Validation residuals of the ridge model with feature mask `code` (bit j = feature j) and lambda index li."""
    idx = np.flatnonzero((code >> np.arange(p_f)) & 1); ym = ytr.mean()
    if idx.size == 0:
        return yva - ym
    mu = Xtr[:, idx].mean(0); Ac = Xtr[:, idx] - mu
    w = np.linalg.solve(Ac.T @ Ac + lambdas[li] * np.eye(idx.size), Ac.T @ (ytr - ym))
    return yva - ym - (Xva[:, idx] - mu) @ w

R2 = {(c, l): residuals(c, l) ** 2 for c in range(2**p_f) for l in range(5)}     # squared residuals, all configs
true = {k: v.mean() for k, v in R2.items()}                                      # exact (full-validation) MSE
opt = min(true, key=true.get); vals = np.sort(np.array(list(true.values())))
half_sd = np.std([R2[opt][np.random.default_rng(s).choice(n_va, n_va // 2, replace=False)].mean() for s in range(2000)])
print(f"optimum {true[opt]:.4f}; best - second best = {vals[1] - vals[0]:.1e}; noise s.d. at the optimum = {half_sd:.3f}")

def search(mode, budget, rng, T0=0.02, T_end=2e-4, r=3):
    """SA over (mask, lambda) with noisy evaluations (MSE on a random half of the validation set).
    plain : compare the candidate's noisy value with the current point's stored first value (1 eval/step)
    reeval: re-evaluate the current point at every step and compare the two fresh values (2 evals/step)
    avg   : evaluate every candidate r times and use the mean (r evals/step)
    noise-free: exact values. Returns (true regret, stored - true value of the returned point, truly-worse acceptances)."""
    evals = 0; meas = {}
    def measure(cfg, k=1):
        nonlocal evals; evals += k
        if mode == "noise-free":
            v = true[cfg]
        else:
            v = np.mean([R2[cfg][rng.choice(n_va, n_va // 2, replace=False)].mean() for _ in range(k)])
        s, c = meas.get(cfg, (0.0, 0)); meas[cfg] = (s + k * v, c + k); return v
    k_eval = {"plain": 1, "noise-free": 1, "reeval": 1, "avg": r}[mode]
    cur = (int(rng.integers(2**p_f)), int(rng.integers(5))); f = measure(cur, k_eval)
    first = {cur: f}; per_step = {"reeval": 2}.get(mode, k_eval)
    n_steps = (budget - k_eval) // per_step; alpha = (T_end / T0) ** (1 / n_steps); T = T0; worse = 0
    for _ in range(n_steps):
        c, l = cur
        cand = (c ^ (1 << int(rng.integers(p_f))), l) if rng.random() < 0.8 else (c, int(np.clip(l + rng.choice([-1, 1]), 0, 4)))
        f2 = measure(cand, k_eval); first.setdefault(cand, f2)
        if mode == "reeval":
            f = measure(cur)
        if f2 <= f or rng.random() < np.exp(-(f2 - f) / T):
            worse += true[cand] > true[cur]; cur, f = cand, f2
        T *= alpha
    if mode == "plain":                     # a configuration keeps the value it had when first evaluated
        best = min(first, key=first.get); stored = first[best]
    else:                                   # the mean of all measurements of a configuration
        best = min(meas, key=lambda k: meas[k][0] / meas[k][1]); stored = meas[best][0] / meas[best][1]
    return true[best] - true[opt], stored - true[best], worse

res = {m: np.array([search(m, 300, np.random.default_rng(s)) for s in range(40)]) for m in ("plain", "reeval", "avg", "noise-free")}
for m, r in res.items():
    q1, md, q3 = np.percentile(r[:, 0], [25, 50, 75])
    p = "" if m in ("plain", "noise-free") else f", Wilcoxon p vs plain = {wilcoxon(r[:, 0], res['plain'][:, 0]).pvalue:.2g}"
    print(f"{m:10s} true regret median {md:.3f} [{q1:.3f}, {q3:.3f}], optimum found {np.mean(r[:, 0] == 0):.0%}, "
          f"stored - true {r[:, 1].mean():+.3f}, truly-worse acceptances/run {r[:, 2].mean():.1f}{p}")
```

| mode | true regret median [IQR] | optimum found | stored minus true value of the returned point | truly uphill acceptances per run | Wilcoxon $p$ vs plain |
|---|---|---|---|---|---|
| plain | 0.031 [0.017, 0.050] | 0% | **−0.119** | 4.2 | – |
| re-evaluate current point | **0.015** [0.004, 0.033] | 2% | −0.057 | 26.3 | 0.004 |
| average of 3 | 0.021 [0.006, 0.040] | 0% | −0.046 | 3.5 | 0.058 |
| (noise-free SA, same budget) | 0.000 [0.000, 0.000] | 95% | 0 | 34.7 | – |

*Diagnosis.* The value plain SA reports for its answer is on average 0.119 below the truth, about 2.3 noise standard
deviations. This is the winner's curse: the configuration with the lowest recorded value is mostly the one whose
single measurement was luckiest. The same bias blocks the search: a lucky current point is rarely left (only 4.2 truly
uphill moves per run, against 34.7 in noise-free SA, whose controlled uphill moves are what lets it find the optimum in
95% of runs).

*Remedies.* Re-evaluating the current point breaks the blocking and, because every configuration visited repeatedly
accumulates measurements, halves the bias; it gives the best median regret and is significantly better than plain SA
($p=0.004$) despite taking half as many steps. Averaging three evaluations reduces the noise of each comparison by
$\sqrt3$ and also improves the regret, but not significantly at this budget ($p=0.058$). No variant reliably finds the
optimum: when the noise is about 75 times larger than the differences that matter, 300 evaluations cannot rank the top
configurations. The effective remedies change the *measurement*: more validation data, fixed validation folds (common
random numbers), or racing/bandit allocation of repeated evaluations (e.g. Hyperband), rather than the acceptance
rule alone.
