# Solutions — Week 3: Tabu Search and Trajectory Metaheuristics

> Try every exercise yourself before reading these solutions. The point is the struggle, not the answer.

Conventions. Every Python block is a standalone script: run it from inside the week folder
(`week3_tabu_search_and_trajectory_methods/`). It carries its own imports and compact copies of the notebook functions it
needs (for example `tabu_search_qap`, `vnd`, `tabucol`). The numbers quoted in the text are what the block prints.
Numbers from random instances depend on the seeds shown. Proofs are self-contained. The heaviest blocks
(Notebook 1 Exercise 2, Notebook 3 Exercise 3, Lab Exercises 5 and 6) need 30–60 s of CPU time each.

---

## Notebook 1 — Tabu Search Fundamentals

### Exercise 1 ★ — Aspiration by objective cannot revisit; aspiration "better than current" can trap TS

**(a)** Let $x_0, x_1, \dots, x_k$ be the solutions visited so far and $x^{\ast}_k$ the best of them, so
$f(x_j) \ge f(x^{\ast}_k)$ for every $j \le k$. A tabu move $m$ is aspirated only if $f(x_k \oplus m) \lt f(x^{\ast}_k)$.
Then $f(x_k \oplus m) \lt f(x_j)$ for every $j \le k$, so $x_k \oplus m \ne x_j$ for every $j$. The aspirated move leads
to a solution that has never been visited. $\square$

**(b)** Take $\lbrace 0,1 \rbrace^n$ with bit-flip moves, $n \ge 3$, and let $\vert x \vert$ be the number of ones:

$$
f(x) = \begin{cases} 1 & \vert x \vert = 0, \\ 0 & \vert x \vert = n, \\ 2 + \vert x \vert & \text{otherwise.} \end{cases}
$$

$x = 0$ is a local minimum but not a global one, since the global minimum is $1^n$. The tabu attribute is "bit $j$
just flipped": the reverse flip of $j$ is tabu for $t$ iterations.

*Aspiration "better than the current solution".* From $0$ every move goes uphill to some $e_j$ with $f = 3$. From
$e_j$ the reverse flip of $j$ is tabu, but it leads to $f = 1 \lt 3 = f(e_j)$, so it is aspirated. It is also the best
move, because every other neighbour has $\vert x \vert = 2 \lt n$ and $f = 4$ (this is where $n \ge 3$ is needed).
TS therefore returns to $0$. There, the flip of $j$ is now tabu, and it is not aspirated ($3 \gt 1$), so TS moves to
some $e_i$ with $i \ne j$ and comes straight back again. If the tenure is so long that every flip is tabu at $0$, the
usual rule "if no move is admissible, take the best move anyway" (the block's `argmin` does the same) again leads to
some $e_i$. Whatever the tenure, TS alternates between $0$ and its neighbours forever, and $f^{\ast} = 1$.

*Aspiration by objective with $t \ge n - 1$ (so in particular $t \ge n$).* From $e_j$ the reverse flip leads to $f = 1$, which is not
$\lt f^{\ast} = 1$, so it stays tabu. TS must climb to $\vert x \vert = 2$. By induction, after $\ell$ iterations
$\vert x \vert = \ell$, and the one-bits were set at iterations $1, \dots, \ell$. At iteration $\ell + 1$, a bit set at
iteration $i \ge 1$ is still tabu, because $i + t \ge 1 + (n-1) \ge \ell + 1$ for $\ell \le n - 1$. Flipping it back
gives $f \ge 1 = f^{\ast}$, so it is not aspirated. Adding a one is never tabu, because those bits have not been
flipped yet. TS keeps adding ones until it reaches $1^n$ with $f = 0$.

Run with $n = 8$ and 50 iterations, where "returns" counts the visits to $x = 0$ (for the objective rule, the returns
happen after $1^n$ has been found, when the search leaves the optimum again):

| rule | $t = 3$ | $t = 7 = n - 1$ | $t = 8$ | $t = 20$ |
|---|---|---|---|---|
| objective: best, returns | 1, 6 | **0**, 3 | **0**, 2 | **0**, 7 |
| current: best, returns | 1, 25 | 1, 25 | 1, 25 | 1, 25 |

```python
import sys; sys.path.insert(0, "..")
import numpy as np

def f_trap(x):
    k = x.sum(); n = len(x)
    return 1.0 if k == 0 else (0.0 if k == n else 2.0 + k)

def ts_bits(n, t, rule, iters=50):
    """Bit-flip TS from x = 0; rule = 'objective' (aspirate if < best) or 'current' (aspirate if < f(x)).
    If every move is tabu and none is aspirated, argmin over all-inf picks bit 0 ('take a move anyway')."""
    x = np.zeros(n, int); fx = f_trap(x); best = fx; tabu = np.zeros(n, int); visits0 = 0
    for k in range(1, iters + 1):
        vals = np.array([f_trap(np.where(np.arange(n) == j, 1 - x, x)) for j in range(n)])
        ref = best if rule == "objective" else fx
        adm = (tabu < k) | (vals < ref)
        j = int(np.argmin(np.where(adm, vals, np.inf)))
        x[j] ^= 1; fx = vals[j]; tabu[j] = k + t; best = min(best, fx); visits0 += fx == 1.0
    return best, visits0

for rule in ("objective", "current"):
    print(rule, [(float(b), int(r)) for b, r in (ts_bits(8, t, rule) for t in (3, 7, 8, 20))])
```

### Exercise 2 ★ — Tenure sweep with the weak ("both") rule

This repeats Section 5 exactly: the same 3 instances with $n = 20$, 8 seeds and 600 iterations. The gaps are
measured against the best value either rule found. The table gives the median gap in %.

| $t$ | 0 | 1 | 2 | 4 | 7 | 10 | 15 | 20 | 30 | 50 | 80 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weak | 1.77 | 1.39 | 1.36 | 1.10 | 0.70 | 0.58 | 0.54 | 0.51 | 0.50 | **0.23** | 0.23 | 0.25 |
| strong | 1.77 | 1.39 | 1.33 | 1.16 | 0.60 | 0.34 | 0.36 | **0.18** | 0.23 | 0.31 | 0.56 | 0.57 |

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import random_qap, qap_cost
def qap_delta_matrix(p, F, D):
    Dp = D[np.ix_(p, p)]; M = F @ Dp; dg = np.diag(M)
    Delta = 2.0 * (M + M.T - dg[:, None] - dg[None, :] + 2.0 * F * Dp)
    np.fill_diagonal(Delta, 0.0); return Delta

def qap_update_after_swap(Delta, q, r, s, F, D):     # Taillard's O(n^2) update; q = permutation after swap
    a = F[r] - F[s]; b = D[q[s], q] - D[q[r], q]
    new = Delta + 2.0 * np.subtract.outer(a, a) * np.subtract.outer(b, b)
    Dq = D[np.ix_(q, q)]; diagM = np.sum(F * Dq, axis=0)
    for i in (r, s):
        row = 2.0 * (F[:, i] @ Dq + Dq[:, i] @ F - diagM[i] - diagM + 2.0 * F[i] * Dq[i]); row[i] = 0.0
        new[i, :] = row; new[:, i] = row
    return new

def tabu_search_qap(F, D, n_iter, tenure, rng, rule="either", p0=None, trace=False):
    """Notebook 1's full-neighbourhood TS (strong 'either' or weak 'both' rule, aspiration by objective)."""
    n = len(F); p = rng.permutation(n) if p0 is None else np.array(p0)
    cost = qap_cost(p, F, D); best = cost; tabu_until = np.zeros((n, n), dtype=np.int64)
    iu, ju = np.triu_indices(n, 1); Delta = qap_delta_matrix(p, F, D); perms, costs = [], []
    for k in range(1, n_iter + 1):
        d = Delta[iu, ju]
        tA, tB = tabu_until[iu, p[ju]] >= k, tabu_until[ju, p[iu]] >= k
        adm = ~((tA & tB) if rule == "both" else (tA | tB)) | (cost + d < best - 1e-9)
        if not adm.any(): adm[:] = True
        j = int(np.argmin(np.where(adm, d, np.inf))); r, s = int(iu[j]), int(ju[j])
        tabu_until[r, p[r]] = tabu_until[s, p[s]] = k + tenure
        cost += d[j]; q = p.copy(); q[[r, s]] = q[[s, r]]
        Delta = qap_update_after_swap(Delta, q, r, s, F, D); p = q; best = min(best, cost)
        if trace: perms.append(tuple(p)); costs.append(cost)
    return dict(best=float(best), perms=perms, costs=costs)

tenures = [0, 1, 2, 4, 7, 10, 15, 20, 30, 50, 80, 120]
insts = [random_qap(20, np.random.default_rng(200 + i)) for i in range(3)]      # Section 5 instances
Rw, Rs = np.zeros((len(tenures), 3, 8)), np.zeros((len(tenures), 3, 8))
for a, t in enumerate(tenures):
    for i, (Fi, Di) in enumerate(insts):
        for sd in range(8):
            Rw[a, i, sd] = tabu_search_qap(Fi, Di, 600, t, np.random.default_rng(1000*i + sd), rule="both")["best"]
            Rs[a, i, sd] = tabu_search_qap(Fi, Di, 600, t, np.random.default_rng(1000*i + sd), rule="either")["best"]
bk = np.minimum(Rw.min(axis=(0, 2)), Rs.min(axis=(0, 2)))           # best value either rule found
for name, R in (("weak", Rw), ("strong", Rs)):
    g = np.median((100 * (R - bk[None, :, None]) / bk[None, :, None]).reshape(len(tenures), -1), axis=1)
    print(f"{name:>6}: " + "  ".join(f"t={t}:{v:.2f}" for t, v in zip(tenures, g)))
```

**Why the optimum shifts to larger tenures.** Each iteration makes 2 of the $n^2 = 400$ facility–location attributes
tabu, so roughly $\rho = 2t/n^2$ of them are tabu at any time. A swap $(r,s)$ checks two attributes, $(r, \pi_s)$ and
$(s, \pi_r)$. Under the strong rule it is tabu if either one is, which happens with probability about
$1 - (1 - \rho)^2 \approx 2\rho$. Under the weak rule both must be tabu, which happens with probability about
$\rho^2$. Reaching the same degree of restriction under the weak rule therefore needs a much longer tenure. The optimum
moves from $t \approx n$ to $t \approx 2.5n$–$6n$, and the right-hand side of the U is much flatter, because
over-restriction sets in far later. For $t \le 1$ the two rules coincide, since the only tabu attributes are those of
the reverse move, and the reverse move hits both of them.

### Exercise 3 ★★ — Swap delta for the asymmetric QAP

Let $\pi'$ be $\pi$ with $\pi_r$ and $\pi_s$ exchanged, so $\pi'_r = \pi_s$ and $\pi'_s = \pi_r$. Only terms with
$i \in \lbrace r,s \rbrace$ or $j \in \lbrace r,s \rbrace$ change. Split them into the four "inner" terms with both
indices in $\lbrace r,s \rbrace$ and the "row" and "column" terms with exactly one index in $\lbrace r,s \rbrace$:

$$
\begin{aligned}
\Delta ={}& F_{rr}(D_{\pi_s\pi_s} - D_{\pi_r\pi_r}) + F_{ss}(D_{\pi_r\pi_r} - D_{\pi_s\pi_s})
 + F_{rs}(D_{\pi_s\pi_r} - D_{\pi_r\pi_s}) + F_{sr}(D_{\pi_r\pi_s} - D_{\pi_s\pi_r}) \\
&+ \sum_{k \ne r,s} \Big[ (F_{kr} - F_{ks})(D_{\pi_k\pi_s} - D_{\pi_k\pi_r}) + (F_{rk} - F_{sk})(D_{\pi_s\pi_k} - D_{\pi_r\pi_k}) \Big].
\end{aligned}
$$

The column term comes from $F_{kr} D_{\pi_k \pi_r} \to F_{kr} D_{\pi_k \pi_s}$ and
$F_{ks} D_{\pi_k \pi_s} \to F_{ks} D_{\pi_k \pi_r}$, and the row term is its mirror image. When $F$ and $D$ are
symmetric with zero diagonals, the inner terms cancel and the two sums coincide, which gives back the notebook's
formula $2\sum_{k \ne r,s}(F_{kr}-F_{ks})(d_{ks}-d_{kr})$. The cost is still $O(n)$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import qap_cost

def qap_delta_asym(p, r, s, F, D):
    pr, ps = p[r], p[s]
    k = np.ones(len(p), bool); k[[r, s]] = False; pk = p[k]
    d = (F[r, r] * (D[ps, ps] - D[pr, pr]) + F[s, s] * (D[pr, pr] - D[ps, ps])
         + F[r, s] * (D[ps, pr] - D[pr, ps]) + F[s, r] * (D[pr, ps] - D[ps, pr]))
    d += np.sum((F[k, r] - F[k, s]) * (D[pk, ps] - D[pk, pr]))      # column terms
    d += np.sum((F[r, k] - F[s, k]) * (D[ps, pk] - D[pr, pk]))      # row terms
    return d

rng = np.random.default_rng(0); err = 0.0
for _ in range(50):                              # random asymmetric instances, non-zero diagonals
    n = int(rng.integers(3, 15))
    F = rng.integers(0, 10, (n, n)).astype(float); D = rng.random((n, n)) * 10
    p = rng.permutation(n); c = qap_cost(p, F, D)
    for r in range(n):
        for s in range(r + 1, n):
            q = p.copy(); q[[r, s]] = q[[s, r]]
            err = max(err, abs(qap_delta_asym(p, r, s, F, D) - (qap_cost(q, F, D) - c)))
print(f"max |delta - (cost(q) - cost(p))| over 50 asymmetric instances: {err:.1e}")
assert err < 1e-9
```

Validation: 50 random asymmetric instances with $3 \le n \le 14$ were used, with integer $F$ and real $D$, both having
non-zero diagonals. Every swap of a random permutation was checked against `qap_cost` differences. The maximum absolute
error was $1.8 \times 10^{-12}$.

### Exercise 4 ★★ — Eventual periodicity of tenure-$t$ TS and a period-4 example

**Theorem.** TS on a finite space with a fixed tenure $t$, aspiration by objective and deterministic tie-breaking
(including the "all moves tabu, take the best anyway" rule) generates a sequence $x_k$ that is eventually periodic.
Its period is at most $\vert S \vert \cdot \vert M \vert^{t}$, where $\vert M \vert = n(n-1)/2$ is the number of swap
moves.

*Proof.* (i) *The memory is a function of the last $t$ moves.* An attribute set at iteration $k_0$ gets
`tabu_until` $= k_0 + t$. At iteration $k+1$ it is tabu iff it was last set at some $k_0 \ge k + 1 - t$, that is,
during the last $t$ iterations. The attributes set at iteration $k_0$ are $(r, \pi^{(k_0-1)}_r)$ and
$(s, \pi^{(k_0-1)}_s)$. They depend on the move and on the solution before it. That solution can be recovered from
$x_k$ by undoing the swaps $m_k, m_{k-1}, \dots$, which are involutions. So the tabu status at iteration $k+1$ is a
function of $\sigma_k = (x_k, m_{k-t+1}, \dots, m_k)$.
(ii) *The aspiration level changes finitely often.* $f^{\ast}$ is non-increasing, and each time it changes it takes a
new value from the finite set $f(S)$. So after some iteration $K_0$ it is constant.
(iii) For $k \ge K_0$, the next move is a deterministic function of $\sigma_k$, and $\sigma_{k+1}$ is a function of
$\sigma_k$ and that move. So $\sigma_{k+1} = \Psi(\sigma_k)$ for a fixed map $\Psi$ on a set of at most
$\vert S \vert \, \vert M \vert^t$ states. By the pigeonhole principle, some state repeats within that many steps. From
then on the state sequence, and with it $x_k$, is periodic, with period at most
$\vert S \vert \, \vert M \vert^t = n! \, (n(n-1)/2)^t$. For $t = 0$ this reduces to the notebook's bound $\vert S \vert$. $\square$

**A tiny instance with period > 2 at $t = 1$.** A search over `random_qap(4, default_rng(seed))`, starting from the
identity permutation, succeeded at the first seed:

```text
F = [[0 6 5 2]        D = [[0.   9.71 9.55 7.82]
     [6 0 0 0]             [9.71 0.   0.52 1.93]
     [5 0 0 9]             [9.55 0.52 0.   1.91]
     [2 0 9 0]]            [7.82 1.93 1.91 0.  ]]
```

Tabu search with $t = 1$ from the identity permutation cycles with **period 4** through the costs
$163.82 \to 130.02 \to 130.14 \to 162.18 \to 163.82 \to \dots$ and the permutations
(1,0,2,3), (3,0,2,1), (3,0,1,2), (2,0,1,3). With $t = 1$ the immediate reversal is forbidden, but a return after two
steps is not, so the orbit has length 4.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import random_qap, qap_cost
def qap_delta_matrix(p, F, D):
    Dp = D[np.ix_(p, p)]; M = F @ Dp; dg = np.diag(M)
    Delta = 2.0 * (M + M.T - dg[:, None] - dg[None, :] + 2.0 * F * Dp)
    np.fill_diagonal(Delta, 0.0); return Delta

def qap_update_after_swap(Delta, q, r, s, F, D):     # Taillard's O(n^2) update; q = permutation after swap
    a = F[r] - F[s]; b = D[q[s], q] - D[q[r], q]
    new = Delta + 2.0 * np.subtract.outer(a, a) * np.subtract.outer(b, b)
    Dq = D[np.ix_(q, q)]; diagM = np.sum(F * Dq, axis=0)
    for i in (r, s):
        row = 2.0 * (F[:, i] @ Dq + Dq[:, i] @ F - diagM[i] - diagM + 2.0 * F[i] * Dq[i]); row[i] = 0.0
        new[i, :] = row; new[:, i] = row
    return new

def tabu_search_qap(F, D, n_iter, tenure, rng, rule="either", p0=None, trace=False):
    """Notebook 1's full-neighbourhood TS (strong 'either' or weak 'both' rule, aspiration by objective)."""
    n = len(F); p = rng.permutation(n) if p0 is None else np.array(p0)
    cost = qap_cost(p, F, D); best = cost; tabu_until = np.zeros((n, n), dtype=np.int64)
    iu, ju = np.triu_indices(n, 1); Delta = qap_delta_matrix(p, F, D); perms, costs = [], []
    for k in range(1, n_iter + 1):
        d = Delta[iu, ju]
        tA, tB = tabu_until[iu, p[ju]] >= k, tabu_until[ju, p[iu]] >= k
        adm = ~((tA & tB) if rule == "both" else (tA | tB)) | (cost + d < best - 1e-9)
        if not adm.any(): adm[:] = True
        j = int(np.argmin(np.where(adm, d, np.inf))); r, s = int(iu[j]), int(ju[j])
        tabu_until[r, p[r]] = tabu_until[s, p[s]] = k + tenure
        cost += d[j]; q = p.copy(); q[[r, s]] = q[[s, r]]
        Delta = qap_update_after_swap(Delta, q, r, s, F, D); p = q; best = min(best, cost)
        if trace: perms.append(tuple(int(v) for v in p)); costs.append(round(float(cost), 2))
    return dict(best=float(best), perms=perms, costs=costs)

def period_of(seq):
    tail = seq[len(seq) // 2:]
    for P in range(1, len(tail) // 2):
        if all(tail[i] == tail[i + P] for i in range(len(tail) - P)):
            return P

for seed in range(2000):
    n = 4 if seed < 1000 else 5
    F, D = random_qap(n, np.random.default_rng(seed))
    res = tabu_search_qap(F, D, 400, 1, np.random.default_rng(0), p0=np.arange(n), trace=True)
    P = period_of(res["perms"])
    if P is not None and P > 2: break
print(f"seed {seed}, n = {n}, period {P}")
print("F =\n", F.astype(int), "\nD =\n", D)
print("costs:", res["costs"][-P:], "\nperms:", res["perms"][-P:])
```

### Exercise 5 ★★ — Elite candidate lists vs random candidate lists

Every $L$ iterations, the search scans all $n(n-1)/2$ swaps and keeps the $m$ best. In between, it re-evaluates only
those $m$ swaps with the $O(n)$ delta. The budget is 60 000 evaluations per run, with tenure 10 and the same
3 instances × 8 seeds as Section 6. Gaps are measured against the best value any of the five variants found on the
instance.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import wilcoxon
from utils import random_qap, qap_cost

def swap_deltas(p, R, S, F, D):                 # O(n) delta of each swap (R[m], S[m]), vectorised
    Dp = D[np.ix_(p, p)]
    return 2.0 * (np.einsum("km,km->m", F[:, R] - F[:, S], Dp[:, S] - Dp[:, R]) + 2.0 * F[R, S] * Dp[R, S])

def ts_step(p, cost, best, R, S, d, tabu_until, k, tenure):
    """Strong tabu rule + aspiration by objective; returns the updated (p, cost, best)."""
    tabu = (tabu_until[R, p[S]] >= k) | (tabu_until[S, p[R]] >= k)
    adm = ~tabu | (cost + d < best - 1e-9)
    if not adm.any(): adm[:] = True
    j = int(np.argmin(np.where(adm, d, np.inf))); r, s = int(R[j]), int(S[j])
    tabu_until[r, p[r]] = tabu_until[s, p[s]] = k + tenure
    p = p.copy(); p[[r, s]] = p[[s, r]]; cost += d[j]
    return p, cost, min(best, cost)

def ts_random_cand(F, D, max_evals, tenure, rng, frac):          # Notebook 1, Section 6
    n = len(F); iu, ju = np.triu_indices(n, 1); m = max(1, int(round(frac * len(iu))))
    p = rng.permutation(n); cost = qap_cost(p, F, D); best = cost
    tabu_until = np.zeros((n, n), dtype=np.int64); evals, k = 0, 0
    while evals + m <= max_evals:
        k += 1; pick = rng.choice(len(iu), m, replace=False); R, S = iu[pick], ju[pick]
        d = swap_deltas(p, R, S, F, D); evals += m
        p, cost, best = ts_step(p, cost, best, R, S, d, tabu_until, k, tenure)
    return best

def ts_elite_cand(F, D, max_evals, tenure, rng, L=10, m=20):
    n = len(F); iu, ju = np.triu_indices(n, 1)
    p = rng.permutation(n); cost = qap_cost(p, F, D); best = cost
    tabu_until = np.zeros((n, n), dtype=np.int64); evals, k = 0, 0
    while True:
        k += 1
        if (k - 1) % L == 0:                                   # full scan -> elite list of the m best swaps
            if evals + len(iu) > max_evals: break
            d_all = swap_deltas(p, iu, ju, F, D); evals += len(iu)
            keep = np.argsort(d_all)[:m]; R, S = iu[keep], ju[keep]; d = d_all[keep]
        else:                                                  # re-evaluate only the elite list
            if evals + m > max_evals: break
            d = swap_deltas(p, R, S, F, D); evals += m
        p, cost, best = ts_step(p, cost, best, R, S, d, tabu_until, k, tenure)
    return best

insts = [random_qap(20, np.random.default_rng(200 + i)) for i in range(3)]      # Notebook 1 instances
variants = {"random 10%": lambda F, D, g: ts_random_cand(F, D, 60_000, 10, g, 0.10),
            "random 25%": lambda F, D, g: ts_random_cand(F, D, 60_000, 10, g, 0.25),
            "elite L=10, m=20": lambda F, D, g: ts_elite_cand(F, D, 60_000, 10, g, 10, 20),
            "elite L=5, m=10": lambda F, D, g: ts_elite_cand(F, D, 60_000, 10, g, 5, 10),
            "elite L=20, m=30": lambda F, D, g: ts_elite_cand(F, D, 60_000, 10, g, 20, 30)}
V = np.array([[[run(F, D, np.random.default_rng(1000*i + sd)) for sd in range(8)] for i, (F, D) in enumerate(insts)]
              for run in variants.values()])
bk = V.min(axis=(0, 2))                                   # best value any variant found, per instance
G = (100 * (V - bk[None, :, None]) / bk[None, :, None]).reshape(len(variants), -1)
for name, g in zip(variants, G):
    print(f"{name:>17}: median gap {np.median(g):.2f}%  IQR [{np.percentile(g, 25):.2f}, {np.percentile(g, 75):.2f}]")
print(f"elite(10,20) vs random 25%: Wilcoxon p = {wilcoxon(G[2], G[1]).pvalue:.2f} over {G.shape[1]} pairs")
```

| variant | median gap | IQR |
|---|---|---|
| random 10% | 0.73% | [0.60, 0.92] |
| random 25% | 0.55% | [0.22, 0.67] |
| elite $L=10, m=20$ | 0.73% | [0.17, 0.90] |
| elite $L=5, m=10$ | 0.58% | [0.00, 0.83] |
| elite $L=20, m=30$ | 0.79% | [0.43, 1.11] |

Elite $(10,20)$ vs random 25% gives a Wilcoxon signed-rank $p$ of 0.24 over 24 pairs. Elite lists reach the
best-known value more often (lower first quartile), because they spend their evaluations on moves that were
recently the best. Their median is no better, however: between refreshes the stored moves go stale, since
their deltas change after every swap. At this budget, neither design is significantly better.

### Exercise 6 ★★★ — Tabu search for hyperparameter search

**Design.** The model is ridge regression on polynomial features of a 2-D input (120 points) whose second
coordinate is scaled by 1000. The target is $\sin(2.5x_1) + 0.6u^2 + 0.5x_1u$ plus noise with s.d. 0.1, where
$u = x_2/1000$. Each configuration is scored by 5-fold CV MSE. The configuration space has 4 coordinates:
$\log_{10}\lambda \in \lbrace -6,\dots,2 \rbrace$ (9 values), degree $1\dots7$, scaling
$\lbrace$none, standard, min–max$\rbrace$ and interaction terms $\lbrace$off, on$\rbrace$, which gives 378 configurations.
Moves change one coordinate. Ordinal coordinates ($\lambda$, degree) move to an adjacent value, and categorical ones
can take any other value. The **attribute** is (coordinate, old value), so TS may not undo a change for a random
tenure in $\lbrace 2,3,4 \rbrace$. Aspiration is by objective. An evaluation is one CV run of a *distinct*
configuration, cached so that revisits are free. Because the space is small, it was enumerated to obtain the exact
optimum.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from itertools import product

# data: 2-D input, second coordinate badly scaled (x1000)
g = np.random.default_rng(42); N = 120
X = np.column_stack([g.uniform(-1, 1, N), 1000 * g.uniform(-1, 1, N)])
u = X[:, 1] / 1000
y = np.sin(2.5 * X[:, 0]) + 0.6 * u**2 + 0.5 * X[:, 0] * u + 0.1 * g.normal(size=N)
folds = np.array_split(g.permutation(N), 5)

def features(Xr, deg, inter):
    cols = [np.ones(len(Xr))]
    for dg in range(1, deg + 1):
        for a in range(dg + 1):
            if inter or a in (0, dg):
                cols.append(Xr[:, 0] ** a * Xr[:, 1] ** (dg - a))
    return np.column_stack(cols)

def cv_mse(c):                                   # c = (lambda index, degree index, scaling, interactions)
    lam, deg, scal, inter = 10.0 ** (c[0] - 6), c[1] + 1, c[2], c[3]
    err = 0.0
    for f in folds:
        tr = np.setdiff1d(np.arange(N), f); Xtr, Xte = X[tr], X[f]
        if scal == 1: mu, sd = Xtr.mean(0), Xtr.std(0); Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        if scal == 2: lo, hi = Xtr.min(0), Xtr.max(0); Xtr, Xte = (Xtr - lo) / (hi - lo), (Xte - lo) / (hi - lo)
        A, B = features(Xtr, deg, inter), features(Xte, deg, inter)
        w = np.linalg.lstsq(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr], rcond=None)[0]
        err += np.sum((B @ w - y[f]) ** 2)
    return err / N

sizes = (9, 7, 3, 2)                             # 9 x 7 x 3 x 2 = 378 configurations
table = {c: cv_mse(c) for c in product(*map(range, sizes))}   # exhaustive, only to know the optimum
opt = min(table.values())

def ts_hpo(budget, rng, tenure=(2, 4)):
    cache = {}
    def f(c):
        if c not in cache: cache[c] = table[c]          # one CV run per *distinct* configuration
        return cache[c]
    cur = tuple(int(rng.integers(s)) for s in sizes); best = f(cur); tabu = {}; k = 0
    while len(cache) < budget:
        k += 1; nb = []
        for j in range(4):
            vals = [cur[j] - 1, cur[j] + 1] if j < 2 else range(sizes[j])     # ordinal: adjacent; categorical: any
            nb += [(j, v, cur[:j] + (v,) + cur[j+1:]) for v in vals if 0 <= v < sizes[j] and v != cur[j]]
        scored = []
        for j, v, c in nb:
            if len(cache) >= budget and c not in cache: continue
            val = f(c)
            if tabu.get((j, v), 0) < k or val < best: scored.append((val, j, c))   # aspiration by objective
        if not scored: break
        val, j, c = min(scored)
        tabu[(j, cur[j])] = k + int(rng.integers(tenure[0], tenure[1] + 1))       # attribute (coordinate, old value)
        cur = c; best = min(best, val)
    return best

keys = list(table)
def random_search(budget, rng):
    return min(table[keys[i]] for i in rng.choice(len(keys), budget, replace=False))

print(f"optimum CV-MSE = {opt:.4f}")
for budget in (20, 40, 80):
    ts = np.array([ts_hpo(budget, np.random.default_rng(s)) for s in range(100)]) - opt
    rs = np.array([random_search(budget, np.random.default_rng(s)) for s in range(100)]) - opt
    print(f"budget {budget:>2}: TS hit rate {np.mean(ts < 1e-12):.2f}, median excess {np.median(ts):.1e} | "
          f"RS hit rate {np.mean(rs < 1e-12):.2f}, median excess {np.median(rs):.1e}")
```

Results over 100 seeds each (optimum CV-MSE = 0.0125). Random search samples distinct configurations uniformly
without replacement, with the same number of CV runs.

| budget (CV runs) | TS: hit rate | TS: median excess | random search: hit rate | RS: median excess |
|---|---|---|---|---|
| 20 | **0.17** | 1.4e-4 | 0.08 | 1.6e-4 |
| 40 | **0.49** | 8.4e-9 | 0.17 | 9.5e-6 |
| 80 | **1.00** | 0 | 0.31 | 9.2e-8 |

With 20 CV runs (5% of the space), TS spends much of its budget climbing out of its random start, and its median
excess is no better than random search's, although it hits the optimum twice as often. From 40 runs onward, the
memory-guided local moves find the exact optimum about three times as often as random search, and at 80 runs (21% of
the space) every TS run finds it. The tiny median excesses (around $10^{-8}$) come from configurations that differ from
the optimum only by a neighbouring $\lambda$ at the small end, where the CV error hardly changes. The lesson for AI
practice: TS pays off when there is structure (adjacent $\lambda$ and degree values have similar CV error) and the
budget allows at least a few dozen evaluations.

---

## Notebook 2 — Intensification, Diversification and Reactive TS

### Exercise 1 ★ — Length of a path-relinking walk between permutations

Let $x$ be the current permutation and $x^G$ the guide. Define the position map $\varphi = x^{-1} \circ x^G$, so
$\varphi(i)$ is the position $j$ where $x_j = x^G_i$. Position $i$ is correct iff $\varphi(i) = i$. The $m$ incorrect
positions split into $c \ge 1$ cycles of $\varphi$ of length $\ge 2$. The map $\varphi$ is conjugate to
$x^G \circ (x^I)^{-1}$, which acts on locations, so both have the same cycle type.

A relinking step at position $i$ swaps $x_i$ with $x_j$, where $j = \varphi(i)$. Afterwards $x_i = x^G_i$, so $i$
becomes a fixed point. The predecessor $k$ of $i$ in its cycle ($\varphi(k) = i$, meaning $x^G_k$ used to sit at $i$)
now points to $j$. The cycle $(\dots, k, i, j, \dots)$ therefore becomes $(\dots, k, j, \dots)$:

- if the cycle had length $\ell \ge 3$, then $m$ decreases by 1 and $c$ is unchanged;
- if $\ell = 2$ (so $k = j$), both positions become correct: $m$ decreases by 2 and $c$ by 1.

In both cases $m - c$ decreases by exactly 1, and the walk ends when $m = c = 0$. So **every** relinking path, greedy or
not, has exactly $m - c$ steps. That is at most $m - 1$, with equality iff $c = 1$, meaning the incorrect positions form
a single cycle of $x^G \circ (x^I)^{-1}$. $\square$ (In the notebook, $m - c = 13$ for $n = 20$.)

### Exercise 2 ★ — Birthday bound for Zobrist hashing

For distinct permutations $\pi \ne \sigma$,
$H(\pi) \oplus H(\sigma) = \bigoplus_{(i,\ell) \in A_\pi \triangle A_\sigma} z_{i\ell}$, where
$A_\pi = \lbrace (i, \pi_i) \rbrace$. The symmetric difference is non-empty. An XOR of one or more independent uniform
64-bit keys is uniform: condition on all keys but one, and XOR-ing a uniform key onto a fixed value is uniform. Hence
$P(H(\pi) = H(\sigma)) = 2^{-64}$. By the union bound over the $\binom{m}{2}$ pairs of $m$ fixed distinct solutions,

$$
P(\text{some collision}) \le \binom{m}{2} 2^{-64} = \frac{m(m-1)}{2^{65}} .
$$

For reactive TS, the visited solutions depend on the hashes. Couple the run with an idealised run that recognises
repetitions with exact keys (the permutations themselves). The two runs agree until the first collision among the
idealised run's first $m$ distinct solutions. Those solutions do not depend on the values of the $z$'s (the keys are
drawn once, before the run, and nothing else uses their values), so they are $m$ fixed distinct permutations and the
same bound holds. $\square$ The keys must really be uniform on 64 bits for this bound, as in the notebook's
`zobrist_table`. With 63-bit keys (for example `integers(0, 2**63 - 1)`) the bound would only be $m(m-1)/2^{64}$.

### Exercise 3 ★★ — Transition vs residence frequency

The transition memory $\tau_i$ counts how often facility $i$ was moved. The penalty on non-improving moves becomes
$\lambda \bar\delta (\tau_r + \tau_s)/k$ instead of $\lambda \bar\delta (h_{r\pi_s} + h_{s\pi_r})/k$. The setup
matches Section 6: $n = 25$, 4 instances × 5 seeds, 450 000 evaluations, strong rule, $t = 8$, $\lambda = 2$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import wilcoxon
from utils import random_qap, qap_cost

def qap_delta_matrix(p, F, D):
    Dp = D[np.ix_(p, p)]; M = F @ Dp; dg = np.diag(M)
    Delta = 2.0 * (M + M.T - dg[:, None] - dg[None, :] + 2.0 * F * Dp)
    np.fill_diagonal(Delta, 0.0); return Delta

def qap_update_after_swap(Delta, q, r, s, F, D):     # Taillard's O(n^2) update; q = permutation after swap
    a = F[r] - F[s]; b = D[q[s], q] - D[q[r], q]
    new = Delta + 2.0 * np.subtract.outer(a, a) * np.subtract.outer(b, b)
    Dq = D[np.ix_(q, q)]; diagM = np.sum(F * Dq, axis=0)
    for i in (r, s):
        row = 2.0 * (F[:, i] @ Dq + Dq[:, i] @ F - diagM[i] - diagM + 2.0 * F[i] * Dq[i]); row[i] = 0.0
        new[i, :] = row; new[:, i] = row
    return new

def ts_freq(F, D, max_evals, rng, kind, lam=2.0, tenure=8):
    """Notebook 2's TS (strong rule, t = 8) with a residence or transition frequency penalty on non-improving moves."""
    n = len(F); iu, ju = np.triu_indices(n, 1)
    p = rng.permutation(n); cost = qap_cost(p, F, D); best = cost
    Delta = qap_delta_matrix(p, F, D); dbar = np.mean(np.abs(Delta[iu, ju]))
    tabu_until = np.zeros((n, n), dtype=np.int64); h = np.zeros((n, n)); tau = np.zeros(n); evals, k = 0, 0
    while evals + len(iu) <= max_evals:
        k += 1; evals += len(iu); d = Delta[iu, ju]
        tabu = (tabu_until[iu, p[ju]] >= k) | (tabu_until[ju, p[iu]] >= k)
        adm = ~tabu | (cost + d < best - 1e-9)
        pen = (h[iu, p[ju]] + h[ju, p[iu]]) if kind == "residence" else (tau[iu] + tau[ju])
        score = np.where(d >= 0, d + lam * dbar * pen / k, d)
        if not adm.any(): adm[:] = True
        j = int(np.argmin(np.where(adm, score, np.inf))); r, s = int(iu[j]), int(ju[j])
        tabu_until[r, p[r]] = tabu_until[s, p[s]] = k + tenure
        cost += d[j]; q = p.copy(); q[[r, s]] = q[[s, r]]
        Delta = qap_update_after_swap(Delta, q, r, s, F, D); p = q
        tau[[r, s]] += 1; h[np.arange(n), p] += 1; best = min(best, cost)
    P = h / h.sum(); ent = -(P[P > 0] * np.log(P[P > 0])).sum() / np.log(n * n)
    return best, ent

insts = [random_qap(25, np.random.default_rng(300 + i)) for i in range(4)]       # Notebook 2, Section 6
kinds = ("residence", "transition")
out = np.array([[[ts_freq(F, D, 450_000, np.random.default_rng(10*i + sd), kind) for sd in range(5)]
                 for i, (F, D) in enumerate(insts)] for kind in kinds])          # (kind, inst, seed, [best, ent])
B, E = out[..., 0], out[..., 1].reshape(2, -1)
bk = B.min(axis=(0, 2))
G = (100 * (B - bk[None, :, None]) / bk[None, :, None]).reshape(2, -1)
for kind, g, e in zip(kinds, G, E):
    print(f"{kind:>10}: median gap {np.median(g):.2f}%  IQR [{np.percentile(g, 25):.2f}, {np.percentile(g, 75):.2f}]"
          f"  entropy {e.mean():.3f} ± {e.std(ddof=1) / np.sqrt(len(e)):.3f}")
print(f"Wilcoxon (20 pairs): p = {wilcoxon(G[0], G[1]).pvalue:.3f}")
```

| memory | median gap | IQR | residence entropy (mean ± s.e.) |
|---|---|---|---|
| residence | **0.25%** | [0.01, 0.59] | **0.745 ± 0.006** |
| transition | 0.42% | [0.11, 0.88] | 0.683 ± 0.007 |

Wilcoxon over the 20 pairs gives $p = 0.016$. Residence frequency diversifies more and gives better final quality.
The reason is that the transition penalty is blind to *where* a facility goes. It discourages moving busy facilities
at all, which slows the search down, but it does not push them towards unused locations.

### Exercise 4 ★★ — Penalty-based oscillation

Let $g_\lambda(x) = v^\top x - \lambda \max(0, w^\top x - C)$ with positive integer $w_i$ and integer $C$.

**Claim.** If $\lambda \gt \max_i v_i$, every local maximum of $g_\lambda$ under single flips is feasible.
*Proof.* Let $x$ be infeasible, with excess $e = w^\top x - C \ge 1$ (an integer). Since $e \gt 0$, $x$ contains an item
$i$. Dropping $i$ gives the excess $\max(0, e - w_i)$, so

$$
g_\lambda(x - e_i) - g_\lambda(x) = -v_i + \lambda \bigl(e - \max(0, e - w_i)\bigr) = -v_i + \lambda \min(e, w_i) \ge -v_i + \lambda \gt 0,
$$

because $\min(e, w_i) \ge 1$. So $x$ is not a local maximum. $\square$

**$\lambda \gt \max_i v_i/w_i$ is not enough.** Take $v = (10, 1)$, $w = (10, 1)$, $C = 9$ and $\lambda = 1.5$, which is
$\gt \max v_i/w_i = 1$. The solution $x = (1,0)$ is infeasible with $g = 10 - 1.5 = 8.5$. Its neighbours are $(0,0)$ with
$g = 0$ and $(1,1)$ with $g = 11 - 3 = 8$. So $x$ is an infeasible local maximum, which the script confirms.
Per-unit-weight arguments fail because dropping an item removes at most $e$ units of excess, not $w_i$ units.

**Adaptive-$\lambda$ tabu search.** Maximise $g_\lambda$ with flip moves, a random tenure in $[3,10]$, and aspiration
for a flip that gives a new best feasible value. $\lambda$ starts at the mean of $v_i/w_i$. It is multiplied by 1.5 when
the last 5 iterates were all infeasible and divided by 1.5 when they were all feasible. The search starts from the
greedy-by-ratio solution, whose $n$ evaluations are charged to the budget.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import random_knapsack, knapsack_dp

# two-item counterexample: lambda = 1.5 > max v_i / w_i = 1, yet (1,0) is an infeasible local maximum
v2, w2, C2, lam2 = np.array([10., 1.]), np.array([10., 1.]), 9, 1.5
g = lambda x: v2 @ x - lam2 * max(0.0, w2 @ x - C2)
x = np.array([1., 0.])
print("g(1,0) =", g(x), " neighbours:", g(np.array([0., 0.])), g(np.array([1., 1.])), " feasible:", w2 @ x <= C2)

def knapsack_so(v, w, C, max_evals, rng, tenure=(3, 10)):
    """Notebook 2's phase-based strategic oscillation with depth 0."""
    n = len(v); ratio = v / w; x = np.zeros(n, bool); val = load = best = 0.0; best_x = x.copy()
    tabu_until = np.zeros(n, dtype=np.int64); phase, k, evals = +1, 0, 0
    while evals + n <= max_evals:
        k += 1; evals += n
        sign = np.where(x, -1.0, 1.0); nval, nload = val + sign * v, load + sign * w
        asp = (nload <= C) & (nval > best); free = tabu_until < k
        score = np.where(~x & (free | asp), ratio, -np.inf) if phase > 0 else np.where(x & (free | asp), -ratio, -np.inf)
        if asp.any(): score = np.where(asp, np.inf, score)
        if not np.any(score > -np.inf):
            evals -= n; k -= 1
            if phase > 0: phase = -1; continue
            break
        j = int(rng.choice(np.flatnonzero(score == score.max())))
        x[j] = ~x[j]; val, load = float(nval[j]), float(nload[j])
        tabu_until[j] = k + int(rng.integers(tenure[0], tenure[1] + 1))
        if load <= C and val > best: best, best_x = val, x.copy()
        if phase > 0 and load > C: phase = -1
        elif phase < 0 and load <= C: phase = +1
    return best, best_x

def knapsack_penalty_ts(v, w, C, max_evals, rng, tenure=(3, 10), K=5, up=1.5):
    n = len(v); x = np.zeros(n, bool)
    for i in np.argsort(-v / w):                         # greedy-by-ratio start (n evaluations)
        if w[x].sum() + w[i] <= C: x[i] = True
    val, load = float(v[x].sum()), float(w[x].sum()); lam = (v / w).mean()
    best, best_x = val, x.copy(); tabu_until = np.zeros(n, dtype=np.int64); hist = []; k, evals = 0, n
    while evals + n <= max_evals:
        k += 1; evals += n
        sign = np.where(x, -1.0, 1.0); nval, nload = val + sign * v, load + sign * w
        gval = nval - lam * np.maximum(0.0, nload - C)
        asp = (nload <= C) & (nval > best); adm = (tabu_until < k) | asp
        j = int(rng.choice(np.flatnonzero(gval == np.where(adm, gval, -np.inf).max())))
        x[j] = ~x[j]; val, load = float(nval[j]), float(nload[j])
        tabu_until[j] = k + int(rng.integers(tenure[0], tenure[1] + 1))
        if load <= C and val > best: best, best_x = val, x.copy()
        hist.append(load <= C)
        if len(hist) >= K:
            if not any(hist[-K:]): lam *= up            # K infeasible iterates in a row: raise the penalty
            elif all(hist[-K:]): lam /= up              # K feasible iterates in a row: lower it
    return best, best_x

for corr in ("uncorrelated", "weak", "strong"):
    res = {"oscillation": [], "penalty": []}
    for inst in range(8):
        v, w, C = random_knapsack(100, np.random.default_rng(500 + inst), correlation=corr)
        opt, _ = knapsack_dp(v, w, C)
        for sd in range(5):
            for name, fn in (("oscillation", knapsack_so), ("penalty", knapsack_penalty_ts)):
                b, xb = fn(v, w, C, 20_000, np.random.default_rng(sd))
                assert w[xb].sum() <= C and abs(v[xb].sum() - b) < 1e-9 and b <= opt + 1e-9
                res[name].append(100 * (opt - b) / opt)
    print(corr, {k: (int(np.sum(np.array(g) < 1e-9)), round(float(np.mean(g)), 3)) for k, g in res.items()})
```

This uses the same 24 instances × 5 seeds and the same 20 000-evaluation budget as the notebook. Every reported
solution is checked for feasibility and against the DP optimum. The block first confirms the two-item counterexample
($g = 8.5$ at $(1,0)$ against 0 and 8 at its neighbours).

| method | exact hits (uncorr., weak, strong) of 40 | mean gap |
|---|---|---|
| phase-based oscillation, depth 0 | 21, **24**, **40** | 0.039%, 0.021%, 0.000% |
| adaptive penalty | **23**, 2, 33 | 0.059%, 0.132%, 0.034% |

The penalty version matches phase-based oscillation on uncorrelated instances, but it is clearly worse on correlated
ones. There the ratios $v_i/w_i$ are nearly equal, and $g_\lambda$ ranks flips by $v_i - \lambda \cdot$(excess added).
That favours large items rather than a fine exchange at the boundary. Starting from the greedy-by-ratio solution
matters: with only 200 iterations, a search started from the empty knapsack would spend a large part of its budget
just filling it.

### Exercise 5 ★★ — Mixed and truncated path relinking

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import friedmanchisquare
from utils import random_qap, qap_cost

def qap_delta_matrix(p, F, D):
    Dp = D[np.ix_(p, p)]; M = F @ Dp; dg = np.diag(M)
    Delta = 2.0 * (M + M.T - dg[:, None] - dg[None, :] + 2.0 * F * Dp)
    np.fill_diagonal(Delta, 0.0); return Delta

def qap_update_after_swap(Delta, q, r, s, F, D):     # Taillard's O(n^2) update; q = permutation after swap
    a = F[r] - F[s]; b = D[q[s], q] - D[q[r], q]
    new = Delta + 2.0 * np.subtract.outer(a, a) * np.subtract.outer(b, b)
    Dq = D[np.ix_(q, q)]; diagM = np.sum(F * Dq, axis=0)
    for i in (r, s):
        row = 2.0 * (F[:, i] @ Dq + Dq[:, i] @ F - diagM[i] - diagM + 2.0 * F[i] * Dq[i]); row[i] = 0.0
        new[i, :] = row; new[:, i] = row
    return new

def qap_delta(p, r, s, F, D):
    return 2.0 * ((F[:, r] - F[:, s]) @ (D[p, p[s]] - D[p, p[r]]) + 2.0 * F[r, s] * D[p[r], p[s]])

def ts_qap(F, D, max_evals, rng, tenure=8, p0=None):
    """Notebook 2's basic TS: strong rule, fixed tenure, aspiration by objective."""
    n = len(F); iu, ju = np.triu_indices(n, 1); p = rng.permutation(n) if p0 is None else np.array(p0)
    cost = qap_cost(p, F, D); best, best_p = cost, p.copy(); Delta = qap_delta_matrix(p, F, D)
    tabu_until = np.zeros((n, n), dtype=np.int64); evals, k = 0, 0
    while evals + len(iu) <= max_evals:
        k += 1; evals += len(iu); d = Delta[iu, ju]
        adm = ~((tabu_until[iu, p[ju]] >= k) | (tabu_until[ju, p[iu]] >= k)) | (cost + d < best - 1e-9)
        if not adm.any(): adm[:] = True
        j = int(np.argmin(np.where(adm, d, np.inf))); r, s = int(iu[j]), int(ju[j])
        tabu_until[r, p[r]] = tabu_until[s, p[s]] = k + tenure
        cost += d[j]; q = p.copy(); q[[r, s]] = q[[s, r]]
        Delta = qap_update_after_swap(Delta, q, r, s, F, D); p = q
        if cost < best - 1e-9: best, best_p = cost, p.copy()
    return dict(best=float(best), perm=best_p, evals=evals)

first_step = []                                   # diagnostic: step index of the best intermediate
def relink_general(xI, xG, F, D, mode="greedy"):
    """greedy: walk xI -> xG; mixed: alternate steps from both ends; truncated: stop after ceil(m/2) steps."""
    a, b = np.array(xI), np.array(xG)
    ca, cb = qap_cost(a, F, D), qap_cost(b, F, D)
    m0 = int(np.sum(a != b)); best_mid, best_c, best_step, evals, step = None, np.inf, None, 0, 0
    while np.any(a != b):
        if mode == "truncated" and step >= int(np.ceil(m0 / 2)): break
        move_a = mode != "mixed" or step % 2 == 0
        x, tgt = (a, b) if move_a else (b, a)
        pos = np.empty(len(x), dtype=int); pos[x] = np.arange(len(x))
        cand = [(qap_delta(x, i, pos[tgt[i]], F, D), i, pos[tgt[i]]) for i in np.flatnonzero(x != tgt)]
        evals += len(cand); dl, i, j = min(cand); x[[i, j]] = x[[j, i]]
        if move_a: ca += dl
        else: cb += dl
        step += 1; c = ca if move_a else cb
        if np.any(a != b) and c < best_c: best_mid, best_c, best_step = x.copy(), c, step
    if best_step is not None: first_step.append(best_step == 1)
    return best_mid, best_c, evals

def ts_with_path_relinking(F, D, max_evals, rng, mode, segment=60_000, n_elite=5):
    """Notebook 2's driver: TS segments, elite pool, restart from the best relinked intermediate."""
    elite, evals, p0, best, best_p = [], 0, None, np.inf, None
    while max_evals - evals >= segment:
        res = ts_qap(F, D, segment, rng, p0=p0); evals += res["evals"]
        if res["best"] < best: best, best_p = res["best"], res["perm"].copy()
        if not any(np.array_equal(res["perm"], e) for _, e in elite):
            elite = sorted(elite + [(res["best"], res["perm"].copy())], key=lambda t: t[0])[:n_elite]
        if len(elite) >= 2:
            mid, mc, e = relink_general(best_p, elite[int(rng.integers(1, len(elite)))][1], F, D, mode)
            evals += e; p0 = mid
            if mid is not None and mc < best: best, best_p = mc, mid.copy()
        else:
            p0 = None
    if max_evals - evals > len(F) ** 2:
        res = ts_qap(F, D, max_evals - evals, rng, p0=p0); best = min(best, res["best"])
    return best

insts = [random_qap(25, np.random.default_rng(300 + i)) for i in range(4)]       # Notebook 2, Section 6
modes = ("greedy", "mixed", "truncated")
Q = np.array([[[ts_with_path_relinking(F, D, 450_000, np.random.default_rng(10*i + sd), mode) for sd in range(5)]
               for i, (F, D) in enumerate(insts)] for mode in modes])
bk = Q.min(axis=(0, 2))
G = (100 * (Q - bk[None, :, None]) / bk[None, :, None]).reshape(3, -1)
for mode, g in zip(modes, G):
    print(f"{mode:>9}: median gap {np.median(g):.2f}%  IQR [{np.percentile(g, 25):.2f}, {np.percentile(g, 75):.2f}]")
print("identical in all runs:", bool(np.all(Q[0] == Q[1]) and np.all(Q[0] == Q[2])),
      "| runs where the variants differ:", int(np.sum((Q[0] != Q[1]) | (Q[0] != Q[2]))))
if not np.all(G == G[0]): print(f"Friedman p = {friedmanchisquare(*G).pvalue:.2f}")
print(f"best intermediate = first step in {sum(first_step)} of {len(first_step)} relinkings")
```

The block copies the notebook's `ts_with_path_relinking` driver (TS segments of 60 000 evaluations, a 5-solution elite
pool, restart from the best relinked intermediate) and swaps in each relinking variant; the relinking evaluations are
charged to the budget. With the Section 6 setup ($n = 25$, 4 × 5 runs, 450 000 evaluations), **the three variants
produced identical results in 19 of the 20 runs**. All three have median gap 0.26% and IQR [0.04, 0.56], and a Friedman
test gives $p = 0.37$. The diagnostic explains why. In 343 of the 357 relinkings of all three variants (96%), the best
intermediate solution was the *first* step from the initiating elite, and all three variants take the same greedy
first step. Between two good QAP local optima, the path climbs
immediately. The only useful restart point is therefore a swap-neighbour of the elite, which the next TS segment would
explore anyway. Relinking pays off when elites share structure that intermediate points can combine. That calls for a
local search applied *along* the path (for example, a short TS from the best $q$ intermediates), not just an
evaluation of the path.

### Exercise 6 ★★★ — Reactive TS for feature selection

The data have $d = 30$ standard normal features. There are 3 suppressor pairs $(0,1), (2,3), (4,5)$, where the second
member is the first plus noise with s.d. 0.15 and the pair enters the target with coefficients $+2, -2$. There are also
three plain signals (features 6, 7, 8 with coefficients 1, 0.6 and 0.4), and the noise has s.d. 0.5. There are 200
training and 200 validation rows. The objective is $J(S)$, the validation MSE of OLS on $S$
plus $0.02\vert S \vert$. Moves are bit flips with aspiration by objective. The Zobrist keys $z_{j,b}$ are 64-bit, and a
flip of bit $j$ updates the hash by $h \leftarrow h \oplus z_{j,S_j} \oplus z_{j,1-S_j}$. Reactive TS uses the
Battiti–Tecchiolli rules ($\times 1.1$ on a repetition with gap $\lt 2(d-1)$, $\times 0.9$ after a quiet period longer
than the moving-average gap), without the escape phase. The comparison is 60 iterations = 1 801 evaluations per run,
20 datasets and seeds, against a fixed tenure $t = 3$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import wilcoxon

d = 30
def make_data(rng, n=400):
    Z = rng.normal(size=(n, d))
    for a in (0, 2, 4):                                   # suppressor pairs (0,1), (2,3), (4,5)
        Z[:, a + 1] = Z[:, a] + 0.15 * rng.normal(size=n)
    y = sum(2.0 * (Z[:, a] - Z[:, a + 1]) for a in (0, 2, 4)) + 1.0 * Z[:, 6] + 0.6 * Z[:, 7] + 0.4 * Z[:, 8]
    y = y + 0.5 * rng.normal(size=n)
    return Z[:200], y[:200], Z[200:], y[200:]

def J(S, data, lam=0.02):                                 # validation MSE of OLS on S + size penalty
    Xtr, ytr, Xva, yva = data; idx = np.flatnonzero(S)
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(ytr)), Xtr[:, idx]]), ytr, rcond=None)[0]
    return float(np.mean((yva - np.column_stack([np.ones(len(yva)), Xva[:, idx]]) @ beta) ** 2) + lam * len(idx))

def ts_fs(data, rng, iters, reactive, tenure=3):
    Z = rng.integers(0, 2**64, size=(d, 2), dtype=np.uint64)       # Zobrist keys z[j, bit value]
    S = rng.random(d) < 0.3; h = 0
    for j in range(d): h ^= int(Z[j, int(S[j])])
    cur = J(S, data); best = cur; tabu_until = np.zeros(d, int); evals = 1
    T, avg, t_change, H, Ts, reps = 1.0, 1.0, 0, {}, [], 0
    for k in range(1, iters + 1):
        vals = np.array([J(np.where(np.arange(d) == j, ~S, S), data) for j in range(d)]); evals += d
        adm = (tabu_until < k) | (vals < best - 1e-12)
        if not adm.any(): adm[:] = True
        j = int(np.argmin(np.where(adm, vals, np.inf)))
        h ^= int(Z[j, int(S[j])]) ^ int(Z[j, 1 - int(S[j])])        # O(1) hash update of the flip
        S[j] = ~S[j]; cur = vals[j]; best = min(best, cur)
        tabu_until[j] = k + (int(np.ceil(T)) if reactive else tenure)
        if reactive:                                                # Battiti-Tecchiolli rules, no escape phase
            if h in H:
                R = k - H[h]; reps += 1
                if R < 2 * (d - 1): avg = 0.1 * R + 0.9 * avg; T = min(1.1 * T, d - 2); t_change = k
            H[h] = k
            if k - t_change > avg: T = max(0.9 * T, 1.0); t_change = k
            Ts.append(T)
    return best, evals, Ts, reps

res_r, res_f, Tfinal, Tmax, reps = [], [], [], [], []
for i in range(20):
    data = make_data(np.random.default_rng(900 + i))
    b, ev, Ts, rp = ts_fs(data, np.random.default_rng(i), 60, True)
    res_r.append(b); Tfinal.append(Ts[-1]); Tmax.append(max(Ts)); reps.append(rp)
    res_f.append(ts_fs(data, np.random.default_rng(i), 60, False)[0])
res_r, res_f = np.array(res_r), np.array(res_f)
print(f"evaluations per run: {ev}")
print(f"reactive better on {np.sum(res_r < res_f - 1e-12)}, fixed better on {np.sum(res_f < res_r - 1e-12)}, "
      f"ties {np.sum(np.abs(res_r - res_f) <= 1e-12)}")
diff = (res_r - res_f)[np.abs(res_r - res_f) > 1e-12]        # drop tied pairs (Wilcoxon's convention)
if len(diff): print(f"exact Wilcoxon on the {len(diff)} untied pairs: p = {wilcoxon(diff, method='exact').pvalue:.4f}")
print(f"repetitions per run (median) {np.median(reps):.0f}; T final median {np.median(Tfinal):.1f}, "
      f"T max median {np.median(Tmax):.1f}")
```

Results: reactive TS was better on 4 datasets, fixed-tenure TS on 0, and 16 were ties. The exact Wilcoxon test on the
4 untied pairs gives $p = 0.125$, which is the smallest $p$ attainable with 4 pairs, so the test cannot reach
significance even though every untied pair favours reactive TS. More datasets, or a harder objective with fewer ties,
would be needed for a significant result. **How $T$ evolves:** it starts at 1. The median number of detected
repetitions was 26 per run, and they push $T$ up to a median maximum of 9.2, about $d/3$. $T$ hardly falls back in the
quiet phases (median final value 8.7), because repetitions keep occurring while the search wanders around the
suppressor-pair plateau. The fixed tenure $t = 3$ is too short for $d = 30$: the search re-enters recently left
subsets. Where the two methods differ, reactive TS wins every time.

---

## Notebook 3 — ILS, VNS and GRASP

### Exercise 1 ★ — The double bridge as two bridges

Write the tour as four segments $A B C D$ with first and last cities $s_X$ and $e_X$. The tour edges between segments
are $e_A s_B$, $e_B s_C$, $e_C s_D$ and $e_D s_A$.

*Bridge 1.* Remove $e_A s_B$ and $e_C s_D$, and add $e_A s_D$ and $e_C s_B$ without reversing anything. The result is
two cycles: $A \to D \to A$ (via $e_A s_D$ and the old edge $e_D s_A$) and $B \to C \to B$ (via the old edge $e_B s_C$
and the new edge $e_C s_B$). This is the "illegal" 2-change. The legal reconnection $e_A e_C$, $s_B s_D$ would reverse
$B C$.

*Bridge 2.* Remove $e_D s_A$ from the first cycle and $e_B s_C$ from the second, and add $e_D s_C$ and $e_B s_A$. The
two cycles merge into $A \to D \to C \to B \to A$, again without reversal. Taken alone (applied to the original
tour), this second 2-change also splits it: $A B$ closes via $e_B s_A$, and $C D$ closes via $e_D s_C$.

Together the two bridges remove $e_A s_B$, $e_B s_C$, $e_C s_D$ and $e_D s_A$ and produce exactly $A D C B$. $\square$
Notebook 3 (Section 3) also verifies the exact edge counts. $A D C B$ changes $4 - s$ edges, where $s$ counts
cyclically adjacent singleton segments. $A C B D$ changes $3 - [\vert B \vert = \vert C \vert = 1]$ edges, because it
keeps $e_D s_A$. It is therefore a 3-opt move and not a double bridge, although many code snippets use it under that
name.

### Exercise 2 ★ — GRASP at the extremes of α

*$\alpha = 1$.* The RCL condition $c(j) \le c_{min} + (c_{max} - c_{min}) = c_{max}$ holds for every unvisited city, so
each step draws uniformly from the $r$ remaining cities. Given the start city $s$, a particular ordering of the other
$n - 1$ cities has probability $\prod_{r=1}^{n-1} 1/r = 1/(n-1)!$. All $(n-1)!$ directed Hamiltonian cycles starting at
$s$ are therefore equally likely. Because $s$ itself is uniform, so is the tour as a sequence.

*$\alpha = 0$ without ties.* The RCL is $\lbrace \arg\min_j c(j) \rbrace$, a single city, so the construction is a
deterministic function of the start city (nearest-neighbour tour). With $n$ possible starts there are at most $n$
distinct tours, and possibly fewer as undirected cycles. All diversity must then come from the start city.

### Exercise 3 ★★ — Don't-look bits and neighbour lists

This is a first-improvement 2-opt with 10-nearest-neighbour lists and a queue of active cities (don't-look bits off).
The search stops scanning a neighbour list once $D_{ac} \ge D_{ab}$, since no gain is possible beyond that point. After a
double bridge, only the 8 endpoints of the changed edges are reactivated. Each examined $(a, c)$ pair counts as one
evaluation.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import random_euclidean_tsp, tour_length, best_so_far_on_grid

class Counter:
    def __init__(self): self.evals = 0

def two_opt_mask(n):
    I, J = np.indices((n, n)); m = J >= I + 2; m[0, n - 1] = False; return m

def or_opt_deltas(t, D, L):
    n = len(t); pos = np.arange(n)
    prev, first, last, nxt = t[(pos - 1) % n], t, t[(pos + L - 1) % n], t[(pos + L) % n]
    gain = D[prev, first] + D[last, nxt] - D[prev, nxt]; tj, tj1 = t, np.roll(t, -1); base = -D[tj, tj1]
    fwd = D[tj[None, :], first[:, None]] + D[last[:, None], tj1[None, :]] + base[None, :] - gain[:, None]
    rev = D[tj[None, :], last[:, None]] + D[first[:, None], tj1[None, :]] + base[None, :] - gain[:, None]
    return fwd, rev, ((pos[None, :] - (pos[:, None] - 1)) % n) > L

def apply_or_opt(t, i, j, L, rev):
    n = len(t); idx = (i + np.arange(L)) % n; seg = t[idx][::-1] if rev else t[idx]
    keep = np.ones(n, bool); keep[idx] = False; rest = t[keep]; k = int(np.flatnonzero(rest == t[j])[0])
    return np.concatenate([rest[:k + 1], seg, rest[k + 1:]])

def vnd(t, D, cnt, mask, tol=1e-10):
    """Notebook 3's best-improvement VND: N1 = 2-opt, N2 = Or-opt (L = 1, 2, 3, both orientations)."""
    t = np.array(t); length = tour_length(t, D); n_valid = int(mask.sum())
    while True:
        a, b = t, np.roll(t, -1); dab = D[a, b]
        M = np.where(mask, D[a[:, None], a[None, :]] + D[b[:, None], b[None, :]] - dab[:, None] - dab[None, :], np.inf)
        cnt.evals += n_valid; k = int(np.argmin(M))
        if M.flat[k] < -tol:
            i, j = divmod(k, len(t)); t = t.copy(); t[i + 1:j + 1] = t[i + 1:j + 1][::-1]; length += M.flat[k]
            continue
        best = (0.0, None)
        for L in (1, 2, 3):
            fwd, rev, valid = or_opt_deltas(t, D, L); cnt.evals += 2 * int(valid.sum())
            for rv, A in ((False, fwd), (True, rev)):
                A = np.where(valid, A, np.inf); kk = int(np.argmin(A))
                if A.flat[kk] < best[0] - tol: best = (A.flat[kk], (kk // len(t), kk % len(t), L, rv))
        if best[1] is None: return t, length
        t = apply_or_opt(t, *best[1]); length += best[0]

def grasp_construct(D, alpha, rng, cnt):
    n = len(D); tour = [int(rng.integers(n))]; unvisited = np.ones(n, bool); unvisited[tour[0]] = False
    for _ in range(n - 1):
        cand = np.flatnonzero(unvisited); c = D[tour[-1], cand]; cnt.evals += len(cand)
        nxt = int(rng.choice(cand[c <= c.min() + alpha * (c.max() - c.min()) + 1e-12]))
        tour.append(nxt); unvisited[nxt] = False
    return np.array(tour)

def at_budget(ev, bs, budget):          # best-so-far after exactly `budget` evaluations
    return float(best_so_far_on_grid(np.asarray(ev), np.asarray(bs), np.array([budget]))[0])

def double_bridge(t, rng):
    a, b, c = np.sort(rng.choice(np.arange(1, len(t)), 3, replace=False))
    return np.concatenate([t[:a], t[c:], t[b:c], t[a:b]]), (a, b, c)

def ils_vnd(D, budget, rng):                     # Notebook 3's ILS: k = 1, 'better' acceptance, full VND
    n = len(D); cnt = Counter(); mask = two_opt_mask(n)
    x, fx = vnd(rng.permutation(n), D, cnt, mask); ev, bs, per = [cnt.evals], [fx], []
    while cnt.evals < budget:
        e0 = cnt.evals; y, fy = vnd(double_bridge(x, rng)[0], D, cnt, mask); per.append(cnt.evals - e0)
        if fy < fx - 1e-12: x, fx = y, fy
        ev.append(cnt.evals); bs.append(fx)
    return at_budget(ev, bs, budget), np.median(per), len(per)

def two_opt_dlb(t, D, nn, cnt, active):
    """First-improvement 2-opt with neighbour lists and don't-look bits (queue of active cities)."""
    n = len(t); t = np.array(t); pos = np.empty(n, int); pos[t] = np.arange(n)
    queue = list(active); inq = np.zeros(n, bool); inq[queue] = True
    while queue:
        a = queue.pop(); inq[a] = False
        for direction in (1, -1):
            b = t[(pos[a] + direction) % n]; dab = D[a, b]; moved = False
            for c in nn[a]:
                if D[a, c] >= dab: break                        # no gain possible beyond this neighbour
                cnt.evals += 1
                d_ = t[(pos[c] + direction) % n]
                if D[a, c] + D[b, d_] - dab - D[c, d_] < -1e-12:
                    r = np.roll(t, -pos[a]); jc = int(np.flatnonzero(r == c)[0])
                    if direction == 1: r[1:jc + 1] = r[1:jc + 1][::-1]      # a b..c d -> a c..b d
                    else: r[jc:] = r[jc:][::-1]                               # a..d c..b -> a..d b..c
                    t = r; pos[t] = np.arange(n)
                    for v in (a, b, c, d_):
                        if not inq[v]: queue.append(v); inq[v] = True
                    moved = True; break
            if moved: break
    return t

def ils_dlb(D, budget, rng):
    n = len(D); nn = np.argsort(D, axis=1)[:, 1:11]; cnt = Counter()
    x = two_opt_dlb(rng.permutation(n), D, nn, cnt, range(n)); fx = tour_length(x, D); ev, bs, per = [cnt.evals], [fx], []
    while cnt.evals < budget:
        y0, (a, b, c) = double_bridge(x, rng)
        ends = {x[a-1], x[a], x[b-1], x[b], x[c-1], x[c], x[-1], x[0]}      # endpoints of the changed edges
        e0 = cnt.evals; y = two_opt_dlb(y0, D, nn, cnt, list(ends)); fy = tour_length(y, D); per.append(cnt.evals - e0)
        if fy < fx - 1e-12: x, fx = y, fy
        ev.append(cnt.evals); bs.append(fx)
    return at_budget(ev, bs, budget), np.median(per), len(per)

# is the DLB result a true 2-opt local optimum? (only guaranteed w.r.t. the moves the DLB queue re-examines)
perm_ok, residual = True, []
for sd in range(20):
    _, D60 = random_euclidean_tsp(60, np.random.default_rng(sd))
    t = two_opt_dlb(np.random.default_rng(0).permutation(60), D60, np.argsort(D60, 1)[:, 1:11], Counter(), range(60))
    perm_ok &= sorted(t) == list(range(60))
    a, b = t, np.roll(t, -1); dab = D60[a, b]
    M = D60[a[:, None], a[None, :]] + D60[b[:, None], b[None, :]] - dab[:, None] - dab[None, :]
    residual.append(np.where(two_opt_mask(60), M, np.inf).min())
residual = np.array(residual)
print(f"permutations: {perm_ok}; improving 2-opt move left on {np.sum(residual < -1e-9)}/20 instances "
      f"(most negative full 2-opt delta {residual.min():+.4f})")

inst = [random_euclidean_tsp(100, np.random.default_rng(1000 + i))[1] for i in range(2)]     # Notebook 3 instances
out = {}
for name, fn, budget in (("VND, 1e7", ils_vnd, 10**7), ("DLB, 1e6", ils_dlb, 10**6)):
    R = np.array([fn(D, budget, np.random.default_rng(sd)) for D in inst for sd in range(3)]); out[name] = R[:, 0]
    print(f"{name}: evals/iteration (median) {np.median(R[:, 1]):.0f}, iterations {np.mean(R[:, 2]):.0f}, "
          f"mean best {R[:, 0].mean():.4f}, per-run {np.round(R[:, 0], 4).tolist()}")
print(f"DLB (1e6) shorter than VND (1e7) on {np.sum(out['DLB, 1e6'] < out['VND, 1e7'])}/6 runs")
```

Correctness check: on 20 random 60-city instances the DLB output is always a permutation. It is, however, a true
2-opt local optimum on only 19 of them: on one instance an improving 2-opt move is left (full delta $-0.0222$). This is
expected, and it is not caused by the 10-NN truncation. Don't-look bits are a heuristic. A 2-opt move on edges
$(a,b)$ and $(c,d)$ reverses the segment between them, and this flips which reconnection of *other* pairs of edges is
legal. A move that becomes improving this way may have no endpoint among the reactivated cities.

The comparison uses the notebook's two $n = 100$ instances × 3 seeds with $k = 1$ and *better* acceptance:

| ILS engine | budget | evaluations per ILS iteration (median) | iterations | mean best length |
|---|---|---|---|---|
| notebook VND (full 2-opt + Or-opt) | $10^7$ | 82 450 | 108 | 7.8235 |
| DLB 2-opt, 10-NN lists | $10^6$ | **245** | 3 911 | **7.6149** |

A kick costs about **340 times fewer evaluations**. Even with a budget 10 times smaller, the DLB version is better on
6 of 6 runs, because it performs 36 times more kicks. At the *equal* budget of $10^7$ it can only do better still: a
run with budget $10^7$ performs exactly the same first $10^6$ evaluations (same seed), and the best-so-far length never
increases. The small loss of local optimality does not matter here. Re-running the whole VND after a kick that changes 4 edges wastes
almost all of its evaluations. This is why every serious ILS for the TSP (for example chained Lin–Kernighan) uses
don't-look bits.

### Exercise 4 ★★ — GRASP with path relinking

Tours are normalised (city 0 first, and the direction with $t_1 \lt t_{n-1}$) and relinked by position swaps, as for
permutations. Each candidate step costs one evaluation. After every GRASP iteration, the new local optimum is relinked
towards a random member of a 5-tour elite pool. The best intermediate tour is polished by VND and replaces the local
optimum if it is better.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import binomtest
from utils import random_euclidean_tsp, tour_length, best_so_far_on_grid

class Counter:
    def __init__(self): self.evals = 0

def two_opt_mask(n):
    I, J = np.indices((n, n)); m = J >= I + 2; m[0, n - 1] = False; return m

def or_opt_deltas(t, D, L):
    n = len(t); pos = np.arange(n)
    prev, first, last, nxt = t[(pos - 1) % n], t, t[(pos + L - 1) % n], t[(pos + L) % n]
    gain = D[prev, first] + D[last, nxt] - D[prev, nxt]; tj, tj1 = t, np.roll(t, -1); base = -D[tj, tj1]
    fwd = D[tj[None, :], first[:, None]] + D[last[:, None], tj1[None, :]] + base[None, :] - gain[:, None]
    rev = D[tj[None, :], last[:, None]] + D[first[:, None], tj1[None, :]] + base[None, :] - gain[:, None]
    return fwd, rev, ((pos[None, :] - (pos[:, None] - 1)) % n) > L

def apply_or_opt(t, i, j, L, rev):
    n = len(t); idx = (i + np.arange(L)) % n; seg = t[idx][::-1] if rev else t[idx]
    keep = np.ones(n, bool); keep[idx] = False; rest = t[keep]; k = int(np.flatnonzero(rest == t[j])[0])
    return np.concatenate([rest[:k + 1], seg, rest[k + 1:]])

def vnd(t, D, cnt, mask, tol=1e-10):
    """Notebook 3's best-improvement VND: N1 = 2-opt, N2 = Or-opt (L = 1, 2, 3, both orientations)."""
    t = np.array(t); length = tour_length(t, D); n_valid = int(mask.sum())
    while True:
        a, b = t, np.roll(t, -1); dab = D[a, b]
        M = np.where(mask, D[a[:, None], a[None, :]] + D[b[:, None], b[None, :]] - dab[:, None] - dab[None, :], np.inf)
        cnt.evals += n_valid; k = int(np.argmin(M))
        if M.flat[k] < -tol:
            i, j = divmod(k, len(t)); t = t.copy(); t[i + 1:j + 1] = t[i + 1:j + 1][::-1]; length += M.flat[k]
            continue
        best = (0.0, None)
        for L in (1, 2, 3):
            fwd, rev, valid = or_opt_deltas(t, D, L); cnt.evals += 2 * int(valid.sum())
            for rv, A in ((False, fwd), (True, rev)):
                A = np.where(valid, A, np.inf); kk = int(np.argmin(A))
                if A.flat[kk] < best[0] - tol: best = (A.flat[kk], (kk // len(t), kk % len(t), L, rv))
        if best[1] is None: return t, length
        t = apply_or_opt(t, *best[1]); length += best[0]

def grasp_construct(D, alpha, rng, cnt):
    n = len(D); tour = [int(rng.integers(n))]; unvisited = np.ones(n, bool); unvisited[tour[0]] = False
    for _ in range(n - 1):
        cand = np.flatnonzero(unvisited); c = D[tour[-1], cand]; cnt.evals += len(cand)
        nxt = int(rng.choice(cand[c <= c.min() + alpha * (c.max() - c.min()) + 1e-12]))
        tour.append(nxt); unvisited[nxt] = False
    return np.array(tour)

def at_budget(ev, bs, budget):          # best-so-far after exactly `budget` evaluations
    return float(best_so_far_on_grid(np.asarray(ev), np.asarray(bs), np.array([budget]))[0])

def normalise(t):                                 # city 0 first, direction with t[1] < t[-1]
    t = np.roll(t, -int(np.flatnonzero(t == 0)[0]))
    return t if t[1] < t[-1] else np.concatenate([[0], t[1:][::-1]])

def tsp_relink(xI, xG, D, cnt):
    """Greedy path relinking by position swaps; returns the best intermediate tour (or None)."""
    x, gd = normalise(xI).copy(), normalise(xG)
    pos = np.empty(len(x), int); pos[x] = np.arange(len(x)); best, best_len = None, np.inf
    while True:
        diff = np.flatnonzero(x != gd)
        if len(diff) == 0: break
        cands = []
        for i in diff:
            y = x.copy(); j = pos[gd[i]]; y[[i, j]] = y[[j, i]]; cands.append((tour_length(y, D), i, j))
        cnt.evals += len(cands)
        L, i, j = min(cands); x[[i, j]] = x[[j, i]]; pos[x[i]], pos[x[j]] = i, j
        if np.any(x != gd) and L < best_len: best, best_len = x.copy(), L
    return best

def grasp(D, budget, rng, alpha=0.2, relink=False, n_elite=5):
    n = len(D); cnt = Counter(); mask = two_opt_mask(n); elite = []; best = np.inf; ev, bs = [], []
    while cnt.evals < budget:
        x, fx = vnd(grasp_construct(D, alpha, rng, cnt), D, cnt, mask)
        if relink and elite:
            mid = tsp_relink(x, elite[int(rng.integers(len(elite)))][1], D, cnt)
            if mid is not None:
                y, fy = vnd(mid, D, cnt, mask)
                if fy < fx: x, fx = y, fy
        if relink and not any(abs(fx - e[0]) < 1e-9 for e in elite):
            elite = sorted(elite + [(fx, x.copy())], key=lambda e: e[0])[:n_elite]
        best = min(best, fx); ev.append(cnt.evals); bs.append(best)
    return at_budget(ev, bs, budget)

inst = [random_euclidean_tsp(100, np.random.default_rng(1000 + i))[1] for i in range(2)]     # Notebook 3 instances
plain = np.array([grasp(D, 10**7, np.random.default_rng(sd)) for D in inst for sd in range(3)])
pr = np.array([grasp(D, 10**7, np.random.default_rng(sd), relink=True) for D in inst for sd in range(3)])
wins = int(np.sum(pr < plain))
print("plain GRASP:", np.round(plain, 4).tolist()); print("GRASP + PR: ", np.round(pr, 4).tolist())
print(f"GRASP+PR shorter on {wins}/6 runs, mean relative change {100 * np.mean(pr / plain - 1):+.2f}%, "
      f"exact sign test p = {binomtest(wins, 6).pvalue:.2f}")
```

On the notebook's two $n = 100$ instances × 3 seeds, with $\alpha = 0.2$ and $10^7$ evaluations, GRASP+PR was shorter on
**5 of 6** runs, with a mean relative change of **−0.39%**. Plain GRASP gave lengths 7.8352, 7.8787, 8.0212, 7.5759,
7.5778, 7.5977; GRASP+PR gave 7.7944, 7.8766, 8.0151, 7.5050, 7.6286, 7.4886. On the TSP, unlike the QAP of
Notebook 2 (Exercise 5), relinking helps. Two locally optimal tours share many edges, so the intermediate tours
inherit good structure, and VND can turn the best of them into a new local optimum. The exact sign test for 5 of 6
gives $p = 0.22$, so a larger study is needed to confirm the effect.

### Exercise 5 ★★ — Reactive GRASP

Following Prais & Ribeiro (2000), $\alpha$ is drawn from $\lbrace 0, 0.1, 0.2, 0.5, 1 \rbrace$ with probabilities
$p_i \propto q_i = (f^{\ast}/A_i)^{\delta}$, where $A_i$ is the mean local-optimum length obtained with $\alpha_i$. The
probabilities are refreshed every 5 iterations with $\delta = 10$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import random_euclidean_tsp, tour_length, best_so_far_on_grid

class Counter:
    def __init__(self): self.evals = 0

def two_opt_mask(n):
    I, J = np.indices((n, n)); m = J >= I + 2; m[0, n - 1] = False; return m

def or_opt_deltas(t, D, L):
    n = len(t); pos = np.arange(n)
    prev, first, last, nxt = t[(pos - 1) % n], t, t[(pos + L - 1) % n], t[(pos + L) % n]
    gain = D[prev, first] + D[last, nxt] - D[prev, nxt]; tj, tj1 = t, np.roll(t, -1); base = -D[tj, tj1]
    fwd = D[tj[None, :], first[:, None]] + D[last[:, None], tj1[None, :]] + base[None, :] - gain[:, None]
    rev = D[tj[None, :], last[:, None]] + D[first[:, None], tj1[None, :]] + base[None, :] - gain[:, None]
    return fwd, rev, ((pos[None, :] - (pos[:, None] - 1)) % n) > L

def apply_or_opt(t, i, j, L, rev):
    n = len(t); idx = (i + np.arange(L)) % n; seg = t[idx][::-1] if rev else t[idx]
    keep = np.ones(n, bool); keep[idx] = False; rest = t[keep]; k = int(np.flatnonzero(rest == t[j])[0])
    return np.concatenate([rest[:k + 1], seg, rest[k + 1:]])

def vnd(t, D, cnt, mask, tol=1e-10):
    """Notebook 3's best-improvement VND: N1 = 2-opt, N2 = Or-opt (L = 1, 2, 3, both orientations)."""
    t = np.array(t); length = tour_length(t, D); n_valid = int(mask.sum())
    while True:
        a, b = t, np.roll(t, -1); dab = D[a, b]
        M = np.where(mask, D[a[:, None], a[None, :]] + D[b[:, None], b[None, :]] - dab[:, None] - dab[None, :], np.inf)
        cnt.evals += n_valid; k = int(np.argmin(M))
        if M.flat[k] < -tol:
            i, j = divmod(k, len(t)); t = t.copy(); t[i + 1:j + 1] = t[i + 1:j + 1][::-1]; length += M.flat[k]
            continue
        best = (0.0, None)
        for L in (1, 2, 3):
            fwd, rev, valid = or_opt_deltas(t, D, L); cnt.evals += 2 * int(valid.sum())
            for rv, A in ((False, fwd), (True, rev)):
                A = np.where(valid, A, np.inf); kk = int(np.argmin(A))
                if A.flat[kk] < best[0] - tol: best = (A.flat[kk], (kk // len(t), kk % len(t), L, rv))
        if best[1] is None: return t, length
        t = apply_or_opt(t, *best[1]); length += best[0]

def grasp_construct(D, alpha, rng, cnt):
    n = len(D); tour = [int(rng.integers(n))]; unvisited = np.ones(n, bool); unvisited[tour[0]] = False
    for _ in range(n - 1):
        cand = np.flatnonzero(unvisited); c = D[tour[-1], cand]; cnt.evals += len(cand)
        nxt = int(rng.choice(cand[c <= c.min() + alpha * (c.max() - c.min()) + 1e-12]))
        tour.append(nxt); unvisited[nxt] = False
    return np.array(tour)

def at_budget(ev, bs, budget):          # best-so-far after exactly `budget` evaluations
    return float(best_so_far_on_grid(np.asarray(ev), np.asarray(bs), np.array([budget]))[0])

def grasp_fixed(D, budget, rng, alpha):
    n = len(D); cnt = Counter(); mask = two_opt_mask(n); best = np.inf; ev, bs = [], []
    while cnt.evals < budget:
        x, fx = vnd(grasp_construct(D, alpha, rng, cnt), D, cnt, mask); best = min(best, fx)
        ev.append(cnt.evals); bs.append(best)
    return at_budget(ev, bs, budget)

def reactive_grasp(D, budget, rng, alphas=(0.0, 0.1, 0.2, 0.5, 1.0), block=5, delta=10):
    """Prais & Ribeiro (2000): p_i proportional to (f*/A_i)^delta, refreshed every `block` iterations."""
    n = len(D); cnt = Counter(); mask = two_opt_mask(n); m = len(alphas)
    p = np.full(m, 1.0 / m); sums, counts = np.zeros(m), np.zeros(m); best = np.inf; ev, bs, used = [], [], []; it = 0
    while cnt.evals < budget:
        i = int(rng.choice(m, p=p)); used.append(i)
        x, fx = vnd(grasp_construct(D, alphas[i], rng, cnt), D, cnt, mask)
        sums[i] += fx; counts[i] += 1; best = min(best, fx); it += 1
        if it % block == 0:
            A = np.where(counts > 0, sums / np.maximum(counts, 1), best)
            q = (best / A) ** delta; p = q / q.sum()
        ev.append(cnt.evals); bs.append(best)
    return at_budget(ev, bs, budget), np.bincount(used, minlength=m), it

inst = [random_euclidean_tsp(100, np.random.default_rng(1000 + i))[1] for i in range(2)]     # Notebook 3 instances
fixed = np.array([grasp_fixed(D, 10**7, np.random.default_rng(sd), 0.0) for D in inst for sd in range(3)])
R = [reactive_grasp(D, 10**7, np.random.default_rng(sd)) for D in inst for sd in range(3)]
react = np.array([r[0] for r in R]); use = np.sum([r[1] for r in R], axis=0)
print(f"reactive shorter on {np.sum(react < fixed)}/6 runs; mean relative change {100 * np.mean(react / fixed - 1):+.2f}%")
print("share of iterations per alpha:", np.round(use / use.sum(), 2).tolist(), "| iterations per run:", [r[2] for r in R])
```

Against the best fixed $\alpha = 0$ (2 instances × 3 seeds, $10^7$ evaluations), reactive GRASP was shorter on only
**2 of 6** runs, with a mean relative change of **+0.35%**. The shares of iterations per $\alpha$ were
0.20, 0.21, 0.24, 0.21 and 0.13. The adaptation barely moves away from uniform, because a run has only 13–18 GRASP
iterations. With so few samples per $\alpha$ the averages $A_i$ are noisy, and they differ by only about 1% (see the
"after VND" column in Section 6), so $(f^{\ast}/A_i)^{10}$ stays almost flat. Reactive GRASP pays off when there are
hundreds of iterations and the $\alpha$ values differ clearly. Otherwise, spending budget on poor $\alpha$ values costs
more than the adaptation gains.

### Exercise 6 ★★★ — ILS for program/prompt-like sequence search

**Design.** A candidate is a sequence of $L = 30$ symbols over an alphabet of 4. It must match a hidden target, but the
score is *deceptive* in blocks of 2. A block with both symbols right costs 0, a block with none right costs 1, and a
block with exactly one right costs 2. Near-misses are therefore penalised, which mimics an evaluation in which half of
an edit pair breaks a program. The objective is to minimise the sum, which is 0 at the optimum. The local search is
best-improvement over single edits ($30 \times 3$ neighbours) on a *noisy* score, true score $+\,\sigma \varepsilon$.
The kick applies $k$ random edits, and acceptance is *better* on the noisy score. The budget is 30 000 evaluations,
with 10 seeds per setting. The true score of the final incumbent is reported.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

Lq, A, BS = 30, 4, 2
target = np.random.default_rng(123).integers(A, size=Lq)
def true_score(s):                                # deceptive blocks of 2: 0 if both right, 2 if one, 1 if none
    u = (s == target).reshape(-1, BS).sum(1)
    return float(np.sum(np.where(u == BS, 0, 1 + u)))

NB_I = np.repeat(np.arange(Lq), A)              # all single edits (position i, symbol a), in scan order
NB_A = np.tile(np.arange(A), Lq)

def local_search(s, rng, sigma, cnt):             # best improvement over single edits on the noisy score
    s = s.copy(); cur = true_score(s) + sigma * rng.normal(); cnt[0] += 1
    while True:
        keep = NB_A != s[NB_I]                    # the Lq * (A - 1) = 90 real edits
        I, Asym = NB_I[keep], NB_A[keep]
        Y = np.repeat(s[None, :], len(I), axis=0); Y[np.arange(len(I)), I] = Asym
        u = (Y == target).reshape(len(I), -1, BS).sum(2)
        v = np.where(u == BS, 0, 1 + u).sum(1) + sigma * rng.normal(size=len(I)); cnt[0] += len(I)
        j = int(np.argmin(v))
        if not v[j] < cur - 1e-12: return s, cur
        s[I[j]] = Asym[j]; cur = v[j]

def ils_seq(k, budget, rng, sigma):
    cnt = [0]; x, fx = local_search(rng.integers(A, size=Lq), rng, sigma, cnt); best = true_score(x); rev = []
    while cnt[0] < budget:
        y = x.copy(); idx = rng.choice(Lq, k, replace=False); y[idx] = rng.integers(A, size=k)   # kick: k random edits
        y, fy = local_search(y, rng, sigma, cnt); rev.append(np.array_equal(y, x))          # revert = back to x
        if fy < fx: x, fx = y, fy                                                             # 'better' on noisy score
        best = min(best, true_score(x))
    return best, np.mean(rev)

for sigma in (0.0, 0.3):
    for k in (1, 2, 4, 8, 16, 30):
        R = np.array([ils_seq(k, 30_000, np.random.default_rng(sd), sigma) for sd in range(10)])
        print(f"sigma={sigma}, k={k:>2}: hits {int(np.sum(R[:, 0] == 0)):>2}/10, revert rate {R[:, 1].mean():.2f}, "
              f"median final true score {np.median(R[:, 0]):.1f}")
```

| $k$ | 1 | 2 | 4 | 8 | 16 | 30 |
|---|---|---|---|---|---|---|
| $\sigma = 0$: hits / revert rate | 7 / 0.85 | 9 / 0.82 | **10** / 0.67 | **10** / 0.34 | 2 / 0.01 | 0 / 0.00 |
| $\sigma = 0.3$: hits / revert rate | 1 / 0.12 | 8 / 0.27 | 9 / 0.42 | **10** / 0.25 | 0 / 0.00 | 0 / 0.00 |

"Hits" is the number of the 10 runs that ended at the optimum (true score 0). The median final true scores at
$k = 16$ and $k = 30$ were 1.0 and 4.0 for $\sigma = 0$, and 1.0 and 4.5 for $\sigma = 0.3$. This reproduces the ILS
picture. Weak kicks are mostly undone: at $\sigma = 0$ and $k = 1$, 85% of kicks return to the same sequence. Kicks that
are too strong ($k \ge 16$, with $k = 30$ being random restart) destroy the solved blocks faster than the local search
can re-derive them. The best $k$ is 4–8, where about a third to two thirds of the kicks are undone. With noise, the
revert rate is no longer a clean diagnostic, because noisy acceptance lets the search drift. Very weak kicks then
suffer most, since the local search accepts noise-driven "improvements", and $k = 8$ is the most robust choice.

---

## Lab — Tabu Search for Graph Colouring and the QAP

### Exercise 1 ★ — Moves of non-conflicting vertices cannot improve

The delta of recolouring $v$ from $c(v)$ to $c'$ is $\gamma_{vc'} - \gamma_{v c(v)}$. If $v$ is not in conflict, then
$\gamma_{v c(v)} = 0$, so the delta is $\gamma_{vc'} \ge 0$. Such a move can never decrease $f$. The best improving
move, and the aspiration candidates (which need $f + \Delta \lt f^{\ast} \le f$), therefore always involve a
conflicting vertex. Restricting to conflicting vertices loses no improving move, although it does exclude "sideways"
moves of conflict-free vertices, which are pure plateau moves. $\square$

### Exercise 2 ★ — DSATUR on bipartite graphs, and a bad first-fit order

**DSATUR is exact on bipartite graphs.** Consider one connected component with bipartition $(X, Y)$. The first vertex
of the component gets colour 0. Invariant: *the coloured vertices of the component form a connected set, coloured
properly with colour $a$ on $X$ and colour $b$ on $Y$, where $\lbrace a,b \rbrace = \lbrace 0,1 \rbrace$.*
While the component is incomplete, connectivity gives some uncoloured vertex with a coloured neighbour, so the maximum
saturation is $\ge 1$. The chosen vertex $u$ therefore has a coloured neighbour, which keeps the coloured set
connected. All coloured neighbours of $u$ lie on the opposite side, so they share one colour, and the saturation of $u$
is exactly 1. $u$ receives the smallest colour not in that set, which is precisely its side's colour, so the invariant
is maintained. When a component is finished, the next vertex has saturation 0 and gets colour 0, and a new component
starts. DSATUR therefore uses at most 2 colours, which is $\chi$ for any bipartite graph with an edge. This holds for
any tie-breaking rule. $\square$ Check: DSATUR used $\le 2$ colours on 200 random bipartite graphs.

**First-fit with $\Theta(n)$ colours on a 2-colourable graph.** The crown graph has vertices
$u_1..u_m, v_1..v_m$ and edges $u_i v_j$ for $i \ne j$. Take the order $u_1, v_1, u_2, v_2, \dots$. By induction,
$u_i$ and $v_i$ both get colour $i-1$. Vertex $u_i$ is adjacent to $v_1..v_{i-1}$, which carry colours $0..i-2$, and
not to $v_i$. So first-fit uses $m = n/2$ colours, while $\chi = 2$. Run output: first-fit used 4, 8 and 16 colours for
$m = 4, 8, 16$, and DSATUR used 2 each time.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

def greedy_colouring(A, order):                  # first-fit in the given vertex order
    col = -np.ones(len(A), dtype=int)
    for v in order:
        used = set(col[A[v] & (col >= 0)].tolist()); c = 0
        while c in used: c += 1
        col[v] = c
    return col

def dsatur(A):                                   # lab's DSATUR (ties: largest degree)
    n = len(A); col = -np.ones(n, dtype=int); deg = A.sum(1); sat = [set() for _ in range(n)]
    for _ in range(n):
        un = np.flatnonzero(col < 0); s = np.array([len(sat[v]) for v in un])
        cand = un[s == s.max()]; v = cand[np.argmax(deg[cand])]; c = 0
        while c in sat[v]: c += 1
        col[v] = c
        for u in np.flatnonzero(A[v]): sat[u].add(c)
    return col

def proper(A, col): return not np.any(A & (col[:, None] == col[None, :]))

# DSATUR on 200 random bipartite graphs (random sides, random density, possibly disconnected)
g = np.random.default_rng(0); worst = 0
for _ in range(200):
    n = int(g.integers(4, 40)); side = g.random(n) < 0.5
    A = np.triu((g.random((n, n)) < g.uniform(0.05, 0.5)) & (side[:, None] != side[None, :]), 1); A = A | A.T
    col = dsatur(A); assert proper(A, col); worst = max(worst, col.max() + 1)
print(f"DSATUR on 200 random bipartite graphs: at most {worst} colours")

# crown graph: u_1..u_m, v_1..v_m, edges u_i v_j for i != j; bad order u1, v1, u2, v2, ...
for m in (4, 8, 16):
    A = np.zeros((2 * m, 2 * m), bool); A[:m, m:] = ~np.eye(m, dtype=bool); A = A | A.T
    bad = [x for i in range(m) for x in (i, m + i)]
    cf, cd = greedy_colouring(A, bad), dsatur(A); assert proper(A, cf) and proper(A, cd)
    print(f"crown graph m = {m:>2} (n = {2*m}): first-fit {cf.max() + 1} colours, DSATUR {cd.max() + 1}")
```

### Exercise 3 ★★ — PartialCol (partial legal colourings)

The search keeps a *proper* partial $k$-colouring and minimises the number of uncoloured vertices $\vert U \vert$. A move
$(v \in U, c)$ colours $v$ with $c$ and uncolours its $\gamma_{vc}$ neighbours of colour $c$, so
$\Delta\vert U \vert = \gamma_{vc} - 1$. The uncoloured neighbours $u$ get the tabu attribute $(u, c)$, with the
TabuCol-style tenure $U\lbrace 0..9 \rbrace + \lfloor 0.6\vert U \vert \rfloor$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

def gnp(n, p, rng):
    A = np.triu(rng.random((n, n)) < p, 1); return A | A.T

def tabucol(A, k, rng, max_iter, nbrs, L=10, lam=0.6):
    """The lab's TabuCol (conflict neighbourhood, gamma matrix, Galinier-Hao dynamic tenure)."""
    n = len(A); col = rng.integers(k, size=n)
    gamma = A.astype(np.int64) @ np.eye(k, dtype=np.int64)[col]
    f = int(gamma[np.arange(n), col].sum() // 2); best = f; tabu = np.zeros((n, k), dtype=np.int64)
    for it in range(1, max_iter + 1):
        if f == 0: break
        own = gamma[np.arange(n), col]; cv = np.flatnonzero(own > 0)
        dl = (gamma[cv] - own[cv][:, None]).astype(float); dl[np.arange(len(cv)), col[cv]] = np.inf
        dl = np.where((tabu[cv] < it) | (f + dl < best), dl, np.inf); m = dl.min()
        if not np.isfinite(m): continue
        ii, cc = np.nonzero(dl == m); j = int(rng.integers(len(ii))); v, c = cv[ii[j]], cc[j]; old = col[v]
        gamma[nbrs[v], old] -= 1; gamma[nbrs[v], c] += 1; col[v] = c; f += int(m)
        tabu[v, old] = it + int(rng.integers(L)) + int(lam * len(cv)); best = min(best, f)
    return f == 0, it

def partialcol(A, k, rng, max_iter, nbrs, L=10, lam=0.6):
    """Proper partial k-colouring; minimise |U| (uncoloured vertices). Move (v in U, c): colour v with c and
    uncolour its neighbours of colour c; (u, c) becomes tabu for every uncoloured neighbour u."""
    n = len(A); col = -np.ones(n, int)
    for v in rng.permutation(n):                                   # greedy start with k colours
        free = [c for c in range(k) if not np.any(col[nbrs[v]] == c)]
        if free: col[v] = free[0]
    gamma = np.zeros((n, k), np.int64)                             # gamma[v, c] = # neighbours of v coloured c
    for v in range(n):
        if col[v] >= 0: gamma[nbrs[v], col[v]] += 1
    tabu = np.zeros((n, k), np.int64); U = np.flatnonzero(col < 0); best = len(U); it = 0
    for it in range(1, max_iter + 1):
        if len(U) == 0: break
        cost = gamma[U].astype(float)                              # Delta|U| = gamma[v, c] - 1
        cost = np.where((tabu[U] < it) | (len(U) + cost - 1 < best), cost, np.inf)
        if not np.isfinite(cost.min()): continue
        ii, cc = np.nonzero(cost == cost.min()); j = int(rng.integers(len(ii))); v, c = U[ii[j]], cc[j]
        for u in nbrs[v][col[nbrs[v]] == c]:
            col[u] = -1; gamma[nbrs[u], c] -= 1; tabu[u, c] = it + int(rng.integers(L)) + int(lam * len(U))
        col[v] = c; gamma[nbrs[v], c] += 1
        U = np.flatnonzero(col < 0); best = min(best, len(U))
    assert not np.any(A & (col[:, None] == col[None, :]) & (col[:, None] >= 0))      # always proper
    return len(U) == 0, it

graphs, k_tabucol = [(100, 0.5), (100, 0.3), (80, 0.5)], [14, 10, 13]      # k = smallest k TabuCol solved in the lab
for gi, ((n, p), k0) in enumerate(zip(graphs, k_tabucol)):
    A = gnp(n, p, np.random.default_rng(100 + gi)); nbrs = [np.flatnonzero(A[v]) for v in range(n)]
    for k in (k0, k0 - 1):
        out = []
        for fn in (tabucol, partialcol):
            R = [fn(A, k, np.random.default_rng(1000 * gi + s), 20_000, nbrs) for s in range(3)]
            out.append(f"{sum(r[0] for r in R)}/3 {[r[1] for r in R if r[0]]}")
        print(f"G({n},{p}), k = {k}: TabuCol {out[0]:<22} PartialCol {out[1]}")
```

This uses the lab's three graphs, 3 seeds (the lab's TabuCol seeds for both methods) and at most 20 000 iterations,
at the smallest $k$ TabuCol solved in the lab and at $k - 1$. The iteration counts are those of the successful runs:

| graph, $k$ | TabuCol solved (iterations) | PartialCol solved (iterations) |
|---|---|---|
| $G(100,0.5)$, 14 | 1/3 (9 408) | 0/3 |
| $G(100,0.5)$, 13 | 0/3 | 0/3 |
| $G(100,0.3)$, 10 | 3/3 (232, 240, 138) | 3/3 (989, 907, 303) |
| $G(100,0.3)$, 9 | 0/3 | 0/3 |
| $G(80,0.5)$, 13 | 3/3 (130, 459, 543) | 3/3 (1 153, 2 020, 457) |
| $G(80,0.5)$, 12 | 0/3 | 0/3 |

PartialCol reaches the same minimal $k$ on $G(100,0.3)$ and $G(80,0.5)$, but it fails at $k = 14$ on $G(100,0.5)$,
where TabuCol succeeds in only one of three runs as well. Where both succeed, TabuCol needs fewer iterations in 5 of 6
runs (by a factor of about 2–9). The two iterations are not equal in cost: a PartialCol iteration scans $\vert U \vert \cdot k$ moves, while a
TabuCol iteration scans $\vert C \vert (k-1)$ moves, and usually $\vert U \vert \ll \vert C \vert$. Blöchliger & Zufferey
report PartialCol's advantage mainly on larger, structured instances.

### Exercise 4 ★★ — Moments of $W^{+}$ and the tie correction

Under $H_0$, $W^{+} = \sum_{i=1}^m r_i I_i$ with independent $I_i \sim$ Bernoulli$(1/2)$ and ranks $r_i$. Hence
$E[W^{+}] = \tfrac12 \sum_i r_i$ and $\mathrm{Var}(W^{+}) = \tfrac14 \sum_i r_i^2$.

*No ties:* $r_i$ runs through $1..m$, so $E = m(m+1)/4$ and $\mathrm{Var} = \tfrac14 \cdot m(m+1)(2m+1)/6 = m(m+1)(2m+1)/24$.

*Ties:* a group of $t$ tied values that would occupy ranks $a+1, \dots, a+t$ all receive the midrank
$\bar r = a + (t+1)/2$. The rank sum is unchanged, so the mean stays $m(m+1)/4$. The sum of squares falls by

$$
\sum_{j=1}^{t} (a+j)^2 - t \bar r^{\,2} = \sum_{j=1}^{t} \Bigl(j - \tfrac{t+1}{2}\Bigr)^2 = \frac{t(t^2-1)}{12} = \frac{t^3 - t}{12},
$$

which is $t$ times the variance of the uniform distribution on $\lbrace 1..t \rbrace$. So

$$
\mathrm{Var}(W^{+}) = \frac{m(m+1)(2m+1)}{24} - \sum_g \frac{t_g^3 - t_g}{48},
$$

the formula implemented in the lab (validated there against `scipy.stats.wilcoxon`). $\square$

### Exercise 5 ★★ — Robustness of the TS-vs-SA verdict to the SA schedule

There are five SA schedules. $T_0$ is scaled by 0.1, 1 or 10 relative to the lab's rule
$T_0 = \bar\delta^{+}/\ln 2$, and $T_f/T_0 \in \lbrace 10^{-2}, 10^{-3}, 10^{-5} \rbrace$. Each is compared with robust
TS on the lab's 12 instances with $10^5$ evaluations, using the lab's own Wilcoxon test, and the five $p$-values are
Holm-adjusted.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy import stats
from utils import random_qap, qap_cost

def qap_delta_matrix(p, F, D):
    Dp = D[np.ix_(p, p)]; M = F @ Dp; dg = np.diag(M)
    Delta = 2.0 * (M + M.T - dg[:, None] - dg[None, :] + 2.0 * F * Dp)
    np.fill_diagonal(Delta, 0.0); return Delta

def qap_update_after_swap(Delta, q, r, s, F, D):     # Taillard's O(n^2) update; q = permutation after swap
    a = F[r] - F[s]; b = D[q[s], q] - D[q[r], q]
    new = Delta + 2.0 * np.subtract.outer(a, a) * np.subtract.outer(b, b)
    Dq = D[np.ix_(q, q)]; diagM = np.sum(F * Dq, axis=0)
    for i in (r, s):
        row = 2.0 * (F[:, i] @ Dq + Dq[:, i] @ F - diagM[i] - diagM + 2.0 * F[i] * Dq[i]); row[i] = 0.0
        new[i, :] = row; new[:, i] = row
    return new

def qap_delta(p, r, s, F, D):
    return 2.0 * ((F[:, r] - F[:, s]) @ (D[p, p[s]] - D[p, p[r]]) + 2.0 * F[r, s] * D[p[r], p[s]])

def robust_ts(F, D, max_evals, rng):             # the lab's robust TS (weak rule, tenure in [0.9n, 1.1n])
    n = len(F); p = rng.permutation(n); cost = qap_cost(p, F, D); best = cost
    Delta = qap_delta_matrix(p, F, D); iu, ju = np.triu_indices(n, 1)
    tabu_until = np.full((n, n), -1, dtype=np.int64); t_lo, t_hi = int(np.floor(0.9 * n)), int(np.ceil(1.1 * n))
    evals, k, t = 0, 0, t_hi
    while evals + len(iu) <= max_evals:
        k += 1; evals += len(iu)
        if (k - 1) % (2 * t_hi) == 0: t = int(rng.integers(t_lo, t_hi + 1))
        d = Delta[iu, ju]
        adm = ~((tabu_until[iu, p[ju]] >= k) & (tabu_until[ju, p[iu]] >= k)) | (cost + d < best - 1e-9)
        if not adm.any(): adm[:] = True
        j = int(np.argmin(np.where(adm, d, np.inf))); r, s = int(iu[j]), int(ju[j])
        tabu_until[r, p[r]] = tabu_until[s, p[s]] = k + t
        cost += d[j]; q = p.copy(); q[[r, s]] = q[[s, r]]
        Delta = qap_update_after_swap(Delta, q, r, s, F, D); p = q; best = min(best, cost)
    return best

def sa_schedule(F, D, max_evals, rng, t0_scale=1.0, final_ratio=1e-3, n_probe=200):
    """The lab's SA with T0 = t0_scale * mean uphill delta / ln 2 and T_f = final_ratio * T0."""
    n = len(F); p = rng.permutation(n); cost = qap_cost(p, F, D)
    probe = [qap_delta(p, *rng.choice(n, 2, replace=False), F, D) for _ in range(n_probe)]
    T = t0_scale * np.mean([d for d in probe if d > 0]) / np.log(2.0)
    steps = max_evals - n_probe; rho = final_ratio ** (1.0 / steps); best = cost
    R = rng.integers(0, n, size=(steps, 2)); U = rng.random(steps)
    for i in range(steps):
        r, s = R[i]
        if r != s:
            d = qap_delta(p, r, s, F, D)
            if d <= 0 or U[i] < np.exp(-d / T):
                p[r], p[s] = p[s], p[r]; cost += d; best = min(best, cost)
        T *= rho
    return best

def wilcoxon_exact(x, y):                        # the lab's exact signed-rank test (no ties / zeros here)
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]; m = len(d)
    r = np.empty(m); r[np.argsort(np.abs(d))] = np.arange(1, m + 1); wp = int(r[d > 0].sum())
    c = np.zeros(m * (m + 1) // 2 + 1); c[0] = 1.0
    for j in range(1, m + 1): c[j:] = c[j:] + c[:-j].copy()
    pmf = c / 2.0 ** m
    return min(1.0, 2 * min(pmf[:wp + 1].sum(), pmf[wp:].sum()))

def holm(pvals):
    p = np.asarray(pvals); order = np.argsort(p); m = len(p); adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj

insts = [random_qap(25, np.random.default_rng(700 + i)) for i in range(12)]          # the lab's instances
ts = np.array([robust_ts(F, D, 100_000, np.random.default_rng(i)) for i, (F, D) in enumerate(insts)])
schedules = [(1.0, 1e-3), (0.1, 1e-3), (10.0, 1e-3), (1.0, 1e-2), (1.0, 1e-5)]
raw = []
for t0s, fr in schedules:
    sa = np.array([sa_schedule(F, D, 100_000, np.random.default_rng(i), t0s, fr) for i, (F, D) in enumerate(insts)])
    p = wilcoxon_exact(ts, sa); raw.append(p)
    assert abs(p - stats.wilcoxon(ts, sa).pvalue) < 1e-10                          # independent reference
    print(f"T0 x{t0s:<4} Tf/T0 = {fr:.0e}: TS better on {np.sum(ts < sa)}/12, "
          f"median (TS-SA)/SA = {np.median(100 * (ts - sa) / sa):+.2f}%, raw p = {p:.3f}")
print("Holm-adjusted p:", np.round(holm(raw), 3).tolist())
```

| SA schedule ($T_0$ factor, $T_f/T_0$) | TS better on | median $(TS-SA)/SA$ | raw $p$ |
|---|---|---|---|
| ×1, $10^{-3}$ (lab) | 4/12 | +0.04% | 0.470 |
| ×0.1, $10^{-3}$ | 6/12 | −0.02% | 0.380 |
| ×10, $10^{-3}$ | 7/12 | −0.05% | 0.622 |
| ×1, $10^{-2}$ | 4/12 | +0.11% | 0.266 |
| ×1, $10^{-5}$ | 4/12 | +0.20% | 0.733 |

All five Holm-adjusted $p$-values are 1.0. The first row reproduces the lab's result exactly (4/12, $p = 0.4697$); the
block copies the lab's robust TS, SA and exact Wilcoxon test, and checks every $p$-value against `scipy.stats.wilcoxon`. The verdict "no detectable difference" holds for every SA schedule tried. The direction of
the small effect even flips with $T_0$. With 12 instances, the test can only detect large effects. Establishing
differences of about 0.1% would need many more instances, or several seeds per instance with a mixed-effects analysis.

### Exercise 6 ★★★ — Feature selection with $d = 60$

The data have $d = 60$ standard normal features: 4 suppressor pairs $(0,1), \dots, (6,7)$ (second member = first
member plus noise with s.d. 0.15, coefficients $+2, -2$), 6 plain signals (features 8–13) with coefficients 1.0 down
to 0.3, and noise with s.d. 0.5. There are 300 training rows and 3 000 test
rows, and 8 datasets. The selection criterion is **3 × repeated 5-fold CV MSE** $+\,0.01\vert S\vert$, and each
distinct subset scored counts as one evaluation (cached). The budget is 1 200 evaluations. The lasso is a from-scratch
coordinate descent on standardised features over 60 penalties. Each distinct support on its path is refitted by OLS
and scored with the same CV criterion, and each such scoring counts as one evaluation. Each CV evaluation is 15 OLS fits,
solved from precomputed per-fold Gram matrices.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

d, pairs, plain = 60, [(0, 1), (2, 3), (4, 5), (6, 7)], np.arange(8, 14)
truth = np.zeros(d, bool); truth[:14] = True

def make_data(rng, n_tr=300, n_te=3000):
    Z = rng.normal(size=(n_tr + n_te, d))
    for a, b in pairs: Z[:, b] = Z[:, a] + 0.15 * rng.normal(size=len(Z))     # suppressor pairs
    y = sum(2.0 * (Z[:, a] - Z[:, b]) for a, b in pairs) + Z[:, plain] @ np.linspace(1.0, 0.3, 6)
    y = y + 0.5 * rng.normal(size=len(Z))
    return Z[:n_tr], y[:n_tr], Z[n_tr:], y[n_tr:]

def ols_mse(Xtr, ytr, Xte, yte, S):
    idx = np.flatnonzero(S)
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(ytr)), Xtr[:, idx]]), ytr, rcond=None)[0]
    return float(np.mean((yte - np.column_stack([np.ones(len(yte)), Xte[:, idx]]) @ beta) ** 2))

class CVObjective:
    """J(S) = 3 x repeated 5-fold CV MSE + 0.01 |S|; each distinct subset scored = one evaluation (cached).
    Each of the 15 OLS fits solves the normal equations from a precomputed per-fold Gram matrix."""
    def __init__(self, X, y, rng, lam=0.01):
        self.lam, self.cache, self.evals, self.n = lam, {}, 0, len(y)
        X1 = np.column_stack([np.ones(len(y)), X]); self.folds = []
        for _ in range(3):
            for f in np.array_split(rng.permutation(len(y)), 5):
                tr = np.setdiff1d(np.arange(len(y)), f)
                self.folds.append((X1[tr].T @ X1[tr], X1[tr].T @ y[tr], X1[f], y[f]))
    def __call__(self, S):
        key = S.tobytes()
        if key not in self.cache:
            self.evals += 1; idx = np.concatenate([[0], 1 + np.flatnonzero(S)]); err = 0.0
            for G, b, Xf, yf in self.folds:
                beta = np.linalg.solve(G[np.ix_(idx, idx)], b[idx])
                err += np.sum((yf - Xf[:, idx] @ beta) ** 2)
            self.cache[key] = err / (3 * self.n) + self.lam * S.sum()
        return self.cache[key]

def flip(S, j): T = S.copy(); T[j] = ~T[j]; return T

def stepwise(J, budget, allow_drop=True):        # best single add (or drop) while it improves
    S = np.zeros(d, bool); cur = J(S)
    while J.evals < budget:
        vals = [(J(flip(S, j)), j) for j in range(d) if (allow_drop or not S[j]) and J.evals < budget]
        if not vals or min(vals)[0] >= cur: break
        cur, j = min(vals); S[j] = ~S[j]
    return S

def tabu(J, budget, rng, tenure=(3, 10)):
    S = np.zeros(d, bool); best_v, best_S = J(S), S.copy(); tu = np.zeros(d, int); k = 0
    while J.evals < budget:
        k += 1
        vals = np.array([J(flip(S, j)) if J.evals < budget else np.inf for j in range(d)])
        adm = (tu < k) | (vals < best_v)
        j = int(np.argmin(np.where(adm, vals, np.inf)))
        if not np.isfinite(vals[j]): break
        S[j] = ~S[j]; tu[j] = k + int(rng.integers(tenure[0], tenure[1] + 1))
        if vals[j] < best_v: best_v, best_S = vals[j], S.copy()
    return best_S

def lasso_path_supports(X, y, n_lam=60):
    """Coordinate-descent lasso on standardised features (warm starts); distinct supports along the path."""
    Xs = (X - X.mean(0)) / X.std(0); yc = y - y.mean(); n = len(y)
    lam_max = np.max(np.abs(Xs.T @ yc)) / n; beta = np.zeros(d); res = yc.copy(); sups = []
    for lam in lam_max * np.logspace(0, -3, n_lam):
        for _ in range(500):
            dmax = 0.0
            for j in range(d):
                z = Xs[:, j] @ res / n + beta[j]                      # column variance is 1
                new = np.sign(z) * max(abs(z) - lam, 0.0)
                if new != beta[j]: res -= Xs[:, j] * (new - beta[j]); dmax = max(dmax, abs(new - beta[j])); beta[j] = new
            if dmax < 1e-7: break
        S = beta != 0
        if not any(np.array_equal(S, T) for T in sups): sups.append(S.copy())
    return sups

rows = {m: [] for m in ("forward", "forward-backward", "lasso path", "tabu search")}
import time; t0 = time.time()
for rep in range(8):
    Xtr, ytr, Xte, yte = make_data(np.random.default_rng(900 + rep))
    for m in rows:
        J = CVObjective(Xtr, ytr, np.random.default_rng(rep))
        if m == "forward": S = stepwise(J, 1200, allow_drop=False)
        elif m == "forward-backward": S = stepwise(J, 1200)
        elif m == "tabu search": S = tabu(J, 1200, np.random.default_rng(rep))
        else: S = min(lasso_path_supports(Xtr, ytr), key=J)            # refit each support by OLS, score by CV
        rows[m].append((J(S), ols_mse(Xtr, ytr, Xte, yte, S), np.sum(S & truth), np.sum(S & ~truth),
                        sum(S[a] and S[b] for a, b in pairs), J.evals))
for m, r in rows.items():
    r = np.array(r, float)
    print(f"{m:>16}: CV-J {np.median(r[:, 0]):.4f}  test MSE {np.median(r[:, 1]):.4f}  true {np.median(r[:, 2]):.1f}/14"
          f"  false pos {np.median(r[:, 3]):.1f}  pairs {r[:, 4].mean():.2f}/4 (mean)  evals {np.median(r[:, 5]):.0f}")
print(f"({time.time() - t0:.0f} s)")
```

(`forward` is `stepwise` with `allow_drop=False`.) Medians over the 8 datasets, except the pair count, which is a mean:

| method | median CV-$J$ | median test MSE | true features found (of 14) | false positives | complete pairs (of 4, mean) | evaluations used |
|---|---|---|---|---|---|---|
| forward | 0.6267 | 0.5954 | 7.0 | 0.0 | 0.50 | 452 |
| forward–backward | 0.6267 | 0.5954 | 7.0 | 0.0 | 0.50 | 468 |
| lasso path | 0.6845 | 0.6420 | 6.0 | 0.0 | 0.00 | 36 |
| tabu search | **0.5850** | **0.5114** | **9.0** | 0.0 | **1.62** | 1 200 |

The greedy methods and the lasso find the 6 plain signals but few suppressor pairs. A pair member helps only a
little on its own, so greedy selection completes a pair only occasionally (0.5 of 4 on average), once the plain signals
are in. The lasso never does: it shrinks each feature by its individual marginal contribution and would need a very
small penalty to admit near-collinear pairs. Forward–backward stops at the same point as forward, because it needs an
improving single move and none exists. Tabu search accepts the uphill "add one member" step. It completes 1.6 of the 4
pairs on average and lowers the median test MSE by about 14% compared with forward selection. It uses its whole budget,
while the greedy methods stop by themselves after about 460 evaluations, so this is "equal *maximum* budget". Greedy
restarts from random subsets could use the remaining budget, but they would still need an uphill step to capture a
pair. With about 20 iterations of 60 flips, tabu search still misses more than half of the pairs, so a frequency-based
diversification (Notebook 2) or a pair-flip move would help. Repeated CV keeps the selection from overfitting a single
validation split: no method selects a false positive in the median run.
