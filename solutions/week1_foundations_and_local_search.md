# Solutions — Week 1: Foundations: Search Spaces, Landscapes and Local Search

> Try each exercise yourself before reading these solutions: the point is the reasoning, not the answer.

Every `python` block below is a **standalone script**. It is meant to be run from inside
`week1_foundations_and_local_search/`, has its own imports (`sys.path.insert(0, "..")` for `utils`), defines every
helper it uses (copied compactly from the notebooks), and is seeded. `tools/run_solution_blocks.py` executes them all.
The numbers quoted in the text are what the blocks print. Where a solution needs a notebook's instance, the block
rebuilds that instance exactly, either from the notebook's fixed seed or by replaying the notebook's random stream.
The only exception is the wall-clock timing of Notebook 1, Exercise 5, which by its nature changes from run to run.

---

## Notebook 1 — Optimization Problems, Black Boxes and No Free Lunch

### Exercise 1 (★) — Orbit of a function = functions with the same histogram; size is multinomial

Let $f: X \to Y$ with $X$ finite, and write $h_y(f) = \vert f^{-1}(y) \vert$ for its histogram.

*(Orbit ⊆ same histogram.)* For a bijection $\sigma$ of $X$,

$$
(f\circ\sigma)^{-1}(y) = \sigma^{-1}\big(f^{-1}(y)\big),
$$

and a bijection preserves cardinalities, so $h_y(f\circ\sigma) = h_y(f)$ for every $y$.

*(Same histogram ⊆ orbit.)* Let $g$ satisfy $h_y(g) = h_y(f)$ for all $y$. For each $y$ pick a bijection
$\sigma_y: g^{-1}(y) \to f^{-1}(y)$, which exists because the sets have equal size. The level sets of $g$ partition $X$,
and so do those of $f$. So $\sigma = \bigcup_y \sigma_y$ is a bijection of $X$, and for $x \in g^{-1}(y)$ we have
$f(\sigma(x)) = y = g(x)$. Hence $g = f \circ \sigma$.

*(Size.)* A function with histogram $(h_y)_{y\in Y}$ is determined by choosing which $h_{y_1}$ points get $y_1$, which
$h_{y_2}$ of the rest get $y_2$, and so on. The number of such functions is therefore

$$
\binom{\vert X\vert}{h_{y_1}}\binom{\vert X\vert - h_{y_1}}{h_{y_2}}\cdots = \frac{\vert X\vert !}{\prod_{y} h_y!}.
$$

For $f_0 = (0,1,1,2,2,2,2,2)$ this gives $8!/(1!\,2!\,5!) = 168$, the value the notebook checks.

### Exercise 2 (★) — A revisiting algorithm can be strictly worse; why NFL assumes non-revisiting

*Example.* Take $X = \lbrace a, b\rbrace$ and $Y = \lbrace 0, 1\rbrace$, which gives 4 functions. Algorithm $A_1$ queries
$a$ and then $b$. Algorithm $A_2$ queries $a$ twice. Measure performance by the best value after $m = 2$ evaluations,
averaged uniformly over the four functions:

$$
A_1: \ \mathbb{E}\big[\min(f(a), f(b))\big] = \tfrac14, \qquad A_2: \ \mathbb{E}\big[f(a)\big] = \tfrac12 .
$$

So $A_2$ is strictly worse *per evaluation*, averaged over all functions. Its second evaluation contains no new
information: the value is already known.

*Why NFL is stated for non-revisiting algorithms.* The proof counts, for a value sequence $y \in Y^m$, the functions
that produce it. It uses the fact that every query is a new point, whose value is still free, so each step multiplies the
count by exactly $1/\vert Y\vert$. A revisit breaks this. Its value is forced by the history, so sequences with an
inconsistent repeated value have count 0 and the others have count $\vert Y\vert^{\vert X\vert - m'}$, where $m'$ is the
number of *distinct* points. The distribution of $d^y_m$ then depends on *when* the algorithm revisits, so it is
algorithm-dependent. Wolpert and Macready avoid this by measuring performance on distinct points. Any algorithm can be
made non-revisiting by caching its history, which never makes it worse: on every $f$ it sees the same distinct values
at no greater cost.

### Exercise 3 (★★) — Exact distribution of $d^y_m$ for the uniformly-random-order algorithm

Picking a uniformly random unvisited point at each step produces a uniformly random permutation $\pi$ of $X$. The
algorithm is therefore a uniform mixture of the $8!$ deterministic, non-adaptive orders:

$$
P(d^y_m = y \mid f, \text{random}) = \frac{1}{8!}\sum_{\pi} \mathbf{1}\big[(f(\pi_1),\ldots,f(\pi_m)) = y\big].
$$

Summed over all $f$, every deterministic component produces each $y \in Y^m$ exactly $3^{8-m}$ times (NFL). The mixture
therefore gives each $y$ probability $3^{8-m}/3^8 = 3^{-m}$ under a uniformly random $f$. The block computes the
mixture exactly, over all $8!$ orders and all 6561 functions.

```python
import sys; sys.path.insert(0, "..")
import itertools
import numpy as np

NX, NY = 8, 3
HAM = np.array([[bin(a ^ b).count("1") for b in range(NX)] for a in range(NX)])
ALL_F = np.array(list(itertools.product(range(NY), repeat=NX)))          # all 3^8 functions
perms = np.array(list(itertools.permutations(range(NX))))                # all 8! orders

# (i) exact distribution of d^y_m for the uniformly-random-order algorithm = uniform mixture over the 8! orders
m, POW = 4, NY ** np.arange(3, -1, -1)
counts = np.zeros(NY**m)
for p in perms:
    counts += np.bincount(ALL_F[:, p[:m]] @ POW, minlength=NY**m)
prob = counts / (len(perms) * len(ALL_F))
print(f"m={m}: {len(prob)} sequences, min prob {prob.min():.7f}, max prob {prob.max():.7f}, 3^-m = {NY**-m:.7f}")
assert np.allclose(prob, NY**-m, atol=1e-15)

# (ii) mean best-so-far of the same mixture on the Hamming-Lipschitz class
smooth = np.array([all(abs(int(f[a]) - int(f[b])) <= 1 for a in range(NX) for b in range(NX) if HAM[a, b] == 1)
                   for f in ALL_F])
Fs = ALL_F[smooth]
curve = np.zeros(NX)
for p in perms:
    curve += np.minimum.accumulate(Fs[:, p], axis=1).mean(axis=0)
curve /= len(perms)
print(f"{smooth.sum()} Hamming-Lipschitz functions; random-order mean best-so-far, m = 1..8:")
print("  " + ", ".join(f"{v:.3f}" for v in curve))
```

```text
m=4: 81 sequences, min prob 0.0123457, max prob 0.0123457, 3^-m = 0.0123457
743 Hamming-Lipschitz functions; random-order mean best-so-far, m = 1..8:
  1.000, 0.717, 0.575, 0.491, 0.436, 0.398, 0.369, 0.346
```

Every one of the 81 sequences has probability exactly $3^{-4}$. On the 743 Hamming-Lipschitz functions, the random
order's mean best-so-far at $m = 3$ is 0.575. That lies between the exploiting greedy-local algorithm (0.556) and the
fixed lexicographic and Gray orders (0.580), and below anti-local (0.585) and history-hash (0.598), all from
Notebook 1's table. A random order ignores the structure, and on this non-c.u.p. class it is neither the best nor the
worst.

### Exercise 4 (★★) — "Only if" of the sharpened NFL on a two-point space

Let $X = \lbrace a, b\rbrace$. The only non-identity permutation is the swap $\tau$. Identify $f$ with the pair
$(f(a), f(b))$, so $f\circ\tau = (f(b), f(a))$. Suppose $F$ is not c.u.p. Then there is $f^\circ \in F$ with
$f^\circ\circ\tau \notin F$. Consider

- $A_1$: query $a$, then $b$; its full value sequence on $f$ is $(f(a), f(b))$, i.e. $f$ itself;
- $A_2$: query $b$, then $a$; its full value sequence on $f$ is $(f(b), f(a)) = f\circ\tau$.

With $m = 2$, the set of sequences that $A_1$ produces over $F$ is $F$, and the set for $A_2$ is
$F\circ\tau = \lbrace f\circ\tau : f\in F\rbrace$. Each sequence occurs once, because distinct functions give distinct
pairs. The sequence $f^\circ\circ\tau$ is produced by $A_2$ on $f^\circ$. It is produced by $A_1$ on no function of $F$,
since that would require $f^\circ\circ\tau \in F$. So

$$
\sum_{f\in F} P(d^y_2 = f^\circ\circ\tau \mid f, A_2) = 1 \ne 0 = \sum_{f\in F} P(d^y_2 = f^\circ\circ\tau \mid f, A_1),
$$

and the performance measure $\Phi(d^y_2) = \mathbf{1}[d^y_2 = f^\circ\circ\tau]$ separates the two algorithms. Note that the
difference may only appear at $m = 2$. For example, $F = \lbrace(0,1),(1,2),(2,0)\rbrace$ is not c.u.p., yet $f(a)$ and
$f(b)$ have the same multiset $\lbrace 0,1,2\rbrace$, so the $m=1$ distributions coincide.

### Exercise 5 (★★) — Wall-clock fits of Held–Karp and why the notebook counts operations

The block times `utils.held_karp` (best of 3) and fits three models by least squares on $\log t$ over $n = 8, \ldots, 15$.
It compares them by AIC, $n\log(\mathrm{RSS}/n) + 2k$. The time per relaxation is $t(n)/R(n)$ with
$R(n) = (n-1)(n-2)2^{n-3}$, the exact count from Notebook 1.

```python
import sys; sys.path.insert(0, "..")
import time
import numpy as np
from utils import held_karp, random_euclidean_tsp

sizes, rng, t = np.arange(5, 16), np.random.default_rng(0), []
for n in sizes:
    _, D = random_euclidean_tsp(int(n), rng)
    reps = []
    for _ in range(3):                                   # best of 3 repetitions reduces load noise
        t0 = time.perf_counter(); held_karp(D); reps.append(time.perf_counter() - t0)
    t.append(min(reps))
t = np.array(t)
R = (sizes - 1) * (sizes - 2) * 2.0 ** (sizes - 3)       # exact number of relaxations (Notebook 1)
print("time per relaxation (us), n = 8..15:", " ".join(f"{v:.2f}" for v in 1e6 * t[3:] / R[3:]))

y, n_ = np.log(t)[3:], sizes[3:]                         # fit on n >= 8 (tiny n are dominated by overhead)
def aic(res, k):
    return len(res) * np.log(np.mean(res**2)) + 2 * k
X = np.c_[np.ones_like(n_), n_]
cA = np.mean(y - 2 * np.log(n_) - n_ * np.log(2))                     # c n^2 2^n         (1 parameter)
bA = np.linalg.lstsq(X, y - 2 * np.log(n_), rcond=None)[0]            # c n^2 2^(beta n)  (2 parameters)
bB = np.linalg.lstsq(X, y, rcond=None)[0]                             # c 2^(beta n)      (2 parameters)
print(f"c n^2 2^n        : AIC {aic(y - cA - 2 * np.log(n_) - n_ * np.log(2), 1):7.2f}  base 2 (fixed)")
print(f"c n^2 2^(beta n) : AIC {aic(y - 2 * np.log(n_) - X @ bA, 2):7.2f}  base {np.exp(bA[1]):.2f}")
print(f"c 2^(beta n)     : AIC {aic(y - X @ bB, 2):7.2f}  base {np.exp(bB[1]):.2f}")
print(f"memory of held_karp at n=25 (float64 C + int64 P, shape 2^24 x 25): {2 * 25 * 2**24 * 8 / 1e9:.2f} GB; "
      f"C alone at n=30: {30 * 2**29 * 8 / 1e9:.0f} GB")
```

One run on a shared, loaded machine printed:

```text
time per relaxation (us), n = 8..15: 0.86 0.92 1.71 1.55 1.93 2.25 1.75 2.42
c n^2 2^n        : AIC  -11.86  base 2 (fixed)
c n^2 2^(beta n) : AIC  -23.32  base 2.36
c 2^(beta n)     : AIC  -21.15  base 2.82
memory of held_karp at n=25 (float64 C + int64 P, shape 2^24 x 25): 6.71 GB; C alone at n=30: 129 GB
```

On this run the model with the $n^2$ factor and a free base fits best. Its base, 2.36, is above 2 because the time
*per relaxation* is not constant: it drifts from about 0.9 to 2.4 µs over this range, through cache effects, Python
overheads and machine load. Without the $n^2$ term the base must also absorb the polynomial factor. Over $n = 8..15$
that factor contributes about $(15/8)^{2/7} \approx 1.20$ per added city, and $2.36 \times 1.20 \approx 2.83$ is the
2.82 of the pure exponential fit. Re-running the block gives different AIC values and bases, and can change the ranking.
This is exactly why a *check* must not use timings: wall time measures the implementation, the interpreter and the
machine at that moment, not the algorithm. The operation count $R(n) + n$ is exact, reproducible, and matches the
closed form on every machine.

*Memory for $n=25$.* `utils.held_karp` stores two arrays of shape $2^{24}\times 25$, a float64 cost table and an int64
predecessor table. That is $2 \times 25 \times 2^{24} \times 8$ bytes $\approx 6.7$ GB. A laptop runs out of memory
before it runs out of patience, and at $n = 30$ the cost table alone is 129 GB.

### Exercise 6 (★★★) — A structured class on $\lbrace 0,1\rbrace^4$, the best adaptive algorithm, and its cost on the complement

*Class.* Let $\mathcal{C}$ be the functions $f:\lbrace 0,1\rbrace^4 \to \lbrace 0,1,2\rbrace$ that are
**Hamming-1-Lipschitz** ($\vert f(x) - f(x')\vert \le 1$ for neighbours) and have a **unique minimiser**
$t$ with $f(t) = 0$. Lipschitz continuity and uniqueness force every neighbour of $t$ to have value 1. The other 11
points take values in $\lbrace 1, 2\rbrace$ freely, because $\vert 1 - 2\vert = 1$. Hence

$$
\vert\mathcal{C}\vert = 16 \cdot 2^{11} = 32768 .
$$

The block also checks this characterisation against the definition on 200 000 random functions. The task is to
minimise the expected number of evaluations $T$ until $t$ is evaluated, uniformly over $\mathcal{C}$.

*Posterior.* Given the observations, $t$ is still possible iff it is unvisited and every observed neighbour of $t$ has
value 1. Every such $t$ leaves $2^{11 - (\text{observed non-neighbours of } t)}$ functions of $\mathcal{C}$ consistent
with the history. With $m$ observations and $w(t)$ observed neighbours of $t$, that is $2^{11 - m + w(t)}$, so

$$
P(t \mid \text{history}) \propto 2^{\,w(t)}\,\mathbf{1}[t \text{ consistent}], \qquad w(t) = \#\lbrace\text{observed neighbours of } t\rbrace .
$$

*Family of algorithms.* The **MAP rule** queries the unvisited point of highest posterior. We search over MAP rules with
44 different tie-breaking orders: lexicographic, reverse, Gray, Hamming weight, and 40 random orders. We also include
the fixed lexicographic and Gray orders and Notebook 1's greedy-local algorithm. $\mathbb{E}[T \mid \mathcal{C}]$ is
computed exactly for each, over all 32768 functions. Outside $\mathcal{C}$ no point may be consistent, and then the MAP
rule queries the first unvisited point in its tie-break order. Every algorithm is therefore a well-defined,
deterministic, non-revisiting algorithm on all $3^{16}$ functions, which the NFL argument below requires.

```python
import sys; sys.path.insert(0, "..")
import itertools
import numpy as np

NX = 16
HAM = np.array([[bin(a ^ b).count("1") for b in range(NX)] for a in range(NX)])

# the class C: Hamming-1-Lipschitz f: {0,1}^4 -> {0,1,2} with a unique minimiser t, f(t) = 0
C = []
for t in range(NX):
    free = [x for x in range(NX) if HAM[t, x] >= 2]
    for vals in itertools.product((1, 2), repeat=len(free)):
        f = np.ones(NX, dtype=int); f[t] = 0; f[free] = vals; C.append(f)
C = np.array(C)
print("|C| =", len(C))
# brute-force check of the characterisation on random functions
r = np.random.default_rng(0)
G = r.integers(0, 3, (200_000, NX))
E = [(a, b) for a in range(NX) for b in range(a + 1, NX) if HAM[a, b] == 1]
lip = np.all([np.abs(G[:, a] - G[:, b]) <= 1 for a, b in E], axis=0)
in_def = lip & ((G == 0).sum(1) == 1)
Cset = set(map(tuple, C))
assert all((tuple(g) in Cset) == bool(k) for g, k in zip(G, in_def)), "characterisation of C is wrong"
print(f"definition vs characterisation agree on 200000 random functions ({in_def.sum()} members)")

def make_map(tiebreak):
    """MAP rule: query the unvisited point of highest posterior 2^w(t); ties broken by `tiebreak` order.
    If no point is consistent (only possible outside C), query the first unvisited point in that order."""
    def pol(xs, ys):
        best, arg = -1, None
        for t in tiebreak:
            if t in xs:
                continue
            nb = [y for x, y in zip(xs, ys) if HAM[t, x] == 1]
            if all(y == 1 for y in nb) and len(nb) > best:
                best, arg = len(nb), t
        return arg if arg is not None else next(t for t in tiebreak if t not in xs)
    return pol

def greedy_local(xs, ys):                                # Notebook 1's greedy-local algorithm, on 4 bits
    if not xs:
        return 0
    b = xs[int(np.argmin(ys))]
    return min((x for x in range(NX) if x not in xs), key=lambda x: (HAM[b, x], x))

def fixed_order(order):
    return lambda xs, ys: next(x for x in order if x not in xs)

def expected_T(policy, funcs):
    """Mean number of evaluations until the first global minimum is evaluated (memoised over histories)."""
    cache, total = {}, 0
    for f in funcs:
        xs, ys, fmin = [], [], f.min()
        while True:
            key = tuple(zip(xs, ys))
            if key not in cache:
                cache[key] = policy(xs, ys)
            x = cache[key]; xs.append(x); ys.append(int(f[x]))
            if ys[-1] == fmin:
                break
        total += len(xs)
    return total / len(funcs)

lex, gray = list(range(NX)), [g ^ (g >> 1) for g in range(NX)]
weight = sorted(range(NX), key=lambda x: (bin(x).count("1"), x))
family = {"lexicographic order": fixed_order(lex), "Gray order": fixed_order(gray), "greedy local": greedy_local,
          "MAP, tie-break lex": make_map(lex), "MAP, tie-break reverse": make_map(lex[::-1]),
          "MAP, tie-break Gray": make_map(gray), "MAP, tie-break Hamming weight": make_map(weight)}
res = {k: expected_T(p, C) for k, p in family.items()}
rr = np.random.default_rng(1)
for i in range(40):                                      # widen the family: MAP with 40 random tie-break orders
    order = [int(v) for v in rr.permutation(NX)]
    res[f"MAP, random tie-break #{i}"] = expected_T(make_map(order), C)
for k, v in list(res.items())[:7]:
    print(f"{k:30s} E[T | C] = {v:.3f}")
rand = np.array([v for k, v in res.items() if "random" in k])
print(f"MAP, 40 random tie-break orders: best {rand.min():.3f}, median {np.median(rand):.3f}, worst {rand.max():.3f}")
best = min(res, key=res.get)
print(f"best algorithm of the family: {best}, E[T | C] = {res[best]:.3f}")

# NFL bookkeeping: the total of T over all 3^16 functions is algorithm-independent; the lexicographic order gives it
# in closed form (first index of the minimum of 16 i.i.d. uniform values on {0,1,2})
ET_all = sum(((2 - v) / 3) ** k * (((3 - v) / 3) ** (NX - k) - ((2 - v) / 3) ** (NX - k))
             for k in range(NX) for v in range(3))
N_all, N_C = 3**NX, len(C)
saving = (res["lexicographic order"] - res[best]) * N_C
comp = lambda eC: (ET_all * N_all - eC * N_C) / (N_all - N_C)
print(f"E[T] over all 3^16 functions (any non-revisiting algorithm) = {ET_all:.5f}")
print(f"best MAP saves {saving:.0f} evaluations in aggregate on C; mean on the complement "
      f"{comp(res[best]):.5f} vs {comp(res['lexicographic order']):.5f} for the lexicographic order "
      f"(complement is {(N_all - N_C) / N_C:.0f}x larger than C)")
```

```text
|C| = 32768
definition vs characterisation agree on 200000 random functions (165 members)
lexicographic order            E[T | C] = 8.500
Gray order                     E[T | C] = 8.500
greedy local                   E[T | C] = 7.903
MAP, tie-break lex             E[T | C] = 5.421
MAP, tie-break reverse         E[T | C] = 5.421
MAP, tie-break Gray            E[T | C] = 5.425
MAP, tie-break Hamming weight  E[T | C] = 5.405
MAP, 40 random tie-break orders: best 5.370, median 5.397, worst 5.450
best algorithm of the family: MAP, random tie-break #22, E[T | C] = 5.370
E[T] over all 3^16 functions (any non-revisiting algorithm) = 2.97412
best MAP saves 102559 evaluations in aggregate on C; mean on the complement 2.97229 vs 2.96991 for the lexicographic order (complement is 1313x larger than C)
```

Exploiting the structure cuts the expected cost from 8.5 evaluations (any fixed order) to 5.37 (the best MAP rule
found). The tie-break matters only in the second decimal: the MAP rules span 5.370–5.450. The MAP rule is greedy, since
it maximises the probability of success at the *next* query, so it is the best of this family but is not proven
optimal among all adaptive algorithms. That would need a dynamic programme over histories.

*Cost on the complement.* $T$ is a function of the full value sequence $d^y_{16}$: it is the index of the first global
minimum. By NFL, its sum over all $3^{16}$ functions is the same for every non-revisiting algorithm. The lexicographic
order gives it in closed form, because its values are 16 i.i.d. uniform draws from $\lbrace 0,1,2\rbrace$:

$$
\mathbb{E}[T] = \sum_{k=0}^{15} P(T \gt k) = \sum_{k=0}^{15}\sum_{v=0}^{2}\Big(\tfrac{2-v}{3}\Big)^{k}\Big[\Big(\tfrac{3-v}{3}\Big)^{16-k} - \Big(\tfrac{2-v}{3}\Big)^{16-k}\Big] = 2.97412 .
$$

The best MAP rule saves $(8.5 - 5.370)\times 32768 = 102\,559$ evaluations in aggregate on $\mathcal{C}$ (unrounded). It
therefore pays exactly $102\,559$ extra evaluations on the $3^{16} - 32768$ other functions: its mean on the complement
is 2.97229, against 2.96991 for the lexicographic order. The price is invisible per function, because the complement is
1313 times larger, but it is exact.

---

## Notebook 2 — Representations, Neighbourhoods and Local Search

### Exercise 1 (★) — Nested neighbourhoods; a 2-opt optimum that is not 3-opt optimal

*Proof.* Let $N(x) \subseteq N'(x)$ for all $x$, and let $x$ be $N'$-locally optimal: $f(x) \le f(y)$ for all
$y\in N'(x)$. Then in particular $f(x) \le f(y)$ for all $y \in N(x)\subseteq N'(x)$, so $x$ is $N$-locally optimal.
The converse fails because $N'$ has more candidate improvements.

*Example.* Or-opt moves (relocate a segment of 1–3 cities, possibly reversed) change at most three edges, so they are
3-opt moves. Running best-improvement 2-opt on random 7-city instances and testing every Or-opt neighbour of the result
quickly finds a counterexample. One, with coordinates rounded to three decimals (index: coordinates):

0 (0.052, 0.680), 1 (0.368, 0.590), 2 (0.670, 0.669), 3 (0.523, 0.555), 4 (0.198, 0.495), 5 (0.125, 0.481), 6 (0.536, 0.774).

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import tour_length

P = np.array([[0.052, 0.680], [0.368, 0.590], [0.670, 0.669], [0.523, 0.555],
              [0.198, 0.495], [0.125, 0.481], [0.536, 0.774]])
D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
t = np.array([5, 4, 3, 2, 6, 1, 0]); n = len(t)
two_opt = [(i, j) for i in range(n - 1) for j in range(i + 2, n) if not (i == 0 and j == n - 1)]
nbrs = [np.concatenate([t[: i + 1], t[i + 1 : j + 1][::-1], t[j + 1 :]]) for i, j in two_opt]
L = tour_length(t, D)
print(f"L(t) = {L:.4f}; {len(nbrs)} 2-opt neighbours, shortest {min(tour_length(s, D) for s in nbrs):.4f}")
assert all(tour_length(s, D) >= L - 1e-12 for s in nbrs), "t is not 2-opt optimal"
t3 = np.array([5, 4, 1, 3, 2, 6, 0])
print(f"Or-opt (3-opt) neighbour {t3.tolist()}: length {tour_length(t3, D):.4f}")
assert tour_length(t3, D) < L
```

```text
L(t) = 1.5508; 14 2-opt neighbours, shortest 1.5683
Or-opt (3-opt) neighbour [5, 4, 1, 3, 2, 6, 0]: length 1.4892
```

The tour $(5,4,3,2,6,1,0)$ has length 1.5508 and is 2-opt locally optimal: all 14 2-opt neighbours, re-evaluated in
full, are longer. Moving city 1 between 4 and 3 gives $(5,4,1,3,2,6,0)$, of length 1.4892. This is a 3-opt improvement:
edges $(6,1),(1,0),(4,3)$ are replaced by $(6,0),(4,1),(1,3)$.

### Exercise 2 (★) — Swap moves connect $\mathfrak{S}_n$ with diameter $n-1$; 2-opt connects all tours

*Swap.* The swap distance between $p$ and $q$ is the minimum number of transpositions whose product is $r = p^{-1}q$.
Write $c(r)$ for the number of cycles of $r$, fixed points included. Composing with a transposition merges two cycles or
splits one, so it changes $c$ by exactly $\pm 1$. The identity has $n$ cycles. Therefore at least $n - c(r)$
transpositions are needed. The bound is achieved: a $k$-cycle is a product of $k-1$ transpositions. So
$d(p,q) = n - c(p^{-1}q) \le n-1$, with equality iff $p^{-1}q$ is an $n$-cycle. The distance is finite for every pair, so
the graph is connected, and the diameter is exactly $n-1$.

*2-opt.* Represent a tour as a sequence with $t_0$ fixed. The 2-opt move $(i, i+2)$ reverses the length-2 segment
$(t_{i+1}, t_{i+2})$, which swaps two adjacent positions. For $n \ge 4$ all moves $(i,i+2)$ with $0 \le i \le n-3$ are
valid, because the only excluded pair, $(0, n-1)$, would need $n-1 = 2$. They realise every adjacent transposition of
positions $1,\ldots,n-1$. Adjacent transpositions generate the symmetric group on those positions, so every sequence
that starts with $t_0$ is reachable. Every tour, being a cyclic order, has such a representative. Hence the 2-opt graph
is connected, even when restricted to segment reversals of length 2.

### Exercise 3 (★★) — Or-opt with delta evaluation vs 2-opt on Held–Karp-validated instances

Remove the segment $s_0\ldots s_1$ of $L \in \lbrace 1,2,3\rbrace$ cities, which sits between $p$ and $n_x$, and
reinsert it, in either orientation, between two consecutive cities $u, v$ of the remaining tour. The change in length is

$$
\Delta = D_{u a} + D_{b v} - D_{u v} - \big(D_{p s_0} + D_{s_1 n_x} - D_{p n_x}\big), \qquad (a,b) \in \lbrace (s_0,s_1), (s_1,s_0)\rbrace ,
$$

which is computed in $O(1)$ time. There are $\sum_{L=1}^{3} 2n(n-L-1) = 6n(n-3)$ candidates per scan, 528 for $n = 11$
(for $L = 1$ both orientations coincide), against $n(n-3)/2 = 44$ for 2-opt.

The block uses the notebook's protocol exactly: 30 instances with $n = 11$ (`default_rng(1000 + inst)`), 20 starts
each, gaps against Held–Karp. It replays the notebook's random stream, so the starts are identical to the notebook's
and the 2-opt row reproduces Notebook 2's printed numbers. "VND" is a 2-opt descent followed by an Or-opt descent,
repeated until neither improves (the variable neighbourhood descent of Week 3). The Or-opt output is also checked
against an independent, explicit enumeration of all Or-opt neighbours with full tour lengths.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import random_euclidean_tsp, tour_length, held_karp

def two_opt_move(t, i, j):
    return np.concatenate([t[: i + 1], t[i + 1 : j + 1][::-1], t[j + 1 :]])

def two_opt_moves(n):
    return [(i, j) for i in range(n - 1) for j in range(i + 2, n) if not (i == 0 and j == n - 1)]

def two_opt_best(t, D):
    """Best-improvement 2-opt with O(1) deltas; returns (tour, #evaluations)."""
    t = np.array(t); n = len(t); m = n * (n - 3) // 2; ev = 0
    I, J = np.indices((n, n)); valid = (J >= I + 2) & ~((I == 0) & (J == n - 1))
    while True:
        a, b = t, np.roll(t, -1)
        dm = D[a[:, None], a[None, :]] + D[b[:, None], b[None, :]] - D[a, b][:, None] - D[a, b][None, :]
        dm = np.where(valid, dm, np.inf); ev += m
        i, j = np.unravel_index(np.argmin(dm), dm.shape)
        if dm[i, j] >= -1e-12:
            return t, ev
        t = two_opt_move(t, i, j)

def two_opt_first(t, D, rng):
    """Notebook 2's first-improvement 2-opt (run only to reproduce the notebook's random stream)."""
    t = np.array(t); n = len(t); moves = np.array(two_opt_moves(n))
    while True:
        for i, j in moves[rng.permutation(len(moves))]:
            a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
            if D[a, c] + D[b, d] - D[a, b] - D[c, d] < -1e-12:
                t = two_opt_move(t, i, j); break
        else:
            return t

def or_opt_ls(t, D):
    """Best-improvement Or-opt (segment of 1-3 cities, either orientation) with O(1) deltas; (tour, #evaluations)."""
    t = list(t); n = len(t); ev = 0
    while True:
        best, arg = -1e-12, None
        for L in (1, 2, 3):
            for i in range(n):
                s0, s1, p, nx = t[i], t[(i + L - 1) % n], t[(i - 1) % n], t[(i + L) % n]
                removed = D[p, s0] + D[s1, nx] - D[p, nx]
                for k in range(n - L - 1):                       # edge (u, v) of the remaining tour
                    u, v = t[(i + L + k) % n], t[(i + L + k + 1) % n]
                    for rev in (False, True):
                        a, b = (s1, s0) if rev else (s0, s1); ev += 1
                        dl = D[u, a] + D[b, v] - D[u, v] - removed
                        if dl < best:
                            best, arg = dl, (L, i, k, rev)
        if arg is None:
            return np.array(t), ev
        L, i, k, rev = arg
        seg = [t[(i + m) % n] for m in range(L)]; rest = [t[(i + L + m) % n] for m in range(n - L)]
        t = rest[: k + 1] + (seg[::-1] if rev else seg) + rest[k + 1 :]

def or_opt_neighbours(t):
    """All Or-opt neighbours built explicitly (independent check, full tour lengths)."""
    t = list(t); n = len(t); out = []
    for L in (1, 2, 3):
        for i in range(n):
            seg = [t[(i + m) % n] for m in range(L)]; rest = [t[(i + L + m) % n] for m in range(n - L)]
            for k in range(n - L - 1):
                for s in (seg, seg[::-1]):
                    out.append(rest[: k + 1] + s + rest[k + 1 :])
    return out

def vnd(t, D):
    """2-opt descent, then Or-opt descent, repeated until neither improves."""
    ev = 0
    while True:
        t, e1 = two_opt_best(t, D); L0 = tour_length(t, D)
        t, e2 = or_opt_ls(t, D); ev += e1 + e2
        if tour_length(t, D) >= L0 - 1e-12:
            return t, ev

N_CITIES, N_INST, N_STARTS = 11, 30, 20
res = {k: {"gap": [], "ev": []} for k in ("2-opt", "Or-opt", "2-opt + Or-opt (VND)")}
for inst in range(N_INST):                               # the notebook's instances AND starts (same random stream)
    irng = np.random.default_rng(1000 + inst)
    _, D = random_euclidean_tsp(N_CITIES, irng)
    L_opt, _ = held_karp(D)
    for s in range(N_STARTS):
        start = irng.permutation(N_CITIES)
        for name, ls in (("2-opt", two_opt_best), ("Or-opt", or_opt_ls), ("2-opt + Or-opt (VND)", vnd)):
            t, ev = ls(start, D)
            L = tour_length(t, D)
            assert L >= L_opt - 1e-9
            res[name]["gap"].append(100 * (L - L_opt) / L_opt); res[name]["ev"].append(ev)
            if name == "Or-opt" and s == 0:                  # independent optimality check of the Or-opt output
                assert all(tour_length(q, D) >= L - 1e-9 for q in or_opt_neighbours(t))
        two_opt_first(start, D, irng)                         # consumes irng exactly as the notebook does
for name, r in res.items():
    g, e = np.array(r["gap"]), np.array(r["ev"])
    q1, med, q3 = np.percentile(g, [25, 50, 75])
    print(f"{name:22s} % above opt median {med:.2f} [IQR {q1:.2f}, {q3:.2f}], mean {g.mean():.2f};  "
          f"P(optimum) {np.mean(g < 1e-7):.3f};  evaluations median {np.median(e):.0f}")
```

```text
2-opt                  % above opt median 0.00 [IQR 0.00, 0.78], mean 0.96;  P(optimum) 0.687;  evaluations median 308
Or-opt                 % above opt median 0.00 [IQR 0.00, 0.00], mean 0.04;  P(optimum) 0.935;  evaluations median 2640
2-opt + Or-opt (VND)   % above opt median 0.00 [IQR 0.00, 0.00], mean 0.07;  P(optimum) 0.922;  evaluations median 880
```

Or-opt local optima are much better on these small instances: the optimum is reached by 93.5% of descents, against
68.7% for 2-opt. But a descent costs about 8.6× more evaluations, because the neighbourhood is 12 times larger. Starting
with the cheap 2-opt and finishing with Or-opt keeps almost all of the quality at a third of the cost. This is the idea
behind the variable neighbourhood descent of Week 3.

### Exercise 4 (★★) — Exhaustive basins of first-improvement descent (fixed scan order)

For first improvement with scan order $0,1,\ldots,13$, the step moves to the first improving bit flip. On the whole state
space this is `np.argmax(V < F[:, None], axis=1)`. The rest is the notebook's pointer jumping. The block rebuilds the
notebook's own SK instance by replaying every random draw Notebook 2 makes before Section 6. It reproduces the
notebook's 11 local optima and global basin of 5252 states.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import random_euclidean_tsp

def two_opt_move(t, i, j):
    return np.concatenate([t[: i + 1], t[i + 1 : j + 1][::-1], t[j + 1 :]])

def notebook2_sk_instance():
    """Rebuild Notebook 2's SK instance by replaying every draw the notebook makes from rng = default_rng(0)."""
    rng = np.random.default_rng(0)
    rng.random(6)                                        # Section 1: random keys
    for n in (6, 7, 9):
        rng.permutation(n)                               # Section 2: neighbourhood-size checks
    _, D = random_euclidean_tsp(12, rng)                 # Section 3: 2-opt checks
    rng.permutation(12)                                  # start of best-improvement (deterministic descent)
    t = rng.permutation(12); n = 12                      # first-improvement 2-opt consumes rng
    moves = np.array([(i, j) for i in range(n - 1) for j in range(i + 2, n) if not (i == 0 and j == n - 1)])
    while True:
        for i, j in moves[rng.permutation(len(moves))]:
            a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
            if D[a, c] + D[b, d] - D[a, b] - D[c, d] < -1e-12:
                t = two_opt_move(t, i, j); break
        else:
            break
    J = np.triu(rng.standard_normal((14, 14)), 1); J = J + J.T      # Section 6
    h = 0.3 * rng.standard_normal(14)
    return J, h

NB = 14
J, h = notebook2_sk_instance()
states = np.arange(2**NB)
S = 2.0 * ((states[:, None] >> np.arange(NB)) & 1) - 1.0
F = -0.5 * np.einsum("si,ij,sj->s", S, J, S) - S @ h
NBR = states[:, None] ^ (1 << np.arange(NB))[None, :]
V = F[NBR]
g = int(np.argmin(F))
def attract(step):
    """Pointer jumping: the fixed point reached from every state."""
    a = step.copy()
    while not np.array_equal(a[a], a):
        a = a[a]
    return a

improving = V < F[:, None]
first_step = np.where(improving.any(1), NBR[states, np.argmax(improving, 1)], states)   # first improving bit, order 0..13
best_step = np.where(V.min(1) < F, NBR[states, V.argmin(1)], states)
att_f, att_b = attract(first_step), attract(best_step)
opt_f, size_f = np.unique(att_f, return_counts=True)
opt_b, size_b = np.unique(att_b, return_counts=True)
print(f"local optima: best-improvement {len(opt_b)}, first-improvement {len(opt_f)}, identical sets: {np.array_equal(opt_b, opt_f)}")
print(f"states whose descent ends at a different optimum: {100 * np.mean(att_f != att_b):.1f}%")
print(f"global-optimum basin: best {size_b[opt_b == g][0]} states ({np.mean(att_b == g):.4f}), "
      f"first {size_f[opt_f == g][0]} states ({np.mean(att_f == g):.4f})")
print(f"largest basin: best {size_b.max()}, first {size_f.max()}")
```

```text
local optima: best-improvement 11, first-improvement 11, identical sets: True
states whose descent ends at a different optimum: 56.1%
global-optimum basin: best 5252 states (0.3206), first 4434 states (0.2706)
largest basin: best 5252, first 4434
```

- **The local optima are identical** (the same 11 states). A fixed point of either descent is a point with no strictly
  improving neighbour. That property belongs to $(f, N)$ alone, not to the pivoting rule.
- **The basins differ**: 56.1% of all states end at a different optimum. The global optimum attracts 32.1% of states
  under best improvement but only 27.1% under first improvement. In both cases it still has the largest basin.

The reason is that a basin is a property of the *dynamics*, not of the landscape. First improvement follows a
different, order-dependent path through the same landscape. Basins, and therefore restart success rates, must always be
reported together with the descent rule.

### Exercise 5 (★★) — A lower bound on the success probability of RRHC

*Proof.* RRHC starts descent $i$ from an independent uniform point. Descent $i$ ends in the global optimum with
probability $p$, independently of the others. If every descent costs at most $c$ evaluations, then with budget $B$ the
first $k = \lfloor B/c\rfloor$ descents all complete: after $j \lt k$ complete descents, at least $B - jc \ge c$
evaluations remain. RRHC fails only if every descent it runs fails, and in particular only if the first $k$ fail.
Therefore

$$
P(\text{fail}) \le (1-p)^{k}, \qquad P(\text{success}) \ge 1 - (1-p)^{\lfloor B/c\rfloor}.
$$

*Check against Section 6.* The block uses the notebook's SK instance (rebuilt as in Exercise 4) with best improvement,
so $p = 0.3206$. The cost of a descent with $s$ moves is $1 + 14(s+1)$ evaluations: the initial point, one full scan per
move, and a final scan. The exhaustive maximum over all $2^{14}$ starts is $c = 183$, and the mean is 89.2. For each
budget, 20 000 RRHC runs are simulated exactly: descents run back to back, and a run succeeds iff the global optimum is
*evaluated* within $B$ evaluations. The optimum is evaluated during the scan of its predecessor, so a descent cut short
by the budget can still succeed.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import random_euclidean_tsp

def two_opt_move(t, i, j):
    return np.concatenate([t[: i + 1], t[i + 1 : j + 1][::-1], t[j + 1 :]])

def notebook2_sk_instance():
    """Rebuild Notebook 2's SK instance by replaying every draw the notebook makes from rng = default_rng(0)."""
    rng = np.random.default_rng(0)
    rng.random(6)                                        # Section 1: random keys
    for n in (6, 7, 9):
        rng.permutation(n)                               # Section 2: neighbourhood-size checks
    _, D = random_euclidean_tsp(12, rng)                 # Section 3: 2-opt checks
    rng.permutation(12)                                  # start of best-improvement (deterministic descent)
    t = rng.permutation(12); n = 12                      # first-improvement 2-opt consumes rng
    moves = np.array([(i, j) for i in range(n - 1) for j in range(i + 2, n) if not (i == 0 and j == n - 1)])
    while True:
        for i, j in moves[rng.permutation(len(moves))]:
            a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
            if D[a, c] + D[b, d] - D[a, b] - D[c, d] < -1e-12:
                t = two_opt_move(t, i, j); break
        else:
            break
    J = np.triu(rng.standard_normal((14, 14)), 1); J = J + J.T      # Section 6
    h = 0.3 * rng.standard_normal(14)
    return J, h

NB = 14
J, h = notebook2_sk_instance()
states = np.arange(2**NB)
S = 2.0 * ((states[:, None] >> np.arange(NB)) & 1) - 1.0
F = -0.5 * np.einsum("si,ij,sj->s", S, J, S) - S @ h
NBR = states[:, None] ^ (1 << np.arange(NB))[None, :]
V = F[NBR]
g = int(np.argmin(F))
# exhaustive descent statistics for best improvement: number of moves s(x), attractor, and when the optimum is first seen
step = np.where(V.min(1) < F, NBR[states, V.argmin(1)], states)
cur, prev, moves = states.copy(), states.copy(), np.zeros(2**NB, int)
while True:
    nxt = step[cur]; moving = nxt != cur
    if not moving.any():
        break
    prev = np.where(moving, cur, prev); moves += moving; cur = nxt
att = cur
cost = 1 + NB * (moves + 1)                              # start + one full scan per move + the final scan
last_bit = np.log2(np.maximum(prev ^ att, 1)).astype(int)
# evaluation index at which the global optimum is first evaluated (it is found inside the scan of its predecessor)
hit = np.where(moves == 0, 1, 1 + NB * (moves - 1) + last_bit + 1)
p, c = np.mean(att == g), cost.max()
print(f"p = {p:.4f};  descent cost: max c = {c}, mean {cost.mean():.1f} (mean {cost[att == g].mean():.1f} in the global basin)")

def rrhc_success(B, rng):
    """One RRHC run with budget B: success iff the global optimum is evaluated within B evaluations."""
    rem = B
    while True:
        x = rng.integers(2**NB)
        if att[x] == g and hit[x] <= rem:
            return True
        if cost[x] > rem:                                # this descent is cut by the budget
            return False
        rem -= cost[x]

rng = np.random.default_rng(1)
RUNS = 20000
for B in (100, 200, 400, 800):
    s = np.mean([rrhc_success(B, rng) for _ in range(RUNS)])
    print(f"B={B:4d}: bound 1-(1-p)^floor(B/c) = {1 - (1 - p) ** (B // c):.3f};  simulated {s:.3f} +- {np.sqrt(s * (1 - s) / RUNS):.3f};"
          f"  mean-cost heuristic 1-(1-p)^(B/mean) = {1 - (1 - p) ** (B / cost.mean()):.3f}")
```

```text
p = 0.3206;  descent cost: max c = 183, mean 89.2 (mean 96.2 in the global basin)
B= 100: bound 1-(1-p)^floor(B/c) = 0.000;  simulated 0.278 +- 0.003;  mean-cost heuristic 1-(1-p)^(B/mean) = 0.352
B= 200: bound 1-(1-p)^floor(B/c) = 0.321;  simulated 0.525 +- 0.004;  mean-cost heuristic 1-(1-p)^(B/mean) = 0.580
B= 400: bound 1-(1-p)^floor(B/c) = 0.538;  simulated 0.806 +- 0.003;  mean-cost heuristic 1-(1-p)^(B/mean) = 0.823
B= 800: bound 1-(1-p)^floor(B/c) = 0.787;  simulated 0.967 +- 0.001;  mean-cost heuristic 1-(1-p)^(B/mean) = 0.969
```

The bound holds at every budget, but it is loose, because it charges every descent the worst-case cost. Replacing $c$ by
the mean cost gives the right order of magnitude, and it becomes accurate for large $B$. By renewal theory the number of
completed descents concentrates around $B/\bar c$. The heuristic overestimates slightly, because the global basin's
descents are not typical: in this instance they cost 96.2 evaluations on average, against 89.2 overall.

### Exercise 6 (★★★) — Binary can beat Gray; the counting argument; Whitley (1999)

*Construction.* Let $f(i) = ((i \bmod 64) - 32)^2 + 0.001\,i$ on $\lbrace 0,\ldots,255\rbrace$. It has 4 local minima on
the integer line, at $i = 32, 96, 160, 224$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

K = 8
ints = np.arange(2**K)
gray_inv = np.empty_like(ints); gray_inv[ints ^ (ints >> 1)] = ints          # genotype -> integer under Gray
NB = ints[:, None] ^ (1 << np.arange(K))[None, :]

def local_minima(values):                                # values indexed by genotype; no strictly better neighbour
    return np.flatnonzero(np.all(values[NB] >= values[:, None], axis=1))

f = ((ints % 64) - 32) ** 2 + 0.001 * ints               # f on the integer line
line_min = [int(i) for i in ints if (i == 0 or f[i] < f[i - 1]) and (i == 255 or f[i] < f[i + 1])]
bin_min = local_minima(f)                                # genotype g decodes to integer g
gray_min = gray_inv[local_minima(f[gray_inv])]           # report the decoded integers
print("integer-line local minima:", line_min)
print("binary local minima (integers):", bin_min.tolist(), "| Gray local minima (integers):", sorted(gray_min.tolist()))

# counting argument: random rankings (uniformly random injective f), same function under both encodings
rng = np.random.default_rng(0)
nb_, ng_ = [], []
for _ in range(2000):
    r = rng.permutation(2**K).astype(float)             # r[i] = rank of integer i
    nb_.append(len(local_minima(r))); ng_.append(len(local_minima(r[gray_inv])))
nb_, ng_ = np.array(nb_), np.array(ng_)
print(f"mean #local minima over 2000 random rankings: binary {nb_.mean():.2f}, Gray {ng_.mean():.2f}, "
      f"theory 256/9 = {256 / 9:.2f};  binary has fewer than Gray on {100 * np.mean(nb_ < ng_):.1f}% of rankings")
```

```text
integer-line local minima: [32, 96, 160, 224]
binary local minima (integers): [31, 32] | Gray local minima (integers): [32, 96, 160, 224]
mean #local minima over 2000 random rankings: binary 28.53, Gray 28.34, theory 256/9 = 28.44;  binary has fewer than Gray on 44.0% of rankings
```

Standard binary induces **2 bit-flip local minima, Gray induces 4**. Under binary, flipping one of the two top bits maps
$i$ to $i \pm 64$ or $i \pm 128$, i.e. to another copy of the same parabola. The tilt $0.001\,i$ makes the lower copy
better, so the copies at 96, 160 and 224 are not minima, and only $i = 32$ survives. There is one extra, false minimum at
$i = 31$ ($00011111_2$), created by the Hamming cliff to $32 = 00100000_2$. Under Gray, flipping the top Gray bit maps $i$
to its mirror image $255 - i$ (for example $32 \leftrightarrow 223$), not to another copy. All four integer-line minima
remain minima. No false ones can appear, by the argument of Section 7: every integer that is not a local minimum on the
line has a strictly better integer neighbour $i \pm 1$, which is a Gray bit-flip neighbour.

*Why such functions must exist.* Take $f$ uniformly random among injective functions on 256 points, i.e. a uniformly
random ranking. Under *any* bijective encoding into $\lbrace 0,1\rbrace^8$ with the bit-flip neighbourhood, a point is a
local minimum iff it is the smallest of itself and its 8 neighbours. Its value is exchangeable with theirs, so this has
probability $1/9$. The expected number of local minima is therefore $256/9 \approx 28.44$ for binary *and* for Gray; the
simulation gives 28.53 and 28.34. Gray is strictly better on some functions: on every strictly unimodal function it has
1 minimum, while binary has up to 8 (Section 7). The averages are equal, so Gray must be strictly worse on some other
functions. In the random sample, binary had fewer minima than Gray on 44.0% of rankings. This is the NFL mechanism: an
encoding cannot be better on average over all functions, only on a class.

*Relation to Whitley (1999).* Whitley's "free lunch" result for Gray codes makes the class precise, by counting the local
optima that each encoding induces in aggregate. Over the set of functions (on the wrapped integer domain) with fewer
than the maximum possible number of optima, Gray coding induces fewer optima than standard binary. The counting identity
above then forces binary to be the better code in aggregate on the remaining, maximally multimodal functions. For
"reasonable" problems with few optima, Gray is the safer choice.

---

## Notebook 3 — Fitness Landscape Analysis

### Exercise 1 (★) — $P_0(1) = 1 - (K+1)/N$; ACF with adjacent neighbourhoods

One step flips a single bit, chosen uniformly among $N$. Contribution $c_i$ depends on the $K+1$ distinct bits
$\lbrace i\rbrace \cup V_i$, and it is unchanged iff the flipped bit is not one of them:

$$
P_0(1) = 1 - \frac{K+1}{N}.
$$

This is exact. It is also what the general formula gives, since $2^{-(K+1)}\sum_j \binom{K+1}{j}(1 - 2j/N) = 1 - \frac{2}{N}\cdot\frac{K+1}{2}$.

The derivation of $\rho(s) = (M P_0(s) - 1)/(M-1)$ used only two facts: (i) each contribution depends on exactly $K+1$
distinct bits, and (ii) distinct contributions have independent tables. Both hold for **adjacent** neighbourhoods
$V_i = \lbrace i+1, \ldots, i+K\rbrace \bmod N$. Hence

$$
\rho(1) = 1 - \frac{M}{M-1}\cdot\frac{K+1}{N}, \qquad M = 2^{K+1},
$$

in both models, and more generally the whole ACF is the same. The numerical check uses $N=64$, $K=4$, 20 landscapes of
each kind, and walks of 20 000 steps.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

class NK:
    """NK landscape; neighbourhood model 'random' (Notebook 3) or 'adjacent' (V_i = i+1..i+K mod N)."""
    def __init__(self, N, K, rng, model="random"):
        if model == "random":
            others = [rng.choice(np.delete(np.arange(N), i), K, replace=False) for i in range(N)]
        else:
            others = [(i + 1 + np.arange(K)) % N for i in range(N)]
        self.nbr = np.array([[i, *o] for i, o in enumerate(others)], dtype=int)
        self.table, self.w, self.N = rng.random((N, 2 ** (K + 1))), 1 << np.arange(K + 1), N
    def __call__(self, X):
        idx = (np.atleast_2d(X)[:, self.nbr] * self.w).sum(-1)
        return self.table[np.arange(self.N), idx].mean(axis=1)

def bitflip_walk(N, L, rng):
    flips = np.zeros((L, N), dtype=np.int64); flips[np.arange(L), rng.integers(0, N, L)] = 1
    return (rng.integers(0, 2, N) + np.cumsum(flips, axis=0)) % 2

def rho1(y):
    y = y - y.mean(); return np.mean(y[:-1] * y[1:]) / np.mean(y * y)

N, K, L, R = 64, 4, 20000, 20
M = 2 ** (K + 1)
rng = np.random.default_rng(0)
for model in ("random", "adjacent"):
    r = np.array([rho1(NK(N, K, rng, model)(bitflip_walk(N, L, rng))) for _ in range(R)])
    print(f"{model:8s}: rho(1) = {r.mean():.4f} +- {r.std(ddof=1) / np.sqrt(R):.4f}")
print(f"theory 1 - M/(M-1) (K+1)/N = {1 - M / (M - 1) * (K + 1) / N:.4f}")
```

```text
random  : rho(1) = 0.9191 +- 0.0007
adjacent: rho(1) = 0.9198 +- 0.0009
theory 1 - M/(M-1) (K+1)/N = 0.9194
```

The ACF is a second-order statistic, so it cannot distinguish the models. Their computational complexity differs
completely: a polynomial DP exists for adjacent neighbourhoods, while the random model is NP-complete for $K\ge2$. This is
a standard warning that correlation length is not a complete measure of hardness.

### Exercise 2 (★) — FDC of OneMax is $-1$; FDC of LeadingOnes in closed form

*OneMax.* The optimum is $1^N$ and $d(x) = N - u(x)$, so $f = u = N - d$ is an exact decreasing affine function of $d$.
For any $N\ge1$ we have $\mathrm{Var}(u) = N/4 \gt 0$, so the correlation is exactly $-1$.

*LeadingOnes.* Let $L(x)$ be the number of leading ones, with optimum $1^N$ and $d = N - u$. Under uniform $x$:

- $P(L \ge k) = 2^{-k}$ for $0\le k\le N$. Hence $\mathbb{E}L = \sum_{k=1}^N 2^{-k} = 1 - 2^{-N}$ and
  $\mathbb{E}L^2 = \sum_{k=1}^N (2k-1)2^{-k} = 3 - (2N+3)2^{-N}$. Therefore

$$
\mathrm{Var}(L) = 2 - (2N+1)2^{-N} - 4^{-N}.
$$

- Given $L = k \lt N$, bit $k+1$ is 0 and the remaining $N-k-1$ bits are uniform, so
  $\mathbb{E}[u\mid L] = \tfrac{N-1}{2} + \tfrac{L}{2} + \tfrac12\mathbf{1}[L=N]$. Since
  $\mathrm{Cov}(L, \mathbf{1}[L=N]) = 2^{-N}(N - \mathbb{E}L)$,

$$
\mathrm{Cov}(L,u) = \tfrac12\mathrm{Var}(L) + \tfrac12\,2^{-N}\,(N - \mathbb{E}L) = 1 - (N+2)\,2^{-N-1}.
$$

With $\sigma_d = \sqrt N/2$,

$$
\mathrm{FDC}_{\text{LeadingOnes}} = -\frac{\mathrm{Cov}(L,u)}{\sigma_L\,\sigma_d} = -\frac{2 - (N+2)2^{-N}}{\sqrt{N}\,\sqrt{2 - (2N+1)2^{-N} - 4^{-N}}} \;\sim\; -\sqrt{2/N}.
$$

```python
import sys; sys.path.insert(0, "..")
import numpy as np

def fdc_lo_formula(N):
    return -(2 - (N + 2) * 2.0**-N) / (np.sqrt(N) * np.sqrt(2 - (2 * N + 1) * 2.0**-N - 4.0**-N))

for N in (8, 12, 16):
    X = (np.arange(2**N)[:, None] >> np.arange(N)) & 1          # all bit strings
    u = X.sum(1)
    L = np.argmin(np.c_[X, np.zeros(2**N, int)], axis=1)        # leading ones = index of the first 0
    om = np.corrcoef(u, N - u)[0, 1]
    print(f"N={N:2d}: OneMax FDC {om:+.6f};  LeadingOnes FDC enumeration {np.corrcoef(L, N - u)[0, 1]:+.6f}, "
          f"formula {fdc_lo_formula(N):+.6f}")
Ns = np.arange(2, 400)
print("smallest N with |FDC_LeadingOnes| < 0.15:", Ns[np.abs(fdc_lo_formula(Ns)) < 0.15][0])
```

```text
N= 8: OneMax FDC -1.000000;  LeadingOnes FDC enumeration -0.498583, formula -0.498583
N=12: OneMax FDC -1.000000;  LeadingOnes FDC enumeration -0.408174, formula -0.408174
N=16: OneMax FDC -1.000000;  LeadingOnes FDC enumeration -0.353549, formula -0.353549
smallest N with |FDC_LeadingOnes| < 0.15: 89
```

Enumeration and formula agree to 6 digits. LeadingOnes is easy for a (1+1) EA, with expected runtime $\Theta(N^2)$, yet
its FDC tends to 0 and falls into the "difficult" band $\vert\mathrm{FDC}\vert \lt 0.15$ from $N = 89$ on. This is another
counterexample to reading FDC as a difficulty measure.

### Exercise 3 (★★) — Mean and variance of the number of local optima when $K = N-1$

Write $I_x$ for the indicator that $x$ is a local maximum. The $2^N$ values are i.i.d. and continuous, so all orderings
are equally likely. Let $p = 1/(N+1)$.

- $\mathbb{E} I_x = p$, since $x$ must be the largest of its closed neighbourhood of $N+1$ values. Hence
  $\mathbb{E}\sum_x I_x = 2^N/(N+1)$.
- $d(x,y) = 1$: $x$ and $y$ cannot both be maxima, since each would have to exceed the other. So $\mathrm{Cov} = -p^2$.
- $d(x,y) = 2$: the closed neighbourhoods share the 2 common neighbours, and their union has $2N$ points. The maximum of
  the union must be $x$ or $y$. Say it is $x$, with probability $\frac{1}{2N}$. Then $y \notin N[x]$ must be the largest
  of its own $N+1$ values, whose relative order is unaffected, with probability $\frac{1}{N+1}$. So
  $P(I_xI_y = 1) = 2\cdot\frac{1}{2N}\cdot\frac{1}{N+1} = \frac{1}{N(N+1)}$.
- $d(x,y) \ge 3$: the closed neighbourhoods are disjoint, so $I_x$ and $I_y$ are independent.

Each $x$ has $N$ points at distance 1 and $\binom N2$ at distance 2, so

$$
\mathrm{Var}\Big(\sum_x I_x\Big) = 2^N\Big[p(1-p) - Np^2 + \binom N2\Big(\frac{1}{N(N+1)} - p^2\Big)\Big] = 2^N\,\frac{N-1}{2(N+1)^2},
$$

because $p(1-p) - Np^2 = p\,(1 - (N+1)p) = 0$ and $\frac{1}{N(N+1)} - \frac{1}{(N+1)^2} = \frac{1}{N(N+1)^2}$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

rng = np.random.default_rng(0)
for N, runs in ((10, 400), (16, 60)):
    nb = np.arange(2**N)[:, None] ^ (1 << np.arange(N))[None, :]
    counts = []
    for _ in range(runs):
        f = rng.random(2**N)                               # i.i.d. landscape = NK with K = N-1
        counts.append(np.sum(np.all(f[nb] < f[:, None], axis=1)))
    c = np.array(counts, float)
    v = c.var(ddof=1)
    se_v = np.std((c - c.mean()) ** 2, ddof=1) / np.sqrt(runs)   # rough s.e. of the sample variance
    print(f"N={N:2d}, {runs} runs: mean {c.mean():.1f} (theory {2**N / (N + 1):.1f});  variance {v:.1f} +- {se_v:.1f} "
          f"(theory {2**N * (N - 1) / (2 * (N + 1) ** 2):.1f})")
N = 16
var = 2**N * (N - 1) / (2 * (N + 1) ** 2)
print(f"N=16: theoretical s.d. {np.sqrt(var):.1f}; s.e. of a mean of 5 instances {np.sqrt(var / 5):.1f}")
```

```text
N=10, 400 runs: mean 93.2 (theory 93.1);  variance 40.3 +- 2.8 (theory 38.1)
N=16, 60 runs: mean 3852.9 (theory 3855.1);  variance 2059.1 +- 349.2 (theory 1700.8)
N=16: theoretical s.d. 41.2; s.e. of a mean of 5 instances 18.4
```

For $N = 16$ the variance is 1700.8, a standard deviation of 41.2, so the mean of 5 instances has s.e. 18.4. The
notebook's 5 exhaustive instances had mean 3864.8 against 3855.1, a deviation of 0.5 theoretical s.e., and sample s.e.
11.7, which is compatible with 18.4 for a sample of five. The simulated variances agree with the formula within about
one standard error.

### Exercise 4 (★★) — NK$q$: strict optima and neutral networks vs $q$ ($N = 14$)

NK$q$ draws each table entry uniformly from $\lbrace 0, 1/q, \ldots, (q-1)/q\rbrace$. The block compares $N q f(x)$ as
exact integers, so that ties are exact ($q = \infty$ means continuous tables). Neutral networks are the connected
components of the graph that joins bit-flip neighbours of equal fitness, computed by union–find. The results are
averages over 3 landscapes per row, with all $2^{14}$ states enumerated.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

def nkq_values(N, K, q, rng):
    """All 2^N values of an NKq landscape as exact integers N*q*f(x) (q=None: continuous tables)."""
    nbr = np.array([[i, *rng.choice(np.delete(np.arange(N), i), K, replace=False)] for i in range(N)])
    table = rng.random((N, 2 ** (K + 1))) if q is None else rng.integers(0, q, (N, 2 ** (K + 1)))
    X = (np.arange(2**N)[:, None] >> np.arange(N)) & 1
    idx = (X[:, nbr] * (1 << np.arange(K + 1))).sum(-1)
    return table[np.arange(N), idx].sum(axis=1)

def neutral_networks(f, nb):
    """Sizes of the connected components of {x ~ y : y in N(x), f(x) = f(y)} (union-find)."""
    parent = np.arange(len(f))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(nb.shape[1]):
        for a in np.flatnonzero(f[nb[:, i]] == f):
            b = nb[a, i]
            if a < b and find(a) != find(b):
                parent[find(a)] = find(b)
    return np.unique([find(a) for a in range(len(f))], return_counts=True)[1]

N, REPS = 14, 3
nb = np.arange(2**N)[:, None] ^ (1 << np.arange(N))[None, :]
rng = np.random.default_rng(0)
print(" K   q   strict max  non-strict max  mean network size  largest network")
for K in (2, 6):
    for q in (2, 3, 4, 8, None):
        rows = []
        for _ in range(REPS):
            f = nkq_values(N, K, q, rng)
            sizes = neutral_networks(f, nb)
            rows.append((np.sum(np.all(f[nb] < f[:, None], 1)), np.sum(np.all(f[nb] <= f[:, None], 1)),
                         sizes.mean(), sizes.max()))
        r = np.mean(rows, axis=0)
        print(f"{K:2d}  {'inf' if q is None else q:>3}  {r[0]:10.1f}  {r[1]:14.1f}  {r[2]:17.1f}  {r[3]:15.0f}")
```

```text
 K   q   strict max  non-strict max  mean network size  largest network
 2    2         2.0           253.3               54.3             2526
 2    3         2.7            77.0                8.0              746
 2    4         8.0            70.0                4.2              130
 2    8         9.7            37.7                1.7               21
 2  inf        13.7            13.7                1.0                1
 6    2        58.3           551.7               17.5             3284
 6    3        92.0           365.3                4.9             1752
 6    4       106.7           306.7                2.8              722
 6    8       134.7           226.0                1.5               25
 6  inf       174.7           174.7                1.0                1
```

Decreasing $q$ adds neutrality. Strict local optima almost disappear: for $K=2$, $q=2$ there are only 2 strict local
maxima per landscape on average. Meanwhile neutral networks percolate, and the giant component holds 15% ($K=2$) to 20%
($K=6$) of the whole space at $q = 2$. Non-strict local maxima, i.e. points with no strictly better neighbour, become
*more* numerous, because they now sit on plateaus. A search that accepts sideways moves can drift along these networks,
as in the MAX-SAT result of the notebook.

### Exercise 5 (★★) — Basin-transition LON vs escape-edge LON

For basins $B_a$ of best-improvement ascent,

$$
p_{ab} = \frac{1}{\vert B_a\vert}\sum_{x\in B_a}\frac{1}{N}\sum_{y \in N(x)}\mathbf{1}[y \in B_b] .
$$

The block rebuilds the notebook's landscape, NK($N=10$, $K=4$) from `default_rng(7)`, and both networks.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

# Notebook 3's LON landscape: NK(N=10, K=4) built from default_rng(7) with the same draws as NKLandscape
N, K = 10, 4
rng = np.random.default_rng(7)
nbr = np.array([[i, *rng.choice(np.delete(np.arange(N), i), K, replace=False)] for i in range(N)])
table = rng.random((N, 2 ** (K + 1)))
X = (np.arange(2**N)[:, None] >> np.arange(N)) & 1
f = table[np.arange(N), (X[:, nbr] * (1 << np.arange(K + 1))).sum(-1)].mean(axis=1)
nb = np.arange(2**N)[:, None] ^ (1 << np.arange(N))[None, :]
st = np.arange(2**N)
step = np.where(f[nb].max(1) > f, nb[st, np.argmax(f[nb], 1)], st)      # best-improvement ascent
att = step.copy()
while not np.array_equal(att[att], att):
    att = att[att]
optima = np.unique(att); n_opt = len(optima)
lab = np.searchsorted(optima, att)                                       # basin label of every state
g = int(np.searchsorted(optima, np.argmax(f)))

# basin-transition LON: p_ab = P(random bit flip from a uniform point of basin a lands in basin b)
Wb = np.zeros((n_opt, n_opt))
np.add.at(Wb, (np.repeat(lab, N), lab[nb].ravel()), 1)
Pb = Wb / (np.bincount(lab)[:, None] * N)
# escape-edge LON of the notebook: 1- and 2-bit kicks from each optimum, then ascent
pert = [1 << i for i in range(N)] + [(1 << i) | (1 << j) for i in range(N) for j in range(i + 1, N)]
We = np.zeros((n_opt, n_opt))
for a, o in enumerate(optima):
    for p in pert:
        We[a, lab[o ^ p]] += 1
Pe = We / len(pert)

def reaches(P, target):
    seen, frontier = {target}, [target]
    while frontier:
        v = frontier.pop()
        for u in np.flatnonzero(P[:, v] > 0):
            if u not in seen:
                seen.add(int(u)); frontier.append(int(u))
    return len(seen)

def stationary(P):
    w, V = np.linalg.eig(P.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1))]); return v / v.sum()

off = ~np.eye(n_opt, dtype=bool)
print(f"{n_opt} local optima; global-optimum basin holds {np.mean(lab == g):.3f} of the states")
for name, P in (("basin-transition", Pb), ("escape edges", Pe)):
    print(f"{name:16s}: off-diagonal edges {np.count_nonzero(P[off])}, median self-loop {np.median(np.diag(P)):.3f}, "
          f"optima reaching the global optimum {reaches(P, g)}/{n_opt}, stationary mass on it {stationary(P)[g]:.3f}")
print(f"correlation of off-diagonal weights: {np.corrcoef(Pb[off], Pe[off])[0, 1]:.3f}")
pi = np.bincount(lab) / 2**N
print(f"reversibility check: max |pi_a p_ab - pi_b p_ba| = {np.abs(pi[:, None] * Pb - (pi[:, None] * Pb).T).max():.1e};"
      f"  max |stationary - basin fractions| = {np.abs(stationary(Pb) - pi).max():.1e}")
```

```text
25 local optima; global-optimum basin holds 0.087 of the states
basin-transition: off-diagonal edges 404, median self-loop 0.322, optima reaching the global optimum 25/25, stationary mass on it 0.087
escape edges    : off-diagonal edges 203, median self-loop 0.291, optima reaching the global optimum 25/25, stationary mass on it 0.101
correlation of off-diagonal weights: 0.833
reversibility check: max |pi_a p_ab - pi_b p_ba| = 1.7e-18;  max |stationary - basin fractions| = 3.3e-16
```

The escape-edge numbers (25 optima, 203 edges, median self-loop 0.29) are Notebook 3's. The basin-transition network
is denser, because every boundary point of a basin creates an edge. The escape network is sparser, but it describes
what an iterated local search can actually do from an optimum. Both networks let every optimum reach the global one,
and their off-diagonal weights are strongly correlated (0.833).

*A small theorem explains the 0.087.* Let $W_{ab}$ be the number of ordered neighbour pairs $(x,y)$ with
$x\in B_a$ and $y \in B_b$. The neighbour relation is symmetric, so $W_{ab} = W_{ba}$. With $\pi_a = \vert B_a\vert/2^N$,

$$
\pi_a\,p_{ab} = \frac{W_{ab}}{N\,2^N}
$$

is symmetric in $a,b$. The basin-transition chain is therefore reversible, and its stationary distribution is exactly
the basin-size distribution: the lumped image of the uniform stationary distribution of the random walk. The block
confirms both facts to rounding error, and the global basin holds 8.7% of the states. The escape chain has no such
identity. Here it puts slightly more mass on the global optimum (0.101), because 2-bit kicks leave the small basins more
easily than they leave the large ones.

### Exercise 6 (★★★) — ACF of the Sphere under the reflected Gaussian walk; an exactly exponential ACF

*Spectral structure of the walk.* Take one coordinate on $[-a, a]$, with a step $x \mapsto x + \sigma z$ followed by
reflection. The transition kernel is the Gaussian kernel with its mirror images (method of images). Consider the functions

$$
\phi_k(x) = \cos\Big(\frac{k\pi (x+a)}{2a}\Big), \qquad k \ge 1 .
$$

Their even $4a$-periodic extension is a pure cosine wave, so they satisfy the Neumann condition at $\pm a$. Convolving a
cosine of frequency $\omega$ with a Gaussian multiplies it by $e^{-\sigma^2\omega^2/2}$. Hence $\phi_k$ is an
eigenfunction of one step:

$$
\mathbb{E}\big[\phi_k(x_{t+1}) \mid x_t\big] = \lambda_k\,\phi_k(x_t), \qquad \lambda_k = \exp\Big(-\frac{\sigma^2 k^2 \pi^2}{8a^2}\Big).
$$

This is exact as long as a step never crosses both walls, which a single reflection assumes anyway. The stationary law
is uniform.

*Sphere.* On $[-a,a]$ the cosine series of $x^2$ is

$$
x^2 = \frac{a^2}{3} + \sum_{m\ge1}\frac{4a^2}{m^2\pi^2}(-1)^m\cos\frac{m\pi x}{a},
$$

and $(-1)^m\cos(m\pi x/a) = \phi_{2m}(x)$. So only the even modes appear, with coefficients $4a^2/(m^2\pi^2)$ and
eigenvalues $\lambda_{2m} = \exp(-\sigma^2 m^2\pi^2/(2a^2))$. The $\phi$'s are orthogonal with $\mathbb{E}\phi^2 = 1/2$,
and $\mathrm{Var}(x^2) = a^4/5 - a^4/9 = 4a^4/45$. Coordinates are independent and identical, so the ACF of the sum
equals that of one coordinate:

$$
\rho_{\text{sphere}}(s) = \frac{90}{\pi^4}\sum_{m\ge1}\frac{1}{m^4}\exp\Big(-\frac{s\,\sigma^2 m^2 \pi^2}{2a^2}\Big).
$$

At $s = 0$ this equals 1, because $\sum_m m^{-4} = \pi^4/90$ (Parseval confirms the variance). The slow decay in the
notebook's figure is the $m=1$ mode: 92.4% of the variance ($90/\pi^4$) relaxes on the *diffusive* time scale
$2a^2/(\sigma^2\pi^2) = 2125$ steps for $\sigma = 0.05$ and $a = 5.12$. The prediction $\rho(10) = 0.993$,
$\rho(100) = 0.939$ is close to the notebook's 0.995 and 0.953. That walk spans only about 28 relaxation times, so its
ACF has a sampling error of a few hundredths.

*Verification with a faster walk, and an exactly exponential ACF.* The block checks the formula with $\sigma = 0.5$ in
$d = 10$ (8 walks of 40 000 steps). It then checks the design below.

Use a single eigenfunction, $f(x) = \sum_{i=1}^d \cos(m\pi x_i/a) = \sum_i (-1)^m \phi_{2m}(x_i)$. Then
$\rho(s) = \lambda_{2m}^s$ exactly, with

$$
\ell = -\frac{1}{\ln\lambda_{2m}} = \frac{2a^2}{\sigma^2 m^2\pi^2} \quad\Longleftrightarrow\quad \sigma = \frac{a}{m\pi}\sqrt{\frac{2}{\ell}} .
$$

For $\ell = 25$ and $m = 3$ this gives $\sigma = 0.1537$.

```python
import sys; sys.path.insert(0, "..")
import numpy as np

a = 5.12

def rho_sphere(s, sigma, M=200):
    m = np.arange(1, M + 1)[:, None]
    return (90 / np.pi**4) * np.sum(m**-4.0 * np.exp(-np.outer(1, np.atleast_1d(s)) * sigma**2 * m**2 * np.pi**2 / (2 * a**2)), axis=0)

def gaussian_walk(d, L, sigma, rng):
    """Reflected Gaussian walk on [-a, a]^d (as in Notebook 3)."""
    X = np.empty((L, d)); x = rng.uniform(-a, a, d); Z = sigma * rng.standard_normal((L, d))
    for t in range(L):
        x = x + Z[t]
        x = np.where(x > a, 2 * a - x, x); x = np.where(x < -a, -2 * a - x, x)
        X[t] = x
    return X

def acf(y, lags):
    y = y - y.mean(); v = np.mean(y * y)
    return np.array([np.mean(y[:-s] * y[s:]) / v for s in lags])

print(f"sigma=0.05: slow-mode share 90/pi^4 = {90 / np.pi**4:.3f}, time scale 2a^2/(sigma^2 pi^2) = "
      f"{2 * a**2 / (0.05**2 * np.pi**2):.0f} steps;  predicted rho(10) = {rho_sphere(10, 0.05)[0]:.3f}, "
      f"rho(100) = {rho_sphere(100, 0.05)[0]:.3f}")

rng = np.random.default_rng(0)
lags = np.array([1, 10, 100, 300, 1000])
W = [gaussian_walk(10, 40000, 0.5, rng) for _ in range(8)]
A = np.array([acf(np.sum(X**2, 1), lags) for X in W])
print("sigma=0.5, d=10, 8 walks x 40000 steps (Sphere):")
for s, m_, se, th in zip(lags, A.mean(0), A.std(0, ddof=1) / np.sqrt(8), rho_sphere(lags, 0.5)):
    print(f"  lag {s:4d}: empirical {m_:+.3f} ({se:.3f})   formula {th:.3f}")

ell, m = 25.0, 3                                          # prescribed correlation length, mode index
sigma = a / (m * np.pi) * np.sqrt(2 / ell)
lags = np.array([1, 10, 25, 60])
W = [gaussian_walk(10, 40000, sigma, rng) for _ in range(8)]
A = np.array([acf(np.sum(np.cos(m * np.pi * X / a), 1), lags) for X in W])
print(f"f(x) = sum_i cos({m} pi x_i / a), sigma = {sigma:.4f}:")
for s, m_, se in zip(lags, A.mean(0), A.std(0, ddof=1) / np.sqrt(8)):
    print(f"  lag {s:3d}: empirical {m_:.3f} ({se:.3f})   exp(-s/ell) = {np.exp(-s / ell):.3f}")
```

```text
sigma=0.05: slow-mode share 90/pi^4 = 0.924, time scale 2a^2/(sigma^2 pi^2) = 2125 steps;  predicted rho(10) = 0.993, rho(100) = 0.939
sigma=0.5, d=10, 8 walks x 40000 steps (Sphere):
  lag    1: empirical +0.939 (0.000)   formula 0.939
  lag   10: empirical +0.588 (0.003)   formula 0.586
  lag  100: empirical +0.009 (0.006)   formula 0.008
  lag  300: empirical -0.010 (0.009)   formula 0.000
  lag 1000: empirical -0.002 (0.004)   formula 0.000
f(x) = sum_i cos(3 pi x_i / a), sigma = 0.1537:
  lag   1: empirical 0.961 (0.001)   exp(-s/ell) = 0.961
  lag  10: empirical 0.670 (0.005)   exp(-s/ell) = 0.670
  lag  25: empirical 0.367 (0.007)   exp(-s/ell) = 0.368
  lag  60: empirical 0.084 (0.008)   exp(-s/ell) = 0.091
```

Both checks agree within about one standard error at every lag. The same analysis explains why Rastrigin's
$\cos(2\pi x)$ has an ACF that is only *approximately* $e^{-2\pi^2\sigma^2 s}$ on $[-5.12, 5.12]$. Its frequency
$2\pi$ corresponds to $m = 2a = 10.24$, which is not an integer, so it is not an exact eigenfunction of the reflected
walk.

---

## Lab — Benchmark Suite, Experiment Harness and Baselines

### Exercise 1 (★) — ERT is the expected cost of the independently restarted algorithm

Run the algorithm repeatedly with independent seeds until the first successful run. Let $G$ be the number of failed
runs before it. Each run succeeds with probability $p_s$ independently, so $G$ is geometric with
$\mathbb{E}G = (1-p_s)/p_s$. The total cost is

$$
T = \sum_{i=1}^{G} \mathrm{RT}^{\text{fail}}_i + \mathrm{RT}^{\text{succ}},
$$

where the $\mathrm{RT}^{\text{fail}}_i$ are i.i.d. with the law of a run's cost conditioned on failure. Conditionally on
the success/failure labels of the runs, the costs are independent, and $G$ is a function of the labels only. Wald's
identity (or simply conditioning on $G$) therefore gives

$$
\mathbb{E}T = \mathbb{E}G\;\mathbb{E}\,\mathrm{RT}^{\text{fail}} + \mathbb{E}\,\mathrm{RT}^{\text{succ}} = \frac{1-p_s}{p_s}\,\mathbb{E}\,\mathrm{RT}^{\text{fail}} + \mathbb{E}\,\mathrm{RT}^{\text{succ}} .
$$

The plug-in estimate from $n$ runs, $n_s$ of them successful, is

$$
\frac{\sum_{\text{succ}}\mathrm{RT}}{n_s} + \frac{n - n_s}{n_s}\cdot\frac{\sum_{\text{fail}}\mathrm{RT}}{n - n_s} = \frac{\sum_{\text{all}}\mathrm{RT}}{n_s},
$$

which is exactly the harness's `ert`.

### Exercise 2 (★) — $\Delta_{12}\equiv0$ implies additive separability; Sphere vs Ellipsoid under rotation

Fix a reference point $(b_1, b_2)$. Setting $a = (x_1, x_2)$ in $\Delta_{12} = 0$ gives
$f(x_1,x_2) + f(b_1,b_2) - f(x_1,b_2) - f(b_1,x_2) = 0$. Hence

$$
f(x_1, x_2) = \underbrace{f(x_1, b_2) - f(b_1, b_2)}_{g(x_1)} + \underbrace{f(b_1, x_2)}_{h(x_2)} .
$$

*Sphere.* For orthogonal $R$, $\Vert R x\Vert^2 = \Vert x\Vert^2 = \sum_i x_i^2$, which is separable for every rotation.

*Ellipsoid.* $f(Rx) = x^\top A x$ with $A = R^\top W R$ and $W$ diagonal with distinct entries. For a quadratic form,

$$
\Delta_{12} = 2A_{12}(a_1-b_1)(a_2-b_2),
$$

which vanishes identically iff $A_{12} = 0$. In $d = 2$ with distinct eigenvalues, that means the coordinate axes are
eigenvectors of $A$, i.e. $R$ is a signed permutation. A generic rotation therefore destroys separability.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import sphere, ellipsoid, shifted_rotated, random_rotation

rng = np.random.default_rng(0)
R = random_rotation(2, rng)
A = R.T @ np.diag([1.0, 1e6]) @ R                        # f(Rx) = x^T A x for the 2-D Ellipsoid (condition 10^6)
a, b = rng.uniform(-5, 5, 2), rng.uniform(-5, 5, 2)
def delta12(f):
    return f(np.array([a[0], a[1]])) + f(np.array([b[0], b[1]])) - f(np.array([a[0], b[1]])) - f(np.array([b[0], a[1]]))
ell_rot = shifted_rotated(ellipsoid, np.zeros(2), R)
sph_rot = shifted_rotated(sphere, np.zeros(2), R)
print(f"rotated Ellipsoid: Delta_12 = {delta12(ell_rot):.4f};  formula 2 A_12 (a1-b1)(a2-b2) = "
      f"{2 * A[0, 1] * (a[0] - b[0]) * (a[1] - b[1]):.4f}")
print(f"rotated Sphere:    Delta_12 = {delta12(sph_rot):.1e}")
```

```text
rotated Ellipsoid: Delta_12 = 1431680.6799;  formula 2 A_12 (a1-b1)(a2-b2) = 1431680.6799
rotated Sphere:    Delta_12 = 3.6e-15
```

For a Haar rotation with condition number $10^6$, the measured $\Delta_{12}$ equals the formula to all digits shown,
while the rotated Sphere's is zero to rounding error.

### Exercise 3 (★★) — Next-order correction to $\mathbb{E}[M_n]$

Write $(1-F)^n = e^{n\ln(1-F)} = e^{-nF}\big(1 - \tfrac{n}{2}F^2 + O(nF^3 + n^2F^4)\big)$ with $F = c\varepsilon^{d/2}$,
and substitute $t = nc\,\varepsilon^{d/2}$, so that $nF^2 = t^2/n$ and
$d\varepsilon = \tfrac2d (nc)^{-2/d} t^{2/d-1}dt$:

$$
\int_0^\infty e^{-nF}\,\frac{n}{2}F^2\,d\varepsilon = \frac{1}{2n}\cdot\frac{2}{d}(nc)^{-2/d}\,\Gamma\Big(2 + \frac2d\Big) = \frac{d+2}{d^2 n}\,\Gamma\Big(1+\frac2d\Big)(nc)^{-2/d},
$$

using $\Gamma(2 + 2/d) = (1 + 2/d)\,\Gamma(1+2/d)$. The remaining terms contribute $O(n^{-2})$ relative to the leading
one, so

$$
\mathbb{E}[M_n] = \Gamma\Big(1+\frac2d\Big)(nc)^{-2/d}\Big(1 - \frac{d+2}{d^2\,n} + O(n^{-2})\Big),
$$

valid while the relevant ball lies inside the box. The block compares both approximations with the `quad` values of
the Lab, for the cases whose neglected tail is negligible.

```python
import sys; sys.path.insert(0, "..")
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

a = 5.12
print(" d       n        exact    rel.err leading   rel.err corrected")
for d, n in ((2, 100), (2, 1000), (5, 100), (5, 10_000), (10, 10_000)):
    c = math.pi ** (d / 2) / gamma(d / 2 + 1) / (2 * a) ** d
    lead = gamma(1 + 2 / d) * (n * c) ** (-2 / d)
    U = min(a * a, 50 * lead)                            # split so quad resolves the peak at 0 (as in the Lab)
    g = lambda e: (1 - c * e ** (d / 2)) ** n
    exact = quad(g, 0, U, limit=200, epsabs=0, epsrel=1e-13)[0] + quad(g, U, a * a, limit=200, epsabs=0, epsrel=1e-13)[0]
    assert (d - 1) * a * a * (1 - c * a**d) ** n < 1e-6 * exact     # neglected tail beyond the inscribed ball
    corr = lead * (1 - (d + 2) / (d * d * n))
    print(f"{d:2d} {n:7d}  {exact:11.6g}   {(lead - exact) / exact:+.2e}          {(corr - exact) / exact:+.2e}")
```

```text
 d       n        exact    rel.err leading   rel.err corrected
 2     100     0.330467   +1.00e-02          -1.00e-04
 2    1000    0.0333439   +1.00e-03          -1.00e-06
 5     100      7.56695   +2.80e-03          -1.23e-05
 5   10000       1.2026   +2.80e-05          -1.23e-09
10   10000      12.6533   +1.20e-05          -3.52e-10
```

The leading term overestimates by exactly $(d+2)/(d^2n)$ to first order: 0.01 for $d = 2$ and $n = 100$. The corrected
formula reduces the error by a further factor of order $1/n$, as the $O(n^{-2})$ remainder predicts.

### Exercise 4 (★★) — The 1/5 success rule

The block copies the Lab's harness and hill climber, and adds a (1+1) hill climber with Rechenberg's 1/5 rule. After
every $d$ steps, $\sigma$ is divided by $c = 0.85$ if more than a fifth of them were strict improvements, and
multiplied by $c$ if fewer were. The settings are those of the Lab: $d=10$, $B=4000$, 11 seeds, starting step
$0.02\,(u-\ell)$. The table gives final errors, median [IQR], and the p-value of a one-sided Mann–Whitney test that
the 1/5 rule's errors are smaller.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import BENCHMARKS, BudgetedObjective, BudgetExhausted, best_so_far_on_grid

# ---- the Lab's course harness and baselines (copied)
def run_optimizer(opt, f, budget, seed, lo, hi, d, f_opt=0.0, **params):
    F = BudgetedObjective(f, budget)
    try:
        opt(F, np.random.default_rng(seed), lo, hi, d, **params)
    except BudgetExhausted:
        pass
    ev, best = F.trace()
    return ev, best - f_opt

def run_experiment(opt, f, budget, seeds, lo, hi, d, f_opt=0.0, grid=None, **params):
    if grid is None:
        grid = np.unique(np.round(np.logspace(0, np.log10(budget), 80)).astype(int))
    traces = [run_optimizer(opt, f, budget, s, lo, hi, d, f_opt, **params) for s in seeds]
    E = np.array([best_so_far_on_grid(ev, err, grid) for ev, err in traces])
    return {"grid": grid, "errors": E, "traces": traces, "budget": budget}

def random_search(F, rng, lo, hi, d, batch=100):
    while True:
        k = min(batch, F.remaining)
        if k <= 0:
            raise BudgetExhausted
        F(rng.uniform(lo, hi, (k, d)))

def hill_climber(F, rng, lo, hi, d, sigma_frac=0.02, patience=None):
    sigma = sigma_frac * (hi - lo)
    while True:
        x = rng.uniform(lo, hi, d); fx = F(x); fails = 0
        while patience is None or fails < patience:
            y = np.clip(x + sigma * rng.standard_normal(d), lo, hi)
            fy = F(y)
            fails = 0 if fy < fx else fails + 1
            if fy <= fx:
                x, fx = y, fy

def rr_local_search(F, rng, lo, hi, d, sigma_frac=0.02, patience=200):
    hill_climber(F, rng, lo, hi, d, sigma_frac, patience)
# ---- end of copied harness
from scipy.stats import mannwhitneyu

def one_fifth(F, rng, lo, hi, d, sigma_frac=0.02, c=0.85):
    """(1+1) hill climber with Rechenberg's 1/5 rule: every d steps, sigma /= c if the success rate > 1/5,
    sigma *= c if < 1/5 (success = strict improvement)."""
    sigma = sigma_frac * (hi - lo); x = rng.uniform(lo, hi, d); fx = F(x); succ = k = 0
    while True:
        y = np.clip(x + sigma * rng.standard_normal(d), lo, hi); fy = F(y); k += 1
        if fy <= fx:
            succ += fy < fx; x, fx = y, fy
        if k == d:
            rate = succ / d
            sigma = sigma / c if rate > 0.2 else (sigma * c if rate < 0.2 else sigma)
            sigma = min(sigma, hi - lo); succ = k = 0

D, B, SEEDS = 10, 4000, range(11)
fmt = lambda e: f"{np.median(e):9.3g} [{np.percentile(e, 25):8.3g}, {np.percentile(e, 75):8.3g}]"
print(f"{'function':11s}{'fixed-step HC':>32s}{'1/5 rule':>32s}   p (1/5 < fixed)")
for name, bm in BENCHMARKS.items():
    e = {}
    for label, opt in (("fixed", hill_climber), ("1/5", one_fifth)):
        e[label] = run_experiment(opt, bm.f, B, SEEDS, bm.lower, bm.upper, D, bm.f_opt)["errors"][:, -1]
    p = mannwhitneyu(e["1/5"], e["fixed"], alternative="less").pvalue
    print(f"{name:11s}{fmt(e['fixed']):>32s}{fmt(e['1/5']):>32s}   {p:.1e}")
```

```text
function                      fixed-step HC                        1/5 rule   p (1/5 < fixed)
sphere          0.0456 [  0.0349,   0.0586]   3.12e-37 [8.85e-38, 1.95e-36]   4.1e-05
ellipsoid     1.67e+03 [     973,  2.3e+03]   1.06e+03 [     666, 1.69e+03]   6.5e-02
rosenbrock        17.5 [    15.8,     23.8]       4.31 [     3.3,     4.67]   4.1e-05
rastrigin          102 [    92.8,      109]       93.5 [    78.1,      101]   1.2e-01
ackley            20.2 [      20,     20.3]       19.6 [    19.5,     19.7]   2.9e-03
griewank          1.17 [    1.15,      1.2]      0.121 [  0.0566,    0.149]   4.1e-05
schwefel      2.02e+03 [1.88e+03, 2.11e+03]   1.96e+03 [1.74e+03, 2.01e+03]   2.6e-01
levy              23.5 [    19.9,     26.1]       21.6 [    17.6,     25.6]   3.0e-01
```

The fixed-step column reproduces the Lab's (1+1) hill climber exactly (same seeds). Step-size adaptation **removes the
stagnation** on the Sphere completely, giving linear convergence down to $10^{-37}$. It also helps greatly on
Griewank, whose large-scale structure is a quadratic bowl, and on Rosenbrock. On the Ellipsoid with condition number
$10^6$ the improvement is not significant ($p = 0.065$), because an *isotropic* step must adapt to the smallest
curvature scale. That is the motivation for covariance adaptation (CMA-ES, Week 6). On Rastrigin the improvement is not
significant either ($p = 0.12$), and the same holds on Schwefel and Levy. The rule shrinks $\sigma$ inside the first
basin it finds, which makes it a faster *local* optimiser but not a better global one. Restarts or a population are
still needed.

### Exercise 5 (★★) — ECDF of runtimes aggregated over the eight functions

For each function, 15 targets are log-spaced between the median initial error and the best final error of any
algorithm, floored at $10^{-8}$ of the initial error. The ECDF at budget $b$ is the fraction of (function, target, run)
triples solved within $b$ evaluations. The last column is the area under the ECDF against $\log b$, normalised by
$\log 4000$, a one-number summary of the whole curve (1 would mean every target solved at the first evaluation).
The block re-runs the Lab's three baselines (about 30 s), prints the table and draws the ECDF plot.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import BENCHMARKS, BudgetedObjective, BudgetExhausted, best_so_far_on_grid

# ---- the Lab's course harness and baselines (copied)
def run_optimizer(opt, f, budget, seed, lo, hi, d, f_opt=0.0, **params):
    F = BudgetedObjective(f, budget)
    try:
        opt(F, np.random.default_rng(seed), lo, hi, d, **params)
    except BudgetExhausted:
        pass
    ev, best = F.trace()
    return ev, best - f_opt

def run_experiment(opt, f, budget, seeds, lo, hi, d, f_opt=0.0, grid=None, **params):
    if grid is None:
        grid = np.unique(np.round(np.logspace(0, np.log10(budget), 80)).astype(int))
    traces = [run_optimizer(opt, f, budget, s, lo, hi, d, f_opt, **params) for s in seeds]
    E = np.array([best_so_far_on_grid(ev, err, grid) for ev, err in traces])
    return {"grid": grid, "errors": E, "traces": traces, "budget": budget}

def random_search(F, rng, lo, hi, d, batch=100):
    while True:
        k = min(batch, F.remaining)
        if k <= 0:
            raise BudgetExhausted
        F(rng.uniform(lo, hi, (k, d)))

def hill_climber(F, rng, lo, hi, d, sigma_frac=0.02, patience=None):
    sigma = sigma_frac * (hi - lo)
    while True:
        x = rng.uniform(lo, hi, d); fx = F(x); fails = 0
        while patience is None or fails < patience:
            y = np.clip(x + sigma * rng.standard_normal(d), lo, hi)
            fy = F(y)
            fails = 0 if fy < fx else fails + 1
            if fy <= fx:
                x, fx = y, fy

def rr_local_search(F, rng, lo, hi, d, sigma_frac=0.02, patience=200):
    hill_climber(F, rng, lo, hi, d, sigma_frac, patience)
# ---- end of copied harness
import matplotlib.pyplot as plt

D, B, SEEDS = 10, 4000, range(11)
algos = {"random search": (random_search, {}), "(1+1) HC": (hill_climber, {}), "RR-LS": (rr_local_search, {})}
res = {(fn, an): run_experiment(opt, bm.f, B, SEEDS, bm.lower, bm.upper, D, bm.f_opt, **kw)
       for fn, bm in BENCHMARKS.items() for an, (opt, kw) in algos.items()}

def ecdf_runtimes(fn_list, an, grid, n_targets=15):
    """Fraction of (function, target, run) triples solved within each budget of `grid`. Per function, the targets
    are log-spaced from the median initial error to the best final error of any algorithm (floored at 1e-8 x initial)."""
    rts = []
    for fn in fn_list:
        e0 = np.median(np.concatenate([res[fn, a]["errors"][:, 0] for a in algos]))
        e_best = min(res[fn, a]["errors"][:, -1].min() for a in algos)
        targets = np.logspace(np.log10(e0), np.log10(max(e_best, 1e-8 * e0)), n_targets)
        for ev, err in res[fn, an]["traces"]:
            for tg in targets:
                hit = np.flatnonzero(err <= tg); rts.append(ev[hit[0]] if hit.size else np.inf)
    rts = np.array(rts)
    return np.array([np.mean(rts <= g) for g in grid])

show = [10, 30, 100, 300, 1000, 3000, 4000]
fine = np.union1d(np.unique(np.round(np.logspace(0, np.log10(B), 400)).astype(int)), show)
print(f"{'':15s}" + "".join(f"{g:>7d}" for g in show) + "   area (log budget)")
fig, ax = plt.subplots()
for an in algos:
    curve = ecdf_runtimes(list(BENCHMARKS), an, fine)
    area = np.trapezoid(curve, np.log(fine)) / np.log(B)            # normalised area under the ECDF on log scale
    print(f"{an:15s}" + "".join(f"{curve[np.searchsorted(fine, g)]:7.3f}" for g in show) + f"   {area:.3f}")
    ax.step(fine, curve, where="post", label=an)
ax.set(xscale="log", xlabel="function evaluations", ylabel="fraction of (function, target, run) solved",
       title=f"ECDF of runtimes, 8 functions x 15 targets x {len(SEEDS)} runs, d = {D}")
ax.legend(); plt.show()
```

```text
                    10     30    100    300   1000   3000   4000   area (log budget)
random search    0.202  0.246  0.314  0.363  0.415  0.464  0.476   0.282
(1+1) HC         0.110  0.202  0.395  0.607  0.655  0.691  0.698   0.357
RR-LS            0.110  0.202  0.395  0.607  0.687  0.765  0.783   0.368
```

Random search leads for the first few tens of evaluations: it covers the easy targets fastest. The hill climbers
overtake it by 100 evaluations. RR-LS and HC are identical until the first restart, which needs at least 200
consecutive failures, and then RR-LS keeps solving targets while HC plateaus. The ECDF shows in one picture what several
fixed-budget tables would.

### Exercise 6 (★★★) — Tuning $(\sigma, P)$ of RR-LS with the harness, and the danger of tuning on the test set

The objective of the tuning is the mean over {Sphere, Rastrigin} of the median $\log_{10}$ final error, over 5 training
seeds with $B = 2000$ and $d = 10$. The tuner is random search over $\log_{10}\sigma_{\text{frac}} \in [-3, -0.5]$ and
$\log_{10}P \in [1, 3]$, with 24 candidates. The tuned and default configurations are then re-evaluated on 11 fresh
seeds and on four held-out functions.

```python
import sys; sys.path.insert(0, "..")
import numpy as np
from utils import BENCHMARKS, BudgetedObjective, BudgetExhausted, best_so_far_on_grid

# ---- the Lab's course harness and baselines (copied)
def run_optimizer(opt, f, budget, seed, lo, hi, d, f_opt=0.0, **params):
    F = BudgetedObjective(f, budget)
    try:
        opt(F, np.random.default_rng(seed), lo, hi, d, **params)
    except BudgetExhausted:
        pass
    ev, best = F.trace()
    return ev, best - f_opt

def run_experiment(opt, f, budget, seeds, lo, hi, d, f_opt=0.0, grid=None, **params):
    if grid is None:
        grid = np.unique(np.round(np.logspace(0, np.log10(budget), 80)).astype(int))
    traces = [run_optimizer(opt, f, budget, s, lo, hi, d, f_opt, **params) for s in seeds]
    E = np.array([best_so_far_on_grid(ev, err, grid) for ev, err in traces])
    return {"grid": grid, "errors": E, "traces": traces, "budget": budget}

def random_search(F, rng, lo, hi, d, batch=100):
    while True:
        k = min(batch, F.remaining)
        if k <= 0:
            raise BudgetExhausted
        F(rng.uniform(lo, hi, (k, d)))

def hill_climber(F, rng, lo, hi, d, sigma_frac=0.02, patience=None):
    sigma = sigma_frac * (hi - lo)
    while True:
        x = rng.uniform(lo, hi, d); fx = F(x); fails = 0
        while patience is None or fails < patience:
            y = np.clip(x + sigma * rng.standard_normal(d), lo, hi)
            fy = F(y)
            fails = 0 if fy < fx else fails + 1
            if fy <= fx:
                x, fx = y, fy

def rr_local_search(F, rng, lo, hi, d, sigma_frac=0.02, patience=200):
    hill_climber(F, rng, lo, hi, d, sigma_frac, patience)
# ---- end of copied harness
D, B = 10, 2000

def score(sig_frac, pat, fnames, seeds):
    """Mean over functions of the median log10 final error of RR-LS (the tuning objective)."""
    out = []
    for fn in fnames:
        bm = BENCHMARKS[fn]
        e = run_experiment(rr_local_search, bm.f, B, seeds, bm.lower, bm.upper, D, bm.f_opt,
                           sigma_frac=sig_frac, patience=pat)["errors"][:, -1]
        out.append(np.median(np.log10(e + 1e-12)))
    return float(np.mean(out))

train_f, train_s = ["sphere", "rastrigin"], range(5)
test_s, held_out = range(100, 111), ["ackley", "levy", "schwefel", "griewank"]
tr = np.random.default_rng(42)
cands = [(10 ** tr.uniform(-3, -0.5), int(round(10 ** tr.uniform(1, 3)))) for _ in range(24)]
scores = [score(s, p, train_f, train_s) for s, p in cands]
best = cands[int(np.argmin(scores))]
print(f"tuned (sigma_frac, P) = ({best[0]:.4f}, {best[1]}) after {len(cands)} candidates x {len(train_f)} functions "
      f"x {len(train_s)} seeds x {B} evaluations")
for label, fn, seeds in (("tuning functions, tuning seeds", train_f, train_s),
                         ("tuning functions, 11 fresh seeds", train_f, test_s),
                         ("held-out functions, fresh seeds", held_out, test_s)):
    print(f"{label:34s} tuned {score(*best, fn, seeds):+.3f}   default (0.02, 200) {score(0.02, 200, fn, seeds):+.3f}")
for fn in train_f:
    print(f"  {fn:9s} (tuning seeds): tuned {score(*best, [fn], train_s):+.3f}   default {score(0.02, 200, [fn], train_s):+.3f}")
```

```text
tuned (sigma_frac, P) = (0.0017, 894) after 24 candidates x 2 functions x 5 seeds x 2000 evaluations
tuning functions, tuning seeds     tuned -0.656   default (0.02, 200) +0.394
tuning functions, 11 fresh seeds   tuned -0.664   default (0.02, 200) +0.346
held-out functions, fresh seeds    tuned +1.453   default (0.02, 200) +1.452
  sphere    (tuning seeds): tuned -3.096   default -1.169
  rastrigin (tuning seeds): tuned +1.967   default +1.958
```

The tuned configuration is an order of magnitude better on the functions it was tuned on, and the gain survives fresh
seeds, so it is not seed noise. On held-out functions it is **no better at all**. The per-function lines show why: the
whole gain comes from the Sphere, where a tiny $\sigma$ with long patience exploits the single smooth basin. On
Rastrigin nothing changed. Reporting the first row as "performance" would be tuning on the test set. The correct
protocol separates training instances from test instances, as in racing methods such as F-Race (Birattari, Stützle,
Paquete & Varrentrapp 2002) and irace, which tune on a training distribution. It also reports the tuning budget
(here $24 \times 2 \times 5 \times 2000 = 480\,000$ evaluations) as part of the algorithm's cost. The same logic applies
to hyperparameter optimisation in ML: never select on the test split.
