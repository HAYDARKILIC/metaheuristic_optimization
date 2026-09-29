# Week 1 — Foundations: Search Spaces, Landscapes and Local Search

## Overview

Before studying any single metaheuristic we need a precise language for what it does. This week fixes that language:
an optimisation problem is a search space with an objective; a metaheuristic is judged in the **black-box model** by
the number of objective evaluations it spends; the difficulty of a problem *for a given algorithm* is a property of the
**fitness landscape** $(S, N, f)$ formed by representation, neighbourhood and objective. The No Free Lunch theorem
explains why this relativity is unavoidable. Local search, the simplest procedure that exploits a neighbourhood, gives
us the first baselines and the first failure mode (local optima) that every later algorithm is designed to overcome.
The week ends with the experiment harness used for all later comparisons.

## Learning objectives

After this week you can:

1. State the black-box optimisation model and explain why algorithms must be compared at equal evaluation budgets.
2. Quantify why exact methods fail at scale, using the exactly counted $\Theta(n^2 2^n)$ operations of Held–Karp.
3. Give the definitions of *metaheuristic* by Glover, Osman & Laporte, and Sörensen & Glover, and describe an algorithm by its mechanisms instead of its metaphor.
4. State and prove the No Free Lunch theorem and its sharpened form, and explain why it does not make algorithm design pointless.
5. Define neighbourhoods and local optima, derive neighbourhood sizes, and implement first- and best-improvement hill climbing with delta evaluation.
6. Measure landscape structure: autocorrelation and correlation length, fitness–distance correlation, number of local optima, basins and local optima networks.
7. Run a statistically honest benchmark: seeds, best-so-far traces, median and IQR, success rates, and expected running time (ERT).

## 1. Optimisation problems and the black-box model

An optimisation problem is a pair $(S, f)$ with $f: S \to \mathbb{R}$; we seek

$$
x^{\ast} \in \arg\min_{x \in S} f(x).
$$

In **continuous** problems $S \subseteq \mathbb{R}^d$ is usually a box. In **combinatorial** problems $S$ is finite
but huge: $2^n$ subsets, $(n-1)!/2$ symmetric tours. In the **black-box** model the algorithm only sees queries
$x \mapsto f(x)$. This fits simulators, compiled programs, and the validation loss in hyperparameter optimisation or
neural architecture search, where a single query can cost GPU-hours. The cost measure is therefore the number of
function evaluations (FEs), and every comparison in this course is made at a fixed budget $B$.

**Why not solve exactly?** The symmetric TSP is NP-hard (Karp 1972). The Held–Karp dynamic programme (Bellman 1962;
Held & Karp 1962),

$$
C(T, k) = \min_{j \in T \setminus \lbrace k \rbrace} \big[ C(T \setminus \lbrace k \rbrace, j) + D_{jk} \big],
$$

runs in $\Theta(n^2 2^n)$ time and $\Theta(n 2^n)$ memory. Wall-clock timings depend on the machine, so Notebook 1
counts operations instead: the implementation performs exactly

$$
R(n) = \sum_{s=1}^{n-1} \binom{n-1}{s}\, s\,(n-1-s) = (n-1)(n-2)\,2^{n-3}
$$

relaxations $C(T,k) + D_{kj}$, and the measured count of distance-matrix reads matches $R(n) + n$ exactly for
$n = 5, \ldots, 15$. Even at an optimistic $10^9$ relaxations per second, $n = 30$ needs about 130 GB of table and
$n = 50$ needs about ten years and $2 \times 10^{8}$ GB.

## 2. What is a metaheuristic?

- **Glover (1986)** coined *meta-heuristic* for a master strategy that guides subordinate heuristics beyond local optimality.
- **Osman & Laporte (1996):** an iterative generation process that guides a subordinate heuristic by combining concepts for exploring and exploiting the search space.
- **Sörensen & Glover (2013):** a high-level, problem-independent algorithmic framework that provides guidelines or strategies for developing heuristic optimisation algorithms.
- **Sörensen (2015)** criticised "novel" metaphor-based methods that relabel known mechanisms.

This course describes every algorithm by five mechanisms: **representation**, **neighbourhood/variation**,
**selection/acceptance**, **memory**, and **stochasticity**. Formally, a search algorithm maps a history of evaluated pairs
$d_m = ((x_1, f(x_1)), \ldots, (x_m, f(x_m)))$ to the next query $x_{m+1}$.

## 3. No Free Lunch

Let $X$ and $Y$ be finite and let $\mathcal{F} = Y^X$ be the set of all functions. For a non-revisiting algorithm $a$, let
$d^y_m$ be the sequence of observed values.

**Theorem (Wolpert & Macready 1997).** For any two algorithms $a_1, a_2$ and any $m$,

$$
\sum_{f \in \mathcal{F}} P(d^y_m \mid f, m, a_1) = \sum_{f \in \mathcal{F}} P(d^y_m \mid f, m, a_2).
$$

*Proof idea.* The next point is always new. Among all functions consistent with the history so far, its value is
uniform on $Y$, whichever point was chosen. By induction, every value sequence $y \in Y^m$ is produced by exactly
$|Y|^{|X|-m}$ functions, for every algorithm. Any performance measure is a function of $d^y_m$, so its average over
$\mathcal{F}$ is the same for all algorithms.

**Sharpened NFL (Schumacher, Vose & Whitley 2001).** The equality holds on a subset $F \subseteq Y^X$ if and only if $F$ is
*closed under permutation*: $f \in F$ implies $f \circ \sigma \in F$ for every bijection $\sigma$ of $X$. A c.u.p. set is a
union of "histogram orbits". Such sets are a vanishing fraction of all subsets (Igel & Toussaint 2003). On
$\lbrace 0,1 \rbrace^3 \to \lbrace 0,1,2 \rbrace$ there are 45 orbits, so there are $2^{45} - 1$ non-empty c.u.p. sets
among $2^{6561} - 1$ subsets.

**Consequences.** NFL does not say that all algorithms are equal on problems we care about. It says that performance
comes only from matching the algorithm to a *structured* problem class, one that is not closed under permutation.
Notebook 1 verifies both theorems exhaustively with five adaptive algorithms. It also shows that on the
Hamming-Lipschitz functions a locality-exploiting algorithm wins, and that it loses exactly as much in aggregate on the
complement, an identity that follows from NFL (the optimisation analogue of Schaffer's 1994 conservation law for
generalisation performance).

## 4. Representations, neighbourhoods and local optima

A **genotype–phenotype map** $\gamma: G \to S$ sits between the search operators and the objective, so the search really
optimises $f \circ \gamma$. Examples are binary decoding of reals and random keys (Bean 1994), where $\mathrm{argsort}(r)$ is a permutation.

| Neighbourhood | Space | Size |
|---|---|---|
| bit flip | $\lbrace 0,1\rbrace^n$ | $n$ |
| swap | permutations | $n(n-1)/2$ |
| insertion | permutations | $(n-1)^2$ distinct |
| 2-opt | symmetric tours | $n(n-3)/2$ |
| Gaussian step | $\mathbb{R}^d$ | a distribution $\mathcal{N}(x, \sigma^2 I)$ |

A point $x$ is a **local minimum with respect to $N$** if $f(x) \le f(y)$ for all $y \in N(x)$. Local optimality is
relative. If $N \subseteq N'$, every $N'$-local optimum is also $N$-local. A global optimum is local for every
neighbourhood. Hill climbing iterates moves to improving neighbours:

```text
HillClimb(x, N, f, rule):
  loop
    if rule == best:  y <- argmin over N(x) of f        # full scan
    else:             y <- first improving neighbour    # stop scan early
    if f(y) >= f(x): return x                           # local optimum
    x <- y
```

For 2-opt, **delta evaluation** computes the length change of a move in $O(1)$:

$$
\Delta(i,j) = D_{t_i t_j} + D_{t_{i+1} t_{j+1}} - D_{t_i t_{i+1}} - D_{t_j t_{j+1}}.
$$

**Basins.** For a deterministic descent $h$, the basin of $x^{\ast}$ is $\lbrace x : h(x) = x^{\ast} \rbrace$. With uniform
restarts, one descent succeeds with probability $p = |B(x^{\ast}_{\mathrm{glob}})| / |S|$, and the expected number of
restarts is $1/p$. Notebook 2 computes all basins of a 14-spin Sherrington–Kirkpatrick instance exhaustively and confirms
the predicted success rate by simulation.

**Encodings matter.** Standard binary coding has Hamming cliffs: $127$ and $128$ differ in all 8 bits. The reflected Gray
code maps consecutive integers to strings one bit flip apart. As a result, every strictly unimodal function of an integer
has exactly one bit-flip local minimum under Gray coding. Standard binary coding gives no such guarantee.

## 5. Fitness landscape analysis

A landscape is the triple $(S, N, f)$ (Stadler 2002). **NK landscapes** (Kauffman & Levin 1987; Kauffman 1993) tune
ruggedness through the epistasis $K$:

$$
f(x) = \frac{1}{N} \sum_{i=1}^{N} c_i(x_i, x_{V_i}), \qquad |V_i| = K.
$$

**Autocorrelation (Weinberger 1990).** Along a random bit-flip walk,

$$
\rho(s) = \frac{\mathrm{Cov}(f(x_t), f(x_{t+s}))}{\mathrm{Var} f}, \qquad \ell = -\frac{1}{\ln \rho(1)}.
$$

For NK landscapes, let $P_0(s)$ be the probability that all $K+1$ bits of a contribution have been flipped an even number
of times:

$$
P_0(s) = 2^{-(K+1)} \sum_{j=0}^{K+1} \binom{K+1}{j} \Big(1 - \frac{2j}{N}\Big)^{s} \approx \Big(1 - \frac{K+1}{N}\Big)^{s}.
$$

Within one landscape, distinct entries of a table with $M = 2^{K+1}$ entries have covariance $-\sigma^2/M$ about the
table mean. The exact ACF is therefore

$$
\rho(s) = \frac{M P_0(s) - 1}{M - 1}.
$$

This reduces to the classic $1 - (K+1)/N$ at $s = 1$ when $M$ is large. It differs visibly for $K = 0$.

**Fitness–distance correlation (Jones & Forrest 1995).** FDC is $\mathrm{corr}(f, d)$, where $d$ is the distance to the
nearest global optimum. For maximisation, $-1$ is ideal (OneMax attains it exactly), and deceptive traps give positive
values. Jones & Forrest's rule of thumb is $\mathrm{FDC} \le -0.15$ straightforward, $\vert \mathrm{FDC} \vert \lt 0.15$ difficult,
$\mathrm{FDC} \ge 0.15$ misleading. The measure has known counterexamples
(Altenberg 1997).

**Counting local optima.** If all $2^N$ values are i.i.d. ($K = N-1$), then

$$
\mathbb{E}[\#\text{local optima}] = \frac{2^N}{N+1}.
$$

At $K = 0$ there is exactly one local optimum. Notebook 3 checks both endpoints by exhaustive enumeration at $N = 16$.

**Local optima networks** (Ochoa et al. 2008; escape edges: Vérel et al. 2011) have the local optima as nodes. Their
edges record which optimum a perturbation followed by descent reaches, which is the landscape as seen by iterated
local search.

**Continuous ruggedness.** Take a Gaussian walk with step $\sigma$. The cosine part of Rastrigin has
$\rho(s) = e^{-2\pi^2 \sigma^2 s}$, while the quadratic part decorrelates only on a diffusive time scale. The result is a
two-scale ACF that one correlation length cannot summarise.

**Neutrality.** Plateaus (MAX-SAT, redundant encodings, NK$p$/NK$q$) allow drift without selection. SAT local search
therefore accepts sideways moves (GSAT; Selman, Levesque & Mitchell 1992).

## 6. Benchmarking methodology

The course harness (Lab) runs an optimiser $R$ times under a `BudgetedObjective` and records best-so-far errors on a
logarithmic evaluation grid. It reports median and IQR, success rates, and the expected running time

$$
\mathrm{ERT}(\varepsilon) = \frac{\sum_r \mathrm{RT}_r(\varepsilon)}{\#\text{successes}},
$$

where unsuccessful runs contribute their whole budget (called ENES by Price 1997 and SP2 by Auger & Hansen 2005; ERT or
aRT in COCO, Hansen et al. 2021). ERT is the expected cost
of the algorithm with independent restarts. For pure random search it equals $1/p$, and the Lab confirms this.

**Random-search scaling.** For uniform samples in $[-a, a]^d$ on the Sphere, $P(f \le \varepsilon) = c\,\varepsilon^{d/2}$ with
$c = V_d/(2a)^d$. Hence

$$
\mathbb{E}\Big[\min_{i \le n} f(x_i)\Big] \approx \Gamma\Big(1 + \frac{2}{d}\Big) (n c)^{-2/d}.
$$

Halving the error in $d = 10$ therefore takes 32 times more samples. The Lab verifies the constant and the slope in
$d = 2, 5, 10$.

The three baselines are random search, a fixed-step (1+1) hill climber, and random-restart local search. The Lab shows
that the naive hill climber beats random search on unimodal functions but can be significantly *worse* than random
search on multimodal ones. Restarts repair part of that. Every later algorithm must beat these baselines at equal budget.

## How the notebooks fit

| Notebook | What you build |
|---|---|
| `01_optimization_problems_and_no_free_lunch` | Held–Karp operation count (measured, matched to a closed form) and extrapolation; exhaustive NFL verification over all 6561 functions $\lbrace 0,1\rbrace^3 \to \lbrace 0,1,2\rbrace$ for five adaptive algorithms; sharpened NFL on c.u.p. and structured classes; conservation law |
| `02_representations_neighbourhoods_and_local_search` | Genotype–phenotype maps; neighbourhood sizes verified by enumeration; first/best-improvement 2-opt validated against Held–Karp; random search vs RRHC at equal budgets; exhaustive basins of a spin glass; Gray vs binary Hamming cliffs |
| `03_fitness_landscape_analysis` | NK landscapes; exact random-walk ACF validated empirically (within standard errors); exhaustive local-optima counts vs $K$; FDC with closed-form checks; a local optima network; continuous ACF of Rastrigin vs Sphere; neutrality in MAX-SAT |
| `lab_benchmark_suite_and_baselines` | Tour of the 8 benchmarks, separability and rotation; the course **experiment harness** (median/IQR, success rate, ERT validated against $1/p$); three baselines in $d = 10$; the $n^{-2/d}$ random-search law |

## Common pitfalls

- **Comparing by iterations or wall time** instead of function evaluations. A population method's "iteration" may cost 100 evaluations.
- **Single-run anecdotes.** Stochastic search has heavy-tailed outcomes. Report median and IQR (or mean ± s.e.) over seeds, and use rank-based tests.
- **Reading NFL as "all algorithms are equal".** It holds only for uniform averages over c.u.p. classes. Real problem classes are structured.
- **Forgetting that local optimality is relative** to the neighbourhood and the encoding. Change either and the landscape changes.
- **Treating FDC or the correlation length as complete difficulty measures.** Both have counterexamples, and FDC needs the optimum.
- **Tuning on the test functions** and then reporting the same functions as evidence.
- **Benchmarking only separable, centred functions.** Use shifted and rotated versions, or coordinate-wise tricks will look like general-purpose strength.

## Reading

1. D. H. Wolpert and W. G. Macready, "No free lunch theorems for optimization", *IEEE Transactions on Evolutionary Computation* 1(1):67–82, 1997; and C. Schumacher, M. D. Vose and L. D. Whitley, "The no free lunch and problem description length", *GECCO 2001*.
2. H. H. Hoos and T. Stützle, *Stochastic Local Search: Foundations and Applications*, Morgan Kaufmann, 2004 — Ch. 1–2 (SLS methods, neighbourhoods), Ch. 4 (empirical analysis of SLS algorithms) and Ch. 5 (search space structure).
3. E.-G. Talbi, *Metaheuristics: From Design to Implementation*, Wiley, 2009 — Ch. 1 (common concepts, representations, performance assessment) and Ch. 2 (single-solution methods, landscape analysis).
4. P. F. Stadler, "Fitness landscapes", in *Biological Evolution and Statistical Physics*, Lecture Notes in Physics 585, Springer, 2002; and E. Weinberger, "Correlated and uncorrelated fitness landscapes and how to tell the difference", *Biological Cybernetics* 63:325–336, 1990.
5. K. Sörensen, "Metaheuristics — the metaphor exposed", *International Transactions in Operational Research* 22(1):3–18, 2015.
6. N. Hansen, A. Auger, R. Ros, O. Mersmann, T. Tušar and D. Brockhoff, "COCO: A platform for comparing continuous optimizers in a black-box setting", *Optimization Methods and Software* 36(1):114–144, 2021.
