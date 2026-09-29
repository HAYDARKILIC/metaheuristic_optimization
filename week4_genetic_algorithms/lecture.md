# Week 4 — Genetic Algorithms and Evolutionary Mechanisms

## Overview

A genetic algorithm (GA) keeps a *population* of candidate solutions and improves it by repeating three steps.
**Selection** looks at fitness and decides who becomes a parent. **Variation** (crossover and mutation) does not look
at fitness; it creates the new candidates. **Replacement** merges parents and offspring into the next population. That
is the whole method: every question about "convergence", "diversity" or "building blocks" comes down to how these
three maps behave. This week takes each mechanism apart and puts it back together. We quantify selection pressure
with takeover times and selection intensities. We derive what crossover operators keep and pass on (positional bias,
variance transfer, edge heritability). We prove and test the classic theory results: the schema theorem, deceptive
traps, exact runtimes of the (1+1) EA, and the Nix–Vose Markov chain. We also study niching, island models and
constraint handling. The lab builds a memetic algorithm for the TSP and a small evolutionary architecture search. Every
claim is checked against an exact solver, a closed form or an independent reference implementation, and every
comparison uses an equal evaluation budget across several seeds.

## Learning objectives

After this week you should be able to:

1. write a GA as a composition of selection, variation, replacement and elitism, and say which parts use fitness;
2. derive the selection probabilities, selection intensity and takeover time of roulette, SUS, ranking, tournament,
   truncation and Boltzmann selection, and prove that SUS has minimal spread;
3. analyse crossover operators by what they transmit, and derive the positional bias of one-point crossover and the
   variance transfer of BLX-$\alpha$ and SBX;
4. state and prove the schema theorem, explain what it does *not* say, and build a deceptive function;
5. reproduce the runtimes $e\,n\ln n$ (OneMax) and $\frac{e-1}{2}n^2$ (LeadingOnes) of the (1+1) EA;
6. choose niching and constraint-handling methods and explain how each one orders solutions;
7. design memetic algorithms and evolutionary NAS experiments with an honest cost model.

## 1. The generational loop

```text
P <- N random genotypes; evaluate
repeat until the evaluation budget is spent:
    M <- select(P, f, N)                      # mating pool
    O <- mutate(crossover(pairs of M))        # N offspring, N evaluations
    P <- replace(P, O)  (+ elitism: keep the best of P)
```

Holland (1975) introduced the GA as a model of adaptation, and De Jong (1975) was the first to study it
systematically as a function optimiser. Goldberg (1989) made it widely known. In mechanism terms the representation
defines the search space, selection is a random map from (population, fitness) to parents, and variation consists of
fitness-blind Markov kernels.

## 2. Selection

Let $e_i$ be the expected number of copies of individual $i$ in the mating pool.

- **Roulette**: $m$ i.i.d. draws with $p_i = f_i / \sum_j f_j$. The copy counts are binomial, so the spread runs from
  $0$ to $m$.
- **SUS** (Baker 1987): one spin, with $m$ equally spaced pointers. Each count lies in
  $\lbrace\lfloor e_i\rfloor, \lceil e_i\rceil\rbrace$, which is the minimal spread, and the method is unbiased. The
  interval of length $e_i$ contains either $\lfloor e_i\rfloor$ or $\lceil e_i\rceil$ points of a unit lattice, and
  averaging over the random offset gives exactly $e_i$.
- **$k$-tournament**: the $i$-th worst of $n$ is selected with probability $\big(i^k-(i-1)^k\big)/n^k$.
- **Linear ranking**, **truncation** and **Boltzmann** selection complete the family. Ranking, tournament and
  truncation depend only on the *order* of fitness values, so they are invariant to monotone rescaling.

**Selection intensity.** For a standard-normal fitness distribution, $I = (\bar f_{\text{sel}} - \bar f)/\sigma$. Binary
tournament gives $I_2 = 1/\sqrt\pi$, truncation with fraction $\tau$ gives $I = \varphi(z_\tau)/\tau$, and linear
ranking gives $I = (s-1)/\sqrt\pi$.

**Takeover time** (Goldberg & Deb 1991). Start from a single copy of the best individual and apply selection only. For
tournaments the non-best proportion follows $Q_{t+1} = Q_t^k$. Solving $Q_t \le 1/n$ gives

$$t^{\ast} \approx \frac{\ln n + \ln\ln n}{\ln k},$$

and for proportional selection with fitness ratio $r$, $t^{\ast} \approx 2\ln(n-1)/\ln r$. Proportional selection
therefore loses its pressure when fitness values are compressed. Notebook 1 shows this directly: on planted
MAX-3SAT, fitness-proportional selection (roulette, SUS) leaves a median of 15–16 clauses unsatisfied, while
rank-based schemes leave 2.

## 3. Variation operators

**Binary.** One-point crossover separates two loci at distance $\delta$ with probability $\delta/(L-1)$. For
two-point crossover the probability is $2\delta(L-1-\delta)/((L-1)(L-2))$, and for uniform crossover it is $1/2$.
Bit-flip mutation with rate $1/L$ leaves a fraction $(1-1/L)^L \approx e^{-1}$ of offspring unchanged.

**Real-valued.** Let the parents be i.i.d. with variance $\sigma^2$. Then

$$\mathrm{Var}_{\text{BLX-}\alpha} = \sigma^2\Big[\tfrac12 + \tfrac{(1+2\alpha)^2}{6}\Big], \qquad \mathrm{Var}_{\text{SBX}} = \sigma^2\,\frac{1 + E[\beta^2]}{2}, \qquad E[\beta^2] = \frac{\eta+1}{2}\Big(\frac{1}{\eta+3} + \frac{1}{\eta-1}\Big).$$

BLX-$\alpha$ preserves variance exactly when $\alpha = (\sqrt3-1)/2$. SBX (Deb & Agrawal 1995) always expands the
variance slightly, and whole arithmetic crossover contracts it by a factor $2/3$. The SBX spread factor has density
$\frac12(\eta+1)\beta^{\eta}$ for $\beta \le 1$ and $\frac12(\eta+1)\beta^{-(\eta+2)}$ for $\beta \gt 1$; this is
verified by a KS test.

**Permutations.** PMX (Goldberg & Lingle 1985) and CX (Oliver, Smith & Holland 1987) preserve absolute positions. OX
(Davis 1985) preserves relative order, and ERX (Whitley, Starkweather & Fuquay 1989) preserves adjacencies. On the TSP,
whose cost is a sum over edges, the key quantity is *heritability*: the fraction of the child's edges that come from
a parent. ERX has the highest heritability and beats CX on every seed. However, operators that cannot create new
edges stagnate unless mutation is added, and too much total disruption also hurts.

## 4. Theory

**Schema theorem** (Holland 1975). For roulette selection, one-point crossover and bit-flip mutation,

$$E[m(H,t+1)\mid P_t] \ge m(H,t)\,\frac{f(H,t)}{\bar f(t)}\Big[1 - p_c\frac{\delta(H)}{L-1}\Big]\,(1-p_m)^{o(H)}.$$

This is a bound for one generation, in expectation. It does not guarantee convergence and cannot simply be iterated (Grefenstette 1993;
Vose 1999); Altenberg (1995) showed that it is a special case of Price's exact covariance theorem. The
**building-block hypothesis** has a precise positive core. On concatenated trap-4 functions, which Notebook 3 checks
exhaustively to be fully deceptive, one-point crossover with *tight* linkage solves all blocks, while loose linkage or
uniform crossover do not.

**Runtime of the (1+1) EA.** On OneMax (Droste, Jansen & Wegener 2002) the expected time is $\Theta(n\log n)$ and
$E[T] \le e\,n(\ln n + 1)$; Sudholt (2013) proved the matching lower bound $e\,n\ln n - O(n\log\log n)$, and Hwang,
Panholzer, Rolin, Tsai & Chen (2018) obtained the expansion
$E[T] = e\,n\ln n + c_1 n + \frac{e}{2}\ln n + c_2 + O(\log n/n)$ with $c_1 \approx -1.89254$. Notebook 3 solves the
Markov chain on the number of ones exactly and confirms this expansion: the residual converges to $c_2 \approx 0.6$. On LeadingOnes (Böttcher, Doerr & Neumann 2010),

$$E[T] = \frac{1}{2p^2}\Big((1-p)^{-n+1} - (1-p)\Big) \;\xrightarrow{p=1/n}\; \frac{e-1}{2}n^2 \approx 0.859\,n^2,$$

and the optimal static rate is $p \approx 1.59/n$, which gives $E[T] \approx 0.772\,n^2$.

**Exact Markov model** (Nix & Vose 1992). The state is the population viewed as a multiset. The transition from $u$
to $v$ is multinomial, with the cell probabilities given by Vose's heuristic map $\mathcal G(u/N)$. For $\ell = 3$
and $N = 4$ there are 330 states, and the exact and simulated trajectories agree within Monte Carlo error.

## 5. Diversity, niching, islands, constraints

- **Drift.** Under neutral selection, $E[h_{t+1}] = (1 - 1/N)h_t$ (the Wright–Fisher law). Crossover does not slow
  this loss.
- **Niching.** Fitness sharing (Goldberg & Richardson 1987) divides fitness by the niche count
  $\sum_j \mathrm{sh}(d_{ij})$. Deterministic crowding (Mahfoud 1995) lets a child replace its closest parent only if
  it is better. Clearing (Pétrowski 1996) keeps only the niche winners. On M1 and Himmelblau, crowding and clearing
  find all optima in every run.
- **Island models.** A ring of subpopulations with periodic migration outperformed the panmictic GA on 10-D
  Rastrigin at equal budget.
- **Constraints.** The available approaches are the death penalty, static and dynamic penalties (Joines & Houck
  1994), repair, Deb's feasibility rules (Deb 2000; a lexicographic order with no penalty coefficient) and stochastic
  ranking (Runarsson & Yao 2000). On knapsack, greedy repair reaches the DP optimum. On g06, whose feasible region is
  0.007% of the box, the death penalty rarely finds a feasible point at all.

## 6. Memetic algorithms and evolutionary NAS

A memetic algorithm (Moscato 1989) applies local search to each offspring. In the **Lamarckian** variant the improved
solution is written back into the genotype; in the **Baldwinian** variant (Hinton & Nowlan 1987) only its fitness is
used. The cost model must be stated. Here a 2-opt delta evaluation costs $4/n$ of a full tour evaluation. The 2-opt search uses don't-look bits, which make it fast but only approximately locally optimal (the lab shows a counterexample and the fix). Under this
model the Lamarckian OX/ERX + 2-opt algorithm beats multistart 2-opt, the Baldwinian variant and a plain GA on a
100-city instance. For NAS, **regularised evolution** (Real et al. 2019) uses tournament selection and single-gene
mutation, and it removes the *oldest* member instead of the worst. On an exhaustively computed table of 216 MLP
configurations (a miniature NAS-Bench-101; Ying et al. 2019), all evolutionary searches beat random search. The
expected regret of random search is validated against its exact order-statistics law.

## How the notebooks fit

| Notebook | What you build |
|---|---|
| `01_ga_anatomy_and_selection` | Six selection operators validated against closed forms; the SUS minimal-spread proof; selection intensities; takeover time (theory vs simulation); a full GA on planted MAX-3SAT comparing selection, elitism and steady-state replacement |
| `02_variation_operators` | Binary, real and permutation operators; positional bias; BLX/SBX variance transfer; KS tests for SBX and polynomial mutation; permutation validity and edge heritability; TSP GA with and without mutation |
| `03_ga_theory` | Monte Carlo check of the schema theorem; exhaustive deception check and the trap-function experiment; exact OneMax chain and LeadingOnes formula vs bit-string simulation; the 330-state Nix–Vose chain |
| `04_diversity_niching_and_constraints` | Drift law; sharing, crowding and clearing on M1 and Himmelblau; island model; six constraint-handling methods on knapsack (vs DP) and g06 (vs SLSQP and the closed form) |
| `lab_memetic_algorithm_for_tsp_and_nas_toy` | 2-opt with don't-look bits; vectorised OX; Lamarckian/Baldwinian memetic GA vs plain GA and multistart 2-opt; a tabular toy NAS with regularised evolution |

## Common pitfalls

- **Using proportional selection on raw objective values.** Pressure then depends on the offset and scale of $f$.
  Use ranking or tournaments, or at least rescale the fitness.
- **Treating the schema theorem as a convergence proof.** It bounds one generation, in expectation.
- **Comparing algorithms by generations or iterations.** Compare them by evaluations, and state how delta evaluations
  and local-search moves are charged.
- **Choosing a crossover without asking what it transmits.** Position-based operators on an edge-based objective,
  or coordinatewise SBX on a rotated problem, discard the structure that makes recombination useful.
- **Mistaking drift for convergence.** A small population loses alleles even without selection. Increase $N$, add
  mutation, or use niching or islands.
- **Using the death penalty when the feasible region is tiny.** Without a gradient towards feasibility, the search
  never finds a feasible point.
- **Seeding instance generation and the algorithm from the same random stream.** The algorithm can then "find" the
  planted solution by coincidence.

## Reading

1. J. H. Holland, *Adaptation in Natural and Artificial Systems*, University of Michigan Press, 1975 (MIT Press ed. 1992), ch. 4–7.
2. D. E. Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning*, Addison-Wesley, 1989, ch. 1–5; and D. E. Goldberg & K. Deb, "A comparative analysis of selection schemes used in genetic algorithms", *FOGA* 1991.
3. A. E. Eiben & J. E. Smith, *Introduction to Evolutionary Computing*, 2nd ed., Springer, 2015, ch. 3–6, 10, 13.
4. K. Deb, *Multi-Objective Optimization Using Evolutionary Algorithms*, Wiley, 2001, ch. 4 (real-coded GAs, SBX); and K. Deb, "An efficient constraint handling method for genetic algorithms", *CMAME* 186, 2000.
5. B. Doerr & F. Neumann (eds.), *Theory of Evolutionary Computation*, Springer, 2020, ch. 1–2 (runtime analysis); and M. D. Vose, *The Simple Genetic Algorithm*, MIT Press, 1999.
6. E. Real, A. Aggarwal, Y. Huang & Q. V. Le, "Regularized evolution for image classifier architecture search", *AAAI* 2019.
