# Week 2 — Simulated Annealing

## Overview

Simulated annealing (SA) is the oldest of the five core metaheuristics in this course and the one with the deepest
theory. It started as a *sampler*: Metropolis, Rosenbluth, Rosenbluth, Teller and Teller (1953) built a Markov chain
whose long-run distribution is the Boltzmann law of statistical mechanics. Thirty years later Kirkpatrick, Gelatt and
Vecchi (1983), and independently Černý (1985), turned the sampler into an *optimiser*: treat the objective as an
energy, run the chain, and lower the temperature so that the Boltzmann law concentrates on the global minima.

Stripped of the metallurgical metaphor, SA is a single-point trajectory method with four mechanisms:

| Mechanism | SA |
|---|---|
| Representation | any search space with a neighbourhood (permutations, bit strings, boxes in $\mathbb{R}^d$) |
| Variation | one random neighbour per step (swap, 2-opt, bit flip, Gaussian/Cauchy step) |
| Acceptance | Metropolis: accept $\Delta \le 0$ always, $\Delta\gt0$ with probability $e^{-\Delta/T}$ |
| Control | a temperature schedule $T_k$ (or an adaptive controller) |

This week proves why the acceptance rule works, states exactly when SA converges (Hajek's theorem), and then shows
why that theory is *not* what decides performance at a finite evaluation budget. We finish with parallel tempering
and with engineered SA for the TSP and MAX-3SAT.

## Learning objectives

After this week you should be able to:

1. Define the Boltzmann distribution and the Metropolis(–Hastings) rule, and prove detailed balance.
2. Compute a stationary distribution, a spectral gap and an exact law $\mu_k$ of a finite Markov chain, and use them to validate a sampler.
3. State Hajek's theorem, compute the depth $d^{\ast}$ of a landscape, and explain the gap between asymptotic guarantees and finite budgets.
4. Choose $T_0$ from a target acceptance ratio, and implement geometric, linear, logarithmic, adaptive and reheating schedules.
5. Implement threshold accepting, great deluge, record-to-record travel and late acceptance, and compare rules fairly (equal budgets, pilot tuning, paired tests).
6. Design proposals for continuous SA (Gaussian, Cauchy, Tsallis) and adapt step sizes to an acceptance target.
7. Derive and implement the replica-exchange rule of parallel tempering.
8. Engineer SA with $O(1)$ delta evaluation for the TSP and MAX-SAT.

## 1. Boltzmann distribution and the Metropolis rule

For a finite space $S$, energy $E$ and temperature $T\gt0$,

$$
\pi_T(x) = \frac{e^{-E(x)/T}}{Z_T}, \qquad Z_T=\sum_{y\in S}e^{-E(y)/T}.
$$

As $T\to\infty$, $\pi_T$ becomes uniform; as $T\to0$, it becomes uniform on the set $S^{\ast}$ of global minima,
because $\pi_T(x)/\pi_T(x^{\ast}) = e^{-(E(x)-E^{\ast})/T}$. The mean energy increases with $T$:
$\frac{d}{dT}\langle E\rangle_T = \mathrm{Var}_T(E)/T^2$.

With a proposal kernel $q$, the Metropolis–Hastings chain moves from $x$ to $y\sim q(x,\cdot)$ with probability

$$
\alpha_T(x,y) = \min\Big(1,\ e^{-(E(y)-E(x))/T}\,\frac{q(y,x)}{q(x,y)}\Big).
$$

**Detailed balance.** For $x\ne y$, $\pi_T(x)P(x,y)=\min(\pi_T(x)q(x,y),\ \pi_T(y)q(y,x))$ is symmetric in $(x,y)$.
Summing over $x$ gives $\pi_TP=\pi_T$. If the chain is also irreducible and aperiodic, it converges to $\pi_T$ from
any start. Only energy *differences* appear, so $Z_T$ is never needed. The Hastings factor matters. With
"uniform over existing neighbours" on a grid, plain Metropolis samples $\propto\deg(x)e^{-E(x)/T}$ rather than
$\pi_T$ (Notebook 1 verifies this exactly).

**Mixing slows exponentially.** The spectral gap of the Metropolis chain behaves like $e^{-m/T}$ as $T\to0$, with
the critical height $m=\max_{x,y}[H(x,y)-E(x)-E(y)]+E^{\ast}$, where $H(x,y)$ is the minimax path energy (Holley & Stroock 1988).
This Arrhenius law is the quantitative form of the exploration/exploitation tension.

```text
SA(x0, E, propose, schedule, budget):
    x <- x0; best <- x
    for k = 1, 2, ... until budget spent:
        y <- propose(x); Δ <- E(y) - E(x)
        if Δ <= 0 or U(0,1) < exp(-Δ / T_k): x <- y
        if E(x) < E(best): best <- x
    return best                      # best-so-far, not the final state
```

## 2. Convergence: SA as an inhomogeneous Markov chain

With changing temperatures the law of $X_k$ is $\mu_k=\mu_0P_{T_1}\cdots P_{T_k}$. Geman and Geman (1984) showed that a
logarithmic schedule with a large enough constant suffices. **Hajek (1988)** gave the exact answer. For
$T_k=c/\log(k+1)$ on a connected, weakly reversible neighbourhood graph,

$$
\lim_{k\to\infty}\Pr(X_k\in S^{\ast})=1 \iff c\ge d^{\ast},
$$

where $d^{\ast}$ is the largest *depth* of a non-global local minimum. The depth is the height one must climb from
the minimum before reaching a strictly better state. The heuristic is Borel–Cantelli: escaping a minimum of depth
$d$ at step $k$ has probability $\approx(k+1)^{-d/c}$, and the series diverges iff $d\le c$.

Notebook 2 propagates the exact law $\mu_k$ on a constructed landscape with $d^{\ast}=2.2$. For $c\lt d^{\ast}$ the
chain freezes in the trap. For $c\ge d^{\ast}$ the probability of being at the optimum keeps rising, but only
logarithmically, and a large $c$ leaves the chain too hot at any finite $k$. The logarithmic schedule therefore buys a
guarantee on every landscape at a price that no finite budget can pay.

## 3. Practical schedules and initial temperature

* **Geometric** $T_k=T_0\alpha^k$ (Kirkpatrick et al. 1983). This is the default choice.
* **Linear** $T_k = T_0(1-k/K)+T_{\text{end}}k/K$.
* **Adaptive**: a controller keeps the observed acceptance rate on a target profile. Examples are Lam & Delosme (1988)
  and the "modified Lam" profile, which has a long plateau at acceptance 0.44.
* **Reheating**: when frozen or stagnant, raise $T$ and cool again.
* **Homogeneous stages** (Aarts & van Laarhoven 1985): run chains of fixed length at fixed $T$ and adapt the decrement from the observed energy variance.

**Initial temperature.** Choose $T_0$ so that a fraction $\chi_0$ of *uphill* moves is accepted. The classical
estimate $T_0=-\overline{\Delta^+}/\ln\chi_0$ is always too hot, by Jensen's inequality. The root of
$\frac1M\sum_m e^{-\Delta^+_m/T}=\chi_0$ is exact on the sample (cf. Ben-Ameur 2004).

**Alternative acceptance rules**, all with deterministic thresholds:

| Rule | Accept if |
|---|---|
| Threshold accepting (Dueck & Scheuer 1990) | $\Delta\lt\tau_k$ |
| Great deluge (Dueck 1993) | $E(y)\le B_k$, a falling "water level" |
| Record-to-record travel (Dueck 1993) | $E(y)\lt E_{\text{best}}+D$ |
| Late acceptance (Burke & Bykov 2008/2017) | $E(y)\le E(x)$ or $E(y)\le$ the current energy $L$ steps ago |

The Notebook 2 comparison runs these rules on an $n=100$ Sherrington–Kirkpatrick spin glass. Each rule was tuned on a
separate pilot instance, and the test runs used common random initial states with paired Wilcoxon tests and a Holm
correction for the nine comparisons. Threshold accepting, record-to-record travel and late acceptance were
statistically indistinguishable from geometric SA; great deluge was worse in median, but not significantly after the
correction. Surprisingly, the logarithmic schedule won, and this difference does survive the correction. The reason is not Hajek: it spent the whole
budget in the window just below the spin-glass transition $T_c=1$, while the hot geometric schedules spent only
10–20% of it there. **At a finite budget, what matters is how much of the budget is spent in the temperature window
where the energy actually decreases.**

## 4. Continuous SA and parallel tempering

In $\mathbb{R}^d$, SA proposes $y=x+\sigma\xi$. The candidate laws are:

* **Gaussian** $\xi$ (classical SA).
* **Cauchy** $\xi$ with $T_k\propto1/k$: *fast SA* (Szu & Hartley 1987).
* **Tsallis** visiting law of *generalised SA* (Tsallis & Stariolo 1996). For $q_v\in(1,3)$ it is exactly a
  multivariate Student-$t$ with $\nu=(3-q_v)/(q_v-1)$ degrees of freedom and scale $T^{1/(3-q_v)}/\sqrt{3-q_v}$.
  Proposition 1 of Notebook 3 proves this and checks it numerically.

**Step-size adaptation.** Random-walk Metropolis in high dimension is most efficient at acceptance rate $0.234$
(Roberts, Gelman & Gilks 1997; $\sigma=2.38/\sqrt d$ for a standard Gaussian target), and in 1-D at $\approx0.44$
(Gelman, Roberts & Gilks 1996). A Robbins–Monro update $\log\sigma\mathrel{+}=\gamma(\mathbf 1_{\text{acc}}-a^{\ast})$
finds the right scale automatically. In SA this makes the step shrink as $T$ falls.

In our experiments the clock-driven FSA/GSA step laws collapsed long before the budget ended. The adaptive
variants were robust. An SA run cannot get much below its *thermal floor* ($dT_{\text{end}}/2$ on a quadratic bowl, by
equipartition).

**Parallel tempering** (Swendsen & Wang 1986; Geyer 1991; Hukushima & Nemoto 1996) runs replicas at
$\beta_1\gt\dots\gt\beta_M$ on the product target $\prod_m\pi_{\beta_m}$. It proposes swaps of neighbouring replicas and accepts them with

$$
\min\Big(1,\ e^{(\beta_i-\beta_j)(E(x_i)-E(x_j))}\Big),
$$

which is the Metropolis–Hastings ratio of the product target under the symmetric swap proposal. PT is a correct
*sampler* of every marginal. On a bimodal target it recovered $\Pr(X\lt0)$ within Monte Carlo error at every
temperature, while a single chain never left its well. As an *optimiser* at equal budgets, PT beat a single SA run
(clearly on Rastrigin, only suggestively on Schwefel once the six tests are Holm-corrected) but tied with SA plus
restarts. A no-swap ablation shows how much the exchange mechanism itself contributes: on Schwefel the swaps make a
significant difference.

## 5. Engineering SA: TSP and MAX-3SAT

* **TSP.** Use 2-opt ($\Delta=D_{ac}+D_{bd}-D_{ab}-D_{cd}$) and or-opt moves with $O(1)$ deltas, $k$-nearest
  neighbour lists, a nearest-neighbour start, and $T_0$ from an acceptance target. SA found the exact Held–Karp
  optimum in every run for $n\le12$. For $n=50$–$200$ the **Held–Karp 1-tree bound** (Held & Karp 1970, 1971),
  maximised by subgradient ascent, turns SA's upper bounds into certified gaps: a single SA run was within about 2%
  (median) of a provable lower bound. The bound also proves that $L^{\ast}_n/\sqrt n$ lies above the
  Beardwood–Halton–Hammersley constant $\approx0.7124$ (Percus & Martin 1996) on every instance, as the boundary
  effects of the unit square predict, and it decreases towards it as $n$ grows.
* **MAX-3SAT** near the threshold $m/n\approx4.26$. Track the number of true literals per clause and the set of
  unsatisfied clauses; then $\Delta=\text{break}-\text{make}$. **Focusing** the proposals on variables in
  unsatisfied clauses is the mechanism that matters most, as in focused Metropolis search (Seitz, Alava & Orponen
  2005) and WalkSAT (Selman, Kautz & Cohen 1994): the two focused methods are the ones whose advantage over plain
  descent survives a multiple-testing correction.
* **AI link.** Feature-subset and hyperparameter search is the same loop. On an exhaustively enumerated
  ridge-regression space, SA found the exact optimum far more often than random search at 150 evaluations. It also
  succeeded more often than descent (77% vs 67% of 30 paired seeds), but an exact McNemar test shows that this edge
  is not significant.

## How the notebooks fit

| Notebook | What you build |
|---|---|
| `01_metropolis_and_simulated_annealing` | Exact Metropolis/Hastings transition matrices on a grid landscape; eigenvector vs Boltzmann vs simulation (Markov-chain CLT); spectral bound and Arrhenius law for the gap; $T\to0$ concentration; generic SA, whose success rate is checked against an exact absorption probability; SA vs restarts vs random search on exactly solved QAPs |
| `02_cooling_schedules_and_convergence` | Exact propagation of the inhomogeneous SA law demonstrating Hajek's $c\ge d^{\ast}$; $T_0$ from acceptance ratio (Jensen); vectorised SK spin-glass engine; geometric/linear/log/adaptive/reheating SA vs TA, GD, RRT, LAHC with pilot tuning and paired tests |
| `03_continuous_sa_and_parallel_tempering` | Gaussian, Cauchy (FSA) and Tsallis (GSA) visiting laws with an exact Student-$t$ sampler; the 0.234 optimal-scaling result and Robbins–Monro step adaptation; continuous SA on Rastrigin, Ackley and Schwefel; PT with exact invariance check, bimodal marginals, and PT vs SA vs restarts vs no-swap ablation |
| `lab_sa_for_tsp_and_max_sat` | TSP SA with 2-opt/or-opt $O(1)$ deltas validated against Held–Karp; schedule-tuning study; Held–Karp 1-tree lower bound certifying SA's gap for $n=50$–$200$; BHH comparison; tours before/after; incremental MAX-3SAT SA validated by enumeration; focused vs unfocused SA vs WalkSAT; SA for feature-subset + ridge-$\lambda$ search |

## Common pitfalls

- **Returning the final state instead of the best-so-far.** SA's final state at a nonzero $T_{\text{end}}$ can be much worse.
- **Asymmetric proposals without the Hastings factor.** They are harmless for pure optimisation but wrong for sampling. Neighbour lists and boundary handling create asymmetry silently; clipping to a box is asymmetric, reflection is not.
- **Starting too hot.** A hot start wastes most of a short budget. $T_0$ from $-\overline{\Delta^+}/\ln\chi_0$ is biased upward, and $\chi_0=0.8$ may be far above the useful temperature window.
- **Citing Hajek as a reason for a log schedule.** The guarantee is asymptotic. At a finite budget, choose the schedule by where it spends the budget.
- **Comparing by iterations or wall-clock time instead of evaluations**, or tuning parameters on the test instance.
- **Single-run anecdotes.** Report medians and IQRs over seeds, use common random numbers, and test with paired or rank tests.
- **Many comparisons, one threshold.** Comparing ten rules pairwise produces "significant" differences by chance; report multiplicity-adjusted $p$-values (e.g. Holm 1979).
- **Trusting upper bounds alone.** An SA tour length says nothing about the optimality gap; a lower bound (Held–Karp for the TSP) certifies it.
- **Trusting a delta-evaluation formula without checking it against full recomputation.**

## Reading

1. S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi (1983). Optimization by simulated annealing. *Science* 220(4598):671–680. Also V. Černý (1985), *J. Optim. Theory Appl.* 45:41–51.
2. N. Metropolis, A. Rosenbluth, M. Rosenbluth, A. Teller, E. Teller (1953). Equation of state calculations by fast computing machines. *J. Chem. Phys.* 21:1087–1092. W. K. Hastings (1970), *Biometrika* 57:97–109.
3. B. Hajek (1988). Cooling schedules for optimal annealing. *Mathematics of Operations Research* 13(2):311–329.
4. E. Aarts, J. Korst (1989). *Simulated Annealing and Boltzmann Machines*. Wiley, Chs. 2–5 (Markov-chain theory and schedules); P. J. M. van Laarhoven, E. Aarts (1987), *Simulated Annealing: Theory and Applications*, Reidel.
5. D. A. Levin, Y. Peres, E. L. Wilmer (2017). *Markov Chains and Mixing Times*, 2nd ed., AMS. Chs. 1–4 and 12 (stationarity, convergence, spectral gap).
6. D. J. Earl, M. W. Deem (2005). Parallel tempering: theory, applications, and new perspectives. *Phys. Chem. Chem. Phys.* 7:3910–3916. G. O. Roberts, A. Gelman, W. R. Gilks (1997), *Ann. Appl. Probab.* 7(1):110–120.
