# Week 6 — Unified Mechanisms, Parameter Control, Multi-Objective Search and Rigorous Benchmarking

## Overview

Weeks 1–5 built five classic metaheuristics one at a time. This final week steps back and asks what they have in
common, how their parameters should be set, how they extend to several objectives, and how to compare them honestly.
The answer to the first question is a single loop: every algorithm keeps a *memory*, samples candidates from a
*distribution parameterised by that memory*, evaluates them, and *selects/updates* the memory. Once algorithms are
described by mechanisms rather than metaphors, several things become visible: "novel" methods that are old ones in
disguise, operators that exploit the benchmark rather than the problem, and the fact that parameter setting is itself
an optimisation problem. We finish with the modern baselines DE and CMA-ES, multi-objective search (NSGA-II, MOEA/D),
and a benchmarking methodology — fixed-target measures, performance profiles and rank-based statistics — applied to a
grand comparison of all algorithms of the course.

## Learning objectives

After this week you should be able to

- write any metaheuristic as (memory, generation distribution, selection/acceptance, memory update) and implement it as
  an ask–tell plug-in of one generic loop;
- measure exploration and exploitation (sampling spread, reach, worsening-acceptance rate) and relate them to parameters;
- show that harmony search is a $(\mu+1)$-ES and detect centre bias with shifted benchmarks;
- implement DE (rand/1, best/1, current-to-pbest/1 with JADE adaptation), the $(1+1)$-ES with the 1/5th rule, and the
  full CMA-ES, and verify their theoretical properties;
- classify parameter-setting methods (tuning vs deterministic/adaptive/self-adaptive control), implement bandit-based
  operator selection, F-Race and a selection hyper-heuristic;
- implement fast non-dominated sorting, crowding distance, exact 2-D hypervolume, IGD, NSGA-II and MOEA/D;
- design a benchmark study, compute ERT, runtime ECDFs and performance profiles, and apply Wilcoxon, Friedman–Nemenyi
  and Holm correctly.

## 1. One loop for every metaheuristic

Let $f:\mathcal{X}\to\mathbb{R}$ be minimised. A metaheuristic is a tuple $(M_0, P, U)$:

$$
X_t \sim P(\cdot \mid M_t), \qquad y_t = f(X_t), \qquad M_{t+1} = U(M_t, X_t, y_t).
$$

| algorithm | memory $M_t$ | generation $P(\cdot\mid M_t)$ | selection / update |
|---|---|---|---|
| SA | current $x$, temperature $T$ | $x+\sigma_t\mathcal{N}(0,I)$ | Metropolis: accept w.p. $\min(1,e^{-\Delta/T})$ |
| TS | current $x$, tabu list | $k$ neighbours | best admissible neighbour, even if worse |
| GA | population | selection, crossover, mutation | generational replacement + elitism |
| PSO | positions, velocities, personal bests | $v\leftarrow wv+c_1r_1(p-x)+c_2r_2(g-x)$ | personal bests replaced if better |
| ACO<sub>ℝ</sub> | archive of $k$ best | Gaussian kernels around archive members | truncation to the $k$ best |
| DE | population | $x_{r_1}+F(x_{r_2}-x_{r_3})$ + binomial crossover | one-to-one greedy |
| CMA-ES | $m,\sigma,C$, two paths | $\mathcal{N}(m,\sigma^2C)$ | weighted recombination, CSA, rank-one/rank-$\mu$ |

This is the ask–tell interface (`ask()` samples, `tell(X, y)` updates). It is also the view of *model-based search*
(Zlochin et al. 2004): each algorithm maintains, implicitly or explicitly, a probability model of where good solutions are.

**Exploration vs exploitation, measured.** Three quantities make the informal terms operational: the *sampling spread*
(root-mean-square distance of the last $w$ evaluated points to their centroid), the *reach* (distance of new candidates
to the best-so-far point) and the *worsening-acceptance rate* $\omega_t$ (fraction of ranks at which the working set
got worse). Elitist memories (DE, ACO<sub>ℝ</sub>) have $\omega\equiv0$ and diversify only through sampling; SA and TS
diversify through acceptance; CMA-ES is non-elitist until its distribution collapses. For PSO, the inertia weight $w$
moves the balance and the second-moment analysis of a stagnant particle gives the stability region (Poli 2009)

$$
c_1+c_2 \lt \frac{24(1-w^2)}{7-5w}, \qquad \lvert w\rvert \lt 1,
$$

which the notebook re-derives from the $3\times3$ moment recursion.

**The metaphor critique.** Sörensen (2015) argued that metaphor-based "novel" algorithms usually repackage known
mechanisms. Weyland (2010) proved that harmony search is a $(\mu+1)$-ES: "memory consideration" is global discrete
recombination, "pitch adjustment" and "random selection" are creep and reset mutations, "replace the worst harmony" is
plus-selection. Camacho-Villalón, Dorigo and Stützle made the same point for intelligent water drops (2019) and for
grey wolf, moth-flame, whale, firefly, bat and antlion optimisers (ITOR, 2022/2023). Notebook 1 codes both harmony
search and the ES in their own vocabularies and shows bit-identical trajectories under common random numbers.

**Centre bias.** Many classic benchmarks have the optimum at the origin, the centre of the box. An operator that
contracts multiplicatively towards $0$ looks brilliant there and fails once the function is shifted (Kudela 2022).
Always benchmark on $f(R(x-o))$ with random shifts $o$ and rotations $R$.

## 2. Differential evolution and CMA-ES

**DE** (Storn & Price 1997) builds mutants from difference vectors. For i.i.d. population members with covariance
$\Sigma$, $x_{r_2}-x_{r_3}$ has covariance $2\Sigma$, so mutation self-scales with the population. Zaharie (2002)
showed that DE/rand/1/bin variation (without selection) changes the expected population variance by the factor

$$
\mathbb{E}\left[\mathrm{Var}(P_{g+1})\right]/\mathrm{Var}(P_g) = 2F^2p - \frac{2p}{N} + \frac{p^2}{N} + 1,
$$

with $p$ the probability of taking a coordinate from the mutant (exact when the base vector $x_{r_1}$ may coincide with the
target; with all indices distinct from each other and from the target the factor changes by $O(p/N^2)$); below $F_{\text{crit}}=\sqrt{(1-p/2)/N}$ the
population contracts even on a flat landscape. **JADE** (Zhang & Sanderson 2009) uses DE/current-to-pbest/1 with an
archive and adapts $\mu_{CR}$ (arithmetic mean of successful $CR$) and $\mu_F$ (Lehmer mean of successful $F$).

**The 1/5th success rule.** For the $(1+1)$-ES on the sphere with normalised step $\sigma^{\ast}=\sigma d/R$, as
$d\to\infty$ (Rechenberg 1973),

$$
\varphi^{\ast}(\sigma^{\ast}) = \frac{\sigma^{\ast}}{\sqrt{2\pi}}e^{-\sigma^{\ast2}/8} - \frac{\sigma^{\ast2}}{2}\Big(1-\Phi\big(\tfrac{\sigma^{\ast}}{2}\big)\Big), \qquad P_s = 1-\Phi\big(\tfrac{\sigma^{\ast}}{2}\big).
$$

The maximum $\varphi^{\ast}\approx0.202$ occurs at $\sigma^{\ast}\approx1.224$, i.e. $P_s\approx0.27$; on the corridor the
optimum is $P_s\approx0.18$. Rechenberg's compromise: increase $\sigma$ if more than 1/5 of the mutations succeed.

**CMA-ES** (Hansen & Ostermeier 2001; Hansen 2016) samples $x_k=m+\sigma y_k$, $y_k\sim\mathcal{N}(0,C)$, and with
$y_w=\sum_{i\le\mu}w_iy_{i:\lambda}$ updates

$$
m \leftarrow m+\sigma y_w, \qquad p_\sigma \leftarrow (1-c_\sigma)p_\sigma+\sqrt{c_\sigma(2-c_\sigma)\mu_{\text{eff}}}\,C^{-1/2}y_w, \qquad \sigma\leftarrow\sigma\exp\Big(\frac{c_\sigma}{d_\sigma}\Big(\frac{\Vert p_\sigma\Vert}{\mathbb{E}\Vert\mathcal{N}(0,I)\Vert}-1\Big)\Big),
$$

$$
C \leftarrow (1-c_1-c_\mu)\,C + c_1\,p_cp_c^{\top} + c_\mu\sum_{i=1}^{\mu}w_i\,y_{i:\lambda}y_{i:\lambda}^{\top}.
$$

(shown for $h_\sigma=1$ and positive weights, $\sum w_i=1$). The notebooks use the default constants of the 2016
tutorial ($\lambda=4+\lfloor3\ln n\rfloor$, $\mu=\lfloor\lambda/2\rfloor$, $w_i\propto\ln\frac{\lambda+1}{2}-\ln i$,
$c_\sigma=\frac{\mu_{\text{eff}}+2}{n+\mu_{\text{eff}}+5}$, $c_1=\frac{2}{(n+1.3)^2+\mu_{\text{eff}}}$, …) with one
simplification: the tutorial's Table 1 also assigns *negative* weights to the $\lambda-\mu$ worst samples (active
CMA), which we omit, as the tutorial's own reference code does. The rank-$\mu$ term is the maximum-likelihood covariance of selected steps (a natural-gradient step on the Gaussian
family), the rank-one term uses the evolution path $p_c$, and cumulative step-size adaptation compares the length of
$p_\sigma$ with its expectation under random selection. Consequences verified in Notebook 2: on a quadratic with Hessian
$H$ the learned $C$ is proportional to $H^{-1}$; CMA-ES is invariant to rotations (the update equations are checked to
be rotation equivariant to rounding error) and to strictly increasing transformations of $f$ (same seed on $f$ and $f^3$
gives a bit-identical run); it converges linearly on the sphere. DE with a small $CR$ is *not* rotation invariant.

## 3. Parameter tuning, parameter control and hyper-heuristics

Eiben, Hinterding & Michalewicz (1999) separate **tuning** (values fixed before the run) from **control** (values
changed during the run), and control into **deterministic** (schedules), **adaptive** (feedback rules such as the 1/5th
rule or JADE) and **self-adaptive** (parameters encoded in individuals). Schwefel's log-normal self-adaptation
$\sigma'=\sigma e^{\tau\xi}$ is unbiased on the log scale. For the $(1,\lambda)$-ES the asymptotic progress
$\varphi^{\ast}=c_{1,\lambda}\sigma^{\ast}-\sigma^{\ast2}/2$ explains why a constant $\sigma$ stalls at distance
$R_\infty=\sigma d/(2c_{1,\lambda})$, while self-adaptation keeps $\sigma^{\ast}$ in the progress region at every scale.

**Adaptive operator selection** is a non-stationary multi-armed bandit (Fialho et al. 2010): probability matching,
adaptive pursuit (Thierens 2005) and UCB-type rules (Auer et al. 2002), the latter with a sliding window for
non-stationarity. **Offline tuning** is best done by racing: F-Race (Birattari et al. 2002) evaluates candidates
instance by instance and discards those that the Friedman test and its post-hoc comparison declare worse; irace
(López-Ibáñez et al. 2016) iterates racing with a sampling model. **Selection hyper-heuristics** (Burke et al. 2013)
choose among low-level heuristics using only objective feedback; their value is robustness across instance classes.
In machine learning this is AutoML: hyperparameter optimisation with Bayesian optimisation (SMAC, Hutter et al. 2011),
successive halving and Hyperband (Li et al. 2017), and population-based training as self-adaptive control.

## 4. Multi-objective metaheuristics

With $F:\mathcal{X}\to\mathbb{R}^M$, $a$ dominates $b$ if $F_m(a)\le F_m(b)$ for all $m$ and $F_m(a)\lt F_m(b)$ for some
$m$. The target is the Pareto front; quality is measured by the **hypervolume** (Pareto compliant, needs a reference
point) and **IGD** (needs a reference front). **NSGA-II** (Deb et al. 2002) combines fast non-dominated sorting
($O(MN^2)$ with domination counters and dominated sets) with crowding distance in a $(\mu+\mu)$ elitist GA.
**MOEA/D** (Zhang & Li 2007) decomposes the problem into $N$ scalar subproblems with weights $\lambda^i$ and the
Tchebycheff function

$$
g^{\text{te}}(x\mid\lambda,z^{\ast}) = \max_m \lambda_m\,\lvert F_m(x)-z^{\ast}_m\rvert,
$$

which, unlike the weighted sum, can reach every Pareto-optimal point, including concave parts of the front. The ZDT
problems have analytic fronts, so hypervolume and IGD can be checked against closed forms (e.g. with reference point
$(1.1,1.1)$, $\mathrm{HV}_{\text{ZDT1}}=0.21+2/3$). In AI, neural architecture search trades accuracy against latency
or size and is naturally multi-objective (NSGA-Net, Lu et al. 2019).

## 5. Rigorous benchmarking

**Protocol.** Equal budgets in function evaluations; shifted and rotated instances generated from recorded seeds;
full best-so-far traces; default parameters or an equal tuning budget for every competitor; random search as a
baseline.

**Two views.** The fixed-budget view (error after $B$ evaluations) is ordinal and budget dependent. The fixed-target
view (evaluations to reach a target) is on a ratio scale. With right-censoring at $B$, the **expected running time**

$$
\mathrm{ERT} = \frac{\sum_i \min(T_i,B)}{\#\text{successes}}
$$

estimates the cost of independent restarts until success; for geometric runtimes it is consistent for $1/p$ at any
budget. **Runtime ECDFs** (COCO; Hansen et al. 2021) aggregate over functions and targets; **performance profiles**
(Dolan & Moré 2002) plot $\rho_s(\tau)$, the share of problems on which solver $s$ is within a factor $\tau$ of the best.

**Statistics** (Demšar 2006). Two algorithms: Wilcoxon signed-rank test. Several algorithms: Friedman test on ranks
within blocks, then the Nemenyi test with critical difference

$$
\mathrm{CD} = q_\alpha\sqrt{\frac{k(k+1)}{6N}},
$$

drawn as a CD diagram, or comparisons with a control corrected by Holm's step-down procedure. A significant result
holds for the tested suite, budget and dimension only; the no-free-lunch theorems (Wolpert & Macready 1997) forbid
universal claims.

## How the notebooks fit

| notebook | what you build |
|---|---|
| `01_a_unified_view_of_metaheuristic_mechanisms` | one ask–tell loop driving SA, TS, GA, PSO, ACO<sub>ℝ</sub>, DE, CMA-ES; mechanism checks against theory (Metropolis rule, tabu admissibility, tournament probability, PSO stability boundary, ACO<sub>ℝ</sub> rank weights, DE crossover); exploration metrics; PSO inertia U-curve; harmony search = $(\mu+1)$-ES; centre-bias experiment |
| `02_differential_evolution_and_cma_es` | DE/rand/1, DE/best/1, JADE; Zaharie's variance law; $(1+1)$-ES progress rate and 1/5th rule; full CMA-ES with checks of $C\propto H^{-1}$, rotation invariance and linear convergence; comparison with the core algorithms |
| `03_parameter_tuning_control_and_hyper_heuristics` | constant vs deterministic vs self-adaptive step sizes; PM, AP, UCB1 and sliding-window UCB with regret and stationarity checks; AOS over step-size operators; Friedman test and F-Race tuning DE; bin-packing selection hyper-heuristic |
| `04_multi_objective_metaheuristics` | fast non-dominated sort vs brute force; crowding distance; ZDT1–3; exact 2-D hypervolume vs closed form and Monte Carlo; IGD; NSGA-II; MOEA/D-Tchebycheff; equal-budget comparison |
| `lab_rigorous_benchmarking_grand_comparison` | seven algorithms on a shifted–rotated suite in $d=10$; convergence graphs; ERT (validated), runtime ECDFs, performance profiles; own Wilcoxon, Friedman, Nemenyi and Holm validated against scipy and published tables; CD diagram; reproducibility checklist |

## Common pitfalls

- Comparing algorithms per *iteration* or per *generation* instead of per function evaluation.
- Benchmarking only on origin-centred, separable functions — rewards centre bias and coordinate-wise operators.
- Tuning your own method and running competitors with arbitrary settings.
- Reporting a single run, or mean ± std of heavy-tailed errors, instead of medians/IQR, ERT and ECDFs.
- Running many pairwise tests without multiple-comparison correction; using Nemenyi with very few blocks and reading
  "not significant" as "equivalent".
- Treating the CMA-ES covariance scale separately from $\sigma$ — only $\sigma^2C$ is meaningful.
- Using the weighted sum to find a concave Pareto front, or computing hypervolume without removing dominated points.
- Calling a mechanism new because it has a new metaphor: always write it as memory, distribution and update first.

## Reading

1. N. Hansen (2016). *The CMA Evolution Strategy: A Tutorial.* arXiv:1604.00772 (Sections 2–4 and Table 1 defaults);
   N. Hansen & A. Ostermeier (2001), *Evolutionary Computation* 9(2):159–195.
2. R. Storn & K. Price (1997). Differential evolution. *Journal of Global Optimization* 11:341–359; J. Zhang & A. C.
   Sanderson (2009), JADE, *IEEE Transactions on Evolutionary Computation* 13(5):945–958.
3. A. E. Eiben, R. Hinterding & Z. Michalewicz (1999). Parameter control in evolutionary algorithms. *IEEE TEC*
   3(2):124–141; M. Birattari et al. (2002), A racing algorithm for configuring metaheuristics, GECCO 2002.
4. K. Deb, A. Pratap, S. Agarwal & T. Meyarivan (2002). NSGA-II. *IEEE TEC* 6(2):182–197; Q. Zhang & H. Li (2007),
   MOEA/D, *IEEE TEC* 11(6):712–731.
5. J. Demšar (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR* 7:1–30; N. Hansen et al.
   (2021), COCO: a platform for comparing continuous optimizers in a black-box setting, *Optimization Methods and
   Software* 36(1):114–144; E. D. Dolan & J. J. Moré (2002), *Mathematical Programming* 91:201–213.
6. K. Sörensen (2015). Metaheuristics — the metaphor exposed. *International Transactions in Operational Research*
   22(1):3–18; D. Weyland (2010), *International Journal of Applied Metaheuristic Computing* 1(2):50–60.
