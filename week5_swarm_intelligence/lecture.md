# Week 5 — Swarm Intelligence: Particle Swarm and Ant Colony Optimization

## Overview

"Swarm intelligence" names two very different algorithm families. **Particle swarm optimization** (PSO) works on
continuous vectors. It keeps a population of trajectories, each with a point memory (its personal best), and links
them through an information graph. **Ant colony optimization** (ACO) builds solutions piece by piece and learns a
*distribution* over those construction decisions (the pheromone matrix). This week drops the metaphors and treats
both in terms of their mechanisms: representation, variation, selection, memory and stochasticity. We derive the
stability theory of PSO, the pheromone bounds of MAX–MIN Ant System, the gradient and cross-entropy view of ACO, and
the convergence theorems. Every claim is checked numerically, and the week ends with a multi-seed comparison at equal
budgets on shifted and rotated benchmarks, plus a hyperparameter-tuning application.

## Learning objectives

After this week you can

1. state the PSO update, prove that the constriction and inertia forms are equivalent, and compute $\chi(4.1)\approx0.7298$;
2. derive the order-1 and order-2 stability regions of the stagnating particle and place standard parameter settings in them;
3. explain how topology, velocity clamping and boundary handling change PSO behaviour, and why standard PSO is not rotation invariant;
4. implement AS, ACS and MMAS for the TSP (vectorised), derive $\tau_{\max}=1/(\rho L_{\text{bs}})$ and the MMAS $\tau_{\min}$, and validate against Held–Karp;
5. interpret a pheromone update as a natural-gradient / cross-entropy step, and state the convergence results precisely;
6. implement ACO$_\mathbb{R}$ and run a fair benchmark: equal budgets, many seeds, IQR bands, runtime ECDFs, rotated functions.

## 1. Particle swarm optimization

PSO was introduced by Kennedy & Eberhart (1995). Particle $i$ has a position $x_i$, a velocity $v_i$, a personal best
$p_i$ and a neighbourhood best $l_i$. With fresh $r_1,r_2\sim U(0,1)^d$ in every coordinate,

$$
v_i\leftarrow w\,v_i+c_1r_1\odot(p_i-x_i)+c_2r_2\odot(l_i-x_i),\qquad x_i\leftarrow x_i+v_i ,
$$

and $p_i$ is replaced greedily. Selection acts only on the memory. The graph behind $l_i$ is the main knob for
exploration: gbest (diameter 1), von Neumann torus (diameter about $\sqrt n$), ring (diameter $n/2$).

```text
initialise x_i ~ U(box), v_i, p_i = x_i
repeat: l_i = best p_j in N(i); update v_i, clamp, x_i += v_i, bound rule; evaluate; update p_i
```

**Inertia and constriction.** Shi & Eberhart (1998) added the inertia weight $w$. Clerc & Kennedy (2002) wrote the update as
$v\leftarrow\chi[v+\varphi_1r_1(p-x)+\varphi_2r_2(l-x)]$ with

$$
\chi=\frac{2}{\left|2-\varphi-\sqrt{\varphi^2-4\varphi}\right|},\qquad\varphi=\varphi_1+\varphi_2\gt4 .
$$

Expanding the bracket gives the inertia form with $w=\chi$ and $c_k=\chi\varphi_k$. The two forms are identical, not
merely similar. Also, $\chi$ is the smaller root of $\chi^2-(\varphi-2)\chi+1=0$. For $\varphi=4.1$ this gives
$\chi=0.729844$ and $c=1.49618$.

**Clamping and bounds.** The 1995 rule ($w=1$, $c=2$) diverges without $v_{\max}$, and even with it the swarm never
contracts. Boundary handling (absorb, reflect, random) is part of the algorithm. It matters most when the optimum
lies near the boundary; on Schwefel, reflecting clearly beats absorbing.

**Rotation.** The coordinate-wise factors $r_k\odot(\cdot)$ confine each step to an axis-aligned box, so performance
depends on the coordinate system. On a separable ellipsoid PSO converges to $10^{-17}$, but after a random rotation
it stalls at errors of a few hundred. SPSO-2011 (Clerc 2011) instead samples uniformly in a hypersphere around
$G=x+c\,[(p-x)+(l-x)]/3$, which makes the update rotation-equivariant. However, invariance by itself does not make
it a better optimiser: with uniform hypersphere sampling the stability region of SPSO-2011 depends on the dimension
(Bonyadi & Michalewicz 2014), and in the notebook its stagnating particle is unstable for $d\ge5$ at the default
parameters, so in $d=10$ the swarm never contracts.

## 2. Dynamics and stability

Assume **stagnation**: $p$ and $l$ are fixed. Each coordinate then follows a random linear recurrence

$$
x_{t+1}=(1+w-\varphi_t)x_t-wx_{t-1}+c_1r_1p+c_2r_2l,\qquad\varphi_t=c_1r_1+c_2r_2 .
$$

With $\varphi$ held constant, the characteristic polynomial is $\lambda^2-(1+w-\varphi)\lambda+w$. The Jury
conditions give the **deterministic stability region**

$$
|w|\lt1,\qquad0\lt\varphi\lt2(1+w).
$$

Setting $\varphi=c_1+c_2$ (all $r=1$) gives the conservative region $c_1+c_2\lt2(1+w)$ (van den Bergh 2002). Because
$r_{k,t}$ is independent of the past, $\mathbb E x_t$ obeys the same recurrence with $\varphi=\mathbb E\varphi_t=(c_1+c_2)/2$,
so the **order-1 (mean) stability region** of the stochastic particle is $|w|\lt1$, $0\lt c_1+c_2\lt4(1+w)$. The
region $c_1+c_2\lt2(1+w)$ is sometimes mislabelled "order-1"; it is a strict subset, valid for the frozen $r\equiv1$ model.

**Order-2 (mean-square) stability** is governed by a $3\times3$ matrix acting on
$(\mathbb E y_t^2,\mathbb E y_{t-1}^2,\mathbb E y_ty_{t-1})$. For $c_1=c_2$, Poli (2009) showed that its spectral
radius is below one iff

$$
c_1+c_2\lt\frac{24(1-w^2)}{7-5w}.
$$

The notebook recovers this boundary to $10^{-8}$ by bisection on the spectral radius, and Monte Carlo simulation
agrees with it on 98% of the grid points that are not too close to the boundary. The Monte Carlo errors come from heavy tails: the sample mean of $y^2$
underestimates $\mathbb E[y^2]$. The default point $(0.7298, 2\times1.49618)$ lies inside every region, with
$\rho(M)\approx0.944$. The linearly decreasing inertia schedule starts *outside* the order-2 region and ends inside it.
The same polynomial describes heavy-ball momentum SGD, with $w=\beta$ and $\varphi=\eta\lambda$.

## 3. Ant colony optimization

Ant System appeared in Dorigo's thesis (1992) and in Dorigo, Maniezzo & Colorni (1996). An ant at city $i$ chooses $j$ with

$$
p_{ij}=\frac{\tau_{ij}^\alpha\eta_{ij}^\beta}{\sum_{l\in\mathcal N_i}\tau_{il}^\alpha\eta_{il}^\beta},\qquad
\tau_{ij}\leftarrow(1-\rho)\tau_{ij}+\sum_{k\in\mathcal U}\frac{\mathbb 1[(i,j)\in T_k]}{L_k}.
$$

| variant | who deposits | exploitation control |
|---|---|---|
| AS (1996) | all ants | none |
| ACS (Dorigo & Gambardella 1997) | best-so-far | greedy choice with prob. $q_0$; local update $\tau\leftarrow(1-\xi)\tau+\xi\tau_0$ |
| MMAS (Stützle & Hoos 2000) | iteration-best | $\tau\in[\tau_{\min},\tau_{\max}]$, initialisation at $\tau_{\max}$, restarts |

**Bounds.** If an edge receives $1/L$ in every iteration, then $\tau_{t+1}=(1-\rho)\tau_t+1/L$ converges to
$1/(\rho L)$. This gives $\tau_{\max}=1/(\rho L_{\text{bs}})$. Next, require the converged colony to rebuild its best
tour with probability $p_{\text{best}}$, using about $n/2$ candidates per decision. That gives

$$
\tau_{\min}=\frac{\tau_{\max}(1-p_{\text{best}}^{1/n})}{(n/2-1)\,p_{\text{best}}^{1/n}} .
$$

The implementation is vectorised over ants *and* independent runs. On 13-city instances AS, ACS and MMAS recover the
Held–Karp optimum in 92–98% of runs within 1500 tours. The one-at-a-time parameter study shows three things.
$\beta=0$ fails, so the heuristic is essential. $\alpha=1$ is best for AS. The long-run MMAS default $\rho=0.02$ is
too slow for a 100-iteration budget. With neighbour-list 2-opt, MMAS+2-opt beats both plain MMAS (even with a tuned $\rho$) and multistart
2-opt at equal cost.

## 4. ACO theory: model-based search, cross-entropy, convergence

ACO is **model-based search** (Zlochin, Birattari, Meuleau & Dorigo 2004): sample from $P_\tau$, then update $\tau$.
For independent binary decisions $P_p(s)=\prod_ip_i^{s_i}(1-p_i)^{1-s_i}$, the score-function gradient of
$J(p)=\mathbb E_p[F(s)]$, preconditioned by the inverse Fisher information $\mathrm{diag}(p_i(1-p_i))$, gives

$$
p\leftarrow(1-\rho)\,p+\rho\sum_k\frac{F(s_k)}{\sum_lF(s_l)}\,s_k .
$$

This is evaporation plus quality-weighted deposit, the hyper-cube ACO update (Blum & Dorigo 2004). In other words,
a pheromone update is a natural-gradient step, and $\rho$ plays the role of the learning rate (Meuleau & Dorigo 2002).
The **cross-entropy method** (Rubinstein 1997, 1999) fits $P_p$ to the elite by maximum likelihood; for this model
the fit is the elite mean, followed by smoothing. That is the same update with the elite as the depositing ants.

**Convergence.** Stützle & Dorigo (2002) consider ACO$_{bs,\tau_{\min}}$: best-so-far deposit with constant
$\tau_{\min}\gt0$. Every construction step then has probability at least some $p_{\min}\gt0$, whatever the history,
and a tour takes $n-1$ decisions, so with $\hat p=1-(1-p_{\min}^{\,n-1})^m$ for $m$ ants

$$
P^{\ast}(t)\ge1-(1-\hat p)^t\to1 .
$$

This is convergence in value. Once $s^{\ast}$ is found, every other edge reaches $\tau_{\min}$ within
$\lceil\ln(\tau_{\min}/\tau_{\max})/\ln(1-\rho)\rceil$ iterations. Gutjahr (2000) proved, for the graph-based Ant
System with a unique optimum and elitist reinforcement, that for every $\varepsilon\gt0$ the probability that a fixed
ant builds the optimum in iteration $t$ is at least $1-\varepsilon$ for all large $t$, provided the number of ants is
large enough or the evaporation rate close enough to $0$. Gutjahr (2002) obtained probability-one convergence with a
slowly decreasing evaporation rate or a slowly decreasing lower bound. These bounds say nothing practical about
runtime. Their structural lesson is that sampling probabilities must stay bounded away from zero. With
$\tau_{\min}=0$, about half of our runs locked into a sub-optimal tour (the probability of building any other tour
is below $10^{-40}$ in each of them).

**Entropy.** The mean row entropy of the normalised pheromone measures how concentrated the model is. For converged
MMAS it equals a closed-form limit that depends only on $n$ and $p_{\text{best}}$; the notebook verifies this to $10^{-6}$.

**ACO$_\mathbb{R}$** (Socha & Dorigo 2008) replaces the pheromone table with an archive of $k$ ranked solutions. An
ant picks a guide $l$ with a Gaussian rank weight
$\omega_l\propto\exp(-(l-1)^2/(2q^2k^2))$, then samples $x^i\sim\mathcal N(s_l^i,\sigma_l^i)$, where
$\sigma_l^i=\xi\sum_e|s_e^i-s_l^i|/(k-1)$. The small parameter $q$ makes it greedy and fast on smooth problems, at
the cost of robustness (Ackley).

## 5. The lab: a fair showdown

The lab compares PSO in three variants (constriction gbest, constriction ring, and LDIW), ACO$_\mathbb{R}$, and a
(1+1)-ES with the 1/5 rule. The test problems are shifted and shifted-rotated ellipsoid, Rosenbrock, Rastrigin and
Ackley, with a budget of $1000d$ at $d=10$ and $500d$ at $d=30$. The results are reported as IQR convergence bands
and COCO-style runtime ECDFs.

- There is no universal winner.
- ACO$_\mathbb{R}$ dominates separable problems, but after rotation it loses by up to 28 orders of magnitude.
- PSO is strongest on Rastrigin.
- The isotropic ES is rotation invariant and wins on Rosenbrock and on the rotated ellipsoid.
- Nothing solves the rotated ill-conditioned ellipsoid; that requires covariance adaptation (CMA-ES, Week 6).

The final section applies PSO to hyperparameter optimisation. It tunes four hyperparameters of an ARD kernel ridge
model by closed-form leave-one-out error, and beats random search at 60 evaluations.

## How the notebooks fit

| notebook | what you build |
|---|---|
| `01_particle_swarm_optimization` | vectorised PSO with 3 topologies, clamping, 3 bound rules, SPSO-2011 update; constriction derivation and equivalence check; topology, parameterisation, boundary and rotation experiments |
| `02_pso_dynamics_and_stability` | one-particle recurrence, order-1 and order-2 regions verified by eigenvalues, bisection and Monte Carlo; swarm explosion; diversity decay vs RMS and Lyapunov rates |
| `03_ant_colony_optimization` | vectorised AS/ACS/MMAS for the TSP, sampler χ² validation, Held–Karp validation, pheromone visualisation, $(\alpha,\beta,\rho)$ study, neighbour-list 2-opt hybrid at equal cost |
| `04_aco_theory_and_extensions` | score-function / natural-gradient view, CE method vs ML fit, CE vs ACO on knapsack vs DP, convergence-in-value checks, pheromone entropy limit, ACO$_\mathbb{R}$ vs PSO |
| `lab_swarm_showdown` | 5 algorithms × shifted/rotated suite (d = 10, 30), IQR curves, runtime ECDFs, rotation analysis, PSO for kernel-ridge hyperparameter tuning vs random search |

## Common pitfalls

- **Comparing by iterations.** A swarm of 40 and a (1+1)-ES spend very different numbers of evaluations per iteration. Always compare at equal evaluation budgets.
- **Origin-centred benchmarks.** Unshifted functions reward a bias towards the centre of the box. Shift every function, and rotate it too, otherwise coordinate-wise methods look far better than they are.
- **$w=1$ without $v_{\max}$.** This setting diverges. Also, an inertia schedule tied to the budget changes behaviour whenever the budget changes.
- **Silent boundary choices.** Absorb, reflect and random give different results; always report which one you used.
- **Pheromone underflow.** In AS with large $\rho$, pheromone can underflow and produce all-zero weight rows. Guard the sampler, or use MMAS bounds.
- **Literature defaults.** MMAS's $\rho=0.02$ assumes thousands of iterations. Tune at the budget you will actually use.
- **Naive Monte Carlo for second moments.** Heavy-tailed products make these estimates optimistic; prefer exact moment recursions.
- **Single runs and means.** Report medians with IQR, or full ECDFs, and test differences (e.g. Mann–Whitney).

## Reading

1. J. Kennedy, R. Eberhart, *Particle swarm optimization*, Proc. IEEE ICNN 1995; Y. Shi, R. Eberhart, *A modified particle swarm optimizer*, IEEE ICEC 1998.
2. M. Clerc, J. Kennedy, *The particle swarm — explosion, stability, and convergence in a multidimensional complex space*, IEEE TEC 6(1), 2002; R. Poli, *Mean and variance of the sampling distribution of particle swarm optimizers during stagnation*, IEEE TEC 13(4), 2009.
3. M. Dorigo, T. Stützle, *Ant Colony Optimization*, MIT Press, 2004 — chapters 2–4 (algorithms, TSP, theory).
4. T. Stützle, H. Hoos, *MAX–MIN Ant System*, Future Generation Computer Systems 16(8), 2000; M. Dorigo, L. Gambardella, *Ant Colony System*, IEEE TEC 1(1), 1997.
5. M. Zlochin, M. Birattari, N. Meuleau, M. Dorigo, *Model-based search for combinatorial optimization: a critical survey*, Annals of OR 131, 2004; P.-T. de Boer et al., *A tutorial on the cross-entropy method*, Annals of OR 134, 2005.
6. K. Socha, M. Dorigo, *Ant colony optimization for continuous domains*, EJOR 185(3), 2008; T. Stützle, M. Dorigo, *A short convergence proof for a class of ACO algorithms*, IEEE TEC 6(4), 2002.
