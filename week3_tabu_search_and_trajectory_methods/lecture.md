# Week 3 — Tabu Search and Trajectory Metaheuristics

## Overview

Week 2 escaped local optima through randomness: simulated annealing sometimes accepts a worse neighbour. This
week covers the other main family of *trajectory* (single-solution) metaheuristics. They escape through **memory**
or through **repeated local search**:

- **Tabu search (TS)** always moves to the best *admissible* neighbour, even when that neighbour is worse. A memory
  of recent move attributes (the tabu list) forbids moves that would undo recent decisions.
- **Iterated local search (ILS)**, **variable neighbourhood search (VNS)** and **GRASP** treat a local search as a
  black-box map from solutions to local optima. They differ only in how they choose the next starting point.

The main ideas of the week are attribute-based memory, the balance between intensification and diversification,
parameters that adapt themselves (reactive TS), and the rule that a kick too weak to escape the local search is
wasted. Everything is implemented in NumPy, validated against exact solvers (brute-force QAP, Held–Karp TSP, DP
knapsack, exact chromatic number, exhaustive feature selection) and compared at equal evaluation budgets.

## Learning objectives

By the end of the week you should be able to:

1. State tabu search precisely in terms of its mechanisms: neighbourhood, candidate list, best-admissible
   selection, tabu attributes and tenure, aspiration.
2. Explain why tabu lists store attributes rather than solutions, and why aspiration is needed as a result.
3. Derive the $O(n)$ swap delta for the QAP and Taillard's $O(1)$-per-entry update of the delta matrix.
4. Prove that fixed-tenure deterministic TS is eventually periodic, and use randomised or reactive tenures to break
   the cycle.
5. Implement long-term memory (frequency penalties, elite path relinking), robust TS, reactive TS with Zobrist
   hashing, and strategic oscillation.
6. Implement ILS (double-bridge kick, acceptance criteria), VNS (VND, shaking) and GRASP (RCL with parameter
   $\alpha$), and compare them fairly.
7. Run a paired comparison of two algorithms with a Wilcoxon signed-rank test that you implemented yourself.

## 1. Tabu search: definition

Let $S$ be finite, $f: S \to \mathbb{R}$ to be minimised, and $M(x)$ the set of moves available at $x$. TS keeps
a current solution $x$, the best solution found $x^{\ast}$, and an array `tabu_until` indexed by *move attributes*:

```text
for k = 1, 2, ...:
    C  <- candidate moves of x                 (full neighbourhood or a candidate list)
    A  <- { m in C : m not tabu at k  or  f(x + m) < f(x*) }        # aspiration by objective
    m* <- argmin_{m in A} f(x + m)             # may be uphill
    make the attributes destroyed by m* tabu until k + t
    x  <- x + m* ;  update x*
```

TS was introduced by **Glover (1986)**, who also coined the name, and developed in *Tabu Search — Part I/II* (ORSA
J. Computing, 1989/1990). Hansen's *steepest ascent / mildest descent* (1986) is a close relative. An *attribute*
is a small fact about a move, for example "facility $i$ left location $\ell$". A list of whole solutions would be
exact but expensive: every one of the $\vert M(x) \vert$ neighbours would have to be hashed and looked up. Such a
list also forbids only exact repeats. An attribute memory costs $O(1)$ per test and forbids a whole region of
recently undone decisions. The cost is **over-restriction**: some never-visited solutions are also forbidden.
**Aspiration by objective** fixes this. A tabu move that would beat $f(x^{\ast})$ cannot lead back to a solution
already visited, so it is always allowed.

**Cycling.** If the tenure is $t = 0$ and ties are broken deterministically, then $x_{k+1} = \Phi(x_k)$ for a fixed
map $\Phi$ on a finite set. By the pigeonhole principle the trajectory is eventually periodic. At a local minimum,
the best move is uphill by $\delta$, and the reverse move then has $\Delta = -\delta$, so the period is usually 2.
For $t \gt 0$ the state (solution, memory) is still finite, so fixed-tenure deterministic TS is also eventually
periodic, but the period can be very long. Notebook 1 shows both effects. With tenure 0 the search gets stuck in a
2-cycle. On tiny QAPs, a short fixed tenure with the weak tabu rule gets trapped in a small set of solutions. This
is the main reason to use randomised (robust) or adaptive (reactive) tenures.

## 2. QAP swap deltas

$C(\pi) = \sum_{i,j} F_{ij} D_{\pi_i \pi_j}$. Take symmetric $F$ and $D$ with zero diagonals, and write
$d_{ij} = D_{\pi_i \pi_j}$. Swapping the locations of facilities $r$ and $s$ changes the cost by

$$
\Delta(\pi; r,s) = 2 \sum_{k \ne r,s} (F_{kr} - F_{ks})(d_{ks} - d_{kr}),
$$

which costs $O(n)$. With $M = F [d_{ij}]$ the whole matrix is
$\Delta_{rs} = 2(M_{rs} + M_{sr} - M_{rr} - M_{ss} + 2F_{rs} d_{rs})$. After the swap $(r,s)$ has been made, giving
$\pi'$, **Taillard (1991)** updates every pair $\lbrace u,v \rbrace$ disjoint from $\lbrace r,s \rbrace$ in $O(1)$:

$$
\Delta(\pi'; u,v) = \Delta(\pi; u,v) + 2(F_{ru} - F_{rv} + F_{sv} - F_{su})(D_{\pi'_s \pi'_u} - D_{\pi'_s \pi'_v} + D_{\pi'_r \pi'_v} - D_{\pi'_r \pi'_u}).
$$

A full-neighbourhood TS iteration therefore costs $O(n^2)$. Every formula in the notebooks is checked against
`qap_cost` of the explicitly swapped permutation.

## 3. Memory beyond the tabu list

| Memory | Content | Role |
|---|---|---|
| recency | when an attribute last changed | tabu status |
| frequency | how often an attribute was present (residence) or changed (transition) | diversification |
| quality | elite solutions | intensification, path relinking |

**Frequency penalty.** Let $h_{i\ell}$ count the iterations in which facility $i$ sat at location $\ell$. Non-improving
moves are ranked by $\Delta + \lambda \bar\delta (h_{r\pi_s} + h_{s\pi_r})/k$, which pushes the search toward
assignments it has rarely used.

**Path relinking** (Glover 1997; Glover, Laguna & Martí 2000) walks from an initiating elite toward a guiding elite.
Each step greedily introduces one attribute of the guide. For permutations, each step fixes at least one position,
so the walk takes at most $n-1$ steps. The best intermediate solution is returned.

**Robust TS** (Taillard 1991) combines four things: the weak rule (a swap is tabu only if *both* facilities would
return to locations they recently left), a tenure drawn uniformly from $[0.9n, 1.1n]$ and re-drawn every $2t_{max}$
iterations, aspiration by objective, and a long-term aspiration that forces the best move placing a facility on a
location it has not occupied for more than $t_{asp}$ iterations. All attributes must be initialised as "used at
iteration 0": if never-used pairs counted as "unused for a long time", the criterion would fire at every iteration
of the first few hundred and force moves regardless of cost.

**Reactive TS** (Battiti & Tecchiolli 1994) adapts the tenure $T$ online:

```text
on revisiting a stored solution (repetition gap R):  if R < 2(n-1): T <- 1.1 T
if no change of T for longer than the moving-average cycle length:  T <- max(0.9 T, 1)
if too many solutions repeat more than REP times:  escape by a random walk
```

Repeated solutions are detected in $O(1)$ with **Zobrist hashing**: $H(\pi) = \bigoplus_i z_{i\pi_i}$, and a swap
updates it by XOR-ing four keys. With $m$ stored 64-bit hashes, a collision has probability at most
$m^2/2^{65}$.

**Strategic oscillation** (Glover 1977; Glover & Laguna 1997) is for constrained problems whose optima lie on the
feasibility boundary. The search crosses the boundary to a controlled depth and then returns. For the 0/1 knapsack,
an ADD phase inserts items by best $v_i/w_i$ until the load exceeds $(1 + \text{depth}) C$, and a DROP phase removes
items by worst $v_i/w_i$ until the solution is feasible again. The best feasible solution seen is kept. The
feasible-only baseline uses the same rules but never overshoots, so the two differ in one mechanism only.

## 4. ILS, VNS and GRASP

All three search the set of local optima $S^{\ast} = LS(S)$ of a fixed local search. Here that local search is
**VND** over 2-opt and Or-opt, and all moves are evaluated with vectorised deltas.

**ILS** (Lourenço, Martin & Stützle 2003):

```text
x <- LS(x0);  repeat:  y <- LS(Perturb(x));  x <- Accept(x, y)
```

The perturbation is the **double bridge** of Martin, Otto & Felten (1991): cut the tour into $A B C D$ and reconnect
it as $A D C B$ without reversing any segment (equivalently, after rotating the labels, $B A D C$). It removes
the four edges between consecutive segments and adds four new ones. It is a non-sequential 4-change: it is the
composition of two "bridge" 2-changes, each of which alone would split the tour into two cycles, and 2-opt, 3-opt or
Lin–Kernighan cannot easily undo it. By contrast, the reconnection $A C B D$, which many texts and code snippets call
the double bridge, keeps the edge from the end of $D$ back to the start of $A$. It changes only three edges and is
the pure 3-opt (Or-opt-type) segment move, which a 3-opt local search undoes in one step. When two adjacent segments
are single cities, one of the "new" edges coincides with an old one. Exactly, $A D C B$ changes $4 - s$ edges, where
$s$ counts cyclically adjacent pairs of singleton segments. The notebook verifies both edge counts on random cuts.
The acceptance rule sets the balance between intensification and diversification:

- *better*: accept only improvements (intensification).
- *random walk*: accept every new local optimum (diversification).
- *LSMC*: accept with the Metropolis probability $e^{-(f(y)-f(x))/T}$, which lies between the two.
- *restart*: replace the kick with a fresh random tour. ILS then becomes random-restart local search.

The **revert rate** is the fraction of kicks after which the local search returns to a solution of the same length.
It measures wasted kicks, and it falls as the kick strength grows.

**VNS** (Mladenović & Hansen 1997) shakes in a sequence of increasingly disruptive neighbourhoods
$N_1, \dots, N_{k_{max}}$, often but not necessarily nested. In the notebook $N_k$ is "$k$ random double bridges":

```text
k <- 1;  repeat:  y <- LS(random point of N_k(x));
                  if f(y) < f(x): x <- y, k <- 1  else: k <- k + 1 (wrap at kmax)
```

VND is the deterministic version: it descends in $N_1$, switches to $N_2$ when stuck, and returns to $N_1$ after
every improvement. A VND local optimum is locally optimal for every neighbourhood in the list.

**GRASP** (Feo & Resende 1989, 1995) repeats a greedy randomised construction followed by local search. At each
construction step, with candidate costs $c(j)$,

$$
RCL = \lbrace j : c(j) \le c_{min} + \alpha (c_{max} - c_{min}) \rbrace ,
$$

and the next element is drawn uniformly from the RCL. $\alpha = 0$ is greedy and $\alpha = 1$ is uniformly random.
Small $\alpha$ gives good starts with little diversity. Large $\alpha$ gives diverse starts that need long, expensive
local searches.

## 5. Graph colouring with TabuCol

For fixed $k$, minimise the number of conflicting edges $f(c)$ (Hertz & de Werra 1987). The neighbourhood recolours
one vertex that is currently in conflict. With $\gamma_{vc} = \vert \lbrace u \in N(v): c(u) = c \rbrace \vert$ the
move delta is $\gamma_{vc'} - \gamma_{vc(v)}$, computed in $O(1)$, and only the rows of $N(v)$ change after a move.
The tabu attribute is $(v, c_{old})$, with dynamic tenure $U\lbrace 0..9 \rbrace + \lfloor 0.6 \vert C \vert \rfloor$
(Galinier & Hao 1999). To bracket $\chi$ from above, decrease $k$ from the DSATUR bound (Brélaz 1979) until TabuCol
fails. A greedy clique gives the lower bound $\omega \le \chi$.

## 6. Fair comparison and the Wilcoxon signed-rank test

Two algorithms run on the same $m$ instances with the same budget give paired differences $d_i$. Drop zeros, rank
$\vert d_i \vert$ with average ranks for ties, and let $W^{+}$ be the sum of the ranks of positive differences. Under
$H_0$ each sign is a fair coin, so without ties $P(W^{+} = w) = c_m(w)/2^m$, where $c_j(w) = c_{j-1}(w) + c_{j-1}(w-j)$.
With ties, use the normal approximation with mean $m(m+1)/4$ and variance
$m(m+1)(2m+1)/24 - \sum_g (t_g^3 - t_g)/48$. The lab implements both and checks them against `scipy.stats.wilcoxon`.

## How the notebooks fit

| Notebook | What you build |
|---|---|
| `01_tabu_search_fundamentals` | QAP swap deltas ($O(n)$, matrix form, Taillard update) validated by recomputation; TS with strong/weak tabu rules, aspiration and candidate lists; exact validation vs `qap_brute_force`; cycling at tenure 0; tenure sweep (U-shape) and candidate-list/aspiration study at equal budget |
| `02_intensification_diversification_and_reactive_ts` | Frequency-based diversification, elite path relinking, robust TS, reactive TS with validated Zobrist hashing; equal-budget comparison on QAP $n = 25$ and attribute-coverage entropy; strategic oscillation on knapsack validated vs `knapsack_dp` with a depth study |
| `03_ils_vns_and_grasp` | Vectorised 2-opt/Or-opt VND validated move by move; double bridge (edge counts verified); ILS with four acceptance rules and kick-strength study (revert rate); VNS; GRASP $\alpha$ trade-off; Held–Karp validation; equal-budget comparison on $n = 100$ |
| `lab_tabu_search_for_graph_colouring_and_qap` | TabuCol with incremental $\gamma$ (validated), exact $\chi$ by backtracking, clique/DSATUR/greedy bounds on $G(n,p)$; robust TS vs SA on QAP $n = 25$ with a from-scratch Wilcoxon test validated against SciPy; TS for feature selection vs forward selection, random search at equal fits and exhaustive search |

## Common pitfalls

- **Off-by-one in the tenure.** If `tabu_until = k + t` is tested with `> k'` instead of `>= k'`, tenure 1 has no
  effect at all. Always test the smallest tenures explicitly.
- **Comparing iterations instead of evaluations.** One full-neighbourhood TS iteration on QAP examines
  $n(n-1)/2$ moves, while one SA step examines 1. Only equal evaluation budgets are fair.
- **Validating an incremental delta against itself.** Check against a full recomputation from the definition, and
  also along a long trajectory, where errors accumulate.
- **Fixed short tenures on small instances.** The search can be trapped in a periodic orbit. Randomise the tenure,
  react to repetitions, or use a stronger tabu-activation rule.
- **Kicks the local search can undo.** Measure the revert rate. A kick that is almost always reverted wastes the
  budget, and a kick that is too strong turns ILS into random restart.
- **Initialising long-term memory as "never".** A long-term aspiration or frequency rule that treats unused
  attributes as used "infinitely long ago" fires on almost every move at the start of the run. Initialise such
  memories as "used at iteration 0".
- **Calling a 3-change a double bridge.** Count the edges that change. $A C B D$ changes three, so a 3-opt local
  search can undo it.
- **Reporting infeasible "best" solutions.** In strategic oscillation, keep the best feasible solution separately
  from the current, possibly infeasible, one.
- **Unpaired tests on paired data.** When both algorithms ran on the same instances, use a signed-rank test and
  report win counts and effect sizes along with the $p$-value.

## Reading

1. F. Glover and M. Laguna, *Tabu Search*, Kluwer, 1997. Ch. 2–4 (short- and long-term memory, aspiration,
   candidate lists, intensification and diversification, strategic oscillation, path relinking).
2. É. Taillard, "Robust taboo search for the quadratic assignment problem", *Parallel Computing* 17 (1991)
   443–455; R. Battiti and G. Tecchiolli, "The reactive tabu search", *ORSA J. Computing* 6(2) (1994) 126–140.
3. H. R. Lourenço, O. C. Martin and T. Stützle, "Iterated local search", in *Handbook of Metaheuristics*
   (Glover & Kochenberger, eds.), Kluwer, 2003, ch. 11.
4. N. Mladenović and P. Hansen, "Variable neighborhood search", *Computers & Operations Research* 24(11) (1997)
   1097–1100.
5. T. A. Feo and M. G. C. Resende, "Greedy randomized adaptive search procedures", *J. Global Optimization* 6
   (1995) 109–133.
6. A. Hertz and D. de Werra, "Using tabu search techniques for graph coloring", *Computing* 39 (1987) 345–351;
   P. Galinier and J.-K. Hao, "Hybrid evolutionary algorithms for graph coloring", *J. Combinatorial Optimization* 3
   (1999) 379–397.
