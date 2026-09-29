# METAHEURISTIC OPTIMIZATION

**The five classic metaheuristics and every mechanism behind them, built from scratch in NumPy.**

A six-week research curriculum on heuristic search: **Simulated Annealing, Tabu Search, Genetic Algorithms,
Particle Swarm Optimization and Ant Colony Optimization**, with Differential Evolution and CMA-ES as modern
baselines. Each algorithm is taken apart into its mechanisms (representation, neighbourhood and variation,
selection and acceptance, memory, stochasticity, parameter control) and each mechanism is analysed on its own.

Metaheuristics solve problems that gradient methods cannot reach: combinatorial scheduling and routing,
black-box engineering design, hyperparameter and neural-architecture search, feature selection, program and
prompt search. They are also easy to misuse. Results often come from a single seed, rest on an unfair evaluation
budget, depend on centre-biased benchmarks, or are hidden behind a new natural metaphor. This course teaches both
how the algorithms work and how to evaluate them honestly.

> **No black boxes.** Every algorithm is derived precisely, implemented in pure NumPy, validated against an exact
> solver (Held–Karp, knapsack DP, brute-force QAP), a closed form or a known theorem, and compared with others only
> at **equal evaluation budgets over multiple seeds, using proper statistical tests**. The 27 notebooks contain
> **575+ automated `[PASS]` checks** and are committed fully executed with their outputs. Worked solutions to all
> **162 exercises** are in [`solutions/`](solutions/); every code block there runs standalone and is executed by CI.

---

## The five algorithms at a glance

| Algorithm | Origin | Search state | Generation | Selection / acceptance | Memory |
|-----------|--------|-------------|-----------|------------------------|--------|
| **Simulated Annealing** | Kirkpatrick, Gelatt & Vecchi 1983; Černý 1985 | one solution | random neighbour | Metropolis rule with a temperature schedule | none (Markov) |
| **Tabu Search** | Glover 1986 | one solution | best admissible neighbour | best non-tabu move, aspiration | short- and long-term attribute memory |
| **Genetic Algorithm** | Holland 1975; Goldberg 1989 | population | crossover + mutation | fitness-based selection, replacement, elitism | the population |
| **Particle Swarm Optimization** | Kennedy & Eberhart 1995 | positions + velocities | velocity update towards personal and social bests | personal/neighbourhood best update | personal bests, topology |
| **Ant Colony Optimization** | Dorigo 1992 | pheromone matrix | probabilistic construction | evaporation + reinforcement by good solutions | pheromone trails |

Week 6 shows that all five (in continuous form: tabu search with a distance-based tabu list, ACO<sub>R</sub> for ACO), together with DE and CMA-ES, are instances of one generic loop:
*sample from a distribution parameterised by memory, then select and update the memory*.

## Curriculum

| Week | Topic | Core ideas | Capstone lab |
|------|-------|-----------|--------------|
| 1 | [Foundations & Local Search](week1_foundations_and_local_search/) | Black-box optimisation, NP-hardness, No Free Lunch (verified exhaustively), representations, neighbourhoods, hill climbing, fitness landscapes (autocorrelation, FDC, NK models, local optima networks) | Benchmark suite, the course's experiment harness (ERT, success rates), random-search scaling law |
| 2 | [Simulated Annealing](week2_simulated_annealing/) | Metropolis rule and detailed balance, Boltzmann concentration, Hajek's cooling theorem, schedules, threshold accepting, great deluge, late acceptance, continuous SA, parallel tempering | SA for TSP (O(1) 2-opt/or-opt deltas, 1-tree lower bounds) and MAX-3SAT at the phase transition |
| 3 | [Tabu Search & Trajectory Methods](week3_tabu_search_and_trajectory_methods/) | Tabu memory, tenure, aspiration, Taillard's QAP deltas, long-term memory, path relinking, robust and reactive TS, strategic oscillation, ILS, VNS, GRASP | TabuCol graph colouring, robust TS vs SA on QAP with Wilcoxon tests, TS for feature selection |
| 4 | [Genetic Algorithms](week4_genetic_algorithms/) | Selection pressure and takeover time, SUS, crossover and mutation for binary/real/permutation encodings, schema theorem, deception, (1+1) EA runtime on OneMax/LeadingOnes, Nix–Vose Markov chain, niching, constraint handling | Memetic algorithm for TSP (Lamarckian vs Baldwinian) and regularised evolution for toy neural architecture search |
| 5 | [Swarm Intelligence](week5_swarm_intelligence/) | PSO: inertia, constriction, topologies, order-1/order-2 stability regions; ACO: Ant System, ACS, MAX–MIN AS, gradient and cross-entropy views, convergence, ACO<sub>R</sub> | Swarm showdown on shifted-rotated benchmarks; PSO for hyperparameter tuning |
| 6 | [Mechanisms & Methodology](week6_mechanisms_and_methodology/) | One generic loop for all metaheuristics, exploration vs exploitation measured, the metaphor critique, DE, JADE, CMA-ES, 1/5th rule, parameter control, bandit operator selection, F-Race, hyper-heuristics, NSGA-II, MOEA/D, hypervolume | Grand comparison with ERT, ECDFs, performance profiles, Friedman + Nemenyi critical-difference diagrams |

## Mechanism index: where each idea is studied

| Mechanism | Where |
|-----------|-------|
| Representation, genotype–phenotype maps, Gray coding | W1 NB2, W4 NB2 |
| Neighbourhood structures (bit-flip, swap, insertion, 2-opt, or-opt, double bridge, Gaussian/Cauchy steps) | W1 NB2, W2 NB3/Lab, W3 NB3 |
| Acceptance criteria (greedy, Metropolis, threshold, great deluge, record-to-record, late acceptance) | W2 NB1–NB2, W3 NB3 |
| Temperature / step-size / parameter control (schedules, 1/5th rule, self-adaptation, CSA, reactive tabu tenure) | W2 NB2–NB3, W3 NB2, W6 NB2–NB3 |
| Memory (tabu lists, frequency memory, elite sets, pheromone, personal bests, evolution paths) | W3 NB1–NB2, W5, W6 NB1–NB2 |
| Selection operators and selection pressure | W4 NB1 |
| Recombination and mutation operators | W4 NB2, W6 NB2 |
| Diversity maintenance (niching, crowding, clearing, islands, restarts, pheromone bounds) | W1 NB2/Lab, W2 NB3, W3 NB3, W4 NB4, W5 NB3 |
| Constraint handling (penalties, repair, feasibility rules, stochastic ranking, strategic oscillation) | W3 NB2, W4 NB4, W5 NB4 |
| Hybridisation (memetic algorithms, ACO + local search, GRASP) | W4 Lab, W5 NB3, W3 NB3 |
| Swarm dynamics and stability | W5 NB2 |
| Model-based search (pheromone as a distribution, cross-entropy, CMA-ES) | W5 NB4, W6 NB2 |
| Multi-objective search (Pareto sorting, crowding, decomposition, hypervolume) | W6 NB4 |
| Adaptive operator selection, tuning, hyper-heuristics | W6 NB3 |
| Theory (NFL, Metropolis chains, Hajek, schema theorem, runtime analysis, Markov-chain models, PSO stability, ACO convergence, ES progress rates, Zaharie's variance law) | W1 NB1, W2 NB1–NB2, W4 NB3, W5 NB2/NB4, W6 NB2–NB3 |
| Benchmarking methodology and statistics | W1 Lab, W6 Lab |

## Repository structure

```
metaheuristic_optimization/
├── README.md
├── LICENSE
├── CITATION.cff
├── Makefile                       # make install | test | notebooks | solutions | check
├── requirements.txt
├── .github/workflows/ci.yml       # unit tests, every notebook, every solution block; fails on any [FAIL]
├── week1_foundations_and_local_search/
│   ├── lecture.md
│   ├── 01_optimization_problems_and_no_free_lunch.ipynb
│   ├── 02_representations_neighbourhoods_and_local_search.ipynb
│   ├── 03_fitness_landscape_analysis.ipynb
│   └── lab_benchmark_suite_and_baselines.ipynb
├── week2_simulated_annealing/
│   ├── lecture.md
│   ├── 01_metropolis_and_simulated_annealing.ipynb
│   ├── 02_cooling_schedules_and_convergence.ipynb
│   ├── 03_continuous_sa_and_parallel_tempering.ipynb
│   └── lab_sa_for_tsp_and_max_sat.ipynb
├── week3_tabu_search_and_trajectory_methods/
│   ├── lecture.md
│   ├── 01_tabu_search_fundamentals.ipynb
│   ├── 02_intensification_diversification_and_reactive_ts.ipynb
│   ├── 03_ils_vns_and_grasp.ipynb
│   └── lab_tabu_search_for_graph_colouring_and_qap.ipynb
├── week4_genetic_algorithms/
│   ├── lecture.md
│   ├── 01_ga_anatomy_and_selection.ipynb
│   ├── 02_variation_operators.ipynb
│   ├── 03_ga_theory.ipynb
│   ├── 04_diversity_niching_and_constraints.ipynb
│   └── lab_memetic_algorithm_for_tsp_and_nas_toy.ipynb
├── week5_swarm_intelligence/
│   ├── lecture.md
│   ├── 01_particle_swarm_optimization.ipynb
│   ├── 02_pso_dynamics_and_stability.ipynb
│   ├── 03_ant_colony_optimization.ipynb
│   ├── 04_aco_theory_and_extensions.ipynb
│   └── lab_swarm_showdown.ipynb
├── week6_mechanisms_and_methodology/
│   ├── lecture.md
│   ├── 01_a_unified_view_of_metaheuristic_mechanisms.ipynb
│   ├── 02_differential_evolution_and_cma_es.ipynb
│   ├── 03_parameter_tuning_control_and_hyper_heuristics.ipynb
│   ├── 04_multi_objective_metaheuristics.ipynb
│   └── lab_rigorous_benchmarking_grand_comparison.ipynb
├── utils/
│   ├── benchmarks.py              # 8 continuous test functions with known optima, shift + rotation
│   ├── problems.py                # TSP, knapsack, QAP, MAX-3SAT generators + exact solvers
│   ├── tracking.py                # BudgetedObjective: evaluation counting, best-so-far traces
│   ├── plotting.py                # shared matplotlib style, savefig -> figures/
│   └── checks.py                  # check_close / check_true validation helpers
├── solutions/                     # worked solutions to every exercise, one file per week
├── tools/
│   └── run_solution_blocks.py     # executes every code block in solutions/ standalone
├── tests/
│   └── test_utils.py              # pytest suite for utils/
├── references/
│   └── reading_list.md
└── figures/                       # every figure produced by the notebooks (w1_*.png ... w6_*.png)
```

## The shared test bed (`utils/`)

- **Continuous benchmarks:** sphere, ill-conditioned ellipsoid, Rosenbrock, Rastrigin, Ackley, Griewank, Schwefel 2.26 and Levy, all with recorded domains and global optima. `shifted_rotated` and `random_rotation` remove separability and centre bias.
- **Combinatorial problems with exact reference solvers:** Euclidean TSP (Held–Karp dynamic programming for n ≤ 13, and the Beardwood–Halton–Hammersley estimate for large n), 0/1 knapsack (Pisinger-style instances and an exact DP), QAP (brute force for n ≤ 9) and random MAX-3SAT.
- **Fair comparisons:** `BudgetedObjective` counts every function evaluation, records best-so-far traces and stops an algorithm when its budget runs out. Every comparison in the course is made at equal evaluation budgets.

## Prerequisites

- **Mathematics:** probability (Markov chains help), basic linear algebra and calculus, and some familiarity with algorithms and complexity.
- **Programming:** Python 3.10+, NumPy, Jupyter.
- **Helpful but optional:** the `optimization_methods`, `probability`, `discrete_mathematics` and `decision_theory` courses in this profile.

## Getting started

```bash
git clone https://github.com/HAYDARKILIC/metaheuristic_optimization.git
cd metaheuristic_optimization
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

To verify the installation, run `make test` (the unit tests for `utils/`), `make notebooks` (re-executes all 27 notebooks, about 15 minutes on a laptop) and `make solutions` (runs every code block in `solutions/`).
Notebooks import the shared helpers with `sys.path.insert(0, "..")`, so run them from inside their week folder (Jupyter does this by default).

## How to use each week

1. Read `lecture.md` (1–2 hours): definitions, pseudocode, theory.
2. Work through the numbered notebooks in order. Predict each experimental result before running it, then compare with the `[PASS]` checks and the statistics.
3. Do the capstone `lab_*.ipynb` as a mini research project.
4. Attempt the six exercises at the end of every notebook (★ warm-up, ★★ standard, ★★★ research-flavoured).
5. Only then compare with the worked solutions in [`solutions/`](solutions/). Proofs are complete, and every coding solution has been run.

## Principles of honest metaheuristic research (followed throughout)

1. Compare algorithms at **equal numbers of function evaluations**, not iterations or wall time.
2. Report **distributions over seeds** (median/IQR, ECDFs), never a single run.
3. Use **paired non-parametric tests** with multiple-comparison correction (Wilcoxon, Friedman + Nemenyi, Holm).
4. **Shift and rotate** benchmarks, since an optimum at the origin rewards centre-biased algorithms.
5. Validate against **exact optima** wherever the instance is small enough.
6. Describe algorithms by their **mechanisms**, not their metaphors (Sörensen 2015).
7. Remember **No Free Lunch**: a claim that one algorithm is better only means something for a stated problem class.

## Where metaheuristics show up in AI

| Application | Technique in this course |
|-------------|-------------------------|
| Hyperparameter optimisation | PSO, CMA-ES, DE, racing (F-Race), bandit-based control, SA over discrete configurations |
| Neural architecture search | Regularised (aging) evolution, GA over discrete configurations, multi-objective NAS |
| Feature selection | Tabu search, SA over subsets |
| Policy search in RL | CMA-ES, evolution strategies |
| Program / prompt search (as exercises) | GA with permutation operators (W4 NB2) and variable-length insert/delete mutation (W4 Lab, Ex. 6), ILS (W3 NB3, Ex. 6) |
| Scheduling, routing and allocation | SA, TS, ACO, memetic GA on TSP, QAP, colouring and bin packing |
| Trade-offs such as accuracy vs latency | NSGA-II, MOEA/D, hypervolume |

## Reading

See [`references/reading_list.md`](references/reading_list.md). The core texts are Talbi, *Metaheuristics: From Design to Implementation*; Gendreau & Potvin (eds.), *Handbook of Metaheuristics*; Hoos & Stützle, *Stochastic Local Search*; Eiben & Smith, *Introduction to Evolutionary Computing*; Dorigo & Stützle, *Ant Colony Optimization*; and Glover & Laguna, *Tabu Search*.

## License

MIT for code; CC-BY-4.0 for prose and figures.

---

**Author:** Haydar Kılıç · [LinkedIn](https://www.linkedin.com/in/haydarkilicai) · haydarkilicinfo@gmail.com
