# Solutions

Worked solutions to every exercise in the course (6 per notebook, 162 in total).

> Attempt each exercise yourself first. Most of what you learn comes from being stuck for a while.

| Week | File | Exercises |
|------|------|-----------|
| 1 | [Foundations & Local Search](week1_foundations_and_local_search.md) | 24 |
| 2 | [Simulated Annealing](week2_simulated_annealing.md) | 24 |
| 3 | [Tabu Search & Trajectory Methods](week3_tabu_search_and_trajectory_methods.md) | 24 |
| 4 | [Genetic Algorithms](week4_genetic_algorithms.md) | 30 |
| 5 | [Swarm Intelligence](week5_swarm_intelligence.md) | 30 |
| 6 | [Mechanisms & Methodology](week6_mechanisms_and_methodology.md) | 30 |

Conventions:
- Solutions follow notebook order. Each exercise keeps its difficulty rating (★ / ★★ / ★★★).
- Proof exercises have complete proofs.
- Coding exercises give concise NumPy code that has actually been run, together with the key numbers it produces. Some results are honest negatives (e.g. a variant that does *not* beat the baseline at the given budget); they are reported as measured.
- Every ```python block is self-contained (its own imports and helpers, seeded randomness) and runs standalone from inside the corresponding week folder. `make solutions` (or `python tools/run_solution_blocks.py`) executes all of them; CI does the same on every push. Outputs are shown in ```text blocks.
