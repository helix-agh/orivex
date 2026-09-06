# Fitness-distance reference fixture

`pflacco_fitness_distance.json` contains all six outputs from the unchanged
`calculate_fitness_distance_correlation` in the sibling `../pflacco` checkout.
Source SHA-256 hashes are recorded in the JSON. The reference function and its type
validator were loaded by AST extraction in the former fitness-std microbenchmark,
avoiding unrelated optional pflacco dependencies. The current
`benchmarks/bbob_correctness.py` imports the complete reference modules directly;
the existing fixture and its recorded source hashes remain unchanged.

Inputs use NumPy `default_rng(73)`: first `uniform(-4, 4, (30, 3))` for X, then
`normal(size=30)` for y. The six cases combine minimization and maximization with proportions 0.1, 0.5,
and 0.99. They use pflacco's default Euclidean distance and best selected observation
as reference. These are the corresponding unchanged outputs from the original fixture;
only cases exercising the removed custom-reference and custom-norm options were dropped.

Objective values are unnormalized. Runtime is omitted. The fixture needs neither
pflacco nor pandas at test time. Full-sample calculations, zero-variance behavior,
normalization, deterministic ties, and Torch gradients have separate tests because
they require explicit semantics beyond these reference cases. In particular, the
inspected pflacco implementation leaves `fopt_idx` undefined at proportion 1.
