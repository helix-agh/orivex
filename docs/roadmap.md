# Roadmap

## Phase 0: correctness foundation

- Define the feature-specification schema.
- Catalogue verified defects and ambiguous semantics in `pflacco` 1.2.2.
- Create reusable analytical landscapes.
- Generate portable differential fixtures from R `flacco`.
- Define metamorphic properties and their precise preconditions.

## Phase 1: execution architecture

- Validate immutable `LandscapeSample` inputs.
- Register individual, versioned feature definitions.
- Resolve shared intermediates through a dependency graph.
- Refuse requests that exceed declared objective-evaluation or memory budgets.
- Return values, statuses, warnings, provenance, and costs separately.

## Phase 2: initial feature profile

Implement and validate candidate zero-additional-evaluation features:

- distribution skewness and kurtosis;
- selected linear and quadratic meta-model fit statistics;
- nearest-better clustering;
- dispersion ratios;
- information content;
- fitness-distance correlation.

PCA, level-set, cell-mapping, and additional-evaluation features remain compatibility or
experimental candidates until the feature-triage study supports their inclusion.

## Phase 3: evidence-driven optimization

- Benchmark across sample size, dimension, and requested feature subsets.
- Remove redundant allocations and use blockwise algorithms where appropriate.
- Parallelize across independent landscapes.
- Add Rust only for measured hotspots that remain material after algorithmic and NumPy/SciPy
  optimization.
