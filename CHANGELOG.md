# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `nbc.*` in the Torch backend. The graph is selected under `no_grad` and the
  selected edge lengths are recomputed from `x`, so the four distance features
  carry a design gradient; `nbc.nb_fitness.cor` also differentiates through `y`
  at a locally fixed graph.
- `nbc_soft.*`: five differentiable surrogates of the nearest-better features,
  in both backends. A smooth first-success distribution over distance-ordered
  candidates replaces the discrete "first strictly better neighbour" search, so
  the statistics have a usable gradient with respect to `y`. They converge to
  their exact `nbc.*` counterparts as the temperature approaches zero for
  distinct objective values. Report the exact group; the surrogate is
  temperature dependent and is not an NBC feature.
- `nbc_soft` options group: `temperature` (default `0.003`) and
  `rank_temperature` (default `0.5`). The gates always rank the objectives;
  min-max gates were measured as the alternative and diverged from the exact
  features 2.4 to 7.8 times further on BBOB, so they are not offered.

### Changed

- `result.metadata.options` now also reports the `nbc_soft` group; compare
  individual groups rather than the whole snapshot.

## 0.1.0

First public release. A selective and fast engine for exploratory landscape
analysis (ELA), and a successor to [`pflacco`](https://github.com/Reiyan/pflacco).

### Highlights

- Selective feature computation: request individual features by name via
  `compute(sample, [...])`, with shared intermediates reused across features,
  explicit cost tiers, and versioned feature definitions.
- 23 features across five families:
  - `ela_distr` — distribution features (skewness, kurtosis).
  - `ela_meta` — meta-model features (linear and quadratic model fits).
  - `ic` — information content of the fitness landscape.
  - `nbc` — nearest-better clustering features.
  - `fitness_distance` — fitness-distance correlation features.
- Objective values are min-max normalized by default; pass `y_normalization=None`
  to use raw canonical objectives.
- Undefined features return an `invalid` status with an explanation instead of
  raising, so partial results survive.
- Optional differentiable PyTorch backend (`pip install "orivex[torch]"`) for
  the distribution, meta-model, and fitness-distance features, with autograd and
  device/dtype support.
- Immutable `LandscapeSample` inputs with fingerprint integrity checks.
- Ships type information (`py.typed`); typed public API surface.

### Public API

- Exports `compute`, `list_features`, `LandscapeSample`, `ObjectiveSense`,
  `YNormalization`, and the feature-specification types (`FeatureSpec`,
  `CostModel`, `CostTier`, `InputRequirement`, `InvarianceClaim`, `MetricKind`,
  `Reference`, `Transformation`, and related capability enums).

### Documentation

- Published documentation at <https://helix-agh.github.io/orivex/>, covering
  installation, feature selection, normalization, fitness-distance conventions,
  the PyTorch backend, and the API reference.

### Tooling

- CI across Python 3.10, 3.12, and 3.14, plus a CPU PyTorch test job.
- PyPI publishing via GitHub Actions using Trusted Publishing (OIDC).
