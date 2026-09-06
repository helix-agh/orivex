# Results and metadata

`ComputationResult.values` is a read-only mapping from resolved feature names to `FeatureValue`
objects. NumPy computations return floating-point scalars; Torch computations return scalar
tensors. A mathematically undefined output has `value=None`, `status=INVALID`, and a message.

`FeatureValue.definition` identifies the mathematical definition. `ExecutionMetadata` records
input and preprocessing fingerprints, options, requested features, evaluated intermediates,
runtime, additional objective evaluations, and backend execution details.

Use the preprocessing fingerprint **together with feature definitions and execution options**
for result caching. See [results and provenance](../user-guide/results.md).

::: orivex.result.ComputationResult

::: orivex.result.FeatureValue

::: orivex.result.FeatureStatus

::: orivex.result.ExecutionMetadata

::: orivex.BackendName

::: orivex.DeviceType

::: orivex.FloatingDType
