# API reference

The reference is built from the source using mkdocstrings. Start with the
[quick start](../getting-started/quickstart.md) for a complete example.

| Area | Public entry points |
| --- | --- |
| [Computation](computation.md) | `orivex.compute`, `orivex.list_features` |
| [Samples](sample.md) | `orivex.LandscapeSample`, `orivex.ObjectiveSense` |
| [Results](results.md) | `orivex.result.ComputationResult`, `FeatureValue`, `ExecutionMetadata` |
| [Specifications](specifications.md) | `orivex.FeatureSpec`, costs, invariances, and capabilities |
| [PyTorch](torch.md) | `orivex.torch.compute`, `TensorLandscapeSample`, discovery functions |

The NumPy API is available from `orivex`. Import the optional tensor backend explicitly
from `orivex.torch`. Import result models from `orivex.result` and feature selection errors
from `orivex.registry`.

Most users need only a sample and `compute`; the engine, planner, and intermediate calculators
are implementation details.
