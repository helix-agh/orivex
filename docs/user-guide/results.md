# Results, status, and provenance

`compute` returns a `ComputationResult`. Its read-only `values` mapping associates each
resolved feature name with a `FeatureValue`; execution information is separate in `metadata`.
The snippets below use the sample from the [quick start](../getting-started/quickstart.md).

## Mathematically undefined outputs

A feature that is undefined for an otherwise valid sample returns a `FeatureValue` with status
`invalid` and an explanation rather than raising or silently producing `NaN`.

```python
from orivex.result import FeatureStatus

flat = LandscapeSample(x, np.zeros(len(x)), lower=lower, upper=upper)
item = compute(flat, "ela_distr.skewness").values["ela_distr.skewness"]

item.status is FeatureStatus.INVALID  # True
item.value  # None
item.message  # 'skewness is undefined for constant objective values'
```

A selector that matches no registered feature is a caller error and raises instead:

```python
from orivex.registry import UnknownFeatureSelection

compute(sample, "ela_meta.nonexistent")
# UnknownFeatureSelection: selector matched no features: ela_meta.nonexistent
```

## Recording a computation

Record requested features, definition identifiers, preprocessing, effective options, backend,
dtype, and execution settings with numerical results. Runtime belongs to metadata and is not
an ELA feature.

```python
metadata = result.metadata
print(metadata.preprocessing_fingerprint)
print(metadata.y_normalization_definition)
print(metadata.options)
print(metadata.backend, metadata.device, metadata.dtype)

for name, item in result.values.items():
    print(name, item.definition, item.status.value, item.message)
```

A raw sample fingerprint alone is insufficient as a result cache key. Include the preprocessing
fingerprint, feature definitions, and effective execution options. See the
[result API](../api/results.md) for all metadata fields.
