# PyTorch backend

Install the optional `torch` extra to run this API. Read the
[PyTorch guide](../user-guide/torch.md) for an example with backpropagation.

`compute` accepts a `TensorLandscapeSample`, feature selectors, `y_normalization`, and `options`.
It returns the same result model as the NumPy backend, with scalar tensors as feature values.
This backend has no `rng` or `workers` parameter.

`list_features()` returns specifications for the Torch profile only. `list_capabilities()`
reports device, dtype, and autograd support including the requested preprocessing mode.
Known features without a Torch implementation raise `UnsupportedFeatureError`; unavailable
devices raise `UnsupportedFeatureDeviceError`.

::: orivex.torch.compute

::: orivex.torch.TensorLandscapeSample
    options:
      members:
        - __init__
        - x
        - y
        - lower
        - upper
        - sense
        - n_observations
        - dimension
        - minimization_y
        - fingerprint
        - device_type
        - device_index
        - dtype_name
        - validate_unchanged

::: orivex.torch.list_features

::: orivex.torch.list_capabilities

::: orivex.torch.UnsupportedFeatureError

::: orivex.torch.UnsupportedFeatureDeviceError
