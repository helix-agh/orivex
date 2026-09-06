# Samples

Construct `LandscapeSample(x, y, lower, upper, sense="minimize")` from paired finite observations.
`x` has shape `(n, d)`, `y` has shape `(n,)`, and both bounds have shape `(d,)`.
The bounds are inclusive and each lower bound must be strictly below its upper bound.

Arrays are copied into read-only `float64` storage. Computation checks that exposed arrays
have not been modified since construction. Read [samples and objective sense](../user-guide/samples.md)
for the integrity and maximization conventions.

::: orivex.LandscapeSample
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
        - validate_unchanged

::: orivex.ObjectiveSense

::: orivex.ObjectiveSenseName
