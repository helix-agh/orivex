# R flacco differential fixtures

This directory will contain portable, versioned fixtures generated from fixed sample matrices.
Every fixture must record:

- R and `flacco` versions;
- orivex feature-definition target;
- full feature parameters;
- sample-generation method, seed, and bounds;
- input checksum;
- output values with round-trip-safe float precision.

R outputs are compatibility evidence. A disagreement is resolved using the mathematical
specification and analytical/metamorphic tests rather than automatically favoring either
implementation.
