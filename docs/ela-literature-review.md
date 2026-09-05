# Exploratory Landscape Analysis features used in benchmark analysis and machine learning

**Purpose:** choose the essential feature surface for `orivex`, a replacement for `pflacco`
**Status:** living report; the paper-level evidence and feature inventory are usable now,
but the corpus can be extended as new application papers are identified
**Literature checked through:** 30 August 2026
**Primary scope:** continuous, single-objective black-box optimization; GECCO, PPSN,
IEEE CEC/TEVC, *Evolutionary Computation*, and closely related work
**Terminology:** feature names use the modern `flacco`/`pflacco` spelling where possible.
Runtime and evaluation-count metadata are not counted as scientific features.

## Executive recommendation

The evidence supports a compact, data-only compatibility core of **38 scientific
features in five families**:

| Family | Scientific outputs | Why it is essential |
|---|---:|---|
| `ela_distr` | 3 | Extremely cheap; individual members recur in selected ML portfolios and real-world/HPO analyses. |
| `ela_meta` | 9 | The most consistently useful family across performance regression, algorithm selection, high-level property prediction, and mixed-variable work. |
| `ic` | 5 | Sensitivity and ruggedness measures repeatedly appear among selected or SHAP-important features. |
| `nbc` | 5 | Strong evidence for funnel/global-structure detection and algorithm selection. |
| `disp` | 16 | Frequently used as a complete family; individual members are important for dimensionality, mixed-variable tasks, and fixed-budget prediction. |

This 38-feature set is not an arbitrary subset of `pflacco`: it is exactly the recurring
combination of three y-distribution features, nine meta-model features, five information
content features, five nearest-better features, and sixteen dispersion features. It is the
most common practical intersection of the papers reviewed below.

Two changes to the current roadmap follow directly from the literature:

1. **Add `ela_distr.number_of_peaks` to the first implementation profile.** The current
   roadmap mentions only skewness and kurtosis. The peak count was selected in early
   algorithm-selection work, separates HPO from BBOB landscapes, and was the first
   selected feature in both mixed-variable algorithm selectors. Its KDE and mode-mass
   semantics must be specified carefully because implementations differ.
2. **Keep fitness-distance correlation, but treat it as an inexpensive adjunct rather than
   part of the dominant 38-feature core.** FDC is interpretable and cheap, and a PPSN 2022
   baseline used its six outputs, but it appears much less often than meta-model, IC, NBC,
   distribution, or dispersion features in recent selected portfolios.

Recommended build order:

| Priority | Families | Decision |
|---|---|---|
| **P0** | `ela_distr`, `ela_meta`, `ic`, `nbc`, `disp` | Implement all 38 scientific outputs, selectable individually. This is the minimum credible ELA/ML library. |
| **P1** | fitness-distance correlation, PCA, basic sample summaries | Implement next. FDC is cheap and interpretable; PCA matters in high dimension and later optimization budgets; basic summaries are useful controls and appear in recent hybrid selectors. |
| **P2** | `ela_level` | Compatibility target, not default core. It is used in many broad baselines and occasionally selected, but is CPU-heavy, failure-prone, and has an unresolved MDA compatibility gap in Python implementations. |
| **P3** | `limo`, cell mapping, convexity, curvature, local search, barrier tree/general cell mapping | Experimental or compatibility-only until demanded. Cell mapping scales exponentially; convexity, curvature, and local search consume extra objective evaluations; evidence of recurring use in modern ML portfolios is weak. |
| **Separate track** | trajectory features, learned representations (Deep-ELA, TransOpt, DoE2Vec, TinyTLA), and MO-ELA | Design extension points, not aliases for classical single-objective ELA. They require different data models, training artifacts, topological computations, or vector-valued objectives. |

## How to read the evidence

It is important to separate three claims that are often conflated:

- **Computed:** a paper calculated a large feature collection. This establishes API demand,
  but not that every member was useful.
- **Selected/important:** feature selection, a decision tree, or SHAP analysis identified a
  feature. This is stronger evidence for an implementation priority.
- **Methodological:** a paper studied stability, scaling, cost, or invariance. This affects
  semantics and metadata even if it did not train the final predictor.

The review therefore gives more weight to repeatedly selected individual features than to
features merely present in a 50--100 dimensional input vector. This is a focused review, not
a bibliometric census. It emphasizes papers that expose enough detail to identify their
feature families or individual inputs.

## Traceability of the supplied reading list

This table records every identifier supplied for the follow-up review. “Excluded” means that
the paper was checked but is not evidence about ELA features; it does not mean the paper is
unimportant in its own area.

| Supplied identifier | What it is | Features or representation | Use in this report |
|---|---|---|---|
| [10.1145/2001576.2001690](https://doi.org/10.1145/2001576.2001690) | Mersmann et al., GECCO 2011, the foundational ELA paper | 50 outputs: meta-model 7, convexity 3, y-distribution 3, level set 12, local search 10, curvature 15 | **Core source.** Defines the historical surface and demonstrates cost-aware feature-group selection for BBOB classification and high-level-property prediction. |
| [10.1162/evco_a_00341](https://doi.org/10.1162/evco_a_00341) | Prager and Trautmann, *Evolutionary Computation* 2024, `pflacco` package paper | Python implementation of classical and later landscape-feature families for continuous and constrained problems | **Compatibility source, not selection evidence.** Use for API/provenance expectations; do not infer that every implemented family is essential. |
| [10.1145/2739480.2754642](https://doi.org/10.1145/2739480.2754642) | Kerschke et al., GECCO 2015, funnel detection | Nearest-better clustering: nearest-neighbour versus nearest-better distances and fitness/indegree structure | **Core source.** Establishes the NBC family. |
| `10.1109/TEVC.2014.2302235` | Unresolvable DOI as supplied (HTTP 404; absent from Crossref) | No paper can be assigned safely | **Needs correction.** Two likely nearby ELA references are the [information-content paper](https://doi.org/10.1109/TEVC.2014.2302006) and the [dispersion/sampling critique](https://doi.org/10.1109/TEVC.2013.2281521); both are discussed, but neither is silently substituted. |
| [10.1007/978-3-030-58112-1_10](https://doi.org/10.1007/978-3-030-58112-1_10) | Hagg et al., PPSN 2020, surrogate-assisted phenotypic niching for air-flow design | Domain descriptors: polygon area and simulated air-flow turbulence/enstrophy; surrogate models predict the descriptors | **Excluded from ELA priority evidence.** It is quality-diversity/behavior-space work, not exploratory landscape features. It does reinforce that expensive descriptors need explicit evaluation cost. |
| [10.1145/3638530.3654359](https://doi.org/10.1145/3638530.3654359) | Kowalczykiewicz and Lipinski, GECCO Companion 2024, vectorized cluster crossover in grammatical evolution | Genotype clustering/crossover; no ELA vector | **Excluded.** Not a landscape-analysis or ELA-for-ML paper. |
| [10.1016/j.swevo.2025.101894](https://doi.org/10.1016/j.swevo.2025.101894) | Cenikj et al., *Swarm and Evolutionary Computation* 2025 | 62 classical outputs from seven families, calculated for raw and min-max-scaled objectives; compared with learned and topological representations | **Core generalization evidence.** More features do not repair out-of-distribution algorithm selection. |
| [10.1016/j.asoc.2020.106138](https://doi.org/10.1016/j.asoc.2020.106138) | Škvorc et al., *Applied Soft Computing* 2020 | Broad `flacco` vector, redundancy filtering, and objective/decision transformation analysis | **Methodological evidence.** Motivates invariance tests, transforms, and feature selection. |
| [10.1016/j.ins.2013.04.015](https://doi.org/10.1016/j.ins.2013.04.015) | Malan and Engelbrecht, *Information Sciences* 2013, fitness-landscape survey | Taxonomy spanning ruggedness, modality, neutrality, evolvability, FDC, autocorrelation, and related landscape measures | **Background, not an ELA input portfolio.** Useful for future extension taxonomy, but it neither computes a modern `flacco` vector nor selects inputs for an ELA model. |
| [10.1145/3594805.3607136](https://doi.org/10.1145/3594805.3607136) | Prager et al., FOGA 2023, neural benchmark-function generation | Exact target vector of 8 outputs: four meta-model adjusted R2 values, distribution skewness, two NBC outputs, and objective-value standard deviation | **Strong individual evidence.** Shows that a small differentiable/optimizable feature surface can drive benchmark generation. |
| [arXiv:2305.15245](https://arxiv.org/abs/2305.15245) | Long et al., ELA-guided genetic-programming function generation | Broad cheaply computed `pflacco` subset on Sobol samples; compares distances between bootstrapped feature distributions | **Methodological evidence.** Equal weighting and the choice of feature-space distance can dominate the generated landscape. |
| [arXiv:2401.01192](https://arxiv.org/abs/2401.01192) | Seiler et al., Deep-ELA | Pretrained transformer maps raw `(X, Y)` point clouds to 24- or 48-dimensional learned vectors | **Separate learned-feature track.** A model/artifact API is required; it is not a replacement definition for classical features. |
| [arXiv:2311.18035](https://arxiv.org/abs/2311.18035) | Cenikj et al., TransOpt | Transformer embeddings pooled with per-coordinate minimum, maximum, mean, and standard deviation; task-trained on BBOB class labels | **Separate learned-feature track.** Useful baseline, but task-specific and trained on function identity. |
| [arXiv:2602.00098](https://arxiv.org/abs/2602.00098) | Preuß et al., MO-ELA | New multi-objective families: non-dominated sorting, non-dominated-point statistics, PCA, MST/kNN graphs, and gradient summaries | **Future multi-objective track.** Strong AAS evidence, but its inputs and semantics do not fit the single-objective `core38` contract. |

## Representative literature and feature usage

### Foundations and feature construction

| Paper | Venue / task | Features used | Library implication |
|---|---|---|---|
| [Mersmann et al., *Exploratory Landscape Analysis*](https://doi.org/10.1145/2001576.2001690) | GECCO 2011; classify BBOB groups and expert properties | 50 outputs: 7 meta-model, 3 convexity, 3 y-distribution, 12 level-set, 10 local-search, and 15 curvature features | Establishes the classical definitions. Only y-distribution, level set, and meta-model are data-only; the other three need objective access and extra evaluations. No feature group was universally required across the cost/accuracy Pareto set. |
| [Bischl et al., *Algorithm Selection Based on ELA and Cost-Sensitive Learning*](https://doi.org/10.1145/2330163.2330209) | GECCO 2012; cost-sensitive BBOB algorithm selection | Classical ELA; compares all features with the cheap subset (y-distribution, level set, meta-model) | Early evidence that useful selectors can avoid evaluation-expensive classical families. |
| [Kerschke et al., *Detecting Funnel Structures by Means of ELA*](https://doi.org/10.1145/2739480.2754642) | GECCO 2015; funnel classification | Introduces nearest-better clustering and combines it with existing ELA | Makes NBC a first-class family rather than a miscellaneous feature. |
| [Kerschke and Trautmann, *Comprehensive Feature-Based Landscape Analysis ... flacco*](https://doi.org/10.1007/978-3-030-05318-5_7) | 2019 reference chapter/package description | 17 `flacco` families, over 300 outputs including metadata | Useful compatibility catalogue, but not evidence that all 17 families belong in a minimal library. |
| [Muñoz et al., *Exploratory Landscape Analysis ... Using Information Content*](https://doi.org/10.1109/TEVC.2014.2302006) | IEEE TEVC 2015; landscape characterization | Information content of fitness sequences: `h_max`, settling/information sensitivities, `m0` | Source semantics for the five IC outputs that recur in later ML work. |
| [Morgan and Gallagher, *Sampling Techniques and Distance Metrics in High Dimensional Continuous Landscape Analysis*](https://doi.org/10.1109/TEVC.2013.2281521) | IEEE TEVC 2014; analyze dispersion in high dimension | Dispersion under uniform random samples and Euclidean distance; proposes corrected methodology | Essential specification warning: distance concentration and boundary/sample effects can bias dispersion, so normalization and the exact estimator are part of the feature definition. |
| [Prager and Trautmann, *Pflacco*](https://doi.org/10.1162/evco_a_00341) | *Evolutionary Computation* 2024; Python package/reference implementation | Classical and extended feature implementations for continuous and constrained optimization | A compatibility and reproducibility source, not evidence that its entire catalogue belongs in the default profile. |
| [Malan and Engelbrecht, *A Survey of Techniques for Characterising Fitness Landscapes*](https://doi.org/10.1016/j.ins.2013.04.015) | *Information Sciences* 2013; broad landscape-analysis survey | Organizes ruggedness, modality, neutrality, evolvability, FDC, autocorrelation, and related measures | Provides an extension taxonomy. Its broader FLA catalogue should not be conflated with the 50 original ELA outputs or modern `pflacco` families. |

The exact historical GECCO 2011 inventory is useful when building a compatibility layer.
Using the paper's original spelling, its 50 outputs are:

- **Meta-model (7):** `approx.linear_ar2`, `approx.lineari_ar2`,
  `approx.linear_min_coef`, `approx.linear_max_coef`, `approx.quadratic_ar2`,
  `approx.quadratici_ar2`, `approx.quadratic_cond`.
- **Convexity (3):** `convex.linear_p`, `convex.convex_p`, `convex.linear_dev`.
- **y-distribution (3):** `distr.skewness_y`, `distr.kurtosis_y`, `distr.n_peaks`.
- **Level set (12):** `levelset.lda_mmce_q`, `levelset.lda_vs_qda_q`,
  `levelset.qda_mmce_q`, and `levelset.mda_mmce_q` for each
  `q` in `{10, 25, 50}`.
- **Local search (10):** `ls.n_local_optima`, `ls.best_to_mean_contrast`,
  `ls.best_basin_size`, `ls.worst_basin_size`, `ls.mean_other_basin_size`, and
  `ls.{min,lq,med,uq,max}_feval`.
- **Curvature (15):** `numderiv.grad_norm_{min,lq,med,uq,max}`,
  `numderiv.grad_scale_{min,lq,med,uq,max}`, and
  `numderiv.hessian_cond_{min,lq,med,uq,max}`.

These are historical identifiers, not the recommended public names. Modern `flacco` and
`pflacco` changed and expanded parts of the meta-model and level-set surfaces. A replacement
should map old names to a documented mathematical specification rather than promise that
similar-looking old and new names are automatically identical.

### Benchmark analysis, classification, and robustness

| Paper | Venue / task | Features used | Main observation for `orivex` |
|---|---|---|---|
| [Renau et al., *Exploratory Landscape Analysis is Strongly Sensitive to the Sampling Strategy*](https://doi.org/10.1007/978-3-030-58115-2_10) | PPSN 2020; BBOB classification and sampling study | 46 features from `disp`, `ic`, `nbc`, `ela_meta`, `ela_distr`, `pca` | Feature values from random, LHS, improved LHS, and Sobol designs do not converge to a common sampling-independent value. The sampling design is part of feature provenance. Sobol gave the best classification accuracy in their experiment. |
| [Škvorc et al., *Understanding the Problem Space ... Using ELA*](https://doi.org/10.1016/j.asoc.2020.106138) | *Applied Soft Computing* 2020; CEC/BBOB visualization and clustering | Broad `flacco` features followed by redundancy/transform analyses | Supports benchmark-space analysis but shows substantial redundancy and lack of invariance under simple transformations. |
| [Renau et al., *Towards Explainable ELA: Extreme Feature Selection*](https://doi.org/10.1007/978-3-030-72699-7_2) | EvoApplications 2021; BBOB function classification | Starts from 46 features in six families; studies ten candidates: `disp.ratio_mean_02`, distribution skewness, four meta-model outputs, two IC sensitivities, `nbc.nb_fitness.cor`, and one PCA output | Often one to four features reach 98% within-suite classification accuracy, but this requires large samples and degrades under leave-one-instance-out evaluation. Invariance becomes decisive. |
| [Tanabe, *Towards ELA for Large-Scale Optimization*](https://doi.org/10.1145/3449639.3459300) | GECCO 2021; high-level BBOB property prediction to 640D | 107 features in `ela_distr`, `ela_level`, `ela_meta`, `nbc`, `disp`, `ic`, `basic`, `limo`, `pca`; also reduced-space cell-mapping variants | `ela_level` and `ela_meta` become expensive at high dimension; distance-based families also need careful scaling. Selective calculation and explicit CPU/memory costs are essential. |
| [Schneider et al., *HPO x ELA*](https://doi.org/10.1007/978-3-031-14714-2_40) | PPSN 2022; compare XGBoost HPO and BBOB landscapes | `ela_meta`, `ic`, `ela_distr`, `nbc`, `disp` on min-max-normalized `X` and standardized `y` | Decision trees used distribution kurtosis, peak count, `nbc.nb_fitness.cor`, NBC distance ratios, and dispersion ratios. IC/meta/NBC/disp loadings describe latent multimodality and dimensionality. |
| [Prager and Trautmann, *Nullifying the Inherent Bias of Non-invariant ELA Features*](https://doi.org/10.1007/978-3-031-30229-9_27) | EvoApplications 2023; AAS invariance/generalization | Broad ELA before and after objective normalization | Objective shift/scale sensitivity can make selectors learn benchmark identifiers. Normalization must be an explicit, reproducible preprocessing option. |
| [Nikolikj et al., *Generalization Ability of Feature-Based Performance Prediction Models*](https://doi.org/10.1109/CEC60901.2024.10611990) | IEEE CEC 2024; cross-benchmark performance regression | One experiment uses 64 ELA features; another 14, with repeated iLHS or Sobol samples | Generalization correlates with similarity of train/test coverage in feature space. A library cannot solve out-of-distribution generalization merely by exposing more features. |
| [Cenikj et al., *Landscape Features ... Have We Hit a Wall in Algorithm Selection Generalization?*](https://doi.org/10.1016/j.swevo.2025.101894) | *Swarm and Evolutionary Computation* 2025; cross-family/cross-suite AS | 62 outputs from `disp`, `ela_distr`, `ela_level`, `ela_meta`, `ic`, `nbc`, `pca`, both raw and min-max-scaled `y`; compared with learned/topological features | Classical ELA remains competitive in-distribution, but no representation beats the single-best-solver baseline out of distribution. This argues for robust semantics and diagnostics, not an ever-growing default vector. |
| [Cenikj et al., *A Survey of Features Used for Representing Black-Box Single-Objective Continuous Optimization*](https://doi.org/10.1016/j.swevo.2026.102288) | *Swarm and Evolutionary Computation* 2026; current survey | Classical ELA, topological/deep features, algorithm and trajectory representations | Confirms the recent shift toward invariance, trajectories, learned features, and cross-benchmark evaluation while retaining classical ELA as the standard baseline. |

### ELA-guided benchmark generation

| Paper | Venue / task | Feature inputs | Main observation for `orivex` |
|---|---|---|---|
| [Prager et al., *Neural Networks as Black-Box Benchmark Functions Optimized for ELA Features*](https://doi.org/10.1145/3594805.3607136) | FOGA 2023; generate BBOB-like and feature-space-filling benchmark functions | Exact 8-output target: `ela_meta.lin_simple.adj_r2`, `ela_meta.lin_w_interact.adj_r2`, `ela_meta.quad_simple.adj_r2`, `ela_meta.quad_w_interact.adj_r2`, `ela_distr.skewness`, `nbc.nb_fitness.cor`, `nbc.nn_nb.sd_ratio`, `fitness_distance.fitness_std`; LHS of `250d`, min-max-scaled `y` | Strong individual evidence for meta-model fit, skewness, and NBC. Also creates demand for stable, optimizable semantics: a feature can become an objective, not just a predictor column. |
| [Long et al., *Challenges of ELA-guided Function Evolution Using Genetic Programming*](https://arxiv.org/abs/2305.15245) | 2023 preprint; evolve functions toward BBOB targets | Broad cheap `pflacco` subset computed on Sobol designs and bootstrapped; excludes the four PCA outputs concerned only with the design coordinates | Feature-space distance and scaling matter. Equal weighting can overemphasize unstable features such as coefficient ratios, and the statistically motivated Wasserstein distance separated same/different BBOB functions less clearly than cosine or correlation distance in this experiment. |

These papers add a requirement that ordinary algorithm-selection studies can hide: outputs
should be sufficiently deterministic, continuous where their mathematics permits it, and
accompanied by valid ranges/status. KDE peak count and fit failures are especially important
because discontinuities or sentinel values can mislead a feature-guided generator.

### ML models for selection, regression, and configuration

| Paper | Venue / task | Feature inputs and selected features | Main observation for `orivex` |
|---|---|---|---|
| [Kerschke and Trautmann, *Automated Algorithm Selection on Continuous Black-Box Problems*](https://doi.org/10.1162/evco_a_00236) | *Evolutionary Computation* 2019; select from 12 BBOB solvers | 102 inputs from classical ELA, basic, cell angle, dispersion, IC, NBC, PCA. Model 1 selected all three distribution outputs, one level-set ratio, `ic.h_max`, `ic.eps_s`, one cell-angle statistic, and best sampled fitness. The better Model 2 retained skewness, level/cell features and added two meta-model plus four NBC features. | Strong individual evidence for all distribution outputs, quadratic/linear meta-model statistics, and all four structural NBC correlations/ratios. The best model needed only nine features. |
| [Jankovic and Doerr, *Landscape-Aware Fixed-Budget Performance Regression and Algorithm Selection*](https://doi.org/10.1145/3377930.3390183) | GECCO 2020; modular CMA-ES performance regression/selection | 56 inputs from distribution, level set, meta-model, dispersion, IC, NBC. A nine-feature model used `disp.diff_mean_02`, `ela_distr.skewness`, four meta-model outputs, `ic.eps_ratio`, `ic.eps_s`, `nbc.nb_fitness.cor`. | The clearest compact portfolio in the literature; it improved over using all features for most reported regression comparisons. |
| [Jankovic et al., *The Impact of Hyper-Parameter Tuning for Landscape-Aware Performance Regression and Algorithm Selection*](https://doi.org/10.1145/3449639.3459406) | GECCO 2021; tune regression models | Reuses broad ELA representations at different sample sizes; compares tree-based regressors | Downstream model and validation design materially affect conclusions about feature usefulness. |
| [Jankovic et al., *Towards Feature-Based Performance Regression Using Trajectory Data*](https://doi.org/10.1007/978-3-030-72699-7_38) | EvoApplications 2021; predict CMA-ES performance from its trajectory | 38 data-only outputs from `ela_distr`, `ela_meta`, `disp`, `nbc`, `ic`, plus CMA-ES state variables | Direct evidence for the 38-feature core and for accepting algorithm trajectories as an alternative sample source. |
| [Kostovska et al., *The Importance of Landscape Features for Performance Prediction of Modular CMA-ES Variants*](https://doi.org/10.1145/3512290.3528832) | GECCO 2022; RF regression plus SHAP | 46 features from distribution, meta-model, IC, NBC, dispersion, PCA | `ic.eps_max`, `ic.eps_ratio`, `ic.eps_s`, meta-model coefficient/condition outputs and later-budget PCA/`ic.m0` repeatedly rank highly. Importance changes with dimension and budget more than with a single module toggle. |
| [Nikolikj et al., *Identifying Minimal Set of ELA Features for Reliable Algorithm Performance Prediction*](https://doi.org/10.1109/CEC55065.2022.9870439) | IEEE CEC 2022; modular CMA-ES performance prediction | Cheap ELA followed by feature selection for six CMA-ES variants | The required number is algorithm-dependent, but selected portfolios overlap substantially. Supports individual selection without claiming one universal minimal vector. |
| [Prager et al., *Automated Algorithm Selection ... Deep Learning and Landscape Analysis Methods*](https://doi.org/10.1007/978-3-031-14714-2_1) | PPSN 2022; cost-sensitive BBOB algorithm selection | Classical ELA, six FDC, 16 dispersion, five IC, five NBC, ten miscellaneous features. After removing 14 mostly level-set outputs with missing values, 48 remained; greedy selection retained 15. | Evidence for FDC/PCA compatibility, and strong evidence that missing-value status must be explicit. |
| [Kostovska et al., *Per-Run Algorithm Selection with Warm-Starting Using Trajectory-Based Features*](https://doi.org/10.48550/arXiv.2204.09483) | PPSN 2022; online algorithm switching | 38 commonly used cheap ELA features from distribution, level/meta-model, dispersion, IC and NBC, combined with CMA-ES time-series features | ELA calculated from only `30d` trajectory observations is useful; combining ELA and state time series is better than either alone. The paper's reported family wording and count do not exactly match current `pflacco` family counts, so reproduce from its artifact rather than infer by count. |
| [Prager and Trautmann, *Exploratory Landscape Analysis for Mixed-Variable Problems*](https://doi.org/10.1109/TEVC.2024.3399560) | IEEE TEVC 2024; HPO algorithm selection | 38 outputs from `ela_distr`, `ela_meta`, `disp`, `ic`, `nbc` after target/one-hot encoding. Greedy selectors retained 12 or 14. Both started with peak count, then several meta-model outputs; dispersion dominated the remainder, with IC last and only one NBC output in one model. | Very strong confirmation of the 38-feature core beyond ordinary continuous BBOB, plus a future need for explicit encoders. |
| [Long et al., *Landscape-Aware Automated Algorithm Configuration Using Multi-Output Mixed Regression and Classification*](https://doi.org/10.1007/978-3-031-70068-2_6) | PPSN 2024; predict modular CMA-ES configuration | `pflacco` ELA from `50d` samples; min-max `y`; features with Pearson correlation above 0.95 removed | Reinforces the need for stable names and matrix-friendly output, but does not establish every input as essential. |
| [Seiler et al., *Learned Features vs. Classical ELA on Affine BBOB Functions*](https://doi.org/10.1007/978-3-031-70068-2_9) | PPSN 2024; algorithm selection on affine BBOB | Classical ELA versus Deep-ELA and TransOpt, alone and combined | Classical ELA is slightly superior alone. In hybrid selectors, the important classical additions are basic, dispersion, meta-model, and in 3D PCA features. |

### Learned representations: complements, not classical feature definitions

| Paper | Representation and training task | Evidence | Library implication |
|---|---|---|---|
| [Seiler et al., *Deep-ELA*](https://arxiv.org/abs/2401.01192) | Self-supervised transformer over `(X, Y)` point clouds with kNN embedding and mean pooling. Medium models emit 24 features for `d + m <= 6`; large models emit 48 for `d + m <= 12`. Models are trained for `25d` or `50d` samples and constrain outputs to `[-1, 1]`. | Tested on high-level property prediction and single-/multi-objective AAS. The pretrained vectors are less correlated and designed for shift/scale/rotation invariance, but require large pretrained artifacts and fixed dimensionality envelopes. | Define a generic `PointCloudRepresentation` extension with model ID/checksum, expected preprocessing, dimensionality limit, sample policy, device, and output schema. Do not expose `deep_ela.0` as if it had a fixed statistical meaning independent of a checkpoint. |
| [Cenikj et al., *TransOpt*](https://arxiv.org/abs/2311.18035) | Supervised transformer trained to classify the 24 BBOB functions. Encoder outputs are pooled by minimum, maximum, mean, and standard deviation. Best reported configuration used `50d`, embedding size 30, one head, and one layer. | Approximately 70--80% BBOB classification accuracy for dimensions 3 and 20; `y` is scaled to `[0,1]`, `X` remains in `[-5,5]`. | Useful learned baseline, but its representation is label/task-specific. Store training corpus/task and preprocessing in provenance; never describe it as an invariant drop-in replacement for classical ELA. |

### Multi-objective ELA extension track

[Preuß et al., *MO-ELA*](https://arxiv.org/abs/2602.00098) is the strongest supplied
evidence for a future native multi-objective surface. Starting from sampled pairs
`(X, Y)` with vector-valued `Y`, it proposes five groups:

- **Non-dominated sorting (19):** non-dominated count, maximum rank, average points
  per layer; hypervolume and Solow--Polasky summaries for the first five layers; and
  R2 values for degree-1 through degree-4 regressions over layer hypervolumes.
- **Non-dominated objective statistics (11 for two objectives, 12 for three):** per-objective
  minimum, maximum, mean, and standard deviation, plus objective correlations and a
  two-objective standard-deviation difference.
- **PCA:** summaries of explained variance on non-dominated decision points, objective
  vectors, and their concatenation. The current preprint is internally inconsistent here:
  Section 3.3 says 9 outputs (min/max/mean for each input), while Table 2 also lists standard
  deviation, which implies 12. A library specification must resolve this before coding.
- **Graphs:** 50 MST and 80 1-NN-graph outputs over decision space, objective space,
  and edge-transferred graphs. They summarize weights, closeness, angles, connected
  components, component sizes, and longest paths, plus cross-space ratios.
- **Gradient (4):** minimum, maximum, mean, and standard deviation of multi-objective
  absolute slopes along objective-space MST edges.

The AAS experiment combines these with earlier Liefooghe-style MO landscape features.
Greedy selection retains 32 of 226 candidates for bi-objective tasks and 36 of 233 for
tri-objective tasks. Among the new features, global NDS structure dominates the bi-objective
selection (7 NDS features), whereas local 1-NN graph structure dominates the tri-objective
selection (11 graph features); PCA and gradient each contribute one output in both cases.

This is promising evidence, but not a reason to enlarge the single-objective P0. Treat MO-ELA
as a separately versioned namespace with objective-vector scaling, dominance/tie rules,
hypervolume reference points, graph degeneracy policy, and a strict distinction between
decision-space and objective-space distances.

## Exact feature inventory recommended for the core

### P0: the recurring 38 data-only features

#### Objective-value distribution: 3

| Canonical name | Meaning | Direct evidence |
|---|---|---|
| `ela_distr.skewness` | skewness of sampled objective values | Selected in the 2019 and GECCO 2020 selectors; used in the 2021 extreme portfolio. |
| `ela_distr.kurtosis` | excess kurtosis of sampled objective values | Selected in the 2019 selector and the HPO-vs-BBOB decision tree. |
| `ela_distr.number_of_peaks` | KDE-based count of significant modes | Selected in the 2019 selector, HPO analysis, and first in both TEVC 2024 mixed-variable selectors. |

Specification warning: skewness and kurtosis have multiple finite-sample conventions.
Peak count depends on KDE bandwidth, grid, boundary extension, mode definition, and mode-mass
threshold. These choices must be versioned rather than inherited implicitly from SciPy or R.

#### Meta-model: 9

- `ela_meta.lin_simple.adj_r2`
- `ela_meta.lin_simple.intercept`
- `ela_meta.lin_simple.coef.min`
- `ela_meta.lin_simple.coef.max`
- `ela_meta.lin_simple.coef.max_by_min`
- `ela_meta.lin_w_interact.adj_r2`
- `ela_meta.quad_simple.adj_r2`
- `ela_meta.quad_simple.cond`
- `ela_meta.quad_w_interact.adj_r2`

Evidence is unusually strong at the individual level. The GECCO 2020 nine-feature portfolio
used adjusted linear fit, maximum coefficient, intercept, and adjusted quadratic fit. The
GECCO 2022 SHAP study repeatedly highlighted minimum coefficient, coefficient ratio,
quadratic condition and interaction fit. The TEVC 2024 selectors covered almost the entire
family.

Specification warnings: define the design matrices and interaction terms exactly; define
adjusted-R2 behavior when degrees of freedom are insufficient; define coefficient treatment
under constant columns, rank deficiency, zero minimum coefficient, and choice of solver.
R `lm`/QR and scikit-learn OLS are not numerically identical on small or degenerate designs.

#### Information content: 5

- `ic.h_max` — maximum information content/entropy.
- `ic.eps_s` — settling sensitivity.
- `ic.eps_max` — epsilon associated with maximum information content.
- `ic.eps_ratio` — partial information sensitivity (often called half-partial sensitivity).
- `ic.m0` — initial partial information content.

`eps_s` and `eps_ratio` recur in compact selectors; `eps_max`, `eps_ratio`, and `eps_s`
dominate several GECCO 2022 SHAP settings; `m0` becomes important at larger budgets.

Specification warnings: path construction, start point, nearest-neighbour tie-breaking,
distance normalization, epsilon grid, logarithm base, and missing-threshold behavior all need
to be explicit. A seed is part of provenance whenever path construction is stochastic.

#### Nearest-better clustering: 5

- `nbc.nn_nb.sd_ratio`
- `nbc.nn_nb.mean_ratio`
- `nbc.nn_nb.cor`
- `nbc.dist_ratio.coeff_var`
- `nbc.nb_fitness.cor`

The first three compare nearest-neighbour and nearest-better-neighbour distances; the fourth
summarizes the variability of distance ratios; the fifth correlates fitness with nearest-better
graph indegree. Four NBC outputs were selected by the best 2019 selector.
`nbc.nb_fitness.cor` recurs in GECCO 2020, the 2021 extreme portfolio, HPO analysis, and
TEVC 2024. `nbc.dist_ratio.coeff_var` and `nbc.nn_nb.mean_ratio` were used to separate
problem dimensionality in the HPO study.

Specification warnings: minimization/maximization orientation, ties in objective and distance,
the global-best point, duplicate `X`, constant arrays, correlation convention, distance metric,
and approximate-neighbour shortcuts must all have defined behavior.

#### Dispersion: 16

For each elite quantile `q` in `{02, 05, 10, 25}`, implement:

- `disp.ratio_mean_q`
- `disp.ratio_median_q`
- `disp.diff_mean_q`
- `disp.diff_median_q`

These compare pairwise decision-space distances within the best 2%, 5%, 10%, and 25% of
the sample to those in the complete sample. `disp.diff_mean_02` is in the GECCO 2020
nine-feature portfolio. Ratio features drive the HPO dimensionality tree, and multiple
difference features dominate both TEVC 2024 mixed-variable selectors.

Specification warnings: state whether diagonal zero distances are excluded, how duplicate
points are treated, how quantile ties are handled, how elite counts are rounded, whether the
median is taken from condensed or square-form distances, and what happens when an elite set
has fewer than two distinct observations. Computation is zero-evaluation but `O(n^2)` in time
and potentially memory.

### P1: inexpensive and useful adjuncts

#### Fitness-distance correlation: 6

- `fitness_distance.fd_correlation`
- `fitness_distance.fd_cov`
- `fitness_distance.distance_mean`
- `fitness_distance.distance_std`
- `fitness_distance.fitness_mean`
- `fitness_distance.fitness_std`

FDC is interpretable and used in the PPSN 2022 classical-ELA baseline. It is also a useful
sanity check on known analytical landscapes. Its evidence frequency is lower than the P0
families, so it should follow rather than delay them. Specify whether distance is measured to
a known global optimum, the best sampled point, or the nearest of several optima, and whether
the statistics apply to all observations or an elite fraction.

#### Principal components: 8

- `pca.expl_var.cov_x`
- `pca.expl_var.cor_x`
- `pca.expl_var.cov_init`
- `pca.expl_var.cor_init`
- `pca.expl_var_PC1.cov_x`
- `pca.expl_var_PC1.cor_x`
- `pca.expl_var_PC1.cov_init`
- `pca.expl_var_PC1.cor_init`

Here `x` means decision variables only and `init` includes objective values. The first four
report the fraction of components required to pass an explained-variance threshold; the last
four report PC1's explained fraction. PCA appears in modern broad baselines, some minimal
BBOB portfolios, high-dimensional analysis, later-budget SHAP rankings, and the 3D hybrid
selector of PPSN 2024.

Specify centering/scaling, covariance versus correlation, eigenvalue ordering, zero-variance
columns, sign/rotation indeterminacy, threshold comparison, and rank-deficient behavior.

#### Basic summaries

Implement sample size, dimension, objective minimum/maximum/mean/standard deviation and
bound/range summaries as cheap controls, but do not confuse benchmark-identifying absolute
values with invariant landscape structure. The best sampled fitness was selected in the 2019
study; basic features also contributed unique information to the PPSN 2024 hybrid selector.
Absolute objective summaries should be opt-in for invariant ML profiles.

### P2: level-set compatibility

For quantiles `{10, 25, 50}`, the full R-compatible scientific surface is:

- `ela_level.mmce_lda_q`, `ela_level.mmce_qda_q`, `ela_level.mmce_mda_q`
- `ela_level.lda_qda_q`, `ela_level.lda_mda_q`, `ela_level.qda_mda_q`

Naming differs across old papers and current packages. Early papers use forms such as
`levelset.lda_mmce_10`; current Python APIs use `ela_level.mmce_lda_10`. Preserve aliases
in a compatibility layer, not in the mathematical specification.

Level-set features have genuine use: the 2019 selector retained an LDA/MDA error ratio and
many 2020--2025 baselines calculate the family. They are nevertheless a second-wave target:

- cross-validated LDA/QDA/MDA fits are CPU-heavy;
- small or imbalanced quantile classes fail;
- covariance singularities are common in high dimension;
- modern `pflacco` does not implement MDA, so its native surface is not R-compatible;
- PPSN 2022 removed 14 mostly level-set outputs because of missing values;
- GECCO 2021 found level-set computation impractical at large dimensions without reduction.

### P3: defer from the default profile

| Family | Reason to defer |
|---|---|
| `ela_conv`, `ela_curv`, `ela_local` | Require additional objective evaluations and are largely absent from recent practical ML portfolios. Keep their costs explicit if implemented. |
| `cm_angle`, `cm_conv`, `cm_grad`, `gcm`, barrier trees | Grid cell count grows exponentially with dimension; one cell-angle output helped the 2019 model, but later work rarely selects these families. |
| `limo` | Computationally affordable and included in GECCO 2021 high-dimensional work, but much less recurring evidence than ordinary meta-model features. |
| local optima networks, length scale, gradient walks, Sobol/sensitivity families | Valuable fitness-landscape methods, but not part of the recurring classical ELA baseline. Add through separate modules and evidence profiles. |

## Cross-paper synthesis

### What is robustly essential

1. **Meta-models are the most defensible first family after simple distribution moments.**
   They recur in virtually every task and individual coefficients/fits survive selection.
2. **IC and NBC provide information that distribution and regression fits do not.** Their
   individual sensitivity, correlation, and graph statistics recur across compact portfolios.
3. **Dispersion is essential as a family, even though its 16 members are redundant in some
   datasets.** Different elite quantiles and mean/median variants are selected in different
   domains. Individual selection makes supporting all 16 reasonable.
4. **Peak count is essential despite being awkward to specify.** Dropping it for ease of
   implementation would remove one of the most consistently selected distribution features.
5. **There is no universal minimal subset smaller than the 38-feature core.** Nine features
   work well in one GECCO 2020 setting; one to four can classify in-distribution BBOB with
   large samples; mixed-variable selectors need 12--14; SHAP importance changes with budget
   and dimension. The library should expose individual features, not hard-code one vector.

### What the feature counts do not prove

- A paper that feeds all 46, 56, 62, or 102 outputs to a random forest does not establish that
  every output is useful.
- High BBOB function-ID accuracy is not the same as accurate algorithm selection on unseen
  function families.
- Repeated instances of the same 24 functions are not independent evidence of
  out-of-distribution generalization.
- Feature selection results depend on objective scaling, sampling design, sample size,
  dimension, algorithm portfolio, budget, model class, and validation split.

## Requirements implied for `orivex`

### 1. Sampling and transforms are part of the feature definition

The API should never return an unqualified feature vector. At minimum, provenance should
record:

- sampling method (`random`, LHS variant, Sobol, or trajectory);
- sample size and its relationship to dimension;
- seed, repetitions, and aggregation across repetitions;
- decision-space normalization and original bounds;
- objective transformation (`raw`, standardized, min-max, log, rank, or user-defined);
- optimization direction and tie policy.

Do not silently min-max-scale `y`. Offer preprocessing as an explicit versioned transform and
allow raw and normalized profiles to coexist. The literature shows that normalization often
helps transfer but not uniformly for every feature family.

### 2. Individual features, shared intermediates

The dependency planner should share, without forcing group-level output:

- objective moments and KDE evaluations for `ela_distr`;
- linear, interaction, and quadratic design matrices for `ela_meta`;
- ranks, nearest-neighbour structures, and path order for IC/NBC;
- pairwise distance reductions for dispersion;
- centered/scaled covariance matrices and eigendecompositions for PCA.

Users must still be able to request one feature, such as `ic.eps_ratio` or
`disp.diff_mean_02`, without receiving the historical family.

### 3. Scientific values are separate from execution metadata

Legacy family counts often include `costs_runtime` and `costs_fun_evals`. Return these as
result metadata rather than ordinary predictors. This avoids accidental leakage of hardware,
implementation, and evaluation-policy information into ML models.

### 4. Failure is data, not `NaN` without explanation

Return a status and diagnostic for at least:

- too few observations or degrees of freedom;
- constant/near-constant objective values;
- singular or ill-conditioned regression/discriminant covariance;
- empty or one-point elite subsets;
- duplicate decision points and distance ties;
- zero denominators and undefined correlations;
- failed KDE or no significant peak;
- memory/budget refusal before an `O(n^2)` allocation.

This is directly motivated by papers dropping entire families after missing values appeared.

### 5. Profiles should be evidence-based conveniences, not semantic bundles

Suggested profiles:

```text
core38 = ela_distr[3] + ela_meta[9] + ic[5] + nbc[5] + disp[16]
core_interpretable = ela_distr + ela_meta + ic + nbc + fdc
classic_data_only = core38 + ela_level + pca + basic
legacy_flacco = compatibility surface, including expensive families when objective access is supplied
```

Profiles should expand to versioned individual feature identifiers. They should not be the
unit of calculation or caching.

### 6. Validation targets

For each feature, use:

- analytical landscapes for known values or qualitative ordering;
- differential fixtures against R `flacco`, with documented intentional deviations;
- metamorphic tests for decision translation/scaling/rotation and objective shift/scale where
  invariance is mathematically expected;
- degeneracy fixtures for ties, constants, duplicates, and insufficient rank;
- sampling-repeat tests that distinguish numerical determinism from statistical stability;
- cost tests that assert zero additional evaluations for the core and prevent accidental
  quadratic memory allocations.

## Proposed change to the existing roadmap

The current roadmap is well aligned with the literature. A sharper order is:

1. Finish distribution semantics and add KDE peak count.
2. Implement all nine meta-model features, beginning with the four members in the GECCO 2020
   compact portfolio.
3. Implement the five IC and five NBC outputs.
4. Implement all sixteen dispersion outputs with a blockwise distance path.
5. Add the six FDC and eight PCA outputs plus basic summaries.
6. Implement level-set LDA/QDA only after specifying how MDA compatibility and singular
   fits are represented.
7. Re-evaluate `limo` and expensive/cell-based families only after usage telemetry or a
   concrete compatibility requirement justifies them.
8. Keep learned point-cloud representations and MO-ELA behind separate interfaces. For
   MO-ELA, first stabilize the preprint's PCA count and degeneracy conventions; for learned
   representations, version model weights and preprocessing together.

## Bottom line

For an initial release that researchers can realistically use to reproduce modern ELA-based
ML work, **the essential target is the 38-feature distribution/meta-model/IC/NBC/dispersion
core, plus explicit sampling, transform, status, and cost metadata**. FDC, PCA, and basic
summaries are the next useful layer. Level set is important for compatibility but should not
block the core. The remaining historical `flacco` catalogue is not justified as a default
implementation target by current application evidence.

The most important architectural conclusion is that feature usefulness is conditional. A
correct replacement should make small, task-specific portfolios cheap and reproducible rather
than optimize for calculating every historical feature at once.
