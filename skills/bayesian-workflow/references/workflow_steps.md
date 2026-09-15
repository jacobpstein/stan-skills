# The Workflow Diagram, Box by Box

*Bayesian Workflow* Figure 2.1 (p. 18). Each box lists the book section, what to do, and where
in these skills the mechanics live. Yellow outcomes ("provisionally accepted") loop back to
"Choose your focus"; pink outcomes ("problematic") flow into diagnosis or modification.

## Pick an initial model (§5.1)

Adapt a template. Start linear/normal (or negative binomial for counts, logistic for binary).
Enumerate assumptions. Write the data model in generative form.
→ [stan-modeling](../../stan-modeling/SKILL.md), [applied-regression](../../applied-regression/SKILL.md)

## Choose your focus

You can enter the loop at model structure, at fitting, at comparison, or at use. A new dataset
for an existing model enters at "Fit to real data"; a new question for a fitted model enters at
"Using the model".

## Model structure

| Sub-box | Do | Skill |
|---|---|---|
| Prior predictive check (§5.9) | simulate ~10 datasets from the prior; plot against data | stan-priors |
| Use simulations (ch. 6) | fake-data recovery; simulate first, summarize last | bayesian-workflow/references/fake_data.md |
| Relationship to scientific goals (ch. 10) | is the quantity of interest a function of the model? can the design inform it? | bayesian-workflow |

Outcomes: *Model is provisionally accepted* or *Model contradicts domain knowledge* → Modify.

## Correctness of computation

| Sub-box | Do | Skill |
|---|---|---|
| Small-scale simulation tests (§6.3) | one or a few fake datasets, check recovery | stan-testing |
| Simulation-based calibration checking (ch. 14) | S prior draws → S fits → rank uniformity | stan-testing |
| Modeling as software development (ch. 15) | git, scripts, tests, modular Stan | stan-testing |

Outcomes: *Computation is not valid* → Diagnose; *Computation is provisionally accepted*.

## Fit to real data (ch. 11)

Fit and diagnose convergence: R-hat < 1.01, ESS > 100/chain, no divergences, no treedepth
saturation, E-BFMI > 0.3, MCSE small enough for the digits reported.
→ [stan-computation](../../stan-computation/SKILL.md)

Outcomes: *Model converges* → Visualize and check; *Computation is not valid* → Diagnose.

## Diagnose and address computational issues (ch. 12)

| Sub-box | When |
|---|---|
| Simplify the model | always first: meet in the middle |
| Implement model components separately | multi-part models (ODE + regression) |
| Run for a small number of iterations | while iterating; 200 is enough to see most failures |
| Run on a subset of data | slow models |
| Plot intermediate quantities | numerical problems, stuck chains |
| Add prior information | non-identification, funnels, weak likelihood |
| Reparameterize | funnel, correlations, aliasing |
| Check for multimodality | tail ESS ≫ bulk ESS, bimodal histograms |
| Stack individual chains | exploration with stuck chains |
| Add more data | small-N funnels that no prior fixes |

Outcomes: *Computation improved* → back to the top; *Give up* → Modify the model (use an
approximation, a simpler component, or a different starting model).

## Visualize and check a fitted model (ch. 8)

| Sub-box | Do | Skill |
|---|---|---|
| Posterior predictive check | y_rep vs y; targeted statistics; grouped checks | stan-model-evaluation |
| Cross validation | PSIS-LOO, Pareto k, LOO-PIT | stan-model-evaluation |
| Influence of single data points | pointwise elpd, k values, LOO-PIT extremes | stan-model-evaluation |
| Influence of prior | power-scaling sensitivity | stan-priors |

Outcomes: *Fit to data is provisionally accepted*; *Fit to data is problematic* → Modify.

## Modify the model

| Sub-box | Typical trigger |
|---|---|
| Pick a new starting model | the template was wrong for the question |
| Replace model component | PPC failure localized to one component (tails, link, dispersion) |
| Enrich/expand the model (§9.7) | grouped residual structure, new data, new question |
| Modify priors (§5.6) | prior predictive absurdities, prior sensitivity, computation |
| Add more data | weakly informed components |
| Use an approximation (ch. 13) | computation too expensive for the exploration phase |

## Comparing and combining models (ch. 9)

Comparing inferences (plot QoI vs model index), multiverse analysis, model averaging/stacking.
→ [stan-model-evaluation/references/model_comparison.md](../../stan-model-evaluation/references/model_comparison.md)

## Using the model (ch. 7)

Postprocessing, prediction, poststratification, causal inference: all computations over
posterior draws, usually in `generated quantities` or with `posterior_epred`/`posterior_predict`.
→ [applied-regression](../../applied-regression/SKILL.md)

## Why the loop is tangled (§2.2)

1. The model you want is more complex than you can fit or than the data support; you start with
   a model you know is missing features.
2. Data are not fixed; new data force extensions.
3. Fitting is itself hard and must be checked.
4. Models are understood by comparison with alternatives fit to the same data.

Shortcuts are legitimate when time, compute, or the cost of error demand them; say which ones you
took.
