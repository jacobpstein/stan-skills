# Posterior Predictive Checking in Detail

Source: *Bayesian Workflow* §8.1–8.3, §6.1; *Regression and Other Stories* §11.4.

## What a PPC is

Replicated data `y_rep ~ p(y_rep | y) = ∫ p(y_rep | θ) p(θ | y) dθ`, usually the same size and
design as `y`. Think of `(θ, y, y_rep)` as a joint distribution `p(θ) p(y|θ) p(y_rep|θ)`;
a PPC asks whether `y` looks like a typical draw alongside the `y_rep`s.

Implementation: simulate `y_rep` in `generated quantities` from the **data model** (the
generative story), one replicate per posterior draw. For a normal regression:

```stan
generated quantities {
  array[N] real y_rep = normal_rng(mu, sigma);   // vectorized rng returns array
}
```

For a zero-inflated Poisson you coded via `log_mix` in the model block, the rng must follow
the generative story, not the marginalized likelihood:

```stan
generated quantities {
  array[N] int y_rep;
  for (n in 1:N)
    y_rep[n] = bernoulli_rng(theta) ? 0 : poisson_log_rng(eta[n]);
}
```

## Choosing test summaries

- Start with the **whole distribution** (density/ECDF overlay). It examines many aspects at
  once and catches gross misspecification (normal fit to lognormal data shows tail mismatch).
- Then pick summaries the model does **not** fit directly and that would matter for the
  question: `sd` (overdispersion), `max`/`min` (tails), proportion of zeros, quantiles,
  autocorrelation of residuals, a between-group sd.
- Prefer **ancillary** summaries whose distribution is independent of the fitted parameters.
  A binomial model fit to beta-binomial data reproduces the mean but `sd(y_rep)` is
  systematically below `sd(y)`.
- Compute summaries by **subgroup**, especially groups defined by a variable **omitted** from
  the model. Extra between-group variability in the data relative to `y_rep` is a direct
  pointer to the next expansion.
- Rarely compute a posterior predictive p-value; when you do, remember it is conservative
  because the data are used twice. Graphical checks make the same comparison and show *how*
  the model misses.

## Discrete and binary outcomes

- Counts with many distinct values and smooth probabilities: KDE overlays are acceptable.
- Counts with few distinct values or sharp probability changes: use a **rootogram**
  (`ppc_rootogram`, `az.plot_ppc_rootogram`), which compares expected and observed frequencies
  of each count with a square-root or log y-axis. Hanging rootograms make deficits at zero
  (missing zero inflation) obvious.
- Binary outcomes: KDE overlays make no sense and proportion bar plots cannot fail (any model
  with an intercept matches the marginal rate). Use a **PAV-adjusted calibration plot**
  (pool-adjacent-violators isotonic regression of `y` on `p_hat`, with consistency bands from
  `y_rep`): `az.plot_ppc_pava` in Python, the `reliabilitydiag` package in R. It needs no bin-width
  choice and the monotonicity assumption tightens the band.
- Categorical outcomes: one-vs-rest probabilities; ordinal: cumulative probabilities.

## PIT-based checks

`PIT_i = Pr(y_rep_i ≤ y_i | y)`. For discrete data use a randomized PIT. If the pointwise
predictive distributions are calibrated, PITs are uniform. Plot the **ECDF difference**
(ECDF minus the uniform CDF) with simultaneous confidence bands rather than a histogram.

Interpretation of the ECDF-difference plot:
- Hump (ECDF above diagonal in the middle, PITs concentrated near 0.5): predictions too wide
  (overdispersed), **or** double use of data in a flexible model.
- U shape (PITs piled at 0 and 1): predictions too narrow.
- Monotone drift: systematic bias.

### Posterior PIT vs LOO-PIT

Posterior PIT conditions on `y_i` when predicting `y_i`. For models where the effective
number of parameters is large relative to n (varying intercepts with one or two observations
per group, splines, GPs), the posterior for unit i is pulled toward `y_i`, so posterior PIT
concentrates near 0.5 even for a calibrated model, and can also hide true miscalibration.

Rule: **when the model is flexible, use LOO-PIT** (`ppc_loo_pit_ecdf`, `az.plot_loo_pit`).
LOO-PIT values are dependent, so use a uniformity test that handles dependence (the POT test
ArviZ prints in the plot corner). A clean posterior PIT plot alone should not be read as
evidence of calibration until LOO agrees. With two parameters and hundreds of observations the
two agree and posterior PIT suffices.

## Hierarchical replication scenarios

With `p(φ) p(α | φ) p(y | α, φ)` (φ = population-level, α = group-level) there are three
replication distributions, each answering a different question:

1. **Existing groups, new observations**: keep `φ^s, α^s`, draw `y_rep ~ p(y | α^s, φ^s)`.
   Checks within-group fit. Uses the data twice at the group level.
2. **New groups from the fitted population**: keep `φ^s`, draw `α_rep ~ p(α | φ^s)`, then
   `y_rep`. Checks whether the population model for groups is adequate. This is the
   distribution you need when the goal is prediction for a new hospital, school, or year.
3. **New population**: draw `φ_rep` from the prior, then `α_rep`, then `y_rep`. This is a
   prior predictive check and requires proper hyperpriors.

Code both 1 and 2 in `generated quantities` when the model will be used for new groups:

```stan
generated quantities {
  vector[N] y_rep_within;
  vector[N] y_rep_newgroup;
  {
    vector[J] alpha_new;
    for (j in 1:J) alpha_new[j] = normal_rng(mu_alpha, sigma_alpha);
    for (n in 1:N) {
      y_rep_within[n]   = normal_rng(alpha[group[n]] + x[n] * beta, sigma);
      y_rep_newgroup[n] = normal_rng(alpha_new[group[n]] + x[n] * beta, sigma);
    }
  }
}
```

## Exploring inferences to find problems

Any simulation-based forecast can be treated as an implicit posterior and checked the same way
(BW §8.1, FiveThirtyEight 2020): compute marginal probabilities, then **condition on tail
events** and recompute; scatter pairs of quantities to see tail dependence that a correlation
matrix hides. Implausible conditional statements (losing New Jersey *raising* the Alaska win
probability) point to structural errors (independent long-tailed state errors) that no
marginal summary reveals.

## When a check fails

There is no universal rule for when a failure requires a model change; it depends on the
purpose and cost. Ask: would this discrepancy change the answer to the question I care about?
If yes, the failure names the next expansion (heavier tails, overdispersion, a missing
grouping, zero inflation, a nonlinearity). Keep the failed model in the report as the
comparison point that motivated the change.

## Recipes

### bayesplot (any interface producing a draws matrix)

```r
yrep <- posterior::as_draws_matrix(fit$draws("y_rep"))   # cmdstanr
yrep <- posterior_predict(fit_rstanarm)                   # rstanarm / brms
ppc_dens_overlay(y, yrep[sample(nrow(yrep), 50), ])
ppc_ecdf_overlay(y, yrep[1:50, ])
ppc_stat(y, yrep, stat = "sd"); ppc_stat_2d(y, yrep, stat = c("mean", "sd"))
ppc_stat_grouped(y, yrep, group = site, stat = "mean")
ppc_intervals(y, yrep, x = x1); ppc_ribbon(y, yrep, x = time)
ppc_rootogram(y, yrep, style = "hanging")
# PAV-adjusted reliability diagram: see the reliabilitydiag recipe in loo_cv.md
ppc_pit_ecdf(y, yrep, prob = 0.99, plot_diff = TRUE)
ppc_loo_pit_ecdf(y, yrep, lw = weights(loo1$psis_object))
ppc_error_scatter_avg_vs_x(y, yrep, x = x1)
```

### ArviZ 1.x

```python
az.plot_ppc_dist(dt, kind="ecdf", num_samples=50)
az.plot_ppc_tstat(dt, t_stat="sd")
az.plot_ppc_tstat(dt, t_stat=lambda x: (x == 0).mean(axis=-1))  # proportion of zeros
az.plot_ppc_rootogram(dt, yscale="sqrt")
az.plot_ppc_pava(dt, data_type="binary")
az.plot_ppc_pit(dt)                # posterior PIT-ECDF difference
az.plot_loo_pit(dt)                # LOO-PIT-ECDF, POT test in corner
az.plot_ppc_interval(dt)           # observed vs predictive intervals per observation
az.plot_ppc_censored(dt)           # censored outcomes
```

Group-wise checks in ArviZ: pass `coords={"obs": idx_for_group}` or reshape `y_rep` with a
`group` dim in `dims=` when calling `az.from_cmdstanpy`.
