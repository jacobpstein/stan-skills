---
name: applied-regression
description: >
  Load when the user is doing applied regression and the question is about how to build,
  transform, interpret, or report the model rather than about Stan syntax: interpreting
  coefficients, centering and standardizing, log and other transformations, interactions,
  the divide-by-4 rule, average predictive comparisons, R2 and LOO R2, logistic and count
  GLMs with offsets and overdispersion, separation, binned residuals, design and sample-size
  analysis by simulation, poststratification and MRP, and causal-regression pitfalls such as
  adjusting for post-treatment variables. Implements Regression and Other Stories (Gelman, Hill,
  Vehtari) in Stan/rstanarm/brms. Triggers include: interpret coefficients, comparisons not
  effects, divide by 4, divide by 2 sd, standardize, center predictors, log transformation,
  elasticity, interaction, overdispersion, offset, exposure, negative binomial, ordered logistic,
  separation, binned residuals, Bayesian R2, power analysis, sample size, type S error,
  type M error, poststratification, MRP, post-treatment variable, confounding.
---

# Applied Regression in Stan

Follows *Regression and Other Stories* (Gelman, Hill, Vehtari, 2020). The book's models are
`stan_glm` calls; everything here translates to brms and to hand-written Stan. For the program
itself see [stan-modeling](../stan-modeling/SKILL.md); for the R interfaces see
[r-stan-interfaces](../r-stan-interfaces/SKILL.md); for the iterative process see
[bayesian-workflow](../bayesian-workflow/SKILL.md).

## The ten quick tips (ROS Appendix B)

1. **Think about variation and replication.** Coefficients vary across datasets. Fitting the same
   model to many datasets ("the secret weapon") tells you more than a standard error from one.
2. **Forget about statistical significance.** Do not discretize by whether an interval excludes
   zero. There are almost no true zeros, the threshold throws away information, and effects vary
   by context.
3. **Graph the relevant, not the irrelevant.** Plot the fitted model over the data, make many
   graphs, and skip automatic diagnostic output (Q-Q plots, influence diagrams) you cannot
   explain.
4. **Interpret coefficients as comparisons, not effects.** See below.
5. **Understand methods through fake-data simulation.** It clarifies assumptions, reveals a
   method's repeated-sampling behavior, debugs code, and is required for design analysis.
6. **Fit many models**, from too simple to too complex, and keep track of what you fit.
7. **Set up a computational workflow** so fits are fast enough to iterate: subset the data, use
   fake data when the fit is stuck, use predictive simulation to locate misfit.
8. **Use transformations**: logs for positive variables, standardizing for comparability,
   interactions and combined predictors.
9. **Do causal inference in a targeted way**, not as a byproduct of one big regression.
10. **Learn methods through live examples** and know the magnitude, not just the sign, of your
    coefficients.

## Interpret coefficients as comparisons

A coefficient is a **between-unit comparison**, not a within-unit effect. For
`earnk = -26.0 + 0.6 * height + 10.6 * male`, the correct statement is: among people of the same
sex who differ by one inch in height, earnings differ by about $600 on average. "The effect of
height is $600" is wrong unless the causal assumptions of ch. 18-21 hold.

The comparison framing is always valid as a description of the model, needs no causal
assumptions, and is the foundation on which causal interpretation is built (with ignorable
assignment and no interaction, the average causal effect equals the treatment coefficient; in
general you work through model predictions).

## Assumptions, in decreasing order of importance (ROS §11.1)

1. **Validity**: the data answer the research question; the outcome measures the phenomenon; the
   model includes the relevant predictors; the results generalize to where you will apply them.
2. **Representativeness**: the sample represents the population of `y` given `X`. Selection on
   `X` is harmless; selection on `y` is not. This is a reason to include more predictors.
3. **Additivity and linearity**: the most important mathematical assumption. Fix with
   transformations, interactions, nonlinear terms, or splines.
4. **Independence of errors**: violated in time series, spatial, and multilevel data.
5. **Equal variance of errors**: matters for predictive intervals, not for the average
   relationship.
6. **Normality of errors**: least important. Do not routinely make Q-Q plots of residuals.

Predictors need not be normally distributed; the marginal distribution of `y` need not be
either.

## Transformations

| Transformation | When | Interpretation |
|---|---|---|
| Center: `x - mean(x)` | always, especially with interactions | intercept becomes the prediction at the mean |
| Standardize by **2 sd**: `(x - mean(x)) / (2 * sd(x))` | comparing continuous and binary predictors | coefficient is comparable to a 0/1 indicator's coefficient |
| `log(y)` | positive outcome, multiplicative structure | coefficient ≈ proportional change when \|β\| < 0.1; otherwise `exp(β)` |
| `log(x)` and `log(y)` | elasticity | % change in y per % change in x |
| `sqrt(y)` | counts, milder than log | compresses less; harder to interpret |
| Bin a continuous predictor | parametric form implausible | flexible, loses within-bin trend |
| Sum related items into a score | many correlated predictors | more stable estimation |

Why 2 sd: a binary predictor with p ≈ 0.5 has sd ≈ 0.5, so dividing a continuous predictor by
2 sd puts it on the same footing as a 0-to-1 comparison.

With interactions, a main-effect coefficient is the comparison **at the other predictor equal to
zero**. If zero is outside the data (IQ = 0, height = 0), center first or the coefficient is
meaningless.

Log-scale coefficients: 0.06 means about 6% higher; 1.0 means `exp(1) = 2.7` times, not double.
Above about 0.1 in magnitude, exponentiate rather than reading the percentage off.

Comparing a log-outcome model to a raw-outcome model with LOO requires a **Jacobian correction**:
subtract `log(y_i)` from each pointwise elpd of the log model before `loo_compare`.

## Logistic regression

- **Divide by 4**: the maximum change in `Pr(y = 1)` per unit of x is `β / 4`, a good
  approximation near p = 0.5. A coefficient of 0.33 on a 1-5 income scale means at most about 8
  percentage points per step.
- Compare predictors on different scales by multiplying each coefficient by that predictor's sd
  before dividing by 4.
- **Average predictive comparison** is the honest general summary: average the predicted
  difference over the empirical distribution of the other predictors, not at their mean.

```r
b <- coef(fit)
delta <- invlogit(b[1] + b[2]*1 + b[3]*d$arsenic + b[4]*d$educ4) -
         invlogit(b[1] + b[2]*0 + b[3]*d$arsenic + b[4]*d$educ4)
mean(delta)     # e.g. -0.21: 100m farther, 21 points less likely to switch
```

  Evaluating at the mean of the other predictors can badly over- or under-state the comparison
  when those predictors are spread out or bimodal.
- **Binned residual plots**, not raw residual plots: bin by fitted probability or by a predictor,
  plot mean residual per bin with `2 * sqrt(p_j (1 - p_j) / n_j)` bounds. Roughly 95% should fall
  inside.
- **Error rate is a poor summary**: it treats p = 0.6 and p = 0.99 identically and can show zero
  improvement over a trivial model in imbalanced problems. Use the log score.
- **Separation**: if a predictor or a linear combination perfectly predicts the outcome, the MLE
  is infinite (`glm()` returns a large finite number, which is an artifact). The fix is a proper
  prior. Never fit logistic regressions with flat priors.

## Other GLMs

| Outcome | Model | Notes |
|---|---|---|
| Counts, no upper limit | negative binomial (`neg_binomial_2_log`) | start here, not Poisson; Poisson's sd is fixed at `sqrt(mean)` |
| Counts out of known trials | binomial/logistic | overdispersion: `z_i = (y_i - n_i p̂_i)/sqrt(n_i p̂_i (1-p̂_i))`, estimate ≈ `sum(z^2)/(N-k)` |
| Counts with exposure | offset `log(exposure)`, coefficient fixed at 1 | `offset = log(exposure)` in rstanarm; `offset()` in brms; add to the intercept term in Stan |
| Ordered categories | ordered logistic (`stan_polr`, `cumulative()`) | cutpoints strictly increasing; probit ≈ logit / 1.6 |
| Unordered categories | separate nested logistic models, or categorical | do not force an ordering |
| Outlier-prone continuous | Student-t | `family = student()`; fix `nu = 4` if it will not identify |
| Outlier-prone binary | robit (t latent errors, ν = 4) | ν → ∞ gives probit, ν ≈ 7 gives logit |

Always check the zero count for count models:

```r
yrep <- posterior_predict(fit)
mean(apply(yrep, 1, function(r) mean(r == 0)))   # compare to mean(y == 0)
```

In the roaches example, Poisson replicates produced 0-0.8% zeros against 36% observed.

## Fit, check, compare

```r
fit <- stan_glm(y ~ x1 + x2, data = d)
print(fit, digits = 2)                # median and MAD-sd, the book's default summary
sims <- as.matrix(fit)

# any function of parameters: compute on the draws, never on the point estimates
z <- sims[, "x1"] / sims[, "x2"]; c(median(z), mad(z))

# three levels of prediction
predict(fit, newdata = new)                  # point
posterior_linpred(fit, newdata = new)        # + coefficient uncertainty
posterior_epred(fit, newdata = new)          # expectation (= linpred for linear models)
posterior_predict(fit, newdata = new)        # + residual noise

# residuals: always against fitted, never against observed
plot(predict(fit), y - predict(fit))

# replication check
ppc_dens_overlay(y, posterior_predict(fit)[1:50, ])
ppc_stat(y, posterior_predict(fit), stat = "min")

median(bayes_R2(fit)); median(loo_R2(fit))
loo_compare(loo(fit1), loo(fit2))
```

Bayesian R² is `V(fitted_s) / (V(fitted_s) + sigma_s^2)` per draw, which stays in [0, 1] even
with strong priors; the plug-in version can exceed 1.

Adding pure-noise predictors always raises within-sample R² and log score but lowers LOO: roughly
+0.5 raw and −0.5 LOO per noise predictor. That gap is the signature of overfitting.

## Priors (rstanarm defaults, ROS §9.5)

For `y = a + b1 x1 + ... + error`:

- each coefficient: `normal(0, 2.5 * sd(y) / sd(x_k))`
- intercept (on centered predictors, i.e. `E[y | x = mean(x)]`): `normal(mean(y), 2.5 * sd(y))`
- residual scale: `exponential(1 / sd(y))`
- logistic regression: `normal(0, 2.5 / sd(x_k))`, no `sd(y)` factor

The 2.5 comes from standardized effects rarely exceeding 1 in magnitude. Check with
`prior_summary(fit)`. With many predictors these independent priors imply an R² prior near 1;
see [stan-priors](../stan-priors/SKILL.md) for the fixes (scale by `sqrt(R2_0 / p)`, or use a
regularized horseshoe or R2D2 prior).

Combining a prior estimate with a data estimate (ROS §9.3):

```
theta_bayes = (theta_prior/se_prior^2 + theta_data/se_data^2) / (1/se_prior^2 + 1/se_data^2)
se_bayes    = sqrt(1 / (1/se_prior^2 + 1/se_data^2))
```

## Design and sample size by simulation

Power analysis is fake-data simulation. Simulate the study under an assumed effect, fit the
intended analysis, and look at the distribution of estimates and standard errors.

Key results:
- A low-power study is not a small chance of success. Conditional on "significance", the estimate
  has a substantial chance of the wrong sign (**Type S error**) and is guaranteed to overstate
  the magnitude (**Type M error**), often by many multiples.
- 80% power needs the true value about **2.8 standard errors** from the comparison point:
  `n = p(1-p) (2.8/(p - p0))^2` for a proportion, `n = (5.6 sigma / delta)^2` total for a
  two-group mean difference.
- **Interactions need 4× the sample size** of a main effect of the same size, and **16×** if the
  interaction is half the size. Design for the main effect, put the large comparison in the main
  effect, and use partial pooling rather than demanding significance for interactions.
- Pre-treatment covariates that genuinely correlate with the outcome reduce the residual sd and
  hence the required n.

See [references/design_analysis.md](references/design_analysis.md) for the simulation recipes.

## Causal regression cautions

- **Do not adjust for post-treatment variables.** Conditioning on a mediator compares units with
  different unobserved potential outcomes for that mediator, and can flip the sign of the
  treatment coefficient even under randomization. Mediation via "add the mediator as a predictor"
  is not valid; principal stratification or instrumental variables are the honest routes.
- Randomization licenses causal claims about the **randomized variable only**, not about other
  predictors in the same regression.
- Rarely can more than one coefficient in a multi-predictor observational regression be given a
  defensible causal reading.
- Choose the treatment variable deliberately, check balance and overlap, and adjust for
  pre-treatment differences.

## Poststratification and MRP

When the sample is unrepresentative but you know the population composition:

1. Fit a regression on all the adjustment variables.
2. Build a table with one row per population cell and its known count `N_j`.
3. Predict cell expectations with `posterior_epred` and take the `N`-weighted average **per
   posterior draw**.

```r
epred <- posterior_epred(fit, newdata = poststrat_table)      # n_sims x J
est   <- epred %*% poststrat_table$N / sum(poststrat_table$N)
c(mean(est), mad(est))
est2 <- est + rnorm(length(est), 0, 0.02)     # add known unmodeled uncertainty
```

With many cells, fit the regression as a **multilevel** model so sparse cells are partially
pooled: multilevel regression and poststratification (MRP). See
[references/poststratification.md](references/poststratification.md).

## References

- [references/transformations.md](references/transformations.md): centering, scaling, logs,
  interactions, worked interpretation patterns, the Jacobian correction.
- [references/glms.md](references/glms.md): logistic, Poisson/NB, binomial, ordinal, robust, with
  Stan and rstanarm/brms code and the diagnostics for each.
- [references/design_analysis.md](references/design_analysis.md): fake-data design analysis,
  coverage checking, Type S and M errors, sample-size formulas.
- [references/poststratification.md](references/poststratification.md): poststratification tables,
  MRP, propagating cell-count uncertainty.
- [references/causal.md](references/causal.md): comparisons vs effects, ignorability, balance and
  overlap, post-treatment adjustment, instrumental variables and regression discontinuity.
