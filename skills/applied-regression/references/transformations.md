# Transformations and Interpretation

Source: *Regression and Other Stories* ch. 12, §6.3, §13.2.

## Linear transformations change interpretation, not fit

Rescaling `x` or `y` linearly leaves predictions and fit identical; only the coefficients move.
So choose the scale that makes the coefficient a quantity you can defend in a sentence.

| Idiom | Code | Coefficient means |
|---|---|---|
| Center | `x_c <- x - mean(x)` | intercept = prediction at the mean |
| Center at a reference | `c_height <- height - 66` | intercept = prediction at 66 inches |
| Standardize (1 sd) | `z <- (x - mean(x)) / sd(x)` | change in y per 1 sd of x |
| **Standardize (2 sd)** | `z <- (x - mean(x)) / (2 * sd(x))` | comparable to a 0/1 indicator's coefficient |
| Convenient units | `income / 10000`, `(pid - 4) / 4` | per $10k, per half-scale |

**Why 2 sd**: a binary predictor with `p ≈ 0.5` has `sd ≈ 0.5`, so its full 0-to-1 comparison
spans 2 sd. Dividing a continuous predictor by 1 sd would make its coefficient correspond to only
half the range a binary coefficient covers. Divide by 2 sd and the two are directly comparable.

Caveat: standardizing by the **sample** mean and sd only makes sense when n is large enough for
those to be stable, and it makes coefficients incomparable across datasets. For small samples or
cross-study comparison, standardize by an external reference scale.

For models without interactions you can skip the transformation and multiply each `beta_k` by
`2 * sd(x_k)` afterward to get the same comparability.

## Centering with interactions

In `y ~ x1 + x2 + x1:x2`, the coefficient on `x1` is the comparison **at `x2 = 0`**. If 0 is
outside the data, that number is meaningless.

```r
d$c_mom_hs <- d$mom_hs - mean(d$mom_hs)
d$c_mom_iq <- d$mom_iq - mean(d$mom_iq)
fit <- stan_glm(kid_score ~ c_mom_hs * c_mom_iq, data = d)
```

Centering does not change the interaction coefficient or the residual sd; it makes each main
effect "the comparison at the other variable's average". Center at a substantively meaningful
value when there is one (`mom_iq - 100`, `mom_hs - 0.5`).

In Stan, do this in `transformed data` and back-transform in `generated quantities`; brms centers
predictors internally for the intercept and reports the uncentered value.

## Correlation and regression to the mean

With `x` and `y` both standardized, the intercept is 0 and the slope **is** the correlation, so
it is always in [-1, 1]. In general `b = rho * sd_y / sd_x`.

The regression line (minimizing vertical distance) is not the principal-component line
(minimizing perpendicular distance). At `rho = 0.5` the regression slope is half the principal
component slope. The regression line is the right one for predicting `y` from `x`.

Regression to the mean: a unit at `k` sd above the mean in `x` is predicted at `rho * k` sd in
`y`. This is a property of conditional means, not of the process; individual realizations are not
pulled toward the mean.

## Logarithms

Use for all-positive variables. Two benefits: back-transformed predictions are automatically
positive, and a linear model on the log scale is a **multiplicative** model on the original
scale:

`log y = b0 + b1 x1 + e` becomes `y = B0 * B1^x1 * E` with `Bk = exp(bk)`.

Interpretation:

| Coefficient | Reading |
|---|---|
| 0.06 | about 6% higher per unit (since `exp(0.06) ≈ 1.06`) |
| −0.06 | about 6% lower |
| 0.30 | `exp(0.30) = 1.35`, 35% higher; the linear approximation is already off |
| 1.00 | 2.7 times, not double; unusually large |

The approximation `exp(b) ≈ 1 + b` is good for `|b| < 0.1`. Above that, exponentiate.

Prefer natural log to log10: the coefficient reads directly as a proportional change. log10 makes
predicted *values* easier to read but coefficients harder.

**Log-log (elasticity)**: `log(y) ~ log(x)` gives percent change in y per percent change in x.

Even when the dynamic range is narrow (log height has sd 0.06, so the fit barely changes), the
log scale can still be the better frame for interpretation.

Always check with a density overlay: a raw-scale linear model on a positive outcome produces
replicated negative values, visible immediately.

```r
ppc_dens_overlay(d$earn[d$earn > 0], posterior_predict(fit_raw)[1:50, ])
ppc_dens_overlay(log(d$earn[d$earn > 0]), posterior_predict(fit_log)[1:50, ])
```

## Comparing models across an outcome transformation

`elpd` values on different outcome scales are not comparable. To compare a model of `log(y)`
against a model of `y`, add the log Jacobian `log|d log(y)/dy| = -log(y)` to each pointwise elpd
of the log model:

```r
loo_log_adj <- loo(fit_log)
loo_log_adj$pointwise[, 1] <- loo_log_adj$pointwise[, 1] - log(d$weight)
loo_compare(loo(fit_raw), loo_log_adj)
```

`loo_compare` warns about differing response variables; that warning is expected once you have
made the adjustment manually.

## Other transformations

- **Square root**: milder than log for counts; predictions are awkward to back-transform, so it
  suits prediction more than interpretation.
- **Two-part / hurdle**: for a point mass at zero plus a continuous positive part, fit a logistic
  model for `Pr(y > 0)` and a log-scale model conditional on `y > 0`.
- **Discretizing a continuous predictor**: justified when no parametric form is plausible (age in
  four bins to allow generational patterns). It loses within-bin trend and the bin boundaries are
  a modeling choice, so show both the binned and the continuous fit.
- **Prefer continuous outcomes to discretized ones**: model vote share and threshold afterward
  rather than modeling the binary winner.
- **Index vs indicator**: for a J-level factor include at most J−1 indicators alongside an
  intercept. Name indicators for their meaning (`male`, not `sex_1`).

## Building a model for prediction (ROS §12.6)

1. Include all substantively relevant inputs.
2. Combine related items into a score rather than entering each separately.
3. Consider interactions between predictors with plausibly large effects.
4. Read standard errors as uncertainty, and expect estimates to move with new data.
5. Inclusion decisions:
   - small SE → keep it;
   - large SE and no substantive reason → dropping it can stabilize the rest;
   - substantively important → keep it even with a large SE, and address the uncertainty with more
     data or a stronger prior;
   - implausible sign with a small SE → investigate; it usually means a real, interpretable
     confounding structure.
6. Record every modeling choice. LOO comparison helps only if you have not tried too many models.

The mesquite-bushes example runs the whole loop: a raw-scale linear model with unstable LOO
(`p_loo` 17.8 against 9 parameters) → the same model on the log scale with all Pareto k fine →
a geometrically motivated single predictor (`log(canopy_volume)`) achieving LOO R² 0.78 against
0.84 for the full model. Near-collinear predictors inflate individual coefficient standard errors
while the joint posterior stays well identified; check with a scatterplot of the two coefficients'
draws before concluding a predictor "does not matter".
