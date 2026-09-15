# Generalized Linear Models: Practice

Source: *Regression and Other Stories* ch. 13-15.

A GLM is: outcome `y`, linear predictor `X beta`, link `g` with `E(y|X) = g^{-1}(X beta)`, a data
distribution `p(y | yhat)`, and possibly extra parameters (dispersion, cutpoints).

## Logistic regression

`Pr(y=1) = inv_logit(X beta)`. The curve is steepest at p = 0.5, where its slope is `beta / 4`.

### Interpreting

- **Divide by 4** for the maximum change in probability per unit of x. `0.33 / 4 = 0.08`: at most
  8 percentage points per step of a 1-5 income scale.
- Multiply by the predictor's sd first to compare predictors on different scales:
  `(-0.90 * 0.38) / 4 = -0.085` vs `(0.46 * 1.10) / 4 = 0.13`.
- Odds ratios (`exp(beta)`) avoid the boundary problem but are less intuitive; prefer the
  probability scale.
- Evaluate predictions at meaningful reference points, not at x = 0.

### Average predictive comparison (ROS §14.4)

The general, honest summary for any nonlinear model. Split inputs into `u` (the one you are
comparing) and `v` (the rest). Average the predicted difference over the **empirical** distribution
of `v`, not at its mean:

```r
b <- coef(fit)
delta <- invlogit(b[1] + b[2]*hi + b[3]*d$arsenic + b[4]*d$educ4) -
         invlogit(b[1] + b[2]*lo + b[3]*d$arsenic + b[4]*d$educ4)
round(mean(delta), 2)
```

With interactions, add the interaction terms to both the hi and lo linear predictors before
differencing. In Bayesian practice, also average over posterior draws:

```r
epred_hi <- posterior_epred(fit, newdata = transform(d, dist100 = 1))
epred_lo <- posterior_epred(fit, newdata = transform(d, dist100 = 0))
delta <- rowMeans(epred_hi - epred_lo)          # one value per draw
quantile(delta, c(0.05, 0.5, 0.95))
```

Evaluating at `mean(v)` can be badly misleading when `v` is spread out or bimodal: if the data sit
at the flat ends of the curve, the comparison at `E(v)` looks artificially large; if the data sit
in the middle but `E(v)` does not, it understates.

### Checking

- **Binned residual plots**: bin by fitted probability (or by a predictor), plot the mean residual
  per bin with bounds `2 * sqrt(p_j (1 - p_j) / n_j)`. About 95% should land inside. A pattern
  against a predictor suggests a transformation of that predictor.
- **Error rate** is a weak summary: it treats p = 0.6 and p = 0.99 alike and can show no
  improvement over predicting the majority class even for a genuinely better model in an
  imbalanced problem. Compare it to the null rate `min(p̄, 1 - p̄)`, then use the log score
  instead.
- **Log score**: `sum(y * log(p) + (1 - y) * log(1 - p))`. As a scale for intuition: a model that
  says p = 0.5 for everything scores `n * log(0.5) = -0.693n`; a well-calibrated model giving 0.8
  to half and 0.6 to the rest scores about `-0.587n`, an improvement of roughly one log-score unit
  per ten observations.

```r
predp <- predict(fit, type = "response")
logscore <- sum(y * log(predp) + (1 - y) * log(1 - predp))
loo(fit)      # elpd_loo, p_loo; p_loo should be near the parameter count
```

### Separation

If any predictor, or any linear combination of predictors, perfectly separates the outcome, the
MLE is `±Inf`. `glm()` returns a large finite number, which is an artifact of stopping the
iteration. The 1964 election survey had 87 Black respondents and zero Republican preferences, so
the MLE for that coefficient is `-Inf`.

**The fix is a proper prior.** rstanarm's default `normal(0, 2.5 / sd(x))` is enough to give a
large but finite, sensible estimate while leaving well-identified coefficients essentially
unchanged. Never fit binary models with `prior = NULL`: separation appears unpredictably whenever
a subgroup is small.

### Latent formulation

`y_i = 1{z_i > 0}`, `z_i = X_i beta + e_i` with `e_i` logistic (sd ≈ 1.6). The scale is not
identified jointly with `beta`, hence the convention. Probit uses a standard normal instead, so
**probit coefficients ≈ logistic coefficients / 1.6**.

## Count data

### Poisson

`y ~ Poisson(exp(X beta))`. No dispersion parameter: `sd(y) = sqrt(E(y))`. Real count data are
almost always **over**dispersed, so Poisson is usually the wrong starting point.

### Negative binomial

`sd(y|x) = sqrt(E(y|x) + E(y|x)^2 / phi)`. Small `phi` means strong overdispersion;
`phi -> Inf` recovers Poisson. Start here.

```r
fit <- stan_glm(y ~ x, family = neg_binomial_2(link = "log"), data = d)
# the auxiliary "reciprocal_dispersion" is phi
```

```stan
y ~ neg_binomial_2_log_glm(X, log_exposure + alpha, beta, phi);
```

Coefficients are multiplicative: `exp(0.012) = 1.012`, a 1.2% higher rate per unit; rescale to
per-10-units (`0.12`) for a more readable 12.7%.

### Exposure and offsets

When unit `i` has baseline exposure `u_i` (trap-days, person-years, vehicle flow), model
`y_i ~ NB(u_i * theta_i, phi)` with `theta_i = exp(X_i beta)`. `log(u_i)` is the **offset**: a
predictor with its coefficient fixed at 1.

```r
fit <- stan_glm(y ~ roach100 + treatment + senior, family = neg_binomial_2,
                offset = log(exposure2), data = roaches)
```

```r
brm(y ~ x + offset(log(exposure)), family = negbinomial(), data = d)
```

### Choosing the count model

- Successes out of a known number of trials → binomial/logistic.
- No natural upper limit → Poisson/negative binomial.
- Natural limit far above the typical count (population 100,000, expected count 4.5) → Poisson/NB
  is a fine approximation.

### Checking counts

```r
yrep <- posterior_predict(fit)
mean(apply(yrep, 1, function(r) mean(r == 0)))     # compare to mean(y == 0)
ppc_dens_overlay(log10(y + 1), log10(yrep[1:50, ] + 1))
ppc_rootogram(y, yrep, style = "hanging")
```

The roaches Poisson model produced replicated zero-proportions of 0 to 0.008 against an observed
36%. Switching to negative binomial fixed the zeros but the maximum counts were still far too
large; always run more than one check.

## Binomial with overdispersion

Standardized residual `z_i = (y_i - n_i p̂_i) / sqrt(n_i p̂_i (1 - p̂_i))`; under a correct model
these are approximately standard normal, so `sum(z^2) / (N - k)` estimates the overdispersion and
can be compared to a chi-squared reference. The overdispersed form is
`sd(y) = sqrt(omega * n * p * (1 - p))` with `omega > 1`. Fit as a beta-binomial in brms
(`family = beta_binomial()`) or directly in Stan.

## Ordered categorical

`Pr(y > k) = inv_logit(X beta - c_k)` with strictly increasing cutpoints.

```r
fit <- stan_polr(factor(y) ~ x, data = d, prior = R2(0.3, "mean"))
brm(y ~ x, family = cumulative("logit"), data = d)
```

```stan
y ~ ordered_logistic(X * beta, cutpoints);      // no intercept: absorbed by the cutpoints
```

Three equivalent parameterizations exist (differing in what is fixed to zero); pick the one whose
cutpoints are interpretable in the units of the problem. Alternatives: probit; linear regression
on a numeric recoding if the categories are numerous and roughly equally spaced; nested binary
regressions for more flexibility at the cost of the latent-cutpoint interpretation.

**Unordered** categories (do nothing / switch to a private well / switch to a community well /
build new): do not force an ordering. Fit separate logistic models for each meaningful binary
sub-decision, or a categorical model.

## Robust regression

Replace normal or logistic errors with Student-t to downweight discordant points.

```r
brm(y ~ x, family = student(), data = d)           # continuous outcome
```

```stan
y ~ student_t(nu, alpha + X * beta, sigma);
nu ~ gamma(2, 0.1);
```

**Robit** is the binary analog: latent errors `t_nu(0, sqrt((nu-2)/nu))`, scaled to sd 1 for any
`nu`. `nu` is noisy to estimate; **fix it at 4** for robustness. `nu -> Inf` gives probit,
`nu ≈ 7` gives logit. On clean data robit and logit agree; with one flipped observation, logit is
visibly pulled and robit is not.

## Constructive choice models

A logistic regression can be derived from a decision rule rather than fitted descriptively: if a
household switches when net benefit exceeds cost, and the population distribution of the
benefit-to-cost ratio is logistic with center `mu` and scale `sigma`, then
`Pr(switch) = inv_logit((mu - x)/sigma)`, exactly a logistic regression with slope `-1/sigma`.
A normal distribution instead gives probit. This is a way to justify functional form from
substance: distance enters linearly because it is a directly perceived cost, arsenic enters
logarithmically because risk is perceived on a compressed scale.
