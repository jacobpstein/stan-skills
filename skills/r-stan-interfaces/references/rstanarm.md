# rstanarm Reference

rstanarm 2.32.2. Precompiled Stan models for standard regressions, run through rstan. No
compilation wait, fixed model menu, and the default priors from *Regression and Other Stories*.

## Model functions

| Function | Model |
|---|---|
| `stan_glm` | GLM (gaussian, binomial, poisson, Gamma, inverse.gaussian) |
| `stan_glm.nb` | negative binomial |
| `stan_lmer`, `stan_glmer`, `stan_glmer.nb` | multilevel, lme4 formula syntax |
| `stan_polr` | ordinal (proportional odds) |
| `stan_betareg` | beta regression for proportions |
| `stan_nlmer` | nonlinear multilevel |
| `stan_gamm4` | generalized additive multilevel |
| `stan_mvmer`, `stan_jm` | multivariate and joint longitudinal-survival |
| `stan_biglm` | large linear models |

```r
fit <- stan_glm(y ~ x1 + x2, data = d, family = gaussian(),
                chains = 4, iter = 2000, seed = 1, refresh = 0)
fit_h <- stan_glmer(y ~ x + (1 + x | g), data = d, family = binomial(),
                    adapt_delta = 0.95)
```

## Default priors and autoscaling

```r
prior_summary(fit)
```

| Argument | Default | Autoscaled to |
|---|---|---|
| `prior` (coefficients) | `normal(0, 2.5, autoscale = TRUE)` | `normal(0, 2.5 * sd(y) / sd(x_k))`; `sd(y)` is replaced by 1 for non-Gaussian families |
| `prior_intercept` | `normal(0, 2.5, autoscale = TRUE)`, applied **on the centered predictors** | `normal(mean(y), 2.5 * sd(y))` for gaussian identity link; location 0 otherwise |
| `prior_aux` (`sigma`, `phi`, `shape`) | `exponential(1, autoscale = TRUE)` | rate `1 / sd(y)` |
| `prior_covariance` (glmer) | `decov(regularization = 1, concentration = 1, shape = 1, scale = 1)` | LKJ on correlations, Dirichlet simplex splitting the total variance, gamma on the total scale |

Key points:

- The intercept prior is on `E[y | x = mean(x)]`. rstanarm centers internally and reports the
  uncentered intercept.
- Autoscaling is what makes `2.5` mean "weakly informative on a standardized scale". Turn it off
  (`autoscale = FALSE`) when you are specifying a prior in the raw units yourself.
- Available prior families: `normal`, `student_t`, `cauchy`, `laplace`, `lasso`, `hs`
  (horseshoe), `hs_plus`, `product_normal`, `exponential`, `decov`, `lkj`, `dirichlet`,
  `R2` (for `stan_lm` / `stan_polr`), and `NULL` for a flat prior (rarely a good idea).

```r
fit <- stan_glm(y ~ x, data = d,
                prior = normal(0, 1, autoscale = FALSE),
                prior_intercept = normal(200, 50, autoscale = FALSE),
                prior_aux = exponential(1/30, autoscale = FALSE))
```

## Useful arguments

- `prior_PD = TRUE`: sample the prior predictive distribution only.
- `QR = TRUE`: QR-decompose the design matrix for correlated or numerous predictors; coefficients
  are transformed back automatically.
- `algorithm = "sampling"` (NUTS), `"meanfield"` or `"fullrank"` (ADVI), `"optimizing"`.
- `adapt_delta`, `iter`, `chains`, `cores`, `seed`, `refresh` pass through to rstan.
- `offset()` in the formula for exposure in count models; `weights =` for weighted likelihoods.

## Post-fit

```r
print(fit, digits = 2)            # median and MAD-sd, the ROS default summary
summary(fit, probs = c(0.05, 0.5, 0.95))    # includes n_eff and Rhat
posterior_interval(fit, prob = 0.9)
as.matrix(fit); as.data.frame(fit); posterior::as_draws_df(fit)

posterior_predict(fit, newdata = nd)        # includes observation noise
posterior_epred(fit, newdata = nd)          # expectation
posterior_linpred(fit, transform = TRUE)
predictive_error(fit)
predictive_interval(fit, prob = 0.9)

pp_check(fit, plotfun = "dens_overlay", nreps = 50)
pp_check(fit, plotfun = "stat", stat = "sd")

loo1 <- loo(fit, save_psis = TRUE)
loo1 <- loo(fit, k_threshold = 0.7)         # refits the bad folds
kfold(fit, K = 10)
loo_compare(loo1, loo2)
bayes_R2(fit); loo_R2(fit)
```

## ROS idioms

```r
# median and MAD-sd summary, which is what the book reports
print(fit, digits = 2)

# simulate first, summarize last
draws <- as.matrix(fit)
ratio <- draws[, "x1"] / draws[, "x2"]
quantile(ratio, c(0.05, 0.5, 0.95))

# predictive comparison at two predictor values
nd <- data.frame(x1 = c(0, 1), x2 = mean(d$x2))
epred <- posterior_epred(fit, newdata = nd)
quantile(epred[, 2] - epred[, 1], c(0.05, 0.5, 0.95))

# LOO-based R2 and log score
loo_R2(fit)
loo(fit)$estimates["elpd_loo", ]
```

## Limits

- No cmdstanr backend: models are precompiled against rstan/StanHeaders at package install.
- You cannot see or edit the Stan program (`stan_glm` shares one large precompiled model per
  family).
- If you need a custom likelihood, a latent variable, or custom generated quantities, move to
  brms (which will show you its generated Stan) or write the program yourself for cmdstanr.
