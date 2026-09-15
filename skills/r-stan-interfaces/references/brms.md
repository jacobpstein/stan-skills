# brms Reference

brms 2.23.0. Formula syntax compiles to a Stan program; `stancode(fit)` shows it.

## Formula syntax

| Syntax | Meaning |
|---|---|
| `y ~ x1 + x2` | population-level (fixed) effects |
| `y ~ x + (1 | g)` | varying intercept by `g` |
| `y ~ x + (1 + x | g)` | varying intercept and slope, correlated |
| `y ~ x + (1 + x || g)` | same, uncorrelated |
| `y ~ x + (1 | g1) + (1 | g2)` | crossed groupings |
| `y ~ x + (1 | g1/g2)` | nested: `g1` and `g1:g2` |
| `y ~ x + (1 | gr(g, by = z))` | separate covariance per level of `z` |
| `y ~ s(x)`, `y ~ t2(x1, x2)` | smooth terms (mgcv bases) |
| `y ~ gp(x)` | approximate Gaussian process |
| `y ~ me(x, sdx)` | predictor measured with error |
| `y ~ mo(x_ordinal)` | monotonic effect of an ordered factor |
| `y ~ ar(p = 1, gr = subject)` | autoregressive residuals |
| `y | trials(n) ~ x` | binomial |
| `y | cens(censored) ~ x` | censoring indicator column |
| `y | trunc(lb = 0) ~ x` | truncation |
| `y | weights(w) ~ x` | weighted likelihood |
| `y | rate(exposure) ~ x` | Poisson/NB offset by exposure |
| `bf(y ~ x, sigma ~ x)` | distributional model: predict sigma too |
| `bf(y ~ a * exp(-b * t), a + b ~ 1 + (1|id), nl = TRUE)` | nonlinear model |

Common families: `gaussian`, `student`, `bernoulli`, `binomial`, `beta_binomial`, `poisson`,
`negbinomial`, `zero_inflated_poisson`, `zero_inflated_negbinomial`, `hurdle_poisson`,
`hurdle_lognormal`, `Beta`, `Gamma`, `lognormal`, `weibull`, `exponential`,
`cumulative` / `sratio` / `cratio` / `acat` (ordinal), `categorical`, `multinomial`, `wiener`.

## Priors

```r
get_prior(y ~ x + (1 + x | g), data = d, family = gaussian())
# (also available as default_prior() in recent versions)

pr <- prior(normal(0, 1), class = b) +
      prior(normal(0, 5), class = Intercept) +
      prior(exponential(1), class = sigma) +
      prior(exponential(1), class = sd, group = g) +
      prior(exponential(2), class = sd, group = g, coef = x) +
      prior(lkj(2), class = cor) +
      prior(normal(0, 1), class = b, dpar = "zi") +
      prior(horseshoe(df = 1, par_ratio = 0.1), class = b)

fit <- brm(y ~ x + (1 + x | g), data = d, prior = pr)
prior_summary(fit)
```

Defaults: **flat on `class = b`** (always override), `student_t(3, median(y), 2.5*mad(y))` on the
intercept, half-`student_t` on `sigma` and `sd`, `lkj(1)` on `cor`. `lb =` / `ub =` truncate.
The `Intercept` prior applies to the intercept after brms centers the predictors, i.e. to the
expected outcome at the predictor means.

Special priors for many coefficients: `horseshoe(df, par_ratio, ...)`,
`R2D2(mean_R2, prec_R2, cons_D2)`, `lasso()`.

## Prior predictive

```r
fit_prior <- brm(y ~ x, data = d, prior = pr, sample_prior = "only")
pp_check(fit_prior, ndraws = 50)
```

`sample_prior = "yes"` keeps prior draws alongside the posterior so `hypothesis()` can compute
evidence ratios.

## Fitting

```r
fit <- brm(
  formula, data = d, family = gaussian(), prior = pr,
  chains = 4, iter = 2000, warmup = 1000, cores = 4, seed = 1,
  backend = "cmdstanr",                       # or "rstan"
  threads = threading(2),                     # within-chain, cmdstanr backend only
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  save_pars = save_pars(all = TRUE),          # required for loo_moment_match
  init = 0.1,
  file = "results/fit_m2", file_refit = "on_change"
)
```

`file =` caches the fit to disk; `file_refit = "on_change"` refits only when the model or data
change. This is the single biggest quality-of-life setting for an iterative workflow.

## Inspecting the Stan code

```r
stancode(fit)                                   # or make_stancode(formula, data, family, prior)
standata(fit)                                   # or make_standata(...)
```

Reading the generated program teaches the idioms: centered intercept `Intercept` with
`b_Intercept` recovered in `generated quantities`, non-centered group effects via `z_1` and
`L_1`, `target += ...` accumulation, and a `lprior` variable that `priorsense` uses.

## Checking and comparison

```r
pp_check(fit, type = "dens_overlay", ndraws = 50)
pp_check(fit, type = "ecdf_overlay")
pp_check(fit, type = "stat", stat = "sd")
pp_check(fit, type = "stat_2d", stat = c("mean", "sd"))
pp_check(fit, type = "stat_grouped", stat = "mean", group = "site")
pp_check(fit, type = "intervals", x = "time")
pp_check(fit, type = "rootogram", style = "hanging")     # counts
pp_check(fit, type = "bars")                             # small integer outcomes
pp_check(fit, type = "error_scatter_avg_vs_x", x = "x1")
pp_check(fit, type = "loo_pit_ecdf", method = "correlated")
pp_check(fit, type = "loo_intervals")

fit <- add_criterion(fit, "loo", moment_match = TRUE, save_psis = TRUE)
fit <- add_criterion(fit, "loo", reloo = TRUE)           # refits the bad folds
kfold(fit, K = 10)
loo_compare(fit1, fit2, model_names = c("M1", "M2"))
bayes_R2(fit); loo_R2(fit)
```

`moment_match = TRUE` requires `save_pars = save_pars(all = TRUE)` at fit time. When many Pareto
k are bad (flexible models with one parameter per observation), go straight to `kfold()`.

## Post-processing

```r
posterior_epred(fit, newdata = nd, re_formula = NA)   # expectation, population-level only
posterior_predict(fit, newdata = nd, allow_new_levels = TRUE)
posterior_linpred(fit, transform = TRUE)
conditional_effects(fit, effects = "x1:x2")
hypothesis(fit, "x1 > 0")
hypothesis(fit, "sd_g__Intercept > sigma", class = NULL)
as_draws_df(fit)
emmeans::emmeans(fit, ~ group)                        # marginal means, if you use emmeans
```

`re_formula = NA` drops group-level effects (population prediction); `re_formula = NULL`
(default) keeps them; `allow_new_levels = TRUE` with `sample_new_levels = "gaussian"` draws
fresh group effects for unseen groups.

Counterfactual contrasts:

```r
nd0 <- transform(d, treatment = 0); nd1 <- transform(d, treatment = 1)
ratio <- rowMeans(posterior_epred(fit, newdata = nd1)) /
         rowMeans(posterior_epred(fit, newdata = nd0))
quantile(ratio, c(0.05, 0.5, 0.95))
```

This is the right way to summarize an effect in a zero-inflated or nonlinear model where the
coefficients split across components.

## Sensitivity

```r
library(priorsense)
powerscale_sensitivity(fit)
powerscale_plot_dens(fit, variable = c("b_x1", "sigma"))
powerscale_sensitivity(fit, prediction = \(x, ...) posterior::rvar(ratio))
```

brms writes an `lprior` variable into the generated code, which is what makes this work without
extra setup.

## When to leave brms

- Custom likelihood that `custom_family()` cannot express cleanly.
- Latent-variable structures (IRT with discrimination, state space, ODE systems).
- Integrated LOO via `integrate_1d`.
- Full control over `generated quantities`.

The migration path is `stancode(fit)` and `standata(fit)`: you start from working, correct Stan
code rather than a blank file.
