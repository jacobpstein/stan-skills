---
name: r-stan-interfaces
description: >
  Load when the user is working with Stan from R: cmdstanr (cmdstan_model, $sample, $summary,
  $draws, $loo, $pathfinder, $laplace, $optimize, $generate_quantities), rstanarm (stan_glm,
  stan_glmer, stan_lmer, stan_polr, default priors and autoscaling, prior_summary, posterior_predict,
  pp_check, loo, bayes_R2), brms (brm, formula syntax, set_prior/get_prior/default_prior,
  make_stancode/stancode, families, add_criterion, conditional_effects, threading, backend),
  and the supporting packages posterior, bayesplot, loo, priorsense, projpred, tidybayes.
  Triggers include: cmdstanr, rstanarm, brms, stan_glm, stan_glmer, brm(, set_prior, get_prior,
  make_stancode, pp_check, posterior_predict, posterior_epred, summarise_draws, as_draws,
  loo_compare, add_criterion, powerscale_sensitivity, cv_varsel, rstan, library(brms).
---

# Stan from R: cmdstanr, rstanarm, brms

Three ways to reach the same sampler. Pick by how much control you need over the model:

| Interface | Use when | Cost |
|---|---|---|
| **rstanarm** | the model is a standard GLM/GLMM and you want sane defaults now | fixed model menu; rstan backend, precompiled |
| **brms** | formula syntax covers it (most multilevel, distributional, nonlinear, survival, ordinal models) | compilation per model; generated Stan code is verbose but readable |
| **cmdstanr** | you wrote the `.stan` file, or brms cannot express the model | you own the program, the priors, and the generated quantities |

A productive pattern: prototype in brms, run `stancode(fit)` to get the Stan program, then edit
that program and continue in cmdstanr when you need something brms does not offer.

Versions current as of September 2026: CmdStan/Stan 2.39.0, cmdstanr 0.9.0 (from the stan-dev
R-universe, not CRAN), rstanarm 2.32.2, brms 2.23.0, posterior 1.7.0, loo 2.10.1,
bayesplot 1.16.0, priorsense 1.2.0, projpred 2.10.0.

```r
install.packages("cmdstanr", repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
cmdstanr::install_cmdstan(cores = 4)
```

## cmdstanr

```r
library(cmdstanr); library(posterior); library(bayesplot)

mod <- cmdstan_model("m.stan",
                     stanc_options = list("warn-pedantic" = TRUE, "O1" = TRUE),
                     cpp_options = list(stan_threads = TRUE))

fit <- mod$sample(
  data = data, chains = 4, parallel_chains = 4,
  iter_warmup = 1000, iter_sampling = 1000,
  seed = 20260914, init = 0.1, adapt_delta = 0.8, max_treedepth = 10,
  refresh = 500
)

fit$diagnostic_summary()          # divergences, treedepth, E-BFMI per chain
fit$cmdstan_diagnose()            # CmdStan's own diagnose report
fit$summary(c("alpha", "beta", "sigma"))
fit$summary(NULL, default_convergence_measures())   # rhat, ess_bulk, ess_tail
fit$summary(NULL, default_mcse_measures())          # mcse_mean, mcse_q5, mcse_q95
fit$save_object("results/m_v1.rds")                 # survives temp-dir cleanup
```

Draws come out as `posterior` objects:

```r
fit$draws()                              # draws_array (iteration, chain, variable)
fit$draws(format = "draws_df")
fit$draws("beta", format = "draws_matrix")
fit$draws(format = "draws_rvars")        # rvar: arithmetic propagates draws automatically
fit$sampler_diagnostics()                # divergent__, treedepth__, energy__, stepsize__
```

Other methods: `$optimize()`, `$laplace(mode = , draws = )`, `$pathfinder(num_paths = 4)`,
`$variational()`, `$generate_quantities(fitted_params = fit, data = data)`, `$loo()`.
`$init_model_methods()` compiles `$log_prob()`, `$grad_log_prob()`, and `$unconstrain_draws()`
into R, which is what `loo_moment_match` needs.

Within-chain threading needs `cpp_options = list(stan_threads = TRUE)` at compile time and
`threads_per_chain =` at sample time; OpenCL needs `cpp_options = list(stan_opencl = TRUE)` and
`opencl_ids = c(platform, device)`.

## rstanarm

Precompiled, so no wait for the compiler, and the defaults are the ones from *Regression and
Other Stories*.

```r
library(rstanarm)
fit <- stan_glm(y ~ x1 + x2, data = d, family = gaussian(),
                prior = normal(0, 2.5, autoscale = TRUE),
                prior_intercept = normal(0, 2.5, autoscale = TRUE),
                prior_aux = exponential(1, autoscale = TRUE),
                chains = 4, seed = 1, refresh = 0)
prior_summary(fit)      # shows the adjusted (autoscaled) priors actually used
```

### Default priors and what autoscaling means

| Target | Default | After autoscaling |
|---|---|---|
| Coefficients | `normal(0, 2.5)` | `normal(0, 2.5 * sd(y) / sd(x_k))`; `sd(y)` replaced by 1 for non-Gaussian families |
| Intercept | `normal(0, 2.5)` **on the centered predictors** | `normal(mean(y), 2.5 * sd(y))` for Gaussian identity link, location 0 otherwise |
| `sigma` (`prior_aux`) | `exponential(1)` | rate `1 / sd(y)` |
| Group-level covariance (`stan_glmer`) | `decov(regularization = 1, concentration = 1, shape = 1, scale = 1)` | LKJ on correlations, Dirichlet simplex splitting the total variance, gamma on the total |

The intercept prior is a prior on `E[y | x = mean(x)]`, not on the value at `x = 0`. Set
`autoscale = FALSE` when you specify priors on the raw scale yourself.

Model functions: `stan_glm`, `stan_glm.nb`, `stan_lmer`, `stan_glmer`, `stan_glmer.nb`,
`stan_polr` (ordinal), `stan_betareg`, `stan_nlmer`, `stan_gamm4`, `stan_mvmer`, `stan_jm`.

Useful arguments: `prior_PD = TRUE` (prior predictive only), `QR = TRUE` (QR reparameterization
for correlated predictors), `algorithm = "sampling" | "meanfield" | "fullrank"`.

Post-fit: `posterior_predict()`, `posterior_epred()`, `posterior_linpred(transform = TRUE)`,
`pp_check()`, `loo()`, `kfold()`, `bayes_R2()`, `loo_R2()`, `predictive_error()`,
`as.matrix()` / `as_draws()`.

rstanarm runs on rstan with models compiled at package install; there is no cmdstanr backend.

## brms

```r
library(brms)
fit <- brm(
  bf(y ~ x1 + x2 + (1 + x1 | group)),
  data = d, family = gaussian(),
  prior = prior(normal(0, 1), class = b) +
          prior(normal(250, 100), class = Intercept) +
          prior(exponential(0.04), class = sigma) +
          prior(exponential(0.04), class = sd, group = group, coef = Intercept) +
          prior(exponential(0.1),  class = sd, group = group, coef = x1) +
          prior(lkj(2), class = cor),
  chains = 4, cores = 4, seed = 1,
  backend = "cmdstanr",
  control = list(adapt_delta = 0.95),
  save_pars = save_pars(all = TRUE),     # needed for loo_moment_match
  file = "results/fit_m2"                # caches the fit
)
```

### Prior classes

| `class` | Applies to |
|---|---|
| `b` | population-level coefficients; narrow with `coef = "x1"` |
| `Intercept` | intercept **after brms centers the predictors** |
| `sigma` | residual sd |
| `sd` | sds of group-level effects; narrow with `group =` and `coef =` |
| `cor` | LKJ prior on group-level correlation matrices |
| `shape`, `phi`, `nu`, `zi`, `hu` | family-specific parameters; use `dpar =` for distributional parameters |

brms defaults are **flat on `class = b`**, so always set coefficient priors explicitly. Inspect
what is settable with `get_prior(formula, data, family)` (also available as `default_prior()` in
recent versions), and check what was used with `prior_summary(fit)`.

`sample_prior = "only"` fits the prior predictive; `sample_prior = "yes"` keeps prior draws
alongside the posterior for `hypothesis()`.

### Inspecting and escaping to Stan

```r
stancode(fit)            # or make_stancode(formula, data, family, prior)
standata(fit)            # or make_standata(...)
```

Read the generated code to learn the idioms brms uses (centered intercept, non-centered group
effects with `z` and `L`), then lift it into your own program when you outgrow the formula.

### Post-fit

```r
pp_check(fit, type = "dens_overlay", ndraws = 50)
pp_check(fit, type = "stat", stat = "sd")
pp_check(fit, type = "stat_grouped", stat = "mean", group = "site")
pp_check(fit, type = "rootogram", style = "hanging")     # counts
pp_check(fit, type = "loo_pit_ecdf", method = "correlated")
pp_check(fit, type = "loo_intervals")

fit <- add_criterion(fit, "loo", moment_match = TRUE, save_psis = TRUE)
loo_compare(fit1, fit2)
conditional_effects(fit)
hypothesis(fit, "x1 > 0")
posterior_epred(fit, newdata = nd)        # expectation
posterior_predict(fit, newdata = nd)      # predictive draws
```

Threading: `threads = threading(2)` with `backend = "cmdstanr"`.

## Supporting packages

```r
library(posterior)
draws <- as_draws_df(fit)                       # works on cmdstanr, brms, rstanarm fits
summarise_draws(draws, mean, sd, ~quantile(.x, c(0.05, 0.95)),
                default_convergence_measures())
rv <- as_draws_rvars(draws)
rv$ratio <- rv$alpha / rv$beta                  # arithmetic on rvars keeps the draws
quantile(rv$ratio, c(0.05, 0.95))

library(bayesplot)
mcmc_rank_overlay(draws, pars = c("alpha", "beta"))   # more sensitive than trace plots
mcmc_pairs(draws, np = nuts_params(fit), pars = c("sigma_alpha", "alpha[1]"))
mcmc_trace(draws, np = nuts_params(fit))

library(loo)
loo_compare(loo1, loo2); loo_model_weights(list(loo1, loo2), method = "stacking")

library(priorsense)
powerscale_sensitivity(fit)
powerscale_plot_dens(fit, variable = c("b_x1", "sigma"))

library(projpred)
cvvs <- cv_varsel(fit, method = "forward", cv_method = "LOO", validate_search = TRUE)
suggest_size(cvvs)
```

`tidybayes` / `ggdist` (`spread_draws`, `add_epred_draws`, `stat_halfeye`) are the tidyverse-side
equivalents when you are plotting with ggplot2.

## Choosing and switching

- Start in brms for anything the formula covers. It gives correct non-centered parameterizations,
  sensible generated quantities, and `loo`/`pp_check` for free.
- Move to cmdstanr when you need a custom likelihood, a latent-variable structure, integrated
  LOO, or full control of `generated quantities`.
- Use rstanarm when the model is a plain GLM/GLMM, compilation time matters, or you are following
  *Regression and Other Stories* directly.
- All three produce `posterior` draws objects, so diagnostics, `loo`, `bayesplot`, and
  `priorsense` work identically downstream.

## References

- [references/cmdstanr.md](references/cmdstanr.md): full method list, threading, OpenCL, saving
  and reloading fits, standalone generated quantities, moment matching setup.
- [references/rstanarm.md](references/rstanarm.md): model functions, the default-prior table with
  worked autoscaling arithmetic, and the ROS idioms.
- [references/brms.md](references/brms.md): formula syntax, family list, prior classes,
  distributional and nonlinear models, `pp_check` types, and reading the generated Stan code.
