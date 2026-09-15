---
name: stan-priors
description: >
  Load when the user is choosing, justifying, checking, or debugging priors for a Stan model
  (CmdStanPy, cmdstanr, brms, rstanarm): weakly informative defaults, scaling priors to the
  data and predictors, priors for intercepts, coefficients, scales, hierarchical variances,
  correlation matrices (LKJ), log-scale/GLM priors, prior predictive checks, prior-likelihood
  conflict and tail behavior, power-scaling sensitivity (priorsense / az.psense), regularizing
  and sparsity priors (horseshoe, R2D2), and brms set_prior / rstanarm default priors.
  Triggers include: prior, weakly informative, flat prior, uniform prior, hyperprior,
  exponential(1), half-normal, LKJ, prior predictive, prior sensitivity, power scaling,
  priorsense, psense, horseshoe, R2D2, autoscale, prior_summary, set_prior, get_prior.
---

# Priors for Stan Models

Follows *Bayesian Workflow* §5.6–5.10, ch. 17, ch. 28 and *Regression and Other Stories* §9.5.
Stan gives a parameter an implicit improper uniform prior when no `~` or `target +=` statement
mentions it. That is almost never what you want: improper priors cannot be simulated from (no
prior predictive check, no SBC), make the posterior improper under separation or collinearity,
and slow or break HMC.

Start with weakly informative priors on unit-scale parameters, check them with a prior
predictive simulation, fit, then check prior sensitivity. Do not over-invest in the priors of
the first model before you have seen how it fits.

## Default recipe

1. **Put parameters on unit or interpretable scale.** Standardize or center predictors; use
   `log(theta / typical_value)` for positive quantities; `(year - 2000) / 10` for time.
2. **Coefficients** (unit-scale predictor and outcome): `normal(0, 1)` gives mild
   regularization; `normal(0, 10)` only rules out absurd values; `normal(0, 2.5)` is the
   rstanarm-style default. Realistic effects are often ~0.1 sd, so `normal(0, 1)` is weak.
3. **Intercept**: center predictors so the intercept is the expected outcome for a typical
   unit, then `normal(mean_y_guess, 2.5 * sd_y_guess)` or, on unit scale, `normal(0, 2.5)`.
4. **Scale parameters** (`sigma`, hierarchical `tau`): `exponential(1)` or `normal(0, 1)` with
   `<lower=0>` on unit scale; `exponential(1/typical_sd)` on the raw scale. Never
   `uniform(0, 100)`.
5. **Correlation matrices**: `lkj_corr_cholesky(1)` (uniform) or `(2)` (toward identity).
6. **Write one sentence per informative prior** explaining the numbers.

```stan
data {
  int<lower=0> N, K;
  matrix[N, K] X;
  vector[N] y;
}
transformed data {
  matrix[N, K] Xc;
  vector[K] x_mean;
  for (k in 1:K) { x_mean[k] = mean(X[, k]); Xc[, k] = X[, k] - x_mean[k]; }
}
parameters {
  real alpha_c;              // intercept at predictor means
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  alpha_c ~ normal(250, 100);    // response-scale guess: 95% in (50, 450)
  beta ~ normal(0, 20);          // per-unit-x change; 95% in (-40, 40)
  sigma ~ exponential(1.0 / 50); // rate = 1/mean; sd of prior = 50
  y ~ normal_id_glm(Xc, alpha_c, beta, sigma);
}
generated quantities {
  real alpha = alpha_c - dot_product(x_mean, beta);   // intercept at x = 0
}
```

## The informativity ladder for a unit-scale coefficient (BW §5.6)

| Prior | Character | Use |
|---|---|---|
| none (flat) | improper | never for final models; causes drift under weak data |
| `normal(0, 1e6)` | super-vague proper | behaves like flat; do not use |
| `normal(0, 10)` | rules out unrealistic regions | fine when data are strong |
| `normal(0, 1)` | rules out unrealistic regions + mild regularization | default |
| `normal(0.4, 0.2)` | specific information | only with documented background knowledge |
| `normal(0, 0.1)` / `normal(0, 0.05)` | "effects are usually small" | noisy studies of small effects |

"Weakly informative" is relative to the question: a prior that is weak for a coefficient can be
very strong for a prediction, and with 15 binary predictors under `normal(0, 1)` the implied
prior on a fitted probability piles up at 0 and 1 (BW Fig. 5.8). As predictors multiply, tighten
per-coefficient priors or use a joint prior on total explained variance.

## Scaling rules

### Regression on the raw scale (BW ch. 17, ROS §9.5)

- Intercept (centered predictors): `normal(m, s)` with 95% covering the plausible range of the
  outcome mean. Sleep study: `normal(250, 100)` ms.
- Slope: think "change in y per unit x"; `normal(0, 20)` ms/day when ±40 is the edge of
  plausibility. Encode direction only when the goal (prediction, decision) warrants it; keep it
  symmetric when the goal is to summarize the data or to detect coding errors.
- Residual sd: `exponential(1/50)` when 50 is a plausible sd (the exponential's sd equals its
  mean, so this is wide).
- rstanarm defaults implement this automatically: `normal(0, 2.5 * sd(y) / sd(x_k))` per
  coefficient, `normal(mean(y), 2.5 * sd(y))` for the intercept, `exponential(1 / sd(y))` for
  sigma. Read them with `prior_summary(fit)`. brms defaults are flat on coefficients and
  `student_t(3, 0, 2.5 * mad(y))`-style on intercept and scales; set coefficient priors
  explicitly.

### Adding hierarchical structure

When you add varying intercepts, the old residual sd splits into within- and between-group
parts. **Halve the prior mean**: `sigma ~ exponential(1/50)` becomes `sigma ~ exponential(1/25)`
and `tau_0 ~ exponential(1/25)`. Varying slopes get their own scale (`tau_1 ~ exponential(1/10)`
for ±10 ms/day). Correlations: `L ~ lkj_corr_cholesky(1)`.

### Moving to a log-scale (lognormal, Poisson, gamma) model

Do **not** reuse raw-scale priors on the log scale; `normal(250, 100)` on a log intercept puts
the prior predictive mean hundreds of orders of magnitude off. Translate by **matching
quantiles**: `normal(250, 100)` has 2.5%/97.5% quantiles 50 and 450; their logs are 3.9 and
6.1, so use `normal(5, 0.55)`. Keep effect priors centered at 0 (`normal(0, 0.2)` on a log
slope means ±40% per unit at the 95% edge). Scale priors shrink accordingly: `exponential(3)`
for sigma, split to `exponential(6)` each when adding varying intercepts, `exponential(10)` for
a varying slope sd. LKJ is scale-free and carries over unchanged.

### Logit-scale models

- `normal(0, 5)` on intercept and slopes rules out probabilities below 0.007 or above 0.993 for
  an average unit; too weak for most problems but safe.
- `normal(0, 2.5)` (rstanarm) is a reasonable default with standardized predictors.
- `normal(0, 50)` looks weak but forces near-step-function predictions: with sparse data the
  posterior concentrates on extreme values.
- A "flat" prior on a logit parameter is `logistic(0, 1)` on the probability; `logistic(0, 2)`
  or wider is **bimodal** on the probability scale. `normal(0, 100)` on a logit with zero
  successes gives an essentially improper posterior (BW §5.6 binomial example).

## Priors for specific structures

| Structure | Recommendation | Notes |
|---|---|---|
| Group-level sd with few groups | `normal(0, 1)` or `exponential(1)` on unit scale; zero-avoiding `lognormal` or `inv_gamma` if funnel persists and you can justify it | say so and run sensitivity if used only for computation |
| Correlation matrix | `lkj_corr_cholesky(eta)`; eta=1 uniform, eta=2 mild shrinkage | marginals concentrate near 0 as dimension grows; check with prior simulation |
| Student-t degrees of freedom | `gamma(2, 0.1)` or `nu ~ exponential(1/30)`; put the scale prior on the actual sd `sigma * sqrt(nu/(nu-2))`, not on `sigma` | BW §5.6 joint-prior tip |
| Negative binomial shape | brms default `inv_gamma(0.4, 0.3)`; small shape = far from Poisson | |
| Bounded parameter (elasticity in (0,1)) | `normal(0.5, 0.5)` soft, **not** `uniform(0, 1)` | hard bounds only for mathematical constraints |
| Mixture component scales | shared or hierarchical `log sigma_k ~ normal(log sigma_0, 1)` | prevents the unbounded-likelihood spike |
| Item-response discrimination | `gamma_k ~ normal(1, sigma_gamma)`, `sigma_gamma ~ exponential(2)` | mean fixed at 1 breaks multiplicative aliasing; allows negative values to detect coding errors |
| Many coefficients, expected sparsity | regularized horseshoe or R2D2 | see [references/regularizing_priors.md](references/regularizing_priors.md) |
| Meta-analysis SNR prior for a new RCT | `0.42 normal(0, 1.5) + 0.58 normal(0, 3.5)` (Cochrane) | BW §5.6 |
| GP length-scale | `inv_gamma` tuned so mass lies between data resolution and data range | |

## Prior predictive checking

Procedure (BW §5.8–5.9): fix unmodeled data (N, x) and unmodeled hyperparameters; draw the
modeled parameters from their priors; simulate y; repeat; plot ~10 simulated datasets next to
the real one. Prefer scatterplots and fitted-line spaghetti; also look at the parameter draws.

In Stan, a prior predictive simulation is the same program with the likelihood switched off:

```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
  int<lower=0, upper=1> prior_only;
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 1.0 / 3);
  sigma ~ exponential(1);
  if (!prior_only) y ~ normal(alpha + beta * x, sigma);
}
generated quantities {
  array[N] real y_rep = normal_rng(alpha + beta * x, sigma);
}
```

```python
prior = model.sample(data={**data, "prior_only": 1}, chains=1, iter_sampling=500, seed=1)
dt_prior = az.from_cmdstanpy(prior=prior, prior_predictive="y_rep", observed_data={"y": data["y"]})
az.plot_ppc_dist(dt_prior, group="prior_predictive", kind="ecdf")
```

```r
prior_fit <- mod$sample(data = c(data, prior_only = 1), chains = 1, iter_sampling = 500)
yrep <- prior_fit$draws("y_rep", format = "draws_matrix")
bayesplot::ppc_dens_overlay(y, yrep[1:20, ])
# brms: brm(..., sample_prior = "only") then pp_check()
```

Red flags: predictions spanning many orders of magnitude; impossible values in more than a
few percent of draws; all simulations identical (dogmatic prior); the temperature regression
with `normal(0, 100)` on slope and intercept implying ±10 000 degrees. When the data clearly
dominate the prior, a strange prior predictive is not a problem; sensitivity analysis will
confirm.

## Tail behavior and prior-likelihood conflict (BW §5.10)

- With `theta ~ normal(0, 1)` and data equally informative at `ybar = 10`, the posterior sits at
  5, contradicting both prior and data, and the computation shows no warning. **Plot prior,
  likelihood, and posterior together**; the posterior alone hides conflict.
- A heavy-tailed prior (`cauchy(0, 1)`, `student_t(3, 0, 1)`) lets the data win when they
  disagree strongly; shrinkage goes to zero as the conflict grows. Cost: some efficiency, and
  Cauchy tails on unbounded parameters slow HMC.
- If you trust the prior and suspect the data, add a bias term instead:
  `y ~ normal(theta + bias, sigma); bias ~ cauchy(0, 0.1);`. The posterior for theta is then
  non-monotone in the data mean, which is the intended behavior.
- Cauchy priors on `<lower=0>` scale parameters are sampled on the log scale and are usually
  fine; on unbounded parameters they produce varying curvature and poor mixing.

## Sensitivity analysis: power-scaling

Power-scaling multiplies the prior or the likelihood by `alpha` in {0.8, 1, 1.25} and
reweights the existing draws by importance sampling; no refits. Reliable when the Pareto k of
the weights is below 0.7.

```r
library(priorsense)
powerscale_sensitivity(fit)                    # brms/rstanarm/cmdstanr fit
powerscale_plot_dens(fit, variable = c("b_Days", "sigma"))
powerscale_sensitivity(fit, prediction = \(x, ...) ratio_of_interest)  # on a derived quantity
```

```python
az.psense_summary(dt)          # needs log_prior and log_likelihood groups
az.plot_psense_dist(dt, var_names=["beta", "sigma"])
```

For CmdStanPy/cmdstanr programs, expose `lprior` (`target += ...` terms accumulated into a
`real lprior` in `transformed parameters` or `generated quantities`) and `log_lik` so the tools
can find them; brms ≥ 2.20 writes `lprior` automatically.

| Diagnosis | Meaning | Response |
|---|---|---|
| sensitive to prior **and** likelihood | prior-likelihood conflict | think about the prior and the data model; not "loosen until quiet" |
| sensitive to prior, not likelihood | weak likelihood | prior is doing the work; report it, consider more data |
| sensitive to likelihood only | informative data | fine |
| neither | prior irrelevant here | can skip prior predictive refinements |

For complex models with correlated parameters (splines), analyze the sensitivity of the
**quantity of interest** (treatment effect ratio, prediction) rather than every coefficient.

## Priors are joint objects

- Independent weak priors on many coefficients are a strong prior on R² near 1 (BW Fig. 8.13).
  Scale each coefficient prior by the number of predictors, or put the prior on R² (R2D2).
- Changing `normal(mu, sigma)` to `student_t(nu, mu, sigma)` changes the implied sd to
  `sigma * sqrt(nu/(nu-2))`; the prior on `sigma` needs revisiting.
- Priors that depend on the data (`normal(0, 0.05 * sd(y))`) are shortcuts that break
  generativity. Test: can you draw theta from the prior without seeing y? If not, make the scale
  a parameter with its own prior or accept the approximation knowingly.
- Reparameterize so that independence is plausible (proportions of body mass instead of
  organ volumes; softmax to remove sum-to-one dependence).

## References

- [references/prior_recommendations.md](references/prior_recommendations.md): a table of
  defaults by parameter type and interface, including rstanarm autoscaling and brms `set_prior`
  syntax.
- [references/prior_predictive.md](references/prior_predictive.md): the five-variable
  classification, the simulation procedure, worked logistic/GP/linear examples.
- [references/regularizing_priors.md](references/regularizing_priors.md): regularized
  horseshoe and R2D2 in Stan, when to use each, sampler settings.
- [references/sensitivity.md](references/sensitivity.md): power-scaling mechanics,
  `lprior` in Stan programs, static sensitivity plots, refitting with alternative priors.
