---
name: stan-modeling
description: >
  Load whenever the user is writing or editing a .stan program or fitting one from Python
  (CmdStanPy) or R (cmdstanr): program blocks, current Stan syntax and types, constraints,
  vectorization and GLM functions, non-centered parameterization, generated quantities for
  predictions and log_lik, and the standard model patterns (linear and logistic regression,
  hierarchical/multilevel, GLMs with offsets, ordinal, censored and truncated, mixtures,
  measurement error, Gaussian processes, splines, time series, ODEs). Also covers moving draws
  into ArviZ or posterior for diagnostics. Triggers include: stan, .stan file, cmdstanpy,
  cmdstanr, stanc, target +=, generated quantities, transformed parameters, simplex, ordered,
  cholesky_factor_corr, reduce_sum, log_lik, y_rep, hierarchical model, varying intercept,
  varying slope, multilevel, non-centered.
---

# Writing and Fitting Stan Programs

A Stan program is a joint model, not a fitting procedure. Write the generative story first,
then translate it. Every parameter needs a proper prior; every quantity you will report needs
to come out of `generated quantities`.

Companion skills: [bayesian-workflow](../bayesian-workflow/SKILL.md) for the process,
[stan-priors](../stan-priors/SKILL.md) for prior choice,
[stan-computation](../stan-computation/SKILL.md) when sampling misbehaves,
[stan-model-evaluation](../stan-model-evaluation/SKILL.md) for checks and comparison,
[r-stan-interfaces](../r-stan-interfaces/SKILL.md) for cmdstanr, rstanarm, and brms.

## Program skeleton

```stan
functions {
  // user-defined functions; _lpdf/_lpmf/_rng/_lp suffixes have special meaning
}
data {
  int<lower=0> N;                    // bounds document and validate inputs
  int<lower=1> K;
  matrix[N, K] X;
  array[N] int<lower=0, upper=1> y;  // modern array syntax
  int<lower=0, upper=1> prior_only;  // switch for prior predictive runs
}
transformed data {
  matrix[N, K] Xc;                   // centering, one-time computation
  vector[K] x_mean;
  for (k in 1:K) { x_mean[k] = mean(X[, k]); Xc[, k] = X[, k] - x_mean[k]; }
}
parameters {
  real alpha_c;
  vector[K] beta;
}
transformed parameters {
  vector[N] eta = alpha_c + Xc * beta;
  real lprior = normal_lpdf(alpha_c | 0, 2.5) + normal_lpdf(beta | 0, 1);
}
model {
  target += lprior;
  if (!prior_only) y ~ bernoulli_logit(eta);
}
generated quantities {
  real alpha = alpha_c - dot_product(x_mean, beta);   // intercept on the original scale
  vector[N] log_lik;
  array[N] int y_rep = bernoulli_logit_rng(eta);
  for (n in 1:N) log_lik[n] = bernoulli_logit_lpmf(y[n] | eta[n]);
}
```

Block rules:
- `transformed data` runs once; `transformed parameters` runs every leapfrog step and its values
  are **saved** in the output; local variables inside `model { { ... } }` are not saved.
- `generated quantities` runs once per saved draw. `_rng` functions are allowed only here (and
  in `transformed data`). Never touch `target` here.
- Statements in `model` accumulate onto `target`; two `~` statements for the same parameter
  multiply their densities (`theta ~ normal(0,1); theta ~ normal(0,1);` is `normal(0, 1/sqrt(2))`).

## Current syntax (Stan 2.33+)

| Do | Not |
|---|---|
| `array[N] int y;` `array[N, K] real z;` | `int y[N];` (removed) |
| `array[N] vector[K] v;` | `vector[K] v[N];` (removed) |
| `target += normal_lupdf(y \| mu, sigma);` | `increment_log_prob(...)` (removed) |
| `x = y;` | `x <- y;` (removed) |
| `a ? b : c` | `if_else(a, b, c)` (removed) |
| `mod$format(canonicalize = TRUE)` / `stanc --print-canonical` to upgrade old code | hand-editing large legacy programs |

Use `~` for readability; use `target += ..._lupdf(...)` when you need the unnormalized version
for speed, or `..._lpdf` when the normalizing constant matters (comparing targets, computing
`log_lik`, user-defined densities with parameters in the constants). In `generated quantities`
always use the fully normalized `_lpdf`/`_lpmf` for `log_lik`.

Compile with checks on:

```python
model = CmdStanModel(stan_file="m.stan", stanc_options={"warn-pedantic": True, "O1": True})
```

```r
mod <- cmdstan_model("m.stan", stanc_options = list("warn-pedantic" = TRUE, "O1" = TRUE))
```

Pedantic mode catches parameters with no prior, missing `<lower=0>` on scale parameters,
unused parameters, and suspicious distribution arguments. Treat its warnings as a checklist.

## Types and constraints

| Type | Use |
|---|---|
| `real<lower=0> sigma;` | scale parameters, always constrained |
| `real<lower=0, upper=1> theta;` | probabilities when not modeled on the logit scale |
| `vector[K]`, `row_vector[K]`, `matrix[N, K]` | linear algebra; prefer over arrays of reals |
| `array[N] int` | integer data and counts |
| `simplex[K]` | mixture weights, category probabilities |
| `ordered[K]`, `positive_ordered[K]` | cutpoints, ordered means in mixtures |
| `cholesky_factor_corr[K]` | correlation matrices (use with `lkj_corr_cholesky`) |
| `cholesky_factor_cov[K]` | covariance matrices |
| `sum_to_zero_vector[K]` | ANOVA-style effects, ICAR components (avoids a soft constraint) |
| `vector<lower=0>[K]`, `vector<offset=mu, multiplier=sigma>[K]` | elementwise constraints and affine reparameterization |
| `tuple(real, vector[N])` | multiple return values from a function |

Constraints are transformations with Jacobian adjustments handled by Stan. Adding a constraint
is not the same as adding a prior: `real<lower=0> b;` with `b ~ normal(0, 5);` is a half-normal.

## Vectorization and speed

Vectorized statements are dramatically faster than loops because the autodiff graph is smaller:

```stan
y ~ normal(alpha + X * beta, sigma);           // good
for (n in 1:N) y[n] ~ normal(alpha + X[n] * beta, sigma);   // slow
```

Use the GLM functions when they apply; they have analytic gradients:

| Model | Function |
|---|---|
| Linear regression | `normal_id_glm(X, alpha, beta, sigma)` |
| Logistic | `bernoulli_logit_glm(X, alpha, beta)` |
| Poisson (log link) | `poisson_log_glm(X, alpha, beta)` |
| Negative binomial | `neg_binomial_2_log_glm(X, alpha, beta, phi)` |
| Ordered logistic | `ordered_logistic_glm(X, beta, cutpoints)` |

Offsets go into the intercept argument as a vector: `poisson_log_glm(X, log_exposure + alpha, beta)`.

Other speed rules: hoist loop-invariant computations into `transformed data`; avoid storing huge
`transformed parameters` you do not need (put them in `generated quantities` or a local block);
use `profile("name") { ... }` blocks to find the expensive part.

## Model patterns

### Hierarchical / multilevel

```stan
data {
  int<lower=0> N, J;
  array[N] int<lower=1, upper=J> group;
  vector[N] x, y;
}
parameters {
  real mu_alpha, beta;
  real<lower=0> sigma_alpha, sigma;
  vector<offset=mu_alpha, multiplier=sigma_alpha>[J] alpha;   // non-centered, written centered
}
model {
  mu_alpha ~ normal(0, 5);
  sigma_alpha ~ exponential(1);
  sigma ~ exponential(1);
  beta ~ normal(0, 1);
  alpha ~ normal(mu_alpha, sigma_alpha);
  y ~ normal(alpha[group] + beta * x, sigma);
}
```

`<offset=, multiplier=>` gives the non-centered geometry while keeping the readable centered
statement. Use it when groups have little data; drop it (plain centered) when each group has many
observations. The explicit form is `alpha = mu_alpha + sigma_alpha * z; z ~ std_normal();`.

### Varying intercepts and slopes with correlation

```stan
parameters {
  vector[2] gamma;                       // population intercept and slope
  vector<lower=0>[2] tau;
  cholesky_factor_corr[2] L;
  matrix[2, J] z;
}
transformed parameters {
  matrix[J, 2] b = (diag_pre_multiply(tau, L) * z)';   // group deviations
}
model {
  to_vector(z) ~ std_normal();
  L ~ lkj_corr_cholesky(2);
  tau ~ exponential(1);
  gamma ~ normal(0, 5);
  {
    vector[N] mu;
    for (n in 1:N) mu[n] = gamma[1] + b[group[n], 1] + (gamma[2] + b[group[n], 2]) * x[n];
    y ~ normal(mu, sigma);
  }
}
generated quantities {
  matrix[2, 2] Omega = multiply_lower_tri_self_transpose(L);
}
```

### Counts with exposure

```stan
y ~ poisson_log_glm(X, log_exposure + alpha, beta);          // Poisson
y ~ neg_binomial_2_log_glm(X, log_exposure + alpha, beta, phi);  // overdispersed
```

Start with negative binomial for counts; check `phi` (small `phi` means far from Poisson). Add
zero inflation only if a rootogram or a calibration plot of `Pr(y > 0)` says so.

### Zero-inflated and hurdle

```stan
model {
  for (n in 1:N) {
    if (y[n] == 0)
      target += log_sum_exp(bernoulli_logit_lpmf(1 | zi[n]),
                            bernoulli_logit_lpmf(0 | zi[n]) + poisson_log_lpmf(0 | eta[n]));
    else
      target += bernoulli_logit_lpmf(0 | zi[n]) + poisson_log_lpmf(y[n] | eta[n]);
  }
}
generated quantities {
  array[N] int y_rep;
  for (n in 1:N)
    y_rep[n] = bernoulli_logit_rng(zi[n]) ? 0 : poisson_log_rng(eta[n]);
}
```

The `_rng` must follow the **generative** story even though the model block uses the marginalized
likelihood.

### Ordinal

```stan
parameters {
  vector[K] beta;               // no intercept: it is absorbed by the cutpoints
  ordered[C - 1] cutpoints;
}
model {
  cutpoints ~ student_t(3, 0, 2.5);
  beta ~ normal(0, 1);
  y ~ ordered_logistic(X * beta, cutpoints);
}
```

### Censored and truncated

```stan
// right-censored at U: observed y[1:N], censored count N_cens all at U
y ~ normal(mu, sigma);
target += N_cens * normal_lccdf(U | mu, sigma);

// truncated sampling (only values above L could be observed)
for (n in 1:N) {
  y[n] ~ normal(mu, sigma) T[L, ];       // equivalent to subtracting normal_lccdf(L | mu, sigma)
}
```

Censoring and truncation are different: censoring means the value exists but is only bounded;
truncation means such units never enter the sample.

### Mixtures

```stan
parameters {
  simplex[K] lambda;
  ordered[K] mu;                  // ordering breaks label switching
  vector<lower=0>[K] sigma;
}
model {
  mu ~ normal(0, 5);
  sigma ~ lognormal(0, 0.5);      // keeps component scales within an order of magnitude
  lambda ~ dirichlet(rep_vector(2, K));
  for (n in 1:N) {
    vector[K] lps = log(lambda);
    for (k in 1:K) lps[k] += normal_lpdf(y[n] | mu[k], sigma[k]);
    target += log_sum_exp(lps);
  }
}
```

The mixture likelihood is **unbounded** (set one mean to a data point and let its scale go to
zero), so component scales need a shared or hierarchical prior. `log_mix(lambda, lp1, lp2)` is
the two-component shortcut.

### Measurement error

```stan
parameters {
  vector[N] x_true;
  real mu_x; real<lower=0> sigma_x;
  real alpha, beta; real<lower=0> sigma;
}
model {
  x_true ~ normal(mu_x, sigma_x);          // population model for the true predictor
  x_obs ~ normal(x_true, sigma_meas);      // measurement model (sigma_meas known or modeled)
  y ~ normal(alpha + beta * x_true, sigma);
}
```

Error in x attenuates the slope if ignored; error in y does not (it merges with the residual).

### Gaussian processes

```stan
transformed data { real delta = 1e-9; }
parameters { real<lower=0> rho, sigma_f, sigma; vector[N] eta; }
model {
  matrix[N, N] L_K = cholesky_decompose(add_diag(gp_exp_quad_cov(x, sigma_f, rho), delta));
  rho ~ inv_gamma(5, 5);          // tuned so mass avoids lengths below the data spacing
  sigma_f ~ normal(0, 1);
  eta ~ std_normal();
  y ~ normal(L_K * eta, sigma);
}
```

Exact GPs are O(N³); beyond a few hundred points use a Hilbert-space basis-function
approximation (HSGP) or splines. See [references/gp_splines.md](references/gp_splines.md).

### Time series

```stan
// AR(1)
y[2:N] ~ normal(alpha + rho * y[1:(N-1)], sigma);
// Gaussian random walk on a latent state (non-centered)
z ~ std_normal();
state = cumulative_sum(z) * sigma_state;
```

See [references/timeseries.md](references/timeseries.md) for AR/MA, local level and trend,
seasonal terms, and forecasting in `generated quantities`.

## Fitting from Python

```python
from cmdstanpy import CmdStanModel, write_stan_json
import arviz as az

model = CmdStanModel(stan_file="m.stan", stanc_options={"warn-pedantic": True})
fit = model.sample(data=data, chains=4, parallel_chains=4,
                   iter_warmup=1000, iter_sampling=1000, seed=1, inits=0.1)
print(fit.diagnose())
fit.save_csvfiles("results/m_v1")

dt = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive="y_rep",
    log_likelihood="log_lik",
    observed_data={"y": data["y"]},
    coords={"obs": range(data["N"]), "pred": predictor_names},
    dims={"y": ["obs"], "y_rep": ["obs"], "log_lik": ["obs"], "beta": ["pred"]},
)
az.summary(dt, var_names=["alpha", "beta", "sigma"])
```

`fit.draws_pd()` gives a DataFrame, `fit.draws_xr()` an xarray Dataset, `fit.stan_variable("beta")`
a numpy array shaped (draws, ...). Pass data as a dict of numpy arrays or a JSON file written with
`write_stan_json`.

## Fitting from R

```r
library(cmdstanr); library(posterior)
mod <- cmdstan_model("m.stan", stanc_options = list("warn-pedantic" = TRUE))
fit <- mod$sample(data = data, chains = 4, parallel_chains = 4, seed = 1, init = 0.1)
fit$diagnostic_summary()
fit$summary(c("alpha", "beta", "sigma"))
draws <- fit$draws(format = "draws_df")
fit$save_object("results/m_v1.rds")
```

## Checklist before you sample

1. Every parameter has a proper prior (`--warn-pedantic` agrees).
2. Scale parameters are `<lower=0>`; probabilities are `<lower=0, upper=1>` or on the logit scale.
3. Predictors are centered and roughly unit scale (in `transformed data`).
4. Data bounds in the `data` block encode what the inputs must satisfy.
5. `log_lik` and `y_rep` exist in `generated quantities` if you will check or compare the model.
6. A `prior_only` switch (or a separate prior predictive program) exists.
7. Non-centered parameterization where groups are small.
8. Run `model.optimize()` once as a smoke test before the first `sample()`.

## References

- [references/syntax.md](references/syntax.md): blocks, types, control flow, functions, common
  compile errors and what they mean, canonicalizing legacy code.
- [references/regression_patterns.md](references/regression_patterns.md): linear, logistic,
  Poisson/NB, ordinal, robust, multivariate outcomes, QR, splines-as-design-matrix, interactions.
- [references/hierarchical.md](references/hierarchical.md): pooling levels, non-centered forms,
  correlated slopes, crossed and nested groupings, group-level predictors, sum-to-zero.
- [references/gp_splines.md](references/gp_splines.md): exact GP, HSGP basis functions,
  penalized splines, choosing length-scale priors.
- [references/timeseries.md](references/timeseries.md): AR/MA, state space, seasonality,
  forecasting, and leave-future-out setup.
- [references/generated_quantities.md](references/generated_quantities.md): log_lik, y_rep,
  predictions for new data, counterfactuals, standalone `generate_quantities`.
