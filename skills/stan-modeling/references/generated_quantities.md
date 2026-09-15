# Generated Quantities: log_lik, Replications, and Predictions

Everything you will report or check comes out of `generated quantities`. It runs once per saved
draw, so it is cheap relative to the model block, and `_rng` functions are allowed.

## The standard trio

```stan
generated quantities {
  vector[N] log_lik;                        // for PSIS-LOO and model comparison
  array[N] real y_rep;                      // for posterior predictive checks
  vector[N_new] y_pred;                     // for predictions at new predictor values
  for (n in 1:N) log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  y_rep = normal_rng(mu, sigma);
  y_pred = to_vector(normal_rng(alpha + X_new * beta, sigma));
}
```

Rules:

- `log_lik[n]` must be the **normalized** log density (`_lpdf`, not `_lupdf`) of one exchangeable
  unit: the unit you would leave out in cross-validation.
- `y_rep` must follow the **generative data model**, not the marginalized likelihood you wrote in
  the model block. For a zero-inflated Poisson the model block uses `log_sum_exp`; the rng draws
  a Bernoulli then a Poisson.
- Vectorized `_rng` functions return an `array[] real`; assign to `array[N] real` or wrap with
  `to_vector`.

## Choosing the leave-one-out unit

| Goal | `log_lik` unit |
|---|---|
| Predict a new observation in an existing group | one observation |
| Predict a whole new group | sum of the group's observation log-densities |
| Forecast the next time point | one time point (with leave-future-out folds) |

```stan
// leave-one-group-out
generated quantities {
  vector[J] log_lik_group = rep_vector(0, J);
  for (n in 1:N)
    log_lik_group[group[n]] += normal_lpdf(y[n] | mu[n], sigma);
}
```

## Predictions for new data

Pass the new design matrix in the `data` block so predictions are computed inside Stan and
propagate all the uncertainty:

```stan
data {
  int<lower=0> N_new;
  matrix[N_new, K] X_new;
  array[N_new] int<lower=1, upper=J> group_new;   // existing groups
}
generated quantities {
  vector[N_new] mu_new = alpha[group_new] + X_new * beta;
  array[N_new] real y_new = normal_rng(mu_new, sigma);     // includes residual noise
}
```

`mu_new` is the expectation (the analog of `posterior_epred`); `y_new` is the predictive draw
(the analog of `posterior_predict`). Uncertainty in the fitted line is much narrower than
predictive uncertainty; report whichever answers the question.

### New groups

Draw a fresh group effect rather than reusing a fitted one:

```stan
generated quantities {
  array[N_new] real y_newgroup;
  for (n in 1:N_new) {
    real alpha_new = normal_rng(mu_alpha, sigma_alpha);
    y_newgroup[n] = normal_rng(alpha_new + X_new[n] * beta, sigma);
  }
}
```

## Counterfactuals and contrasts

Compute the comparison inside Stan so the posterior of the contrast is available directly:

```stan
generated quantities {
  vector[N] mu_treated   = alpha + beta_t * 1 + X * beta;
  vector[N] mu_untreated = alpha + beta_t * 0 + X * beta;
  real ate = mean(exp(mu_treated) - exp(mu_untreated));   // on the outcome scale
  real ratio = mean(exp(mu_treated)) / mean(exp(mu_untreated));
}
```

This is the right way to summarize an effect that splits across components of a model (a
zero-inflated model's count and zero parts, or an interaction): the coefficients are not
interpretable separately, but the predicted contrast is.

Never compute a function of posterior means. `mean(a) / mean(b)` is not `mean(a / b)`.

## Poststratification

```stan
data {
  int<lower=0> J;                    // cells
  vector[J] N_pop;                   // population counts per cell
  matrix[J, K] X_cell;
}
generated quantities {
  vector[J] theta_cell = inv_logit(alpha[cell_group] + X_cell * beta);
  real theta_pop = dot_product(N_pop, theta_cell) / sum(N_pop);
}
```

## Derived parameters

```stan
generated quantities {
  real ld50 = -alpha / beta;
  corr_matrix[2] Omega = multiply_lower_tri_self_transpose(L);
  real r2 = variance(mu) / (variance(mu) + square(sigma));    // Bayesian R-squared per draw
  vector[K] beta_orig = beta ./ x_sd;                          // back to original units
  real alpha_orig = alpha_c - dot_product(x_mean ./ x_sd, beta);
}
```

## Standalone generated quantities

Run `generated quantities` against an existing fit without resampling. Useful for adding
predictions, log-likelihoods you forgot, or new counterfactuals to a fit that took hours.

```stan
// m_gq.stan: same data and parameters blocks, only generated quantities differ
```

```python
gq = model_gq.generate_quantities(previous_fit=fit, data=data)
y_new = gq.stan_variable("y_new")
```

```r
gq <- mod_gq$generate_quantities(fitted_params = fit, data = data)
gq$draws("y_new")
```

The `parameters` block of the GQ program must match the original exactly.

## Wiring into ArviZ and posterior

```python
dt = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive="y_rep",
    predictions="y_new",
    log_likelihood="log_lik",
    observed_data={"y": data["y"]},
    constant_data={"x": data["x"]},
    coords={"obs": range(N), "new": range(N_new), "pred": names},
    dims={"y": ["obs"], "y_rep": ["obs"], "log_lik": ["obs"],
          "y_new": ["new"], "beta": ["pred"]},
)
```

```r
draws <- fit$draws(c("y_rep", "log_lik"), format = "draws_matrix")
loo1  <- fit$loo(variables = "log_lik", r_eff = TRUE, save_psis = TRUE)
bayesplot::ppc_dens_overlay(y, draws[1:50, grep("^y_rep", colnames(draws))])
```

## Cost control

`generated quantities` runs once per saved draw, not per leapfrog step, so quadrature, ODE
solves, and large matrix operations that would be prohibitive in the model block are affordable
here. That is exactly why integrated LOO (`integrate_1d` per observation) is placed here rather
than in the model block.

But the output is written to CSV: `N_new` × draws values per variable. For large N, thin the
draws or compute the summaries you need (means, quantiles) inside Stan instead of exporting
every replicate.
