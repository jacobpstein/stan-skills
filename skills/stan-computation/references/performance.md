# Performance Tuning

Source: Stan User's Guide (Efficiency Tuning); *Bayesian Workflow* §12.4.

Order of operations: make the model right, then make it fast. Most "slow model" problems are
model problems (see [failure_modes.md](failure_modes.md)). When the model is right and it is
still slow, work through this list.

## Measure first

```stan
model {
  profile("priors") {
    beta ~ normal(0, 1);
    sigma ~ exponential(1);
  }
  profile("likelihood") {
    y ~ normal_id_glm(X, alpha, beta, sigma);
  }
}
```

```python
fit = model.sample(data=data, save_profile=True)
print(fit.profiles())      # per-block forward/reverse autodiff time and node counts
```

```r
fit <- mod$sample(data = data, ...)
fit$profiles()
```

The profile output tells you which block dominates and whether the cost is in the forward pass or
the autodiff tape. Optimizing anything else is wasted effort.

## Vectorize

Vectorized distribution statements build a much smaller autodiff graph than loops.

```stan
y ~ normal(mu, sigma);                           // one node set for all N
for (n in 1:N) y[n] ~ normal(mu[n], sigma);      // N node sets
```

Vectorize arithmetic too: `X * beta` rather than a loop of `dot_product`; `log1m_exp`, `log_sum_exp`,
`log_inv_logit` instead of composing primitives.

## Use GLM functions

These have hand-written analytic gradients and are typically 2–5× faster than the equivalent
composition:

```stan
y ~ normal_id_glm(X, alpha, beta, sigma);
y ~ bernoulli_logit_glm(X, alpha, beta);
y ~ poisson_log_glm(X, log_exposure + alpha, beta);
y ~ neg_binomial_2_log_glm(X, alpha, beta, phi);
y ~ ordered_logistic_glm(X, beta, cutpoints);
```

## Move work out of the inner loop

- Constants and data transformations belong in `transformed data` (computed once).
- Quantities you only need for reporting belong in `generated quantities` (computed once per saved
  draw), not `transformed parameters` (computed every leapfrog step **and** stored).
- Use local blocks inside `model { { ... } }` for temporaries you do not want written to the CSV.
- Precompute Cholesky factors, log-offsets, and index arrays in `transformed data`.

## Sufficient statistics

For repeated identical observations, replace N likelihood evaluations with one weighted term:

```stan
// instead of bernoulli_logit for each of n_trials rows with the same predictor
successes ~ binomial_logit(trials, eta);
// or a weighted log-likelihood
target += counts[g] * normal_lpdf(value[g] | mu, sigma);
```

## Within-chain parallelism: reduce_sum

```stan
functions {
  real partial_sum_lpmf(array[] int y_slice, int start, int end,
                        matrix X, real alpha, vector beta) {
    return bernoulli_logit_glm_lupmf(y_slice | X[start:end], alpha, beta);
  }
}
model {
  target += reduce_sum(partial_sum_lpmf, y, grainsize, X, alpha, beta);
}
```

```python
model = CmdStanModel(stan_file="m.stan", cpp_options={"STAN_THREADS": True})
fit = model.sample(data={**data, "grainsize": 1}, chains=4, parallel_chains=4,
                   threads_per_chain=2)
```

```r
mod <- cmdstan_model("m.stan", cpp_options = list(stan_threads = TRUE))
fit <- mod$sample(data = data, chains = 4, parallel_chains = 4, threads_per_chain = 2)
```

Worth it when the likelihood dominates the gradient cost and N is large. Start with
`grainsize = 1` and let the scheduler choose. Total cores used is `parallel_chains ×
threads_per_chain`; between-chain parallelism is more efficient per core, so only add threads
after all chains are already parallel.

`map_rect` is the older, more manual alternative; prefer `reduce_sum` in new code.

## GPU

```python
model = CmdStanModel(stan_file="m.stan",
                     cpp_options={"STAN_OPENCL": True, "OPENCL_DEVICE_ID": 0, "OPENCL_PLATFORM_ID": 0})
```

Helps for large matrix operations and the GLM likelihoods with big design matrices. Not useful for
small models.

## Compiler optimization

```python
model = CmdStanModel(stan_file="m.stan", stanc_options={"O1": True},
                     cpp_options={"STAN_CPP_OPTIMS": True})
```

`--O1` applies safe stanc3 optimizations (lazy code motion, dead code elimination). Compilation
gets slower; sampling gets faster. Verify results are unchanged when you turn it on.

## Expensive special cases

| Cost source | Mitigation |
|---|---|
| Exact GP, O(N³) per evaluation | Hilbert-space basis approximation (HSGP), inducing points, or splines |
| ODE solvers | use `ode_rk45` for non-stiff and `ode_bdf` for stiff; loosen tolerances during exploration; analytic solutions where they exist; `ode_adjoint_tol_ctl` for many parameters |
| `integrate_1d` | put it in `generated quantities` if it is only needed for `log_lik` (once per draw instead of once per leapfrog step) |
| Large `transformed parameters` | move to `generated quantities`, or compute in a local block |
| Matrix inverse | use `mdivide_left_spd`, `\`, or a Cholesky solve instead of `inverse()` |
| Repeated `cholesky_decompose` of a data-only matrix | do it in `transformed data` |

## Sampling settings that are not speed fixes

Raising `adapt_delta` makes sampling slower and only helps with a small number of divergences
caused by mild curvature. Raising `max_treedepth` makes sampling slower and usually hides a
non-identification problem. Reducing warmup below 1000 typically costs more in sampling
efficiency than it saves. Reducing `iter_sampling` is legitimate during exploration: 200 draws
is enough to see most failures.

## Rough targets

For a model you will iterate on, aim for a fit in under a minute during exploration. If a single
fit takes more than about 15 minutes, work on a subset or a simplified model until the structure
is settled, then run the full model once for the final results.
