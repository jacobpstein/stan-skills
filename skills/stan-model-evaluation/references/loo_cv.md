# PSIS-LOO Cross-Validation in Depth

Source: *Bayesian Workflow* §8.3, §9.4, ch. 24; Vehtari, Gelman & Gabry (2017); Paananen et al.
(2021); Sivula et al. (2025).

## Definitions

- `p(y_i | y_{-i}) = ∫ p(y_i | θ) p(θ | y_{-i}) dθ` (leave-one-out predictive density).
- `elpd_loo = Σ_i log p(y_i | y_{-i})`; higher is better. It is on the log-score scale, so
  differences between models are what matter, not absolute values.
- `p_loo = lpd − elpd_loo` (in-sample minus LOO log predictive density) is the effective number
  of parameters. Compare it to the actual parameter count:
  - `p_loo ≫ number of parameters` → misspecification (the roaches Poisson model: p_loo 273 with
    4 parameters).
  - `p_loo` large relative to N → very flexible model; expect PSIS trouble and uninformative PPCs.
  - `p_loo` ≈ parameter count → healthy sign.

## PSIS mechanics and Pareto k

The LOO posterior is approximated by importance weighting the full posterior draws with
`r_i^s = 1 / p(y_i | θ^s)`. The right tail of the weights is fitted with a generalized Pareto
distribution; its shape parameter k per observation is the reliability diagnostic.

Threshold in the `loo` package: `min(1 − 1/log10(S), 0.7)`; with S = 4000 that is 0.7. In ArviZ
1.x the same rule applies via `loo1.pareto_k` and `az.plot_khat`.

| k | Interpretation |
|---|---|
| ≤ threshold | reliable |
| (0.7, 1] | unreliable estimate for that point; variance of the weights may be infinite |
| > 1 | PSIS fails for that point |

The MCSE of `elpd_loo` is reported as `NA` when any k > 0.7 because it cannot be computed.

### Relative efficiency

Pass the relative ESS of the likelihood draws so the Monte Carlo SE accounts for autocorrelation:

```r
ll <- fit$draws("log_lik", format = "draws_matrix")
r_eff <- loo::relative_eff(exp(ll), chain_id = rep(1:fit$num_chains(), each = fit$metadata()$iter_sampling))
loo1 <- loo::loo(ll, r_eff = r_eff, save_psis = TRUE)
# equivalent shortcut
loo1 <- fit$loo(variables = "log_lik", r_eff = TRUE, save_psis = TRUE)
```

ArviZ computes `reff` from the chain dimension automatically when the DataTree has chains.

## Repair ladder for high k

1. **Moment matching** (Paananen et al. 2021): affine transformations of the posterior draws
   matched to the LOO posterior's mean and covariance, applied only to high-k observations.
   Needs access to the log-density of the model at transformed draws.
   - brms: `loo(fit, moment_match = TRUE)` or `add_criterion(fit, "loo", moment_match = TRUE)`.
     Models with varying coefficients need `save_pars = save_pars(all = TRUE)` at fit time.
   - cmdstanr: `loo_moment_match(x, loo1, post_draws, log_lik_i, unconstrain_pars, log_prob_upars,
     log_lik_i_upars)`: you supply functions that use `fit$init_model_methods()`,
     `fit$unconstrain_draws()`, `fit$log_prob()`.
   - ArviZ: `az.loo_moment_match(dt, loo_orig, log_prob_upars_fn, log_lik_i_upars_fn, ...)`.
   Moment matching fixed all high k for the roaches Poisson and NB models; for the per-observation
   varying-intercept model it reduced 204 bad k to 46, which is still too many.
2. **Exact refits for the bad folds** (`reloo`): brms `add_criterion(fit, "loo", reloo = TRUE)`;
   ArviZ `az.reloo(wrapper, loo_orig)` with a `SamplingWrapper` subclass. Fine for a handful of
   folds; 46 refits is not.
3. **K-fold**: `kfold(fit, K = 10)` (brms/rstanarm), `az.loo_kfold(dt, wrapper, k=10)`. Results
   from K-fold can be mixed with PSIS-LOO in `loo_compare`.
4. **Integrated LOO** for models with one group-level parameter per few observations: see below.

Do not skip the repair when comparing flexible models. In BW ch. 24, ignoring the warnings gave
`elpd_diff = −275 (18.5)` for a true difference of `−0.7 (7.3)`; WAIC was worse (−311).
A large difference (thousands) survives imperfect computation; a small one does not.

## Integrated (marginalized) LOO for varying intercepts

With one latent `z_i` per observation, removing `y_i` changes `p(z_i | ·)` so drastically that
importance sampling fails. Integrate `z_i` out of the pointwise likelihood by adaptive
quadrature inside `generated quantities`. This does not change the posterior; it only changes the
`log_lik` used by PSIS. Pattern from BW ch. 24:

```stan
functions {
  real integrand(real z, real notused, array[] real theta,
                 array[] real X_i, array[] int y_i) {
    real sigmaz = theta[1];
    real mu_i = theta[2];
    real p = exp(normal_lpdf(z | 0, sigmaz) + poisson_log_lpmf(y_i | z + mu_i));
    return (is_inf(p) || is_nan(p)) ? 0 : p;
  }
}
data {
  int<lower=0> N;
  int<lower=0> P;
  matrix[N, P] X;
  array[N] int<lower=0> y;
  vector[N] offsett;          // "offset" is reserved
  real integrate_1d_reltol;   // e.g. 1e-6; increase if you see error-estimate messages
}
parameters {
  real alpha;
  vector[P] beta;
  vector[N] z;
  real<lower=0> sigmaz;
}
model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  z ~ normal(0, sigmaz);
  sigmaz ~ normal(0, 1);
  y ~ poisson_log_glm(X, z + offsett + alpha, beta);
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_loorep;
  for (i in 1:N) {
    real mu_i = offsett[i] + alpha + X[i, ] * beta;
    log_lik[i] = log(integrate_1d(integrand, negative_infinity(), positive_infinity(),
                                  append_array({sigmaz}, {mu_i}), {0}, {y[i]},
                                  integrate_1d_reltol));
    // LOO predictive replicate: draw a fresh z, not the fitted z[i]
    y_loorep[i] = poisson_log_rng(normal_rng(0, sigmaz) + mu_i);
  }
}
```

Use `y_loorep` (not `y_rep`) together with the integrated `log_lik` for LOO-PIT and LOO
reliability diagrams:

```r
loo_int <- fit$loo(save_psis = TRUE)
ppc_loo_pit_ecdf(y = y, yrep = fit$draws("y_loorep", format = "matrix"),
                 psis_object = loo_int$psis_object, method = "correlated")
```

Result in BW: all k < 0.7, p_loo 4.7, elpd within 0.7 of the 10-fold estimate. For two group
parameters use nested quadrature; beyond that, use K-fold.

## Grouped, temporal, and large data

- **Leave-one-group-out** when the goal is prediction for new groups:
  `kfold(fit, folds = loo::kfold_split_grouped(K = J, x = group))` or compute `log_lik` per
  group in Stan (sum over the group's observations) and run PSIS on that. Importance sampling is
  much harder for whole groups; expect refits.
- **Leave-future-out** for forecasting; for *comparing* time-series models, h-block CV with the
  joint log score is more efficient (Cooper et al. 2025). Both need refits; the `loo` package has
  `loo_approximate_posterior` and `psis` helpers for approximate LFO.
- **Large N**: `loo_subsample(ll_fn, draws, data, observations = 400)` (loo) or
  `az.loo_subsample(dt, observations=400)`.

## LOO-based summaries

- `E_loo(x, psis_object)` (loo) and `az.loo_expectations(dt, kind="mean")`: LOO-weighted
  expectations of any function of `y_rep`, e.g. `Pr(y > 0)` for reliability diagrams.
- `loo_predictive_metric`, `az.loo_metrics(dt, kind="rmse"|"mae"|"acc")`.
- `az.loo_score(dt, score_func="crps")`.
- `loo_R2` (rstanarm/brms) and `az.loo_r2(dt, var_name="y")`.
- `az.loo_influence(dt)`: per-observation influence from the importance weights.

## Python: computing LOO without ArviZ

If `log_lik` is only available as a CmdStanPy variable:

```python
import arviz as az
dt = az.from_cmdstanpy(posterior=fit, log_likelihood="log_lik", observed_data={"y": y})
loo1 = az.loo(dt, pointwise=True)
bad = (loo1.pareto_k > 0.7).sum().item()
```

Always build the DataTree with the `log_likelihood=` argument rather than leaving `log_lik` in
the posterior group; `az.loo` looks in the `log_likelihood` group.

## PAV-adjusted reliability diagram for a binary event

For a discrete outcome, calibration of a specific event (`Pr(y > 0)`, `Pr(y = k)`) is checked
with a pool-adjacent-violators reliability diagram on **LOO** predicted probabilities. In R this
is the `reliabilitydiag` package fed by `E_loo()` (BW ch. 24):

```r
library(reliabilitydiag)
p_loo_event <- E_loo((posterior_predict(fit) > 0) + 0, loo1$psis_object)$value
rd <- reliabilitydiag(EMOS = p_loo_event, y = as.numeric(y > 0))
autoplot(rd) + labs(x = "Predicted probability of non-zero",
                    y = "Conditional event probability")
```

The consistency band comes from the model's own replicates; a curve outside it is miscalibration
for that event. In the roaches example this is what motivated zero inflation, when the rootogram
and LOO-PIT had both looked acceptable.

In Python, `az.plot_ppc_pava(dt, data_type="binary")` draws the equivalent PAV calibration curve
with credible bands directly from the DataTree.
