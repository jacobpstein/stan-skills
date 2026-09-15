# Prior and Likelihood Sensitivity

Source: *Bayesian Workflow* §8.5, §17.1, §24.5; Kallioinen, Paananen, Bürkner & Vehtari (2024).

## Why check both

Sensitivity to the prior arises from either a **prior-likelihood conflict** or a **weak
likelihood**. You cannot tell which without also checking sensitivity to the likelihood. Test
both at once with power-scaling.

## Power-scaling by importance sampling

Replace `p(theta) p(y | theta)` with `p(theta)^alpha p(y | theta)` (prior scaling) or
`p(theta) p(y | theta)^alpha` (likelihood scaling), for `alpha` near 1 (e.g. 0.8 and 1.25, or
0.99/1.01 for derivative-based summaries). Reweight the existing posterior draws with
`w^s ∝ p(theta^s)^(alpha − 1)` (or the likelihood analog) and compare the reweighted posterior to
the original using a distance (cumulative Jensen-Shannon). Pareto-smooth the weights; trust the
result when k < 0.7. No refitting.

### Requirements in the Stan program

The tools need the log prior and the pointwise log likelihood as draws:

```stan
transformed parameters {
  real lprior = 0;
  lprior += normal_lpdf(alpha | 0, 5);
  lprior += normal_lpdf(beta | 0, 1);
  lprior += exponential_lpdf(sigma | 1);
}
model {
  target += lprior;
  y ~ normal_id_glm(X, alpha, beta, sigma);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) log_lik[n] = normal_lpdf(y[n] | alpha + X[n] * beta, sigma);
}
```

Truncated priors: `lprior` must include the normalizing constant for truncation when it depends
on parameters (`- normal_lccdf(0 | 0, 1)` for a half-normal is a constant and can be dropped).
brms ≥ 2.20 generates `lprior` automatically.

### R

```r
library(priorsense)
ps <- powerscale_sensitivity(fit)                 # cmdstanr, brms, rstanarm
ps
#  variable   prior likelihood diagnosis
#  b_Days     0.02  0.11       -
powerscale_plot_dens(fit, variable = c("b_Intercept", "b_Days"))
powerscale_plot_ecdf(fit, variable = "sigma")
powerscale_plot_quantities(fit, variable = "b_Days", quantity = c("mean", "sd"))
# sensitivity of a derived quantity instead of raw coefficients
powerscale_sensitivity(fit, prediction = \(x, ...) posterior::rvar(ratio_draws))
```

For raw cmdstanr fits, `priorsense` looks for `lprior` and `log_lik` variables; override with
`log_prior_name =` / `log_lik_name =`.

### Python (ArviZ 1.x)

```python
dt = az.from_cmdstanpy(posterior=fit, log_likelihood="log_lik",
                       observed_data={"y": y})
dt["log_prior"] = dt["posterior"][["lprior"]].ds   # or pass log_prior= if supported by your version
print(az.psense_summary(dt))                      # columns: prior, likelihood, diagnosis
az.plot_psense_dist(dt, var_names=["beta", "sigma"])
az.plot_psense_quantities(dt, var_names=["beta"])
```

### Reading the diagnosis

| prior | likelihood | diagnosis | Meaning |
|---|---|---|---|
| ≥ 0.05 | ≥ 0.05 | prior-data conflict | prior and data disagree; revisit both |
| ≥ 0.05 | < 0.05 | weak likelihood | the prior is doing the work; report it |
| < 0.05 | ≥ 0.05 | "-" | data-informed, prior not influential |
| < 0.05 | < 0.05 | "-" | well determined by neither? check the parameter is used |

Default threshold 0.05 on the cumulative Jensen-Shannon distance. The goal is understanding, not
silencing the diagnostic: conflict can mean a bad prior or genuinely surprising data. For models
with correlated parameters (splines, many coefficients) analyze the derived quantity you care
about (treatment effect, a prediction) rather than every coefficient.

## Refitting with alternative priors

When power-scaling k values are large (the posteriors are too different to bridge) or the
alternative prior is a different family, refit:

```r
fit_wide  <- update(fit, prior = prior(normal(0, 10), class = b))
fit_t     <- update(fit, prior = prior(student_t(3, 0, 1), class = b))
bayesplot::mcmc_intervals(list(orig = as_draws(fit), wide = as_draws(fit_wide), t = as_draws(fit_t)), pars = "b_Days")
```

Robust conclusions hold across reasonable priors. Sensitive conclusions are reported as
sensitive, with the prior choice justified.

Tail behavior matters as much as scale (BW §17.1): `normal(0, 1)` vs `student_t(7, 0, 1)` on a
slope whose data say 11 gave posteriors of 3.8 and 9.2.

## Static sensitivity analysis (no refit, no reweighting)

Scatter posterior draws of a quantity of interest against a parameter. Read it two ways: as the
direct dependence, and as an implicit reweighting (changing that parameter's prior reweights the
points by the prior ratio, so the plot shows how far the quantity could move).

## Prior information vs likelihood information

If data are uninformative about some aspect, add prior information there. Prefer models whose
parameters the data can update over models that are closer to the truth but cannot be informed.
Skip prior predictive refinement when sensitivity analysis shows the data dominate.
