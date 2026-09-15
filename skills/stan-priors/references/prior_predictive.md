# Prior Predictive Checking

Source: *Bayesian Workflow* §5.8–5.9, §8.5; Gabry et al. (2019).

## Five kinds of variables in a Stan program (BW §5.8)

Stan does not distinguish them, but the simulation does:

```stan
data {
  int N;                          // 1. unmodeled data
  vector[N] x, y;                 // x: unmodeled data;  y: 5. modeled data
  real mu_a, mu_b;                // 2. unmodeled parameters (hyperparameters passed as data)
  real<lower=0> sigma_a, sigma_b;
}
parameters {
  real a, b;                      // 4. modeled parameters
  real<lower=0> sigma_y;          // 3. parameter with an improper prior (no statement below)
}
model {
  a ~ normal(mu_a, sigma_a);
  b ~ normal(mu_b, sigma_b);
  y ~ normal(a + b * x, sigma_y);
}
```

Procedure: fix (1) and (2); choose a value for (3) since it cannot be simulated (better: give it a
proper prior); draw (4) from the prior; draw (5) from the data model. Repeat. Compare the
simulated datasets with the real one and look at the simulated parameters to understand how the
model works.

## Implementations

### `prior_only` flag (single program, recommended)

```stan
data { ... int<lower=0, upper=1> prior_only; }
model {
  ...priors...
  if (!prior_only) y ~ normal(mu, sigma);
}
generated quantities { array[N] real y_rep = normal_rng(mu, sigma); }
```

Sampling with `prior_only = 1` runs HMC on the prior; it is fast and reuses the exact prior code.

### `fixed_param` with `_rng` in generated quantities

A separate program that draws parameters with `_rng` calls in `generated quantities` and runs
with `fixed_param=True` (CmdStanPy) / `fixed_param = TRUE` (cmdstanr). Faster for expensive
likelihoods and guarantees independent draws, but duplicates the prior code (which is itself a
useful consistency check; see SBC).

### brms / rstanarm

`brm(..., sample_prior = "only")` then `pp_check(fit)`; rstanarm `stan_glm(..., prior_PD = TRUE)`.

## What to plot

- ~10 simulated datasets as scatterplots or lines, side by side with the real data on common axes.
- For regression, spaghetti of `a + b * x` over the observed x range.
- For GPs and splines, draws of the latent function.
- Parameter draws themselves (histograms) to confirm the joint prior is what you intended.
- Summary statistics of `y_rep` (range, proportion of zeros, sd) against the observed.

## Worked examples (BW §5.9)

**Logistic regression, x standardized, n = 32:**

| Prior on a, b | Prior predictive behavior |
|---|---|
| `normal(0, 0.5)` | every item near 50% correct, no relation to x: too strong |
| `normal(0, 5)` | range of item difficulties and discriminations, some negative: weakly informative |
| `normal(0, 50)` | near-perfect threshold discrimination in every simulated dataset: weak in parameter space, far too strong in data space |

**Linear trend in temperature, x centered:** `normal(0, 100)` on both parameters implies ±10 000
degree changes; `a ~ normal(0, 10)`, `b ~ normal(0, 1/3)` per year is weakly informative (a
change over a century unlikely to exceed 10 degrees, yearly variation under 3).

**Many predictors:** logistic regression with independent `normal(0, 1)` on 2, 4, and 15 binary
predictors: prior predictive probabilities spread out for 2–4, pile up at 0 and 1 for 15. Scale
per-coefficient sd by `1/sqrt(K)` or use a joint prior.

**GP:** plot prior draws of the function for a few amplitude and length-scale values before
fitting; "useful with any model and essential with unfamiliar ones".

## Reading the results

| Pattern | Diagnosis | Action |
|---|---|---|
| Simulated outcomes span many orders of magnitude | priors too wide on a link or log scale | tighten; quantile-match from the outcome scale |
| Impossible values (negative counts, probabilities beyond [0,1]) in > a few % | wrong family or scale | change family, add constraints, rescale |
| All datasets look identical | dogmatic prior | widen |
| Datasets look plausible but far from the observed | miscentered prior | recenter using domain knowledge, not the data |
| Data clearly dominate the prior (posterior unchanged under power-scaling) | prior predictive absurdity is harmless | move on |

## When to skip

If sensitivity analysis shows the data dominate, a strange prior predictive distribution is not a
problem (BW §8.5). Prior predictive checks remain essential before collecting data, for
understanding an unfamiliar model, and for components expected to be weakly informed by data
(group-level scales with few groups, GP hyperparameters, mixture weights).

## Thought experiment version

Even without running code: write down what data the priors would generate. If the answer is
"anything at all", the priors are not yet doing their job.
