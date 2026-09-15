# Failure Modes and Steps Forward

Source: *Bayesian Workflow* ch. 12. Each entry gives the symptom as Stan reports it, the cause,
what to look at, and the fix. The general mechanism behind most of them: poorly identified
parameters create a ridge or plateau in the posterior, and no single step size or metric works
across it.

## Improper posterior

**Symptoms**

```
Warning: 1580 of 4000 (40.0%) transitions ended with a divergence.
Warning: 2420 of 4000 (60.0%) transitions hit the maximum treedepth limit of 10.
```

R-hat around 3; parameter histograms reaching 1e35.

**Cause**: a parameter with no prior (implicit improper uniform) combined with a likelihood that
does not constrain it: complete separation in logistic regression, perfect collinearity, an
all-successes binomial cell.

**Look at**: marginal histograms and pairs plots of the parameters. Values of 1e35 are obviously
nonsense, and no sampler setting will fix them.

**Fix**: add proper priors.

```stan
alpha ~ normal(0, 10);
beta ~ normal(0, 10);
```

`--warn-pedantic` reports "The parameter beta has no priors" before you ever sample.

**Rule**: if more than about 1% of transitions diverge, raising `adapt_delta` will not help.

## Declared but unused parameter

**Symptom**: `Warning: 1686 of 4000 (42.0%) transitions hit the maximum treedepth limit of 10.`
Diagnostics are fine for every parameter except one, which has terrible R-hat and ESS and a
trace wandering to ±1e20.

**Fix**: remove it, or give it a prior if it is meant to be there. Per-parameter R-hat and ESS
identify the culprit; the trace plot explains what is happening. Pedantic mode says "The
parameter gamma was declared but was not used in the density calculation."

## Aliasing and competing parameters

**Symptom**: no warnings at all. Diagnostics pass. But a trivial model takes seconds per chain
instead of milliseconds, and ESS is lower than you would expect. A pairs plot shows correlation
near −0.999.

**Causes**

- An explicit intercept plus a column of ones in the design matrix.
- A predictor with a single unique value.
- Additive aliasing: `y_jk = alpha_j - beta_k` is unchanged when a constant is added to every
  `alpha` and subtracted from every `beta`.
- Multiplicative aliasing: `y_jk = delta_k (alpha_j - beta_k)`; scale `delta` up and the rest down.
- Label switching in mixtures: `K!` identical posterior modes.

**Fixes**

- Drop the redundant column.
- Pin one level (`beta[1] = 0`) or use `sum_to_zero_vector` for the whole block.
- Fix a scale: give `gamma` a prior centered at 1 rather than a free scale.
- Mixtures: `ordered[K] mu;` or component-distinguishing priors (`mu_k ~ normal(a + b k, s)` with
  `b > 0`); or relabel draws in post-processing. Both get harder in higher dimensions.

## High posterior correlation from uncentered predictors

**Symptom**: treedepth warnings, ESS lower than expected, slow chains. Pairs plot of intercept
against a slope is a tight diagonal.

**Cause**: the intercept is the prediction at `x = 0`, and `x` (say, calendar years 1952–2013) is
far from zero.

**Fix**: center the predictors.

```stan
transformed data {
  vector[N] x_c = x - mean(x);
}
...
generated quantities {
  real alpha = alpha_c - beta * mean(x);   // intercept on the original scale
}
```

In the book's example this raised bulk ESS threefold and cut time per chain from 1.3 s to under
0.05 s. For correlated predictors more generally, use the QR reparameterization.

## Multimodality

**Symptom**: R-hat 1.7, bulk ESS 6, tail ESS 163. Bimodal histogram; chains stuck in separate
regions. **Tail ESS much larger than bulk ESS is a signature of multimodality.**

**Steps**

- Run more chains. With 4 randomly initialized chains there is a 12.5% chance of missing a
  two-mode posterior entirely.
- Run many Pathfinder paths from different inits to find the modes cheaply, then initialize
  chains from them; Pathfinder's importance weights also estimate the relative mass of each mode.
- More chains diagnose but do not fix: the fraction of chains in a mode is not its posterior mass.
- If chains do jump between nearby modes occasionally, running longer helps; rank ECDF-difference
  plots show the residual non-uniformity.
- Chain stacking (cross-validation weights per chain) approximately discards chains stuck in
  low-mass modes. Useful during exploration, not a substitute for full Bayes.
- Understand *where* the multimodality comes from. Each mode is a different explanation of the
  data; the fix is usually a model change (continuous expansion, stronger prior, a constraint).

Four situations and their responses:

| Situation | Response |
|---|---|
| Effectively disjoint volumes, all but one with near-zero mass | avoid minor modes with inits, prior information, or hard constraints |
| Symmetric modes (label switching) | identify the model by ordering or relabeling |
| Genuinely different high-mass modes | stacking; or split with a strong mixture prior and fit components separately |
| Single volume with an unstable tail | initialize near the mass; reparameterize if the tail matters |

## Distant initial values and overflow

**Symptom**

```
Chain 1 Rejecting initial value:
Chain 1   Log probability evaluates to log(0), i.e. negative infinity.
Chain 1   Stan can't start sampling from this initial value.
```

Large R-hat, small ESS, some chains frozen at their starting point.

**Cause**: Stan's default inits are uniform(−2, 2) on the unconstrained scale. With a predictor
reaching 2000, `exp(alpha + beta * x)` overflows for `beta > 0.3`.

**Fixes**, in order of preference:

1. Scale the predictors to unit scale. This is the durable fix.
2. Narrow the inits: `inits=0.001` / `init = 0.001`.
3. Supply explicit inits per chain, or Pathfinder inits.

## Varying curvature (heavy-tailed priors)

**Symptom**: R-hat near the threshold, low ESS, thick tails in pairs plots, rank histograms bad
especially for large values.

**Cause**: `cauchy(0, 10)` on an unbounded parameter. HMC needs a much smaller step size in the
tail than in the bulk, and adaptation picks one step size.

**Fixes**: use `normal` or `student_t(4, ...)`; or reparameterize the Cauchy as
`phi ~ uniform(0, 1); theta = tan(pi() * (phi - 0.5));`. Note that a Cauchy prior on a
`<lower=0>` parameter is sampled on the log scale and usually behaves fine.

## The funnel

**Symptom**: divergences clustered where the group-level scale is small; `sigma0` with R-hat 1.17
and ESS 19.

```
Warning: 406 of 4000 (10.0%) transitions ended with a divergence.
```

**Cause**: in a hierarchical model, `mu_k ~ normal(mu0, sigma0)` means the `mu_k` are pinned when
`sigma0 → 0` and free when it is large. The posterior is a funnel in `(mu, log sigma0)` and its
curvature changes continuously along the edge.

Raising `adapt_delta` removes the divergence warnings without fixing the mixing (R-hat 1.13,
ESS 28). That is not a fix.

**Centered** (use when each group has many observations):

```stan
parameters {
  real mu0;
  real<lower=0> sigma0;
  vector[K] mu;
  real<lower=0> sigma;
}
model {
  mu0 ~ normal(10, 10);
  sigma0 ~ normal(0, 10);
  mu ~ normal(mu0, sigma0);
  sigma ~ lognormal(0, 0.5);
  y ~ normal(mu[x], sigma);
}
```

**Non-centered** (use when groups have little data):

```stan
parameters {
  real mu0;
  real<lower=0> sigma0;
  vector[K] z;
  real<lower=0> sigma;
}
transformed parameters {
  vector[K] mu = mu0 + sigma0 * z;
}
model {
  mu0 ~ normal(10, 10);
  sigma0 ~ normal(0, 10);
  z ~ std_normal();
  sigma ~ lognormal(0, 0.5);
  y ~ normal(mu[x], sigma);
}
```

In the book's example (71 groups, 3 observations each) this took `sigma0` from ESS 19 and
R-hat 1.17 to ESS 1382 and R-hat 1.00.

**Mixed case** (some groups data-rich, others data-poor) has no automatic answer; parameterize
different groups differently if it matters. `vector<offset=mu0, multiplier=sigma0>[K] mu;` gives
the non-centered geometry while keeping the centered statement.

## Missing positivity constraint

**Symptom**

```
Exception: normal_id_glm_lpdf: Scale vector is -0.747476, but must be positive finite!
```

Sporadic occurrences during early warmup are normal. Frequent ones mean the sampler keeps
proposing infeasible values, rejections pile up, and estimates are biased.

**Fix**: `real<lower=0> sigma;`. Pedantic mode catches this.

## Unbounded mixture likelihood

Setting one component mean to a data point and letting its scale go to zero sends the likelihood
to infinity, at every data point. **Fix**: constrain the component scales to be equal, or give
them a shared hierarchical prior (`log sigma_k ~ normal(log sigma_0, 1)`), or use
application-driven informative priors on the component means.

## Slow models

Slow HMC means expensive gradients, high dimension, or geometry where a good step size in one
region is bad in another. Slow computation is usually a symptom of another problem.

Checklist:

1. Simulate data from the model and fit that. Misspecified models are slow.
2. Build up: no varying intercepts, then one batch, then the next.
3. Run 200 iterations, not 2000, while diagnosing.
4. Put at least moderately informative priors on coefficients and group-level scales.
5. Question the model (is a purely additive 14-predictor model plausible?).
6. Fit on a subset of the data first.

## Meet in the middle

Simplify the failing complex model step by step until it works; build the working simple model up
until it breaks. The problem lives between. For multi-component models (ODE plus regression),
unit-test each component separately on simulated data.

## Modulating the prior as a diagnostic

- Add `normal(0, 100)` to every parameter (assuming unit scale). It has no effect unless the
  model is blowing up from non-identification, in which case you can see the problem without the
  model actually diverging.
- When freeing a parameter that was fixed at 4, do not jump to a flat prior: try `normal(4, 0.1)`,
  then `normal(4, 1)`. Set inits and scaling to match: `real<offset=4, multiplier=0.1> theta;`

## What to do about convergence problems (BW §12.5)

- Concern threshold: R-hat > 1.01 or ESS < 100.
- R-hat 1.2 after a default run: high autocorrelation; doubling the run may suffice.
- R-hat 2 after 1000 iterations: more iterations will not help. Diagnose.
- Look at trace plots during warmup and sampling, the log joint density per chain, and per-chain
  posterior summaries. Simplify the model (fix some parameters) and see whether the sampler
  copes.
- Only after you understand why minor modes arise should you discard stuck chains or constrain
  the inits.
- It is legitimate to carry an imperfect fit forward in a series of models, interpreting it in
  the context of simpler models that did fit.
