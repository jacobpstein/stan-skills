# Hierarchical and Multilevel Models in Stan

Source: *Bayesian Workflow* §12.3, ch. 17; *Regression and Other Stories* ch. 22; Stan User's
Guide.

## Build-up order

1. **Complete pooling**: one intercept, group structure ignored.
2. **No pooling**: group-specific intercepts with independent priors.
3. **Partial pooling**: hierarchical intercepts. Compare to both.
4. **Varying slopes**: only if the question is about how the slope varies.
5. **Group-level predictors**: explain the between-group variation.

Fit each, compare via LOO, and check whether the added structure reduces between-group variance.
Steps 1 and 2 are scaffolds: they exist to make step 3 interpretable, not to be reported as
competitors.

## Varying intercepts

```stan
data {
  int<lower=0> N, J, K;
  array[N] int<lower=1, upper=J> group;
  matrix[N, K] X;
  vector[N] y;
}
parameters {
  real mu_alpha;
  real<lower=0> sigma_alpha, sigma;
  vector[K] beta;
  vector<offset=mu_alpha, multiplier=sigma_alpha>[J] alpha;
}
model {
  mu_alpha ~ normal(0, 5);
  sigma_alpha ~ exponential(1);
  sigma ~ exponential(1);
  beta ~ normal(0, 1);
  alpha ~ normal(mu_alpha, sigma_alpha);
  y ~ normal(alpha[group] + X * beta, sigma);
}
```

`alpha[group]` is vectorized multiple indexing: one element per observation. The
`<offset=, multiplier=>` declaration gives the non-centered geometry while the statement stays
readable.

Prior scaling when you add a level: the residual sd is now within-group only, so split the
previous prior. `sigma ~ exponential(1/50)` becomes `sigma ~ exponential(1/25)` and
`sigma_alpha ~ exponential(1/25)`.

## Centered vs non-centered

| Regime | Parameterization |
|---|---|
| Few observations per group, weak likelihood | non-centered (`offset`/`multiplier`, or explicit `z`) |
| Many observations per group, strong likelihood | centered |
| Mixed | no automatic answer; split the groups if it matters |

Explicit non-centered form:

```stan
parameters { vector[J] z; }
transformed parameters { vector[J] alpha = mu_alpha + sigma_alpha * z; }
model { z ~ std_normal(); }
```

The funnel that motivates this and the diagnostics that reveal it are in
[stan-computation/references/failure_modes.md](../../stan-computation/references/failure_modes.md).

## Varying intercepts and slopes with correlation

```stan
data {
  int<lower=0> N, J;
  int<lower=1> P;                          // number of varying coefficients (incl. intercept)
  array[N] int<lower=1, upper=J> group;
  matrix[N, P] Z;                          // design for the varying part (column 1 = ones)
  vector[N] y;
}
parameters {
  vector[P] gamma;                         // population-level values
  vector<lower=0>[P] tau;                  // sds of the group deviations
  cholesky_factor_corr[P] L;
  matrix[P, J] z;
  real<lower=0> sigma;
}
transformed parameters {
  matrix[J, P] b = (diag_pre_multiply(tau, L) * z)';    // group deviations, non-centered
}
model {
  to_vector(z) ~ std_normal();
  L ~ lkj_corr_cholesky(2);
  tau ~ exponential(1);
  gamma ~ normal(0, 5);
  sigma ~ exponential(1);
  {
    vector[N] mu;
    for (n in 1:N) mu[n] = Z[n] * (gamma + b[group[n]]');
    y ~ normal(mu, sigma);
  }
}
generated quantities {
  matrix[P, P] Omega = multiply_lower_tri_self_transpose(L);
  matrix[P, P] Sigma = quad_form_diag(Omega, tau);
}
```

Prior notes: `tau` for a slope gets its own scale (`exponential(1/10)` for ±10 units/predictor);
`lkj_corr_cholesky(1)` is uniform over correlation matrices, `(2)` shrinks toward independence.
As the dimension grows, LKJ(1) marginals concentrate near zero anyway; check with prior
predictive simulation.

## Crossed and nested groupings

Stan does not distinguish them; you index them. Two crossed factors:

```stan
parameters {
  vector<offset=0, multiplier=sigma_a>[J] a;     // e.g. subject
  vector<offset=0, multiplier=sigma_b>[K] b;     // e.g. item
  real mu;
  real<lower=0> sigma_a, sigma_b, sigma;
}
model {
  a ~ normal(0, sigma_a);
  b ~ normal(0, sigma_b);
  y ~ normal(mu + a[subject] + b[item], sigma);
}
```

Center the deviations at zero and keep a single global intercept `mu`; otherwise `mu`, `a`, and
`b` are additively aliased. `sum_to_zero_vector[J] a;` enforces it exactly.

Nested groupings (classrooms within schools) are just two index arrays:

```stan
y ~ normal(mu + school_eff[school] + class_eff[classroom], sigma);
class_eff ~ normal(0, sigma_class);        // classroom index is globally unique
```

## Group-level predictors

Explain the between-group variation instead of only absorbing it:

```stan
data {
  matrix[J, Q] U;              // group-level predictors, one row per group
}
parameters {
  vector[Q] delta;
}
model {
  alpha ~ normal(U * delta, sigma_alpha);
  delta ~ normal(0, 1);
}
```

`sigma_alpha` should shrink when the group-level predictors explain real variation. This is also
what makes predictions for **new** groups informative rather than pure prior draws.

## Non-nested and multiple-membership

```stan
// each observation belongs to several groups with weights w
for (n in 1:N) mu[n] = alpha0 + dot_product(w[n], alpha[members[n]]);
```

## Hierarchical GLMs

Everything above composes with a link function:

```stan
y ~ bernoulli_logit(alpha[group] + X * beta);
y ~ poisson_log(log_exposure + alpha[group] + X * beta);
y ~ neg_binomial_2_log(log_exposure + alpha[group] + X * beta, phi);
```

Priors on the logit scale: `sigma_alpha ~ exponential(1)` means group intercepts typically vary
by about one logit, which is already substantial.

## Item-response / two-way latent structure

```stan
parameters {
  real<lower=0> sigma_alpha, sigma_beta, sigma_gamma;
  real mu_beta;
  vector<offset=0, multiplier=sigma_alpha>[J] alpha;        // person ability, centered at 0
  vector<offset=mu_beta, multiplier=sigma_beta>[K] beta;    // item difficulty
  vector<offset=1, multiplier=sigma_gamma>[K] gamma;        // discrimination, centered at 1
}
model {
  alpha ~ normal(0, sigma_alpha);
  beta ~ normal(mu_beta, sigma_beta);
  gamma ~ normal(1, sigma_gamma);      // normal, not lognormal: allows negative discrimination
  y ~ bernoulli_logit(gamma[item] .* (alpha[student] - beta[item]));
}
```

Identification: fixing the ability mean at 0 resolves the additive shift; fixing the
discrimination prior mean at 1 resolves the multiplicative scaling. Allowing negative
discrimination is deliberate: it is how you discover miscoded items.

## Partial pooling diagnostics

- Compare `sigma_alpha` to the residual `sigma`: if it is near zero the groups are not
  distinguishable and the hierarchical layer may not be needed.
- Plot the group estimates against their no-pooling counterparts: partial pooling shrinks small
  groups more. That shrinkage pattern is the model working, not a bug.
- With few groups (J < 5) the group-level scale is weakly identified; report an interval, expect
  a long right tail, and consider a zero-avoiding prior.
- For LOO on hierarchical models, decide whether the prediction task is new observations or new
  groups; see [stan-model-evaluation/references/loo_cv.md](../../stan-model-evaluation/references/loo_cv.md).
