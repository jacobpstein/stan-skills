# Reparameterization

Source: *Bayesian Workflow* §12.4; Stan User's Guide, Efficiency Tuning.

HMC works best when the posterior is close to a unit-scale independent normal without sharp
corners. Reparameterization is how you get there. The shared idea across all of these: **find
the quantities the data actually identify and make them the parameters.**

## Centered vs non-centered hierarchical

```stan
// centered: good when each group has many observations (strong likelihood)
vector[J] alpha;
alpha ~ normal(mu, tau);

// non-centered: good when groups have few observations (weak likelihood)
vector[J] z;
z ~ std_normal();
vector[J] alpha = mu + tau * z;

// affine declaration: non-centered geometry, centered statement
vector<offset=mu, multiplier=tau>[J] alpha;
alpha ~ normal(mu, tau);
```

The `<offset=, multiplier=>` form is preferred in new code: Stan samples
`(alpha - mu) / tau` internally while the model block stays readable, and switching between
parameterizations is a one-line change.

Mixed regimes (some groups rich, some poor) have no automatic solution. If it matters, split the
groups into two blocks with different parameterizations.

## QR decomposition for correlated predictors

```stan
transformed data {
  matrix[N, K] Q = qr_thin_Q(X) * sqrt(N - 1.0);
  matrix[K, K] R = qr_thin_R(X) / sqrt(N - 1.0);
  matrix[K, K] R_inv = inverse(R);
}
parameters {
  real alpha;
  vector[K] theta;            // coefficients in the rotated space
  real<lower=0> sigma;
}
model {
  theta ~ normal(0, 5);       // note: the prior is on theta, not beta
  y ~ normal_id_glm(Q, alpha, theta, sigma);
}
generated quantities {
  vector[K] beta = R_inv * theta;
}
```

The rotated coefficients are approximately uncorrelated, which removes the dominant geometry
problem in regressions with correlated columns. The catch is that a simple prior on `theta` is
not a simple prior on `beta`; use it when the priors are weak or when you construct the prior on
`beta` explicitly.

## Centering and scaling predictors

Always the first thing to try:

```stan
transformed data {
  matrix[N, K] Xc;
  vector[K] x_mean, x_sd;
  for (k in 1:K) {
    x_mean[k] = mean(X[, k]);
    x_sd[k] = sd(X[, k]);
    Xc[, k] = (X[, k] - x_mean[k]) / x_sd[k];
  }
}
generated quantities {
  vector[K] beta_orig = beta ./ x_sd;
  real alpha_orig = alpha_c - dot_product(x_mean ./ x_sd, beta);
}
```

## Cholesky factors for correlation and covariance

Never sample a `corr_matrix` or `cov_matrix` directly when you can sample its Cholesky factor:

```stan
parameters {
  cholesky_factor_corr[K] L;
  vector<lower=0>[K] tau;
  matrix[K, J] z;
}
transformed parameters {
  matrix[J, K] b = (diag_pre_multiply(tau, L) * z)';
}
model {
  to_vector(z) ~ std_normal();
  L ~ lkj_corr_cholesky(2);
  tau ~ exponential(1);
}
generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L);
  matrix[K, K] Sigma = quad_form_diag(Omega, tau);
}
```

Use `multi_normal_cholesky(mu, diag_pre_multiply(tau, L))` rather than `multi_normal` with a
constructed covariance: it avoids a factorization per leapfrog step.

## Well-identified quantities as parameters

**Start and end times** → start time and **log duration**. Duration is positive by construction,
additive shifts act on the start, and multiplicative changes act on the duration.

**Variance components** (hierarchical, state space, GP with several sources) → **total variance
plus a simplex** splitting it:

```stan
parameters {
  real<lower=0> sigma_total;
  simplex[3] p;                    // shares of the total variance
}
transformed parameters {
  vector[3] sigma = sigma_total * sqrt(p);
}
```

The total is usually well identified even when the split is not, and the simplex prior
(`dirichlet`) is an interpretable statement about how variation divides.

**Three-parameter logistic curve** `E(y) = m * inv_logit(a + b x)` is poorly identified when the
data do not span the full range. Reparameterize with the value at a reference point and the
inflection point:

```stan
// gamma = E(y | x = 0), eta = -a / b (inflection)
real gamma;
real eta;
real b;
// E(y) = gamma * inv_logit(b * (x - eta)) / inv_logit(-b * eta);
```

Then a prior on `eta` can suppress inflection points far outside the observed x range.

**Sum-to-zero effects**: use the built-in type rather than a soft constraint.

```stan
parameters { sum_to_zero_vector[K] effect; }     // replaces effect ~ normal(0, 5) plus a Potential
```

## Marginalization

If part of the model can be integrated out analytically, the remaining posterior is usually much
better behaved.

- Hierarchical normal-normal: sample `p(phi | y)` and draw the group means from
  `p(theta | phi, y)` in `generated quantities`.
- GP with a normal likelihood: integrate the latent function out (`gp_exp_quad_cov` plus a
  diagonal, then `multi_normal_cholesky` on `y` directly).
- Logit-normal varying intercepts on binomial data → beta-binomial, which marginalizes the
  intercept analytically (awkward when there are group-level predictors).
- Discrete latent states (mixtures, HMMs) **must** be marginalized: Stan has no discrete
  parameters. Use `log_sum_exp` / `log_mix` in the model block and recover state probabilities in
  `generated quantities`.

```stan
// two-component mixture, marginalized
target += log_mix(lambda,
                  normal_lpdf(y[n] | mu[1], sigma[1]),
                  normal_lpdf(y[n] | mu[2], sigma[2]));
```

## Adding prior information as a reparameterization of the problem

The ladder of abstraction: poor mixing → difficult geometry → weakly informative data for parts
of the model → substantive prior information. Fitting a sum of two declining exponentials works
when the rates are 0.1 and 2.0 and fails when they are 0.1 and 0.2, because the curve is then
indistinguishable from a single exponential. `normal(0, 1)` priors on unit-scale parameters are
enough regularization to fix the geometry; check sensitivity with `normal(0, 0.5)` and
`normal(0, 2)`.

Zero-avoiding priors (`lognormal`, `inv_gamma`) on group-level scales remove the funnel neck.
Legitimate when such information exists; if you use them purely for computation, say so and run
a prior sensitivity analysis.

## Adding data

Some geometries cannot be reparameterized away, only informed away: a normal likelihood with
N = 1 is unbounded, N = 2 gives a funnel, N = 8 is fine. Lotka-Volterra with 6 points diverges,
with 9 works, with 21 is well behaved. Sometimes the answer is that the data do not support the
model and you should fit a simpler one.
