# Regression Patterns in Stan

Each pattern gives the Stan code, the priors that go with it, and the checks that matter. All
assume centered and scaled predictors; see [syntax.md](syntax.md) for the `transformed data`
idiom.

## Linear regression

```stan
model {
  alpha_c ~ normal(mean_y_guess, 2.5 * sd_y_guess);
  beta ~ normal(0, 1);                        // standardized predictors
  sigma ~ exponential(1);
  y ~ normal_id_glm(Xc, alpha_c, beta, sigma);
}
generated quantities {
  vector[N] log_lik;
  array[N] real y_rep = normal_rng(alpha_c + Xc * beta, sigma);
  real r2 = variance(alpha_c + Xc * beta) / (variance(alpha_c + Xc * beta) + square(sigma));
  for (n in 1:N) log_lik[n] = normal_lpdf(y[n] | alpha_c + Xc[n] * beta, sigma);
}
```

Checks: residuals vs fitted; `ppc_dens_overlay`; `ppc_stat(y, yrep, "sd")`; LOO-PIT.

## Robust regression (Student-t errors)

```stan
parameters { real<lower=1> nu; ... }
model {
  nu ~ gamma(2, 0.1);
  y ~ student_t(nu, alpha_c + Xc * beta, sigma);
}
```

Use when LOO-PIT is S-shaped because a few outliers inflate `sigma`, or when `pp_check(type =
"loo_intervals")` shows points far outside. A posterior for `nu` concentrated below ~5 is strong
evidence for heavy tails. Note that under Student-t the residual sd is `sigma * sqrt(nu/(nu-2))`,
so the prior on `sigma` means something different than under the normal model.

## Logistic regression

```stan
model {
  alpha_c ~ normal(0, 2.5);
  beta ~ normal(0, 2.5);            // standardized predictors
  y ~ bernoulli_logit_glm(Xc, alpha_c, beta);
}
generated quantities {
  vector[N] p = inv_logit(alpha_c + Xc * beta);
  array[N] int y_rep = bernoulli_rng(p);
  vector[N] log_lik;
  for (n in 1:N) log_lik[n] = bernoulli_logit_lpmf(y[n] | alpha_c + Xc[n] * beta);
}
```

Interpretation: divide a coefficient by 4 for the maximum change in probability per unit of the
predictor. Never use a flat prior: complete or quasi-complete separation makes the posterior
improper and the chains drift to infinity. Checks: PAV calibration plot
(`az.plot_ppc_pava` in Python; `reliabilitydiag::reliabilitydiag` on `E_loo()` values in R),
not a density overlay.

Binomial (grouped) form:

```stan
y ~ binomial_logit(trials, alpha_c + Xc * beta);
```

## Poisson and negative binomial with exposure

```stan
data { vector[N] log_exposure; }
model {
  alpha_c ~ normal(0, 2.5);
  beta ~ normal(0, 1);
  phi ~ inv_gamma(0.4, 0.3);                    // brms default; small phi = far from Poisson
  y ~ neg_binomial_2_log_glm(Xc, log_exposure + alpha_c, beta, phi);
}
generated quantities {
  array[N] int y_rep = neg_binomial_2_log_rng(log_exposure + alpha_c + Xc * beta, phi);
}
```

Start with negative binomial, not Poisson. Diagnose overdispersion with a rootogram and by
comparing `p_loo` to the parameter count: a Poisson fit to overdispersed data gave `p_loo = 273`
for 4 parameters. Coefficients are multiplicative: `exp(beta_k)` is the rate ratio per unit.

## Zero-inflated and hurdle

```stan
parameters { vector[K] beta, beta_zi; real alpha, alpha_zi; }
model {
  vector[N] eta = alpha + Xc * beta + log_exposure;
  vector[N] eta_zi = alpha_zi + Xc * beta_zi;
  for (n in 1:N) {
    if (y[n] == 0)
      target += log_sum_exp(bernoulli_logit_lpmf(1 | eta_zi[n]),
                            bernoulli_logit_lpmf(0 | eta_zi[n])
                              + neg_binomial_2_log_lpmf(0 | eta[n], phi));
    else
      target += bernoulli_logit_lpmf(0 | eta_zi[n])
                  + neg_binomial_2_log_lpmf(y[n] | eta[n], phi);
  }
}
```

**Zero-inflated**: zeros come from either a structural process or the count process.
**Hurdle**: a separate process decides zero vs positive, and the count part is truncated at 1
(`neg_binomial_2_log_lpmf(y | eta, phi) - neg_binomial_2_lccdf(0 | ...)`).

Motivate zero inflation with a calibration plot of `Pr(y > 0)`, not with the raw zero count.
Summarize treatment effects with `posterior_epred`-style contrasts in `generated quantities`,
because the coefficient signs in the two components are opposite and not separately meaningful.

## Ordinal

```stan
parameters {
  vector[K] beta;               // no intercept: absorbed by the cutpoints
  ordered[C - 1] cutpoints;
}
model {
  cutpoints ~ student_t(3, 0, 2.5);
  beta ~ normal(0, 1);
  y ~ ordered_logistic(Xc * beta, cutpoints);
}
generated quantities {
  array[N] int y_rep;
  for (n in 1:N) y_rep[n] = ordered_logistic_rng(Xc[n] * beta, cutpoints);
}
```

`ordered_probit` is the normal-latent version. Category-specific effects require the full
multinomial or a "unequal slopes" extension; check the proportional-odds assumption with grouped
PPCs before adding that complexity.

## Multinomial / categorical

```stan
parameters { matrix[K, C - 1] beta_raw; }
transformed parameters {
  matrix[K, C] beta = append_col(rep_vector(0, K), beta_raw);   // reference category = 1
}
model {
  to_vector(beta_raw) ~ normal(0, 2.5);
  for (n in 1:N) y[n] ~ categorical_logit(beta' * Xc[n]');
}
```

Fixing one column at zero identifies the model; a softmax without that constraint is aliased.

## Censored and truncated

```stan
// right censoring at U: y_obs observed, N_cens units known only to exceed U
y_obs ~ normal(mu_obs, sigma);
for (m in 1:N_cens) target += normal_lccdf(U | mu_cens[m], sigma);

// interval censoring in (L, U)
target += log_diff_exp(normal_lcdf(U | mu, sigma), normal_lcdf(L | mu, sigma));

// truncation: units below L never enter the sample
for (n in 1:N) y[n] ~ normal(mu[n], sigma) T[L, ];
```

Survival models are the common case: `weibull_lccdf`, `lognormal_lccdf`, `exponential_lccdf` for
the survival function. Report the data model (which observations were censored and why), not just
the likelihood terms.

## Measurement error

```stan
parameters {
  vector[N] x_true;
  real mu_x; real<lower=0> sigma_x;
}
model {
  x_true ~ normal(mu_x, sigma_x);          // population model for the latent predictor
  x_obs ~ normal(x_true, sigma_meas);      // measurement model
  y ~ normal(alpha + beta * x_true, sigma);
}
```

Ignoring error in `x` attenuates `beta` toward zero; error in `y` merges with the residual and
does not bias the slope. `sigma_meas` can be data (from a calibration study) or a parameter with
an informative prior; it is rarely identified from the main data alone.

## Missing data

Stan has no `NA`. Split the vector:

```stan
data {
  int<lower=0> N_obs, N_mis;
  array[N_obs] int<lower=1> ii_obs;
  array[N_mis] int<lower=1> ii_mis;
  vector[N_obs] y_obs;
}
parameters { vector[N_mis] y_mis; }
transformed parameters {
  vector[N_obs + N_mis] y;
  y[ii_obs] = y_obs;
  y[ii_mis] = y_mis;
}
```

Missing *outcomes* under MAR can simply be dropped; missing *predictors* need a model for the
predictor, which means the program is no longer conditional on x.

## QR reparameterization for correlated predictors

```stan
transformed data {
  matrix[N, K] Q = qr_thin_Q(Xc) * sqrt(N - 1.0);
  matrix[K, K] R = qr_thin_R(Xc) / sqrt(N - 1.0);
  matrix[K, K] R_inv = inverse(R);
}
parameters { vector[K] theta; }
model { y ~ normal_id_glm(Q, alpha_c, theta, sigma); }
generated quantities { vector[K] beta = R_inv * theta; }
```

## Interactions and polynomials

Build the columns in `transformed data` (centered first, so the main effects stay interpretable
as comparisons at the mean of the other predictor):

```stan
transformed data {
  vector[N] x1x2 = x1_c .* x2_c;
}
```

Hierarchical shrinkage by term order is a good default: give interaction coefficients a tighter
prior than main effects. Interactions need roughly 16 times the sample size of a main effect for
the same precision, so expect wide posteriors.

## Splines as a design matrix

Build the basis in R or Python (`splines::bs`, `patsy.bs`, `scipy.interpolate`), pass it as a
matrix, and put a hierarchical prior on the coefficients:

```stan
data { int<lower=1> B; matrix[N, B] basis; }
parameters { vector[B] w; real<lower=0> tau; }
model {
  w ~ normal(0, tau);        // or a random-walk prior on differences for a smoothing spline
  tau ~ exponential(1);
  y ~ normal(alpha + basis * w, sigma);
}
```

Penalized (P-spline) version: `w[2:B] - w[1:(B-1)] ~ normal(0, tau);` penalizes wiggliness
instead of magnitude. See [gp_splines.md](gp_splines.md).
