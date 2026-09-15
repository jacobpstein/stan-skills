# Gaussian Processes and Splines in Stan

Both give flexible nonlinear functions. Splines are cheaper and usually the right first move;
GPs are more interpretable in terms of length scale and amplitude.

## Exact Gaussian process

```stan
data {
  int<lower=1> N;
  array[N] real x;
  vector[N] y;
}
transformed data {
  real delta = 1e-9;                   // jitter for numerical stability
}
parameters {
  real<lower=0> rho;                   // length scale
  real<lower=0> alpha;                 // marginal sd of the function
  real<lower=0> sigma;                 // observation noise
  vector[N] eta;
}
transformed parameters {
  vector[N] f;
  {
    matrix[N, N] K = gp_exp_quad_cov(x, alpha, rho);
    matrix[N, N] L_K = cholesky_decompose(add_diag(K, delta));
    f = L_K * eta;                     // non-centered: eta ~ std_normal()
  }
}
model {
  rho ~ inv_gamma(5, 5);
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 1);
  eta ~ std_normal();
  y ~ normal(f, sigma);
}
```

Cost is O(N³) per gradient evaluation from the Cholesky factorization. Practical limit is a few
hundred observations.

**Marginalized form** (normal likelihood only, much better behaved):

```stan
model {
  matrix[N, N] K = add_diag(gp_exp_quad_cov(x, alpha, rho), square(sigma) + delta);
  y ~ multi_normal_cholesky(rep_vector(mu, N), cholesky_decompose(K));
}
```

Recover `f` in `generated quantities` if you need it.

### Length-scale priors matter

A GP is identified only when the true length scale sits between the data spacing and the data
range. Outside that window the posterior runs off to an unidentified extreme, so use a prior with
little mass below the minimum spacing and above the range:

```r
# choose inv_gamma(a, b) with ~1% mass below lb and ~1% above ub
lb <- min(diff(sort(unique(x)))); ub <- diff(range(x))
```

`inv_gamma` is the standard choice because it has no mass at zero and a light right tail.
Always run a prior predictive check: draw `rho`, `alpha` from their priors and plot the implied
functions.

### Covariance functions

| Function | Behavior |
|---|---|
| `gp_exp_quad_cov(x, alpha, rho)` | infinitely smooth; often too smooth for real data |
| `gp_matern32_cov`, `gp_matern52_cov` | once/twice differentiable; a better default for physical data |
| `gp_exponential_cov` | rough (Ornstein-Uhlenbeck) |
| `gp_periodic_cov(x, alpha, rho, period)` | periodic |

Sum kernels for additive structure (long-term trend plus seasonal plus short-term wiggle);
multiply for interactions. The birthday time-series example in *Bayesian Workflow* ch. 27 builds
exactly this way, one component at a time.

## Hilbert space basis-function approximation (HSGP)

The practical replacement for exact GPs with 1-3D inputs. Approximate the GP by a finite basis of
eigenfunctions of the Laplacian on a box of half-width `c * max|x|`:

```stan
functions {
  vector phi_j(real L, int j, vector x) {
    return sin(j * pi() / (2 * L) * (x + L)) / sqrt(L);
  }
  real lambda_j(real L, int j) {
    return square(j * pi() / (2 * L));
  }
  real spd_exp_quad(real alpha, real rho, real w) {   // spectral density
    return square(alpha) * sqrt(2 * pi()) * rho * exp(-0.5 * square(rho) * w);
  }
}
data {
  int<lower=1> N, M;                 // M basis functions
  vector[N] x, y;
  real<lower=0> L;                   // boundary, e.g. c * max(abs(x)) with c = 1.5
}
transformed data {
  matrix[N, M] PHI;
  for (m in 1:M) PHI[, m] = phi_j(L, m, x);
}
parameters {
  vector[M] beta;
  real<lower=0> rho, alpha, sigma;
}
model {
  vector[M] diagSPD;
  for (m in 1:M) diagSPD[m] = sqrt(spd_exp_quad(alpha, rho, lambda_j(L, m)));
  beta ~ std_normal();
  rho ~ inv_gamma(5, 5);
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 1);
  y ~ normal_id_glm(PHI, 0, diagSPD .* beta, sigma);
}
```

Cost is O(N·M) instead of O(N³). Rules of thumb: `c` (boundary factor) around 1.5, and `M` large
enough that `M >= 1.75 * c / (rho / L)`; in practice start with `M = 20-40` in one dimension and
check that the posterior for `rho` is not pushing against the approximation's resolution. brms
implements this as `gp(x, k = M, c = 1.5)`.

## Splines

Build the basis outside Stan and pass it as a matrix. This is usually the cheapest flexible model.

```r
library(splines)
B <- bs(x, df = 12, intercept = FALSE)
data <- list(N = length(x), B = ncol(B), basis = B, y = y)
```

```stan
data {
  int<lower=1> N, B;
  matrix[N, B] basis;
  vector[N] y;
}
parameters {
  real alpha;
  vector[B] w;
  real<lower=0> tau, sigma;
}
model {
  w ~ normal(0, tau);              // hierarchical shrinkage on the coefficients
  tau ~ exponential(1);
  sigma ~ exponential(1);
  alpha ~ normal(0, 5);
  y ~ normal_id_glm(basis, alpha, w, sigma);
}
```

**Penalized (P-)spline**: penalize differences rather than magnitudes, which enforces smoothness
rather than shrinking toward zero.

```stan
w[2:B] - w[1:(B-1)] ~ normal(0, tau);     // first-order random walk on coefficients
w[1] ~ normal(0, 5);
```

Second-order differences (`w[3:B] - 2*w[2:(B-1)] + w[1:(B-2)]`) penalize curvature and give a
smoother fit that extrapolates linearly.

Use enough basis functions (df 10-20) and let the penalty do the work; choosing the knot count by
cross-validation is fragile.

## Choosing between them

| Situation | Choice |
|---|---|
| 1D smooth trend, N up to millions | penalized spline |
| Need interpretable length scale / amplitude | GP or HSGP |
| N up to a few hundred, need exactness | exact GP |
| N large, 1-3 inputs | HSGP |
| Periodic structure | `gp_periodic_cov` or a Fourier basis |
| Additive decomposition (trend + season + noise) | sum of GP/HSGP components, built up one at a time |
| Many inputs | not a GP; consider a regression with interactions, or BART-style methods outside Stan |

## Checking a fitted smooth

- Plot posterior draws of the function against the data, not just the mean.
- Check that the length-scale posterior is inside the identifiable window; if it piles against the
  prior boundary, the data do not determine the smoothness and the prior is doing the work.
- Compare to a simple parametric alternative (linear, quadratic) with LOO. A flexible fit that
  does not beat a line is telling you something.
- For hierarchical smooths, watch the same funnel issues as any hierarchical model and use
  non-centered forms.
