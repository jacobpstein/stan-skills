# Time Series in Stan

## Autoregressive models

```stan
data {
  int<lower=2> T;
  vector[T] y;
}
parameters {
  real alpha;
  real<lower=-1, upper=1> rho;      // stationarity constraint
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 5);
  rho ~ normal(0, 0.5);
  sigma ~ exponential(1);
  y[2:T] ~ normal(alpha + rho * y[1:(T-1)], sigma);
}
```

AR(p) with a coefficient vector:

```stan
parameters { vector[P] rho; }
model {
  for (t in (P+1):T)
    y[t] ~ normal(alpha + dot_product(rho, y[(t-P):(t-1)]), sigma);
}
```

The `<lower=-1, upper=1>` bound enforces stationarity only for AR(1). For AR(p), either accept
that the posterior may wander into nonstationary territory (and check), or parameterize by partial
autocorrelations.

The initial observations are conditioned on rather than modeled. For a fully generative model,
give `y[1]` its stationary distribution: `y[1] ~ normal(alpha / (1 - rho), sigma / sqrt(1 - rho^2))`.

## Random walk and local level

```stan
parameters {
  vector[T] mu_raw;
  real<lower=0> sigma_level, sigma_obs;
}
transformed parameters {
  vector[T] mu = mu_0 + cumulative_sum(mu_raw) * sigma_level;   // non-centered
}
model {
  mu_raw ~ std_normal();
  sigma_level ~ exponential(1);
  sigma_obs ~ exponential(1);
  y ~ normal(mu, sigma_obs);
}
```

The non-centered `cumulative_sum` form is essential: the centered version
(`mu[t] ~ normal(mu[t-1], sigma_level)`) has the same funnel geometry as any hierarchical model,
with `sigma_level` in the role of the group scale.

**Local linear trend** adds a slope state:

```stan
transformed parameters {
  vector[T] delta = delta_0 + cumulative_sum(delta_raw) * sigma_slope;
  vector[T] mu = mu_0 + cumulative_sum(delta) ;
}
```

## Seasonality

```stan
// dummy-variable seasonality with a sum-to-zero constraint
parameters { sum_to_zero_vector[S] season; }
model { season ~ normal(0, sigma_season); }
// use season[month[t]] in the linear predictor
```

Or a Fourier basis, which is smoother and needs fewer parameters:

```stan
transformed data {
  matrix[T, 2*K] X_seas;
  for (k in 1:K) {
    X_seas[, 2*k-1] = sin(2 * pi() * k * to_vector(time) / period);
    X_seas[, 2*k]   = cos(2 * pi() * k * to_vector(time) / period);
  }
}
```

Time-varying seasonality: give the seasonal coefficients their own random walk.

## Structural decomposition

Build the series as a sum of components and add them one at a time, checking after each:

```
y[t] = trend[t] + seasonal[t] + holiday[t] + regression[t] + noise
```

The *Bayesian Workflow* birthdays case study (ch. 27) is this pattern with GP components: start
with a slow trend, add yearly seasonality, add day-of-week, add special days, comparing LOO at
each step. Fit each expansion and keep the previous model for comparison.

## Forecasting

Forecast in `generated quantities`, propagating the state forward one step at a time:

```stan
data { int<lower=0> T_ahead; }
generated quantities {
  vector[T_ahead] y_future;
  {
    real y_prev = y[T];
    for (h in 1:T_ahead) {
      y_future[h] = normal_rng(alpha + rho * y_prev, sigma);
      y_prev = y_future[h];        // feed the simulated value forward
    }
  }
}
```

The common error is to use the observed `y[T]` at every horizon, which understates the forecast
variance. For a state-space model, propagate the latent state:

```stan
generated quantities {
  vector[T_ahead] mu_future, y_future;
  real mu_prev = mu[T];
  for (h in 1:T_ahead) {
    mu_prev = normal_rng(mu_prev, sigma_level);
    mu_future[h] = mu_prev;
    y_future[h] = normal_rng(mu_prev, sigma_obs);
  }
}
```

Posterior predictive checks for time series need the same care: replicate the whole series
sequentially conditional on the previous *simulated* value, not the observed one, then compare a
structural statistic (number of direction changes, autocorrelation at lag 1, longest run) between
the observed and replicated series.

## Cross-validation for time series

LOO is usually the wrong target: leaving out an interior point when the neighbors are present is
not the prediction task. Use:

- **Leave-future-out (LFO)**: refit at increasing time origins and score the next h steps. Approximate
  it with PSIS between refits; refit when the Pareto k of the importance weights exceeds 0.7.
- **h-block cross-validation** with the joint log score for *comparing* models; more efficient
  than LFO for that purpose.

Both need refits. Expect to write a loop rather than call `loo()`.

```r
# leave-future-out sketch
L <- 100                                   # minimum training length
elpds <- rep(NA, T - L)
for (t in L:(T - 1)) {
  fit_t <- mod$sample(data = list(T = t, y = y[1:t]), refresh = 0)
  lp <- fit_t$draws("log_lik_next", format = "matrix")   # log p(y[t+1] | y[1:t])
  elpds[t - L + 1] <- log_mean_exp(lp)
}
sum(elpds)
```

## Hidden Markov models

Discrete states must be marginalized; Stan has the forward algorithm built in:

```stan
data {
  int<lower=1> T, K;
  vector[T] y;
}
parameters {
  array[K] simplex[K] Gamma_rows;   // transition probabilities out of each state
  simplex[K] rho;                   // initial state distribution
  ordered[K] mu;                    // state means, ordered for identifiability
  real<lower=0> sigma;
}
transformed parameters {
  matrix[K, K] Gamma;
  matrix[K, T] log_omega;           // log p(y[t] | state = k)
  for (k in 1:K) Gamma[k] = Gamma_rows[k]';
  for (t in 1:T)
    for (k in 1:K)
      log_omega[k, t] = normal_lpdf(y[t] | mu[k], sigma);
}
model {
  mu ~ normal(0, 5);
  sigma ~ exponential(1);
  target += hmm_marginal(log_omega, Gamma, rho);
}
generated quantities {
  array[T] int states = hmm_latent_rng(log_omega, Gamma, rho);
  matrix[K, T] state_prob = hmm_hidden_state_prob(log_omega, Gamma, rho);
}
```

Order the state means (or use state-distinguishing priors) or the labels will switch between
chains.

## Practical notes

- Center time at a meaningful origin (`(year - 2000) / 10`), never leave raw years in a linear
  predictor: the intercept becomes an extrapolation to year zero and the posterior correlation
  wrecks sampling.
- Missing time points: index observations by their time and let the state process run over the
  full grid.
- Irregular spacing: use a GP or a continuous-time state model rather than a discrete AR.
- Long series with per-time-point states are exactly the case where the per-observation latent
  variable breaks PSIS-LOO; see the integrated-LOO pattern in
  [stan-model-evaluation/references/loo_cv.md](../../stan-model-evaluation/references/loo_cv.md).
