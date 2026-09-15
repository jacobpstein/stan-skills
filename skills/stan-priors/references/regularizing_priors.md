# Regularizing and Sparsity Priors in Stan

Source: *Bayesian Workflow* §5.6 (piranha principle), §8.5, §9.7, ch. 28; Piironen & Vehtari
(2017); Zhang et al. (2022); Aguilar & Bürkner (2023).

## Why joint priors

Many independent weak priors are a strong joint prior on predictions. With K standardized
predictors and `beta_k ~ normal(0, 1)`, the implied prior on R² concentrates near 1 as K grows
(BW Fig. 8.13). The **piranha principle**: many large effects cannot coexist; encode "the total
explained variation is bounded" rather than "each coefficient is unbounded".

Options in order of complexity:

1. Scale per-coefficient sd with K: `beta ~ normal(0, tau / sqrt(K))`.
2. Hierarchical shrinkage with a learned scale: `beta ~ normal(0, tau); tau ~ normal(0, 1);`
   (ridge-like; fine when effects are similar in size).
3. **R2D2**: put the prior on R² and derive the coefficient prior; predictively consistent as
   predictors are added.
4. **Regularized (Finnish) horseshoe**: few large effects, most near zero.

## Regularized horseshoe

```stan
data {
  int<lower=0> N, K;
  matrix[N, K] X;                   // standardized
  vector[N] y;
  real<lower=0> scale_global;       // tau0 = p0 / (K - p0) * sigma / sqrt(N)
  real<lower=1> nu_global;          // 1 = half-Cauchy on tau
  real<lower=1> nu_local;           // 1 = half-Cauchy on lambda
  real<lower=0> slab_scale;         // e.g. 2: scale of the regularizing slab
  real<lower=0> slab_df;            // e.g. 4
}
parameters {
  real alpha;
  vector[K] z;
  real<lower=0> tau;                // global shrinkage
  vector<lower=0>[K] lambda;        // local shrinkage
  real<lower=0> caux;
  real<lower=0> sigma;
}
transformed parameters {
  real c = slab_scale * sqrt(caux);
  vector<lower=0>[K] lambda_tilde = sqrt(c^2 * square(lambda) ./ (c^2 + tau^2 * square(lambda)));
  vector[K] beta = z .* lambda_tilde * tau;
}
model {
  z ~ std_normal();
  lambda ~ student_t(nu_local, 0, 1);
  tau ~ student_t(nu_global, 0, scale_global * sigma);
  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);
  alpha ~ normal(0, 5);
  sigma ~ exponential(1);
  y ~ normal_id_glm(X, alpha, beta, sigma);
}
```

Choosing `scale_global`: with a prior guess `p0` of the number of relevant predictors,
`tau0 = p0 / (K - p0) * sigma / sqrt(N)`; pass `scale_global = p0 / (K - p0) / sqrt(N)` and
multiply by `sigma` inside (as above). For logistic regression drop `sigma` and use a
pseudo-variance `4` in the derivation (`scale_global = p0 / (K - p0) * 2 / sqrt(N)`).

Sampling: the horseshoe creates a double funnel; expect to need `adapt_delta = 0.95–0.99` and
to see a few divergences. Always use the non-centered form above. brms: `prior(horseshoe(df = 1,
scale_global = ..., df_global = 1, scale_slab = 2, df_slab = 4), class = b)`.

## R2D2 (R² induced Dirichlet decomposition)

Put a `beta(a, b)` prior on R², convert to total signal variance `W = R² / (1 − R²)` (times
`sigma²`), split it across coefficients with a Dirichlet simplex `phi`, and set
`beta_k ~ normal(0, sqrt(W * phi_k) * sigma)`.

```stan
data {
  int<lower=0> N, K;
  matrix[N, K] X;
  vector[N] y;
  real<lower=0> r2_mean_a, r2_mean_b;      // beta(a, b) prior on R2, e.g. (1, 2) mean 1/3
  real<lower=0> dirichlet_conc;            // e.g. 0.5 (sparser) to 1
}
parameters {
  real alpha;
  vector[K] z;
  real<lower=0, upper=1> R2;
  simplex[K] phi;
  real<lower=0> sigma;
}
transformed parameters {
  real tau2 = R2 / (1 - R2);
  vector[K] beta = z .* sqrt(tau2 * phi) * sigma;
}
model {
  z ~ std_normal();
  R2 ~ beta(r2_mean_a, r2_mean_b);
  phi ~ dirichlet(rep_vector(dirichlet_conc, K));
  alpha ~ normal(0, 5);
  sigma ~ exponential(1);
  y ~ normal_id_glm(X, alpha, beta, sigma);
}
```

The R² prior is interpretable ("I expect this regression to explain about a third of the
variance") and stays the same when predictors are added, which makes model expansion
predictively consistent. brms ≥ 2.20: `prior(R2D2(mean_R2 = 0.3, prec_R2 = 3, cons_D2 = 0.5),
class = b)`. For multilevel models the R2D2M2 extension shares the variance budget with the
group-level terms.

## When to use which

| Situation | Prior |
|---|---|
| Few predictors, all plausibly relevant | `normal(0, 1)` on standardized coefficients |
| Many predictors, effects similar in size | hierarchical normal with learned `tau` |
| Many predictors, believe most are irrelevant | regularized horseshoe (then projpred for selection) |
| Want to reason about explained variance; expanding the predictor set over time | R2D2 |
| Time series coefficients (AR lags) | ARR2 (R2D2 for autoregressions) |
| Interactions and polynomial terms | hierarchical shrinkage by term order (`tau_main > tau_interaction`) |

Regularization is a modeling statement about a population of effects. Report it as such, and
check with prior predictive simulation that the implied R² prior is what you meant.
