# Prior Recommendations by Parameter Type

Sources: *Bayesian Workflow* §5.6, ch. 17; *Regression and Other Stories* §9.5; rstanarm
`priors` vignette; brms `set_prior`.

## Quick table (unit-scale parameters unless noted)

| Parameter | Weakly informative default | Tighter alternative | Avoid |
|---|---|---|---|
| Regression coefficient, standardized x and y | `normal(0, 1)` | `normal(0, 0.5)`; `normal(0, 0.1)` when effects are known small | flat, `normal(0, 1e6)` |
| Coefficient, raw scales | `normal(0, 2.5 * sd_y / sd_x)` | domain-based | `normal(0, 100)` on unscaled x |
| Intercept, centered predictors | `normal(mean_y, 2.5 * sd_y)` or `normal(guess, wide)` | | prior on the uncentered intercept when x is far from 0 |
| Logistic coefficient, standardized x | `normal(0, 2.5)` | `normal(0, 1)` | `normal(0, 50)`; Cauchy on unbounded params |
| Logistic intercept | `normal(0, 2.5)` to `normal(0, 5)` | | `normal(0, 100)` (improper-like with zero successes) |
| Residual sd | `exponential(1)` (unit) / `exponential(1/sd_guess)` | `normal(0, 1)` truncated | `uniform(0, 100)`, `inv_gamma(0.001, 0.001)` |
| Group-level sd | `exponential(1)` or `normal(0, 1)` truncated; halve the residual prior mean when adding a level | `lognormal` or `inv_gamma` zero-avoiding when funnel and justification | `uniform`, `half-Cauchy(0, 25)` for few groups |
| Correlation matrix | `lkj_corr_cholesky(1)` | `lkj_corr_cholesky(2)` to `(4)` | flat on each pairwise correlation (inconsistent in ≥ 3D) |
| Student-t df | `gamma(2, 0.1)`; brms default `gamma(2, 0.1)` | | flat |
| NB shape (`phi`) | `inv_gamma(0.4, 0.3)` (brms default) or `exponential(1)` on `1/sqrt(phi)` | | flat |
| Simplex / mixture weights | `dirichlet(rep_vector(2, K))` | | `dirichlet(0.01)` |
| Probability (no covariates) | `beta(2, 2)`, or logit-scale `normal(0, 1.5)` | | `uniform` if the value can be near 0 or 1 with little data |
| Rate / elasticity believed in (0, 1) | `normal(0.5, 0.5)` | | `uniform(0, 1)` |
| GP amplitude | `normal(0, 1)` truncated | | |
| GP length scale | `inv_gamma(a, b)` with 1% mass below the data spacing and 1% above the range | | flat (mass at unidentified extremes) |
| Spline coefficients | hierarchical: `b ~ normal(0, tau)`, `tau ~ exponential(1)` | | independent flat |
| Effect size in a noisy study | `normal(0, 0.1)` sd units | `normal(0, 0.05)` | `normal(0, 1)` ("most mass on huge effects") |
| SNR prior for a new RCT (meta-analytic) | `0.42 normal(0, 1.5) + 0.58 normal(0, 3.5)` on z = effect/SE (Cochrane) | psychology replications: `0.57 normal(0, 1.2) + 0.43 normal(0, 4.1)` | |

## rstanarm defaults (autoscaled)

For `stan_glm(y ~ x1 + x2, family = gaussian)`:

- `prior = normal(0, 2.5)` with `autoscale = TRUE`: each coefficient gets sd `2.5 * sd(y) / sd(x_k)`
  (for gaussian; for other families `2.5 / sd(x_k)`).
- `prior_intercept = normal(0, 2.5)`, autoscaled to `normal(mean(y), 2.5 * sd(y))` and placed on the
  intercept **after centering** the predictors.
- `prior_aux = exponential(1)`, autoscaled to rate `1 / sd(y)`.
- `stan_glmer`: `prior_covariance = decov(regularization = 1, concentration = 1, shape = 1, scale = 1)`
  (LKJ(1) on correlations, a simplex split of total variance, gamma on the total scale).

Inspect with `prior_summary(fit)`. Turn autoscaling off with `autoscale = FALSE` when you specify
priors on the raw scale yourself.

## brms

Defaults: **flat** on population-level coefficients (`class = b`); `student_t(3, median(y), 2.5 * mad(y))`
on `Intercept` (after internal centering) and on `sigma`, `sd` (half-t); `lkj(1)` on `cor`. Always
set coefficient priors:

```r
prior <- prior(normal(0, 1), class = b) +
  prior(normal(250, 100), class = Intercept) +
  prior(exponential(0.04), class = sigma) +
  prior(exponential(0.04), class = sd, group = Subject, coef = Intercept) +
  prior(exponential(0.1),  class = sd, group = Subject, coef = Days) +
  prior(lkj(1), class = cor, group = Subject)
get_prior(Reaction ~ Days + (Days | Subject), data = sleepstudy)   # see what can be set
fit <- brm(Reaction ~ Days + (Days | Subject), data = sleepstudy, prior = prior,
           sample_prior = "yes")          # keeps prior draws for plot(hypothesis(...))
make_stancode(..., prior = prior)         # inspect the generated Stan
```

`class = b` covers all slopes unless `coef =` narrows it. Distributional parameters use
`dpar = "zi"`, `dpar = "sigma"` etc. `lb = 0` truncates.

## Stan idioms

```stan
// scale parameters
real<lower=0> sigma;         sigma ~ exponential(1);
real<lower=0> tau;           tau ~ normal(0, 1);           // half-normal via constraint
// correlation
cholesky_factor_corr[K] L;   L ~ lkj_corr_cholesky(2);
// vector of coefficients, one statement
vector[K] beta;              beta ~ normal(0, 1);
// several scalars at once
{a, b} ~ normal(0, 5);
// t degrees of freedom
real<lower=1> nu;            nu ~ gamma(2, 0.1);
// zero-avoiding group scale
tau ~ lognormal(0, 1);
// scaled coefficient by number of predictors
beta ~ normal(0, 1 / sqrt(K));
```

The exponential is parameterized by **rate**: `exponential(1/50)` has mean 50.

## Justification sentences (template)

Write one per informative prior, e.g.:

- "`b_0 ~ normal(250, 100)` ms: reaction times for adults are 150–350 ms; 95% prior mass in
  (50, 450) rules out impossible values without favoring any plausible one."
- "`b_1 ~ normal(0, 20)` ms/day: a change of 40 ms/day over 8 days would double reaction time,
  at the edge of plausibility; centered at zero because the analysis should be able to detect a
  coding error that flips the sign."
- "`gamma_1 ~ normal(0, 0.2)` on log(BVA/50): sd 0.2 ≈ log(60/50), so we are fairly sure the
  population mean is between 40 and 60."
