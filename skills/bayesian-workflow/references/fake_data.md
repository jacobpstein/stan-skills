# Fake-Data Simulation and Simulation Experiments

Source: *Bayesian Workflow* ch. 6, §4.4, §10.5; *Regression and Other Stories* §5.5, §7.2, §16.6.
"You don't understand your model until you can simulate from it."

## Simulate first, summarize last (BW §6.1)

- Never compute a function of posterior means. `E[a/b] ≠ E[a]/E[b]`; compute `a/b` per draw,
  then summarize. Parameters are correlated and the mistake is common.
- Predictions belong in `generated quantities` with `_rng` functions, one replicate per draw:

```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
  int<lower=0> N_tilde;
  vector[N_tilde] x_tilde;
}
parameters {
  real a, b;
  real<lower=0> sigma;
}
model {
  a ~ normal(0, 10);
  b ~ normal(0, 1);
  sigma ~ exponential(1);
  y ~ normal(a + b * x, sigma);
}
generated quantities {
  array[N_tilde] real y_tilde = normal_rng(a + b * x_tilde, sigma);
}
```

- Point summaries: mean ± sd or median ± MAD-sd (`1.483 * median(|z - median(z)|)`); central 50%
  and 90% intervals; HPD when the posterior hits a boundary; never the mode. For a positive
  scale parameter with few groups, report an interval, not ± sd.
- The posterior of a prediction is flatter than the prediction at the posterior mean (logistic
  curve averaged over draws vs. curve at `â, b̂`).

## The fake-data check (BW §6.3)

Before fitting real data, or whenever a fit is suspicious:

1. Choose plausible parameter values (domain knowledge, or the posterior medians from a real fit).
2. Simulate one dataset with the **same size, shape, and structure** as the real data (same N,
   same groups, same predictors; simulate x only if there is no design to reuse).
3. Fit the model.
4. Check:
   - The posterior differs from the prior for every parameter. If not, a parameter is declared
     but unused, or the design cannot inform it.
   - The true values lie inside reasonable intervals (roughly half in 50% intervals, most in 90%).
   - Recovery quality across the regions of parameter space that matter (a hierarchical model
     behaves differently in the funnel neck and mouth; a GP fails when the true length scale is
     shorter than the data resolution or longer than the data range; sums of exponentials are
     identifiable only when rates are well separated).
5. If it fails: simplify the model step by step until it works; the problem is in the last step.

One dataset catches blatant errors (wrong model, indexing bugs). Subtle miscalibration needs
SBC; see [stan-testing](../../stan-testing/SKILL.md). But "a few simple checks are just about
always worthwhile before investing in elaborate suites of simulations."

### Two-step procedure

Fit the model to real data → draw parameter values from that posterior → use them as truth for
simulations. This concentrates the checks in the region of parameter space the data care about.

### Upper bound interpretation

Good recovery on data simulated from the model is an **upper bound** on what you can learn
from real data. Failure on the model's own data means the real-data analysis is hopeless.
Simulated data are the only place inference on latent variables can be checked directly.

## Python and R harnesses

```python
import numpy as np, arviz as az
rng = np.random.default_rng(1)
true = dict(a=50.0, b=2.0, sigma=10.0)
N = 100
x = rng.uniform(0, 10, N)
y = rng.normal(true["a"] + true["b"] * x, true["sigma"])
fit = model.sample(data=dict(N=N, x=x, y=y, N_tilde=1, x_tilde=[20.0]), seed=1)
post = fit.draws_pd()
for k, v in true.items():
    lo, hi = post[k].quantile([0.05, 0.95])
    print(f"{k}: true {v:.2f}  90% ({lo:.2f}, {hi:.2f})  {'ok' if lo <= v <= hi else 'MISS'}")
```

```r
a <- 50; b <- 2; sigma <- 10; N <- 100
x <- runif(N, 0, 10)
y <- rnorm(N, a + b * x, sigma)
fit <- mod$sample(data = list(N = N, x = x, y = y, N_tilde = 1, x_tilde = 20), refresh = 0)
sims <- posterior::as_draws_rvars(fit$draws())
print(quantile(sims$a, c(0.05, 0.95))); print(quantile(sims$a / sims$b, c(0.1, 0.9)))
```

Use `rvar` objects (posterior package) or xarray in Python so functions of draws stay draws.

## Designing a simulation experiment (BW §4.4, §6.4)

State the aim first. Examples:
- Can this design detect an item with zero or negative discrimination? Set `sigma_gamma = 0.5`
  so a few items are near zero, fix other hyperparameters at real-fit medians, simulate J = 32
  students, fit, compare true vs estimated. Then repeat with J = 100 to see what more data buys.
- How does adjusting for a pre-treatment covariate behave under unbalanced assignment and a
  nonlinear outcome? Simulate 500 students with latent ability, assign treatment with
  probability depending on the midterm, compare raw difference vs regression-adjusted estimates.
- What does measurement error in x do to a slope? Simulate `x_star = x + noise` and refit;
  error in x attenuates the slope, error in y does not.

Rules:
- Vary one factor at a time; a brute-force grid is expensive and uninformative.
- Use the same plotting code for real and simulated fits; compare panels with common axes.
- N = 1000 replications are not needed to see a pattern with N = 1000 observations; one
  realization often shows it, and SBC handles the "is it calibrated" question separately.
- Also simulate from a **wrong** data-generating process (Student-t errors when fitting a normal
  model) and measure how coverage degrades. Robustness is an empirical property.

## Hierarchical replication scenarios (BW §6.1)

For a model `p(phi) p(alpha | phi) p(y | alpha, phi)` there are three things "new data" can mean:

| Scenario | Draw | Use |
|---|---|---|
| New observations in existing groups | keep `phi^s, alpha^s`; draw `y` | within-group PPC |
| New groups from the fitted population | keep `phi^s`; draw `alpha_new ~ p(alpha | phi^s)`; then `y` | predictions for a new hospital/school/year |
| New population | draw `phi` from the prior, then `alpha`, then `y` | prior predictive; needs proper hyperpriors |

Predictive simulation also needs values for unmodeled data (new x): choose values of interest,
give x a model, or bootstrap the observed x.

## Design analysis by simulation (ROS ch. 16)

Power calculations are fake-data simulations: assume an effect size and sd, simulate the study
many times, fit the intended analysis, and record the distribution of estimates, standard
errors, and the probability that the 95% interval excludes zero. Use it to choose sample size
and to see how noisy the estimate would be even when "significant" (Type M and Type S errors).
Interactions need roughly 16× the sample size of main effects for the same precision.
