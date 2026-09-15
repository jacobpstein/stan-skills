---
name: stan-testing
description: >
  Load when the user is testing, validating, or setting up infrastructure for Stan code:
  parameter-recovery tests, simulation-based calibration (SBC) and rank histograms, pytest or
  testthat harnesses around CmdStanPy/cmdstanr models, fixed_param simulation programs,
  standalone generate_quantities, fast tests in CI, reproducibility, seeds, and project
  structure for a series of models. Triggers include: simulation-based calibration, SBC, rank
  statistic, rank histogram, parameter recovery, recover the parameters, test my Stan model,
  pytest stan, testthat stan, fixed_param, generate_quantities, CI for Bayesian models,
  reproducible analysis, targets, Snakemake, golden test, regression test for a model.
---

# Testing Stan Models as Software

Statistical modeling is software development (BW ch. 15). A Stan program can be wrong in three
distinct ways, and each needs a different test:

| Failure | Test |
|---|---|
| The program does not express the intended model | parameter recovery on simulated data; separate prior-predictive generator vs posterior code |
| The intended model is expressed but the algorithm does not sample it correctly | SBC, convergence diagnostics |
| Downstream code (data prep, post-processing, plots) is wrong | ordinary unit tests with fixtures |

Fast tests belong in CI; slow inference tests belong behind a marker. See
[bayesian-workflow](../bayesian-workflow/SKILL.md) for where these sit in the loop.

## The test ladder, cheapest first

1. **Compiles with `--warn-pedantic`** and no warnings. Seconds.
2. **Optimizes**: `model.optimize()` returns a finite mode from several inits. Seconds.
3. **Fixed-param generator runs**: the prior predictive program produces data in the right shape
   and range. Seconds.
4. **One-dataset parameter recovery**: fit simulated data, check the truth is inside the
   intervals and the posterior differs from the prior. Minutes.
5. **SBC**: many prior draws, ranks uniform. Hours; run before release, not on every commit.

## Parameter recovery test

```python
# tests/test_recovery.py
import numpy as np, pytest
from cmdstanpy import CmdStanModel

@pytest.fixture(scope="session")
def model():
    return CmdStanModel(stan_file="stan/linear.stan",
                        stanc_options={"warn-pedantic": True})

def simulate(rng, N=200, alpha=1.0, beta=2.0, sigma=0.5):
    x = rng.normal(size=N)
    y = rng.normal(alpha + beta * x, sigma)
    return dict(N=N, x=x, y=y, prior_only=0), dict(alpha=alpha, beta=beta, sigma=sigma)

@pytest.mark.slow
def test_recovers_parameters(model):
    rng = np.random.default_rng(0)
    data, truth = simulate(rng)
    fit = model.sample(data=data, chains=4, iter_sampling=500, seed=1, show_progress=False)
    assert fit.diagnose().find("no problems detected") >= 0 or fit.divergences.sum() == 0
    summ = fit.summary()
    assert (summ.loc[list(truth), "R_hat"] < 1.01).all()
    for name, value in truth.items():
        draws = fit.stan_variable(name)
        lo, hi = np.quantile(draws, [0.005, 0.995])
        assert lo <= value <= hi, f"{name}: {value} outside ({lo:.3f}, {hi:.3f})"
```

Use a wide interval (99%) for a single dataset so the test is not flaky; a single 90% interval
misses the truth 10% of the time by design. Fix the simulation seed; vary the sampler seed only
if you want to detect seed sensitivity.

```r
# tests/testthat/test-recovery.R
test_that("linear model recovers parameters", {
  skip_on_cran(); skip_if_not(nzchar(Sys.getenv("RUN_SLOW_TESTS")))
  set.seed(0)
  N <- 200; alpha <- 1; beta <- 2; sigma <- 0.5
  x <- rnorm(N); y <- rnorm(N, alpha + beta * x, sigma)
  fit <- mod$sample(data = list(N = N, x = x, y = y, prior_only = 0),
                    chains = 4, iter_sampling = 500, seed = 1, refresh = 0)
  s <- fit$summary(c("alpha", "beta", "sigma"))
  expect_true(all(s$rhat < 1.01))
  expect_true(all(s$q5 <= c(alpha, beta, sigma) & c(alpha, beta, sigma) <= s$q95))
})
```

### What the check actually tells you

- **Posterior equals prior for a parameter** → it is declared but unused, or the design cannot
  inform it. This is the single most valuable signal in the test.
- **Truth far outside the interval** → indexing bug, wrong link, wrong likelihood, or a
  transposed design matrix.
- **Estimates compressed toward the population mean** → shrinkage, expected, not a bug. Check
  calibration (how often intervals cover), not point accuracy.
- Recovery on the model's own data is an **upper bound** on real-data performance.

## Simulation-based calibration (SBC)

SBC checks the whole inference pipeline: if the generator and the posterior code define the same
model and the sampler is correct, then ranks of prior draws within posterior draws are uniform.

Procedure (BW ch. 14, Modrák et al. 2025):

1. Draw `theta_s` from the prior, `s = 1..S` (S ≈ 100 for a smoke check, 1000 for a real check).
2. Simulate `y_s ~ p(y | theta_s)`.
3. Fit the model to each `y_s`, keeping `M` approximately independent posterior draws (thin
   correlated MCMC draws; `M = 99` or `M = 999` makes the rank histogram bins even).
4. For each test quantity `T`, compute the rank of `T(theta_s)` among the `M` posterior values.
   Break ties at random for discrete quantities.
5. The `S` ranks should be uniform on `{0, ..., M}`.

```python
import numpy as np, arviz as az
from cmdstanpy import CmdStanModel

gen = CmdStanModel(stan_file="stan/linear_prior_sim.stan")   # _rng only, fixed_param
post = CmdStanModel(stan_file="stan/linear.stan")
S, M, THIN = 200, 99, 10
ranks = {k: [] for k in ["alpha", "beta", "sigma"]}
for s in range(S):
    sim = gen.sample(data={"N": N, "x": x}, fixed_param=True, chains=1,
                     iter_sampling=1, seed=1000 + s, show_progress=False)
    truth = {k: sim.stan_variable(k)[0] for k in ranks}
    y = sim.stan_variable("y_sim")[0]
    fit = post.sample(data={"N": N, "x": x, "y": y, "prior_only": 0}, chains=4,
                      iter_sampling=M * THIN // 4, thin=THIN, seed=2000 + s,
                      show_progress=False)
    for k in ranks:
        draws = fit.stan_variable(k)
        ranks[k].append(int((draws < truth[k]).sum()))
# uniformity check
az.plot_ecdf_pit(az.from_dict({k: (np.array(v) + 0.5) / (M + 1) for k, v in ranks.items()}))
```

The generator program mirrors the priors exactly, which is the point: writing the prior twice
and checking the two agree catches transcription errors.

```stan
// stan/linear_prior_sim.stan  — run with fixed_param
data { int<lower=0> N; vector[N] x; }
generated quantities {
  real alpha = normal_rng(0, 2.5);
  real beta  = normal_rng(0, 1);
  real<lower=0> sigma = fabs(normal_rng(0, 1));
  array[N] real y_sim = normal_rng(alpha + beta * x, sigma);
}
```

In R use the **SBC** package (`hyunjimoon/SBC`), which handles the plumbing, caching, and plots:

```r
library(SBC)
gen <- SBC_generator_function(simulate_one_dataset, N = 200)
ds  <- generate_datasets(gen, n_sims = 200)
bkd <- SBC_backend_cmdstan_sample(mod, chains = 4, iter_sampling = 1000, thin = 10)
res <- compute_SBC(ds, bkd)
plot_rank_hist(res); plot_ecdf_diff(res); res$stats
```

### Reading the rank histogram

| Shape | Diagnosis |
|---|---|
| Flat within the band | calibrated |
| Piled at low ranks | posterior systematically **over**estimates |
| Piled at high ranks | posterior systematically **under**estimates |
| U shape | posterior too **narrow** (overconfident) |
| Inverted U (∩) | posterior too **wide** |
| Flat plus a spike at one end | a few extreme fits, often non-convergence |

Prefer ECDF-difference plots with simultaneous bands over histograms. With many test quantities,
rank them by the γ statistic (the likelihood of the most extreme ECDF point under uniformity)
and inspect the worst.

### Practical SBC advice

- Wide priors make SBC correct but irrelevant: `normal(0, 100)` on a logistic coefficient
  generates all-zero or all-one datasets that no algorithm handles. Either tighten the prior or
  **reject unrealistic simulated datasets** using a criterion that depends on the **data only**
  (not on the latent parameters); that leaves the posterior unchanged and keeps SBC valid.
- Too slow for 1000 replications? Run 20. Extreme ranks (0 or M) appearing repeatedly already
  indicate a problem. Start with SBC on a smaller submodel.
- Add derived test quantities (log-likelihood, a prediction) as well as raw parameters;
  miscalibrations in different regions can cancel in the marginals.
- **Posterior SBC**: after seeing real data, use the fitted posterior as the prior, simulate new
  data, refit, and check ranks. This checks coherence of the algorithm in the region that
  matters, which is what you care about when only a couple of divergences appear.

## Fast tests for CI

Keep the full pipeline out of CI; test the parts that break most often.

- **Compilation** of every `.stan` file with `--warn-pedantic`, asserting no warnings.
- **Data preparation**: given a small synthetic frame, the dict/list handed to Stan has the right
  keys, shapes, dtypes, and satisfies the declared bounds. This catches most real failures.
- **Standalone generated quantities**: run `generate_quantities` against a tiny cached fit to
  test post-processing without resampling.

```python
gq = model_gq.generate_quantities(previous_fit=cached_fit, data=data)
```

```r
gq <- mod_gq$generate_quantities(fitted_params = cached_fit, data = data)
```

- **Very short sampling** (`chains=1, iter_warmup=50, iter_sampling=50`) purely to assert the
  program runs and produces the expected variables, with no assertions about values.
- Mark the real inference tests `@pytest.mark.slow` / `skip_if_not(Sys.getenv(...))` and run them
  nightly.

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = ["slow: needs real MCMC (deselect with -m 'not slow')"]
```

## Project structure and reproducibility

```
project/
├── stan/            one file per distinct model; shared code in stan/functions/*.stanfunctions
├── R/ or src/       data prep, plotting, shared helpers
├── scripts/         self-contained, runnable in a clean environment, numbered by stage
├── tests/
├── results/         fit objects and CSVs, git-ignored, regenerable
└── report/
```

- Git from the first commit. Keep clearly different models in **different files** so they can be
  diffed and compared; use the commit history as the record of what you found and decided.
- Version the data, figures, and report alongside the code; tag the commit matching each
  milestone report rather than naming files `_final_final`.
- Every script must run in a clean session with no dependence on preset globals. Use `targets`
  (R) or `Snakemake`/`make` for simulation studies so only the changed stages recompute.
- Share code across models with `#include` of `.stanfunctions` files
  (`stanc_options = list("include-paths" = "stan/functions")`) rather than copying blocks; a bug
  found in one place is then fixed in one place.
- Bit-level reproducibility across machines is not achievable. Aim for agreement within the
  digits you report, justified by MCSE. Fix seeds while iterating; for final results either
  report the seed or show the digits are stable across seeds.

## Readable Stan is tested Stan

Stan's type system is documentation. Prefer declarations over comments:

```stan
real<lower=0> oxygen_level;              // not: real x17;  // oxygen, should be positive
target += normal_lpdf(y | mu, sigma);    // not: target += -0.5 * (y - mu)^2 / sigma^2;  // normal
```

Document user-facing functions at the API level (argument types, return, error conditions).
Replace a long expression you want to comment with a well-named function.

## References

- [references/sbc.md](references/sbc.md): full SBC implementations in Python and R, rank
  statistics, the γ statistic, rejection sampling of datasets, posterior SBC.
- [references/test_patterns.md](references/test_patterns.md): pytest and testthat harnesses,
  fixtures, caching compiled models, CI configuration, golden-file tests for generated
  quantities.
