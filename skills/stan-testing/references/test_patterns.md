# Test Patterns for Stan Projects

## Layer the test suite by cost

| Layer | Runtime | Runs |
|---|---|---|
| Compile with `--warn-pedantic` | seconds | every commit |
| Data-prep unit tests | seconds | every commit |
| `optimize()` smoke test | seconds | every commit |
| Short sampling smoke test (50 + 50 iterations) | ~10 s | every commit |
| Standalone `generate_quantities` against a cached fit | seconds | every commit |
| Parameter recovery | minutes | nightly / pre-merge |
| SBC | hours | before release |

Only the first five belong in a pull-request check.

## Python: pytest

```python
# tests/conftest.py
import pytest
from pathlib import Path
from cmdstanpy import CmdStanModel

STAN_DIR = Path(__file__).parent.parent / "stan"

@pytest.fixture(scope="session")
def model():
    return CmdStanModel(stan_file=STAN_DIR / "linear.stan",
                        stanc_options={"warn-pedantic": True})

@pytest.fixture(scope="session")
def cached_fit(model, sim_data):
    """One short fit reused by every downstream test."""
    return model.sample(data=sim_data, chains=2, iter_warmup=200,
                        iter_sampling=200, seed=1, show_progress=False)

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs real MCMC")
```

```python
# tests/test_program.py
import subprocess, numpy as np

def test_compiles_without_pedantic_warnings(model):
    out = subprocess.run(
        [model.exe_info()["STANC"] if False else "stanc", "--warn-pedantic",
         str(model.stan_file)],
        capture_output=True, text=True)
    assert "Warning" not in out.stderr, out.stderr

def test_optimize_runs(model, sim_data):
    opt = model.optimize(data=sim_data, seed=1)
    assert np.isfinite(opt.optimized_params_np).all()

def test_expected_variables_present(cached_fit):
    for v in ("alpha", "beta", "sigma", "log_lik", "y_rep"):
        assert v in cached_fit.stan_variables()

def test_log_lik_shape(cached_fit, sim_data):
    assert cached_fit.stan_variable("log_lik").shape[1] == sim_data["N"]
```

```python
# tests/test_data_prep.py  -- the tests that catch the most real bugs
def test_stan_data_contract(raw_frame):
    d = build_stan_data(raw_frame)
    assert d["N"] == len(raw_frame)
    assert d["X"].shape == (d["N"], d["K"])
    assert d["group"].dtype.kind == "i"
    assert d["group"].min() >= 1 and d["group"].max() <= d["J"]   # 1-based, in range
    assert np.isfinite(d["X"]).all()
    assert (d["y"] >= 0).all()                                    # matches <lower=0> in the program
```

Run the fast suite with `pytest -m "not slow"`.

## R: testthat

```r
# tests/testthat/setup.R
library(cmdstanr)
mod <- cmdstan_model(testthat::test_path("..", "..", "stan", "linear.stan"),
                     stanc_options = list("warn-pedantic" = TRUE))
sim  <- simulate_dataset(seed = 1)

# tests/testthat/test-program.R
test_that("model compiles cleanly and optimizes", {
  expect_silent(mod$check_syntax(pedantic = TRUE))
  opt <- mod$optimize(data = sim$data, seed = 1)
  expect_true(all(is.finite(opt$summary()$estimate)))
})

test_that("generated quantities have the right shape", {
  fit <- mod$sample(data = sim$data, chains = 2, iter_warmup = 200,
                    iter_sampling = 200, seed = 1, refresh = 0)
  d <- fit$draws("log_lik", format = "draws_matrix")
  expect_equal(ncol(d), sim$data$N)
})

test_that("parameters are recovered", {
  skip_if_not(nzchar(Sys.getenv("RUN_SLOW_TESTS")))
  fit <- mod$sample(data = sim$data, chains = 4, iter_sampling = 1000, seed = 1, refresh = 0)
  s <- fit$summary(c("alpha", "beta", "sigma"))
  expect_true(all(s$rhat < 1.01))
  expect_true(all(s$ess_bulk > 400))
  expect_true(all(s$q5 <= sim$truth & sim$truth <= s$q95))
})
```

## Caching the compiled model

Compilation dominates test time. Cache it:

- pytest: a `scope="session"` fixture, plus an `exe_file` that persists between runs (CmdStanPy
  reuses the executable when the source hash is unchanged).
- CI: cache the `stan/` directory's compiled binaries and the CmdStan installation keyed on the
  CmdStan version and the `.stan` file hashes.
- cmdstanr: `cmdstan_model()` reuses the executable unless the source changed.

## Golden-file tests for generated quantities

Post-processing (predictions, contrasts, poststratification) is ordinary code and deserves
ordinary tests. Run it against a small cached fit and compare to a stored expected output:

```python
def test_ate_calculation(cached_fit, tmp_path):
    ate = compute_ate(cached_fit, newdata=fixture_frame)
    expected = np.load("tests/golden/ate.npy")
    np.testing.assert_allclose(np.quantile(ate, [0.05, 0.5, 0.95]),
                               expected, rtol=1e-6)
```

Regenerate the golden file deliberately, with the same seed, and review the diff. If the numbers
move, either the model changed or something broke; the test's job is to make you look.

## Testing model *behavior*, not just plumbing

Assertions worth writing once the program runs:

```python
def test_posterior_differs_from_prior(model, sim_data):
    """A parameter the data cannot inform is almost always a bug."""
    prior = model.sample(data={**sim_data, "prior_only": 1}, chains=2,
                         iter_sampling=500, seed=1, show_progress=False)
    post  = model.sample(data={**sim_data, "prior_only": 0}, chains=2,
                         iter_sampling=500, seed=1, show_progress=False)
    for p in ("alpha", "beta", "sigma"):
        assert post.stan_variable(p).std() < 0.9 * prior.stan_variable(p).std(), \
            f"{p} was not informed by the data"

def test_no_divergences_on_clean_data(cached_fit):
    assert cached_fit.divergences.sum() == 0

def test_shrinkage_direction(model):
    """Group estimates should be pulled toward the population mean."""
    fit = model.sample(data=hierarchical_sim, seed=1, show_progress=False)
    alpha = fit.stan_variable("alpha").mean(axis=0)
    assert np.std(alpha) < np.std(hierarchical_sim["group_means_nopool"])
```

## CI configuration

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - name: Cache CmdStan
        uses: actions/cache@v4
        with:
          path: ~/.cmdstan
          key: cmdstan-2.39.0
      - run: pip install cmdstanpy pytest arviz
      - run: python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.39.0')"
      - run: pytest -m "not slow" -q
```

Run the slow suite on a schedule, not on every push.

## Reproducibility checklist

- Fix the simulation seed in tests; fix the sampler seed while iterating.
- Pin the CmdStan version in CI and record it in the results.
- Make every script runnable in a clean session, with no reliance on the interactive workspace.
- Use a pipeline tool (`targets`, `Snakemake`, `make`) for anything with more than three stages,
  so only the changed steps recompute.
- Record the CmdStan version, seed, and data hash alongside saved fits:

```python
meta = dict(cmdstan=cmdstanpy.cmdstan_version(),
            seed=SEED,
            data_sha=hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest(),
            git=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip())
(Path(outdir) / "meta.json").write_text(json.dumps(meta, indent=2))
```

Bit-level reproducibility across machines is not achievable. Agreement within the digits you
report, justified by MCSE, is the achievable standard.
