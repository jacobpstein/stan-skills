---
name: stan-computation
description: >
  Load when the user is running or debugging HMC/NUTS in Stan (CmdStanPy, cmdstanr, rstan,
  brms, rstanarm): choosing chains, warmup and iterations, initial values, reading R-hat,
  ESS bulk/tail, MCSE, divergences, max treedepth, E-BFMI, "Rejecting initial value" messages,
  slow sampling, multimodality, label switching, funnels and non-centered parameterization,
  reparameterization, adapt_delta, and approximate algorithms (optimize, Laplace, Pathfinder,
  ADVI, reduce_sum/threads). Triggers include: divergent transitions, treedepth, rhat, ess_bulk,
  ess_tail, mcse, warmup, adaptation, step size, metric, inits, convergence, mixing, funnel,
  reparameterize, multimodal, pathfinder, laplace_sample, variational, folk theorem.
---

# Fitting and Diagnosing Stan Models

Follows *Bayesian Workflow* Part 3 (ch. 11–13). Two rules organize everything here:

- **Fit fast, fail fast.** Run short chains on a new model, look at the diagnostics, and fix
  the model before spending compute on precise inference for something you may discard.
- **The folk theorem of statistical computing.** When you have computational problems, there
  is usually a problem with your model. The first instinct should be to look at the posterior
  and the model, not to raise `adapt_delta`, `max_treedepth`, or the iteration count.

For writing the program see [stan-modeling](../stan-modeling/SKILL.md); for interface details
in R see [r-stan-interfaces](../r-stan-interfaces/SKILL.md).

## Default run settings

| Setting | Exploration | Final results |
|---|---|---|
| chains | 4 | 4 (or 8; more parallel chains beat longer chains) |
| iter_warmup | 200–500 | 1000 (Stan default; do not shorten) |
| iter_sampling | 200 | 1000+ as needed for MCSE |
| adapt_delta | 0.8 | 0.8; raise to 0.9–0.99 only when divergences are rare (< 1%) and no geometry fix applies |
| max_treedepth | 10 | 10; do not raise to hide problems |
| inits | `0.1` (uniform(−0.1, 0.1) unconstrained) or Pathfinder | same, or explicit per-chain |
| seed | fixed while iterating | fixed but reported, or unfixed with MCSE-justified digits |

```python
fit = model.sample(
    data=data, chains=4, parallel_chains=4,
    iter_warmup=1000, iter_sampling=1000,
    seed=20260914, inits=0.1, adapt_delta=0.8, show_progress=False,
)
fit.save_csvfiles("results/model_v1")   # save immediately
```

```r
fit <- mod$sample(data = data, chains = 4, parallel_chains = 4,
                  iter_warmup = 1000, iter_sampling = 1000,
                  seed = 20260914, init = 0.1, refresh = 500)
fit$save_object("results/model_v1.rds")
```

Stan's default init range, uniform(−2, 2) on the unconstrained scale, assumes unit-scale
parameters. The book now recommends `init = 0.1`; with predictors far from unit scale even
that can overflow (`exp(alpha + beta * x)` with x ~ 2000). The durable fix is to put data and
parameters on unit scale.

## Reading the diagnostics

Check in this order after every fit. Both interfaces print warnings; also call
`fit.diagnose()` (CmdStanPy) or `fit$diagnostic_summary()` / `fit$cmdstan_diagnose()` (cmdstanr).

```python
print(fit.diagnose())
summ = fit.summary()                     # Mean, MCSE, StdDev, 5%, 50%, 95%, ESS_bulk, ESS_tail, R_hat
print(summ.loc[summ["R_hat"] > 1.01])
print(fit.divergences, fit.max_treedepths)
```

```r
fit$diagnostic_summary()                 # divergences, treedepth, E-BFMI per chain
fit$summary(NULL, posterior::default_convergence_measures())
```

| Diagnostic | Threshold | Meaning when violated |
|---|---|---|
| R-hat (rank-normalized) | < 1.01 all quantities (≈ 1.1 acceptable early) | chains disagree in location or scale |
| ESS_bulk, ESS_tail | > 100 per chain (≥ 400 for 4 chains) for reliable diagnostics | too few effective draws; tail ≫ bulk suggests multimodality |
| Divergent transitions | 0, or a handful randomly scattered | step size too large for local curvature: funnel, boundary, improper posterior |
| Max treedepth hits | 0 | very long trajectories: unused/unidentified parameter, flat direction, strong correlation |
| E-BFMI | > 0.3 (some use 0.2) | energy transitions too small; heavy tails or poor adaptation |
| MCSE | small relative to digits you report | not enough draws for that quantity |
| Time per chain | consistent with model size | a trivial model taking seconds hints at aliasing or correlation |

Rank plots (`az.plot_rank`, `mcmc_rank_overlay`) are more sensitive than trace plots. Trace
plots are for understanding a failure, not for certifying success.

## How many iterations and digits

Decide the digits first (BW §11.4–11.6). Two significant digits is usually enough; one is fine
while exploring.

1. Run with defaults; check R-hat and ESS for all quantities.
2. Confirm ESS is high enough for the diagnostics themselves to be reliable (≥ 100 per chain).
3. Look at the posterior sd of each quantity of interest and pick the digits worth reporting.
4. Check `mcse_mean`, `mcse_q5`, `mcse_q95` (posterior R package: `summarise_draws(draws,
   default_mcse_measures())`; ArviZ: `az.summary(..., kind="diagnostics")`). Twice the MCSE is
   the likely variation across seeds.
5. If the last reported digit moves, report fewer digits or run more. Halving MCSE needs 4×
   the draws.

Rules of thumb: 100 independent draws give one significant digit for a mean; 2000 give two.
Stan's NUTS often yields ESS > S/2, so 4 × 1000 draws is close to two digits for means and one
to two for 5%/95% quantiles. Indicators (`Pr(beta > 0)`) need many more draws for the same
digits. For quantities with infinite variance use median and MAD; quantile MCSE is always
defined.

## Failure modes and the step forward

Full worked examples in [references/failure_modes.md](references/failure_modes.md).

| Symptom | Likely cause | Step forward |
|---|---|---|
| 40% divergences, 60% treedepth, R-hat ≈ 3, values ~1e35 | improper posterior (no prior, separation, collinearity) | add proper priors; `--warn-pedantic` finds parameters without priors |
| Treedepth warnings; one parameter with R-hat ≫ 1 and huge range | declared but unused parameter, or one with no likelihood information | remove it or give it a prior; per-parameter R-hat/ESS points to the culprit |
| Diagnostics pass but a trivial model takes seconds; ESS lower than expected; corr ≈ −0.999 in pairs plot | aliasing (intercept plus column of ones; IRT shift) or high posterior correlation | drop redundant column; fix one level or sum-to-zero; **center predictors** (100× speedup in BW §12.3); QR |
| R-hat ≈ 1.7, tiny ESS_bulk, ESS_tail ≫ ESS_bulk, bimodal histogram | multimodality | more chains or multi-path Pathfinder to find modes; understand the source; ordering constraints or relabeling for label switching; stacking during exploration |
| `Rejecting initial value: Log probability evaluates to log(0)` | overflow from inits × unscaled predictors | scale predictors; narrow inits |
| Repeated `Scale vector is -0.7, but must be positive` | missing `<lower=0>` | add the constraint |
| Divergences clustered where a group scale is small | funnel (weak likelihood per group) | non-centered parameterization |
| Divergences persist after non-centering with many obs per group | non-centering hurts with strong likelihood | use centered for those groups |
| R-hat near threshold, thick tails in pairs plot, poor rank plots | heavy-tailed prior (Cauchy) on an unbounded parameter | normal or Student-t prior, or reparameterize the Cauchy |
| R-hat 1.2 after a default run | slow mixing, not broken | run 2× longer |
| R-hat 2 after 1000 iterations | broken | do not just run longer; diagnose |

Divergence rule (BW §12.3): **if more than about 1% of transitions diverge, raising
`adapt_delta` will not help.** Look at pairs plots colored by divergence
(`az.plot_pair(dt, visuals={"divergence": True})`, `mcmc_pairs(np = nuts_params(fit))`,
`mcmc_scatter(..., np = ...)`) and find where they cluster.

## Reparameterization toolbox

- **Non-centered hierarchical**: `mu = mu0 + sigma0 * z; z ~ std_normal();` when groups have
  little data. Centered when groups have lots of data. Mixed cases have no automatic fix;
  parameterize groups differently if needed.
- **`<offset=, multiplier=>`** on declarations lets Stan sample on the scaled space while you
  write the centered model:
  `vector<offset=mu0, multiplier=sigma0>[K] mu;`
- **Center and scale predictors** in `transformed data`; back-transform coefficients in
  `generated quantities`.
- **QR decomposition** for correlated predictors: `Q = qr_thin_Q(X) * sqrt(N-1)`,
  `R = qr_thin_R(X) / sqrt(N-1)`, model on `theta`, recover `beta = R \ theta`.
- **Well-identified quantities as the basis**: start time + log duration; total variance +
  simplex split across variance components; for a sigmoid, the value at a reference x and the
  inflection point instead of raw (a, b).
- **Marginalize** the local parameters when possible (normal-normal, GP with normal
  likelihood, beta-binomial instead of logit-normal binomial); sample the hyperparameters, then
  draw locals in `generated quantities`.
- **Zero-avoiding priors** (lognormal, inverse-gamma) on group scales remove the funnel neck;
  legitimate when there is such information, but say so and run a sensitivity check if used
  only for speed.

See [references/reparameterization.md](references/reparameterization.md) for Stan code.

## Debugging strategy

- **Meet in the middle**: simplify the failing model until it works; build up the working simple
  model until it breaks. The bug lives between.
- **Simulate from the model** and fit the simulated data; misspecified models are slow and
  simulated data separate computation from modeling problems.
- **Modulate the prior**: add normal(0, 100) to every parameter as a diagnostic (only matters if
  the posterior is improper); when freeing a fixed parameter, go normal(4, 0.1) then normal(4, 1),
  not straight to flat.
- **Fit a subset**, 200 iterations, fewer chains, while diagnosing.
- **Optimize first**: `model.optimize()` / `mod$optimize()` smoke-tests the program in seconds.
- **Print intermediates** for numerical problems: evaluate `log_prob` at a problematic draw
  (`model.log_prob(params, data)`, `fit$log_prob()`), look for Inf, 0, NaN.
- Slow model checklist (BW §12.4): simulate, build up one component at a time, 200 iterations,
  informative priors on coefficients and group scales, question the model (interactions?),
  subset the data.

## Approximate algorithms

| Algorithm | Call | Use it for | Do not use it for |
|---|---|---|---|
| L-BFGS optimize | `.optimize()` / `$optimize()` | smoke test; MAP with `jacobian=True` for penalized MLE | uncertainty; hierarchical scale parameters (mode at 0) |
| Laplace | `.laplace_sample(mode=...)` / `$laplace()` | near-normal posteriors, quick draws; check Pareto k of density ratios, importance-reweight if k < 0.7 | skewed or hierarchical posteriors |
| Pathfinder | `.pathfinder()` / `$pathfinder()` | inits for HMC (`inits=fit_pf.create_inits()`), sanity check, finding modes and their relative mass | final inference |
| ADVI | `.variational()` / `$variational()` | repeated fits on similar data **after** validating by simulation for that model class | anything untested; it underestimates variance |
| Chain stacking | `loo::stacking_weights` on per-chain pointwise log-lik | exploration with poorly mixing chains | replacing full Bayes |

Approximate algorithm = exact algorithm for an approximate model. Validate any approximation
the same way as a model: simulate data with known parameters, run the pipeline, compare.

## Parallelism

- Between chains: `parallel_chains=4`.
- Within chain: `reduce_sum` over the likelihood with `cpp_options={"STAN_THREADS": True}` and
  `threads_per_chain=`. Worthwhile when the likelihood dominates cost (large N, ODEs). See
  [references/performance.md](references/performance.md).
- GPU/OpenCL for large matrix and GLM ops via `STAN_OPENCL`.

## Replicability

Bit-level reproducibility across machines is not achievable; aim for agreement within the
reported digits, justified by MCSE. Fix seeds while iterating and in demos; for final results
either report the seed or show that the digits are stable across seeds.

## References

- [references/failure_modes.md](references/failure_modes.md): each failure mode with the Stan
  code, the warning text, what the diagnostics show, and the fix.
- [references/reparameterization.md](references/reparameterization.md): centered vs non-centered,
  offset/multiplier, QR, marginalization, mixture identifiability.
- [references/approximate_inference.md](references/approximate_inference.md): optimize, Laplace,
  Pathfinder, ADVI, divide-and-conquer, PSIS updating, model simplifications.
- [references/performance.md](references/performance.md): vectorization, GLM functions,
  reduce_sum, profiling, compile flags.
