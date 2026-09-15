---
name: stan-model-evaluation
description: >
  Load when the user is checking or comparing fitted Stan models: posterior predictive checks,
  PSIS-LOO cross-validation, Pareto k diagnostics, moment matching, k-fold or leave-one-group-out
  CV, LOO-PIT calibration, loo_compare / az.compare, elpd differences, stacking and model
  averaging, Bayesian R2, projection predictive variable selection. Covers the loo/bayesplot R
  packages, cmdstanr $loo(), brms add_criterion, and ArviZ 1.x (arviz-stats/arviz-plots).
  Triggers include: ppc, pp_check, plot_ppc, LOO, PSIS, elpd, Pareto k, khat, reloo,
  moment_match, kfold, loo_compare, stacking, model weights, LOO-PIT, calibration, rootogram,
  PAV calibration plot, log score, projpred, cv_varsel, WAIC, Bayes factor.
---

# Model Checking and Comparison for Stan Models

Follows *Bayesian Workflow* (Gelman, Vehtari, McElreath et al. 2026) ch. 8–9 and ch. 24.
Convergence is **not** evidence that the model is fit for purpose. After every successful
fit, three checks are almost always necessary:

1. Do the parameter estimates make sense (fit to implicit prior knowledge)?
2. Prior, posterior, and **cross-validation** predictive checks (fit to data).
3. Look at the quantities the model will actually be used for (predictions, ratios,
   poststratified averages), not only the named parameters.

For building the model see [stan-modeling](../stan-modeling/SKILL.md); for the overall
iterative loop see [bayesian-workflow](../bayesian-workflow/SKILL.md); for sampler problems
see [stan-computation](../stan-computation/SKILL.md).

## Prerequisites in the Stan program

Model checking needs replicated data and pointwise log-likelihood from `generated quantities`.
Add them to every model you intend to check or compare:

```stan
generated quantities {
  vector[N] log_lik;
  array[N] real y_rep;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
    y_rep[n] = normal_rng(mu[n], sigma);
  }
}
```

Rules:
- `log_lik[n]` must be the log density of **one exchangeable unit** (the unit you would leave
  out). For grouped data where you want to predict new groups, compute `log_lik` per group.
- `y_rep` must come from the **data model**, not from the likelihood you coded (mixtures,
  censoring): simulate the generative process you would present in the paper.
- Use `_rng` functions only in `generated quantities`; never touch `target` there.

## Posterior predictive checking (PPC)

Simulate `y_rep ~ p(y_rep | y)` and compare it to `y` graphically. The authors rarely use
posterior predictive p-values; graphical checks are the default.

### Which check to run

| Data / question | Check | bayesplot | ArviZ 1.x |
|---|---|---|---|
| Continuous, overall shape | density overlay of ~50 `y_rep` vs `y` | `ppc_dens_overlay`, `ppc_ecdf_overlay` | `az.plot_ppc_dist(dt, kind="kde")` / `kind="ecdf"` |
| Specific feature (sd, max, skew, prop of zeros) | histogram of `T(y_rep)` with `T(y)` line | `ppc_stat(y, yrep, stat = "sd")` | `az.plot_ppc_tstat(dt, t_stat="sd")` |
| Counts with few distinct values | rootogram | `ppc_rootogram` | `az.plot_ppc_rootogram(dt)` |
| Binary outcomes | PAV-adjusted calibration plot | `reliabilitydiag()` on `E_loo()` values | `az.plot_ppc_pava(dt)` |
| Subgroups, especially groupings **not** in the model | grouped stat / intervals | `ppc_stat_grouped`, `ppc_intervals_grouped` | `az.plot_ppc_tstat(..., coords=...)` |
| Pointwise calibration | PIT-ECDF (posterior or LOO) | `ppc_pit_ecdf`, `ppc_loo_pit_ecdf` | `az.plot_ppc_pit(dt)`, `az.plot_loo_pit(dt)` |

Guidance from BW §8.2:
- Prefer test summaries that are **ancillary or nearly so** (their distribution should not
  depend on the parameters). Summaries of subgroups defined by an omitted predictor expose
  missing structure.
- For **binary data**, density overlays and bar plots of proportions are useless (an
  intercept-only model gets the proportion right). Use PAV-adjusted calibration plots.
- For **counts with few distinct values** or sharp probability changes, KDEs oversmooth; use a
  rootogram (`yscale="sqrt"` or log).
- Aim for *severe tests*: checks that would reveal a discrepancy **if the model would mislead
  you on the question you care about**. A generic density overlay is a safeguard against gross
  misspecification, not a certificate.
- Hierarchical models have three replication scenarios (BW §6.1, §8.2): new data for existing
  groups (posterior predictive), new groups from the fitted population (mixed), and a new
  population (prior predictive). Choose the one matching how the model will be used and
  simulate it in `generated quantities`.

### Python (CmdStanPy + ArviZ 1.x)

```python
import arviz as az

dt = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive="y_rep",
    log_likelihood="log_lik",
    observed_data={"y": data["y"]},
    coords={"obs": range(data["N"])},
    dims={"y": ["obs"], "y_rep": ["obs"], "log_lik": ["obs"]},
)
az.plot_ppc_dist(dt, kind="ecdf")
az.plot_ppc_tstat(dt, t_stat="sd")
az.plot_ppc_rootogram(dt)          # counts
az.plot_ppc_pava(dt)               # binary
```

`az.from_cmdstanpy` maps `y_rep` onto `y` by stripping the `_rep` suffix; keep that naming or
pass `data_pairs`-style dims explicitly.

### R (cmdstanr + bayesplot, or rstanarm/brms `pp_check`)

```r
library(bayesplot); library(posterior)
yrep <- fit$draws("y_rep", format = "draws_matrix")   # cmdstanr
ppc_dens_overlay(y, yrep[1:50, ])
ppc_stat(y, yrep, stat = "sd")
ppc_rootogram(y, yrep)                                # counts
# binary, PAV-adjusted reliability diagram (reliabilitydiag package):
# reliabilitydiag(EMOS = E_loo(yrep_indicator, loo1$psis_object)$value, y = as.numeric(y))
ppc_pit_ecdf(y, yrep, prob = 0.99)

pp_check(fit_brms, type = "dens_overlay", ndraws = 50)  # brms / rstanarm wrapper
pp_check(fit_brms, type = "stat", stat = "sd")
pp_check(fit_brms, type = "stat_grouped", stat = "mean", group = "site")
```

### PIT and calibration

`PIT_i = F_i(y_i)`, the predictive CDF evaluated at the observation. Calibrated pointwise
predictions give uniform PITs. Display as **PIT-ECDF difference plots** with simultaneous
confidence bands (`ppc_pit_ecdf`, `az.plot_ppc_pit`, `az.plot_loo_pit`); ArviZ prints the
POT uniformity test p-value, which stays valid under the dependence induced by LOO.

Decision rule (BW §8.2–8.3): **posterior PIT uses the data twice**. When the effective number of
parameters is high relative to n (hierarchical, spline, GP models), posterior PIT can both
fake miscalibration (S-shape with too many values near 0.5) and hide it. Use **LOO-PIT**
instead, and do not read a clean posterior PIT-ECDF as evidence of calibration until LOO
agrees.

## Cross-validation: PSIS-LOO

Leave-one-out expected log predictive density,
`elpd_loo = sum_i log p(y_i | y_{-i})`, estimated by Pareto-smoothed importance sampling
without refitting. See [references/loo_cv.md](references/loo_cv.md) for details.

```r
# R: cmdstanr, loo package
library(loo)
loo1 <- fit$loo(variables = "log_lik", r_eff = TRUE)      # cmdstanr helper
# or manually
ll   <- fit$draws("log_lik", format = "draws_matrix")
r_eff <- relative_eff(exp(ll), chain_id = rep(1:4, each = nrow(ll) / 4))
loo1 <- loo(ll, r_eff = r_eff)
print(loo1); pareto_k_table(loo1); plot(loo1)

# brms / rstanarm
loo1 <- loo(fit_brms)                     # or add_criterion(fit_brms, "loo")
```

```python
# Python: ArviZ 1.x
loo1 = az.loo(dt, pointwise=True)        # needs log_likelihood group
print(loo1)
az.plot_khat(loo1)
```

### Pareto k diagnostics

The `loo` package threshold is `min(1 - 1/log10(S), 0.7)`, which is 0.7 for S ≥ 2200 draws.

| k | Meaning | Action |
|---|---|---|
| < 0.7 (S-dependent) | PSIS reliable | none |
| 0.7–1 | estimate for that point unreliable, variance finite-ish | moment matching, or refit for those folds (`reloo`) |
| > 1 | importance weights have infinite variance | refit those folds or switch to k-fold |

What to do, in order (BW ch. 9, ch. 24):
1. **Few bad points**: `loo_moment_match(fit, loo1)` (brms: `loo(fit, moment_match = TRUE)`;
   ArviZ: `az.loo_moment_match`). Cheap, usually fixes k in 0.7–1.
2. **Still bad, few points**: refit without each bad observation (`reloo = TRUE` in brms;
   `az.reloo` with a refit wrapper).
3. **Many bad points**: PSIS is the wrong tool; use `kfold(fit, K = 10)` / `az.loo_kfold`.
4. **Ask why**: high k flags observations the model finds surprising. In the roaches example
   (BW ch. 24) the Poisson model had hundreds of high k values because it was severely
   overdispersed; the fix was a negative binomial model, not a better LOO estimator. High k
   for a flexible model with n ≈ p (e.g., varying intercepts with one observation per group)
   signals that leaving one point out changes the posterior a lot; **integrated LOO**
   (marginalize the group parameter analytically or by quadrature inside `generated
   quantities`) fixes this.

### Integrated (marginalized) LOO for varying intercepts

When each group has few observations, `p(y_i | y_{-i})` under a varying-intercept model needs
the group effect integrated out. BW ch. 24 computes it inside Stan with quadrature over the
group parameter; see [references/loo_cv.md](references/loo_cv.md) for the pattern.

## Comparing models

```r
loo_compare(loo1, loo2, loo3)          # sorted; elpd_diff and se_diff relative to the best
loo_model_weights(list(loo1, loo2, loo3), method = "stacking")
```

```python
cmp = az.compare({"m1": dt1, "m2": dt2, "m3": dt3}, method="stacking")
print(cmp[["rank", "elpd", "elpd_diff", "dse", "weight", "warning"]])
az.plot_compare(cmp)
```

### Reading the difference (BW §9.4, Sivula et al. 2025)

The normal approximation `elpd_diff ± se_diff` is well calibrated only when **all** hold:
- `|elpd_diff| > 4` (models are not nearly identical),
- both models are reasonably well specified with no gross outliers,
- `n > 100`.

Decision rules:
- `|elpd_diff| < 4`: no practical or statistical difference. Proceed with either; prefer the
  one that is simpler or better matches the scientific goal. Do not report "model 2 wins".
- `|elpd_diff| > 4` and `n > 100`, models pass checks: use `elpd_diff / se_diff` as a z-score;
  `Pr(better) = Phi(z)`.
- Small n: multiply `se_diff` by 2 for a conservative estimate; consider collecting more data.
- Misspecified models (many high k, outliers): `se_diff` is inflated, so a large observed
  difference is, if anything, understated.

Never compare models by raw `elpd` values across different datasets or different `log_lik`
units. Never use WAIC for anything PSIS-LOO can do; WAIC has no reliability diagnostic.

### Avoiding overfitting to the comparison (BW §9.5)

- **Few candidate models, or one clearly best**: selection overfitting is negligible; pick it.
- **Many models with similar performance**: selecting the best overfits. Use stacking
  (`loo_model_weights(..., method = "stacking")`, `az.compare(method="stacking")`), a
  multiverse summary over the plausible set, or continuous model expansion (fit a model that
  contains both alternatives).
- Exclude scaffolds (deliberately over-simple comparison models, buggy versions) from any
  average.
- Stacking weights that differ from 0/1 indicate **heterogeneity of fit**: some observations
  prefer each model. Treat that as a pointer toward a hierarchical or mixture expansion, not
  as the end of the analysis.
- Bayes factors and marginal likelihoods are not recommended for model weighting: changing an
  irrelevant coefficient prior from normal(0,10) to normal(0,100) divides the marginal
  likelihood by roughly 10^k for k coefficients while predictions are unchanged.

### Variable selection

Selecting predictors by minimum CV error over many submodels overfits. Use projection
predictive selection (`projpred::cv_varsel`, `suggest_size`) on a well-fitting reference
model, or stop selection early. See [references/model_comparison.md](references/model_comparison.md).

## Influence of individual observations

- Flag observations with very low pointwise `elpd_loo_i` or LOO-PIT near 0 or 1.
- When comparing two models, scatter `elpd_loo_i(m1)` against `elpd_loo_i(m2)`, and plot the
  difference against a key predictor colored by outcome (BW Fig 8.10). A total difference is
  often driven by a handful of points.
- The Pareto k of each observation is itself an influence measure (distribution of importance
  weights). `az.loo_influence` and `loo::pareto_k_values` expose it.
- Grouped or time-structured data: use leave-one-group-out or leave-future-out CV; PSIS is
  much less reliable for these, so expect refits. For model comparison on time series, h-block
  CV with the joint log score is more efficient than leave-future-out.

## Other CV variants

| Situation | Method |
|---|---|
| Predict new observations in existing groups | PSIS-LOO |
| Predict **new groups** | leave-one-group-out (`kfold(fit, folds = kfold_split_grouped(K, x = group))`; `az.loo_kfold(..., group_by=)`) |
| Time series forecasting | leave-future-out; h-block CV for comparison |
| Many high k | k-fold, `K = 10` |
| Large n (> 10k) | `az.loo_subsample`, `loo_subsample()` |

## Bayesian R² and scores

- `bayes_R2(fit)` (rstanarm/brms) or compute from `mu` draws: `var(mu) / (var(mu) + sigma^2)`
  per draw, then summarize. `loo_R2` / `az.loo_r2` for the LOO version.
- `az.loo_score(dt, score_func="crps")` and `az.loo_metrics(dt, kind="rmse")` for proper
  scores on the outcome scale.

## Reporting

Report the **sequence** of models, not only the winner: a table with one row per model, its
description, `elpd_loo`, `elpd_diff` and `se_diff` versus the best, and which checks it
failed. Include the pointwise plots that motivated each expansion. Say what the difference
means for the quantity of interest, which is often "the conclusions did not change".

## References

- [references/loo_cv.md](references/loo_cv.md): PSIS-LOO mechanics, Pareto k, moment matching,
  integrated LOO Stan code, k-fold and grouped folds.
- [references/ppc.md](references/ppc.md): choosing test statistics, PIT and PAV plots, hierarchical
  replication scenarios, bayesplot and ArviZ recipes.
- [references/model_comparison.md](references/model_comparison.md): elpd difference calibration,
  stacking, multiverse, projpred, worked examples from BW (milk, sleep study, roaches).
