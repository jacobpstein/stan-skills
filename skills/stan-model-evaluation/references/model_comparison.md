# Model Comparison, Averaging, and Selection

Source: *Bayesian Workflow* ch. 9, ch. 24, ch. 28; Sivula et al. (2025); Yao et al. (2018);
McLatchie & Vehtari (2024); Piironen, Paasiniemi & Vehtari (2020).

## The elpd difference and its standard error

`elpd_loo(M_a, M_b) = Σ_i [log p_a(y_i | y_{-i}) − log p_b(y_i | y_{-i})]`, with

`SE = sqrt( n/(n−1) Σ_i (d_i − d̄)² )` over the pointwise differences `d_i`.

`loo_compare` reports each model relative to the best (`elpd_diff ≤ 0`, `se_diff`). ArviZ
`az.compare` reports `elpd`, `elpd_diff`, `dse`, `weight`, and `warning` (True when any Pareto k
is bad).

### When the normal approximation is calibrated

All three must hold (Sivula et al. 2025):

1. `|elpd_diff| > 4`. When the models make nearly identical predictions the pointwise
   differences are tiny and skewed; `se_diff` is underestimated. The good news: such a small
   difference is not practically meaningful either way.
2. Both models are reasonably well specified; no gross outliers dominating the pointwise
   differences; finite variance (check Pareto k of both).
3. `n > 100`.

When they hold, `Pr(M_a better) ≈ Φ(elpd_diff / se_diff)`. With small n, multiply `se_diff` by 2
(Bengio & Grandvalet heuristic) for a conservative statement, and consider collecting more data.

### Worked comparisons from the book

| Case | Result | Reading |
|---|---|---|
| Primate milk, n = 17, M4 (both predictors) vs M1 | +4.2 (2.4) | clears the 4 threshold but n small → doubled SE gives Pr > 0.81; collinear predictors make marginal posteriors overlap zero while the joint does not; predictive comparison is the right tool |
| Sleep study, M3 varying slopes vs M2 varying intercepts | −12.7 (9.8) for M2 | two outliers → refit with Student-t; under t the slope structure matters more: M2t −45.4 (8.5) |
| Roaches Poisson vs NB | −4633 (685) for Poisson | misspecified Poisson inflates SE; the difference is so large it does not matter |
| Roaches NB vs ZINB | −23.0 (7.0) for NB | clear, and the PAV reliability diagram shows *where* ZINB helps |
| Roaches ZINB with vs without treatment | −8.4 (4.6) without | Pr ≈ 0.96 that treatment improves prediction |
| Roaches sqrt vs spline for baseline | −2.4 (3.0) | no practical difference, proceed with either |

## Interpreting a comparison table

```r
loo_compare(fit1, fit2, fit3)
#        elpd_diff se_diff
# fit3     0.0       0.0
# fit2   -12.7       9.8
# fit1   -77.8      20.9
```

- Row order is by `elpd_loo`; the best model is the reference.
- Report `elpd_diff` with `se_diff` in the text, not the raw `elpd_loo` values.
- Look at `p_loo` per model; a `p_loo` far above the parameter count flags misspecification.
- Check `pareto_k_table()` for every model before believing the table.

## Selection vs averaging vs expansion

Decision rules (BW §9.5–9.6):

- **One model clearly best, or few candidates**: select it. The overfitting from selection is
  negligible (McLatchie & Vehtari 2024). Discarding a model that fails PPC and loses badly on
  LOO is joint inference, not cheating.
- **Several models with similar elpd**: do not pick the single best. Options:
  - **Stacking**: `loo_model_weights(list(loo1, loo2, loo3), method = "stacking")`;
    `az.compare(..., method="stacking")` (default). Weights minimize the LOO log score of the
    mixture predictive. Pseudo-BMA+ (`method = "pseudobma"`, `"BB-pseudo-BMA"`) is the
    alternative when you want Bayesian-bootstrap uncertainty on weights.
  - **Multiverse**: report the quantity of interest across all plausible models (plot it against
    model index), possibly filtering out models that fail checks.
  - **Continuous expansion**: fit a model that contains both alternatives (e.g., `λA + (1−λ)B`,
    or a hierarchical model when stacking weights show heterogeneity by observation).
- Never average in scaffolds (deliberately too-simple comparison models, buggy versions).
- Stacking weights far from 0/1 mean different observations prefer different models: a signal to
  improve the model, not a final answer.
- **Bayes factors**: not recommended for routine comparison. The marginal likelihood depends on
  prior scale even when the prior does not affect predictions (normal(0,10) → normal(0,100) on k
  coefficients divides it by ~10^k) and is unstable under misspecification.

### Averaged predictions

```r
w <- loo_model_weights(list(loo1, loo2), method = "stacking")
yrep <- rbind(posterior_predict(fit1)[sample(S, round(w[1] * S)), ],
              posterior_predict(fit2)[sample(S, round(w[2] * S)), ])
```

```python
cmp = az.compare({"m1": dt1, "m2": dt2})
mixed = az.extract(dt1, group="posterior_predictive", num_samples=int(cmp.loc["m1","weight"]*S))
# concatenate with the m2 draws weighted analogously
```

## Plotting models in relation to each other

Plot the quantity of interest (treatment effect, LD50, predicted rate) with ±1 and ±2 sd bars
against model index for the sequence of models (BW Fig. 9.1). It shows which expansion moved
the inference and which did not; that is the report.

```python
import matplotlib.pyplot as plt
means = [az.extract(d)["ate"].mean().item() for d in dts]
sds   = [az.extract(d)["ate"].std().item() for d in dts]
plt.errorbar(range(len(dts)), means, yerr=sds, fmt="o"); plt.xticks(range(len(dts)), names)
```

## Variable selection

Selecting the subset of predictors with the lowest CV error overfits badly when there are many
candidates (Piironen & Vehtari 2017). Prefer:

- **Projection predictive selection** (`projpred`): fit a good *reference* model with all
  predictors and a regularizing prior (regularized horseshoe or R2D2), then project it onto
  submodels and pick the smallest submodel whose predictive performance matches the reference:

```r
library(projpred)
ref <- get_refmodel(fit_brms)           # or rstanarm fit
cvvs <- cv_varsel(ref, method = "forward", cv_method = "LOO", validate_search = TRUE)
plot(cvvs, stats = c("elpd", "rmse"), deltas = TRUE)
nsel <- suggest_size(cvvs)
proj <- project(ref, nterms = nsel)
```

  Use `validate_search = TRUE` so the reported performance accounts for the search.
- **Stop early**: estimate the magnitude of selection-induced overfitting (McLatchie & Vehtari
  2024) and stop adding terms once improvements are within that noise.
- **Do not fit all 2^k subsets**; many are scientifically nonsensical and the multiverse of
  results is not a bracket on total uncertainty.

## Predictive consistency when expanding

Choose priors so that adding components keeps the prior predictive distribution similar:
R2D2-type priors keep the implied prior on R² stable as predictors are added; regularized
horseshoe encodes "few large effects"; ARR2 does the same for autoregressive time series. Weak
independent normal priors on many coefficients silently push the prior on R² toward 1 (BW
Fig. 8.13). See [stan-priors](../../stan-priors/SKILL.md).

## Against parsimony

Statistics is not physics; a simple model that beats a complex one usually means the complex
model was the wrong expansion, not that simplicity is a virtue. The goal is a model that
"unfolds" with more data and shrinks toward the prior with little data. Build up from simple
models for engineering reasons (debuggability, computation, understanding), not because
simpler models are more likely true.
