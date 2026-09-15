# Model Expansion, Comparison of Series, and Multiverse

Source: *Bayesian Workflow* ch. 9, §5.1, §12.4, ch. 17, ch. 24.

## Why build a series (BW §9.1, §9.3)

1. Human cognition is limited; models are measurement instruments learned gradually.
2. Time: stop when the model is good enough for the purpose.
3. Computation: no algorithm works for all models; simpler versions make diagnosis possible.
4. Disentangling computational problems, modeling problems, data quality, and bugs.

Also: placeholders get replaced; new data arrive; a well-fitting model leaves room; simple
models serve as comparisons (unadjusted vs adjusted causal estimates); computational trouble
forces a rethink.

## Topology of models (BW §9.2)

Models within a class form a partial order (AR(1) < AR(2) < ARMA(2,1); the 2^k predictor
subsets are corners of a cube). The interest is in **navigating** among neighbors and in how
parameters in neighboring models relate, not in assigning probabilities to models. Priors add a
continuous dimension between models: `normal(0, 0.01)` on a coefficient is nearly "excluded",
`normal(0, 10)` is nearly "included".

## Expansion triggers and the matching response

| Observation | Expansion |
|---|---|
| PPC: replicated sd too small (counts) | Poisson → negative binomial (start there next time) |
| PPC: too few zeros | zero-inflated or hurdle component |
| PPC grouped by an omitted variable shows extra spread | varying intercepts by that variable |
| Residuals vary by group slope | varying slopes with LKJ correlation |
| LOO-PIT S-shaped (too wide) after outliers inflate sigma | Student-t residuals |
| Predictions negative for a positive outcome | log-scale model (lognormal, gamma) with quantile-matched priors |
| Predictor is measured with error | latent true predictor with a measurement model |
| Predictor is a function of the outcome (total score) | latent-variable model (IRT) |
| Divergences in the funnel | non-centered parameterization (a computational expansion) |
| Very slow fit, implausible posterior wandering | stronger priors: the model is weakly identified |

## Roaches sequence (BW ch. 24)

| Model | Check that motivated it | Result |
|---|---|---|
| Poisson | density overlay and rootogram: far underdispersed; p_loo 273 for 4 parameters | discard |
| Negative binomial | rootogram fine; LOO-PIT fine; PAV reliability of Pr(y > 0) off | keep, improve zeros |
| Poisson + per-observation varying intercepts | PSIS-LOO fails (204 bad k); PPC uninformative for a flexible model | need integrated LOO or k-fold; elpd ≈ NB |
| Zero-inflated NB | reliability diagram improves; elpd +23 (7) over NB; p_loo 10 ≈ 9 parameters | keep |

The treatment effect ratio was similar under NB and ZINB and overconfident under Poisson.
Summarize effects that split across mixture components via predictions (`posterior_epred` with
treatment set to 0 and 1), and power-scale the prior on that derived quantity.

## Sleep study sequence (BW ch. 17)

| Model | Change | Prior adjustments |
|---|---|---|
| M1 | `Reaction ~ Days` | intercept normal(250, 100), slope normal(0, 20), sigma exponential(1/50) |
| M2 | `+ (1 | Subject)` | split sigma prior: sigma and tau_0 both exponential(1/25) |
| M3 | `+ (Days | Subject)` | tau_1 exponential(1/10), LKJ(1) |
| M3t | Student-t residuals | motivated by LOO-PIT S-shape; nu ≈ 2.6 |
| lognormal M5 | log-scale outcome | quantile-matched intercept normal(5, 0.55), slope normal(0, 0.2), sigma exponential(3) → exponential(6) split, tau_1 exponential(10) |

Under Student-t residuals the varying slopes matter more (outliers no longer inflate sigma).
Comparing a log-outcome model to a raw-outcome model requires adding the Jacobian `−log(y_i)`
to the pointwise elpd of the log model.

## Continuous expansion and predictive consistency (BW §9.7)

- Instead of choosing A vs B, fit `lambda * A + (1 - lambda) * B`.
- Build with simulation in parallel: each added feature gets a fake-data test.
- Choose priors that keep the prior predictive distribution stable as components are added
  (R2D2, regularized horseshoe, ARR2). Independent normal(0, 1) on 26 coefficients puts the
  prior on R² near 1.
- Against parsimony: if a simple model beats a complex one, the complex model was the wrong
  expansion; define a different complex model that captures what made the simple one work.

## Multiverse and selection safety (BW §9.3, §9.5)

- Plot the quantity of interest against model index for the series (BW Fig. 9.1: SATE and PATE
  under y ~ z; y ~ z + x; + x:z; + x²...). Adjusting for the imbalanced covariate moved the
  estimate; further adjustments did not.
- Multiverse: fit the plausible alternatives (preprocessing, response distribution, predictor
  sets that make scientific sense) and show the conclusion across them, filtering out models
  that fail checks. If conclusions do not change, the choice of "best" model does not matter.
- Selection among few models, or with a clear winner, does not overfit. Selection among many
  similar models does; stack or average instead.
- Interpret poorly mixed fits in the context of the simpler models already fit; it is
  acceptable to carry an imperfect model forward while the next expansion is built.
