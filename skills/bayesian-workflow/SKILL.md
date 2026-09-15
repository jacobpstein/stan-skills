---
name: bayesian-workflow
description: >
  Load whenever the user is building, checking, expanding, or reporting a Bayesian model
  (Stan, brms, rstanarm, or any PPL) and the question is about process rather than a single
  API call: where to start, what to check first, how to simulate fake data, when to expand the
  model, how to compare a series of models, and how to write up the analysis. Implements the
  Gelman, Vehtari, McElreath et al. Bayesian Workflow (2026). Triggers include: Bayesian
  workflow, statistical workflow, start simple, iterative model building, fake-data simulation,
  simulated data, parameter recovery, prior predictive check, posterior predictive check, model
  expansion, series of models, sequence of models, model criticism, model checking, reporting a
  Bayesian analysis, "does my model make sense", "what should I do next".
---

# Bayesian Workflow

Bayesian **inference** is fitting one model and using it. Bayesian **data analysis** adds
model building, checking, and improvement. Bayesian **workflow** adds the comparison of a
series of models under realistic computation, in order to understand each one. In practice you
will fit many models for any problem; some will be wrong in retrospect, and those are
unavoidable steps toward the ones worth reporting.

This skill is the map. The other skills are the territory:
[stan-modeling](../stan-modeling/SKILL.md) (writing the program),
[stan-priors](../stan-priors/SKILL.md), [stan-computation](../stan-computation/SKILL.md),
[stan-model-evaluation](../stan-model-evaluation/SKILL.md),
[applied-regression](../applied-regression/SKILL.md), [stan-testing](../stan-testing/SKILL.md).

## The loop

```
Pick an initial model
  → Model structure: prior predictive check, simulate, relate to scientific goals
  → Correctness of computation: fit simulated data, SBC, software practice
  → Fit to real data: diagnose convergence
      ↳ computation invalid → Diagnose & address computational issues
                               (simplify, fewer iterations, subset, reparameterize,
                                add prior info, plot intermediates, check multimodality)
  → Visualize and check the fitted model: PPC, cross-validation, influence of points, prior
      ↳ fit problematic or contradicts domain knowledge → Modify the model
                               (replace component, expand, modify priors, add data,
                                use an approximation, pick a new starting model)
  → Compare and combine models: compare inferences, multiverse, stacking
  → Use the model: postprocess, predict, poststratify, causal inference
```

Every "accepted" state is provisional. The diagram (BW Fig. 2.1) shows paths an analysis *may*
take; no real analysis takes all of them.

## Four scenarios you are in at any moment (BW §2.3)

1. The model contains the true data-generating process (only when fitting simulated data).
2. The model is not true but captures the important aspects and gives reasonable inferences.
   This is the goal.
3. The model is bad: it misses or mischaracterizes an important aspect.
4. There is a conceptual or programming error: the coded model is not the intended model.

The typical path starts at 3, dips into 4 and 1, and works toward 2. Treat every model as
provisional. If a fit is slow to converge, assume 3 or 4 and investigate immediately; do not
assume 2 and tweak sampler settings.

## Step by step

### 1. Pick an initial model (BW §5.1)

- Adapt a template: a textbook model, a case study, a published analysis of a similar problem.
- Start linear and normal unless the outcome type forbids it: it is often a good approximation,
  you will compare against it anyway, and it is easiest to debug. Start with negative binomial
  rather than Poisson for counts.
- Two directions both work: start simple and add, or start with the big model in mind and strip
  it to something that fits. Prefer whichever is easiest to build, debug, and expand.
- Every component is a placeholder: normal → t or mixture; linear → spline or GP; exact
  predictor → measurement-error model; weak prior → informative prior.
- Enumerate the assumptions, justify each, and ask where a violation would matter (BW §5.2,
  eight schools).
- Write the **data model** (the generative story), even where the Stan code implements a
  marginalized likelihood (mixtures, censoring). You need the generative form for checks.

### 2. Check the model structure before real data

- **Prior predictive check**: simulate ~10 datasets from the prior with the real design and plot
  them next to the real data. Adjust priors that produce impossible or absurd data. See
  [stan-priors](../stan-priors/SKILL.md).
- **Fake-data check** (BW §6.3): pick plausible parameter values, simulate a dataset with the
  same size and structure, fit, and verify that (a) the posterior differs from the prior for
  every parameter, (b) the true values fall inside reasonable intervals, (c) recovery holds in
  the regions of parameter space you care about. If it fails, simplify until it works. Details
  in [references/fake_data.md](references/fake_data.md).
- Design the simulation with an aim (e.g., can the design identify a negative discrimination?),
  not a grid over everything. Draw other parameters from the posterior of a real fit, or fix
  them at sensible values, rather than from very wide priors.
- **Try to break it**: construct a dataset on which the model should fail (BW §4.5). Knowing
  where a model fails is how you understand it.

### 3. Fit fast, fail fast (BW ch. 11–12)

- Short runs first (4 chains × 200 iterations); R-hat ≈ 1.1 is acceptable during exploration.
- Fix inits and scale (unit-scale parameters, `init = 0.1`), use proper priors, run pedantic
  mode.
- If diagnostics fail, look at the posterior and the model before touching `adapt_delta`. See
  [stan-computation](../stan-computation/SKILL.md).
- Save draws immediately after the first successful fit.

### 4. Visualize and check the fitted model (BW ch. 8)

- Do the parameter estimates make sense against domain knowledge?
- Posterior predictive checks with test statistics aimed at what could go wrong; group by
  variables not in the model.
- LOO cross-validation and Pareto k; LOO-PIT for flexible models; pointwise influence.
- Prior and likelihood sensitivity by power-scaling.
- Look at the quantities the model is *for* (predictions, contrasts, poststratified means), not
  only the named parameters. See [stan-model-evaluation](../stan-model-evaluation/SKILL.md).

### 5. Modify and expand (BW ch. 9)

Triggers for expansion: new data, a failed check, computational trouble, or a scientific
question the current model cannot answer.

- Change **one thing at a time**, and keep the previous model as the comparison point.
- Expand the fake-data simulation in parallel so each new feature is tested.
- Prefer continuous expansion (a model that contains both alternatives) over discrete choice.
- Keep priors predictively consistent as components are added (per-coefficient scale shrinking
  with the number of predictors, R2D2, regularized horseshoe).
- When a posterior predictive check points at the data rather than the model, check the data.
  In BW ch. 4, plotting fitted curves for all 24 exam items revealed three answer-key coding
  errors.

The multiple-choice exam case study (BW ch. 4) is the canonical sequence: per-item logistic
regression → standardized predictor with weakly informative priors → guessing floor →
multilevel over items → correlated intercepts and slopes → fully generative item-response model
→ discrimination parameters → simulation experiment → breaking the model. Stan code for each step
is in [references/case_study_exam.md](references/case_study_exam.md).

### 6. Compare, combine, decide (BW §9.4–9.6)

- Compare on LOO elpd among models that pass the checks; `|elpd_diff| < 4` means no practical
  difference.
- One clear winner: select it. Many similar: stack, multiverse, or expand.
- Plot the quantity of interest against model index for the whole series (BW Fig. 9.1). That
  plot *is* the finding.
- Do not average in scaffolds. Do not use Bayes factors for routine comparison.

### 7. Use and report (BW ch. 7, ch. 15)

- Simulate first, summarize last: never compute a function of posterior means; compute the
  function per draw. Put predictions in `generated quantities`.
- Report the **series** of models: what each expansion fixed or did not change, the prior and
  posterior predictive findings that motivated each step, and the final inferences with the
  digits justified by MCSE (usually two significant digits).
- Report point estimates as mean ± sd or median ± MAD-sd, with central 50% and 90% intervals;
  intervals rather than ± sd for skewed scale parameters; HPD when a posterior sits against a
  boundary.
- Prediction, poststratification, and causal contrasts are computations on posterior draws.
  Generalizing to a new population needs a model for how the population differs (regression on
  group characteristics plus poststratification).

## Reporting template

```
| Model | Change from previous | Motivation (which check) | elpd_loo diff (se) | QoI mean (sd) |
|-------|----------------------|--------------------------|--------------------|---------------|
| M1    | complete pooling     | starting template        | -77.8 (20.9)       | 11.3 (1.9)    |
| M2    | + varying intercepts | PPC by subject           | -12.7 (9.8)        | 11.4 (1.1)    |
| M3    | + varying slopes     | subject-level residuals  | 0                  | 11.3 (2.0)    |
| M3t   | Student-t residuals  | LOO-PIT S-shape          | +41.7 vs M3        | ...           |
```

State what changed for the conclusion and what did not; "the treatment effect estimate was
stable across all models that passed checks" is a result.

## Software habits that make the loop possible (BW ch. 15)

- Git from day one; one file per distinct model; version data and figures; tag milestones.
- Self-contained scripts that run in a clean environment; a pipeline tool (`targets`,
  `Snakemake`, `make`) for simulation studies.
- Test as you go: fake-data recovery per component; SBC for the final model; readable Stan
  with constrained declarations instead of comments. See [stan-testing](../stan-testing/SKILL.md).

## References

- [references/workflow_steps.md](references/workflow_steps.md): the full diagram box by box,
  with which skill and function covers each step.
- [references/fake_data.md](references/fake_data.md): fake-data checks, simulation experiment
  design, the two-step posterior-as-truth procedure, hierarchical replication scenarios.
- [references/case_study_exam.md](references/case_study_exam.md): the BW ch. 4 model sequence
  with Stan code and the lesson from each step.
- [references/model_expansion.md](references/model_expansion.md): topology of models, expansion
  triggers, continuous expansion, predictive consistency, multiverse, and the roaches and sleep
  study sequences.
