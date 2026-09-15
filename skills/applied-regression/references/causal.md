# Causal Inference with Regression

Source: *Regression and Other Stories* ch. 18-21, §6.3, §19.6.

## Comparisons, then effects

A regression coefficient always describes a **between-unit comparison**. It describes a causal
**effect** only under additional assumptions. The safest interpretation is the comparison; state
the causal assumptions explicitly when you make the stronger claim.

For `earnk = -26.0 + 0.6 * height + 10.6 * male`:
- Valid: among people of the same sex differing by an inch in height, earnings differ by about
  $600 on average.
- Not valid without assumptions: "the effect of height is $600."

"What if this person were an inch taller" may not even be a well-defined question without saying
how the change would come about.

## What randomization buys

With randomized `z`, the potential outcomes are independent of assignment, so:

- `y ~ z` gives an unbiased treatment effect.
- `y ~ z + x` with **pre-treatment** `x` is also unbiased, and has a smaller standard error when
  `x` predicts `y`.

Randomization licenses causal claims about the **randomized variable only**. Other coefficients
in the same regression are still comparisons.

## Do not adjust for post-treatment variables (ROS §19.6)

Do not include any variable measured after treatment assignment: a mediator, an intermediate
outcome, anything the treatment could have changed. **This holds even under randomization.**

Why: conditioning on a post-treatment variable `q` compares treated and control units within
levels of `q`, but `q` itself was shifted by the treatment. Two families with the same observed
`q = 0.5` are not comparable: the treated one must have had a lower underlying `q` potential,
since treatment raised it. You are comparing different kinds of units, and the treatment
coefficient absorbs that selection.

The book's worked example produces a treatment coefficient of −1.5 ("a negative direct effect on
IQ") when the true total effect is a uniformly positive +10 to +15 points.

**Mediation analysis** by adding the mediator as a predictor and reading `tau*` as "the direct
effect net of the mediator", or `tau*/tau` as "the fraction mediated", is generally not valid.

What can be estimated: **principal strata**, defined by each unit's joint potential outcomes for
the mediator (what `q` would be under treatment and under control). These are pre-treatment
characteristics and are legitimate to condition on, but they are unobserved, so you need extra
assumptions or an instrumental-variables design (ch. 21).

Corollary for observational work: adjusting for a post-treatment variable adds this
selection problem on top of whatever confounding you already had.

## Observational studies

Assume ignorability conditional on measured confounders, then:

1. **Choose the treatment variable deliberately.** Do not answer several causal questions from
   one big regression.
2. **Check balance**: compare the distribution of each pre-treatment covariate between treated
   and control.
3. **Check overlap**: is there common support? Where there is none, no amount of modeling helps;
   restrict the analysis to the region of overlap and say so.
4. **Adjust** by regression, subclassification, matching, or weighting. Regression adjustment is
   a model-based extrapolation, so it is only as good as the functional form where overlap is
   thin.
5. Report the estimand: sample average treatment effect, population average, effect on the
   treated. They differ when effects vary.

```r
# average treatment effect from a fitted model, per posterior draw
epred1 <- posterior_epred(fit, newdata = transform(d, z = 1))
epred0 <- posterior_epred(fit, newdata = transform(d, z = 0))
ate <- rowMeans(epred1 - epred0)
quantile(ate, c(0.05, 0.5, 0.95))
```

Computing the contrast from predictions rather than reading a coefficient is what makes it work
with interactions, nonlinear links, and mixture components.

In Stan:

```stan
generated quantities {
  real ate;
  {
    vector[N] mu1 = inv_logit(alpha + beta_z * 1 + X * beta);
    vector[N] mu0 = inv_logit(alpha + beta_z * 0 + X * beta);
    ate = mean(mu1 - mu0);
  }
}
```

## Varying treatment effects

Treatment effects usually vary. Interactions between treatment and pre-treatment covariates are
the way to model that, but interactions need much larger samples (see
[design_analysis.md](design_analysis.md)): 4× for an interaction the size of the main effect, 16×
for one half the size. Use partial pooling and informative priors rather than demanding
significance.

## Designs that identify effects without full ignorability

| Design | Idea | Key assumption |
|---|---|---|
| **Instrumental variables** | an instrument affects treatment but not the outcome except through treatment | exclusion restriction, monotonicity; estimates the local effect on compliers |
| **Regression discontinuity** | assignment determined by a threshold on a running variable | continuity of potential outcomes at the threshold; local to the cutoff |
| **Difference in differences** | compare changes over time between groups | parallel trends absent treatment |
| **Within-unit comparison** | use repeated measurements of the same unit | no time-varying confounding |

Each is a targeted design, not something you read off a large regression. Each estimates a
specific, often local, estimand; say which.

## Checklist before making a causal claim

- Is the treatment variable well defined as an intervention?
- Was it randomized? If not, what makes assignment ignorable given the measured covariates?
- Are all adjustment variables **pre-treatment**?
- Is there overlap between treated and control on those variables?
- Does the estimand match the question (sample, population, treated)?
- Have you computed the contrast from predictions rather than reading a single coefficient?
- Is only one coefficient in this regression being given a causal reading?

Rarely can more than one coefficient in a multi-predictor observational regression be given a
defensible causal reading at the same time. Even the arsenic well-switching regression, a purely
descriptive model, should not be given a "holding everything else fixed" causal reading for each
of distance, arsenic, and education simultaneously.
