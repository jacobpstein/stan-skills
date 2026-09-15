# Poststratification and MRP

Source: *Regression and Other Stories* §17.1-17.2; *Bayesian Workflow* ch. 7.

## The idea

If the sample is unrepresentative but you (a) fit a regression including the variables on which it
is unrepresentative and (b) know the population distribution of those variables, you can correct
the estimate: predict for every population cell, then average the cell predictions weighted by
population counts.

The poststratified estimate is

```
population estimate = sum_j (N_j / N) * ybar_j
```

not the naive `sum_j (n_j / n) * ybar_j`, which just reproduces the raw sample mean.

In the 2016 CBS example, the raw survey gave 45% Trump support; the sample was
disproportionately Democratic, and reweighting to the population's 33/36/31 party split gave 47%.

## Three-step recipe

**1. Fit a regression on all the adjustment variables.**

```r
fit <- stan_glm(vote ~ factor(pid), data = poll, refresh = 0)
```

**2. Build the poststratification table**: one row per population cell, with a column of known
population counts or proportions.

```r
poststrat <- data.frame(
  pid = c("Republican", "Democrat", "Independent"),
  N   = c(0.33, 0.36, 0.31)
)
```

**3. Predict cell expectations and take the weighted average per posterior draw.**

```r
epred <- posterior_epred(fit, newdata = poststrat)        # n_sims x J
est   <- epred %*% poststrat$N / sum(poststrat$N)         # length n_sims
c(mean(est), mad(est))
```

Use `posterior_epred`, not `posterior_linpred` or `predict`: you want `E[y]` per cell. For linear
models they coincide; for GLMs they do not, and for a binary outcome the cell quantity you want is
a **probability**.

## Adding known unmodeled uncertainty

If you believe the posterior SE understates reality (polls shift, there is house effect bias),
add it explicitly:

```r
est2 <- est + rnorm(length(est), 0, 0.02)     # 2 extra percentage points
```

Same mean, wider interval. This is an honest way to acknowledge error sources outside the model.

## Multiple factors and MRP

Extend the table to one row per **combination** of factor levels, with the population count for
each combination. Everything downstream is identical.

With many factors the table explodes: the Xbox survey example used eight factors and 13,824
cells. Sparse cells make unpooled estimates unstable, so fit the regression as a **multilevel**
model and let the group-level parameters partially pool. That combination is multilevel
regression and poststratification, MRP.

```r
fit <- stan_glmer(
  vote ~ (1 | state) + (1 | age) + (1 | educ) + (1 | ethnicity) +
         (1 | age:educ) + male + prev_vote_share,
  family = binomial(link = "logit"), data = poll
)
epred <- posterior_epred(fit, newdata = poststrat)
est   <- epred %*% poststrat$N / sum(poststrat$N)
```

Include state-level predictors (previous election results, region) as group-level covariates so
that states with little data are shrunk toward an informed prediction rather than the grand mean.

In Stan, the same thing is a hierarchical logistic regression plus a weighted average in
`generated quantities`:

```stan
data {
  int<lower=0> J;                 // poststratification cells
  matrix[J, K] X_pop;
  array[J] int<lower=1> cell_state;
  vector[J] N_pop;
}
generated quantities {
  vector[J] theta_cell = inv_logit(alpha[cell_state] + X_pop * beta);
  real theta_pop = dot_product(N_pop, theta_cell) / sum(N_pop);
}
```

## What it does and does not do

The Xbox 2012 poll was drawn from gamers: younger, more male, less party-affiliated than the
electorate, and its raw daily estimates showed Obama losing badly. After regression on
demographics and politics and poststratification against a population table, the corrected daily
series tracked the actual 52% two-party outcome and matched contemporaneous telephone polls.

But in 2016 a similarly adjusted online poll still missed by several points in some states,
because the adjustment variables did not capture the real sample-population discrepancy that
year. Poststratification corrects for the variables you adjust on and nothing else.

## Practical complications

1. **Binary outcomes need logistic regression**, so the poststratified quantity is a probability
   per cell.
2. **Cell counts are themselves estimates** (from a census or a previous election) with their own
   uncertainty; propagate it when it matters.
3. **Continuous adjustment variables** are usually discretized into cells, with the regression
   supplying a smooth prediction within each.
4. **Pooling over time**: fit one multilevel model to many days with time-varying coefficients
   rather than a separate regression per day.

## Generalizing beyond surveys

The same machinery answers "what would the average outcome be in a different population": fit a
model with the relevant predictors, get the target population's composition, and average the
predictions over it. It is also how you turn an individual-level causal model into a
population-average treatment effect, and how *Bayesian Workflow* frames generalization from the
eight schools to new schools: regress on school characteristics, then poststratify.
