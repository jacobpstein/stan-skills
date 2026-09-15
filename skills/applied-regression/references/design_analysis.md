# Design Analysis and Fake-Data Simulation

Source: *Regression and Other Stories* §5.5, §7.2, ch. 16.

Fake-data simulation is not about learning the real problem. It is about evaluating the
properties of the method, given an assumed generative model.

## The four-step recipe (ROS §7.2)

**1. Create the pretend world.** Fix true values for every parameter, usually from a previous
fit, and keep the real predictors.

```r
a <- 46.3; b <- 3.0; sigma <- 3.9
x <- hibbs$growth; n <- length(x)
```

**2. Simulate fake data.**

```r
y <- a + b * x + rnorm(n, 0, sigma)
fake <- data.frame(x, y)
```

**3. Fit and compare to the assumed truth.** The fit must see only `x` and `y`.

```r
fit <- stan_glm(y ~ x, data = fake, refresh = 0)
b_hat <- coef(fit)["x"]; b_se <- se(fit)["x"]
abs(b - b_hat) < b_se        # inside the 68% interval?
```

A single run checks the ballpark. It does **not** verify that the intervals are honest.

**4. Loop and check coverage.** This is the real test.

```r
n_fake <- 1000
cover_68 <- cover_95 <- rep(NA, n_fake)
for (s in 1:n_fake) {
  y <- a + b * x + rnorm(n, 0, sigma)
  fit <- stan_glm(y ~ x, data = data.frame(x, y), refresh = 0)
  b_hat <- coef(fit)["x"]; b_se <- se(fit)["x"]
  cover_68[s] <- abs(b - b_hat) < b_se
  cover_95[s] <- abs(b - b_hat) < 2 * b_se
}
c(mean(cover_68), mean(cover_95))
```

With n = 16 the book got 62.8% and 92.8%, not 68% and 95%. The cause is using normal multipliers
when the right reference is `t` with `n - 2` degrees of freedom:

```r
t_68 <- qt(0.84, n - 2); t_95 <- qt(0.975, n - 2)
```

After which coverage matches. This loop is the standard way to check that a procedure's stated
uncertainty is real, and it is the same machinery as simulation-based calibration (see
[stan-testing](../../stan-testing/SKILL.md)).

## Why simulate

1. It forces you to state your assumptions in numbers: how big could this effect plausibly be,
   what correlations are realistic.
2. It reveals the method's repeated-sampling behavior. You can simulate from a **richer** process
   than the model you fit, to see what misspecification costs.
3. It debugs code: with large n or small residual sd, the fit must recover the truth. If it does
   not, you have a bug.
4. It is required for design analysis.

## Power, Type S, and Type M

Statistical power is the pre-data probability of "significance" given an assumed true effect. The
conventional 80% target is the wrong goal.

The winner's curse: when the true effect is small relative to the standard error, a significant
result from a low-power study is doubly misleading. It has a substantial probability of the
**wrong sign** (Type S error) and its magnitude is inflated (Type M error), often many-fold.

Worked example: a true effect of at most 2 percentage points with SE 8.1 gives power under 6%.
Conditional on significance, the estimate has at least a 24% chance of the wrong sign and is
guaranteed to be more than 8 times too large.

In the beauty/sex-ratio case, with SE 3% and a literature-informed true effect below 0.5 points,
the Type S error rate is 31-42% and any significant estimate must be 12-30 times the true effect.

**A low-power study is not a small chance of success. It is close to guaranteed to produce either
nothing or something misleading.**

## Sample-size formulas

All normal-approximation; the conservative bound for a proportion is `sqrt(p(1-p)/n) <= 0.5/sqrt(n)`.

| Goal | Formula |
|---|---|
| Target SE for a proportion | `n = p(1-p) / se^2`, conservatively `(0.5/se)^2` |
| 80% power vs a reference proportion | `n = p(1-p) (2.8 / (p - p0))^2` |
| 80% power comparing two proportions | `n = 2 p̄(1-p̄) (2.8 / (p1 - p2))^2` total |
| Target SE, continuous | `n = (sigma / se)^2` |
| 80% power vs a reference mean | `n = (2.8 sigma / (theta - theta0))^2` |
| 80% power, two-group difference | `n = (5.6 sigma / delta)^2` total |

The 2.8 is `qnorm(0.975) + qnorm(0.8) = 1.96 + 0.84`. For very small samples replace it with the
`t` equivalent (`qt(0.975, df) + qt(0.8, df)`, about 3.1 at df = 10).

Increasing the **effect size** (stronger dose, more responsive population) beats increasing n,
since SE scales as `1/sqrt(n)` but required n scales as `1/effect^2`. Published effects are
systematically overestimates, both because they come from idealized conditions and because of
publication filtering.

Pre-treatment covariates that genuinely predict the outcome reduce the residual sd and therefore
the required n: if they explain half the variance, n falls by roughly half. Use the regression's
residual sd in the formulas, not the raw within-group sd.

Post-hoc power calculations that plug in an estimated (and uncertain) effect are close to
meaningless as a rigorous exercise, though useful for intuition.

## Interactions need much larger samples

An interaction is a difference of differences, so its standard error is roughly **twice** that of
a main effect of the same size.

- Same size as the main effect: **4× the sample size** for equal precision.
- Half the size (often realistic): **16×**.

Consequences:
1. Studies powered for main effects are almost always underpowered for interactions.
2. Failing to find a main effect and then hunting through interactions is a forking-paths trap
   built on noisier information than it appears.
3. Code the inputs so the large, well-powered comparison is the main effect.
4. When interactions matter, use partial pooling and informative priors rather than demanding
   that an interval exclude zero.

## Design analysis for causal questions (ROS §16.6)

Simulate potential outcomes, then the assignment mechanism, then the observed outcome, then fit
the candidate analyses.

```r
n <- 100
true_ability   <- rnorm(n, 50, 16)
x              <- true_ability + rnorm(n, 0, 12)     # pre-test, correlated with outcome
y_if_control   <- true_ability + rnorm(n, 0, 12)
y_if_treated   <- y_if_control + 5                   # true effect

# (a) randomized assignment
z <- sample(rep(c(0, 1), n / 2))
# (b) confounded assignment: weaker students more likely treated
# z <- rbinom(n, 1, invlogit(-(x - 50) / 20))

y <- ifelse(z == 1, y_if_treated, y_if_control)
fit_simple   <- stan_glm(y ~ z, data = data.frame(y, z), refresh = 0)
fit_adjusted <- stan_glm(y ~ z + x, data = data.frame(y, z, x), refresh = 0)
```

Wrap it in a function and loop it; a single run's point estimate says nothing.

Results the book demonstrates:
- Under randomization, both analyses are unbiased, but adjusting for a pre-treatment covariate
  that genuinely correlates with the outcome cuts the standard error substantially (about a third
  in the worked example). A covariate simulated **independently** of the outcome gives no such
  gain, so the simulation must reproduce the real covariate-outcome correlation to be informative.
- Under confounded assignment, the unadjusted comparison is badly biased and can have the wrong
  sign (−6.4 when the truth is +5.0), while adjusting for the covariate recovers about 4.6.
- That rescue is simulation-specific: with a nonlinear covariate-outcome relationship or an
  unmeasured confounder, the adjusted estimate can still be badly biased.

## Bootstrap caveats

Resampling requires deciding **what** to resample: residuals rather than (x, y) pairs in some
regression settings; blocks for time series; clusters for multilevel data. The bootstrap does not
fix separation, since a resample of data with an empty cell still has an empty cell.
