# Case Study: Building Up an Item-Response Model (BW ch. 4)

24 multiple-choice items, 32 students, 4 options each. Question: do the items discriminate
between weak and strong students? This sequence illustrates almost every step of the workflow.
Stan programs follow the book (current syntax).

## Model 1: per-item logistic regression, flat priors

```stan
data {
  int J;
  array[J] int<lower=0, upper=1> y;
  vector[J] x;                 // student's total score
}
parameters {
  real a, b;
}
model {
  y ~ bernoulli_logit(a + b * x);
}
```

Works for one item but shows strong negative posterior correlation between `a` and `b` because
x (11–23) is far from zero. Looping over all 24 items **fails for the two items everyone
answered correctly**: the likelihood allows an arbitrarily large intercept, the posterior is
improper, and HMC drifts to infinity (complete separation).

**Lesson:** flat priors plus sparse binary data give an improper posterior.

## Model 2: standardized predictor and weakly informative priors

```stan
data {
  int J;
  array[J] int<lower=0, upper=1> y;
  vector[J] x;
  real mu_a, mu_b;
  real<lower=0> sigma_a, sigma_b;
}
transformed data {
  vector[J] x_adj = (x - mean(x)) / sd(x);
}
parameters {
  real a, b;
}
model {
  a ~ normal(mu_a, sigma_a);   // normal(0, 5): average student Pr(correct) in (0.007, 0.993)
  b ~ normal(mu_b, sigma_b);   // normal(0, 5): kept symmetric to detect negative discrimination
  y ~ bernoulli_logit(a + b * x_adj);
}
```

Now `a` is the log-odds for an average student and `b` the change for one sd of ability.
Posterior correlation drops; priors barely affect the fit. Plotting fitted curves for all 24
items revealed that item N had **negative** discrimination and E and R near zero. The data were
checked: **three answer-key coding errors**. Recoded and refit.

**Lesson:** standardize predictors so priors are interpretable; posterior predictive plots find
errors in the data, not only the model.

## Model 3: guessing floor

```stan
  y ~ bernoulli(0.25 + 0.75 * inv_logit(a + b * x_adj));
```

Bounded between 0.25 and 1. Easy to write, impossible in a GLM package. But with 32 students,
several items now show substantial posterior probability of negative discrimination: once
guessing is allowed, "they got it right by luck" becomes a plausible story. Mixture-like models
average over alternative explanations and are unstable with sparse data.

**Lesson:** a more realistic component can increase uncertainty; that motivates pooling.

## Model 4: multilevel over items

```stan
data {
  int N, J, K;
  array[N] int<lower=1, upper=J> student;
  array[N] int<lower=1, upper=K> item;
  array[N] int<lower=0, upper=1> y;
  vector[J] x;
  real mu_mu_a, mu_mu_b;
  real<lower=0> sigma_mu_a, sigma_mu_b, mu_sigma_a, mu_sigma_b;
}
transformed data {
  vector[J] x_adj = (x - mean(x)) / sd(x);
}
parameters {
  real mu_a, mu_b;
  real<lower=0> sigma_a, sigma_b;
  vector<offset=mu_a, multiplier=sigma_a>[K] a;   // non-centered via offset/multiplier
  vector<offset=mu_b, multiplier=sigma_b>[K] b;
}
model {
  a ~ normal(mu_a, sigma_a);
  b ~ normal(mu_b, sigma_b);
  mu_a ~ normal(mu_mu_a, sigma_mu_a);            // normal(0, 5)
  mu_b ~ normal(mu_mu_b, sigma_mu_b);            // normal(0, 5)
  sigma_a ~ exponential(1 / mu_sigma_a);         // mean 5 logits
  sigma_b ~ exponential(1 / mu_sigma_b);
  y ~ bernoulli(0.25 + 0.75 * inv_logit(a[item] + b[item] .* x_adj[student]));
}
```

Long format (`student[n]`, `item[n]`) handles unbalanced data. Result: `mu_a ≈ 0.8`,
`sigma_a ≈ 1.6` (items vary a lot in difficulty), `mu_b ≈ 1.1`, `sigma_b ≈ 0.2` (slopes hardly
vary; the apparent variation in unpooled fits was noise). Reading the summary: median as point
estimate, MAD-sd, q5/q95, R-hat ≈ 1, ESS > 100.

**Lesson:** partial pooling stabilizes sparse per-item estimates; hyperparameter naming
(`mu_mu_a`, `sigma_mu_a`) is typo-prone, so keep it systematic.

## Model 5: correlated intercepts and slopes

```stan
parameters {
  vector[2] mu_ab;
  vector<lower=0>[2] sigma_ab;
  array[K] vector[2] e_ab;
  cholesky_factor_corr[2] L_ab;
}
transformed parameters {
  vector[K] a, b;
  for (k in 1:K) {
    a[k] = mu_ab[1] + sigma_ab[1] * e_ab[k][1];
    b[k] = mu_ab[2] + sigma_ab[2] * e_ab[k][2];
  }
}
model {
  e_ab ~ multi_normal_cholesky(rep_vector(0, 2), L_ab);
  mu_ab ~ normal(mu_mu_ab, sigma_mu_ab);
  sigma_ab ~ exponential(1 ./ mu_sigma_ab);
  L_ab ~ lkj_corr_cholesky(1);                    // uniform over correlation matrices
  y ~ bernoulli(0.25 + 0.75 * inv_logit(a[item] + b[item] .* x_adj[student]));
}
generated quantities {
  corr_matrix[2] Omega_ab = multiply_lower_tri_self_transpose(L_ab);
}
```

Some correlation between difficulty and discrimination, but fitted curves are indistinguishable
from Model 4.

**Lesson:** an expansion that does not change the conclusions is still informative: the data
cannot resolve that correlation.

## Model 6: start over with a generative item-response model

Models 1–5 use total score as a predictor, which is a function of the outcomes: not generative
(you cannot simulate y from parameters). Replace it with latent ability and difficulty:

```stan
data {
  int N, J, K;
  array[N] int<lower=1, upper=J> student;
  array[N] int<lower=1, upper=K> item;
  array[N] int<lower=0, upper=1> y;
  real mu_mu_beta;
  real<lower=0> sigma_mu_beta, mu_sigma_alpha, mu_sigma_beta;
}
parameters {
  real mu_beta;
  real<lower=0> sigma_alpha, sigma_beta;
  vector<offset=0, multiplier=sigma_alpha>[J] alpha;
  vector<offset=mu_beta, multiplier=sigma_beta>[K] beta;
}
model {
  alpha ~ normal(0, sigma_alpha);                // ability centered at 0 fixes the shift
  beta ~ normal(mu_beta, sigma_beta);
  mu_beta ~ normal(mu_mu_beta, sigma_mu_beta);   // normal(0, 5)
  sigma_alpha ~ exponential(1 / mu_sigma_alpha); // mean 5
  sigma_beta ~ exponential(1 / mu_sigma_beta);
  y ~ bernoulli(0.25 + 0.75 * inv_logit(alpha[student] - beta[item]));
}
```

Identification: adding a constant to all `alpha` and all `beta` leaves the likelihood unchanged.
Centering `alpha` at 0 with a hierarchical prior resolves it symmetrically instead of pinning one
item.

## Model 7: discrimination parameters

`Pr(y = 1) = 0.25 + 0.75 inv_logit(gamma_k (alpha_j - beta_k))`, with
`gamma_k ~ normal(1, sigma_gamma)` and `sigma_gamma ~ exponential(2)` (mean 0.5). Fixing the
prior mean of `gamma` at 1 breaks the multiplicative aliasing (scale `alpha, beta` up and
`gamma` down). A **normal, not lognormal,** prior deliberately allows negative discrimination so
miscoded items can be found. Result: `sigma_gamma ≈ 0.3`, discriminations mostly 0.7–1.3.
Abilities overlap heavily (the exam does not discriminate students well); difficulties are
better separated; items everyone got right have broad `beta` posteriors held from −∞ only by
the hierarchical prior.

## Simulation experiment (§4.4)

Aim: can the model recover items with zero or negative discrimination? Set `sigma_gamma = 0.5`,
other hyperparameters at real-fit posterior medians (`mu_beta = −0.82, sigma_alpha = 0.82,
sigma_beta = 1.56`), J = 32, K = 24. Draw parameters, simulate y, fit.

Findings: hyperparameters recovered within 95% intervals; **but** all fitted curves sloped
upward even for items with true `gamma ≈ 0`: the posterior knows some items discriminate poorly
but cannot say which. Estimates of `gamma` are compressed to 0.6–1.4 against a true range of
0–2 (shrinkage). Coverage was fine (~half of 50% intervals, ~95% of 95% intervals). With
J = 100 students, difficulties and discriminations become accurate; abilities stay noisy
because there are still only 24 items.

**Lesson:** a design can be adequate for population parameters and inadequate for unit-level
ones; simulation tells you which before you promise results.

## Breaking the model (§4.5)

1. Remove regularization: flat priors on items everyone got right → improper posterior.
2. The guessing model with `normal(0, 1000)` priors on data simulated from itself (n = 32,
   `a = −6, b = 0.4`): R-hat 1.15, ESS 19, `a ≈ −1052 ± 807`, fitted curves are near step
   functions ("pure guessing below a threshold, perfect knowledge above"). The likelihood does
   not go to zero as parameters → ∞, so a weak prior does not constrain it.

**Lesson:** construct a scenario where each model fails; that is how you learn what it needs.

## What each step taught (§4.6)

1. Single-item logistic regression: interpret and graph fit with uncertainty.
2. All items: degeneracy for all-correct items.
3. Standardizing: enabled interpretable weakly informative priors.
4. Plotting all items in informative order: three data coding errors.
5. Guessing term: more realistic, more uncertain.
6. Multilevel: stability from pooling.
7. Item-response: generative, gives ability uncertainty, removes the circular predictor.
8. Breaking the model: understanding its limits.

At each step compare inferences to the previous model; large differences mean either an
important expansion or a bug, and you have to find out which.
