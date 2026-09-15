# Approximate Algorithms

Source: *Bayesian Workflow* ch. 13.

MCMC error goes to zero with more draws; approximate algorithms have an error floor you cannot
sample away. The useful framing: **an approximate algorithm is an exact algorithm for an
approximate model.** Empirical Bayes is a data-dependent point-mass prior; Laplace is a
data-dependent normal fit; early stopping makes the starting point a prior.

Requirements depend on the workflow stage. Early on, large-scale features estimated cheaply are
enough ("fit fast, fail fast"). For final inference about fine-scale features, use MCMC.

## Optimization (L-BFGS)

```python
opt = model.optimize(data=data, jacobian=False)   # penalized MLE
opt = model.optimize(data=data, jacobian=True)    # MAP in the constrained space
print(opt.optimized_params_dict)
```

```r
opt <- mod$optimize(data = data, jacobian = TRUE)
opt$summary()
```

Uses:
- **Smoke test**: does the program compile, run, and produce a finite, sensible mode in seconds?
  Do this before the first `sample()` on any new model.
- Point estimates when the posterior is near-normal and the likelihood is highly informative
  relative to the number of parameters.

Do not use it for hierarchical scale parameters (the joint mode is often at `tau = 0`), for
skewed posteriors, or for any uncertainty statement. Stan's optimizer uses a looser tolerance
than classical software on purpose; it is far more robust to starting values than IWLS.

## Laplace approximation

```python
lap = model.laplace_sample(data=data, draws=1000)      # optimizes then samples the normal
```

```r
lap <- mod$laplace(data = data, draws = 1000)
```

A normal centered at the mode with covariance from the Hessian. **Check it**: draw from the
approximation, compute the ratio of the true target to the approximating density, and apply the
Pareto k diagnostic. If k < 0.7, use the ratios as importance weights to improve the
approximation; if k > 0.7 the approximation is not usable for that posterior.

Nested Laplace (integrate local parameters, keep hyperparameters by quadrature or MCMC) is the
basis of INLA, TMB, GPstuff. The conditional posterior of local parameters given hyperparameters
is often close to normal even when the joint posterior is not.

## Pathfinder

```python
pf = model.pathfinder(data=data, num_paths=4, draws=1000)
fit = model.sample(data=data, inits=pf.create_inits(), chains=4)
```

```r
pf <- mod$pathfinder(data = data, num_paths = 4, draws = 1000)
fit <- mod$sample(data = data, init = pf, chains = 4)
```

Runs L-BFGS and picks the best normal approximation along the optimization path by a stochastic
KL estimate; multi-path Pathfinder uses the normals as a mixture importance-sampling proposal.

Three uses:
1. Quickly check that the model produces a sensible posterior.
2. **Initialize HMC near the typical set**, which shortens warmup and avoids adaptation being
   corrupted by a long transient.
3. With suspected multimodality, run many paths to find modes and estimate their relative mass by
   importance sampling; discard modes with negligible mass.

Not for final inference.

## ADVI (variational inference)

```python
vb = model.variational(data=data, algorithm="meanfield")   # or "fullrank"
```

```r
vb <- mod$variational(data = data, algorithm = "meanfield")
```

Minimizes reverse KL over a normal family. Even the global optimum is a misspecified
approximation: mean-field recovers marginal precisions but **underestimates marginal variances
and entropy**. The optimization itself may not converge, and accuracy is hard to assess in high
dimensions.

Legitimate use: repeated inference on similar datasets **after** the accuracy for that model
class and those quantities has been validated by simulation experiments. Otherwise use it for
rough exploration only.

## Simulation-based and amortized inference

For models where you can simulate but cannot evaluate the likelihood:

- ABC: draw from the prior, simulate, keep draws whose simulated data are close to the observed.
  Struggles in high dimensions.
- Neural density estimators trained on model simulations; more efficient than ABC when the
  observed data lie in the typical set of the training simulations.
- Amortized Bayesian inference: an expensive training phase, then near-instant posteriors for new
  datasets. Predictive checks still apply; posterior-simulation calibration checking is the
  recommended validation.

## Divide and conquer

**Sequential updating**: given draws from `p(theta | y1)`, obtain `p(theta | y1, y2)` by
Pareto-smoothed importance sampling with weights `p(y2 | theta^s)`. Works when the new data are
much less informative than the old (10 000 points updated with 100). High Pareto k means poor
overlap: either chance or a model violation. Next step is a particle filter, or approximating the
old posterior parametrically (multivariate normal on the unconstrained scale) and using it as the
prior for a fresh fit.

**Parallel**: partition the data, fit each subset, combine. There is no generally good partition
(clustering, time, space all matter). Early in the workflow, independent fits per subset plus a
simple meta-analysis of summaries is often enough.

## Fitting a simpler model on purpose

"If you can't bring the computation to the model, bring the model to the computation":

- Normal likelihood even for bounded or discrete outcomes when the quantities of interest are
  insensitive to it.
- Drop a nonlinear term; varying intercepts and slopes → varying intercepts only; measurement
  error model → treat the measurement as exact.
- Pin a parameter at a fixed value (including a nonzero hyperparameter: fixing the group-level
  variance removes the funnel entirely).
- Strong priors as soft constraints (`theta ~ normal(0, 0.1)` is more flexible than `theta = 0`).
- Pre-process: combine related predictors into a score; replace a multivariate outcome with a
  total.
- Fit on a subset.

Caveats: a restricted family may fit the data worse and therefore be *harder* to fit; constraints
and strong priors speed things up when consistent with the data and create multimodality when they
conflict with it; pre-processing removes the intermediate quantities you would have used to
diagnose problems; subsets have less-constrained posteriors.

## Validating any approximation

Simulate from the model with known parameters, run the full computational pipeline, and compare
the recovered inferences to the truth. Use SBC to check that the approximation reproduces the
features you care about, and Pareto k diagnostics wherever importance sampling is involved. Report
which algorithm produced the reported results.
