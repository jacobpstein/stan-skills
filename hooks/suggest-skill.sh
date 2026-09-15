#!/usr/bin/env bash
# Suggest Stan skills based on keywords in the user's prompt.
# Runs as a UserPromptSubmit hook and receives JSON on stdin.
# Must exit 0 regardless of match (hooks must not fail).

set -euo pipefail

input=$(cat)
prompt=$(echo "$input" | jq -r '.prompt // .user_prompt // empty' 2>/dev/null || true)

if [ -z "$prompt" ]; then
  exit 0
fi

prompt_lower=$(echo "$prompt" | tr '[:upper:]' '[:lower:]')
directives=()

matches_any() {
  local kw
  for kw in "$@"; do
    if echo "$prompt_lower" | grep -qE "$kw"; then
      return 0
    fi
  done
  return 1
}

stan_modeling_keywords=(
  "\\bstan\\b" "\\.stan\\b" "cmdstan" "cmdstanpy" "cmdstanr" "rstan" "brms" "rstanarm"
  "stanc" "target \\+=" "generated quantities" "transformed parameters"
  "hierarchical model" "multilevel" "varying intercept" "varying slope"
  "mixed effects" "random effects" "gaussian process" "mixture model"
  "censored" "truncated" "ordinal" "ordered logistic" "zero.inflated"
  "hurdle" "negative binomial" "item response" "hidden markov" "state space"
  "bayesian" "mcmc" "posterior" "\\bnuts\\b" "\\bhmc\\b" "probabilistic programming"
  "from_cmdstanpy" "draws_pd" "draws_xr" "summarise_draws"
)

bayesian_workflow_keywords=(
  "bayesian workflow" "statistical workflow" "fake.data" "simulated data"
  "simulate data" "model expansion" "iterative model" "build up the model"
  "start simple" "model building" "gelman" "vehtari" "mcelreath"
  "posterior predictive check" "prior predictive check" "model checking"
  "topology of models" "series of models" "sequence of models"
)

stan_priors_keywords=(
  "prior" "weakly informative" "informative prior" "flat prior" "uniform prior"
  "prior predictive" "prior sensitivity" "power.scaling" "priorsense"
  "regularizing prior" "horseshoe" "\\br2d2\\b" "half.normal" "half.cauchy"
  "lkj" "scale prior" "prior on sigma" "default prior"
)

stan_computation_keywords=(
  "divergen" "treedepth" "tree depth" "max_treedepth" "adapt_delta"
  "r.hat" "rhat" "ess_bulk" "ess_tail" "effective sample size" "mcse"
  "monte carlo standard error" "warmup" "warm-up" "adaptation" "step size"
  "mass matrix" "metric" "initial values" "\\binits\\b" "convergence" "mixing"
  "funnel" "non.centered" "noncentered" "reparameteri" "multimodal"
  "label switching" "e.bfmi" "energy" "chains" "iterations" "slow sampling"
  "pathfinder" "\\badvi\\b" "variational" "laplace" "optimiz" "\\bmap\\b estimate"
  "reduce_sum" "map_rect" "within.chain parallel" "threads"
  "\\bslow\\b" "speed ?up" "faster" "takes (too )?long" "performance"
  "parallel_chains" "adapt_engaged" "step_size" "num_warmup" "iter_warmup"
)

stan_model_evaluation_keywords=(
  "\\bloo\\b" "psis" "elpd" "pareto.k" "k.hat" "khat" "loo_compare" "loo_moment_match"
  "k.fold" "kfold" "cross.validation" "model comparison" "compare models"
  "stacking" "model averaging" "model weight" "bayes factor" "\\bwaic\\b"
  "loo.pit" "calibration" "posterior predictive" "\\bppc\\b" "ppc_" "plot_ppc"
  "log score" "predictive accuracy" "projpred" "variable selection"
  "integrated loo" "marginal likelihood" "bayesian r2" "loo_r2"
)

applied_regression_keywords=(
  "regression and other stories" "\\bros\\b" "regression coefficient" "interpret.*coefficient"
  "divide.by.4" "divide by four" "standardi[sz]" "center(ing)? predictors"
  "log transform" "interaction" "logistic regression" "poisson regression"
  "linear regression" "\\bglm\\b" "stan_glm" "stan_lmer" "stan_glmer"
  "poststratif" "\\bmrp\\b" "treatment effect" "causal" "post.treatment"
  "confound" "propensity" "instrumental variable" "regression discontinuity"
  "power analysis" "sample size" "design analysis" "overdispersion" "offset"
  "residual plot" "r.squared" "\\br2\\b" "separation" "average predictive comparison"
)

r_stan_keywords=(
  "cmdstanr" "cmdstan_model" "rstanarm" "stan_glm" "stan_glmer" "stan_lmer" "stan_polr"
  "brms" "brm\\(" "set_prior" "get_prior" "make_stancode" "posterior_predict"
  "posterior_epred" "posterior_linpred" "pp_check" "bayesplot" "summarise_draws"
  "as_draws" "loo_compare" "loo_model_weights" "priorsense" "projpred" "cv_varsel"
  "tidybayes" "[$]sample[(]" "[$]summary[(]" "library\\(rstan" "rstan"
)

stan_testing_keywords=(
  "simulation.based calibration" "\\bsbc\\b" "parameter recovery" "recover.*parameters"
  "test.*stan" "stan.*test" "pytest.*stan" "testthat.*stan" "unit test.*model"
  "reproducib" "seed" "version control.*model" "ci.*stan" "stan.*ci\\b"
  "rank statistic" "rank histogram" "calibration check" "fixed_param"
  "regression test.*model" "golden" "snapshot test"
)

if matches_any "${stan_modeling_keywords[@]}"; then
  directives+=("Load the stan-modeling skill before responding. The user is working with Stan (CmdStanPy/cmdstanr/brms); this skill covers current Stan syntax, program structure, parameterization, common model patterns, and ArviZ/posterior integration.")
fi
if matches_any "${bayesian_workflow_keywords[@]}"; then
  directives+=("Load the bayesian-workflow skill before responding. The user is building or checking a Bayesian model; this skill covers the Gelman/Vehtari/McElreath iterative workflow: start simple, fake-data simulation, prior/posterior predictive checks, expand one step at a time, compare, report the series.")
fi
if matches_any "${stan_priors_keywords[@]}"; then
  directives+=("Load the stan-priors skill before responding. The user is choosing or checking priors; this skill covers weakly informative defaults, scaling rules, prior predictive checks, tail behavior, and power-scaling sensitivity.")
fi
if matches_any "${stan_computation_keywords[@]}"; then
  directives+=("Load the stan-computation skill before responding. The user is fitting or diagnosing HMC/NUTS; this skill covers warmup, chains, ESS/MCSE, divergences, treedepth, reparameterization, failure modes, and approximate algorithms.")
fi
if matches_any "${stan_model_evaluation_keywords[@]}"; then
  directives+=("Load the stan-model-evaluation skill before responding. The user is checking or comparing models; this skill covers posterior predictive checks, PSIS-LOO, Pareto k, LOO-PIT, loo_compare, stacking, and projpred.")
fi
if matches_any "${applied_regression_keywords[@]}"; then
  directives+=("Load the applied-regression skill before responding. The user is doing applied regression; this skill covers Regression and Other Stories guidance: interpreting coefficients as comparisons, transformations, GLMs, fake-data simulation, poststratification, and causal regression pitfalls.")
fi
if matches_any "${r_stan_keywords[@]}"; then
  directives+=("Load the r-stan-interfaces skill before responding. The user is using the R Stan stack (cmdstanr, rstanarm, brms, posterior, bayesplot, loo); this skill covers fitting, priors, prediction, and diagnostics from R.")
fi
if matches_any "${stan_testing_keywords[@]}"; then
  directives+=("Load the stan-testing skill before responding. The user is testing or validating Stan code; this skill covers parameter recovery tests, simulation-based calibration, pytest/testthat harnesses, and reproducibility practices.")
fi

if [ ${#directives[@]} -gt 0 ]; then
  combined=$(printf '%s\n' "${directives[@]}")
  jq -n --arg ctx "$combined" '{
    "systemMessage": $ctx,
    "hookSpecificOutput": {
      "hookEventName": "UserPromptSubmit",
      "additionalContext": $ctx
    }
  }'
fi

exit 0
