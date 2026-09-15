# cmdstanr Reference

Install from the stan-dev R-universe (cmdstanr is not on CRAN):

```r
install.packages("cmdstanr", repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
cmdstanr::check_cmdstan_toolchain()
cmdstanr::install_cmdstan(cores = 4)
cmdstanr::cmdstan_version()
```

## Compiling

```r
mod <- cmdstan_model(
  "m.stan",
  compile = TRUE,
  stanc_options = list("warn-pedantic" = TRUE, "O1" = TRUE),
  cpp_options = list(stan_threads = TRUE, stan_opencl = TRUE),
  include_paths = "stan/functions",
  force_recompile = FALSE
)
mod$print()                     # the Stan source
mod$check_syntax(pedantic = TRUE)
mod$format(canonicalize = TRUE, overwrite_file = TRUE)   # upgrade deprecated syntax
mod$exe_file()
```

Recompilation is triggered by a changed source hash. `force_recompile = TRUE` when you change
compiler options.

## Sampling

```r
fit <- mod$sample(
  data = data,                 # named list, or a path to a JSON file
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = NULL,    # requires stan_threads at compile time
  seed = 20260914,
  init = 0.1,                  # scalar = uniform(-0.1, 0.1) unconstrained; or a list per chain
  iter_warmup = 1000,
  iter_sampling = 1000,
  save_warmup = FALSE,
  thin = 1,
  adapt_delta = 0.8,
  max_treedepth = 10,
  metric = "diag_e",           # or "dense_e", "unit_e"
  step_size = NULL,
  refresh = 500,
  show_messages = TRUE,
  output_dir = "results/csv"   # otherwise a temp dir that disappears
)
```

Explicit inits per chain:

```r
inits <- lapply(1:4, function(i) list(alpha = 0, beta = rep(0, K), sigma = 1))
fit <- mod$sample(data = data, chains = 4, init = inits)
```

## Diagnostics

```r
fit$diagnostic_summary()
#> $num_divergent  [1] 0 0 0 0
#> $num_max_treedepth [1] 0 0 0 0
#> $ebfmi [1] 1.02 0.98 1.05 1.01
fit$cmdstan_diagnose()
fit$summary(NULL, posterior::default_convergence_measures())
fit$sampler_diagnostics(format = "draws_df")
fit$time()                       # per-chain warmup and sampling seconds
fit$metadata()$step_size_adaptation
fit$init()                       # the inits actually used
```

## Draws

```r
fit$draws()                                  # draws_array
fit$draws(variables = c("alpha", "beta"), format = "draws_df")
fit$draws(format = "draws_matrix")
fit$draws(format = "draws_rvars")            # rvar objects
posterior::as_draws_df(fit)                  # equivalent
fit$summary(variables = "beta", "mean", "sd", ~quantile(.x, c(0.05, 0.95)))
```

## Saving and reloading

The CSV files live in a temp directory by default and are deleted when the session ends.

```r
fit$save_object("results/fit_m1.rds")        # portable, includes the draws
fit2 <- readRDS("results/fit_m1.rds")

fit$save_output_files(dir = "results/csv", basename = "m1")
fit3 <- as_cmdstan_fit(list.files("results/csv", pattern = "m1.*csv", full.names = TRUE))
```

`$save_object()` is the safe default; `as_cmdstan_fit()` is what you use to reload raw CSVs
produced elsewhere (a cluster run, CmdStan on the command line).

## Other inference methods

```r
opt <- mod$optimize(data = data, jacobian = TRUE, seed = 1)
opt$summary()

lap <- mod$laplace(data = data, mode = opt, draws = 1000, jacobian = TRUE)

pf  <- mod$pathfinder(data = data, num_paths = 4, draws = 1000, seed = 1)
fit <- mod$sample(data = data, init = pf, chains = 4)        # Pathfinder inits for HMC

vb  <- mod$variational(data = data, algorithm = "meanfield")

gq  <- mod_gq$generate_quantities(fitted_params = fit, data = data)
gq$draws("y_new")
```

## LOO and moment matching

```r
loo1 <- fit$loo(variables = "log_lik", r_eff = TRUE, save_psis = TRUE)
print(loo1); loo::pareto_k_table(loo1)
```

Moment matching from cmdstanr needs the model's log-density exposed:

```r
fit$init_model_methods()          # compiles log_prob / grad_log_prob / unconstrain_draws

log_lik_i <- function(draws, i, ...) draws[, paste0("log_lik[", i, "]")]
post_draws <- function(x, ...) posterior::as_draws_matrix(x$draws())
unconstrain_pars <- function(x, pars, ...) x$unconstrain_draws(draws = pars)
log_prob_upars <- function(x, upars, ...) apply(upars, 1, x$log_prob)

loo_mm <- loo::loo_moment_match(
  x = fit, loo = loo1,
  post_draws = post_draws, log_lik_i = log_lik_i,
  unconstrain_pars = unconstrain_pars, log_prob_upars = log_prob_upars,
  log_lik_i_upars = log_lik_i_upars     # your function evaluating log_lik[i] at upars
)
```

This is enough friction that brms (`moment_match = TRUE`) is often the easier route when the
model can be expressed there.

## Parallelism

```r
# between chains
fit <- mod$sample(chains = 4, parallel_chains = 4)

# within chain (model must use reduce_sum and be compiled with stan_threads)
mod <- cmdstan_model("m.stan", cpp_options = list(stan_threads = TRUE))
fit <- mod$sample(data = c(data, list(grainsize = 1)),
                  chains = 4, parallel_chains = 4, threads_per_chain = 2)

# GPU
mod <- cmdstan_model("m.stan", cpp_options = list(stan_opencl = TRUE))
fit <- mod$sample(data = data, opencl_ids = c(0, 0))
```

Total cores used is `parallel_chains * threads_per_chain`. Fill the chains first.

## Debugging a bad point

```r
fit$init_model_methods()
upars <- fit$unconstrain_variables(list(alpha = 0, beta = rep(0, K), sigma = 1))
fit$log_prob(upars)
fit$grad_log_prob(upars)
```

## Data handling

```r
cmdstanr::write_stan_json(data, "data/m1.json")
fit <- mod$sample(data = "data/m1.json")
```

Stan wants a flat named list of numerics; integers must be integers (`as.integer`), factors must
be converted to integer codes, and matrices are passed as matrices (R is column-major, cmdstanr
handles the transposition).
