# Stan Language Reference for Everyday Use

Covers the syntax you need for the model patterns in this skill, the deprecations that break old
code, and the compile errors you will actually hit.

## Blocks, in order

| Block | Runs | Values saved | Allowed |
|---|---|---|---|
| `functions` | — | — | user functions; `_lpdf`/`_lpmf`/`_lp`/`_rng` suffixes |
| `data` | once at load | yes (as inputs) | declarations with bounds; no assignment |
| `transformed data` | once at load | no | any computation on data; `_rng` allowed |
| `parameters` | — | yes | unconstrained-space declarations with constraints; no assignment |
| `transformed parameters` | every leapfrog step | **yes** | deterministic functions of parameters; `target +=` allowed |
| `model` | every leapfrog step | no | `~`, `target +=`, local blocks |
| `generated quantities` | once per saved draw | yes | `_rng`, predictions, `log_lik`; **no** `target` |

Local variables declared inside `model { { ... } }` are not written to the output. Use that for
temporaries; use `transformed parameters` only for things you want in the draws.

## Types

```stan
int<lower=0> N;
real<lower=0, upper=1> theta;
vector[K] beta;                    // column vector
row_vector[K] r;
matrix[N, K] X;
array[N] int<lower=0> counts;      // array of ints
array[N] vector[K] v;              // array of vectors
array[N, K] real z;                // 2-D array of reals
simplex[K] lambda;                 // sums to 1, all positive
ordered[K] cut;                    // strictly increasing
positive_ordered[K] pos;
unit_vector[K] u;
sum_to_zero_vector[K] eff;         // Stan 2.36+
cholesky_factor_corr[K] L;
cholesky_factor_cov[K] Lc;
corr_matrix[K] Omega;              // prefer the Cholesky version
cov_matrix[K] Sigma;               // prefer the Cholesky version
complex z1;                        // and complex_vector, complex_matrix
tuple(real, vector[N]) pair;       // Stan 2.33+
```

Elementwise constraints and affine transforms:

```stan
vector<lower=0>[K] tau;
vector<lower=0, upper=1>[K] p;
vector<offset=mu, multiplier=sigma>[J] alpha;    // non-centered parameterization
array[N] real<lower=lb, upper=ub> bounded;       // bounds may be data or parameters
```

Vectors and matrices are faster than arrays of reals for arithmetic; arrays are required for
integers and for ragged-style indexing.

## Distribution statements and target

```stan
y ~ normal(mu, sigma);                          // drops constants
target += normal_lpdf(y | mu, sigma);           // keeps constants
target += normal_lupdf(y | mu, sigma);          // explicitly unnormalized, same as ~
```

Use `_lpdf` (normalized) whenever the value of `target` itself matters: `log_lik` in generated
quantities, `lprior` for priorsense, a user-defined density whose parameters appear in the
normalizing constant, or bridge sampling.

Truncation modifies the normalization automatically:

```stan
y ~ normal(mu, sigma) T[L, U];      // only valid for a scalar y in a loop
```

Multiple statements about the same parameter multiply densities. This is intentional and
sometimes useful (combining independent sources of prior information), but
`theta ~ normal(0, 1)` written twice is `normal(0, 1/sqrt(2))`, not a no-op.

## Indexing and slicing

```stan
y[3]                 // element
y[2:5]               // slice
y[idx]               // multiple indexing with an int array; idx may repeat values
X[, 2]               // column
X[3, ]               // row
alpha[group]         // vectorized lookup: one element per observation
b[group[n], 1]
segment(v, start, len)
head(v, n); tail(v, n)
to_vector(m); to_matrix(v, r, c); to_array_1d(x)
append_row(a, b); append_col(a, b); append_array(a, b)
```

Ragged structures: Stan has no ragged arrays. Store data in long format with a group index, or
store a flat vector plus start/length arrays and use `segment`.

## Control flow and functions

```stan
functions {
  real my_lpdf(real y, real mu, real sigma) {     // name must end in _lpdf to use with ~
    return -0.5 * square((y - mu) / sigma) - log(sigma);
  }
  vector shift(vector x, real c) { return x + c; }
  real check(real x) {
    if (x < 0) reject("x must be non-negative, got ", x);
    return sqrt(x);
  }
}
```

- A function ending in `_lpdf` can be used as `y ~ my(mu, sigma)` (drop the suffix).
- A function ending in `_rng` may only be called from `transformed data`, `generated quantities`,
  or another `_rng` function.
- A function ending in `_lp` may modify `target` and may only be called from
  `transformed parameters` or `model`.
- `print("label ", x)` writes to the console; `reject(...)` rejects the current proposal;
  `fatal_error(...)` aborts.
- `#include "functions/helpers.stanfunctions"` with
  `stanc_options = list("include-paths" = "stan")` shares code across models.

## Deprecations removed in 2.33+

| Removed | Replacement |
|---|---|
| `int y[N];` `real z[N, K];` | `array[N] int y;` `array[N, K] real z;` |
| `vector[K] v[N];` | `array[N] vector[K] v;` |
| `<-` assignment | `=` |
| `increment_log_prob(x)` | `target += x` |
| `get_lp()` | `target()` |
| `if_else(c, a, b)` | `c ? a : b` |
| `abs(real)` | `fabs(real)` (or `abs` for ints) |
| `multiply_log`, `binomial_coefficient_log` | `lmultiply`, `lchoose` |
| `#` comments | `//` |

Upgrade legacy code automatically:

```bash
stanc --print-canonical old_model.stan > new_model.stan
```

```r
mod$format(canonicalize = TRUE, overwrite_file = TRUE)
```

## Compiler flags worth using

```python
CmdStanModel(stan_file="m.stan",
             stanc_options={"warn-pedantic": True, "O1": True,
                            "include-paths": "stan/functions"},
             cpp_options={"STAN_THREADS": True})
```

| Flag | Effect |
|---|---|
| `--warn-pedantic` | parameters with no prior, missing positivity constraints, unused parameters, suspicious distribution arguments |
| `--O1` | safe stanc3 optimizations; slower compile, faster sampling |
| `--print-canonical` | rewrite deprecated syntax |
| `--include-paths` | resolve `#include` |
| `--warn-uninitialized` | flags variables read before assignment |

## Common errors and what they mean

| Message | Cause |
|---|---|
| `Ill-typed arguments supplied to function` | wrong container type (array vs vector) or wrong argument order |
| `Variable "x" does not exist` | declared in a later block, or a typo; blocks are ordered |
| `Cannot assign to variable outside of declaration block` | assigning to a `data` or `parameters` variable |
| `Random number generators are not allowed in the model block` | move the `_rng` call to `generated quantities` |
| `Exception: ... is -0.7, but must be positive finite!` | missing `<lower=0>`, or a computed scale going negative |
| `Exception: ... Initialization failed` | the target is `-inf` or `nan` at the inits: data outside the support, overflow, or an improper posterior |
| `Log probability evaluates to log(0)` | as above; scale predictors or narrow inits |
| `Rejecting initial value` repeatedly | same; check `model.log_prob` at a candidate point |
| `mismatch in dimension declared and found` | data shape does not match the declaration; check the bounds you declared |

Debug a bad point directly:

```python
model.log_prob(params={"alpha": 0.0, "beta": [0.1, 0.2], "sigma": 1.0}, data=data)
```

```r
fit$init_model_methods(); fit$log_prob(unconstrained_variables = upars)
```

## Style

- One statement per line; 2-space indent; `snake_case` names.
- Name variables for what they are (`oxygen_level`, not `x17`); the type declaration is the
  documentation (`real<lower=0>` says more than a comment).
- Keep the `model` block short by pushing computation into `transformed parameters` or functions.
- Comment the *why*, not the *what*; if a long expression needs a comment, make it a function.
