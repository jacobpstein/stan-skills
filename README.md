# Stan Skills

A plugin for Claude Code and other AI coding agents providing [Agent Skills](https://agentskills.io) for Bayesian modeling in **Stan**, informed by the workflow and applied-regression guidance in

- Gelman, Vehtari, McElreath et al., *Bayesian Workflow* (CRC Press, 2026), and
- Gelman, Hill, Vehtari, *Regression and Other Stories* (Cambridge, 2020).

The skills cover the Python interface (CmdStanPy + ArviZ) and the R interfaces (cmdstanr, rstanarm, brms, posterior, bayesplot, loo) on equal footing. The structure mirrors [pymc-labs/python-analytics-skills](https://github.com/pymc-labs/python-analytics-skills).

## Skills

| Skill | Description |
|-------|-------------|
| [bayesian-workflow](skills/bayesian-workflow/) | The full Gelman/Vehtari/McElreath workflow: pick an initial model, prior predictive check, fake-data simulation, fit and diagnose, posterior predictive check, cross-validation, expand one step at a time, compare, and report the *series* of models. |
| [stan-modeling](skills/stan-modeling/) | Writing Stan programs: blocks, current syntax, types and constraints, vectorization, non-centered parameterization, GLMs, hierarchical models, GPs, time series, mixtures, censoring, ordinal, measurement error; fitting with CmdStanPy/cmdstanr and moving draws into ArviZ/posterior. |
| [stan-priors](skills/stan-priors/) | Prior specification: weakly informative defaults and scaling rules, priors for regression/multilevel/GLMM models, prior predictive checks, tail behavior and prior-likelihood conflict, power-scaling sensitivity (priorsense), regularizing and sparsity priors. |
| [stan-computation](skills/stan-computation/) | Fitting and diagnosing HMC/NUTS: warmup and adaptation, chains and iterations, R-hat/ESS/MCSE thresholds, divergences and treedepth, the fail-fast loop, named failure modes with fixes, reparameterization, multimodality, approximate algorithms (Laplace, Pathfinder, ADVI), and parallelization. |
| [stan-model-evaluation](skills/stan-model-evaluation/) | Model checking and comparison: posterior predictive checks and test statistics, PSIS-LOO and Pareto k, moment matching, k-fold, integrated LOO, LOO-PIT calibration, `loo_compare`, stacking, Bayesian R², projection predictive variable selection. |
| [applied-regression](skills/applied-regression/) | *Regression and Other Stories* in Stan/rstanarm: coefficients as comparisons, divide-by-4, centering/standardizing, log transformations, interactions, GLMs (logistic, Poisson, negative binomial, ordered), fake-data simulation, design analysis, poststratification, and causal regression pitfalls. |
| [r-stan-interfaces](skills/r-stan-interfaces/) | The R stack: cmdstanr, rstanarm (`stan_glm`, `stan_glmer`, default priors), brms (`brm`, `set_prior`, `make_stancode`), posterior, bayesplot, loo, priorsense, projpred. |
| [stan-testing](skills/stan-testing/) | Statistical modeling as software development: parameter-recovery tests, simulation-based calibration (SBC), pytest/testthat harnesses for Stan programs, `fixed_param` and `generate_quantities` for fast tests, reproducibility and version control. |

## Installation

### Via npx

```bash
npx skills add jacobpstein/stan-skills
```

### As a Claude Code plugin

```bash
/plugin marketplace add jacobpstein/stan-skills
/plugin install stan@stan-skills
```

Installs all skills plus the keyword-suggestion hook.

### Manual installation

```bash
git clone https://github.com/jacobpstein/stan-skills.git
cd stan-skills
./install.sh claude                     # Claude Code
./install.sh all                        # All platforms
./install.sh claude -- stan-modeling    # A single skill
```

### Utility commands

```bash
./install.sh --list        # List skills with descriptions
./install.sh --validate    # Validate skill structure
./scripts/validate-skills.sh
```

## Platform support

| Platform | Install location | Auto-discovered |
|----------|-----------------|-----------------|
| Claude Code | `~/.claude/skills/` | Yes |
| OpenCode | `~/.config/opencode/skills/` | Yes |
| Gemini CLI | `~/.gemini/skills/` | Yes |
| Cursor | `~/.cursor/skills/` | Yes |
| VS Code Copilot | `~/.copilot/skills/` | Yes |

## Project structure

```text
stan-skills/
├── .claude-plugin/
│   ├── marketplace.json      # Plugin registry metadata
│   ├── plugin.json           # Plugin configuration
│   └── CLAUDE.md             # Plugin development guidance
├── hooks/
│   ├── hooks.json            # UserPromptSubmit hook configuration
│   └── suggest-skill.sh      # Keyword-based skill suggestion
├── scripts/
│   └── validate-skills.sh    # Repository validation
├── skills/
│   ├── bayesian-workflow/
│   ├── stan-modeling/
│   ├── stan-priors/
│   ├── stan-computation/
│   ├── stan-model-evaluation/
│   ├── applied-regression/
│   ├── r-stan-interfaces/
│   └── stan-testing/
├── install.sh
├── package.json
└── skills.json
```

Each skill is a `SKILL.md` entry point plus `references/*.md` for detail the agent loads on demand.

## Hooks

A `UserPromptSubmit` hook suggests relevant skills when it detects keywords such as `stan`, `cmdstanpy`, `cmdstanr`, `rstanarm`, `brms`, `divergences`, `R-hat`, `loo`, `Pareto k`, `prior predictive`, `fake data`, `poststratification`, `SBC`. Test it directly:

```bash
echo '{"prompt": "my Stan model has divergences"}' | bash hooks/suggest-skill.sh
```

## Sources

The workflow content follows *Bayesian Workflow* (Gelman, Vehtari, McElreath, with Simpson, Margossian, Yao, Kennedy, Gabry, Bürkner, Modrák, Leos Barajas; CRC Press 2026; code at https://avehtari.github.io/Bayesian-Workflow/) and *Regression and Other Stories* (Gelman, Hill, Vehtari; Cambridge 2020; code at https://avehtari.github.io/ROS-Examples/). The skills paraphrase and operationalize the guidance; they do not reproduce book text.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
