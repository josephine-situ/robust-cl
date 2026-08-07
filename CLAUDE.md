# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for **robust constraint learning**: when a trained ML model
$\hat{f}(x;\theta)$ is embedded as a constraint $\hat{f}(x)\le b$ inside an
optimization problem, noisy training labels produce models whose "optimal"
decisions violate the true constraint. This repo implements and benchmarks four
methods that make the embedded-constraint decision robust to label uncertainty,
across a synthetic problem and the OptiCL gastric-cancer chemotherapy case study
(replicating Maragno et al. 2025 Table 6).

The flagship contribution is **Cutting Planes** (`src/methods/cp.py`): a trilevel
formulation solved by iteratively separating worst-case models found via
*localized bootstrap resampling* near the current prescription.

## Environment & commands

Python **≥3.14**, managed with **uv** (see `uv.lock`, `pyproject.toml`). Prefix
Python invocations with `uv run`. **Gurobi license required** (all MIP solving
goes through `gurobipy`).

```bash
uv sync                                          # install deps into .venv
uv run python experiments/run_chemo_robust.py --quick   # gastric smoke test (5 cohorts, 4 methods, all_constraints only)
uv run python experiments/run_chemo_robust.py           # full gastric run, 3 methods (long; use SLURM)
uv run python experiments/run_all.py                    # synthetic single run, all methods
uv run python experiments/run_cv.py                     # cross-validate model types/hyperparams
uv run python experiments/summarize_table6.py           # post-process Table 6 CSV -> .csv/.tex
sbatch experiments/submit_chemo_robust.sh               # full run on SLURM (12h, 128G, 16 cpu)
```

There is **no test suite** and no linter configured. `main.py` is a stub —
entry points are the scripts in `experiments/`. Verify changes by running the
`--quick` gastric path or `run_all.py` and checking the CSVs in `results/`.

Prefer the **Bash** tool for these scripts (POSIX). The `results/` and `logs/`
CSV/artifact outputs are the primary way to inspect behavior.

## Architecture

### The four methods (`src/methods/`)

All share the same MIP scaffolding in `nominal.py` (`build_decision_vars`,
`add_problem_constraints`, `embed_constraints`, objective builders). Each method
returns a `SolutionResult`.

- **`nominal.py`** — plain constraint learning, no robustness. Also the shared
  toolkit every other method imports.
- **`robust_regression.py`** — train one noise-robust model, embed it.
- **`wrapper.py`** — Maragno et al. ensemble chance constraint: require
  $(1-\alpha)$ of $P$ bootstrap models to satisfy the constraint. Also provides
  `solve_tree_violation_wrapper` and the bootstrap-index helpers reused by CP.
- **`cp.py`** — Cutting Planes (the contribution). See below.

### `solve_cp` — one driver, auto-selected strategy

`solve_cp` (`src/methods/cp.py:1187`) handles every problem shape with an
identical loop — *train nominal → build master → solve for optimal $x^*$ →
separate → add cuts → terminate* — and **auto-selects** the separation strategy
from the problem shape (number of learned constraints × number of optimal
solutions). **There is no separation flag.**

- **basic** — single LP, single learned constraint (synthetic). Retrain each
  localized bootstrap resample, score all *candidates* at $x^*$, and cut the
  single worst — ranked by the *actual* constraint model (not a CART proxy).
  The candidates are a separation pool, not an embedded ensemble: only the argmax
  becomes a cut. ("Ensemble" in this repo means the GT evaluator; see below.)
- **coherent** — multiple constraints / multiple $x^*$ / learned objective
  (gastric). A *scenario* is one **shared** localized bootstrap relabeling used
  to train every constraint (and the epigraph objective) jointly. Each iteration
  cuts the single worst scenario, ranked by **normalized average distance**
  (mean relative exceedance over all $(x^*,\text{outcome})$ cells, 0–1 scale).
  Stops when that distance ≤ `cp.dist_tol`, or no scenario fits under the
  coverage cap `cp_alpha` (max fraction of $x^*$ allowed infeasible).

Key shared knobs (all under `cp:` in `config.yaml`): **anchors** (`anchor_source`
train/test, `n_anchors`, `anchor_method` kmedoids/sample/all) select where the
$x^*$ are collected for contextual problems; **localization** (`distance`
full/context/auto) picks the bootstrap pool; `robustify_objective` toggles the
epigraph objective robustification; `eval_mode` (`global` vs
`per_anchor_nearest`) controls whether one shared master or per-anchor masters
are trained, with `nearest_distance` for prescribe-time anchor assignment.

The *marketing* (in-LP context, "coupled") setting is described in the README
but **not implemented** (no data loader).

### Data & problem instances (`src/data/`)

- **`generate.py`** — the core abstraction. `ProblemInstance` is the universal
  container (decision vs context var indices, list of `LearnedConstraint`s each
  holding one or more `MLModelData`, ground-truth models, `EvalOutcome`s for
  Table 6, trust-region hull). `synthetic_nonlinear()` and `gastric_cancer()`
  build instances; `filter_constraints()` subsets them (e.g. DLT-only).
- **`gastric_v11.py`** — gastric data processing aligned with the
  constraint-learning v11 notebook / Maragno Appendix D.1 (imputation, ECOG/KPS
  mapping, context + treatment columns).
- **`gastric_model_specs.py`** — hyperparameters replicating the paper's
  embedded constraint models (Table EC.10) and 6-model ground-truth ensemble
  (Table EC.12). Training seed = 1 to match CL `run_MLmodels.py`.

### Model training & embedding (`src/models/`)

- **`train.py`** — trains Linear (ElasticNet), SVM (LinearSVR), CART, RF, GBM,
  XGB, MLP; plus bootstrap-sample helpers and `retrain_on_perturbed`.
- **`embed.py`** — encodes each trained model type as Gurobi MIO constraints
  (trees via big-M leaf-selection binaries, ensembles as averages/sums, MLP ReLU
  via binary activations, Pipelines recurse through the scaler). **When adding a
  new model type, both `train.py` and `embed.py` must support it.**

### Evaluation (`src/evaluation/`)

- **`chemo_metrics.py`** — paper-aligned Table 6 metrics. All reported outcomes
  use the **ground-truth ensemble** (6 sklearn models per outcome), *not* the
  embedded MIP models. `evaluate_given_table6` / `evaluate_prescribed_table6`
  produce the given-vs-prescribed comparison over a shared samestore cohort.
- **`metrics.py`** — simpler synthetic-problem feasibility/violation metrics.

### Calibration (`src/methods/calibrate.py`)

Every non-CP baseline has one monotone robustness knob (wrapper/tree $\alpha$,
robust_param $\rho$). `calibrate_strength` picks the *strongest* setting whose
training-set infeasible fraction is ≤ the shared `uncertainty.alpha`, so all
methods are compared at a matched robustness level. CP is exempt (it
self-regulates via its `p_infeas` cap); nominal has no knob.

## Config

`config.yaml` drives everything. Notable structure: `data.type` switches
synthetic vs gastric; `uncertainty.alpha` is the **shared coverage cap** that CP
enforces directly and the baselines are calibrated to; `methods.cp.*` holds the
CP knobs above; `methods.chemo.methods_to_run` / `constraint_modes` select what
the gastric runner executes, and `methods.chemo.quick` overrides them for
`--quick`. Cross-validated model selections are read from
`results/cv/*_selected_configs.json` (constraint models) and
`*_gt_ensemble_configs.json` (GT ensemble) when present, via `--cv-configs`.

## Conventions / gotchas

- **Ground truth for evaluation is fixed and separate** from the embedded
  models. The GT ensemble is refit on the full clean cohort; only constraint/fit
  rows are resampled in robustness realizations. Don't conflate the embedded
  constraint model with the GT evaluator.
- Robustness (per the memory note & `submit_chemo_robust.sh`) is measured by
  **outer m-out-of-n subsampling without replacement** of training rows, with
  the GT ensemble as a fixed oracle — this is uncertainty over *training draws*,
  distinct from the inner bootstrap the methods use.
- Uncertainty is **data-driven** (bootstrap resamples of observed labels), not a
  parametric label-noise model or preset perturbation wrapper.
- `trust_region.py` / `add_trust_region` constrains decisions to the convex hull
  of observed treatment vectors (gastric).
- Parameter robustness (`methods.robust_param.rho`) shrinks decision-tree leaf
  regions by a margin from the split thresholds; applied across all methods.
