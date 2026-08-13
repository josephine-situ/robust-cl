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
  $(1-\alpha)$ of $P$ models to satisfy the constraint. Also provides
  `solve_tree_violation_wrapper` and the bootstrap-index helpers. **The P models
  do not come from bootstrap under the default**: `scenario_source` defaults to
  `"noise"`, so they are a prefix of CP's `ScenarioBank`. The bootstrap helpers
  (and `uncertainty.bootstrap_frac`) are reached only under
  `scenario_source: "bootstrap"`. `run_all.py:96` and `run_chemo_robust.py:515`
  build a bootstrap cache unconditionally and pass it in; under `"noise"` it is
  **dead** — same shape as the `cp_alpha` dead argument documented below, and it
  is not evidence the production wrapper bootstraps.
- **`cp.py`** — Cutting Planes (the contribution). See below.

### `uncertainty.py` — the shared uncertainty set D

All three uncertainty-aware methods face **one** set and differ only in what they
do with it (cut lazily / chance-constrain / robustify the fit):

$$D_c = \{\delta: |\delta_i| \le \varepsilon_c,\ \|\delta\|_1 \le \texttt{budget\_frac}\cdot n\,\varepsilon_c\},\quad \varepsilon_c = \texttt{eps\_0}\cdot \mathrm{scale}(y_c)$$

- **The scale is the out-of-fold residual sd** (`scale_stat: "oof_sd"`), so
  `eps_0 = 1` means *one unexplained standard deviation* — a unit whose meaning,
  not merely whose units, transfers across problems. δ is added to labels *before*
  training and the model is retrained, so the radius must be a label-space
  quantity; both `oof_sd` and marginal `sd(y)` are, and they differ in what they
  measure.
- **Not marginal `sd(y)`** (kept as the `"sd"` ablation): it is mostly *signal*
  wherever the model fits. On synthetic `sd(y) = 0.545` against a true noise of
  0.100, so `eps_0 = 1` there corrupts labels 4× harder than the data supports and
  **CP could not converge against it** in 20 iterations. Under `oof_sd` the same
  `eps_0 = 1` gives 0.128 — it nearly recovers the DGP without being told it.
- The model dependence is bounded and deliberate: `run_cv.py` freezes the model
  class *before* any robustness and all three methods embed that same frozen
  model, so D comes from a shared pre-committed choice, not one method's tuning.
  It does conflate label noise with misspecification — on gastric the models
  explain almost nothing (unexplained ratio 0.78–1.02, one outcome above 1.0), so
  D stays nearly as wide as `sd(y)` there. That is the honest answer when the fit
  is that poor. Folds follow the problem's own scheme (temporal on gastric, so the
  estimate cannot leak future information). **No coverage claim** is made.
- **`ScenarioBank`** draws B **vertices** of D (±eps on `budget_frac`·n rows, 0
  elsewhere — matching robust_reg's adversary; interior draws would be a weaker
  adversary at the same D) and trains one model per draw per outcome, with
  `random_state` fixed across members so the scenario is the only variation.
  Draw *b* is a pure function of `(seed, b)`, so **the wrapper's P models are a
  nested prefix of CP's B** — which is what makes the α=0 ≡ τ→0 equivalence exact.
- **`coherent`** — one flag, all three methods. Coherent shares one standardized
  direction across outcomes (scaled by each `eps_c`); incoherent draws
  independently. Vacuous on synthetic (one outcome) — literally, not
  approximately: with one `MLModelData` both branches make the same single
  `_vertex_direction` call against the same rng, so the banks are bit-identical.

  **OPEN QUESTION (raised 2026-08-11, unresolved — do not treat the two bullets
  above as settled).** Two objections, both open:

  1. **Does coherence mean anything across different targets?** What is shared is
     a dimensionless ±1 direction over *rows*, not a label shift — all six
     gastric outcomes are built on the same `X_fit` in the same row order
     (`generate.py:533`, `:552`), so row *i* is the same trial arm everywhere,
     and magnitudes stay per-outcome via `eps_c`. The story that justifies it is
     *record-level* mismeasurement: a study that under-reports adverse events
     under-reports across all five toxicity endpoints. **That story does not
     cover OS.** Sign is per-outcome-label, not clinical valence: `+1` on a
     toxicity percentile is worse, `+1` on OS months is better, so a coherent
     draw is not "these arms get uniformly worse" — it is "these arms shift up
     in each column's own signed units." Whether OS belongs on the shared
     direction at all is unresolved. Note the cross-outcome residual
     correlation is never estimated: coherent asserts +1, incoherent asserts 0,
     and the truth is in between.
  2. **Incoherent is not the per-constraint worst case, though it should
     arguably be.** Its *set* is the product `D_1 × … × D_C` and does strictly
     contain coherent's diagonal — but the separation never exploits that. The
     loop scores one scenario index at a time and pulls every constraint's model
     from `bank.models_for(b)` at one shared `b` (`cp.py:1468`,
     `uncertainty.py:366`), then cuts that whole `b` (`cp.py:1523`). Outcome 1
     from draw 3 cannot be paired with outcome 2 from draw 17, so the adversary
     searches B joint points, not `B^C` combinations — a legal region of its own
     D that it never visits. **No flag reaches the mixed adversary**;
     `cut_whole_scenario=False` still cuts a single `b`.
  3. **The flag means structurally different things in CP and the wrapper**,
     which follows from (2). The wrapper's incoherent arm gives each constraint
     its own indicator `z[c, p]` (`wrapper.py:232`, `:234-235`), so different
     constraints may be satisfied by *different* replicates — close to the
     per-constraint worst case objection (2) asks for. CP's incoherent arm has
     no analogue: independent draws, but still one shared `b` per cut. So
     incoherent CP is a strictly weaker object than incoherent wrapper, while
     coherent CP and coherent wrapper match by construction. **The α=0 ≡ τ→0
     equivalence is therefore a coherent-arm result** — `cp.py:1511-1514`
     derives it precisely from the whole-scenario cut matching the wrapper's
     single joint indicator. Whether it survives on the incoherent arm has not
     been tested, and the reasoning above suggests it does not.

  Changing (2) is not free: the shared-`b` cut is what makes CP at τ→0 identical
  to the wrapper at α=0 and what makes permanent scenario exclusion sound
  (reasoning at `cp.py:1506-1522`). A per-constraint argmax would break both and
  would defend against points that correspond to no single relabeling of one
  trial. Cheapest next step is measurement, not a rewrite: build both banks at
  the same seed and B, score every draw at the nominal $x^*$, and compare the max
  and the full distance distribution — a bank scan, no MIP resolves.
- `uncertainty.eps_0` and `budget_frac` are shared **constants, not knobs** —
  each method keeps exactly one conservatism dial (CP τ, wrapper α, robust_reg
  `label_eps`). `eps_0` is deliberately *not* pinned to robust_reg's calibrated
  ε*, which would conflate D's definition with one method's tuning.

### `solve_cp` — one driver, auto-selected strategy

`solve_cp` (`src/methods/cp.py:1187`) handles every problem shape with an
identical loop — *train nominal → build master → solve for optimal $x^*$ →
separate → add cuts → terminate* — and **auto-selects** the separation strategy
from the problem shape (number of learned constraints × number of optimal
solutions). **There is no separation flag.**

- **basic** — single LP, single learned constraint (synthetic). Score every bank
  draw at $x^*$ and cut the single worst — ranked by the *actual* constraint model
  (not a CART proxy). The bank is a separation pool, not an embedded ensemble:
  only the argmax becomes a cut. ("Ensemble" in this repo means the GT evaluator.)
- **coherent** — multiple constraints / multiple $x^*$ / learned objective
  (gastric). A *scenario* is one **shared** relabeling used to train every
  constraint (and the epigraph objective) jointly. Each iteration cuts the single
  worst scenario, ranked by **normalized average distance** (mean relative
  exceedance over all $(x^*,\text{outcome})$ cells, 0–1 scale). Stops when that
  distance ≤ the tolerance, or no scenario fits under the coverage cap `cp_alpha`
  (pinned at 0 in production, so: no scenario can be cut without breaking a
  protected anchor).

**Scenarios come from a fixed bank** (`cp.scenario_source: "noise"`, default).
The legacy `"bootstrap"` path redrew every iteration while `d0` stayed frozen from
iteration 0, so the stopping rule compared *different samples*. Kept as an ablation.

### Four things that had to be right before CP converged on gastric

Each was found by a run, not by reading. In combination they took gastric from an
exact period-4 cycle (never converging, τ inert across its whole grid) to
`status=optimal` in 19 iterations.

1. **`cp.mip_gap` (default `1e-4`, was hard-coded `0.01`).** The single biggest
   one. At 1% on a gastric objective of ~10 the solver returns any incumbent
   within ~0.1, while the distances being separated are ~0.007 — so cuts an order
   of magnitude below the solver's own tolerance left $x^*$ unmoved and different
   cut sets produced identical solutions. **Synthetic never hit this** (objective
   ~1.2, distances ~0.1, so its cuts sit above the gap): same code, opposite
   regimes. The loop now matches the final and prescribe-time solves.
2. **Nothing is removed from the master** under a fixed bank — `prune_slack_cuts`
   is off and `cut_eviction: "reject"`. Removing a cut lets a previous $x^*$ recur,
   which is what a cycle *is*. With nothing removed the eligible set strictly
   shrinks, so CP terminates in ≤ B iterations on both problems.
3. **The protected anchor set is fixed**, not recomputed per iteration — the
   anchors the *nominal* fit could serve (8/10 on gastric), tested **set-wise** so
   CP can't trade one patient's feasibility for another's. The old per-iteration
   baseline was a ratchet.
4. **Rejections are cached** (113 of 200 draws on gastric), which (3) makes sound:
   with a fixed protected set and a monotone master, a cut that breaks a protected
   anchor always will. Without this, rollbacks grew 8 → 44 per iteration.

`cp.objective_monotone` (default off) adds a no-deterioration cut in both paths.
It is redundant while nothing is removed, and is the lever for re-enabling
pruning/eviction safely. Note it constrains $x$ in both settings — `obj_expr` is
`c'x` on synthetic and `−OS(x)` on gastric under `robustify_objective: false` — but
would be **vacuous** under `robustify_objective: true`, where `obj_expr` becomes
the free epigraph variable `t_obj`.

`_report_cp_diagnostics` prints per-run cycle status, objective regressions,
permanent rejections, and anchor-infeasibility / bound-blocked rates. These are
reported, never acted on automatically.

**`d0` is a high quantile** (`cp.d0_quantile`, default 0.9) of the iteration-0
scenario distances, **not their max** — the max grows with B, and CP (B=200) and
the wrapper (P=20) run at different B by design. The quantile is over the *draws*
(the coherent path means over anchors × outcomes inside each entry first).

**τ=1 is the weak end of the grid but is not nominal.** The tolerance comes from
`q0.9` while the stopping statistic is the **max** over the bank, on both paths
(`cp.py:1076`, `cp.py:1530`), so iteration 0 fails its own test and τ=1 still
separates the worst ~decile. **No τ in `[0.1, 1.0]` reproduces nominal** — run
nominal itself for that endpoint. This is known and deliberately left as is: the
max is the worst-case statistic the contribution rests on, and matching the two
sides would either weaken it to a percentile claim or make `d0` seed-noisy. The
consequence is that τ is a *ratio to each problem's own d0*, never a shared
physical quantity — the two paths don't even share units (basic scores **raw
signed** violations; coherent scores **normalized, anchor-averaged** distances
gated at 0). Also, coherent drops permanently-rejected draws after iteration 0
(113/200 on gastric) and rejection correlates with severity, so gastric's max is
measured with its worst tail deleted and the same τ over-cuts *less* there.
`_resolve_d0` and `config.yaml`'s `d0_quantile` carry the full statement.

**Why B differs, and why it matters.** The wrapper embeds all P models, so P is
capped by MIP size; CP embeds one cut per iteration and evicts/prunes. Measured on
synthetic: wrapper at P=20 is 31,917 vars / 101,172 constrs; CP at B=200 is 6,380
vars / 20,236 constrs. At B = P on the identical prefix, CP at τ→0 and the wrapper
at α=0 return the same $x^*$ exactly — **CP is a lazy wrapper at α=0**.

Key shared knobs (all under `cp:` in `config.yaml`): **anchors** (`anchor_source`
train/test, `n_anchors`, `anchor_method` kmedoids/sample/all) select where the
$x^*$ are collected for contextual problems; `robustify_objective` toggles the
epigraph objective robustification; `eval_mode` (`global` vs
`per_anchor_nearest`) controls whether one shared master or per-anchor masters
are trained, with `nearest_distance` for prescribe-time anchor assignment.
`uncertainty.cp_k_neighbors_*`, `cp_n_candidates` and `cp.distance` apply **only**
to the legacy `scenario_source: "bootstrap"` path — kept so prior results
reproduce, unused under the default.

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
training-set infeasible fraction is ≤ `uncertainty.alpha`, so all methods are
compared at a matched robustness level. Nominal has no knob.

**This is the legacy path.** `calibration.method` defaults to `"cv"`
(`cv_calibrate.py`), which selects each knob on held-out folds instead;
`calibrate_strength` runs only under `calibration.method: "alpha"`.

**CP is exempt, and does not read `uncertainty.alpha`.** Its coverage cap
`cp_alpha` is **pinned at 0 at every call site** (`run_all.py`, `run_sweep.py`,
`run_chemo_robust.py:267`), so a cut that would break the protected anchor set is
rolled back and its draw permanently rejected — τ is CP's only lever. The
`cp_alpha > 0` machinery in `_CoherentSeparation` (admit the worst scenario
affordable while `p_infeas ≤ alpha`) is reachable only by calling `solve_cp`
directly. Note `run_chemo_robust.py:244`/`:363` still *pass* `settings["alpha"]`
into `_cp_solver`'s `cp_alpha` parameter, but the body hard-codes `0.0` and the
argument is dead — do not read those call sites as evidence alpha reaches CP.

## Config

`config.yaml` drives everything. Notable structure: `data.type` switches
synthetic vs gastric; `uncertainty.{eps_0, budget_frac, coherent, scale_stat}`
define the **shared set D**; `uncertainty.alpha` is the **legacy-calibration
target only** (baselines under `calibration.method: "alpha"` — CP never reads it,
see Calibration); `methods.cp.*`
holds the CP knobs above; `methods.chemo.methods_to_run` / `constraint_modes`
select what the gastric runner executes, and `methods.chemo.quick` overrides them
for `--quick`. Cross-validated model selections are read from
`results/cv/*_selected_configs.json` (constraint models) and
`*_gt_ensemble_configs.json` (GT ensemble) when present, via `--cv-configs`.

**Robustness-knob CV is per (method, coherence) cell**, keyed
`method@coherent` / `method@incoherent` (`cv_calibrate.knob_key` / `lookup_knob`,
which falls back to a bare `method` key for older JSONs). The two cells are not
interchangeable — coherence and conservatism would be confounded by reusing one
θ* — so they are calibrated separately. `cv_calibration.coherence_cells` controls
which cells run; synthetic stays single-cell (coherence is vacuous there).

Which cell is the *stronger* adversary is **not settled** (see the OPEN QUESTION
under `uncertainty.py`). Coherent is stronger under the implementation as it
stands — finite B covers the diagonal far better than the product set, and the
mean-over-cells scoring (`cp.py:1481-1504`) has a heavier right tail when the
per-outcome exceedances move together. That is a property of *(mean scoring ×
shared `b` × finite B)*, not of the sets: incoherent's set strictly contains
coherent's, so under max-over-outcomes scoring or a per-constraint argmax the
ordering would flip. Earlier revisions of this file asserted coherent-is-stronger
flatly; that was underspecified.

## Presentations (`presentations/`)

Two kinds of deck, with opposite update rules.

- **`method.tex` is the standing method reference and must always be true of the
  code at HEAD.** It is the one deck that *is* edited in place. **Any change to a
  method, to D, to the calibration or evaluation protocol, or to a default in
  `config.yaml` that the deck states, is not finished until `method.tex` says the
  new thing** — same change, not a follow-up. When in doubt, re-read the slide
  that covers what you touched and check its numbers against the config/JSON you
  changed. It carries no results: dated numbers and figures belong in the update
  decks, because they go stale and `method.tex` may not.
- **`research_update_YYYY-MM-DD.tex`** are dated snapshots of what changed since
  the last one. A new deck supersedes the last rather than editing it in place —
  edit an existing update deck only to correct something that was wrong or has
  since landed, not to keep it current.

Both:

- **Be very concise.** Terse bullet fragments, not prose. No lengthy sentences,
  no sentences spanning several lines. One claim per bullet, numbers over
  adjectives. Slides must fit — check the log for `Overfull \vbox` and cut text
  (don't just shrink the font) until there are none.
- **Build with `latexmk`, never bare `pdflatex`.** `presentations/.latexmkrc` sets
  `$aux_dir = 'build'` and `$out_dir = '.'`, so any latexmk run whose cwd is
  `presentations/` — including an editor's compile-on-save — keeps the `.pdf` (and
  `.synctex.gz`) next to the `.tex` and puts `.aux`, `.log`, `.fls`,
  `.fdb_latexmk`, `.nav`, `.out`, `.snm`, `.toc` in `build/`. Bare `pdflatex`
  ignores the rc file and litters. `presentations/build.ps1` compiles every deck
  this way. (A distinct `$aux_dir` relies on `-aux-directory`, which is
  MiKTeX-only.)

## Conventions / gotchas

- **Ground truth for evaluation is fixed and separate** from the embedded
  models. The GT ensemble is refit on the full clean cohort; only constraint/fit
  rows are resampled in robustness realizations. Don't conflate the embedded
  constraint model with the GT evaluator.
- Robustness (per the memory note & `submit_chemo_robust.sh`) is measured by
  **outer m-out-of-n subsampling without replacement** of training rows, with
  the GT ensemble as a fixed oracle — this is uncertainty over *training draws*,
  distinct from the inner bootstrap the methods use.
- Uncertainty is the **shared set D** (`src/methods/uncertainty.py`), scaled per
  outcome by the out-of-fold residual sd. The older data-driven bootstrap
  resampling survives as the `scenario_source: "bootstrap"` ablation.
- **`uncertainty.bootstrap_frac` (default 0.5) is the rows drawn per bootstrap
  replicate**, as a proportion of `n_train`. 0.5 is what Maragno et al. (2025)
  Sec. 4.4.1 actually specifies — "a bootstrap sample (proportion = 0.5) of the
  underlying data" — for the WFP wrapper experiment; the paper gives **no**
  proportion for the chemo case study, because §5 never uses the wrapper (Table 6
  there comes from one selected model per outcome, not an ensemble). Applying the
  wrapper to gastric at all is our extension. Half-size replicates carry ~39%
  unique rows against ~63% for n-out-of-n, so they overlap less, the P models
  spread wider, and the constraint binds harder: measured on synthetic at P=5,
  `obj = -1.2533` at 0.5 vs `-1.2769` at 1.0. Set `1.0` to reproduce results
  predating this knob.
- **`eps_0` is in units of the scale, so it interacts with `scale_stat`.**
  `eps_0 = 1` under `oof_sd` is the intended operating point; `eps_0 = 1` under
  marginal `"sd"` is 4× too wide on synthetic and CP will not converge. If you
  switch `scale_stat`, re-check `eps_0`.
- Never set the radius from synthetic's known `noise_std` — that calibrates D to
  the data-generating process and CP wins by construction. `oof_sd` arriving near
  `noise_std` is a *validation* of the estimator, not an input to it.
- `trust_region.py` / `add_trust_region` constrains decisions to the convex hull
  of observed treatment vectors (gastric).
- Parameter robustness (`methods.robust_param.rho`) shrinks decision-tree leaf
  regions by a margin from the split thresholds; applied across all methods.
