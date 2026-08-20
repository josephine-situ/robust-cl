# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for **robust constraint learning**: a trained model `f(x;theta)` is
embedded as a constraint `f(x) <= b` inside an optimization problem, and noisy
training labels yield models whose "optimal" decisions violate the *true*
constraint. Four methods are benchmarked on a synthetic LP and the OptiCL
gastric-cancer case study (Maragno et al. 2025, Table 6).

The contribution is **Cutting Planes** (`src/methods/cp.py`): separate the
worst-case model over a fixed bank of relabelings drawn from a shared uncertainty
set D, adding one cut per iteration.

`presentations/method.tex` is the prose version of this file and must stay true of
HEAD (see Presentations). Dated `research_update_*.tex` decks carry the numbers;
current is **2026-08-19**, the shared-D rho sweep.

## Environment & commands

Python **>=3.14** via **uv**; prefix with `uv run`. **Gurobi license required.**
No test suite, no linter. `main.py` is a stub — entry points are `experiments/`.
Verify changes with the `--quick` gastric path or `run_all.py`, then read the CSVs
in `results/`. Prefer the **Bash** tool (POSIX scripts).

```bash
uv sync
uv run python experiments/run_chemo_robust.py --quick   # gastric smoke test (5 cohorts, 4 methods)
uv run python experiments/run_chemo_robust.py           # full gastric (long; use SLURM)
uv run python experiments/run_all.py                    # synthetic, all methods
uv run python experiments/run_cv.py                     # model type/hyperparameter CV
uv run python experiments/summarize_table6.py           # Table 6 CSV -> .csv/.tex
uv run python experiments/verify_embedding.py           # MIP vs sklearn/xgb agreement
uv run python experiments/run_adversary_probe.py        # is the random bank a weak adversary?
sbatch experiments/submit_chemo_robust.sh               # 12h, 128G, 16 cpu
```

**The rho sweep is the current headline experiment.** `run_rho_sweep.py` **forces
`geometry="ellipsoid"` regardless of `config.yaml`**:

```bash
uv run python experiments/run_rho_sweep.py --problem gastric --ablate    # coherent cell (default)
uv run python experiments/run_rho_sweep.py --problem gastric --incoherent --ablate
uv run python experiments/run_rho_sweep.py --problem gastric --match-bank    # B=P
uv run python experiments/run_rho_sweep.py --problem synthetic --n-folds 10  # 4 folds quantizes feas to 0.25
uv run python experiments/run_rho_sweep.py --rho-star-only --feas-target 0.8 --out-suffix _t080
uv run python experiments/plot_rho_sweep.py --suffix _coh    # -> results/figures/fig_rho_*.pdf
sbatch experiments/submit_rho_sweep.sh                       # PROBLEM/COHERENCE/MATCH_BANK/RHO_GRID env
```

Grids: rho `[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]`, tau `[1.0, 0.1, 0.01, 0.001]`,
alpha `[0.0, 0.1, 0.2, 0.3, 0.5]`. Runs resume from a checkpoint keyed
`(method@rho, knob)` **only**, so **always pass the cell flags** — otherwise a
second cell resumes the first's rows and overwrites its curve.

**Only `results/rho_sweep/` and `results/figures/fig_rho_*` are current** —
ellipsoid geometry and fixed temporal folds (2026-08-17/19). Everything else
(`results/gastric/` all <= 2026-08-13, `results/synthetic/`, `adversary_probe/`,
`cv/*_robustness_knobs.json`) is `box_l1` and/or random-KFold-scaled. **Never read
a `box_l1` number against an ellipsoid one**; re-run before citing.

## Architecture

### The four methods (`src/methods/`)

Shared MIP scaffolding lives in `nominal.py` (`build_decision_vars`,
`add_problem_constraints`, `embed_constraints`, objective builders); every method
returns a `SolutionResult`.

- **`nominal.py`** — no robustness, plus the toolkit everything else imports.
- **`robust_regression.py`** — robustify the *fit*:
  `min_theta max_{delta in D_c} L(theta; X, y_c + delta)`. **Linear** is exact: on
  the ball the inner max is closed form (`delta* = R*r/||r||`, giving
  `(||r||_2 + R)^2`), so one second-order cone `nu >= ||r||_2` turns the ElasticNet
  QP into an **SOCP** (`_label_robust_linear`); `R=0` recovers sklearn exactly.
  **Trees/ensembles** alternate worst-delta / retrain `robust_reg.K` (=5) times.
  Its dial `label_eps` **is** D's radius, so on the sweep it tracks rho.
- **`wrapper.py`** — Maragno ensemble chance constraint: `(1-alpha)` of `P` models
  must satisfy each constraint. **The P models are not bootstrapped by default** —
  `scenario_source: "noise"` makes them a prefix of CP's `ScenarioBank`.
  `run_all.py` and `run_chemo_robust.py` build a bootstrap cache unconditionally
  and pass it in; under `"noise"` it is **dead code**, not evidence of
  bootstrapping. OS is never chance-constrained (`constraint_idxs` drops anything
  with `obj_weight != 0`); under `robustify_objective: false` the objective embeds
  a freshly trained **nominal** OS model. It used to embed perturbed bank draw 0 —
  **wrapper objective numbers from before 2026-08-15 carry that bias** (up to 1.08
  months on gastric).
- **`cp.py`** — Cutting Planes; see below.

### `uncertainty.py` — the shared set D

All three uncertainty-aware methods face one set and differ only in what they do
with it (cut lazily / chance-constrain / robustify the fit).

**Default (`geometry: "ellipsoid"`, since 2026-08-19):**
`D_c = {delta : ||delta||_2 <= R_c}`, `R_c = rho * scale(y_c) * sqrt(n)`.
It reads `rho` **alone**. `sqrt(n)` (not `sqrt(m)`) makes `rho=1` mean "L2 norm of
iid noise at one unexplained sd per row", which transfers across n=200 synthetic
and n=320 gastric.

**Ablation (`"box_l1"`):** `|delta_i| <= eps_c`, `||delta||_1 <= budget_frac*n*eps_c`,
`eps_c = eps_0 * scale(y_c)`. `eps_0`/`budget_frac` are **ignored** under the
ellipsoid — `budget_frac` cannot constrain an L2 ball, so the pair would be
non-identifiable there. Caveat: `uncertainty_set_from_config`'s *code* fallback is
still `box_l1`; only `config.yaml` carries the ellipsoid default.

- **Scale = out-of-fold residual sd** (`scale_stat: "oof_sd"`), so `rho = 1` is one
  unexplained sd per row. delta is added to labels *before* training and the model
  refit, so the radius must be label-space. Measured: gastric toxicities
  `sd(y)=0.288` vs `oof_sd` 0.249-0.293 (unexplained ratio 0.78-1.02); synthetic
  `sd(y)=0.545` vs `oof_sd` **0.128** against a true sigma of 0.100. Per outcome
  (gastric, temporal folds): dlt 0.2547, blood 0.2495, constitutional 0.2912,
  infection 0.2580, gi 0.2680, OS 2.05.
- **Not marginal `sd(y)`** (the `"sd"` ablation): mostly *signal*. At `eps_0=1` it
  is 4x too wide on synthetic and **CP will not converge** in 20 iterations.
- **Folds follow the problem's own scheme and the rows in hand** — temporal on
  gastric, so no leakage; a CV fold or `train_subsample_frac` draw estimates
  `scale(y_c)` from *its own* rows with cutoffs re-derived (`_cutoffs_from_years`).
  Broken until 2026-08-17 (`filter_constraints` dropped `train_pub_years`, so
  gastric silently fell back to random KFold). Nothing leaked, but no pre-fix
  gastric run used the temporal scheme; radii differ by -7.1% to +6.1%. Synthetic
  was always KFold by design. **No coverage claim is made.**
- **`chi2_radius` is unused (zero call sites) and must stay that way** while the
  scale is `oof_sd`. Three of its four assumptions fail — sigma is noise *plus*
  misspecification (~90% the latter on gastric), residuals are heteroskedastic
  (per-row SEs span 2.2x) and not independent (effective df < n); only Gaussianity
  holds, benignly. Coverage of delta is not coverage of the *decision* anyway. And
  chi2 concentrates: the 50% and 99% radii differ by 9.3% at n=320, so the level is
  an inert knob, never a conservatism dial.
- **Worst case: matched in magnitude, deliberately NOT in direction.** robust_reg
  is **directed** (`worst_case_label_shift`, or `R*r/||r||` on the ball); CP and the
  wrapper take **random** boundary draws. The reason is feasibility, not fairness:
  CP/wrapper scenarios become *embedded constraints*, and a directed adversary
  makes each as tight as D allows until the master admits nothing; robust_reg's
  adversary only shapes the fit, so it cannot cause infeasibility. Cost, measured
  on synthetic: best-of-B random reaches **1.07 eps** vs directed **1.67 eps**
  (~64%), and the gap *widens* under the ellipsoid. **Shared D means a shared set
  and equal budget, not equal adversary strength.**
- **`ScenarioBank`** draws B **vertices** of D and trains one model per draw per
  outcome, `random_state` fixed so the scenario is the only variation. Draw `b` is
  a pure function of `(seed, b)`, so the wrapper's P models are a **nested prefix**
  of CP's B — which is what makes the `alpha=0` == `tau->0` equivalence exact.
- **`coherent` is a grouping, not a flag.** The group shares one standardized
  direction (scaled by each outcome's radius); names in `coherent_exclude`
  (production: `["os_constraint"]`) draw independently. Vacuous on synthetic —
  bit-identical banks. Unknown names are ignored, since one config drives both
  problems. Justification and open objections:

  1. **RESOLVED (2026-08-15): OS does not belong on the shared direction.** OOF
     residual correlation (n=145, forward-chaining) is **+0.28** across non-DLT
     toxicity pairs on percentile labels, vs **+0.06** for OS against every
     toxicity. Matches the record-level-mismeasurement story, which never covered
     survival. Still *asserted*: truth is +0.28, we impose +1 in-group.
  2. **OPEN — incoherent is not the per-constraint worst case.** Its set is the
     product `D_1 x ... x D_C`, but separation pulls every constraint's model from
     one shared `b` and cuts that whole `b`, so the adversary searches B joint
     points, not `B^C`. No flag reaches the mixed adversary
     (`cut_whole_scenario=False` still cuts one `b`).
  3. **OPEN — the flag means different things in CP and the wrapper.** The
     wrapper's incoherent arm gives each constraint its own `z[c,p]`; CP's has no
     analogue. So the `alpha=0` == `tau->0` equivalence is a **coherent-arm
     result**, untested (and likely false) on the incoherent arm.
  4. **OPEN — DLT's draw is inconsistent.** `DLT = 1 - prod(1 - tox)` holds exactly,
     yet DLT takes the shared direction: collinear at +1.0000 before clipping. The
     group spends five radii on four d.o.f., and **no delta in D preserves the
     identity**. Right sign, overstated magnitude. Fixing it means changing
     `ScenarioBank._draw` and the label construction — not a config flip.

  Changing (2) is not free: the shared-`b` cut is what makes CP-at-`tau->0` equal
  the wrapper-at-`alpha=0` and makes permanent scenario exclusion sound. Cheapest
  next step is measurement (score both banks at the nominal `x*`), not a rewrite.
- **`rho`/`eps_0`/`budget_frac` are shared constants, not knobs.** Each method
  keeps exactly one dial: CP `tau`, wrapper `alpha`, robust_reg `label_eps`. Never
  pin D to robust_reg's calibrated optimum, to the GT ensemble (tunes to the
  judge), or to synthetic's known `noise_std` (CP then wins by construction). **Do
  not fix `tau->0` and `alpha=0` together**: at B=P that collapses CP into the
  wrapper. The sweep's fixed dials are **tau = 0.01** (`methods.cp.dist_tol_rel`)
  and **alpha = 0.2** (`methods.wrapper.alpha`), read from `config.yaml` by
  `_fixed_knobs` — *not* from `results/cv/*_robustness_knobs.json`, whose tau was
  selected under the old `d0` basis and means a different quantity
  (`--knobs-from-cv` restores it, with a warning; `--knobs` overrides).

### `solve_cp` — one driver, auto-selected strategy

`solve_cp` (`src/methods/cp.py`) runs one loop for every problem shape — *train
nominal -> build master -> solve for `x*` -> separate -> cut -> terminate* — and
**auto-selects** the separation strategy from the problem shape. **There is no
separation flag.**

- **basic** (synthetic: single LP, one learned constraint) — score every bank draw
  at `x*`, cut the single worst, ranked by the actual constraint model. The bank is
  a separation *pool*, not an embedded ensemble ("ensemble" here means the GT
  evaluator).
- **coherent** (gastric: many constraints / many `x*` / learned objective) — a
  scenario is one shared relabeling training every constraint and the epigraph
  objective jointly. Cut the worst scenario per iteration, ranked by **mean
  relative exceedance over (anchor x outcome) cells**.

Other CP settings:

- **Fixed bank** (`scenario_source: "noise"`). The legacy `"bootstrap"` path redrew
  each iteration while `d0` stayed frozen at iteration 0, comparing different
  samples; kept as an ablation. `uncertainty.cp_k_neighbors_*`, `cp_n_candidates`
  and `cp.distance` apply to that legacy path **only**.
- **`cut_whole_scenario: true`** cuts all of an accepted scenario's constraints —
  what makes permanent exclusion sound and matches the wrapper's per-replicate
  indicator.
- **`robustify_objective` stays off**: the ablation gave worse test feasibility
  *and* worse OS, and costs P extra OS embeddings in the wrapper.
- **`objective_monotone`** (off) adds a no-deterioration cut; redundant while
  nothing is removed, and the lever for re-enabling pruning/eviction safely. It
  would be **vacuous** under `robustify_objective: true` (`obj_expr` becomes the
  free epigraph variable).
- **Anchors** (`anchor_source`, `n_anchors` = 10, `anchor_method` = kmedoids) pick
  where `x*` is collected on contextual problems; `eval_mode` (`global` vs
  `per_anchor_nearest`) picks one shared master vs per-anchor masters.
- `_report_cp_diagnostics` prints cycles, objective regressions, permanent
  rejections and anchor-infeasibility rates — reported, never acted on.

### Four things that had to be right before CP converged on gastric

Each found by a run, not by reading. Together: from an exact period-4 cycle to
`status=optimal` in 19 iterations.

1. **`cp.mip_gap` = `1e-4`** (was hard-coded 0.01). At 1% on a gastric objective of
   ~10 the solver returns anything within ~0.1 while the distances separated are
   ~0.007, so cuts below the solver's own tolerance left `x*` unmoved. **Synthetic
   never hit this** (objective ~1.2, distances ~0.1) — same code, opposite regimes.
2. **Nothing is removed from the master** (`prune_slack_cuts` off,
   `cut_eviction: "reject"`). Removing a cut lets a previous `x*` recur — that is
   what a cycle is. The eligible set then strictly shrinks, so CP terminates in
   <= B iterations.
3. **The protected anchor set is fixed**, not recomputed — the anchors the *nominal*
   fit could serve (8/10 on gastric), tested **set-wise** so CP cannot trade one
   patient's feasibility for another's.
4. **Rejections are cached** (113/200 draws on gastric), which (3) makes sound.
   Without it rollbacks grew 8 -> 44 per iteration.

### tau — the CP stopping tolerance

**tau is measured in unexplained standard deviations** (`tolerance_basis: "scale"`,
default): each exceedance is divided by its own outcome's `oof_sd` and averaged
over (anchor x outcome) cells. So tau is a physical quantity in the **same units as
`rho`**, independent of seed/bank/B, and **the grid spans nominal** — a tau above
the iteration-0 distance stops before any cut (verified on synthetic: tau=1.0
returns the nominal objective in 1 iteration). The basic path keeps violations raw
for logging and multiplies instead of divides; the two paths log different units
but tau means the same thing.

- **One decade grid, wide range** (`[1.0, 0.1, 0.01, 0.001]`). Both paths max over
  draws, but a draw *scores* differently: basic has one cell (so, the raw
  violation); coherent means over `n_anchors x n_outcomes`, i.e. (violating
  fraction) x (mean exceedance among violators). Gastric's ~0.1 violating fraction
  is why its maxima sit near 0.03 against synthetic's ~0.98. That is **range, not
  meaning** — breadth is information. Read `[cp] basis=scale ... max iter-0 dist=`
  off a real run before assuming the grid brackets a **new** problem.
- **Do not pin the grid from measured distances**: those scale with D, i.e. with
  rho, so a fitted grid needs refitting after every rho and is circular. Order is
  (1) rho sweep at fixed tau, (2) tau ablation at the chosen rho.
- **The mean is the right statistic and is anchor-count stable.** A scenario
  breaking 2 of 20 cells badly is less dangerous than one breaking all 20
  moderately; a max ranks those backwards. Measured max iteration-0 distance:
  `n_anchors` 4 -> 0.0307, 8 -> 0.0315, 16 -> 0.0178 — no 1/n trend.
- **The scale is per outcome** (`_build_scale_map`), so cells are dimensionless and
  commensurable. Only a 1.17x spread on gastric (the percentile transform makes
  every toxicity `sd(y) ~ 0.2887`), but it would matter if OS were a constraint
  (`oof_sd` 2.05, ~8x); OS is the objective and is excluded from the map.
- **Expect `status="max_iterations"` at the small-tau end.** CP returns its
  incumbent, so sweeps do not crash — but report those cells as **capped**, not as
  a converged answer.
- **`d0` is a high quantile** (`d0_quantile`, 0.9) of iteration-0 distances, not
  their max — the max grows with B, and CP (B=200) and the wrapper (P=20) run at
  different B by design.
- **Legacy `tolerance_basis: "d0"`** makes tau a ratio to each problem's own `d0`
  while the stopping statistic is the **max**, so iteration 0 fails its own test and
  **no tau in [0.1, 1.0] reproduces nominal** (it takes tau=2.0).

**Why B differs.** The wrapper embeds all P models, so P is capped by MIP size; CP
embeds one cut per iteration. Measured on synthetic: wrapper at P=20 is 31,917
vars / 101,172 constrs, CP at B=200 is 6,380 / 20,236. At B=P on the identical
prefix, CP at `tau->0` and the wrapper at `alpha=0` return the same `x*` — **CP is
a lazy wrapper at alpha=0**.

### Data & problem instances (`src/data/`)

- **`generate.py`** — `ProblemInstance` is the universal container (decision vs
  context indices, `LearnedConstraint`s holding `MLModelData`, ground-truth models,
  `EvalOutcome`s, trust-region hull). `synthetic_nonlinear()` / `gastric_cancer()`
  build them; `filter_constraints()` subsets them.
- **`gastric_v11.py`** — data processing per the CL v11 notebook / Maragno D.1.
- **`gastric_model_specs.py`** — Table EC.10 embedded models and the EC.12 6-model
  GT ensemble; training seed 1 to match `run_MLmodels.py`.

**Frozen gastric picks** (`results/cv/gastric_selected_configs.json`, by `run_cv.py`
on R^2, *before* any robustness): **XGB** for DLT/blood/constitutional/infection,
**linear (ElasticNet)** for GI and OS. Synthetic does **not** go through
`run_cv.py` — its model is hard-coded in `config.yaml` (`rf`, 50 trees, depth 5)
and has never been CV'd.

The *marketing* (in-LP context) setting in the README is **not implemented**.

### Model training & embedding (`src/models/`)

- **`train.py`** — Linear (ElasticNet), SVM (LinearSVR), CART, RF, GBM, XGB, MLP;
  bootstrap helpers; `retrain_on_perturbed`.
- **`embed.py`** — Gurobi MIO encodings (trees as big-M leaf binaries, ensembles as
  averages/sums, MLP ReLU via binaries, Pipelines recurse through the scaler).
  **A new model type needs both `train.py` and `embed.py`.**

**Trees embed exactly, and the tolerances are load-bearing.** Leaf boxes get a
`SPLIT_EPS = 1e-5` band on *both* sides of each split (a bound left at theta is
still met at theta - `FeasibilityTol`, routing `x` to the wrong leaf), clamped so
no training row loses its leaf. Branch routing is done in **float32** because both
libraries traverse there. `IntFeasTol` is pinned to `1e-9`: big-M turns integrality
slack into `M * IntFeasTol` of x-slack and `M ~ 46`, so Gurobi's `1e-5` default
lets a free-`x` adversary extract `1.6e-2`. Verified to `<= 3e-7` over all rows and
perturbed refits — run `verify_embedding.py` after touching any of this.

### Evaluation (`src/evaluation/`)

- **`chemo_metrics.py`** — Table 6 metrics. All reported outcomes use the **GT
  ensemble** (6 sklearn models/outcome), *not* the embedded MIP models.
- **`metrics.py`** — synthetic feasibility/violation metrics.

**Protocol.** The GT ensemble is fit on all **416** gastric arms (train + test), a
*superset* of the constraint fit rows — which is why one full-data draw would
favour nominal and why the headline is **m-out-of-n subsampling**:
`--n-realizations 10 --subsample-frac 0.5`, common random numbers across methods,
`all_constraints` only. Every number is a mean over the **samestore cohort** (test
arms *every* method could prescribe for, recomputed per draw), so all of it is
**conditional on solvability**, with the solved fraction reported beside it.

### Calibration (`src/methods/calibrate.py`, `cv_calibrate.py`)

**The calibration target is rho\*, not each method's knob.** `run_rho_sweep.py`
sweeps the shared rho with tau/alpha fixed and reports **rho\*(method)** — the
largest rho meeting the target. D is shared at **every point of the sweep**, and
that curve is where the shared-D comparison is read. rho is never *fitted*; tau and
alpha become ablations at one rho (`--ablate`).

Rule: **largest grid rho with held-out feasibility >= 0.9 AND solved fraction
>= 0.5** (both defaults). The solved floor is the artifact guard — high feasibility
over few survivors is not a win, and it is what binds the wrapper. Measured
(2026-08-17, coherent, `results/rho_sweep/*_rho_star_coh.csv`):

| problem | method | rho\* | feas | obj | solved | master s | `bound` |
|---|---|---|---|---|---|---|---|
| gastric | cp | 1.0 | 0.934 | 9.63 | 0.59 | 494 | `grid_max` (censored) |
| gastric | robust_reg | 0.75 | 0.933 | 10.55 | 0.91 | 1.0 | `feasibility` |
| gastric | wrapper | 0.5 | 0.940 | 10.57 | 0.60 | 24 | `solved_floor` |
| gastric | nominal | — | | | | | never reaches 0.9 |
| synthetic | cp | 1.0 | 1.00 | -1.207 | 1.00 | 123 | `grid_max` (censored) |
| synthetic | others | — | | | | | never reach 0.9 |

Read `bound` before quoting a rho\*: `grid_max` means the grid ran out. **CP is
censored on both problems.**

- **Evaluation then runs each method at its own rho\***, so the match at evaluation
  is on held-out feasibility, **not** on D. **This is not wired up yet** — no runner
  reads `*_rho_star*.csv`; `run_chemo_robust.py` / `run_all.py` take D from
  `config.yaml` and do not force the ellipsoid. `method.tex` states the protocol;
  the code does not implement it.
- **rho\* is re-derivable without re-solving**: `{problem}_rho_curve{cell}.csv`
  carries every column, and `--rho-star-only` recomputes under a new
  `--feas-target` / `--min-solved` / `--exclude-capped` into `--out-suffix`. The
  criteria are written back as columns.
- Every cell carries `status`, `n_capped`, and a wall clock split into the
  **master** phase (for CP, the whole cut loop) and the **test-point** phase — the
  comparison CP's MIP-size claim rests on. Capped cells are **kept and flagged**.
- Outputs are scoped by cell (`_coh`/`_incoh`, `_matchbank`). CP samples D with
  B=200 against the wrapper's P=20, so a rho\* gap between them is confounded with
  sampling density — `--match-bank` removes it.
- **CP is exempt from `uncertainty.alpha`.** Its `cp_alpha` is **pinned at 0 at
  every call site**: a cut breaking the protected anchor set is rolled back and its
  draw permanently rejected. `run_chemo_robust.py` still *passes* `settings["alpha"]`
  into `cp_alpha`, but the body hard-codes `0.0` — the argument is **dead**.
- **Legacy paths.** `calibrate_strength` (strongest knob with training infeasible
  fraction <= `uncertainty.alpha`) runs only under `calibration.method: "alpha"`;
  `cv_calibrate.py` knob CV is per (method, coherence) cell, keyed
  `method@coherent` / `method@incoherent`.

## Config

`config.yaml` drives everything: `data.type` switches problem; `uncertainty.*`
defines the **shared D**; `uncertainty.alpha` is the **legacy-calibration target
only**; `methods.cp.*` holds the CP knobs; `methods.chemo.methods_to_run` /
`constraint_modes` select what the gastric runner executes, with
`methods.chemo.quick` overriding for `--quick`. CV model selections come from
`results/cv/*_selected_configs.json` and `*_gt_ensemble_configs.json` via
`--cv-configs`.

**Which coherence cell is the stronger adversary is not settled.** Coherent is
stronger *as implemented* — finite B covers the diagonal better than the product
set, and mean-over-cells scoring has a heavier right tail when exceedances move
together. That is a property of (mean scoring x shared `b` x finite B), not of the
sets: incoherent's set strictly contains coherent's, so under max-over-outcomes
scoring the ordering would flip.

## Known gaps (2026-08-19 deck's next steps)

Stated limitations of the current numbers, not bugs to fix silently.

1. **The rho axis is unresolved** — CP censored at the grid max on both problems;
   extend past rho=1. Unexplained: CP's dip at rho=0.5 on gastric; synthetic
   robust_reg at rho >= 0.5 (objective better than nominal at feasibility 0).
2. **The synthetic embedded model has never been CV'd** (hard-coded `rf` 50/5), and
   the sweep would ignore a CV result if one existed.
3. **Synthetic CV scoring is optimistic** — `make_cv_oracle` builds the oracle from
   the *same* model class it judges, so oracle and candidate share errors (the
   analytic `gt_constraints` are final-eval only). Val rows are unused and
   feasibility is quantized to `1/n_folds`. **Read the curve, not rho\***, there.
4. **The DLT draw is inconsistent** — objection (4) above.
5. **Nothing is confirmed on the test set under the ellipsoid**, so rho\* has no
   training-draw error bars.
6. **C-MICL (Ovalle et al. 2025) is not implemented** — no calibration split, no
   width model, and our folds are temporal while conformal needs exchangeability.
   Expected infeasible on gastric anyway: the 0.78-1.02 unexplained ratio gives a
   ~1.65 sd half-width on five constraints at once.
7. **Only two instances.** WFP food basket (Maragno's own wrapper setting) is the
   intended third.

## Presentations (`presentations/`)

- **`method.tex` is the standing method reference and must be true of HEAD.** It is
  the one deck edited in place. **A change to a method, to D, to the calibration or
  evaluation protocol, or to a config default the deck states, is not finished
  until `method.tex` says the new thing** — same change, not a follow-up. It
  carries no results.
- **`research_update_YYYY-MM-DD.tex`** are dated snapshots; a new deck supersedes
  the last rather than editing it. Current: **2026-08-19** (supersedes 08-10, then
  08-07). It includes `../results/figures/fig_rho_*.pdf`, so regenerating figures
  changes the deck — re-check it. Its "Next steps" slide mirrors **Known gaps**
  above; keep them in step. `chemo_replication_gaps.tex` and
  `robustcl_chemo_regimen.tex` are standalone one-offs.

Both kinds:

- **Be very concise.** Terse bullet fragments, one claim each, numbers over
  adjectives. Slides must fit — check the log for `Overfull \vbox` and cut text
  rather than shrinking the font.
- **Build with `latexmk`, never bare `pdflatex`.** `.latexmkrc` sets
  `$aux_dir = 'build'`, `$out_dir = '.'`, so the `.pdf` sits next to the `.tex` and
  aux files go to `build/`. `build.ps1` compiles every deck.

## Conventions / gotchas

- **GT is fixed and separate from the embedded models.** The GT ensemble is refit on
  the full clean cohort; only constraint/fit rows are resampled in realizations.
- **Robustness = outer m-out-of-n subsampling** (without replacement) of training
  rows against the fixed GT oracle — uncertainty over *training draws*, distinct
  from the inner set D the methods use.
- **`uncertainty.bootstrap_frac` (0.5)** is rows per bootstrap replicate as a
  proportion of `n_train`, per Maragno Sec. 4.4.1 — which specifies it for the WFP
  wrapper only; the chemo case study never uses the wrapper, so applying it to
  gastric is our extension. Half-size replicates carry ~39% unique rows vs ~63%, so
  the constraint binds harder (synthetic P=5: `obj = -1.2533` at 0.5 vs `-1.2769`
  at 1.0). Set `1.0` to reproduce pre-knob results.
- **`eps_0` interacts with `scale_stat`** — re-check it if you switch.
- **`variable_lb`/`variable_ub` split by role.** Treatment (decision) columns take
  their box from `X_fit`, because the optimizer *chooses* them; context columns stay
  on train+test **deliberately** — contexts are never chosen, the box only has to
  contain them, and narrowing to train would make every test solve infeasible since
  the split is temporal.
- `trust_region.py` confines gastric decisions to the convex hull of observed
  treatment vectors.
- `methods.robust_param.rho` shrinks decision-tree leaf regions by a margin from the
  split thresholds; applied across all methods.
