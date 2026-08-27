# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Ask me questions about anything unclear before implementing.
This file should only have necessary information and remain under 500 lines.

## What this is

Research code for **robust constraint learning**: a trained model `f(x;theta)` is
embedded as a constraint `f(x) <= b` inside an optimization problem, and noisy
training labels yield models whose "optimal" decisions violate the *true*
constraint. Six methods (`nominal`, `robust_reg`, `wrapper`, `cp`, `margin`,
`cmicl`) are benchmarked on a synthetic LP, the C-MICL DMA-MR reactor (Ovalle et
al. 2025, Sec. 5.1) and the OptiCL gastric-cancer case study (Maragno et al. 2025,
Table 6). The contribution is **Cutting Planes** (`src/methods/cp.py`): separate
the worst-case model over a fixed bank of relabelings drawn from a shared
uncertainty set D, adding one cut per iteration.

**Read `presentations/method.tex` first.** It is the standing method reference —
formulation, the three instances, each method, D and the scenario bank, the
calibration and evaluation protocols — and **it must stay true of HEAD**: a change
to a method, to D, to a protocol, or to a config default the deck states is not
finished until `method.tex` says the new thing (same change, not a follow-up). It
carries no results. This file is the *operational* layer: commands, cell naming,
which results are current, and the traps that reading the code would not reveal —
the code itself is well commented, so don't restate it here.

Dated `research_update_YYYY-MM-DD.tex` decks carry the numbers; a new deck
supersedes the last rather than editing it. Current is **2026-08-28** (the dial
sweep, 8 frames, complete); **2026-08-19** is the last rho-sweep deck.
**Known gaps = the "Next steps" frame of the current deck** — keep the two in step.

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
uv run python experiments/run_cv.py --problem {synthetic,reactor,gastric} [--ensemble]  # model / GT-ensemble CV
uv run python experiments/verify_embedding.py [--problem synthetic|reactor]  # MIP vs sklearn/xgb agreement
uv run python experiments/summarize_table6.py           # Table 6 CSV -> .csv/.tex
uv run python experiments/run_adversary_probe.py        # is the random bank a weak adversary?
uv run python experiments/probe_cmicl_cost_sampling.py  # does a SAMPLED c restore C-MICL's rate?
uv run python experiments/measure_clip_fraction.py      # how much of D the label bounds remove, per rho
sbatch experiments/submit_chemo_robust.sh               # 12h, 128G, 16 cpu
```

**SLURM**: every `submit_*.sh` sources `experiments/_activate_env.sh` rather than
running `module load miniforge` — that name is **not on every node of `mit_normal`**
(a 2026-08-25 rho sweep lost 2 of 6 tasks to it). The helper tries a cached conda
base, then several module names, then `conda.sh` by absolute path, and caches the
answer in `logs/.conda_base`. **Prime it once from a login node**
(`conda info --base > logs/.conda_base`) and no task depends on a module name
again. On failure it prints the node and `MODULEPATH` and **aborts the task**
rather than running python against the wrong interpreter.

## The two sweeps

Described in `method.tex` ("Calibration: primary axis is each method's own dial" /
"the rho axis (supporting)"). Operationally:

**Primary — the dial sweep** (`run_dial_sweep.py`): rho **fixed** at one of two
columns per problem, each method walking **its own** dial. The deliverable is the
**curve in objective x held-out feasibility**; the derived point `dial*` is, among
cells meeting `--feas-target` (0.9) at `solved_frac >= --min-solved` (0.5), the
best objective in the problem's own sense.

**Supporting — the rho sweep** (`run_rho_sweep.py`): the shared rho with tau/alpha
fixed, reporting `rho*(method)` — a question about D, not about decisions. It
**forces `geometry="ellipsoid"` regardless of `config.yaml`**.

`rho*` is a capacity, `m*` a price, `dial*` a point on a frontier; **none of them
is the comparison** — the curve is, in both files.

```bash
uv run python experiments/run_dial_sweep.py --problem gastric               # the primary axis
uv run python experiments/run_dial_sweep.py --problem reactor --rho-columns 1 2 3 4
uv run python experiments/run_dial_sweep.py --problem gastric --coherent    # the ablation cell
uv run python experiments/run_dial_sweep.py --problem gastric --cp-alpha-ablate   # coverage cap; walked whole
uv run python experiments/run_dial_sweep.py --problem reactor --search grid  # no monotonicity assumed
uv run python experiments/run_dial_test.py --problem reactor                # the TEST stage at dial*
uv run python experiments/run_dial_test.py --problem gastric --phases full  # OOS: the 96 X_test arms
uv run python experiments/plot_dial_sweep.py --all --suffix _incoh          # frontier + solved figures
uv run python experiments/plot_dial_sweep.py --problem gastric --suffix _incoh_s42 --xlim auto --compact  # DECK figure
sbatch experiments/submit_dial_sweep.sh                  # one array task per problem; test stage after

uv run python experiments/run_rho_sweep.py --problem gastric --ablate        # incoherent cell (default)
uv run python experiments/run_rho_sweep.py --problem gastric --coherent --ablate
uv run python experiments/run_rho_sweep.py --problem gastric --match-bank    # B=P
uv run python experiments/run_rho_sweep.py --rho-star-only --feas-target 0.8 --out-suffix _t080
uv run python experiments/pool_rho_seeds.py --problem gastric --cell _incoh  # spread across seeds
uv run python experiments/plot_rho_sweep.py --suffix _incoh
sbatch experiments/submit_rho_sweep.sh                   # PROBLEM(S)/COHERENCE/MATCH_BANK/RHO_GRID/SEEDS env
```

### Cells, checkpoints, and the resume trap

Both sweeps resume from a checkpoint keyed `(method@rho, dial)` **only** — no cell
token. **Always pass the cell flags**, or a second cell silently resumes the
first's rows and overwrites its curve. The cell suffix is `_coh`/`_incoh` +
`_matchbank` + `_f<n>` + `_m<model>` + `_s<seed>`.

**`--refresh` clears EVERY output of the cell** (scores, contexts, curve, star,
skipped). Pass it whenever a previous run of the same cell is **not comparable** —
a changed grid, a changed scoring rule. The star table is written once at the *end*
of the sweep, so a refreshed run that timed out leaves the **previous** star on
disk, which is the file `run_dial_test.py` reads.

**Seeds**: the dial grids run at **seed 42 only** (already |rho columns| x |dial
grid|). The rho sweep is where `--seed` is swept.

### The dial grid

| method | dial | rho columns | notes |
|---|---|---|---|
| `cp` | tau | gastric {0.5, 1.0}, reactor {2, 3} | **one fixed tau grid**, same on every column |
| `wrapper` | alpha | same | its P models are a prefix of CP's bank |
| `margin` | m | — | scored once; faces no D |
| `cmicl` | alpha | — | scored once; alpha = `1 - feas_target` is the protocol point |
| `nominal` | none | — | single reference point |
| `robust_reg` | — | — | **dropped**: its dial IS rho, so at fixed rho it has none |

**The reactor's rho columns are {2, 3}, measured rather than guessed** (2026-08-27).
They were {1, 2} on the reasoning that nominal misses the benzene target by ~4 units
of `F` and rho=1 buys ~2.2; the run at {3, 4} then showed rho=3 is already **past**
the transition — CP delivers 0.9 at the loosest tau on the grid and 1.0 everywhere
below it — so 4 only buys objective CP is not being asked to pay. {2, 3} brackets
the transition instead of sitting above it. `--rho-columns 1 2 3 4` for the whole span.

**Every grid spans its dial's full usable range** (widened 2026-08-27, each still a
strict superset of its predecessor so checkpoints resume rather than orphan). The
adaptive search makes length cheap — bracketing is O(log n) and the dead tails are
pruned unscored — and the 2026-08-27 curves showed the old grids were the binding
limit, not the methods: the reactor's margin (feasibility **0.1** at the old max
m=1.5), the reactor's CP (delivering at the old max tau=1.0) and gastric's wrapper
(delivering at the old max alpha=0.5) all had `bound="grid_end"`.

| dial | grid | the ends, and why they are there |
|---|---|---|
| `cp` tau | 10 … 1e-4 | bottom is the **mip_gap floor** `_resolve_tolerance` clamps to, so nothing below it is a distinct tolerance and a lower grid value would be a *mislabelled* tau; top is well above any iteration-0 distance, where CP stops before any cut and its curve meets nominal |
| `wrapper` alpha | 0 … 0.95 | only multiples of **1/P** (=0.05) are distinct levels; 0.95 = "1 of the P models must hold". **1.0 is excluded**: it requires none, removing the learned constraint entirely — weaker than nominal, not a looser wrapper |
| `margin` m | 0 … 5 | m=0 **is** nominal; the top is set by the reactor (`s_c`=2.19, so m=5 is `F_C6H6 >= 60.9`). Large m goes **infeasible, not conservative**, which shows as a falling solved fraction under `--min-solved` |
| `cmicl` alpha | 0.02 … 1.0 | 0.02 is the finest level `n_cal=80` can certify; 1.0 is `q = s_(1)`, the smallest nonconformity score — a real, very loose tightening, not a removed constraint (`conformal_quantile` takes `k = max(ceil((n+1)(1-alpha)), 1)`) |

- **The grid is SEARCHED, not walked** (`--search adaptive`, default): cells are
  ordered by robustness (`ROBUSTNESS_SIGN`), feasibility 0 prunes the less-robust
  side and `solved_frac < --min-solved` the more-robust side, so it bisects to the
  delivering interval's **least-robust end** (the protocol point) and spends the
  rest of `--max-evals` filling the band around it — the deliverable is a *curve*,
  so a bare bisection answers the wrong question. Checkpointed cells are free;
  C-MICL's protocol point is a `must_visit`, scored even when it does not solve.
- The prunes rest on **structural** monotonicity of objective and solvability (each
  dial nests the optimizer's feasible set), *not* on held-out feasibility, whose
  monotonicity is empirical. So the order check reports two things: a **violation**
  is a *verdict* inversion (`[search] NON-MONOTONE`, `monotone_note`) and the
  signal to re-run that series with `--search grid`; a **wobble** is a numeric dip
  changing no verdict (`feas_wobble`), recorded but not acted on. Over the
  committed curves 6 of 11 series wobble, none changes a verdict, and the search
  reproduces the exhaustive `dial*` in 11/11.
- Unscored cells go to `{problem}_dial_skipped{cell}.csv` with a reason, **never**
  into the curve as NaNs — the plot reads non-finite feasibility as "no solution on
  any fold", which is a *result*, and the two claims are opposites.
- `dial*` assumes **no** monotonicity (best objective among whatever delivered) and
  a `none` row is not a row of NaNs — `best_feasibility` / `best_feas_dial` /
  `best_feas_objective` / `best_feas_solved_frac` keep "0.88 everywhere" and "0.00
  everywhere" distinguishable.
- Both rho columns share one output CSV (the `rho` column tells them apart) and
  **one bank per (rho column, fold)** — a pure function of `(instance, D, seed, B)`,
  so one B=200 bank serves CP's whole tau grid and the wrapper's alpha grid.
- **`--cp-alpha-ablate` is walked whole, never searched** (its question is the
  *shape* of both columns) and is structurally inert on synthetic and the reactor,
  where the runner skips rather than emit a flat curve reading as a measurement.

**Plots**: `fig_dial_frontier_*` carries the **objective only** and
`fig_dial_solved_*` plots `solved_frac` on the **same x axis**, so a frontier point
is read by dropping straight down — load-bearing, because gastric margin at m=0.75
(obj 8.78, feas 1.000) reads as dominating until you see it solved **20.3%** of the
cohort. Colour = method, shade = rho column, shape = whether the method faces D,
**hollow = `solved_frac < --min-solved` and nothing else**; the panel is captioned
as TUNING scores. `--xlim auto` is **needed on gastric** (every cell sits between
0.85 and 1.00, so the fixed span hides the crossing) and changes no number;
`--compact` is the **deck aspect** and writes `..._slide.png` rather than
overwriting the report figure.

### The test stage — the sweep TUNES, `run_dial_test.py` TESTS

Every sweep number is a fold score under the judge that instance tunes against, and
`dial*` is fitted to exactly that column, so quoting a tuned dial's own tuning
score is the error this stage exists to avoid. It reads
`{problem}_dial_star{cell}.csv`, holds each method at its `dial*`, and re-scores
under a judge the dial never faced. Three phases, not interchangeable:

| phase | what it is | what it is not |
|---|---|---|
| `folds` | the sweep's own folds re-solved at `dial*`, truth-judged; a rate with a spread over folds | **not held out** — `dial*` was chosen on these folds |
| `full` | one refit on **all** rows, one decision per method | one **bit** of feasibility per method; no spread |
| `subsample` | **gastric only, default-on**: `full` repeated over `--n-realizations` (10) m-out-of-n draws of `--subsample-frac` (0.5) of the constraint fit rows | refused elsewhere — no other instance has a cohort to prescribe for |

The judge: **synthetic** the analytic `f_true`; **reactor** the **ODE**
(`make_gt_oracle`), never the proxy that tuned `dial*`; **gastric** the
**full-cohort GT ensemble fit on all 416 arms** (`instance.eval_outcomes[*].gt_fn`),
prescribing for the **96 `X_test` arms** — **not** `make_cv_oracle`'s train-only
320-arm ensemble, which is the tuning proxy. Two separate objects, and confusing
them is the easy mistake here; on gastric both judge *and* cohort change between
tuning and test.

`subsample` is the repo's standing robustness protocol (the one every Table 6
number is reported over) applied at `dial*`, and the only place the stage can
honestly report **training-draw** variation. It resamples the fit rows only — the
oracle is built once off the full-data instance — under **CRN**
(`subsample_seed = bootstrap_seed + 1000*(r+1)`, byte-identical to
`run_chemo_robust.run_robustness_probe`), so realization `r` is the same draw there
and the comparison is paired. `feasibility` is conditional on each series' own
solved arms; `feasibility_samestore` is over the arms **every tested series** solved
that realization — the Table 6 convention, and the one that does not flatter
whoever solved least. **The tail is the point**: `feas_worst_case` and `feas_q10`
sit beside the mean for both cohorts, and at 10 draws the min *is* one draw, a
range and never a bound. A series with no `dial*` is skipped with the reason
printed; `RUN_TEST=0` / `TEST_PHASES=full` narrow the stage in
`submit_dial_sweep.sh`.

**Neither single-decision instance has a row-level train/test split, and that is
structural**: `synthetic_nonlinear` and `reactor_micl` set `X_test` empty, so the
CV folds are the only row structure and the separation is the *judge*, not the
rows. Gastric alone has a held-out cohort, and its fold means resolve to ~0.003
(4 temporal folds x ~96 contexts) where synthetic (5 folds) and the reactor (10)
have one decision per fold — feasibility **quantized to 0.2 / 0.1**, one fold
flipping being the whole difference, so **read the curve there, not the protocol
point**. What rescues the *ordering* is that the comparison is **paired**:
`su.folds` is built once and `FoldCache` shares the fold instance and its bank
across the whole dial grid.

### rho sweep specifics

Grids: rho `[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]`, tau `[1.0, 0.1, 0.01, 0.001]`,
alpha `[0.0, 0.1, 0.2, 0.3, 0.5]`. It is **per-method parameter** (`SWEEP_PARAM`):
rho for cp/wrapper/robust_reg, the RHS margin `m` for margin, nothing for
nominal/cmicl (scored once as a reference level; `rho_star` NaN,
`bound="no_param"`, since `grid_max` would assert they absorbed the whole grid).
The `param_swept` column keeps them apart — **never read an `m` as a rho**.
`rho*` = largest grid value with feasibility >= 0.9 **and** solved fraction >= 0.5
(the artifact guard); **read `bound` before quoting one** — `grid_max` means the
grid ran out, and CP is censored on both problems. `--rho-star-only` recomputes it
from `{problem}_rho_curve{cell}.csv` without re-solving. Fixed dials come from
`config.yaml` (`tau = 0.01`, `alpha = 0.2`), **not**
`results/cv/*_robustness_knobs.json`, whose tau was selected under the old `d0`
basis and means a different quantity.

**`--seed` is the bank axis.** D is *sampled* (CP cuts against B=200, the wrapper
embeds P=20), so one curve cannot separate the method from the bank it drew. The
flag reseeds the `ScenarioBank` **and** every model's `random_state` (so `oof_sd`,
hence `R_c`, wobbles a few percent) while data and evaluation folds stay
bit-identical; `pool_rho_seeds.py` writes `*_pooled.csv`, and with ~3 seeds that is
a **range, not a CI**. `submit_rho_sweep.sh` runs one **(problem, seed)** pair per
array task, seed fastest; ablations run on the **first seed of each problem**
(tasks 0 and 3), not global task 0. `PROBLEM=` singular wins over `PROBLEMS` but
**`--array` must be narrowed with it**. It omits `robust_reg`.

**A seed is not a draw for every method.** It moves cp/wrapper's bank,
robust_reg's refits and cmicl's calibration split; it does **not** reach `nominal`
at all, and reaches `margin` only through the fold split — which on **gastric is
temporal and ignores the seed**. A pooled sd of 0 for nominal, or for gastric
margin, is **structural**, not evidence of stability.

## Which results are current

- **Only `results/rho_sweep/` and `results/figures/fig_rho_*` / `fig_dial_*`** —
  ellipsoid geometry, fixed temporal folds. Inside it the **2026-08-27 dial cells
  are newest and the only tested ones**: gastric with all three phases, the reactor
  with a full curve, a star table and an ODE-judged test stage.
- **Both committed dial cells are now GRID-SUPERSEDED, not invalidated**
  (2026-08-27). The dial grids were widened and the reactor's rho columns moved
  {3, 4} -> {2, 3}, so the committed curves and star tables are the answer to a
  narrower question than the code now asks: on the reactor `margin`, reactor `cp`
  and gastric `wrapper` the star sits at `bound="grid_end"`, and gastric `cmicl`
  has no star at all. **Re-run both cells WITHOUT `--refresh`** — every new grid is
  a strict superset of the one that produced those rows and the scoring rule is
  unchanged, so the search replays them free and spends its budget only on the new
  cells. The reactor's rho=4 series simply drops out of the rewritten curve (the
  curve is built from the run's own rows; the checkpoint keeps them, unread).
- **No gastric result outside those dial cells is current** (2026-08-21): the
  production draw is now **incoherent** and DLT is **derived** rather than drawn,
  so the default cell is `_incoh` and the committed `_coh` curves are neither the
  production cell nor a valid `--coherent` ablation of it — re-run both.
  `gastric_dial_*_incoh_s42.csv` / `gastric_dial_test*_incoh_s42.csv` **are**
  current. Synthetic and the reactor declare no label link and have one outcome, so
  coherence is vacuous there and they are unaffected.
- Everything else (`results/gastric/` all <= 2026-08-13, `results/synthetic/`,
  `adversary_probe/`, `cv/*_robustness_knobs.json`) is `box_l1` and/or
  random-KFold-scaled. **Never read a `box_l1` number against an ellipsoid one.**
- **Gastric `robust_reg` rows predate `clip_labels`** (2026-08-21), so its curve is
  against the raw ball while CP's and the wrapper's are against the clipped one.
- **Pre-2026-08-20 objective columns are not gap-comparable across methods**
  (nominal/robust_reg ran at `mip_gap` 0.01 while CP/wrapper ran at 1e-4).
- **Nothing synthetic from before 2026-08-21 carries over**: `n_train` went
  200 -> 2500, and the dropped-hyperparameter fix changed which model is embedded
  (the CV *JSONs* reproduce bit-identically; results against them do not).
- **Any result read against a tuned GT ensemble from before 2026-08-21 is stale** —
  members are now tuned inside their deployment `Pipeline`, and on gastric that
  ensemble is the *evaluation* oracle.
- **Every gastric number from before the 2026-08-21 `IterativeImputer` fix is
  against different labels**: blood imputation never converged and the labels were
  an arbitrary iterate; 394/495 rows moved on at least one outcome.

## Repo map

- `src/data/generate.py` — `ProblemInstance`, the universal container, plus
  `synthetic_nonlinear()`, `gastric_cancer()`, `filter_constraints()`. Beside it:
  `gastric_v11.py` (CL v11 / Maragno D.1 processing), `dma_mr.py` (vendored reactor
  RHS — `opyrability` is **not** a dependency), the `*_model_specs.py`.
- `src/models/train.py` / `embed.py` — Linear/SVM/CART/RF/GBM/XGB/MLP and their
  Gurobi encodings. **A new model type needs both files.**
- `src/methods/` — `nominal.py` holds the MIP scaffolding every method imports;
  then one file per method, plus `uncertainty.py` (the shared D and `ScenarioBank`)
  and `calibrate.py` / `cv_calibrate.py`.
- `src/evaluation/` — `chemo_metrics.py` (Table 6; every reported outcome uses the
  **GT ensemble**, never the embedded models) and `metrics.py` (synthetic; the
  **only** place the analytic `f_true` is allowed).
- `experiments/method_builders.py` — **every runner builds its solvers here**
  (`build_method`, `cp_solver`): the one place a solver's argument list and the
  cross-cutting decisions (single `mip_gap`, `cp_alpha` pinned at 0, the shared
  uncertainty set) live, after four near-copies each needed patching separately.
  The *problem*-specific half stays with the problem (`_resolve_run_settings` for
  gastric, `method_builders.synth_settings` otherwise).

**Frozen CV picks** (`results/cv/*_selected_configs.json`, chosen on R^2 *before*
any robustness): gastric **XGB** for DLT/blood/constitutional/infection and
**linear (ElasticNet)** for GI and OS; synthetic **`mlp` (50,), lbfgs, alpha
0.01**; reactor **`mlp` (10,5,2)**. All three CV stages reproduce bit-identically,
so a re-run that differs means a **code or label** change, not noise. `run_cv.py`
**overwrites** those files — `git checkout` them after an inspection run unless you
mean to re-freeze.

## Config

**Every block is named for its scope.** `problem.type` switches problem;
`synthetic.*` / `reactor.*` carry instance settings; `default_model` is the
embedded-model fallback, beaten by `reactor.model` then by
`results/cv/*_selected_configs.json`; `optimization.mip_gap` is the **one solver
gap** every method runs at; `uncertainty.*` defines the **shared D** (`alpha` there
is the legacy-calibration target only, and `clip_labels`, `derive_linked_labels`
and `coherent` are shared in scope but in effect on **gastric alone**);
`cv_calibration.*` holds the knob-CV folds and grids;
`methods.{cp,wrapper,robust_reg,cmicl,margin}` one dial each; `methods.chemo.*` is
gastric only.

- **`methods.cmicl` and `methods.margin` read (almost) nothing from
  `uncertainty.*`** — the first calibrates against held-out residuals, the second
  shifts the RHS by a fitted dial. The exception is `methods.margin.scale_stat`
  (default `null` = `uncertainty.scale_stat`): the margin is quoted in the same
  unexplained-sd units as rho and tau, so it must read the same estimator. There is
  deliberately **no** `methods.margin.robustify_objective` — a margin on a learned
  objective term is a constant, moving the reported objective but not `x*`.
- CP's `separation` is **not** a dial (defaults `"auto"`, follows
  `uncertainty.coherent`); `cut_rollback` is structural, and so are cmicl's
  `cal_frac` / `width_*` / `multiplicity`. None of them is swept.
- `uncertainty_set_from_config`'s *code* fallback is still `box_l1` (only
  `config.yaml` carries the ellipsoid default), and the `clip_labels` /
  `derive_linked_labels` dataclass fields still default `False`.

## Conventions / gotchas

- **tau is FIXED before the run**, on both sweeps, every problem, both separation
  paths — one grid, the same at every rho. **Never pin it from measured
  distances**: that makes tau a function of the bank, of B and of which folds were
  looked at. `CPHistory.iter0_tau` is a *diagnostic* saying whether the fixed grid
  brackets the problem and **must not be fed back**; `run_dial_sweep.py` did place
  the grid that way between 2026-08-25 and 2026-08-26, and that is removed and
  **must not come back**.
- **tau does not transfer between CP's separation paths**: the coherent path
  averages a draw over units, the incoherent one does not, so incoherent distances
  run ~C times larger (the same gastric bank reads max iteration-0 distance
  **0.0623 coherent vs 0.2441 incoherent**). Read `[cp] basis=scale ... max iter-0
  dist=` off a real run before assuming the grid brackets a new problem or path.
- **There is no separation flag on either sweep runner** (removed 2026-08-26);
  `methods.cp.separation` in `config.yaml` is the only way to force a mismatch, and
  a forced cell is not comparable to the matched pair it would be read against —
  `run_dial_test.py` never had the flag, so it cannot find a `_sep*` cell.
- **Expect `status="max_iterations"` at the small-tau end.** CP returns its
  incumbent so sweeps do not crash — report those cells as **capped**.
- **`rho`/`eps_0`/`budget_frac` are shared constants, not knobs.** Never pin D to
  robust_reg's calibrated optimum, to the GT ensemble (tunes to the judge), or to
  synthetic's known `noise_std` (CP then wins by construction). **`eps_0` interacts
  with `scale_stat`** — re-check it if you switch. **`chi2_radius` has zero call
  sites and must stay that way** while the scale is `oof_sd`: three of its four
  assumptions fail, and it concentrates (50% and 99% radii differ by 9.3% at
  n=320), so its level is an inert knob, not a conservatism dial.
- **`cp_alpha` is pinned at 0 for every result.** It became a real count budget in
  `cp.py` on 2026-08-25 so the coverage-cap ablation can walk it; at 0 every
  existing call is bit-identical. `run_chemo_robust.py` still passes
  `settings["alpha"]` into it and the body ignores it — that argument stays dead.
- **GT is fixed and separate from the embedded models**, refit on the full clean
  cohort; only constraint/fit rows are resampled. **Robustness = outer m-out-of-n
  subsampling** without replacement against that fixed oracle — uncertainty over
  *training draws*, distinct from the inner set D. `uncertainty.bootstrap_frac`
  (0.5) is Maragno Sec. 4.4.1's WFP-wrapper setting; applying it to gastric is our
  extension (set `1.0` to reproduce pre-knob results).
- **Tree/MLP embedding tolerances are load-bearing**: `SPLIT_EPS = 1e-5` on both
  sides of each split, float32 branch routing, and `IntFeasTol` pinned to `1e-9`
  because big-M turns integrality slack into `M * IntFeasTol` of x-slack. **Run
  `verify_embedding.py` after touching any of it**, and route its `--problem
  synthetic|reactor` through `run_sweep._synth_instance` / `_reactor_instance`,
  **not** `run_adversary_probe.build_instance`, which never loads the CV selection.
- **`variable_lb`/`variable_ub` split by role**: treatment columns take their box
  from `X_fit` (the optimizer *chooses* them); context columns stay on train+test
  **deliberately** — the box only has to contain them, and the split is temporal.
- **The reactor's `cost_vector` is FIXED at ones** (a new `c` is a different
  problem, not a new sample of this one), and C-MICL's reactor feasibility is a
  property of that distribution: **0.99** under `c_i ~ U(0,1)` vs **0.11** under
  `c_i ~ U(0,1)/span_i` (`probe_cmicl_cost_sampling.py`, a diagnostic that changes
  nothing in the evaluation). **Never quote a C-MICL reactor feasibility without
  naming the `c` distribution**, and do not quote our `F_C6H6` values as
  reproductions of theirs (Table 1 agrees to a consistent -1.6%).
- **Gastric C-MICL is measured infeasible** at alpha=0.1 under both multiplicity
  settings (half-widths 1.33-1.73 sd(y), i.e. 0.38-0.50 against an rhs of 0.6, on
  five constraints at once), which is why its grid runs the **full [0.02, 1]** of a
  miscoverage level — where it *first solves* is the result. On the 2026-08-27 run
  alpha 0.1 and 0.3 solved **nothing** and 0.5 (the old grid top) solved **13.8%**
  of contexts, under the 0.5 floor, so its star row was empty and the answer was
  known only to be "above 0.5". Budget for it: proving the marginal case infeasible
  costs **176 s** against nominal's 0.9 s.
- **The evaluate-at-`rho*` protocol is stated in `method.tex` but NOT wired up** —
  no runner reads `*_rho_star*.csv`; `run_chemo_robust.py` / `run_all.py` take D
  from `config.yaml` and do not force the ellipsoid.
- The *marketing* (in-LP context) setting described in the README is **not
  implemented**.

## Presentations

- Build with **`latexmk`, never bare `pdflatex`** (`.latexmkrc` sets
  `$aux_dir = 'build'`, `$out_dir = '.'`); `build.ps1` compiles every deck.
- **Be very concise**: terse bullet fragments, one claim each, numbers over
  adjectives. Slides must fit — check the log for `Overfull \vbox` and **cut text
  rather than shrinking the font**.
- **Regenerating a figure changes a deck.** 08-19 includes `fig_rho_*.pdf`; 08-28
  includes `fig_dial_frontier_gastric_incoh_s42_slide.png` and
  `fig_dial_frontier_reactor_incoh_f10_mmlp_s42_slide.png` — the **`--compact`
  slide variants** (a deck pointed at the report figure of the same stem squeezes
  the panel to under half the frame), and **PNG**, not PDF. Rebuild and re-check.
- `chemo_replication_gaps.tex` and `robustcl_chemo_regimen.tex` are standalone
  one-offs.
