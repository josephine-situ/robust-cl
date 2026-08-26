# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Ask me questions about anything unclear before implementing.

## What this is

Research code for **robust constraint learning**: a trained model `f(x;theta)` is
embedded as a constraint `f(x) <= b` inside an optimization problem, and noisy
training labels yield models whose "optimal" decisions violate the *true*
constraint. Six methods are benchmarked on a synthetic LP, the C-MICL DMA-MR
reactor (Ovalle et al. 2025, Sec. 5.1) and the OptiCL gastric-cancer case study
(Maragno et al. 2025, Table 6).

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
uv run python experiments/run_cv.py --problem synthetic --ensemble  # + the synthetic CV oracle
uv run python experiments/run_cv.py --problem reactor --ensemble    # DMA-MR model + proxy judge
uv run python experiments/summarize_table6.py           # Table 6 CSV -> .csv/.tex
uv run python experiments/verify_embedding.py           # MIP vs sklearn/xgb agreement
uv run python experiments/verify_embedding.py --problem synthetic   # CV-selected mlp
uv run python experiments/verify_embedding.py --problem reactor     # CV-selected mlp
uv run python experiments/run_adversary_probe.py        # is the random bank a weak adversary?
uv run python experiments/probe_cmicl_cost_sampling.py  # does a SAMPLED c restore C-MICL's rate? (diagnostic; c stays ones everywhere else)
sbatch experiments/submit_chemo_robust.sh               # 12h, 128G, 16 cpu
```

**Every `submit_*.sh` sources `experiments/_activate_env.sh`** rather than running
`module load miniforge` itself. That name is **not present on every node of**
**`mit_normal`**: on 2026-08-25 a 6-task rho sweep lost tasks 4-5 (reactor, seeds 7
and 13) to `Lmod ... module(s) are unknown: "miniforge"` while tasks 0-3 ran on the
same script. The helper tries a cached conda base, then several module names (plain
and `--ignore_cache`), then conda.sh by absolute path, and on success caches the
resolved base in `logs/.conda_base` so later tasks skip Lmod entirely. Prime it once
from a login node — `conda info --base > logs/.conda_base` — and no task depends on a
module name again. On failure it prints the node, `MODULEPATH` and the conda modules
that node *does* have, then **aborts the task** (the caller's `set -e`) instead of
running python against the wrong interpreter.

**The DIAL sweep is the headline experiment; the rho sweep is now supporting
evidence** (2026-08-25). `run_dial_sweep.py` walks each method along **its own
dial at a FIXED rho** and the primary output is a scatter in **objective x
held-out feasibility** — because the claim the contribution rests on is "at equal
feasibility, whose decisions are better", which is a statement about two axes at
once. `run_rho_sweep.py` answers a different question ("how much assumed
uncertainty does each method absorb"), which is about D; it is still run, still
current, and is no longer the headline.

```bash
uv run python experiments/run_dial_sweep.py --problem gastric               # the primary axis
uv run python experiments/run_dial_sweep.py --problem reactor --rho-columns 1 2 3
uv run python experiments/run_dial_sweep.py --problem gastric --coherent    # the ablation cell
uv run python experiments/run_dial_sweep.py --problem gastric --cp-alpha-ablate
uv run python experiments/run_dial_test.py --problem reactor                # the TEST stage at each dial*
uv run python experiments/run_dial_test.py --problem gastric --phases full  # OOS: the 96 X_test arms
uv run python experiments/plot_dial_sweep.py --all --suffix _incoh          # -> fig_dial_frontier_*.pdf
uv run python experiments/measure_clip_fraction.py       # how much of D the label bounds remove, per rho
sbatch experiments/submit_dial_sweep.sh                  # one array task per problem; runs the test stage after
```

**The sweep TUNES; `run_dial_test.py` TESTS** (2026-08-26). Every number in the
sweep is a fold score under the judge that instance tunes against, and `dial*` is
fitted to exactly that column — so quoting a tuned dial's own tuning score as the
result is the error the test stage exists to avoid. It reads
`{problem}_dial_star{cell}.csv`, holds each method at its own `dial*`, and scores
again under a judge the dial never faced. Two phases, not interchangeable:

| phase | what it is | what it is not |
|---|---|---|
| `folds` | the sweep's own folds re-solved at `dial*`, truth-judged. A rate, with a spread over folds | **not held out** — `dial*` was chosen on these folds |
| `full` | one refit on **all** rows, one decision per method | one **bit** of feasibility per method; no spread |

The judge per problem: **synthetic** the analytic `f_true` (`gt_constraints[0]`),
**reactor** the **ODE** (`make_gt_oracle`) — never the proxy that tuned `dial*`,
whose own boundary error is Known gap #8 — and **gastric** the **full-cohort GT
ensemble, fit on all 416 arms** (`instance.eval_outcomes[*].gt_fn`, the fixed
evaluation oracle every Table 6 number is against), prescribing for the **96
`X_test` arms**. That is **not** `make_cv_oracle`'s train-only 320-arm ensemble,
which is the tuning proxy the sweep scores `dial*` against — the two are separate
objects and confusing them is the easy mistake here. So on gastric both the judge
and the cohort change between tuning and test.
Outputs: `{problem}_dial_test{cell}.csv` (summary per (method, phase)) and
`{problem}_dial_test_points{cell}.csv` (per fold/context). A series with no
`dial*` is **skipped with the reason printed** — there is no tuned dial to test.
`RUN_TEST=0` / `TEST_PHASES=full` narrow it in `submit_dial_sweep.sh`.

**Neither single-decision instance has a row-level train/test split, and that is
structural**: `synthetic_nonlinear` and `reactor_micl` set `X_test` empty (there
are no contexts to hold out), so the CV folds are the only row structure and the
test stage's separation is the *judge*, not the rows. Gastric is the one instance
where a held-out cohort exists.

The grid:

| method | dial | rho columns | notes |
|---|---|---|---|
| `cp` | tau | gastric {0.5, 1.0}, reactor {1, 2} | **one fixed tau grid**, same on every column |
| `wrapper` | alpha | same | its P models are a prefix of CP's bank |
| `margin` | m | — | scored once; faces no D |
| `cmicl` | alpha | — | scored once; alpha=0.1 flagged as the protocol point |
| `nominal` | none | — | single reference point |
| `robust_reg` | — | — | **dropped**: its dial IS rho, so at fixed rho it has none |

- **Both rho columns share one output CSV.** The `rho` column tells them apart and
  one file per (problem, coherence, seed) is what the plot reads. The checkpoint
  key is `(method@rho, dial)`, so resume is unchanged.
- **tau is FIXED BEFORE THE RUN.** One absolute grid, `TAU_GRID` =
  `[1.0, 0.1, 0.01, 0.001]`, the same on every rho column and every problem, in
  unexplained-sd units (`--tau-grid` to change it). **tau is a parameter of the
  method**, set in advance exactly as rho and the margin's `m` are — it is never
  read back off the run.
  **Placing the grid from the iteration-0 separation distance was tried and is
  removed (2026-08-26); do not reintroduce it.** It made tau a function of the
  bank, of `B` and of which folds were probed, so the same nominal tau meant a
  different tolerance in every cell and the primary figure's x-axis stopped being
  one quantity. It also cost an extra CP run per (rho column, fold), and it
  silently misplaced the grid whenever the probed fold was not the fold with the
  largest distance — measured on the 2026-08-26 run, the `tau_frac=1` endpoint
  cut on **5/10** reactor folds at rho=1, **6/10** at rho=2 and **3/4** gastric
  folds at rho=1.0 (there, objective **10.87 against nominal's 11.30**). Whether
  the top of the fixed grid happens to stop before any cut is a **property of the
  run**, reported by `[cp] ... max iter-0 dist=` and by `status` — not something
  the grid is bent to guarantee. `{problem}_tau_probe{cell}.json` is gone, and a
  stale one is deleted on the next run.
  **Every dial curve produced before 2026-08-26 was scored on a probe-placed
  grid** and its tau axis is not comparable across rho columns — re-run.
- **One shared bank per (rho column, fold)** (`cv_calibrate.FoldCache`,
  `uncertainty.build_bank_for_instance`). A bank is a pure function of
  `(instance, D, seed, B)` — neither tau nor alpha reaches it — so one B=200 bank
  serves CP's whole tau grid *and* the wrapper's alpha grid. Verified on
  synthetic: **2 constructions for 2 folds across 5 solves** (was 10), and CP at
  the smallest tau still equals the wrapper at alpha=0 to the last digit. Cost:
  the cache holds every fold's bank at once, so memory is `len(folds)` x the
  models; it is dropped between rho columns for that reason.
- **Per-context records**: `{problem}_dial_contexts{cell}.csv` carries
  `(fold, context_idx, solved, feasible, objective)` per cell. Primary scoring is
  **unchanged** — still conditional on each cell's own solved contexts, and that
  independence is load-bearing — but the objective is the deliverable now, and a
  conditional mean of it flatters whoever solved least. The same-cohort comparison
  is derivable from these rows afterwards rather than made primary.
- **The plot encodes solved fraction as marker size.** Without it gastric margin
  at m=0.75 (obj 8.78, feas 1.000, solved 0.203) reads as a dominating point when
  it is a 20% cohort. Pareto direction follows `oracle.objective_sense`, carried
  as a column, because gastric maximises survival and the reactor minimises cost.
- **Seeds**: the full dial grids run at **seed 42 only**. Repeating three seeds
  triples a grid that is already |rho columns| x |dial grid|; re-run the protocol
  points at 7 and 13 instead if the curves come out non-monotone.

Pre-registered expectations, so a surprise reads as a result and not a bug:
**gastric C-MICL will be infeasible over much of its alpha grid** (measured
infeasible at 0.1 under both multiplicity settings; `n_cal = 80` means
alpha >= 0.02 for a finite `q` at all), which is why its grid is extended
**upward** to 0.2/0.3/0.5 — where it *first solves* is the result, and proving the
marginal case infeasible costs 176 s against nominal's 0.9 s. **Reactor rho=2 may
be short**: nominal misses by ~4 units of `F` and rho=1 buys ~2.2, so add 3.

**The rho sweep.** `run_rho_sweep.py` **forces
`geometry="ellipsoid"` regardless of `config.yaml`**:

```bash
uv run python experiments/run_rho_sweep.py --problem gastric --ablate    # incoherent cell (default since 2026-08-21)
uv run python experiments/run_rho_sweep.py --problem gastric --coherent --ablate    # the ablation
uv run python experiments/run_rho_sweep.py --problem gastric --match-bank    # B=P
uv run python experiments/run_rho_sweep.py --problem synthetic   # 10 folds; feas quantized to 0.1
uv run python experiments/run_rho_sweep.py --problem reactor     # ODE ground truth
uv run python experiments/run_rho_sweep.py --seed 7          # repeat the sweep on another bank
# The sweep is PER-METHOD PARAMETER (run_rho_sweep.SWEEP_PARAM): rho for
# cp/wrapper/robust_reg, the RHS margin m for margin, nothing for nominal/cmicl
# (scored once, repeated as a reference level). One grid, because all of them are
# in unexplained-sd units. --margin-grid gives margin its own values.
uv run python experiments/run_rho_sweep.py --problem gastric --methods nominal cp wrapper margin --ablate  # + the tuned-nominal baseline
uv run python experiments/run_rho_sweep.py --problem reactor --methods nominal cp wrapper cmicl --ablate
uv run python experiments/run_rho_sweep.py --rho-star-only --feas-target 0.8 --out-suffix _t080
uv run python experiments/pool_rho_seeds.py --problem gastric --cell _incoh  # spread across seeds
uv run python experiments/plot_rho_sweep.py --suffix _incoh  # -> results/figures/fig_rho_*.pdf
sbatch experiments/submit_rho_sweep.sh                       # seed job array; PROBLEM/COHERENCE/MATCH_BANK/RHO_GRID/SEEDS env
```

Grids: rho `[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]`, tau `[1.0, 0.1, 0.01, 0.001]`,
alpha `[0.0, 0.1, 0.2, 0.3, 0.5]`. Runs resume from a checkpoint keyed
`(method@rho, knob)` **only**, so **always pass the cell flags** — otherwise a
second cell resumes the first's rows and overwrites its curve. The cell is
`_coh`/`_incoh` + `_matchbank` + `_f<n>` (`--n-folds`) + `_m<model>` + `_s<seed>`
(`--seed`). CP's separation path follows the coherence the cell is already named
for, so it needs no token of its own; only a **forced** `--separation` mismatch
adds one (`_sepcoher`/`_sepincoh`).

**`--seed` is the bank axis** (since 2026-08-21). D is *sampled*: CP cuts against
B=200 draws, the wrapper embeds P=20, so one curve cannot separate the method from
the bank it drew. The flag reseeds the `ScenarioBank` **and** every model's
`random_state` — which moves `oof_sd`, so `R_c` wobbles a few percent between
seeds (synthetic fold 1: 0.1314 at seed 7 vs 0.1238 at seed 42) — while the data
and the evaluation folds keep `uncertainty.bootstrap_seed` and stay bit-identical.
`pool_rho_seeds.py` writes `*_pooled.csv`: mean/sd of feasibility per (method,
rho) and rho* per seed. With ~3 seeds that is a **range, not a CI**, and it leaves
the training-draw half of Known gap #5 open (the folds are fixed by construction).
`submit_rho_sweep.sh` runs one **(problem, seed) pair** per array task since
2026-08-22 — `PROBLEMS="gastric reactor"` x `SEEDS="42 7 13"`, `--array=0-5%2`,
seed varying fastest, so a truncated array still yields whole seed-sweeps of the
first problem. Ablations run on the **first seed of each problem** (tasks 0 and
3), not on global task 0, or the reactor would never get one. `PROBLEM=` singular
still narrows to one problem and wins over `PROBLEMS`; **--array must be narrowed
with it** (surplus tasks exit 0 saying "nothing to do", rather than failing).
It **omits `robust_reg`** — see Known gaps #1; put it back with
`METHODS="nominal cp wrapper robust_reg"`.

**A seed is not a draw for every method.** `--seed` moves the METHODS'
randomness: cp/wrapper's bank, robust_reg's refits, cmicl's calibration split. It
does **not** reach `nominal` (`solve_nominal` takes no seed) and reaches `margin`
only through the fold split behind `scale(y_c)` — which on **gastric is temporal
and ignores the seed entirely**. Measured: gastric margin scales are
bit-identical across seeds 42/7/13 (dlt 0.249513, blood 0.239812); synthetic
moves ~0.6% (0.100790 / 0.100581 / 0.101148). So a pooled sd of 0 for nominal, or
for margin on gastric, is **structural, not evidence of stability** — read it as
"this method has nothing to resample", and do not report it beside a sampling
method's spread as if the two were the same measurement.

**No gastric result in the repo is current** (2026-08-21). Two changes hit every
gastric bank at once: the production draw is now **incoherent** (`coherent:
false`) and DLT is **derived** rather than drawn (`derive_linked_labels`). The
default sweep cell is therefore `_incoh`, not `_coh`, and the committed `_coh`
curves were produced with DLT on the shared direction — so they are neither the
new production cell *nor* a valid `--coherent` ablation of it. Re-run both.
Synthetic and the reactor declare no link and have one outcome, so coherence is
vacuous there: **they are unaffected by this pair of changes.**

**The 2026-08-22 incoherent separation path does NOT add to that.** CP's separation
now follows the bank (below), so a **coherent** bank still cuts one shared scenario
exactly as before — verified identical to the pre-change code on gastric, so every
committed `_coh` curve reproduces. What changed is the `_incoh` cell, of which
nothing was committed. The other three methods are untouched either way.

**Only `results/rho_sweep/` and `results/figures/fig_rho_*` are current** —
ellipsoid geometry and fixed temporal folds (2026-08-17/19), and on gastric only
up to the paragraph above. Everything else
(`results/gastric/` all <= 2026-08-13, `results/synthetic/`, `adversary_probe/`,
`cv/*_robustness_knobs.json`) is `box_l1` and/or random-KFold-scaled. **Never read
a `box_l1` number against an ellipsoid one**; re-run before citing.
One exception inside the current set: **gastric `robust_reg` rows predate
`clip_labels` (2026-08-21)**, so its gastric curve and rho\* are against the raw
ball while CP's and the wrapper's are against the clipped one — re-run the gastric
sweep before reading robust_reg against them. CP, the wrapper and everything
synthetic are unaffected.

## Architecture

### The five methods (`src/methods/`)

Shared MIP scaffolding lives in `nominal.py` (`build_decision_vars`,
`add_problem_constraints`, `embed_constraints`, objective builders); every method
returns a `SolutionResult`.

**Every runner builds its solvers through `experiments/method_builders.py`**
(`build_method(method, knob, ...)`, `cp_solver`) — the one place a solver's
argument list is written, and the one place the cross-cutting decisions live: the
single `mip_gap`, `cp_alpha` pinned at 0, the shared `uncertainty_set`. It replaced
four near-copies (`run_all`, `run_sweep._synth_build`,
`run_chemo_robust._build_solvers` / `_method_build_map`) that each had to be
patched separately — the one-MIP-gap fix landed in all four. The **problem**-specific
half stays with the problem: a settings resolver per problem
(`_resolve_run_settings` for gastric, `method_builders.synth_settings` for
synthetic/reactor) and the knob `ranges` / strength maps in `run_chemo_robust`.

- **`nominal.py`** — no robustness, plus the toolkit everything else imports.
- **`robust_regression.py`** — robustify the *fit*:
  `min_theta max_{delta in D_c} L(theta; X, y_c + delta)`. **Linear with unbounded
  labels** is exact: on the ball the inner max is closed form (`delta* = R*r/||r||`,
  giving `(||r||_2 + R)^2`), so one second-order cone `nu >= ||r||_2` turns the
  ElasticNet QP into an **SOCP** (`_label_robust_linear`); `R=0` recovers sklearn
  exactly. **Trees/ensembles** alternate worst-delta / retrain `robust_reg.K` (=5)
  times. **A linear outcome with `label_bounds` also takes the loop** — the closed
  form is exact on the raw ball only, and `delta*` is unrealizable once labels are
  bounded; on gastric that is **GI** (the coherent arm never used the SOCP anyway).
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
- **`margin.py`** — **feasibility-tuned nominal**, the cheap baseline the rest
  have to beat, and the second method that never faces D. Same nominal fit, same
  nominal MIP, RHS moved in:
  `sum_m w_m f_m(x) <= rhs - m_c`, `m_c = margin * sum_m |w_m| * scale(y_m)`.
  Deck 2026-08-19 next step 7, RHS-margin half only — the penalty-multiplier
  variant is deliberately **not** implemented.
  - **One dimensionless dial for the whole problem.** Each constraint is scaled
    by its **own** `scale(y_c)`, so gastric's five toxicities take five different
    absolute margins off one knob — five separate margins would be a
    five-parameter fit against one feasibility number, which is no longer a
    baseline. `margin=1` is one unexplained sd of headroom per constraint, so it
    reads on the **same axis as `rho` and `tau`**. Measured, gastric: `rhs = 0.6`
    -> 0.35 (dlt), 0.36 (blood), 0.31 (constitutional), 0.34 (infection),
    0.33 (gi) at `margin=1`.
  - **It is swept on `m`, not on rho** (`SWEEP_PARAM["margin"] = "margin"`), so
    its sweep curve is a genuine conservatism curve and `m*` is read off it by
    the same rule as `rho*`. The axis value is in the same unexplained-sd units
    as rho and tau — which is what puts them on one plot — but a point on the
    margin curve is a **tightening the optimizer pays for**, not an assumed
    radius. The `param_swept` column on every row is what keeps the two apart;
    never read an `m` as a rho. `--margin-grid` gives it its own values when the
    useful range of a direct RHS shift differs from that of a radius.
  - **The scale is literally D's scale** — `uncertainty.instance_label_scales`,
    the same function `ScenarioBank` calls (verified bit-identical: reactor
    `2.1902848847140364` both ways). Two independent copies could drift on a
    fold scheme or a stat with nothing to flag it; sharing the estimator is what
    lets rho, tau and m be quoted on one axis.
  - **`|w_m|` keeps the sign right.** The reactor states a *lower* bound
    (`weight = -1`, `rhs = -50`), and `|w|` tightens it to `F_C6H6 >= 52.19` at
    `margin=1` rather than loosening it. Same worst-case-direction weighting
    C-MICL uses for its half-width.
  - **`margin=0` IS nominal** — same fit, same MIP, same `x*`; verified
    bit-identical on synthetic and reactor (obj diff `0.0e0`, x diff `0.0e0`), so
    the baseline's own curve starts *at* the nominal point rather than near it.
  - **Monotone in `m`**, so an `m*` at any feasibility target always exists. That
    is the whole point of it as a "tuned" baseline — contrast gastric
    `robust_reg`, whose feasibility *falls* with rho. Measured on synthetic
    (3 folds, `rho` inert): feas 0.000 -> 0.667 -> 1.000 and obj -1.2628 ->
    -1.2269 -> -1.1910 at m = 0, 0.5, 1.0.
  - **The objective term is deliberately untouched**, and there is no
    `robustify_objective` flag: a margin on a learned objective model adds
    `margin * |a| * scale`, a **constant**, which shifts the reported objective
    without moving the argmin. It would corrupt the column the methods are
    compared on while changing no decision — vacuous, not merely off.
  - **It carries NO guarantee.** `m` is fitted to held-out feasibility and means
    only what that fit means — which is also true of a fitted rho, and is the
    honest comparison.
  - **Large margins go infeasible, not conservative.** Once `rhs - m_c` drops
    below anything the label range can produce the MIP has no solution;
    `_unreachable_note` says so up front (a diagnostic, not a check — a linear or
    MLP fit can predict outside the range) and it shows as a falling solved
    fraction, which `--min-solved` guards.
- **`cmicl.py`** — Ovalle et al.'s conformal MICL, and **the one method that does
  not face D at all**. Per constraint model: split the training rows, fit `h` on
  proper-train (this is what gets embedded), fit a **width model** `u` there on
  `|y - h(x)|`, take the split-conformal quantile `q = s_(k)`,
  `k = ceil((n_cal+1)(1-alpha))`, of `s_i = |y_i - h| / max(u, floor)` on the
  held-out calibration rows, then embed **both** models as
  `sum_m [w_m h_m(x) + |w_m| q_m u_m(x)] <= rhs`. Its dial is **`alpha`**, the
  miscoverage level. Consequences that decide how its numbers may be read:
  - **It is an EVALUATION method, not a rho one.** `rho` never reaches it, so a
    rho curve for it is flat by construction and every cell re-measures the same
    number — it is **not** in the sweep's default methods and has **no ablation**.
    Its `alpha` is not a dial to be chosen either: it is **pinned to
    `1 - feas_target`** (0.1 against the 0.9 target), because the conformal level
    and the feasibility target are the same quantity. That is the comparison it
    is here for — the shared-D methods **search** for the rho that delivers 0.9,
    C-MICL **asserts** 0.9 from the calibration set and is scored on whether it
    arrives. `--methods ... cmicl` still runs it as a flat reference line if you
    want one; the sweep prints a note saying it is not part of the protocol.
  - **The guarantee is marginal** over `P_XY`, while `x*` is an argmin sitting
    **on** the constraint — exactly where Known gap #8 says a marginal statement
    says least. Feasibility measured here is the empirical consequence, not a
    delivered guarantee.
  - **Exchangeability is assumed and is false on gastric** (temporal folds). The
    calibration split is random *within* a fold's training rows, so coverage is
    marginal over the training years, not the validation year.
  - **`h` is fit on `1 - cal_frac` of the rows** (0.75), fewer than every other
    method's model. Intrinsic to split conformal, which is why `cal_frac` is
    structural and not a second dial.
  - The interval is intersected with `label_bounds`, the conformal analogue of
    `clip_labels`. With one model per constraint — every instance here — that clip
    is either **vacuous** (`w*hi <= rhs`, constraint dropped) or **inert**, so it
    needs no binaries; on gastric `hi = 1.0` against `rhs = 0.6`, so it is inert.
  - `multiplicity: "bonferroni"` (alpha/C) makes the **joint** statement hold at
    `1 - alpha`. On gastric that level is usually finer than `n_cal = 80` can
    resolve — `ceil(81*(1-0.02)) = 80` is the edge — and there is then no finite
    `q`. That is **reported as an infeasible solve with the reason printed**,
    never clipped to the largest score.
  - **Measured, reactor, ODE-judged, 10 folds, n=1000, CV-selected `mlp` base
    (2026-08-22): it does NOT reproduce the paper's Figure 1, and the diagnostic
    says why.** At their own alpha=0.1: ODE feasibility **0.60** against their
    reported **>= 0.90**, objective 3061, versus nominal at feasibility **0.00**
    and objective 2948. The conformal machinery is not the problem — **marginal
    coverage on held-out rows is 0.899** (0.949 at alpha=0.05), i.e. exactly
    `1 - alpha`. What collapses is coverage **at the optimum**:
    `|F_ODE(x*) - h(x*)| <= q*u(x*)` holds on only **0.50** of folds (0.60 at
    alpha=0.05), and the mean slack `F_ODE(x*) - (h - q*u)` is **+0.27** — the
    optimizer drives the embedded bound right up to the truth and sits there, so
    a small coverage failure flips feasibility. Feasibility (0.60) exceeds
    coverage-at-x\* (0.50) because coverage is two-sided while the constraint is a
    lower bound only. **That 0.899 -> 0.50 drop is their Assumption 4.1**
    (conditional independence of feasibility and coverage) failing on this
    instance, and it is the same object as Known gap #8.
    **TESTED (2026-08-22), and the answer depends entirely on the cost
    distribution** — `experiments/probe_cmicl_cost_sampling.py`, a diagnostic
    that changes nothing in the evaluation (`cost_vector` stays ones everywhere a
    result is produced; the probe passes its own). They average over **100 sampled
    `c`** with the model held fixed; we average over **training draws at one `c`**.
    Reproducing their average, alpha=0.1, N=100, full-data 800/200 fit:

    | `c` scheme | C-MICL feas | coverage at x\* | mean slack | x\* mean pairwise |
    |---|---|---|---|---|
    | `unit` (`c_i ~ U(0,1)`, the literal reading) | **0.99** | 0.99 | +1.22 | 0.145 |
    | `scaled` (`c_i ~ U(0,1)/span_i`) | **0.11** | 0.11 | -1.05 | 0.330 |
    | nominal, either | 0.00 | | | |

    1. **Their >= 0.90 IS reproducible here** (0.99 under the literal reading), so
       our 0.60 is a different average, not a defect. The anchor agrees: at the
       full-data fit with `c` = ones, C-MICL is feasible (`F = 51.79`) while
       nominal is not (45.85) — so much of the 0.60 is **training-draw
       variability at 720-row fold fits**, not the argmin effect.
    2. **Feasibility EQUALS coverage-at-`x*` to two decimals under both schemes.**
       Feasibility here is not mostly about coverage at the optimum, it **is**
       coverage at the optimum. Mechanism, measured.
    3. **The rate is a property of the cost distribution.** Under `unit` the
       optimum barely moves and never leaves the well-fitted region (T pinned to a
       bound, sd 9e-16); under `scaled` it roams over `d_t` and `L` (sd 0.17 each)
       and coverage collapses. Assumption 4.1 holds in the first case by an
       accident of variable scaling, not because the method secured it.

    **Never quote a C-MICL reactor feasibility without naming the `c`
    distribution it came from.** The paper does not state theirs (App. D.1 gives
    the decision-variable ranges; Table 5 says only "operational or design cost
    coefficient"), so both readings above are ours. Feasibility on the fold-based
    number is quantized to 1/10, so +-0.1 there is one fold.
    **The wrapper undershoots its own published band on the same folds, which is
    what points at the instance rather than at either method.** Our wrapper is
    W-MICL, whose Figure 1 range the repo records as 0.45-0.85; here it reaches
    **0.30** (alpha=0.2, P=20, rho=1, objective 2997). So on this instance BOTH
    our C-MICL and our W-MICL land below the paper's values for them, while the
    conformal calibration itself is exact (0.899) — a common-cause pattern, and
    the fixed `c` is the obvious common cause. **The W-MICL number is confounded
    on its own terms** and is not a reproduction attempt: their W-MICL(P)
    bootstraps ensembles of P in {5,10,25,50}, ours embeds P=20 draws from the
    shared D at rho=1, which is a different ensemble.
    Timing, same folds: C-MICL **25 s** vs the wrapper **905 s**, a 36x gap in
    the direction of their Figure 3 (they claim two orders of magnitude).
  - **Measured, gastric, `rf`, seed 42, `all_constraints` (2026-08-22): the
    prediction holds — it is INFEASIBLE.** At alpha=0.1, `multiplicity: "none"`,
    mean half-widths are **1.33-1.73 sd(y)** per toxicity (dlt 1.51, blood 1.33,
    constitutional 1.39, infection 1.73, gi 1.46) against `method.tex`'s predicted
    ~1.65 sd; on the percentile scale that is **0.38-0.50 against an rhs of 0.6**,
    on five constraints at once, and the master solve returns `infeasible`.
    Bonferroni at the same alpha (`alpha_eff = 0.02`, and `ceil(81*0.98) = 80 <=
    n_cal = 80` is exactly the edge where a finite `q` still exists) widens those
    to **2.05-2.99 sd(y)** and is infeasible too. Note the **cost of the negative
    result**: proving the marginal case infeasible took **176 s** against
    nominal's 0.9 s, while the Bonferroni case fell out in 2.9 s. A sweep that
    includes gastric C-MICL should budget for that.

### `uncertainty.py` — the shared set D

**Three** of the uncertainty-aware methods face this one set and differ only in
what they do with it (cut lazily / chance-constrain / robustify the fit).
**`cmicl` and `margin` are not among them** — the first calibrates against
held-out residuals, the second just shifts the RHS by a fitted dial, and neither
reads any `uncertainty.*` key except `scale_stat`. `margin` is still **swept**,
on its own parameter rather than on rho; `cmicl` has no swept parameter at all.

`instance_label_scales` is the **one place `scale(y_c)` is estimated** — D's
radius, CP's tau basis and the margin's tightening are all multiples of its
output, so they cannot drift onto different fold schemes or stats.

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

- **D is intersected with the label range** (`clip_labels`, on in `config.yaml`
  since **2026-08-21**; the dataclass field still defaults `False`). Applies to
  every method that faces D: `ScenarioBank` has always clipped its draws
  (`_clip_to_bounds`), and robust_reg's training adversary now clips too — before
  this, CP and the wrapper faced `D n [0,1]` on the gastric toxicities while
  robust_reg faced the raw ball, so **shared D held only up to the bounds**. It
  binds hardest exactly there. **Re-measured 2026-08-25**
  (`experiments/measure_clip_fraction.py`, production cell: ellipsoid,
  incoherent, DLT derived, B=30, seed 42) on the four freely drawn toxicities:

  | rho | rows leaving `[0,1]` | reach = `\|clipped\|/\|raw\|` |
  |---|---|---|
  | 0.5 | 9.6-11.8% | 0.921-0.934 |
  | 0.75 | 14.5-17.7% | 0.878-0.899 |
  | 1.0 | 19.2-23.1% | 0.834-0.864 |

  **rho=0.5 is now read, not inferred** — it is a headline dial-sweep column. The
  coherent cell is within a percentage point at every rho (0.8-23.4% at rho=1),
  so coherence does not move this marginal statistic. **Derived DLT barely clips
  at all** (0.07-0.16% of rows, reach 1.000): the identity maps in-range
  components to an in-range DLT by construction, which is a property of the link
  rather than of D.
  **The figures previously recorded here — 45-49% at rho=1, and `||delta||`
  4.56 -> 2.56 — do not reproduce on current code** under either geometry or
  either coherence cell (`box_l1` at `eps_0=1` gives 12-16%). They predate both
  the `IterativeImputer` label correction and `derive_linked_labels`, and the
  4.56 was DLT's *free-draw* radius under an older `oof_sd`. Treat them as
  superseded; the cause has not been isolated further.
  Clipping only shrinks `|delta_i|`, so the shift stays in D. Bites
  only where `label_bounds` is set — gastric's five toxicities; **gastric OS and
  synthetic carry none and do not move**. **Gastric robust_reg numbers predating
  2026-08-21 are not comparable across this switch**; set `clip_labels: false` to
  reproduce them.
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
  of CP's B.
- **Draws are INCOHERENT in production** (`coherent: false`, since 2026-08-21).
  Coherence remains a *grouping*, not a global flag — under `coherent: true` the
  group shares one standardized direction scaled by each outcome's radius, and
  names in `coherent_exclude` (`["os_constraint"]`) draw independently — but that
  is now the **ablation** (`run_rho_sweep.py --coherent`), not the default cell.
  Vacuous on synthetic and the reactor either way (one outcome; bit-identical
  banks). Unknown `coherent_exclude` names are ignored, since one config drives
  both problems.

  **Why the flip.** Coherent asserts +1 in-group; the measured OOF residual
  correlation on gastric (n=145, forward-chaining) is **+0.28** across non-DLT
  toxicity pairs on percentile labels. Incoherent asserts 0. Neither fits, and the
  choice was previously defended by the one block that genuinely *is* near +1 —
  DLT against its four components. **That block is no longer a draw**
  (`derive_linked_labels`, below), so what a shared direction is left modelling is
  the +0.28 block alone, where 0 is the closer of the two available assertions.
  Justification and open objections:

  1. **RESOLVED (2026-08-15): OS does not belong on the shared direction.** OOF
     residual correlation is **+0.28** across non-DLT toxicity pairs, vs **+0.06**
     for OS against every toxicity. Matches the record-level-mismeasurement story,
     which never covered survival. Kept as `coherent_exclude` so flipping
     `coherent` back restores the measured grouping, not a global +1.
  2. **MOSTLY RESOLVED for CP (2026-08-22): the incoherent separation path.**
     The objection was that the incoherent set is the product `D_1 x ... x D_C`
     while separation pulled every constraint's model from one shared `b` and cut
     that whole `b`, so the adversary searched B joint points, not `B^C`. On an
     incoherent bank CP now ranks the draws **per constraint** and admits a model
     for **each** constraint per iteration, so its search is B points per
     constraint. What is **still asserted**: the cut set is the per-outcome worst,
     a point of the product set that need not be any single sampled draw, so a CP
     cut is no longer a relabeling of any trial — the mirror of the old objection,
     traded deliberately. And **the wrapper is unchanged**: its `z[c,p]` still
     gates on one draw per replicate, so this closes the gap on CP's side only.
  3. **RESOLVED (2026-08-21): DLT is derived, not drawn.** See
     `derive_linked_labels` below.

- **`derive_linked_labels` — DLT follows the identity instead of drawing** (on in
  `config.yaml` since 2026-08-21; the dataclass field still defaults `False`).
  `DLT_PROP = 1 - prod(1 - tox)` holds exactly (2e-16) over the four modeled
  toxicities, so DLT carries no degree of freedom. It used to take the shared `u`
  anyway, which spent five outcomes' radius on four d.o.f. and made every draw a
  relabeling of no trial. The instance now declares a `LabelLink`
  (`src/data/generate.py`) and `ScenarioBank._draw` hands DLT
  `derive(perturbed components) - baseline`: perturbed percentile labels back to
  raw proportions (`gastric_v11.percentile_inverse`), the identity, then
  re-percentile against DLT's own reference. Measured, gastric, B=8 at seed 42,
  production cell (incoherent, `clip_labels` on):

  | | identity error (max) | norm vs own `R_dlt` | corr(DLT, blood) |
  |---|---|---|---|
  | linked, rho=0.25 | 6.25e-3 | 1.19x | +0.43 |
  | linked, rho=1.0 | 6.25e-3 | 0.89x | +0.46 |
  | free draw, rho=0.25 | 0.430 | 0.96x | +0.02 |
  | free draw, rho=1.0 | 0.994 | 0.85x | +0.04 |

  The 6.25e-3 floor is **not** the link's error: it is 2/320 of a percentile rank,
  because recomputing the identity in float differs from the stored `DLT_PROP` by
  ~2e-16 and that flips ties inside `percentileofscore` on 44 of 320 rows. Hence
  `baseline` (the link at the *unperturbed* sources) rather than `y_train` — a zero
  component shift then derives a zero DLT shift to **0.0e0**. Notes:
  - **The target's radius is reported, not imposed** (`_log_links`). `R_dlt`
    described a *free* outcome; under the link DLT's shift is an output of the four
    component radii, so it may exceed it (1.19x at rho=0.25).
  - **Only DLT moves.** The discarded free draw still consumes its `rng` call (and,
    under `--coherent`, still creates the shared direction), so at the same seed
    the other four outcomes' shifts are bit-identical to a bank built with the link
    off.
  - **Bank only** — CP and the wrapper. robust_reg's per-outcome training adversary
    is unlinked: its inner max would have to become joint across outcomes, with no
    closed form.
  - **Dropped under `dlt_only`**, where the components are not modeled;
    `filter_constraints` removes the link and DLT draws freely again.
  - `LabelLink.derive` must be **row-wise**, so `cv_calibrate._fold_instance` can
    carry a fold by subsetting `baseline` rather than rebuilding it.
- **`rho`/`eps_0`/`budget_frac` are shared constants, not knobs.** Each method
  keeps exactly one dial: CP `tau`, wrapper `alpha`, robust_reg `label_eps`. Never
  pin D to robust_reg's calibrated optimum, to the GT ensemble (tunes to the
  judge), or to synthetic's known `noise_std` (CP then wins by construction).
  The sweep's fixed dials are **tau = 0.01** (`methods.cp.dist_tol_rel`)
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
- **coherent / incoherent** (gastric: many constraints / many `x*` / learned
  objective) — every draw is scored **per unit** (one unit per learned constraint,
  plus the epigraph objective when robustified): its normalized exceedance averaged
  over the anchors. **Which of the two runs is read off the bank's own geometry**
  (`methods.cp.separation: "auto"`), not chosen per run — the adversary should
  match the set it is drawn from, and `--coherent` then flips the draws *and* the
  adversary as one ablation.
  - **coherent bank** (`uncertainty.coherent: true`): the draws lie on a shared
    direction, so a draw *is* one relabeling of the whole trial. One scenario is
    cut per iteration, ranked by the **mean** of its unit distances (equivalently,
    mean relative exceedance over (anchor x outcome) cells), and
    `cut_whole_scenario` cuts all of it. **Unchanged from before 2026-08-22** —
    verified identical to the pre-change code on gastric.
  - **incoherent bank** (`uncertainty.coherent: false`, the production cell): D is
    the product set, so **the draws are considered per constraint**. Each unit ranks
    the whole eligible bank on its own outcome and one model is admitted for
    **each** unit above tau — 5 gastric constraints give up to 5 cuts from up to 5
    different draws. tau is met **per unit**, and CP stops only when *every* one is
    within it.

  `separation: "coherent"|"incoherent"` forces the path against the bank. Legal,
  and the log says `[FORCED: bank is ...]`, but there is no result it produces that
  a matched pair does not produce more honestly.

  **tau does not transfer between the two.** The coherent path averages over the
  units, the incoherent one does not, so incoherent distances run about C x larger
  and the same tau separates far harder. Measured on gastric (B=30, 6 anchors,
  seed 42): max iteration-0 distance **0.0623 coherent vs 0.2441 incoherent**.
  Re-read the grid off a run before reusing it across paths.

Other CP settings:

- **Fixed bank** (`scenario_source: "noise"`). The legacy `"bootstrap"` path redrew
  each iteration while `d0` stayed frozen at iteration 0, comparing different
  samples; kept as an ablation. `uncertainty.cp_k_neighbors_*`, `cp_n_candidates`
  and `cp.distance` apply to that legacy path **only**, and since 2026-08-21 live
  in the code defaults rather than `config.yaml` — same values, so the ablation
  still reproduces.
- **`cut_whole_scenario: true`** cuts all of an accepted scenario's constraints —
  what makes permanent exclusion sound and matches the wrapper's per-replicate
  indicator. **Read on the coherent path only**; an incoherent cut is one
  (constraint, draw) pair by construction, so there is no scenario left to complete.
- **`cut_rollback`** (incoherent path only) decides what happens when a
  constraint's model breaks a protected anchor. Either way **only that attempt's
  embedded models are removed** — `remove_scenario` drops exactly the vars and
  constraints `add_scenario` created for it, so the constraint's earlier cuts, the
  nominal base and every other unit's cut stay put (audited: the master's
  var/constr counts return exactly).
  - **`"forward"`** (default): walk the constraints most-violating first; for each,
    try its most-violating model and re-solve, and if a protected anchor breaks,
    roll **that model** back, permanently reject that `(unit, draw)`, and try the
    constraint's **next** most-violating model — until one is admitted or its
    candidates run out. Then move to the next constraint. So an iteration ends with
    a model added for **each** constraint that has an admissible one, rather than
    dropping a constraint on its first failure.
  - **`"peel"`**: stage every constraint's top model, test once, then drop from the
    **least-violating** end until they fit. One anchor sweep when the whole set
    fits, but **no fallback** to a constraint's next candidate, and the peeled model
    is rejected on a **heuristic** attribution (it is never shown to be the
    culprit).
  Measured, gastric B=30 / 6 anchors / tau=0.001: forward added **5/5** constraint
  models at iteration 0 (blood's draw 2 broke an anchor, its draw 0 got in) and ran
  5 iterations; peel added **3/5** there (it dropped infection and blood outright)
  and stopped at iteration 2. **Forward is the more expensive lever**: exhausting a
  constraint's ranking cost 18 rollbacks in one iteration and 184s against peel's
  24s. The cost is bounded across the *run*, not the iteration — rejections are
  permanent, so at most `C x B` rollbacks are ever paid.
  **`cut_eviction: "evict_slack"` is ignored on the incoherent path** (a warning is
  printed); models are rolled back per constraint instead.
- **Rejections are keyed `(unit, draw)` on the incoherent path**, not by draw, so
  the same relabeling can be the legitimate worst case for a constraint it has not
  been cut for. A draw is skipped outright only when every unit excludes it — which
  is why `scanned=` no longer shrinks each iteration there.
- **Both paths are vacuous with one unit** (synthetic, reactor: one constraint, no
  robustified objective) — the per-constraint ranking *is* the shared ranking, and
  those problems take the **basic** path anyway. `separation` reaches gastric only.
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

1. **`optimization.mip_gap` = `1e-4`** (was hard-coded 0.01). At 1% on a gastric
   objective of ~10 the solver returns anything within ~0.1 while the distances
   separated are ~0.007, so cuts below the solver's own tolerance left `x*`
   unmoved. **Synthetic never hit this** (objective ~1.2, distances ~0.1) — same
   code, opposite regimes. Since 2026-08-20 this is **one gap for every method**
   (see Conventions).
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
over the anchors. On the **incoherent** path that per-outcome mean **is** the
statistic tau is tested against, one test per constraint; on the **coherent** path
the per-outcome means are averaged again over the outcomes, which is the old "mean
over (anchor x outcome) cells". So tau is a physical quantity in the **same units
as `rho`**, independent of seed/bank/B, but **its numeric scale differs by ~C
between the two paths** — see `solve_cp` above. **The grid spans nominal** — a tau
above
the iteration-0 distance stops before any cut (verified on synthetic: tau=1.0
returns the nominal objective in 1 iteration). The basic path keeps violations raw
for logging and multiplies instead of divides; the two paths log different units
but tau means the same thing.

**One tolerance rule, `_resolve_tolerance` (2026-08-25):**
`tolerance = tau * conv`, floored at `mip_gap * the SAME conv`. **Multiplying tau
is the primitive**; `conv` converts it into whatever units that path compares in:

| path | `conv` | because |
|---|---|---|
| basic (synthetic, reactor) | `s_c` | violations are kept **raw** |
| contextual (gastric) | `1.0` | exceedances were **already** `/ s_c` |

The surviving division is **structural, in exactly one place**: the coherent path
averages a draw across outcomes whose `s_c` differ, so there is no single factor
to multiply tau by and the normalization must happen per cell, *before* that mean.
The incoherent path's mean is within one outcome, so its division **is** a
`tau * s_c` written the other way round — left as a division only because flipping
it would restate every logged distance without moving a decision.

**Both sides of the `max` now use the same `conv`, so the floor is 1e-4 in tau
units on every problem** and `tau < mip_gap` is the floored region everywhere.
Before this the basic path converted tau with `s_c` and its floor with
`tol_scale = max(1, |rhs|)` — the **legacy d0 basis's** normalizer, which the
coherent path only uses under `tolerance_basis: "d0"`. Measured floors in tau
units before -> after: gastric 1e-4 -> 1e-4, synthetic 9.7e-4 -> 1e-4, reactor
**2.3e-3 -> 1e-4** (its `rhs` of -50 gave `tol_scale = 50` against `s_c = 2.19`,
so a factor of 23 came from the constraint's right-hand side rather than from
anything about solver resolution). **It floored a committed cell**: the tau=0.001
row of `reactor_ablations_incoh_f10_mmlp_s42.csv` actually ran at **tau=0.00228**
(objective 3052.118 vs 3052.081 at tau=0.01, feasibility 0 either way, so no
conclusion moved — but a mislabelled tau matters now that tau is the swept **axis**
of `run_dial_sweep.py`). Nothing at `tau >= 2.3e-3` (reactor) or `9.7e-4`
(synthetic) changes, and the contextual path is bit-identical; verified by
reproducing all six synthetic CP cells of the dial sweep exactly. The legacy `d0`
basis keeps `tol_scale` untouched, since reproducing prior runs is its only job.

- **One decade grid, wide range** (`[1.0, 0.1, 0.01, 0.001]`). Both paths max over
  draws, but a draw *scores* differently: basic has one cell (so, the raw
  violation); the multi-constraint paths mean over the anchors, i.e. (violating
  fraction) x (mean exceedance among violators), and the **coherent** one divides
  again by `n_outcomes`. Gastric's ~0.1 violating fraction is why its coherent
  maxima sit near 0.03 against synthetic's ~0.98. That is **range, not meaning** —
  breadth is information. Read `[cp] basis=scale ... max iter-0 dist=` off a real
  run before assuming the grid brackets a **new** problem **or the other separation
  path**: the same gastric bank reads 0.0623 coherent and 0.2441 incoherent.
- **NEVER pin the grid from measured distances.** tau is **fixed before the run**
  and is a parameter of the method, on the rho sweep and on the dial sweep alike:
  one grid, the same at every rho, on every problem, on both separation paths.
  A grid read back off the run makes tau a function of the bank, of `B` and of
  which folds were looked at, so the same nominal tau means a different tolerance
  in every cell and the axis stops being one quantity. This holds **whether or not
  rho is moving** — a fixed rho does not license it. `run_dial_sweep.py` did place
  the grid from each column's iteration-0 distance between 2026-08-25 and
  2026-08-26; that is removed and must not come back. On the rho sweep the order
  stays (1) rho sweep at fixed tau, (2) tau ablation at the chosen rho.
- **`CPHistory.iter0_tau`** is the iteration-0 separation distance expressed in
  tau's own units, set by whichever strategy ran, so it reads the same off the
  basic path (max raw violation / `scale(y_c)`) and the contextual ones (the
  distance itself under `basis="scale"`). It is a **diagnostic**: it says whether
  the fixed grid brackets the problem, which is a fact about the run to be
  reported. It **must not be fed back into the grid** — see the bullet above.
- **The mean is the right statistic and is anchor-count stable.** A scenario
  breaking 2 of 20 anchors badly is less dangerous than one breaking all 20
  moderately; a max ranks those backwards. This is the *anchor* mean and it stands
  on both paths; what the incoherent path drops is the second mean, the one **over
  outcomes** — a draw that breaks one constraint badly and four not at all is that
  constraint's adversary rather than being averaged down to a fifth. Measured max iteration-0 distance:
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
  **The blood `IterativeImputer` did not converge, and the labels were an
  arbitrary iterate** (fixed 2026-08-21). `Lympho34`/`Lympho4` carry **9 and 8
  observed values out of 355**; each round refit them on ~8 rows against 14
  predictors and wrote ~347 fabricated values back into the frame, which then fed
  every other column's regression. With the `min_value=0` clip on top the
  iteration **cycled**: it hit `max_iter` at 500, 1000, 2000 *and* 5000, the
  per-round change plateauing at 5e-3–6e-2 against a 1e-3 tolerance, and `BLOOD_4`
  moved **up to 0.32** between the 500- and 5000-round answers. A fabricated
  `Lympho4` was the max of the five grade-4 columns on **27/355** rows (surviving
  the `min` with `BLOOD_34` on 22), and with no `max_value` the imputer emitted
  proportions up to **1.808**. The fix: exclude columns observed on
  `< IMPUTE_MIN_OBS_FRAC` (0.05) of rows from the imputation *model* — their real
  observations are rejoined and `.max(axis=1)` skips the NaNs — and clip to
  `[0, 1]`. Converges in ~150 rounds, no warning, nothing out of range.
  **This relabelled the cohort**, so every gastric number from before it is
  against different labels: rows changed (of 495) `BLOOD_4` **264** (mean 0.0066,
  max 0.202), `DLT_PROP` **394** (max 0.236), the three non-blood toxicities
  55–221 (max 0.072); **394/495 rows moved on at least one outcome**. Raising
  `max_iter` does not fix it and `tol` only silences it. The other two imputers
  were always fine (tox 8 rounds, context 1).
  **What this deviates from is the v11 notebook, not Maragno D.1.** The D.1
  citation in this repo (`generate.py:189`) covers the **context** fill (`imp_x`),
  which converges in one round; `_build_outcomes` is documented as "v11
  definitions" and imputes *outcome* columns that are then max-reduced into a
  label — label construction, not covariate fill. OptiCL's source is not vendored
  here, so whether their published blood labels share the arbitrary-iterate
  property is **open**. Note also that `IterativeImputer` at its default
  `sample_posterior=False` is a single deterministic MICE pass, not the multiple
  imputation D.1 describes.
- **`gastric_model_specs.py`** — Table EC.10 embedded models and the EC.12 6-model
  GT ensemble; training seed 1 to match `run_MLmodels.py`.

**Frozen gastric picks** (`results/cv/gastric_selected_configs.json`, by `run_cv.py`
on R^2, *before* any robustness): **XGB** for DLT/blood/constitutional/infection,
**linear (ElasticNet)** for GI and OS.

**Gastric CV reproduces bit-identically**, like synthetic and reactor — verified
2026-08-21 by running `run_cv.py --problem gastric` twice on the current code and
diffing `gastric_cv_scores.csv` and `gastric_selected_configs.json` at full
precision. `run_cv.py --problem gastric` still **overwrites** those files, so
`git checkout` them after an inspection run unless you mean to re-freeze. If a
re-run does not reproduce the committed JSON, the cause is a **code or label**
change since it was written, not run-to-run noise: the picks above were re-frozen
on 2026-08-21 against the corrected `IterativeImputer` labels, which moved
dlt `max_depth` 4 -> 3 and blood `n_estimators` 20 -> 10 (winners unchanged;
`os_constraint` scores moved by ~1e-9, confirming the relabel touched only the
toxicities). Blood is the outcome the fix helped most: `xgb` CV R^2 0.175 -> 0.204.

**Synthetic now goes through `run_cv.py` too** (2026-08-21), on the same grids and
the same 5-fold R^2 criterion: `results/cv/synthetic_selected_configs.json`, loaded
by `run_sweep.synth_model_spec` and pushed onto the instance as
`constraint_model_configs`, so one assignment reaches every method and every
`ScenarioBank` refit. Current pick: **`mlp`, one hidden layer of 50, lbfgs,
alpha 0.01** (CV R^2 0.9639, test R^2 0.9990 on an independent 1000-row draw). It
beats `gbm` 0.9603 and the old hard-coded `rf` 0.9247. Absent that JSON, every
caller still falls back to `config.yaml`'s `default_model` block (renamed from
`model` on 2026-08-21), which reproduces pre-CV runs.

**The dropped-hyperparameter bug never touched the selection.** `train_model`
ignored `solver`/`alpha` for `mlp`, `epsilon` for `svm` and `subsample` for `gbm`
until 2026-08-21, but `train_best_model_cv` compares models through
`GridSearchCV`, which sets those on the pipeline directly, and `test_r2` comes off
the same `best_estimator_` — so both CV stages were always fitting what they
reported. Re-running `run_cv.py --problem synthetic --ensemble` and `--problem
reactor --ensemble` after the fix reproduces **every** `results/cv/` artifact
bit-identically (winners, CV R^2, test R^2, GT ensemble configs). What the bug
broke was everything *downstream* of the JSON — the embedded nominal fit and every
`ScenarioBank` refit went through `train_model` and got adam/`alpha=1e-4` instead.
So the JSONs are sound and the **results** computed against them are not: any
synthetic or reactor number produced before the fix embedded a different model
than the one named in `*_selected_configs.json`. The same applies to the proxy CV
oracles, whose members are fitted through `train_model` (`svm` `epsilon`, `gbm`
`subsample`) even though their config JSONs are unchanged.

Two things had to be true first, and neither was at `n_train = 200`:

- **`n_train` is 2500, up from 200.** At 200 the CV winner (`mlp` (10,5,2)) had a
  refit-to-refit prediction spread of **0.62** on the training rows against a
  per-row label shift of 0.128 at `rho=1` — **4.9x amplification**, vs ~2x for the
  tree ensembles — so 4 of 12 refits disagreed with the nominal fit about the box
  corner and **the wrapper prescribed x=(1,1)**, where `f_true = 2.5` against
  `rhs = 1.0`. At 2500 that spread is **0.092**, the second lowest of the seven
  candidates, and the nominal `x*` moves off the boundary to the interior. The
  pathology was small-sample, not model choice, so **no selection screen is
  needed** — plain CV R^2 picks the model. Verified downstream: the wrapper returns
  `feas=1.000`, `obj=-1.2504` at `rho=0.2`, and bank training fell 43.1s -> 1.7s on
  12.5x the data.
- **The embedding is exact for MLP**: max `|embedded - sklearn|` is `6.7e-16` over
  training rows and ReLU kinks (2026-08-21, `verify_embedding.py --problem
  synthetic`, against the lbfgs/alpha=0.01 net that is now actually fitted; the
  earlier `7.4e-14` was the adam net the dropped-hyperparameter bug produced). The
  corner solution was never an encoding error.

`rho` keeps its meaning across the `n` change **by construction** — `R_c = rho *
scale(y_c) * sqrt(n)` fixes the per-row shift at `rho * oof_sd` for any `n`, which
is what the `sqrt(n)` convention is for. The radius grows (1.81 -> 5.63 at
`rho=1`); that is the same quantity, not a wider set. `oof_sd` fell 0.128 -> 0.113
against a true sigma of 0.100, so the instance is better determined too.
**Nothing in `results/synthetic/` or the synthetic `results/rho_sweep/` cells from
before 2026-08-21 carries over.**

The *marketing* (in-LP context) setting in the README is **not implemented**.

### The reactor instance (`src/data/dma_mr.py`, `reactor_model_specs.py`)

**The C-MICL regression case study** (Ovalle et al. 2025 Sec. 5.1 / App. D.1; the
reactor of Carrasco & Lima 2017), added **2026-08-21**. Five design variables
`(v0, v_He, T, dt, L)`, minimize a linear operating cost, subject to a learned
constraint that outlet benzene flow reach 50.

**It is here because its oracle is MECHANISTIC.** Every other judge in this repo is
fitted, so it carries an error of its own exactly where a constrained optimum
lives — on the boundary. Here ground-truth feasibility is *integrated*, so that
failure mode is absent, and it is published ground: C-MICL's Figure 1 reports
W-MICL — **our wrapper** — reaching only 0.45–0.85 ground-truth feasibility here,
never the 0.9 target.

- **Two judges, and they must not be confused.** `make_gt_oracle` returns the
  **ODE** (`ReactorODEOracle`) — final evaluation and audits only. `make_cv_oracle`
  returns the **proxy** six-class ensemble for **rho tuning**, because tuning rho
  against the exact truth is the same error as pinning D to synthetic's known
  `noise_std`. `make_gt_oracle` returns `None` on every other problem; callers must
  handle that rather than fall back to the proxy.
- **Sign convention.** The requirement is a *lower* bound, so the single model
  carries `weight = -1` against `rhs = -50`, and `gt_constraints` returns
  `-F_C6H6`. `SyntheticOracle` now takes that `weight`; hard-coding `+1` would
  silently invert the feasibility column.
- **`cost_vector` is FIXED at ones.** C-MICL redraw it per instance, but a new `c`
  is a *different problem*, not a new sample of this one. Note the units are not
  commensurate (`T ~ 1e3` vs `dt ~ 1`), so with ones the objective is effectively
  about `v0`, `v_He` and `T` — a property of the stated formulation, not a bug.
- **`noise_std = 2.0`**, calibrated to their **model fit** and nothing else: the
  paper says only that noise "was added". 2.0 reproduces their ReLU-NN (our 5-fold
  CV R^2 0.958 vs their 0.954, read off Table 2 as `1 - MSE/1.2871`). Their stated
  "variance 1.2871" cannot be this quantity — their own Table 1 spans 28–43 — it is
  a standardized target, which is why R^2 is the check.
- **The design is uniform over the box; the optimizer is not.** Only **216/1000
  (21.6%)** of the sampled rows satisfy the D.1b–e ratio constraints, and only
  **30 (3.0%)** satisfy them *and* reach `F_C6H6 >= 50`. Binding: `v0/L >= 20`
  (43.4%), `v0/v_He >= 0.75` (69.8%), `v0 <= 1.1 T` (81.0%). **Kept, not corrected**
  — uniform sampling is what the paper specifies. It is also why the instance is
  worth having: boundary uncertainty here is genuine, and it is the concrete form
  of the caveat C-MICL raise in their own Discussion.
- **Validation.** RF 0.855 vs their 0.864 and GBM 0.912 vs 0.907 on 5-fold CV R^2;
  CART 0.797 is below their LMDT 0.951 because a linear-model tree is a richer
  class, not because of a discrepancy. Table 1 agrees to a consistent **-1.6%**
  (not scatter; not the negative-flow guard — probably their fixed 2000-point
  integration grid). **Do not quote our `F_C6H6` values as reproductions of
  theirs.**
- **Cost.** Each oracle call is a stiff ODE solve at `rtol=atol=1e-10`, ~160 ms, so
  the design is **cached** under `results/reactor/` keyed by `(n, seed)`.
- `opyrability` is **not** a dependency. The model lives in that project's
  `tests/dma_mr.py`, is not exposed by the distribution, and each of its entry
  points varies only a subset of the five inputs — so the RHS is vendored with
  attribution and the package was removed again after being tried.

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

**MLP is now load-bearing on two of the three instances**, so `verify_embedding.py`
covers it too (2026-08-21). Its boundary case is per model family: a split
threshold for a tree, and for an MLP a hidden unit at **zero pre-activation**,
where the big-M ReLU binary is free — found by bisecting between two training rows
on a pre-activation sign change, so it reaches kinks at **any** layer, not just the
first. The tie itself is benign (both branches give `h = 0`); what it measures is
the `M * IntFeasTol` slack around it. Measured, at the CV-selected nets:

| problem | model | max err | max spread |
|---|---|---|---|
| synthetic | `mlp` (50,) | 6.7e-16 | 8.9e-16 |
| reactor | `mlp` (10,5,2) | 4.6e-12 | 2.6e-12 |

Reactor is four orders looser than synthetic purely because of scale (`F ~ 50`,
`T ~ 1e3`), and both are far inside the `1e-6` bar.

`--problem synthetic` / `--problem reactor` build the instance through
`run_sweep._synth_instance` / `_reactor_instance`, so the model verified is the
CV-selected one every method embeds. **Do not route them through
`run_adversary_probe.build_instance`**: it never loads the CV selection, so it
embeds `default_model` (`rf`), not the `mlp` every method embeds — before
2026-08-21 `verify_embedding.py --problem synthetic` was checking that `rf`, on a
200-row instance besides. The row count is fixed (the `data:` -> `synthetic:`
rename made it read the real block, so 2500); the CV-selection half stands.

### Evaluation (`src/evaluation/`)

- **`chemo_metrics.py`** — Table 6 metrics. All reported outcomes use the **GT
  ensemble** (6 sklearn models/outcome), *not* the embedded MIP models.
- **`metrics.py`** — synthetic feasibility/violation metrics. This is where the
  analytic `f_true` is allowed; the CV oracle below is not.

**Protocol.** The GT ensemble is fit on all **416** gastric arms (train + test), a
*superset* of the constraint fit rows — which is why one full-data draw would
favour nominal and why the headline is **m-out-of-n subsampling**:
`--n-realizations 10 --subsample-frac 0.5`, common random numbers across methods,
`all_constraints` only. Every number is a mean over the **samestore cohort** (test
arms *every* method could prescribe for, recomputed per draw), so all of it is
**conditional on solvability**, with the solved fraction reported beside it.

### Calibration (`src/methods/calibrate.py`, `cv_calibrate.py`)

**Two axes, and since 2026-08-25 the DIAL one is primary.**

**Primary — the dial sweep** (`run_dial_sweep.py`). rho is **fixed** at one of two
columns per problem and each method walks its own dial. The reported object is the
**curve in objective x held-out feasibility**, and the derived one is each
series' **protocol point**: among cells meeting the target on enough solved
contexts, the one with the **best objective** in the problem's own sense
(`{problem}_dial_star{cell}.csv`, `dial_star` / `bound`). That rule needs no
monotonicity — unlike rho, a dial has no direction that is automatically "more"
(alpha and tau run the opposite way round from `m`) — and it is the decision a
user would actually take from the series. `bound` is `interior`, `grid_end` (the
dial grid may be the limit rather than the method) or `none`.
**A `none` row is not a row of NaNs** (since 2026-08-26): every row also carries
`best_feasibility` / `best_feas_dial` / `best_feas_objective` /
`best_feas_solved_frac` — how close the series got at `solved_frac >= min_solved`,
whether or not it delivered. "0.88 at every dial" and "0.00 at every dial" are
different results and the table could not previously tell them apart; on the
2026-08-26 reactor run, where nothing reached 0.9, this is the difference between
reading CP at rho=2 (**0.80**) against C-MICL (0.30) and margin (0.00), and
reading seven blanks. Per-cell feasibility itself was always saved — it is in
`{problem}_dial_curve{cell}.csv` and `{problem}_dial_scores{cell}.csv`; only the
*derived* star table dropped it.

**Supporting — the rho sweep** (`run_rho_sweep.py`). Sweeps the shared rho with
tau/alpha fixed and reports **rho\*(method)** — the largest rho meeting the
target. D is shared at **every point of the sweep**, and that curve is where the
shared-D comparison is read. rho is never *fitted*; tau and alpha become ablations
at one rho (`--ablate`).

**Neither table is the comparison.** `rho*` is a capacity, `m*` a price, and a
`dial*` is a point on a frontier; the thing they are all read off — objective at
equal feasibility — is the curve, which is why the curve is the primary output in
both files.

**The sweep is per-method parameter, so the star table is too.** Each method
walks its own dial (`SWEEP_PARAM`) and the same rule below is applied to it, with
a `param_swept` column saying which quantity the star is in:

- `rho` for **cp / wrapper / robust_reg**. Within one grid value these three
  still face the **same D**, which is what the shared-D comparison rests on.
  `rho*` is **capacity**: the largest assumed uncertainty the method still
  delivers under.
- `margin` for **margin**. `m*` is **price**: the largest RHS shift that still
  solves on enough contexts. It is the number the shared-D `rho*` results have to
  be defended against — the same feasibility bought by a one-line RHS shift.
- **none** for **nominal / cmicl**, which have no conservatism parameter. They
  are scored **once** and repeated across the axis as a reference level;
  `rho_star` is `NaN` with `bound="no_param"`. Reporting `grid_max` for them
  would assert they absorbed the whole grid, the opposite of the truth.

**`rho*` and `m*` are not comparable to each other as numbers** — one is an
assumption, the other a tightening. The comparison that matters is **objective at
equal feasibility**, i.e. the curve, which is why the curve is the primary output
and the star table the derived one.

Rule: **largest grid rho with held-out feasibility >= 0.9 AND solved fraction
>= 0.5** (both defaults). The solved floor is the artifact guard — high feasibility
over few survivors is not a win, and it is what binds the wrapper. Measured
(2026-08-17, coherent, `results/rho_sweep/*_rho_star_coh.csv`):

| problem | method | rho\* | feas | obj | solved | master s | `bound` |
|---|---|---|---|---|---|---|---|
| gastric | cp | 1.0 | 0.934 | 9.63 | 0.59 | 494 | `grid_max` (censored) |
| gastric | robust_reg | 0.75 | 0.933 | 10.55 | 0.91 | 1.0 | `feasibility` (pre-`clip_labels`) |
| gastric | wrapper | 0.5 | 0.940 | 10.57 | 0.60 | 24 | `solved_floor` |
| gastric | nominal | — | | | | | never reaches 0.9 |
| synthetic | cp | 1.0 | 1.00 | -1.207 | 1.00 | 123 | `grid_max` (censored) |
| synthetic | others | — | | | | | never reach 0.9 |

Read `bound` before quoting a rho\*: `grid_max` means the grid ran out. **CP is
censored on both problems.** The gastric `robust_reg` row is **pre-`clip_labels`**
(raw ball, so a wider adversary than it now faces) — stale until that sweep is
re-run; the other rows are unaffected.

- **Evaluation then runs each method at its own rho\***, so the match at evaluation
  is on held-out feasibility, **not** on D. **This is not wired up yet** — no runner
  reads `*_rho_star*.csv`; `run_chemo_robust.py` / `run_all.py` take D from
  `config.yaml` and do not force the ellipsoid. `method.tex` states the protocol;
  the code does not implement it.
- **rho\* is re-derivable without re-solving**: `{problem}_rho_curve{cell}.csv`
  carries every column, and `--rho-star-only` recomputes under a new
  `--feas-target` / `--min-solved` / `--exclude-capped` into `--out-suffix`. The
  criteria are written back as columns.
- **The synthetic CV oracle is a mixed-type ensemble** (2026-08-21). It was a
  single model of *the same class the candidate embeds* — an `rf` judging an `rf` —
  so oracle and candidate shared their approximation error. It is now **seven**
  classes (`linear, svm, cart, rf, gbm, xgb, mlp`), tuned
  by `run_cv.py --problem synthetic --ensemble` into
  `synthetic_gt_ensemble_configs.json` with fallback specs in
  `src/data/synthetic_model_specs.py`. Two
  properties are kept deliberately: it is fit on the **noisy** `y_train` (the
  analytic `gt_constraints` stay final-eval only, or D would be calibrated against
  the very truth it is judged by), and on the **full** training rows, a superset of
  any fold's — exactly as gastric's oracle is. So synthetic held-out feasibility is
  an **m-out-of-n** statement, not a genuinely held-out one; that is the protocol
  on both problems, not a synthetic defect.
- **MLP is an ensemble member everywhere except gastric** (2026-08-21).
  `GT_CV_PARAM_GRIDS` carries an `mlp` entry; `GASTRIC_GT_CV_PARAM_GRIDS` is that
  dict minus `mlp` and is the only grid the gastric runner uses, because gastric's
  ensemble is a **replication of Table EC.12** (six types) and its tuned JSON is
  the *final evaluation* oracle there, not a tuning proxy. Elsewhere MLP is in for
  the opposite reason: synthetic and reactor both **embed** an MLP after CV, and a
  judge with no MLP cannot follow the candidate into the region where the candidate
  is wrong — which near a constrained optimum is the boundary, where the verdict is
  decided (Known gaps #8). This **gives up the class-disjointness** the 2026-08-21
  mixed-type switch had bought on those two problems; the trade is measured, R^2
  against **noiseless** truth, 6 members vs 7:

  | problem | judge error sd, 6 (no mlp) | 7 (+mlp) | R^2 6 -> 7 |
  |---|---|---|---|
  | synthetic (independent 1000-row draw) | 0.0531 | **0.0466** | 0.9897 -> 0.9921 |
  | reactor (vs the ODE, on the design rows) | 2.075 | **1.802** | 0.9663 -> 0.9746 |

  A shared class is diluted to 1/7 of the average; a systematic blind spot is not.
  **Caveat, reactor**: the tuned `mlp` member came out `(10,5,2)`/`alpha=0.01` —
  *identical* to the embedded candidate, same rows and same `random_state`, so that
  member is the candidate and contributes no independent error. The judge's
  independence there rests on the other six. Synthetic drew `(100,)`/`alpha=0.1`
  against the candidate's `(50,)`/`alpha=0.01`, so its member is genuinely distinct.
- **Ensemble CV members are tuned inside their deployment pipeline** (fixed
  2026-08-21). `run_cv_for_ensemble` tuned bare estimators while
  `train_fixed_ensemble` -> `train_model(normalize=True)` deploys every member
  inside `Pipeline(StandardScaler, est)` — so a scale-dependent penalty
  (ElasticNet `alpha`/`l1_ratio`, SVR `C`/`epsilon`, MLP `alpha`) was selected for
  a different fit than the one that ran. It is now wrapped exactly as
  `train_best_model_cv` wraps the embedded candidates. Measured on the reactor
  `mlp`: **CV R^2 0.849 unscaled vs 0.962 scaled**, a different architecture and a
  100x different `alpha`, with lbfgs hitting `max_iter`. **Every
  `*_gt_ensemble_configs.json` changed** — trees are scale-invariant and did not
  move, `linear` and `svm` did, on all three problems, **gastric included** (its
  member list is still the paper six). So any result read against a tuned GT
  ensemble from before 2026-08-21 is stale, and on gastric that is the *evaluation*
  oracle `run_chemo_robust.py` loads by default. The `--no-cv-configs` path
  (paper `GT_ENSEMBLE_SPECS`) is untouched, as are all
  `*_selected_configs.json` — re-running model selection reproduces them
  bit-identically, since that path always scaled.
- **Synthetic folds are 5** (`cv_calibration.n_kfold`, was 4), matching
  `run_cv.py`'s model-selection CV so both CV stages read the same folds. One solve
  per fold means feasibility takes only `0, 0.2, ..., 1.0`, so **the 0.9 target is
  met only by 1.0**: synthetic `rho*` reads as "the largest rho at which every fold
  is feasible". Read the curve there.
- Every cell carries `status`, `n_capped`, and a wall clock split into the
  **master** phase (for CP, the whole cut loop) and the **test-point** phase — the
  comparison CP's MIP-size claim rests on. Capped cells are **kept and flagged**.
- Outputs are scoped by cell (`_coh`/`_incoh`, `_matchbank`). CP samples D with
  B=200 against the wrapper's P=20, so a rho\* gap between them is confounded with
  sampling density — `--match-bank` removes it.
- **CP is exempt from `uncertainty.alpha`, and `cp_alpha` is pinned at 0 for
  every result** — a cut breaking the protected anchor set is rolled back and its
  draw permanently rejected, so tau is the only robustness knob.
  `run_chemo_robust.py` still *passes* `settings["alpha"]` into `cp_alpha` and the
  body ignores it; that argument stays dead.
  **It is no longer dead in `cp.py`, though** (2026-08-25). It used to be:
  `_protected_still_feasible` returned on the first broken anchor whatever alpha
  said, so the only place `self.alpha` appeared was a print. It is now a real
  **count budget** — `max_broken = floor(alpha * n_anchors) - (anchors nominal
  already failed)`, floored at 0 — threaded into all three call sites, so the
  **coverage-cap ablation** (`run_dial_sweep.py --cp-alpha-ablate`) can hold tau at
  tau* and walk it. At `max_broken = 0` the loop still returns on the first break,
  so **every existing call is bit-identical** and no committed number moves.
  Permanence survives the change: the protected set and the budget are both fixed
  and the master only tightens, so the broken count is monotone in the cut set and
  a candidate that exceeds the budget once exceeds it always. What alpha > 0 buys
  is exactly what the set-wise test forbade — CP trading one patient's
  feasibility for another's — which is the property the ablation exists to price.
  **Structurally inert on synthetic and the reactor**: both take the basic
  separation path, which has no protected-anchor test; the runner says so and
  skips rather than emitting a flat curve that reads as a measurement.
- **Legacy paths.** `calibrate_strength` (strongest knob with training infeasible
  fraction <= `uncertainty.alpha`) runs only under `calibration.method: "alpha"`,
  a key that left `config.yaml` on 2026-08-21 and now needs adding back (live
  default `"cv"`);
  `cv_calibrate.py` knob CV is per (method, coherence) cell, keyed
  `method@coherent` / `method@incoherent`.

## Config

**Every block is named for its scope** (restructured 2026-08-21).
`problem.type` switches problem (the reactor is `--problem reactor`);
`synthetic.*` and `reactor.*` carry their own instance settings; `default_model`
is the shared embedded-model fallback, beaten by `reactor.model` and then by
`results/cv/*_selected_configs.json`; `optimization.mip_gap` is the **one solver
gap** every method runs at; `uncertainty.*` defines the **shared D**, with `alpha`
the **legacy-calibration target only** and `clip_labels` / `derive_linked_labels`
shared in scope but in effect on **gastric alone** (`coherent` is likewise vacuous
off gastric); `cv_calibration.*` holds the knob-CV folds and grids;
`methods.{cp,wrapper,robust_reg,cmicl,margin}` one dial each (CP's `separation`
is **not** a dial — it defaults to `"auto"` and follows `uncertainty.coherent`;
`cut_rollback` is structural and is not swept, and neither are `cmicl`'s
`cal_frac` / `width_*` / `multiplicity`). **`methods.cmicl` and `methods.margin`
are the two blocks that read (almost) nothing from `uncertainty.*`** — the first
calibrates against held-out residuals, the second shifts the RHS by a fitted
dial, so `rho`, `coherent`, `clip_labels` and the rest do not reach either.
`methods.margin.scale_stat` is the one exception and defaults to `null`, meaning
`uncertainty.scale_stat`: the margin is quoted in the same unexplained-sd units
as `rho` and `tau`, so it must read the same estimator. Overriding it decouples
them — don't, without saying so. There is deliberately no
`methods.margin.robustify_objective`: a margin on a learned objective term is a
constant and would move the reported objective without moving `x*`;
`methods.chemo.*` is **gastric
only** (`methods_to_run` / `constraint_modes`, `quick` for `--quick`). CV model
selections come from `results/cv/*_selected_configs.json` and
`*_gt_ensemble_configs.json` via `--cv-configs`.

Renamed in that pass: `data.type` -> `problem.type`, `data.*` -> `synthetic.*`,
`model` -> `default_model`. Removed, each onto a code default carrying the same
value so nothing moved: `calibration`, `conservativeness_sweep`,
`methods.chemo_wrapper`, `methods.cp.distance`, the `uncertainty.cp_*`
localized-bootstrap knobs, and the never-read `optimization.constraint_rhs` /
`variable_bounds` (the real RHS is `0.5 * n_features` in `src/data/generate.py`).

**Which coherence cell is the stronger adversary is not settled, and the
production cell is now the weaker-as-implemented one.** Coherent is stronger *as
implemented* — finite B covers the diagonal better than the product set, and
mean-over-cells scoring has a heavier right tail when exceedances move together.
That is a property of (mean scoring x shared `b` x finite B), not of the sets:
incoherent's set strictly contains coherent's, so under max-over-outcomes scoring
the ordering would flip. The 2026-08-21 flip to `coherent: false` was made on
**what the correlations say**, not on adversary strength, so expect the incoherent
curves to sit above the coherent ones at the same rho; that is the sampling
property above, not a method result. Run `--coherent` alongside.

## Known gaps (2026-08-19 deck's next steps)

Stated limitations of the current numbers, not bugs to fix silently.

1. **The rho axis is unresolved** — CP censored at the grid max on both problems;
   extend past rho=1. Unexplained: CP's dip at rho=0.5 on gastric; synthetic
   robust_reg at rho >= 0.5 (objective better than nominal at feasibility 0).
2. **RESOLVED (2026-08-21). The synthetic embedded model is CV'd** and the sweep
   reads the result: `mlp` (50,), CV R^2 0.9639 / test 0.9990, via
   `synthetic_selected_configs.json` -> `constraint_model_configs`. Required raising
   `n_train` 200 -> 2500; at 200 the winner amplified a 0.128/row label shift into a
   0.62 refit spread and the wrapper prescribed the box corner. Still **asserted**:
   `n_features` is 2 and `noise_std` 0.1 by fiat, and `--seed`'s effect on `mlp`
   refits has not been re-measured at the new `n`.
3. **PARTLY RESOLVED (2026-08-21). Synthetic CV scoring** — the oracle is no
   longer the candidate's own class (six-class ensemble, and no MLP in it, so no
   class is shared), and `n_kfold` is 5 rather than 4. What **remains open** is the
   part a fold scheme cannot fix: on a single-decision problem the fold-val rows
   are unused, the oracle is fit on the full training rows, and feasibility is
   quantized to `1/n_folds` — so a synthetic feasibility is an **m-out-of-n**
   statement over 6 possible values. **Read the curve, not rho\***, there.
4. **RESOLVED (2026-08-21). DLT is derived through its identity**
   (`uncertainty.derive_linked_labels`, `LabelLink`), and the production draw is
   now **incoherent** — the +1 in-group assertion is gone along with the one
   block that justified it. Identity error falls from up to 0.994 of a percentile
   rank to a 6.25e-3 floor that is a `percentileofscore` tie, not the link. What
   is **still asserted**: incoherent imposes 0 where the non-DLT toxicity pairs
   measure +0.28, and the derived DLT shift is no longer confined to its own
   `R_dlt` (1.19x at rho=0.25) because that radius described a free outcome.
   robust_reg is unlinked. **Every gastric bank number changes** — re-run the
   sweep before reading anything against a pre-2026-08-21 gastric curve.
   **NEW (2026-08-22): the incoherent separation path.** On an incoherent bank CP
   ranks the draws per constraint and admits a model for each, which answers the
   product-set half of objection (2) on CP's side — separation searches B points
   per constraint rather than B joint points. Selected from the bank, so the
   coherent arm is untouched and every committed `_coh` curve reproduces. Three
   things are **asserted, not shown**: (a) the cut set is the per-outcome worst, a
   product-set point that need not be any sampled relabeling, so a CP cut is no
   longer a relabeling of a trial; (b) the wrapper is unchanged, so shared D no
   longer implies a shared *adversary shape* between them, on top of the direction
   gap already documented under `uncertainty.py`; (c) tau's numeric scale moved by
   ~C, so the `[1.0, 0.1, 0.01, 0.001]` grid is untested on this path — re-read
   `max iter-0 dist` before the tau ablation. First measurement (gastric, B=30, 6
   anchors, tau=0.001, seed 42): incoherent/forward ran to `coverage_cap` in 5
   iterations, adding 5/5, 4/4, 3/3, 2/3 constraint models. **The readier
   `coverage_cap` is expected** — the per-outcome worst is a strictly stronger
   adversary, so more models break a protected anchor — but whether that costs
   feasibility or buys it is **unmeasured**: no rho sweep has been run on this path.
5. **Nothing is confirmed on the test set under the ellipsoid**, so rho\* has no
   training-draw error bars.
6. **MOSTLY RESOLVED (2026-08-22). C-MICL is implemented** (`src/methods/cmicl.py`,
   `methods.cmicl` in `config.yaml`, `--methods ... cmicl` on every runner): the
   calibration split and the width model both exist, and **the predicted gastric
   infeasibility is confirmed** — 1.33-1.73 sd(y) half-widths on five constraints
   at once, master infeasible at alpha=0.1 either multiplicity. What is **still
   asserted, not fixed**: (a) our gastric folds are temporal while conformal needs
   exchangeability, so the split is random *within* a fold's training rows and
   coverage is marginal over the training years, not the validation year; (b) the
   guarantee is marginal over `P_XY` while `x*` is an argmin sitting on the
   constraint — and on the reactor that is now **measured, not asserted**:
   marginal coverage 0.899 against coverage-at-`x*` of 0.50 (see the `cmicl.py`
   bullet above). C-MICL is deliberately **not** on the rho axis and has no
   ablation: its alpha is pinned to `1 - feas_target`, so it enters at evaluation
   only.
   **The reactor gap is now diagnosed, not open** (2026-08-22,
   `experiments/probe_cmicl_cost_sampling.py`). We reach 0.60 at alpha=0.1 where
   they report >= 0.90, but reproducing *their* average — 100 sampled `c`, model
   fixed — gives **0.99** under `c_i ~ U(0,1)` and **0.11** under a
   scale-normalized `c`. So the implementation is fine and the number is a
   property of the cost distribution, which the paper does not state. See the
   `cmicl.py` bullet above for the table and the mechanism (feasibility equals
   coverage-at-`x*` to two decimals under both schemes). **What this leaves
   genuinely open** is a claim the paper's Figure 1 cannot settle either: the
   feasibility guarantee is not robust to the instance distribution, and on this
   instance a cost distribution that actually explores the design box takes it to
   0.11. That is worth a slide, and it is the strongest argument in the repo for
   why a method whose conservatism is indexed by an assumed set may be preferable
   to one indexed by a marginal coverage level.
7. **PARTLY RESOLVED (2026-08-21). A third instance exists** — the C-MICL DMA-MR
   reactor, with a mechanistic ODE oracle. WFP food basket (Maragno's own wrapper
   setting) remains unimplemented, and no rho sweep has been *run* on the reactor
   yet — only the instance, both judges and the wiring are in place.
8. **NEW (2026-08-21). A fitted judge cannot score a boundary optimum.** A
   constrained optimum sits ON the constraint by construction, which is exactly
   where a fitted oracle's own error decides the verdict. Measured on synthetic:
   the proxy judge's error in the decision band has sd 0.039 against margins of
   0.015–0.020, it flips **31%** of verdicts against the analytic truth inside that
   band, and it called **5 of 5** nominal decisions infeasible that `f_true` calls
   feasible. A 0.0017 change in `sum(x)` moved reported wrapper feasibility from
   0.000 to 1.000. **The bias is not neutral between methods**: robust methods
   leave slack, where the judge is reliable, while nominal sits on the boundary,
   where it is not — so it flatters the contribution. The reactor is the instance
   where this is measurable: there the proxy agrees with the ODE on 5/5 nominal
   decisions (margins of 4–7 against judge sd 1.6), while still flipping **25.9%**
   of verdicts inside its own `|F - 50| < 5` band. Synthetic feasibility numbers
   should be read as **judge-dominated near the boundary**; gastric is unmeasurable
   and deliberately left alone.
   **Adding the `mlp` member helps and does not fix it** (2026-08-21, re-measured
   over 200k uniform draws with `|f_true - b| < 0.05`): band error sd **0.0375 ->
   0.0328**, bias +0.0111 -> +0.0098, flipped verdicts **28.3% -> 26.4%**. The
   original 0.039 / 31% above were the six-member judge; the reactor's 25.9% is
   also pre-MLP. Tightening the band to 0.02 raises the flip rate to 39-40% for
   both judges — the failure is the band, not the member list.

9. **PARTLY RESOLVED (2026-08-22). The feasibility-tuned nominal baseline
   exists** (`src/methods/margin.py`, `methods.margin`, `--methods ... margin`) —
   deck 2026-08-19 next step 7. `f(x) <= b - m*scale(y_c)`, one dimensionless
   dial, monotone, `m=0` bit-identical to nominal. What is **NOT** done: (a) the
   **penalty-multiplier** variant (`c*alpha` on the outcome's regularization) is
   deliberately not implemented — the deck itself notes it only ever *shrinks*
   `alpha` (`alpha_eff = alpha/(1 + rho/||r*||)`), so it is a weaker lever than
   the margin and was dropped rather than deferred; (b) **no rho sweep has been
   run with it on any problem**, so there is no `m*` beside the published `rho*`
   yet, and the comparison the baseline exists to make — does a shared-D curve
   sit above a plain RHS shift at equal feasibility? — is **unmeasured**;
   (c) it is absent from `submit_rho_sweep.sh`'s default `METHODS`, so it must
   be asked for explicitly. **Since 2026-08-22 the sweep is per-method parameter**
   (`SWEEP_PARAM`): margin is swept on `m` rather than held flat while rho moves,
   so `m*` comes off the main curve and its `--ablate` pass was removed as
   redundant. That also stopped `nominal`/`cmicl` being re-solved once per grid
   point for an identical answer.
   **NEW (2026-08-25): the comparison now has a place to be read** —
   `run_dial_sweep.py` puts margin on the same objective-x-feasibility axes as CP
   and the wrapper, with `margin` in the default `METHODS` of
   `submit_dial_sweep.sh`, so (c) is closed there. **(b) is still open**: the
   machinery exists and no dial sweep has been *run* on any problem, so the
   question the baseline exists to ask remains unanswered.

10. **NEW (2026-08-25). Everything in the dial sweep is machinery, not results.**
    `run_dial_sweep.py`, `plot_dial_sweep.py`, `submit_dial_sweep.sh`, the shared
    bank, the per-context records, the tau probe and the coverage-cap budget are
    all in place and smoke-tested on synthetic; **no gastric or reactor dial sweep
    has been run.** Everything above about what the figure will show is a design
    statement. Specifically unmeasured: whether CP's curve sits above the margin's
    at equal feasibility on either instance; where gastric C-MICL first solves;
    whether the reactor needs rho=3; and whether relaxing CP's coverage cap lifts
    its feasibility ceiling.
    Two things are **asserted rather than shown** even once it is run:
    (a) the protocol point is "best objective among the cells that deliver", which
    is a *reporting* rule chosen here and not derived from anything;
    (b) the coverage cap above 0 lets CP drop one patient to serve another, and
    **which** patient is an artefact of the anchor ordering — the ablation prices
    that trade, it does not make it principled.
    **REMOVED (2026-08-26): tau is no longer placed from the iteration-0
    distance.** The probe was a mistake and is deleted, not merely defaulted
    differently — tau is a parameter of the method, fixed before the run, one grid
    for every rho column. See the dial-sweep section for what it cost.
    **The 2026-08-26 run is void on both counts.** The gastric task **crashed**
    at `cmicl` alpha=0.02 (`solve_for_context` dereferenced `result.x` on a
    C-MICL result that returns `status="infeasible"` with no MIP when
    `ceil((n_cal+1)(1-alpha)) > n_cal`) — fixed, it now scores as unsolvable like
    any other infeasible cell — and the reactor task ran on a probe-placed tau
    grid. Re-run both.

12. **NEW (2026-08-26). The test stage exists and has not been run.**
    `run_dial_test.py` and the `RUN_TEST` block of `submit_dial_sweep.sh` are in
    place and smoke-tested on the reactor (nominal: ODE-infeasible on 10/10 folds
    *and* on the full-data refit, objective 2947.7 / 2941.5). **No method has a
    `dial*` to test yet** — the reactor sweep reached 0.9 feasibility nowhere and
    the gastric sweep crashed — so every test-stage number is still unmeasured.
    Two things are **asserted**: (a) the `folds` phase re-solves the folds
    `dial*` was chosen on, so it is a truth-judged rate and **not** a held-out
    estimate, and the file says so rather than fixing it; (b) on gastric the judge
    is the same GT ensemble that tuned `dial*`, so only the **cohort** (the 96
    `X_test` arms) is held out — there is no ground truth there to appeal to.

11. **NEW (2026-08-25). The recorded gastric clip fractions did not reproduce.**
    CLAUDE.md carried 45-49% of shifted labels leaving `[0,1]` at rho=1 and a
    roughly halved realizable shift; re-measuring on current code gives **19-23%**
    and a reach of 0.83-0.86, in either coherence cell, and `box_l1` at `eps_0=1`
    gives 12-16% — so the old numbers do not come back under any switch tried.
    They predate the `IterativeImputer` label correction and `derive_linked_labels`
    and are treated as superseded. **The cause has not been isolated**, which is
    the open part: if some other pre-2026-08-21 measurement in this file was made
    the same way, it is suspect for the same reason.

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

- **One MIP gap for every method** (`optimization.mip_gap`, `1e-4`), read by
  `resolve_mip_gap` (`src/methods/nominal.py`) and passed to nominal, robust_reg,
  the wrapper and CP alike; it also covers CP's cut loop, CP's final solve and the
  prescribe-time re-solve in both evaluators. The methods are compared on their
  objective, so a per-method gap confounds that comparison: 1% of a gastric
  objective of ~10 is ~0.1 months, the same order as the differences reported
  (9.63 vs 10.55). Until **2026-08-20** nominal and robust_reg solved at **0.01**
  while the wrapper and CP solved at `1e-4`, and `metrics.py` coarsened the
  synthetic prescribe-time re-solve to **0.01** for every method — so **objective
  columns in any pre-2026-08-20 result are not gap-comparable across methods**;
  re-run before reading small objective gaps off them. `methods.cp.mip_gap` is
  still honoured as a legacy fallback if the `optimization` key is absent.
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
