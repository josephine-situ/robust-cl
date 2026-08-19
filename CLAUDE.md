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
uv run python experiments/run_rho_sweep.py --problem synthetic   # shared-D rho axis + rho*(method)
uv run python experiments/plot_rho_sweep.py             # rho-sweep figures (one cell) -> results/figures/
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

  **The objective is the nominal fit, and OS is never chance-constrained.**
  `constraint_idxs` excludes any constraint with `obj_weight != 0`, so OS gets no
  `z` indicator. Under `robustify_objective: false` (the default) the objective
  embeds one **freshly trained nominal** OS model. It used to embed `ensemble[0]`
  — bank draw 0, which is itself perturbed, since `_draw(0)` produces a nonzero δ
  — so the wrapper maximized an arbitrary perturbed OS while CP maximized the true
  nominal one, an uncontrolled and seed-dependent asymmetry in Table 6's objective
  column (measured: draw 0's OS differed from nominal by up to 1.08 months on
  gastric). **Wrapper objective numbers produced before 2026-08-15 carry that
  bias.**
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
  estimate cannot leak future information) **and follow the rows in hand**: a CV
  fold or a `train_subsample_frac` draw estimates `scale(y_c)` from *its own* rows,
  with forward-chaining cutoffs re-derived from them (`_cutoffs_from_years`), never
  from rows held out of it. **Two bugs made that false until 2026-08-17**, and only
  the second was reachable in production: `_fold_instance` left `train_pub_years` at
  full length (folds would index past the fold's rows — `IndexError` on all four
  gastric folds), and `filter_constraints` — which `cv_score_knob` calls right
  after, and which every gastric run calls per `constraint_mode` — dropped
  `train_pub_years` entirely, so `instance_folds` saw `None` and fell back to
  **random KFold**. The second masked the first. Nothing held out leaked in (the
  rows were always the fold's own), but the temporal scheme was not in force in any
  gastric run. The per-outcome scales quoted in this file (dlt 0.2547, blood 0.2495,
  constitutional 0.2912, infection 0.2580, gi 0.2680, OS 2.05) are the **temporal**
  ones and are what HEAD now produces; every gastric artifact in `results/`
  predating the fix was built on the KFold scales (0.2568 / 0.2546 / 0.2745 /
  0.2777 / 0.2609, OS 1.9962) — a −7.1% to +6.1% per-outcome difference in D's
  radius. Re-run gastric (Table 6, `--calibrate-cv`, and both rho-sweep cells)
  before citing those numbers. Synthetic is unaffected: no `train_pub_years`, so it
  was always KFold by design. **No coverage claim** is made.
- **`uncertainty.geometry`** (default `"ellipsoid"` **since 2026-08-18**; was
  `"box_l1"`) selects D's *shape*, and the two shapes are **parameterized
  separately**. `"box_l1"` is kept as an ablation and keeps `eps_0`/`budget_frac`
  untouched, so every artifact in `results/` reproduces under it. `"ellipsoid"` is the ball
  ‖δ‖₂ ≤ R_c with **R_c = ρ·scale(y_c)·√n**, and reads `rho` *alone* — `eps_0` and
  `budget_frac` are ignored under it. That is not a convenience: `budget_frac`
  cannot constrain an L2 ball (no L1 face, no support restriction), so it could
  only ever scale the radius, leaving `(eps_0, budget_frac)` **non-identifiable**
  there with only their product observable. `rho` replaces both and discards
  nothing. `√n` (not `√m`) makes ρ=1 mean "the L2 norm of iid noise at one
  unexplained sd per row", which transfers across n=200 synthetic and n=320
  gastric. The matched-size correspondence `√m·eps_c` survives as
  `UncertaintySet.radius_from_eps` for equal-budget geometry comparisons in the
  probe, but no longer parameterizes the set.

  Geometry reaches all three methods at once (bank draws, CP/probe separation,
  robust_reg's training adversary and its linear counterpart, which becomes an
  SOCP), because one that reached only some would break the shared-D comparison.
  The ellipsoid is **strictly stronger** at equal budget against a linear
  objective (Cauchy–Schwarz), while random sampling is *no better* in it, so it
  widens the random-vs-directed gap rather than closing it. Measured on synthetic:
  directed/FW 1.675 → 3.389 eps (2.0×), because there nnz(g)=57 < m=100 and the
  box wastes budget on rows that cannot move f(x\*); gastric binds (nnz 313–320 >
  m=160) so expect less. Compounding that, **ρ=1 is √2 wider than eps_0=1** at
  `budget_frac` 0.5 — so ρ=1 is ~2.8× the box's effective adversary and is *not*
  an operating point; read ρ\* off the sweep. **Nothing in `results/` was produced
  under `"ellipsoid"`, so no artifact there reflects the current default** — re-run
  before comparing, and never read a `box_l1` number against an ellipsoid one.
  Measured quantities quoted below (distances, τ ranges, iteration counts) were
  taken under `box_l1` unless stated otherwise.

  **ρ is swept, not fitted — and ρ\*(method) is what the evaluation run uses**
  (`experiments/run_rho_sweep.py`). D is literally shared at **every point of the
  sweep**, and that curve is where the shared-D comparison is read: at one ρ the
  methods differ only in what they do with the same set. The derived ρ\*(method) —
  the largest ρ whose held-out feasibility still meets the target — is then fixed
  per method for evaluation, so **evaluation matches held-out feasibility, not D**:
  each method faces a ball of its own radius there, and a cross-method objective
  gap is read at matched robustness rather than matched uncertainty. What keeps
  that honest is the criterion, held-out feasibility on training folds. Never fit ρ
  against the GT ensemble (tunes to the judge) or against synthetic's known
  `noise_std` (calibrates D to the DGP, so CP wins by construction). Note
  `robust_reg`'s `label_eps` **is** the D radius, so it tracks ρ through the sweep rather than staying fixed; τ and α are
  separate dials and do stay fixed. And CP samples D with B=200 against the
  wrapper's P=20, so a ρ\* gap between them is confounded with sampling density —
  `--match-bank` sets B=P to remove it.

- **`chi2_radius` is unused (zero call sites) and must stay that way while the
  scale is `oof_sd`.** It was documented as the coverage claim an ellipsoid makes
  available. It is not one here. Of the four assumptions, three fail: σ is
  `oof_sd`, which is label noise *plus* misspecification and on gastric is ~90% the
  latter (the nameable label noise, from binomial sampling SEs of the toxicity
  proportions, is 1.7–3.1× smaller); the noise is heteroskedastic (arm sizes 26–92
  give per-row SEs spanning 2.2×) while χ²_n assumes one σ; and OOF residuals are
  not independent across rows, so effective df < n. Only Gaussianity holds, and
  benignly — gastric toxicity residuals are symmetric (|skew| ≤ 0.33) and
  thin-tailed (excess kurtosis −0.24 to −1.04), so a χ² radius would err
  conservative; OS does not qualify (skew +0.64, shapiro_p 6e-3). Beyond all four,
  coverage of δ is not coverage of the *decision*: under misspecification there is
  no true θ for D to cover. Also note χ² concentration makes the coverage *level* a
  nearly inert knob — at n=320 the 50% and 99% radii differ by 9.3% — so it must
  never be used as a conservatism dial. A real claim would come from the
  measurement rather than the fit (per-row binomial σ_i, weighted ellipsoid); that
  is a genuinely narrower D and a different paper, not a drop-in.
- **How each method samples the worst case — matched in magnitude, deliberately
  NOT in direction.** robust_reg uses a **directed** adversary (`worst_case_label_shift`
  greedy top-m by residual, or the closed form `R·r/‖r‖` under `ellipsoid`). CP and
  the wrapper use **random** boundary draws from the bank. Same D, same budget
  spent, different alignment — and that asymmetry is intentional.

  The reason is feasibility, not fairness. CP and the wrapper turn scenarios into
  *embedded constraints* — CP a cut per accepted scenario, the wrapper a joint
  chance constraint over P models. A directed adversary makes each of those as
  tight as D allows, and they accumulate until the master admits no prescription
  at all. That is exactly what `run_adversary_probe.py` Part C/D measures, and
  what CP's rollback / permanent-rejection machinery already contains at *random*
  draws. robust_reg is immune because its adversary never becomes a constraint: it
  shapes the **fit**, one model per outcome is retrained on the shifted labels, and
  that single model is embedded. A worst-case shift moves where the model sits; it
  cannot make the optimization infeasible.

  The cost is real and should be reported rather than hidden: the best of B random
  draws reaches **1.07 eps** against a directed adversary's **1.67 eps** on
  synthetic (~64%), and the gap *widens* under `"ellipsoid"` (`g'u` has the same sd
  under both geometries while the attainable max rises). **"Shared D" guarantees a
  shared set and equal budget, not equal adversary strength.**

- **`ScenarioBank`** draws B **vertices** of D (±eps on `budget_frac`·n rows, 0
  elsewhere — matching robust_reg's adversary in magnitude; interior draws would be
  weaker still at the same D) and trains one model per draw per outcome, with
  `random_state` fixed across members so the scenario is the only variation.
  Draw *b* is a pure function of `(seed, b)`, so **the wrapper's P models are a
  nested prefix of CP's B** — which is what makes the α=0 ≡ τ→0 equivalence exact.
- **`coherent` is a *grouping*, not a flag** (`uncertainty.coherent_exclude`).
  Coherent shares one standardized direction across the group (scaled by each
  outcome's own radius); constraints named in `coherent_exclude` draw
  independently even under `coherent: true`. Production sets it to
  `["os_constraint"]`. Vacuous on synthetic (one outcome) — literally, not
  approximately: with one `MLModelData` both branches make the same single
  `_vertex_direction` call against the same rng, so the banks are bit-identical.
  An empty list reproduces pre-grouping banks bit-for-bit; unknown names are
  reported and ignored rather than raising, because one `config.yaml` drives both
  problems and `os_constraint` legitimately does not exist on synthetic.

  **Objection 1 below is now RESOLVED by measurement (2026-08-15); objections 2,
  3 and 4 remain open.**

  1. **RESOLVED — OS does not belong on the shared direction.** The cross-outcome
     residual correlation, previously never estimated, now is: out-of-fold
     residuals over the same rows/folds/frozen configs the bank uses (n=145 under
     forward-chaining) give **+0.28** across non-DLT toxicity pairs on the
     percentile labels δ actually perturbs (+0.22 raw), against **+0.06** for OS
     versus every toxicity (+0.02 raw, 3 of 5 pairs negative). DLT is excluded
     from that average because `DLT_PROP = 1 − ∏(1 − tox)` makes it a
     deterministic function of the other four — verified exact to 2e-16, over
     exactly the four modeled outcomes (`gastric_v11.py:215` builds it from
     `BLOOD_4`, `CONSTITUTIONAL_34`, `INFECTION_34`, `GI_34`) — so its +0.44–0.80
     row is construction, not evidence. **Excluded from the average, not from the
     group**; see (4). Coherent asserts +1 and incoherent asserts 0;
     neither fits both blocks, hence the grouping. This matches the story that
     justified coherence in the first place — *record-level* mismeasurement, a
     study that under-reports adverse events under-reporting across all five
     toxicity endpoints — which never covered survival. (Sign is
     per-outcome-label, not clinical valence: `+1` on a toxicity percentile is
     worse, `+1` on OS months is better.) **Still asserted, not estimated:** the
     truth within the toxicity group is +0.28 and we impose +1. Correlated draws
     (a copula on the estimated Σ) remain the unexplored option; the grouping is
     the cheap 90%.
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
  4. **DLT is excluded from the correlation average but stays *in* the coherent
     group**, and the identity that disqualified its correlation also makes its
     draw inconsistent. `coherent_exclude` is the only exclusion and it names OS
     alone, so `dlt_constraint` takes the shared direction: `δ_dlt = R_dlt · u`
     with the same `u` as the other four, hence exactly collinear before clipping
     (measured +1.0000; clipping to [0,1] is the only decorrelator — +0.94–+0.96
     at ρ=1 with ~20% of rows clipped, +0.97–+0.99 at ρ=0.25). Two consequences.
     The group spends **five outcomes' radius on four degrees of freedom**. And
     the draw is not a *consistent* relabeling: δ perturbs each outcome's
     percentile labels independently, so **no δ ∈ D leaves perturbed-DLT equal to
     `1 − ∏(1 − perturbed tox)`** — a study that under-reports the four
     components has a fully *determined* DLT shift, not a free one, so coherent
     hands DLT a full-radius shift the mismeasurement story does not license.
     Affects both arms (incoherent draws DLT independently, which is no more
     consistent). Keeping DLT in the group is still the right **sign** — its
     residuals do move with the others — and it is the magnitude that is
     asserted, so this is a known overstatement, not a bug. The consistent
     alternative (perturb the four components only, re-derive DLT through the
     identity, re-percentile) is a change to `ScenarioBank._draw` and the label
     construction, **not a config flip**.

  Changing (2) is not free: the shared-`b` cut is what makes CP at τ→0 identical
  to the wrapper at α=0 and what makes permanent scenario exclusion sound
  (reasoning at `cp.py:1506-1522`). A per-constraint argmax would break both and
  would defend against points that correspond to no single relabeling of one
  trial. Cheapest next step is measurement, not a rewrite: build both banks at
  the same seed and B, score every draw at the nominal $x^*$, and compare the max
  and the full distance distribution — a bank scan, no MIP resolves.
- `uncertainty.eps_0` / `budget_frac` (box) and `rho` (ellipsoid) are shared
  **constants, not knobs** — each method keeps exactly one conservatism dial (CP
  τ, wrapper α, robust_reg `label_eps`). `eps_0` is deliberately *not* pinned to
  robust_reg's calibrated ε*, which would conflate D's definition with one
  method's tuning. `rho` is swept rather than fitted for the same reason (see the
  geometry bullet). **Do not fix τ→0 and α=0 together** when choosing fixed dials
  for a sweep: at B=P that makes CP identical to the wrapper by the α=0 ≡ τ→0
  equivalence, collapsing two of the three methods into one solver and leaving
  nothing to compare on decisions. The CV picks (τ=0.1, α=0.2–0.3) are clear of
  that corner. Note τ\* sits at the *strong end* of its grid on both gastric
  coherence cells, i.e. CV is saturating rather than selecting — which is why
  fixing τ costs nothing measurable.

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

**τ is measured in unexplained standard deviations** (`cp.tolerance_basis:
"scale"`, the default). Each exceedance is divided by its outcome's own
`scale(y_c)` — the same scale that sets D's radius — and averaged over (anchor ×
outcome) cells, so τ reads as "stop when the mean exceedance is below τ sd". Three
consequences: τ is a physical quantity in the **same units as `rho`**, not a ratio
to a bank statistic; it does not move with the seed, the bank, or B; and **the
grid spans nominal**, because a τ above the iteration-0 distance stops before any
cut. Verified on synthetic (scale 0.1281, max iter-0 violation 0.1259): τ=1.0
returns the nominal objective exactly in 1 iteration.

The basic path keeps its violations **raw** for logging, so it converts τ into
constraint units by multiplying by the scale instead of dividing; the two paths
therefore log different units but τ means the same thing in both.

**One τ grid, but it needs log-scale range.** The sd basis makes τ a single
physical quantity everywhere — "stop when the population-mean exceedance is below
τ unexplained sd" — so a shared grid is meaningful. What it does not do is put
both problems in the same *part* of that axis. Both paths max over draws; they
differ in what one draw *scores as*. The basic path has a single (anchor ×
constraint) cell, so a draw's score is that raw violation — effectively
conditioned on violating. The coherent path means over `n_anchors × n_outcomes`
cells, i.e. **(violating cell fraction) × (mean exceedance among violators)** — a
factor the basic path has no room for. Gastric's violating fraction (~0.1) is why
its maxima sit at ~0.03 against synthetic's ~0.98.

That is a difference in **range, not meaning**, and the fraction is information
(breadth), not a unit mismatch — gastric scenarios genuinely break a smaller share
of the anchor population. The same way ρ\* differs across problems without making
ρ incomparable. Hence `knob_grids.cp` is one plain **decade** grid
(`[1.0, 0.1, 0.01, 0.001]`), which brackets both without calibration.

**Do not pin the τ grid from measured distances.** The iteration-0 distance range
is a function of D's size, i.e. of `rho` — so a grid fitted at one ρ would need
refitting after every ρ, and reading τ off a run τ helped produce is circular. The
order is: **(1)** ρ sweep at one fixed reasonable τ, **(2)** τ ablation at the
chosen ρ on the decade grid.

**The scale is per outcome, not one number.** `_build_scale_map` returns
`{c_idx → scale(y_c)}` and each cell is divided by *its own* outcome's `oof_sd` —
so the mean averages dimensionless quantities and outcomes on different label
scales are commensurable. Measured on gastric: dlt 0.2547, blood 0.2495,
constitutional 0.2912, infection 0.2580, gi 0.2680 — a 1.17× spread, so the
normalization does almost nothing *there*, because the percentile transform makes
every toxicity's `sd(y) ≈ 0.2887` by construction. It would matter a great deal if
OS were a constraint (`oof_sd = 2.05` months, ~8× the toxicities); OS is the
objective, so it is excluded from the map and only enters under
`robustify_objective`, where its own scale is fetched separately.

**Expect `status="max_iterations"` at the small-τ end.** CP still returns its
incumbent, so a sweep will not crash — but those cells are *not converged* and
must be reported as capped rather than as a converged answer at that τ.

The **mean is the right statistic and is anchor-count stable.** Breadth is what it
is for: a scenario breaking 2 of 20 cells badly is less dangerous for a policy
serving a population than one breaking all 20 moderately, and a max would rank
those backwards. It is stable because the numerator scales with the denominator —
both factors are sample statistics converging to a population quantity that does
not depend on `n_anchors`. Measured on gastric (B=25, kmedoids train anchors), max
iteration-0 distance: `n_anchors=4 → 0.0307`, `8 → 0.0315` (1.03×; a divisor
artifact would predict ~0.5×), `16 → 0.0178`. No 1/n trend; the residual spread is
anchor-*set* variation and shrinks as the estimator converges.

Verified that τ=0.1 on gastric `--quick` added **zero** scenario cuts
(`dist_tol=0.100 > max dist 0.0233`), i.e. returned nominal — that is τ=0.1 being
gastric's *nominal endpoint* on the shared grid, not the grid failing. Read
`[cp] basis=scale … max iter-0 dist=` off a real run before assuming the grid
brackets a **new** problem.

`"scale"` also retires the per-cell divisor `max(1, |rhs|)`, which was a **no-op
on gastric**: `rhs = 0.6`, so `max(1, 0.6) = 1` and the "normalized" distances
were raw percentile units all along.

**Legacy `tolerance_basis: "d0"`** reproduces the old behavior: tolerance = τ ·
`q0.9` of the iteration-0 distances, while the stopping statistic is the **max**
over the bank — so iteration 0 fails its own test and **no τ in `[0.1, 1.0]`
reproduces nominal** (measured: τ=1.0 still cuts, obj −1.2757 vs nominal −1.3055;
it takes τ=2.0 there). Under that basis τ is a ratio to each problem's own `d0`,
never a shared physical quantity. Also note coherent drops permanently-rejected
draws after iteration 0 (113/200 on gastric) and rejection correlates with
severity, so gastric's max is measured with its worst tail deleted.
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

**Under the ρ parameterization the calibration target is ρ\*, not each method's
knob.** `experiments/run_rho_sweep.py` is what stage-1 knob CV used to be: it
sweeps the shared ρ with every method's own dial held fixed (τ = 0.01, α = 0.2)
and reports **ρ\*(method)** — the largest ρ whose held-out feasibility still meets
the target. τ and α move to *ablations at one chosen ρ* (`--ablate`), which is all
that is needed to show the fixed values were not cherry-picked. ρ itself is never
*fitted* — ρ\* is read off the reported curve at a fixed feasibility target, and
the curve stays the primary result. But it **is** the point the evaluation run
uses: each method is evaluated at its own ρ\*, so D is shared across methods on
the sweep and **not** at evaluation, where the match is on held-out feasibility
instead (see the geometry bullet in `uncertainty.py`).

Every swept cell carries `status`, `n_capped`, and the wall clock split into the
**master** phase (train + build + solve to the final master; for CP the whole cut
loop) and the **test-point** phase (one prescribe solve per held-out context) —
`cv_score_knob(..., return_details=True)`. The split is the comparison CP's MIP-size
claim rests on: CP pays up front in the cut loop then prescribes from a small
master, while the wrapper embeds all P models and pays again on every test point.
Cells with `n_capped > 0` hit `max_iterations`; they are **kept and flagged**, not
dropped — the incumbent is still usable, and the flag travels with the row.

**ρ\* is a reporting choice and is re-derivable without re-solving.**
`{problem}_rho_curve{cell}.csv` carries every column the criteria could need, and
`--rho-star-only` recomputes the table from it under a new `--feas-target` /
`--min-solved` / `--exclude-capped`, writing to `--out-suffix` so several criteria
coexist. The chosen criteria are written back as columns, so no ρ\* table is
ambiguous about the rule that produced it.

**Every sweep output is scoped by its cell** — `_coh`/`_incoh`, plus `_matchbank`
under `--match-bank` (`_variant_suffix`). The coherent/incoherent and B=200/B=P
runs are *different experiments* that the workflow asks you to run as a pair, and
sharing one filename failed silently in both directions: the resume checkpoint is
keyed `(method@rho, knob)` **only**, so a second cell resumed the first's rows and
reported them as its own, and the curve is written rather than appended, so the
second cell overwrote the first. `--rho-star-only` takes the same cell flags to
pick which curve it reads.

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
- **`variable_lb`/`variable_ub` split by variable role.** Treatment (decision)
  columns take their box from `X_fit` — the fit rows, subsample included — because
  the optimizer *chooses* them; they used to come from `X_valid` (train+test), which
  let held-out arms widen the action space (4 of 84 columns: S-1 and Trimetrexate
  dose ceilings). The feasible set is unchanged, since the trust region already
  confines decisions to the hull of those same rows; the box is now tight, which
  only prunes more unreachable leaves in `embed._embed_leaves`. Context columns stay
  on train+test **deliberately**: contexts are never chosen (evaluation overwrites
  them with `lb=ub=` the given test row), the box only has to *contain* them, and
  narrowing it to train would make every test solve infeasible — the split is
  temporal, so every test row's `Pub_Year` exceeds the training max by construction
  and leaf pruning would delete every leaf covering it.
- Parameter robustness (`methods.robust_param.rho`) shrinks decision-tree leaf
  regions by a margin from the split thresholds; applied across all methods.
