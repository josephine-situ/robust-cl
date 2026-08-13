# Robust Constraint Learning via Iterative Scenario Generation

## Overview

This repository implements and compares approaches for making
constraint learning (Maragno et al., 2023/2025) robust to label
uncertainty. When a trained ML model $\hat{f}(x;\theta)$ is embedded as a
constraint $\hat f(x) \le b$ inside an optimization problem, noisy training
labels can produce models whose "optimal" decisions violate the true
constraint.

We formulate this as a trilevel optimization problem and solve it via
**Cutting Planes** (`src/methods/cp.py`) — the contribution of this repo —
benchmarked against three baselines, across a synthetic problem and the
OptiCL gastric-cancer chemotherapy case study (replicating Maragno et al.
2025, Table 6).

## Formulations

**Nominal Constraint Learning**

Embeds a trained ML model $\hat{f}(x;\theta^\ast)$ as a constraint in an optimization problem:

```math
\min_{x \in \mathcal{X}} \; c^\top x \quad \text{s.t.} \quad \hat{f}(x;\theta^{\ast}) \leq b
```

**The Vulnerability:** Labels $y$ are noisy ($y^\ast = y + \delta$). Different perturbations $\delta$ lead to different trained models $\theta^\ast(\delta)$ and vastly different "optimal" decisions. The optimizer frequently exploits the errors in the nominal model.

**Robust Constraint Learning (Trilevel Formulation):**

We seek a decision $x$ that remains feasible for *every* model resulting from a plausible label perturbation $\delta \in \mathcal{D}$.

```math
\begin{aligned}
\min_{x \in \mathcal{X}} \quad & c^\top x \\
\text{s.t.} \quad & \max_{\delta \in \mathcal{D}} f(x;\theta^{\ast}(\delta)) \leq b \\
\text{where} \quad & \theta^{\ast}(\delta) = \arg\min_{\theta \in \Theta_{\mathrm{feas}}} \mathcal{L}(\theta; X, y+\delta)
\end{aligned}
```

$\mathcal{D}$ is never a parametric noise model here — it is realized as **bootstrap resamples of the observed labels** (see "Robust Methods" below). Cutting Planes iteratively approximates the reachability set of models $\Theta^\ast$: on each step, a separation oracle resamples training data local to the current $x^k$ to find the worst-case model, and adds it as a cut.

## Problem Settings

The scenarios differ only in the mapping between models, constraint terms, and LPs. Let $\hat{f}_m(\cdot;\theta_m)$ index trained models, $q \in \mathcal{Q}$ index the LPs solved at evaluation (with context $z_q$, decisions $x_q$), and each learned constraint take the form

```math
\sum_{(m,k) \in \mathcal{T}_c} w_{c,m,k}\,\hat{f}_m\bigl(a_{c,k}(x_q,z_q);\theta_m\bigr) \leq b_c
```

where $\mathcal{T}_c$ is the set of model occurrences in constraint $c$. The robust oracle separates $\max_{\theta_m \in \Theta_m^\ast}(\cdot)$ at the incumbent; its granularity is set by $Q_m$, the query points where model $m$ is evaluated, and whether they share $\theta_m$.

**Notation glossary:**

| Symbol | What it indexes |
|---|---|
| $m$ | a **trained model** — one specific fitted $\hat f_m(\cdot;\theta_m)$ out of the whole catalog used anywhere in the problem (e.g. gastric's separate DLT model, Blood model, OS model, ...) |
| $\theta_m$ | model $m$'s fitted parameters |
| $c$ | a **learned constraint** — one row of the optimization problem, e.g. gastric's `dlt_constraint`, `blood_constraint`, etc. |
| $k$ | an **occurrence** of a model *within* constraint $c$ — the same model can appear more than once in one constraint's sum, evaluated at different input slices |
| $\mathcal T_c$ | the set of $(m,k)$ pairs — i.e. which models, and how many times each, get summed together to form constraint $c$'s left-hand side |
| $w_{c,m,k}$ | the weight multiplying that occurrence's prediction in constraint $c$'s sum |
| $a_{c,k}(x_q,z_q)$ | the function that builds the actual input vector fed to model $m$ for occurrence $k$ of constraint $c$, from the LP's decisions $x_q$ and context $z_q$ (a slice/rearrangement — the model doesn't necessarily see the whole $(x_q,z_q)$ vector) |
| $b_c$ | constraint $c$'s right-hand side bound |
| $q$ | one **separate LP** solved at evaluation — e.g. one per patient cohort in gastric, or the single instance in synthetic |
| $x_q, z_q$ | the decision variables and context specific to LP $q$ |
| $\mathcal Q$ | the whole set of LPs solved at evaluation time |
| $\Theta_m^\ast$ | the reachability set for model $m$ specifically — the set of models reachable by retraining on some admissible label perturbation; the separation oracle solves $\max_{\theta_m\in\Theta_m^\ast}(\cdot)$ |
| $Q_m$ | the set of query points where model $m$ actually gets evaluated, across every $q$ and $k$ it appears in — the number that determines separation difficulty |

**Worked example — gastric.** Constraint $c=$ `dlt_constraint` has $\mathcal T_c = \{(m_{\text{dlt}}, 1)\}$: one model, occurring once, weight $w=1$, $a_{c,1}$ picks out the treatment-arm features from $(x_q,z_q)$ — the sum collapses to $\hat f_{m_\text{dlt}}(x_q;\theta_{m_\text{dlt}}) \le b_c$. This repeats per patient cohort $q$ (each cohort is its own LP with its own $x_q,z_q$), so $Q_{m_\text{dlt}}$ spans many query points across $q$ — but each cohort gets its **own**, independently-found worst-case $\theta_{m_\text{dlt}}$; no requirement that one $\theta$ be simultaneously worst for every cohort. That independence is "replicated per LP (parallel)": $Q_m$ is large, but it factors into separate, unlinked sub-problems.

**Worked example — marketing** (unimplemented, but the point of the notation). Here $\lVert\mathcal Q\rVert=1$ — one LP — but a single constraint $c$ sums over many customers/segments: $\mathcal T_c=\{(m,1),(m,2),\dots,(m,K)\}$, all the *same* model $m$, each $a_{c,k}$ picking out a different customer's context, all sharing **one** $\theta_m$. Now $Q_m$ has many points, but they can't be separated independently — the oracle must find the single worst-case $\theta_m$ that's simultaneously bad across the whole weighted sum $\sum_k w_{c,m,k}\hat f_m(a_{c,k}(x,z_k);\theta_m)$. That's "coupled": one harder joint cut instead of many easy parallel ones.

| Axis | Synthetic | Gastric cancer | Marketing |
|------|-----------|----------------|-----------|
| Datasets → models | 1 → 1 | $M$ → $M$ (one per constraint) | 1 → 1 |
| Model occurrences per constraint | 1 | 1 | many (summed) |
| Learned constraints per LP | 1 | many | 1 (+ model's own) |
| Decision vars | per-LP | shared across models | shared; multiple input copies |
| LPs at evaluation $\lVert\mathcal{Q}\rVert$ | 1 | many (one per cohort) | 1 |
| Query points $Q_m$ | one point | one per LP (independent) | many in one constraint (shared $\theta$) |
| Separation *(structure permits)* | single point-localized cut | **replicated** per LP (parallel) | **coupled** over the point set |

**Two roles of context.** *Parametric* context (gastric) indexes separate LPs: each $z_q$ yields its own solution ${x_q}^{*}(z_q)$ and an independent, point-localized oracle — granularity scales *out* (more solutions, parallel cuts). *In-LP* context (marketing) places several context points in one constraint sharing a single $\theta$: there is one solution, but the oracle must choose one worst-case model maximizing the *aggregate* over $Q_m$ — granularity *couples* (one harder cut over a point set, not term-by-term). Synthetic is the degenerate case of both ($\lVert Q_m \rVert = 1$). **Marketing is described here as a target setting but is not implemented** (no data loader).

**What CP actually implements is more coupled than gastric's structure requires.** The last table row describes what each problem's *structure* permits, not what the solver does. Gastric's parametric context would allow a fully independent worst-case $\theta$ per LP, but the implemented `coherent` strategy (default `eval_mode: global`) deliberately couples the anchors instead: it solves **one shared master**, evaluates every candidate scenario at **all** $x^\ast_1,\dots,x^\ast_K$, ranks scenarios by the **mean** normalized exceedance across all (anchor × outcome) cells, and adds a **single cut** — a fully embedded model valid at *every* context, not a per-anchor constraint. That is precisely what keeps a robust region built from *training* anchors sound when `evaluate_prescribed_table6` later re-solves for unseen test cohorts. (A literally per-anchor variant exists — `eval_mode: per_anchor_nearest` trains one master per anchor — but an ablation found it hurt both feasibility and OS, so it is not the default.)

### Concrete formulations

**Synthetic** — a single global LP, no context, the degenerate case of every axis above. The true constraint function is known exactly (used only for final evaluation, never for training):
```math
f_{\text{true}}(x) = \sum_{j=1}^d x_j^2 \;+\; 0.5\prod_{j=1}^d x_j, \qquad x\in[0,1]^d
```
Training labels are noisy draws around it, $y_i = f_{\text{true}}(x_i)+\varepsilon_i,\ \varepsilon_i\sim\mathcal N(0,\sigma^2)$ ($\sigma=$ `noise_std`); the embedded constraint $\hat f(x;\theta)$ is fit on these noisy $y$, never on $f_\text{true}$ directly. The optimization problem:
```math
\min_{x\in[0,1]^d} \; -\sum_{j=1}^d x_j \quad\text{s.t.}\quad \hat f(x;\theta)\le b, \qquad b = 0.5\,d
```
($b$ is set so the constraint boundary sits strictly inside the box rather than at a corner.) One constraint, one model, one LP — the setting CP's **basic** separation targets.

**Gastric** — many LPs (one per patient cohort $q$), many learned constraints per LP, plus a learned objective. Decision variables $x$ are the treatment/drug-dose block (binary drug-on indicators + weekly average doses), restricted to the convex hull of observed treatment vectors (the trust region, `src/utils/trust_region.py` — this subsumes the per-coordinate box bounds, which remain only to tighten the embedding's big-M constants) plus one domain constraint capping the regimen at **three drugs** ($\sum_j x_j^{\text{Ind}} \le 3$); context $z$ is nine patient covariates (`Pub_Year`, `Asia`, `N_Patient`, `FRAC_MALE`, `AGE_MED`, `Prior_Palliative_Chemo`, `Primary_Stomach`, `Primary_GEJ`, `ECOG_MEAN`) — fixed per cohort, never optimized. Five toxicity outcomes are each converted to a **percentile score** relative to the training distribution and capped at the 60th percentile (`GASTRIC_TOX_UB`, overridable by `--rhs-grid`); overall survival (OS) is the learned objective, maximized:
```math
\max_{x\,\in\,\mathrm{hull}(X_\text{train}),\;\sum_j x_j^{\text{Ind}} \le 3} \widehat{\text{OS}}(x,z;\theta_\text{os})
\quad\text{s.t.}\quad \widehat{\text{pctl}}_c(x,z;\theta_c) \le 0.6, \;\; c\in\{\text{DLT, Blood, Constit., Infection, GI}\}
```
Five learned constraints plus a learned objective, evaluated at many context
anchors — the setting CP's **coherent** separation targets. The cohort splits
temporally (`split_gastric_v11`, train = `Pub_Year` $\le 2008$, test = the
following four years): **320 training arms, 96 test arms**, 416 in total.

## Robust Methods

All four methods share the same MIP scaffolding in `src/methods/nominal.py`
(`build_decision_vars`, `add_problem_constraints`, `embed_constraints`,
objective builders) and the same MIO embedding in `src/models/embed.py`
(trees via big-M leaf-selection binaries, ensembles as averages/sums, MLP
ReLU via binary activations, pipelines recursing through the scaler). Each
returns a `SolutionResult`. Uncertainty is **data-driven** throughout —
bootstrap resamples of observed labels — never a parametric noise model.

### Nominal (`nominal.py::solve_nominal`)

No robustness. The baseline every other method is compared against.

1. Train one model per learned-constraint outcome on the full training data.
2. Embed each model as MIP constraints; any model carrying a non-zero `obj_weight` (gastric's OS) becomes an objective term instead, under the epigraph $\min c^\top x + t,\ t \ge \sum_i w_i \hat f_i(x)$ — the same $t$ CP and Wrapper later raise with worst-case cuts.
3. Solve for $x \in \mathcal{X}$ ($\mathcal{X}$ = box bounds + domain constraints + trust region).

### Robust Regression (`robust_regression.py::solve_robust_regression`)

Trains a single model *robust to label noise* per outcome (Bertsimas, Dunn,
Pawlowski & Zhuo 2019, Sec. 5, adapted from classification to regression),
then embeds it nominally — robustifies the **model**, not the decision.

1. Define a bounded additive label-uncertainty set per outcome,
   $\mathcal{D} = \{\delta : \lVert\delta\rVert_1 \le \gamma,\ |\delta_i| \le \varepsilon\}$,
   with $\varepsilon = \texttt{label\_eps} \cdot \mathrm{std}(y)$ (unitless knob, scaled per outcome) and $\gamma = \texttt{budget\_frac} \cdot n \cdot \varepsilon$.
2. Solve the label-robust training counterpart $\min_\theta \max_{\delta \in \mathcal{D}} \mathcal{L}(\theta; X, y+\delta)$:
   - **Linear** (ElasticNet loss): the inner max has a closed convex form attained at a vertex of $\mathcal{D}$ (a top-$m$-largest-residuals penalty term) — solved exactly as one Gurobi QP, no iteration.
   - **Tree / ensemble / MLP**: no closed form — an adversarial-training loop alternates finding the worst-case label shift for the current model (`worst_case_label_shift`) and retraining on it (`retrain_on_perturbed`), for `K` iterations or until the shift stops changing.
3. Embed the resulting single robust model per outcome (same nominal MIP embedding).
4. Solve for $x^\ast$.

Knob: `label_eps` (perturbation radius). **Not coherent across constraints**: each outcome's worst-case label shift is solved independently, with its own residuals and its own $\varepsilon$ — unlike Wrapper and CP's coherent path, there is no single shared relabeling required to be simultaneously worst for every constraint.

#### Deriving the linear closed form

For linear/ElasticNet regression the inner maximization has an exact convex reformulation, so the whole robust counterpart collapses to **one QP, no iteration** (`_label_robust_linear`, [robust_regression.py:48-98](src/methods/robust_regression.py#L48-L98)):

1. **The adversary's problem, for a fixed $\theta$.** With residuals $r_i = y_i - \hat y_i(\theta)$, the inner max is
   ```math
   \max_{\delta \in \mathcal D} \sum_i (r_i + \delta_i)^2, \qquad \mathcal D = \{\delta : |\delta_i|\le\varepsilon,\ \lVert\delta\rVert_1\le\gamma\}
   ```
   Squaring is convex increasing in $|r_i+\delta_i|$, so the optimum sits at a **vertex** of $\mathcal D$: push $\delta_i = \pm\varepsilon$ (same sign as $r_i$, to grow $|r_i|$) on the largest-$|r_i|$ coordinates first, greedily, until the $L_1$ budget $\gamma$ is spent. Writing $m=\gamma/\varepsilon$ (the number of points that receive the *full* $\varepsilon$ shift; the $m$-th may be fractional), this is exactly `worst_case_label_shift` ([perturbations.py:33-59](src/utils/perturbations.py#L33-L59)) — an exact vertex solution, not a heuristic.
2. **Plugging the vertex back in.** Substituting $\delta_i^\ast$ gives
   ```math
   \sum_i (r_i+\delta_i^\ast)^2 = \sum_i r_i^2 \;+\; 2\varepsilon\, S_m(|r|) \;+\; \varepsilon^2 m
   ```
   where $S_m(|r|)$ is the sum of the $m$ largest $|r_i|$. The last term is a **constant** (doesn't depend on $\theta$, only on the fixed knobs $\varepsilon,\gamma$), so it drops out of $\arg\min_\theta$, leaving the tractable data term $\tfrac{1}{2n}\big[\sum_i r_i^2 + 2\varepsilon S_m(|r|)\big]$.
3. **Making $S_m(\cdot)$ convex-representable.** The sum of the $m$ largest elements of a vector has the standard epigraph form
   ```math
   S_m(a) = \min_t \Big\{ m\,t + \sum_i \max(0,\, a_i - t) \Big\}
   ```
   (search over a threshold $t$; anything above $t$ contributes its excess, and $m$ is exactly calibrated so the optimal $t$ sits at the $m$-th largest value). This turns $S_m(|r|)$ into a small LP nested inside the outer problem.
4. **One joint QP.** Because $r_i$ is affine in $(\beta,b_0)$ and every other piece above is convex piecewise-linear, minimizing over $\theta=(\beta,b_0)$ **and** the epigraph variables $(t, q_i, a_i)$ *simultaneously* is jointly convex — there's no need to alternate between adversary and model. The code builds exactly this: `a[i]=|r_i|` via two linear inequalities, `q[i] >= a[i]-t` for the epigraph, `top_m = m*t + sum(q)`, plus the ElasticNet $\ell_1/\ell_2$ penalty terms, all as one Gurobi QP solved once.

This exactness is specific to squared loss over a box-∩-$L_1$-ball set — it's *why* robust regression needs an explicit, simple uncertainty-set geometry (see "Why do we choose a label uncertainty set for robust regression but not CP?" — no such closed form exists for CP's embedded tree/ensemble models, and CP additionally needs the worst case localized to the current decision $x^k$, which a fixed global set doesn't support).

#### The adversarial-training loop (non-linear model classes)

Trees, RF, XGB, and MLPs have no such convex reformulation, so `_label_robust_loop` ([robust_regression.py:101-116](src/methods/robust_regression.py#L101-L116)) approximates the same minimax by alternating the two half-problems it *can* solve exactly, holding the other fixed:

```
model = train_model(X, y)                                    # start at the nominal fit
repeat up to K times:
    residuals = y - model.predict(X)
    delta = worst_case_label_shift(residuals, eps, gamma)     # exact vertex solve, same rule as above, given the CURRENT model
    if delta == previous delta: break                          # fixed point reached
    model = retrain_on_perturbed(X, y, delta)                 # retrain on y + delta (replaces the model)
```

Each step is individually exact — the adversary step reuses the identical greedy top-$m$ vertex rule from the linear derivation (it only needs $r_i$, which is well-defined for any model), and the retraining step is an ordinary model fit. What's *not* exact is the alternation itself: for a non-convex model class there's no guarantee this fixed-point iteration converges to the true joint minimax (it can cycle rather than converge), which is why it's capped at a small `K` (default 5) rather than iterated to convergence — the `np.allclose` check only gives an early exit if $\delta$ happens to stabilize.

This is a different kind of iteration from Cutting Planes, worth contrasting directly: robust regression's loop alternates over $(\theta,\delta)$ **before any decision exists** — $x$ never appears in it, and the single resulting model is embedded once, unchanged, regardless of what $x^\ast$ the optimizer later picks. CP's loop alternates over $(x^\ast, \text{worst-case models})$ — the adversary step is anchored to whatever $x^\ast$ the master just solved for, uses a finite *sampled* search over bootstrap resamples rather than an exact argmax, and **accumulates** cuts into a growing master constraint set rather than replacing a single model outright.

### Wrapper (`wrapper.py::solve_wrapper`)

Maragno et al. (2025) ensemble chance constraint, made *coherent* across
constraints in this codebase (one shared bootstrap replicate must satisfy
every toxicity constraint jointly, rather than independently per constraint).

1. Draw $P$ bootstrap resamples of the training rows. The gastric runner passes `_coherent_bootstrap_indices`, so the *same* $P$ index vectors drive every outcome and replicate $p$ is one coherent relabeling of the whole cohort; `solve_wrapper`'s own default (`_get_shared_bootstrap_indices`) resamples per `MLModelData` independently, which is equivalent on synthetic (one model) but not on gastric.
2. Train $P$ models per outcome on these shared resamples.
3. Embed all $P \times (\text{outcomes})$ models; add a binary indicator $z_p = 1$ iff decision $x$ satisfies *every* toxicity constraint under replicate $p$'s models.
4. Impose the joint chance constraint $\tfrac{1}{P}\sum_p z_p \ge 1-\alpha$.
5. Robustify the learned objective (OS) with a worst-case epigraph over the same $P$ replicates.
6. Solve for $x^\ast$.

Knob: `alpha` (max fraction of replicates allowed to violate). $P$ comes from
`uncertainty.n_bootstrap` on gastric (`methods.wrapper.n_estimators` is used only
by the synthetic runner). A second variant, `solve_tree_violation_wrapper`,
reproduces OptiCL's literal per-tree random-forest chance constraint (knob
`methods.chemo_wrapper.alpha`) — used only as the paper-replication baseline for
Table 6, not one of the four compared methods.

### Cutting Planes — ours (`cp.py::solve_cp`)

One driver handles every problem shape and **auto-selects** the separation
strategy from it — there is no separation flag. `basic` requires *all four* of:
non-contextual, $\le 1$ learned constraint, a single anchor, and no robustified
learned objective (`_run_cp_loop`); everything else, including every contextual
gastric configuration, gets `coherent`.

1. Train nominal models and build the master MIP; for contextual problems (gastric) also select representative **context anchors** $z_1,\dots,z_K$ (`select_anchor_contexts`, k-medoids over the context columns by default). Each anchor's context is pinned in turn, yielding one optimal solution $x^\ast_k$ per anchor.
2. Solve the master for the current optimal solution(s) $x^\ast$.
3. **Separate**: resample training data localized near each $x^\ast$ to find the worst-case constraint model(s).
   - **basic** (non-contextual single LP with one learned constraint and no learned objective — synthetic only): `localized_bootstrap_separation` takes the independent worst-case (max) model per constraint, ranked by the *actual* embedded model type (not a CART proxy).
   - **coherent** (everything else, including *every* contextual gastric run): one **shared** localized-bootstrap relabeling ("scenario") retrains every constraint and the epigraph objective jointly, so the adversary is a single plausible relabeling rather than independent worst cases. Each iteration draws `cp_n_candidates` scenarios over a pool localized to the union of the anchors' neighborhoods and ranks them by **normalized average distance** (mean relative exceedance over all $(x^\ast,\text{outcome})$ cells, 0–1 scale).
4. If the worst distance exceeds the tolerance, add that scenario as a cut. Tolerance is resolved once, at iteration 0: given a relative knob `tau` (`cp_dist_tol_rel`) it is `tau * d0`, where $d_0$ is *this problem's own* iteration-0 worst distance — so one grid transfers across datasets whose noise scales differ by orders of magnitude. With no `tau` supplied it falls back to the absolute `methods.cp.dist_tol` (coherent) or `1e-6` (basic). In the coherent path a cut that would make a **currently-feasible** anchor infeasible is not simply dropped: under `cut_eviction: "evict_slack"` the stalest (most-slack) non-nominal cut is **evicted** to make room; if that still fails, the loop rolls the cut back and tries the next-worst candidate scenario.
5. Repeat 2–4 until the worst distance is within tolerance, no candidate scenario can be admitted without new infeasibility, or `max_iterations` is reached.
6. Re-solve the master at a tight MIP gap (`1e-4`) and return $x^\ast$. (Basic path only: if the loop exits without converging, the best feasible iterate seen is returned instead of the over-cut final master's solve.)

**Knob: `tau` only.** CP is *single-lever* — the coverage side is a fixed
**rule**, not a tunable: a cut may never make a currently-feasible anchor
infeasible, while anchors already infeasible under the nominal fit are
tolerated. The budget is recomputed each iteration from that baseline
(`feas_alpha = n_infeas_base / n_anchors` in `_CoherentSeparation.step`), so the
`conservativeness_sweep.cp_alpha_max` value is **never read** — the gastric runner
hard-codes `cp_alpha=0.0` and `_CoherentSeparation.self.alpha` is stored but
unused. Under the legacy alpha calibration CP was *exempt* (that rule already
keeps the training set feasible); under CV calibration its `tau` **is** selected
like every other method's knob.

Note that `tau` is **not** a `methods.cp:` key in `config.yaml`. It reaches
`solve_cp` as the `cp_dist_tol_rel` argument, and is supplied by
`cv_calibration.knob_grids.cp` (CV calibration / centered Pareto) or by
`conservativeness_sweep.cp_dist_tol_rel_{min,max}` (fixed-strength sweep).
`methods.cp.dist_tol` is the absolute fallback used when neither path supplies a
`tau`.

Further CP knobs (all under `cp:` in `config.yaml`): **anchors**
(`anchor_source` train/test, `n_anchors`, `anchor_method`) select where $x^\ast$
are collected; **localization** (`distance` full/context/auto) picks the
bootstrap pool; `robustify_objective` toggles epigraph objective
robustification; `eval_mode` (`global` vs `per_anchor_nearest`) chooses one
shared master vs. per-anchor masters. Pool size comes from
`uncertainty.cp_k_neighbors_frac` / `cp_k_neighbors_min`, candidate count from
`uncertainty.cp_n_candidates`.

### Robust Parameter (cross-cutting `rho`, in `src/models/embed.py`)

Not a separate training procedure — a deterministic, decision-side
robustification applied *inside the MIP embedding* of any of the above
methods (their `rho` argument, default `0`, no data resampling involved). It
shrinks each tree/leaf's effective split thresholds by a margin governed by
`rho` (e.g. `lb/(1-rho)`, `ub/(1+rho)`), so learned splits are penalized
against threshold uncertainty. It only bites on **tree-based** embeddings (CART,
RF, GBM, XGB — it flows through `_embed_leaves`); for linear/SVM/MLP constraints
`rho` is silently a no-op. It is also run **on its own** as a fifth
baseline ("`robust_param`" = nominal training + `rho > 0` embedding) for
comparison. Knob: `rho`; calibrated only via the legacy alpha grid (not
currently in the CV knob grids, since a poorly-chosen `rho` can make the
embedded MIP infeasible).

## Calibration

Each method exposes exactly one monotone robustness knob
(`robust_reg`→`label_eps`, `wrapper`→`alpha`, `robust_param`→`rho`, `cp`→`tau`);
nominal has no knob. **Which methods get calibrated depends on the method:**
under the legacy alpha path CP is exempt (`CALIBRATED_METHODS` omits it — its
coverage rule already keeps the training set feasible), but under CV
calibration CP's `tau` is in `cv_calibration.knob_grids` and is selected like
any other. `config.yaml`'s `calibration.method` selects how the knob is picked:

- **`"cv"` (default) — held-out robustness-parameter CV** (`src/methods/cv_calibrate.py`, run via `--calibrate-cv`). For each knob in a grid, fit on fold-train, score `(feasibility, objective)` on fold-val against a **train-only oracle** (a GT-ensemble proxy for gastric, a proxy model for synthetic — never the final analytic/GT truth), averaged over folds. Folds are temporal forward-chaining for gastric (`fold_cutoffs`, train = years ≤ cutoff) or KFold for synthetic. Picks $\theta^\ast$ = the knob with max feasibility subject to the objective staying within `os_tolerance_frac` of nominal. Writes `results/cv/*_robustness_knobs.json` (+ a resumable scores checkpoint).
- **`"alpha"` (legacy) — training-infeasible-fraction heuristic** (`src/methods/calibrate.py::calibrate_strength`). Grid-scans strength from strongest to weakest and picks the strongest setting whose *training-set* infeasible fraction is $\le$ `uncertainty.alpha`. This is the **only** consumer of `uncertainty.alpha`: CP's own coverage rule is analogous in spirit but pinned at zero new infeasibility, and does not read the config value.

**Calibration picks a point; it does not choose a trade-off.** Both rules
*maximize a robustness quantity subject to a constraint* — alpha subject to
in-sample solvability, CV subject to an objective budget. In particular
`select_knob_cv` sorts by `(feasibility, grid_index)` — the objective appears
only in the filter that decides which knobs are eligible, never in the ranking.
Since the grids are ordered *strongest last*, ties therefore break toward the
**more robustified** knob, with no check on what that robustification costs:
the rule cannot distinguish a free robustness gain from an expensive one.
(It does *not* follow that the stronger knob has worse OS — in the gastric CV
scores the relationship is non-monotone, e.g. CP's $\tau{=}0.1$ beats
$\tau{=}0.75$ on *both* feasibility and OS.) Neither rule
asks whether a feasibility gain was *worth* its OS cost, so a calibrated point
alone cannot separate a genuinely more robust method from a merely more
conservative one. That is what the **Pareto sweep** is for: it traces each
method's feasibility–objective frontier (centered on $\theta^\ast$ via
`--pareto-center-cv`), enabling comparison by *frontier dominance* and a check
that $\theta^\ast$ sits somewhere sensible on its own curve.

This runs as **stage 1** of the gastric pipeline; stage 2 (the CV-calibrated
headline table + CV-centered Pareto sweep) consumes its output. See
`experiments/submit_pipeline.sh` for the chained SLURM version.

### Distribution-free formalization (design notes, not implemented)

The coverage rule (as implemented: admit no cut that newly makes a feasible
anchor infeasible; in general, keep $\ge (1-\alpha)$ of patients feasible) is the
lightweight, implemented version of a broader idea: a **distribution-free**
constraint that assumes no parametric label-noise model and relies only on
the data. The orthogonal *per-point* robustness (worst-case over the
localized ensemble) could likewise be replaced by a distribution-free
predictive bound:

- **Split-conformal upper bound.** With a held-out calibration split, $\hat f(x^k) + Q_{1-\alpha}(\text{residuals}) \le b$ gives finite-sample coverage with no distributional assumption; a localized (Mondrian) variant uses calibration residuals from context-neighbors of $x^k$.
- **Jackknife+ / CV+.** Distribution-free predictive bands without a separate calibration split — useful for the scarce gastric training split ($n{=}320$).
- **Wasserstein DRO.** Worst-case over a data-radius ball around the empirical distribution, radius calibrated by CV; principled but heavier to embed.

## Evaluation

Every method's final score comes from an evaluator that is **kept separate
from the embedded constraint models** — never grade a method against the same
(possibly noisy-labeled) model it just trained, since that would let an
overfit model mark its own homework. What that fixed evaluator *is*, and how
"robustness" gets measured on top of it, differs between the two problems for
a structural reason: gastric's data is a fixed, finite real cohort that can
never be regenerated, while synthetic's data-generating process is fully
known and can be resampled at will.

### Gastric: a fixed GT ensemble + resampling the real cohort

**The oracle.** All reported given/prescribed outcomes (`src/evaluation/chemo_metrics.py`)
use a **ground-truth (GT) ensemble** — six sklearn model types per outcome
(Table EC.12 of the paper) — fit **once, on the full clean cohort**. This GT
ensemble is a completely different object from the embedded constraint
models used inside the MIP (which may be a different model type, e.g. a
single CART, per Table EC.10), and it never changes across robustness
realizations below — it plays the role of "the real answer," to whatever
extent a second, independently-fit ensemble is trustworthy given there is no
analytic ground truth for real patients.

**The headline comparison.** `evaluate_given_table6` / `evaluate_prescribed_table6`
compare the GT ensemble's prediction on the *observed* (given) historical
treatment against its prediction on the *optimizer's prescribed* decision,
over a shared **samestore** cohort — test rows that are feasible under every
constraint mode being compared (`samestore_eval_mask`), so different methods
and constraint modes are judged on exactly the same patients.

**Why a single training draw is not enough — it is biased toward nominal.**
The GT ensemble is fit on `X_valid`, the full cohort ([generate.py](src/data/generate.py) step 11),
while the constraint models are fit on `X_fit`, the training rows — a *subset*
of the oracle's own fitting data. So on a single draw, "does the prescription
satisfy the GT constraint" largely measures how closely the embedded model
agrees with a model fit on those same rows, which the **nominal** fit maximizes
by construction. Worse, there is only one labeling in play, so there is nothing
for a robust method to hedge *against*: every deliberate hedge is a deviation
from the fit the oracle rewards, and can only look worse. Robustness is
invisible — not absent — under single-draw evaluation.

**The robustness probe: outer m-out-of-n subsampling.** This supplies the
missing variation. To ask "how sensitive
is this method to *which* patients happened to be in the training data,"
`run_chemo_robust_realizations` repeats the whole pipeline $R$ times
(`--n-realizations`), each time fitting the **embedded/robust constraint
models only** on an m-out-of-n subsample of the training rows, drawn
**without replacement** (`train_subsample_frac` / `subsample_seed`,
`--subsample-frac` / `--frac-grid`) — the GT ensemble oracle is refit once on
the full cohort and held fixed throughout. Realizations sharing an index $r$
use the **same subsample seed** regardless of the `rhs`/`frac` grid cell being
swept (common random numbers), so every cell in a sweep is compared against
the *same* set of training draws rather than independent noise. `aggregate_realizations`
then reports, per method/constraint-mode/outcome, the distribution across
realizations: mean, SD, **worst-case** (min), and a low quantile (default
10th percentile, a CVaR-style tail stat) — a genuinely robust method should
show a *higher worst-case* and *lower SD* on the joint feasibility row than
nominal, even when the means look similar.

This is a **different kind of uncertainty than the inner bootstrap** the
methods themselves use to build robustness (Wrapper/CP resample *inside* one
fixed training set to approximate $\Theta^\ast$): the outer probe instead asks
what happens if the *training draw itself* had been different — i.e., it's
evaluating whether a method's robustness generalizes across plausible
alternate cohorts, not just within the one cohort it saw.

### Synthetic: the exact analytic truth + fresh noise draws

**The oracle.** Synthetic's constraint is generated from a known nonlinear
function, so the ground truth (`instance.gt_constraints`) is exact — no
fitting, no approximation, no separate ensemble needed. `evaluate_all`
(`src/evaluation/metrics.py`) checks the optimizer's $x^\ast$ against this
analytic function directly.

**The robustness probe: regenerate, don't resample.** Because the data-generating
process is known, "what if the training draw had been different" doesn't
need subsampling a fixed cohort — it's answered by literally drawing a fresh
noisy dataset. `run_synthetic_centered_pareto` (`--pareto`) retrains each
method from scratch on `n_real` independently-reseeded synthetic instances
(same configured `noise_std`; the seed redraws both the design points $X$ and
the label noise) at a given knob, and scores the resulting $x^\ast$ against the
fixed analytic truth for feasibility and against the (constant) cost vector for
the objective; `worst_case_feas` is the min feasibility rate across those draws. The
separate noise **sweep** (`--sweep noise`) instead varies `noise_std` itself,
showing how each method degrades as the *label noise level* grows rather than
as the *training draw* varies at fixed noise.

**Calibration still holds out the truth.** Even though the analytic truth is
available, CV calibration (`--calibrate-cv`) deliberately does *not* use it —
it scores knobs against a train-only proxy oracle (`SyntheticOracle`, a model
fit on the training labels), exactly mirroring gastric's train-only-GT-ensemble
discipline, so knob selection never peeks at the ground truth it will later be
graded against. The analytic truth is reserved for the final Pareto/sweep
evaluation only.

| | Gastric | Synthetic |
|---|---|---|
| Ground truth | GT ensemble (6 models/outcome), fit once on the full real cohort | Exact analytic function (`gt_constraints`) |
| "Different training draw" | m-out-of-n subsample of the *same* fixed cohort, without replacement | Fresh noisy dataset regenerated from the known generator |
| Held fixed across realizations | GT ensemble | The analytic truth (always fixed; nothing to refit) |
| Calibration oracle | Train-only GT-ensemble proxy (never the final GT ensemble) | Train-only proxy model (never the analytic truth) |

## Setup

```bash
uv sync                 # installs dependencies into .venv from pyproject.toml / uv.lock
```

Requires a Gurobi license (free academic license available) — all MIP
solving goes through `gurobipy`. Prefix Python invocations with `uv run`.

## Gastric Cancer Chemotherapy Experiment

Compares all methods on the OptiCL gastric cancer case study (Table 6 metrics).

```bash
# Model-type/hyperparameter CV (run once; selects constraint-model + GT-ensemble configs)
uv run python experiments/run_cv.py --ensemble

# Stage 1: robustness-parameter CV (selects each method's knob)
uv run python experiments/run_chemo_robust.py --calibrate-cv

# Stage 2: CV-calibrated Table 6 comparison
uv run python experiments/run_chemo_robust.py --quick   # smoke run: 5 cohorts, 4 methods, all_constraints only
uv run python experiments/run_chemo_robust.py           # full run (long; use SLURM)

# CV-centered Pareto sweep (worst-case feasibility vs. OS, centered on each theta*)
uv run python experiments/run_chemo_robust.py --pareto-center-cv

# Fixed-threshold sweeps (rhs / data-scarcity / conservativeness axes)
uv run python experiments/run_chemo_robust.py --rhs-grid 0.3 0.4 0.5 0.6
uv run python experiments/run_chemo_robust.py --frac-grid 0.3 0.5 0.7 0.9
uv run python experiments/run_chemo_robust.py --conservativeness-grid 0 0.25 0.5 0.75 1

# OptiCL replication baseline only (RF tree-violation wrapper, alpha=0.25)
uv run python experiments/run_chemo_replication.py

# Post-processing
uv run python experiments/summarize_table6.py       # Table 6 CSV -> presentation .csv/.tex
uv run python experiments/make_paper_figures.py     # fig_headline/fig_tradeoff/fig_rhs_frontier/...

# SLURM
sbatch experiments/submit_cv_calibrate.sh                 # stage 1 alone
bash experiments/submit_pipeline.sh                       # stage 1 -> stage 2 (afterok dependency)
sbatch experiments/submit_chemo_robust.sh                 # full run, no CV chaining
```

Uncertainty is **data-driven**: bootstrap resamples of observed training
labels (no parametric label noise). Key config: `uncertainty.n_bootstrap`,
`cp_k_neighbors_frac`, `cp_k_neighbors_min`, `cp_n_candidates`,
`cv_calibration.*` in `config.yaml`.

## Synthetic Experiment

A synthetic nonlinear problem (see "Concrete formulations" above): the true
constraint is $f_\text{true}(x)\le 0.5\,d$ (so $\le 1.0$ at the default
$d = 2$; the RHS is derived from `data.n_features` inside
`synthetic_nonlinear`, **not** read from `optimization.constraint_rhs`), with
decision variables bounded in $[0,1]^d$. Label noise is injected at
dataset-generation time via a configurable standard deviation $\sigma$
(`data.noise_std`), so — unlike gastric — the analytic ground truth is known and
can be used to validate the final prescriptions (CV calibration on synthetic
still uses a train-only proxy oracle, matching the gastric CV path; the analytic
truth is reserved for final evaluation only).

```bash
# Single run, all methods
uv run python experiments/run_all.py

# Robustness-parameter CV (KFold + proxy oracle) then CV-centered Pareto
uv run python experiments/run_sweep.py --calibrate-cv
uv run python experiments/run_sweep.py --pareto

# Label-noise sigma sweep (degradation as sigma grows)
uv run python experiments/run_sweep.py --sweep noise
uv run python experiments/run_sweep.py --sweep noise --plot-only   # replot existing results

# Legacy Gamma sweep — see the caveat below
uv run python experiments/run_sweep.py --sweep gamma
uv run python experiments/run_sweep.py --sweep all    # gamma + noise
```

> **`--sweep gamma` is currently inert.** `run_gamma_sweep` sets
> `config["uncertainty"]["gamma"]`, but no solver reads that key — the methods'
> uncertainty is data-driven (bootstrap resamples), with no scalar budget
> $\Gamma$. Every value on the grid therefore re-runs the *same* experiment, and
> `sweep_results.csv` / `gamma_sweep.png` differ only by solver noise. It is kept
> as a stub from the earlier budgeted-uncertainty formulation; use
> `--sweep noise` (noise level) or `--pareto` (training draw at fixed noise) for
> real robustness axes.

## Configuration

Edit `config.yaml` to change:
- **Data**: `data.type` (`synthetic` / `gastric_cancer`), `n_train`, `n_features`, `noise_std`.
- **Model**: type (`cart`/`rf`/`xgb`/...), hyperparameters. Cross-validated selections are read from `results/cv/*_selected_configs.json` / `*_gt_ensemble_configs.json` when present (`--cv-configs`).
- **Uncertainty**: `uncertainty.alpha` (coverage cap for the *legacy* alpha calibration of the baselines — CP does not read it, see the CP section), `n_bootstrap`, `cp_k_neighbors_frac`/`cp_k_neighbors_min`/`cp_n_candidates`.
- **Calibration**: `calibration.method` (`"cv"` default, `"alpha"` legacy) and their respective bounds/grids; `cv_calibration.knob_grids` per method (`cp` in `tau` units, `robust_reg` in `label_eps`, `wrapper` in chance `alpha`).
- **Method-specific**: `methods.robust_param.rho`, `methods.robust_reg.{label_eps,budget_frac,K}`, `methods.wrapper.{n_estimators,alpha}`, `methods.cp.*` (`max_iterations`, `anchor_source`, `n_anchors`, `anchor_method`, `distance`, `dist_tol`, `robustify_objective`, `eval_mode`, `nearest_distance`, `cut_eviction`). `solve_cp` auto-selects basic vs. coherent separation from the problem shape — there is no separation flag.
- **Gastric-specific**: `methods.chemo.methods_to_run` / `constraint_modes`, overridden by `methods.chemo.quick` for `--quick`.
- **Not read by any code path**: `optimization.constraint_rhs` / `optimization.variable_bounds` (the synthetic instance derives both internally) and `conservativeness_sweep.cp_alpha_max` (retained for reference only).

## Project Structure

```
robust-cl/
├── config.yaml                    # all experiment parameters
├── pyproject.toml / uv.lock        # dependencies (uv-managed)
├── src/
│   ├── data/
│   │   ├── generate.py             # ProblemInstance; synthetic_nonlinear() / gastric_cancer() builders
│   │   ├── gastric_v11.py          # gastric cohort processing (imputation, ECOG/KPS, v11 alignment)
│   │   └── gastric_model_specs.py  # embedded-constraint + 6-model GT-ensemble hyperparameters (paper Tables EC.10/EC.12)
│   ├── models/
│   │   ├── train.py                # train/retrain models; bootstrap-sample helpers
│   │   └── embed.py                # MIO embedding (trees/ensembles/MLP/pipelines) + rho margin tightening
│   ├── methods/
│   │   ├── nominal.py              # shared MIP scaffolding + plain constraint learning
│   │   ├── robust_regression.py    # label-robust training counterpart
│   │   ├── wrapper.py              # Maragno et al. ensemble chance constraint + OptiCL tree-violation variant
│   │   ├── cp.py                   # Cutting Planes (basic + coherent separation) — the contribution
│   │   ├── calibrate.py            # legacy alpha-target knob calibration
│   │   └── cv_calibrate.py         # CV-based robustness-knob calibration (current default)
│   ├── evaluation/
│   │   ├── metrics.py              # synthetic feasibility/violation metrics
│   │   └── chemo_metrics.py        # Table 6 given-vs-prescribed metrics (GT ensemble)
│   └── utils/
│       ├── perturbations.py        # label perturbation sampling
│       └── trust_region.py         # convex-hull trust region (gastric)
├── experiments/
│   ├── run_all.py                  # single synthetic run, all methods
│   ├── run_sweep.py                # synthetic sweeps + CV calibration + CV-centered Pareto
│   ├── run_chemo_robust.py         # gastric Table 6 comparison + CV calibration + sweeps
│   ├── run_chemo_replication.py    # OptiCL Table 6 replication baseline only
│   ├── run_cv.py                   # model-type/hyperparameter CV (constraint models + GT ensemble)
│   ├── summarize_table6.py         # Table 6 CSV -> presentation-ready .csv/.tex
│   ├── make_paper_figures.py       # gastric + synthetic paper figures
│   ├── plot_results.py             # basic synthetic bar plot + CP convergence from cp_trace.csv
│   └── submit_*.sh                 # SLURM job scripts (see below)
└── results/                        # CSV/figure outputs, see "Expected Outputs"
```

**SLURM scripts**: `submit_cv_calibrate.sh` (stage 1: robustness-knob CV) →
`submit_pipeline.sh` (chains stage 1 into stage 2 via `afterok`) →
`submit_cp_final.sh` (stage 2 array: headline + frontiers + CV-centered
Pareto); `submit_chemo_robust.sh` (full run, no CV chaining);
`submit_cp_confirm_ablation.sh` (CP ablation confirmation sweep);
`submit_chemo.sh` (legacy OptiCL replication run).

## Expected Outputs

| Location | Contents |
|----------|----------|
| `results/synthetic/` | `results.csv` (single run), `noise_sweep_results.csv` + `noise_sweep.png` (noise sweep), `synthetic_pareto.csv` (CV-centered Pareto), `comparison.png` (`plot_results.py`); `sweep_results.csv` / `gamma_sweep.png` only from the inert Gamma stub |
| `results/gastric/` | `chemo_robust_table6*.csv` (Table 6), `chemo_robust_realizations_*.csv` / `chemo_robust_robustness_summary_*.csv` (sweeps), `summary_*.csv` / `.tex` (`summarize_table6.py`), `cp_trace.csv` (`iteration, max_violation, objective`), `ccg_convergence.png` (`plot_results.py`), `prescriptions/*.csv` |
| `results/cv/` | `*_selected_configs.json` / `*_gt_ensemble_configs.json` (model-type CV), `*_robustness_knobs.json` / `*_robustness_cv_scores.csv` (robustness-parameter CV), SHAP plots |
| `results/figures/` | `fig_headline`, `fig_tradeoff`, `fig_rhs_frontier`, `fig_frac_frontier`, `fig_pareto` (gastric); `fig_synthetic`, `fig_synthetic_pareto` (synthetic) — PDF + PNG |

## Notes / Gotchas

- **Ground truth for evaluation is fixed and separate** from the embedded models. The gastric GT ensemble is refit on the full clean cohort; only constraint/fit rows are resampled in robustness realizations. Don't conflate the embedded constraint model with the GT evaluator.
- Robustness (gastric) is measured by **outer m-out-of-n subsampling without replacement** of training rows, with the GT ensemble as a fixed oracle — uncertainty over *training draws*, distinct from the inner bootstrap the methods use to build their own robustness.
- `trust_region.py` constrains gastric decisions to the convex hull of observed treatment vectors.
