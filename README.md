# Robust Constraint Learning via Iterative Scenario Generation

## Overview

This repository implements and compares approaches for making
constraint learning (Maragno et al., 2023/2025) robust to label
uncertainty. When ML models are embedded as constraints in
optimization problems, noisy training labels can lead to solutions
that violate the true constraints.

We formulate this as a trilevel optimization problem and solve it
via Cutting Planes.

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

We iteratively approximate the reachability set of models $\Theta^\ast$ through **Cutting Planes**. On each step, a separation oracle searches over the data perturbations (e.g., via targeted removal/addition of influential training points or adversarial continuous label shifts) to find the worst-case constraints for the current $x^k$.

## Problem Settings

The scenarios differ only in the mapping between models, constraint terms, and LPs. Let $\hat{f}_m(\cdot;\theta_m)$ index trained models, $q \in \mathcal{Q}$ index the LPs solved at evaluation (with context $z_q$, decisions $x_q$), and each learned constraint take the form

```math
\sum_{(m,k) \in \mathcal{T}_c} w_{c,m,k}\,\hat{f}_m\bigl(a_{c,k}(x_q,z_q);\theta_m\bigr) \leq b_c
```

where $\mathcal{T}_c$ is the set of model occurrences in constraint $c$. The robust oracle separates $\max_{\theta_m \in \Theta_m^\ast}(\cdot)$ at the incumbent; its granularity is set by $Q_m$, the query points where model $m$ is evaluated, and whether they share $\theta_m$.

| Axis | Synthetic | Gastric cancer | Marketing |
|------|-----------|----------------|-----------|
| Datasets → models | 1 → 1 | $M$ → $M$ (one per constraint) | 1 → 1 |
| Model occurrences per constraint | 1 | 1 | many (summed) |
| Learned constraints per LP | 1 | many | 1 (+ model's own) |
| Decision vars | per-LP | shared across models | shared; multiple input copies |
| LPs at evaluation $\lVert\mathcal{Q}\rVert$ | 1 | many (one per cohort) | 1 |
| Query points $Q_m$ | one point | one per LP (independent) | many in one constraint (shared $\theta$) |
| Separation | single point-localized cut | **replicated** per LP (parallel) | **coupled** over the point set |

**Two roles of context.** *Parametric* context (gastric) indexes separate LPs: each $z_q$ yields its own solution ${x_q}^{*}(z_q)$ and an independent, point-localized oracle — granularity scales *out* (more solutions, parallel cuts). *In-LP* context (marketing) places several context points in one constraint sharing a single $\theta$: there is one solution, but the oracle must choose one worst-case model maximizing the *aggregate* over $Q_m$ — granularity *couples* (one harder cut over a point set, not term-by-term). Synthetic is the degenerate case of both ($\lVert Q_m \rVert = 1$).

## Experimental Setup & Methods Compared

Following our latest experimental design, we employ data-driven uncertainty calibration (cross-validation and bootstrap variation) instead of static preset perturbation wrappers. We compare four main approaches:

| Method | Calibration | Robustness Mechanism | Hyperparameter Tuning |
|--------|-------------|----------------------|-----------------------|
| **Nominal**| None | None | --- |
| **Robust Regression**| CV / Bootstrap | Train one model robust to data noise, then embed | CV predictive accuracy |
| **Wrapper** | Bootstrap | Maragno et al. ensemble chance constraint ($\alpha$ viol.) | CV feasibility + cost |
| **Cutting Planes (Ours)**| Bootstrap | Localized bootstrap separation at $x^k$ | CV feasibility + cost |

**Separation Oracle:** 
The separation oracle uses **localized bootstrap resampling**: at each iteration, training arms nearest to the current prescription $x^k$ are resampled to find worst-case constraint models. Wrapper and robust regression share a fixed set of $P$ bootstrap resamples; CP generates fresh localized candidates each iteration.

### Unified driver with an auto-selected separation strategy

A single driver, `solve_cp` (`src/methods/cp.py`), covers every setting. The loop is identical — *train nominal → build master → solve for the optimal solution(s) $x^\*$ → separate → add cuts → terminate* — over shared scaffolding (`_build_master_with_nominal`, `_setup_anchors`, `_solve_all_anchors`, `_resolve_distance`, `_finalize`); each strategy implements just one iteration via a small `step()` contract. The strategy is **chosen automatically from the problem shape** — no separation flag — based on the number of learned constraints and the number of optimal solutions $x^\*$ (one global $x^\*$ for non-contextual problems; one per context anchor for parametric-context problems like gastric):

- **basic** — *single LP, single learned constraint* (synthetic). Plain **worst-case (max)** separation over the localized bootstrap ensemble at $x^\*$, ranked by the *actual* constraint model (not a CART proxy, so it is correct for non-tree constraints). Cut whatever violates; stop when nothing does.
- **coherent** — *multiple constraints, multiple $x^\*$, and/or a learned objective* (gastric). A *scenario* is one **shared** localized bootstrap relabeling used to train every constraint (and the epigraph objective) jointly, so the adversary is a single plausible relabeling rather than an independent worst case per constraint. Each iteration draws $B$ scenarios from a pool localized to the union of the current $x^\*$ neighborhoods, trains one model per constraint per scenario, and cuts the single worst scenario. The worst scenario is the one with the largest **normalized average distance**: each constraint exceedance is divided by $\max(1,|\text{rhs}|)$ and each objective exceedance by $\max(1,|t\_\text{val}|)$, then averaged over all $(x^\*, \text{outcome})$ cells (so the metric is on a 0–1 scale regardless of the number of constraints or patients). We **stop** when either (1) the worst scenario's normalized average distance is $\le$ `cp.dist_tol` (some residual violation is tolerated), or (2) no sampled scenario can be cut without pushing the fraction of infeasible $x^\*$ (`p_infeas`) above `cp_alpha` (the **coverage cap**).
  - **`cp_alpha`** only caps feasibility (multiple $x^\*$): adding a scenario's constraint cuts is rolled back if it pushes `p_infeas` above $\alpha$, falling back to the next-worst affordable scenario. It never affects ranking or the objective (objective cuts only raise the epigraph $t$ and never reduce feasibility).
  - **Single $x^\*$** (non-contextual, multiple constraints): `p_infeas` is degenerate $(0/1)$ so there is no coverage cap; we cut the worst scenario by normalized average distance each iteration and stop once it falls within `cp.dist_tol` (or the master goes infeasible).

Two further knobs are shared:

- **Anchor set (where we collect $x^\*$).** For the *parametric-context* gastric case the master is solved once per representative context anchor, giving $x^\*_1,\dots,x^\*_K$. Anchors are chosen by `select_anchor_contexts` (k-medoids over the context columns by default; `sample`/`all` available) from either the training or test contexts (`cp_anchor_source`). Training anchors give an **offline** robust region (no test labels, precomputable); each cut is a full embedded model valid at every context, so it remains sound when `evaluate_prescribed_table6` re-solves per test cohort.
- **Localization distance (which pool).** `cp_distance` defaults to **`"full"`** for all scenarios (context + decision distance, so the pool follows $x^\*$). `"context"` localizes on the context columns only (models trained on arms for *similar patients*), and `"auto"` uses context-only when the problem is contextual, full otherwise. Training points are only the bootstrap *pool*, never feasibility targets.
- **Objective robustification.** `cp.robustify_objective` (default `true`) uses the epigraph reformulation and worst-case objective cuts in coherent separation. Set `false` to embed OS nominally (constraints only are robustified).
- **Per-anchor evaluation mode.** `cp.eval_mode: "per_anchor_nearest"` trains one CP master per training anchor (each with single-$x^*$ coherent separation, so no coverage-cap deadlock). At prescribe time, `evaluate_prescribed_table6` picks the nearest training anchor (by `cp.nearest_distance`, default context-only) and re-solves that anchor's MIP. Cost is $K\times$ the single-anchor CP train time. Default `eval_mode: "global"` keeps one shared master with global cuts.

The *marketing* (in-LP context, **coupled**) setting — one constraint summing a shared $\theta$ over many in-LP context points, $\sum_q f(a(x,z_q)) \le b$ — would add a third strategy (worst-case model over the aggregate); it is not implemented pending a marketing data loader.

### Distribution-free formalization

The coverage cap (keep $\ge (1-\alpha)$ of patients feasible) is the lightweight, implemented version of a broader idea: a **distribution-free** constraint that assumes no parametric label-noise model and relies only on the data. The orthogonal *per-point* robustness (worst-case over the localized ensemble) can likewise be replaced by a distribution-free predictive bound. Theoretically grounded alternatives:

- **Split-conformal upper bound.** With a held-out calibration split, $\hat f(x^k) + Q_{1-\alpha}(\text{residuals}) \le b$ gives finite-sample coverage with no distributional assumption; a localized (Mondrian) variant uses calibration residuals from context-neighbors of $x^k$.
- **Jackknife+ / CV+.** Distribution-free predictive bands without a separate calibration split — useful for the scarce gastric training split ($n{=}320$).
- **Wasserstein DRO.** Worst-case over a data-radius ball around the empirical distribution, radius calibrated by CV; principled but heavier to embed.

## Gastric Cancer Chemotherapy Experiment

Compare all robust methods on the OptiCL gastric cancer case study (Table 6 metrics):

```bash
# Local smoke run (5 test cohorts, 3 methods)
python experiments/run_chemo_robust.py --quick

# Full comparison (cluster or long local run)
python experiments/run_chemo_robust.py

# OptiCL replication baseline only
python experiments/run_chemo_replication.py

# SLURM (12h, full run)
sbatch experiments/submit_chemo_robust.sh
```

Uncertainty is **data-driven**: bootstrap resamples of observed training labels (no parametric label noise). Config: `uncertainty.n_bootstrap`, `cp_k_neighbors_frac`, `cp_k_neighbors_min`, `cp_n_candidates` in `config.yaml`.

## Synthetic Experiment

The current codebase includes a synthetic nonlinear experiment. We generate a dataset using an underlying nonlinear function where $y$ values simulate constraints $f(x) \leq 0.5$. The variables $x$ are bounded within $[0, 1]$. Label noise is injected during dataset generation based on a configurable noise standard deviation $\sigma$. The models trained are constraint learning classifiers.
The experiments compare Nominal constraint learning, Robust Regression, Wrapper (ensemble chance constraints), and our robust Cutting Planes approach. We evaluate the performance under different constraint violation vulnerability criteria.

## Setup

```bash
pip install -r requirements.txt
```

Requires a Gurobi license (free academic license available).

## Running Experiments
Currently just using run_all.py

### Single experiment (fixed parameters)

```bash
python experiments/run_all.py
```

### Sweep over uncertainty budget Gamma

```bash
python experiments/run_sweep.py --sweep gamma
```

### Sweep over label noise level sigma

```bash
python experiments/run_sweep.py --sweep noise
```

### All sweeps

```bash
python experiments/run_sweep.py --sweep all
```

### Plot from existing results

```bash
python experiments/run_sweep.py --sweep all --plot-only
```

## Configuration

Edit `config.yaml` to change:
- Data: number of training points, features, noise level
- Model: type (cart/rf/xgb), hyperparameters
- Uncertainty: bootstrap resamples (`n_bootstrap`, `cp_k_neighbors_frac`, `cp_k_neighbors_min`, `cp_alpha`)
- Method-specific: wrapper alpha/P, number of scenarios, Cutting Planes settings (`cp.anchor_source`, `cp.n_anchors`, `cp.anchor_method`, `cp.distance`, `cp.dist_tol`, `cp.robustify_objective`, `cp.eval_mode`, `cp.nearest_distance`). `solve_cp` auto-selects basic vs coherent separation from the problem shape; there is no separation flag.
- Evaluation: CV folds, Bootstrap resamples

## Project Structure

```
robust-cl/
├── config.yaml              # All experiment parameters
├── src/
│   ├── data/
│   │   └── generate.py      # Problem instance generation
│   ├── models/
│   │   ├── train.py          # Train / retrain ML models
│   │   └── embed.py          # MIO embedding of trees
│   ├── methods/
│   │   ├── nominal.py        # Standard constraint learning
│   │   ├── robust_regression.py
│   │   ├── wrapper.py        # Maragno et al. ensemble wrapper
│   │   └── cp.py             # Cutting Planes / Cutting Planes
│   ├── evaluation/
│   │   └── metrics.py        # Feasibility and robustness metrics
│   └── utils/
│       └── perturbations.py  # Label perturbation sampling
├── experiments/
│   ├── run_all.py            # Single experiment runner
│   ├── run_sweep.py          # Parameter sweeps
│   └── plot_results.py       # Basic plotting
└── results/                  # Output CSVs and plots
```

## Expected Outputs

After running all sweeps, the `results/` directory will contain:

| File | Description |
|------|-------------|
| `results.csv` | Single-run comparison of all baseline methods |
| `cp_trace.csv` | Cutting Planes iteration history (violation, objective) |
| `sweep_results.csv` | Results across Gamma values |
| `noise_sweep_results.csv` | Results across noise levels |
| `comparison.png` | Bar chart of single-run results |
| `cp_convergence.png` | Cutting Planes violation and objective per iteration |
| `gamma_sweep.png` | Price of robustness curves |
| `noise_sweep.png` | Degradation under increasing noise |

## Parameter Robustness
An alternative form of robustness is robustness on the parameters, implemented here via multiplicative uncertainty. By configuring the rho parameter, decision tree splits are robustified against coefficient uncertainty. This penalizes the nominal splits, shrinking the feasible leaf regions by a margin dependent on the feature threshold and rho. This is implemented automatically across all constraint learning methods.
