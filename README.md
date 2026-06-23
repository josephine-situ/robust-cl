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

Embeds a trained ML model $\hat{f}(x;\theta^*)$ as a constraint in an optimization problem:

$$
\min_{x \in \mathcal{X}} \; c^\top x \quad \text{s.t.} \quad \hat{f}(x;\theta^*) \leq b
$$

**The Vulnerability:** Labels $y$ are noisy ($y^* = y + \delta$). Different perturbations $\delta$ lead to different trained models $\theta^*(\delta)$ and vastly different "optimal" decisions. The optimizer frequently exploits the errors in the nominal model.

**Robust Constraint Learning (Trilevel Formulation):**

We seek a decision $x$ that remains feasible for *every* model resulting from a plausible label perturbation $\delta \in \mathcal{D}$.

$$
\begin{aligned}
\min_{x \in \mathcal{X}} \quad & c^\top x \\
\text{s.t.} \quad & \max_{\delta \in \mathcal{D}} f(x;\theta^*(\delta)) \leq b \\
\text{where} \quad & \theta^*(\delta) = \arg\min_{\theta \in \Theta_{\mathrm{feas}}} \mathcal{L}(\theta; X, y+\delta)
\end{aligned}
$$

We iteratively approximate the reachability set of models $\Theta^*$ through **Cutting Planes**. On each step, a separation oracle searches over the data perturbations (e.g., via targeted removal/addition of influential training points or adversarial continuous label shifts) to find the worst-case constraints for the current $x^k$.

## Problem Settings

The scenarios differ only in the mapping between models, constraint terms, and LPs. Let $\hat{f}_m(\cdot;\theta_m)$ index trained models, $q \in \mathcal{Q}$ index the LPs solved at evaluation (with context $z_q$, decisions $x_q$), and each learned constraint take the form

$$
\sum_{(m,k) \in \mathcal{T}_c} w_{c,m,k}\,\hat{f}_m\!\big(a_{c,k}(x_q,z_q);\theta_m\big) \leq b_c
$$

where $\mathcal{T}_c$ is the set of model *occurrences* in constraint $c$. The robust oracle separates $\max_{\theta_m \in \Theta_m^*}(\cdot)$ at the incumbent; its granularity is set by $Q_m$, the query points where model $m$ is evaluated, and whether they share $\theta_m$.

| Axis | Synthetic | Gastric cancer | Marketing |
|------|-----------|----------------|-----------|
| Datasets → models | 1 → 1 | $M$ → $M$ (one per constraint) | 1 → 1 |
| Model occurrences per constraint | 1 | 1 | many (summed) |
| Learned constraints per LP | 1 | many | 1 (+ model's own) |
| Decision vars | per-LP | shared across models | shared; multiple input copies |
| LPs at evaluation $\|\mathcal{Q}\|$ | 1 | many (one per cohort) | 1 |
| Query points $Q_m$ | one point | one per LP (independent) | many in one constraint (shared $\theta$) |
| Separation | single point-localized cut | **replicated** per LP (parallel) | **coupled** over the point set |

**Two roles of context.** *Parametric* context (gastric) indexes separate LPs: each $z_q$ yields its own solution $x_q^*(z_q)$ and an independent, point-localized oracle — granularity scales *out* (more solutions, parallel cuts). *In-LP* context (marketing) places several context points in one constraint sharing a single $\theta$: there is one solution, but the oracle must choose one worst-case model maximizing the *aggregate* over $Q_m$ — granularity *couples* (one harder cut over a point set, not term-by-term). Synthetic is the degenerate case of both ($\|Q_m\| = 1$).

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

Uncertainty is **data-driven**: bootstrap resamples of observed training labels (no parametric label noise). Config: `uncertainty.n_bootstrap`, `cp_k_neighbors_frac`, `cp_n_candidates` in `config.yaml`.

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
- Uncertainty: bootstrap resamples (`n_bootstrap`, `cp_k_neighbors_frac`)
- Method-specific: wrapper alpha/P, number of scenarios, Cutting Planes settings
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
