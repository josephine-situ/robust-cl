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
$$ \min_{x \in \mathcal{X}} \; c^\top x \quad \text{s.t.} \quad \hat{f}(x;\theta^*) \leq b $$

**The Vulnerability:** Labels $y$ are noisy ($y^* = y + \delta$). Different perturbations $\delta$ lead to different trained models $\theta^*(\delta)$ and vastly different "optimal" decisions. The optimizer frequently exploits the errors in the nominal model.

**Robust Constraint Learning (Trilevel Formulation):**
We seek a decision $x$ that remains feasible for *every* model resulting from a plausible label perturbation $\delta \in \mathcal{D}$.
$$ \min_{x \in \mathcal{X}} \quad c^\top x $$
$$ \text{s.t.} \quad \max_{\delta \in \mathcal{D}} \quad f(x;\,\theta^*(\delta)) \leq b $$
$$ \text{where} \quad \theta^*(\delta) = \arg\min_{\theta \in \Theta_{\mathrm{feas}}} \quad \mathcal{L}(\theta;\, X,\, y+\delta) $$

We iteratively approximate the reachability set of models $\Theta^*$ through **Cutting Planes**. On each step, a separation oracle searches over the data perturbations (e.g., via targeted removal/addition of influential training points or adversarial continuous label shifts) to find the worst-case constraints for the current $x^k$.

## Experimental Setup & Methods Compared

Following our latest experimental design, we employ data-driven uncertainty calibration (cross-validation and bootstrap variation) instead of static preset perturbation wrappers. We compare four main approaches:

| Method | Calibration | Robustness Mechanism | Hyperparameter Tuning |
|--------|-------------|----------------------|-----------------------|
| **Nominal**| None | None | --- |
| **Robust Classification**| CV / Bootstrap | Train one model robust to data noise, then embed | CV predictive accuracy |
| **Wrapper** | Bootstrap | Maragno et al. ensemble chance constraint ($\alpha$ viol.) | CV feasibility + cost |
| **Cutting Planes (Ours)**| CV / Bootstrap | Adversarial scenarios ($\Gamma$ perturbation budget) | CV feasibility + cost |

**Separation Oracle:** 
Our current separation oracle uses an uncertainty set over continuous label perturbations (with a $L_1$ budget $\Gamma$ and $L_\infty$ bounds $\bar{\delta}$) to find the worst-case constraints for the current $x^k$.
*TODO: Update the separation oracle to search directly over training samples (e.g. adversarial bootstrap/resampling) as a discrete strategy better suited for tree models.*

## Synthetic Experiment

The current codebase includes a synthetic nonlinear experiment. We generate a dataset using an underlying nonlinear function where $y$ values simulate constraints $f(x) \leq 0.5$. The variables $x$ are bounded within $[0, 1]$. Label noise is injected during dataset generation based on a configurable noise standard deviation $\sigma$. The models trained are constraint learning classifiers.
The experiments compare Nominal constraint learning, Robust Classification, Wrapper (ensemble chance constraints), and our robust Cutting Planes approach. We evaluate the performance under different constraint violation vulnerability criteria.

## Setup

```bash
pip install -r requirements.txt
```

Requires a Gurobi license (free academic license available).

## Running Experiments

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
- Uncertainty: perturbation budgets and bounds
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
│   │   ├── robust_classification.py
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

## Key Hypotheses

1. **Nominal CL is fragile:** Even moderate label noise ($\sigma > 0$) causes the nominal solution $x^*$ to frequently violate the true underlying constraint.
2. **Cutting Planes restores true feasibility at modest cost:** By injecting adversarial structure directly targeting optimization blindspots instead of random points, Cutting Planes achieves 100% feasibility efficiently, while objective costs increase gracefully with robustness parameter $\Gamma$.
3. **Cutting Planes > Wrapper under label noise:** Wrapper strategies diversify heavily across model classes or bootstrap folds but are bound by shared bias from the noisy data. Cutting Planes explicitly builds constraints over generated scenarios spanning label perturbations, giving stronger worst-case reachability coverage.
4. **Decision-aware robustness provides superiority:** Unlike robust classification which trains one globally robust model unaware of downstream constraints, Cutting Planes uses the separation oracle dynamically reacting to the optimizer's active region—focusing analytical power strictly where $x^*$ is vulnerable.
