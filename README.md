# Robust Constraint Learning via Iterative Scenario Generation

## Overview

This repository implements and compares approaches for making
constraint learning (Maragno et al., 2023/2025) robust to label
uncertainty. When ML models are embedded as constraints in
optimization problems, noisy training labels can lead to solutions
that violate the true constraint.

We formulate this as a trilevel optimization problem and solve it
via Column-and-Constraint Generation (C&CG).

## Methods Compared

| Method | Description |
|--------|-------------|
| **Nominal** | Standard constraint learning; single model, no robustness |
| **Robust Classification** | Train one model robust to label noise, then embed (not decision-aware) |
| **Wrapper** | Maragno et al. (2025) ensemble chance constraint on P bootstrap estimators |
| **Random Scenarios** | Embed S models from random label perturbations, require all feasible |
| **C&CG** | Adversarial scenario generation (proposed); iteratively finds worst-case perturbations |

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

### Efficiency comparison (C&CG vs random at equal k)

```bash
python experiments/run_sweep.py --sweep efficiency
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
- Uncertainty: delta_bar, Gamma
- Method-specific: wrapper alpha/P, number of scenarios, C&CG settings
- Evaluation: number of held-out perturbations

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
│   │   ├── random_scenarios.py
│   │   └── ccg.py            # Column-and-Constraint Generation
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
| `results.csv` | Single-run comparison of all 5 methods |
| `ccg_trace.csv` | C&CG iteration history (violation, objective) |
| `sweep_results.csv` | Results across Gamma values |
| `noise_sweep_results.csv` | Results across noise levels |
| `efficiency.csv` | C&CG vs random at equal model count |
| `comparison.png` | Bar chart of single-run results |
| `ccg_convergence.png` | C&CG violation and objective per iteration |
| `gamma_sweep.png` | Price of robustness curves |
| `noise_sweep.png` | Degradation under increasing noise |
| `efficiency.png` | Adversarial vs random scenario selection |

## Key Hypotheses

1. **Nominal is fragile:** moderate label noise causes infeasibility
2. **Robust classification is over-conservative:** protects uniformly, not at the optimizer's solution
3. **Wrapper has shared bias:** all estimators trained on same noisy data
4. **C&CG dominates:** better feasibility at lower cost; fewer scenarios needed than random sampling
