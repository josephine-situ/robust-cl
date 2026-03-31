"""
Data generation for constraint learning experiments.

We need:
1. Training data (X, y) where y = f_true(X) + noise
2. A known ground truth f_true for synthetic experiments
3. A downstream optimization problem: min c'x s.t. f(x) <= b
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ProblemInstance:
    """Complete problem instance for constraint learning."""
    # Training data
    X_train: np.ndarray  # (n, d)
    y_train: np.ndarray  # (n,) - noisy labels
    y_true: np.ndarray   # (n,) - true labels (if synthetic)

    # Problem definition
    cost_vector: np.ndarray     # c in min c'x
    constraint_rhs: float       # b in f(x) <= b
    variable_lb: np.ndarray     # lower bounds on x
    variable_ub: np.ndarray     # upper bounds on x
    n_features: int

    # Ground truth (if available)
    f_true: Optional[Callable] = None


def synthetic_nonlinear(n_train: int = 200,
                        n_features: int = 2,
                        noise_std: float = 0.1,
                        seed: int = 42) -> ProblemInstance:
    """
    Synthetic problem with known ground truth.

    f_true(x) = sum_j x_j^2 + 0.5 * prod_j x_j
    Nonlinear but smooth; easy to visualize in 2D.

    Optimization problem:
        min  -sum(x)          (want x large)
        s.t. f_true(x) <= b   (constraint limits x)
             0 <= x <= 1
    """
    rng = np.random.RandomState(seed)

    # Ground truth function
    def f_true(x):
        """x can be (d,) or (n, d)."""
        x = np.atleast_2d(x)
        return np.sum(x ** 2, axis=1) + 0.5 * np.prod(x, axis=1)

    # Generate training data spread over [0, 1]^d
    X_train = rng.uniform(0, 1, size=(n_train, n_features))
    y_true = f_true(X_train)
    y_train = y_true + rng.normal(0, noise_std, size=n_train)

    # Cost vector: minimize -sum(x) (i.e., maximize sum)
    cost_vector = -np.ones(n_features)

    # Variable bounds
    variable_lb = np.zeros(n_features)
    variable_ub = np.ones(n_features)

    # Constraint RHS: set so that the true optimum is interior
    # f_true at x=1 is n_features + 0.5, so b < that
    constraint_rhs = 0.5 * n_features

    return ProblemInstance(
        X_train=X_train,
        y_train=y_train,
        y_true=y_true,
        cost_vector=cost_vector,
        constraint_rhs=constraint_rhs,
        variable_lb=variable_lb,
        variable_ub=variable_ub,
        n_features=n_features,
        f_true=f_true,
    )