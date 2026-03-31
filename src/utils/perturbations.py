"""
Label perturbation utilities.

Uncertainty set D = {delta : |delta_i| <= delta_bar_i, ||delta||_1 <= Gamma}
"""

import numpy as np
from typing import List


def sample_random_perturbation(n: int,
                               delta_bar: float,
                               gamma: float,
                               rng: np.random.RandomState
                               ) -> np.ndarray:
    """
    Sample a random perturbation delta in D.

    Strategy: sample direction, then project onto D.
    """
    # Sample raw perturbation
    delta = rng.uniform(-delta_bar, delta_bar, size=n)

    # Project onto L1 ball of radius gamma
    delta = project_l1_ball(delta, gamma)

    # Clip to box
    delta = np.clip(delta, -delta_bar, delta_bar)

    return delta


def project_l1_ball(v: np.ndarray, radius: float) -> np.ndarray:
    """Project v onto the L1 ball of given radius."""
    if np.sum(np.abs(v)) <= radius:
        return v

    # Standard algorithm: sort and threshold
    u = np.sort(np.abs(v))[::-1]
    cumsum = np.cumsum(u)
    n = len(v)
    rho = np.max(
        np.where((u - (cumsum - radius) / np.arange(1, n + 1)) > 0)[0]
    )
    theta = (cumsum[rho] - radius) / (rho + 1)
    return np.sign(v) * np.maximum(np.abs(v) - theta, 0)


def sample_multiple_perturbations(n: int,
                                  delta_bar: float,
                                  gamma: float,
                                  n_samples: int,
                                  seed: int = 42
                                  ) -> List[np.ndarray]:
    """Sample multiple perturbations from D."""
    rng = np.random.RandomState(seed)
    return [
        sample_random_perturbation(n, delta_bar, gamma, rng)
        for _ in range(n_samples)
    ]


def greedy_adversarial_perturbation(X, y, x_current, delta_bar,
                                    gamma, model_type, model_params,
                                    n_candidates=20, seed=42):
    """
    Greedy separation oracle.

    For fixed x_current, find delta in D that maximizes
    f(x_current; theta*(delta)) by greedily flipping/shifting
    labels.

    Parameters
    ----------
    X : training features
    y : original (noisy) training labels
    x_current : current solution from master problem
    delta_bar : per-label perturbation bound
    gamma : L1 budget
    model_type, model_params : for retraining
    n_candidates : number of greedy steps

    Returns
    -------
    best_delta : best perturbation found
    best_value : f(x_current; theta*(best_delta))
    best_model : retrained model
    """
    from src.models.train import retrain_on_perturbed

    n = len(y)
    rng = np.random.RandomState(seed)
    x_2d = np.atleast_2d(x_current)

    best_delta = np.zeros(n)
    base_model = retrain_on_perturbed(
        X, y, best_delta, model_type, model_params
    )
    best_value = base_model.predict(x_2d)[0]
    best_model = base_model

    # Greedy: try perturbing each candidate label
    # Focus on points near x_current in feature space
    distances = np.linalg.norm(X - x_current, axis=1)
    candidate_indices = np.argsort(distances)[:n_candidates]

    current_delta = np.zeros(n)
    current_l1 = 0.0

    for idx in candidate_indices:
        if current_l1 >= gamma:
            break

        # Try both positive and negative perturbation
        for sign in [+1, -1]:
            trial_delta = current_delta.copy()
            max_shift = min(delta_bar, gamma - current_l1)
            trial_delta[idx] = sign * max_shift

            model = retrain_on_perturbed(
                X, y, trial_delta, model_type, model_params
            )
            value = model.predict(x_2d)[0]

            if value > best_value:
                best_value = value
                best_delta = trial_delta.copy()
                best_model = model
                current_delta = trial_delta.copy()
                current_l1 = np.sum(np.abs(current_delta))
                break  # Accept this perturbation, move to next

    return best_delta, best_value, best_model