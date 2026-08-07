"""The shared label-uncertainty model D, and the scenario bank drawn from it.

Every uncertainty-aware method in this repo faces the same set

    D_c = { delta : |delta_i| <= eps_c,  ||delta||_1 <= gamma_c },
    eps_c = eps_0 * scale(y_c),   gamma_c = budget_frac * n * eps_c

and differs only in what it *does* with it: ``cp`` separates it lazily one cut at
a time, ``wrapper`` chance-constrains a fixed sample of it, ``robust_regression``
robustifies the fit against it. Each keeps exactly one conservatism dial (tau,
alpha, label_eps); the shape of D -- its scale, its budget, and whether outcomes
move together -- is shared, so a difference between methods is a difference in
*method*, not in the uncertainty each happened to be handed.

Two design points worth keeping straight:

**Why sd(y), not a residual-based scale.** delta is added to labels *before*
training and the model is retrained (:func:`retrain_on_perturbed`) -- this is not
a post-hoc perturbation of predictions. The radius therefore belongs in label
space, and sd(y) is the natural scale there: model-free, computable before any
fit, identical for all three methods. An out-of-fold residual radius would make D
depend on model class and hyperparameters, which is circular -- robust_reg's own
model would define the set robust_reg is robust to -- and would silently move D
whenever ``run_cv.py`` re-selected a model type. The residual estimator is kept
as ``stat="oof_quantile"`` for ablation, and :func:`label_scale_report` logs both
either way. **No coverage claim is made or implied**; this is a calibrated scale.

On gastric that means sd(y) ~ 0.289 for all five toxicity outcomes, since
``train_percentile_scores`` makes them percentile ranks. That is a feature, not a
degeneracy: ``eps_c = eps_0 * sd(y_c)`` is then the same *rank* shift in every
outcome, which is exactly what makes a shared coherent draw a coherent statement.
OS, in raw months, gets its own scale.

**Why draws are vertices, not interior points.** ``sample_random_perturbation``
draws uniform then L1-projects, spreading mass thinly over all n rows.
robust_reg's adversary instead takes the vertex -- +/-eps on m = budget_frac * n
rows, 0 elsewhere (:func:`worst_case_label_shift`). Sampling the interior would
hand CP and the wrapper a systematically weaker adversary than robust_reg at the
*same* D, confounding the comparison this module exists to enable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.data.generate import ProblemInstance
from src.models.train import retrain_on_perturbed, train_model


# ---------------------------------------------------------------------------
# Label scale
# ---------------------------------------------------------------------------
def label_scale(y: np.ndarray,
                stat: str = "sd",
                X: Optional[np.ndarray] = None,
                model_type: Optional[str] = None,
                model_params: Optional[dict] = None,
                folds: Optional[Sequence] = None,
                level: float = 0.9) -> float:
    """Scale of the label-uncertainty radius for one outcome.

    ``stat="sd"`` (default) is ``np.std(y)`` -- model-free, and what
    ``robust_regression`` has always used. ``stat="oof_quantile"`` is the
    ``level`` quantile of out-of-fold absolute residuals, an ablation that needs
    ``X`` / ``model_type`` / ``folds``. In-sample residuals are deliberately not
    an option: XGB with ``n_estimators=20`` on a few hundred arms drives them
    toward 0 and would collapse D.
    """
    y = np.asarray(y, dtype=float)
    if stat == "sd":
        return float(np.std(y))
    if stat != "oof_quantile":
        raise ValueError(f"unknown label scale stat {stat!r} (sd | oof_quantile)")
    if X is None or model_type is None or folds is None:
        raise ValueError("stat='oof_quantile' needs X, model_type and folds")
    resid = _oof_residuals(X, y, model_type, model_params, folds)
    return float(np.quantile(np.abs(resid), level))


def _oof_residuals(X, y, model_type, model_params, folds) -> np.ndarray:
    """Out-of-fold residuals over the caller's fold scheme (rows may repeat/omit)."""
    res = []
    for tr, va in folds:
        if len(tr) == 0 or len(va) == 0:
            continue
        model = train_model(X[tr], y[tr], model_type, model_params)
        res.append(y[va] - model.predict(X[va]))
    if not res:
        raise ValueError("no usable folds for out-of-fold residuals")
    return np.concatenate(res)


def label_scale_report(y, X=None, model_type=None, model_params=None,
                       folds=None, level: float = 0.9) -> dict:
    """``{sd, oof_sd, oof_quantile}`` for one outcome -- the diagnostic behind the
    choice of ``stat``. The out-of-fold entries are ``None`` when no folds are
    available. Cheap enough to log once per run; if the OOF numbers turn out as
    flat across outcomes as sd(y) is, the residual estimator buys nothing."""
    out = {"sd": float(np.std(np.asarray(y, dtype=float))),
           "oof_sd": None, "oof_quantile": None}
    if X is not None and model_type is not None and folds is not None:
        try:
            r = _oof_residuals(X, np.asarray(y, dtype=float),
                               model_type, model_params, folds)
            out["oof_sd"] = float(np.std(r))
            out["oof_quantile"] = float(np.quantile(np.abs(r), level))
        except (ValueError, RuntimeError):
            pass
    return out


# ---------------------------------------------------------------------------
# The uncertainty set D
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UncertaintySet:
    """D's shape, shared by every method. ``eps_0`` is a radius in units of
    ``scale(y)``; ``budget_frac`` is the fraction of rows the adversary may move
    (m = budget_frac * n), fixed at 0.5 and deliberately not a knob."""
    eps_0: float = 1.0
    budget_frac: float = 0.5
    coherent: bool = True
    scale_stat: str = "sd"

    def eps(self, scale: float) -> float:
        return float(self.eps_0) * float(scale)

    def gamma(self, scale: float, n: int) -> float:
        return float(self.budget_frac) * int(n) * self.eps(scale)


def uncertainty_set_from_config(config: dict, coherent: Optional[bool] = None) -> UncertaintySet:
    """Build the shared D from ``config['uncertainty']``."""
    unc = (config or {}).get("uncertainty", {}) or {}
    return UncertaintySet(
        eps_0=float(unc.get("eps_0", 1.0)),
        budget_frac=float(unc.get("budget_frac", 0.5)),
        coherent=bool(unc.get("coherent", True)) if coherent is None else bool(coherent),
        scale_stat=str(unc.get("scale_stat", "sd")),
    )


# ---------------------------------------------------------------------------
# The scenario bank
# ---------------------------------------------------------------------------
def _vertex_direction(n: int, budget_frac: float, rng: np.random.RandomState) -> np.ndarray:
    """One vertex of the box-cap-L1 set, in units of eps: +/-1 on m randomly chosen
    rows and 0 elsewhere, with m = budget_frac * n. ``||u||_1 = m`` exactly
    exhausts the budget, matching robust_reg's adversary."""
    m = int(round(float(budget_frac) * n))
    m = max(1, min(n, m))
    u = np.zeros(n, dtype=float)
    rows = rng.choice(n, size=m, replace=False)
    u[rows] = rng.choice([-1.0, 1.0], size=m)
    return u


class ScenarioBank:
    """B label-shift draws from D, and the models trained on them.

    Built once and reused: with the draws fixed, ``theta(delta_b)`` no longer
    depends on the incumbent ``x*``, so CP's worst-violation-over-the-bank is
    monotone across iterations and tau means something. (Redrawing every
    iteration is what made CP's trace oscillate rather than converge.)

    Draw ``b`` is a pure function of ``(seed, b)``, so a bank of size B and a bank
    of size P < B built from the same seed agree on their first P draws. The
    wrapper therefore takes a genuine *prefix* of CP's bank -- nested, not merely
    identically distributed -- which is what makes the alpha=0 / tau->0
    equivalence test compare like with like.

    B differs between the two by design. The wrapper embeds all P of its models in
    the MIP, so P is capped by MIP size; CP embeds one extra scenario per
    iteration and evicts/prunes cuts, so its master stays small at any B. CP can
    thus search a far denser sample of the same D at a fraction of the MIP.
    """

    def __init__(self,
                 instance: ProblemInstance,
                 model_config_map: dict,
                 uset: UncertaintySet,
                 n_scenarios: int = 200,
                 seed: int = 42,
                 scale_stat: Optional[str] = None,
                 folds: Optional[Sequence] = None,
                 verbose: bool = True):
        self.instance = instance
        self.uset = uset
        self.n_scenarios = int(n_scenarios)
        self.seed = int(seed)
        self.coherent = bool(uset.coherent)
        self._model_config_map = model_config_map

        # Ordered, de-duplicated MLModelData handles. Order fixes the incoherent
        # draw sequence, so it must not depend on dict iteration order.
        self._mds: List = []
        seen = set()
        for constraint in instance.constraints:
            for md in constraint.models_data:
                if id(md) not in seen:
                    seen.add(id(md))
                    self._mds.append(md)

        stat = scale_stat or uset.scale_stat
        self.scales: Dict[int, float] = {}
        self.reports: Dict[int, dict] = {}
        for md in self._mds:
            m_type, m_params = model_config_map[id(md)]
            self.scales[id(md)] = label_scale(
                md.y_train, stat=stat, X=md.X_train, model_type=m_type,
                model_params=m_params, folds=folds,
            )
            self.reports[id(md)] = label_scale_report(
                md.y_train, X=md.X_train, model_type=m_type,
                model_params=m_params, folds=folds,
            )
        degenerate = [k for k, v in self.scales.items() if not np.isfinite(v) or v <= 0]
        if degenerate:
            raise ValueError(
                f"{len(degenerate)} outcome(s) have a non-positive label scale under "
                f"stat={stat!r}; D would be empty. Check the training labels."
            )

        self._deltas: Dict[int, List[np.ndarray]] = {id(md): [] for md in self._mds}
        self._models: Dict[int, List] = {id(md): [] for md in self._mds}
        self._built = 0
        if verbose:
            self._log_scales()
        self.extend(self.n_scenarios, verbose=verbose)

    # -- construction --------------------------------------------------------
    def _log_scales(self) -> None:
        print(f"    [bank] D: eps_0={self.uset.eps_0:g} x scale(y), "
              f"budget_frac={self.uset.budget_frac:g}, "
              f"coherent={self.coherent}, stat={self.uset.scale_stat}", flush=True)
        for c in self.instance.constraints:
            for md in c.models_data:
                r = self.reports.get(id(md), {})
                oof_sd = r.get("oof_sd")
                oof_q = r.get("oof_quantile")
                extra = ""
                if oof_sd is not None:
                    extra = f"  oof_sd={oof_sd:.4f} oof_q90={oof_q:.4f}"
                print(f"    [bank]   {c.name}: sd(y)={r.get('sd', float('nan')):.4f}"
                      f"  eps={self.uset.eps(self.scales[id(md)]):.4f}{extra}", flush=True)

    def extend(self, n_total: int, verbose: bool = False) -> "ScenarioBank":
        """Grow the bank to ``n_total`` draws, keeping existing ones untouched."""
        n_total = int(n_total)
        if n_total <= self._built:
            return self
        import time as _time
        t0 = _time.time()
        for b in range(self._built, n_total):
            for md_id, delta in self._draw(b).items():
                self._deltas[md_id].append(delta)
            for md in self._mds:
                md_id = id(md)
                m_type, m_params = self._model_config_map[md_id]
                params = dict(m_params or {})
                # Fixed across bank members: the scenario must be the ONLY source
                # of variation, otherwise model-seed noise contaminates the delta
                # effect. (train_bootstrap_models deliberately does the opposite.)
                params["random_state"] = self.seed
                self._models[md_id].append(retrain_on_perturbed(
                    md.X_train, md.y_train, self._deltas[md_id][b], m_type, params,
                ))
        grown = n_total - self._built
        self._built = n_total
        self.n_scenarios = max(self.n_scenarios, n_total)
        if verbose:
            print(f"    [bank] trained {grown} scenario(s) x {len(self._mds)} model(s) "
                  f"in {_time.time() - t0:.1f}s", flush=True)
        return self

    def _draw(self, b: int) -> Dict[int, np.ndarray]:
        """Label shift per MLModelData for draw ``b``; a pure function of (seed, b)."""
        rng = np.random.RandomState(self.seed + b)
        out = {}
        shared_u = None
        for md in self._mds:
            n = len(md.y_train)
            if self.coherent:
                # One standardized direction shared across outcomes, scaled per
                # outcome by its own eps_c -- so the draw sits inside every
                # outcome's own D while all models move together.
                if shared_u is None or len(shared_u) != n:
                    shared_u = _vertex_direction(n, self.uset.budget_frac, rng)
                u = shared_u
            else:
                u = _vertex_direction(n, self.uset.budget_frac, rng)
            delta = self.uset.eps(self.scales[id(md)]) * u
            out[id(md)] = _clip_to_bounds(md, delta)
        return out

    # -- consumption ---------------------------------------------------------
    def __len__(self) -> int:
        return self._built

    def models_for(self, b: int) -> Dict[int, object]:
        """``{md_id -> model}`` for draw ``b`` -- one coherent (or incoherent)
        relabeling of the whole trial, which is what CP cuts on."""
        return {md_id: models[b] for md_id, models in self._models.items()}

    def model(self, model_data, b: int):
        return self._models[id(model_data)][b]

    def delta(self, model_data, b: int) -> np.ndarray:
        return self._deltas[id(model_data)][b]

    def as_ensembles_cache(self, n: Optional[int] = None) -> Dict[int, list]:
        """``{md_id -> [model_0 .. model_{n-1}]}`` -- the shape
        ``train_bootstrap_ensembles_for_instance`` already accepts, so the wrapper
        takes its bank as a parameter pass. ``n=None`` uses the whole bank."""
        n = self._built if n is None else min(int(n), self._built)
        return {md_id: models[:n] for md_id, models in self._models.items()}


def _clip_to_bounds(model_data, delta: np.ndarray) -> np.ndarray:
    """Return the delta actually realizable given the outcome's label bounds.

    Percentile-scored outcomes live in [0, 1]; shifting a rank past either end is
    not a plausible relabeling. Clipping ``y + delta`` and taking the difference
    keeps the applied shift inside D (clipping only shrinks |delta_i|).
    """
    bounds = getattr(model_data, "label_bounds", None)
    if bounds is None:
        return delta
    lo, hi = bounds
    y = np.asarray(model_data.y_train, dtype=float)
    return np.clip(y + delta, lo, hi) - y
