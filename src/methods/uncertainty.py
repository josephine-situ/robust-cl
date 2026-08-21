"""The shared label-uncertainty model D, and the scenario bank drawn from it.

Every uncertainty-aware method in this repo faces the same set, in one of two
parameterizations selected by ``uncertainty.geometry``:

    ellipsoid  D_c = { d : ||d||_2 <= R_c },  R_c = rho * scale(y_c) * sqrt(n)
               -- the DEFAULT since 2026-08-18
    box_l1     D_c = { d : |d_i| <= eps_c,  ||d||_1 <= budget_frac * n * eps_c },
                     eps_c = eps_0 * scale(y_c)  -- kept as an ablation

and differs only in what it *does* with it: ``cp`` separates it lazily one cut at
a time, ``wrapper`` chance-constrains a fixed sample of it, ``robust_regression``
robustifies the fit against it. Each keeps exactly one conservatism dial (tau,
alpha, label_eps); the shape of D -- its scale, its budget, and whether outcomes
move together -- is shared, so a difference between methods is a difference in
*method*, not in the uncertainty each happened to be handed.

``ellipsoid`` exists to make D a **one-parameter family**. ``budget_frac`` cannot
constrain an L2 ball -- no L1 face, no support restriction -- so it could only
ever scale the radius, leaving ``(eps_0, budget_frac)`` non-identifiable with
only their product observable. ``rho`` replaces both, which is why it is now the
default. ``box_l1`` is untouched and remains the geometry every artifact
currently in ``results/`` was produced under -- those numbers do not carry across
the switch (the ball is ~2.8x the box's effective adversary at rho = eps_0 = 1).

**rho is swept, not fitted -- and rho*(method) is what the evaluation run
uses.** It defines the problem all three methods solve, and D is literally shared
at every point of the swept axis: that curve, produced by
``experiments/run_rho_sweep.py``, is where the shared-D comparison is read. The
derived rho*(method) -- the largest rho whose held-out feasibility still meets the
target -- is then fixed per method for evaluation, so **evaluation matches
held-out feasibility, not D**: each method faces a ball of its own radius there.
The criterion is what keeps that honest. Never fit rho against the GT ensemble (it
tunes to the judge) or against synthetic's known ``noise_std`` (it calibrates D to
the data-generating process, which CP would then win by construction).

Two design points worth keeping straight:

**Why the scale is the out-of-fold residual sd.** delta is added to labels
*before* training and the model is retrained (:func:`retrain_on_perturbed`) --
this is not a post-hoc perturbation of predictions -- so the radius belongs in
**label space**. Both candidate statistics live there; they differ in what they
measure:

- ``sd(y)`` is the *marginal* spread of the labels. Most of it is signal the
  features explain, so ``eps_0 = 1`` corrupts labels far harder than the data
  supports wherever the model fits well. Measured: on synthetic (true noise 0.100)
  ``sd(y) = 0.545`` against an out-of-fold residual of ``0.128`` -- a factor of
  four. CP could not converge against that set in 20 iterations.
- ``oof_sd`` (the default) is the *unexplained* spread -- how much of a label the
  frozen model cannot account for. ``eps_0 = 1`` then means "one unexplained
  standard deviation", a unit whose **meaning** transfers across problems rather
  than merely its units. On synthetic it recovers 0.128 against a true noise of
  0.100 without ever being told the data-generating process.

The model dependence is bounded and deliberate: ``run_cv.py`` freezes the model
class *before* any robustness and all three methods embed that same frozen model,
so D is defined by a shared, pre-committed choice rather than by any one method's
tuning. What the residual does conflate is label noise with model
misspecification: on gastric the models explain almost nothing (one outcome's
residual exceeds ``sd(y)``), so D there stays nearly as wide as the marginal
spread -- which is the honest answer when the fit is that poor.

``stat="sd"`` (marginal) and ``stat="oof_quantile"`` remain as ablations, and
:func:`label_scale_report` logs all three. **No coverage claim is made or
implied**; this is a calibrated scale, not a conformal guarantee.

Note the percentile transform makes ``sd(y) ~ 0.289`` for all five gastric
toxicities. Under either statistic the per-outcome scaling
``delta^c = eps_c * u`` keeps a shared coherent draw meaning the same *relative*
shift in every outcome; OS, in raw months, gets its own scale.

**Why draws are vertices, not interior points.** ``sample_random_perturbation``
draws uniform then L1-projects, spreading mass thinly over all n rows.
robust_reg's adversary instead takes the vertex -- +/-eps on m = budget_frac * n
rows, 0 elsewhere (:func:`worst_case_label_shift`). Sampling the interior would
hand CP and the wrapper a systematically weaker adversary than robust_reg at the
*same* D, confounding the comparison this module exists to enable.

**Why the draws are nonetheless RANDOM in direction, while robust_reg's adversary
is DIRECTED. This asymmetry is deliberate.** The three methods are matched in
*magnitude* -- every draw is a boundary point of D, spending the same budget
robust_reg's inner max spends -- but they are not matched in *alignment*. CP and
the wrapper draw directions at random; robust_reg aims its shift along the
residuals (:func:`worst_case_label_shift` / :func:`l2_worst_case_shift`).

The reason is structural, and it is about feasibility rather than fairness. CP and
the wrapper turn each scenario into *embedded constraints*: CP adds a cut per
accepted scenario, the wrapper requires (1-alpha) of P models to hold jointly. A
directed adversary would make each of those constraints as tight as D allows, and
tight constraints accumulate -- the master stops admitting any prescription at all.
That is not hypothetical: it is the failure ``run_adversary_probe.py`` Part C/D
exists to measure ("with EVERY constraint at its own worst case simultaneously, is
the master still feasible at each anchor?"), and it is what CP's rollback and
permanent-rejection machinery already spends effort containing at *random* draws.

robust_reg is exposed to none of that, because the adversary never becomes a
constraint. It shapes the *fit*: one model per outcome is retrained on the shifted
labels, and that single model is what gets embedded. A worst-case shift moves
where the model sits; it cannot make the optimization infeasible. So robust_reg can
afford a directed adversary at the same D that would render CP and the wrapper
unsolvable.

The cost of the asymmetry is real and should be reported, not hidden: random
sampling recovers only part of the attainable worst case. Measured on synthetic by
the probe, the best of B random draws reaches ``1.07 eps`` against a directed
adversary's ``1.67 eps`` (~64%), and the gap *widens* under ``"ellipsoid"``,
since ``g'u`` has the same sd under both geometries while the attainable maximum
rises. So "shared D" guarantees a shared *set*, and equal *budget*, but not equal
adversary strength -- by design.

**Geometry (``uncertainty.geometry``, default ``"ellipsoid"``).** It replaces the
older box-cap-L1 set with the ball ``||d||_2 <= R_c = rho*scale*sqrt(n)``.
Draws become uniform on the unit sphere (:func:`_sphere_direction`); the
adversary's argmax becomes ``R g / ||g||_2`` (:func:`l2_worst_case_shift`), which
is closed form rather than a greedy top-m search. The size correspondence to the
box -- ``sqrt(m)*eps_c``, the L2 length of a vertex -- survives as
:meth:`UncertaintySet.radius_from_eps` for equal-budget geometry comparisons
(``run_adversary_probe.py``), but no longer parameterizes the set. Three
consequences, all measured or provable:

1. The ellipsoid is **stronger** at matched size, always (Cauchy-Schwarz). How
   much stronger depends on the influence vector's shape and must be measured per
   problem: the box is only efficient when g has exactly m equally-sized nonzeros.
   Gastric has nnz(g) = 313-320 > m = 160, so the box abandons influential rows
   (gap -> sqrt(n/m) as g flattens); synthetic has nnz(g) = 57 < m = 100, so the
   box wastes budget on rows that cannot move f(x*) at all (measured: 2.0x).
2. Random sampling is **no better** in it: ``g'u`` has sd ``||g|| sqrt(m/n)``
   under both geometries, so the bank's best-of-B improves not at all while the
   attainable worst case rises. The random-bank-is-weak gap widens.
3. It is the only geometry here that *could* carry a **coverage** statement
   (:func:`chi2_radius`), because ``||d||_2^2/sigma^2 ~ chi^2_n`` under Gaussian
   noise -- but with ``scale = oof_sd`` it cannot, and no such claim is made. The
   binding failure is not the Gaussian assumption (gastric toxicity residuals are
   symmetric and thin-tailed: |skew| <= 0.33, excess kurtosis -0.24 to -1.04,
   which errs conservative) but sigma itself: an out-of-fold residual sd is label
   noise *plus* misspecification, and on gastric it is ~90% the latter
   (unexplained 0.76-1.02). A coverage number built on it would cover the wrong
   random variable. Two further violations: the noise is heteroskedastic (arm
   sizes 26-92 give binomial SEs spanning 2.2x) while chi^2_n assumes one sigma,
   and OOF residuals are not independent across rows, so the effective df is
   below n. See :func:`chi2_radius` for the route to a real claim.

The flag changes the *set*, not the separation loop: CP still cuts one shared
draw ``b`` at a time, so the joint-vs-product-set objection is untouched by it.
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
                stat: str = "oof_sd",
                X: Optional[np.ndarray] = None,
                model_type: Optional[str] = None,
                model_params: Optional[dict] = None,
                folds: Optional[Sequence] = None,
                level: float = 0.9) -> float:
    """Scale of the label-uncertainty radius for one outcome.

    - ``"oof_sd"`` (default): standard deviation of out-of-fold residuals -- the
      *unexplained* label spread, so ``eps_0 = 1`` is one unexplained sd.
    - ``"oof_quantile"``: the ``level`` quantile of ``|residual|``, a
      heavier-tailed variant.
    - ``"sd"``: ``np.std(y)``, the *marginal* spread. Model-free but mostly
      signal wherever the model fits (see the module docstring).

    The out-of-fold variants need ``X`` / ``model_type``; ``folds`` defaults to a
    4-fold KFold over the rows. In-sample residuals are deliberately not an
    option: XGB with ``n_estimators=20`` on a few hundred arms drives them toward
    0 and would collapse D.
    """
    y = np.asarray(y, dtype=float)
    if stat == "sd":
        return float(np.std(y))
    if stat not in ("oof_sd", "oof_quantile"):
        raise ValueError(
            f"unknown label scale stat {stat!r} (oof_sd | oof_quantile | sd)")
    if X is None or model_type is None:
        raise ValueError(f"stat={stat!r} needs X and model_type")
    if folds is None:
        folds = default_folds(len(y))
    resid = _oof_residuals(X, y, model_type, model_params, folds)
    if stat == "oof_sd":
        return float(np.std(resid))
    return float(np.quantile(np.abs(resid), level))


def default_folds(n: int, n_splits: int = 4, seed: int = 42):
    """Plain KFold row indices, used when the caller supplies no fold scheme.

    Callers holding a :class:`ProblemInstance` should prefer
    :func:`instance_folds`, which picks the problem's own scheme (temporal
    forward-chaining on gastric, KFold on synthetic).
    """
    from sklearn.model_selection import KFold
    n_splits = max(2, min(int(n_splits), int(n)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(int(n))))


def _cutoffs_from_years(years) -> tuple:
    """Forward-chaining cutoffs that are non-empty for *these* rows.

    The default cutoffs (2004-2007) describe the full gastric training set. A
    subset of it -- an outer CV fold, or a ``train_subsample_frac`` draw -- can end
    before 2004, in which case every default cutoff yields an empty validation year
    and the temporal scheme collapses to a KFold fallback that ignores time. Deriving
    the cutoffs from the rows in hand keeps forward-chaining wherever the rows can
    support it, so the scale estimate never sees a later year than it trains on.
    """
    years = np.asarray(years, dtype=float)
    usable = [c for c in np.unique(years)[:-1]
              if np.any(years <= c) and np.any((years > c) & (years <= c + 1))]
    return tuple(usable[-4:])


def instance_folds(instance: ProblemInstance, seed: int = 42):
    """The problem's own fold scheme, for estimating the label scale.

    Temporal forward-chaining on gastric (``Pub_Year`` is a feature, so random
    folds would leak future information into the scale estimate) and KFold on
    synthetic. Falls back to :func:`default_folds` if the instance cannot supply
    folds -- the scale is a nuisance parameter, not worth failing a solve over.

    The years are read off ``instance``, so a fold/subsample instance yields folds
    indexed into *its own* rows: D's radius is estimated from the rows that instance
    fits on, never from rows held out of it.
    """
    try:
        from src.methods.cv_calibrate import make_folds
        years = getattr(instance, "train_pub_years", None)
        if years is not None:
            cutoffs = _cutoffs_from_years(years)
            if cutoffs:
                return make_folds(instance, "temporal", cutoffs, seed=seed)
        return make_folds(instance, "auto", seed=seed)
    except (ImportError, ValueError, AttributeError, IndexError):
        n = len(instance.constraints[0].models_data[0].y_train)
        return default_folds(n, seed=seed)


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
    choice of ``stat``, and the evidence for how much of the label spread the
    frozen model actually explains. Logged once per bank build."""
    out = {"sd": float(np.std(np.asarray(y, dtype=float))),
           "oof_sd": None, "oof_quantile": None,
           "resid_skew": None, "resid_kurtosis": None, "shapiro_p": None}
    if X is not None and model_type is not None and folds is not None:
        try:
            r = _oof_residuals(X, np.asarray(y, dtype=float),
                               model_type, model_params, folds)
            out["oof_sd"] = float(np.std(r))
            out["oof_quantile"] = float(np.quantile(np.abs(r), level))
            out.update(_normality_report(r))
        except (ValueError, RuntimeError):
            pass
    return out


def _normality_report(resid: np.ndarray) -> dict:
    """Shape of the out-of-fold residuals: the evidence for or against
    :func:`chi2_radius`.

    ``||delta||_2^2 / sigma^2 ~ chi^2_n`` needs the noise to be Gaussian (spherical
    is enough for the *direction*, but the radius quantile is chi^2-specific), so
    these three numbers are what decides whether an ellipsoidal D may carry a
    coverage claim. Reported, never acted on: nothing downstream branches on them.
    """
    r = np.asarray(resid, dtype=float)
    out = {"resid_skew": None, "resid_kurtosis": None, "shapiro_p": None}
    try:
        from scipy import stats as _st
        out["resid_skew"] = float(_st.skew(r))
        out["resid_kurtosis"] = float(_st.kurtosis(r))          # excess; 0 = normal
        # Shapiro-Wilk is unreliable past ~5000 points; subsample deterministically.
        rs = r if len(r) <= 5000 else np.random.RandomState(0).choice(r, 5000, False)
        out["shapiro_p"] = float(_st.shapiro(rs).pvalue)
    except (ImportError, ValueError, RuntimeError):
        pass
    return out


# ---------------------------------------------------------------------------
# The uncertainty set D
# ---------------------------------------------------------------------------
GEOMETRIES = ("box_l1", "ellipsoid")


@dataclass(frozen=True)
class UncertaintySet:
    """D's shape, shared by every method. ``geometry`` selects between

    - ``"box_l1"``:    D = {|d_i| <= eps_c, ||d||_1 <= budget_frac*n*eps_c},
                       eps_c = eps_0 * scale(y_c)
    - ``"ellipsoid"``: D = {||d||_2 <= R_c},  R_c = rho * scale(y_c) * sqrt(n)

    **The two parameterizations are separate on purpose.** ``box_l1`` keeps
    ``eps_0`` and ``budget_frac`` exactly as they were so every result in
    ``results/`` reproduces. ``ellipsoid`` reads ``rho`` alone and ignores both:
    an L2 ball has no L1 face and no support restriction, so ``budget_frac``
    could only ever have entered as a scale factor on the radius -- meaning
    ``(eps_0, budget_frac)`` was **non-identifiable** there, with only
    ``eps_0*sqrt(budget_frac)`` observable. Collapsing them into one ``rho``
    discards nothing.

    **Why ``sqrt(n)``.** For iid noise at one scale unit per row, ``||d||_2 ~
    scale*sqrt(n)``, so ``rho = 1`` is "the typical L2 norm of one unexplained sd
    of label noise" -- a meaning that transfers across problems (n=200 synthetic,
    n=320 gastric) rather than only its units. This is a **size convention, not a
    coverage claim**: see :func:`chi2_radius` for why a chi^2 quantile cannot
    supply one here while the scale is an out-of-fold *residual* sd (label noise
    confounded with misspecification).

    Note ``rho = 1`` is NOT the old operating point. Against ``eps_0 = 1`` at
    ``budget_frac = 0.5`` it is sqrt(2) wider (sqrt(n) vs sqrt(n/2)) before the
    ellipsoid's own Cauchy-Schwarz advantage over the box (measured 2.0x on
    synthetic, :mod:`src.utils.perturbations`). Expect rho* well below 1; read it
    off a sweep, do not assume it.

    ``clip_labels`` intersects D with the outcome's ``label_bounds`` for the
    *training* adversary, matching what :meth:`ScenarioBank.draw` has always done
    to every bank draw (:func:`_clip_to_bounds`). **``config.yaml`` turns it on**
    (2026-08-21): with it off, robust_reg trained against the raw ball while CP
    and the wrapper faced the clipped one, so "shared D" held only up to the
    bounds -- and it binds hardest exactly there. Measured on the five gastric
    toxicities at ``rho=1``: 45--49% of the shifted labels fall outside [0, 1] and
    clipping roughly **halves** the realizable shift (DLT ``||delta||`` 4.56 ->
    2.56); at ``rho=0.75``, 39--41% and 3.42 -> 2.22. OS, unbounded, is untouched.
    The field itself still defaults ``False`` so an old config loads unchanged.

    It bites only where ``label_bounds`` is set: gastric's five toxicities
    (percentile ranks). Gastric OS and the synthetic constraint carry none, so
    their numbers do not move. On the linear arm it costs the closed form --
    ``robust_regression._train_label_robust_model`` routes a bounded linear
    outcome (on gastric, GI) to the alternating loop, because
    ``max ||r + delta||^2`` over a ball **intersected with a box** is not
    ``(||r|| + R)^2``. Gastric robust_reg numbers predating 2026-08-21 are not
    comparable across this switch.

    ``coherent_exclude`` names constraints drawn **independently** even when
    ``coherent=True`` -- the coherence grouping, not a global flag. Empty by
    default so existing banks are bit-identical; ``config.yaml`` sets it to the
    OS outcome on gastric. See :meth:`ScenarioBank._draw` for the measurement.
    """
    eps_0: float = 1.0
    budget_frac: float = 0.5
    coherent: bool = True
    scale_stat: str = "oof_sd"
    geometry: str = "box_l1"
    rho: float = 1.0
    coherent_exclude: tuple = ()
    clip_labels: bool = False

    def __post_init__(self):
        if self.geometry not in GEOMETRIES:
            raise ValueError(
                f"unknown uncertainty geometry {self.geometry!r} "
                f"(expected one of {GEOMETRIES})")
        object.__setattr__(self, "coherent_exclude",
                           tuple(self.coherent_exclude or ()))

    def eps(self, scale: float) -> float:
        """Per-row cap (box_l1) / the unit the L2 radius is built from (ellipsoid)."""
        return float(self.eps_0) * float(scale)

    def n_moved(self, n: int) -> int:
        """m = budget_frac * n, clamped to [1, n]. The number of rows a box vertex
        moves, and the ``sqrt(m)`` that sets the matched ellipsoid radius."""
        return max(1, min(int(n), int(round(float(self.budget_frac) * int(n)))))

    def gamma(self, scale: float, n: int) -> float:
        """L1 budget. ``box_l1`` only -- an ellipsoid has no L1 face."""
        return float(self.budget_frac) * int(n) * self.eps(scale)

    def radius(self, scale: float, n: int) -> float:
        """L2 radius R_c = rho * scale * sqrt(n). ``ellipsoid`` only.

        Does not consult ``budget_frac``: an L2 ball has no L1 face, so the only
        way that parameter could enter is as a factor on the radius, which
        ``rho`` already supplies (see the class docstring on identifiability).
        """
        return float(self.rho) * float(scale) * float(np.sqrt(int(n)))

    def radius_from_eps(self, eps: float, n: int) -> float:
        """The *matched-size* radius sqrt(m)*eps: the L2 length of the box-cap-L1
        vertex at per-row cap ``eps``.

        This is the box-to-ball size correspondence, kept for callers that hold an
        ``eps_c`` and want the equivalent ball -- ``run_adversary_probe.py`` uses
        it to compare geometries at equal budget. It is **not** how the ellipsoid
        set is parameterized any more; use :meth:`radius` for that.
        """
        return float(np.sqrt(self.n_moved(n))) * float(eps)

    def magnitude(self, scale: float, n: int) -> float:
        """Multiplier applied to a standardized draw from :func:`_draw_direction`.

        The two geometries standardize differently -- box directions carry the
        vertex's own sqrt(m) length in units of eps, sphere directions are unit
        -- so this is the one place that difference is reconciled.
        """
        if self.geometry == "ellipsoid":
            return self.radius(scale, n)
        return self.eps(scale)


def uncertainty_set_from_config(config: dict, coherent: Optional[bool] = None) -> UncertaintySet:
    """Build the shared D from ``config['uncertainty']``."""
    unc = (config or {}).get("uncertainty", {}) or {}
    return UncertaintySet(
        eps_0=float(unc.get("eps_0", 1.0)),
        budget_frac=float(unc.get("budget_frac", 0.5)),
        coherent=bool(unc.get("coherent", True)) if coherent is None else bool(coherent),
        scale_stat=str(unc.get("scale_stat", "oof_sd")),
        geometry=str(unc.get("geometry", "box_l1")),
        rho=float(unc.get("rho", 1.0)),
        coherent_exclude=tuple(unc.get("coherent_exclude", ()) or ()),
        clip_labels=bool(unc.get("clip_labels", False)),
    )


def chi2_radius(scale: float, n: int, level: float = 0.9) -> float:
    """L2 radius of the ball that covers a Gaussian noise vector with prob ``level``.

    **Not wired into the default path**, and deliberately so -- it is the one thing
    an ellipsoid makes available that the box-cap-L1 set does not, and it is only
    licensed if the residuals are actually near-Gaussian.

    If the true label noise is ``delta ~ N(0, sigma^2 I_n)`` then
    ``||delta||_2^2 / sigma^2 ~ chi^2_n`` exactly, so

        R = sigma * sqrt(chi2.ppf(level, n))   =>   P(delta in D) = level

    which turns D from "a calibrated scale, no coverage claim" into a set with a
    stated confidence level. Note the concentration: chi^2_n has mean n and sd
    sqrt(2n), so R ~ sigma*sqrt(n) almost regardless of ``level`` -- at n=320 the
    50% and 99% radii differ by about 6%. The coverage is over the *label noise
    vector*, not over the fitted model; the chain to the model runs through
    ``retrain_on_perturbed``, i.e. "if delta is in D then theta(delta) is in the
    retrained image of D".

    **Unused: zero call sites, and it should stay that way while ``scale`` is an
    out-of-fold residual sd.** Four assumptions are needed and three fail here:

    - *sigma known* -- FAILS, and this is the binding one. ``oof_sd`` is label
      noise plus misspecification; on gastric the fits explain almost nothing
      (unexplained 0.76-1.02), so the radius is set almost entirely by what the
      model cannot fit. The nameable label noise is smaller by 1.7-3.1x (binomial
      sampling SE of the toxicity proportions at their arm sizes, pushed through
      the percentile map). A "90% coverage" built on ``oof_sd`` covers the wrong
      random variable.
    - *homoskedastic* -- FAILS. ``N_Patient`` runs 26-92, so per-row binomial SEs
      span 2.2x (q10 0.040, q90 0.088 on DLT). One sigma is the wrong object.
    - *independent components* -- FAILS. df = n needs n independent coordinates;
      OOF residuals share a fitted model within each fold, and forward-chaining
      trains on growing prefixes. Effective df < n by an unquantified amount.
    - *Gaussian* -- HOLDS, benignly. Measured on gastric: the five toxicities are
      symmetric (|skew| <= 0.33) with negative excess kurtosis (-0.24 to -1.04),
      so ``||d||^2`` concentrates more than chi^2 and the radius errs conservative.
      OS does not qualify (skew +0.64, exkurt +0.58, shapiro_p 6e-3).

    And one that survives even if all four held: coverage of ``delta`` is not
    coverage of the *decision*. The chain runs delta -> ``retrain_on_perturbed``
    -> theta(delta) -> constraint -> x*, and under misspecification there is no
    true theta for D to cover.

    A real claim would come from the measurement rather than the fit: the gastric
    toxicities are proportions over ``N_Patient`` patients, so
    ``sigma_i = sqrt(p_i(1-p_i)/N_i)`` is known per row and
    ``D = {sum_i d_i^2/sigma_i^2 <= chi2.ppf(level, n)}`` handles the
    heteroskedasticity correctly and covers a nameable quantity. Its own limits:
    sampling noise only -- not between-trial heterogeneity, not the
    ``IterativeImputer`` fill, not misspecification -- and each sigma_i still has
    to cross the percentile map, whose local slope is unstable where labels pile
    up near zero.

    Note also the concentration: chi^2_n has mean n and sd sqrt(2n), so R is
    nearly independent of ``level`` -- at n=320 the 50% and 99% radii differ by
    9.3%, at n=145 by 14%. The coverage level is therefore a nearly inert knob and
    must not be used as a conservatism dial; ``rho`` is the axis with real range.
    """
    from scipy.stats import chi2
    return float(scale) * float(np.sqrt(chi2.ppf(float(level), df=int(n))))


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


def _sphere_direction(n: int, rng: np.random.RandomState) -> np.ndarray:
    """One uniform draw from the **unit** sphere ``||u||_2 = 1``.

    Unit-length, unlike :func:`_vertex_direction`, which carries its own sqrt(m)
    in units of eps. :meth:`UncertaintySet.magnitude` reconciles the two, and it
    is what lets ``rho`` be the ellipsoid's single size parameter -- the radius
    lives entirely in the multiplier rather than being split between the
    multiplier and the draw. Uniformity comes free from spherical symmetry of the
    standard normal: normalizing a N(0, I_n) sample discards the magnitude and
    keeps the direction.

    Worth knowing before reading a bank built this way: a random direction in high
    dimension is nearly orthogonal to any fixed influence vector g, so ``g'u`` has
    sd ``||g||/sqrt(n)`` and the *realized* shift ``R g'u`` has sd ``rho*scale*||g||``
    -- matching the vertex draw's ``||g||*sqrt(m/n)`` scale up to the radius
    convention. Random sampling is no better here than there, while the attainable
    maximum is larger, so switching geometry *widens* the gap between the bank's
    best draw and the true worst case rather than closing it.
    """
    g = np.asarray(rng.normal(size=int(n)), dtype=float)
    nrm = float(np.linalg.norm(g))
    if nrm <= 0:                     # probability zero; keep the draw well-defined
        return np.ones(int(n), dtype=float) / float(np.sqrt(int(n)))
    return g / nrm


def _draw_direction(n: int, uset: UncertaintySet,
                    rng: np.random.RandomState) -> np.ndarray:
    """One standardized draw from D's boundary, per ``geometry``.

    The two standardizations differ -- box vertices are +/-1 on m rows (units of
    eps_c, L2 length sqrt(m)); sphere draws are unit vectors -- and
    :meth:`UncertaintySet.magnitude` supplies the matching multiplier. Both
    branches consume the rng, but differently, so a bank at a given seed is NOT
    comparable draw-for-draw across geometries -- only distributionally. Draw
    ``b`` remains a pure function of ``(seed, b)`` within a geometry, which is all
    the wrapper's nested-prefix property needs.
    """
    if uset.geometry == "ellipsoid":
        return _sphere_direction(n, rng)
    return _vertex_direction(n, uset.budget_frac, rng)


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
        self._md_name: Dict[int, str] = {}
        seen = set()
        for constraint in instance.constraints:
            for md in constraint.models_data:
                if id(md) not in seen:
                    seen.add(id(md))
                    self._mds.append(md)
                    self._md_name[id(md)] = constraint.name
        # Constraints that keep their own independent direction under coherence.
        self._excluded = {id(md) for md in self._mds
                          if self._md_name[id(md)] in set(uset.coherent_exclude)}
        # Reported, not raised: one config.yaml drives both problems, and
        # "os_constraint" legitimately does not exist on synthetic. A typo should
        # still be visible, hence the note rather than silence.
        unknown = set(uset.coherent_exclude) - set(self._md_name.values())
        if unknown and verbose:
            print(f"    [bank] note: coherent_exclude {sorted(unknown)} not in this "
                  f"instance (have {sorted(set(self._md_name.values()))}); ignored",
                  flush=True)

        stat = scale_stat or uset.scale_stat
        # Out-of-fold scales need a fold scheme; use the problem's own (temporal on
        # gastric, so the scale estimate cannot leak future information).
        if folds is None and stat != "sd":
            folds = instance_folds(instance, seed)
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
        if self.uset.geometry == "ellipsoid":
            size = f"rho={self.uset.rho:g} x scale(y) x sqrt(n)"
        else:
            size = (f"eps_0={self.uset.eps_0:g} x scale(y), "
                    f"budget_frac={self.uset.budget_frac:g}")
        excl = (f", coherent_exclude={list(self.uset.coherent_exclude)}"
                if self.uset.coherent_exclude else "")
        print(f"    [bank] D: {size}, geometry={self.uset.geometry}, "
              f"coherent={self.coherent}{excl}, "
              f"stat={self.uset.scale_stat}", flush=True)
        for c in self.instance.constraints:
            for md in c.models_data:
                r = self.reports.get(id(md), {})
                sd, oof_sd, oof_q = r.get("sd"), r.get("oof_sd"), r.get("oof_quantile")
                extra = ""
                if oof_sd is not None:
                    # explained/unexplained ratio: near 1 means the model accounts
                    # for almost none of the label spread, so D stays wide.
                    extra = (f"  oof_sd={oof_sd:.4f} oof_q90={oof_q:.4f}"
                             f"  unexplained={oof_sd / sd:.2f}" if sd else "")
                n = len(md.y_train)
                if self.uset.geometry == "ellipsoid":
                    size = f"  R={self.uset.radius(self.scales[id(md)], n):.4f}"
                    # Residual shape: reported so the Gaussian reading behind any
                    # chi^2 radius can be checked. Nothing branches on it.
                    sk, ku, sh = (r.get("resid_skew"), r.get("resid_kurtosis"),
                                  r.get("shapiro_p"))
                    if sk is not None:
                        extra += (f"  [resid skew={sk:+.2f} exkurt={ku:+.2f}"
                                  f" shapiro_p={sh:.3g}]")
                else:
                    size = f"  eps={self.uset.eps(self.scales[id(md)]):.4f}"
                if self.coherent and id(md) in self._excluded:
                    extra += "  [independent direction]"
                print(f"    [bank]   {c.name}: sd(y)={sd:.4f}{size}{extra}",
                      flush=True)

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
        """Label shift per MLModelData for draw ``b``; a pure function of (seed, b).

        Coherence is a **grouping**, not a global flag. Under ``coherent=True``
        every constraint outside ``coherent_exclude`` shares one standardized
        direction, scaled per outcome by its own radius, so the draw sits inside
        every outcome's own D while those models move together; each excluded
        constraint draws independently.

        The grouping is what the data supports. Measured on gastric (out-of-fold
        residuals, same rows/folds/frozen configs the bank uses, n=145 under
        forward-chaining): the non-DLT toxicity pairs correlate +0.28 on the
        percentile labels the bank perturbs (+0.22 on raw), while OS against every
        toxicity is +0.06 (+0.02 raw) -- indistinguishable from zero, and three of
        five pairs negative on raw. DLT's 0.44-0.80 row is excluded from that
        average because ``DLT_PROP = 1 - prod(1 - tox)`` makes it a deterministic
        function of the others (exact to 2e-16 over exactly the four modeled
        toxicities), so its correlation is construction, not evidence. Coherent
        asserts +1 and incoherent asserts 0; neither fits both blocks, and the
        record-level mismeasurement story that licenses coherence ("a study that
        under-reports adverse events under-reports across all toxicity endpoints")
        never covered survival anyway.

        **DLT is excluded from that average, not from the group.**
        ``coherent_exclude`` names OS alone, so the branch below hands
        ``dlt_constraint`` the shared ``u``: ``delta_dlt = R_dlt * u``, exactly
        collinear with the other four before clipping (measured +1.0000; the
        ``_clip_to_bounds`` call is the only decorrelator, leaving +0.94..+0.96 at
        rho=1 with ~20% of rows clipped, +0.97..+0.99 at rho=0.25). Two
        consequences: the group spends five outcomes' radius on four degrees of
        freedom, and the draw is not a *consistent* relabeling -- delta perturbs
        each outcome's percentile labels independently, so no delta in D leaves
        perturbed-DLT equal to ``1 - prod(1 - perturbed tox)``. The under-reporting
        story determines DLT's shift from the four components rather than leaving
        it free, so coherent overstates it; the sign is still right and only the
        magnitude is asserted. Fixing it means perturbing the four components,
        re-deriving DLT through the identity and re-percentiling -- a change here
        and in the label construction, not a config flip. See objection (4) in
        CLAUDE.md.
        """
        rng = np.random.RandomState(self.seed + b)
        out = {}
        shared_u = None
        for md in self._mds:
            n = len(md.y_train)
            if self.coherent and id(md) not in self._excluded:
                if shared_u is None or len(shared_u) != n:
                    shared_u = _draw_direction(n, self.uset, rng)
                u = shared_u
            else:
                u = _draw_direction(n, self.uset, rng)
            delta = self.uset.magnitude(self.scales[id(md)], n) * u
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
