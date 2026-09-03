"""C-MICL: conformal mixed-integer constraint learning (Ovalle et al. 2025).

The fourth uncertainty-aware method, and the only one that does **not** face the
shared set D. Where ``cp`` / ``wrapper`` / ``robust_reg`` are handed one D and
differ only in what they do with it, C-MICL builds its own tightening out of
held-out residuals: no relabelings, no bank, no radius.

Per learned constraint model, on that constraint's own training rows:

1. split the rows into PROPER-TRAIN and CALIBRATION (one shared split, see
   :func:`_split_rows`);
2. fit ``h`` on proper-train -- this is the model that gets embedded;
3. fit the WIDTH model ``u`` on proper-train, target ``|y - h(x)|``;
4. score the held-out calibration rows with
   ``s_i = |y_i - h(x_i)| / u(x_i)`` and take the split-conformal
   quantile ``q = s_(k)``, ``k = ceil((n_cal + 1) * (1 - alpha_eff))``;
5. embed ``h`` AND ``u``, and require the conservative end of the predictive
   interval to satisfy the constraint::

       sum_m [ w_m * h_m(x) + |w_m| * q_m * u_m(x) ]  <=  rhs

   which is ``sup { sum_m w_m f_m : f_m in [h_m - q_m u_m, h_m + q_m u_m] }``.

``alpha`` is the method's ONE dial, in the same role as CP's tau and the
wrapper's alpha: the miscoverage level of the predictive interval.

WHAT THIS IS AND IS NOT
-----------------------
- **The guarantee is marginal** over ``P_XY`` (and over the calibration draw): the
  interval covers ``y`` at a fresh exchangeable ``(x, y)`` with probability
  ``1 - alpha_eff``. It does NOT say ``x*`` is feasible. ``x*`` is an argmin, not
  a random draw, and it sits ON the constraint by construction -- exactly where a
  marginal statement says least (CLAUDE.md, Known gaps #8). Feasibility read off
  this method is the empirical consequence of that statement, not a delivered
  guarantee.
- **Exchangeability is assumed and is false on gastric**, whose folds are
  temporal by design (train = rows up to a cutoff year, val = the next). The
  split below is random *within* the fold's training rows, so coverage is
  marginal over the training years, not over the validation year. That is
  Known gap #6 and implementing the method does not make the assumption true.
- **h is fit on fewer rows than every other method's model.** Split conformal
  spends ``cal_frac`` of the training rows on calibration and does not give them
  back. Intrinsic to the method, not a handicap imposed here -- and the reason
  ``cal_frac`` is a structural setting rather than a second dial.
- **D plays no part.** On ``run_rho_sweep.py`` C-MICL's curve is FLAT IN RHO up to
  fold noise, by construction. That is why it is worth running there: it is the
  axis-free competitor the shared-D methods are read against, not another point
  on the same axis. Its own dial (alpha) is what moves it.

FIDELITY TO THE PAPER (arXiv:2506.03531, Sec. 4.1 and 5.1)
----------------------------------------------------------
Matched deliberately, so a disagreement with their Figure 1 is about the setting
and not about the recipe:

- the score is `s = |h(x_i) - y_i| / u(x_i)` with `u` trained on the ABSOLUTE
  RESIDUALS of `h` over the training rows (their two-step procedure) -- in-sample,
  as they specify;
- their quantile is `Quantile(s_1..s_N; ceil((N+1)(1-alpha))/N)` taken with
  numpy's `interpolation="higher"`, which lands ONE ORDER STATISTIC ABOVE the
  `s_(k)`, `k = ceil((N+1)(1-alpha))`, computed here -- see DIFFERENCES #1;
- their constraint is `[h(x) +- q u(x)] subset Y`, i.e. the whole interval must
  lie in the feasible region. For the reactor's lower bound `F >= 50` that is
  `h - q u >= 50`, which is exactly `w*h + |w|*q*u <= rhs` at `w = -1`,
  `rhs = -50`;
- `alpha = 0.1` is their main-text target (they report `alpha = 0.05` in an
  appendix), and it is pinned to `1 - feas_target` here for the same reason;
- `u` is a **32x32 ReLU MLP with a LINEAR output**, the same architecture for
  every base model, and it is **not selected** -- see WIDTH MODEL below;
- the width is **not floored**, and the MIP instead requires `u(x) >= 0` -- see
  WIDTH MODEL;
- `cal_frac = 0.2` is their 80/20 split, and their `n_cal` is close to ours:
  their sheet is **2000 rows** (`data/unscaled_noisy_reactor_data.xlsx`,
  `A1:F2001`), `regression.py:571` throws half of it away as `X_unseen` --
  which is **never read again anywhere in the file** -- and splits 80/20 within
  the rest, so `h` sees **800** rows and **`n_cal = 200`**. Ours is a CV fold of
  a 1000-row design: 900 fold-train rows, so `h` sees 720 and `n_cal = 180`,
  which is what sets the reactor's alpha floor (`CMICL_ALPHA_GRID_EXTRA` in
  `run_dial_sweep.py`).

**Not matched, and it is the protocol, not the method**: they average empirical
ground-truth feasibility over **100 randomly sampled cost vectors** with the
calibration set held fixed. This repo fixes `cost_vector` at ones on purpose (a
new `c` is a different problem, see `src/data/generate.py`), so our rate is over
**training/calibration draws at one `c`**. Both are "empirical ground-truth
feasibility"; they are not the same average, and a gap between the two numbers
should be read with that in front. Their `c` is now known -- `U(-4, 4)` with
negative components divided by 10, over their scaled variables
(`regression.py:713-719`) -- and `probe_cmicl_cost_sampling.py --schemes paper`
measures it.

**Verified against their code** --
`https://github.com/dovallev/c-micl` (commit `b44fe53`, 2026-07-14),
`regression.py` -- not inferred from the text. The instance is theirs to the
digit: identical
variable order `(v0, v_He, T, dt, L)`, identical box, all seven domain
constraints identical once their `/100` input scaling is undone, and our
vendored ODE reproduces their labels to a **-1.37% mean** offset with a residual
sd of 1.94 that is their own label noise (`reactor.noise_std: 2.0`).

DIFFERENCES FROM THEIR CODE
---------------------------
Audited 2026-09-03 against `github.com/dovallev/c-micl` at commit `b44fe53`,
files `regression.py` and `notebooks/regression/03_feasibility_verification.ipynb`.
Every difference found is below; none is a disagreement about the recipe, and
only the first three move `q` or `h`.

1. **`q` is one order statistic lower here.** `regression.py:669` is the widely
   copied conformal idiom, `np.quantile(s, ceil((n+1)(1-alpha))/n,
   interpolation="higher")`. numpy puts a quantile at virtual index `p*(n-1)`
   and `"higher"` rounds that up, so at `p = k/n` it returns `s_(k+1)` rather
   than the `s_(k)` the level names: at `n=200, alpha=0.1` theirs is `s_(182)`
   where `conformal_quantile` gives `s_(181)`. Theirs is conservative by one row
   and BOTH are valid -- a larger `q` only over-covers -- and the gap in `q`
   shrinks like 1/n. Kept at `s_(k)` because that is the order statistic the
   guarantee is stated over; changing it would move every `cmicl` number again.
2. **`h`'s hyperparameters are hand-fixed there, CV-selected here.** Their
   `models_and_params` (`regression.py:585-621`) gives each family exactly one
   grid point -- LinearDT `max_depth 5 / min_samples_split 10 / max_bins 40`;
   RF `15 trees / depth 5 / min_samples_split 3 / max_features 0.6`; GBM
   `15 trees / lr 0.2 / depth 5 / min_samples_split 5 / max_features 0.6`; MLP a
   Keras `32-32-1` (ReLU hidden, linear output, Adam lr 1e-3, batch 32, 2000
   epochs, L2 0.01 on the hidden kernels only) -- and they report all four
   surrogate families separately. Ours is one model per constraint out of
   `results/cv/*_selected_configs.json`: on the reactor sklearn
   `MLPRegressor((10, 5, 2), solver="lbfgs", alpha=0.01)` inside
   `Pipeline(StandardScaler, ...)`. Different family, depth, optimizer and
   feature scaling -- so `h` is not their `h` even where both are "an MLP".
3. **`u` is their architecture under a different trainer.** `(32,32)`,
   `alpha=0.01`, `max_iter=2000` reproduces their grid point, but sklearn's
   `MLPRegressor` is not Keras: `alpha` penalises EVERY layer where theirs
   penalises only the hidden kernels, the default batch is `min(200, n)` against
   their 32, `max_iter` is a CAP that `tol=1e-4` / `n_iter_no_change=10`
   normally stops early against their fixed 2000 epochs, and
   `train_model(normalize=True)` standardises the features where they hand the
   net their own `/100` scaling. Same shape of `u`, a differently fitted `u`.
4. **Units.** They train and optimize on scaled columns -- inputs `/100` except
   `dt`, labels `/10`, hence the floor `min_req = 5.0` and a cost vector that
   multiplies scaled `x` (`regression.py:563-565`, `:724`). We stay in physical
   units with `rhs = 50`. `q` is a ratio and carries over; big-M widths, the
   net's conditioning and the meaning of a cost coefficient do not.
5. **Their MIP also forces `h(x) >= 0`** (`m.y_f` is `NonNegativeReals`,
   `regression.py:485`); ours leaves `h` free. Inert on the reactor, where
   `h >= 50 + q u >= 0` already, and it would bite only on a constraint with a
   non-positive rhs. Their own notebook declares `y_f` free, so this one is not
   consistent even inside their repo.
6. **Their big-M box is the data hull of whichever split trained the model** --
   `np.min/np.max` over `X_train` for C-MICL but over `X_rest` for MICL
   (`regression.py:697-707`) -- and `x` carries no bound of its own beyond
   `NonNegativeReals`. Ours is `DECISION_RANGES` for every method, so their
   C-MICL and MICL do not optimize over the same box and ours do. Numerically it
   is a rounding difference on this instance: their sampled hull matches
   `DECISION_RANGES` to under 0.1% per face. It is a difference in what defines
   the box, not in where the box is.
7. **Solver settings**: theirs `MIPGap 0.01`, `Threads 8` (`:734-735`); ours the
   single `optimization.mip_gap` every method here shares, which is what keeps
   the objective column comparable across methods (CLAUDE.md, Conventions).
8. **Their notebook is a third implementation**, not a demo of the script:
   `notebooks/regression/03` floors the width at `np.maximum(u, 1e-6)` in the
   SCORE while still embedding `u >= 0`, trains both nets 300 epochs on Keras
   defaults, and leaves `y_f` free. We follow the script -- it is what produced
   the paper's numbers.
9. **Neither implementation is Mondrian.** The paper's guarantee rests on a
   Mondrian (group-conditional) conformal set plus conditional independence of
   coverage and feasibility; `regression.py` calibrates ONE global quantile, and
   their notebook says as much ("this notebook's simplified (non-Mondrian)
   conformal set"). `conformal_quantile` here is global too, so this gap is
   SHARED -- worth naming, because it is a reason an empirical rate can sit
   below `1 - alpha` without either implementation being wrong.

**Ours, with no counterpart in their code** (this repo's problems need them):
the `multiplicity="bonferroni"` joint level, the label-range clip and its
vacuous-constraint drop, one calibration split shared across a problem's
outcomes, `q = inf` refused instead of a numpy error when `k > n_cal`, and alpha
walked as a DIAL rather than fixed at `{0.1, 0.05}`.

WIDTH MODEL -- THEIRS, NOT OURS
-------------------------------
``u`` is fit in-sample on ``h``'s own proper-train residuals, which biases it
small. That costs nothing in validity -- split conformal keeps its coverage for
ANY deterministic score function, and ``q`` absorbs the bias globally -- but it
means ``u`` shapes the half-width better than it sizes it.

The architecture is **fixed at 32x32, never cross-validated**, and that is
faithful rather than lazy: their ``train_model(..., cv=True)`` searches a grid of
**exactly one point** (``hidden_layer_sizes [(32,32)], alpha [0.01], epochs
[2000]``), so the five folds it fits select nothing and the deployed ``u`` is a
single net refit on all proper-train rows. Every grid in their file is a single
point, ``h``'s included, so our CV-selected ``h`` is the *more* tuned of the two.

**A ReLU NN with a LINEAR output layer** (``regression.py:96``,
``model.add(Dense(1))`` with no activation), so ``u`` can and does predict
NEGATIVE widths. Neither of the two guards this repo used to apply is theirs, and
the three designs fail in different directions:

- **theirs (what runs here now)**: score ``s = |y - h(x)| / u(x)``, unguarded, so
  a negative ``u`` gives a NEGATIVE score which sorts to the *bottom* and can
  never be the binding order statistic -- the row's information is silently
  dropped and ``q = s_(k)`` comes out **lower** than it should, biasing coverage
  DOWN. In the MIP ``u_eff`` is a NonNegativeReals variable set *equal* to the
  network output (``regression.py:484``, ``:535``), which is not a clamp: it
  makes every ``x`` with ``u(x) < 0`` **infeasible**, restricting the decision
  space to wherever the width model happens to be non-negative.
- **their notebook** (``notebooks/regression/03``): floors at an absolute
  ``1e-6``, a divide-by-zero guard only, so a near-zero ``u`` still explodes the
  score and inflates ``q`` for every well-fit row.
- **this repo before 2026-09-03**: floored at 5% of the mean absolute
  proper-train residual, applied identically in the score and in the MIP. Valid
  (one deterministic score function in both places) and every ``x`` stayed
  feasible, but the floored rows produced huge scores that biased ``q`` **UP** --
  measured on the reactor at 32x32: min ``u = -5.37``, 26% of calibration rows on
  the floor, ``q = 27.6``, mean half-width **3.9 sd(y)** on labels of 11-70.

Note what the first bullet costs THEM: the score function used to calibrate
(raw ``u``, negatives allowed) is not the one deployed in the MIP (restricted to
``u >= 0``), and split conformal's guarantee wants the same deterministic score
in both places. That is a real gap in the reference implementation and it is
reported, not silently patched -- ``calibrate_conformal_model`` counts the
negative-``u`` calibration rows and prints a warning when there are any.

**``u_eff == u(x)`` with ``u_eff >= 0`` lets the optimizer WALK THE TIGHTENING
OFF, and on the reactor it does so immediately** (measured 2026-09-03, the first
run under these semantics). The tightening is ``q * u_eff``, which the optimizer
wants SMALL, and the only thing stopping it at zero is where the width model
happens to vanish. Since a linear output crosses zero -- 11/200 calibration rows
predict ``u <= 0`` here, so the zero set cuts through the box -- the MIP parks
``x*`` exactly on it: ``u(x*) = 1e-15`` on **every** solve, and ``q = 6.08`` is
embedded and unused. Measured feasibility 0.00.

**This is NOT "C-MICL becomes nominal".** At ``u(x*) = 0`` the *conformal* part
of the constraint is gone -- what binds is ``h(x) >= 50``, nominal's constraint
-- but ``u(x) >= 0`` is an ADDITIONAL constraint nominal does not have, and
``x*`` sits **on its boundary**, so it is active, not vacuous. What runs is
"nominal on ``h``, intersected with ``{x : u(x) >= 0}``": a strictly smaller
feasible set, cut by where a residual regression happens to cross zero. Measured
(``--n-instances 3 --schemes fixed_ones paper``, alpha=0.1, seed 42):

    scheme/inst   objective vs nominal   F_ODE(x*)   nominal F_ODE
    fixed_ones 0       +3.48%              47.70        45.85
    paper 0           +11.21%              46.50        45.87
    paper 1            +3.17%              49.15        45.87
    paper 2            +2.27%              49.15        45.87

So the restriction **costs 2-11% of objective and buys 0.6-3.3 units of truth**,
landing short of the floor of 50 -- a weak tightening, not an absent one.

**And it is arbitrary rather than accidentally protective.** The tempting reading
is that ``u < 0`` marks an extrapolation region, so excluding it is an unintended
trust region. Measured on the calibration rows, it is not: the 11 rows with
``u <= 0`` have a mean TRUE ``|residual|`` of **1.900** against **1.981** for the
189 with ``u > 0``, and sit no nearer the box faces in any load-bearing sense
(mean distance to the nearest face 0.059 vs 0.091, and the ``u > 0`` set contains
a row at 0.0000). The sign of ``u`` carries no information about the error there;
it is where a mis-specified regression undershoots an ordinary residual past
zero. Caveat: that characterises the negative region **near the data**, while
``x*`` sits on the ``u = 0`` surface, which may be elsewhere in the box.

The old floor was not only a numerical guard -- ``u_eff >= floor > 0`` is what
made the tightening un-escapable. Their formulation has the same hole; whether it
binds depends on whether their fitted ``u``'s zero set intersects the region
where ``h >= 50``, which on this instance ours does. Reporting this rather than
re-adding the floor is deliberate: the method under test is theirs.

MULTIPLICITY
------------
``multiplicity="bonferroni"`` divides alpha by the number of learned constraints
so the JOINT statement holds at ``1 - alpha``; ``"none"`` (default, and what
Ovalle et al. report) leaves each constraint marginal at ``1 - alpha``. On
gastric's five toxicity constraints the Bonferroni level is often finer than the
calibration set can resolve (``ceil((n_cal + 1) * (1 - alpha/C)) > n_cal``), and
there is then no finite quantile to embed. That is reported as an infeasible
solve with the reason printed, never clipped to the largest score.
"""

import math
import time
from typing import Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.data.generate import ProblemInstance
from src.methods.nominal import (
    DEFAULT_MIP_GAP,
    SolutionResult,
    resolve_constraint_config,
    build_decision_vars,
    add_problem_constraints,
    build_and_set_objective,
    model_X_ref,
)
from src.models.train import train_model
from src.models.embed import embed_model


def _split_rows(n: int, cal_frac: float, seed: int):
    """One (proper-train, calibration) split of ``n`` rows, SHARED across outcomes.

    Gastric's outcomes are five labels on the same patients, so splitting each
    one independently would put a patient in the fit for one toxicity and in the
    calibration set for another: no single set of trials would be held out from
    the problem as a whole, and the joint statement would be about nothing in
    particular. Keying the permutation on ``(seed, n)`` alone gives every outcome
    of the same length the same split.
    """
    n_cal = int(round(cal_frac * n))
    n_cal = max(1, min(n - 1, n_cal))
    perm = np.random.default_rng(seed + 1000 * n).permutation(n)
    return perm[n_cal:], perm[:n_cal]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile ``s_(k)``, ``k = ceil((n + 1) * (1 - alpha))``.

    Returns ``inf`` when ``k > n``: the calibration set is too small to certify
    that level, so no finite tightening carries the guarantee. Clipping to
    ``max(scores)`` instead would name a level the data cannot support, which is
    why the caller fails the solve rather than pretending.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(s[max(k, 1) - 1])


def calibrate_conformal_model(X, y, model_type, model_params, *,
                              alpha, cal_frac, seed,
                              width_model_type=None, width_model_params=None,
                              label=""):
    """Fit ``h``, fit the width model ``u``, calibrate ``q``.

    Returns ``(h, u, q, info)``. ``h`` sees the proper-train rows ONLY --
    holding the calibration rows out of it is what makes their scores
    exchangeable with a fresh point's.

    The score is ``|y - h(x)| / u(x)`` with ``u`` used RAW, which is Ovalle et
    al.'s ``regression.py:665`` exactly -- no floor, no ``abs``, no epsilon. A
    negative ``u`` therefore yields a negative score that sorts below every
    honest one and can never be the binding order statistic, so ``q`` comes out
    too small. That is a property of the reference implementation, not a choice
    made here, and the ``n_neg_u`` diagnostic is how it stays visible.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    tr_idx, cal_idx = _split_rows(len(y), cal_frac, seed)

    h = train_model(X[tr_idx], y[tr_idx], model_type, model_params)

    resid_tr = np.abs(y[tr_idx] - h.predict(X[tr_idx]))
    u = train_model(X[tr_idx], resid_tr,
                    width_model_type or model_type,
                    width_model_params if width_model_type else model_params)

    u_cal = np.asarray(u.predict(X[cal_idx]), dtype=float)
    n_neg_u = int(np.sum(u_cal <= 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.abs(y[cal_idx] - h.predict(X[cal_idx])) / u_cal
    # A 0/0 row would break the sort; an exact-zero width is an infinite score,
    # which `conformal_quantile` can order and the caller can reject.
    scores = np.where(np.isnan(scores), np.inf, scores)
    q = conformal_quantile(scores, alpha)

    half = q * float(np.mean(u_cal)) if np.isfinite(q) else float("inf")
    sd_y = float(np.std(y))
    info = {
        "n_train": int(len(tr_idx)), "n_cal": int(len(cal_idx)),
        "alpha_eff": float(alpha), "q": q,
        "mean_width": float(np.mean(u_cal)), "min_width": float(np.min(u_cal)),
        "n_neg_u": n_neg_u, "mean_half_width": half, "sd_y": sd_y,
    }
    print(
        f"    [cmicl] {label}: n_fit={info['n_train']} n_cal={info['n_cal']} "
        f"alpha_eff={alpha:.4g} q={q:.4g} mean_u={info['mean_width']:.4g} "
        f"-> mean half-width {half:.4g} ({half / max(sd_y, 1e-12):.2f} sd(y))",
        flush=True,
    )
    if n_neg_u:
        print(
            f"    [cmicl] WARNING {label}: {n_neg_u}/{len(cal_idx)} calibration "
            f"rows have u <= 0 (min {info['min_width']:.4g}), so their scores are "
            f"negative and sink below the quantile -- q is biased DOWN and the "
            f"MIP's u >= 0 constraint will exclude part of the decision space. "
            f"This is Ovalle et al.'s behaviour (regression.py:665, :484); their "
            f"linear output layer is what allows it.",
            flush=True,
        )
    return h, u, q, info


def solve_cmicl(instance: ProblemInstance,
                model_type: str = "rf",
                model_params: dict = None,
                alpha: float = 0.1,
                cal_frac: float = 0.25,
                width_model_type: Optional[str] = None,
                width_model_params: Optional[dict] = None,
                multiplicity: str = "none",
                seed: int = 42,
                rho: float = 0.0,
                robustify_objective: bool = False,
                mip_gap: float = DEFAULT_MIP_GAP) -> SolutionResult:
    """Ovalle et al.'s conformal MICL: embed the conservative end of a
    split-conformal predictive interval instead of the point prediction.

    ``alpha`` is the single dial. ``cal_frac``, ``width_*`` and ``multiplicity``
    are structural settings with one production value each, in the same sense as
    CP's ``separation``: they are not swept.

    ``robustify_objective`` (default ``False``, matching CP and the wrapper)
    leaves the learned objective a plain nominal fit on ALL training rows, so the
    objective column stays comparable across methods. Under ``True`` the
    objective term takes its own conformal bound, in whichever direction makes it
    worse, and is then fit on the proper-train rows like every constraint model.
    """
    start = time.time()
    if multiplicity not in ("none", "bonferroni"):
        raise ValueError(f"unknown multiplicity: {multiplicity!r}")

    constraint_idxs = [
        c_idx for c_idx, c in enumerate(instance.constraints)
        if not any(md.obj_weight != 0 for md in c.models_data)
    ]
    n_units = max(len(constraint_idxs), 1)
    alpha_eff = (alpha / n_units) if multiplicity == "bonferroni" else alpha

    bonf = (f" -> alpha_eff={alpha_eff:.4g} over {n_units} constraints"
            if multiplicity == "bonferroni" else "")
    print(
        f"    [cmicl] Calibrating (alpha={alpha:g}, "
        f"multiplicity={multiplicity}{bonf}, cal_frac={cal_frac:g})...",
        flush=True,
    )

    # ---- calibrate one (h, u, q) per constraint model ---------------------
    fitted = {}          # id(model_data) -> (h, u, q); u None = nominal
    warned_multi_clip = False
    config_idx = 0
    for constraint in instance.constraints:
        is_obj = any(md.obj_weight != 0 for md in constraint.models_data)
        for model_data in constraint.models_data:
            md_id = id(model_data)
            m_type, m_params = resolve_constraint_config(
                instance, config_idx, model_type, model_params)
            config_idx += 1
            if md_id in fitted:
                continue
            if is_obj and not robustify_objective:
                # Nominal objective, fit on ALL rows: the split belongs to the
                # conformal CONSTRAINT machinery, and shrinking the objective fit
                # too would move the objective column for a reason that has
                # nothing to do with the method being compared.
                fitted[md_id] = (
                    train_model(model_data.X_train, model_data.y_train,
                                m_type, m_params),
                    None, 0.0,
                )
                continue
            h, u, q, _info = calibrate_conformal_model(
                model_data.X_train, model_data.y_train, m_type, m_params,
                alpha=alpha_eff, cal_frac=cal_frac, seed=seed,
                width_model_type=width_model_type,
                width_model_params=width_model_params,
                label=constraint.name,
            )
            if not np.isfinite(q):
                print(
                    f"    [cmicl] INFEASIBLE: {constraint.name} needs "
                    f"ceil((n_cal+1)*(1-{alpha_eff:.4g})) <= n_cal calibration "
                    f"rows and has {int(round(cal_frac * len(model_data.y_train)))}. "
                    f"Raise alpha, raise cal_frac, or drop "
                    f"multiplicity='bonferroni'.",
                    flush=True,
                )
                return SolutionResult(
                    x_opt=np.zeros(instance.n_features), obj_value=np.inf,
                    status="infeasible", models_embedded=0,
                    solve_time=time.time() - start,
                )
            fitted[md_id] = (h, u, q)

    # ---- build the MIP ----------------------------------------------------
    opt = gp.Model("cmicl")
    opt.Params.OutputFlag = 0
    opt.Params.MIPGap = mip_gap
    opt.Params.MIPFocus = 1

    x = build_decision_vars(opt, instance)
    models_embedded = 0
    obj_terms = []
    n_vacuous = 0

    def _embed(ml_model, prefix, c_idx, m_idx):
        nonlocal models_embedded
        models_embedded += 1
        return embed_model(
            opt, ml_model, x, instance.variable_lb, instance.variable_ub,
            name_prefix=prefix, rho=rho,
            X_ref=model_X_ref(instance, c_idx, m_idx),
        )

    def _width_var(u_model, prefix, c_idx, m_idx):
        """``u(x)`` held in a NONNEGATIVE variable, by EQUALITY.

        This is Ovalle et al.'s formulation verbatim (``regression.py:484``,
        ``m.y_u = pyo.Var(within=pyo.NonNegativeReals)``, and ``:535``,
        ``m.y_u == m.u_surrogate.outputs[0]``). It is NOT a floor: an equality
        against a non-negative variable makes every ``x`` where the width model
        predicts ``u(x) < 0`` **infeasible**, so the decision space is restricted
        to wherever ``u`` happens to be non-negative. On an instance where ``u``
        goes negative over part of the box that is a real, unannounced constraint
        on ``x`` -- and it is also why a C-MICL solve can come back infeasible
        with a perfectly finite ``q``.

        The repo's own former alternative -- ``u_eff >= u(x)`` with ``lb=floor``,
        which left every ``x`` feasible and clamped the width instead -- is gone
        on purpose (2026-09-03): one implementation of this method, and it is
        theirs.
        """
        u_raw = _embed(u_model, prefix, c_idx, m_idx)
        u_eff = opt.addVar(lb=0.0, name=f"{prefix}_ueff")
        opt.addConstr(u_eff == u_raw, name=f"{prefix}_ueq")
        return u_eff

    for c_idx, constraint in enumerate(instance.constraints):
        is_obj = any(md.obj_weight != 0 for md in constraint.models_data)
        single = len(constraint.models_data) == 1

        if is_obj:
            for m_idx, model_data in enumerate(constraint.models_data):
                h, u, q = fitted[id(model_data)]
                a = model_data.obj_weight * model_data.weight
                h_var = _embed(h, f"cmicl_obj_c{c_idx}_m{m_idx}", c_idx, m_idx)
                if u is None:
                    obj_terms.append(a * h_var)
                else:
                    u_eff = _width_var(u, f"cmicl_objw_c{c_idx}_m{m_idx}",
                                       c_idx, m_idx)
                    obj_terms.append(a * h_var + abs(a) * q * u_eff)
            continue

        terms = []
        vacuous = False
        for m_idx, model_data in enumerate(constraint.models_data):
            h, u, q = fitted[id(model_data)]
            w = model_data.weight
            h_var = _embed(h, f"cmicl_c{c_idx}_m{m_idx}", c_idx, m_idx)
            u_eff = _width_var(u, f"cmicl_w_c{c_idx}_m{m_idx}",
                               c_idx, m_idx)
            terms.append(w * h_var + abs(w) * q * u_eff)

            # The interval is intersected with the label range -- the conformal
            # analogue of uncertainty.clip_labels. For a SINGLE-model constraint
            # that clip is one of exactly two things, and neither needs a binary:
            # the worst clipped value is w*hi (w > 0) or w*lo (w < 0), so either
            # it already satisfies the rhs -- the constraint is vacuous and is
            # dropped -- or the clip cannot bind and the raw bound is exactly
            # equivalent. Every instance in this repo has one model per
            # constraint. With more than one the exact clip needs a per-term
            # min/max; the raw bound stays VALID there (it dominates the clipped
            # one), only more conservative, so it is used and said so.
            if model_data.label_bounds is not None:
                lo, hi = model_data.label_bounds
                worst_clipped = w * hi if w > 0 else w * lo
                if single:
                    if worst_clipped <= constraint.rhs:
                        vacuous = True
                elif not warned_multi_clip:
                    warned_multi_clip = True
                    print(
                        f"    [cmicl] NOTE: {constraint.name} combines several "
                        f"models with label_bounds; the interval clip is dropped "
                        f"(conservative) rather than encoded exactly.",
                        flush=True,
                    )
        if vacuous:
            # min(h + q u, hi) <= rhs holds at every x: the label range alone
            # satisfies it. Adding the raw bound instead would impose a
            # constraint the method does not make.
            n_vacuous += 1
            continue
        opt.addConstr(gp.quicksum(terms) <= constraint.rhs,
                      name=f"cmicl_constr_{c_idx}")

    add_problem_constraints(opt, x, instance)
    build_and_set_objective(opt, x, instance, obj_terms)
    opt.update()
    vac = (f", {n_vacuous} constraint(s) vacuous under the label range"
           if n_vacuous else "")
    print(
        f"    [cmicl] MIP built (alpha_eff={alpha_eff:.4g}, "
        f"{models_embedded} models embedded{vac}; "
        f"{opt.NumVars} vars / {opt.NumConstrs} constrs); solving...",
        flush=True,
    )
    opt.optimize()
    elapsed = time.time() - start

    if opt.Status == GRB.OPTIMAL:
        return SolutionResult(
            x_opt=np.array([v.X for v in x]),
            obj_value=opt.ObjVal,
            status="optimal",
            models_embedded=models_embedded,
            solve_time=elapsed,
            opt=opt,
            x=x,
        )
    return SolutionResult(
        x_opt=np.zeros(instance.n_features),
        obj_value=np.inf,
        status="infeasible",
        models_embedded=models_embedded,
        solve_time=elapsed,
        opt=opt,
        x=x,
    )
