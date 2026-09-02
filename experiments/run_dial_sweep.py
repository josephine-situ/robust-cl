"""PRIMARY axis: every method walked along its OWN dial at a FIXED rho.

This is the headline experiment. ``run_rho_sweep.py`` is now supporting evidence:
it answers "how much assumed uncertainty does each method absorb", which is a
question about D, while the question the contribution actually has to answer is
"at equal held-out feasibility, whose decisions are better". That one is read off
a curve in **objective x feasibility** space, and each method reaches its points
by moving the dial it really has -- not by being dragged along a rho axis two of
them do not face at all.

The grid::

    method      dial      rho columns                       notes
    ---------------------------------------------------------------------------
    cp          tau       gastric {0.5, 1.0}, reactor {2,3}  ONE fixed tau grid
    wrapper     alpha     same                               P is a bank prefix
    margin      m         --                                 faces no D
    cmicl       alpha     --                                 alpha=0.1 = protocol
    nominal     none      --                                 one reference point

``robust_reg`` is deliberately **absent**. Its dial ``label_eps`` IS D's radius,
so at a fixed rho it has no dial to move -- there is no curve for it on this axis.
It keeps its place on the rho sweep, where the axis is the quantity it actually
walks. Asking for it here is an error rather than a silent flat line.

**Both rho columns share one output CSV.** The ``rho`` column already tells them
apart, and one file per (problem, coherence, seed) is what the plot reads. The
checkpoint cell key is ``(method@rho, dial)``, so resume is unchanged.

Four things this file exists to get right, in the order they had to be fixed:

1. **Per-context records.** ``{problem}_dial_contexts{cell}.csv`` carries
   ``(fold, context_idx, solved, feasible, objective)`` for every cell. Primary
   scoring is unchanged -- each cell is still scored conditional on its OWN solved
   contexts, and that independence is load-bearing. But the objective is now the
   deliverable rather than a side column, and a conditional mean of it rewards
   whichever cell solved least. The same-cohort comparison is derivable from these
   rows afterwards; making it primary would couple every cell to every other.

2. **One bank per (rho column, fold).** A bank is a pure function of
   ``(instance, D, seed, B)``: neither tau nor alpha reaches it. So ONE bank of
   B=200 serves CP's whole tau grid and the wrapper's whole alpha grid -- the
   wrapper's P models are a prefix of CP's B. Gastric drops from ~14 bank builds
   per rho column to 1 per fold. See :class:`src.methods.cv_calibrate.FoldCache`.

3. **tau is FIXED BEFORE THE RUN.** One absolute grid (``TAU_GRID``), the same on
   every rho column and every problem, in unexplained-sd units. tau is a
   PARAMETER OF THE METHOD, set in advance like rho and like the margin's m -- it
   is never read back off the run, and in particular never placed from an
   iteration-0 separation distance. Doing that made tau a function of the bank,
   of B and of which folds were probed, so the same nominal tau meant a different
   tolerance in every cell and the primary figure's axis stopped being one
   quantity. Whether the top of the grid happens to stop before any cut is a
   PROPERTY OF THE RUN, reported by the ``[cp] ... max iter-0 dist=`` line, not
   something the grid is bent to guarantee.

4. **The grid is SEARCHED, not walked** (``--search adaptive``, the default).
   A dial grid is ordered by robustness, and two of its answers say where NOT to
   look: a cell scoring feasibility 0 means nothing LESS robust can deliver, and
   a cell below ``--min-solved`` means nothing MORE robust can solve. So the
   cells that deliver the target at an acceptable solved fraction are an interval
   of that order, and its least-robust end is the protocol point (robustness is
   what the objective pays for). The search bisects to bracket that end, then
   spends the rest of ``--max-evals`` filling the band around it -- the
   deliverable is a CURVE, so resolution goes where the frontier bends and none
   of it into the two dead tails. That let the grids get FINER at lower cost
   (``TAU_GRID`` is half-decades where it was decades) and then WIDER at lower
   cost again (2026-08-27: every grid now spans its dial's full usable range,
   because bracketing is O(log n) and the dead tails are pruned unscored). Every
   old value is still in every grid, so an existing checkpoint resumes rather
   than being orphaned. Widening was not cosmetic -- on the 2026-08-27 curves the
   reactor's margin, the reactor's CP and gastric's wrapper each had their
   protocol point pinned at ``bound="grid_end"``, and gastric's C-MICL had no
   protocol point at all because it first solves above the old grid top.

   This ASSUMES the dial is monotone, which the rho axis is not always (CP's dip
   at rho=0.5 on gastric; robust_reg's feasibility falling with rho) and which
   has never been tested on the dials. So: violations among the cells actually
   scored are printed and land in the star table's ``monotone_note``; every
   unscored cell is written to ``{problem}_dial_skipped{cell}.csv`` with the
   reason, never into the curve as a row of NaNs (the plot reads a non-finite
   feasibility as "no solution on any fold", which is a RESULT); and
   ``--search grid`` walks the whole grid, assuming nothing.

Ablation (``--cp-alpha-ablate``): sweep tau, read tau* at the target, hold tau*
and walk ``cp_alpha``. It asks whether relaxing the coverage cap lifts CP's
feasibility ceiling and what that costs in solved fraction. One rho column only
(``--cp-alpha-rho``; gastric 1.0, reactor 2.0). **Structurally inert on
synthetic and the reactor** -- both take CP's basic separation path, which has no
protected-anchor test to relax -- and the runner says so and skips rather than
producing a flat curve that looks like a measurement.

Outputs, all scoped by the same cell suffix ``run_rho_sweep`` uses
(``_coh``/``_incoh``[``_matchbank``][``_f<n>``][``_m<model>``][``_s<seed>``]):

  ``{problem}_dial_scores{cell}.csv``   resume checkpoint, keyed (method@rho, dial)
  ``{problem}_dial_contexts{cell}.csv`` per-context records
  ``{problem}_dial_curve{cell}.csv``    THE primary output; what the plot reads
  ``{problem}_dial_star{cell}.csv``     derived: each series' protocol point
  ``{problem}_dial_skipped{cell}.csv``  cells the search did not score, and why
  ``{problem}_cp_alpha{cell}.csv``      the coverage-cap ablation

Usage::

    python experiments/run_dial_sweep.py --problem gastric
    python experiments/run_dial_sweep.py --problem reactor --rho-columns 1 2 3 4
    python experiments/run_dial_sweep.py --problem gastric --coherent
    python experiments/run_dial_sweep.py --problem gastric --cp-alpha-ablate
    python experiments/run_dial_sweep.py --problem reactor --search grid
    python experiments/plot_dial_sweep.py --problem gastric --suffix _incoh
"""

import argparse
import dataclasses
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods.cv_calibrate import (
    FoldCache, cv_score_knob, load_detail_checkpoint, append_score,
    append_contexts, append_judge,
)
from src.data.instances import load_config
from experiments.run_rho_sweep import (
    _setup_synthetic, _setup_reactor, _setup_gastric, _variant_suffix,
    _bank_seed, _synth_n_folds,
)

OUT_DIR = "results/rho_sweep"

# Which dial each method is walked along at a fixed rho, and whether it faces D.
#
#   cp / wrapper  face D. One rho column is one shared set; within it the two are
#                 separated only by what they DO with it, which is the comparison.
#   margin        the RHS shift m. Faces no D, so it is scored once per problem
#                 and its curve is the same one against every rho column.
#   cmicl         the conformal miscoverage alpha. Faces no D either; alpha=0.1 is
#                 the PROTOCOL point (the conformal level and the feasibility
#                 target are the same quantity) and is flagged as such -- the rest
#                 of its grid is there to find where it first solves, not to be
#                 chosen from.
#   nominal       nothing to move: one reference point.
DIAL = {"cp": "tau", "wrapper": "alpha", "margin": "margin",
        "cmicl": "alpha", "nominal": None}
FACES_D = ("cp", "wrapper")
# The keyword each D-facing solver takes a prebuilt ScenarioBank under.
BANK_KWARG = {"cp": "cp_bank", "wrapper": "bank"}

# rho columns per problem. Two per problem: enough to show whether the ordering of
# the methods is a property of the methods or of the assumed radius, without
# doubling into a rho sweep by the back door.
#   gastric  0.5 / 1.0 -- where CP's committed curve is strongest, and the point
#            below it. Both are headline columns now, not sensitivity checks.
#   reactor  2 / 3 -- MEASURED, not guessed. rho=1 and 2 were the original pair
#            because nominal misses the benzene target by ~4 units of F and
#            rho=1 buys ~2.2; the 2026-08-27 run at {3, 4} then showed rho=3 is
#            already past the edge (CP delivers feasibility 0.9 at the LOOSEST
#            tau on the grid and 1.0 everywhere below it), so 4 only buys
#            objective CP is not being asked to pay. 2/3 brackets the transition
#            instead of sitting above it. --rho-columns overrides.
DEFAULT_RHO_COLUMNS = {"gastric": [0.5, 1.0], "reactor": [2.0, 3.0],
                       "synthetic": [0.5, 1.0]}

# PER-METHOD overrides of the above, for the case where one method's usable range
# simply is not the shared one. Only entry: the reactor's wrapper.
#
# The shared {2, 3} brackets CP's transition, but the wrapper never gets there --
# it is out of DIAL at alpha=0 (where all P=20 models must hold; there is no
# stricter level) at feasibility 0.10 on rho=2 and 0.40 on rho=3. Its alpha=0 end
# only clears the 0.9 target at rho=6, measured on the 2026-08-28 {4,5,6} probe:
# 0.50 / 0.70 / 0.90 as rho goes 4 / 5 / 6. So on the shared column the wrapper
# has no dial* at all, and reporting it there measures the column rather than the
# method. {5, 6} brackets ITS transition the way {2, 3} brackets CP's -- and the
# comparison the deliverable makes is at equal FEASIBILITY, not at equal rho, so
# the two methods sitting on different columns is the point rather than a
# confound. The cost of that capacity is what the curve then shows.
#
# --rho-columns overrides BOTH: it puts every method on the given columns, which
# is what a span probe (`--rho-columns 1 2 3 4`) wants.
METHOD_RHO_COLUMNS = {"reactor": {"wrapper": [5.0, 6.0]}}


def _rho_columns_for(problem, methods, override):
    """``{method: [rho, ...]}`` for the D-facing methods, and their sorted union.

    ``override`` (``--rho-columns``) wins for every method; otherwise a method
    takes its ``METHOD_RHO_COLUMNS`` entry if it has one and the problem default
    if it does not. The union is what the run loops over -- one bank per rho,
    shared by whichever methods sit on it.
    """
    if override:
        cols = [float(r) for r in override]
        by_method = {m: list(cols) for m in methods}
    else:
        shared = [float(r) for r in DEFAULT_RHO_COLUMNS[problem]]
        per = METHOD_RHO_COLUMNS.get(problem, {})
        by_method = {m: [float(r) for r in per.get(m, shared)] for m in methods}
    union = sorted({r for cols in by_method.values() for r in cols})
    return by_method, union

# CP's tau grid. ABSOLUTE, FIXED BEFORE THE RUN, and the SAME on every rho column.
#
# tau is a stopping tolerance in unexplained-sd units (cp.py, `tolerance_basis:
# "scale"`), which is what makes it commensurable with rho and with the margin's
# m. It is a parameter of the method, set in advance -- NOT a quantity read back
# off the run. An earlier version placed this grid as fractions of each column's
# own iteration-0 separation distance; that is gone, and it must stay gone:
#
#   - it makes tau depend on the bank, on B, and on which folds were probed, so
#     the same nominal "tau" means a different tolerance in every cell and the
#     axis of the primary figure stops being one quantity;
#   - it cost an extra CP run per (rho column, fold) purely to place a grid;
#   - and it silently misplaced the grid whenever the probed fold was not the
#     fold with the largest distance (measured 2026-08-26: the endpoint cut on
#     5/10 and 6/10 reactor folds and 3/4 gastric folds at rho=1.0).
#
# One decade grid, wide range, the same one the rho sweep's tau ablation uses.
# Whether the top of it stops before any cut is a PROPERTY OF THE RUN, reported
# by the `[cp] ... max iter-0 dist=` line and by `status`, not something the grid
# is bent to guarantee.
# HALF-decade since 2026-08-26, and a strict SUPERSET of the old
# ``[1.0, 0.1, 0.01, 0.001]`` -- every previously scored cell keeps its dial
# value, so a checkpoint from an earlier run resumes into this grid instead of
# being orphaned. The grid got finer because the ADAPTIVE search (below) no
# longer pays for its length: it walks O(log n) cells to bracket the target and
# spends what is left filling the band around it, so resolution near the knee of
# the frontier is now nearly free while the dead tails cost nothing at all.
#
# WIDENED AT BOTH ENDS on 2026-08-27, for a reason the committed curves state
# outright: the 2026-08-27 reactor run has CP DELIVERING at the loosest tau on
# the grid (feasibility 0.9 at tau=1.0, rho=3), so the least-robust delivering
# cell -- which is exactly what `_dial_star` returns, because the objective is
# what robustness is paid for -- sat at `bound="grid_end"`. The grid was the
# limit, not the method. tau=3, 10 give the search somewhere to bracket it, and
# they are also where CP's curve joins the nominal point: a tau above the
# iteration-0 separation distance stops before any cut.
#
# The bottom is the mip_gap floor and stops there. `_resolve_tolerance` floors
# `tau * conv` at `mip_gap * conv` (= 1e-4 in tau units on EVERY problem since
# 2026-08-25), so 1e-4 IS the smallest distinct setting: a grid value below it
# would run at 1e-4 while being written to the curve as something else, which is
# the mislabelled-tau failure that section exists to prevent.
TAU_GRID = [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]

# All three below are supersets of their pre-2026-08-26 values, for the same
# resume reason, and each was widened again on 2026-08-27 -- see TAU_GRID for
# why widening is now cheap and why it was needed.
#
# The wrapper's alpha runs to the far end of its OWN resolution. It is a chance
# constraint over P models, so the only distinct levels are multiples of 1/P
# (=0.05 at P=20) and every value here is one; alpha=0.95 requires exactly ONE
# of the P models to hold. 1.0 is deliberately absent: it requires none, which
# removes the learned constraint from the MIP altogether, so it is not a looser
# wrapper but the absence of one -- weaker than nominal, and not a point on this
# method's frontier. Grid-limited before this: at rho=1.0 on gastric the old top
# (alpha=0.5) still delivered 0.919, so the search wanted a LESS robust cell and
# the grid had none.
DEFAULT_ALPHA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                      0.9, 0.95]
# m=0 is bit-identical to nominal (same fit, same MIP, same x*), so the baseline's
# curve starts AT the nominal point rather than near it. The top end is set by the
# REACTOR, where the old grid ran out with the baseline still climbing: m=1.5 (the
# old max) scored feasibility 0.1 against a 0.9 target, so the margin's own
# protocol point was unreachable and the one comparison it exists for -- the same
# feasibility bought by a one-line RHS shift -- could not be made. m is in
# unexplained-sd units (reactor s_c=2.19, so m=5 is F_C6H6 >= 60.9). Large m goes
# INFEASIBLE rather than conservative once `rhs - m_c` leaves the label range;
# that shows up as a falling solved fraction and is what --min-solved guards, so
# the tail costs at most the cells the search spends discovering it.
DEFAULT_MARGIN_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75, 0.875, 1.0,
                       1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
# Extended UPWARD on purpose, now over the full [0, 1] of a miscoverage level.
# C-MICL is measured infeasible on gastric at alpha=0.1 under both multiplicity
# settings, and n_cal=80 means alpha >= 0.02 is needed for a finite conformal
# quantile at all. Where it FIRST solves is the result; 0.1 is the protocol point
# whether or not it solves there. Measured on the 2026-08-27 gastric run: alpha
# 0.1 and 0.3 solved NOTHING and 0.5 -- the old grid top -- solved 13.8% of
# contexts, under the 0.5 floor, so C-MICL's row of the star table was empty and
# the "where does it first solve" question was answered only with "above 0.5".
# alpha=1.0 is well defined and is kept as the endpoint: `conformal_quantile`
# takes k = max(ceil((n+1)(1-alpha)), 1), so it is the SMALLEST nonconformity
# score -- a real, very loose tightening, not a removed constraint.
DEFAULT_CMICL_ALPHA_GRID = [0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3,
                            0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
# ...and the BOTTOM is per problem, because n_cal sets it rather than taste.
# `conformal_quantile` takes k = ceil((n_cal + 1)(1 - alpha)) and returns inf once
# k > n_cal, so the finest CERTIFIABLE level is 1/(n_cal + 1). Below it no finite
# tightening carries the guarantee and the solve fails outright. Above it, only
# alphas landing on DISTINCT k are distinct levels: a finer value inside the same
# k-band runs at the same q while being written to the curve as something else,
# which is the mislabelled-dial failure TAU_GRID's bottom exists to prevent.
#
#   gastric  n_cal =  80  floor 1/81 = 0.0123; every alpha in [0.0123, 0.0247)
#                         gives k = 80 = s_(max), and 0.02 IS that level. Nothing
#                         distinct sits below it, so there is no extension.
#   reactor  n_cal = 180  floor 1/181 = 0.0055; TWO distinct levels sit below
#                         0.02 -- k=179 over [0.01105, 0.01657) and k=180 over
#                         [0.00552, 0.01105). One representative each, and 0.02
#                         is k=178, so the three are consecutive levels.
#   synthetic             no dial cell has ever been run, so its n_cal is not
#                         measured and nothing is asserted here. Read the
#                         `n_fit=... n_cal=...` line off the first C-MICL fold
#                         before extending it.
#
# This is NOT expected to move the reactor verdict, and was not added on a guess
# that it would: on the 2026-08-27 run C-MICL peaked at feasibility 0.400 at
# alpha=0.03 and had ALREADY turned by 0.02 (solved 1.000 -> 0.800, feasible
# folds 4/10 -> 3/8), i.e. the band is wide enough there to be deleting
# prescriptions rather than protecting them. It is here so that "the grid ran
# out" and "the method ran out" are distinguishable on that series instead of
# assumed apart -- the same question `bound` answers for a delivering series.
CMICL_ALPHA_GRID_EXTRA = {"reactor": [0.0075, 0.015]}


def cmicl_alpha_grid(problem):
    """The shared alpha grid plus the finer levels this problem's n_cal certifies."""
    return sorted(set(DEFAULT_CMICL_ALPHA_GRID)
                  | set(CMICL_ALPHA_GRID_EXTRA.get(problem, ())))
# The coverage cap, as a fraction of the anchors a cut may newly break.
DEFAULT_CP_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Adaptive dial search
# ---------------------------------------------------------------------------
# Sorting a grid by ``ROBUSTNESS_SIGN[method] * dial`` ascending puts the LEAST
# robust cell first. tau, the wrapper's alpha and C-MICL's alpha all run
# backwards (smaller = fewer decisions admitted); the margin's m runs forwards.
#
#   cp        smaller tau   -> CP keeps cutting for longer        -> more robust
#   wrapper   smaller alpha -> more of the P models must hold     -> more robust
#   margin    larger m      -> the RHS is moved further in        -> more robust
#   cmicl     smaller alpha -> a higher conformal quantile        -> more robust
ROBUSTNESS_SIGN = {"cp": -1.0, "wrapper": -1.0, "margin": 1.0, "cmicl": -1.0}

# What one scored cell says about where to look next.
#
#   dead        feasibility 0 at an acceptable solved fraction. Nothing LESS
#               robust can deliver, so that whole tail is pruned outright.
#   under       0 < feasibility < target. Only more robust cells can help -- but
#               the cell itself is a real point of the frontier, and its
#               neighbours are worth filling in.
#   delivers    feasibility >= target at solved_frac >= min_solved. A candidate
#               protocol point; what is left to ask is whether a LESS robust cell
#               also delivers, since that one has the better objective.
#   unsolvable  solved_frac < min_solved -- the same floor _dial_star applies, so
#               a cell here can never be a protocol point anyway. Nothing MORE
#               robust can solve, so that tail is pruned outright.


def _cell_tag_token(args):
    """``--cell-tag`` as a filename token, or ``""``.

    The rest of the cell suffix names things the RUN cannot choose without
    changing what a number means (coherence, fold count, embedded model, seed).
    This one names a question instead: a probe that deliberately runs a subset of
    the sweep -- one method, or a rho column the shared grid does not carry -- and
    therefore must not be written into the cell the full sweep owns. See
    ``_guard_curve_rewrite`` for what goes wrong without it.
    """
    tag = (getattr(args, "cell_tag", None) or "").strip().strip("_")
    if not tag:
        return ""
    keep = "".join(c for c in tag if c.isalnum() or c in "-")
    if not keep:
        raise SystemExit(f"[dial-sweep] --cell-tag {tag!r} has no usable "
                         f"characters (letters, digits and '-' only)")
    return f"_{keep}"


def _guard_curve_rewrite(problem, var, planned, args):
    """Refuse to rewrite a finished cell's curve with FEWER series than it holds.

    ``{problem}_dial_curve{cell}.csv`` is written, not appended, and it is built
    from the rows THIS run scored. So ``--methods wrapper --rho-columns 5 6`` does
    not add a wrapper column to a finished cell: it REPLACES the curve, and the
    star table with it, with the two series it just ran. The score checkpoint
    survives -- it is keyed ``(method@rho, dial)``, so nothing is unrecoverable --
    but the committed curve is gone until a full re-run rebuilds it, and every
    series the run did not touch has to be re-solved to get it back.

    That is the resume trap the cell suffix already guards, reached from the other
    side: the suffix stops a second cell READING the first's rows, and this stops
    it OVERWRITING them. A partial run is a legitimate thing to want -- it just
    needs its own cell, which ``--cell-tag`` gives it (and which skips this check,
    since a tagged cell owns nothing yet).

    ``--refresh`` also skips it, having already deleted the curve on purpose.

    ``--drop-series cp@2 wrapper@3`` is the narrow version of ``--refresh``: it
    says "I mean to lose exactly these", and the guard still fires on anything
    else that would go. It exists because the two ways of meaning it differ in
    cost -- ``--refresh`` clears the score CHECKPOINT too, so every surviving
    series is re-solved from scratch, which on a cell whose expensive series are
    the ones being KEPT is a large price for a bookkeeping statement. Naming the
    casualties keeps the checkpoint and still cannot silently drop a series the
    caller forgot about. A name that is not on disk is an error, not a no-op:
    it usually means a typo, and ignoring it would re-arm the trap.
    """
    curve_path = os.path.join(OUT_DIR, f"{problem}_dial_curve{var}.csv")
    if getattr(args, "cell_tag", None) or args.refresh:
        return
    if not os.path.exists(curve_path):
        return
    try:
        old = pd.read_csv(curve_path)
    except Exception:
        return
    if "method" not in old:
        return
    have = {(str(m), "" if not np.isfinite(r) else f"{float(r):g}")
            for m, r in zip(old["method"], old.get("rho", [np.nan] * len(old)))}
    missing = sorted(have - set(planned))

    # Series the caller has explicitly said they mean to lose.
    dropped = set()
    for token in (getattr(args, "drop_series", None) or []):
        m, _, r = str(token).partition("@")
        key = (m.strip(), f"{float(r):g}" if r.strip() else "")
        if key not in have:
            raise SystemExit(
                f"[dial-sweep] --drop-series {token!r} names a series the curve "
                f"does not hold. It has: "
                + ", ".join(sorted(f"{a}@{b}" if b else a for a, b in have))
                + ".\n  Nothing was run: fix the name rather than widening the "
                  "flag, or the guard stops protecting the rest of the cell.")
        dropped.add(key)
    if dropped:
        print("[dial-sweep] --drop-series: intentionally discarding "
              + ", ".join(sorted(f"{a}@rho={b}" if b else a for a, b in dropped))
              + " from the curve and star table (the score checkpoint keeps "
                "their rows, unread)", flush=True)
    missing = [k for k in missing if k not in dropped]

    if not missing:
        return
    lost = ", ".join(f"{m}@rho={r}" if r else m for m, r in missing)
    raise SystemExit(
        f"[dial-sweep] REFUSING to rewrite {curve_path}: it holds {len(have)} "
        f"series and this run scores only {len(planned)}, so {len(missing)} would "
        f"be DELETED from the curve and the star table -- {lost}." + "\n"
        f"  The scored cells survive in {problem}_dial_scores{var}.csv, but the "
        f"curve is rebuilt from this run's rows alone, so recovering them "
        f"means re-running them." + "\n"
        "  Run the probe in its OWN cell:   --cell-tag <name>\n"
        "  or put the missing series back:  --methods / --rho-columns "
        "covering them\n"
        "  or name what you mean to lose:   --drop-series "
        + " ".join(f"{m}@{r}" if r else m for m, r in missing) + "\n"
        "  or discard the whole cell:       --refresh (re-solves everything: it "
        "clears the score checkpoint too)")


def _series_key(method, rho):
    """One series is one (method, rho column); the flat methods have no rho."""
    return (method, "" if not np.isfinite(rho) else f"{float(rho):g}")


def _verdict(d, target, min_solved):
    feas = float(d.get("feas", np.nan))
    solved = float(d.get("solved", 0.0) or 0.0)
    if not np.isfinite(feas) or solved < min_solved:
        return "unsolvable"
    if feas >= target:
        return "delivers"
    return "dead" if feas <= 0.0 else "under"


def _fold_counts(d, n_folds):
    """``(feasible, solved)`` fold counts behind one cell's scores, or ``None``.

    Only meaningful on a single-decision problem, where a fold contributes ONE
    binary outcome; ``n_folds=0`` (contextual) returns ``None``.

    The denominator is the whole point. ``feas`` is conditional on the folds that
    SOLVED, so a dip from 0.400 to 0.375 is not 0.025 of a fold: it is 4 of 10
    against 3 of 8. Quoting it against a fixed ``1/n_folds`` printed the
    2026-08-27 reactor C-MICL wobble as "0 fold(s)" -- which reads as "nothing
    moved" when in fact one fold had flipped and two more had gone unsolvable,
    the very thing that made that cell the end of the useful dial.
    """
    if not n_folds:
        return None
    feas = float(d.get("feas", np.nan))
    solved = float(d.get("solved", np.nan) or np.nan)
    if not (np.isfinite(feas) and np.isfinite(solved)):
        return None
    n_solved = int(round(solved * n_folds))
    return int(round(feas * n_solved)), n_solved


def _order_check(dname, dial_at, seq, seen, detail, target, min_solved,
                 n_folds=0):
    """Did the order the pruning rests on hold, among the cells actually scored?

    Returns ``(violations, wobbles)`` -- TWO different things, kept apart on
    purpose, because only the first can mislead the search.

    A **violation** is a VERDICT inversion. The two prune rules claim "nothing
    less robust delivers" and "nothing more robust solves", so what falsifies
    them is a delivering cell sitting BELOW a non-delivering one, or a solvable
    cell sitting ABOVE an unsolvable one. Those, and only those, mean a pruned
    tail might have held the answer.

    A **wobble** is a numeric dip in feasibility that leaves both cells on the
    SAME side of the target. It is real and worth recording -- the committed
    gastric dial curve has several, between -0.003 and -0.014 -- but it changes
    no verdict, so it invalidates no prune. Reporting it as a reason to re-run
    the whole grid would be crying wolf: on that curve it would fire on 6 of 11
    series, none of which the search would have got wrong.

    ``n_folds`` is the fold count on a single-decision problem and 0 on a
    contextual one, where a fold is a mean over its own held-out cohort and no
    fold quantum exists. Where it is known a wobble is quoted as
    ``feasible/solved -> feasible/solved folds`` rather than in decimals, because
    "0.800 -> 0.700" on the 10-fold reactor is one fold and reads as far more than
    that as a decimal -- and because BOTH denominators can move at once
    (``_fold_counts``).
    """
    viol, wobble = [], []
    solv, deliv = {}, {}
    for p in seq:
        sv = float(detail[p].get("solved", np.nan) or np.nan)
        solv[p] = bool(np.isfinite(sv) and sv >= min_solved)
        deliv[p] = seen[p] == "delivers"
    for i, x in enumerate(seq):
        for y in seq[i + 1:]:                      # y is strictly MORE robust
            if not solv[x] and solv[y]:
                viol.append(f"{dname}={dial_at(y):.5g} clears the solved floor "
                            f"while the less robust {dname}={dial_at(x):.5g} "
                            f"does not")
            if deliv[x] and solv[y] and not deliv[y]:
                viol.append(f"{dname}={dial_at(x):.5g} reaches the target "
                            f"({float(detail[x]['feas']):.3f}) while the more "
                            f"robust {dname}={dial_at(y):.5g} does not "
                            f"({float(detail[y]['feas']):.3f})")
    for x, y in zip(seq, seq[1:]):
        fx = float(detail[x].get("feas", np.nan))
        fy = float(detail[y].get("feas", np.nan))
        if not (np.isfinite(fx) and np.isfinite(fy)) or fy >= fx - 1e-9:
            continue
        if deliv[x] and not deliv[y]:
            continue                                # already a violation above
        cx = _fold_counts(detail[x], n_folds)
        cy = _fold_counts(detail[y], n_folds)
        how = (f"{cx[0]}/{cx[1]} -> {cy[0]}/{cy[1]} folds" if cx and cy
               else f"{fx - fy:.3f}")
        side = "above" if deliv[x] else "below"
        wobble.append(f"feasibility dips {fx:.3f}->{fy:.3f} ({how}) between "
                      f"{dname}={dial_at(x):.5g} and {dial_at(y):.5g}, both "
                      f"{side} the target")
    return viol, wobble


def _search_dials(method, grid, evaluate, known, target, min_solved,
                  max_evals=None, must_visit=(), label="", n_folds=0):
    """Walk one series' dial grid in the order the answers actually decide.

    **This assumes the dial is monotone**, in both directions at once: held-out
    feasibility rises with robustness and the solved fraction falls with it. The
    cells that both deliver the target and stay solvable are then an INTERVAL of
    the robustness-ordered grid, and the protocol point ``_dial_star`` picks --
    best objective among the delivering cells -- is that interval's LEAST robust
    end, because the objective is what robustness is paid for with.

    That assumption is not free, and this repo has counterexamples on the rho
    axis (CP's dip at rho=0.5 on gastric; robust_reg's feasibility FALLING with
    rho). It has never been tested on the dials. Two things are done about that
    rather than nothing: every cell actually evaluated is checked for order
    violations afterwards, and any is printed and carried into the star table's
    ``monotone_note``; and ``--search grid`` still walks the whole grid, which is
    the fallback whenever a violation shows up.

    Phase 1 BRACKETS by bisection: ``under``/``dead`` sends the window to the
    more-robust half, ``delivers``/``unsolvable`` to the less-robust half. O(log
    n) cells, landing on the transition.

    Phase 2 FILLS what is left of the eval budget outwards from the transition,
    inside the two hard prune bounds. The deliverable here is a CURVE, not a
    single number, so a bare bisection leaving three points per series would
    answer the wrong question. Filling puts the resolution where the frontier
    bends and buys none of it in the two dead tails.

    Cells already on the score checkpoint are FREE: they are replayed first, so a
    resumed run both re-reads its own answers and starts with the window already
    narrowed.

    ``must_visit`` names dial values scored whatever the search thinks -- C-MICL's
    protocol point ``1 - target``, which is asserted rather than chosen and has to
    be on the record even when it does not solve.

    Returns ``(seen, detail, info)``: verdict and scored dict per grid position,
    plus a summary carrying ``skipped`` (a reason per unscored cell) and
    ``violations``.
    """
    grid = [float(v) for v in grid]
    order = sorted(range(len(grid)),
                   key=lambda i: ROBUSTNESS_SIGN[method] * grid[i])
    n = len(order)
    dname = DIAL[method] or "dial"

    def dial_at(pos):
        return grid[order[pos]]

    budget = (int(max_evals) if max_evals is not None
              else max(4, int(np.ceil(np.log2(max(n, 2)))) + 2))

    seen, detail = {}, {}
    spent = 0
    # The HARD prune bounds: the two rules that exclude a cell outright rather
    # than merely deprioritise it. Nothing outside them is ever scored.
    keep_lo, keep_hi = 0, n - 1

    def visit(pos):
        nonlocal spent, keep_lo, keep_hi
        dial = dial_at(pos)
        free = known(dial)
        d = evaluate(dial)
        if not free:
            spent += 1
        v = _verdict(d, target, min_solved)
        seen[pos], detail[pos] = v, d
        if v == "dead":
            keep_lo = max(keep_lo, pos + 1)
        elif v == "unsolvable":
            keep_hi = min(keep_hi, pos - 1)
        return v

    def window():
        """The bisection window implied by EVERY verdict seen so far."""
        a, b = keep_lo, keep_hi
        for pos, v in seen.items():
            if v in ("dead", "under"):
                a = max(a, pos + 1)
            else:
                b = min(b, pos - 1)
        return a, b

    # 0. Replay the checkpoint, then anything the protocol demands.
    for pos in range(n):
        if known(dial_at(pos)):
            visit(pos)
    for dial in must_visit:
        pos = next((p for p in range(n) if np.isclose(dial_at(p), float(dial))),
                   None)
        if pos is not None and pos not in seen:
            visit(pos)

    # 1. Bracket.
    while spent < budget:
        a, b = window()
        cand = [p for p in range(a, b + 1) if p not in seen]
        if not cand:
            break
        mid = (a + b) // 2
        visit(min(cand, key=lambda p: (abs(p - mid), p)))

    # 2. Fill outwards from the transition; less-robust side first on a tie,
    #    which is the side the better objective is on.
    #
    # The prune bounds are re-read on every step, not just used to build the
    # order: filling can itself discover the end of the solvable range, and a
    # materialised order would then walk straight past it (measured -- with the
    # order fixed up front, m=16 scoring unsolvable did not stop m=32 being
    # scored too).
    notunder = [p for p, v in seen.items() if v in ("delivers", "unsolvable")]
    trans = min(notunder) if notunder else (keep_lo + keep_hi) // 2
    while spent < budget:
        cand = [q for q in range(keep_lo, keep_hi + 1) if q not in seen]
        if not cand:
            break
        visit(min(cand, key=lambda q: (abs(q - trans), q)))

    # ---- what was skipped, and why ----------------------------------------
    skipped = []
    for pos in range(n):
        if pos in seen:
            continue
        if pos < keep_lo:
            why = (f"pruned: less robust than {dname}={dial_at(keep_lo - 1):.5g}, "
                   f"which scored feasibility 0")
        elif pos > keep_hi:
            why = (f"pruned: more robust than {dname}={dial_at(keep_hi + 1):.5g}, "
                   f"which fell below the solved floor")
        else:
            why = f"not reached: eval budget {budget} spent"
        skipped.append(dict(dial=dial_at(pos), robustness_rank=pos, reason=why))

    # ---- did the order the pruning rests on hold? -------------------------
    viol, wobble = _order_check(dname, dial_at, sorted(seen), seen, detail,
                                target, min_solved, n_folds)

    pruned = sum(1 for r in skipped if r["reason"].startswith("pruned"))
    info = dict(n_grid=n, n_evaluated=len(seen), spent=spent, budget=budget,
                skipped=skipped, n_pruned=pruned,
                n_unreached=len(skipped) - pruned, violations=viol,
                wobbles=wobble, keep_lo=keep_lo, keep_hi=keep_hi)
    print(f"[search] {label or method}: evaluated {len(seen)}/{n} cells "
          f"({spent} solved, {len(seen) - spent} resumed; budget {budget}); "
          f"pruned {pruned}, not reached {len(skipped) - pruned}", flush=True)
    _report_order(label or method, viol, wobble)
    return seen, detail, info


def _report_order(label, viol, wobble):
    """Violations are a reason to re-run; wobbles are a note. Never swap them."""
    for v in viol:
        print(f"[search] NON-MONOTONE {label}: {v} -- the pruning rests on that "
              f"order, so a pruned tail may hold the answer. Re-run this series "
              f"with --search grid before reading its protocol point", flush=True)
    for w in wobble:
        print(f"[search] wobble {label}: {w} -- no verdict changes, so no prune "
              f"is affected; recorded, not acted on", flush=True)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def _setup(config, args):
    return {"synthetic": _setup_synthetic, "reactor": _setup_reactor,
            "gastric": _setup_gastric}[args.problem](config, args)


def _bank_size(config, args):
    """B for the SHARED bank: enough for CP's separation pool and the wrapper's P.

    CP separates over B draws and the wrapper embeds the first P of them, so one
    bank of ``max(B, P)`` serves both. ``--match-bank`` sets CP's B to P, which is
    the whole point of that flag; the max then collapses to P and the two methods
    sample D at the same density.
    """
    unc = config.get("uncertainty", {}) or {}
    methods = config.get("methods", {}) or {}
    p = int((methods.get("wrapper", {}) or {}).get("n_estimators",
                                                   unc.get("n_bootstrap", 20)))
    p = max(p, int(unc.get("n_bootstrap", 20)))
    b = p if args.match_bank else int((methods.get("cp", {}) or {})
                                      .get("n_scenarios", 200))
    return max(b, p)


def _make_bank_factory(config, args, uset, model_spec, seed):
    """``fold_instance -> ScenarioBank`` for one rho column."""
    from src.methods.uncertainty import build_bank_for_instance

    model_type, model_params = model_spec
    n = _bank_size(config, args)

    def factory(fold_instance):
        return build_bank_for_instance(fold_instance, model_type, model_params,
                                       uset, n_scenarios=n, seed=seed,
                                       verbose=True)
    return factory


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run(config, args):
    from src.methods.uncertainty import uncertainty_set_from_config

    os.makedirs(OUT_DIR, exist_ok=True)
    problem = args.problem
    su = _setup(config, args)
    base_uset = dataclasses.replace(
        uncertainty_set_from_config(config),
        geometry="ellipsoid", coherent=bool(args.coherent),
    )
    bank_seed = _bank_seed(config, args)
    var = _variant_suffix(args) + _cell_tag_token(args)

    scores_path = os.path.join(OUT_DIR, f"{problem}_dial_scores{var}.csv")
    ctx_path = os.path.join(OUT_DIR, f"{problem}_dial_contexts{var}.csv")
    # The judge audit: per-decision slack, member instability and the decision
    # vector. Derived data like the contexts file, cleared by --refresh with it,
    # and read only by experiments/audit_judge.py.
    judge_path = os.path.join(OUT_DIR, f"{problem}_dial_judge{var}.csv")
    if args.refresh:
        # EVERY output of this cell, not just the two that feed the resume. A
        # --refresh that left the curve and the star behind is a trap: the curve
        # is rewritten after each cell, so it recovers, but `{problem}_dial_star`
        # is written only at the END of the sweep -- so a refreshed run that
        # times out leaves the PREVIOUS run's star on disk, and
        # `run_dial_test.py` reads exactly that file to decide where to hold each
        # method. Deleting it up front makes a missing star an obvious failure
        # rather than a silently stale one.
        for pth in (scores_path, ctx_path, judge_path,
                    os.path.join(OUT_DIR, f"{problem}_dial_curve{var}.csv"),
                    os.path.join(OUT_DIR, f"{problem}_dial_star{var}.csv"),
                    os.path.join(OUT_DIR, f"{problem}_dial_skipped{var}.csv")):
            if os.path.exists(pth):
                os.remove(pth)
                print(f"[dial-sweep] --refresh: removed {pth}", flush=True)
    ckpt = load_detail_checkpoint(scores_path) if not args.refresh else {}
    # A stale `{problem}_tau_probe{cell}.json` from the removed grid-placement
    # probe is deleted rather than ignored: leaving it implies the grid still
    # comes from an iteration-0 distance, and it does not.
    stale = os.path.join(OUT_DIR, f"{problem}_tau_probe{var}.json")
    if os.path.exists(stale):
        os.remove(stale)
        print(f"[dial-sweep] removed {stale}: tau is fixed before the run and is "
              f"no longer placed from an iteration-0 distance", flush=True)

    methods = list(args.methods or ["nominal", "cp", "wrapper", "margin", "cmicl"])
    bad = [m for m in methods if m not in DIAL]
    if bad:
        extra = (" robust_reg has no dial at a fixed rho -- its dial IS the "
                 "radius. It belongs on run_rho_sweep.py."
                 if "robust_reg" in bad else "")
        raise SystemExit(f"[dial-sweep] not on this axis: {bad}.{extra}")
    d_methods = [m for m in methods if m in FACES_D]
    flat_methods = [m for m in methods if m not in FACES_D]
    rho_by_method, rho_cols = _rho_columns_for(problem, d_methods,
                                               args.rho_columns)
    # Every (method, rho) series this run will write, in the same key form the
    # curve is read back in. Checked against what is already on disk BEFORE any
    # solving, so a partial run fails in a second rather than after an hour.
    _guard_curve_rewrite(
        problem, var,
        {_series_key(m, r) for m, cols in rho_by_method.items() for r in cols}
        | {_series_key(m, np.nan) for m in flat_methods},
        args)

    print(f"[dial-sweep] problem={problem} geometry=ellipsoid "
          f"coherent={args.coherent} folds={len(su.folds)} seed={bank_seed} "
          f"sense={su.oracle.objective_sense}", flush=True)
    if len({tuple(c) for c in rho_by_method.values()}) > 1:
        print("[dial-sweep] rho columns are PER METHOD: "
              + "; ".join(f"{m}={[float(f'{r:g}') for r in c]}"
                          for m, c in sorted(rho_by_method.items()))
              + " -- the comparison is at equal FEASIBILITY, not equal rho",
              flush=True)
    else:
        print(f"[dial-sweep] rho columns={rho_cols} "
              f"(D-facing: {d_methods or 'none'})", flush=True)
    print(f"[dial-sweep] dials: "
          + ", ".join(f"{m}={DIAL[m] or 'none'}" for m in methods), flush=True)
    print(f"[dial-sweep] shared bank B={_bank_size(config, args)} per "
          f"(rho column, fold); CP's tau grid and the wrapper's alpha grid both "
          f"read it", flush=True)
    if not su.contextual:
        print(f"[dial-sweep] WARNING: single-decision problem -- feasibility is "
              f"quantized to 1/{len(su.folds)} = {1/len(su.folds):.2f}. Read the "
              f"scatter, not the protocol point.", flush=True)

    rows = []

    def score(tag, method, uset, dial, cache=None, cp_alpha=None, phase="dial",
              rho=np.nan, note=""):
        """One scored cell, resumable, with per-context records."""
        ckey = (tag, float(dial))
        cell = f"{tag} {DIAL[method] or 'dial'}={dial:.6g}"
        if ckey not in ckpt:
            # Open the cell BEFORE solving. The one-line summary below is printed
            # after the fact, so without this the hundreds of solver lines a cell
            # emits belong to nothing until it ends -- and a job killed mid-cell
            # left no marker at all for which cell it died in.
            print(f"\n[cell] BEGIN {cell}"
                  + (f"  (cp_alpha={cp_alpha:g})" if cp_alpha else "")
                  + f"  [{len(su.folds)} folds]", flush=True)
            build = su.make_build(method, uset, cp_alpha=cp_alpha)
            d = cv_score_knob(
                build, dial, su.folds, su.oracle, su.instance,
                constraint_names=su.constraint_names, contextual=su.contextual,
                return_details=True, return_contexts=True,
                fold_cache=cache, bank_kwarg=BANK_KWARG.get(method),
                label=cell,
            )
            append_score(scores_path, tag, dial, d["feas"], d["obj"], d["solved"], d)
            append_contexts(ctx_path, tag, dial, d.pop("contexts", []))
            append_judge(judge_path, tag, dial, d.pop("judge", []))
            ckpt[ckey] = d
        else:
            print(f"\n[cell] RESUMED {cell} (from the score checkpoint)",
                  flush=True)
        d = ckpt[ckey]
        cap = f" CAPPED({d['n_capped']}/{len(su.folds)})" if d.get("n_capped") else ""
        print(f"[cell] END   {tag:<28s} {DIAL[method] or 'dial'}={dial:<10.5g} "
              f"feas={d['feas']:.3f} obj={d['obj']:+.4f} solved={d['solved']:.3f} "
              f"master={d['master_time_s']:.1f}s "
              f"test/pt={d['test_time_per_point_s']:.2f}s "
              f"[{d['status']}]{cap}", flush=True)
        rows.append(dict(
            problem=problem, method=method, rho=rho, dial=float(dial),
            dial_name=DIAL[method] or "none", faces_D=method in FACES_D,
            feasibility=d["feas"], objective=d["obj"], solved_frac=d["solved"],
            status=d["status"], n_capped=d["n_capped"],
            master_time_s=d["master_time_s"], test_time_s=d["test_time_s"],
            test_time_per_point_s=d["test_time_per_point_s"],
            objective_sense=su.oracle.objective_sense,
            coherent=bool(args.coherent), matched_bank=bool(args.match_bank),
            seed=bank_seed, cp_alpha=(np.nan if cp_alpha is None else float(cp_alpha)),
            phase=phase, note=note,
        ))
        # Rewrite the curve after EVERY cell, not once at the end. A gastric run
        # is |rho columns| x |dial grid| cells against a 12h wall clock, so it may
        # well be requeued; the score checkpoint already made that cheap, but a
        # curve written only on the final line meant a timed-out job left nothing
        # to plot even though every cell it finished was on disk. Rewriting a
        # few-dozen-row CSV per cell costs nothing next to a master solve.
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, f"{problem}_dial_curve{var}.csv"), index=False)
        return d

    # The full configured grid, the search summary and the unscored cells, per
    # series. All three exist because the curve CSV can no longer be read as "the
    # grid": under --search adaptive it is a SUBSET of it, and which subset --
    # and why the rest is missing -- is part of the result.
    grids, search_info, skipped_rows = {}, {}, []
    # The fold count a wobble is quoted against. On a single-decision problem
    # each fold contributes ONE binary outcome, so a cell score is a proportion
    # over folds and a dip is readable as a count of them -- but conditional on
    # the folds that SOLVED, which is why `_fold_counts` renders both counts
    # rather than a fixed 1/len(folds) quantum. On a contextual problem a fold is
    # a mean over its own held-out cohort, so there is no such quantum: 0 disables
    # the rendering.
    n_folds_res = 0 if su.contextual else len(su.folds)

    def walk(tag, method, grid, uset, rho=np.nan, cache=None, must_visit=(),
             note_fn=None):
        """Score one series' grid: adaptively by default, whole on --search grid."""
        key = _series_key(method, rho)
        grids[key] = [float(v) for v in grid]

        def ev(dial):
            return score(tag, method, uset, dial, cache=cache, rho=rho,
                         note=(note_fn(dial) if note_fn else ""))

        if args.search == "grid":
            # Walked whole -- but still ORDER-CHECKED. A full grid is where a
            # violation is most worth knowing about: nothing was pruned on the
            # assumption, so what comes out is a clean statement about the dial
            # itself rather than a warning about this run.
            order = sorted(grids[key], key=lambda v: ROBUSTNESS_SIGN[method] * v)
            seen, detail = {}, {}
            for pos, dial in enumerate(order):
                d = ev(dial)
                seen[pos] = _verdict(d, float(args.feas_target),
                                     float(args.min_solved))
                detail[pos] = d
            viol, wob = _order_check(
                DIAL[method] or "dial", lambda q: order[q], sorted(seen), seen,
                detail, float(args.feas_target), float(args.min_solved),
                n_folds_res)
            _report_order(tag, viol, wob)
            search_info[key] = dict(search="grid", n_grid=len(grid),
                                    n_evaluated=len(grid), spent=len(grid),
                                    budget=len(grid), skipped=[], n_pruned=0,
                                    n_unreached=0, violations=viol, wobbles=wob)
            return
        _, _, info = _search_dials(
            method, grids[key], ev,
            known=lambda d, _t=tag: (_t, float(d)) in ckpt,
            target=float(args.feas_target), min_solved=float(args.min_solved),
            max_evals=args.max_evals, must_visit=must_visit, label=tag,
            n_folds=n_folds_res)
        info["search"] = "adaptive"
        search_info[key] = info
        for r in info["skipped"]:
            skipped_rows.append(dict(problem=problem, method=method, rho=rho,
                                     dial_name=DIAL[method] or "none",
                                     search="adaptive", **r))

    # ---- the D-facing methods, one rho column at a time --------------------
    cache = None
    for rho in rho_cols:
        if not d_methods:
            break
        # Only the methods whose OWN columns include this rho. With per-method
        # columns the union is looped, so a rho the wrapper alone sits on must
        # not drag CP onto it (and vice versa) -- that would silently re-add the
        # very series `_guard_curve_rewrite` was told this run does not score.
        here = [m for m in d_methods if rho in rho_by_method[m]]
        if not here:
            continue
        uset = dataclasses.replace(base_uset, rho=rho)
        # A fresh cache per column: the bank is keyed on the fold index alone, so
        # carrying one across columns would hand rho=1.0 the rho=0.5 bank.
        if cache is not None:
            cache.clear()
        cache = FoldCache(
            bank_factory=_make_bank_factory(config, args, uset, su.model_spec,
                                            bank_seed),
            key=(rho, bool(args.coherent), bank_seed, _bank_size(config, args)),
        )
        print(f"\n[dial-sweep] === rho = {rho:g} ===", flush=True)

        tau_vals = None
        if "cp" in here:
            # tau is FIXED BEFORE THE RUN and is the same on every rho column.
            # It is not read off any iteration-0 distance -- see TAU_GRID.
            tau_vals = [float(t) for t in (args.tau_grid or TAU_GRID)]
            print(f"[dial-sweep] tau grid (fixed, same on every rho column): "
                  f"{tau_vals} -- unexplained-sd units, set before the run",
                  flush=True)

        for method in here:
            grid = (tau_vals if method == "cp"
                    else [float(a) for a in (args.alpha_grid or DEFAULT_ALPHA_GRID)])
            if grid is None:
                continue
            walk(f"{method}@rho={rho:g}", method, grid, uset, rho=rho,
                 cache=cache)

    # ---- the methods that face no D: scored ONCE -------------------------
    # No rho column: neither reads any uncertainty.* key that a column moves, so a
    # per-column repeat would re-measure one number. The plot draws them once and
    # they sit against every column.
    for method in flat_methods:
        if DIAL[method] is None:
            print(f"\n[dial-sweep] === {method} (no dial: one reference point) ===",
                  flush=True)
            score(method, method, base_uset, 0.0, rho=np.nan, phase="reference",
                  note="no dial")
            continue
        grid = {
            "margin": args.margin_grid or DEFAULT_MARGIN_GRID,
            "cmicl": args.cmicl_alpha_grid or cmicl_alpha_grid(problem),
        }[method]
        print(f"\n[dial-sweep] === {method} (dial={DIAL[method]}, faces no D) ===",
              flush=True)
        if method == "cmicl":
            print(f"[dial-sweep] NOTE: cmicl's PROTOCOL point is "
                  f"alpha={1 - float(args.feas_target):g} -- the conformal level "
                  f"and the feasibility target are the same quantity, so it is "
                  f"asserted, not chosen. The rest of the grid is here to find "
                  f"where it first SOLVES; on gastric it is measured infeasible "
                  f"at 0.1 under either multiplicity setting.", flush=True)
        # C-MICL's protocol point is ASSERTED, not chosen, so it is scored
        # whatever the search would have done with it -- including when it does
        # not solve, which on gastric is the measured expectation.
        prot = 1 - float(args.feas_target)
        must = [prot] if method == "cmicl" else []
        note_fn = ((lambda d: "protocol point" if np.isclose(d, prot) else "")
                   if method == "cmicl" else None)
        walk(method, method, [float(v) for v in grid], base_uset, rho=np.nan,
             must_visit=must, note_fn=note_fn)

    curve = os.path.join(OUT_DIR, f"{problem}_dial_curve{var}.csv")
    # A cell the search never scored is NOT a cell that produced no solution --
    # the plot reads a non-finite feasibility as "infeasible on every fold",
    # which is a result. So skipped cells go to their own file with the reason
    # each, and never into the curve as a row of NaNs. A stale file is deleted
    # rather than left, or the next --search grid run would still look pruned.
    skip_path = os.path.join(OUT_DIR, f"{problem}_dial_skipped{var}.csv")
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(skip_path, index=False)
        print(f"\n[dial-sweep] wrote {skip_path}: {len(skipped_rows)} cells "
              f"not scored, with the reason for each", flush=True)
    elif os.path.exists(skip_path):
        os.remove(skip_path)
    star = _dial_star(pd.DataFrame(rows), problem, float(args.feas_target),
                      su.oracle.objective_sense, float(args.min_solved),
                      out_suffix=var, grids=grids, search_info=search_info)

    # ---- the coverage-cap ablation ---------------------------------------
    # Runs AFTER the protocol table, because it holds tau at the tau* that table
    # reports. Its rows land in the same curve CSV under phase="cp_alpha_ablation"
    # -- the plot draws them as their own series and _dial_star excludes them.
    if args.cp_alpha_ablate:
        _cp_alpha_ablation(su, config, args, base_uset, star, score, problem,
                           bank_seed, var)

    df = pd.DataFrame(rows)
    df.to_csv(curve, index=False)
    print(f"\n[dial-sweep] wrote {curve}", flush=True)
    return df


def _cp_alpha_ablation(su, config, args, base_uset, star, score, problem,
                       bank_seed, var):
    """Hold tau at tau*, walk ``cp_alpha``.

    The question: CP's held-out feasibility tops out near 0.984 on gastric, and the
    cap is the obvious suspect -- a cut that would break one protected anchor is
    rolled back whole, so the adversary CP is allowed to admit is bounded by the
    single most fragile patient. Relaxing the cap admits stronger cuts at the price
    of dropping anchors, which shows up as solved fraction. Both columns are
    reported; neither is a free win.

    tau* comes from the sweep just run, at ``--cp-alpha-rho``, so the ablation
    varies ONE thing against a point the main curve already reports.
    """
    if not su.contextual:
        print(f"\n[dial-sweep] SKIPPING the cp_alpha ablation on {problem}: it is "
              f"structurally inert here. Single-decision problems take CP's BASIC "
              f"separation path, which has no protected-anchor test -- there is no "
              f"coverage cap to relax, so every alpha would return the same run. "
              f"See cp._BasicSeparation.", flush=True)
        return
    # CP's OWN top column, not the union's: with per-method columns the union can
    # hold a rho no CP series was ever scored at, and the star lookup below would
    # then miss and skip the ablation for the wrong reason.
    rho_a = float(args.cp_alpha_rho if args.cp_alpha_rho is not None
                  else max(_rho_columns_for(problem, ["cp"],
                                            args.rho_columns)[0]["cp"]))
    sel = star[(star["method"] == "cp") & np.isclose(star["rho"], rho_a)]
    if sel.empty or not np.isfinite(sel["dial_star"].iloc[0]):
        print(f"\n[dial-sweep] SKIPPING the cp_alpha ablation: cp has no tau* at "
              f"rho={rho_a:g} (it never reaches feasibility "
              f">= {args.feas_target:g} on enough solved contexts there), so "
              f"there is no point to hold tau at.", flush=True)
        return
    tau_star = float(sel["dial_star"].iloc[0])
    uset = dataclasses.replace(base_uset, rho=rho_a)
    cache = FoldCache(
        bank_factory=_make_bank_factory(config, args, uset, su.model_spec,
                                        bank_seed),
        key=("cp_alpha", rho_a, bool(args.coherent), bank_seed),
    )
    print(f"\n[dial-sweep] === coverage-cap ablation: rho={rho_a:g}, "
          f"tau*={tau_star:.5g} held fixed ===", flush=True)
    print(f"[dial-sweep] cp_alpha > 0 lets a cut break up to that fraction of the "
          f"anchors. It buys a stronger adversary by dropping patients, so read "
          f"feasibility and solved_frac together -- neither alone is the result.",
          flush=True)
    # Walked WHOLE, never searched. The question here is the SHAPE of both
    # columns against alpha -- "what does relaxing the cap buy, and what does it
    # cost in solved fraction" -- so pruning on a feasibility target would throw
    # away the half of the answer the ablation exists for.
    for a in [float(v) for v in (args.cp_alpha_grid or DEFAULT_CP_ALPHA_GRID)]:
        score(f"cp@rho={rho_a:g}@cpalpha={a:g}", "cp", uset, tau_star,
              cache=cache, cp_alpha=a, phase="cp_alpha_ablation", rho=rho_a,
              note=f"tau*={tau_star:.5g}")
    cache.clear()


def _dial_star(df, problem, target, sense, min_solved, out_suffix="",
               grids=None, search_info=None):
    """The PROTOCOL POINT of each series: the best objective that still delivers.

    A series here is one (method, rho column). Unlike ``rho*`` -- "the largest
    assumed radius the method absorbs" -- a dial has no direction that is
    automatically "more"; alpha and tau run opposite ways round from m. So the
    point picked is the operationally meaningful one: among the cells meeting the
    feasibility target on enough solved contexts, the one with the **best
    objective** in the problem's own sense. That is the decision a user would take
    from this series, and it needs no monotonicity assumption.

    ``solved_frac >= min_solved`` is applied FIRST and is the artefact guard: a
    dial that renders most contexts unsolvable and gets the survivors right scores
    1.0 feasibility. It is why the scatter encodes solved fraction too -- the
    table's floor is a cliff, the plot's marker size is the gradient.

    ``bound`` says what stopped the search, because the cases mean different
    things: ``grid_end`` (the winning cell sits at an end of the dial grid, so the
    grid may be the limit rather than the method) and ``interior`` (a genuine
    optimum) for a series that delivers; ``none_grid_end`` and ``none_interior``
    for one that never reaches the target at all. That last split matters as much
    as the first and used to be missing -- a flat ``none`` said only "no", so the
    2026-08-27 reactor C-MICL row (best feasibility 0.400, at alpha=0.03, one cell
    INSIDE a grid running to 0.02) was indistinguishable from a series still
    climbing when the grid ran out. Only the second is a reason to widen a grid,
    and ``grid_end`` on a delivering series is exactly the signal the 2026-08-27
    widening pass was read off.

    Under ``--search adaptive`` the rows in ``df`` are a SUBSET of the grid, so
    ``grids`` carries the full configured grid per series (``bound`` is a claim
    about the grid, not about what was scored) and ``search_info`` carries how
    many cells were evaluated, how many were pruned, and any monotonicity
    violation the search found in what it did measure. The rule below is
    unchanged and still assumes no monotonicity itself -- it just has fewer cells
    to apply itself to, and ``monotone_note`` is what says when that matters.
    """
    rows = []
    main = df[df["phase"] == "dial"] if "phase" in df else df
    ref = df[df["phase"] == "reference"] if "phase" in df else df.iloc[0:0]
    better = (lambda a, b: a < b) if sense == "min" else (lambda a, b: a > b)

    for (method, rho), g_all in main.groupby(["method", "rho"], dropna=False):
        g_all = g_all.sort_values("dial")
        # Ends come from the FULL CONFIGURED grid, not from the cells scored. The
        # two differ under an adaptive search, and reading `grid_end` off the
        # evaluated subset would report it every time a tail was pruned -- which
        # is the opposite of what the flag means.
        key = _series_key(method, rho)
        full = (grids or {}).get(key)
        ends = ({float(min(full)), float(max(full))} if full
                else {float(g_all["dial"].min()), float(g_all["dial"].max())})
        si = (search_info or {}).get(key) or {}
        prov = dict(search=str(si.get("search", "grid")),
                    n_grid=int(si.get("n_grid", len(g_all))),
                    n_evaluated=int(si.get("n_evaluated", len(g_all))),
                    n_pruned=int(si.get("n_pruned", 0)),
                    n_unreached=int(si.get("n_unreached", 0)),
                    monotone_note="; ".join(si.get("violations", []) or []),
                    feas_wobble="; ".join(si.get("wobbles", []) or []))
        g = g_all[g_all["solved_frac"] >= min_solved]
        # How close the series got, whether or not it delivered. A series that
        # misses the target used to leave a row of NaNs, which says only "no" --
        # 0.88 at every dial and 0.00 at every dial are different results and the
        # table could not tell them apart. Reported on EVERY row (it is the
        # series' own best, not the protocol point) so the two never get confused.
        elig = g[np.isfinite(g["feasibility"])]
        # The most-feasible cell, ties broken on OBJECTIVE in the problem's own
        # sense. Ties are the rule, not the exception, on the single-decision
        # instances -- feasibility there is quantized to 1/n_folds, so a whole
        # stretch of the dial can share one rate -- and `idxmax` alone would pick
        # whichever happened to sort first, making the row an artifact of grid
        # order. Breaking on objective makes this the exact analogue of `dial*`
        # with the target lowered to the rate actually achieved, which is what
        # `run_dial_test.py --fallback-best-feas` then tests.
        if elig.empty:
            best_feas = None
        else:
            top = elig[np.isclose(elig["feasibility"],
                                  elig["feasibility"].max())]
            best_feas = top.loc[top["objective"].idxmin() if sense == "min"
                                else top["objective"].idxmax()]
        close = dict(
            best_feasibility=(float(best_feas["feasibility"])
                              if best_feas is not None else np.nan),
            best_feas_dial=(float(best_feas["dial"])
                            if best_feas is not None else np.nan),
            best_feas_objective=(float(best_feas["objective"])
                                 if best_feas is not None else np.nan),
            best_feas_solved_frac=(float(best_feas["solved_frac"])
                                   if best_feas is not None else np.nan),
        )
        ok = g[g["feasibility"] >= target]
        if ok.empty:
            # Did the series run out of GRID or out of METHOD? Read off where its
            # best feasibility sits: at an end of the full configured grid, the
            # curve was still being cut off and a wider grid is the next move; one
            # cell inside it, the dial turned on its own and widening buys only
            # more of the falling tail.
            at_end = (np.isfinite(close["best_feas_dial"])
                      and float(close["best_feas_dial"]) in ends)
            reached = ("no cell scored a feasibility at all "
                       f"(solved_frac>={min_solved:g} everywhere unmet)"
                       if best_feas is None else
                       f"best was {close['best_feasibility']:.3f} at "
                       f"{g_all['dial_name'].iloc[0]}={close['best_feas_dial']:.5g}"
                       + (" -- AT A GRID END, so the grid may be the limit rather "
                          "than the method" if at_end else
                          " -- inside the grid, so the dial turned before the grid "
                          "ran out"))
            rows.append(dict(method=method, rho=rho, dial_name=g_all["dial_name"].iloc[0],
                             dial_star=np.nan, feasibility=np.nan, objective=np.nan,
                             solved_frac=np.nan, n_capped=0, master_time_s=np.nan,
                             test_time_per_point_s=np.nan,
                             bound=("none_grid_end" if at_end else "none_interior"),
                             note=f"never reaches feas>={target:g} at "
                                  f"solved_frac>={min_solved:g}; {reached}",
                             **prov, **close))
            continue
        best = ok.iloc[0]
        for _, r in ok.iterrows():
            if better(float(r["objective"]), float(best["objective"])):
                best = r
        notes = []
        if prov["monotone_note"]:
            # The search prunes on monotonicity. If the cells it DID score are
            # not ordered, the pruned ones are not evidence and this point is
            # provisional -- say so on the row, not only in the log.
            notes.append("NON-MONOTONE among the scored cells -- re-run this "
                         "series with --search grid")
        if float(best["dial"]) in ends:
            notes.append("at a grid end -- the dial grid may be the limit, "
                         "not the method")
        if int(best.get("n_capped", 0) or 0):
            notes.append(f"n_capped={int(best['n_capped'])} (incumbent, not converged)")
        rows.append(dict(method=method, rho=rho, dial_name=str(best["dial_name"]),
                         dial_star=float(best["dial"]),
                         feasibility=float(best["feasibility"]),
                         objective=float(best["objective"]),
                         solved_frac=float(best["solved_frac"]),
                         n_capped=int(best.get("n_capped", 0) or 0),
                         master_time_s=float(best["master_time_s"]),
                         test_time_per_point_s=float(best["test_time_per_point_s"]),
                         bound=("grid_end" if float(best["dial"]) in ends
                                else "interior"),
                         note="; ".join(notes), **prov, **close))
    for _, r in ref.iterrows():
        # nominal: no dial, so no protocol point -- carried as the reference level
        # the whole plot is read against.
        rows.append(dict(method=r["method"], rho=np.nan, dial_name="none",
                         dial_star=np.nan, feasibility=float(r["feasibility"]),
                         objective=float(r["objective"]),
                         solved_frac=float(r["solved_frac"]),
                         n_capped=int(r.get("n_capped", 0) or 0),
                         master_time_s=float(r["master_time_s"]),
                         test_time_per_point_s=float(r["test_time_per_point_s"]),
                         bound="no_dial", note="reference level, nothing to move",
                         search="none", n_grid=1, n_evaluated=1, n_pruned=0,
                         n_unreached=0, monotone_note="", feas_wobble="",
                         best_feasibility=float(r["feasibility"]),
                         best_feas_dial=np.nan,
                         best_feas_objective=float(r["objective"]),
                         best_feas_solved_frac=float(r["solved_frac"])))
    out = pd.DataFrame(rows)
    out.insert(0, "feas_target", target)
    out.insert(1, "min_solved", min_solved)
    out.insert(2, "objective_sense", sense)
    path = os.path.join(OUT_DIR, f"{problem}_dial_star{out_suffix}.csv")
    out.to_csv(path, index=False)
    print(f"\n[dial-sweep] protocol point per series "
          f"(feasibility >= {target:g}, solved_frac >= {min_solved:g}, "
          f"best objective, sense={sense})")
    print(out.to_string(index=False))
    print(f"[dial-sweep] wrote {path}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=("synthetic", "reactor", "gastric"),
                   default="gastric")
    p.add_argument("--cell-tag", default=None,
                   help="extra token on the cell suffix, for a PROBE that runs "
                        "part of the sweep -- one method, or a rho column the "
                        "shared grid does not carry. Without it such a run is "
                        "refused, because the curve and star are rewritten from "
                        "the rows the run itself scored and the untouched series "
                        "would be dropped. A tagged cell is its own experiment: "
                        "it resumes nothing from the untagged one, and "
                        "run_dial_test.py does not look for it")
    p.add_argument("--methods", nargs="+", default=None,
                   help="default: nominal cp wrapper margin cmicl. robust_reg is "
                        "NOT on this axis -- its dial is the radius, so at a fixed "
                        "rho it has none; it stays on run_rho_sweep.py")
    p.add_argument("--drop-series", nargs="+", default=None, metavar="METHOD@RHO",
                   help="series this run intentionally removes from the cell's "
                        "curve and star table, e.g. `wrapper@2 wrapper@3` after "
                        "moving a method to its own rho columns. The narrow "
                        "--refresh: the guard still fires on anything else that "
                        "would be lost, and the score checkpoint is KEPT, so the "
                        "series that survive are not re-solved")
    p.add_argument("--rho-columns", type=float, nargs="+", default=None,
                   help=f"rho values to run the D-facing methods at "
                        f"(default {DEFAULT_RHO_COLUMNS}, with per-method "
                        f"overrides {METHOD_RHO_COLUMNS}). Passing this puts "
                        f"EVERY method on the given columns, overriding both. "
                        f"All columns share one output CSV; the rho column tells "
                        f"them apart")
    p.add_argument("--tau-grid", type=float, nargs="+", default=None,
                   help=f"CP's tau grid, absolute, in unexplained-sd units "
                        f"(default {TAU_GRID}). FIXED BEFORE THE RUN and the same "
                        f"on every rho column: tau is a parameter of the method, "
                        f"not a statistic read back off it")
    p.add_argument("--alpha-grid", type=float, nargs="+", default=None,
                   help=f"wrapper alpha grid (default {DEFAULT_ALPHA_GRID}); "
                        "0/0.1/0.2/0.5 are OptiCL's published WFP values. Only "
                        "multiples of 1/P are distinct levels; 1.0 is excluded "
                        "because it removes the constraint rather than loosening "
                        "it")
    p.add_argument("--margin-grid", type=float, nargs="+", default=None,
                   help=f"margin m grid, unexplained-sd units "
                        f"(default {DEFAULT_MARGIN_GRID}); m=0 IS nominal")
    p.add_argument("--cmicl-alpha-grid", type=float, nargs="+", default=None,
                   help=f"C-MICL miscoverage grid (default "
                        f"{DEFAULT_CMICL_ALPHA_GRID}, plus any per-problem lower "
                        f"extension in CMICL_ALPHA_GRID_EXTRA="
                        f"{CMICL_ALPHA_GRID_EXTRA}). Extended upward on purpose -- "
                        f"where it FIRST solves on gastric is the result; the "
                        f"bottom is what n_cal can certify")
    p.add_argument("--cp-alpha-ablate", action="store_true",
                   help="after the sweep, hold tau at tau* and walk CP's coverage "
                        "cap. Contextual problems only (gastric)")
    p.add_argument("--cp-alpha-rho", type=float, default=None,
                   help="rho column for the coverage-cap ablation (default: the "
                        "largest column). gastric 1.0, reactor 3.0")
    p.add_argument("--cp-alpha-grid", type=float, nargs="+", default=None,
                   help=f"coverage-cap values (default {DEFAULT_CP_ALPHA_GRID}); "
                        "0 is the production pin")
    p.add_argument("--search", choices=("adaptive", "grid"), default="adaptive",
                   help="how each series' dial grid is walked. adaptive "
                        "(default): bisect on the robustness order to bracket "
                        "the feasibility target, then spend what is left of "
                        "--max-evals filling the band around it; a cell scoring "
                        "feasibility 0 prunes everything LESS robust, and one "
                        "below --min-solved prunes everything MORE robust. That "
                        "ASSUMES the dial is monotone -- violations among the "
                        "cells actually scored are printed and land in the star "
                        "table's monotone_note. grid: walk the whole grid, which "
                        "assumes nothing and is the fallback when it does")
    p.add_argument("--max-evals", type=int, default=None,
                   help="cells scored per series under --search adaptive "
                        "(default ceil(log2 n) + 2, floored at 4). Cells already "
                        "on the score checkpoint are free and do not count")
    p.add_argument("--feas-target", type=float, default=0.9,
                   help="held-out feasibility target (default 0.9). Also fixes "
                        "C-MICL's protocol point at 1 - target")
    p.add_argument("--min-solved", type=float, default=0.5,
                   help="drop cells below this solved fraction before reading the "
                        "protocol point (default 0.5)")
    p.add_argument("--incoherent", dest="coherent", action="store_false",
                   default=False)
    p.add_argument("--coherent", dest="coherent", action="store_true")
    p.add_argument("--match-bank", action="store_true",
                   help="set CP's bank B to the wrapper's P, removing the B!=P "
                        "confound")
    p.add_argument("--n-folds", type=int, default=None,
                   help="single-decision problems only; default "
                        "cv_calibration.n_kfold")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the DRAWS FROM D only; the data and the folds "
                        "keep the config seed. Scopes every output with _s<seed>")
    p.add_argument("--refresh", action="store_true",
                   help="discard EVERY output of this cell first -- scores, "
                        "contexts, curve, star and skipped. Needed whenever a "
                        "previous run of the same cell is not comparable "
                        "(changed grid, changed scoring); without it the "
                        "checkpoint resumes those cells silently")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--cv-configs", default="results/cv/gastric_selected_configs.json")
    args = p.parse_args()

    config = load_config(args.config)

    # Config-only, exactly as run_dial_test.py resolves it -- there is no
    # `--separation` flag on either, so the two stages cannot disagree about the
    # cell name. See run_rho_sweep.main for why the flag was removed.
    args.separation = (
        config.get("methods", {}).get("cp", {}).get("separation", "auto"))
    if args.problem in ("synthetic", "reactor"):
        args.n_folds = _synth_n_folds(config, args)
        from src.data.instances import synth_model_spec, reactor_model_spec
        spec = (synth_model_spec if args.problem == "synthetic"
                else reactor_model_spec)
        args.synth_model = spec(config)[0]

    run(config, args)


if __name__ == "__main__":
    main()
