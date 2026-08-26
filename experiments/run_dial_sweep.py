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
    cp          tau       gastric {0.5, 1.0}, reactor {1,2}  ONE fixed tau grid
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
   of it into the two dead tails. That let the grids get FINER at lower cost:
   ``TAU_GRID`` is 7 half-decades where it was 4 decades, and every old value is
   still in it, so an existing checkpoint resumes rather than being orphaned.

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
    python experiments/run_dial_sweep.py --problem reactor --rho-columns 1 2 3
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
    append_contexts,
)
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
#   reactor  1 / 2 -- nominal misses the benzene target by ~4 units of F and
#            rho=1 buys ~2.2, so 2 is right at the edge. Add 3 if it is short:
#            --rho-columns 1 2 3.
DEFAULT_RHO_COLUMNS = {"gastric": [0.5, 1.0], "reactor": [1.0, 2.0],
                       "synthetic": [0.5, 1.0]}

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
TAU_GRID = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

# All three below are supersets of their pre-2026-08-26 values, for the same
# resume reason.
DEFAULT_ALPHA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
# m=0 is bit-identical to nominal (same fit, same MIP, same x*), so the baseline's
# curve starts AT the nominal point rather than near it.
DEFAULT_MARGIN_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75, 0.875, 1.0,
                       1.25, 1.5]
# Extended UPWARD on purpose. C-MICL is measured infeasible on gastric at
# alpha=0.1 under both multiplicity settings, and n_cal=80 means alpha >= 0.02 is
# needed for a finite conformal quantile at all. Where it FIRST solves is the
# result; 0.1 is the protocol point whether or not it solves there.
DEFAULT_CMICL_ALPHA_GRID = [0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3,
                            0.4, 0.5]
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


def _order_check(dname, dial_at, seq, seen, detail, target, min_solved,
                 feas_resolution=0.0):
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

    ``feas_resolution`` is the smallest feasibility difference that is not one
    fold flipping (``1/n_folds`` on a single-decision problem, where each fold
    contributes ONE binary outcome; 0 on a contextual one, where a fold is a mean
    over its held-out cohort). Wobbles are quoted in folds where it is known,
    because "0.800 -> 0.700" on the 10-fold reactor is one fold and reads as far
    more than that in decimals.
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
        how = (f"{(fx - fy) / feas_resolution:.0f} fold(s)"
               if feas_resolution > 0 else f"{fx - fy:.3f}")
        side = "above" if deliv[x] else "below"
        wobble.append(f"feasibility dips {fx:.3f}->{fy:.3f} ({how}) between "
                      f"{dname}={dial_at(x):.5g} and {dial_at(y):.5g}, both "
                      f"{side} the target")
    return viol, wobble


def _search_dials(method, grid, evaluate, known, target, min_solved,
                  max_evals=None, must_visit=(), label="",
                  feas_resolution=0.0):
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
                                target, min_solved, feas_resolution)

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
    var = _variant_suffix(args)

    scores_path = os.path.join(OUT_DIR, f"{problem}_dial_scores{var}.csv")
    ctx_path = os.path.join(OUT_DIR, f"{problem}_dial_contexts{var}.csv")
    if args.refresh:
        # EVERY output of this cell, not just the two that feed the resume. A
        # --refresh that left the curve and the star behind is a trap: the curve
        # is rewritten after each cell, so it recovers, but `{problem}_dial_star`
        # is written only at the END of the sweep -- so a refreshed run that
        # times out leaves the PREVIOUS run's star on disk, and
        # `run_dial_test.py` reads exactly that file to decide where to hold each
        # method. Deleting it up front makes a missing star an obvious failure
        # rather than a silently stale one.
        for pth in (scores_path, ctx_path,
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
    rho_cols = [float(r) for r in (args.rho_columns
                                   or DEFAULT_RHO_COLUMNS[problem])]
    d_methods = [m for m in methods if m in FACES_D]
    flat_methods = [m for m in methods if m not in FACES_D]

    print(f"[dial-sweep] problem={problem} geometry=ellipsoid "
          f"coherent={args.coherent} folds={len(su.folds)} seed={bank_seed} "
          f"sense={su.oracle.objective_sense}", flush=True)
    print(f"[dial-sweep] rho columns={rho_cols} (D-facing: {d_methods or 'none'})",
          flush=True)
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
    # The smallest feasibility difference that is not one fold flipping. On a
    # single-decision problem each fold contributes ONE binary outcome, so the
    # cell score is a binomial proportion over len(folds) draws and 1/len(folds)
    # is the resolution limit -- 0.1 on the 10-fold reactor, where "0.800 ->
    # 0.700" is one fold. On a contextual problem a fold is a mean over its own
    # held-out cohort, so there is no such quantum.
    feas_res = 0.0 if su.contextual else 1.0 / max(len(su.folds), 1)

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
                feas_res)
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
            feas_resolution=feas_res)
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
        if "cp" in d_methods:
            # tau is FIXED BEFORE THE RUN and is the same on every rho column.
            # It is not read off any iteration-0 distance -- see TAU_GRID.
            tau_vals = [float(t) for t in (args.tau_grid or TAU_GRID)]
            print(f"[dial-sweep] tau grid (fixed, same on every rho column): "
                  f"{tau_vals} -- unexplained-sd units, set before the run",
                  flush=True)

        for method in d_methods:
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
            "cmicl": args.cmicl_alpha_grid or DEFAULT_CMICL_ALPHA_GRID,
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
    rho_a = float(args.cp_alpha_rho if args.cp_alpha_rho is not None
                  else max(args.rho_columns or DEFAULT_RHO_COLUMNS[problem]))
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

    ``bound`` says what stopped the search, because the three cases mean different
    things: ``grid_end`` (the winning cell sits at an end of the dial grid, so the
    grid may be the limit rather than the method), ``interior`` (a genuine
    optimum), ``none`` (the series never reaches the target at all).

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
        best_feas = (elig.loc[elig["feasibility"].idxmax()]
                     if not elig.empty else None)
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
            reached = ("no cell scored a feasibility at all "
                       f"(solved_frac>={min_solved:g} everywhere unmet)"
                       if best_feas is None else
                       f"best was {close['best_feasibility']:.3f} at "
                       f"{g_all['dial_name'].iloc[0]}={close['best_feas_dial']:.5g}")
            rows.append(dict(method=method, rho=rho, dial_name=g_all["dial_name"].iloc[0],
                             dial_star=np.nan, feasibility=np.nan, objective=np.nan,
                             solved_frac=np.nan, n_capped=0, master_time_s=np.nan,
                             test_time_per_point_s=np.nan, bound="none",
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
    p.add_argument("--methods", nargs="+", default=None,
                   help="default: nominal cp wrapper margin cmicl. robust_reg is "
                        "NOT on this axis -- its dial is the radius, so at a fixed "
                        "rho it has none; it stays on run_rho_sweep.py")
    p.add_argument("--rho-columns", type=float, nargs="+", default=None,
                   help=f"rho values to run the D-facing methods at "
                        f"(default {DEFAULT_RHO_COLUMNS}). Both columns share one "
                        f"output CSV; the rho column tells them apart")
    p.add_argument("--tau-grid", type=float, nargs="+", default=None,
                   help=f"CP's tau grid, absolute, in unexplained-sd units "
                        f"(default {TAU_GRID}). FIXED BEFORE THE RUN and the same "
                        f"on every rho column: tau is a parameter of the method, "
                        f"not a statistic read back off it")
    p.add_argument("--alpha-grid", type=float, nargs="+", default=None,
                   help=f"wrapper alpha grid (default {DEFAULT_ALPHA_GRID}); "
                        "0/0.1/0.2/0.5 are OptiCL's published WFP values")
    p.add_argument("--margin-grid", type=float, nargs="+", default=None,
                   help=f"margin m grid, unexplained-sd units "
                        f"(default {DEFAULT_MARGIN_GRID}); m=0 IS nominal")
    p.add_argument("--cmicl-alpha-grid", type=float, nargs="+", default=None,
                   help=f"C-MICL miscoverage grid (default "
                        f"{DEFAULT_CMICL_ALPHA_GRID}). Extended upward on purpose: "
                        f"where it FIRST solves on gastric is the result")
    p.add_argument("--cp-alpha-ablate", action="store_true",
                   help="after the sweep, hold tau at tau* and walk CP's coverage "
                        "cap. Contextual problems only (gastric)")
    p.add_argument("--cp-alpha-rho", type=float, default=None,
                   help="rho column for the coverage-cap ablation (default: the "
                        "largest column). gastric 1.0, reactor 2.0")
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
    p.add_argument("--separation", dest="separation", default=None,
                   choices=("auto", "coherent", "incoherent"))
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

    import yaml
    config = yaml.safe_load(open(args.config))

    if args.separation is None:
        args.separation = (
            config.get("methods", {}).get("cp", {}).get("separation", "auto"))
    if args.problem in ("synthetic", "reactor"):
        args.n_folds = _synth_n_folds(config, args)
        from experiments.run_sweep import synth_model_spec, reactor_model_spec
        spec = (synth_model_spec if args.problem == "synthetic"
                else reactor_model_spec)
        args.synth_model = spec(config)[0]

    run(config, args)


if __name__ == "__main__":
    main()
