# Reactor and C-MICL: instance facts

Moved verbatim from `CLAUDE.md` on 2026-11-16. Still binding for Paper A, where the
reactor is the lead instance and C-MICL a baseline. Paper A runs use Ovalle et al.'s
100 cost vectors (`--costs paper`); the fixed `balanced` cost below is the legacy
pipeline's.

## Frozen CV picks

**Frozen CV picks** (`results/cv/*_selected_configs.json`, chosen on R^2 *before*
any robustness): gastric **XGB** for DLT/blood/constitutional/infection and
**linear (ElasticNet)** for GI and OS; synthetic **`mlp` (32,32), lbfgs, alpha
0.01**; reactor **`mlp` (10,5,2)**. Synthetic's was `(50,)` until 2026-09-03,
when Ovalle et al.'s own `(32,32)` went into the MLP grid as a candidate (the
note beside `CV_PARAM_GRIDS`) and won by **6e-5** R^2 — a tie broken in the last
decimal, kept only because it reproduces bit-identically AND no current
synthetic result was against the old pick. The reactor keeps `(10,5,2)` with
their point in the grid (every reactor CV output byte-identical) and gastric
keeps its six. All three CV stages reproduce bit-identically,
so a re-run that differs means a **code or label** change, not noise. `run_cv.py`
**overwrites** those files — `git checkout` them after an inspection run unless you
mean to re-freeze.

## Reactor instance and C-MICL

- **The reactor's `cost_vector` is FIXED — at `1/span_i` since 2026-09-03**
  (`reactor.cost_vector: "balanced"`, resolved by `instances.reactor_cost_vector`).
  A new `c` is a different problem, not a new sample of this one, so it is fixed
  for the life of a result set — but it is **no longer ones**: the raw units are
  not commensurate (`v0`, `v_He` span 1050 each, `T` 351, against `dt`'s 1.5), so
  under ones `dt` carried **0.06%** of the objective range and `L` 3.5%, and the
  comparison ran on three of the five variables. Balanced gives each 20% and puts
  the objective in box-widths, range `[4.143, 9.143]` where ones gave
  `[1908, 4450]`. **Every reactor number from before 2026-09-03 is against a
  different objective** — feasibility included, since `x*` moves. `"ones"`
  reproduces them. This is NOT C-MICL's per-instance draw and deliberately so:
  theirs is an average over 100 cost vectors, which is the probe's job (below),
  and their seed-0 draw is itself unbalanced (`v_He` 70% of the span, `L` 0.2%).
  D is untouched — rho is in units of `scale(y)*sqrt(n)` on the benzene labels —
  so the tau/alpha/margin/cmicl grids stay placed, but **no measured rho column
  transfers**: both were re-derived on the balanced instance by the 2026-09-04
  full-span run — CP {1, 2}, wrapper {3, 4} (see `METHOD_RHO_COLUMNS`).
- C-MICL's reactor feasibility is a
  property of that distribution: **0.99** under `c_i ~ U(0,1)` vs **0.11** under
  `c_i ~ U(0,1)/span_i` (`probe_cmicl_cost_sampling.py`, a diagnostic that changes
  nothing in the evaluation). **Never quote a C-MICL reactor feasibility without
  naming the `c` distribution.** Ovalle et al.'s own draw is now known from their
  code — `U(-4,4)` with negative components `/10`, against their `/100`-scaled
  variables (`regression.py:713-719`) — which is **`--schemes paper`** and lands
  *between* those two, so **neither 0.99 nor 0.11 is their number** and both
  bullets above are superseded as answers to "does the reactor match the paper".
- **We are on their exact reactor instance, verified 2026-09-03** against the
  dataset in their repo: same column order `(v0, v_He, T, dt, L, F_C6H6)`, same
  box as `DECISION_RANGES`, all seven domain constraints identical once their
  `/100` input scaling is undone (their `L <= 1.5 dt` *is* our `L/dt <= 150`), and
  `benzene_flow` reproduces their labels over 120 rows at a mean of **-1.37%**
  with residual sd **1.94** — their own label noise (`noise_std: 2.0`), not our
  ODE disagreeing. So the -1.6% Table 1 offset is real and systematic, and our
  `F_C6H6` still should not be quoted as a reproduction of theirs.
- **Their code is `https://github.com/dovallev/c-micl`**, audited 2026-09-03 at
  commit `b44fe53`: `regression.py` is the script their README designates for
  the paper's regression case study — its grid matches the paper's protocol (100
  cost draws at seed 0, four surrogate families, `P` ∈ {1,5,10,25,50}, alpha ∈
  {0.1, 0.05}) but it **aggregates nothing, ships no outputs, and is
  non-Mondrian**, so "it produced the paper's numbers" is an inference from
  structure and **not** a verified reproduction (cmicl.py DIFFERENCES #8). The
  repo's only committed number, notebook 03's `0.800`, disagrees with the
  paper's `>=0.90`. Also:
  `notebooks/regression/03` is a *second, different* implementation of the same
  method (a `1e-6` width floor, 300 epochs, free `y_f`) — we follow the script.
  The point-by-point diff against ours — `q` one order statistic apart, `h`
  hand-fixed there vs CV-selected here, Keras vs sklearn `u`, their `/100`-`/10`
  scaled units, and that neither side is Mondrian — is the **DIFFERENCES**
  section of `src/methods/cmicl.py`. **Read it before quoting a `cmicl` number
  as theirs.**
- **Gastric C-MICL only solves at a loose alpha**, which is why its grid runs the
  **full [0.02, 1]** of a miscoverage level — where it *first solves* is the
  result. Measured under Ovalle et al.'s raw-`u` semantics with batch 32
  (2026-09-23): alpha=0.6 solves **83.9%** of contexts and delivers (feas
  0.929), while the protocol alpha=0.1 solves **5.6%** — below `--min-solved`,
  which is why the whole alpha<0.1 tail is *pruned* rather than scored (89.3% /
  8.1% before batch 32). That ~6% is the finding, not a gap: at the level it
  asserts, C-MICL declines to prescribe for ~94% of the cohort, and the
  feasibility it reports (1.000 tuned) is conditional on the few it solved. The older reading — infeasible at alpha=0.1 under both
  multiplicity settings, half-widths 1.33-1.73 sd(y) against an rhs of 0.6 on five
  constraints at once, 13.8% solved at alpha=0.5 — is the **floored-`u`** regime
  and does not carry over. Budget for it either way: a cell that proves infeasible
  cost **176 s** against nominal's 0.9 s.
