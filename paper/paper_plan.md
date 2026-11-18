# Paper A plan

Revised 2026-11-18. Outline: `paper_outline.md` (version 4). Proofs:
`paper_a_guarantee.md` (its Sections 5 and 6 are superseded by this file).
Results so far: the go/no-go probe (job 29141177, summarized below; full
write-up in `notes/progress.md`, 2026-11-14) and the local checks under
"Results so far". No baseline has been run on the paper's design.

## After the meeting with DB (2026-11-16), revised after review (2026-11-18)

**Decisions.**

- Exploratory: edit the outline and this plan, not `paper.tex` (DB).
- The RC stays. DB also wants a non-parametric method: cutting planes whose
  separation is solved by first-order methods (FO-CP; his sketch is
  `c:\Users\jsitu\Downloads\robust-cl-cp.jpeg`). **One paper, two methods.**
  In the paper FO-CP is "for any model trained by a differentiable loss",
  not "non-parametric": it needs a differentiable training map. The RC
  shares its constraint form, not its radius argument (see Radius).
- Semi-synthetic data first, real data later (DB). Instances come from those
  widely used in the optimization literature, not from new constraints on
  the reactor. **WFP as regression, with its closed-form truth, is the
  second instance.** Semi-synthetic chemotherapy (SS2) and the infeasibility
  probe move to the RC track. Workload dispatching (48 learned constraints)
  is optional. Classification comes after regression.
- The objective stays nominal; a robust objective (epigraph) is a deferred
  variant.
- Literature: `paper_a_literature.md`, "FO-CP, robust-optimization machinery
  and instances". Every building block of FO-CP is prior work (bilevel label
  poisoning, influence functions, cutting sets with local oracles); the
  combination was not found. **Read Kinsey et al. 2022 and Zhang, Amin and
  Perakis 2026 in full before claiming it.**

**FO-CP.**

- **Uncertainty set: one ellipsoid per outcome**, as in the RC:
  `‖Σₖₖ^{-1/2}Δyₖ‖ ≤ Rₖ`, `Δyₖ` one entry per training row of outcome k,
  `σᵢₖ²` on the diagonal (Ben-Tal, El Ghaoui and Nemirovski 2009, Sec. 1.2:
  the RC is constraint-wise). At equal R it gives the same robust
  constraints as one ellipsoid over all outcomes; the model-free radius is
  smaller (Radius). A constraint containing several outcomes' models gets
  one set over those outcomes' labels.
- **Loop.** Solve the nominal problem for `x⁰`. At `x^t`, for each protected
  constraint k, maximize the retrained prediction `ĥₖ(x^t; ȳ + Δy)` over
  its set. If the maximum exceeds `bₖ + τ`, add the model retrained on
  `ȳ + Δy*` as a cut on k. Re-solve; stop when no cut is added, or at a
  fixed iteration cap. (DB's weighted ReLU sum, split by constraint:
  per-constraint cuts are deeper and have no zero gradient.)
- **Stopping and τ.** The guarantee is stated for `bₖ + τ` (or b is
  tightened by τ in the loop). Separation is local, so stopping certifies
  nothing by itself (Mutapcic and Boyd 2009). The final decision is checked
  a posteriori with a stronger attack (more starts, more steps), and the gap
  is reported.
- **Weights are binary, chosen by the master** (Gorissen et al. 2015,
  "K out of N"; ours adds the nominal rows): `zₖ ∈ {0, 1}`,
  `Σₖ zₖ ≥ ⌈pK⌉`, k's robust cuts enforced only if `zₖ = 1` (big-M). The
  guarantee covers the protected set: a protected constraint is violated
  only if the noise leaves its set (for the RC's radius, `Wₖ > Rₖ`; for the
  model-free radius, `‖Σₖₖ^{-1/2}eₖ‖ > Rₖ`), whichever constraints the
  master chose **[to prove: short]**.
- **Training map** (the review found it ill-defined; fixed here):
  - every retraining starts from one fixed `θ₀` (the nominal fit), never
    from the previous ascent step, so the map is a function of the labels;
  - lbfgs to a gradient tolerance, not an iteration cap;
  - label standardization uses constants fixed from `ȳ` (the probe
    standardizes by each label vector's own mean and sd,
    `probe_rc_ridge.py:106`, which is not affine);
  - damped Hessian `H + λI` for the implicit gradient (dead ReLUs make H
    singular), checked against finite-difference retraining;
  - the clean-label reference is the same map on `y*`; the cold-start fit
    on `y*` (what R1 to R4 used) is reported beside it.
- **Ascent.** Gradient by implicit differentiation,
  `∇_Δy ĥ(x^t) = J (H + λI)⁻¹ ∇_θ f(x^t; θ)`, one linear solve. In whitened
  coordinates step to `z = R g̃/‖g̃‖`, or a projected step if that
  oscillates. Starts: zero, the linearized direction, random. The best
  candidate is confirmed by retraining before it becomes a cut.
- **Relation to the RC and the old CP.** For a model affine in its labels
  the gradient is `s(x)` and one step gives `R s(x)/‖s(x)‖`, the RC's cut.
  On a network the first step is the RC on the linearized network (Koh and
  Liang's influence). The random bank becomes an ablation of the
  separation. Retraining the whole network avoids the honest-fitting
  problem that broke the last-layer RC under standard fitting (R3: 0.76
  against a 0.90 target).
- **Cost.** Per separation: starts × steps × (retraining + one solve), per
  violated constraint, per iteration. Separations depend on `x^t`, hence on
  the cost vector, so they are not shared across the 100 cost vectors
  (one RC radius serves all of them). Each cut on an end-to-end network
  adds a full ReLU embedding (64 binaries for the (32, 32) network) to a
  master whose single solve already takes 15 to 100 seconds. F2 is piloted
  on 10 cost vectors to measure this.

**Radius.** Theorem 1 does not carry over. The RC's simulated radius (3.6σ
on cubic ridge) is not the radius of a ball containing the noise
(`‖e‖ ≈ 32σ`); it works because an affine model sees the noise only through
`s(x)ᵀe`, and W is the largest normalized error over the domain. FO-CP has
no `s(x)`.

| Option | Radius | Guarantee | Status |
|---|---|---|---|
| (a) Model-free ball | `√χ²(n, 1−ε/K)` per outcome, whitened (union bound) | Exact for any model, if separation were exact | **F−1 (job 29561590): impractical.** 32.5σ at n = 1,000. Infeasible on cubic ridge in 100 of 100 redraws; on the frozen network feasible, 1.00 against the ODE, but +15.7% of the objective range against +2.2% at the simulated radius. Stays as the exact but conservative reference |
| (b) Linearized subspace | `√χ²(d_eff, 1−ε)`, `Δy` in the linearized map's span | Exact for affine maps only | If F1 finds d_eff ≈ n, (b) ≈ (a) |
| **(c) Sampled W (chosen 2026-11-19)** | For N sampled noise draws e′, retrain on `ȳ + e′` and compute `W_c = max over x ∈ X of [ĥ(x; ȳ+e′) − ĥ(x; ȳ)] / ‖s̃(x)‖`, with s̃ the linearized sensitivity at θ̂ and X the whole decision domain (fixed, label-free); R is the ⌈(N+1)(1−ε)⌉-th smallest. **Computed once, before the loop**; one R for every iteration and every cost vector; the ascent searches the ball of that radius | Exact for affine models: the numerator is `s(x)ᵀe′`, so this is Theorem 1's simulated radius. For networks, first order (a parametric bootstrap; s̃ depends on the fitted weights) | F2's radius. Scenario optimization (imposing about 361 retrained models) is not pursued |

**Instances.**

| # | Instance | Truth | Role |
|---|---|---|---|
| SS1 | Reactor, Ovalle et al.'s 100 cost vectors, σ = 2 | Two arms, each with labels generated by its own truth: (i) the ODE; (ii) a (32, 32) network fitted to noise-free ODE labels | (ii) has (nearly) no model bias, so it tests label-noise protection alone; report the gap `ĥ(y*) − truth` |
| WFP | Palatable diet, regression (OptiCL; the most reused case in constraint learning) | OptiCL App. C.1's closed-form palatability function; its constants are not published: recover them from the public 5,000-row data or ask the authors | Second instance, for both methods |
| SS2 | Chemotherapy, real covariates, five toxicity constraints | a fixed fitted model per outcome; binomial noise from arm sizes | RC track: several constraints against a known truth, infeasibility handling (P2) |
| — | Workload dispatching (48 constraints; EML, OptiCL, SurrogateLIB) | thermal simulator, apparently not public | Optional |
| — | Student enrollment (JANOS, OptiCL) | fixed logistic model | Later, with classification |

**Probes.** F−1 to F2 decide whether FO-CP is worth building fully.

| # | Question | Design | Decides |
|---|---|---|---|
| F−1 | Is the model-free radius (a) affordable? | Existing code: `probe_rc_ridge.py --stage modelfree` at R = √χ²(n, 0.9) on `cubic` and `net_frozen`, the 100 redraws | **Done (job 29561590): no.** Cubic infeasible in all 100; network +15.7% against +2.2% (simulated). (a) is out as a practical radius; (c) must be defined |
| F0 | Is the machinery right? | (i) Affine models: one step equals `R‖s(x)‖`. (ii) End-to-end network: implicit gradient against finite-difference retraining. (iii) Loop on `net_frozen`: x* and objective converge to the RC's cone solution as τ → 0; record cuts | Code correct; gate |
| F1 | Does ascent beyond the first step find worse relabelings? | End-to-end (32, 32), a few redraws, at the nominal x* and at one robust iterate. Increase in the retrained prediction, in σ units, at equal R, for: the linearized step, FO multi-start, the random bank (B = 200), the true relabeling `Δy = −e`, cold restarts with other seeds. Also d_eff | If FO ≈ linearized, FO-CP reduces to one-step separation in the same loop. If cold restarts move ĥ as much, training randomness dominates label noise |
| F2 | Does FO-CP deliver on SS1, at what cost? | Radius (c), with (b) as a closed-form check where d_eff is small. Pilot: 10 cost vectors, SS1(ii). Then 100 cost vectors, both arms. On the end-to-end network: FO-CP, nominal, margin, wrapper, C-MICL (one model for every method); on `net_frozen`: FO-CP beside the last-layer RC. Judged against the clean-label model and the arm's truth | H4; the radius |

**Tasks.**

- **T7.** What stopping shows: with exact separation and τ > 0, robust
  feasibility at `b + τ` after finitely many cuts; with continuous decisions
  cuts converge only in the limit (Kelley-type; T3's K-iteration bound is
  for trees). With local separation, nothing beyond the a posteriori check.
  The protected-set guarantee for both radius events.
- **T8.** Radius (c): exact for affine models (reduces to T1); a
  first-order argument for networks. (a) is short and kept as the
  conservative reference.
- **I11.** The training map in sklearn (lbfgs, `warm_start` from the fixed
  `θ₀`) and the Hessian in numpy (1,281 weights). PyTorch only if that is too
  slow.
- **I12.** The separation: implicit gradient, normalized step, starts,
  retraining check, a posteriori attack.
- **I13.** A slim master (about 200 lines) on the probe's own embedding, not
  `cp.py` (`embed_model` handles only sklearn types; `cp.py` carries anchors,
  rollback and eviction). Hidden layers embedded once when only the last
  layer changes; `zₖ` big-M.
- **I14.** SS1's realizable truth; the WFP regression instance (recover the
  truth function); SS2's generator.

**Risks.**

- Local separation gives no certificate; the a posteriori attack measures
  the gap but does not close it.
- Radius (a) may be too large to use (F−1), and (c) guarantees only against
  a resampled model.
- FO-CP sees label noise, not bias: on the end-to-end network it may miss
  against the ODE as the network RC did. SS1(ii) separates the two.
- The master grows by one embedded network per cut, and separations are not
  shared across cost vectors.
- Referees will ask: convergence and what is certified at stopping (T7);
  a scenario-optimization baseline; links to adversarial training (min-max);
  runtime against n and cuts, against the RC; the separation gap;
  robustness to training randomness; why an ℓ₂ label ball (it contains
  coherent shifts independent noise does not produce).

**Open:** the reactor's lead model (network or cubic ridge; below).

## Results of job 29141177 and the lead-model proposal (2026-11-14)

Reactor, Ovalle et al.'s 100 cost vectors, noise redrawn per run, ε = 0.1,
σ = 2 known, `experiments/probe_rc_ridge.py`. The network is (32, 32): R2
freezes its hidden layers on an independent noise draw (the theorem holds
exactly), R4 fits them on half the rows and the last layer on the other half,
R3 fits everything on the same labels (standard fitting). Feasibility against
the clean-label model / the ODE; cost as % of the objective range. Outputs:
`results/paper_a/prelim/redraw_*_paper_eps0.1_*.json`.

| Run | Nominal, ODE | Pointwise | Simulated | Closed-form | Cost, simulated / closed-form |
|---|---|---|---|---|---|
| R1 cubic ridge | 0.66 | 0.90 / 0.80 | **1.00 / 0.95** | 1.00 / 1.00 | 4.2% / 8.6% |
| R2 network, frozen | 0.18 | 0.91 / 0.22 | 1.00 / 0.52 | 1.00 / 0.79 | 2.2% / 3.3% |
| R4 network, sample-split | 0.25 | 0.92 / 0.28 | 1.00 / 0.42 | 1.00 / 0.61 | 3.1% / 5.2% |
| R3 network, standard | 0.23 | 0.42 / 0.29 | 0.76 / 0.33 | 0.98 / 0.45 | 1.9% / 3.6% |

- **Cubic ridge works**: nominal fails (0.66), the RC delivers 0.95.
- **The network fails against the ODE because of bias**: at the nominal
  optimum its clean-label model overstates the ODE in 80 to 90% of runs, so
  the optimizer selects on misfit that a label-noise margin cannot see.
- **Standard fitting breaks the guarantee** (R3: 0.76 against a 0.90 target).
- **No selection effect at ε = 0.1**: the pointwise radius already meets the
  target against the clean-label model (also at ε = 0.05 and 0.2, jobs
  29142767-8). The simulated radius's extra margin pays for bias spread.
- **The solver-certified radius is needed**: enumeration understates `W` by
  0.10 to 0.24σ on the network.

**Proposed for DB, not adopted** (still open after 2026-11-16):

- Cubic ridge leads on the reactor; the network becomes the example of where
  a label-noise margin is not enough. E1's baselines on cubic ridge, plus
  C-MICL as published on the network.
- The bias margin `B(x)` may need to be more than a remark.
- Try Ovalle et al.'s own surrogate families (`regression.py`): LinearDT
  (depth 5), random forest and gradient boosting (15 trees, depth 5) and the
  (32, 32) MLP. The tree families are affine in the labels with splits fixed;
  LinearDT is at the noise floor here (calibration MAE 2.08 at σ = 2) and is
  the candidate lead. Needs `linear-tree` and the leaf-vector code (I6).

## What the paper has to show

The paper is a practical method with a guarantee behind it. H1 to H3 are the
RC's; H4 is FO-CP's (added 2026-11-18).

| # | Point | Experiment | Our own go or no-go |
|---|---|---|---|
| H1 | **It delivers.** At its simulated radius for ε = 0.1, the robust counterpart (RC) reaches 0.9 ground-truth feasibility with no tuning, where the nominal decision does not. | E1 | nominal already near 0.9, or the RC under it |
| H2 | **It is cheaper.** At the same ground-truth feasibility, the RC has the best objective among the constant margin, the wrapper and C-MICL. | E1 | another method is cheaper at 0.9 |
| H3 | **It certifies more of what it prescribes.** With five learned toxicity constraints, the RC certifies a regimen at the target level for more cohorts than any baseline, gives every other cohort the regimen with the best certified level (or protects a fraction of the constraints), and says which cohorts no regimen can be certified for. Tested first on SS2 (known truth, P2), then on real data (E2). | P2, E2 | a baseline's decisions certify at least as well as the RC's at the same level |
| H4 | **FO-CP extends it.** For a model trained end to end, FO-CP reaches the target feasibility against the clean-label model where the last-layer RC under standard fitting does not (R3: 0.76), at a cost comparable to the baselines'. (Added 2026-11-18.) | F2 | radius (a) unaffordable (F−1) and (c) undefinable; or ascent finds nothing beyond the linearized step (F1) |

The last column is for us. The paper reports means with standard errors and
makes no formal tests. Feasibility is over solved runs, with the fraction
solved beside it.

How our margin differs from each baseline's:

| Method | Margin varies with `x`? | Where the margin comes from | Models embedded | Rows used to fit |
|---|---|---|---|---|
| Constant margin | no | chosen by the user | 1 | all |
| Wrapper | yes, implicitly | disagreement of P bootstrapped models | P | all (resampled) |
| C-MICL | yes | a second model trained on the first one's residuals, scaled by a conformal quantile | 2 | 80% (20% held out) |
| RC (ours) | yes | computed exactly from the fitted model and training inputs | 1 | all |

C-MICL's margin is also decision-dependent, so that is not our advantage over
it. The differences are that our width is computed, not learned, that the
guarantee holds at `x*`, and that one model is trained and embedded. C-MICL's
width has one advantage over ours: it can pick up model misfit, which `‖s(x)‖`
does not see.

## Why the earlier approaches are not the paper's method

The work up to September compared cutting planes over a random bank, the
wrapper, a right-hand-side margin and C-MICL, each tuned on its own parameter
(the dial sweep). Two of those stay as baselines. None is the method. Numbers
below are from the reactor dial cell (balanced cost, 10 folds, ODE test stage
of 2026-09-07) and the gastric cell (2026-09-08).

| Earlier approach | What it did | Why it is not the method | In the paper |
|---|---|---|---|
| Cutting planes over a random bank | Drew 200 relabelings from an ellipsoid of radius rho, retrained the model on each, cut the worst one at `x*`, stopped at tolerance tau. With several constraints, a cut that left a protected cohort infeasible was rolled back and rejected. | No guarantee: the bank is a finite sample of the set, so stopping shows `x*` is feasible for the drawn models only. Two parameters (rho, tau) with no rule tying either to a violation level; both were set against the ground truth, and the rho that works had to be re-measured per instance (rho = 1 never passed 0.30, rho = 2 delivered). The result also depends on the bank's seed. For a model affine in its labels the worst relabeling has a closed form, `R s(x)/‖s(x)‖`, so sampling is unnecessary. The rollback kept the master feasible by giving up robustness wherever it was unaffordable, so what was delivered had no stated level. | Out as a method. One sentence in Section 4: random-draw cuts carry no guarantee. **2026-11-16:** the cutting-plane loop returns with first-order separation (FO-CP, top of this file), and the random bank becomes its separation ablation. The rollback's role (prescribing when full robustness is infeasible) passes to the certified fallback; see "Infeasibility on chemotherapy". |
| Right-hand-side margin | Tightened the constraint by a constant `m` | The margin is the same at every `x`, and nothing says how large to make it: `m* = 4.0` was found only with the oracle. At 0.9 feasibility it was the most expensive of the three tuned methods (6.043, against 5.816 for cutting planes and 5.885 for the wrapper). On gastric a large `m` makes the problem infeasible instead of conservative (`m = 0.75` solved 20% of contexts). | Baseline. |
| Wrapper | Embedded 20 bootstrapped models and required a fraction to hold | The fraction is not tied to a violation level. At its strictest setting it reached 0.00 and 0.10 on the two rho columns where cutting planes operated, and it embeds 20 models. On gastric it solved 64 to 68% of contexts. | Baseline. |
| Robust regression | Trained a label-robust model and embedded it nominally | It acts on the learner, not the embedding. | Out. Paper B. |
| The dial sweep as the protocol | Tuned each method's parameter against an oracle and compared the tuned points | The paper's claim is that no tuning is needed, so a protocol built on tuning cannot test it. It also had one cost vector and 10 folds, so feasibility moved in steps of 0.1. | Its design survives as the equal-feasibility comparison (Figure 2), on 100 redraws. |
| Analytic synthetic instance | Known `f_true` | The reactor gives an exact truth on an instance from the literature. | Out. |

What the earlier work contributes to the paper: the failure itself (nominal
ODE-feasible in 0 of 10), and the finding that every tuned safeguard reached
0.9 only with ground truth to tune against. That is the motivation in
Section 1.

Since 2026-11-16 the cutting-plane loop is back as FO-CP: it retrains the
whole model on each relabeling, as the old method did, but finds the
relabeling by gradient ascent instead of sampling it, and takes its radius
from ε instead of tuning rho. The random bank appears only as FO-CP's
separation ablation (F1), not as a baseline.

## Scope (narrowed 2026-11-14, revised 2026-11-18)

Two methods; semi-synthetic instances first, real data last. The number of
displays is to be fixed once F−1 to F2 have run.

| # | Experiment | Problem | Design | Output | Paper |
|---|---|---|---|---|---|
| E1 | **Feasibility, cost and the radius** | reactor (SS1) | RC on the lead model (open: network or cubic ridge) and FO-CP on the end-to-end network. Ovalle et al.'s 100 cost vectors, noise redrawn for each. RC at the pointwise, simulated and closed-form radii for ε ∈ {0.05, 0.1, 0.2}, and on a radius grid. Constant margin, wrapper and C-MICL on the same model as the method they are compared with, at their prescribed settings and over their ranges. Nominal. | Table 1, Figure 2, Figure 3 | 6.1, 6.2 |
| E3 | **Second instance** | WFP, regression, closed-form truth | as E1, one ε, both methods | Table 3 | 6.4 |
| P2 | **Several constraints, known truth** | chemotherapy, semi-synthetic (SS2) | RC; certified fallback against protecting a fraction p of the constraints | appendix or 6.3 | 6.3 |
| E2 | **Real data** | chemotherapy | Maragno et al.'s protocol. RC with the joint radius over the test cohorts' treatment slices, certified fallback where infeasible; margin, wrapper, C-MICL, nominal on the same fixed models, each decision given its certified level | Table 2 | 6.3 |

Appendix, one table each, on the reactor:

- A1. Standard against honest fitting for a forest.
- A2. Non-Gaussian noise (Student-t, uniform).
- A3. Cubic ridge, the case where the theorem holds with no caveat (unless it
  becomes the lead model).

Cut from the paper:

- Baselines on two model classes (one model per comparison).
- Forest runs on the reactor other than A1.
- Sample-size, heteroscedastic-noise and cross-validated-ridge checks.
- Bonferroni as a reported result (it stays in the theory).
- The width-at-the-optimum comparison with C-MICL.
- Classification (after regression; student enrollment is the candidate).
- New constraints on the reactor.
- The analytic synthetic instance, the dial sweep, and the random bank as a
  method (it is FO-CP's separation ablation).

The several-constraints result is tested against a known truth on SS2 (P2);
workload dispatching (48 constraints) is the optional literature instance.

## Results so far (2026-11-14, local)

Script: `experiments/probe_rc_ridge.py`. Outputs: `results/paper_a/prelim/`.
All on the reactor with **cubic ridge** (56 features, fixed penalty, train R²
0.957), fixed balanced cost, σ = 2 known. This is not the paper's lead model.

**The solver works.** The RC is a nonconvex problem with one second-order
cone. Gurobi solves it to proven optimality in about 1 second, and finds
decisions about 0.1 cheaper in cost than enumeration over 42,000 sampled
points at every radius.

**The simulated radius works through the solver but is slow.** About 43
seconds per noise draw (four or five global solves), with upper and lower
bounds on `W` agreeing to 0.001. By enumeration over 1,000 draws the radius is
3.59σ at ε = 0.1 (3.84σ at 0.05, 3.37σ at 0.2). On 10 solver draws the
solver's `W` was 0.03 to 0.15σ above enumeration, and above 0.1σ in six of the
ten (`radius.json`). So the 0.1σ added to an enumerated radius does not cover
the gap, and the gap has not been measured for the network at all. The
closed-form radius is 8.36σ and the pointwise value 1.28σ.

**100 noise redraws, ε = 0.1, every run solved:**

| Radius | R | Feasible vs clean-label model | Feasible vs ODE | Mean cost | Mean margin at `x*` |
|---|---|---|---|---|---|
| Nominal | 0 | 0.30 | 0.96 | 5.792 | 0 |
| Pointwise | 2.56 | 0.87 | 0.99 | 5.831 | 0.71 |
| Simulated (3.7σ) | 7.40 | 1.00 | 1.00 | 5.902 | 1.91 |
| Closed-form | 16.72 | 1.00 | 1.00 | 6.039 | 4.14 |

What it says:

- Against the clean-label model the theory behaves as expected. Nominal is
  feasible in 30% of redraws. The pointwise radius gets 0.87 against a target
  of 0.90, the direction selection predicts, but within one standard error
  (0.03). The simulated radius had no violations, so it is conservative here.
- Against the ODE there is no failure to fix: the cubic's bias at the optimum
  is on the safe side (about 1.8 flow units on the first draw) and larger than
  the noise effect. This is why cubic ridge is not the lead model.
- The simulated radius costs 1.9% over nominal; the closed-form radius 4.3%.

From the old harness (embedded network, one fixed balanced cost, 10 folds, ODE
judge): nominal feasible in 0 of 10; C-MICL at alpha = 0.1 in 3 of 10; a tuned
margin and a tuned wrapper reach 0.9 only with ground truth to tune against.
So the failure exists for the network on one cost direction.

On Ovalle et al.'s cost draw there is already one measurement, from the C-MICL
width-model probe (`notes/progress.md`, 2026-09-23; 100 cost vectors, one
dataset, the repo's embedded network, ODE judge): nominal is feasible in 0.00
of runs and C-MICL at alpha = 0.1 in 0.81 (0.83 on the randomized balanced
draw). Two consequences. The premise very likely holds on their costs; what E1
still has to establish is that it holds with the (32, 32) network and the noise
redrawn. And C-MICL's as-prescribed number to beat is about 0.8, not 0.3, so
the gap to a 0.9 target is two to three standard errors at 100 runs.

## Decisions

- **Practical results lead.** Experiments are about 40% of the paper. The
  theory is one section; proofs go to the appendix.
- **The paper stands alone.** C-MICL is one of three baselines, not the frame.
- **The reactor's lead model is open** (network or cubic ridge). The network's
  last-layer RC reaches only 0.42 to 0.52 against the ODE, cubic ridge 0.95;
  see "Results of job 29141177" for the proposal to switch.
- **Models are fixed before the pipeline runs**, as Ovalle et al. do: class,
  architecture and hyperparameters per problem.
- **Cost vectors follow Ovalle et al.** (100 draws, their distribution), with
  the noise redrawn for each. Their feasibility rate is over cost vectors on
  one dataset and our theorem's probability is over the noise; redrawing both
  covers both.
- **One model per comparison**: every baseline embeds the same model as the
  method it is compared with.
- **The guarantee is against the clean-label model.** Going further needs an
  assumption on the true function. The paper says this in a remark, measures
  the gap against the truth, and puts the Lipschitz margin in an appendix if
  it gets proved.
- **The RC's headline radius is the simulated one.** Its finite-sample
  statement is an order statistic of the simulated draws (T1). FO-CP's radius
  is open (top of this file).
- **Semi-synthetic first** (DB, 2026-11-16): reactor, WFP with its
  closed-form truth, semi-synthetic chemotherapy; real chemotherapy last.
- **Classification after regression.**
- **No formal hypothesis tests.** Means with standard errors.
- Everything decided from 2026-11-16 on is at the top of this file.

## Open design points from the 2026-11-14 review

1. **Where the hidden layers come from.** E1 as written trains them on a
   second noise draw of the same inputs, which a practitioner does not have.
   That makes "uses every training row" and the rows-used column against
   C-MICL unfair to C-MICL. R4 tests the one-dataset alternative (sample
   split). Until R4 is read, the abstract's "uses every training row" and the
   80%-against-all row in the comparison table are provisional.
2. **Is 100 redraws enough?** C-MICL as prescribed was at 0.81 on their costs
   (2026-09-23 probe). At 100 runs the standard error near 0.85 is 0.036, so
   a 0.9 against 0.81 difference is about two standard errors. Decide after R2
   to R4: if the RC lands within 0.05 of C-MICL, raise E1 to 300 redraws for
   the as-prescribed rows (Table 1) and keep 100 for the sweep (Figure 2).
3. **C-MICL as published.** Running C-MICL on our shared network (frozen
   hidden layers, ridge last layer, sklearn) is not their setting. Add one row
   with C-MICL fitted end to end as they fit it (their network trained in
   full, their width model at batch 32), so the baseline cannot be called
   handicapped. The shared-model rows stay the comparison of margins.
4. **Pointwise radius.** The Gaussian value `σ z(1−ε)` (1.28σ at ε = 0.1) is
   the one used everywhere. `paper_a_guarantee.md`'s table had the
   sub-Gaussian `√(2 ln(1/ε))`; it is now marked as such.
5. **Novelty against simultaneous bands and post-selection inference:
   checked 2026-11-14** (`paper_a_literature.md`; outline Section 1.2). Two of
   the three claimed novelties were not new. Bands restricted to a region with
   a constant simulated by optimizing over it are Liu et al. (2005) for linear
   models over a box. Validity at the selected point is the PoSI / inference-
   on-winners projection argument. The uniform "mean + β·sd" constraint is
   safe Bayesian optimization; Fiedler et al. (2021) has our noise term for
   kernel ridge. What survives: the clean-label target (no function-norm
   bound in the radius), the mixed-integer region with each supremum a global
   MIP, one form for ridge, last-layer networks and forests, and the band
   embedded as a constraint. The contribution is now argued from those. Two
   things follow for the method. The simulated radius must use the solver's
   bound on each `W`, as I4 already plans. And a referee will ask why not
   Fiedler et al.'s radius. Its noise part bounds the whole noise vector
   (about σ√n), which is already looser than our closed-form σ√χ²(d) with
   d ≪ n, so one sentence answers it.

## Honest fitting

Not required by the method. It is a condition of the theorem for models whose
structure is learned from the labels.

| Model | Structure learned from labels? | Theorem holds with standard fitting? |
|---|---|---|
| Ridge (features and penalty fixed in advance) | no | yes, exactly |
| Last-layer network | hidden layers | only with the hidden layers fitted on separate rows |
| Tree, forest | splits | only with honest fitting |

What goes wrong without it: the theorem needs `s(x)` to be independent of the
noise. Structure chosen on the noisy labels already chases the noise, so the
margin understates the error.

How the paper handles it:

- **Reactor (E1):** R2 (hidden layers frozen on an independent draw) and R4
  (sample split) both met the theorem against the clean-label model but
  reached only 0.52 and 0.42 against the ODE (job 29141177). If the lead
  model stays a network, the RC uses R2 or R4; a network trained end to end
  goes to FO-CP, which needs no honest fitting. Tied to the open lead-model
  decision.
- **Chemotherapy (E2):** the RC is applied to the fitted models with their
  structure frozen, with standard fitting. The paper says the guarantee is then
  empirical. A1 measures on the reactor what standard fitting costs in
  violations and what honest fitting costs in accuracy.

## Chemotherapy on real data

The protocol is Maragno et al.'s, which the repo already implements:

- Fit on the training arms, prescribe for the 96 held-out arms.
- Judge with the ensemble fitted on all 416 arms.
- Repeat over subsamples of the training rows; report the mean and the worst.

What changes for the RC:

- **Models.** Fixed before any robust run: a forest for the four outcomes
  where the existing selection picked XGBoost, ridge for the two where it
  picked elastic net. Every baseline uses the same models. The accuracy given
  up against the existing picks is reported, and the paper states that the
  choice used the labels. This departs from step 0 of the recipe (fix the
  model before looking at the labels); the outline says so in Section 6.3.
- **σ per arm, binomial, for the main result** (decided 2026-11-14):
  `σᵢ = √(pᵢ(1−pᵢ)/n_armᵢ)`, with the weighted norm. It is the label noise the
  theorem is about: the sampling error of an arm's toxicity rate. It uses the
  noisy `p`, so T2 has to cover the plug-in. The out-of-fold residual sd is a
  second row only. It mixes misfit into σ, which the theorem does not ask for,
  and it is likely to make the RC infeasible the way C-MICL is.
- **Objective.** Survival is a learned objective. The default is no margin on
  it, as for the other methods. A robust objective (epigraph form) is a
  deferred variant.
- **What it can show.** The certified level of each method's prescriptions
  (the main measure), the fraction prescribed, toxicity limits met under the
  judge, and survival. The judge is itself fitted, so this experiment does not
  test the theorem. E1 does.

## Infeasibility on chemotherapy (decided 2026-11-14)

The instance has little headroom: nominal already scores 0.89 to 0.96, and
C-MICL at alpha = 0.1 prescribes for about 6% of cohorts. Five simultaneous
margins may leave the RC infeasible for many cohorts too. This section is how
the paper handles that. Nothing in it has been run.

**CP's rollback is not carried over.** With several constraints, the
random-bank cutting planes kept the master feasible by rolling back any cut
that left a protected cohort infeasible (`presentations/method.tex`, "Cutting
Planes: coverage"). That gives up robustness wherever robustness is
unaffordable, so what was delivered has no stated level. It stays in the
thesis.

**The RC may be much less infeasible than C-MICL.**
- C-MICL's width is a prediction interval for a new outcome. It covers noise
  plus misfit and does not shrink with n.
- The RC's margin `R σ‖s(x)‖` is a confidence band on the fitted model.
  `‖s(x)‖` is about √(leverage) for ridge and 1/√(n_leaf) for a tree, so it
  shrinks with n.
- Which of the two the chemo RC resembles is decided by the choice of σ (see
  "σ per arm" above). With the binomial σ the margin should be well below
  C-MICL's. This is a guess, not a measurement; probe P1 below measures it.
- The tension, which the reactor result (job 29141177) makes concrete: a
  label-noise margin does not see misfit. Under the binomial σ the RC may
  prescribe often and still miss against the judge where the models misfit.
  So Table 2 reports both the certified level and the judge's verdict.

**Three layers, in order.**

1. **Shrink the radius, keeping the guarantee.**
   - The joint simulated radius instead of Bonferroni (T1).
   - Take W's supremum over the decision slices actually prescribed for, not
     the whole box. A cohort's context is fixed, so its decision space is the
     treatment slice. Over the union of the 96 test cohorts' slices, the
     radius holds for all cohorts at once and costs about one MIP per draw
     (a selection variable picks the cohort). Per cohort it is smaller still,
     but costs 96 times the solves. Use the union for the main result.
   - The per-arm weighted norm (above).
   - An unequal split of ε across the five constraints if one dominates.
     This is still a guarantee (weighted Bonferroni, or the joint radius with
     per-constraint scaling).
2. **The certified fallback.** For any decision `x` define the certified
   radius `r̂(x) = minⱼ (bⱼ − f̂ⱼ(x)) / ‖Σⱼ^½ sⱼ(x)‖`, in each constraint's
   standardized units. `Σⱼ` is the diagonal of the per-arm noise variances
   for outcome j; with a single σⱼ the denominator is `σⱼ‖sⱼ(x)‖`. Theorem 1's proof gives violation ⇒ `W > r̂(x*)`, so
   `P(violation and r̂(x*) ≥ r) ≤ P(W > r)` for every r, however `x*` was
   chosen, including by maximizing `r̂` (T6). When no regimen meets the target
   radius for a cohort:
   - solve `max t` s.t. `f̂ⱼ(x) + t·‖Σⱼ^½ sⱼ(x)‖ ≤ bⱼ` for all j, by
     bisection on t over feasibility problems (each is the RC at radius t);
   - prescribe the result and report `ε̂ = P(W > r̂)`, read off the simulated
     draws of W that the radius already needs.

   Robustness degrades gracefully, as CP's rollback did, but every
   prescription carries its own level. The statement is joint, not
   conditional on reaching `r̂`, and the paper must word it that way.
3. **Abstention as an outcome.** A cohort with no certifiable regimen at any
   useful level is reported as such: the data cannot support a safe regimen
   there. C-MICL's 94% non-prescription is read the same way.

**Table 2 changes.** `r̂(x)` can be computed for any method's decision, so
every method is put on one scale. Columns:
- the fraction of cohorts certified at ε ∈ {0.05, 0.1, 0.2};
- the fraction prescribed at all;
- toxicity limits met under the judge;
- predicted survival;
- the worst subsample beside each mean.

This also sidesteps the judge's weak separation: the certified level does not
depend on the judge. The judge's verdict stays beside it as the check against
misfit.

**Probe P1 (local, minutes; not yet run; step 1 of the Sequence).**
- Fit ridge on the training arms for the five toxicities.
- For each of the 96 test cohorts, solve the max-t problem.
- Report the fraction whose `r̂` reaches the joint radius at ε = 0.1, under
  the binomial σ and under the out-of-fold residual sd. For the radius, use
  enumeration, which is good enough for a go/no-go.

This decides H3 before I6 and I7 are built. If even the binomial σ certifies
few cohorts, E2 is reframed as a demonstration of the fallback and of
abstention, and H3 is weakened to the comparison of certified levels.

## Theory tasks

| # | Task | Status | Needed for |
|---|---|---|---|
| T1 | Simulated radius: the ⌈(N+1)(1−ε)⌉-th smallest of N simulated `W` gives level ε by exchangeability, with each `W` replaced by the solver's upper bound; joint version for several constraints | drafted 2026-11-15 (Prop. simulated radius; the joint version is held in paper_a_guarantee_held.tex until P1); not checked by JS | E1, E2 |
| T2 | Estimated σ: upper confidence bound and its effect on ε; unequal known `σᵢ`; the binomial plug-in `σᵢ` from noisy `pᵢ`. Alternative route: simulate `σ̂/σ` jointly with the noise, as Liu et al. (2005) do for their critical constant | to prove | the recipe in Section 2; E2 |
| T3 | Directed cuts terminate in at most K iterations | drafted 2026-11-15 (paper Sec. 4, proof in appendix); not checked by JS | Section 4; E2 |
| T4 | Sub-Gaussian version of the subspace radius | to prove | appendix |
| T5 | Lipschitz bias margin | to prove | appendix, optional |
| T6 | Certified level of any decision: `P(violation and r̂(x*) ≥ r) ≤ P(W > r)` for all r, `x*` chosen in any way; the joint version over several constraints | drafted 2026-11-15, held until P1 in paper_a_guarantee_held.tex; not checked by JS | Section 3; E2 fallback and Table 2 |
| T7 | FO-CP: what stopping certifies (exact separation, τ > 0, continuous decisions converge in the limit); the protected-set guarantee for both radius events | to prove | FO-CP, P2 |
| T8 | FO-CP's radius (c): exact for affine models (reduces to T1), first order for networks; (a) as the reference | to prove | F2 |

Theorem 1 and the finite, subspace and Bonferroni corollaries are proved.
Dropped from this paper: classification, the ridge bias bound, counting K for
forests.

## Implementation tasks

| # | Task | Status | Needed for |
|---|---|---|---|
| I1 | RC for ridge: second-order cone on `‖s(x)‖`, polynomial features | done as a probe (`probe_rc_ridge.py`) | A3 |
| I2 | RC for a last-layer network: train and freeze the hidden layers, refit the last layer by ridge, ReLU embedding plus one cone | to build | E1 |
| I3 | Redraw harness: draw a cost vector and noise, refit, solve, judge against the clean-label model and the ODE | probe version exists for fixed cost | E1 |
| I4 | Simulated radius, single and joint, each maximization solved to a bound; start the iterations from the enumeration value to cut solves | probe version exists; slow | E1, E2 |
| I5 | Baselines on the shared network with frozen hidden layers | to check against the existing methods | E1 |
| I6 | Forest with leaf vectors; RC by directed cuts; honest variant | to build | E2, A1 |
| I7 | RC in the chemotherapy runner; per-outcome σ; fixed models | to build | E2 |
| I8 | WFP instance, regression, with OptiCL App. C.1's truth function (constants recovered from the public data or from the authors) | to build | E3 |
| I9 | Certified radius `r̂(x)` for any decision; the max-t fallback by bisection; `ε̂` from the stored W draws | to build | E2, Table 2 |
| I10 | Probe P1: ridge on the gastric training arms, max-t per test cohort, two choices of σ | to build; local, minutes; **still not run** | H3 go/no-go |
| I11–I14 | FO-CP: training map, separation, slim master, SS1 truth / WFP / SS2 | to build (top of this file) | F0–F2, E3, P2 |

Everything beyond a probe goes through `sbatch`. E1 is many short solves and
fits `mit_quicktest` or `mit_normal` arrays split by method and cost vector.
The simulated radius at 200 draws is a cluster job.

## Data and what it supports

| Problem | Have | Enough for |
|---|---|---|
| Reactor | ODE oracle; generator for any n and σ; 1,000 cached designs; all baselines implemented; Ovalle et al.'s cost draw reproduced in an earlier probe | E1 once I2 to I5 exist |
| Chemotherapy | 416 real trial arms, six outcomes, arm sizes, judge ensemble, subsampling protocol, all baselines implemented | E2 once I6, I7 exist. No ground truth: supports practical claims, not the theorem |
| WFP | nothing in the repo; OptiCL's public data (5,000 baskets) | E3 once I8 recovers the truth function |
| Chemotherapy, semi-synthetic (SS2) | the real arms' covariates and arm sizes; a fitted model per outcome as truth | P2 once I14 builds the generator |

## Sequence

Revised 2026-11-18. Step 1 of the old sequence and a small version of
step 2 were job 29141177.

1. **Cheap checks now.** F−1 (existing code, cluster) and P1 (local,
   minutes). F−1 decides whether radius (a) survives; P1 decides H3.
2. **The reactor's lead model**, with DB, from R1 to R4 and F−1.
3. **FO-CP gate.** I11, I12, F0 (local), then F1 (go or no-go: does ascent
   add anything over the linearized step?). Then I13 and the F2 pilot
   (10 cost vectors, SS1(ii)); the full F2 after it. T7, T8 alongside.
4. **RC on the reactor** (can run in parallel on the cluster): I2 to I5, then
   E1. T1 alongside. Draft Sections 2, 3, 6.1, 6.2.
5. **WFP.** I8, then E3 with both methods.
6. **Several constraints, known truth.** I14 (SS2 generator), then P2 on the
   RC track. T2, T3, T6.
7. **Real chemotherapy.** I6, I7, I9, then E2. Draft Sections 4 and 6.3.
8. **Appendix tables A1 to A3.**

## Risks

- **The nominal decision does not fail on the paper's design.** It did not on
  cubic ridge. Step 1 tests this for the network.
- **The gap between the clean-label model and the truth is large.** Then
  feasibility against the ODE is decided by bias, in either direction, and not
  by the margin. On cubic ridge the gap (about 1.8) was as large as the
  simulated margin (1.9).
- **The radius overshoots.** The simulated radius gave no violations in 100
  redraws at ε = 0.1. Then the result is the margin's shape at matched
  feasibility (H2), and the guarantee is described as conservative.
- **The last-layer RC covers only noise that enters through the last layer.**
  If most of the network's error comes from the hidden layers, the margin is
  too small against the ODE even though the theorem holds.
- **The RC is not the cheapest at matched feasibility.** `‖s(x)‖` sees label
  noise but not model misfit, which C-MICL's width model can pick up.
- **The simulated radius is expensive to certify.** 43 seconds per draw on
  cubic ridge.
- **Five margins make the RC infeasible on chemotherapy.** Mitigated by the
  binomial σ, the radius over the prescribed slices and the certified
  fallback ("Infeasibility on chemotherapy"); P1 measures how much is left.
- **Under the binomial σ the RC prescribes but misses against the judge**
  where the models misfit, as the network did against the ODE. Report the
  judge's verdict beside the certified level.
- **The chemotherapy judge cannot separate methods.** The audit put the 0.9
  target inside the judge's uncertainty band for most cells. Report orderings
  and the fraction prescribed there, and keep feasibility levels to the reactor.
- **WFP's truth function has unpublished constants.** If they cannot be
  recovered from the public data, ask OptiCL's authors.
- **FO-CP's own risks** are listed in its section at the top.
