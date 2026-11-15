# Paper A plan

Revised 2026-11-14 (late, after the first solver test and a narrowing of
scope). Outline: `paper_outline_v3.md`. Proofs: `paper_a_guarantee.md` (its
Sections 5 and 6 are superseded by this file). Results: the go/no-go probe
(job 29141177, under "Tonight") and the earlier local checks under "Results
so far". No baseline has been run on the paper's design.

## Tonight (2026-11-14), before the meeting with DB

The question for tonight: is the paper's direction worthwhile? That needs two
answers. Does the nominal decision fail on the model the literature uses, and
does the RC fix it?

**Runs.** On the cluster: `experiments/submit_rc_probe.sh`, job 29141177, a
32-task array on `mit_preemptable` with `--requeue` (8h wall, 8 cores each),
running `experiments/probe_rc_ridge.py`. Tasks 0 to 9 are R2, 10 to 19 R3,
20 to 29 R4 (ten redraws each), task 30 is R1, task 31 the radius check. Each
redraw task saves after every redraw and resumes from its own file, so a
preemption costs one redraw. (The first submission, job 29138280 on
`mit_normal`, was cancelled: the scheduler put its start about 21 hours out.)
SLURM emails when the array ends. To read the results, on the cluster:
`python experiments/probe_rc_ridge.py --stage collect --costs paper`.

Reactor, ε = 0.1, Ovalle et al.'s cost vectors (their exact random stream), one
cost vector and one fresh noise draw per redraw. Each redraw is solved at four
radii (nominal, pointwise, simulated, closed-form) and judged against the
clean-label model and the ODE.

| Run | Model | Redraws | What it answers | Log |
|---|---|---|---|---|
| R1 | Cubic ridge | 100 | Was the 0.96 nominal feasibility under the balanced cost a property of the model or of that one cost direction? | `rc_probe_29141177_30.out` |
| R2 | Network (32, 32), hidden layers frozen on an independent noise draw, last layer refit by ridge | 100 | The case where the theorem holds exactly for a network. Does the last-layer margin deliver against the ODE? Idealized: a practitioner has no second noise draw. | `rc_probe_29141177_{0..9}.out` |
| R3 | Same network, hidden layers retrained on each redraw's own labels (how it would be fitted in practice) | 100 | How far is standard fitting from the theorem: violation rate against the clean-label model with the same hidden layers, and against the ODE | `rc_probe_29141177_{10..19}.out` |
| R4 | Same network, sample-split on each redraw: hidden layers on a random half of the rows, last layer on the other half | 100 | The theorem's condition met with one dataset, which is what a practitioner can do. Costs half the rows in the last layer (R² 0.944 on its half in a local check, against 0.958 for the full network). This is the candidate design for E1. | `rc_probe_29141177_{20..29}.out` |
| — | Radius check: solver-certified `W` against enumeration for the R2 network, 10 noise draws | — | Whether the enumerated radius plus 0.1σ covers the solver's `W` on the network (on the cubic it did not in six of ten draws) | `rc_probe_29141177_31.out`, `radius_net_frozen.json` |

Logs are in `logs/` and JSON outputs in `results/paper_a/prelim/`, both on the
cluster. One network solve takes 15 to 100 seconds, so a ten-redraw task is
expected to take 30 to 60 minutes. With 100 redraws the standard error on a
rate near 0.9 is about 0.03. The simulated radius is by enumeration plus 0.1σ,
not solver-certified. The frozen network is trained once and saved
(`net_frozen_weights.npz`) so every R2 task embeds the same one.

**How to read them.**

| Outcome | Meaning for the paper |
|---|---|
| R2 and R3: nominal fails against the ODE, RC at the simulated radius reaches about 0.9 | The direction holds with the network as lead model. |
| R2 delivers, R3 does not | The method needs honest fitting of the hidden layers in practice, not only for the theorem. Still a paper, with that as a stated requirement, and R4 decides whether the requirement is affordable. |
| R4 delivers at a cost close to R2's | E1 uses the sample split. The "uses every training row" claim goes; the comparison with C-MICL's 80% becomes half the rows against 80%, stated as such. |
| R2 delivers, R4 does not | Half the rows is too few for the last layer. Try a smaller structure split (for example 30/70) before giving up the network. |
| Neither delivers against the ODE, both deliver against the clean-label model | The last-layer margin is too small against the network's own error. The network cannot be the lead model. Fall back to models where `s(x)` captures all of the fit (ridge on fixed features, forests), and the paper has to say why. |
| R1: nominal fails under their costs and the RC fixes it | Cubic ridge can serve as a lead or second model after all. |

First single-draw look at the network (their first cost vector): with frozen
hidden layers the nominal decision has ODE flow 49.4 and the simulated radius
gives 51.2; with standard fitting nominal has 44.4 and the simulated radius
46.6, with a margin of only 1.1. So the concern about the last-layer RC is
real, and R3 is the run that settles it.

Every summary table now reports, beside each violation rate, the mean shortfall
below the floor among the violating runs and the worst one, against both the
clean-label model and the ODE. A rate of 0.85 reads differently when the
misses are 0.2 flow units than when they are 5.

### Results of job 29141177 (2026-11-14)

All 32 tasks completed, every run solved to optimality. Reactor, Ovalle et
al.'s 100 cost vectors with the noise redrawn for each, ε = 0.1, σ = 2 known.
Feasibility over 100 redraws; standard error at most 0.05. "Cost" is the
increase over nominal as a percentage of the objective's range (`|c|·span`).
Outputs: `results/paper_a/prelim/redraw_*_paper_eps0.1_*.json`.

| Model | Nominal, ODE | Pointwise: clean / ODE | Simulated: clean / ODE | Closed-form: clean / ODE | Cost, simulated / closed-form |
|---|---|---|---|---|---|
| R1 cubic ridge | 0.66 | 0.90 / 0.80 | **1.00 / 0.95** | 1.00 / 1.00 | 4.2% / 8.6% |
| R2 network, frozen | 0.18 | 0.91 / 0.22 | 1.00 / 0.52 | 1.00 / 0.79 | 2.2% / 3.3% |
| R4 network, sample-split | 0.25 | 0.92 / 0.28 | 1.00 / 0.42 | 1.00 / 0.61 | 3.1% / 5.2% |
| R3 network, standard fitting | 0.23 | 0.42 / 0.29 | 0.76 / 0.33 | 0.98 / 0.45 | 1.9% / 3.6% |

Why the network fails: bias at the nominal optimum, where the fitted model
equals the floor. (Clean-label model − 50) is the label-noise effect the
margin is built for; (ODE − clean-label model) is the model's bias, which no
label-noise margin sees.

| Model | Noise effect, mean | Bias, mean (sd) | Bias negative in |
|---|---|---|---|
| Cubic ridge | +0.74 | −0.05 (1.30) | 43% of runs |
| Network, frozen | +0.98 | −1.56 (1.67) | 90% |
| Network, sample-split | +0.88 | −2.78 (3.22) | 79% |
| Network, standard | +0.26 | −2.23 (2.72) | 80% |

Radius check (R2 network, 10 draws): the solver's `W` is 0.10 to 0.24σ above
enumeration (mean 0.17σ), so enumeration plus 0.1σ is too small on every
draw. Each certified `W` took 4 to 19 seconds on the network, against 43 on
the cubic.

**What it says.**

1. **The direction holds, with cubic ridge.** Under Ovalle et al.'s costs the
   nominal cubic decision is ODE-feasible in only 0.66 of runs (it was 0.96
   on the balanced cost, which was a property of that one cost direction).
   The RC at the simulated radius reaches 0.95, and the misses are small (mean
   0.14 flow units, worst 0.32), at 4.2% of the objective range. This is the
   reading-table row "R1: nominal fails under their costs and the RC fixes
   it".
2. **The network cannot be the lead model.** Every honest network variant
   meets the theorem against the clean-label model (1.00 at the simulated
   radius) and misses badly against the ODE (0.42 to 0.52). This is the
   reading-table row "neither delivers against the ODE, both deliver against
   the clean-label model". The cause is measured, not inferred: at the
   optimum the network's bias is negative in 80 to 90% of runs, so the
   optimizer is exploiting the network's misfit, which a label-noise margin
   cannot see. The cubic's bias has mean zero and no selection. The sample
   split makes it worse (a hidden layer fitted on 500 rows misfits more).
3. **Standard fitting breaks the guarantee itself**: 0.76 against the
   clean-label model at the simulated radius, target 0.90. This is the
   evidence for the honest-fitting condition, and it can replace A1 (forest)
   with the network as the example.
4. **The selection effect does not show at ε = 0.1.** The pointwise radius
   already meets the target against the clean-label model on every honest fit
   (0.90 to 0.92). So the simulated radius is conservative for its own target
   here, and on the cubic its extra margin buys ODE feasibility (0.80 to
   0.95) by absorbing the bias's spread, not by correcting for selection.
   Figure 3's argument for the simulated radius needs either a setting where
   pointwise falls short (smaller n, higher dimension) or a different
   framing. Cubic at ε = 0.05 and 0.2 (jobs 29142767, 29142768, 100 redraws
   each) confirms it across ε:

   | ε | Target | Pointwise: clean / ODE | Simulated: clean / ODE | Closed-form: clean / ODE | Cost, simulated |
   |---|---|---|---|---|---|
   | 0.05 | 0.95 | 0.95 / 0.81 | 1.00 / 0.97 | 1.00 / 1.00 | 4.5% |
   | 0.1 | 0.90 | 0.90 / 0.80 | 1.00 / 0.95 | 1.00 / 1.00 | 4.2% |
   | 0.2 | 0.80 | 0.77 / 0.76 | 1.00 / 0.93 | 1.00 / 1.00 | 4.0% |

   The pointwise radius sits on its target against the clean-label model at
   every ε (0.77 against 0.80 is within one standard error). The simulated
   radius never violates the clean-label model and moves little with ε (6.9
   to 8.0 in R, 4.0 to 4.5% in cost), so it overshoots most at large ε.
   Against the ODE it holds 0.93 to 0.97, the pointwise radius 0.76 to 0.81.
   So on this instance the simulated radius pays for the bias's spread, not
   for selection.
5. **The certified radius is needed and cheap enough.** Enumeration
   understates `W` by about 0.17σ on the network. At 4 to 43 seconds per
   draw, a certified radius from 200 draws is under three hours on one task,
   and it splits across an array.

**Proposed for DB (not yet adopted):**

- Lead model on the reactor: cubic ridge (the theorem holds with no caveat,
  the nominal fails, the RC delivers). The network moves to a short section
  or appendix as the case where the label-noise margin is not enough because
  the optimizer exploits bias, which is also a finding about C-MICL's
  advantage (its width model can learn misfit).
- E1's baselines on cubic ridge. The C-MICL-as-published row on the network
  stays, as the comparison a reader will ask for.
- The bias margin `B(x)` may need to be more than a remark: the network
  result is direct evidence of the optimizer selecting on bias.
- **Use Ovalle et al.'s own four surrogate families for `h`** (`regression.py`,
  each at one fixed setting): LinearDT (linear model tree, depth 5), random
  forest (15 trees, depth 5), gradient boosting (15 trees, depth 5, learning
  rate 0.2) and the 32-32 MLP. Their `u` is a 32-32 network for every `h`,
  which is not why we chose the network (the RC has no `u`). All three tree
  families are affine in the labels with their splits fixed, so the RC
  applies, with the splits fitted on separate rows. LinearDT is the candidate
  lead: an ordinary regression per leaf (a cone per leaf, selected by the leaf
  binaries), and at the noise floor on this instance (calibration MAE 2.08
  with σ = 2, `cmicl.py` DIFFERENCES #2), so its bias should be small, as the
  cubic's is. A depth-5 CART was well above the floor (MAE 4.48), so the
  forest and boosting may fail as the network did. Next probe: the RC on each
  family, same design as job 29141177. Needs `linear-tree` as a dependency
  and the leaf-vector code (I6).

**Writing.** Start `paper/paper.tex`. Sections that are settled are written as
prose: the problem, the sensitivity vector and the counterpart, Theorem 1 and
the closed-form radii with proofs. Everything else is an outline in the draft,
marked as such: the introduction's numbers, the simulated-radius statement,
the solution methods, and all of Section 6.

**Not tonight.** Baselines, the solver-certified radius, chemotherapy, WFP.

## What the paper has to show

The paper is a practical method with a guarantee behind it. It makes three
points, each with one experiment.

| # | Point | Experiment | Our own go or no-go |
|---|---|---|---|
| H1 | **It delivers.** At its simulated radius for ε = 0.1, the robust counterpart (RC) reaches 0.9 ground-truth feasibility with no tuning, where the nominal decision does not. | E1 | nominal already near 0.9, or the RC under it |
| H2 | **It is cheaper.** At the same ground-truth feasibility, the RC has the best objective among the constant margin, the wrapper and C-MICL. | E1 | another method is cheaper at 0.9 |
| H3 | **It still prescribes.** With six learned constraints on real data, the RC prescribes for most cohorts. | E2 | RC is infeasible for most cohorts |

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
| Cutting planes over a random bank | Drew 200 relabelings from an ellipsoid of radius rho, retrained the model on each, cut the worst one at `x*`, stopped at tolerance tau | No guarantee: the bank is a finite sample of the set, so stopping shows `x*` is feasible for the drawn models only. Two parameters (rho, tau) with no rule tying either to a violation level; both were set against the ground truth, and the rho that works had to be re-measured per instance (rho = 1 never passed 0.30, rho = 2 delivered). The result also depends on the bank's seed. For a model affine in its labels the worst relabeling has a closed form, `R s(x)/‖s(x)‖`, so sampling is unnecessary. | Out. One sentence in Section 4: random-draw cuts carry no guarantee. Stays in the thesis. |
| Right-hand-side margin | Tightened the constraint by a constant `m` | The margin is the same at every `x`, and nothing says how large to make it: `m* = 4.0` was found only with the oracle. At 0.9 feasibility it was the most expensive of the three tuned methods (6.043, against 5.816 for cutting planes and 5.885 for the wrapper). On gastric a large `m` makes the problem infeasible instead of conservative (`m = 0.75` solved 20% of contexts). | Baseline. |
| Wrapper | Embedded 20 bootstrapped models and required a fraction to hold | The fraction is not tied to a violation level. At its strictest setting it reached 0.00 and 0.10 on the two rho columns where cutting planes operated, and it embeds 20 models. On gastric it solved 64 to 68% of contexts. | Baseline. |
| Robust regression | Trained a label-robust model and embedded it nominally | It acts on the learner, not the embedding. | Out. Paper B. |
| The dial sweep as the protocol | Tuned each method's parameter against an oracle and compared the tuned points | The paper's claim is that no tuning is needed, so a protocol built on tuning cannot test it. It also had one cost vector and 10 folds, so feasibility moved in steps of 0.1. | Its design survives as the equal-feasibility comparison (Figure 2), on 100 redraws. |
| Analytic synthetic instance | Known `f_true` | The reactor gives an exact truth on an instance from the literature. | Out. |

What the earlier work contributes to the paper: the failure itself (nominal
ODE-feasible in 0 of 10), and the finding that every tuned safeguard reached
0.9 only with ground truth to tune against. That is the motivation in
Section 1.

Two points DB may press on:

- **Cutting planes won the old comparison.** They were the cheapest method at
  0.9 on the reactor. That result needed an oracle to pick rho and tau. The
  robust counterpart is the same idea made exact, with frozen structure: the
  closed-form worst case replaces the bank and a radius from ε replaces rho.
  It is not literally the limit of the old method, which retrained the whole
  model on each relabeling.
- **Should the random bank be a baseline?** Version 2 of the outline had it as
  one; version 3 drops it. Recommendation: leave it out. It needs full
  retraining per cut and hours per cell, it would have to be re-run on the
  shared network, and it has no prescribed setting. Raise with DB.

## Scope (narrowed 2026-11-14)

Three problems, four displays in the main text.

| # | Experiment | Problem | Design | Output | Paper |
|---|---|---|---|---|---|
| E1 | **Feasibility, cost and the radius** | reactor | Ovalle et al.'s network, RC on the last layer. Their 100 cost vectors, noise redrawn for each. RC at the pointwise, simulated and closed-form radii for ε ∈ {0.05, 0.1, 0.2}, and on a radius grid. Constant margin, wrapper and C-MICL on the same network, at their prescribed settings and over their ranges. Nominal. | Table 1, Figure 2, Figure 3 | 6.1, 6.2 |
| E2 | **Six constraints, real data** | chemotherapy | Maragno et al.'s protocol. RC with the simulated joint radius; margin, wrapper, C-MICL, nominal, all on the same fixed models | Table 2 | 6.3 |
| E3 | **Second case study** | WFP | as E1, one ε | Table 3 | 6.4 |

Appendix, one table each, all on the reactor:

- A1. Standard against honest fitting for a forest.
- A2. Non-Gaussian noise (Student-t, uniform).
- A3. Cubic ridge, the case where the theorem holds with no caveat.

Cut from the paper:

- The separate fixed-cost design with noise redraws (merged into E1).
- Baselines on two model classes (one shared model per problem).
- Cubic ridge as the lead model (now A3).
- Forest runs on the reactor other than A1.
- Sample-size, heteroscedastic-noise and cross-validated-ridge checks.
- Bonferroni as a reported result (it stays in the theory).
- The width-at-the-optimum comparison with C-MICL.
- Classification as a section (one paragraph in the discussion).
- The analytic synthetic instance, the dial sweep and the random-bank cutting
  planes (these stay in the thesis).

Open, to raise with DB: no experiment tests the several-constraints result
against a ground truth (the reactor and WFP have one learned constraint each,
chemotherapy has no truth). Options: accept it and say so, or add a second ODE
output as a constraint on the reactor.

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
- **The lead model on the reactor is Ovalle et al.'s network**, with the RC on
  the last layer. It is the model the reference paper uses and the one whose
  nominal decision fails. **Under review after job 29141177**: the network's
  RC reaches only 0.42 to 0.52 against the ODE, and cubic ridge reaches 0.95.
  See the results under "Tonight" for the proposal to switch.
- **Models are fixed before the pipeline runs**, as Ovalle et al. do: class,
  architecture and hyperparameters per problem.
- **Cost vectors follow Ovalle et al.** (100 draws, their distribution), with
  the noise redrawn for each. Their feasibility rate is over cost vectors on
  one dataset and our theorem's probability is over the noise; redrawing both
  covers both.
- **One model per problem for every method.**
- **The guarantee is against the clean-label model.** Going further needs an
  assumption on the true function. The paper says this in a remark, measures
  the gap against the ODE, and puts the Lipschitz margin in an appendix if it
  gets proved.
- **The simulated radius is the headline radius.** Its finite-sample statement
  is an order statistic of the simulated draws (T1).
- **Chemotherapy uses the real data**, under Maragno et al.'s protocol.
- **WFP is in the paper.** It is the last build and not needed for the first
  draft.
- **Classification is a paragraph in the discussion.**
- **No formal hypothesis tests.** Means with standard errors.

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

- **Reactor (E1):** to be decided by R2 against R4. Either the hidden layers
  are trained once on an independent noise draw and frozen (R2: the condition
  holds exactly and every row is used, but it needs a second dataset), or each
  redraw is sample-split (R4: one dataset, half the rows in the last layer).
  R4 is the version a practitioner can run, so it is the default if it
  delivers.
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
- **σ is estimated**, per outcome. Two estimates are available: out-of-fold
  residuals, and the binomial `p(1−p)/n_arm` (toxicities only, and it uses the
  noisy `p`). Pick one for the main result. Open.
- **Objective.** Survival is a learned objective. The default is no margin on
  it, as for the other methods.
- **What it can show.** Practical behaviour: fraction prescribed (the main
  measure), toxicity limits met, survival. The judge is itself fitted, so this
  experiment does not test the theorem. E1 does.

Known difficulty: the instance has little headroom. Nominal already scores
0.89 to 0.96, and C-MICL at alpha = 0.1 prescribes for about 6% of cohorts.
Five simultaneous margins may leave the RC infeasible for many cohorts. The
simulated joint radius is the mitigation.

## Theory tasks

| # | Task | Status | Needed for |
|---|---|---|---|
| T1 | Simulated radius: the ⌈(N+1)(1−ε)⌉-th smallest of N simulated `W` gives level ε by exchangeability, with each `W` replaced by the solver's upper bound; joint version for several constraints | to prove, short | E1, E2 |
| T2 | Estimated σ: upper confidence bound and its effect on ε; unequal known `σᵢ` | to prove | the recipe in Section 2; E2 |
| T3 | Directed cuts terminate in at most K iterations | to prove, short | Section 4; E2 |
| T4 | Sub-Gaussian version of the subspace radius | to prove | appendix |
| T5 | Lipschitz bias margin | to prove | appendix, optional |

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
| I8 | WFP instance | to build | E3 |

Everything beyond a probe goes through `sbatch`. E1 is many short solves and
fits `mit_quicktest` or `mit_normal` arrays split by method and cost vector.
The simulated radius at 200 draws is a cluster job.

## Data and what it supports

| Problem | Have | Enough for |
|---|---|---|
| Reactor | ODE oracle; generator for any n and σ; 1,000 cached designs; all baselines implemented; Ovalle et al.'s cost draw reproduced in an earlier probe | E1 once I2 to I5 exist |
| Chemotherapy | 416 real trial arms, six outcomes, arm sizes, judge ensemble, subsampling protocol, all baselines implemented | E2 once I6, I7 exist. No ground truth: supports practical claims, not the theorem |
| WFP | nothing in the repo | nothing until I8 |

## Sequence

1. **Check the premise.** Nominal on the network under Ovalle et al.'s cost
   draw with the noise redrawn: how often is it ODE-feasible? Needs no new
   method code. If it is already near 0.9, the reactor needs a harder setting
   (fewer rows or more noise) before anything else is built.
2. **I2 to I4, then the RC half of E1** (the three radii, no baselines). This
   is the go or no-go for H1 and gives Figure 3.
3. **I5, then the rest of E1.** T1 alongside. Draft Sections 2, 3, 6.1, 6.2.
4. **T2, T3. I6, I7, then E2.** Draft Sections 4 and 6.3.
5. **Appendix tables A1 to A3.**
6. **I8 and E3.**

Step 1 and a small version of step 2 are tonight's runs (top of this file).

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
- **Six constraints make the RC infeasible on chemotherapy.**
- **The chemotherapy judge cannot separate methods.** The audit put the 0.9
  target inside the judge's uncertainty band for most cells. Report orderings
  and the fraction prescribed there, and keep feasibility levels to the reactor.
- **WFP is a build from nothing.**
