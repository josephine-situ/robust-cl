# Paper A plan

Revised 2026-11-14 (late, after the first solver test and a narrowing of
scope). Outline: `paper_outline_v3.md`. Proofs: `paper_a_guarantee.md` (its
Sections 5 and 6 are superseded by this file). Nothing below has been run on
the paper's lead model. The only results are under "Results so far".

## Tonight (2026-11-14), before the meeting with DB

The question for tonight: is the paper's direction worthwhile? That needs two
answers. Does the nominal decision fail on the model the literature uses, and
does the RC fix it?

**Runs.** On the cluster: `experiments/submit_rc_probe.sh`, job 29138280, a
21-task array on `mit_normal` (3h wall, 8 cores each), running
`experiments/probe_rc_ridge.py`. Tasks 0 to 9 are R2, 10 to 19 are R3 (ten
redraws each), task 20 is R1. Each task saves after every redraw and resumes
from its own file. All 21 tasks were still pending when checked later on
2026-11-14, with no output yet. SLURM emails when the array ends. To read the results, on
the cluster: `python experiments/probe_rc_ridge.py --stage collect --costs paper`.

Reactor, ε = 0.1, Ovalle et al.'s cost vectors (their exact random stream), one
cost vector and one fresh noise draw per redraw. Each redraw is solved at four
radii (nominal, pointwise, simulated, closed-form) and judged against the
clean-label model and the ODE.

| Run | Model | Redraws | What it answers | Log |
|---|---|---|---|---|
| R1 | Cubic ridge | 100 | Was the 0.96 nominal feasibility under the balanced cost a property of the model or of that one cost direction? | `rc_probe_29138280_20.out` |
| R2 | Network (32, 32), hidden layers frozen on an independent noise draw, last layer refit by ridge | 100 | The case where the theorem holds exactly for a network. Does the last-layer margin deliver against the ODE? | `rc_probe_29138280_{0..9}.out` |
| R3 | Same network, hidden layers retrained on each redraw's own labels (how it would be fitted in practice) | 100 | How far is standard fitting from the theorem: violation rate against the clean-label model with the same hidden layers, and against the ODE | `rc_probe_29138280_{10..19}.out` |

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
| R2 delivers, R3 does not | The method needs honest fitting of the hidden layers in practice, not only for the theorem. Still a paper, with that as a stated requirement. |
| Neither delivers against the ODE, both deliver against the clean-label model | The last-layer margin is too small against the network's own error. The network cannot be the lead model. Fall back to models where `s(x)` captures all of the fit (ridge on fixed features, forests), and the paper has to say why. |
| R1: nominal fails under their costs and the RC fixes it | Cubic ridge can serve as a lead or second model after all. |

First single-draw look at the network (their first cost vector): with frozen
hidden layers the nominal decision has ODE flow 49.4 and the simulated radius
gives 51.2; with standard fitting nominal has 44.4 and the simulated radius
46.6, with a margin of only 1.1. So the concern about the last-layer RC is
real, and R3 is the run that settles it.

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
  nominal decision fails.
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

- **Reactor (E1):** the hidden layers are trained once on an independent noise
  draw and frozen; each redraw refits only the last layer. The condition holds
  exactly, and every row of the redraw is used.
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
  choice used the labels.
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
