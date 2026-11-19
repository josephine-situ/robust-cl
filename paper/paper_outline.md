# Constraint Learning under Label Noise

Target: *Operations Research*. Josephine Situ and Dimitris Bertsimas.
Version 4 (2026-11-16, after the meeting with DB). Version 3 (2026-11-14) is
this file's previous commit; it superseded `paper_outline_v2.md`.
Experiments and sequencing: `paper_plan.md`. Proofs: `paper_a_guarantee.md`.

**What changed from v3 (DB, 2026-11-16).** Exploratory: outline and plan
only, `paper.tex` untouched.

- A second method, for any model trained by a differentiable loss (DB's
  "non-parametric" method): cutting planes whose separation problem is
  solved by first-order methods through the training map (FO-CP; Section
  2.2). One paper, two methods. The RC shares FO-CP's constraint form, not
  its radius argument; FO-CP's radius is **[open]** (Section 3).
- Semi-synthetic instances come first (Section 6.0): the reactor, WFP as
  regression with its closed-form truth, and semi-synthetic chemotherapy.
  Real chemotherapy data (6.3) comes last.
- The reactor's lead model (network or cubic ridge) is **[open]**
  (`paper_plan.md`, "Results of job 29141177").
- The abstract and contributions 1–4 still describe the RC alone. Rewrite
  them for two methods.

**What changed from v2 (in v3).**

- The paper is organized around a practical claim, with the theory as its
  justification. Experiments are about 40% of the paper.
- The paper stands on its own: the claim is that the method delivers the
  feasibility it targets, without tuning, at a lower cost than existing
  safeguards. C-MICL, the wrapper and a constant margin are baselines on equal
  footing.
- Theory is one section: one theorem, the radii, several constraints. Proofs go
  to the appendix. The guarantee is against the clean-label model; the gap to
  the truth is measured, and the bias margin is a remark.
- Chemotherapy uses the real data, with Maragno et al.'s evaluation protocol.
- The paper is not tied to the existing dial-sweep experiments. The random-bank
  cutting planes are out.

**Narrowed on 2026-11-14 (late), after a first solver test.**

- Three problems and four displays in the main text. The reactor is one set of
  runs that gives Table 1, Figure 2 and Figure 3.
- The lead model on the reactor is Ovalle et al.'s network with the robust
  counterpart on its last layer, not cubic ridge. With cubic ridge the nominal
  decision was already feasible under the ODE in 96 of 100 redraws, so it does
  not show the failure, and no reference paper embeds that model.
- The reactor uses Ovalle et al.'s 100 cost vectors, with the noise redrawn for
  each. The separate fixed-cost design is dropped.
- Every baseline runs on the same model as the robust counterpart.
- Models are fixed before the pipeline runs, as Ovalle et al. do.
- Forests appear once, on chemotherapy.
- Classification is a paragraph in the discussion, not a section.
- Robustness checks are cut to three appendix tables.
- Results are means with standard errors. No formal hypothesis tests.

Status tags: **[proved]** checked by hand, **[to prove]** believed true and not
written out, **[open]** not known to work, **[to run]** experiment not run.
Nothing in Section 6 has been run on the paper's models. The claims there are
hypotheses, not results.

## Abstract

Draft, with placeholders for the numbers:

> In constraint learning, a constraint with no known form is replaced by a model
> fitted on data and embedded in a mixed-integer optimization problem. When the
> training labels are noisy, the optimizer selects the point where the model
> errs in its favour, so the prescribed decision often violates the true
> constraint. We propose a robust counterpart whose margin `R‖s(x)‖` grows with
> the sensitivity of the prediction at `x` to the training labels. It applies to
> ridge regression, trees and forests, and last-layer networks, uses every
> training row [provisional: false for a network fitted by sample split, see
> R4 in `paper_plan.md`], and is solved as a mixed-integer second-order cone problem or by
> finitely many cuts. We prove that, at any feasible decision, however it was
> chosen, the probability of violating the constraint of the model fitted on
> noise-free labels is at most ε, and size the radius from sampled noise
> scenarios using only the fitted model and the noise level. On a membrane reactor design problem
> with a mechanistic ground truth, the nominal decision is feasible in [N]% of
> runs and the robust counterpart in [X]% at a target of 90%, at [Z]% lower
> cost than a constant margin, the bootstrap wrapper and conformal constraint
> learning at the same feasibility. On chemotherapy regimen design with real
> trial data and five learned toxicity constraints, it certifies a regimen at
> the 90% level for [A]% of cohorts, against [C]% for the best baseline. Every
> other cohort receives the regimen with the highest certified level, or is
> reported as one for which the data support no safe regimen.

## 1. Introduction

- **Open with the failure, in numbers.** On the reactor with the network of
  Ovalle et al., the nominal decision is infeasible under the ODE in most runs.
  **[to run]** on the paper's design; the existing 0 of 10 is for one fixed
  cost in the old harness.
- **Why it happens.** The optimizer searches for the point where the model is
  most optimistic. A bound that holds at a fixed `x` does not hold at `x*`.
- **Why existing safeguards fall short.**
  - A constant margin pays the same everywhere, and nothing says how large to
    make it.
  - The wrapper (Maragno et al.) embeds P bootstrapped models and requires a
    fraction of them to hold; nothing ties that fraction to a violation level.
  - C-MICL (Ovalle et al.) does vary its margin with `x`, through a second
    model trained to predict the size of the first model's error. That width is
    itself learned, so it has errors of its own, and the optimizer can exploit
    a point where the width is underestimated just as it exploits the
    prediction. It also holds out a calibration split and doubles the embedded
    model.
- **What we do.** A margin that varies with the decision and is computed, not
  learned: `R‖s(x)‖` follows exactly from the fitted model and the training
  inputs. It is large where few training labels determine the prediction and
  small where many do, by construction. One model is trained and embedded.
- **What a practitioner gets.** A radius set from the target violation level,
  with no ground truth, no calibration split and no tuning run.

### 1.1 Contributions
1. A robust counterpart for learned constraints under label noise, for every
   model that is affine in its training labels once its structure is fixed.
2. Exact solution methods that keep the problem in a class solvers handle: one
   second-order cone for linear models and last-layer networks, finitely many
   cuts for tree ensembles.
3. A violation guarantee at the optimizer's own choice of `x*`, with a radius
   sized from sampled noise scenarios that is much smaller than the
   closed-form bounds.
4. Experiments on a design problem with a mechanistic ground truth and on two
   case studies from the constraint-learning literature, showing that the
   method reaches its target feasibility without tuning and at lower cost than
   a constant margin, the wrapper and conformal constraint learning. Where the
   target cannot be met (several binding constraints), it prescribes with a
   certified level per decision instead of failing. **[to run]**
5. **[added 2026-11-16, open]** A counterpart for models trained by a
   differentiable loss that are not affine in their labels (networks trained
   end to end): cutting planes whose worst-case relabeling is found by
   gradient ascent through retraining. On affine models its cut is the RC's.

### 1.2 Literature (one page)

Checked 2026-11-14. Per-paper notes, quotes and how much of each was read:
`paper_a_literature.md`. Every reference is in `paper_a.bib` and in Zotero
(Robust CL collection).

- **Constraint learning.** Fajemisin et al. (survey); OptiCL (Maragno et al.):
  trust region (convex hull, no probabilistic content) and the bootstrap
  wrapper (α is a conservatism knob; the paper never ties it to a violation
  probability).
- **Conformal.** C-MICL (Ovalle et al.). Its Theorem 4.1 holds for an
  exchangeable test point and reaches the optimum through Assumption 4.1
  (feasibility and coverage conditionally independent), which the authors say
  "could potentially fail" (Sec. 6). Still an arXiv preprint.
- **Other uncertainty-aware constraint learning.** None models the fitted
  model's error under label noise. Alcántara and Ruiz embed an estimated
  conditional quantile (the outcome's own spread; the quantile's estimation
  error is untreated). Cremer et al. tune per-split margins on embedded trees
  against a simulator: a precedent for an x-dependent margin, found
  empirically, and a clean statement of the selection problem (99.9% accurate
  rule, 0% at optimized points). DCL (Oztekin et al., 2026) localizes the
  surrogate and names feasibility certification as open. **Xavier et al. is
  learning-to-optimize, not constraint learning: dropped.**
- **For FO-CP** (checked 2026-11-18; `paper_a_literature.md`). Its
  separation is bilevel label poisoning (Mei and Zhu 2015; Biggio et al.
  2012; Muñoz-González et al. 2017; Jagielski et al. 2018), its first step
  the influence function (Koh and Liang 2017), and its loop a cutting-set
  method with a local pessimizing oracle (Mutapcic and Boyd 2009; Bertsimas,
  Nohadani and Teo 2010). Closest in purpose: poisoning a predict-then-
  optimize decision (Kinsey et al. 2022) and influence of noisy points on
  decisions (Zhang, Amin and Perakis 2026); both read in full 2026-11-19:
  in both the learned quantity is the objective over a known feasible set,
  and neither robustifies a constraint (Kinsey et al. attack; Zhang et al.
  reweight training points against a clean validation set and bound
  regret). **Not claimed:** any of the parts. **Claimed (not found):** adversarially
  retrained models as cuts protecting a learned constraint. Protecting a
  fraction of the constraints is Gorissen et al.'s (2015) "K out of N";
  imposing sampled retrained models is scenario optimization (Campi,
  Garatti and Ramponi 2018 for non-convex problems).
- **Robust optimization.** Bertsimas and den Hertog; Bertsimas, Gupta and
  Kallus; Hong, Huang and Lam (data-splitting guarantees for the optimized
  decision, the closest OR precedent for a guarantee at the optimum). For a
  linear model the label ball is an ellipsoid on the coefficients, so with the
  closed-form radius the RC *is* Goldfarb and Iyengar's ellipsoidal
  counterpart; Hüllen et al. do RO on surrogate parameters. Cutting sets:
  Mutapcic and Boyd; Bertsimas, Dunning and Lubin.
- **Simultaneous inference.** Bands restricted to a region of the covariates
  are classical (Wynn and Bloomfield 1971; Naiman 1987), and Liu et al. (2005)
  already simulate the critical constant by optimizing over the region on each
  draw (a QP over a box): that is our scenario radius for a linear model.
  Liu (2010) surveys it. Sup-t bands: Montiel Olea and Plagborg-Møller.
  Scheffé. Validity at a data-selected point via a simultaneous band is the
  projection argument of PoSI (Berk et al.) and inference on winners
  (Andrews, Kitagawa and McCloskey). Optimizer's curse: Smith and Winkler.
- **Uniform bounds in safe learning.** Safe BO (Srinivas et al.; Sui et al.;
  Berkenkamp et al.) and safe linear bandits (Amani et al.) impose
  "mean + β·sd ≤ b" with β valid over the whole domain. **Closest precedent
  for the margin: Fiedler, Scherer and Trimpe (2021)**, whose noise term for
  kernel ridge is a radius times the same ‖s(x)‖, uniform over the domain. It
  is not constraint learning: the setting is learning-based robust control,
  where a GP learns unknown dynamics and a robust controller is designed
  against the resulting set of models. No paper in the constraint-learning
  literature has the margin. GP chance
  constraints in global optimization (Schweidtmann et al.; Wiebe et al.) have
  the same form with a Bayesian, pointwise probability. Thebelt et al. solve
  mean + κ·sd over tree kernels as an MISOCP with κ a tuning constant.
- **Statistical ingredients.** Honest forests (Wager and Athey); forest
  weights (Meinshausen).
- **What is new, stated against these.** The target is the clean-label model,
  so no bound on the true function's norm enters the radius (and the claim is
  weaker, which we say). The decisions the protection must cover are the
  mixed-integer feasible set, so each sampled scenario's worst-case decision
  problem is a global MIP solved to a bound. One label-space form covers
  ridge, last-layer networks and tree ensembles. The protection is imposed as
  a robust constraint, so the optimizer's choice inherits it. **Not
  claimed:** restricting a confidence band to a region or simulating its
  width by optimization (Liu et al.), the ellipsoidal counterpart for linear
  models, or validity at a selected point from protection that holds at every
  decision.

## 2. Problem and method

Written so that a reader can implement the method from this section alone.

- **Nominal problem.** Embedded model, domain constraints, trust region.
- **Label noise.** `y = y* + ε`, features certain. One sentence per case study
  on where the noise comes from (measurement noise in the reactor, sampling
  error of a trial arm's toxicity rate, survey noise in palatability).
- **Sensitivity vector.** `f(x; θ(y)) = s(x)ᵀy + c(x)`. Table of `s(x)` and
  `‖s(x)‖` per model class: ridge, last-layer network (hidden layers frozen,
  last layer refit by ridge), tree, forest.
- **Robust counterpart.** `f(x; θ̂) + R‖s(x)‖₂ ≤ b`. **[proved]** as the exact
  counterpart of an ℓ₂ ball on the labels. With known unequal noise levels
  (trial arms of different sizes) the norm is weighted by `σᵢ`.
- **Reading the margin.** For a tree, `‖s(x)‖ = 1/√n_leaf`: the margin is a
  standard error. For ridge and a last-layer network it is the square root of
  the leverage. The margin is large exactly where the optimizer is
  extrapolating.
- **Figure 1.** The reactor's feasible region under the nominal model, a
  constant margin and the robust counterpart, with the true boundary.
- **Choosing R: the recipe.**
  0. Fix the model before looking at the labels: class, architecture and
     hyperparameters. Ovalle et al. fix their networks the same way. For a
     last-layer network, the hidden layers must not depend on the noise
     `s(x)` is meant to bound. The sample split (R4) meets that with one
     dataset but reached only 0.42 against the ODE; a network trained end to
     end goes to FO-CP instead. **[open: tied to the lead model]**
  1. Estimate σ from out-of-fold residuals (an upper confidence bound).
  2. Size the radius for the target ε from sampled noise scenarios (Section 3).
  3. Solve. No ground truth and no held-out calibration rows are used.
  - Alternative for a practitioner with a validation oracle: choose `R` on it.

### 2.2 First-order cutting planes (added 2026-11-16) **[open]**

For a model trained by a differentiable loss that is not affine in its
labels. Specification, open choices and probes: `paper_plan.md`, "FO-CP".

- **Robust constraint.** `ĥₖ(x; ȳₖ + Δyₖ) ≤ bₖ` for all
  `‖Σₖₖ^{-1/2}Δyₖ‖ ≤ R`, where `ĥₖ(·; y)` is the model **retrained** on
  labels `y` and `Δyₖ` has one entry per training row of outcome k: one
  ellipsoid per outcome, as in the RC.
- **Loop.** At `x^t`, for each protected constraint, maximize the retrained
  prediction `ĥₖ(x^t; ȳ + Δy)` over the set by projected gradient ascent,
  a step accepted only if the retrained prediction increases (implicit
  differentiation through training, one linear solve per step; settings in
  the plan). If it
  exceeds `bₖ`, add the retrained model as a cut, enforced at `bₖ − τ`;
  cuts accumulate and are never removed; re-solve; stop when no cut is
  added. With exact separation this certifies robust feasibility at `bₖ`
  (restriction of the right-hand side); the separation is local, so the
  final decision is checked a posteriori by a stronger attack.
- **Which constraints are protected.** Binary `zₖ` chosen by the master,
  `Σ zₖ ≥ ⌈pK⌉`; the alternative is the certified fallback (largest feasible
  R).
- **Special case.** For a model affine in its labels, one step gives
  `R s(x)/‖s(x)‖`, the RC's cut. On a network the first step is the RC on
  the linearized network. The radius does not carry over (Section 3).
- **Training map** for nonconvex models: every retraining starts from one
  fixed `θ₀` (the nominal fit), runs lbfgs to a gradient tolerance, and
  standardizes labels with constants fixed from `ȳ`; the implicit gradient
  uses a damped Hessian and is checked against finite differences.

## 3. Guarantee at the optimum

Kept to about four pages. Proofs in the appendix.

- **Theorem 1.** At any RC-feasible `x*`, however chosen, the violation
  probability is at most `P(W > R)`. **[proved]**
- **Radii.** The RC at radius `R` protects against every noise scenario in an
  effective uncertainty set `𝒰_R` (a scenario is in it if no RC-feasible `x`
  is pushed past its margin); `𝒰_R` contains the ball of radius `R` and is
  larger. `W`, the radius the realized noise requires, is the largest
  normalized error of the model over the whole domain, so any `R` that `W`
  exceeds with probability at most ε works. Two
  ways to get one, in one table. (The pointwise value `σ z(1−ε)`, valid at a
  fixed `x` only, is not offered as a radius; it appears once in Section 6.2
  as the control that ignores selection.)
  - **Closed-form:** bound `W` without looking at the problem. Finite case
    `σ√(2 ln(K/ε))` (union bound over the K cells of a tree ensemble); subspace
    case `σ√χ²(d, 1−ε)` (a linear model or last layer with d features).
    **[proved]** These protect against every direction the noise could take in
    the model's parameter space, whether or not any feasible `x` can reach it.
  - **Scenario radius:** size `𝒰_R` so it contains a `1−ε` fraction of
    sampled noise scenarios, as data-driven RO sizes an uncertainty set
    (Hong, Huang and Lam). Smaller, because the feasible decisions reach only
    part of the parameter space. This is the radius the experiments use.
    Rule: sample N noise scenarios, solve each one's worst-case decision
    problem (the radius it requires), take the ⌈(N+1)(1−ε)⌉-th smallest. The
    realized noise and the N scenarios are exchangeable, so the violation
    probability is at most ε with no extra sampling-error term. **[to prove:
    short]** Each worst-case problem maximizes over the domain with the
    scenario fixed (the reverse of a separation problem); it is solved with
    the solver's upper bound, not estimated from a sample of decisions (a
    sample understates the required radius and so the radius). Needs the
    noise distribution up to scale (Gaussian in the main text).
- **Several constraints.** Risk allocation, ε/m per constraint (union bound)
  **[proved]**; the joint scenario radius as the tighter option **[to
  prove]** with the above. The experiments use the joint radius. When the
  decision has fixed context variables (a cohort's characteristics), each
  scenario's worst-case problem is taken over the treatment slices of the
  contexts prescribed for, not the whole box: valid for all of them at once,
  and smaller.
- **Protecting a fraction of the constraints (added 2026-11-16;** Gorissen
  et al. 2015, "K out of N", plus the nominal rows). However the optimizer
  chooses the protected set, a protected constraint is violated only if the
  noise leaves its set: `Wₖ > Rₖ` for the RC's radius,
  `‖Σₖₖ^{-1/2}eₖ‖ > Rₖ` for FO-CP's model-free radius. **[to prove: short]**
  Unprotected constraints carry only their certified level.
- **Remark: a different level per constraint (2026-11-18; remark only, no
  experiment).** With one set per outcome, constraint k can take its own
  radius Rₖ sized for its own εₖ (a severe toxicity stricter than a mild
  one). Sized one constraint at a time, this guarantees each constraint
  separately; with Σₖ εₖ ≤ ε (risk allocation) it guarantees them jointly.
  An unprotected constraint is the case εₖ = 1.
- **Remark: several models in one constraint (2026-11-18; remark only).**
  None of our instances has one as the literature writes them. Such a
  constraint gets one set over the labels of all its models; there the joint
  set is the tighter choice (margin `R‖(s_A, s_B)‖` rather than
  `R‖s_A‖ + R‖s_B‖`).
- **Certified level of any decision.** **[to write: the argument is Theorem
  1's proof]** For a decision `x`, let
  `r̂(x) = minⱼ (bⱼ − f̂ⱼ(x)) / ‖Σⱼ^½ sⱼ(x)‖`. Violation implies `W > r̂(x*)`,
  so `P(violation and r̂(x*) ≥ r) ≤ P(W > r)` for every r, however `x*` was
  chosen. Two uses:
  - It certifies any method's decision on one scale (Section 6.3).
  - It is the fallback when the RC at the target radius is infeasible:
    prescribe the decision with the largest `r̂` and report its level
    `P(W > r̂)`.

  The statement is joint, not conditional on reaching `r̂`; the wording must
  say so. This is the principled form of the feasibility rollback in the
  earlier cutting planes, which gave up robustness with no stated level.
- **Estimated σ.** Guarantee with an upper confidence bound. **[to prove]**
- **When the structure is learned from the labels.** For ridge with its
  features and penalty fixed in advance the theorem holds as stated. For trees,
  forests and networks it holds when the structure (splits, hidden layers) is
  chosen on rows whose noise is independent of the rows that set the leaf
  values or last layer (honest fitting). The method itself runs on any fitted
  model with its structure frozen; an appendix table measures what standard
  fitting costs in violations and what honest fitting costs in accuracy.
- **Guarantee for FO-CP. [open, added 2026-11-16]** Theorem 1 does not
  carry over: the RC's simulated radius is not the radius of a ball
  containing the noise (3.6σ against `‖e‖ ≈ 32σ`); it works because an
  affine model sees the noise only through `s(x)ᵀe`. Three options: the
  model-free ball `√χ²(n, 1−ε/K)` per outcome (exact for any model if
  separation were exact; impractical: infeasible on cubic ridge, +15.7% of
  the objective range on the frozen network, `paper_plan.md` F−1); the linearized
  subspace (first order); or **the chosen one (2026-11-19)**, a sampled
  version of the simulated radius: for noise draws e′, retrain on `ȳ + e′`
  and take the quantile of `W_c = max over x ∈ X of [ĥ(x; ȳ+e′) − ĥ(x; ȳ)] /
  ‖s̃(x)‖` (s̃ the linearized sensitivity, X the whole domain), N = 200,
  each maximum replaced by Gurobi's upper bound. R is computed once, before
  the loop. Exact for affine models, where it is the
  simulated radius; first order for networks. Separation is local, so stopping certifies nothing for
  a nonlinear model; the final decision gets an a posteriori attack.
- **Remark: the true function.** The guarantee is against the clean-label
  model. Extending it to `h*` requires an assumption on `h*` (for example
  Lipschitz, or linear in the features), which gives a bound `B(x)` on the
  approximation error to add to the margin. Without such an assumption no
  method can bound the error at a point the data do not cover. Section 6
  measures the gap against the ODE directly.

## 4. Solving the robust counterpart

- **Linear models and last-layer networks.** Mixed-integer second-order cone
  problem; the cone has the dimension of the feature vector or last hidden
  layer. The hidden layers use the standard ReLU embedding.
- **Trees and forests.** `‖s(x)‖` through the leaf binaries; cut
  `δ* = R s(x)/‖s(x)‖`; at most `K` iterations. **[to prove: short]**
- **Cuts from random relabelings carry no guarantee.** A finite sample of the
  label set approximates it from the inside, so stopping does not imply the
  decision satisfies the RC. Only the directed cut does. (One sentence; this
  is where the earlier random-bank cutting planes are placed. See "Why the
  earlier approaches are not the paper's method" in `paper_plan.md`.)
- **FO-CP's separation (added 2026-11-16).** Gradient ascent on `Δy`, with
  the gradient of the retrained prediction by implicit differentiation, run
  from several starts. Each candidate is checked by full retraining before it
  becomes a cut. Each cut is a retrained model, so the loop discretizes the
  uncertainty set (Blankenship and Falk 1976), not Kelley's linearizations.
  Cuts accumulate; the master enforces them at `b − τ` and the separation
  tests against `b` (restriction of the right-hand side, Mitsos 2011), so
  with exact separation it stops after finitely many cuts at a decision
  feasible at `b`; an iteration cap bounds the run. The tree case's bound
  of K iterations does not apply. The random bank comes
  back here only as an ablation: the same master with sampled rather than
  optimized relabelings.
- **Certifying any decision.** One evaluation of `r̂(x)` (Section 3) gives the
  level at which a decision is certified, whichever method produced it.
- **When the RC is infeasible (maximum-radius fallback).** `max t` subject to
  the RC at radius t, by
  bisection over feasibility problems: the decision with the highest
  certified level. If no t of any use is feasible, report abstention.
- Solve times are reported in Section 6, not analysed here.

## 5. Baselines

- **Nominal.**
- **Constant margin.** The natural comparison: same model, a margin that does
  not depend on `x`.
- **Wrapper** (OptiCL): P bootstrapped models.
- **C-MICL** (Ovalle et al.): their published settings.
- **One model per comparison.** Every baseline embeds the same model as the
  method it is compared with, so a difference in the results is a difference
  in the margin: the RC's model (the lead model, open) for the RC, the
  network trained end to end for FO-CP. C-MICL also gets one row fitted end
  to end as published, so that the baseline is not handicapped.
- **Conservatism parameter.** Each method has one parameter that trades cost for
  safety: the radius `R` (ours), the margin `m`, the wrapper's violation limit
  α, C-MICL's miscoverage α. Two comparisons, both defined here:
  - *As prescribed:* each method at the setting its authors give for a 90%
    target (C-MICL α = 0.1; the RC at its scenario radius for ε = 0.1). The
    margin and the wrapper have no such setting, which is part of the result;
    Table 1 shows them at the setting the oracle picks, labelled as such.
  - *At equal feasibility:* each method's parameter is varied over its range,
    and cost is compared at the same ground-truth feasibility. This uses the
    oracle and is a comparison device, not a tuning procedure we propose.

## 6. Computational experiments

Three problems. Each subsection opens with its question.

### 6.0 Semi-synthetic instances first (added 2026-11-16) **[to build]**
DB is happy to start on semi-synthetic data: real covariates, a known truth,
injected noise.
- **SS1, reactor.** As 6.1, plus a second arm whose labels come from a
  realizable truth: a (32, 32) network fitted to noise-free ODE labels.
  With the truth (nearly) inside the model class, bias, which sank the
  network RC against the ODE, is removed, so the label-noise protection is
  tested by itself. Each arm is judged against its own truth.
- **WFP (6.4), regression.** The most reused instance in constraint
  learning; OptiCL generates its palatability from a closed-form function,
  so it has an exact truth. Both methods.
- **SS2, chemotherapy (RC).** The real arms' covariates and treatments, one
  fixed model per outcome treated as the truth, binomial noise from the
  real arm sizes. Several constraints against a known truth, and the test
  of infeasibility handling.
- Real data (6.3) last. Workload dispatching (48 learned constraints) is
  optional. Classification comes after regression; student enrollment
  (JANOS, OptiCL; a logistic model, so FO-CP's case) is the candidate.
- The objective stays nominal; a robust objective is a deferred variant.

### 6.1 Membrane reactor design (the lead result)
*Does the method deliver the feasibility it targets, and what does it cost?*
**[to run]**
- Reactor of Carrasco and Lima, as set up by Ovalle et al.: one learned
  constraint, ODE oracle, σ = 2. Chosen because the truth is integrated, not
  fitted, so the training noise can be redrawn and every decision judged exactly.
- Model: **[open]** the RC's lead model is the network's last layer or cubic
  ridge (`paper_plan.md`, "Results of job 29141177": the last-layer RC
  reached 0.42 to 0.52 against the ODE, cubic ridge 0.95). FO-CP runs on
  Ovalle et al.'s network trained end to end.
- Design: Ovalle et al.'s 100 cost vectors, with the training noise redrawn for
  each. The radius does not depend on the cost, so one radius serves all 100.
- **Table 1.** Per method at target 0.9: ground-truth feasibility, cost,
  fraction solved, solve time, training rows used, and the mean shortfall of
  the violating runs. Means with standard errors. Feasibility is over solved
  runs, with the fraction solved beside it. The RC row uses the estimated σ;
  the known σ = 2 is a second row. Number of redraws: 100, raised to 300 for
  the as-prescribed rows if the RC and C-MICL land within 0.05 of each other.
- **Figure 2.** Cost against ground-truth feasibility as each method's
  conservatism parameter varies. The claim is that the RC curve lies below the
  others.

### 6.2 Is the guarantee tight enough to use?
*How conservative is the radius?* **[to run]** The same runs as Section 6.1,
repeated at three radii and three values of ε.
- **Figure 3.** Violation rate against ε for the pointwise, scenario and
  closed-form radii, against the clean-label model and against the ODE.
- Cost of each radius. Expect the closed-form radius to overpay; the scenario
  radius is the recommendation.
- The pointwise radius is the control: it is what a margin sized to protect a
  fixed `x` gives. If it violates more than ε, that is the direct evidence that selection
  by the optimizer matters. If it does not, the scenario radius is
  conservative on this problem and we say so.
- Gap to the truth at the optimum: clean-label model against the ODE at `x*`,
  beside the margin there. This decides whether the bias margin stays a remark.

### 6.3 Chemotherapy regimen design (real data, five learned constraints)
*When several learned constraints bind and σ has to be estimated, how many
cohorts can the method certify a regimen for, and what does it do with the
rest?* **[to run]** The design is in `paper_plan.md`, "Infeasibility on
chemotherapy".
- Go/no-go (P1, ridge, enumerated radius; not a paper result): under the
  binomial σ, 50% of the 96 test cohorts are certified at ε = 0.1 (43% at
  0.05, 61% at 0.2), with the median cohort on the radius, so E2 needs the
  solver-certified radius; under the out-of-fold σ, none.
- Maragno et al.'s gastric cancer case study on the real trial data: 416 arms,
  five toxicity constraints and overall survival as the objective.
- Their evaluation protocol: prescribe for the 96 held-out arms; judge with an
  ensemble fitted on all arms; repeat over subsamples of the training rows.
- Model per outcome fixed before any robust run: a forest where the existing
  selection picked a tree model, ridge where it picked a linear one. The same
  models for every baseline. This is the paper's test of the tree-ensemble
  formulation. That choice used the labels (the existing selection is on R²),
  which departs from step 0 of the recipe; say so.
- **σ per arm**, the binomial `√(p(1−p)/n_arm)` with the weighted norm: the
  sampling error of a trial arm's toxicity rate, which is the label noise the
  theorem is about. The out-of-fold residual sd, which mixes in misfit, is a
  second row.
- **Infeasibility.** The joint radius is taken over the 96 test cohorts'
  treatment slices. Where the RC is infeasible, the certified fallback
  prescribes the decision with the highest certified level. A cohort with no
  useful level is reported as abstention. The random-bank cutting planes'
  rollback is not used: it had no stated level.
- **Table 2.** Every method's prescriptions are put on one scale by their
  certified level `r̂(x)`. Columns:
  - the fraction of cohorts certified at ε ∈ {0.05, 0.1, 0.2} (the main
    measure, which does not depend on the judge);
  - the fraction prescribed at all;
  - toxicity limits met under the judge (the check against misfit, which no
    label-noise margin sees);
  - predicted survival;
  - the worst subsample beside each mean.
- Stated limits: the judge is a fitted model, so this section shows practical
  behaviour and does not test the theorem. Section 6.2 does that.

### 6.4 World Food Programme diet
*Does the reactor result carry over to a larger problem?*
- The most reused case study in constraint learning (OptiCL and at least
  five later papers as regression; C-MICL as classification). Regression,
  with OptiCL App. C.1's closed-form palatability function as the truth
  (constants recovered from the public data or from the authors), noise
  injected at chosen levels. Both methods.
- **Table 3.** Same columns as Table 1.
  **[to run; instance not built]**

## 7. Discussion

- When to use which safeguard: the RC when the model is affine in the labels
  and the noise scale can be bounded; conformal methods when it is not.
- Limitations: the guarantee is against the clean-label model; learned
  structure needs honest fitting for the theorem; models not affine in the
  labels (FO-CP runs on them, but with the guarantee only as strong as the
  answer to Section 3's open question).
- Classification (one paragraph): with Bernoulli labels and a linear
  probability model or classification forest, the same counterpart applies and
  the noise is bounded, so `σ = 1/2` is known. Logistic and softmax models are
  not affine in the labels. Left to future work.
- Future work: feature noise, model selection for embedding, recourse.

## Appendix

- A. Proofs. Sub-Gaussian version of the subspace radius.
- B. Formulations per model class.
- C. The bias margin `B(x)` for Lipschitz `h*` (if proved; otherwise omitted).
- D. Experimental detail, and three checks on the reactor, one table each:
  standard against honest fitting for a forest; non-Gaussian noise; ridge as
  the case where the theorem holds with no caveat.
