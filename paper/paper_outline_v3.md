# Constraint Learning under Label Noise: A Robust Counterpart that Stays Feasible at the Optimum

Target: *Operations Research*. Josephine Situ and Dimitris Bertsimas.
Version 3 (2026-11-14, last revised late that night). Supersedes
`paper_outline_v2.md`.
Experiments and sequencing: `paper_plan.md`. Proofs: `paper_a_guarantee.md`.

**What changed from v2.**

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
> training row, and is solved as a mixed-integer second-order cone problem or by
> finitely many cuts. We prove that, at any feasible decision, however it was
> chosen, the probability of violating the constraint of the model fitted on
> noise-free labels is at most ε, and give a radius computed by simulation from
> the fitted model and the noise level. On a membrane reactor design problem
> with a mechanistic ground truth, the nominal decision is feasible in [N]% of
> runs and the robust counterpart in [X]% at a target of 90%, at [Z]% lower
> cost than a constant margin, the bootstrap wrapper and conformal constraint
> learning at the same feasibility. On chemotherapy regimen design with real
> trial data and six learned constraints, it prescribes for [A]% of cohorts and
> [B]% of its prescriptions satisfy every toxicity limit.

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
   computed by simulation that is much smaller than the closed-form bounds.
4. Experiments on a design problem with a mechanistic ground truth and on two
   case studies from the constraint-learning literature, showing that the
   method reaches its target feasibility without tuning and at lower cost than
   a constant margin, the wrapper and conformal constraint learning.
   **[to run]**

### 1.2 Literature (one page)
- **Constraint learning.** Fajemisin et al. (survey); OptiCL (Maragno et al.):
  trust region and the bootstrap wrapper.
- **Conformal.** C-MICL (Ovalle et al.). Its guarantee at the optimum rests on
  exchangeability plus a conditional-independence assumption the authors say
  may fail.
- **Robust optimization.** Probabilistic guarantees for robust counterparts
  (Bertsimas and den Hertog, Ch. 15); Goldfarb and Iyengar for linear models;
  cutting-set methods (Mutapcic and Boyd; Bertsimas, Dunning and Lubin).
- **Other robust constraint learning.** Alcantara and Ruiz; Xavier et al.;
  Cremer et al.; Ostekin et al. (read in full before stating the difference).
- **Statistical ingredients.** Honest forests (Wager and Athey); forest weights
  (Meinshausen).
- **Simultaneous bands and selection.** The closed-form radius for a linear
  model is Scheffé's simultaneous band; the simulated radius is a sup-t band,
  computed the way the post-selection-inference constant is (Berk et al.). The
  optimizer's curse (Smith and Winkler) names the selection effect. What is new
  here: the band is restricted to the decision domain, embedded as a constraint
  in a mixed-integer problem, and extended to tree ensembles. (Read these in
  full before stating the difference.)

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
     hyperparameters. Ovalle et al. fix their networks the same way.
  1. Estimate σ from out-of-fold residuals (an upper confidence bound).
  2. Simulate the radius for the target ε (Section 3).
  3. Solve. No ground truth and no held-out calibration rows are used.
  - Alternative for a practitioner with a validation oracle: choose `R` on it.

## 3. Guarantee at the optimum

Kept to about four pages. Proofs in the appendix.

- **Theorem 1.** At any RC-feasible `x*`, however chosen, the violation
  probability is at most `P(W > R)`. **[proved]**
- **Radii.** `W` is the largest standardized error of the model over the whole
  domain, so any `R` that `W` exceeds with probability at most ε works. Two
  ways to get one, in one table. (The pointwise value `σ z(1−ε)`, valid at a
  fixed `x` only, is not offered as a radius; it appears once in Section 6.2
  as the control that ignores selection.)
  - **Closed-form:** bound `W` without looking at the problem. Finite case
    `σ√(2 ln(K/ε))` (union bound over the K cells of a tree ensemble); subspace
    case `σ√χ²(d, 1−ε)` (a linear model or last layer with d features).
    **[proved]** These protect against every direction the noise could take in
    the model's parameter space, whether or not any feasible `x` can reach it.
  - **Simulated:** the `1−ε` quantile of `W` itself, by drawing noise vectors
    and maximizing over the actual domain for each. Smaller, because the
    feasible decisions reach only part of the parameter space. This is the
    radius the experiments use. Rule: draw N noise vectors, compute `W` for
    each, take the ⌈(N+1)(1−ε)⌉-th smallest. The true `W` and the N draws are
    exchangeable, so the violation probability is at most ε with no extra
    sampling-error term. **[to prove: short]** Each `W` is a maximization over
    the domain; it is solved with the solver's upper bound, not estimated from
    a sample of points (a sample underestimates `W` and so the radius). Needs
    the noise distribution up to scale (Gaussian in the main text).
- **Several constraints.** Bonferroni **[proved]**; the simulated joint radius
  as the tighter option **[to prove]** with the above. The experiments use the
  joint radius.
- **Estimated σ.** Guarantee with an upper confidence bound. **[to prove]**
- **When the structure is learned from the labels.** For ridge with its
  features and penalty fixed in advance the theorem holds as stated. For trees,
  forests and networks it holds when the structure (splits, hidden layers) is
  chosen on rows whose noise is independent of the rows that set the leaf
  values or last layer (honest fitting). The method itself runs on any fitted
  model with its structure frozen; an appendix table measures what standard
  fitting costs in violations and what honest fitting costs in accuracy.
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
- **Certifying any decision.** One evaluation of the RC at a given `x` checks
  whether the guarantee applies, whichever method produced `x`.
- Solve times are reported in Section 6, not analysed here.

## 5. Baselines

- **Nominal.**
- **Constant margin.** The natural comparison: same model, a margin that does
  not depend on `x`.
- **Wrapper** (OptiCL): P bootstrapped models.
- **C-MICL** (Ovalle et al.): their published settings.
- **One model for everyone.** On each problem every method embeds the same
  model class, so a difference in the results is a difference in the margin.
  On the reactor that is Ovalle et al.'s network, which is also C-MICL's
  published setting.
- **Conservatism parameter.** Each method has one parameter that trades cost for
  safety: the radius `R` (ours), the margin `m`, the wrapper's violation limit
  α, C-MICL's miscoverage α. Two comparisons, both defined here:
  - *As prescribed:* each method at the setting its authors give for a 90%
    target (C-MICL α = 0.1; the RC at its simulated radius for ε = 0.1). The
    margin and the wrapper have no such setting, which is part of the result;
    Table 1 shows them at the setting the oracle picks, labelled as such.
  - *At equal feasibility:* each method's parameter is varied over its range,
    and cost is compared at the same ground-truth feasibility. This uses the
    oracle and is a comparison device, not a tuning procedure we propose.

## 6. Computational experiments

Three problems. Each subsection opens with its question.

### 6.1 Membrane reactor design (the lead result)
*Does the method deliver the feasibility it targets, and what does it cost?*
**[to run]**
- Reactor of Carrasco and Lima, as set up by Ovalle et al.: one learned
  constraint, ODE oracle, σ = 2. Chosen because the truth is integrated, not
  fitted, so the training noise can be redrawn and every decision judged exactly.
- Model: Ovalle et al.'s network, fixed in advance. Hidden layers are trained
  once on an independent noise draw and frozen; the last layer is refit on each
  redraw.
- Design: Ovalle et al.'s 100 cost vectors, with the training noise redrawn for
  each. The radius does not depend on the cost, so one radius serves all 100.
- **Table 1.** Per method at target 0.9: ground-truth feasibility, cost,
  fraction solved, solve time, training rows used. Means with standard errors.
  Feasibility is over solved runs, with the fraction solved beside it. The RC
  row uses the estimated σ; the known σ = 2 is a second row.
- **Figure 2.** Cost against ground-truth feasibility as each method's
  conservatism parameter varies. The claim is that the RC curve lies below the
  others.

### 6.2 Is the guarantee tight enough to use?
*How conservative is the radius?* **[to run]** The same runs as Section 6.1,
repeated at three radii and three values of ε.
- **Figure 3.** Violation rate against ε for the pointwise, simulated and
  closed-form radii, against the clean-label model and against the ODE.
- Cost of each radius. Expect the closed-form radius to overpay; the simulated
  radius is the recommendation.
- The pointwise radius is the control: it is what a fixed-`x` confidence bound
  gives. If it violates more than ε, that is the direct evidence that selection
  by the optimizer matters. If it does not, the simulated radius is
  conservative on this problem and we say so.
- Gap to the truth at the optimum: clean-label model against the ODE at `x*`,
  beside the margin there. This decides whether the bias margin stays a remark.

### 6.3 Chemotherapy regimen design (real data, six learned constraints)
*Does the method still prescribe, and prescribe safely, when several learned
constraints bind and σ has to be estimated?* **[to run]**
- Maragno et al.'s gastric cancer case study on the real trial data: 416 arms,
  five toxicity constraints and overall survival as the objective.
- Their evaluation protocol: prescribe for the 96 held-out arms; judge with an
  ensemble fitted on all arms; repeat over subsamples of the training rows.
- Model per outcome fixed before any robust run: a forest where the existing
  selection picked a tree model, ridge where it picked a linear one. The same
  models for every baseline. This is the paper's test of the tree-ensemble
  formulation.
- **Table 2.** Fraction of cohorts prescribed for (the main measure here,
  since the judge separates methods weakly on feasibility), fraction of
  prescriptions meeting every toxicity limit, and predicted survival, with the
  worst subsample beside the mean.
- Stated limits: the judge is a fitted model, so this section shows practical
  behaviour and does not test the theorem. Section 6.2 does that.

### 6.4 World Food Programme diet
*Does the reactor result carry over to a larger problem?*
- Second case study from the literature (Maragno et al.; Ovalle et al.).
  Palatability simulator as oracle, noise injected at chosen levels.
- **Table 3.** Same columns as Table 1.
  **[to run; instance not built]**

## 7. Discussion

- When to use which safeguard: the RC when the model is affine in the labels
  and the noise scale can be bounded; conformal methods when it is not.
- Limitations: the guarantee is against the clean-label model; learned
  structure needs honest fitting for the theorem; models not affine in the
  labels.
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
