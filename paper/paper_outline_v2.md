# Constraint Learning under Label Noise: Robust Counterparts with Guarantees at the Optimum

Target: *Operations Research*. Josephine Situ and Dimitris Bertsimas.
Version 2 (2026-11-14). Supersedes `paper_outline.md`. Theory: `paper_a_guarantee.md`.
Experiments and sequencing: `paper_plan.md`.

Status tags: **[proved]** checked by hand, **[to prove]** believed true and not
written out, **[open]** not known to work.

## Abstract

## 1. Introduction
- Constraint learning: a constraint has no known form, so a model fitted on data
  is embedded in a mixed-integer optimization (MIO) problem.
- Training labels are noisy, and the optimizer selects the point where the model
  errs in its favour. A bound that holds at a fixed x does not hold at x*.
- One-paragraph preview: a robust counterpart (RC) whose margin `R‖s(x)‖` scales
  with how sensitive the prediction at x is to the labels, and a violation bound
  that holds at any RC-feasible x*, however it was chosen.

### 1.1 Literature
- **Constraint learning.** Survey (Fajemisin et al.); OptiCL (Maragno et al.):
  trust region, and the wrapper that embeds P bootstrapped models.
- **Conformal.** C-MICL (Ovalle et al.) covers the true function, and claims
  validity at the optimum under exchangeability plus a conditional-independence
  assumption the authors say may fail. Our position: a different assumption set,
  not a strictly stronger result.
- **Chance-constrained CL** (Alcantara and Ruiz); problem-specific robust CL
  (Xavier et al., Cremer et al.).
- **Distributional CL** (Ostekin et al.): closest to our cutting planes. State
  the difference precisely once read in full.
- **Robust optimization.** Probabilistic guarantees for RCs (Bertsimas and
  den Hertog, Ch. 15); Goldfarb and Iyengar for linear models; cutting-set methods
  (Mutapcic and Boyd; Bertsimas, Dunning and Lubin).
- **Uniform confidence bounds.** Self-normalized bounds for ridge
  (Abbasi-Yadkori et al.); honest forests (Wager and Athey); forest weights
  (Meinshausen). These supply the pieces for Sections 3 and 4.

### 1.2 Contributions
1. An RC for learned constraints under label noise, for every model that is
   affine in its training labels once its structure is fixed: ridge, honest
   trees and forests, boosting with frozen splits, last-layer neural networks.
2. A violation bound that holds at a data-dependent x*, with explicit radii, for
   regression and for classification. In classification the noise scale is
   known (1/4), so nothing has to be estimated.
3. An extension from the clean-label model to the true function, by adding a
   bias margin under a stated regularity assumption.
4. Exact solution methods: a mixed-integer second-order cone program for linear
   models and finitely terminating directed cuts for tree ensembles.
5. Computational comparison with the wrapper, C-MICL and an RHS margin on
   problems with a ground-truth oracle, at matched ground-truth feasibility.

## 2. Formulation
- Nominal problem with embedded models and the trust region.
- Label-noise model: `y = y* + ε`, features certain. Motivate from each case
  study (measurement noise in the reactor, survey noise in palatability,
  trial-arm sampling in toxicity).
- Linear smoothers: `f(x; θ(y)) = s(x)ᵀy + c(x)`. Table of `s(x)` per model class.
- Two targets, stated up front: the clean-label model `θ* = θ(y*)` (Section 3)
  and the true function `h*` (Section 4).

## 3. Robust counterpart and guarantee at the optimum
- RC: `f(x; θ̂) + R‖s(x)‖₂ ≤ b`. **[proved]**
- Theorem 1: violation probability at any RC-feasible x* is at most `P(W > R)`. **[proved]**
- Corollary 1, finite case (trees, forests, boosting): `R = σ√(2 ln(K/ε))`. **[proved]**
- Corollary 2, subspace case (ridge, last-layer NN), Gaussian noise:
  `R = σ√χ²(d, 1−ε)`. **[proved]** Sub-Gaussian version by a net argument. **[to prove]**
- Corollary 3, several constraints, Bonferroni. **[proved]**
- Simulated radius: the `1−ε` quantile of `W` (jointly over constraints) by
  Monte Carlo, each draw a maximization over X. Tighter than the union bound and
  than Bonferroni; needs a sampling-error term. **[to prove]**
- Honest fitting as the condition that makes A2 true for trees and networks.
- Estimated σ: guarantee with an upper confidence bound on σ. **[to prove]**

## 4. From the clean-label model to the true function
- Theorem 2: if `|h*(x) − f(x; θ*)| ≤ B(x)` on X, then
  `f(x; θ̂) + R‖s(x)‖ + B(x) ≤ b` gives `P(h*(x*) > b) ≤ ε`. **[proved, one line]**
- Instances of `B(x)`:
  - **Trees and forests, h\* L-Lipschitz:** `B(x) ≤ L Σᵢ sᵢ(x)‖xᵢ − x‖`, at most
    `L · diam(leaf)` for a tree. A constant per leaf, so linear in the leaf
    binaries. **[to prove]**
  - **Ridge, h\* linear in the features with ‖θ‖ ≤ S:** the self-normalized
    bound gives a margin `β‖φ(x)‖_{V⁻¹}` that covers noise and shrinkage bias
    together and is uniform in x. Still a second-order cone. **[to prove: adapt
    Abbasi-Yadkori et al., Thm 2]**
  - **Last-layer NN:** approximation error η of the frozen features, assumed. **[open]**
- What cannot be done: no bound on the bias without some assumption on `h*`.
  The paper trades C-MICL's exchangeability at x* for regularity of `h*`.
- The trust region as the device that keeps `B(x)` finite.

## 5. Classification
- Labels `yᵢ ~ Bernoulli(p*(xᵢ))`, constraint `p(x) ≤ b`.
- Linear smoothers in the labels: honest classification trees and forests (leaf
  value is a label average), linear probability models.
- `yᵢ − p*(xᵢ)` is bounded, so sub-Gaussian with proxy 1/4. Corollary 1 applies
  with `σ = 1/2` and no noise estimate. **[to prove: short]**
- Clean-label target is the smoother applied to `p*`; Section 4's Lipschitz
  margin carries over.
- Label flips at rate η shift the mean affinely; correctable when η is known.
  **[to prove]**
- Out of scope: logistic regression and softmax networks (not affine in the
  labels). State as a limitation; a local-linearization result would be
  asymptotic only. **[open]**

## 6. Solving the robust counterpart
- Linear models: mixed-integer second-order cone program, p-dimensional cone.
- Trees and forests: `‖s(x)‖` through leaf binaries; directed cut
  `δ* = R s(x)/‖s(x)‖`; at most K iterations. **[to prove: short proposition]**
- Tolerance τ: guarantee stated for `b + τ`.
- Direct check: any method's output can be certified by evaluating the RC once.

## 7. Baselines and calibration
- Wrapper, C-MICL, RHS margin, and cutting planes over a random scenario bank
  (an inner approximation of the RC, so no guarantee).
- Each baseline has one dial. Dials are tuned to a target ground-truth
  feasibility so that methods are compared at equal feasibility.
- The RC is reported twice: at its theoretical R (no tuning) and at a tuned R
  (the price of the guarantee).

## 8. Computational experiments
All problems have a ground-truth oracle and synthetically generated data, so
noise can be redrawn and `h*` evaluated. Probability statements are over
training-noise redraws unless stated.

### 8.1 Membrane reactor (Ovalle et al.)
- One learned constraint, ODE oracle, Gaussian noise with known σ.
- Role: theorem validation, price of selection, bias, all ablations.

### 8.2 World Food Programme diet (Maragno et al.; also in Ovalle et al.)
- Palatability constraint with a simulator as oracle; noise injected at chosen levels.
- Role: lead practical case; larger MIO; solve-time comparison.
- Classification version: palatable or not, same instance.

### 8.3 Chemotherapy regimen design (Maragno et al.)
- Real data, several learned constraints, no ground truth.
- Semi-synthetic: fit a model to the real data, treat it as `h*`, redraw noise
  at the estimated σ. Role: several constraints at once (Corollary 3 and the
  simulated joint radius).

### 8.4 What each subsection reports
- Violation rate against `θ*` and against `h*`, beside ε.
- Objective, fraction of instances solved, solve time.
- Realized `W` against `R`.

## 9. Discussion
- Price of robustness and price of selection.
- Which assumption would you rather make: regularity of `h*` or exchangeability at x*.
- Limitations: honest fitting costs accuracy; `K` for large ensembles; models
  not affine in the labels.
- Future work: feature noise, model selection for embedding, recourse.
