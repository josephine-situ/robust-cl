# Research plan: robust constraint learning

The standing plan behind the thesis. Adopted 2026-09-28 (see that entry in
`progress.md`). Edit this file when the plan changes, and log the change in
`progress.md`.

## Context

This project extends the OptiCL framework (Maragno et al., *Operations Research*,
2025) for mixed-integer optimization with learned constraints. A trained ML model
ĥ(x) is embedded in a mixed-integer program as a constraint, e.g. ĥ(x) ≤ τ.
Because the optimizer seeks extreme points, it exploits model error and produces
solutions that violate the true constraint.

- **Existing code:** compares robustness *methods* for a fixed model class. It
  does **not** yet compare model classes.
- **Primary references:** the OptiCL paper and appendix; C-MICL (Ovalle et al.,
  2025, conformal prediction for constraint learning); Bertsimas & den Hertog,
  *Robust and Adaptive Optimization* ("the textbook").
- **Motivating gap:** OptiCL's conclusion says robust optimization against
  prediction uncertainty has been handled for linear models (Goldfarb & Iyengar,
  2003) but is open for general ML methods.

| Acronym | Meaning |
|---|---|
| ARO | Adaptive robust optimization |
| CV | Cross-validation |
| C-MICL | Conformal mixed-integer constraint learning |
| DRO | Distributionally robust optimization |
| LDR | Linear decision rule |
| MIO | Mixed-integer optimization |
| RC | Robust counterpart |

## Thesis argument

**"Where should robustness live in a constraint-learning pipeline?"**

Three locations, each applicable to any constraint-learning problem:

| Location | What it controls | When applied | Sees the decision x? | Paper |
|---|---|---|---|---|
| Learner | *Which* function (model class, regularization) | Training | No | B |
| Embedding | *How much* margin around the learned constraint | Optimization | Yes | A |
| Domain | *Where* the optimizer may go (trust region) | Optimization | Yes | A or synthesis |

**Adaptivity** is not a fourth location. It is the condition that changes how much
robustness each location needs when the problem has recourse (Paper C).

**Core tension:** the learner reduces error on average but cannot see x; the
optimizer travels to where error is largest.

**Illustration (linear learner):**
- Learner robustness (textbook Ch. 26): robust regression ≡ regularization, which
  penalizes model complexity ‖β‖.
- Embedding robustness: with β ∈ {β̂ + Pz : z ∈ Z}, the RC is
  `β̂ᵀx + δ*(Pᵀx | Z) ≤ τ`, which penalizes the *decision* in uncertain
  directions, ‖Pᵀx‖.

## Paper A: probabilistic guarantees for embedding robustness

**Priority:** first. It is the theory anchor, and Paper B Section 3 depends on it.

**Technical crux:** textbook Ch. 15 assumes uncertainty enters as `ā + Pz̃`. Here
the uncertainty is in the learned function ĥ. For linear learners the map to P is
immediate; for trees and NNs, constructing P and Z *is* the contribution.

Deliverables:
1. A construction of P and Z from model error for at least two model classes.
2. An a priori bound via robust complexity ρ(Z), plus the tighter a posteriori bound.
3. A multiple-constraint treatment that beats the Bonferroni union bound.
4. A head-to-head comparison with C-MICL, framed as "which assumption would you
   rather make?" (C-MICL needs exchangeability plus a conditional-independence
   assumption it admits may fail).
5. *Stretch:* show the trust region is a limiting case of an embedding
   uncertainty set (zero margin inside the convex hull, infinite outside). If it
   doesn't fit, move it to the thesis synthesis chapter.

Reading:
- [ ] Textbook Ch. 2: support functions, RC reformulations
- [ ] Ch. 3, especially 3.3 (selecting set parameters) and 3.5 (uncertainty sets from data)
- [ ] Ch. 5: robust MIO
- [ ] Ch. 14: concave uncertainty, conjugates
- [ ] **Ch. 15: probabilistic guarantees (core)**
- [ ] C-MICL paper in full, including Appendix B (Mondrian conformal sets)
- [ ] Bertsimas, Gupta & Kallus, *Data-driven robust optimization* (Math. Programming, 2018)
- [ ] Goldfarb & Iyengar (2003)
- [ ] Esfahani & Kuhn (Wasserstein DRO), for positioning

## Paper B: optimization-aware model selection (learner)

**Claim:** CV error is the wrong criterion for choosing a learner to embed, because
the optimizer exploits exactly the errors CV averages away (the optimizer's curse
applied to constraint learning).

Structure:
1. **The gap.** Rank models by CV error and by post-embedding performance, and
   compute the rank correlation. Low correlation is the headline result.
2. **The fix.** Propose a selection criterion that measures error where the
   optimizer goes (e.g., near the trust-region boundary or along its direction of
   travel).
3. **The interaction.** Test whether the best model changes once embedding
   robustness is added. Hypothesis: NNs win on CV, but linear models and trees win
   after robustification because their uncertainty is cheaper to characterize.
   Takeaway: select and robustify jointly, not sequentially.

**Guardrail:** without Section 2 this is a benchmark. Don't write it up until the
criterion exists.

Core experiment, the model × method grid:

| | Nominal | Wrapper | Robust counterpart | C-MICL |
|---|---|---|---|---|
| Linear | | | | |
| XGB | | | | |
| NN | | | | |
| Stable regression | | | | |

Metrics per cell: CV error, out-of-sample feasibility, objective value, solve
time. Section 1 uses the Nominal column; Section 3 reads across rows.

Reading:
- [ ] **Textbook Ch. 26: robust regression ↔ regularization**
- [ ] Bertsimas & Paskov, *Stable Regression* (JMLR, 2020)
- [ ] Smith & Winkler, *The Optimizer's Curse* (Management Science, 2006)
- [ ] OptiCL §2.2 (embeddings) and Appendix A.4 (model wrapper baseline)
- [ ] Mišić, *Optimization of tree ensembles* (Operations Research, 2020)
- [ ] Anderson et al., strong mixed-integer formulations for trained NNs
- [ ] Elmachtoub & Grigas, SPO+ (positioning)

## Paper C: extension to problems with recourse

**Question:** when decisions are made in stages, how much less robustness does
each location need?

**Novelty:** in standard ARO, a parameter is revealed between stages. In
constraint learning, what is revealed is the model's error itself (e.g., observed
toxicity after chemotherapy cycle 1). This information structure is what separates
the paper from applying textbook ARO.

**Technical crux:** LDRs need fixed recourse and continuous adaptive variables, but
embedded trees and NNs introduce binaries. Finite adaptability is the likely route,
with LDRs as the tractable special case for linear learners.

**Applications:** OptiCL's chemotherapy and World Food Programme case studies (both
natively multi-stage).

Reading:
- [ ] Textbook Ch. 6, especially 6.4 (finite adaptability) and 6.5 (lower bounds)
- [ ] Ch. 7, especially 7.5 (nonlinear decision rules) and 7.6 (LDRs with inexact data)
- [ ] Ch. 9: integer adaptive variables
- [ ] Ch. 23: Benders decomposition for two-stage problems
- [ ] Zeng & Zhao, column-and-constraint generation (Operations Research Letters, 2013)
- [ ] Ch. 20: adaptive DRO (optional)

## Sequencing

| Phase | Paper A | Paper B | Paper C | Reading |
|---|---|---|---|---|
| 1 | Construct P, Z | Fill grid (Nominal column first) | — | Ch. 15, 26, 2, 3; Smith & Winkler |
| 2 | Bounds + C-MICL comparison | Develop selection criterion | — | Ch. 14, 5; Bertsimas, Gupta & Kallus |
| 3 | Submit | Interaction section (needs A's method); write up | Start reading | Ch. 6, 7 |
| 4 | — | Submit | Formulation | Ch. 9 |
| 5 | — | — | Computation + thesis synthesis | Ch. 23; Zeng & Zhao |

## Risks

- **Paper A:** the NN construction may not work cleanly. Decide early whether to
  scope to linear + tree ensembles.
- **Paper B:** most at risk of being dismissed as a benchmark or scooped. Don't let
  it slip past Phase 4.
- **Paper C:** finite adaptability may scale poorly with embedded binaries. Keep the
  LDR-for-linear-learners case as a fallback result.
