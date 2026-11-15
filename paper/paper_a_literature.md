# Paper A literature check (2026-11-14)

The check that `paper_plan.md` (open point 5) and outline Section 1.2 asked
for: the robust constraint-learning papers marked "read in full", and novelty
against simultaneous bands, post-selection inference and safe learning. The
outcome is in outline Section 1.2 and in the introduction of `paper.tex`.
This file keeps the evidence.

Every reference is in `paper_a.bib` and in Zotero (Robust CL collection; the
new ones are tagged `lit-check-2026-11-14`, with metadata from Crossref by
DOI). "Read" says how much of each paper was read. Page numbers near a page
break should be checked against the PDF before they are quoted in the paper.

## Verdict

| Claim in the old outline | Status | Prior work |
|---|---|---|
| The band is restricted to the decision domain | **Not new** | Wynn and Bloomfield 1971; Naiman 1987 (polyhedral regions); Liu et al. 2005; Liu 2010 |
| The simulated radius is computed by optimizing over the domain for each draw | **Not new for linear models over a box** | Liu et al. 2005: quantile of a simulated sup over the box, each draw a QP |
| A guarantee at the optimizer's `x*` because the bound is simultaneous | **Known argument** | Projection argument: Berk et al. 2013; Andrews, Kitagawa and McCloskey 2024. Also implicit in every uniform bound in safe BO |
| `f̂(x) + R‖s(x)‖ ≤ b` as a robust counterpart | **Known for linear models** | Ellipsoidal coefficient uncertainty (Goldfarb and Iyengar 2003); safe linear bandits (Amani et al. 2019) |
| Embedded as a constraint in a mixed-integer problem | **New in this form** | Thebelt et al. 2022 solve mean + κ·sd over trees as an MISOCP, but κ is a tuning constant and there is no guarantee |
| Extended to tree ensembles and last-layer networks | **New** | No paper found that puts the label vector in an uncertainty set and propagates it through a model affine in its labels |
| Target is the clean-label model | **New, and weaker** | Safe BO and Fiedler et al. target the true function and need an RKHS-norm bound `B` |

"Not found" means not found in this search, not proven absent.

**Closest precedent for the margin.** Fiedler, Scherer and Trimpe (AAAI 2021).
**It is not constraint learning.** Its setting is learning-based robust
control: a GP learns an unknown dynamical system, a set of models containing
the truth with high probability is derived from it, and a controller is
designed to work for every model in the set (read from the PDF, Introduction).
Nothing in the constraint-learning literature found here has the margin.

For kernel ridge regression, with fixed inputs and independent R-sub-Gaussian
noise, their Proposition 2 bounds, uniformly over the domain,
`|μ_N(x) − f(x)| ≤ B σ_N(x) + R ‖s(x)‖ · ρ(N, δ)` with `s(x) = (K + λI)⁻¹ k(x)`:
the noise term is a radius times our `‖s(x)‖`. The statement of Proposition 2
was checked against the PDF; the exact form of `ρ` was not. The differences:
- They target the true `f`, so a function-norm bound `B` enters.
- Their radius bounds the whole noise vector (about σ√N), not a supremum over
  a restricted `X`.
- They cover kernels only.
- There is no MIP embedding.

**What the paper can claim.**
1. The clean-label target (no norm bound; the weaker claim stated openly).
2. The region is the mixed-integer feasible set, and each simulated supremum
   is a global MIP solved to a bound.
3. One label-space form for ridge, last-layer networks and forests.
4. The band is imposed as a constraint.

**Consequences for the method.**
- Liu et al. (2005, Sec. 5) found that iterative optimizers "may not converge
  to the global maximum": gradient projection returned about 2.5% non-global
  solutions on one example. The simulated radius has to be computed from the
  MIP's upper bound on each `W`, not its incumbent, or it is anticonservative.
  I4 in the plan already does this.
- Liu et al. handle an estimated σ by simulating `σ̂/σ` (a scaled χ) jointly
  with the normal vector, so that `T = Q·‖N‖/(σ̂/σ)`. This is a ready-made
  route for T2 (estimated σ), as an alternative to an upper confidence bound
  on σ.

## Constraint learning (read from Zotero)

**Maragno et al. 2025, OptiCL** (*OR* 73(2):1011–1028). Read: pp. 1–16 of
19; the online appendix (A.4, the wrapper) is not in the PDF.
- The trust region is the convex hull of the observations (Sec. 2.3), with
  an ε-enlarged version (Sec. 3.2). It is geometric, with no probabilistic
  content.
- The wrapper (Sec. 3.1, eq. 4) requires at most a fraction α of P
  bootstrapped estimators to violate the constraint.
- α is never tied to a violation probability. "This experiment demonstrates
  how the α parameter effectively controls the model's robustness"
  (p. 1023); "There is not a single set of optimal parameters" (p. 1022).

**Ovalle et al. 2025, C-MICL** (arXiv:2506.03531; no journal version found).
Read: pp. 1–11, App. B, App. E.
- Assumption 4.1 (p. 6): "The events of C-MICL feasibility and conformal
  coverage are conditionally independent given ground-truth feasibility."
- Theorem 4.1: "for any feasible solution … it is ground-truth feasible with
  probability at least 1−α".
- The authors say it may fail (Sec. 6, p. 11): it "could potentially fail in
  scenarios where the feasible region is biased towards data regions where
  the conformal coverage does not hold … an unavoidable theoretical
  compromise".
- The probability is "with respect to both the calibration data D_cal and
  the test point" (App. B.2): an exchangeable test point, not the optimizer's
  `x*`.
- Two possible gaps, in the reading agent's judgment (verify before using):
  - Lemma 3.1 covers the noisy label but Theorem 4.1 is stated for
    `h(x) = E[Y|X=x]`.
  - Lemma 3.1 needs Mondrian quantiles, but Sec. 4.1 pools one `q̂` (this
    matches cmicl.py DIFFERENCES #8).

**Alcántara and Ruiz 2023** (*Appl. Math. Model.* 121:445–462). Read: full
text (arXiv v1).
- Replaces `f(x) ≤ k` by `P(y ≤ k) ≥ β` by embedding a model of the
  conditional β-quantile: linear QR, SVQR, quantile forests, GBQR, QRNN.
- The uncertainty is the outcome's spread given `x`; the quantile model's own
  estimation error is not treated.
- There is no finite-sample or at-optimum guarantee; the equivalence holds
  only if the quantile is exact (Sec. 2.1, p. 7).

**Cremer et al. 2019** (*IEEE TPWRS* 34(1):791–801). Read: pp. 1–10 of 11.
- Decision-tree security rules embedded in a DC-OPF MILP.
- "a 99.9% accurate security rule … results in 0% accuracy when used to
  derive optimal control actions" (p. 792). Optimized points "accumulate upon
  the binding hyperplanes" (p. 794).
- Fixes:
  - per-split margins `ε_{n,m}`, tuned iteratively with simulator labels at
    the optimized points (Sec. III);
  - an AdaBoost and Platt-calibrated insecurity probability added to the
    objective (Sec. IV).
- No formal guarantee. Cite as motivation and as precedent for a margin that
  varies with `x`.

**Oztekin, Oztoprak and Birbil 2026, DCL** (arXiv:2607.25719). Read: full
text.
- Each iteration fits a local surrogate on nearest neighbours and solves
  inside a convex-hull or Mahalanobis trust region on `x`.
- Label noise is acknowledged but not modelled, and there is no guarantee.
- Uncertainty quantification is left as future work (p. 13): "A principled
  treatment would require quantifying the uncertainty of the learned
  surrogate … and propagating that uncertainty into a feasibility
  certificate."

**Xavier, Qiu and Ahmed 2021** (*IJOC* 33(2):739–756). Read: pp. 1–10 of 19.
- Learning-to-optimize for security-constrained unit commitment: screening
  constraints by kNN, warm starts, SVM-predicted affine subspaces.
- No learned constraint and no label noise. **Not constraint learning:
  dropped from the literature section.**

## Simultaneous inference and selection (web search)

- **Wynn and Bloomfield 1971** (*JRSS-B* 33(2):202–217): exact bands with the
  predictor restricted to an interval.
- **Naiman 1987** (*JASA* 82(397):214–219): bands over a polyhedral convex
  predictor set, narrower than unrestricted ones.
- **Liu, Jamshidian, Zhang and Donnelly 2005** (*JCGS* 14(2):459–484). Read:
  abstract, Secs. 1–2, the numerical results (Sec. 3) and the conclusions,
  from the PDF in Zotero.
  - **Region:** a rectangular box, `aᵢ ≤ xᵢ ≤ bᵢ`, only. General polyhedra
    are not handled; Naiman (1987) handles polyhedral regions, but
    conservatively.
  - **Model:** linear in `x`, with σ estimated.
  - **Critical constant:** simulated from
    `T = sup_box |xᵀ(b̂−b)| / (σ̂ √(xᵀ(XᵀX)⁻¹x))` (eq. 1.3). It is two-sided;
    ours is one-sided.
  - **Each draw:** the sup is the smallest angle between the normal vector
    and the cone `{Px : x in the box}` (eq. 2.2). It is solved to the exact
    global maximum in finitely many steps, by branching or by an active-set
    QP.
  - **Monte Carlo error:** the empirical quantile `ĉ` is reported with a
    standard error (Sec. 2.4), and simulation stops when it falls below about
    0.01 (50,000 to 100,000 draws). There is no exact finite-sample rule like
    our ⌈(N+1)(1−ε)⌉ order statistic. A minor difference, but ours needs no
    Monte Carlo error term.
  - **Verdict:** this is our simulated radius for a linear model over a box.
    Ours differs in the region (a mixed-integer feasible set, so each draw is
    a global MIP, not a finite QP), the model classes, and the finite-sample
    rule.
- **Liu 2010**, *Simultaneous Inference in Regression* (CRC): the survey.
  **Not read: no full text found.** Its chapters on polynomial and nonlinear
  bands are the place to look for a band over a nonconvex region.
- **Montiel Olea and Plagborg-Møller 2019** (*JAE* 34(1):1–17): sup-t band
  over a finite set, narrower than Bonferroni or Scheffé. Our `W` is its
  continuum analogue over `X`.
- **Berk et al. 2013** (*Ann. Stat.* 41(2):802–837): the PoSI constant is a
  quantile of the max of `|lᵀZ|/‖l‖` over a set of directions. Our `W` is the
  same max over `s(x)/‖s(x)‖`, `x ∈ X`.
- **Andrews, Kitagawa and McCloskey 2024** (*QJE* 139(1):305–358): projection
  intervals, which "form a simultaneous confidence band for all potential
  targets and take the implied set of values for the target of interest".
  This is the logic of our Theorem 1.
- **Smith and Winkler 2006** (*Mgmt. Sci.* 52(3):311–322): the optimizer's
  curse, for the objective.
- Related, not cited: Ito, Yabe and Fujimaki (ICML 2018); Gupta, Huang and
  Rusmevichientong (*OR* 2024); Iyengar, Lam and Wang (arXiv:2306.10081). All
  concern objective bias. No paper names a feasibility version of the curse.
- Related, not cited: confidence sets for the *location* of a response
  surface optimum (Peterson, Cahya and del Castillo 2002; Wan et al. 2016).
  These are not about feasibility.

## Robust optimization (web search)

- **Goldfarb and Iyengar 2003** (*MOR* 28(1):1–38): uncertainty sets are
  regression confidence regions; SOCP counterparts.
- **Bertsimas, Gupta and Kallus 2018** (*Math. Prog.* 167(2):235–292): sets
  from hypothesis tests, with guarantees for any feasible solution.
- **Hong, Huang and Lam 2021** (*Mgmt. Sci.* 67(6):3447–3467): a learned
  uncertainty set calibrated by data splitting so that the optimized decision
  meets a chance constraint, uniformly over decisions. The closest OR
  precedent for a guarantee at the optimum.
- **Hüllen et al. 2020** (*Comput. Chem. Eng.* 136:106519): RO on
  polynomial-surrogate parameters. **Full text not read**; the description is
  from the abstract and Wiebe et al.'s summary.
- Contextual and conformal RO (Patel et al. 2024; Johnstone and Cox 2021;
  Sun et al. 2023; Chenreddy and Delage 2024) cover a future random
  parameter, so the set does not depend on `x`. Not cited for now.

## Safe learning and GP-constrained optimization (web search)

- **Srinivas et al. 2012** (*IEEE TIT* 58(5):3250–3265),
  **Sui et al. 2015** (ICML, PMLR 37:997–1005) and **Berkenkamp et al. 2023**
  (*Mach. Learn.* 112:3713–3747): the safe set is where the confidence bound
  meets the threshold, with β valid uniformly.
  - Fiedler, Menn, Kreisköther and Trimpe (TMLR 2024, arXiv:2403.12948)
    document that implementations replace the theoretical β with heuristics.
    Not yet cited; useful if the paper argues that β is uncalibrated in
    practice.
- **Amani, Alizadeh and Thrampoulidis 2019** (NeurIPS 32:9256–9266):
  - The safe set is `μ̂ᵀBx + β‖Bᵀx‖_{A⁻¹} ≤ c`, an SOC counterpart of a
    learned linear constraint with a whole-space radius
    (Abbasi-Yadkori et al. 2011).
  - Online setting; targets the true θ.
- **Fiedler, Scherer and Trimpe 2021** (AAAI 35:7439–7447): see the Verdict
  section.
- **Schweidtmann et al. 2021** (*Math. Prog. Comput.* 13(3):553–581).
  Read: Sec. 6.2, from the PDF in Zotero.
  - Deterministic global optimization with embedded GPs.
  - Case study (N-benzylation, 78 data points): `P(f_impurity(x) ≤ c) ≥ 95%`,
    which "corresponds to the constraint m_D + 1.96 · k_D ≤ c", where `m_D`
    and `k_D` are the GP's prediction and its uncertainty.
  - The probability is under the GP posterior, at each `x`, with a fixed
    multiplier. There is no account of the optimizer selecting `x`; the
    authors note only that the constraint "is active at the optimal solution
    point".
  - The paper calls `k_D` the variance; the 1.96 multiplier implies a
    standard deviation. Quote the formula as printed, not reinterpreted.
- **Wiebe et al. 2022** (*Math. Prog.* 196:805–839):
  `μᵀx + F⁻¹(1−α)√(xᵀΣx) ≤ b`, and RO over a χ² ellipsoid for warped GPs.
  The probability is the GP posterior: Bayesian and pointwise.
- **Thebelt et al. 2022** (NeurIPS 35:37401–37415): mean + κ·sd over a
  tree-kernel GP as an MISOCP. κ is a hyperparameter and there is no
  guarantee. Precedent for the encoding.
- Related, not cited: ENTMOOT (Thebelt et al., *Comput. Chem. Eng.* 2021) and
  Mistry et al. (*IJOC* 2021), which add distance or penalty terms to embedded
  trees with no guarantee.

## Still to read before submission

Updated 2026-11-14: Liu et al. (2005), Schweidtmann et al. (2021) and Fiedler
et al. (2021) are now read from the PDFs in Zotero (above). No full text was
found for these two:

- Liu (2010), the chapters on polynomial and nonlinear bands.
- Hüllen et al. (2020): what its uncertainty set is. Until it is read, the
  paper cites it only as "RO on surrogate parameters".
