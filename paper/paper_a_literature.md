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

## FO-CP, robust-optimization machinery and instances (2026-11-18, web search)

Checked for the second method (first-order cutting planes, FO-CP) and the
instance choice. Not yet in `paper_a.bib` or Zotero. DOIs were checked in
Crossref or on the publisher's page unless marked †. "Read" says how much.

### Verdict for FO-CP

| Claim | Status | Prior work |
|---|---|---|
| Worst-case relabeling in a norm ball, by gradient through retraining, at a fixed decision | **Not new as a technique** | Label/data poisoning as a bilevel problem: Biggio et al. 2012; Mei and Zhu 2015; Muñoz-González et al. 2017; Jagielski et al. 2018; Koh and Liang 2017 (targeted at one test point); Xiao et al. 2012 (label flips) |
| The retrained model on `ȳ + Δy*` added as a cut in a MIP | **Partly new** | Cutting sets with approximate oracles (Mutapcic and Boyd 2009; Bertsimas, Nohadani and Teo 2010); several retrained models embedded together is the OptiCL wrapper. Choosing them adversarially, as cuts: not found |
| First linearized step = influence / leverage margin | **Known argument** | Koh and Liang 2017; ridge prediction `αᵀy` with `α = X(XᵀX+λI)⁻¹h(x)` in Rosenfeld et al. 2020 |
| The combination as RO for constraint learning | **Not found** (checked against both closest papers in full, 2026-11-19) | Closest: Kinsey et al. 2022 (poisoning a predict-then-optimize pipeline's decision); Johnson-Yu et al. 2023 (robust predict-then-optimize against label drift, in training); Zhang, Amin and Perakis 2026 (influence of noisy points on LP decisions) |

**Kinsey et al. 2022 and Zhang, Amin and Perakis 2026 read in full
(2026-11-19).** Neither has the combination. In both, the learned quantity is
the **objective** (cost vector) of a problem whose feasible set is known; no
learned constraint, no robust counterpart, no cuts, no feasibility guarantee.
Kinsey et al. attack (no defense); Zhang et al. defend by reweighting training
points with a clean validation set, and bound regret, not violation. Entries
below.

### Poisoning, influence, implicit differentiation

- **Biggio, Nelson and Laskov 2012**, ICML, pp. 1467–1474, arXiv:1206.6389.
  Read: abstract. Gradient ascent through the SVM's optimality conditions to
  raise test error.
- **Mei and Zhu 2015**, AAAI, pp. 2871–2877, doi:10.1609/aaai.v29i1.9569.
  Read: abstract. Optimal training-set attack as a bilevel problem with
  implicit-function gradients (SVM, logistic, linear regression): FO-CP's
  separation with a different attacker objective.
- **Muñoz-González et al. 2017**, AISec, pp. 27–38, arXiv:1708.08689,
  doi:10.1145/3128572.3140451†. Read: abstract. Back-gradient (unrolled)
  differentiation, the alternative when a network has no clean optimum.
- **Jagielski et al. 2018**, IEEE S&P, arXiv:1804.00308,
  doi:10.1109/SP.2018.00057†. Read: abstract. Poisoning ridge and LASSO by
  inserted points (not a label ball); the TRIM defense.
- **Koh and Liang 2017**, ICML (PMLR 70†), arXiv:1703.04730. Read: abstract.
  The influence of training points on a test prediction: FO-CP's first
  gradient.
- **Blondel et al. 2022**, NeurIPS, arXiv:2105.15183. Read: abstract. Tooling
  for implicit differentiation.
- **Rosenfeld, Winston, Ravikumar and Kolter 2020**, ICML (PMLR 119),
  arXiv:2002.03018. Read: method section. Certifies a prediction (not a
  decision) against label flips, using ridge on a fixed feature map, the RC's
  `s(x)`.
- **Kinsey, Tuck, Sinha and Nguyen 2022**, GameSec, LNCS 13727, pp. 231–252,
  doi:10.1007/978-3-031-26369-9_12. Zotero FDKL8KS9 (no PDF; Springer).
  Read: in full, as Ch. III of Kinsey's 2024 Oregon PhD thesis
  (cs.uoregon.edu/Reports/PHD-202411-Kinsey.pdf, pp. 60–84, 148–150),
  "adapted from" the GameSec paper; check the Springer text before quoting.
  - Setting: a network predicts objective coefficients θ; the decision is
    `max f(x, θ̂)` over a known `Ax ≤ b` (convex QP). Learners attacked:
    two-stage, decision-focused, and a network outputting x directly.
  - Attack: box perturbation of training **features** `uᵢ ∈ [lb, ub]`
    (labels "can be extended" in a footnote; the portfolio case perturbs raw
    prices, hence features and labels). Goal: a bad decision at one target
    test point. Three-level problem (adversary, decision argmax, training
    argmin), solved by projected gradient descent.
  - Gradient: through training by **unrolled** gradient descent (Higher),
    each retraining **from a random initialization** with restarts, keeping
    the best; through the decision by the implicit function theorem on the
    QP's KKT system (qpth).
  - Findings: attacks on the two-stage learner transfer to decision-focused
    ones; attacking decision-focused learners directly is expensive and
    unreliable; **larger budgets often gave weaker attacks** (local optima);
    MetaPoison does not carry over. Future work asks what guarantees a
    defense could give: open.
  - For FO-CP: cite as "worst-case training perturbations against a
    predict-then-optimize decision, as an attack, with learned objective
    coefficients and a known feasible set; no defense". Their
    non-monotone-in-budget attacks are evidence for the local-separation risk
    (multi-start, a posteriori attack). Their random-start retraining is the
    ill-defined training map our fixed `θ₀` avoids.
- **Johnson-Yu, Finocchiaro, Wang et al. 2023**, GameSec (earlier at AAMAS
  2023). Read: abstract; author list and DOI not verified.
- **Zhang, Amin and Perakis 2026**, "Reduced Cost Influence Functions for
  Predict-then-Optimize under Noisy Data", NeurIPS (OpenReview CiBySOXzK9),
  all three at MIT ORC. Zotero 8QJEQ4QN. Code:
  github.com/william-zhang64/Reduced-Cost-Influence-Functions. Read: in full
  (38 pp. with appendix).
  - Setting: a model `f_θ(z)` predicts the **cost vector** of a
    nondegenerate LP `min cᵀx, Ax = b, x ≥ 0`; noisy training set (a fraction
    β of labels corrupted) plus a small **clean validation set**.
  - Method: RCIF = influence of upweighting training point j (Koh and Liang)
    on a reduced cost `δ(B,k)ᵀ f_θ(z_v)` at a validation point, i.e. on its
    distance to the optimality-cone boundary. Scores sum RCIFs over
    validation points (signed by whether the predicted basis is correct;
    "directed" variant picks the facet the true cost violates); weights
    `∝ exp(λΨ)`, refit, iterate (λ = 0.05, τ = 5).
  - Theory (linear model, squared loss): for OLS, `RCIF = −δᵀ rⱼ κ(z_v, zⱼ)`
    with `κ = z_vᵀ Σ̂⁻¹ zⱼ`, which is n times the RC's `sⱼ(x)` for OLS; a
    corrupted point's score drops w.h.p. under directional corruption (D4);
    regret **upper bounds** beat unweighted and validation-only ERM for a
    middle range of validation size. No feasibility statement (the feasible
    set is known).
  - Experiments: random LP, PyEPO shortest path (linear and a small tanh
    network), misspecification; gains are small but significant. Iterating
    hurts with the network (influence error compounds; they cite Basu et al.
    2020 on fragile influence in deep nets). Against minimum-norm
    cone-flipping corruption no first-order method helps (App. B.5): "the
    influence of a training point scales with its perturbation magnitude".
  - For FO-CP: closest in machinery (influence through the trained model to
    a decision-relevant linear functional), different in everything else:
    learned objective not constraint, reweighting not robust counterpart,
    clean validation data required, regret not violation. Two warnings for
    F0/F1: influence on networks is fragile, so the implicit gradient must be
    checked against retraining (F0(ii)) and each candidate confirmed by
    retraining (already planned); and first-order signals miss small,
    decision-targeted perturbations. Cite Basu et al. 2020 beside Koh and
    Liang. Possibly worth mentioning to DB (MIT ORC colleagues).
- Not cited unless needed: Meyer, Albarghouthi and D'Antoni 2022
  (arXiv:2206.03575, certifying linear regression against label bias;
  perturbation set not confirmed); Bertsimas and Boucher 2025
  (arXiv:2503.02173, uncertainty set on the predicted output).

### Robust-optimization machinery

- **Per-constraint sets.** Ben-Tal, El Ghaoui and Nemirovski 2009, *Robust
  Optimization*, Princeton, doi:10.1515/9781400831050, **Ch. 1, Sec. 1.2,
  item B, pp. 11–12** (Example 1.2.6): the RC "is a purely 'constraint-wise'
  construction" and "remains intact when U is extended to the direct product
  of its projections". Read: those pages.
- **Risk allocation.** Nemirovski and Shapiro 2006, *SIAM J. Optim.*
  17(4):969–996, doi:10.1137/050622328, eq. (2.14)–(2.15) and Remark 2.1
  (p. 977): Σαᵢ ≤ α by the union bound; αᵢ as decision variables destroys
  convexity, so αᵢ = α/m. Read: Sec. 2. Joint calibration instead: Chen, Sim,
  Sun and Teo 2010, *OR* 58(2):470–485, doi:10.1287/opre.1090.0712 (not
  read).
- **Protecting a fraction of the constraints.** Gorissen, Yanıkoğlu and den
  Hertog 2015, *Omega* 53:124–137, doi:10.1016/j.omega.2014.12.006, Sec. 6,
  "K out of N constraints": exactly the big-M model with Σzᵢ ≥ K; they note
  it is more conservative than "for every ζ, K of N hold". Ours adds the
  nominal rows. Related: light robustness (Fischetti and Monaci 2009, LNCS
  5868:61–84, doi:10.1007/978-3-642-05465-5_3, abstract only); the
  comprehensive RC (Ben-Tal, Boyd and Nemirovski 2006, *Math. Program.*
  107:63–89, doi:10.1007/s10107-005-0679-z); soft robustness (Ben-Tal,
  Bertsimas and Brown 2010, *OR* 58(4):1220–1234,
  doi:10.1287/opre.1100.0821); maximum feasible subsystem (Amaldi, Pfetsch
  and Trotter 2003, *Math. Program.* 95(3):533–554).
- **Scenario optimization.** Calafiore and Campi 2006, *IEEE TAC*
  51(5):742–753, doi:10.1109/TAC.2006.875041; Campi and Garatti 2008, *SIAM
  J. Optim.* 19(3):1211–1230, doi:10.1137/07069821X (metadata only). **Campi,
  Garatti and Ramponi 2018**, *IEEE TAC* 63(12):4067–4078,
  doi:10.1109/TAC.2018.2808446 (read in full): for nonconvex or
  mixed-integer decisions the guarantee is a posteriori, from the size s* of
  a support sub-sample, and it bounds violation under a **fresh draw**, not
  violation of the true constraint. Sample sizes at ε = 0.1, β = 10⁻³
  (computed): convex d = 5, 143; a posteriori s* = 5, 361; s* = 10, 600;
  mixed-integer a priori (5 continuous, 3 binary), 698. Also Calafiore,
  Lyons and Fagiano 2012 (CDC, doi:10.1109/CDC.2012.6426905), Mohajerin
  Esfahani, Sutter and Lygeros 2015 (*IEEE TAC* 60(1):46–58).
- **Inexact separation.** Mutapcic and Boyd 2009, *Optim. Methods Softw.*
  24(3):381–406, doi:10.1080/10556780802712889 (Secs. 1, 4 read): with an
  approximate pessimizing oracle "we cannot say that the method solves the
  robust problem; indeed, we cannot even (surely) determine if a point x is
  feasible"; they certify the final point separately. Bertsimas, Nohadani
  and Teo 2010, *OR* 58(1):161–178, doi:10.1287/opre.1090.0715 (preprint
  read; constrained version *IJOC* 22(1):44–58): multistart gradient ascent
  for the worst case, "a heuristic strategy". Guarantees under inexact
  worst cases: Kuchlbauer, Liers and Stingl 2022 (*JOTA* 195:1056–1086;
  *IJOC* 34(4):2106–2124; not read); Mitsos 2011 (semi-infinite programs).

### Instances used across constraint learning

Fajemisin, Maragno and den Hertog 2024, *EJOR* 314(1):1–14,
doi:10.1016/j.ejor.2023.04.041: the survey does not organize by application;
its lead example is the WFP diet. C-MICL (Ovalle et al.) has exactly two case
studies: the reactor (regression, ODE truth) and the **WFP baskets as
classification** (4 palatability classes, a network fitted on all baskets as
truth).

| Instance | Used by | Learned constraints | Truth | Data |
|---|---|---|---|---|
| **WFP palatable diet** (regression) | the survey, OptiCL, Alcántara and Ruiz 2023, SurrogateLIB, DCL 2026, Liu et al. 2026 (C-MICL as classification) | 1 (25 decisions) | **closed form**: OptiCL App. C.1 generates palatability from a known function (constants γ_g not published) | OptiCL repo, 5,000 rows |
| Workload dispatching (Lombardi, Milano and Bartolini 2017, *AIJ*) | EML, OptiCL App. E, SurrogateLIB | 48 (one per core) | thermal simulator, apparently not public | SurrogateLIB / PySCIPOpt-ML |
| Concrete mix (UCI) | Alcántara and Ruiz 2023, DCL 2026 | 1 | none; semi-synthetic | UCI |
| Oil-well production (Grimstad and Andersson 2019) | that paper, Liu et al. 2026 | about 10 | noise-free simulator data | location not found |
| Student enrollment (JANOS) | JANOS, OptiCL, Gurobi ML | classification | — | synthetic |

Not verified: page numbers for JANOS, EML, Peters et al.; the WDP simulator.

## Still to read before submission

Updated 2026-11-14: Liu et al. (2005), Schweidtmann et al. (2021) and Fiedler
et al. (2021) are now read from the PDFs in Zotero (above). No full text was
found for these two:

- Liu (2010), the chapters on polynomial and nonlinear bands.
- Hüllen et al. (2020): what its uncertainty set is. Until it is read, the
  paper cites it only as "RO on surrogate parameters".
- Blankenship and Falk (1976, JOTA, discretization for semi-infinite
  programs) and Mitsos (2011, restriction of the right-hand side): FO-CP's
  stopping rule and T7 now rest on them; both cited from memory
  (added 2026-11-19).
- Kinsey et al. (2022): the Springer text, to confirm the thesis chapter
  matches before quoting (read in full from the thesis 2026-11-19; Zhang,
  Amin and Perakis read in full the same day).
