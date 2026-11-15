# Progress log: robust constraint learning

These are notes for weekly advisor updates. The `/weekly-update` skill in
`../research-updates` reads everything dated after the last deck. Decks up to
2026-09-08 are in `../research-updates/`.

Keep entries short: what changed, the evidence (a path, commit, or number), and
what's open. For theory worked out in the Claude.ai project, paste a dated summary
here with the statement, its assumptions, and whether it's proven or conjectured.
The standing thesis plan is in `research_plan.md`; log changes to it here.

## 2026-09-23: C-MICL's coverage gap was the width model's training batch

**Finding.** Our C-MICL fell short of Ovalle et al.'s coverage because of how the
width net `u` is trained, not because of the data or `h`'s architecture. sklearn's
default batch is `min(200, n)`; theirs (Keras) is 32. With the default batch, the
fitted width surface has a dip. The MIP walks into it, `u >= 0` clamps the width
to zero, and the tightening disappears. We adopted `batch_size: 32`. We did not
adopt their other two trainer differences: fixed 2000 epochs measured worse
(0.37), and their `/100` input scaling needs their `h` (nothing solvable).

Reactor probe, N=100 cost draws, alpha=0.1, ground-truth feasibility:

| cost draw | before | after | mean u(x*) | optima with u(x*) < 0.1 |
|---|---|---|---|---|
| `scaled` (randomized `balanced`, what we report), job 23497509 | 0.45 | 0.83 | 0.044 → 1.539 | 98% → 0% |
| `paper` (their draw), job 22854117 | 0.53 | 0.81 | | negative cal widths 9/200 → 0/200 |

`nominal` is 0.00 under both.

**Dial cells re-scored** (commit caa7a15). Only the `cmicl` series changed; every
other series came back bit-identical.
- **Reactor** (job 23500609): the tuning curve now climbs to 0.6 at alpha=0.05
  (it was flat at 0.0–0.1), but there is still no `dial*`. ODE feasibility is
  0.2 → 0.3 at the protocol alpha=0.1 and 0.1 → 0.4 at the best-feasibility cell.
  The probe's 0.83 is a rate over random cost directions, while the cell has one
  fixed direction. Batch 32 helps but doesn't rescue C-MICL on this instance.
- **Gastric** (job 23501706): `dial*` stays at alpha=0.6, feasibility
  0.919 → 0.929 (objective 9.781 → 9.776). The wider `u` makes tight alphas solve
  less: alpha=0.4 0.77 → 0.57, and the protocol alpha=0.1 0.081 → 0.056 of contexts.

**Deferred.** The gastric test stage (job 23579054) has finished, but its results
are not pulled or read yet. Leave them out of this week's deck; the focus is the
research plan (2026-09-28).

## 2026-09-28: Thesis plan and Paper A guarantee (Claude.ai project)

### Thesis plan

Full plan in `research_plan.md`. The organizing question: **where should
robustness live in a constraint-learning pipeline?**
- **Learner** (model class, regularization; can't see x): Paper B.
- **Embedding** (margin around the learned constraint; sees x): Paper A.
- **Domain** (trust region; sees x): Paper A or the synthesis chapter.
- **Recourse** changes how much each location needs: Paper C.

Core tension: the learner reduces average error, but the optimizer travels to
where error is largest.

- **Paper A (first):** probability guarantees for embedding robustness. Construct
  P, Z from model error for at least two model classes, give a priori and
  a posteriori bounds, beat Bonferroni across constraints, and compare against
  C-MICL's assumptions.
- **Paper B:** CV error is the wrong selection criterion for an embedded learner.
  Show a low rank correlation between CV and post-embedding performance, propose
  an optimization-aware criterion, and test whether the best model changes under
  robustification. Needs a model × method grid (linear, XGB, NN, stable
  regression × nominal, wrapper, RC, C-MICL). The current code compares methods
  for a fixed model class only.
- **Paper C:** recourse. What is revealed between stages is the model's error
  (e.g., toxicity after cycle 1). Finite adaptability, with LDRs for linear
  learners as the fallback.

Risks: the NN construction for A may not work (decide early whether to scope to
linear + trees); B could be read as a benchmark if the criterion doesn't
materialize.

**Where the existing work sits in the plan.**
- CP, the wrapper, the margin and C-MICL all robustify the *decision*, so they
  are embedding-location methods (Paper A). CP's random bank approximates the
  exact RC from the inside, and Paper A's analysis gives the exact separating cut
  (below).
- `robust_reg` trains a label-robust model and embeds it nominally
  (`src/methods/robust_regression.py`), so it is the learner location (Paper B).
  It was dropped from the dial sweep because its dial is rho.
- The dial-sweep curves (objective × held-out feasibility, reactor and gastric)
  are Paper A's empirical side, for one fixed model class per problem. Nothing
  yet varies the model class, which is Paper B's grid.

**Asks for the advisor.**
- Does the three-location framing and the A → B → C order make sense?
- For Paper A, should the NN construction be in scope, or should it be linear +
  tree ensembles?
- Paper B's grid has a "Robust counterpart" column. Should that be CP (the
  repo's method, an inner approximation of the RC) or the exact RC where one
  exists (the $\ell_2$-ball MISOCP for linear models)?

Next: run the Nominal column of the grid across all four model classes; read
textbook Ch. 15 and Ch. 26.

### Paper A: pointwise violation bound under label noise

Covers points 1–2 of the plan. The per-method table, level-3 set containment and
the caveats section are not yet worked through.

**Setup and assumptions.**
- Observed labels $y = y^* + \varepsilon$, with $\varepsilon_i$ independent,
  zero-mean, $\text{subG}(\sigma^2)$. For heteroscedastic noise, replace
  $\sigma^2\|s\|^2$ with $\sum_i \sigma_i^2 s_i^2$.
- **Target: the clean-label model** $\theta^* = \theta(y^*)$, the same estimator
  refit on noise-free labels, not $h^*$. The bias $h^* - f(\theta^*)$ is
  excluded; only noise propagation $f(\theta^*) - f(\hat\theta)$ is covered.
- So well-specification isn't needed (it would require realizability plus exact
  recovery, which fails for ridge, EN and trees), and the pointwise bound needs
  neither $R_c$ nor a set $\mathcal D$.
- Models: linear/ridge, elastic net (EN), XGBoost (XGB), random forest (RF),
  neural network (NN).

**Result (derived).** If the model is a linear smoother in the labels,
$f(x;\theta(y)) = s(x)^\top y + c(x)$, with $s(x)$ non-random with respect to
$\varepsilon$ (fixed structure, fixed $x$), then
$f(x;\theta^*) = f(x;\hat\theta) - s(x)^\top\varepsilon$. Prop. 15.2(a) gives
$-s^\top\varepsilon \sim \text{subG}(\sigma^2\|s\|^2)$ (independence makes the
MGF factor), and Prop. 15.1 with slack $t(x) = b - f(x;\hat\theta)$ gives

$$P(\text{violation}) \le \exp\!\left(-\frac{t(x)^2}{2\sigma^2\|s(x)\|^2}\right).$$

The affine-map step is ours, not the textbook's.

**Sensitivity vectors** ($\phi(x)$ is the preprocessed input row, $Z$ stacks the
training rows):

| Model | $s(x)$ | Exact when |
|---|---|---|
| OLS/ridge | $Z(Z^\top Z + \lambda I)^{-1}\phi(x)$ | Always |
| EN | Ridge formula on active set $A$ | Active set and signs stable |
| Single tree | $\mathbf 1\{x_i \in \ell(x)\}/n_{\ell(x)}$, so $\|s\|^2 = 1/n_{\ell(x)}$ | Frozen splits |
| RF | Forest weights $\alpha_i(x) = \frac1B\sum_b c_{bi}\mathbf 1\{\cdot\}/N_b(x)$ | Frozen structure + honest |
| XGB | $m_0 + \eta\sum_t r_{t,\ell_t(x)}$ | Frozen splits, `reg_alpha` = 0 |
| NN | Last-layer ridge in penultimate activations $\psi(x)$ | Frozen hidden layers |

- With an intercept, $\|s\|^2 = 1/n + (\phi - \bar\phi)^\top M_c(\phi - \bar\phi)$;
  the terms are orthogonal because $Z_c$'s columns are centered.
- $\|s\| = \|L^\top\phi(x)\|$ with $M = LL^\top$ shrinks the SOC from $n$ to $p$
  dimensions ($p+1$ with an intercept).
- RF: $c_{bi}$ counts draws of row $i$ for tree $b$ (trees weighted evenly, labels
  not). The weights are nonnegative and sum to 1, so $n_\text{eff} = 1/\|s\|^2$
  (Kish). One bootstrapped tree has variance $\sigma^2\sum_i (c_{bi}/N_b)^2$, not
  $\sigma^2/N_b$. With precomputed leaf vectors $v_{b\ell}$,
  $s(x) = \frac1B\sum_{b,\ell} z_{b\ell} v_{b\ell}$ is linear in OptiCL's leaf
  binaries.

**Exact robust counterparts.** For any frozen-structure linear smoother and any
convex compact $\mathcal D$, the RC is $f(x;\hat\theta) + \delta^*(s(x)\,|\,\mathcal D) \le b$
(Ch. 2 support function).
- An $\ell_2$ ball gives $R\|s(x)\|$: a $p$-dimensional SOC for linear models, so
  a MISOCP. In coefficient space it equals a Goldfarb–Iyengar confidence ellipsoid.
- For RF the RC is exact for "same trees, leaves refit," not for retraining, and
  needs an $n$-dimensional cone (no $p$-dimensional reduction).
- **Link to CP:** random draws approximate the RC from the inside. The exact
  separating cut (§11.1) is $\delta^* = R\,s/\|s\|$.

**Where it breaks.**
- **Trees need honesty.** Splits are chosen on the noisy $y$, so
  $E[\alpha^\top\varepsilon \mid \text{structure}] \ne 0$. Conditioning on
  bootstrap counts and feature draws is fine. The fix is honest (sample-split)
  forests (Wager–Athey), at the cost of a larger $\|s\|$.
- **Data-dependent $x^*$.** A union bound over $K$ cells gives
  $\Gamma = \sqrt{2\ln(K/\epsilon)}$.
- Toy (2 trees, 4 training points, 4 cells): $\|s\|^2$ ranges over 3/8–1,
  $P(\text{violation}) \approx 0.006$ in the example cell, and $\Gamma$ is 2.72
  with the union bound against 2.15 pointwise.

**References.** Textbook Def. 15.1 (sub-Gaussian; the printed
$\exp(-s^2\sigma^2/2)$ should read $+$), Prop. 15.1, Prop. 15.2(a), Thm. 15.1
($\hat\theta$ as the nominal value, $\sigma s(x)$ in place of $P^\top x$), Ch. 2,
§11.1. External: Meinshausen (2006) on forest weights, Wager–Athey on honest
forests, Lee et al. (2016) for EN without sample splitting.

**Open.** Should honest fitting be part of Paper A's method, or should
frozen-structure selection bias be stated as a limitation?

## 2026-09-29: Debrief: thesis direction approved
- DB agreed with "where should robustness live in the pipeline" as the thesis direction: theory and modeling first, computational comparisons paused.
- Next steps: finish the embedding-robustness theory (per-method bounds, set containment, caveats).

## 2026-11-14: Paper A re-scoped; computational comparisons resume

Outline: `paper/paper_outline_v2.md`. Tasks, experiments, sequence:
`paper/paper_plan.md`. Nothing run yet.

**Methodology changes.**
- **The method is the exact robust counterpart**, `f + R‖s(x)‖ ≤ b`, solved by a
  second-order cone (linear models) or directed cuts (tree ensembles). Cutting
  planes over the random bank become a baseline: they approximate the RC from
  the inside and carry no guarantee.
- **Bias: add a margin `B(x)`** so the guarantee is against `h*`, under a
  regularity assumption (Lipschitz for honest trees and forests; realizable with
  bounded norm for ridge, via a self-normalized bound). Stated, not yet proved.
- **Classification in scope** for models affine in the labels (honest
  classification forests, linear probability models). Bernoulli labels have
  variance proxy 1/4, so no noise estimate is needed. Logistic and softmax
  models are out.
- **The analytic synthetic instance is dropped.** Validation moves to the
  reactor (ODE oracle, known σ) and WFP (simulator), with semi-synthetic
  chemotherapy for several constraints.
- **Evaluation changes probability space:** rates over training-noise redraws
  (500 per cell), judged against both the clean-label model and `h*`.

**Data against the new plan.**
- Reactor: generator and ODE oracle cover every validation experiment, but the
  current embedded network (10,5,2) does not satisfy A2. Needs honest trees and
  forests and a ridge or last-layer model first.
- WFP: not in the repo. Largest build.
- Chemotherapy: 416 arms, six outcomes; usable for the method comparison as is,
  and for the several-constraints test once a semi-synthetic `h*` is fitted.
  The gastric test stage (job 23579054) is still unread.

**Open (advisor).** Is a Lipschitz constant acceptable for the headline
guarantee? Which A2 model classes for the reactor? Does the existing dial-sweep
comparison go in the paper?

Next: fit the A2 model classes on the reactor and measure their accuracy cost;
build the honest forest, the `rc` method and the noise-redraw harness; run the
theorem-validation experiment.

### Same day, evening: the paper leads with practical results

Outline: `paper/paper_outline_v3.md` (supersedes v2). Plan rewritten in
`paper/paper_plan.md`. Nothing run.

**Methodology changes.**
- **The paper's claim stands alone:** the robust counterpart reaches its target
  ground-truth feasibility without tuning, costs less at matched feasibility
  than a constant margin, the wrapper and C-MICL, and still prescribes with six
  constraints. Each is a hypothesis with a stated failure condition.
- **Chemotherapy stays on the real data**, under Maragno et al.'s protocol
  (96 held-out arms, full-cohort ensemble as judge, subsampled training rows).
  It supports practical claims only; the theorem is tested on the reactor.
- **Honest fitting is a condition of the theorem for trees, not of the
  method.** The reactor uses polynomial ridge, where it is not needed.
- **Theory is one section.** Theorem 1, the radii, several constraints. The
  bias margin becomes a remark; the gap to the truth is measured against the
  ODE. Classification is deferred.
- **Experiments cut from nine to four** (feasibility and cost on the reactor,
  radius tightness, six constraints on chemotherapy, WFP last). The dial sweep and
  the random-bank cutting planes are not in the paper.
- **The simulated radius is the recommended one.**

**Two local checks on the reactor** (one noise draw, nothing saved):
- Polynomial ridge is affine in the labels and fits nearly as well as the
  network: five-fold R² 0.949 at degree 3 and 0.958 at degree 4, against the
  network's 0.958.
- At ε = 0.1 for the cubic, the radius is 1.28σ pointwise, 3.6σ simulated and
  8.4σ from the chi-square bound. The worst-case radius is too conservative to
  lead with.

Next: the `rc` method for ridge, the redraw harness, then the head-to-head.
