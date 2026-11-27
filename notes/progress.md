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

### Same day, late: the lead model is the network; first go/no-go probe

Outline v3 and `paper/paper_plan.md` revised. Nothing on the paper's design
has finished running.

**Methodology changes.**
- **Lead model on the reactor: Ovalle et al.'s (32, 32) network with the RC on
  its last layer**, not cubic ridge. Cubic ridge on the balanced cost: nominal
  ODE-feasible in 96 of 100 redraws, because the cubic's bias at the optimum
  (about 1.8 flow units) is on the safe side. No failure to fix.
- **Reactor design: Ovalle et al.'s 100 cost vectors, noise redrawn for each.**
  The fixed-cost design is dropped.
- **Every baseline embeds the same model.** C-MICL also gets one row as
  published, so the baseline is not handicapped.
- **Where the network's hidden layers come from is open.** Freezing them on a
  second noise draw meets the theorem but needs data a practitioner lacks. A
  sample split (half the rows each) meets it with one dataset; in a local check
  the last layer reaches R² 0.944 on its half, against 0.958 for the full
  network.
- **Why the earlier methods are out:** random-bank cutting planes carry no
  guarantee and needed the oracle to set rho and tau; the RHS margin and the
  wrapper have no setting tied to a violation level. The margin and the
  wrapper stay as baselines. Table in `paper_plan.md`.

**Data.** The reactor (1,000 designs, σ = 2, ODE judge) is enough for E1. On
Ovalle et al.'s costs, the 2026-09-23 probe already shows nominal at 0.00 and
C-MICL at alpha = 0.1 at 0.81, so the premise likely holds and C-MICL is the
close competitor.

**Running.** Job 29141177 (`mit_preemptable`, 32 tasks): 100 redraws each for
cubic ridge, the frozen network, the standard network and the sample-split
network, at the nominal, pointwise, simulated and closed-form radii, plus a
check of the enumerated radius against the solver for the network.

**Also found.** The gastric test stage (job 23579054) hit its 2-day wall on
2026-09-26; its partial output and checkpoint are on the cluster, unread.

### Same day, night: go/no-go probe results (job 29141177)

Reactor, Ovalle et al.'s 100 cost vectors, noise redrawn per run, ε = 0.1,
σ = 2 known. Feasibility against the clean-label model / the ODE.

| Model | Nominal (ODE) | RC at simulated radius | RC at closed-form radius |
|---|---|---|---|
| Cubic ridge | 0.66 | 1.00 / 0.95 (+4.2% of objective range) | 1.00 / 1.00 (+8.6%) |
| Network, hidden layers frozen | 0.18 | 1.00 / 0.52 | 1.00 / 0.79 |
| Network, sample-split | 0.25 | 1.00 / 0.42 | 1.00 / 0.61 |
| Network, standard fitting | 0.23 | 0.76 / 0.33 | 0.98 / 0.45 |

**Findings.**
- **The method works on cubic ridge**: under Ovalle et al.'s costs its nominal
  decision fails (0.66; the 0.96 earlier was the balanced cost), and the RC
  delivers 0.95 against the ODE with misses of 0.14 flow units on average.
- **On the network it does not, and the cause is bias, not noise.** At the
  nominal optimum the network's clean-label model overstates the ODE in 80 to
  90% of runs (by 1.6 to 2.8 flow units on average); the cubic's error there
  has mean zero. The optimizer selects on the network's misfit, which a
  label-noise margin cannot see. Proposal for DB: cubic ridge leads, and the
  network becomes the example of where the margin is not enough.
- **Standard fitting breaks the guarantee**: 0.76 against the clean-label
  model at a 0.90 target, which supports the honest-fitting condition.
- **No selection effect at ε = 0.1**: the pointwise radius already reaches
  0.90 to 0.92 against the clean-label model. The simulated radius is
  conservative for its own target; on the cubic its extra margin is what
  lifts ODE feasibility from 0.80 to 0.95. Cubic at ε = 0.05 and 0.2 (jobs
  29142767, 29142768) says the same: pointwise 0.95 and 0.77 against the
  clean-label model (targets 0.95, 0.80), simulated 1.00 at every ε and 0.93
  to 0.97 against the ODE. So the simulated radius pays for bias spread on
  this instance, not for selection. Figure 3's framing is open.
- **The enumerated radius understates the certified one** by 0.10 to 0.24σ on
  the network, so E1 needs the solver-certified radius (4 to 19 s per draw).

Detail and the proposal: `paper/paper_plan.md`, "Results of job 29141177".

## 2026-11-14: Paper A literature check narrows the novelty claim

**Finding.** Two of the three things the outline called new are prior work.
- Bands restricted to a region of the covariates, with the critical constant
  simulated by optimizing over that region on each draw, are Liu et al.
  (JCGS 2005) for linear models over a box.
- Validity at the optimizer's chosen point because the bound is simultaneous
  is the projection argument of post-selection inference (Berk et al. 2013)
  and inference on winners (Andrews, Kitagawa and McCloskey 2024).
- The closest single paper is Fiedler, Scherer and Trimpe (AAAI 2021), from
  learning-based robust control, not constraint learning. Their uniform bound
  for kernel ridge has a noise term equal to a radius times our `‖s(x)‖`, but
  it targets the true function, so it needs a function-norm bound. No
  constraint-learning paper has the margin.

**What survives.**
- The clean-label target: no norm bound, and a weaker claim, stated openly.
- The decision set as a mixed-integer region, with each simulated supremum a
  global MIP solved to a bound.
- One form for ridge, last-layer networks and forests.
- The band imposed as a constraint.

The robust constraint-learning papers (Alcántara and Ruiz, Cremer et al.,
Oztekin et al.) do not overlap. Xavier et al. is learning-to-optimize and is
dropped. C-MICL's at-optimum guarantee rests on an assumption its authors say
may fail, as the outline said.

**Evidence.** `paper/paper_a_literature.md`; outline Section 1.2; 28 new
entries in `paper/paper_a.bib`, also in Zotero.

**Read from the full texts since.** Liu et al. (2005) works over a box only,
solves each draw exactly with a finite QP, and handles an estimated σ by
simulating `σ̂/σ` jointly. That last part is a route for T2.

**Open.** Liu (2010) and Hüllen et al. (2020) have no full text.

## 2026-11-14: How the RC paper handles infeasibility on chemotherapy

**Methodology change** (design only; nothing run). Details in `paper/paper_plan.md`,
"Infeasibility on chemotherapy".
- **CP's feasibility rollback is not carried over.** It gave up robustness
  wherever robustness was unaffordable, with no stated level.
- **Its replacement is a certified level for any decision.**
  `r̂(x) = minⱼ (bⱼ − f̂ⱼ(x)) / ‖Σⱼ^½ sⱼ(x)‖` satisfies
  `P(violation and r̂ ≥ r) ≤ P(W > r)` however `x` was chosen (T6, from
  Theorem 1's proof). Where the RC at the target is infeasible, prescribe the
  decision with the largest `r̂` and report its level; otherwise abstain.
- **σ per arm from the binomial sampling error** is the main choice. The RC's
  margin is a confidence band, which shrinks with n; C-MICL's is a prediction
  interval, which does not. So the RC may be far less infeasible than
  C-MICL's 6% prescribed, but only if σ is the label noise and not the
  residual sd.
- **H3 and Table 2 move from "fraction prescribed" to "fraction certified at
  each level, for every method"**, with the judge's verdict beside it. The
  certified level does not depend on the weak chemo judge.

**Next (after the meeting).** Probe P1: ridge on the gastric training arms,
max-t per test cohort, under the binomial σ and the residual sd. It is the
go/no-go for H3 before I6, I7 and I9 are built.

## 2026-11-14: C-MICL's Assumption 4.1 by surrogate family (reactor)

Probe `experiments/probe_cmicl_family_coverage.py`, job 29144969. Ovalle et
al.'s 100 cost vectors, their fixed rf / gbm / 32x32-MLP settings plus a
depth-5 CART, `u` the production width net, alpha = 0.1, noise redrawn 5 times
on the seed-42 designs (rf and gbm have 4: their seed-4 tasks crashed because no
point in the bisection pool was C-MICL-feasible, and the probe does not handle
an empty boundary set). Coverage and feasibility under the ODE.

| Family | Coverage: random / boundary / x* | C-MICL feasible at x* | Nominal feasible | Error h − F: boundary → x* |
|---|---|---|---|---|
| MLP | 0.96 / 0.94 / 0.51 | 0.60 (sd 0.22) | 0.21 | +0.2 → +1.4 |
| Gradient boosting | 0.92 / 0.77 / 0.57 | 0.72 (0.05) | 0.46 | −0.8 → +1.1 |
| Random forest | 0.92 / 0.53 / 0.40 | 0.59 (0.16) | 0.41 | −0.7 → +1.6 |
| CART depth 5 | 0.90 / 0.29 / 0.40 | 0.45 (0.09) | 0.19 | +5.6 → +4.5 |

**Findings.**
- **No family reaches 0.9 on their own costs.** C-MICL is at 0.45 to 0.72.
- **Trees are not uniformly worse.** The single tree is the worst; the
  ensembles are level with the MLP (rf) or better (gbm). Differences between
  rf, gbm and the MLP are within the spread over redraws.
- **The failure is a different mechanism by family.** MLP: the boundary is
  covered (0.94) and selection alone breaks it, with `u` at `x*` about half its
  value at random boundary points (the width-dip mechanism, still present at
  batch 32). Ensembles: the error at random boundary points is pessimistic,
  and at `x*` it turns optimistic, so selection flips its sign. CART: the
  boundary itself is already optimistic, before any selection.
- **The corner mechanism is real, and large for the nominal decision.** The
  nominal tree `x*` is the cheapest point of its cell in every run, and most of
  its optimistic error is position within the cell: 3.9 (gbm), 5.1 (rf), 11.4
  (CART) flow units, against 1.8 to 2.4 from the choice of cell. C-MICL's
  smooth width pulls `x*` off the corner (14 to 16% of its cell is cheaper),
  which cuts that term to 0.8 to 1.0.
- **For Paper A:** the RC's tree margin is constant on a cell and its guarantee
  is against the clean-label tree, so it cannot see the within-cell error.
  Forests and boosting on the reactor are likely to fail the ODE as the network
  did. LinearDT remains untested.
- Marginal coverage with noisy labels is 0.88 in every family, slightly under
  0.90; consistent with `q` biased down by negative-width calibration rows
  (cmicl.py, WIDTH MODEL), not checked.

**Follow-up (same day, local, refits reproduce every run's `q` exactly).**
- **C-MICL's feasible region is tiny**: 0.2% (gbm), 0.3% (rf), 0.4% (CART) and
  2.8% (MLP) of the domain, against 5 to 12% where the nominal `h >= 50`. It
  holds 0 to 2 of the 200 calibration rows for the trees and 12 for the MLP
  (seed 0). So `q` is calibrated almost entirely where no decision can go.
- **The "boundary" reference is weaker than the table suggests.** The
  bisection had only 11 to 16 C-MICL-feasible anchors for the trees (91 for the
  MLP) out of about 4,000 domain points, so its boundary points cluster around
  a few places. The MLP's boundary-to-`x*` drop (0.94 to 0.51) is the
  trustworthy one; the tree rows of that column are rough.
- **Assumption 4.1 cannot be tested in its own probability space here**: of
  1,000 random reference points, 0 to 5 per run are C-MICL-feasible. For the
  MLP (about 25 points pooled) coverage given C-MICL-feasible is 0.86 against
  0.96 given ground-truth feasible: the direction of a violation, too few
  points to call. What the probe does establish is that the guarantee does not
  transfer to the optimizer's `x*` (feasibility 0.45 to 0.72 against 0.90).

**Assumption 4.1 tested directly** (job 29148520, stage `a41`). 300 random
points uniform inside C-MICL's feasible region per run (rejection from the
design box), against 2,000 random domain points. Means over 4 to 5 redraws.

| Family | Truth feasible: random feasible point / `x*` | Coverage given truly feasible: in C-MICL region / anywhere |
|---|---|---|
| MLP | 0.99 / 0.60 | 0.93 / 0.97 |
| Gradient boosting | 0.96 / 0.72 | 0.66 / 0.94 |
| Random forest | 0.93 / 0.59 | 0.49 / 0.93 |
| CART depth 5 | 0.45 / 0.45 | 0.66 / 0.96 |

- **The theorem's conclusion holds at random points for the MLP and the
  ensembles** (0.93 to 0.99), and fails for the single tree (0.45). For those
  three the shortfall at `x*` is the optimizer's selection.
- **Assumption 4.1 fails for every tree family**: coverage among truly feasible
  points is 0.49 to 0.66 inside C-MICL's region against 0.93 to 0.96 overall.
  It nearly holds for the MLP (0.93 against 0.97). The ensembles still meet the
  conclusion because their miscoverage there is on the safe side: the
  assumption is sufficient, not necessary.
- Not used: the infeasible half of the assumption. With coverage defined for
  the true value, covered and C-MICL-feasible imply truly feasible, so that
  half is 0 by construction (0 in every run). Whether their coverage event is
  for the true value or the noisy label decides whether that is a flaw in the
  assumption or in this measurement; check against their App. B first.

## 2026-11-15: First H2 check: the RC against a constant margin on cubic ridge

Job 29223094 (`--stage frontier`, `submit_rc_frontier.sh`). Same 100 redraws as
R1 (Ovalle et al.'s costs, noise redrawn, ε = 0.1, σ = 2), each solved on an RC
radius grid and a constant-margin grid `F ≥ 50 + m`, judged by the ODE. The RC
at its simulated radius reproduces R1 (0.95, +4.24%). Cost is the increase
over nominal as % of the objective range; SE on feasibility ≤ 0.03.

| ODE feasibility | RC (radius, cost) | Constant margin (m, cost) |
|---|---|---|
| ~0.90 | 3σ: 0.92, 3.54% | 1.75: 0.90, 3.41% |
| ~0.95 | simulated 3.71σ: 0.95, 4.24% | 2.5: 0.96, 4.86% |
| 1.00 | 4.5σ: 1.00, 4.96% | 3.5: 1.00, 6.79% |

- **At 0.9 the two frontiers coincide** (within one SE). H2 as stated ("the
  RC is cheapest at matched feasibility") is not supported at the target on
  this model.
- **The RC is cheaper at higher feasibility**: 4.96% against 6.79% to reach
  1.00. Where its margin shape matters is the tail.
- **The margin's m was picked with the ODE; the RC's radius was not.** No rule
  sets m. The natural one, `m = z₀.₉ σ = 2.56`, gives about 0.96 at 4.9%,
  against 0.95 at 4.24% for the simulated radius.
- Why so close: with a uniform design on the box, `‖s(x)‖` varies little
  near the optima, so the cone margin is nearly constant there. A design with
  uneven coverage (or smaller n) is where the shape should pay.

Outputs: `results/paper_a/prelim/frontier_cubic_paper_eps0.1*.{json,csv}`.

### Same day: the frontier on thinner designs (jobs 29224939, 29224940)

Same probe and redraws, two training designs chosen on x alone (Theorem 1
holds): **n = 200** random rows (42 above flow 45), and **thin**: 790 rows,
all with flow ≤ 45 plus 20% of the rest (44 above 45). Full design: 1,000
rows, about 250 above 45. ODE feasibility, cost as % of the objective range.

| Design | Nominal | RC simulated radius (untuned) | RC, cheapest ≥ 0.9 | Margin, cheapest ≥ 0.9 (ODE-tuned) | Margin m = z·σ = 2.56 (untuned) |
|---|---|---|---|---|---|
| full, 1,000 | 0.66 | 0.95, 4.24% | 3σ: 0.92, 3.54% | 1.75: 0.90, 3.41% | about 0.96, 4.9% |
| n = 200 | 0.23 | 0.98, 12.9% | 3σ: 0.94, **11.2%** | 7: 0.93, **14.9%** | about 0.60 |
| thin | 0.65 | 0.95, 7.67% | 3σ: 0.90, **6.08%** | 2.5: 0.90, **5.25%** | about 0.90 |

- **The untuned RC lands at 0.95 to 0.98 in every design.** No constant m
  does: the m that delivers ranges from 1.75 to 7, and the σ-based rule gives
  0.96, 0.60 and 0.90. This is H2's defensible form: the RC adapts its margin
  to the design without the oracle.
- **At matched feasibility the RC is cheaper only at n = 200** (11.2% against
  14.9% at 0.93 to 0.94, and 12.9% at 0.98 against 21.5% for the margin at
  0.97). On the thin design the tuned margin is cheaper (5.25% against 6.08%
  at 0.90; 6.35% at 0.96 against 7.67% at 0.95). On the full design they tie.
  So "cheaper" is design-dependent, and thinning only near the boundary is
  not where it pays.
- **The selection effect appears on both thin designs**: the pointwise radius
  reaches 0.84 (n = 200) and 0.82 (thin) against the clean-label model, below
  the 0.90 target; the simulated radius reaches 1.00. On the full design it
  was 0.90. This gives Figure 3 its argument.
- **Bias dominates at n = 200**: at R = 1.5σ, 0.90 against the clean-label
  model but 0.62 against the ODE, with misses up to 13 flow units.

Outputs: `results/paper_a/prelim/frontier_cubic_{n200,thin0.2}_paper_eps0.1*`.

### Same day: quick C-MICL and split conformal on the same redraws (jobs 29226346-48)

`--stage conformal`. Cubic ridge `h` on a random 80% of the design's rows,
calibrated on the other 20%, score `|y − h|/u`, constraint `h − q·u ≥ 50`.
**C-MICL-poly**: `u` a cubic polynomial on `h`'s absolute in-sample
residuals, used raw with `u ≥ 0` (Ovalle et al.'s semantics, but NOT their
32x32 width network). **Split conformal**: `u = 1`, a constant margin set by
held-out residuals. Cost against the frontier's all-row nominal.

At each method's own protocol setting (ε = α = 0.1, nothing tuned), ODE
feasibility / cost:

| Design | RC, simulated radius | C-MICL-poly, α = 0.1 | Split conformal, α = 0.1 |
|---|---|---|---|
| full, 1,000 | **0.95 / 4.2%** | 0.91 / 6.7% | 0.98 / 7.7% |
| n = 200 | **0.98 / 12.9%** | 0.55 / 7.5% | 0.81 / 11.4% |
| thin | **0.95 / 7.7%** | 0.87 / 8.2% | 0.98 / 9.2% |

- **Only the RC reaches the target in all three designs, and it is the
  cheapest delivering method at its protocol point in each.** Split conformal
  overshoots where data is ample (0.98, paying 1.5 to 3.5 points more) and
  falls short at n = 200 (0.81). C-MICL-poly falls short at n = 200 and thin.
- **Along the whole α grid:** full design, C-MICL-poly reaches 0.90 at 4.1%
  (α = 0.3) and 0.95 at 8.3% (α = 0.05), against the RC's 3.5% (0.92) and 4.2%
  (0.95). At n = 200 it never passes 0.57 at any α, and `u(x*)` falls from 0.92
  to 0.39 as α tightens: the optimizer goes where the width is small, the
  family probe's width-dip mechanism. Thin: it never passes 0.91.
- **Split conformal beats the RC on the thin design at 0.95** (6.75% at
  α = 0.2 against 7.67%), as the ODE-tuned margin did. Elsewhere the RC is
  cheaper at matched feasibility.
- Caveats: polynomial `u`, not their network; C-MICL fits `h` on 80% of the
  rows as published, which costs it most at n = 200 (n_cal = 40, so α = 0.02
  is not certifiable there).

Outputs: `results/paper_a/prelim/conformal_cubic_*`.

## 2026-11-16: Meeting with DB: a non-parametric method joins the RC

**Decisions.**

- Still exploratory: work on the outline and plan, not the draft.
- We went through C-MICL and the old random-bank cutting planes. DB is
  reasonably happy with the robust counterpart, but it is parametric (the
  prediction must be affine in the training labels), so he also wants a
  non-parametric method.
- His proposal (sketch: `c:\Users\jsitu\Downloads\robust-cl-cp.jpeg`): cutting
  planes in which the model is retrained on each relabeling and the worst
  relabeling is found by first-order methods (for example PyTorch). The
  separation is `max_{‖Δy‖≤R} Σₖ wₖ max(ĥₖ(x*; ȳ + Δy) − bₖ, 0)`, a ReLU
  with one weight per constraint so infeasibility can be handled. The master
  adds the model retrained on `ȳ + Δy*` and re-solves.
- Semi-synthetic data first; real data later.

**Methodology.**

- The sketch writes `bₖ − ĥₖ`; it is the violation `ĥₖ − bₖ` (confirmed).
- The uncertainty set is the RC's: `‖Σ^{-1/2}Δy‖ ≤ R`, jointly over all
  outcomes' labels, with Σ the noise covariance (confirmed).
- The RC is the method's closed-form case: for an affine map, one ascent step
  gives the RC's directed cut `R s(x)/‖s(x)‖`. On a network the first step,
  by implicit differentiation, is an RC with the linearized `s(x)`. The
  non-parametric method adds whatever lies beyond that step.
- It runs on a network trained end to end, so it avoids the honest-fitting
  problem that broke the last-layer RC under standard fitting (R3: 0.76
  against the clean-label model, target 0.90).
- Open: the guarantee. An exact one needs the model-free radius
  `σ√χ²(n, 1−ε)` (about 32σ at n = 1,000, against 3.6σ for the RC's
  simulated radius on cubic ridge) and exact separation, which a local search
  does not give. The cheaper radii (linearized subspace, scenario/bootstrap)
  are approximate.
- Consequence for the weights: over that joint set, a weighted hinge sum of 0
  means every constraint holds under every relabeling, whatever the
  weights. So in the separation the weights choose which cut is found, not
  where the loop ends, and they cannot resolve infeasibility. The same
  argument recovers the RC as written: each constraint's worst case uses only
  its own block of Σ. With one model per outcome, one cut per violated
  constraint is deeper than the weighted-sum cut, so FO-CP uses
  per-constraint cuts. Infeasibility is handled by the certified fallback
  (largest feasible R) or by protecting a fraction of the constraints
  (binary per constraint, at least ⌈pK⌉ protected; the guarantee then covers
  the protected set). Penalizing violations in the master was considered
  and dropped: no guarantee, and the weights become another dial.
- The objective stays nominal; a robust objective is a deferred variant.

**Data for the plan.** SS1: the reactor (ODE truth, Ovalle et al.'s 100 cost
vectors, σ = 2), plus a realizable truth (a network fitted to noise-free ODE
labels) to separate label noise from the bias that sank the network RC. SS2:
the 416 real chemotherapy arms' covariates, a fixed fitted model per outcome
as the truth, binomial noise from the real arm sizes. This gives the plan its
first several-constraints experiment with a ground truth. SS1 has everything
built except the new method; SS2 needs a generator.

**Next.** PyTorch training map and separation (not installed yet); F0, the
check that the new method reproduces the RC on cubic ridge; F1, whether the
non-parametric step adds anything over the linearized one. Details:
`paper/paper_plan.md`, "After the meeting with DB (2026-11-16)".

## 2026-11-18: Paper A decisions on the non-parametric method

- **One paper, two methods**: the RC for models affine in their labels, the
  first-order cutting planes (FO-CP) for the rest, with the RC as FO-CP's
  closed-form case.
- **FO-CP's radius is chosen by how it performs on the reactor** (model-free
  ball, linearized subspace or scenario radius; probe F2), not fixed in
  advance.
- **Infeasibility**: binary weights, a protected fraction p of the
  constraints chosen by the master, compared with the certified fallback
  (probe F3 on semi-synthetic chemotherapy).
- **Instances**: new ones come from those widely used in the optimization
  literature (WFP; student enrollment, a logistic model, as FO-CP's case), not
  from new constraints on the reactor.
- The plan's top section is condensed to what is settled, and the 2026-11-14
  run notes are reduced to the results table and the lead-model proposal,
  which is still open.

### Same day: FO-CP uses one uncertainty set per outcome

- Switched from one ellipsoid over all outcomes' labels to one per outcome,
  `‖Σₖₖ^{-1/2}Δyₖ‖ ≤ R`, the form the RC already used. The robust
  constraints are identical (each model sees only its own labels), it is the
  standard RO form (a robust constraint depends only on the set's projection
  onto its own parameters), and the model-free radius is much smaller: on
  chemotherapy (n = 320, K = 5, ε = 0.1) 19.3 per outcome with ε split by a
  union bound, against 40.9 for one joint ellipsoid. Correlation across
  outcomes enters only through the jointly calibrated scenario radius.

### Same day: literature check and plan review; FO-CP's radius is the crux

- **Literature** (`paper/paper_a_literature.md`, new section). Every part of
  FO-CP is prior work: its separation is bilevel label poisoning (Mei and Zhu
  2015 and others), its first step the influence function (Koh and Liang
  2017), its loop a cutting-set method with a local oracle (Mutapcic and Boyd
  2009). The combination (adversarially retrained models as cuts protecting
  a learned constraint) was not found; Kinsey et al. 2022 and Zhang, Amin and
  Perakis 2026 are the closest and must be read in full. Protecting a
  fraction of the constraints is Gorissen et al.'s (2015) "K out of N".
- **Radius.** The RC's simulated radius (3.6σ on cubic ridge) is not the
  radius of a ball that contains the noise (`‖e‖ ≈ 32σ` at n = 1,000); it
  works because an affine model sees noise only through `s(x)ᵀe`. So the RC
  shares FO-CP's constraint form, not its guarantee. FO-CP's only exact
  radius is the model-free ball (about 32σ); new probe F−1 measures its cost
  with existing code before any FO-CP code is written. The scenario
  alternative guarantees violation only for a resampled model (Campi,
  Garatti and Ramponi 2018; about 361 retrained models at ε = 0.1).
- **Training map fixed**: every retraining from one fixed start, lbfgs to a
  gradient tolerance, label standardization constants fixed from `ȳ` (the
  probe standardized each label vector by its own mean and sd, not affine),
  damped Hessian checked against finite differences.
- **Instances.** WFP becomes regression with OptiCL's closed-form truth (the
  most reused instance in constraint learning, six regression papers), the
  second instance for both methods. Semi-synthetic chemotherapy and the
  infeasibility probe move to the RC track (FO-CP needs a differentiable
  training map; those models are forests and ridge).
- In the paper FO-CP is "for any model trained by a differentiable loss",
  not "non-parametric".
- Next: F−1 (cluster, existing code) and P1 (local, still not run).

## 2026-11-19: Probe F-1: the model-free radius is impractical

Job 29561590 (`probe_rc_ridge.py --stage modelfree`). The RC at the
model-free radius σ√χ²(n, 0.9) = 32.5σ (n = 1,000): the label ball that
contains the whole noise vector, valid for any training map, i.e. the new
method's exact radius (a). Same 100 redraws and Ovalle et al.'s cost vectors
as job 29141177, so the rows pair with R1 and R2. Feasibility against the
clean-label model / the ODE; cost as % of the objective range.

| Model | Radius | R (flow) | Clean / ODE | Cost | Margin at x* |
|---|---|---|---|---|---|
| Cubic ridge | simulated 3.7σ | 7.43 | 1.00 / 0.95 | +4.2% | 2.51 |
| Cubic ridge | closed-form 8.4σ | 16.72 | 1.00 / 1.00 | +8.6% | 4.65 |
| Cubic ridge | model-free 32.5σ | 65.05 | infeasible in 100 of 100 | — | — |
| Frozen network | simulated 3.5σ | 7.05 | 1.00 / 0.52 | +2.2% | 1.17 |
| Frozen network | closed-form 5.8σ | 11.52 | 1.00 / 0.79 | +3.3% | 1.91 |
| Frozen network | model-free 32.5σ | 65.05 | 1.00 / 1.00 | +15.7% | 8.45 |

- **Radius (a) is out as a practical radius.** On cubic ridge no decision
  satisfies it in any redraw (solver-certified; the largest affordable radius
  is about 18–23σ by enumeration). On the frozen network it is feasible and
  reaches 1.00 against the ODE, but at 7 times the cost of the simulated
  radius: its 8.45-unit margin swamps the network's bias (mean −1.6), which
  is why it delivers against the ODE.
- So the new method needs a practical radius, option (c), which bounds
  violation only for a resampled model unless a sharper argument is found.
  That definition is the next decision; it gates FO-CP code.

### Same day: FO-CP's radius

- Chosen: a sampled version of the RC's simulated radius. For noise draws e′,
  retrain on `ȳ + e′` and record `W_c = max over the whole domain of
  [ĥ(x; ȳ+e′) − ĥ(x; ȳ)] / ‖s̃(x)‖`, with s̃ the linearized sensitivity; R is
  the (1−ε) order statistic. Computed once before the loop, over a domain
  fixed without the labels (re-sizing R around the iterates would bring the
  selection problem back); gradient ascent then searches the ball of that
  radius at every iteration. For affine models it is exactly Theorem 1's
  simulated radius; for networks the argument is first order.

## 2026-11-19: Probe P1: the RC certifies half the gastric cohorts under the binomial σ

Job 29576766 (`experiments/probe_p1_gastric.py`, 8 `mit_quicktest` tasks).
Ridge (fixed penalty) on the five toxicity percentile labels, fit on the 320
training arms (training R² 0.37–0.48). For each of the 96 test cohorts, the
certified radius r̂ = the largest RC radius at which the cohort still has a
regimen (exact, by Dinkelbach over MISOCPs with the domain constraint and
trust region; 5–6 solves, 10–20 s per cohort). Joint radius over all five
toxicities by enumeration (96 contexts × 282 training regimens, 1,000 draws).
Every cohort is nominally feasible.

| σ | Joint radius, ε = 0.1 | Certified at ε = 0.05 / 0.1 / 0.2 | Median r̂ | Median level |
|---|---|---|---|---|
| Binomial per arm, delta method to the percentile scale (median 0.10–0.26) | 4.29 | 0.43 / 0.50 / 0.61 | 4.31 | 0.09 |
| Out-of-fold residual sd (0.31–0.33) | 4.19 | 0 / 0 / 0 | 2.73 | 1.00 |

- **Under the label noise the theorem is about (binomial σ), half the
  cohorts get a regimen certified at 90%.** H3 stays viable.
- **But the median cohort sits on the radius**: 0.47 / 0.43 / 0.22 are
  certified if the radius is 0.1 / 0.2 / 0.5σ larger, and enumeration
  understated the solver's W by 0.10–0.24σ on the reactor network. E2 needs
  the solver-certified radius before any fraction is quoted.
- **Under the out-of-fold σ nothing is certified** (best level 0.53): the
  residual sd is about the label sd, so σ is mostly ridge's misfit. This is
  the misfit C-MICL's width also absorbs, and why it prescribes for ~6%.
- Data: 416 real arms (320 train, 96 test), five toxicities as percentile
  ranks, arm sizes 11–289. Not yet measured: baselines' certified levels,
  forests in place of ridge.

## 2026-11-19: Probe F0: FO-CP's machinery, and the network's L2 penalty

**Built** `experiments/probe_focp.py`: the training map (I11; L-BFGS-B on the
exact sklearn objective from one fixed θ₀, gradients and Hessian by torch
autograd; torch is now a CPU-only dependency) and the separation (I12;
implicit gradient, monotone projected ascent).

**F0(i) passes.** On cubic ridge and the frozen network the generic implicit
gradient equals `s(x)` to 4e-14, and one ascent step reproduces the RC's
margin `R σ ‖s(x)‖` exactly, with no further step accepted.

**The end-to-end network's training map is not exact.** A ReLU network's
minimizer sits on kinks, so max|grad| stalls near 1e-3 while the loss is
flat to 8 digits: the plan's gradient-tolerance stop is unattainable, and the
map now stops on loss change. With the repo's α = 0.01, retraining on the same
labels moves h(x*) by up to 0.09 flow units, more than one label's effect
(0.005 to 0.05), so F0(ii)'s per-row check against finite differences cannot
be measured. Run instead along whole directions (redraws 0 to 2, R = 3.59σ):

| | α = 0.01 | α = 1 |
|---|---|---|
| drift of the map at x* | up to 0.09 | ≤ 0.002 |
| drop of h(x*) along the implicit gradient at R | 5.5 to 8.2 | 1.4 to 2.3 |
| along random directions | ≤ 1.5 | ≤ 0.08 |
| frozen-network RC margin, same x | 0.84 to 0.92 | 0.95 to 1.38 |
| ODE at the nominal x* (floor 50) | 39.3 to 42.3 | 48.6 to 50.1 |

The implicit gradient is a good ascent direction at both. α = 0.01 (the
synthetic instance's CV pick) makes the end-to-end network so label-sensitive
that FO-CP's margin would be 6 to 9 times the RC's, and its nominal decision
misses the ODE by 8 to 11. At α = 1 both are of the RC's size; there the
gradient overstates small steps (kinks), which a kink-aware sensitivity
would fix. **Open for JS: the network's α.** Three redraws only.

## 2026-11-19: Network α by CV; F0(iii) passes; F1 submitted

- **α = 1 by 5-fold CV** on the dataset's noisy labels (`cv_alpha.json`; CV MSE
  4.37 against σ² = 4; the repo's 0.01 scores 8.97). JS chose CV. The end-to-end
  network for F1 and F2 uses α = 1. R2 to R4's networks used α = 0.01.
- **F0(iii) passes** (`f0_loop_r0.json`): on the frozen network and cubic ridge
  the cutting-plane loop converges to the RC's own solution as τ → 0 (objective
  gap +2.7% / +1.6% at τ = 0.5σ, +0.005% at τ = 0.001σ; 1 to 10 cuts), and each
  loop decision satisfies the RC with slack about τ.
- **F1 submitted**, job 29664832 (redraws 0 to 4). torch is installed (CPU) in
  the cluster's `robcl_env`; jobs that import it need the env's `lib` first on
  `LD_LIBRARY_PATH` (in `submit_f1.sh`).

## 2026-11-19: Probe F1: ascent adds about 20% over the linearized step

Job 29667805; end-to-end network at α = 1, redraws 0 to 4, R = 3.59σ, at the
nominal decision and one robust iterate (`results/paper_a/prelim/f1_r*_a1.json`).
Drop of the prediction, σ units, mean over 10 points:

| linearized step | multistart ascent | random bank (B = 200) | true relabeling | cold restarts (5 seeds, max abs.) |
|---|---|---|---|---|
| 1.03 | 1.25 (median 1.20× linearized) | 0.13 | 0.27 | 0.68 |

- **Go for FO-CP's multistep separation**, though the gain over one step is
  modest (2% to 88%, median 20%).
- **The random bank finds a tenth** of the worst case: the old random-bank
  cutting planes could not have found it.
- **Training randomness is half the label-noise worst case**, and larger than
  the effect of retraining on the clean labels. The guarantee is against the
  fixed-θ₀ map on clean labels; the seed spread must be reported beside it.
- At 1 of 10 points the clean-label retrain lowers the prediction more than the
  worst case in the ball: radius (c)'s question, for F2.
- d_eff ≈ 45 to 57 directions for 90% of the energy, so radius (b) ≈ 8 to 18σ.
- 44 to 117 min per point including the bank and restarts; F2 must time the
  separation alone.

## 2026-11-19: Radius (c) pilot: enumeration understates it, and retraining is chaotic

Jobs 29707708, 29708663, 29709194 (enumeration, 10 redraws × 2 SS1 arms, N = 200)
and 29710101 (local refinement on 5 redraws); `results/paper_a/prelim/radius_*`.

- **R ≈ 3.5 to 5.5σ by enumeration**, but the margin it imposes (R × median
  ‖s(x)‖) is steadier: about 2 flow units, coefficient of variation 0.06 to 0.09.
- **Enumeration misses the maximum.** Local search on the same retrained model
  raises a draw's W by a median 1.0 to 1.6σ (up to 3.6σ); R rises 10 to 17%.
  The ratio peaks where the linearized sensitivity is small.
- **Retraining is chaotic for full-size noise.** A 1e-10 change to the labels
  moves the retrained prediction by up to 0.8 flow units; so does the thread
  count, and another cluster node reproduced W for only 18 of 125 draws. The
  clean-label reference T(y*) is a relabeling of the same size, so fixing θ₀
  does not make it deterministic.
- **Open for JS:** how to define radius (c) and the clean-label reference for a
  network. F2 pilot (job 29709201) is running with the enumerated radii.

## 2026-11-20: F2 pilot: FO-CP reaches the target on both SS1 arms

Job 29709201; 10 cost vectors per arm, end-to-end network at α = 1, radius (c) per
redraw by enumeration, τ = 0.05σ (`results/paper_a/prelim/f2_*_a1.json`).

| arm | feasible vs clean-label model | vs truth | nominal (clean / truth) | cost over nominal |
|---|---|---|---|---|
| (ii) truth = network | 0.90 | 0.90 | 0.30 / 0.30 | +3.5% |
| (i) truth = ODE | 1.00 | 0.90 | 0.30 / 0.20 | +3.5% |

- Multistart separation; every run converged within 20 master solves (mean 4.2
  and 7.4 iterations). The a posteriori attack finds a residual violation at 3 of
  10 decisions, by 0.04 to 0.06σ.
- One-step separation reaches 0.8 to 0.9 at +2.3 to 2.8%, but the attack finds a
  violation at 9 of 10 of its decisions: an ablation, not the method.
- n = 10, so the standard error is about 0.1; the radius is understated by
  enumeration and the clean-label reference is noisy (2026-11-19 entry).

## 2026-11-25: F2 on 100 redraws: FO-CP over-delivers on both SS1 arms

Job 30384811; 100 redraws per SS1 arm (Ovalle et al.'s cost vector k with redraw k),
end-to-end (32, 32) network at α = 1, radius (c) per redraw by enumeration (mean
4.3σ and 4.5σ), τ = 0.05σ. 190 runs reported; 10 rerun in job 30530397 (below).
Details: `paper/paper_plan.md`, "F2 (job 30384811)".

| arm, separation | vs clean-label model | vs truth | nominal (clean / truth) | cost over nominal |
|---|---|---|---|---|
| (ii) truth = network, multistart | 0.98 | 0.98 | 0.47 / 0.37 | +4.4% |
| (i) truth = ODE, multistart | 0.99 | 0.97 | 0.31 / 0.24 | +4.7% |
| (ii), one step | 0.92 | 0.85 | | +3.2% |
| (i), one step | 0.94 | 0.79 | | +3.3% |

- **Above the 0.9 target, not at it.** Radius (c) is a quantile of a supremum
  over the domain, conservative like the RC's simulated radius (also 1.00 against
  the clean-label model in job 29141177), even though enumeration understates it.
- **No bias failure against the ODE** (0.97), unlike the last-layer RC on the
  frozen network (0.52 at +2.2%); cubic-ridge RC: 0.95 at +4.2%. The frozen-network
  comparison will separate the end-to-end fit from FO-CP's larger margin.
- **One-step separation** meets 0.9 against the clean-label model only because the
  radius is conservative: it falls to 0.79 to 0.85 against the truth, and the
  attack finds a violation inside the ball at 87 to 90% of its decisions.
- **The attack does not predict violations**: none of the 62 multistart decisions
  it flags violates the clean-label model, and it cleared the 3 that do.
- **Clean-label reference (open question):** at the final decisions, `T(y*)` and a
  cold-start clean fit differ by a median 0.1σ against 1.1σ of slack, and disagree
  on feasibility in 4 of 190 runs. Here the choice does not move the result.
- 188 of 190 converged within 20 master solves (mean 6); median 22 minutes per run.
- **Reproducibility:** a requeued task refits θ₀ on another node, and the fit
  differs across nodes, so 9 runs mixed two nominal networks (plus task 86, which
  crashed on resume). They are rerun with the fits stored per redraw; including
  them moves no rate by more than 0.01.
- **Next:** baselines on the end-to-end network (margin, wrapper, C-MICL) and the
  frozen-network comparison, on the same redraws and cost vectors.

## 2026-11-27: F2 baselines: FO-CP matches the tuned margin, beats it against the ODE

Jobs 30532214 (baselines, 200 tasks) and 30530397 (the 10 F2 reruns). Data: SS1,
100 redraws per arm, Ovalle et al.'s 100 cost vectors, σ = 2, 1,000 training rows;
every method sees the same noisy labels and cost vector and is judged against the
same clean-label model `T(y*)` and the arm's truth. F2 on all 200 runs moves no
rate by more than 0.01 from the 2026-11-25 entry. Details: `paper/paper_plan.md`,
"F2 baselines".

| method | (ii) truth net: clean / truth / cost | (i) ODE: clean / truth / cost |
|---|---|---|
| FO-CP, multistart (no tuning) | 0.98 / 0.98 / +4.7% | 0.99 / 0.96 / +4.8% |
| margin tuned on the truth to FO-CP's level | 0.98 / +5.0% | 0.96 / +8.1% |
| split conformal, α = 0.1 | 1.00 / 1.00 / +6.3% | 0.99 / 0.89 / +6.6% |
| C-MICL, α = 0.1 (best α no better) | 0.75 / 0.72 / +3.9% | 0.72 / 0.67 / +3.9% |
| wrapper, all 20 bootstrap models | 1.00 / 0.97 / +3.5% | 0.98 / 0.90 / +3.5% |
| frozen network, RC at radius (c) | own 1.00 / 0.75 / +2.3% | own 1.00 / 0.47 / +1.8% |

- **H4 holds so far.** At equal feasibility against the truth, FO-CP costs the
  same as the oracle-tuned margin on arm (ii) and 3.3 points less on the ODE arm.
  Against the clean-label model the tuned margin is 0.4 points cheaper. A margin
  that meets the clean-label model on the ODE arm reaches only 0.76 against the
  ODE: FO-CP's x-dependent margin keeps decisions out of regions where the model
  errs (our reading; untested).
- **C-MICL never reaches 0.9**: lowering α widens q, but the optimizer moves to
  where the width net is small (median u(x*) 1.0 → 0.26).
- **Split conformal** covers a fresh noisy label, so it pays for σ on top of model
  error: conservative for a constraint on the noise-free flow.
- **Frozen network**: the RC holds exactly against its own clean-label model
  (Theorem 1) and FO-CP reproduces it, but its biased clean-label model leaves it
  at 0.47 to 0.75 against the truth.
- **Caveats.** The wrapper with a > 0 hit the 120 s MIP limit in 8 to 70% of runs,
  so those rows are not optima (a = 0 is). FO-CP and the baselines share θ₀ in only
  43 and 60 of 100 redraws (θ₀ had to be refit), so the comparison is across
  redraws, not paired. FO-CP has one point per arm; comparing at exactly 0.9 needs
  it over a radius grid.
- **Next:** wrapper rerun with a longer limit; FO-CP on a radius grid (frontier).
