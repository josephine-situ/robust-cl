# Paper A plan

Revised 2026-11-16 (after the meeting with DB; the new section below comes
first). Before that, 2026-11-14 (late, after the first solver test and a
narrowing of scope). Outline: `paper_outline.md` (version 4). Proofs:
`paper_a_guarantee.md` (its Sections 5 and 6 are superseded by this file).
Results: the go/no-go probe (job 29141177, under "Tonight") and the earlier
local checks under "Results so far". No baseline has been run on the paper's
design.

## After the meeting with DB (2026-11-16)

**What DB said.**

- Still exploratory. Work on the outline and this plan, not on `paper.tex`.
- We went through C-MICL and the old (random-bank) cutting planes in detail.
- DB is reasonably happy with the robust counterpart (RC). It is parametric:
  it needs the prediction to be affine in the training labels,
  `f(x; θ(y)) = s(x)ᵀy + c(x)`. He also wants a non-parametric method.
- His proposal: a new cutting-plane method whose separation problem is solved
  by first-order methods (for example in PyTorch), with the loss in his sketch
  (`c:\Users\jsitu\Downloads\robust-cl-cp.jpeg`, transcribed below). The
  `max(·, 0)` is a ReLU, and each constraint has its own weight so that
  infeasibility can be handled.
- Semi-synthetic data first. Real data later.

**The method as sketched (FO-CP, working name).** Data `(xᵢ, ȳᵢ)`. Write
`ĥₖ(x; y)` for constraint k's model trained on labels `y`, so `ĥₖ(x; ȳ)` is
the nominal model. The robust constraint is `ĥₖ(x; ȳ + Δy) ≤ bₖ` for every
`‖Δy‖ ≤ R`: the model is **retrained** on each relabeling, so nothing is
assumed about how predictions depend on labels.

1. Solve the nominal problem (`Δy = 0`). Call the solution `x^t`.
2. **Separation, by first-order methods.**
   `Δy* = argmax_{‖Δy‖ ≤ R} Σₖ wₖ max(ĥₖ(x^t; ȳ + Δy) − bₖ, 0)`.
3. **Master.** Add the cut `ĥₖ(x; ȳ + Δy*) ≤ bₖ`, an embedded copy of the
   model retrained on `ȳ + Δy*`, and re-solve `min obj` over all cuts so far
   to get `x^{t+1}`.
4. Stop when the separation value is at most τ.

**Settled with the user (2026-11-16).**

- **Sign.** The violation `ĥₖ − bₖ`. (The sketch writes `bₖ − ĥₖ`, which
  as a maximization would find the most favourable relabeling.)
- **The set is the RC's.** `Δy` stacks every outcome's labels, and the norm
  is the RC's ellipsoid, `‖Σ^{-1/2}Δy‖ ≤ R`, with Σ the noise covariance. So
  FO-CP and the RC protect against the same uncertainty set.

**What follows for the weights (my derivation; to put to DB).**

- **They do not move the end point.** Each robust constraint must hold for
  every `Δy` in the set. With exact separation, a maximum of 0 for the
  weighted hinge sum means every hinge is 0 at every `Δy`, so every
  constraint is robust-feasible, whatever the positive weights. The weights
  change which cut is found, not the robust feasible set where the loop
  stops. So they cannot make an infeasible robust problem feasible.
- **Consistent with the RC.** Constraint k's worst case over the joint
  ellipsoid uses only its own block `Σₖₖ` (the ellipsoid's projection), which
  gives `fₖ + R‖Σₖₖ^{1/2} sₖ(x)‖ ≤ bₖ`, the RC as written. Correlation across
  outcomes does not change the RC. It enters only the joint radius
  (outline, Section 3, several constraints).
- **With one model per outcome, the sum gives shallower cuts.** `ĥₖ`
  depends only on `Δyₖ`, so the separation splits into K problems, each with
  the full budget. The weighted sum finds a compromise direction that
  violates several constraints at once, each by less than its own worst case.
  One cut per violated constraint per iteration, as the old incoherent CP
  did, is deeper. **Decided (2026-11-16): use per-constraint cuts; no
  experiment compares the two.** The joint objective is needed only when one
  model is trained on several outcomes (a multi-output network), because
  then `Δyⱼ` moves `ĥₖ`.
- **Infeasibility: two options, both with a stated level.**
  - The RC's certified fallback carried over: the largest R at which the
    robust problem is feasible (bisection), reported as the decision's level.
  - Protect a fraction of the constraints (from the user), below.
  - *Dropped (2026-11-16): the soft master*, which moves the hinges into the
    master's objective as penalties, `min obj + Σₖ wₖ Σⱼ max(ĥₖ(x; ȳ + Δyʲ) − bₖ, 0)`.
    It is never infeasible, but it has no guarantee and the weights become
    one more dial to tune.
- **Protect a fraction of the constraints (user, 2026-11-16).** Binary `zₖ` per constraint (constraints, not cohorts). The
  robust cuts for k hold only if `zₖ = 1` (big-M), the nominal constraint
  always holds, and `Σₖ zₖ ≥ ⌈pK⌉`. The separation uses `wₖ = zₖ`. Unlike
  the weights, this moves the end point: the optimizer chooses which
  constraints it can afford to protect. Guarantee: Theorem 1's bound is
  uniform over the domain, so the data-driven choice of the protected set
  costs nothing, and P(a protected constraint is violated) ≤ P(maxₖ Wₖ > R)
  **[to prove: short]**. Unprotected constraints get only their certified
  level `r̂ₖ(x*)`. Compare with the fallback (shrink R for all) and with an
  unequal split of ε (`Σ εₖ ≤ ε`, which keeps a joint guarantee but cannot
  drop a constraint). RO precedents to check: light robustness (Fischetti and
  Monaci 2009), the globalized robust counterpart (Ben-Tal, Boyd and
  Nemirovski 2006); the wrapper's α is the same idea over bootstrap models.

**How it relates to what we have.**

| | Old CP (random bank) | RC (current paper) | FO-CP (new) |
|---|---|---|---|
| Model | any, retrained per relabeling | affine in labels (ridge, last-layer network, trees with fixed splits) | any whose training is differentiable in the labels (networks trained end to end, GLMs including logistic) |
| Separation | best of B = 200 sampled relabelings | closed form, `Δy = R s(x)/‖s(x)‖` | gradient ascent on `Δy` through the training map |
| Separation exact? | no (inner sample) | yes | no (local search) |
| Stopping certifies `x*`? | no | yes (T3) | no, unless the map is affine |
| Guarantee | none | Theorem 1 | open (below) |

- **The RC is FO-CP's closed-form case.** For an affine map with one
  constraint, the hinge's gradient is `s(x)`, and one normalized ascent step
  lands on `R s(x)/‖s(x)‖`, the RC's directed cut. So FO-CP on ridge must
  reproduce the RC's decision exactly. That is the correctness test (F0).
- **The first gradient is a linearized RC.** By implicit differentiation at
  the trained weights, `∇_Δy ĥ(x) = J H⁻¹ ∇_θ f(x; θ̂)`, with `J` the Jacobian
  of the training predictions and `H` the training loss's Hessian. With the
  Gauss–Newton `H = JᵀJ + λI` this is ridge's `s(x)` on the network's gradient
  features (the influence function of Koh and Liang). So one step from
  `Δy = 0` is the RC on a linearized network. What the non-parametric method
  adds is everything beyond that step: retraining and ascending again where
  the map bends. Probe F1 measures how much that is.
- **The old bank becomes an ablation of the separation**, on the same master:
  random sampling against gradient ascent, at the same `R`.
- **It removes the honest-fitting problem for networks.** Probe R3 (hidden
  layers trained on each redraw's own labels) broke the last-layer RC's
  guarantee: 0.76 against the clean-label model at the simulated radius,
  target 0.90. FO-CP retrains the whole network on every relabeling, so the
  standard fit is the method's own model, every row is used, and the R4
  sample split is not needed.

**What has to be defined before anything runs.**

- **The training map.** Network training is nonconvex, so `y ↦ θ(y)` is not
  unique. Proposal: full-batch, deterministic training, warm-started from the
  nominal weights `θ̂`. The clean-label reference is the same map applied to
  `y*`. Also measure from-scratch training on `y*`, and retraining on `ȳ`
  with only the seed changed: if the seed alone moves `ĥ(x*)` as much as the
  label noise does, the label ball is not modelling the main uncertainty.
- **The gradient.** (G1) Implicit differentiation at the retrained optimum.
  The reactor's (32, 32) network has 1,281 weights, so its Hessian can be
  formed exactly. (G2) Unrolled differentiation through T warm-started
  training steps (back-gradient). G1 first.
- **The ascent (added 2026-11-16).** The set of relabelings is infinite but
  finite-dimensional: `Δy` has one entry per training label (n = 1,000 on
  the reactor), and the retrained prediction at `x^t` is a smooth function of
  it. So the separation is a continuous optimization, not an enumeration.
  - With per-constraint cuts and one model per outcome, constraint k's
    separation is `max ĥₖ(x^t; ȳ + Δyₖ)` over `‖Σₖₖ^{-1/2}Δyₖ‖ ≤ R`: the
    retrained prediction itself, with no hinge, so no zero gradient. A cut
    is added if the maximum exceeds `bₖ + τ`.
  - Gradient by G1: retrain warm-started on `ȳ + Δy` to get `θ`, then
    `∇_Δy ĥ(x^t) = J H⁻¹ ∇_θ f(x^t; θ)` (squared loss; `J` stacks the
    training rows' `∇_θ f(xᵢ; θ)`). One linear solve per step, whatever n is.
  - Step in whitened coordinates `z = Σ^{-1/2}Δy`: `g̃ = Σ^{1/2}∇`, then
    `z ← R g̃/‖g̃‖` (the linear maximizer over the ball, exact in one step for
    an affine map), or a projected step `z ← Proj_{‖z‖≤R}(z + η g̃)` if that
    oscillates. Stop when `ĥ` stops increasing.
  - Cost per step: one warm-started retraining plus one solve (seconds on the
    reactor). Starts × steps retrainings per separation, per violated
    constraint.
- **The norm.** The RC's ellipsoid `‖Σ^{-1/2}Δy‖`, jointly over outcomes
  (settled above). Σ is diagonal with σᵢ² per row where noise levels differ
  (chemotherapy's binomial σ per arm). Cross-outcome covariance is open for
  chemotherapy, where toxicities in one arm are correlated.
- **Search effort.** Multi-start: from `Δy = 0`, from the linearized
  direction, and from random points on the sphere. Each candidate is checked
  by full retraining before it becomes a cut.

**The guarantee is the open question.** Theorem 1 uses that the error of an
affine model is `s(x)ᵀe`. For a retrained nonlinear model, three options:

| Option | Radius | What it guarantees | Catch |
|---|---|---|---|
| (a) Model-free ball | `√χ²(nK, 1−ε)` in the whitened norm: the norm of the whole noise vector over all K outcomes (`σ√χ²(n, 1−ε)` for one constraint) | If separation were exact, the clean-label model is feasible at `x*` with probability ≥ 1−ε, for **any** training map. Short proof. | Separation is local. And the radius protects every direction in label space: about 32σ at n = 1,000, against the RC's 8.4σ closed-form and 3.6σ simulated radii on cubic ridge. On ridge it would make the margin about 9 times the simulated one. |
| (b) Linearized subspace | `σ√χ²(d_eff, 1−ε)`, with `Δy` restricted to the span of the linearized map | Exact for affine maps (it is the RC's subspace radius); approximate otherwise | Valid only to first order |
| (c) Scenarios | Sample noise vectors, retrain on each, and size `R` so the robust set covers a 1−ε fraction of them. Or impose the N retrained models directly (scenario optimization, Calafiore and Campi; Campi and Garatti) | A bound for a fresh **resampled** model | That is a parametric bootstrap, not the clean-label model. With integer decisions it needs the non-convex scenario theory. The old random bank was this, without the theory. |

Recommendation: state (a) as the exact result. Test in F2 how conservative
it is, and use (c) as the practical radius if (a) overpays badly. This is
DB's call: does the non-parametric method need a finite-sample guarantee, or
is an approximate one acceptable?

**Semi-synthetic instances (DB: first).** Real covariates, a known truth
`h*`, labels `h*(xᵢ) + noise` at a chosen σ, so every decision is judged
exactly.

| # | Instance | Truth | Why |
|---|---|---|---|
| SS1 | Reactor, Ovalle et al.'s design and 100 cost vectors, σ = 2 | (i) the ODE; (ii) a realizable truth: a (32, 32) network fitted to noise-free ODE labels | (i) as in E1. (ii) puts the truth inside the model class, so the model has no bias. The network RC missed against the ODE because of bias (bias at `x*` negative in 80 to 90% of runs, job 29141177), so (ii) is where FO-CP's label-noise protection can be tested by itself. |
| SS2 | Chemotherapy: the 416 real arms' covariates and treatments, five toxicity constraints, survival objective | One fixed model per outcome, fitted on the real labels, then treated as the truth. Noise σᵢ binomial from the real arm sizes. | Gives the several-constraints experiment with a ground truth that the plan lacked (see "Scope"), and is where the weights and infeasibility handling are tested. |

SS1 first: the ODE, the cost draws, the redraw harness (`probe_rc_ridge.py`)
and the baselines exist. SS2 needs a generator. Real chemotherapy data (E2)
comes after.

**Candidate instances beyond these (proposed 2026-11-16, not decided).** What
the plan lacks is several learned constraints against an exact truth, and a
constraint or model that couples several outcomes (where the joint set's
shape and cross-outcome covariance matter).

| Candidate | What it adds | Truth | Cost to build | Verdict |
|---|---|---|---|---|
| **SS1+: reactor with more ODE outputs as constraints** | Several constraints with an exact truth. The ODE already integrates eight flows (tube and shell CH4, C2H6, H2, C6H6, `src/data/dma_mr.py`), so e.g. a methane-conversion floor or an ethane cap is one more output of the same solve. Fit them separately or as one multi-output network (shared training: the case where the joint separation is needed) | ODE | Low: same harness | **Recommended first.** The constraints are ours, not Ovalle et al.'s; say so |
| SS2: chemotherapy with DLT as a function of the four toxicities | `1 − Πₖ(1 − ĥₖ(x)) ≤ b`: several models in one constraint, from the literature's own instance | fitted, fixed | Medium (I14) | Already planned; add the DLT variant |
| Student enrollment (JANOS, Bergman et al.; Gurobi ML example; OptiCL compares against it) | Logistic model: not affine in labels, so FO-CP's case; a constraint summing predictions over many students | a fixed logistic model, Bernoulli labels: noise known exactly | Medium; data are synthetic in the public example | Good FO-CP showcase, but classification, which the outline keeps to a paragraph |
| Autothermal reformer (IDAES/OMLT example) | Multi-output network trained on 2,800 simulated points; maximize H2 s.t. an N2 concentration cap (GAMSPy version) | IDAES flowsheet | High: idaes-pse and Ipopt; only two decision variables | Not now |
| Power-system security rules (Cremer et al. 2019 and successors) | The literature's clearest at-optimum failure | power-flow simulator | High; classification | Motivation only |
| WFP (E3) | Second literature case | palatability simulator | High (I8) | As planned |

**Probes, in order.**

| # | Question | Design | Decides |
|---|---|---|---|
| F0 | Does FO-CP reproduce the RC where the RC is exact? | Cubic ridge and the last-layer network on one reactor redraw. Gradient ascent's maximum against `R‖s(x)‖`; FO-CP's final decision against the RC's cone solution. Local, minutes. | The code is right. Gate for everything else. |
| F1 | Is the training map nonlinear enough for the non-parametric method to matter, and how strong is the attack? | (32, 32) network trained end to end, nominal `x*`, a few redraws, several R. Violation found by: the linearized step; FO multi-start with retraining; the random bank (B = 200) in the same ball; seed-only retraining. Cluster quick-test. | If multi-start ≈ the linearized step, FO-CP collapses to an RC on the linearized network (simpler, and it keeps a guarantee). If the seed-only change is comparable, training randomness, not label noise, dominates. |
| F2 | Does FO-CP deliver on SS1, and at what cost? | 100 cost vectors, noise redrawn. FO-CP at radii (a) and (c); the last-layer RC; nominal; margin, wrapper and C-MICL on the same network. Feasibility against the clean-label model (same training map on `y*`), the ODE, and the realizable truth; cost; cuts; time. | H1 and H2 for the non-parametric method |
| F3 | Several constraints and infeasibility | SS2, joint ellipsoid, per-constraint cuts. Where the robust problem is infeasible: the certified fallback against protecting a fraction p of the constraints | Infeasibility handling; H3 on a known truth |

**Theory tasks added.**

- **T7.** If separation is exact, termination gives robust feasibility (as T3).
  With local separation, state what stopping does and does not show. Also:
  over the joint ellipsoid, the weighted hinge sum has maximum 0 exactly when
  every constraint is robust-feasible, for any positive weights (short; see
  "What follows for the weights"). For affine maps with one model per
  outcome, each constraint's separation is closed form. The weighted sum
  itself is a convex function, so maximizing it is not a concave problem, but
  it can be solved exactly by enumerating which constraints are violated
  (2^K − 1 subsets; K = 5 on chemotherapy). That is an exact checker for F3
  on ridge.
- **T8.** Guarantee options (a) to (c) above. (a) is short. (c) needs either
  a linearization argument or scenario theory with the bootstrap caveat.

**Implementation tasks added.**

- **I11.** PyTorch as a dependency (not installed; check wheels for Python
  3.14, else JAX). The training map: deterministic, full-batch, warm-started.
  Export the trained weights to the existing MLP embedding in
  `src/models/embed.py`.
- **I12.** The separation: projected gradient ascent with G1, multi-start, a
  retraining check of each candidate.
- **I13.** The master loop. `cp.py`'s incremental master already embeds one
  retrained model per cut, so swap in the new separation there or in a slim
  copy. To check which is less work.
- **I14.** SS2 generator; the realizable-truth variant of SS1.

**Literature to add (to check; none read yet, none in `paper_a.bib`).** The
separation is a targeted label-poisoning attack with a norm budget, solved
as a bilevel problem: Biggio, Nelson and Laskov (2012); Mei and Zhu (2015);
Muñoz-González et al. (2017, back-gradient); Jagielski et al. (2018,
regression); Koh and Liang (2017, influence functions). The novelty has to be
argued against them. There the attack targets a prediction; here it targets
the optimizer's decision, and the cut goes into a MIP. Scenario optimization:
Calafiore and Campi (2006), Campi and Garatti (2008), Campi, Garatti and
Ramponi (2018, non-convex). Implicit differentiation: Blondel et al. (2022).

**New risks.**

- No certificate. Local separation can miss the worst relabeling, which is the
  same objection the plan makes to the random bank, only weaker.
- The exact radius (a) may be useless in practice (about 32σ at n = 1,000).
- Bias. FO-CP protects against label noise only, so on the end-to-end network
  it will miss against the ODE for the same reason the network RC did. SS1(ii)
  separates the two effects.
- The master grows by one embedded (32, 32) network per cut and constraint.
  One network solve already takes 15 to 100 seconds.
- The hinge has zero gradient at constraints with slack, and a kink at `Δy = 0`
  for a binding one. Starting from the linearized direction avoids both.

**Questions for DB.**

1. One paper with two methods (the RC as the exact case, FO-CP for the rest),
   or the RC paper first and FO-CP as its own paper? Recommendation: one
   paper, the RC presented as FO-CP's closed form.
2. What guarantee must the non-parametric method carry: exact (radius (a)),
   approximate (b, c), or none beyond experiments?
3. The weights. Under the RC's joint ellipsoid they change only which cut is
   found, not where the loop ends (above). We propose handling infeasibility
   instead with the certified fallback (largest feasible radius) or by
   protecting a fraction of the constraints. Is that what he had in mind?
4. Not answered on 2026-11-16 and still open: the reactor's lead model
   (network or cubic ridge, see "Proposed for DB" under "Tonight").

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
| H3 | **It certifies more of what it prescribes.** With five learned toxicity constraints on real data, the RC certifies a regimen at the target level for more cohorts than any baseline, gives every other cohort the regimen with the best certified level, and says which cohorts no regimen can be certified for. (Was: "prescribes for most cohorts"; changed 2026-11-14, see "Infeasibility on chemotherapy".) | E2 | a baseline's decisions certify at least as well as the RC's at the same level |

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
| Cutting planes over a random bank | Drew 200 relabelings from an ellipsoid of radius rho, retrained the model on each, cut the worst one at `x*`, stopped at tolerance tau. With several constraints, a cut that left a protected cohort infeasible was rolled back and rejected. | No guarantee: the bank is a finite sample of the set, so stopping shows `x*` is feasible for the drawn models only. Two parameters (rho, tau) with no rule tying either to a violation level; both were set against the ground truth, and the rho that works had to be re-measured per instance (rho = 1 never passed 0.30, rho = 2 delivered). The result also depends on the bank's seed. For a model affine in its labels the worst relabeling has a closed form, `R s(x)/‖s(x)‖`, so sampling is unnecessary. The rollback kept the master feasible by giving up robustness wherever it was unaffordable, so what was delivered had no stated level. | Out as a method. One sentence in Section 4: random-draw cuts carry no guarantee. **2026-11-16:** the cutting-plane loop returns with first-order separation (FO-CP, top of this file), and the random bank becomes its separation ablation. The rollback's role (prescribing when full robustness is infeasible) passes to the certified fallback; see "Infeasibility on chemotherapy". |
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
| E2 | **Six learned models, real data** | chemotherapy | Maragno et al.'s protocol. RC with the simulated joint radius over the test cohorts' treatment slices, and the certified fallback where it is infeasible; margin, wrapper, C-MICL, nominal, all on the same fixed models, each decision given its certified level | Table 2 (fraction of cohorts certified at each level, per method) | 6.3 |
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
- **A non-parametric method joins the RC** (DB, 2026-11-16): cutting planes
  with first-order separation (FO-CP). Its place in the paper and its
  guarantee are open. See the section at the top.
- **Semi-synthetic data first** (DB, 2026-11-16), real data later.
- **Infeasibility options to test include protecting a fraction of the
  constraints** (user, 2026-11-16), beside the certified fallback.
- **The objective stays nominal; a robust objective is a deferred variant**
  (user, 2026-11-16).

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
- **σ per arm, binomial, for the main result** (decided 2026-11-14):
  `σᵢ = √(pᵢ(1−pᵢ)/n_armᵢ)`, with the weighted norm. It is the label noise the
  theorem is about: the sampling error of an arm's toxicity rate. It uses the
  noisy `p`, so T2 has to cover the plug-in. The out-of-fold residual sd is a
  second row only. It mixes misfit into σ, which the theorem does not ask for,
  and it is likely to make the RC infeasible the way C-MICL is.
- **Objective.** Survival is a learned objective. The default is no margin on
  it, as for the other methods. **A robust objective (epigraph form,
  `max t` s.t. `ĥ_OS(x; ȳ + Δy) ≥ t` over the set) is a variant, deferred**
  (user, 2026-11-16). It would give a certified lower bound on clean-label
  survival. If it is ever run, the epigraph constraint is never droppable
  under the fraction-protected option, and the survival labels join the
  joint set.
- **What it can show.** The certified level of each method's prescriptions
  (the main measure), the fraction prescribed, toxicity limits met under the
  judge, and survival. The judge is itself fitted, so this experiment does not
  test the theorem. E1 does.

## Infeasibility on chemotherapy (decided 2026-11-14)

The instance has little headroom: nominal already scores 0.89 to 0.96, and
C-MICL at alpha = 0.1 prescribes for about 6% of cohorts. Five simultaneous
margins may leave the RC infeasible for many cohorts too. This section is how
the paper handles that. Nothing in it has been run.

**CP's rollback is not carried over.** With several constraints, the
random-bank cutting planes kept the master feasible by rolling back any cut
that left a protected cohort infeasible (`presentations/method.tex`, "Cutting
Planes: coverage"). That gives up robustness wherever robustness is
unaffordable, so what was delivered has no stated level. It stays in the
thesis.

**The RC may be much less infeasible than C-MICL.**
- C-MICL's width is a prediction interval for a new outcome. It covers noise
  plus misfit and does not shrink with n.
- The RC's margin `R σ‖s(x)‖` is a confidence band on the fitted model.
  `‖s(x)‖` is about √(leverage) for ridge and 1/√(n_leaf) for a tree, so it
  shrinks with n.
- Which of the two the chemo RC resembles is decided by the choice of σ (see
  "σ per arm" above). With the binomial σ the margin should be well below
  C-MICL's. This is a guess, not a measurement; probe P1 below measures it.
- The tension, which the reactor result (job 29141177) makes concrete: a
  label-noise margin does not see misfit. Under the binomial σ the RC may
  prescribe often and still miss against the judge where the models misfit.
  So Table 2 reports both the certified level and the judge's verdict.

**Three layers, in order.**

1. **Shrink the radius, keeping the guarantee.**
   - The joint simulated radius instead of Bonferroni (T1).
   - Take W's supremum over the decision slices actually prescribed for, not
     the whole box. A cohort's context is fixed, so its decision space is the
     treatment slice. Over the union of the 96 test cohorts' slices, the
     radius holds for all cohorts at once and costs about one MIP per draw
     (a selection variable picks the cohort). Per cohort it is smaller still,
     but costs 96 times the solves. Use the union for the main result.
   - The per-arm weighted norm (above).
   - An unequal split of ε across the five constraints if one dominates.
     This is still a guarantee (weighted Bonferroni, or the joint radius with
     per-constraint scaling).
2. **The certified fallback.** For any decision `x` define the certified
   radius `r̂(x) = minⱼ (bⱼ − f̂ⱼ(x)) / ‖Σⱼ^½ sⱼ(x)‖`, in each constraint's
   standardized units. `Σⱼ` is the diagonal of the per-arm noise variances
   for outcome j; with a single σⱼ the denominator is `σⱼ‖sⱼ(x)‖`. Theorem 1's proof gives violation ⇒ `W > r̂(x*)`, so
   `P(violation and r̂(x*) ≥ r) ≤ P(W > r)` for every r, however `x*` was
   chosen, including by maximizing `r̂` (T6). When no regimen meets the target
   radius for a cohort:
   - solve `max t` s.t. `f̂ⱼ(x) + t·‖Σⱼ^½ sⱼ(x)‖ ≤ bⱼ` for all j, by
     bisection on t over feasibility problems (each is the RC at radius t);
   - prescribe the result and report `ε̂ = P(W > r̂)`, read off the simulated
     draws of W that the radius already needs.

   Robustness degrades gracefully, as CP's rollback did, but every
   prescription carries its own level. The statement is joint, not
   conditional on reaching `r̂`, and the paper must word it that way.
3. **Abstention as an outcome.** A cohort with no certifiable regimen at any
   useful level is reported as such: the data cannot support a safe regimen
   there. C-MICL's 94% non-prescription is read the same way.

**Table 2 changes.** `r̂(x)` can be computed for any method's decision, so
every method is put on one scale. Columns:
- the fraction of cohorts certified at ε ∈ {0.05, 0.1, 0.2};
- the fraction prescribed at all;
- toxicity limits met under the judge;
- predicted survival;
- the worst subsample beside each mean.

This also sidesteps the judge's weak separation: the certified level does not
depend on the judge. The judge's verdict stays beside it as the check against
misfit.

**Probe P1, first thing after the meeting (local, minutes; not yet run).**
- Fit ridge on the training arms for the five toxicities.
- For each of the 96 test cohorts, solve the max-t problem.
- Report the fraction whose `r̂` reaches the joint radius at ε = 0.1, under
  the binomial σ and under the out-of-fold residual sd. For the radius, use
  enumeration, which is good enough for a go/no-go.

This decides H3 before I6 and I7 are built. If even the binomial σ certifies
few cohorts, E2 is reframed as a demonstration of the fallback and of
abstention, and H3 is weakened to the comparison of certified levels.

## Theory tasks

| # | Task | Status | Needed for |
|---|---|---|---|
| T1 | Simulated radius: the ⌈(N+1)(1−ε)⌉-th smallest of N simulated `W` gives level ε by exchangeability, with each `W` replaced by the solver's upper bound; joint version for several constraints | drafted 2026-11-15 (Prop. simulated radius; the joint version is held in paper_a_guarantee_held.tex until P1); not checked by JS | E1, E2 |
| T2 | Estimated σ: upper confidence bound and its effect on ε; unequal known `σᵢ`; the binomial plug-in `σᵢ` from noisy `pᵢ`. Alternative route: simulate `σ̂/σ` jointly with the noise, as Liu et al. (2005) do for their critical constant | to prove | the recipe in Section 2; E2 |
| T3 | Directed cuts terminate in at most K iterations | drafted 2026-11-15 (paper Sec. 4, proof in appendix); not checked by JS | Section 4; E2 |
| T4 | Sub-Gaussian version of the subspace radius | to prove | appendix |
| T5 | Lipschitz bias margin | to prove | appendix, optional |
| T6 | Certified level of any decision: `P(violation and r̂(x*) ≥ r) ≤ P(W > r)` for all r, `x*` chosen in any way; the joint version over several constraints | drafted 2026-11-15, held until P1 in paper_a_guarantee_held.tex; not checked by JS | Section 3; E2 fallback and Table 2 |

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
| I9 | Certified radius `r̂(x)` for any decision; the max-t fallback by bisection; `ε̂` from the stored W draws | to build | E2, Table 2 |
| I10 | Probe P1: ridge on the gastric training arms, max-t per test cohort, two choices of σ | to build; local, minutes; **not before the meeting** | H3 go/no-go |

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
4. **I10 (probe P1) first: does the RC certify chemo cohorts?** Then T2, T3,
   T6. I6, I7, I9, then E2. Draft Sections 4 and 6.3.
5. **Appendix tables A1 to A3.**
6. **I8 and E3.**

Step 1 and a small version of step 2 are tonight's runs (top of this file).

**Added 2026-11-16, alongside the RC track:** I11 and I12, then F0 (local),
then F1 (go or no-go: does the non-parametric method add anything over a
linearized RC?), then I13 and I14, F2 on SS1, and F3 on SS2. The real-data E2
comes after F3.

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
- **Five margins make the RC infeasible on chemotherapy.** Mitigated by the
  binomial σ, the radius over the prescribed slices and the certified
  fallback ("Infeasibility on chemotherapy"); P1 measures how much is left.
- **Under the binomial σ the RC prescribes but misses against the judge**
  where the models misfit, as the network did against the ODE. Report the
  judge's verdict beside the certified level.
- **The chemotherapy judge cannot separate methods.** The audit put the 0.9
  target inside the judge's uncertainty band for most cells. Report orderings
  and the fraction prescribed there, and keep feasibility levels to the reactor.
- **WFP is a build from nothing.**
