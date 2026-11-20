# Paper A plan

Revised 2026-11-19. Method reference: `paper_outline.md` (v4). Proofs:
`paper_a_guarantee.md` (its Sections 5 and 6 are superseded here). Literature:
`paper_a_literature.md`. Dated results: `notes/progress.md`. No baseline has
been run on the paper's design.

## Decisions

Since the meeting with DB (2026-11-16):

- Exploratory: edit the outline and this plan, not `paper.tex`.
- One paper, two methods. The RC covers models affine in their labels. FO-CP
  (cutting planes with first-order separation; DB's sketch is
  `c:\Users\jsitu\Downloads\robust-cl-cp.jpeg`) covers any model trained by a
  differentiable loss. The paper does not call FO-CP non-parametric, since it
  needs a differentiable training map. The two methods share the constraint
  form, not the radius argument.
- Semi-synthetic instances first, real data last, taken from instances used
  in the optimization literature rather than new constraints on the reactor.
  Order: reactor (SS1); WFP as regression with its closed-form truth; SS2,
  semi-synthetic chemotherapy (RC only); real chemotherapy. Workload
  dispatching is optional. Classification comes after regression.
- The objective stays nominal; a robust objective (epigraph) is deferred.
- Every part of FO-CP is prior work (bilevel label poisoning, influence
  functions, cutting sets with local oracles); the combination was not found
  (`paper_a_literature.md`, "FO-CP, robust-optimization machinery and
  instances"). The two closest papers, Kinsey et al. 2022 and Zhang, Amin and
  Perakis 2026, were read in full on 2026-11-19: both learn the objective over
  a known feasible set, with no robust constraint.

From 2026-11-14 and earlier, still in force:

- Experiments lead, at about 40% of the paper. Theory is one section, with
  proofs in the appendix.
- The paper stands alone; C-MICL is one of three baselines, not the frame.
- Each problem's model (class, architecture, hyperparameters) is fixed before
  the pipeline runs, as Ovalle et al. do.
- Ovalle et al.'s 100 cost vectors, with the noise redrawn for each. Their
  feasibility rate is over cost vectors, our probability over the noise.
- One model per comparison: each baseline embeds the same model as the
  method it is compared with.
- The guarantee is against the clean-label model. The paper says so in a
  remark, measures the gap to the truth, and gives a Lipschitz margin in an
  appendix if T5 is proved.
- The RC's headline radius is the simulated one (T1).
- Means with standard errors, no formal tests. Feasibility is over solved
  runs, with the fraction solved beside it.

Open: the reactor's lead model, cubic ridge or the network (Results of job
29141177).

## What the paper has to show

H1 to H3 are the RC's, H4 is FO-CP's. The last column is our own go/no-go
and does not appear in the paper.

| # | Claim | Experiment | No-go |
|---|---|---|---|
| H1 | At its simulated radius for ε = 0.1, the RC reaches 0.9 ground-truth feasibility with no tuning, where the nominal decision does not | E1 | nominal already near 0.9, or the RC below it |
| H2 | At equal ground-truth feasibility, the RC has the best objective among the constant margin, the wrapper and C-MICL | E1 | another method is cheaper at 0.9 |
| H3 | With five learned toxicity constraints, the RC certifies a regimen at the target level for more cohorts than any baseline, gives every other cohort its best-certified regimen (or protects a fraction of the constraints), and names the cohorts no regimen can be certified for. SS2 first (P2), then real data (E2) | P2, E2 | a baseline certifies at least as well at the same level |
| H4 | For a model trained end to end, FO-CP reaches the target feasibility against the clean-label model, where the last-layer RC under standard fitting does not (R3: 0.76), at a cost comparable to the baselines' | F2 | radius (c) cannot be computed, or ascent adds nothing beyond the linearized step (F1) |

How the margins differ:

| Method | Varies with `x` | Source of the margin | Models embedded | Rows used to fit |
|---|---|---|---|---|
| Constant margin | no | chosen by the user | 1 | all |
| Wrapper | implicitly | disagreement of P bootstrapped models | P | all (resampled) |
| C-MICL | yes | a second model fitted to the first one's residuals, scaled by a conformal quantile | 2 | 80% |
| RC | yes | computed from the fitted model and training inputs | 1 | all |

C-MICL's margin also varies with x. Ours is computed rather than learned,
holds at `x*`, and needs one model. C-MICL's width can pick up model misfit,
which `‖s(x)‖` cannot.

## FO-CP

### Uncertainty set

One ellipsoid per outcome, as in the RC: `‖Σₖₖ^{-1/2}Δyₖ‖ ≤ Rₖ`, with one
entry of `Δyₖ` per training row of outcome k and `σᵢₖ²` on the diagonal (the
RC is constraint-wise: Ben-Tal, El Ghaoui and Nemirovski 2009, Sec. 1.2). At
equal R this gives the same robust constraints as one ellipsoid over all
outcomes, and the model-free radius is smaller. A constraint containing
several outcomes' models gets one set over those outcomes' labels.

### Algorithm (settled 2026-11-19)

`T(y)` is the training map, `ĥₖ(x; y)` the prediction of `T(y)`, `Cₖ` the
stored cut models for constraint k, and `zₖ ∈ {0, 1}` whether k is protected.

0. Compute the radius R before the loop (Radius, option (c)).
1. Set `Cₖ = ∅`. The first master solve is then the nominal problem.
2. Master: `min cᵀx` s.t. `ĥₖ(x; ȳ) ≤ bₖ` for all k; `ĥ(x; θ) ≤ bₖ − τ`
   for every `θ ∈ Cₖ`, enforced only if `zₖ = 1` (big-M);
   `Σₖ zₖ ≥ ⌈pK⌉`; `x ∈ X`.
3. Separation at `xᵗ`, for each k with `zₖ = 1`: maximize `ĥₖ(xᵗ; ȳ + Δy)`
   over `‖Σₖₖ^{-1/2}Δy‖ ≤ R` by projected gradient ascent (Separation). If
   the maximum exceeds `bₖ`, add `T(ȳ + Δy*)` to `Cₖ`, at most one cut per
   constraint per iteration. Cuts are per constraint rather than DB's
   weighted ReLU sum, because per-constraint cuts are deeper and have no zero
   gradient.
4. Cuts accumulate and are never removed. A constraint the master stops
   protecting keeps its cuts, switched off by big-M. Removing cuts would
   allow cycling, and the convergence argument for discretization
   (Blankenship and Falk 1976; to check) needs a growing cut set. The master
   grows by one embedded network per cut.
5. Stop when an iteration adds no cut, or after 20 master solves (reported
   as not converged). Then attack the final decision a posteriori.

τ uses restriction of the right-hand side (Mitsos 2011; to check): cuts are
enforced at `bₖ − τ` and separation tests against `bₖ`, so exact separation
would certify robust feasibility at `bₖ` after finitely many cuts, given a
point strictly feasible by τ. τ = 0.05σ to start; F0(iii) takes τ → 0. τ
must exceed the noise in evaluating a retrained prediction (F0 measures it);
Gurobi's tolerances are far smaller. The separation is local, so stopping
certifies nothing by itself (Mutapcic and Boyd 2009).

The a posteriori attack uses the zero start plus 20 random starts, up to 50
accepted steps each, with the step length halved down to R/256. It reports,
in σ units, the violation gap `(max ĥₖ − bₖ)/σ` and the separation gap (the
attack's maximum minus the loop's last maximum at the same x).

Protected set (Gorissen et al. 2015, "K out of N", plus the nominal rows):
whichever constraints the master protects, a protected constraint is
violated only if the noise leaves its set (`Wₖ > Rₖ` for the RC's radius,
`‖Σₖₖ^{-1/2}eₖ‖ > Rₖ` for the model-free one). [to prove: short]

### Training map

- Every fit starts from one fixed `θ₀` (the nominal fit), so `T` is a
  function of the labels.
- scipy L-BFGS-B on the exact sklearn objective (`probe_focp.py`), stopped
  by the relative change in the loss (scipy's default ftol), not a gradient
  tolerance: a ReLU network's minimizer sits on kinks, where max|grad|
  stalls near 1e-3 while the loss is flat to 8 digits (F0, 2026-11-19). A
  fit that hits the iteration cap (20,000) is discarded. The map is
  deterministic, but not exact: on the repo's α = 0.01 network, retraining
  on the same labels moves h(x*) by 0.04 flow units (0.002 at α = 1).
- Labels are standardized with constants fixed from `ȳ`. The probe
  standardizes each label vector by its own mean and sd
  (`probe_rc_ridge.py:106`), which is not affine.
- The clean-label reference is `T(y*)`; the cold-start fit on `y*` used in
  R1 to R4 is reported beside it.

### Separation

- Gradient by implicit differentiation,
  `∇_Δy ĥ(xᵗ) = J (H + λI)⁻¹ ∇_θ f(xᵗ; θ)`, one linear solve. Dead ReLUs
  make H singular, hence the damping: start at `λ = 10⁻⁴ tr(H)/p`
  (p weights); F0(ii) keeps the smallest λ in `{10⁻⁶, …, 10⁻³} × tr(H)/p`
  at which the implicit and finite-difference gradients agree (cosine
  ≥ 0.99, norms within 5%).
- Whitened coordinates `z = Σₖₖ^{-1/2}Δy` turn the set into the ball
  `‖z‖ ≤ R`, with gradient `g̃ = Σₖₖ^{1/2} g`.
- Step: try `z′ = Proj_R(z + ℓ g̃/‖g̃‖)` with ℓ = ∞ (the jump to
  `R g̃/‖g̃‖`), then ℓ = R, R/2, …, R/32. Accept the first z′ whose retrained
  prediction exceeds the current one by at least 0.001σ; if none does, the
  start ends. At most 20 accepted steps per start.
- Starts: zero, whose first step is the linearized direction; the stored
  relabeling of k's tightest cut at `xᵗ`, if `Cₖ` is not empty; four random
  points on the sphere `‖z‖ = R`.
- Every iterate is a retrained model, so the cut candidate needs no separate
  confirmation.

For a model affine in its labels the gradient is `s(x)` and the first step
gives `R s(x)/‖s(x)‖`, the RC's cut. On a network the first step is the RC
on the linearized network (Koh and Liang's influence). Retraining the whole
network avoids the honest-fitting problem that broke the last-layer RC under
standard fitting (R3: 0.76 against 0.90). The old random bank becomes an
ablation of the separation (F1).

Cost: starts × steps × (retraining + one linear solve), per protected
constraint, per iteration. Separations depend on `xᵗ`, hence on the cost
vector, so they are not shared across the 100 cost vectors. Each cut on the end-to-end network adds a ReLU embedding
(64 binaries for the (32, 32) network) to a master whose single solve
already takes 15 to 100 seconds. The F2 pilot measures this.

### Radius

Theorem 1 does not carry over. The RC's simulated radius (3.6σ on cubic
ridge) is far smaller than a ball containing the noise (`‖e‖ ≈ 32σ`). It
works because an affine model sees the noise only through `s(x)ᵀe`, and
FO-CP has no `s(x)`.

| Option | Radius | Guarantee | Status |
|---|---|---|---|
| (a) Model-free ball | `√χ²(n, 1−ε/K)` per outcome, whitened (union bound) | exact for any model, given exact separation | F−1 (job 29561590): 32.5σ at n = 1,000. Infeasible on cubic ridge in 100 of 100 redraws; on the frozen network 1.00 against the ODE at +15.7% of the objective range (simulated radius: +2.2%). Kept as the conservative reference |
| (b) Linearized subspace | `√χ²(d_eff, 1−ε)`, `Δy` in the linearized map's span | exact for affine maps only | closed-form check in F2 where d_eff is small; ≈ (a) if d_eff ≈ n |
| (c) Sampled W (chosen 2026-11-19) | below | exact for affine models, where it is Theorem 1's simulated radius; first order for networks (a parametric bootstrap) | F2's radius |

Radius (c): for N = 200 noise draws e′ (as for the RC), retrain on
`ȳ + e′` and compute
`W_c = max over x ∈ X of [ĥ(x; ȳ+e′) − ĥ(x; ȳ)] / ‖s̃(x)‖`, with s̃ the
linearized label sensitivity at θ̂ and X the whole domain. R is the
⌈(N+1)(1−ε)⌉-th smallest W_c. R depends on the data through θ̂ and
`s̃`, so with the noise redrawn per cost vector it is not one number for all
100 runs (unlike the RC on fixed features). Decided 2026-11-19: compute R on
the F2 pilot's 10 redraws; if its spread is small, reuse the largest value
for every run, otherwise compute it per redraw. Within a run, one R serves
every iteration. Each W_c is replaced by Gurobi's upper bound from Dinkelbach
iterations, as in `solve_W` (`probe_rc_ridge.py:252`). On the end-to-end
network each iteration is a mixed-integer cone program with two ReLU
embeddings, with `s̃(x)` linearized exactly within each activation pattern.
Enumeration or multistart gradient on x only warm-starts it, since alone
they understate R. The cost per draw is unmeasured (F2 pilot). Scenario
optimization, imposing about 361 retrained models, is not pursued.

### Probes

F−1 to F2 decide whether FO-CP is worth building fully.

| # | Question | Design | Decides |
|---|---|---|---|
| F−1 | Is radius (a) affordable? | `probe_rc_ridge.py --stage modelfree` at R = √χ²(n, 0.9) on `cubic` and `net_frozen`, 100 redraws | Done (job 29561590): no (Radius table) |
| F0 | Is the machinery right? | (i) Affine models: one step gives `R‖s(x)‖`. (ii) End-to-end network: implicit gradient against retraining. (iii) Loop on `net_frozen`: x* and objective converge to the RC's solution as τ → 0; record cuts | (i) passed; (ii) run in directional form; (iii) passed. Results below |
| F1 | Does ascent beyond the first step find worse relabelings? | End-to-end (32, 32), a few redraws, at the nominal x* and one robust iterate. Increase in the retrained prediction, in σ units, at equal R, for: the linearized step, multistart ascent, the random bank (B = 200), the true relabeling `Δy = −e`, cold restarts with other seeds. Also d_eff | Done (job 29667805): ascent adds about 20% over the linearized step; the random bank finds a tenth of either; cold restarts move ĥ about half as much as the worst case. Results below |
| F2 | Does FO-CP deliver on SS1, and at what cost? | Radius (c), with (b) as a check where d_eff is small. Pilot: 10 cost vectors, SS1(ii); then 100 cost vectors, both arms. End-to-end network: FO-CP, nominal, margin, wrapper, C-MICL (one model for all). `net_frozen`: FO-CP beside the last-layer RC. Judged against the clean-label model and the arm's truth | H4. Pilot done (job 29709201): results below |

## Instances

| # | Instance | Truth | Role |
|---|---|---|---|
| SS1 | Reactor, Ovalle et al.'s 100 cost vectors, σ = 2 | two arms, labels generated by (i) the ODE; (ii) a (32, 32) network fitted to noise-free ODE labels | arm (ii) has almost no model bias, so it tests label-noise protection alone; report the gap `ĥ(y*) − truth` |
| WFP | Palatable diet, regression (OptiCL; the most reused case in constraint learning) | OptiCL App. C.1's closed-form palatability function; its constants are unpublished, so recover them from the public 5,000 rows or ask the authors | second instance, both methods |
| SS2 | Chemotherapy, real covariates, five toxicity constraints | a fixed fitted model per outcome; binomial noise from arm sizes | RC: several constraints against a known truth, infeasibility (P2) |
| — | Workload dispatching (48 constraints; EML, OptiCL, SurrogateLIB) | thermal simulator, apparently not public | optional |
| — | Student enrollment (JANOS, OptiCL) | fixed logistic model | later, with classification |

## Experiments

The number of displays is fixed once F−1 to F2 have run.

| # | Experiment | Problem | Design | Output | Section |
|---|---|---|---|---|---|
| E1 | Feasibility, cost and the radius | reactor (SS1) | RC on the lead model and FO-CP on the end-to-end network; 100 cost vectors, noise redrawn for each. RC at the pointwise, simulated and closed-form radii for ε ∈ {0.05, 0.1, 0.2} and on a radius grid. Constant margin, wrapper and C-MICL on the same model as the method compared, at their prescribed settings and over their ranges. Nominal | Table 1, Figures 2–3 | 6.1, 6.2 |
| E3 | Second instance | WFP, regression | as E1, one ε, both methods | Table 3 | 6.4 |
| P2 | Several constraints, known truth | SS2 | RC; certified fallback against protecting a fraction p of the constraints | appendix or 6.3 | 6.3 |
| E2 | Real data | chemotherapy | Maragno et al.'s protocol. RC with the joint radius over the test cohorts' treatment slices, certified fallback where infeasible; margin, wrapper, C-MICL and nominal on the same fixed models, each decision given its certified level | Table 2 | 6.3 |

Appendix, one table each, on the reactor: A1, standard against honest
fitting for a forest; A2, non-Gaussian noise (Student-t, uniform); A3, cubic
ridge, where the theorem holds with no caveat (unless it becomes the lead
model).

Cut: baselines on two model classes; forest runs on the reactor other than
A1; sample-size, heteroscedastic-noise and cross-validated-ridge checks;
Bonferroni as a reported result (it stays in the theory); the
width-at-the-optimum comparison with C-MICL.

## Results

### Probe F0 (2026-11-19, local)

`experiments/probe_focp.py`; outputs `results/paper_a/prelim/f0_*`. Redraws
0 to 2, Ovalle et al.'s cost vectors 0 to 2, σ = 2, R = 3.59σ.

- F0(i) passes. On cubic ridge and `net_frozen` the generic implicit
  gradient equals `s(x)` (relative error ≤ 4e-14), one ascent step lowers h
  by exactly `R σ ‖s(x)‖`, and no second step is accepted.
- F0(ii) as written (per-row cosine ≥ 0.99 against finite differences) is
  not measurable on the end-to-end network: one label of 1,000 moves h(x*)
  by about 0.005 to 0.05, while retraining on the same labels moves it by up
  to 0.09 (the training map's own drift). Central differences at different
  steps disagree (cosine 0.15 to 0.8). The test was run instead along whole
  directions, which is how the ascent uses the gradient:

| | α = 0.01 (repo) | α = 1 |
|---|---|---|
| Drift of T at x* (same labels) | −0.02 to −0.09 | ≤ 0.002 |
| Drop of h(x*) along −g at t = R | 5.5 to 8.2 | 1.4 to 2.3 |
| Same, as a fraction of the linear prediction | 0.87 to 0.99 | 0.32 to 0.97 |
| Drop along random directions at t = R | ≤ 1.5 | ≤ 0.08 |
| Frozen-network RC margin at the same x | 0.84 to 0.92 | 0.95 to 1.38 |
| ODE at the nominal x* (floor 50) | 39.3 to 42.3 | 48.6 to 50.1 |

- The implicit gradient is a good ascent direction at both α. At α = 1 it
  overstates the change at small steps, consistent with a minimizer on
  ReLU kinks, which the smooth implicit-function formula ignores.
- With α = 0.01 (the synthetic instance's CV pick, not a reactor choice)
  the end-to-end network is so label-sensitive that one step at R lowers
  h by 6 to 9 times the frozen network's RC margin, and its nominal decision
  misses the ODE by 8 to 11. At α = 1 both are of the RC's size.
- F0(iii) passes (redraw 0, R = 3.59σ). On `net_frozen` and cubic ridge the
  loop (cuts at 50 + τ, separation against 50) converges to the RC's cone
  solution as τ → 0: objective gap +2.7% / +1.6% at τ = 0.5σ, +0.005% at
  τ = 0.001σ, with 1 to 2 cuts (frozen) and 2 to 10 (cubic); every loop
  decision is RC-feasible with slack about τ.
- α chosen by 5-fold CV on the noisy labels (`cv_alpha.json`): α = 1, CV
  MSE 4.37 against σ² = 4; α = 0.01 scores 8.97. F1 runs at α = 1 (job
  29664832). The implicit gradient is the derivative of the current piece:
  about 45 (row, unit) pre-activations sit within 1e-4 of a kink, so it
  matters for radius (c)'s denominator, not for the ascent, which uses only
  the direction.

### F2 pilot (job 29709201, 2026-11-19)

Ten cost vectors (redraws 0 to 9) on both SS1 arms; end-to-end network at
α = 1; radius (c) per redraw by enumeration (3.5 to 5.5σ; understated, see
below); τ = 0.05σ; cap 20 master solves. `f2_*_a1.json`. Feasibility of the
final decision against the clean-label model `T(y*)` from θ₀, a cold-start
clean fit, and the arm's truth; cost as % of the objective range over the
nominal decision.

| Arm, separation | Converged | Iterations | Clean | Cold | Truth | Nominal: clean / truth | Cost | Attack below 50 | Separation per iteration |
|---|---|---|---|---|---|---|---|---|---|
| (ii) truth net, multistart | 10/10 | 4.2 (1 to 7) | 0.90 | 1.00 | 0.90 | 0.30 / 0.30 | +3.5% | 3/10, gap 0.04σ | 6 min |
| (ii) truth net, one step | 10/10 | 3.1 | 0.90 | 0.80 | 0.80 | | +2.8% | 9/10, gap 0.38σ | 13 s |
| (i) ODE, multistart | 10/10 | 7.4 (1 to 19) | 1.00 | 1.00 | 0.90 | 0.30 / 0.20 | +3.5% | 3/10, gap 0.06σ | 16 min |
| (i) ODE, one step | 10/10 | 3.6 | 0.90 | 0.90 | 0.90 | | +2.3% | 9/10, gap 0.54σ | 49 s |

- On ten cost vectors FO-CP with multistart separation reaches the 0.9
  target against both the clean-label model and the truth, on both arms,
  where nominal reaches 0.2 to 0.3, at +3.5% of the objective range
  (cubic-ridge RC: +4.2%; frozen-network RC: +2.2%, job 29141177). The
  standard error at n = 10 is about 0.1.
- The ODE arm has no bias problem at α = 1: the end-to-end network's
  decisions meet the ODE as often as the clean-label model.
- One-step separation is cheaper and stops earlier, but the a posteriori
  attack still finds a violation at 9 of 10 final decisions (multistart: 3
  of 10, by 0.04 to 0.06σ). Multistart is the method; one step is an
  ablation.
- Every run converged within the cap, but the ODE arm needed up to 19
  master solves; wall time per task 6 minutes to 7.6 hours.
- The full F2 (100 cost vectors × 2 arms, plus the baselines on the
  end-to-end network) is about 200 tasks of 1 to 8 hours on
  mit_preemptable.

### Radius (c) pilot (jobs 29707708, 29708663, 29709194, 29710101; 2026-11-19)

Enumeration over 20,000 domain points, N = 200 draws, end-to-end network at
α = 1, redraws 0 to 9 on both SS1 arms (`radius_c_*`, `radius_refine_*`).

- R by enumeration: 3.47 to 5.48σ (mean 4.44) on arm (i), 3.75 to 4.63σ
  (mean 4.22) on arm (ii). Its product with the median `‖s(x)‖`, the margin
  actually imposed, is steadier: 1.90 to 2.59 flow units (coefficient of
  variation 0.09) and 1.93 to 2.29 (0.06). Per-redraw R is used (F2 pilot).
- Enumeration understates W badly. Local maximization (SLSQP from the 10
  best sampled points) on the same retrained model raises a draw's W by a
  median 1.0 to 1.6σ, up to 3.6σ; R rises by 0.45 to 0.89σ on the five
  redraws checked (10 to 17%). The maximum sits in narrow regions where the
  linearized sensitivity `‖s(x)‖` is small, which a sample misses, and the
  refined value is still only a lower bound.
- The training map is chaotic for full-size noise draws (‖e′‖ ≈ 32σ):
  retraining after a 1e-10 change to the labels moves the prediction by up
  to 0.8 flow units and one draw's W from 2.19 to 3.48; a change of thread
  count does the same, and recomputing on another cluster node reproduced W
  exactly for only 18 of 125 draws. Small relabelings (F0's drift, 0.002)
  are stable; long L-BFGS paths on a flat landscape end in different nearby
  minima. This applies to the clean-label reference `T(y*)` too
  (‖y* − ȳ‖ ≈ 32σ), so the fixed θ₀ does not make it deterministic.
- Open (for JS): radius (c)'s ratio form and the clean-label reference for
  a network.

### Probe F1 (job 29667805, 2026-11-19)

End-to-end (32, 32) network at α = 1, redraws 0 to 4, Ovalle et al.'s cost
vectors 0 to 4, R = 3.59σ; at the nominal decision x0 and at x1, the master
with one cut. Drop of h(x) in σ units (`f1_r*_a1.json`):

| | Mean | Range |
|---|---|---|
| Linearized step | 1.03 | 0.60 to 1.65 |
| Multistart ascent (zero + 4 random starts) | 1.25 | 0.65 to 1.98 |
| Ascent / linearized | 1.20 (median) | 1.02 to 1.88 |
| Random bank, worst of B = 200 | 0.13 | 0.06 to 0.18 |
| True relabeling (retrain on clean labels; ‖e‖ ≈ 32σ, outside the ball) | 0.27 | −0.57 to 1.01 |
| Cold restarts, largest of 5 seeds (abs.) | 0.68 | 0.30 to 1.50 |

- Ascent beyond the first step finds about 20% worse relabelings, so FO-CP
  does not reduce to one-step separation; a one-step variant would capture
  about 83% of the worst case at a fraction of the cost. Accepted steps per
  start: 1 to 7. The best start was the zero start at 4 of 10 points.
- Random relabelings find a tenth of what ascent finds: the random-bank
  cutting planes of the earlier work could not have found the worst case.
- Training randomness is large. Cold restarts on the same labels move h by
  about half the label-noise worst case, and more than retraining on the
  clean labels does. The guarantee is stated against `T(y*)` from the fixed
  θ₀, which removes seed randomness by definition; the cold-start spread is
  the caveat to report beside it.
- At one point of ten (redraw 4, x1) retraining on the clean labels lowers h
  more than the worst case found in the ball (0.76 against 0.65σ). Whether
  R covers the real noise is radius (c)'s question (F2).
- d_eff: rank 161 to 283, participation ratio 67 to 87, 90% of the energy in
  44 to 57 directions; radius (b) would be about 8 to 18σ.
- ODE at x0: 45.7 to 50.0 (4 of 5 below 50); at x1 46.0 to 50.8. One cut
  does not fix ODE feasibility, as expected from bias and a single cut.
- Cost: 44 to 117 minutes per point with the bank and cold restarts, 2 on
  redraw 0. The split between separation and the extras was not recorded;
  F2's pilot must time the separation alone.

### Results of job 29141177 (2026-11-14)

Reactor, 100 cost vectors, noise redrawn per run, ε = 0.1, σ = 2 known,
`experiments/probe_rc_ridge.py`. Network (32, 32): R2 freezes its hidden
layers on an independent noise draw (the theorem holds exactly), R4 fits
them on half the rows and the last layer on the other half, R3 fits
everything on the same labels (standard fitting). Feasibility against the
clean-label model / the ODE; cost as % of the objective range. Outputs:
`results/paper_a/prelim/redraw_*_paper_eps0.1_*.json`.

| Run | Nominal, ODE | Pointwise | Simulated | Closed-form | Cost, simulated / closed-form |
|---|---|---|---|---|---|
| R1 cubic ridge | 0.66 | 0.90 / 0.80 | 1.00 / 0.95 | 1.00 / 1.00 | 4.2% / 8.6% |
| R2 network, frozen | 0.18 | 0.91 / 0.22 | 1.00 / 0.52 | 1.00 / 0.79 | 2.2% / 3.3% |
| R4 network, sample-split | 0.25 | 0.92 / 0.28 | 1.00 / 0.42 | 1.00 / 0.61 | 3.1% / 5.2% |
| R3 network, standard | 0.23 | 0.42 / 0.29 | 0.76 / 0.33 | 0.98 / 0.45 | 1.9% / 3.6% |

- Cubic ridge: nominal reaches 0.66 against the ODE, the RC 0.95.
- The network misses against the ODE because of bias. At the nominal
  optimum its clean-label model overstates the ODE in 80 to 90% of runs, so
  the optimizer selects on misfit a label-noise margin cannot see.
- Standard fitting breaks the guarantee (R3: 0.76 against 0.90).
- No selection effect at ε = 0.1: the pointwise radius already meets the
  target against the clean-label model, also at ε = 0.05 and 0.2 (jobs
  29142767-8). The simulated radius's extra margin pays for bias spread.
- Enumeration understates W by 0.10 to 0.24σ on the network, so the radius
  must come from the solver.

Proposed to DB, not adopted (open since 2026-11-16):

- Cubic ridge leads on the reactor, and the network becomes the example
  where a label-noise margin is not enough. E1's baselines on cubic ridge,
  plus C-MICL as published on the network.
- The bias margin `B(x)` may need to be more than a remark.
- Try Ovalle et al.'s surrogate families (`regression.py`): LinearDT
  (depth 5), random forest and gradient boosting (15 trees, depth 5), and
  the (32, 32) MLP. The tree families are affine in the labels once splits
  are fixed. LinearDT is at the noise floor (calibration MAE 2.08 at σ = 2)
  and is the candidate lead. Needs `linear-tree` and the leaf-vector code
  (I6).

### Cubic ridge at a fixed cost (2026-11-14, local)

Cubic ridge (56 features, fixed penalty, train R² 0.957), the fixed balanced
cost, σ = 2. Outputs in `results/paper_a/prelim/`.

- Gurobi solves the RC (nonconvex, one second-order cone) to proven
  optimality in about 1 s, about 0.1 cheaper than enumeration over 42,000
  sampled points at every radius.
- The solver's simulated radius takes about 43 s per draw (four or five
  global solves), with bounds on W agreeing to 0.001. Enumeration over 1,000
  draws gives 3.59σ at ε = 0.1 (3.84σ at 0.05, 3.37σ at 0.2). On 10 solver
  draws W was 0.03 to 0.15σ above enumeration, and above 0.1σ in six
  (`radius.json`), so adding 0.1σ to an enumerated radius does not cover the
  gap. Closed form 8.36σ, pointwise 1.28σ.

100 noise redraws, ε = 0.1, every run solved:

| Radius | R | Feasible vs clean-label model | Feasible vs ODE | Mean cost | Mean margin at `x*` |
|---|---|---|---|---|---|
| Nominal | 0 | 0.30 | 0.96 | 5.792 | 0 |
| Pointwise | 2.56 | 0.87 | 0.99 | 5.831 | 0.71 |
| Simulated (3.7σ) | 7.40 | 1.00 | 1.00 | 5.902 | 1.91 |
| Closed-form | 16.72 | 1.00 | 1.00 | 6.039 | 4.14 |

- Against the clean-label model: pointwise 0.87 against 0.90, the direction
  selection predicts but within one standard error (0.03); the simulated
  radius had no violations, so it is conservative here.
- Against the ODE nothing fails: the cubic's bias at the optimum is on the
  safe side (about 1.8 flow units on the first draw) and larger than the
  noise effect.
- Cost over nominal: 1.9% at the simulated radius, 4.3% at the closed form.

### Earlier measurements on the network

- Old harness (one balanced cost, 10 folds, ODE judge): nominal feasible in
  0 of 10, C-MICL at alpha = 0.1 in 3 of 10; a tuned margin and a tuned
  wrapper reach 0.9 only when tuned against the truth.
- C-MICL width-model probe (`notes/progress.md`, 2026-09-23; Ovalle et al.'s
  100 cost vectors, one dataset, the repo's network, ODE judge): nominal
  0.00, C-MICL at alpha = 0.1 0.81 (0.83 on the randomized balanced draw).
  The premise likely holds on their costs; E1 has to confirm it with the
  (32, 32) network and redrawn noise. C-MICL's number to beat is about 0.8,
  two to three standard errors below a 0.9 target at 100 runs.

## Open design points from the 2026-11-14 review

1. Hidden layers. E1 trains them on a second noise draw, which a
   practitioner does not have; that makes "uses every training row" and the
   rows-used column unfair to C-MICL. R4 (sample split) is the one-dataset
   alternative. Both claims stay provisional until the lead model is
   settled.
2. Number of redraws. At 100 runs the standard error near 0.85 is 0.036, so
   0.9 against C-MICL's 0.81 is about two standard errors. If the RC lands
   within 0.05 of C-MICL, use 300 redraws for the as-prescribed rows
   (Table 1) and 100 for the sweep (Figure 2).
3. C-MICL as published. Add one row with C-MICL fitted end to end as they
   fit it (their network, width model at batch 32). The shared-model rows
   remain the comparison of margins.
4. Pointwise radius. The Gaussian `σ z(1−ε)` (1.28σ at ε = 0.1) is used
   throughout; `paper_a_guarantee.md`'s sub-Gaussian `√(2 ln(1/ε))` is
   marked as such.
5. Novelty (checked 2026-11-14; outline Section 1.2). Not new: bands over a
   region with a constant simulated by optimizing over it (Liu et al. 2005,
   linear models over a box); validity at the selected point (PoSI,
   inference on winners); the uniform mean + β·sd constraint (safe Bayesian
   optimization; Fiedler et al. 2021 has our noise term for kernel ridge).
   What remains: the clean-label target, with no function-norm bound in the
   radius; a mixed-integer region with each supremum a global MIP; one form
   for ridge, last-layer networks and forests; the band embedded as a
   constraint. Hence the simulated radius uses the solver's bound on each W
   (I4). Fiedler et al.'s noise term bounds the whole noise vector (about
   σ√n), looser than our closed form σ√χ²(d) with d ≪ n; one sentence in
   the paper.

## Honest fitting

A condition of the theorem for models whose structure is learned from the
labels, not of the method. The theorem needs `s(x)` independent of the
noise; structure chosen on the noisy labels chases the noise, so the margin
understates the error.

| Model | Structure learned from labels | Theorem holds with standard fitting |
|---|---|---|
| Ridge, features and penalty fixed in advance | no | yes |
| Last-layer network | hidden layers | only with the hidden layers fitted on separate rows |
| Tree, forest | splits | only with honest fitting |

- Reactor: R2 and R4 meet the theorem against the clean-label model (job
  29141177). If the lead model stays a network, the RC uses one of them, and
  the end-to-end network goes to FO-CP, which needs no honest fitting.
- Chemotherapy: the RC runs on the fitted models with their structure
  frozen, under standard fitting, and the paper calls the guarantee
  empirical there. A1 measures what standard fitting costs in violations
  and honest fitting in accuracy.

## Chemotherapy

### Protocol and models

Maragno et al.'s protocol, already in the repo: fit on the training arms,
prescribe for the 96 held-out arms, judge with the ensemble fitted on all
416 arms, repeat over subsamples of the training rows and report the mean
and the worst.

- Models are fixed before any robust run: a forest for the four outcomes
  where the existing selection picked XGBoost, ridge for the two where it
  picked elastic net, the same for every baseline. The paper reports the
  accuracy given up against the existing picks and states that the choice
  used the labels, a departure from fixing the model first (outline
  Section 6.3).
- σ per arm, binomial, for the main result (2026-11-14):
  `σᵢ = √(pᵢ(1−pᵢ)/n_armᵢ)`, with the weighted norm. It is the sampling
  error of an arm's toxicity rate, the noise the theorem is about. It uses
  the noisy `p`, so T2 must cover the plug-in. The out-of-fold residual sd
  is a second row only, since it mixes misfit into σ.
- Survival is a learned objective with no margin, as for the other methods.
- Measures: the certified level of each method's prescriptions (main), the
  fraction prescribed, toxicity limits met under the judge, survival. The
  judge is itself fitted, so this experiment does not test the theorem; E1
  does.

### Infeasibility on chemotherapy (2026-11-14)

The instance has little headroom: nominal already scores 0.89 to 0.96, and
C-MICL at alpha = 0.1 prescribes for about 6% of cohorts. C-MICL's width is
a prediction interval for a new outcome, covering noise and misfit, and
does not shrink with n. The RC's margin `R σ‖s(x)‖` is a confidence band on
the fitted model and shrinks with n (`‖s(x)‖` ≈ √leverage for ridge,
1/√n_leaf for a tree). A label-noise margin does not see misfit, so the RC
may prescribe often and still miss against the judge; Table 2 reports both.
The random-bank rollback is not carried over (Why the earlier approaches
are not the paper's method).

Three layers, in order:

1. Shrink the radius, keeping the guarantee:
   - the joint simulated radius instead of Bonferroni (T1);
   - W's supremum over the treatment slices of the cohorts prescribed for,
     not the whole box. Over the union of the 96 test cohorts' slices the
     radius holds for all of them at once, at about one MIP per draw (a
     selection variable picks the cohort). Per cohort it is smaller but
     costs 96 times the solves. The main result uses the union;
   - the per-arm weighted norm;
   - an unequal split of ε across the five constraints if one dominates
     (weighted Bonferroni, or the joint radius with per-constraint scaling).
2. The certified fallback. For any decision x,
   `r̂(x) = minⱼ (bⱼ − f̂ⱼ(x)) / ‖Σⱼ^½ sⱼ(x)‖`, with `Σⱼ` the diagonal of
   outcome j's per-arm noise variances. Theorem 1's proof gives violation ⇒
   `W > r̂(x*)`, so `P(violation and r̂(x*) ≥ r) ≤ P(W > r)` for every r,
   however `x*` was chosen, including by maximizing `r̂` (T6). When no
   regimen meets the target radius, solve `max t` s.t.
   `f̂ⱼ(x) + t‖Σⱼ^½ sⱼ(x)‖ ≤ bⱼ` for all j by bisection on t (each step is
   the RC at radius t), prescribe the result and report `ε̂ = P(W > r̂)` from
   the stored draws of W. Every prescription carries its own level. The
   statement is joint, not conditional on reaching `r̂`, and must be worded
   that way.
3. Abstention. A cohort with no regimen certifiable at a useful level is
   reported as such. C-MICL's 94% non-prescription is read the same way.

Table 2 puts every method on one scale, since `r̂(x)` applies to any
decision. Columns: fraction of cohorts certified at ε ∈ {0.05, 0.1, 0.2};
fraction prescribed; toxicity limits met under the judge; predicted
survival; the worst subsample beside each mean. The certified level does not
depend on the judge, whose separation is weak; the judge's verdict stays
beside it as the check against misfit.

### Probe P1 (job 29576766, 2026-11-19)

`experiments/probe_p1_gastric.py`. Ridge (fixed penalty) on the five
toxicity percentile labels. For each of the 96 test cohorts, r̂ by
Dinkelbach over its regimens (domain constraint and trust region); the joint
radius by enumeration over 96 contexts × 282 training regimens, 1,000
draws.

| σ | Median σ (percentile scale) | Joint radius, ε = 0.1 | Certified at ε = 0.05 / 0.1 / 0.2 | Median r̂ | Median level |
|---|---|---|---|---|---|
| Binomial per arm (delta method) | 0.10–0.26 by outcome | 4.29 | 0.43 / 0.50 / 0.61 | 4.31 | 0.09 |
| Out-of-fold residual sd | 0.31–0.33 | 4.19 | 0.00 / 0.00 / 0.00 | 2.73 | 1.00 |

Every cohort is nominally feasible. Ridge fits poorly (training R² 0.37 to
0.48).

- Binomial σ: half the cohorts are certified at ε = 0.1, so H3 stays
  viable. The median r̂ sits on the radius, so the fraction is sensitive to
  it: 0.47 / 0.43 / 0.22 if the radius is 0.1 / 0.2 / 0.5σ larger. E2 needs
  the solver's radius.
- Out-of-fold σ: nothing is certified at any level (best level 0.53). The
  residual sd (0.31) exceeds the label sd (about 0.29), so σ is mostly
  misfit. The binomial σ stays the main result.
- Not yet measured: the baselines' certified levels, and forests in place of
  ridge.

## Theory tasks

| # | Task | Status | Needed for |
|---|---|---|---|
| T1 | Simulated radius: the ⌈(N+1)(1−ε)⌉-th smallest of N simulated W gives level ε by exchangeability, with each W replaced by the solver's upper bound; joint version for several constraints | drafted 2026-11-15 (the joint version held in `paper_a_guarantee_held.tex`); not checked by JS | E1, E2 |
| T2 | Estimated σ: an upper confidence bound and its effect on ε; unequal known `σᵢ`; the binomial plug-in. Alternative: simulate `σ̂/σ` jointly with the noise, as Liu et al. (2005) do | to prove | Section 2; E2 |
| T3 | Directed cuts terminate in at most K iterations | drafted 2026-11-15; not checked by JS | Section 4; E2 |
| T4 | Sub-Gaussian subspace radius | to prove | appendix |
| T5 | Lipschitz bias margin | to prove | appendix, optional |
| T6 | Certified level of any decision, `P(violation and r̂(x*) ≥ r) ≤ P(W > r)` for all r and any choice of `x*`; joint version | drafted 2026-11-15, held in `paper_a_guarantee_held.tex`; not checked by JS | Section 3; E2, Table 2 |
| T7 | FO-CP stopping: with exact separation and the master at `b − τ`, feasibility at `b` after finitely many cuts (Mitsos 2011; cuts discretize the set as in Blankenship and Falk 1976, not Kelley's linearizations); with local separation, only the a posteriori check. The protected-set guarantee for both radius events | to prove | FO-CP, P2 |
| T8 | Radius (c): exact for affine models (reduces to T1), first order for networks; (a) as the reference | to prove | F2 |

Proved: Theorem 1 and its finite, subspace and Bonferroni corollaries.
Dropped: classification, the ridge bias bound, counting K for forests.

## Implementation tasks

| # | Task | Status | Needed for |
|---|---|---|---|
| I1 | RC for ridge: cone on `‖s(x)‖`, polynomial features | done as a probe (`probe_rc_ridge.py`) | A3 |
| I2 | RC for a last-layer network: hidden layers frozen, last layer refit by ridge, ReLU embedding plus one cone | to build | E1 |
| I3 | Redraw harness: cost vector and noise drawn, refit, solve, judge against the clean-label model and the ODE | probe version for a fixed cost | E1 |
| I4 | Simulated radius, single and joint, each maximum solved to a bound, warm-started from enumeration | probe version; slow | E1, E2 |
| I5 | Baselines on the shared network with frozen hidden layers | to check against the existing methods | E1 |
| I6 | Forest with leaf vectors; RC by directed cuts; honest variant | to build | E2, A1 |
| I7 | RC in the chemotherapy runner; per-outcome σ; fixed models | to build | E2 |
| I8 | WFP regression instance with OptiCL App. C.1's truth function | to build | E3 |
| I9 | `r̂(x)` for any decision; the max-t fallback; `ε̂` from the stored W draws | to build | E2, Table 2 |
| I10 | Probe P1 | done (`probe_p1_gastric.py`, job 29576766) | H3 |
| I11 | FO-CP training map; gradients and the Hessian (1,281 weights) by PyTorch autograd on the exact training objective, for correctness (decided 2026-11-19) | to build | F0–F2 |
| I12 | FO-CP separation: implicit gradient, projected ascent, starts, a posteriori attack | to build | F0–F2 |
| I13 | Master of about 200 lines on the probe's embedding, not `cp.py` (`embed_model` handles only sklearn types; `cp.py` carries anchors, rollback and eviction). Hidden layers embedded once when only the last layer changes; `zₖ` big-M | to build | F2 |
| I14 | SS1's realizable truth (arm ii); SS2's generator | to build | F2, P2 |

Everything beyond a probe goes through `sbatch`. E1 is many short solves,
split by method and cost vector into `mit_quicktest` or `mit_normal`
arrays. The simulated radius at 200 draws is a cluster job.

## Data

| Problem | Have | Enough for |
|---|---|---|
| Reactor | ODE oracle; generator for any n and σ; 1,000 cached designs; all baselines; Ovalle et al.'s cost draw | E1 once I2 to I5 exist |
| Chemotherapy | 416 real arms, six outcomes, arm sizes, judge ensemble, subsampling protocol, all baselines | E2 once I6 and I7 exist; no ground truth, so practical claims only |
| WFP | nothing in the repo; OptiCL's public data (5,000 baskets) | E3 once I8 recovers the truth function |
| SS2 | the real arms' covariates and sizes; a fitted model per outcome as truth | P2 once I14 builds the generator |

## Sequence

1. Done: F−1 (radius (a) impractical) and P1 (binomial σ certifies half
   the cohorts at ε = 0.1).
2. The reactor's lead model, with DB.
3. FO-CP gate: I11, I12, F0(i) and F0(ii) (local), then F1 (go/no-go).
   Then I13, F0(iii) (it needs the master), the F2 pilot (which also
   measures how much radius (c) varies across redraws), and the full F2.
   T7 and T8 alongside.
4. RC on the reactor, in parallel on the cluster: I2 to I5, then E1. T1
   alongside. Draft Sections 2, 3, 6.1, 6.2.
5. WFP: I8, then E3 with both methods.
6. Several constraints, known truth: I14, then P2. T2, T3, T6.
7. Real chemotherapy: I6, I7, I9, then E2. Draft Sections 4 and 6.3.
8. Appendix tables A1 to A3.

## Risks

RC and the reactor:

- The nominal decision may not fail. It did not on cubic ridge at the fixed
  cost (0.96 against the ODE); on Ovalle et al.'s costs it did (0.66 cubic,
  0.18 to 0.25 network; job 29141177).
- The gap between the clean-label model and the truth may dominate, so that
  feasibility against the ODE is decided by bias in either direction. On
  cubic ridge the gap (about 1.8) matched the simulated margin (1.9).
- The radius may overshoot: no violations in 100 redraws at ε = 0.1. The
  result is then the margin's shape at matched feasibility (H2), and the
  guarantee is described as conservative.
- The last-layer RC covers only noise entering through the last layer. If
  most of the network's error comes from the hidden layers, the margin is
  too small against the ODE even though the theorem holds.
- The RC may not be cheapest at matched feasibility, since `‖s(x)‖` misses
  model misfit that C-MICL's width model can pick up.
- The solver's radius is slow: 43 s per draw on cubic ridge; FO-CP's W_c is
  unmeasured.

Chemotherapy:

- Five margins may leave the RC infeasible for many cohorts. Mitigations:
  the binomial σ, the radius over prescribed slices, the certified fallback.
- Under the binomial σ the RC may prescribe and still miss against the judge
  where the models misfit; the judge's verdict is reported beside the
  certified level.
- The judge cannot separate methods: the audit put the 0.9 target inside its
  uncertainty band for most cells. Report orderings and the fraction
  prescribed there, and keep feasibility levels to the reactor.

FO-CP:

- Local separation gives no certificate; the a posteriori attack measures
  the gap but does not close it.
- Radius (c) is first order for networks and guarantees only against a
  resampled model (T8).
- FO-CP sees label noise, not bias, so on the end-to-end network it may miss
  against the ODE as the network RC did. SS1(ii) separates the two.
- The master grows by one embedded network per cut, and separations are not
  shared across cost vectors.
- Referees will ask about convergence and what stopping certifies (T7); a
  scenario-optimization baseline; links to adversarial training (min-max);
  runtime against n and the number of cuts, and against the RC; the
  separation gap; robustness to training randomness; and why an ℓ₂ label
  ball, which contains coherent shifts independent noise does not produce.

WFP:

- Its truth function has unpublished constants. If the public data does not
  recover them, ask OptiCL's authors.

## Why the earlier approaches are not the paper's method

Up to September we compared cutting planes over a random bank, the wrapper,
a right-hand-side margin and C-MICL, each tuned on its own parameter (the
dial sweep). Numbers are from the reactor dial cell (balanced cost, 10
folds, ODE test stage of 2026-09-07) and the gastric cell (2026-09-08).

| Approach | What it did | Why it is not the method | In the paper |
|---|---|---|---|
| Cutting planes over a random bank | Drew 200 relabelings from an ellipsoid of radius rho, retrained on each, cut the worst at `x*`, stopped at tolerance tau. With several constraints, a cut that left a protected cohort infeasible was rolled back | The bank is a finite sample of the set, so stopping shows feasibility for the drawn models only. rho and tau were tied to no violation level and were set against the ground truth; the working rho had to be re-measured per instance (rho = 1 never passed 0.30, rho = 2 delivered), and results depended on the bank's seed. For affine models the worst relabeling has a closed form, `R s(x)/‖s(x)‖`. The rollback gave up robustness wherever it was unaffordable, with no stated level | Out as a method; one sentence in Section 4. FO-CP brings the loop back with optimized separation and a radius set by ε; the bank is its ablation (F1). The rollback's role passes to the certified fallback |
| Right-hand-side margin | Tightened the constraint by a constant m | Same at every x, with no rule for its size: `m* = 4.0` was found only with the oracle. At 0.9 feasibility the most expensive tuned method (6.043, against 5.816 for cutting planes and 5.885 for the wrapper). On gastric a large m makes the problem infeasible (m = 0.75 solved 20% of contexts) | Baseline |
| Wrapper | Embedded 20 bootstrapped models and required a fraction to hold | The fraction is tied to no violation level. At its strictest it reached 0.00 and 0.10 on the two rho columns where cutting planes operated; it embeds 20 models; on gastric it solved 64 to 68% of contexts | Baseline |
| Robust regression | Trained a label-robust model and embedded it nominally | Acts on the learner, not the embedding | Out; Paper B |
| The dial sweep | Tuned each method's parameter against an oracle and compared the tuned points | The paper claims no tuning is needed, which a tuned protocol cannot test. One cost vector and 10 folds, so feasibility moved in steps of 0.1 | Survives as the equal-feasibility comparison (Figure 2), on 100 redraws |
| Analytic synthetic instance | Known `f_true` | The reactor gives an exact truth on an instance from the literature | Out |

The earlier work supplies the motivation in Section 1: nominal was
ODE-feasible in 0 of 10, and every tuned safeguard reached 0.9 only when
tuned against the truth.
