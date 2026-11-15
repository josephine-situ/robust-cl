# Paper A: selection-robust violation guarantee

Working note for "Constraint Learning under Uncertainty" (target: Operations Research).
Companion LaTeX block: `paper_a_guarantee.tex`.

Status: proofs checked by hand in discussion; not yet validated numerically.

---

## 1. Setting and notation

| Symbol | Meaning |
|---|---|
| `n`, `p` | training rows, features |
| `y = y* + eps` | observed labels = clean labels + noise |
| `h*` | true function, `y*_i = h*(x_i)` |
| `theta_hat = theta(y)` | model fit on observed labels |
| `theta* = theta(y*)` | **clean-label model**: same estimator, hyperparameters and frozen structure, fit on `y*` |
| `s(x) in R^n` | sensitivity vector: `f(x; theta(y)) = s(x)^T y + c(x)` |
| `sigma` | sub-Gaussian variance proxy of the label noise (sd if Gaussian) |
| `R` | radius of the robust counterpart (RC), fixed before seeing the noise |
| `X` (calligraphic) | decision domain (including trust region) |

Key identity (affine map evaluated at `y` and `y*`):

$$f(x;\theta^*) = f(x;\hat\theta) - s(x)^\top \varepsilon$$

The target is the clean-label model, **not** $h^*$. The bias $h^*(x) - f(x;\theta^*)$ is excluded
from the guarantee and must be measured separately (Section 5).

## 2. Assumptions

**A1 (noise).** $\varepsilon_i$ independent, zero-mean, sub-Gaussian with variance proxy $\sigma^2$.

**A2 (affine predictor).** For all $x \in \mathcal X$: $f(x;\theta(y)) = s(x)^\top y + c(x)$, with
$s(x)$ and $c(x)$ not depending on $\varepsilon$.

A2 holds exactly for OLS/ridge. For elastic net, trees, forests, boosting and neural nets it
needs the structure (active set, splits, hidden layers) to be chosen independently of
$\varepsilon$, i.e. **honest / sample-split fitting**: structure on one split, coefficients or
leaf values on another.

## 3. Results

**Robust counterpart (upper-bound constraint $f \le b$):**

$$f(x;\hat\theta) + R\,\|s(x)\|_2 \le b$$

For a general learned term $w\,f(x;\theta) + d(x) \le b$ with known $d(x)$, the margin is
$R\,|w|\,\|s(x)\|_2$. A lower-bound constraint $f \ge b$ is the case $w = -1$; all bounds
below are unchanged by symmetry.

**Theorem 1.** Under A1-A2, let $x^*$ be any RC-feasible point, possibly chosen as a function
of the data. Then

$$P\big(f(x^*;\theta^*) > b\big) \le P(W > R), \qquad
W = \sup_{x \in \mathcal X,\ s(x) \neq 0} \frac{-s(x)^\top\varepsilon}{\|s(x)\|_2}.$$

*Proof.* If $f(x^*;\theta^*) > b$ then

$$-s(x^*)^\top\varepsilon = f(x^*;\theta^*) - f(x^*;\hat\theta) > b - f(x^*;\hat\theta) \ge R\,\|s(x^*)\|_2 ,$$

so $s(x^*) \neq 0$ and the ratio at $x^*$ exceeds $R$; hence $W > R$. The event $\{W > R\}$
does not involve $x^*$. ∎

**Corollary 1 (finite case: trees, forests, boosting).** If $s(x)$ takes at most $K$ distinct
values on $\mathcal X$, then

$$R = \sigma\sqrt{2\ln(K/\epsilon)} \;\Rightarrow\; P(f(x^*;\theta^*) > b) \le \epsilon .$$

*Proof.* $W$ is a max of $K$ numbers $a_k = -s_k^\top\varepsilon/\|s_k\|$. Each $a_k$ is
subG($\sigma^2$) (textbook Prop. 15.2(a)), so $P(a_k > R) \le e^{-R^2/2\sigma^2}$
(Prop. 15.1). Union bound. ∎

**Corollary 2 (subspace case: OLS, ridge, last-layer NN).** If $s(x)$ lies in a subspace $S$
of dimension $d$ for all $x$, and $\varepsilon \sim N(0, \sigma^2 I)$, then

$$R = \sigma\sqrt{\chi^2_{d,\,1-\epsilon}} \;\Rightarrow\; P(f(x^*;\theta^*) > b) \le \epsilon .$$

*Proof.* Let $\Pi$ project onto $S$. Since $s(x) \in S$, $s^\top\varepsilon = s^\top\Pi\varepsilon$,
and Cauchy-Schwarz gives $W \le \|\Pi\varepsilon\|_2$. With Gaussian noise
$\|\Pi\varepsilon\|^2/\sigma^2 \sim \chi^2_d$. ∎

For ridge, $s(x) = Z M \phi(x)$ with $M = (Z^\top Z + \lambda I)^{-1}$, so $S = \mathrm{col}(Z)$
and $d = \mathrm{rank}(Z) \le p$ ($p+1$ with an intercept). Ridge shrinks $\|s(x)\|$, not $d$.

**Corollary 3 (multiple constraints).** For constraints $j = 1..m$ with their own
$s_j, \sigma_j, R_j$, choose each $R_j$ at level $\epsilon/m$:

- finite case: $R_j = \sigma_j\sqrt{2\ln(m K_j/\epsilon)}$
- subspace case: $R_j = \sigma_j\sqrt{\chi^2_{d_j,\,1-\epsilon/m}}$

Then $P(x^* \text{ violates any constraint}) \le \epsilon$.

*Proof.* $\{x^* \text{ violates some } j\} \subseteq \bigcup_j \{W_j > R_j\}$; union bound. ∎

Notes: no assumption on dependence across outcomes; cost is logarithmic in $m$; one learned
model used in several constraints needs a single $W$ (no correction); a two-sided constraint
counts as $m = 2$.

### Radii at a glance

| Case | Radius $R/\sigma$ | Example |
|---|---|---|
| Pointwise (fixed $x$, not valid for $x^*$) | $\sqrt{2\ln(1/\epsilon)}$ | 2.45 at $\epsilon = 0.05$ |
| Subspace, Gaussian | $\sqrt{\chi^2_{d,1-\epsilon}}$ | 4.28 at $d = 10$ |
| Subspace, $m = 5$ constraints | $\sqrt{\chi^2_{d,1-\epsilon/m}}$ | 4.82 at $d = 10$ |
| Finite | $\sqrt{2\ln(K/\epsilon)}$ | 3.90 at $K = 100$ |

### Open items on the theory

- **Sub-Gaussian version of Corollary 2.** A net argument gives
  $R = \sigma(4\sqrt d + 2\sqrt{2\ln(1/\epsilon)})$. Constants quoted from memory
  (Rigollet's notes, Thm 1.19); verify before use.
- **Counting $K$ for forests.** $K \le \prod_b L_b$ (leaves per tree), so
  $\ln K \le \sum_b \ln L_b$. Alternative: all $s(x)$ lie in the span of the leaf vectors,
  so Corollary 2 applies with $d \le \sum_b L_b$. Use the smaller radius. The trust region
  reduces the reachable cells.
- **Estimated $\sigma$.** The guarantee assumes a valid $\sigma$; an upper bound on $\sigma$
  is needed for full coverage.
- **Conservatism.** The bound treats $x^*$ as adversarially chosen. Experiment 2 measures
  how much smaller $R$ can be in practice.

## 4. Computing the ingredients

- **Ridge / OLS:** $\|s(x)\|_2 = \|L^\top \phi(x)\|_2$ where $M Z^\top Z M = L L^\top$
  ($p$-dimensional second-order cone). With intercept:
  $\|s\|^2 = 1/n + (\phi - \bar\phi)^\top M_c Z_c^\top Z_c M_c (\phi - \bar\phi)$.
- **Single tree:** $s(x)$ has entries $1/n_\ell$ on the training rows in the leaf of $x$;
  $\|s\|^2 = 1/n_\ell$.
- **Forest:** $s(x) = \frac{1}{B}\sum_{b,\ell} z_{b\ell}\, v_{b\ell}$ with precomputed leaf
  vectors $v_{b\ell}$ (bootstrap counts included) and OptiCL leaf binaries $z_{b\ell}$.
- **Clean-label model in experiments:** refit the same frozen structure on $y^*$
  (oracle labels without noise).

## 5. Experiment plan

### Problems

| Problem | Ground truth | Noise | Role |
|---|---|---|---|
| Synthetic | known function | known $\sigma$, controllable | theorem validation, ablations |
| Reactor (membrane reactor, from the C-MICL paper; Carrasco & Lima) | ODE oracle | $N(0, 2^2)$ on $n = 1000$ designs; constraint $\hat h(x) \ge 50$ | main validation: Gaussian noise with known $\sigma$ matches Corollary 2 |
| World Food Programme (OptiCL) | palatability simulator | inject at controlled levels | main practical case |
| Chemotherapy (OptiCL, real data) | none | large, unknown | semi-synthetic version, or short stress test |

### Experiments

1. **Validate the theorem.** Redraw label noise 500+ times at several $\sigma$; refit;
   solve the RC with the theoretical $R$. Record the rate at which $x^*$ violates the
   clean-label model. Expect rate $\le \epsilon$ across $\epsilon \in \{0.01, 0.05, 0.1, 0.2\}$.
   Models: ridge, honest tree, honest forest.
2. **Price of selection.** Same setup with $R \in \{0,\ \text{pointwise},\ \text{uniform}\}$.
   Report violation rate and objective for each. Shows whether the pointwise radius
   suffices in practice.
3. **Method comparison.** RC vs wrapper (OptiCL ensemble), cutting planes, C-MICL,
   RHS margin. Compare at matched ground-truth violation rate: objective, feasibility
   against $h^*$, solve time.
4. **Ablations.**
   - honest vs standard fitting (does A2 matter empirically?)
   - violations of $h^*$ vs of $\theta^*$ (size of the bias term)
   - estimated vs true $\sigma$
   - non-Gaussian noise (uniform, Student-t)
   - $m > 1$ constraints with Corollary 3
   - scaling in $n$ and $p$
5. **Case study.** WFP as the lead; semi-synthetic chemotherapy (fit a model to the real
   data, treat it as $h^*$, redraw noise at the estimated $\sigma$).

### Metrics to log per run

- violation of clean-label constraint at $x^*$ (0/1), violation of $h^*$ (0/1)
- slack $b - f(x^*;\hat\theta)$, $\|s(x^*)\|_2$, margin $R\|s(x^*)\|_2$
- objective value, solve time, RC feasibility status
- realized $W$ (or $\|\Pi\varepsilon\|$) to compare against $R$

### Honest-fitting detail for redraw experiments

Structure must be independent of the estimation noise. Either hold the structure split
fixed across redraws and redraw only the estimation-split noise, or redraw both splits
independently. Both satisfy A2.

## 6. Framing decisions (for the paper text)

- State guarantees as **probability of violation** of $x^*$ over the training noise.
- Do not treat $\hat\theta$ as a fixed nominal value; Theorem 1 is the actual-case result.
- Textbook results used: Definition 15.1, Propositions 15.1 and 15.2(a), and the
  Chapter 2 support function for the RC form.
