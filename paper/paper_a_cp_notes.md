Very little is needed, and none of it is hard. The guarantee attaches to RC-feasibility of the output $x^*$, not to how the algorithm got there, so the main things to show are that termination implies RC-feasibility and that the algorithm terminates.

## Worth proving (one short proposition)

**1. Correctness at termination.** If separation is exact and finds no violated cut, then $x^*$ satisfies the RC and Theorem 1 applies.

- **Exact separation is closed form:** at the current $x$, the worst-case perturbation is $\delta = R\,s(x)/\|s(x)\|$.
- **Tolerance:** if you stop at violation $\le \tau$, then $x^*$ satisfies the RC with $b + \tau$. Either state the guarantee for $b + \tau$, or tighten $b$ by $\tau$ in the algorithm.

**2. Finite termination for trees and forests.** With frozen structure, $s(x)$ takes at most $K$ values. The cut generated in cell $k$ enforces the RC exactly for every $x$ in that cell, so each cell is cut at most once. The algorithm stops after at most $K$ iterations.

## Worth citing, not proving

**3. Continuous case (linear models).** Cuts approximate a second-order cone from outside, so convergence is finite only for a tolerance $\tau > 0$. This is standard (Kelley's method; Mutapcic & Boyd on cutting-set methods for robust optimization). In this case you can also solve the mixed-integer second-order cone program directly, so cutting planes are optional.

**4. Reformulation versus cuts.** Bertsimas, Dunning & Lubin (2016) compare the two empirically. Cite this to justify reporting both.

## Not worth proving

**5. Complexity.** The master problem is already a mixed-integer program, so no useful polynomial bound exists. Reviewers at *Operations Research* will expect iteration counts and solve times, not a complexity theorem.

## Two things to state clearly

- **Random-draw cuts give no guarantee.** They are an inner approximation of the uncertainty set, so termination doesn't imply RC-feasibility. Only directed (exact-separation) cuts do.
- **Verify the output directly.** Whatever method produced $x^*$, check $f(x^*;\hat\theta) + R\|s(x^*)\| \le b$ at the end. It costs one evaluation, and it makes Theorem 1 apply to any method's output, including the wrapper or heuristic cuts, without analyzing that method.

## When cutting planes need more care

If cuts come from fully retraining the model on perturbed labels (structure not frozen), the worst-case $\delta$ has no closed form and separation is heuristic. Termination then proves nothing, and Assumption 2 fails anyway. I'd present that variant as a heuristic and rely on the direct check above.

Which variant does your current implementation use: frozen-structure directed cuts, or full retraining?