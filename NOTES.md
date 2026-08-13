The structural point: it's a sort, not a MIP

Freeze the model's structure (linear active set, or tree split topology). Then $\theta(\delta)$ is affine in $\delta$, so

$$f_{\theta(\delta)}(x^) = f_0 + g^\top \delta, \qquad g_i = \frac{\partial f_{\theta(\delta)}(x^)}{\partial \delta_i}$$

and maximizing over $D = {|\delta_i|\le\varepsilon,\ |\delta|_1\le\Gamma}$ has a closed form: sort by $|g_i|$ descending, set $\delta_i = \varepsilon,\mathrm{sign}(g_i)$ on the top $m=\Gamma/\varepsilon$ rows. That is exactly the loop already in worst_case_label_shift — same code, g substituted for r. No MIP, no solver, $O(n\log n)$.

What $g$ is, per class:

- Linear (ElasticNet; gi and os on gastric, both l1_ratio=1.0): with active set $A$ and signs fixed, $\theta_A$ is affine in $y$, so $g = X_A(X_A^\top X_A + \lambda I)^{-1}x^_A$. One $d\times d$ solve, reusable across all $x^$. Exact, not heuristic, as long as the active set doesn't flip — which you can check post-retrain for free.
- Trees / RF / GBM / XGB (dlt, blood, constitutional, infection — 4 of 6 on gastric): a leaf value is an average of its members' labels, so for one tree with $x^$ landing in leaf $L$, $g_i = \mathbb{1}[i\in L]/|L|$. For an ensemble, $g_i = \frac{1}{T}\sum_t \mathbb{1}[i \in L_t(x^)]/|L_t(x^*)|$. Computed from .apply(X_train) (sklearn) / pred_leaf=True (xgb) plus a bincount — milliseconds, no retraining. GBM/XGB leaf values are shrunk residual sums rather than plain means, so unrolling stages is fiddly, but they remain affine in $y$ at fixed topology.

The tree case also shows why greedy_adversarial_perturbation's feature-distance candidate selection is the wrong proxy: leaf co-membership is the model's own notion of locality, and it's what actually moves the prediction. A row can be Euclidean-near $x^*$ and land in a different leaf, contributing exactly zero.
