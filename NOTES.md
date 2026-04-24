Progress this week
- Make cutting planes approach tractable (>8000s to <200s)
    - Master problem: warm start, relaxed MIP gap. Prune inactive scenarios.
    - Separation problem (finding adversarial points): retrain using smaller proxy model (e.g., CART instead of RF)
- Add robust param method: on synthetic example fast, with better feasibility rate than wrapper, but true constraint violation not good. (robust on splits, not on y value)

Robust CL
- Identify worst-case sample from training data using optimization.
- Replicate chemotherapy results from CL paper (will need to account for features that are not decision variables).
- Trust region
- Added robust parameters (like global optimization paper), maybe add relaxation.
- PGD? But need gradients.
- Ensemble (both in nominal case and with robust wrapper approach)

Global Optimization
- Iteratively improve solution
    - Create new samples based on MIO solution, embed new ML models (cutting planes)
    - Create new samples based on PGD solution, embed new ML models (cutting planes, or start from scratch)
- Alternative robustness
    - Robust wrapper approach (bootstrapped models), or cutting planes as above.