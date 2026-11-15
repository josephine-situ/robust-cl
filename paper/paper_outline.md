# Constraint Learning under Uncertainty
To submit to Operations Research, by Josephine Situ and Dimitris Bertsimas

# Abstract
# Introduction
- Constraint learning is used when we have an optimization problem where we do not know the exact form of constraints, but have data to fit an ML model.
- We embed the ML model into an MIO. Many different ML models can be used (e.g., elastic net, trees, neural networks).
- Some examples taken from the survey paper.
- Data inherently is noisy. We want a method that deals with uncertainties in the data.
- Especially important as optimization can exploit errors in the constraint model.

## Literature Review
- To this end, several approaches have been used.
- Maragno et al. use a trust-region approach, where we constrain our decision to fit within the convex hull of the training data.
    They also use wrappers where P boostrapped models are embedded and optimization constrained so that 1-alpha of these models must satisfy the constraints.
- To address the lack of guarantees and scalability of embedding so many models, Ovalle et al. intoduce C-MICL. However, their guarantees are pointwise, so may not hold at the optimal solution x^*?
- Chance constraint learning (Alcantara and Ruiz).
- Other literature on robustness in CL (with a bit about each and their limitations we're addressing - also they seem to be problem specific): Xavier et al., Cremer et al., 
- Optimization w/ DCL (Ostekin et al.) seems quite similar to the CP approach we use.
- Robust optimization literature w/ robust counterparts and prob guarantees, including CP approaches.

## Contributions
- Furthermore, robust optimization literature is extensive, but has not yet been applied to CL. We create a new approach that uses robust counterparts for CL with data uncertainties, providing probabilitic guarantees that hold at the x^*
- Empirically validating our results on a wide range of problem cases, including those that would be infeasible with other methods.

# Robust Constraint Learning Formulation
- Nominal problem: embedding predictive models (can reference surveys, Maragno et al.)
- Label noise uncertainty set (in our case the x values are treated as certain, but toxicity or overall survival could be noisy depending on how the study measures it, and similar for other case studies?).

# Robust counterparts with probabilistic guarantees
- See paper_a_guarantee

# Other empirically / differently robust methods
- Wrapper (Maragno)
- C-MICL (Ovalle)
- RHS margin (simple baseline)
- Cutting planes (new proposed)
    - Separation over the bank
    - Cut generation and rollback
    - Termination (see paper_a_cp_notes)

# Empirical calibration
- Tuning a dial to reach proposed % coverage.

# Results / Case studies
- Each from previous literature.
- Each w/ conceptual model, data, predictive model
- Each w/ evaluation
    - Ground truth.
    - Robustness over training draws / sampled noise.

## Reactor
- Like Ovalle et al.
- Synthetically generated data points.
- Single constraint.
- Have GT equation.

## WFP
- Maragno et al. Also used in Ovalle et al.
- Not implemented yet?

## Chemotherapy Regimen Design
- Maragno et al.
- Multiple constraints.
- GT ensemble.

# Discussion
- Price of robustness
- Limitations and future work: models to select (can be new type of model, like mentioned in the survey paper), adaptive constraint learning.
- Feature noise
