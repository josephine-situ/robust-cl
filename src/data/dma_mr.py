"""Direct Methane Aromatization membrane reactor (DMA-MR) -- the mechanistic oracle.

This is the ground truth for the ``reactor`` problem instance: the C-MICL
regression case study (Ovalle et al. 2025, Sec. 5.1 / App. D.1), which is itself
the reactor of Carrasco & Lima (2017). Methane is converted to hydrogen and
benzene; hydrogen is drawn off through a membrane, so Le Chatelier drives the
equilibrium forward.

WHY THIS INSTANCE EXISTS. Every other oracle in this repo is FITTED -- gastric's
GT ensemble, synthetic's proxy ensemble -- so the judge carries an error of its
own, and a constrained optimum sits ON the constraint boundary by construction,
exactly where that error decides the verdict. Measured on synthetic (2026-08-21):
the proxy judge's error in the decision band has sd 0.039 against margins of
0.015-0.020, and it disagreed with the analytic truth on 5 of 5 nominal
decisions. Here the oracle is a system of ODEs, so ground-truth feasibility is
INTEGRATED, not predicted, and that failure mode is absent by construction.

PROVENANCE. The right-hand side ``_dma_mr_rhs`` is vendored from ``tests/dma_mr.py``
in the opyrability repository (CODES-group/opyrability, BSD-3), which is the code
the C-MICL authors cite for their data generation. It is copied rather than
imported because it lives in that repo's ``tests/`` directory and is NOT part of
the installed ``opyrability`` package -- nothing in the distribution exposes it.
Only the parameterisation is ours: opyrability's own entry points each vary a
SUBSET of the inputs (``dma_mr_design`` takes ``[L, dt]``, ``dma_mr_mvs`` takes
``[v0, v_He]``, both fixing the rest), whereas C-MICL varies all five. The
reaction kinetics, membrane transport and molar balances below are unchanged.

AGREEMENT with C-MICL Table 1, at RTOL = ATOL = 1e-10 (their five quoted rows):
diffs -0.50, -0.74, -0.46, -0.44, -0.96, i.e. a consistent **-1.6% offset**, not
scatter. It is NOT the negative-flow guard (flooring at 1e-9 as their jax path
does reproduces our numbers to 3dp); the likely cause is that they integrate on a
fixed ``linspace(0, L, 2000)`` grid via ``jax.experimental.ode.odeint`` rather
than adaptively. Untracked, and it does not matter for our use: what this instance
needs is A well-defined mechanistic ground truth that no fitted model has seen,
and 1.6% against an illustrative table rounded to 2dp does not threaten that.
Do NOT quote our F_C6H6 values as reproductions of theirs.

Their App. D.1 also reports "the variance of the output for the 1,000
data-points is 1.2871", which cannot be the variance of this quantity -- their own
Table 1 spans 28.26 to 42.57 across five rows. Read against Table 2's MSEs it is
clearly a STANDARDIZED target (MSE 0.0597 / 1.2871 -> R^2 0.954 for the ReLU NN,
0.864 for the RF), which is the scale-free check to hold ourselves to.

    Carrasco & Lima (2017), "Modeling and Design Optimization of Multifunctional
    Membrane Reactors for Direct Methane Aromatization", Ind. Eng. Chem. Res.
    Ovalle et al. (2025), "Conformal Mixed-Integer Constraint Learning", App. D.1.
"""

from __future__ import annotations

import numpy as np
import scipy.integrate as spint

# --- kinetics and fixed operating parameters (Carrasco & Lima) --------------
R_GAS = 8.314e6      # [Pa.cm^3/(K.mol)]
K1 = 0.04            # [1/s]
K1_INV = 6.40e6      # [cm^3/(s.mol)]
K2 = 4.20            # [1/s]
K2_INV = 56.38       # [cm^3/(s.mol)]
MM_B = 78.00         # benzene molar mass [g/mol]
Q_MEM = 3600 * 0.01e-4   # membrane permeance [mol/(h.cm^2.atm^0.25)]
SELEC = 1500         # membrane selectivity (H2 over the rest)
PT = 101325          # tube pressure [Pa] (1 atm)
PS = 101325          # shell pressure [Pa] (1 atm)

# Decision variables, in the order the ProblemInstance uses them, with the
# uniform sampling ranges of C-MICL App. D.1.
DECISION_NAMES = ("v0", "v_He", "T", "dt", "L")
DECISION_RANGES = {
    "v0":   (450.0, 1500.0),        # inlet CH4 volumetric flow  [cm^3/h]
    "v_He": (450.0, 1500.0),        # sweep gas volumetric flow  [cm^3/h]
    "T":    (997.18, 1348.12),      # operating temperature      [K]
    "dt":   (0.5, 2.0),             # tube diameter              [cm]
    "L":    (10.0, 100.0),          # reactor length             [cm]
}
# ODE solver tolerances, matching opyrability's dma_mr_design.
RTOL = ATOL = 1e-10

# Product-flow requirement: F_C6H6 >= PRODUCT_FLOOR  (C-MICL D.1g)
PRODUCT_FLOOR = 50.0


def _dma_mr_rhs(z, F, k1, k1_Inv, k2, k2_Inv, T, Q, Pt, v0, At, Ft0, Ps,
                v_He, F_He, dt, selec):
    """Molar balances along the reactor. Vendored unchanged from opyrability.

    ``F`` holds eight molar flowrates: tube side CH4/C2H6/H2/C6H6 (0-3) and
    shell side the same four (4-7). Negative flows are clipped because the
    ``^0.25`` partial-pressure terms would otherwise go complex in the first
    integration steps.
    """
    F = np.asarray(F, dtype=float).copy()
    F[F < 0] = 0.0

    Ft = F[0] + F[1] + F[2] + F[3]
    Fs = F[4] + F[5] + F[6] + F[7] + F_He
    if Ft <= 0 or Fs <= 0:
        return np.zeros(8)
    v = v0 * (Ft / Ft0)
    C = F[:4] / v

    P0t = (Pt / 101325) * (F[0] / Ft)
    P1t = (Pt / 101325) * (F[1] / Ft)
    P2t = (Pt / 101325) * (F[2] / Ft)
    P3t = (Pt / 101325) * (F[3] / Ft)
    P0s = (Ps / 101325) * (F[4] / Fs)
    P1s = (Ps / 101325) * (F[5] / Fs)
    P2s = (Ps / 101325) * (F[6] / Fs)
    P3s = (Ps / 101325) * (F[7] / Fs)

    if C[0] == 0:
        r0 = 0.0
    else:
        r0 = 3600 * k1 * C[0] * (1 - ((k1_Inv * C[1] * C[2] ** 2) / (k1 * C[0] ** 2)))
    if C[1] == 0:
        r1 = 0.0
    else:
        r1 = 3600 * k2 * C[1] * (1 - ((k2_Inv * C[3] * C[2] ** 3) / (k2 * C[1] ** 3)))

    # Molar balance adjustment with experimental data (bed voidage, efficiency).
    eff, vb = 0.9, 0.5
    Cat = (1 - vb) * eff

    dF0 = -Cat * r0 * At - (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * np.pi * dt
    dF1 = (0.5 * Cat * r0 * At - Cat * r1 * At
           - (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * np.pi * dt)
    dF2 = (Cat * r0 * At + Cat * r1 * At
           - Q * ((P2t ** 0.25) - (P2s ** 0.25)) * np.pi * dt)
    dF3 = ((1 / 3) * Cat * r1 * At
           - (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * np.pi * dt)
    dF4 = (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * np.pi * dt
    dF5 = (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * np.pi * dt
    dF6 = Q * ((P2t ** 0.25) - (P2s ** 0.25)) * np.pi * dt
    dF7 = (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * np.pi * dt
    return np.array([dF0, dF1, dF2, dF3, dF4, dF5, dF6, dF7])


def benzene_flow(u: np.ndarray) -> float:
    """Outlet benzene flow ``F_C6H6`` [mol/h] for one design ``u``.

    ``u = (v0, v_He, T, dt, L)`` in the units of :data:`DECISION_RANGES`. This is
    the oracle ``h(x)``: it INTEGRATES the reactor rather than predicting it, so
    it is exact up to solver tolerance and independent of every training row.
    Returns ``nan`` if the integration fails, which callers must treat as an
    unusable point rather than as a zero.
    """
    v0, v_He, T, dt, L = (float(t) for t in u)
    At = 0.25 * np.pi * (dt ** 2)         # cross-sectional area [cm^2]
    Ft0 = PT * v0 / (R_GAS * T)           # inlet molar flow, pure CH4 [mol/h]
    F_He = PS * v_He / (R_GAS * T)        # sweep gas molar flow [mol/h]

    y0 = np.hstack((Ft0, np.zeros(7)))
    try:
        sol = spint.solve_ivp(
            _dma_mr_rhs, (0.0, L), y0,
            args=(K1, K1_INV, K2, K2_INV, T, Q_MEM, PT, v0, At, Ft0, PS,
                  v_He, F_He, dt, SELEC),
            # opyrability's own jax entry point integrates at 1e-10, and
            # solve_ivp's defaults (1e-3/1e-6) are not enough: one C-MICL Table 1
            # row came out 3.83 high at the defaults, vs a worst case of 0.96
            # here. See AGREEMENT in the module docstring.
            rtol=RTOL, atol=ATOL,
        )
    except Exception:
        return float("nan")
    if not sol.success:
        return float("nan")
    # Tube-side benzene at the outlet, converted mol/h -> g/h via the molar mass
    # (C-MICL reports F_C6H6 on this scale; their Table 1 values are ~28-43).
    return float(sol.y.T[-1, 3] * 1000 * MM_B)


def benzene_flow_batch(U: np.ndarray) -> np.ndarray:
    """:func:`benzene_flow` over rows of ``U``; ``nan`` where integration failed."""
    U = np.atleast_2d(np.asarray(U, dtype=float))
    return np.array([benzene_flow(row) for row in U])


def sample_designs(n: int, seed: int = 42) -> np.ndarray:
    """``n`` designs drawn uniformly from :data:`DECISION_RANGES`, in
    :data:`DECISION_NAMES` order (C-MICL App. D.1)."""
    rng = np.random.RandomState(seed)
    return np.column_stack([
        rng.uniform(*DECISION_RANGES[k], size=n) for k in DECISION_NAMES
    ])
