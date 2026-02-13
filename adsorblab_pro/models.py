# models.py
"""
AdsorbLab Pro - Adsorption Models
=======================================================

Comprehensive collection of isotherm and kinetic models with:
- Non-linear regression with confidence intervals
- Bootstrap parameter estimation
- Model-specific diagnostics
- Physical parameter validation
- PRESS/Q² cross-validation

Isotherm Models (Single-Component):
- Langmuir (Type I, II)
- Freundlich
- Temkin
- Sips (Langmuir-Freundlich)

Multi-Component Competitive Models:
- Extended Langmuir (Butler-Ockrent)
- Extended Freundlich (Sheindorf-Rebhun-Sheintuch)
- Selectivity coefficient calculation

Kinetic Models (Pseudo-Models):
- Pseudo-First Order (PFO/Lagergren)
- Pseudo-Second Order (PSO/Ho-McKay)
- Revised PSO (rPSO/Bullen et al., 2021) - concentration-corrected
- Elovich
- Intraparticle Diffusion (IPD/Weber-Morris)

Diffusion Analysis:
- Biot number calculation
- Rate-limiting step identification (Boyd plot analysis)

"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import linregress
from scipy.stats import t as t_dist

# =============================================================================
# CONSTANTS - Import from central config for consistency
# =============================================================================
from .config import EPSILON_LOG, MAX_FIT_ITERATIONS, PI_SQUARED, R_GAS_CONSTANT

# Aliases for backward compatibility
EPSILON = EPSILON_LOG  # Use most conservative epsilon for model calculations
R_GAS = R_GAS_CONSTANT
MAX_ITER = MAX_FIT_ITERATIONS

# =============================================================================
# OPTIONAL STREAMLIT IMPORT (enables caching when available)
# =============================================================================
from adsorblab_pro.streamlit_compat import st, STREAMLIT_AVAILABLE as _STREAMLIT_AVAILABLE


# =============================================================================
# MODEL REGISTRY - Maps model names to functions for cache-friendly lookups
# =============================================================================
_MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a model function in the registry."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _MODEL_REGISTRY[name] = func
        return func

    return decorator


def get_model_by_name(name: str) -> Callable[..., Any] | None:
    """Retrieve a model function by name from the registry."""
    return _MODEL_REGISTRY.get(name)


# =============================================================================
# ISOTHERM MODELS - NON-LINEAR
# =============================================================================


@register_model("Langmuir")
def langmuir_model(
    Ce: NDArray[np.floating[Any]], qm: float, KL: float
) -> NDArray[np.floating[Any]]:
    """
    Langmuir isotherm: qe = (qm × KL × Ce) / (1 + KL × Ce)

    Assumes monolayer adsorption on homogeneous surface.

    Parameters
    ----------
    Ce : equilibrium concentration (mg/L)
    qm : maximum monolayer capacity (mg/g)
    KL : Langmuir affinity constant (L/mg)

    Reference: Langmuir, I. (1918). J. Am. Chem. Soc., 40(9), 1361-1403.
    """
    Ce = np.maximum(np.asarray(Ce), 0)
    qm = max(qm, EPSILON)
    KL = max(KL, EPSILON)

    denom = 1 + KL * Ce
    return (qm * KL * Ce) / np.maximum(denom, EPSILON)


@register_model("Freundlich")
def freundlich_model(
    Ce: NDArray[np.floating[Any]], KF: float, n_inv: float
) -> NDArray[np.floating[Any]]:
    """
    Freundlich isotherm: qe = KF × Ce^(1/n)

    Empirical model for heterogeneous surfaces.

    Parameters
    ----------
    Ce : equilibrium concentration (mg/L)
    KF : Freundlich capacity ((mg/g)(L/mg)^(1/n))
    n_inv : 1/n heterogeneity factor

    Note: n > 1 indicates favorable adsorption

    Reference: Freundlich, H. (1906). Z. Phys. Chem., 57, 385-470.
    """
    Ce = np.maximum(np.asarray(Ce), EPSILON)
    KF = max(KF, EPSILON)
    n_inv = np.clip(n_inv, 0.01, 5)

    return KF * np.power(Ce, n_inv)


@register_model("Temkin")
def temkin_model(Ce: NDArray[np.floating[Any]], B1: float, KT: float) -> NDArray[np.floating[Any]]:
    """
    Temkin isotherm: qe = B1 × ln(KT × Ce)

    Where B1 = RT/bT accounts for heat of adsorption.

    Parameters
    ----------
    Ce : equilibrium concentration (mg/L)
    B1 : RT/bT constant (mg/g)
    KT : Temkin binding constant (L/mg)

    Reference: Temkin, M.I. & Pyzhev, V. (1940). Acta Physicochim. USSR, 12, 217-222.
    """
    Ce = np.maximum(np.asarray(Ce), EPSILON)
    KT = max(KT, EPSILON)

    result = B1 * np.log(KT * Ce)

    # Do NOT clamp negatives to zero here.  Clamping distorts the residual
    # surface seen by curve_fit: the optimizer receives false feedback that a
    # physically-invalid parameter set is acceptable, and may converge on
    # parameters that produce qe < 0 across the data range — then report them
    # as a successful fit with converged=True.
    #
    # Raising ValueError instead:
    #   • during curve_fit: scipy propagates ValueError out of the optimizer,
    #     which the except block in _fit_model_core catches as a fitting failure.
    #   • post-fit (y_pred = model_func(x_data, *popt)): the same except block
    #     catches it, so a converged popt that still produces negative qe also
    #     surfaces as {"converged": False} rather than silently wrong output.
    #
    # Physical root cause: qe < 0 means KT × Ce < 1 somewhere in the data,
    # i.e. the binding constant KT is too low for the measured Ce range.
    # The right remedies are tighter lower bounds on KT, better p0, or a
    # different model — not clamping.
    if np.any(result < 0):
        n_negative = int(np.sum(result < 0))
        raise ValueError(
            f"Temkin model produced {n_negative} negative qe value(s) "
            f"(B1={B1:.4g}, KT={KT:.4g}). "
            f"KT \u00d7 Ce must be > 1 across the full data range. "
            f"Increase the lower bound on KT or adjust initial parameters."
        )

    return result


@register_model("Sips")
def sips_model(
    Ce: NDArray[np.floating[Any]], qm: float, Ks: float, ns: float
) -> NDArray[np.floating[Any]]:
    """
    Sips isotherm (Langmuir-Freundlich): qe = qm × (Ks×Ce)^ns / (1 + (Ks×Ce)^ns)

    Combines Langmuir and Freundlich; reduces to Langmuir when ns = 1.

    Parameters
    ----------
    qm : maximum capacity (mg/g)
    Ks : Sips affinity constant (L/mg)^ns
    ns : heterogeneity parameter

    Reference: Sips, R. (1948). J. Chem. Phys., 16, 490-495.
    """
    Ce = np.maximum(np.asarray(Ce), EPSILON)
    qm = max(qm, EPSILON)
    Ks = max(Ks, EPSILON)
    ns = np.clip(ns, 0.1, 5)

    # Naive form:  qm * (Ks·Ce)^ns / (1 + (Ks·Ce)^ns)
    #
    # Failure mode: np.power(Ks*Ce, ns) overflows to inf when Ks*Ce is large
    # and ns > 1 (e.g. Ks=10, Ce=1e70, ns=3 → power_term=inf).
    # NumPy then evaluates inf/(1+inf) = inf/inf = nan, which silently
    # corrupts every downstream statistic while curve_fit reports convergence.
    #
    # The fraction x/(1+x) is the logistic sigmoid evaluated at u = ln(x).
    # Substituting u = ns·ln(Ks·Ce) rewrites the model as qm·eᵘ/(1+eᵘ),
    # which has a standard stable two-branch evaluation:
    #
    #   u >= 0  →  qm · 1          / (1 + e^{-u})   [e^{-|u|} ∈ (0,1]]
    #   u <  0  →  qm · e^{-|u|}   / (1 + e^{-|u|}) [e^{-|u|} ∈ (0,1))
    #
    # safe_exp = e^{-|u|} is always in (0, 1], so neither branch can overflow
    # or produce nan.  The result is numerically identical to the naive form
    # everywhere the naive form is finite (max difference ~1e-16).
    #
    # Physical interpretation: as Ks·Ce → ∞ the model correctly saturates at
    # qm (all sites occupied); the stable form returns exactly qm there.
    u = ns * np.log(Ks * Ce)          # finite: Ks >= EPSILON, Ce >= EPSILON
    safe_exp = np.exp(-np.abs(u))     # always in (0, 1] — overflow impossible
    return qm * np.where(u >= 0, 1.0 / (1.0 + safe_exp), safe_exp / (1.0 + safe_exp))


# =============================================================================
# MULTI-COMPONENT COMPETITIVE ADSORPTION MODELS
# =============================================================================
# Real wastewaters contain multiple contaminants competing for adsorption sites.
# These models predict competitive adsorption from single-component parameters.
# =============================================================================


def extended_langmuir_multicomponent(
    Ce_i: np.ndarray, qm_i: float, KL_i: float, Ce_all: list[np.ndarray], KL_all: list[float]
) -> NDArray[np.floating[Any]]:
    """
    Extended Langmuir model for competitive multi-component adsorption.

    Predicts adsorption of component i in the presence of competing adsorbates.
    Uses single-component Langmuir parameters for each species.

    Model equation:
        qe_i = (qm_i × KL_i × Ce_i) / (1 + Σ(KL_j × Ce_j))

    Parameters
    ----------
    Ce_i : np.ndarray
        Equilibrium concentration of component i (mg/L)
    qm_i : float
        Maximum adsorption capacity for component i from single-component
        isotherm (mg/g)
    KL_i : float
        Langmuir affinity constant for component i from single-component
        isotherm (L/mg)
    Ce_all : List[np.ndarray]
        List of equilibrium concentrations for ALL components including i
        [Ce_1, Ce_2, ..., Ce_n]
    KL_all : List[float]
        List of Langmuir constants for ALL components including i
        [KL_1, KL_2, ..., KL_n]

    Returns
    -------
    np.ndarray
        Equilibrium adsorption capacity for component i (mg/g)

    Notes
    -----
    - Assumes ideal behavior (no interaction effects between adsorbates)
    - Requires single-component isotherm parameters determined separately
    - Most accurate when adsorbates have similar molecular properties
    - For non-ideal systems, consider IAST (Ideal Adsorbed Solution Theory)

    Example
    -------
    >>> # Two-component system: dye + heavy metal
    >>> Ce_dye = np.array([10, 20, 30, 40, 50])  # mg/L
    >>> Ce_metal = np.array([5, 10, 15, 20, 25])  # mg/L
    >>> qm_dye, KL_dye = 100, 0.05     # From single-component isotherms
    >>> qm_metal, KL_metal = 50, 0.1
    >>>
    >>> # Predict dye adsorption with metal competition
    >>> qe_dye = extended_langmuir_multicomponent(
    ...     Ce_i=Ce_dye, qm_i=qm_dye, KL_i=KL_dye,
    ...     Ce_all=[Ce_dye, Ce_metal], KL_all=[KL_dye, KL_metal]
    ... )

    Reference
    ---------
    Butler, J.A.V. & Ockrent, C. (1930). Studies in electrocapillarity. Part III.
    The surface tensions of solutions containing two surface-active solutes.
    J. Phys. Chem., 34(12), 2841-2859.
    """
    Ce_i = np.maximum(np.asarray(Ce_i), EPSILON)
    qm_i = max(qm_i, EPSILON)
    KL_i = max(KL_i, EPSILON)

    # Calculate competition term: 1 + Σ(KL_j × Ce_j)
    #
    # Both Ce_j and KL_j floor at EPSILON, not 0.  Using 0 as the floor was
    # inconsistent with:
    #   • Ce_i (line above): np.maximum(..., EPSILON)
    #   • KL_i (line above): max(..., EPSILON)
    #   • Every other Ce / KL clamp in this file
    #
    # Allowing KL_j = 0 silently zeroes out a competing species' contribution
    # to the denominator, under-counting competition and over-predicting qe_i.
    # Allowing Ce_j = 0 is safe arithmetically but breaks the invariant that
    # all concentrations are strictly positive, making the model behave
    # differently for component j than for component i with identical inputs.
    denominator: NDArray[np.floating[Any]] = np.ones_like(Ce_i)
    for Ce_j, KL_j in zip(Ce_all, KL_all):
        Ce_j_arr = np.maximum(np.asarray(Ce_j), EPSILON)
        KL_j_val = max(KL_j, EPSILON)
        denominator = denominator + KL_j_val * Ce_j_arr

    numerator = qm_i * KL_i * Ce_i

    return numerator / np.maximum(denominator, EPSILON)


def extended_freundlich_multicomponent(
    Ce_i: np.ndarray,
    Kf_i: float,
    n_i: float,
    Ce_all: list[np.ndarray],
    Kf_all: list[float],
    n_all: list[float],
) -> NDArray[np.floating[Any]]:
    """
    Extended Freundlich model for competitive multi-component adsorption.

    Sheindorf-Rebuhn-Sheintuch (SRS) modification of Freundlich for
    multi-component systems.

    Model equation:
        qe_i = Kf_i × Ce_i × (Σ(aij × Ce_j))^(1/n_i - 1)

    Where aij are competition coefficients (aij = 1 for i=j assumed here).

    Parameters
    ----------
    Ce_i : np.ndarray
        Equilibrium concentration of component i (mg/L)
    Kf_i : float
        Freundlich constant for component i (mg/g)/(mg/L)^(1/n)
    n_i : float
        Freundlich exponent for component i
    Ce_all : List[np.ndarray]
        List of equilibrium concentrations for all components
    Kf_all : List[float]
        List of Freundlich constants for all components
    n_all : List[float]
        List of Freundlich exponents for all components

    Returns
    -------
    np.ndarray
        Equilibrium adsorption capacity for component i (mg/g)

    Notes
    -----
    - Less thermodynamically rigorous than Extended Langmuir
    - Useful for heterogeneous surfaces with competitive adsorption
    - Competition coefficients calculated as: aij = (Kf_i/Kf_j)^(n_j/n_i)
    - Higher aij indicates component i outcompetes component j

    Reference
    ---------
    Sheindorf, C., Rebhun, M., & Sheintuch, M. (1981). A Freundlich-type
    multicomponent isotherm. J. Colloid Interface Sci., 79(1), 136-142.
    """
    Ce_i = np.maximum(np.asarray(Ce_i), EPSILON)
    Kf_i = max(Kf_i, EPSILON)
    n_i = max(n_i, 0.1)

    # Calculate weighted sum using competition coefficients
    # aij = (Kf_i / Kf_j)^(n_j/n_i) - relative affinity of component i vs j
    #
    # Ce_j floors at EPSILON (not 0) to match Ce_i above and every other
    # concentration clamp in this file.
    Ce_weighted_sum = np.zeros_like(Ce_i)

    for j, Ce_j_raw in enumerate(Ce_all):
        Ce_j_safe = np.maximum(np.asarray(Ce_j_raw), EPSILON)
        Kf_j = max(Kf_all[j], EPSILON)
        n_j = max(n_all[j], 0.1)

        # Competition coefficient (aij = 1 when i = j)
        aij = np.power(Kf_i / Kf_j, n_j / n_i)
        Ce_weighted_sum = Ce_weighted_sum + aij * Ce_j_safe

    # SRS equation with competition coefficients
    exponent = (1 / n_i) - 1
    competition_term = np.power(np.maximum(Ce_weighted_sum, EPSILON), exponent)

    return Kf_i * Ce_i * competition_term


def calculate_selectivity_coefficient(
    qe_i: float | NDArray[np.floating[Any]],
    Ce_i: float | NDArray[np.floating[Any]],
    qe_j: float | NDArray[np.floating[Any]],
    Ce_j: float | NDArray[np.floating[Any]],
) -> float | NDArray[np.floating[Any]]:
    """
    Calculate selectivity coefficient between two adsorbates.

    The selectivity coefficient (separation factor) indicates the
    preferential adsorption of component i over component j.

    Model equation:
        α_ij = (qe_i / Ce_i) / (qe_j / Ce_j) = (qe_i × Ce_j) / (qe_j × Ce_i)

    Parameters
    ----------
    qe_i : float
        Equilibrium adsorption capacity of component i (mg/g)
    Ce_i : float
        Equilibrium concentration of component i (mg/L)
    qe_j : float
        Equilibrium adsorption capacity of component j (mg/g)
    Ce_j : float
        Equilibrium concentration of component j (mg/L)

    Returns
    -------
    float
        Selectivity coefficient α_ij
        - α_ij > 1: Preferential adsorption of component i
        - α_ij < 1: Preferential adsorption of component j
        - α_ij = 1: No selectivity (equal preference)

    Example
    -------
    >>> # Compare MB vs Pb adsorption selectivity
    >>> alpha = calculate_selectivity_coefficient(
    ...     qe_i=80, Ce_i=10,   # MB: high uptake, low Ce remaining
    ...     qe_j=20, Ce_j=20    # Pb: low uptake, high Ce remaining
    ... )
    >>> print(f"α_MB/Pb = {alpha:.2f}")  # > 1 means MB preferred
    """
    # Convert to numpy arrays for consistent handling
    qe_i = np.asarray(qe_i)
    Ce_i = np.asarray(Ce_i)
    qe_j = np.asarray(qe_j)
    Ce_j = np.asarray(Ce_j)

    # Handle invalid values
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = (qe_i * Ce_j) / (qe_j * Ce_i)

    # Replace invalid values with NaN
    invalid_mask = (Ce_i <= 0) | (Ce_j <= 0) | (qe_j <= 0)
    if np.any(invalid_mask):
        alpha = np.where(invalid_mask, np.nan, alpha)

    # Return scalar if inputs were scalars
    if alpha.ndim == 0:
        return float(alpha)
    return alpha


# =============================================================================
# KINETIC MODELS - NON-LINEAR
# =============================================================================


@register_model("PFO")
def pfo_model(t: NDArray[np.floating[Any]], qe: float, k1: float) -> NDArray[np.floating[Any]]:
    """
    Pseudo-First Order kinetic model (Lagergren):
    qt = qe × (1 - exp(-k1 × t))

    Parameters
    ----------
    t : time (min)
    qe : equilibrium capacity (mg/g)
    k1 : rate constant (min⁻¹)

    Reference: Lagergren, S. (1898). Kungliga Svenska Vetenskapsakademiens Handlingar, 24, 1-39.
    """
    t = np.maximum(np.asarray(t), 0)
    qe = max(qe, EPSILON)
    k1 = max(k1, EPSILON)

    return qe * (1 - np.exp(-k1 * t))


@register_model("PSO")
def pso_model(t: NDArray[np.floating[Any]], qe: float, k2: float) -> NDArray[np.floating[Any]]:
    """
    Pseudo-Second Order kinetic model (Ho-McKay):
    qt = (qe² × k2 × t) / (1 + qe × k2 × t)

    Parameters
    ----------
    t : time (min)
    qe : equilibrium capacity (mg/g)
    k2 : rate constant (g/(mg·min))

    Reference: Ho, Y.S. & McKay, G. (1999). Process Biochem., 34, 451-465.

    WARNING: PSO model fit does NOT imply chemisorption mechanism.
    High R² for PSO is observed in ~90% of kinetic studies regardless
    of actual mechanism—a statistical artifact, not mechanistic evidence.
    See: Hubbe et al. (2019). BioResources, 14(3), 7582-7626.
    """
    t = np.maximum(np.asarray(t), 0)
    qe = max(qe, EPSILON)
    k2 = max(k2, EPSILON)

    numer = qe**2 * k2 * t
    denom = 1 + qe * k2 * t

    return numer / np.maximum(denom, EPSILON)


@register_model("rPSO")
def revised_pso_model(
    t: NDArray[np.floating[Any]], qe: float, k2: float, C0: float, m: float, V: float
) -> NDArray[np.floating[Any]]:
    """
    Revised Pseudo-Second-Order (rPSO) kinetic model.

    Incorporates concentration dependence for improved prediction across
    varying experimental conditions. Addresses the well-documented artifact
    where standard PSO appears to fit ~90% of kinetic data.

    Model equation:
        qt = (qe² × k2 × t) / (1 + qe × k2 × t × φ)

    where φ = 1 + (qe × m) / (C0 × V) is the concentration correction factor

    Parameters
    ----------
    t : np.ndarray
        Time (min)
    qe : float
        Equilibrium adsorption capacity (mg/g)
    k2 : float
        Rate constant (g/(mg·min))
    C0 : float
        Initial adsorbate concentration (mg/L)
    m : float
        Adsorbent mass (g)
    V : float
        Solution volume (L)

    Returns
    -------
    np.ndarray
        Adsorption capacity at time t (mg/g)

    Notes
    -----
    The rPSO model reduces residual sum of squares by ~66% compared to
    standard PSO when using a single rate constant to model multiple
    experiments with varying initial conditions.

    The standard PSO model's near-universal "best fit" (~90% of studies)
    is a methodological artifact, not evidence of chemisorption mechanism.

    This function is registered as ``"rPSO"`` in the model registry for
    API discoverability.  Because the fitting parameters are only (qe, k2),
    use :func:`revised_pso_model_fixed_conditions` to create a
    curve_fit-compatible closure with fixed C0, m, V; those closures are
    automatically registered for Streamlit caching.

    Reference
    ---------
    Bullen, J.C., Saleesongsom, S., Galber, K., & Weiss, D.J. (2021).
    A Revised Pseudo-Second-Order Kinetic Model for Adsorption, Sensitive
    to Changes in Adsorbate and Adsorbent Concentrations.
    Langmuir, 37(10), 3189-3201. DOI: 10.1021/acs.langmuir.1c00142
    """
    t = np.maximum(np.asarray(t), 0)
    qe = max(qe, EPSILON)
    k2 = max(k2, EPSILON)
    C0 = max(C0, EPSILON)
    m = max(m, EPSILON)
    V = max(V, EPSILON)

    # Concentration correction factor (φ)
    # Accounts for the effect of changing solution concentration
    # as adsorption progresses
    phi = 1 + (qe * m) / (C0 * V)

    numerator = qe**2 * k2 * t
    denominator = 1 + qe * k2 * t * phi

    return numerator / np.maximum(denominator, EPSILON)


def revised_pso_model_fixed_conditions(C0: float, m: float, V: float) -> Callable:
    """
    Create an rPSO model function with fixed experimental conditions.

    This is useful for fitting when C0, m, and V are known constants
    and only qe and k2 need to be estimated.

    The returned closure is automatically registered in the model registry
    under a condition-specific key (e.g. ``"rPSO_C0=100_m=0.5_V=0.1"``)
    so that :func:`fit_model_with_ci` and
    :func:`~adsorblab_pro.utils.bootstrap_confidence_intervals` can use
    Streamlit caching for it.

    Parameters
    ----------
    C0 : float
        Initial concentration (mg/L)
    m : float
        Adsorbent mass (g)
    V : float
        Solution volume (L)

    Returns
    -------
    Callable
        Model function with signature f(t, qe, k2) -> qt

    Example
    -------
    >>> model = revised_pso_model_fixed_conditions(C0=100, m=0.5, V=0.1)
    >>> qt = model(t_data, qe=50, k2=0.01)
    """
    C0 = max(C0, EPSILON)
    m = max(m, EPSILON)
    V = max(V, EPSILON)

    def model(t: NDArray[np.floating[Any]], qe: float, k2: float) -> NDArray[np.floating[Any]]:
        return revised_pso_model(t, qe, k2, C0, m, V)

    # Register the closure so Streamlit caching works in fit_model_with_ci
    # and bootstrap_confidence_intervals.  The key encodes the experimental
    # conditions so different (C0, m, V) combinations get separate entries.
    registry_key = f"rPSO_C0={C0}_m={m}_V={V}"
    _MODEL_REGISTRY[registry_key] = model

    return model


@register_model("Elovich")
def elovich_model(
    t: NDArray[np.floating[Any]], alpha: float, beta: float
) -> NDArray[np.floating[Any]]:
    """
    Elovich kinetic model:
    qt = (1/β) × ln(1 + αβt)

    Suitable for chemisorption on heterogeneous surfaces.

    Parameters
    ----------
    t : time (min)
    alpha : initial rate (mg/(g·min))
    beta : desorption constant (g/mg)

    Reference: Zeldowitsch, J. (1934). Acta Physicochim. USSR, 1, 364-449.
    """
    t = np.maximum(np.asarray(t), EPSILON)
    alpha = max(alpha, EPSILON)
    beta = max(beta, EPSILON)

    return (1 / beta) * np.log(1 + alpha * beta * t)


@register_model("IPD")
def ipd_model(t: NDArray[np.floating[Any]], kid: float, C: float) -> NDArray[np.floating[Any]]:
    """
    Intraparticle Diffusion model (Weber-Morris):
    qt = kid × t^0.5 + C

    Parameters
    ----------
    t : time (min)
    kid : intraparticle diffusion rate (mg/(g·min^0.5))
    C : boundary layer constant (mg/g)

    Reference: Weber, W.J. & Morris, J.C. (1963). J. Sanit. Eng. Div. Am. Soc. Civ. Eng., 89, 31-60.
    """
    t = np.maximum(np.asarray(t), 0)  # t=0 is the valid initial boundary condition

    # kid=0 is physically meaningless (zero diffusion rate) and numerically
    # degenerate: the model collapses to qt = C (a flat line), making kid
    # completely unidentifiable.  During curve_fit, any negative trial value
    # is silently mapped to the same 0, so the optimizer sees a zero gradient
    # across the entire negative half of the kid axis — pcov[0,0] then diverges
    # to inf, corrupting SEs and CIs while still reporting converged=True.
    #
    # Clamping to EPSILON preserves a real (infinitesimal) slope so the
    # Jacobian remains informative.  Note: C is intentionally left unclamped —
    # a negative intercept is physically meaningful in Weber-Morris analysis
    # (it indicates pore diffusion, not film diffusion, is rate-limiting).
    kid = max(kid, EPSILON)

    return kid * np.sqrt(t) + C


def calculate_biot_number(kf: float, Dp: float, r: float) -> float:
    """
    Calculate the Biot number for mass transfer.

    The Biot number indicates the relative importance of external
    (film) vs. internal (pore) diffusion resistance.

    Model equation:
        Bi = kf × r / Dp

    Parameters
    ----------
    kf : float
        Film mass transfer coefficient (cm/min or appropriate units)
    Dp : float
        Pore diffusion coefficient (cm²/min)
    r : float
        Particle radius (cm)

    Returns
    -------
    float
        Biot number (dimensionless)

    Interpretation
    --------------
    - Bi >> 1: Internal (pore) diffusion controls
    - Bi << 1: External (film) diffusion controls
    - Bi ≈ 1: Both mechanisms are important

    Typical guidance:
    - Bi > 100: Pore diffusion dominates (>90% of resistance)
    - Bi < 0.1: Film diffusion dominates (>90% of resistance)
    - 0.1 < Bi < 100: Mixed control

    Example
    -------
    >>> Bi = calculate_biot_number(kf=0.01, Dp=1e-8, r=0.05)
    >>> print(f"Bi = {Bi:.0f}")  # If >> 1, pore diffusion controls

    Reference
    ---------
    Ruthven, D.M. (1984). Principles of Adsorption and Adsorption Processes.
    """
    kf = max(kf, EPSILON)
    Dp = max(Dp, EPSILON)
    r = max(r, EPSILON)

    return kf * r / Dp


def identify_rate_limiting_step(
    t: NDArray[np.floating[Any]],
    qt: NDArray[np.floating[Any]],
    qe: float,
    particle_radius: float | None = None,
) -> dict[str, Any]:
    """
    Analyze kinetic data to identify the rate-limiting step.

    Performs multiple diagnostic tests to determine whether film diffusion,
    pore diffusion, or mixed control governs the adsorption kinetics.

    Parameters
    ----------
    t : np.ndarray
        Time data (min)
    qt : np.ndarray
        Adsorption capacity data (mg/g)
    qe : float
        Equilibrium capacity (mg/g), use qt[-1] if unknown
    particle_radius : float, optional
        Particle radius (cm) for diffusion coefficient estimation

    Returns
    -------
    dict
        Diagnostic results containing:
        - 'F': Fractional attainment array
        - 'weber_morris': Weber-Morris IPD analysis results
        - 'boyd_plot': Boyd plot analysis results
        - 'mechanism_suggestion': Most likely rate-limiting mechanism
        - 'confidence': Confidence level in suggestion

    Notes
    -----
    Diagnostic criteria used:
    1. Weber-Morris plot (qt vs t^0.5):
       - Linear through origin → pore diffusion controls
    2. Boyd plot (Bt vs t):
       - Linear through origin → film diffusion controls
       - Linear, non-zero intercept → pore diffusion controls

    Reference
    ---------
    Qiu, H., et al. (2009). Critical review in adsorption kinetic models.
    J. Zhejiang Univ. Sci. A, 10(5), 716-724.
    """
    t = np.asarray(t)
    qt = np.asarray(qt)

    valid = (t > 0) & (qt > 0)
    t = t[valid]
    qt = qt[valid]

    if len(t) < 5:
        return {"error": "Insufficient data points (need at least 5)"}

    qe = max(qe, qt.max())
    F = qt / qe
    F = np.clip(F, EPSILON, 1 - EPSILON)

    results: dict[str, Any] = {"F": F, "t": t, "qt": qt, "qe": qe}

    # 1. Weber-Morris IPD analysis
    sqrt_t = np.sqrt(t)
    slope_wm, intercept_wm, r_wm, _, _ = linregress(sqrt_t, qt)
    weber_morris: dict[str, Any] = {
        "kid": slope_wm,
        "C": intercept_wm,
        "r_squared": r_wm**2,
        "passes_origin": abs(intercept_wm) < 0.1 * qt.max(),
    }
    results["weber_morris"] = weber_morris

    # Calculate effective diffusion coefficient if particle radius provided
    if particle_radius is not None and particle_radius > 0:
        # D_eff = (kid² × r²) / (6 × qe²) in cm²/min
        kid = slope_wm
        D_eff = (kid**2 * particle_radius**2) / (6 * qe**2)
        weber_morris["D_eff_cm2_min"] = D_eff
        weber_morris["D_eff_cm2_s"] = D_eff / 60  # Convert to cm²/s
        weber_morris["particle_radius_cm"] = particle_radius
    # 2. Boyd plot analysis
    # Bt = -0.4977 - ln(1-F) for F < 0.85
    # Bt ≈ 2π - 2π²F/3 for F > 0.85 (approximation)
    Bt = np.where(F < 0.85, -0.4977 - np.log(1 - F), 2 * np.pi - 2 * PI_SQUARED * F / 3)

    slope_boyd, intercept_boyd, r_boyd, _, _ = linregress(t, Bt)
    boyd_plot: dict[str, Any] = {
        "slope": slope_boyd,
        "intercept": intercept_boyd,
        "r_squared": r_boyd**2,
        "passes_origin": abs(intercept_boyd) < 0.1 * Bt.max(),
    }
    results["boyd_plot"] = boyd_plot

    # 3. Mechanism identification
    wm_origin = weber_morris["passes_origin"]
    boyd_origin = boyd_plot["passes_origin"]
    wm_r2 = weber_morris["r_squared"]
    boyd_r2 = boyd_plot["r_squared"]

    if boyd_origin and boyd_r2 > 0.95:
        mechanism = "Film diffusion (external mass transfer)"
        confidence = "High" if boyd_r2 > 0.98 else "Medium"
    elif wm_origin and wm_r2 > 0.95:
        mechanism = "Pore diffusion (intraparticle diffusion)"
        confidence = "High" if wm_r2 > 0.98 else "Medium"
    elif not wm_origin and wm_r2 > 0.90:
        if weber_morris["C"] > 0:
            mechanism = "Pore diffusion with boundary layer effect"
        else:
            mechanism = "Mixed film and pore diffusion"
        confidence = "Medium"
    else:
        mechanism = "Mixed or complex mechanism"
        confidence = "Low"

    results["mechanism_suggestion"] = mechanism
    results["confidence"] = confidence

    # Additional guidance
    recommendations: list[str] = []
    if confidence == "Low":
        recommendations.append("Perform particle size variation study to confirm mechanism")
    if not wm_origin:
        recommendations.append("Non-zero Weber-Morris intercept suggests boundary layer resistance")
    if "Mixed" in mechanism:
        recommendations.append("Consider double-exponential model for two-site kinetics")
    if particle_radius is None:
        recommendations.append(
            "Provide particle radius to estimate effective diffusion coefficient"
        )
    results["recommendations"] = recommendations

    return results


# =============================================================================
# CACHED MODEL FITTING
# =============================================================================


def _fit_model_core(
    model_func: Callable[..., NDArray[np.floating[Any]]],
    x_data: NDArray[np.floating[Any]],
    y_data: NDArray[np.floating[Any]],
    p0: list[float],
    bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    param_names: list[str] | None = None,
    confidence: float = 0.95,
) -> dict[str, Any] | None:
    """
    Core model fitting logic (internal use).

    This is the actual fitting implementation, separated from caching logic.
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    n = len(x_data)
    n_params = len(p0)

    if n < n_params + 2:
        return None

    try:
        # Suppress OptimizeWarning during curve_fit (covariance estimation issues);
        # RuntimeWarnings from model_func (overflow, underflow) are intentionally
        # left unfiltered so they remain visible to callers.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=OptimizeWarning,
                message="Covariance of the parameters could not be estimated",
            )
            if bounds:
                popt, pcov = curve_fit(
                    model_func, x_data, y_data, p0=p0, bounds=bounds, maxfev=MAX_ITER
                )
            else:
                popt, pcov = curve_fit(model_func, x_data, y_data, p0=p0, maxfev=MAX_ITER)

        # Standard errors
        perr = np.sqrt(np.diag(pcov))

        # Guard against a singular / ill-conditioned covariance matrix.
        # scipy sets pcov entries to `inf` when the Jacobian is rank-deficient
        # or the optimisation did not converge; np.sqrt then propagates those
        # infinities (and any NaNs) into perr.  Letting them through means every
        # CI, SE, and the `converged: True` flag reported to the caller would be
        # silently wrong.  Raising here lets the existing except-block return a
        # clean {"converged": False, "error": ...} dict instead.
        if not np.all(np.isfinite(perr)):
            raise RuntimeError(
                "Covariance matrix is singular or infinite; fitting did not "
                "converge reliably. Try different initial parameters (p0), "
                "tighter bounds, or a simpler model."
            )

        # Predictions and residuals
        y_pred = model_func(x_data, *popt)
        residuals = y_data - y_pred

        # Sum of squares
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)

        # R² and Adjusted R²
        r_squared = 1 - ss_res / ss_tot if ss_tot > EPSILON else 0
        if n > n_params + 1:
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_params - 1)
        else:
            adj_r_squared = r_squared

        # RMSE
        rmse = np.sqrt(np.sum(residuals**2) / n)

        # Chi-squared
        y_pred_safe = np.where(np.abs(y_pred) < EPSILON, EPSILON, y_pred)
        chi_sq = np.sum(residuals**2 / np.abs(y_pred_safe))

        # AIC, AICc, BIC
        if ss_res > 0:
            log_lik = -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(ss_res / n) - n / 2
            aic = -2 * log_lik + 2 * n_params
            aicc = (
                aic + (2 * n_params * (n_params + 1)) / (n - n_params - 1)
                if n > n_params + 1
                else aic
            )
            bic = -2 * log_lik + n_params * np.log(n)
        else:
            aic = aicc = bic = np.inf

        # Confidence intervals
        dof = n - n_params
        # Guard before calling t_dist.ppf.  When dof <= 0, ppf returns NaN
        # and silently corrupts every CI in the returned dict.  When dof == 1
        # the fit is technically determined (zero residual freedom) and ppf
        # returns a finite but Cauchy-wide value (~12.7 at 95%) that is
        # statistically meaningless.  Both cases are symptoms of the same
        # root problem: too few data points relative to the number of free
        # parameters.
        #
        # Note: the early-return guard at the top of this function (n < n_params + 2)
        # means dof >= 2 under normal execution.  This check is intentionally
        # kept here as a defence-in-depth measure — matching the adj_r_squared
        # guard above — so that any future loosening of the early guard cannot
        # silently produce NaN CIs.  The resulting ValueError is caught by the
        # existing except block and surfaces as {"converged": False, "error": ...}.
        if dof <= 1:
            raise ValueError(
                f"Insufficient data: need at least n_params + 2 points for "
                f"meaningful confidence intervals, but got n={n}, "
                f"n_params={n_params}, dof={dof}."
            )
        t_val = t_dist.ppf(1 - (1 - confidence) / 2, dof)

        ci_95 = {}
        params_dict = {}
        if param_names is None:
            param_names = [f"p{i}" for i in range(n_params)]

        for i, name in enumerate(param_names):
            params_dict[name] = popt[i]
            params_dict[f"{name}_se"] = perr[i]
            ci_95[name] = (popt[i] - t_val * perr[i], popt[i] + t_val * perr[i])

        return {
            "params": params_dict,
            "popt": popt,
            "pcov": pcov,
            "perr": perr,
            "ci_95": ci_95,
            "y_pred": y_pred,
            "y_data": y_data,
            "x_data": x_data,
            "residuals": residuals,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "rmse": rmse,
            "chi_squared": chi_sq,
            "aic": aic,
            "aicc": aicc,
            "bic": bic,
            "sse": ss_res,
            "sst": ss_tot,
            "n_points": n,
            "num_params": n_params,
            "dof": dof,
            "converged": True,
        }

    except (RuntimeError, ValueError, TypeError) as e:
        return {"converged": False, "error": str(e)}


def _fit_model_cached_impl(
    model_name: str,
    x_data_tuple: tuple[float, ...],
    y_data_tuple: tuple[float, ...],
    p0_tuple: tuple[float, ...],
    bounds_lower: tuple[float, ...] | None = None,
    bounds_upper: tuple[float, ...] | None = None,
    param_names_tuple: tuple[str, ...] | None = None,
    confidence: float = 0.95,
) -> dict[str, Any] | None:
    """
    Cached model fitting with hashable arguments.

    This function accepts only hashable types to enable Streamlit caching.
    """
    # Resolve model function from name
    model_func = get_model_by_name(model_name)
    if model_func is None:
        return {"converged": False, "error": f"Unknown model: {model_name}"}

    # Convert tuples back to arrays/lists
    x_data = np.array(x_data_tuple)
    y_data = np.array(y_data_tuple)
    p0 = list(p0_tuple)
    if bounds_lower is not None and bounds_upper is not None:
        bounds = (bounds_lower, bounds_upper)
    else:
        bounds = None
    param_names = list(param_names_tuple) if param_names_tuple else None

    return _fit_model_core(model_func, x_data, y_data, p0, bounds, param_names, confidence)


# Apply caching if Streamlit is available
if _STREAMLIT_AVAILABLE:
    fit_model_with_ci_cached = st.cache_data(show_spinner=False)(_fit_model_cached_impl)
else:
    fit_model_with_ci_cached = _fit_model_cached_impl


def fit_model_with_ci(
    model_func: Callable[..., NDArray[np.floating[Any]]],
    x_data: NDArray[np.floating[Any]],
    y_data: NDArray[np.floating[Any]],
    p0: list[float],
    bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    param_names: list[str] | None = None,
    confidence: float = 0.95,
    use_cache: bool = True,
) -> dict[str, Any] | None:
    """
    Fit model with confidence intervals for parameters.

    This function automatically uses caching when possible to avoid
    recomputing results for the same data. Caching is enabled when:
    - Streamlit is available
    - use_cache=True (default)
    - The model function is registered in the model registry

    Parameters
    ----------
    model_func : callable
        Model function f(x, *params). For caching to work, this should
        be a registered model (Langmuir, Freundlich, etc.)
    x_data : np.ndarray
        Independent variable data
    y_data : np.ndarray
        Dependent variable data
    p0 : list
        Initial parameter guesses
    bounds : tuple, optional
        Parameter bounds ((lower,...), (upper,...))
    param_names : list, optional
        Names for parameters
    confidence : float
        Confidence level (default 0.95)
    use_cache : bool
        Whether to use caching (default True). Set to False to force
        recomputation.

    Returns
    -------
    dict
        Comprehensive fitting results including:
        - params: fitted parameters with standard errors
        - popt, pcov, perr: scipy curve_fit outputs
        - ci_95: 95% confidence intervals
        - r_squared, adj_r_squared: goodness of fit
        - aic, aicc, bic: information criteria
        - converged: bool indicating success
    """
    # Try to find model name for caching
    model_name = None
    if use_cache and _STREAMLIT_AVAILABLE:
        # Search for model function in registry
        for name, func in _MODEL_REGISTRY.items():
            if func is model_func:
                model_name = name
                break

    # Use cached version if model is registered
    if model_name is not None:
        x_tuple = tuple(np.asarray(x_data).round(10).tolist())
        y_tuple = tuple(np.asarray(y_data).round(10).tolist())
        p0_tuple = tuple(p0)
        bounds_lower = tuple(bounds[0]) if bounds else None
        bounds_upper = tuple(bounds[1]) if bounds else None
        param_names_tuple = tuple(param_names) if param_names else None

        return fit_model_with_ci_cached(
            model_name,
            x_tuple,
            y_tuple,
            p0_tuple,
            bounds_lower,
            bounds_upper,
            param_names_tuple,
            confidence,
        )

    # Fall back to direct computation for unregistered models
    return _fit_model_core(model_func, x_data, y_data, p0, bounds, param_names, confidence)


# =============================================================================
# 3D VISUALIZATION MODELS
# =============================================================================


def langmuir_3d_surface(
    Ce_range: tuple[float, float],
    temp_range: tuple[float, float],
    qm: float,
    KL: float,
    delta_H: float = -25000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D Langmuir surface with temperature dependence.

    Uses Van't Hoff: KL(T) = KL_ref × exp(-ΔH/R × (1/T - 1/T_ref))
    """
    n_points = 30
    Ce_vals = np.linspace(max(Ce_range[0], 0.1), Ce_range[1], n_points)
    temp_vals_C = np.linspace(temp_range[0], temp_range[1], n_points)
    temp_vals_K = temp_vals_C + 273.15

    Ce_grid, temp_grid_K = np.meshgrid(Ce_vals, temp_vals_K)
    qe_grid = np.zeros_like(Ce_grid)

    T_ref = 298.15

    # VECTORIZED computation (Performance Optimization)
    # Calculate KL_adj for all temperatures at once using broadcasting
    # temp_vals_K shape: (n_points,), need it as column vector for broadcasting
    KL_adj_all = KL * np.exp(-delta_H / R_GAS * (1 / temp_vals_K - 1 / T_ref))  # Shape: (n_points,)

    # Apply Langmuir model using broadcasting
    # Ce_grid shape: (n_points, n_points), KL_adj_all[:, np.newaxis] shape: (n_points, 1)
    # This broadcasts KL_adj across all Ce values for each temperature
    KL_Ce = KL_adj_all[:, np.newaxis] * Ce_grid  # Shape: (n_points, n_points)
    qe_grid = qm * KL_Ce / (1 + KL_Ce)

    return Ce_grid, temp_grid_K - 273.15, qe_grid


def ph_temperature_response_surface(
    pH_range: tuple[float, float],
    temp_range: tuple[float, float],
    optimal_pH: float = 6.0,
    optimal_temp: float = 40.0,
    max_capacity: float = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate pH-temperature response surface."""
    n_points = 25
    pH_vals = np.linspace(pH_range[0], pH_range[1], n_points)
    temp_vals = np.linspace(temp_range[0], temp_range[1], n_points)

    pH_grid, temp_grid = np.meshgrid(pH_vals, temp_vals)

    pH_response = np.exp(-((pH_grid - optimal_pH) ** 2) / 4)
    temp_response = np.exp(-((temp_grid - optimal_temp) ** 2) / 200)

    response = pH_response * temp_response * max_capacity

    return pH_grid, temp_grid, response


def parameter_space_visualization(
    model_func: Callable[..., Any],
    param1_range: tuple[float, float],
    param2_range: tuple[float, float],
    Ce_fixed: float = 50,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Generate 3D visualization of parameter space effects.

    Performance Optimization: Attempts vectorized computation first,
    falls back to element-wise for functions that don't support arrays.
    """
    n_points = 25
    param1_vals = np.linspace(param1_range[0], param1_range[1], n_points)
    param2_vals = np.linspace(param2_range[0], param2_range[1], n_points)

    param1_grid, param2_grid = np.meshgrid(param1_vals, param2_vals)

    # Try vectorized computation first (much faster if supported)
    try:
        qe_grid = model_func(Ce_fixed, param1_grid, param2_grid)
        if not np.any(np.isnan(qe_grid)) and not np.any(np.isinf(qe_grid)):
            return param1_grid, param2_grid, qe_grid
    except (ValueError, TypeError):
        pass  # Fall back to element-wise

    # Fallback: element-wise computation with error handling
    qe_grid = np.zeros_like(param1_grid)
    for i in range(n_points):
        for j in range(n_points):
            try:
                qe_grid[i, j] = model_func(Ce_fixed, param1_vals[j], param2_vals[i])
            except (RuntimeError, ValueError, TypeError, ZeroDivisionError):
                qe_grid[i, j] = np.nan

    return param1_grid, param2_grid, qe_grid


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def calculate_initial_rate(t: NDArray[np.floating[Any]], qt: np.ndarray) -> float:
    """Calculate initial adsorption rate from first few points."""
    if len(t) < 3:
        return 0.0

    n_pts = min(4, len(t))
    t_init = t[:n_pts]
    qt_init = qt[:n_pts]

    if len(np.unique(t_init)) < 2:
        return 0.0

    try:
        slope, _, r_val, _, _ = linregress(t_init, qt_init)
        return slope if abs(r_val) > 0.7 else 0.0
    except (ValueError, RuntimeError, TypeError):
        return 0.0


def identify_equilibrium_time(
    t: NDArray[np.floating[Any]], qt: NDArray[np.floating[Any]], threshold: float = 0.95
) -> float:
    """Find time when qt reaches threshold fraction of max."""
    if len(qt) == 0:
        return 0.0

    q_max = np.max(qt)
    if q_max <= 0:
        return t[-1] if len(t) > 0 else 0.0

    for i, q in enumerate(qt):
        if q >= threshold * q_max:
            return t[i]

    return t[-1]


def get_model_info() -> dict[str, dict]:
    """
    Return information about all available models.

    Useful for documentation and UI generation.
    """
    return {
        "isotherms": {
            "Langmuir": {
                "equation": r"q_e = \frac{q_m K_L C_e}{1 + K_L C_e}",
                "params": ["qm (mg/g)", "KL (L/mg)"],
                "description": "Monolayer adsorption on homogeneous surface",
            },
            "Freundlich": {
                "equation": r"q_e = K_F C_e^{1/n}",
                "params": ["KF ((mg/g)(L/mg)^(1/n))", "n (dimensionless)"],
                "description": "Empirical model for heterogeneous surfaces",
            },
            "Temkin": {
                "equation": r"q_e = B_1 \ln(K_T C_e)",
                "params": ["B1 (mg/g)", "KT (L/mg)"],
                "description": "Heat of adsorption decreases linearly with coverage",
            },
            "Sips": {
                "equation": r"q_e = \frac{q_m (K_s C_e)^{n_s}}{1 + (K_s C_e)^{n_s}}",
                "params": ["qm (mg/g)", "Ks (L/mg)", "ns (dimensionless)"],
                "description": "Combines Langmuir and Freundlich",
            },
        },
        "kinetics": {
            "PFO": {
                "equation": r"q_t = q_e (1 - e^{-k_1 t})",
                "params": ["qe (mg/g)", "k1 (min⁻¹)"],
                "description": "Pseudo-first order (Lagergren)",
            },
            "PSO": {
                "equation": r"q_t = \frac{q_e^2 k_2 t}{1 + q_e k_2 t}",
                "params": ["qe (mg/g)", "k2 (g/(mg·min))"],
                "description": "Pseudo-second order (Ho-McKay)",
            },
            "rPSO": {
                "equation": r"q_t = \frac{q_e^2 k_2 t}{1 + q_e k_2 t \cdot \varphi}",
                "params": ["qe (mg/g)", "k2 (g/(mg·min))"],
                "conditions": ["C0 (mg/L)", "m (g)", "V (L)"],
                "description": "Revised PSO with concentration correction (Bullen et al., 2021)",
                "note": "φ = 1 + (qe·m)/(C0·V); reduces ~66% fitting error vs standard PSO",
            },
            "Elovich": {
                "equation": r"q_t = \frac{1}{\beta} \ln(1 + \alpha \beta t)",
                "params": ["α (mg/(g·min))", "β (g/mg)"],
                "description": "Chemisorption on heterogeneous surfaces",
            },
            "IPD": {
                "equation": r"q_t = k_{id} t^{0.5} + C",
                "params": ["kid (mg/(g·min⁰·⁵))", "C (mg/g)"],
                "description": "Intraparticle diffusion (Weber-Morris)",
            },
        },
        "competitive": {
            "Extended-Langmuir": {
                "equation": r"q_{e,i} = \frac{q_{m,i} K_{L,i} C_{e,i}}{1 + \sum_j K_{L,j} C_{e,j}}",
                "params": ["qm_i (mg/g)", "KL_i (L/mg)"],
                "description": "Butler-Ockrent competitive Langmuir",
            },
            "Extended-Freundlich": {
                "equation": r"q_{e,i} = K_{F,i} C_{e,i} \left(\sum_j a_{ij} C_{e,j}\right)^{1/n_i - 1}",
                "params": ["Kf_i ((mg/g)(L/mg)^(1/n))", "n_i (dimensionless)"],
                "description": "Sheindorf-Rebhun-Sheintuch (SRS) model",
            },
        },
    }