# Scientific integration notes

These notes describe the scientific calculations, transport diagnostics and
workflow corrections in the current Streamlit application.

## Changes affecting existing results

- AIC, AICc and BIC now share one unweighted Gaussian-likelihood implementation.
  The parameter count includes estimated residual variance (`k = p + 1`).
  Undefined AICc is infinite; numerically exact fits use a positive variance floor.
  Old model rankings may change. Compare identical observations and error models.
- The comparison screens, Study Overview, exports and DOCX rank AICc only among fits
  to the same observations (and error model) whose AICc is defined; fits to other
  observations (e.g. Temkin without Ce = 0 rows) are listed separately, and undefined
  AICc never produces a "best" model. Elovich, whose implemented form
  q = ln(1 + αβt)/β is defined at t = 0, is fitted to the same observations as the
  other kinetic models.
- Missing/non-finite thermodynamic quantities are unavailable, not positive or
  exothermic. Exact zero is labelled separately. Apparent distribution-ratio
  thermodynamics must not be represented as standard-state quantities.
- Boyd/Weber–Morris plots are conditional transport diagnostics, not proof of
  mechanism. Intercept uncertainty replaces arbitrary origin thresholds.
- Synthetic 3D surfaces and automatic mechanistic confidence claims are removed
  from the stable workflow. No-checks results are not presented as successful checks.

## Fit limits and parameter identifiability

Isotherm (Langmuir, Freundlich, Temkin, Sips) and kinetic (PFO, PSO, Elovich) fits
use starting values and limits derived from the data (`isotherm_fit_setup`,
`kinetic_fit_setup` in `models.py`) instead of fixed ceilings such as KL ≤ 100 L/mg,
k2 ≤ 10 g/(mg·min) or β ≤ 10 g/mg, which pinned valid fits at low concentrations or
low uptake. Physical constraints remain: non-negative capacities, affinities and rate
constants; Freundlich 1/n in [0.01, 5] and Sips ns in [0.1, 5] (plausibility ranges);
Temkin B1 ≥ 0 and KT ≥ 1/min(Ce), so that predicted qe ≥ 0 at every observation.
Upper limits on scale-dependent parameters are numerical search guards 10⁶ times
beyond the data scale (`SEARCH_RANGE`), e.g. KL ≤ 10⁶/min(Ce); KF, B1, KT and α have
none.

Every fit reports, separately from convergence, each parameter's status: *at limit*
(stopped on a limit, so the value is set by the limit and its SE/CI are not valid),
*poorly identified* (the 95 % CI half-width reaches the estimate), *strongly
correlated* (|r| ≥ 0.99 with another parameter; the linearised CI can then understate
the uncertainty, e.g. qe and k1 from early-time-only kinetics) or *identified*. A close
curve fit (high R²) with poorly identified parameters — e.g. Langmuir in the Henry
region — is not reported as a precise parameter estimate, and a fit stopped at a limit
is not presented as evidence against the model. The status appears in the model displays, the comparison table
notes and the parameter exports. The fit configuration (starting values and limits)
is stored with each result.

## Legacy rPSO quarantine

The implemented legacy equation is mathematically a reparameterised PSO curve:
`q(t) = k2*qraw^2*t / (1 + k2*qraw*phi*t)`, `phi = 1 + qraw/Q`, `Q = C0*V/m`.
Its asymptote is `qraw/phi`, not `qraw`. Plotting/export now evaluate that same
equation with the fitted conditions, and refuse missing conditions. The arbitrary
99% capacity cap is removed from the legacy fitting path.

New fits from the stable UI exclude this model until an independent check against
the original publication establishes the intended equation and parameter meanings.
The legacy API remains for reproducibility, not as an endorsed implementation of
Bullen et al. Numerical tests are not validation of the literature attribution.

## API change and deferred work

`determine_adsorption_mechanism` is intentionally removed: threshold-generated
mechanistic scores/confidences have no justified replacement. External callers
must stop using it; zero confidence or empty scores would not be a faithful shim.
Descriptive sign labels are available separately through `sign_label`.

Project save/load and integrated replicate/uncertainty handling remain separate
future work. Neither is claimed as implemented by these corrections.
