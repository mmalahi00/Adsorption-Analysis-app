# Scientific integration notes

This integration combines the transport/workflow corrections with Claude's
scientific-corrections branch. No Next.js migration or new simulation module is included.

## Changes affecting existing results

- AIC, AICc and BIC now share one unweighted Gaussian-likelihood implementation.
  The parameter count includes estimated residual variance (`k = p + 1`).
  Undefined AICc is infinite; numerically exact fits use a positive variance floor.
  Old model rankings may change. Compare identical observations and error models.
- Missing/non-finite thermodynamic quantities are unavailable, not positive or
  exothermic. Exact zero is labelled separately. Apparent distribution-ratio
  thermodynamics must not be represented as standard-state quantities.
- Boyd/Weber–Morris plots are conditional transport diagnostics, not proof of
  mechanism. Intercept uncertainty replaces arbitrary origin thresholds.
- Synthetic 3D surfaces and automatic mechanistic confidence claims are removed
  from the stable workflow. No-checks results are not presented as successful checks.

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
