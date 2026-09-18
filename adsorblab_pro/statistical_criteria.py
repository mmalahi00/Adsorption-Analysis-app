"""Information criteria for unweighted Gaussian least-squares fits."""

import numpy as np


def information_criteria(sse: float, n: int, p: int) -> tuple[float, float, float]:
    """Return AIC, AICc, BIC; count the estimated residual variance in k.

    Compare only fits to the same observations under the same error model.
    The variance floor handles numerically exact fits, not measurement precision.
    """
    if n <= p or not np.isfinite(sse) or sse < 0:
        return np.inf, np.inf, np.inf
    k = p + 1
    variance = max(float(sse) / n, float(np.finfo(float).tiny))
    deviance = n * (np.log(2 * np.pi) + np.log(variance) + 1)
    aic = float(deviance + 2 * k)
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
    return aic, float(aicc), float(deviance + k * np.log(n))
