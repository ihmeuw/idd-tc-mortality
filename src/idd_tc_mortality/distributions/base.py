"""
Base dataclass for all distribution fit results.

Every distribution module returns a FitResult. Downstream code (predict, assemble,
eventually uncertainty draws) depends only on this interface, never on raw statsmodels
or scipy result objects.

cov is stored as a placeholder (None) until the uncertainty module is built. Any code
that tries to use cov before it is populated will fail loudly rather than silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FitResult:
    """Uniform return type for all distribution fit functions.

    Attributes
    ----------
    params:
        1-D array of fitted coefficients, in the same order as the columns of
        the design matrix passed to fit().
    param_names:
        Names corresponding to each element of params. Must have the same length.
    cov:
        Placeholder for the parameter covariance matrix. Populated by the
        uncertainty module (not yet built). None until then.
    fitted_values:
        In-sample predictions on the scale of the outcome (rate scale for bulk/tail
        rate models, probability scale for s1/s2, count scale for nb).
    converged:
        True if the optimizer reported convergence. False if it did not but a result
        was still returned (e.g. GLM iteration limit hit). Used to flag suspect fits
        in the grid without crashing.
    family:
        String identifier matching the distributions registry key, e.g. 'gamma',
        's1', 'nb'. Used for logging and result indexing.
    meta:
        Arbitrary dict for family-specific diagnostics: iteration counts, warning
        messages, clipping rates, GPD shape parameter, etc. Never required by
        downstream code — for human inspection only.
    """

    params: np.ndarray
    param_names: list[str]
    fitted_values: np.ndarray
    family: str
    converged: bool = True
    cov: np.ndarray | None = None  # populated later by uncertainty module
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.params) != len(self.param_names):
            raise ValueError(
                f"params length {len(self.params)} != "
                f"param_names length {len(self.param_names)}"
            )
