"""
DoubleHurdleModel: fitted four-component double-hurdle model.

Stores the four fitted components (S1, S2, bulk, tail) and the metadata
needed to reproduce predictions (threshold, covariate_combo, family names).

predict(df) assembles the unconditional expected death rate by:
  1. Building design matrices from df via features.build_X / align_X.
  2. Predicting each component using its family's predict function.
  3. Combining via combine.assemble_dh_prediction.

S1 uses predict_binomial_cloglog (offset model: log_exposed is separate).
S2 uses s2.predict (log_exposed is a free covariate in X, no offset).
Bulk and tail use get_family(family)["predict"]. NB is the one bulk/tail family
whose predict takes log_exposed as a third argument; all others take only (result, X).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idd_tc_mortality.combine import assemble_dh_prediction
from idd_tc_mortality.distributions import get_family
from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.features import align_X, build_X
from idd_tc_mortality import s2 as s2_mod


class DoubleHurdleModel:
    """Fitted double-hurdle mortality model.

    Parameters
    ----------
    s1_result:
        FitResult from fitting S1 (P(deaths >= 1)) via binomial cloglog.
    s2_result:
        FitResult from fitting S2 (P(rate >= threshold | deaths >= 1)) via
        binomial cloglog.
    bulk_result:
        FitResult from fitting the bulk rate component.
    bulk_family:
        Registry key for the bulk distribution (e.g. 'gamma', 'lognormal').
    tail_result:
        FitResult from fitting the tail rate component.
    tail_family:
        Registry key for the tail distribution (e.g. 'gpd', 'gamma').
    threshold:
        Rate threshold separating bulk from tail observations (rate scale).
    covariate_combo:
        Boolean flag dict passed to features.build_X (wind_speed, sdi,
        basin, is_island). The same combo is used for all four components.
    """

    def __init__(
        self,
        s1_result: FitResult,
        s2_result: FitResult,
        bulk_result: FitResult,
        bulk_family: str,
        tail_result: FitResult,
        tail_family: str,
        threshold: float,
        covariate_combo: dict[str, bool],
    ) -> None:
        self.s1_result = s1_result
        self.s2_result = s2_result
        self.bulk_result = bulk_result
        self.bulk_family = bulk_family
        self.tail_result = tail_result
        self.tail_family = tail_family
        self.threshold = threshold
        self.covariate_combo = covariate_combo

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict unconditional expected death rates for rows in df.

        Parameters
        ----------
        df:
            DataFrame with columns matching covariate_combo flags plus
            'exposed'. Basin must contain values from BASIN_LEVELS.

        Returns
        -------
        np.ndarray
            Unconditional expected death rates, shape (len(df),). Non-negative.
        """
        log_exposed = np.log(df["exposed"].to_numpy(dtype=float))

        # S1: cloglog offset model — log_exposed is separate, not in X
        p_s1 = self._predict_s1(self.s1_result, df, log_exposed)
        # S2: free covariate model — log_exposed is in X, no separate offset
        p_s2 = self._predict_s2(self.s2_result, df)

        # Bulk and tail: registry predict — include_log_exposed inferred from param_names
        rate_bulk = self._predict_rate(self.bulk_result, self.bulk_family, df, log_exposed)
        rate_tail = self._predict_rate(self.tail_result, self.tail_family, df, log_exposed)

        return assemble_dh_prediction(p_s1, p_s2, rate_bulk, rate_tail)

    def metrics(self, df: pd.DataFrame, observed_rates: np.ndarray) -> dict:
        """Compute predictive metrics. Not yet implemented."""
        raise NotImplementedError("metrics() is not yet implemented.")

    def diagnostics(self) -> dict:
        """Return diagnostic summaries for all components. Not yet implemented."""
        raise NotImplementedError("diagnostics() is not yet implemented.")

    def plot(self) -> None:
        """Plot model diagnostics. Not yet implemented."""
        raise NotImplementedError("plot() is not yet implemented.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_s1(
        self,
        result: FitResult,
        df: pd.DataFrame,
        log_exposed: np.ndarray,
    ) -> np.ndarray:
        """Build X (no log_exposed in matrix) and predict S1 via cloglog offset."""
        from idd_tc_mortality.distributions.binomial_cloglog import predict_binomial_cloglog
        X = build_X(df, self.covariate_combo, include_log_exposed=False)
        X = align_X(X, result.param_names)
        return predict_binomial_cloglog(result, X, log_exposed)

    def _predict_s2(
        self,
        result: FitResult,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Build X (log_exposed as free covariate) and predict S2."""
        X = build_X(df, self.covariate_combo, include_log_exposed=True)
        X = align_X(X, result.param_names)
        return s2_mod.predict(result, X)

    def _predict_rate(
        self,
        result: FitResult,
        family: str,
        df: pd.DataFrame,
        log_exposed: np.ndarray,
    ) -> np.ndarray:
        """Build X and call the registry predict for a bulk or tail component.

        include_log_exposed is inferred from param_names: if 'log_exposed'
        appears in the fitted model's param_names, it was a free covariate and
        must be included in the prediction matrix. NB is the exception — it uses
        log_exposed as an offset (not in param_names) and its predict takes a
        separate log_exposed argument.
        """
        include_log_exposed = "log_exposed" in result.param_names
        X = build_X(df, self.covariate_combo, include_log_exposed=include_log_exposed)
        X = align_X(X, result.param_names)

        family_info = get_family(family)
        pred_fn = family_info["predict"]
        if family_info["log_exposed"]:
            return pred_fn(result, X, log_exposed)
        return pred_fn(result, X)
