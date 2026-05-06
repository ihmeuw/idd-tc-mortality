"""
Distribution registry.

Imports are deferred to avoid circular imports (distribution modules import from lib,
which imports base, which lives in this package). Call get_family() rather than
accessing the registry directly.
"""

from __future__ import annotations

from typing import Callable

from idd_tc_mortality.distributions.base import FitResult  # noqa: F401 — re-exported

_REGISTRY: dict[str, dict[str, Callable]] | None = None


def _build_registry() -> dict[str, dict[str, Callable]]:
    from idd_tc_mortality.distributions import gamma  # noqa: PLC0415
    from idd_tc_mortality.distributions import lognormal  # noqa: PLC0415
    from idd_tc_mortality.distributions import beta  # noqa: PLC0415
    from idd_tc_mortality.distributions import scaled_logit  # noqa: PLC0415
    from idd_tc_mortality.distributions import nb  # noqa: PLC0415
    from idd_tc_mortality.distributions import gpd  # noqa: PLC0415
    from idd_tc_mortality.distributions import poisson  # noqa: PLC0415
    from idd_tc_mortality.distributions import truncated_normal  # noqa: PLC0415
    from idd_tc_mortality.distributions import weibull  # noqa: PLC0415
    from idd_tc_mortality.distributions import log_logistic  # noqa: PLC0415
    # tail_outcome: "excess" means the distribution is fit on (death_rate - threshold)
    # and predict() returns excess rates. predict_component adds threshold_rate back.
    # Absent key = not a tail rate family (bulk-only, count model, or raw-rate tail).
    # truncated_normal has no tail_outcome flag: it fits on raw log(rate), not excess_rate.
    # fit_component.py special-cases it (like beta/scaled_logit) to pass threshold_rate and
    # truncation_side directly.
    return {
        "gamma":             {"fit": gamma.fit,             "predict": gamma.predict,             "log_exposed": False, "tail_outcome": "excess"},
        "lognormal":         {"fit": lognormal.fit,         "predict": lognormal.predict,         "log_exposed": False, "tail_outcome": "excess"},
        "beta":              {"fit": beta.fit,              "predict": beta.predict,              "log_exposed": False},
        "scaled_logit":      {"fit": scaled_logit.fit,      "predict": scaled_logit.predict,      "log_exposed": False},
        "nb":                {"fit": nb.fit,                "predict": nb.predict,                "log_exposed": True},
        "gpd":               {"fit": gpd.fit,               "predict": gpd.predict,               "log_exposed": False, "tail_outcome": "excess"},
        "poisson":           {"fit": poisson.fit,           "predict": poisson.predict,           "log_exposed": True},
        "truncated_normal":  {"fit": truncated_normal.fit,  "predict": truncated_normal.predict,  "log_exposed": False},
        "weibull":           {"fit": weibull.fit,           "predict": weibull.predict,           "log_exposed": False, "tail_outcome": "excess"},
        "log_logistic":      {"fit": log_logistic.fit,      "predict": log_logistic.predict,      "log_exposed": False, "tail_outcome": "excess"},
    }


def get_family(name: str) -> dict[str, Callable]:
    """Return the fit/predict dict for a named family."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown distribution family '{name}'. "
            f"Registered families: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]
