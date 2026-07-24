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
    from idd_tc_mortality.distributions import weibull_mean  # noqa: PLC0415
    from idd_tc_mortality.distributions import gpd_cens  # noqa: PLC0415
    from idd_tc_mortality.distributions import log_logistic_cens  # noqa: PLC0415
    from idd_tc_mortality.distributions import gpd_shadow  # noqa: PLC0415
    from idd_tc_mortality.distributions import log_logistic_shadow  # noqa: PLC0415
    from idd_tc_mortality.distributions import gpd_trunc  # noqa: PLC0415 — variant B spike
    from idd_tc_mortality.distributions import log_logistic_trunc  # noqa: PLC0415 — variant B spike
    # tail_outcome: "excess" means the distribution is fit on (death_rate - threshold)
    # and predict() returns excess rates. predict_component adds threshold_rate back.
    # Absent key = not a tail rate family (bulk-only, count model, or raw-rate tail).
    # truncated_normal has no tail_outcome flag: it fits on raw log(rate), not excess_rate.
    # fit_component.py special-cases it (like beta/scaled_logit) to pass threshold_rate and
    # truncation_side directly.
    #
    # needs_cap: True marks a bounded/finite-mean tail variant whose predict() takes a
    # third argument — the per-row excess-scale physical cap H_i = c_i - threshold_rate
    # (see tail_cap.excess_cap). predict_component supplies it. These variants form the
    # parallel bounded-tail selection run and leave the base families untouched.
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
        # Bounded / finite-mean tail variants (parallel selection run; base families above untouched).
        "weibull_mean":      {"fit": weibull_mean.fit,      "predict": weibull_mean.predict,      "log_exposed": False, "tail_outcome": "excess"},
        "gpd_cens":          {"fit": gpd_cens.fit,          "predict": gpd_cens.predict,          "log_exposed": False, "tail_outcome": "excess", "needs_cap": True},
        "log_logistic_cens": {"fit": log_logistic_cens.fit, "predict": log_logistic_cens.predict, "log_exposed": False, "tail_outcome": "excess", "needs_cap": True},
        "gpd_shadow":        {"fit": gpd_shadow.fit,        "predict": gpd_shadow.predict,        "log_exposed": False, "tail_outcome": "excess", "needs_cap": True},
        "log_logistic_shadow": {"fit": log_logistic_shadow.fit, "predict": log_logistic_shadow.predict, "log_exposed": False, "tail_outcome": "excess", "needs_cap": True},
        # Variant B (renormalized truncated likelihood) — validated on hard-truncated real data
        # (identifiable shape, analytic gradient); folds into the honest tail sweep alongside A/C/D.
        "gpd_trunc":         {"fit": gpd_trunc.fit,         "predict": gpd_trunc.predict,         "log_exposed": False, "tail_outcome": "excess", "needs_cap": True},
        "log_logistic_trunc": {"fit": log_logistic_trunc.fit, "predict": log_logistic_trunc.predict, "log_exposed": False, "tail_outcome": "excess", "needs_cap": True},
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
