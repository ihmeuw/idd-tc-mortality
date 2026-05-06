"""
Full four-stage double-hurdle integration test on synthetic TC-realistic data.

Pipeline under test
-------------------
S1  — cloglog + log_exposed offset
S2  — cloglog + log_exposed offset  (on deaths >= 1 subset)
Bulk — lognormal (rate model, log_exposed free covariate, weights = exposed)
Tail — gamma    (excess rate model, log_exposed free covariate, weights = exposed)

These are the actual families used for a preliminary run: lognormal is in
BULK_FAMILIES; gamma is in TAIL_FAMILIES.  Using gamma for bulk is explicitly
NOT tested here because gamma was removed from BULK_FAMILIES.

DGP
---
Two real covariates: wind_speed ~ N(0,1), sdi ~ N(0,1).
exposed ~ Uniform(10K, 1M).  n = 2000.

S1:   eta = -11 + 0.6·wind - 0.3·sdi + log_exposed  → cloglog → Bernoulli
Rate: log(rate) = -20 + 0.4·wind - 0.3·sdi + 0.8·log_exposed + N(0, 0.5²/phi)
      phi = exposed / 5000  (heteroscedastic: large-exposure storms more precise)
      deaths = max(1, round(rate·exposed)) for s1=1 rows; 0 otherwise

Threshold computed at q=0.75 of positive death_rate values.

S2:   eta = -11 + 0.5·wind - 0.3·sdi + log_exposed  → cloglog → Bernoulli
      generated independently on the s1=1 subset; exceeds_threshold drawn from it.
      (Conceptual note: in a fully consistent DGP, S2 would be implied by the rate
       exceeding the threshold.  Generating S2 from an independent cloglog lets us
       assert coefficient recovery with known parameters.)

Bulk: fitted on deaths >= 1 AND rate < threshold rows.
Tail: fitted on rate >= threshold rows; y = excess rate = rate - threshold (> 0).

At mean log_exposed ≈ 11.5:
  rate ≈ exp(-20 + 0.8·11.5) = exp(-10.8) ≈ 2e-5  ← TC-realistic bulk death rate

Coefficient assertions
----------------------
S1 / S2   : intercept ±0.3, wind ±0.3, sdi ±0.3
Bulk (wind, sdi, logexp): ±0.3  (intercept may be biased due to truncation at threshold)
Tail: no tight assertions — DGP is lognormal above threshold; excess is not exactly gamma.
      Fit must converge and predictions must be positive + finite.
Prediction correlation > 0.5 (Pearson r on s1=1 rows vs true death rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.distributions import gamma as tail_mod
from idd_tc_mortality.distributions import lognormal as bulk_mod
from idd_tc_mortality.distributions.binomial_cloglog import fit_binomial_cloglog
from idd_tc_mortality.features import build_X
from idd_tc_mortality.model import DoubleHurdleModel
from idd_tc_mortality.thresholds import compute_thresholds

# ---------------------------------------------------------------------------
# True DGP parameters
# ---------------------------------------------------------------------------
TRUE_S1_INTERCEPT   = -11.0
TRUE_S1_WIND        =   0.6
TRUE_S1_SDI         =  -0.3

TRUE_S2_INTERCEPT   = -11.0
TRUE_S2_WIND        =   0.5
TRUE_S2_SDI         =  -0.3

TRUE_BULK_INTERCEPT = -20.0
TRUE_BULK_WIND      =   0.4
TRUE_BULK_SDI       =  -0.3
TRUE_BULK_LOGEXP    =   0.8
TRUE_BULK_SIGMA     =   0.5

COV_COMBO = {"wind_speed": True, "sdi": True, "basin": False, "is_island": False}


# ---------------------------------------------------------------------------
# Module-scoped fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    """Generate TC-realistic synthetic data and fit all four components."""
    rng = np.random.default_rng(42)
    n = 2000

    # ---- Covariates --------------------------------------------------------
    exposed    = rng.uniform(10_000, 1_000_000, n)
    wind_speed = rng.normal(0, 1, n)
    sdi        = rng.normal(0, 1, n)
    log_exposed = np.log(exposed)
    phi         = exposed / 5_000.0   # precision ∝ exposure

    df = pd.DataFrame({
        "wind_speed": wind_speed,
        "sdi":        sdi,
        "basin":      "NA",
        "is_island":  0.0,
        "exposed":    exposed,
    })

    # ---- S1 DGP: cloglog + log_exposed offset ------------------------------
    eta_s1    = TRUE_S1_INTERCEPT + TRUE_S1_WIND * wind_speed + TRUE_S1_SDI * sdi + log_exposed
    p_s1      = 1.0 - np.exp(-np.exp(eta_s1))
    any_death = rng.binomial(1, np.clip(p_s1, 1e-9, 1 - 1e-9)).astype(float)
    s1_mask   = any_death == 1

    # ---- Bulk rate DGP: lognormal, TC-realistic scale ----------------------
    # log(rate) = -20 + 0.4·wind - 0.3·sdi + 0.8·log_exposed + noise
    # At mean log_exposed ≈ 11.5: rate ≈ exp(-10.8) ≈ 2e-5
    log_rate_true = (TRUE_BULK_INTERCEPT
                     + TRUE_BULK_WIND  * wind_speed
                     + TRUE_BULK_SDI   * sdi
                     + TRUE_BULK_LOGEXP * log_exposed)
    noise = rng.normal(0, TRUE_BULK_SIGMA / np.sqrt(phi))
    death_rate = np.zeros(n)
    death_rate[s1_mask] = np.exp(log_rate_true[s1_mask] + noise[s1_mask])

    # ---- Integer deaths consistent with the rate ---------------------------
    deaths = np.zeros(n)
    deaths[s1_mask] = np.maximum(1, np.round(death_rate[s1_mask] * exposed[s1_mask]))
    df["deaths"] = deaths
    # Recompute rate from integer deaths for consistency with fit_component logic
    death_rate = df["deaths"].values / exposed

    # ---- Threshold at q=0.75 of positive rates -----------------------------
    positive_rates = death_rate[death_rate > 0]
    threshold = compute_thresholds(positive_rates, quantile_levels=np.array([0.75]))[0.75]

    # ---- S2 DGP: cloglog on s1=1 subset ------------------------------------
    # Generated independently from a known cloglog model to allow coefficient recovery.
    s1_idx = np.where(s1_mask)[0]
    eta_s2 = (TRUE_S2_INTERCEPT
              + TRUE_S2_WIND * wind_speed[s1_mask]
              + TRUE_S2_SDI  * sdi[s1_mask]
              + log_exposed[s1_mask])
    p_s2               = 1.0 - np.exp(-np.exp(eta_s2))
    exceeds_threshold  = rng.binomial(1, np.clip(p_s2, 1e-9, 1 - 1e-9)).astype(float)

    # ---- Bulk / tail subsets -----------------------------------------------
    bulk_mask = s1_mask & (death_rate < threshold)
    tail_mask = s1_mask & (death_rate >= threshold)

    # ---- Fit S1 (full dataset) ---------------------------------------------
    X_s1      = build_X(df, COV_COMBO, include_log_exposed=False)
    s1_result = fit_binomial_cloglog(X_s1, any_death, log_exposed, "s1")

    # ---- Fit S2 (s1=1 subset) ----------------------------------------------
    df_s1     = df[s1_mask].reset_index(drop=True)
    X_s2      = build_X(df_s1, COV_COMBO, include_log_exposed=False)
    s2_result = fit_binomial_cloglog(X_s2, exceeds_threshold, log_exposed[s1_mask], "s2")

    # ---- Fit bulk lognormal (deaths>=1, rate < threshold) ------------------
    df_bulk   = df[bulk_mask].reset_index(drop=True)
    X_bulk    = build_X(df_bulk, COV_COMBO, include_log_exposed=True)
    bulk_result = bulk_mod.fit(
        X_bulk,
        death_rate[bulk_mask],
        weights=exposed[bulk_mask],
    )

    # ---- Fit tail gamma (rate >= threshold) on excess rate -----------------
    df_tail   = df[tail_mask].reset_index(drop=True)
    X_tail    = build_X(df_tail, COV_COMBO, include_log_exposed=True)
    y_excess  = death_rate[tail_mask] - threshold
    # Floor at half the minimum positive excess to guard against boundary zeros.
    pos_excess = y_excess[y_excess > 0]
    _floor = pos_excess.min() / 2 if len(pos_excess) > 0 else threshold * 1e-6
    y_excess = np.maximum(y_excess, _floor)
    tail_result = tail_mod.fit(
        X_tail,
        y_excess,
        weights=exposed[tail_mask],
    )

    # ---- Assemble and predict ----------------------------------------------
    model = DoubleHurdleModel(
        s1_result=s1_result,
        s2_result=s2_result,
        bulk_result=bulk_result,
        bulk_family="lognormal",
        tail_result=tail_result,
        tail_family="gamma",
        threshold=threshold,
        covariate_combo=COV_COMBO,
    )
    predictions = model.predict(df)

    return {
        "model":        model,
        "df":           df,
        "any_death":    any_death,
        "death_rate":   death_rate,
        "s1_mask":      s1_mask,
        "bulk_mask":    bulk_mask,
        "tail_mask":    tail_mask,
        "predictions":  predictions,
        "threshold":    threshold,
        "n_s1":         int(s1_mask.sum()),
        "n_bulk":       int(bulk_mask.sum()),
        "n_tail":       int(tail_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Coefficient recovery — S1
# ---------------------------------------------------------------------------

def test_s1_intercept_recovery(pipeline):
    s1 = pipeline["model"].s1_result
    est = s1.params[s1.param_names.index("const")]
    assert abs(est - TRUE_S1_INTERCEPT) < 0.3, f"S1 intercept: true={TRUE_S1_INTERCEPT}, est={est:.3f}"


def test_s1_wind_recovery(pipeline):
    s1 = pipeline["model"].s1_result
    est = s1.params[s1.param_names.index("wind_speed")]
    assert abs(est - TRUE_S1_WIND) < 0.3, f"S1 wind: true={TRUE_S1_WIND}, est={est:.3f}"


def test_s1_sdi_recovery(pipeline):
    s1 = pipeline["model"].s1_result
    est = s1.params[s1.param_names.index("sdi")]
    assert abs(est - TRUE_S1_SDI) < 0.3, f"S1 sdi: true={TRUE_S1_SDI}, est={est:.3f}"


# ---------------------------------------------------------------------------
# Coefficient recovery — S2
# ---------------------------------------------------------------------------

def test_s2_wind_recovery(pipeline):
    s2 = pipeline["model"].s2_result
    est = s2.params[s2.param_names.index("wind_speed")]
    assert abs(est - TRUE_S2_WIND) < 0.3, f"S2 wind: true={TRUE_S2_WIND}, est={est:.3f}"


def test_s2_sdi_recovery(pipeline):
    s2 = pipeline["model"].s2_result
    est = s2.params[s2.param_names.index("sdi")]
    assert abs(est - TRUE_S2_SDI) < 0.3, f"S2 sdi: true={TRUE_S2_SDI}, est={est:.3f}"


# ---------------------------------------------------------------------------
# Coefficient recovery — Bulk (lognormal)
# Wind, sdi, log_exposed should recover despite fitting on the truncated subset.
# Intercept may be biased (log_exposed has wide range far from zero → intercept shifts)
# so we only assert the slope coefficients.
# ---------------------------------------------------------------------------

def test_bulk_wind_recovery(pipeline):
    bulk = pipeline["model"].bulk_result
    est  = bulk.params[bulk.param_names.index("wind_speed")]
    assert abs(est - TRUE_BULK_WIND) < 0.3, f"Bulk wind: true={TRUE_BULK_WIND}, est={est:.3f}"


def test_bulk_sdi_recovery(pipeline):
    bulk = pipeline["model"].bulk_result
    est  = bulk.params[bulk.param_names.index("sdi")]
    assert abs(est - TRUE_BULK_SDI) < 0.3, f"Bulk sdi: true={TRUE_BULK_SDI}, est={est:.3f}"


def test_bulk_logexp_recovery(pipeline):
    """log_exposed coefficient recovered from bulk lognormal fit.

    This is the key exposure test: if log_exposed were absent from X or
    wired incorrectly, this assertion would fail.
    """
    bulk = pipeline["model"].bulk_result
    est  = bulk.params[bulk.param_names.index("log_exposed")]
    assert abs(est - TRUE_BULK_LOGEXP) < 0.3, f"Bulk logexp: true={TRUE_BULK_LOGEXP}, est={est:.3f}"


# ---------------------------------------------------------------------------
# Tail (gamma): convergence and sanity only — DGP is lognormal above threshold,
# excess is not exactly gamma, so tight coefficient recovery is not expected.
# ---------------------------------------------------------------------------

def test_tail_converged(pipeline):
    assert pipeline["model"].tail_result.converged, "Tail gamma GLM did not converge"


def test_tail_fitted_values_positive(pipeline):
    fv = pipeline["model"].tail_result.fitted_values
    assert np.all(fv > 0) and np.all(np.isfinite(fv)), "Tail fitted values not all positive/finite"


# ---------------------------------------------------------------------------
# Prediction quality
# ---------------------------------------------------------------------------

def test_predictions_nonnegative(pipeline):
    assert np.all(pipeline["predictions"] >= 0), "predict() returned negative values"


def test_predictions_finite(pipeline):
    assert np.all(np.isfinite(pipeline["predictions"])), "predict() returned non-finite values"


def test_predictions_lower_for_no_death_rows(pipeline):
    """Mean prediction for any_death=0 rows must be below median for any_death=1 rows."""
    preds     = pipeline["predictions"]
    any_death = pipeline["any_death"]

    mean_no_death = float(np.mean(preds[any_death == 0]))
    median_death  = float(np.median(preds[any_death == 1]))

    assert mean_no_death < median_death, (
        f"Mean prediction for any_death=0 ({mean_no_death:.2e}) should be below "
        f"median for any_death=1 ({median_death:.2e})."
    )


def test_predictions_correlate_with_true_rates(pipeline):
    """Pearson r between assembled predictions and true death rates (s1=1 rows) > 0.5."""
    preds      = pipeline["predictions"]
    death_rate = pipeline["death_rate"]
    s1_mask    = pipeline["s1_mask"]

    r = float(np.corrcoef(preds[s1_mask], death_rate[s1_mask])[0, 1])
    assert r > 0.5, (
        f"Pearson r = {r:.3f}. Expected > 0.5 — broken wiring or severely mis-specified DGP."
    )


# ---------------------------------------------------------------------------
# Sanity: subset sizes and all components converged
# ---------------------------------------------------------------------------

def test_subset_sizes_reasonable(pipeline):
    n = len(pipeline["df"])
    assert 0.4 * n <= pipeline["n_s1"] <= 0.99 * n, f"n_s1={pipeline['n_s1']} unexpected"
    assert pipeline["n_bulk"] > pipeline["n_tail"], "Expected bulk > tail at q=0.75"


def test_all_components_converged(pipeline):
    m = pipeline["model"]
    assert m.s1_result.converged,   "S1 GLM did not converge"
    assert m.s2_result.converged,   "S2 GLM did not converge"
    assert m.bulk_result.converged, "Bulk lognormal did not converge"
    assert m.tail_result.converged, "Tail gamma GLM did not converge"
