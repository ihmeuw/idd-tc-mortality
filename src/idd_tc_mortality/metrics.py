"""
Predictive metrics for all four components of the double-hurdle model.

Five public functions:
  calc_s1_metrics         — binary metrics for S1 (P(deaths >= 1)), full dataset.
  calc_s2_metrics         — binary metrics for S2 (P(rate >= threshold | S1=1)).
  calc_continuous_metrics — MAE/RMSE/correlation for bulk or tail subset.
  calc_s2_forward_metrics — coverage-at-X% for any_death=1 rows.
  calc_full_model_metrics — whole-dataset metrics including false positives.

calc_s1_metrics and calc_s2_metrics are intentionally separate even though they
share implementation: call sites should be explicit about which stage they are
evaluating, and the two functions operate on different subsets.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calc_s1_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    """Binary metrics for the S1 component on the full dataset.

    Parameters
    ----------
    y_true:
        Binary outcome (0/1), length n.
    p_pred:
        Predicted probabilities in [0, 1], length n.

    Returns
    -------
    dict with keys:
        brier                  — Brier score: mean((p_pred - y_true)^2).
        auroc                  — Area under ROC curve (NaN if only one class).
        fpr                    — False positive rate at threshold 0.5.
        fnr                    — False negative rate at threshold 0.5.
        predicted_positive_rate — Fraction of observations with p_pred >= 0.5.
    """
    y, p = _validate_binary(y_true, p_pred)
    return _binary_metrics(y, p)


def calc_s2_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    """Binary metrics for the S2 component (called on the any_death=1 subset).

    Identical signature and return keys to calc_s1_metrics. Kept as a separate
    function so call sites are explicit about which hurdle stage is evaluated.
    """
    y, p = _validate_binary(y_true, p_pred)
    return _binary_metrics(y, p)


def calc_continuous_metrics(
    y_true_rate: np.ndarray,
    y_pred_rate: np.ndarray,
    exposed: np.ndarray,
) -> dict:
    """Rate and count-scale metrics for a bulk or tail subset.

    Parameters
    ----------
    y_true_rate, y_pred_rate:
        Observed and predicted death rates (non-negative), length n.
    exposed:
        Population exposed for each observation (positive), length n.

    Returns
    -------
    dict with keys:
        mae_rate, rmse_rate, cor_rate     — on the rate scale.
        mae_count, rmse_count, cor_count  — on the count scale (rate * exposed).
    """
    yt, yp, e = _validate_continuous(y_true_rate, y_pred_rate, exposed)
    return {
        **_continuous_stats(yt, yp, "rate"),
        **_continuous_stats(yt * e, yp * e, "count"),
    }


def calc_s2_forward_metrics(
    y_true_rate: np.ndarray,
    y_pred_rate: np.ndarray,
    exposed: np.ndarray,
) -> dict:
    """MAE, RMSE, and top-X% coverage for any_death=1 rows.

    Coverage at X% (X in 1..20): let A = indices of top X% rows by observed
    rate, B = indices of top X% rows by predicted rate. Coverage = |A ∩ B| / |A|.
    The same coverage calculation is repeated on the count scale (rate * exposed).

    Parameters
    ----------
    y_true_rate, y_pred_rate:
        Observed and predicted death rates, length n (any_death=1 subset).
    exposed:
        Population exposed, length n.

    Returns
    -------
    dict with keys:
        mae_rate, rmse_rate,
        coverage_rate_1  .. coverage_rate_20,
        coverage_count_1 .. coverage_count_20.
    """
    yt, yp, e = _validate_continuous(y_true_rate, y_pred_rate, exposed)
    obs_total  = float(np.sum(yt * e))
    pred_total = float(np.sum(yp * e))
    out: dict = {
        "mae_rate":       float(np.mean(np.abs(yt - yp))),
        "rmse_rate":      float(np.sqrt(np.mean((yt - yp) ** 2))),
        "pred_obs_ratio": pred_total / obs_total if obs_total > 0 else float("nan"),
    }
    for x in range(1, 21):
        out[f"coverage_rate_{x}"]  = _coverage_at_pct(yt, yp, x)
        out[f"coverage_count_{x}"] = _coverage_at_pct(yt * e, yp * e, x)
    return out


def calc_full_model_metrics(
    y_true_rate: np.ndarray,
    y_pred_rate: np.ndarray,
    exposed: np.ndarray,
    any_death: np.ndarray,
) -> dict:
    """Whole-dataset metrics for the assembled double-hurdle model.

    Parameters
    ----------
    y_true_rate, y_pred_rate:
        Observed and predicted death rates (non-negative), length n.
    exposed:
        Population exposed (positive), length n.
    any_death:
        Binary indicator (0/1), length n. Used to identify false positives.

    Returns
    -------
    dict with keys:
        false_positives         — count where predicted count > 1 and any_death=0.
        zero_acc                — among any_death=0 rows, fraction where pred*exposed < 1.
        mae_rate, rmse_rate, cor_rate,
        mae_count, rmse_count, cor_count,
        total_observed_deaths   — sum(y_true_rate * exposed).
        total_predicted_deaths  — sum(y_pred_rate * exposed).
        pred_obs_ratio          — total_predicted_deaths / total_observed_deaths.
        coverage_rate_1  .. coverage_rate_20,
        coverage_count_1 .. coverage_count_20.
    """
    yt, yp, e = _validate_continuous(y_true_rate, y_pred_rate, exposed)
    ad = np.asarray(any_death, dtype=float)
    if len(ad) != len(yt):
        raise ValueError(
            f"any_death length {len(ad)} != y_true_rate length {len(yt)}."
        )

    fp = int(np.sum((yp * e > 1.0) & (ad == 0)))

    no_death_mask = ad == 0
    n_no_death = int(no_death_mask.sum())
    if n_no_death > 0:
        zero_acc = float(np.mean((yp[no_death_mask] * e[no_death_mask]) < 1.0))
    else:
        zero_acc = float("nan")

    obs_total  = float(np.sum(yt * e))
    pred_total = float(np.sum(yp * e))

    out: dict = {
        "false_positives": fp,
        "zero_acc":        zero_acc,
        **_continuous_stats(yt, yp, "rate"),
        **_continuous_stats(yt * e, yp * e, "count"),
        "total_observed_deaths":  obs_total,
        "total_predicted_deaths": pred_total,
        "pred_obs_ratio": pred_total / obs_total if obs_total > 0 else float("nan"),
    }
    for x in range(1, 21):
        out[f"coverage_rate_{x}"]  = _coverage_at_pct(yt, yp, x)
        out[f"coverage_count_{x}"] = _coverage_at_pct(yt * e, yp * e, x)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_binary(
    y_true: np.ndarray,
    p_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    if len(y) == 0:
        raise ValueError("y_true is empty.")
    if len(y) != len(p):
        raise ValueError(f"y_true length {len(y)} != p_pred length {len(p)}.")
    return y, p


def _validate_continuous(
    y_true_rate: np.ndarray,
    y_pred_rate: np.ndarray,
    exposed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = np.asarray(y_true_rate, dtype=float)
    yp = np.asarray(y_pred_rate, dtype=float)
    e  = np.asarray(exposed, dtype=float)
    if len(yt) == 0:
        raise ValueError("y_true_rate is empty.")
    if len(yt) != len(yp):
        raise ValueError(
            f"y_true_rate length {len(yt)} != y_pred_rate length {len(yp)}."
        )
    if len(yt) != len(e):
        raise ValueError(
            f"y_true_rate length {len(yt)} != exposed length {len(e)}."
        )
    return yt, yp, e


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    brier = float(np.mean((p - y) ** 2))
    auroc = _auroc(y, p)

    predicted_pos = p >= 0.5
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    fp    = int((predicted_pos & (y == 0)).sum())
    fn    = int((~predicted_pos & (y == 1)).sum())

    fpr = float(fp / n_neg) if n_neg > 0 else float("nan")
    fnr = float(fn / n_pos) if n_pos > 0 else float("nan")
    ppr = float(predicted_pos.sum() / len(y))

    return {
        "brier":                  brier,
        "auroc":                  auroc,
        "fpr":                    fpr,
        "fnr":                    fnr,
        "predicted_positive_rate": ppr,
    }


def _auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via Mann-Whitney U: P(score_pos > score_neg)."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    return float(u / (len(pos) * len(neg)))


def _continuous_stats(a: np.ndarray, b: np.ndarray, suffix: str) -> dict:
    """MAE, RMSE, Pearson r between a (observed) and b (predicted)."""
    diff = a - b
    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        cor = float("nan")
    else:
        cor = float(np.corrcoef(a, b)[0, 1])
    return {f"mae_{suffix}": mae, f"rmse_{suffix}": rmse, f"cor_{suffix}": cor}


def _coverage_at_pct(observed: np.ndarray, predicted: np.ndarray, pct: int) -> float:
    """Overlap fraction between top-pct% rows by observed and by predicted.

    n_top = ceil(n * pct / 100), minimum 1.
    Returns |A ∩ B| / n_top where A = top n_top observed indices,
    B = top n_top predicted indices.
    """
    n = len(observed)
    n_top = max(1, int(np.ceil(n * pct / 100)))
    top_obs  = set(np.argsort(observed,  kind="stable")[-n_top:])
    top_pred = set(np.argsort(predicted, kind="stable")[-n_top:])
    return float(len(top_obs & top_pred) / n_top)
