"""
Coefficient-draw uncertainty for the double-hurdle model.

Four toggles, all orthogonal:

  toggle 1 (draw_coefs):         do the N draw models differ in their
                                 linear-predictor coefficients (β) per stage,
                                 or share the MLE point estimate?
  toggle 2 (draw_scale):         do the N draw models differ in their
                                 dispersion / scale parameters per stage,
                                 or share the MLE?
  toggle 3 (outcome_draw):       at predict time, for bulk/tail, sample a
                                 realization from the predictive distribution,
                                 or return its analytical mean?
  toggle 4 (expected_bernoulli): at predict time, for S1/S2, contribute the
                                 stage's probability directly, or Bernoulli-
                                 flip it?

The (T, F, F, F) combination — outcome_draw=False, expected_bernoulli=True,
draw_coefs=False, draw_scale=False — reproduces
evaluate.assemble.assemble_predictions exactly. See test_draw_models.py.

Family coverage. The TOPSIS winner of the half-coupled run is:
    S1 = logit / free
    S2 = logit / free
    bulk = scaled_logit / free
    tail = gamma / free+weight

Anything outside this set raises NotImplementedError at construction time so
silent misbehaviour is impossible. Extend the LINK / FAMILY dispatches below
when a new winner needs draws.

Asymptotic distributions used per family:

  logit GLM (Binomial)        β ~ MVN(params, cov_params). No dispersion to draw.
  scaled_logit (Gaussian WLS) β ~ MVN(params, cov_params).
                              σ² ~ scale × df_resid / chi²(df_resid).
  gamma GLM (log link)        β ~ MVN(params, cov_params).
                              φ ~ scale × df_resid / chi²(df_resid).

Predictive realizations (toggle 3 = True):

  bulk (scaled_logit/Gaussian)
      z   = eta + N(0, σ²_eff)       σ²_eff = scale / weight_storm (= scale for unit weights)
      rate = threshold / (1 + exp(-z))

  tail (gamma/log)
      excess ~ Gamma(shape = w_eff / φ,
                     scale = mu × φ / w_eff)
      rate    = excess + threshold_rate

Reproducibility. Every randomness path (coefficient draws, scale draws,
S1/S2 Bernoulli flips, bulk/tail outcome draws) is driven from a single
``seed: int`` via ``numpy.random.SeedSequence``. Same seed → identical
results across runs and machines.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from idd_tc_mortality.features import align_X, build_X


# ---------------------------------------------------------------------------
# Stage-level container
# ---------------------------------------------------------------------------

@dataclass
class StageDraw:
    """One draw of one stage's parameters.

    Holds enough information to (a) build a design matrix from a storm DataFrame
    that matches the original training design matrix, (b) compute the linear
    predictor, (c) apply the inverse link to get the mean prediction, and (d)
    sample a realization from the predictive distribution.

    Attributes
    ----------
    stage:
        's1', 's2', 'bulk', or 'tail'.
    family:
        Family name as used in the spec: 'logit', 'scaled_logit', 'gamma'.
        (cloglog S1/S2 not implemented yet — add when a winner needs it.)
    exposure_mode:
        How log_exposed enters X / weights — 'free', 'free+weight', 'weight',
        'excluded', or 'offset' (S1 cloglog only).
    covariate_combo:
        Dict of covariate flags (e.g. {"basin": True, "is_island": True, ...}).
        Used by features.build_X.
    params:
        1-D array of coefficients — possibly drawn from MVN(mle, cov), or just
        the MLE if draw_coefs=False at build time.
    param_names:
        Column names in the same order as ``params``. Used to align X.
    scale:
        Dispersion / scale parameter — possibly drawn, possibly None for stages
        that have no dispersion (logit S1/S2).
    threshold_rate:
        For bulk/tail only; the rate-scale threshold the model was fit at. None
        for S1/S2.
    train_weight_mean:
        Mean of training-time observation weights. Needed only for predictive
        draws when exposure_mode is in ('weight', 'free+weight') — used to
        normalise a new storm's weight against the training distribution.
        None when weights were uniform.
    df_resid:
        Residual degrees of freedom — used to scale-draw σ² / φ. None for
        stages with no dispersion.
    """

    stage: str
    family: str
    exposure_mode: str
    covariate_combo: dict
    params: np.ndarray
    param_names: list[str]
    scale: float | None = None
    threshold_rate: float | None = None
    train_weight_mean: float | None = None
    df_resid: int | None = None


# ---------------------------------------------------------------------------
# Draw-model container
# ---------------------------------------------------------------------------

@dataclass
class DrawModel:
    """One complete coefficient draw across all four stages.

    Constructed by ``build_draw_models``. The ``predict`` method applies the
    full DH prediction pipeline to a DataFrame of storms in a single
    vectorised pass. Calling predict twice with the same storm_df and same
    seed yields byte-identical outputs.

    Attributes
    ----------
    draw_id:
        Integer index into the parent list of draw models (0 .. N-1).
    threshold_quantile:
        The quantile level the model was fit at (e.g. 0.70).
    threshold_rate:
        The rate-scale threshold computed from training data. Shared by
        bulk and tail stages.
    s1, s2, bulk, tail:
        StageDraw for each stage.
    """

    draw_id: int
    threshold_quantile: float
    threshold_rate: float
    s1: StageDraw
    s2: StageDraw
    bulk: StageDraw
    tail: StageDraw

    # -------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------

    def predict(
        self,
        storm_df: pd.DataFrame,
        *,
        outcome_draw: bool = True,
        expected_bernoulli: bool = False,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Predict per-storm deaths and death rate under this draw model.

        Parameters
        ----------
        storm_df:
            DataFrame of storms. Must contain the covariates flagged True
            in any stage's ``covariate_combo`` plus an ``exposed`` column
            (used both for log_exposed construction and for converting
            predicted rate to deaths).
        outcome_draw:
            Toggle 3. If True, bulk and tail return a realization from the
            predictive distribution; if False, they return the analytical
            mean. Independent of ``expected_bernoulli``.
        expected_bernoulli:
            Toggle 4. If True, S1 and S2 contribute their probabilities
            (p_s1, p_s2) directly rather than Bernoulli flips. The assembly
            becomes the closed-form expected hurdle:
                rate = p_s1 * (p_s2 * tail_rate + (1 - p_s2) * bulk_rate)
            With ``outcome_draw=False, expected_bernoulli=True,
            draw_coefs=False, draw_scale=False`` the result reproduces
            ``evaluate.assemble.assemble_predictions`` exactly. Default
            False preserves the stochastic-hurdle behaviour.
        seed:
            Seed for the per-call RNG. Same seed → same prediction.

        Returns
        -------
        pd.DataFrame
            Index matches ``storm_df``. Columns:
              ``p_s1``         — P(deaths>=1) for this storm under this draw.
              ``s1_flip``      — Bernoulli realization (0 or 1), NaN when
                                 ``expected_bernoulli=True``.
              ``p_s2``         — P(rate>=threshold | deaths>=1). NaN where
                                 s1_flip=0 (stochastic hurdle); defined for
                                 every row when ``expected_bernoulli=True``.
              ``s2_flip``      — Bernoulli realization, NaN where s1_flip=0,
                                 and NaN for every row when
                                 ``expected_bernoulli=True``.
              ``rate``         — predicted death rate.
              ``deaths``       — rate × exposed.
        """
        if "exposed" not in storm_df.columns:
            raise ValueError("storm_df must contain an 'exposed' column.")

        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(4)  # one stream per stage's random ops
        rng_s1, rng_s2, rng_bulk, rng_tail = (
            np.random.default_rng(s) for s in child_seeds
        )

        n = len(storm_df)
        log_exposed = np.log(storm_df["exposed"].values)

        # ---------- S1 ----------
        eta_s1 = _stage_eta(self.s1, storm_df, log_exposed)
        p_s1 = _inverse_link(self.s1, eta_s1)

        # ---------- S2 ----------
        eta_s2 = _stage_eta(self.s2, storm_df, log_exposed)
        p_s2 = _inverse_link(self.s2, eta_s2)

        # ---------- bulk and tail predictions (analytical or drawn) ----------
        eta_bulk = _stage_eta(self.bulk, storm_df, log_exposed)
        eta_tail = _stage_eta(self.tail, storm_df, log_exposed)

        if outcome_draw:
            bulk_rate = _bulk_draw(self.bulk, eta_bulk, storm_df, rng_bulk)
            tail_rate = _tail_draw(
                self.tail, eta_tail, storm_df, self.threshold_rate, rng_tail,
            )
        else:
            bulk_rate = _bulk_mean(self.bulk, eta_bulk)
            tail_rate = _tail_mean(self.tail, eta_tail, self.threshold_rate)

        # ---------- Assemble ----------
        if expected_bernoulli:
            rate = p_s1 * (p_s2 * tail_rate + (1.0 - p_s2) * bulk_rate)
            s1_flip_out = np.full(n, np.nan)
            s2_flip_out = np.full(n, np.nan)
            p_s2_out = p_s2
        else:
            s1_flip = rng_s1.binomial(1, p_s1)
            s2_flip_all = rng_s2.binomial(1, p_s2)
            rate = np.where(
                s1_flip == 0,
                0.0,
                np.where(s2_flip_all == 0, bulk_rate, tail_rate),
            )
            s1_flip_out = s1_flip
            p_s2_out = np.where(s1_flip == 0, np.nan, p_s2)
            s2_flip_out = np.where(s1_flip == 0, np.nan, s2_flip_all.astype(float))

        deaths = rate * storm_df["exposed"].values

        return pd.DataFrame(
            {
                "p_s1":    p_s1,
                "s1_flip": s1_flip_out,
                "p_s2":    p_s2_out,
                "s2_flip": s2_flip_out,
                "rate":    rate,
                "deaths":  deaths,
            },
            index=storm_df.index,
        )


# ---------------------------------------------------------------------------
# Linear predictor + family-specific helpers
# ---------------------------------------------------------------------------

def _stage_eta(
    stage: StageDraw,
    storm_df: pd.DataFrame,
    log_exposed: np.ndarray,
) -> np.ndarray:
    """Build X for a stage and return eta = X @ params (+ offset if applicable)."""
    include_log_exposed = _include_log_exposed(stage)
    X = build_X(storm_df, stage.covariate_combo, include_log_exposed=include_log_exposed)
    X = align_X(X, stage.param_names)
    eta = np.asarray(X) @ stage.params
    # S1 cloglog/offset adds log_exposed as a constrained offset, not a column.
    if stage.stage == "s1" and stage.exposure_mode == "offset":
        eta = eta + log_exposed
    return eta


def _include_log_exposed(stage: StageDraw) -> bool:
    """Mirror the logic in fit_one_component / predict_one_component."""
    if stage.stage in ("s1", "s2"):
        return stage.exposure_mode == "free"
    # bulk / tail rate models
    return stage.exposure_mode in ("free", "free+weight")


def _inverse_link(stage: StageDraw, eta: np.ndarray) -> np.ndarray:
    """Map linear predictor to probability for S1/S2."""
    if stage.stage in ("s1", "s2"):
        if stage.family == "logit":
            return 1.0 / (1.0 + np.exp(-eta))
        if stage.family == "cloglog":
            return 1.0 - np.exp(-np.exp(eta))
        raise NotImplementedError(
            f"S1/S2 family {stage.family!r} not supported by DrawModel. "
            "Add a branch to _inverse_link when needed."
        )
    raise ValueError(f"_inverse_link is for S1/S2; got stage={stage.stage!r}.")


def _bulk_mean(stage: StageDraw, eta: np.ndarray) -> np.ndarray:
    """Analytical mean rate from a bulk stage."""
    if stage.family == "scaled_logit":
        if stage.threshold_rate is None:
            raise ValueError("scaled_logit bulk requires threshold_rate.")
        return stage.threshold_rate / (1.0 + np.exp(-eta))
    raise NotImplementedError(
        f"Bulk family {stage.family!r} not supported by DrawModel. "
        "Add a branch to _bulk_mean / _bulk_draw when needed."
    )


def _bulk_draw(
    stage: StageDraw,
    eta: np.ndarray,
    storm_df: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """Realization rate from a bulk stage's predictive distribution."""
    if stage.family == "scaled_logit":
        # Gaussian on link scale: z* ~ N(eta, sigma_eff²).
        sigma2 = _effective_sigma2(stage, storm_df)
        z = eta + rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=eta.shape)
        return stage.threshold_rate / (1.0 + np.exp(-z))
    raise NotImplementedError(
        f"Bulk family {stage.family!r} not supported by DrawModel."
    )


def _tail_mean(stage: StageDraw, eta: np.ndarray, threshold_rate: float) -> np.ndarray:
    """Analytical mean rate from a tail stage."""
    if stage.family == "gamma":
        return np.exp(eta) + threshold_rate
    if stage.family == "log_logistic":
        # The fitted model's point prediction is the log-logistic median,
        # alpha = exp(eta). Matches distributions.log_logistic.predict.
        return np.exp(eta) + threshold_rate
    raise NotImplementedError(
        f"Tail family {stage.family!r} not supported by DrawModel. "
        "Add a branch to _tail_mean / _tail_draw when needed."
    )


def _tail_draw(
    stage: StageDraw,
    eta: np.ndarray,
    storm_df: pd.DataFrame,
    threshold_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Realization rate from a tail stage's predictive distribution."""
    if stage.family == "gamma":
        mu = np.exp(eta)
        phi = stage.scale
        w_eff = _effective_weight(stage, storm_df)
        # Variance of y = phi * mu^2 / w_eff (statsmodels Gamma + var_weights).
        # Equivalent Gamma parameterisation:
        #   shape       = w_eff / phi
        #   scale_param = mu * phi / w_eff
        shape = w_eff / phi
        scale_param = mu * phi / w_eff
        excess = rng.gamma(shape=shape, scale=scale_param)
        return excess + threshold_rate
    if stage.family == "log_logistic":
        # y ~ LogLogistic(alpha=exp(eta), k=stage.scale).
        # Inverse-CDF sampling: y = alpha * (u / (1-u))^(1/k), u ~ Uniform(0,1).
        # AFT predictive distribution doesn't scale with per-obs weight,
        # so train_weight_mean is unused here even when exposure_mode='free+weight'.
        alpha = np.exp(eta)
        k = stage.scale
        u = rng.uniform(low=0.0, high=1.0, size=eta.shape)
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        excess = alpha * (u / (1.0 - u)) ** (1.0 / k)
        return excess + threshold_rate
    raise NotImplementedError(
        f"Tail family {stage.family!r} not supported by DrawModel."
    )


def _effective_weight(stage: StageDraw, storm_df: pd.DataFrame) -> np.ndarray:
    """w_eff for predictive variance: 1 for uniform-weighted fits, else
    storm_exposed / train_weight_mean."""
    if stage.exposure_mode in ("weight", "free+weight"):
        if stage.train_weight_mean is None:
            raise ValueError(
                f"{stage.stage!r} ({stage.family}/{stage.exposure_mode}) was built "
                "without train_weight_mean; cannot compute predictive variance."
            )
        return storm_df["exposed"].values / stage.train_weight_mean
    # Uniform fit weights -> effective weight 1.
    return np.ones(len(storm_df))


def _effective_sigma2(stage: StageDraw, storm_df: pd.DataFrame) -> np.ndarray:
    """σ²_eff = scale / w_eff for a Gaussian WLS bulk stage."""
    w_eff = _effective_weight(stage, storm_df)
    return stage.scale / w_eff


# ---------------------------------------------------------------------------
# Build draw models from refit_with_objects output
# ---------------------------------------------------------------------------

def build_draw_models(
    refit_out: dict,
    focus_model: dict,
    data: pd.DataFrame,
    *,
    n_draws: int = 100,
    draw_coefs: bool = True,
    draw_scale: bool = True,
    seed: int = 42,
) -> list[DrawModel]:
    """Build N draw models from the IS fits of a focus model.

    Parameters
    ----------
    refit_out:
        Output of ``idd_tc_mortality.refit_with_objects.refit_model_with_objects``.
        Only the ``refit_out['is']`` entries are consumed (one per stage).
    focus_model:
        The dict that produced ``refit_out`` — needed to read covariate combos
        and exposure modes per stage. Same shape as ``topsis_df.iloc[0].to_dict()``.
    data:
        Training DataFrame. Used to (a) read the rate-scale threshold from
        ``refit_out['is']['combined']['threshold_rate']`` and (b) compute the
        training mean of observation weights for stages fit with
        ``exposure_mode in ('weight', 'free+weight')``.
    n_draws:
        Number of draw models. Default 100. When both toggles are off, the
        returned list still has length n_draws — all identical.
    draw_coefs:
        Toggle 1. If True, β is independently drawn from each stage's MVN
        for every draw. If False, every draw uses the MLE β.
    draw_scale:
        Toggle 2. If True, the dispersion / scale parameter for stages that
        have one (scaled_logit, gamma) is drawn from its asymptotic
        distribution. If False, every draw uses the MLE scale. No effect on
        logit S1/S2 (no dispersion).
    seed:
        Master seed. Same value → identical draw models. Default 42.

    Returns
    -------
    list[DrawModel]
        Length ``n_draws``.

    Raises
    ------
    RuntimeError
        If any of the four IS stages failed during the refit.
    """
    for stage in ("s1", "s2", "bulk", "tail"):
        if refit_out["is"][stage].get("failed", False):
            raise RuntimeError(
                f"Cannot build draw models: stage {stage!r} IS fit failed. "
                f"Error: {refit_out['is'][stage]['metrics'].get('__failed__')}"
            )

    threshold_rate = refit_out["is"]["combined"]["threshold_rate"]
    threshold_quantile = float(focus_model["threshold_quantile"])

    # Pre-compute per-stage MLE arrays + draws (all independent across stages
    # by construction since the four fits are independent).
    ss = np.random.SeedSequence(seed)
    stage_seeds = dict(zip(("s1", "s2", "bulk", "tail"), ss.spawn(4)))

    stage_kits = {}
    for stage in ("s1", "s2", "bulk", "tail"):
        stage_kits[stage] = _prepare_stage(
            stage=stage,
            refit_entry=refit_out["is"][stage],
            focus_model=focus_model,
            data=data,
            threshold_rate=threshold_rate,
            n_draws=n_draws,
            draw_coefs=draw_coefs,
            draw_scale=draw_scale,
            seed=stage_seeds[stage],
        )

    # Pack one DrawModel per draw id, pulling draw k from each stage's kit.
    models: list[DrawModel] = []
    for k in range(n_draws):
        s1_d   = _stage_draw_from_kit(stage_kits["s1"],   k)
        s2_d   = _stage_draw_from_kit(stage_kits["s2"],   k)
        bulk_d = _stage_draw_from_kit(stage_kits["bulk"], k)
        tail_d = _stage_draw_from_kit(stage_kits["tail"], k)
        models.append(
            DrawModel(
                draw_id=k,
                threshold_quantile=threshold_quantile,
                threshold_rate=threshold_rate,
                s1=s1_d, s2=s2_d, bulk=bulk_d, tail=tail_d,
            )
        )
    return models


# ---------------------------------------------------------------------------
# Per-stage preparation: extract MLE + draws from refit raw object
# ---------------------------------------------------------------------------

def _prepare_stage(
    *,
    stage: str,
    refit_entry: dict,
    focus_model: dict,
    data: pd.DataFrame,
    threshold_rate: float,
    n_draws: int,
    draw_coefs: bool,
    draw_scale: bool,
    seed: np.random.SeedSequence,
) -> dict:
    """Pull params / cov / scale / df_resid from a raw statsmodels result and
    pre-compute the N draws of (β, scale) under the current toggles."""
    raw = refit_entry["raw_object"]
    spec = refit_entry["spec"]
    fit_result = refit_entry["fit_result"]

    if raw is None:
        raise RuntimeError(
            f"Stage {stage!r} has no raw_object — cannot extract cov_params. "
            "Family with custom optimiser? Extend _prepare_stage if so."
        )

    family = spec["family"] if stage not in ("s1", "s2") else focus_model[f"{stage}_family"]
    exposure_mode = focus_model[f"{stage}_exposure_mode"]
    covariate_combo = spec["covariate_combo"]
    param_names = list(fit_result.param_names)
    params_mle = np.asarray(fit_result.params)

    # log_logistic uses scipy BFGS, not statsmodels; cov_params is the BFGS
    # hess_inv covering (beta, log_k). Branch early so the rest of the function
    # can keep the statsmodels assumptions.
    if family == "log_logistic":
        return _prepare_stage_log_logistic(
            stage=stage,
            raw=raw,
            fit_result=fit_result,
            exposure_mode=exposure_mode,
            covariate_combo=covariate_combo,
            param_names=param_names,
            params_mle=params_mle,
            spec=spec,
            data=data,
            threshold_rate=threshold_rate,
            n_draws=n_draws,
            draw_coefs=draw_coefs,
            draw_scale=draw_scale,
            seed=seed,
        )

    has_dispersion = family in ("scaled_logit", "gamma")
    scale_mle: float | None
    df_resid: int | None
    try:
        cov_params = np.asarray(raw.cov_params())
    except Exception as exc:
        raise RuntimeError(
            f"Stage {stage!r} ({family}): could not extract cov_params from raw object: {exc}"
        ) from exc

    if has_dispersion:
        # WLSResults and Gamma GLMResults both expose .scale and .df_resid.
        scale_mle = float(raw.scale)
        df_resid = int(raw.df_resid)
    else:
        scale_mle = None
        df_resid = None

    # Training-weight mean — needed at predict time for variance scaling.
    train_weight_mean = _compute_train_weight_mean(
        stage=stage, exposure_mode=exposure_mode, spec=spec, data=data,
        threshold_rate=threshold_rate,
    )

    rng = np.random.default_rng(seed)

    # ----- Coefficient draws -----
    if draw_coefs:
        params_draws = rng.multivariate_normal(
            mean=params_mle, cov=cov_params, size=n_draws,
        )
    else:
        params_draws = np.tile(params_mle, (n_draws, 1))

    # ----- Scale draws -----
    scale_draws: np.ndarray | None
    if scale_mle is None:
        scale_draws = None
    elif draw_scale:
        # σ² (or φ) follows scale × df_resid / chi²(df_resid) asymptotically.
        chi2 = rng.chisquare(df=df_resid, size=n_draws)
        # Guard against pathological chi2=0 (would give inf scale).
        chi2 = np.where(chi2 < 1e-12, 1e-12, chi2)
        scale_draws = scale_mle * df_resid / chi2
    else:
        scale_draws = np.full(n_draws, scale_mle, dtype=float)

    return {
        "stage":             stage,
        "family":            family,
        "exposure_mode":     exposure_mode,
        "covariate_combo":   covariate_combo,
        "param_names":       param_names,
        "params_draws":      params_draws,        # shape (n_draws, p)
        "scale_draws":       scale_draws,         # shape (n_draws,) or None
        "threshold_rate":    threshold_rate if stage in ("bulk", "tail") else None,
        "train_weight_mean": train_weight_mean,
        "df_resid":          df_resid,
    }


def _psd_project(cov: np.ndarray) -> np.ndarray:
    """Project a near-symmetric, near-PSD matrix onto the PSD cone.

    Symmetrizes (fixes round-off asymmetries) then clamps any negative
    eigenvalues to zero. Used to clean up scipy BFGS hess_inv before treating
    it as an asymptotic covariance.
    """
    sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * eigvals) @ eigvecs.T


def _prepare_stage_log_logistic(
    *,
    stage: str,
    raw,
    fit_result,
    exposure_mode: str,
    covariate_combo: dict,
    param_names: list[str],
    params_mle: np.ndarray,
    spec: dict,
    data: pd.DataFrame,
    threshold_rate: float,
    n_draws: int,
    draw_coefs: bool,
    draw_scale: bool,
    seed: np.random.SeedSequence,
) -> dict:
    """Kit builder for log_logistic tails.

    raw is the scipy.optimize.OptimizeResult from BFGS. raw.x = [beta, log_k];
    raw.hess_inv is the (p+1)x(p+1) joint covariance over (beta, log_k).
    FitResult.params is just beta; shape_param k lives in meta['shape_param'].
    """
    cov_full = np.asarray(raw.hess_inv)
    n_beta = len(params_mle)
    if cov_full.shape != (n_beta + 1, n_beta + 1):
        raise RuntimeError(
            f"Stage {stage!r} (log_logistic): expected hess_inv shape "
            f"{(n_beta + 1, n_beta + 1)}, got {cov_full.shape}."
        )
    # BFGS hess_inv can drift slightly asymmetric / non-PSD over iterations
    # (the update preserves symmetry only in exact arithmetic). Symmetrize and
    # project to PSD before using as a covariance, so multivariate_normal
    # doesn't emit RuntimeWarnings and the draws are valid.
    cov_full = _psd_project(cov_full)
    shape_param = float(fit_result.meta["shape_param"])
    log_k_mle = float(np.log(shape_param))

    train_weight_mean = _compute_train_weight_mean(
        stage=stage, exposure_mode=exposure_mode, spec=spec, data=data,
        threshold_rate=threshold_rate,
    )

    rng = np.random.default_rng(seed)

    if draw_coefs and draw_scale:
        joint = rng.multivariate_normal(
            mean=np.concatenate([params_mle, [log_k_mle]]),
            cov=cov_full,
            size=n_draws,
        )
        params_draws = joint[:, :n_beta]
        scale_draws  = np.exp(joint[:, n_beta])
    elif draw_coefs:
        cov_beta = cov_full[:n_beta, :n_beta]
        params_draws = rng.multivariate_normal(
            mean=params_mle, cov=cov_beta, size=n_draws,
        )
        scale_draws  = np.full(n_draws, shape_param, dtype=float)
    elif draw_scale:
        log_k_var = float(cov_full[n_beta, n_beta])
        params_draws = np.tile(params_mle, (n_draws, 1))
        log_k_draws  = rng.normal(
            loc=log_k_mle,
            scale=np.sqrt(max(log_k_var, 0.0)),
            size=n_draws,
        )
        scale_draws  = np.exp(log_k_draws)
    else:
        params_draws = np.tile(params_mle, (n_draws, 1))
        scale_draws  = np.full(n_draws, shape_param, dtype=float)

    return {
        "stage":             stage,
        "family":            "log_logistic",
        "exposure_mode":     exposure_mode,
        "covariate_combo":   covariate_combo,
        "param_names":       param_names,
        "params_draws":      params_draws,
        "scale_draws":       scale_draws,
        "threshold_rate":    threshold_rate,
        "train_weight_mean": train_weight_mean,
        "df_resid":          None,
    }


def _compute_train_weight_mean(
    *, stage: str, exposure_mode: str, spec: dict,
    data: pd.DataFrame, threshold_rate: float,
) -> float | None:
    """Mean of the observation weights used at fit time.

    Only meaningful for rate-model stages with ``exposure_mode in
    ('weight', 'free+weight')`` — those fits used ``exposed`` as
    ``var_weights`` (gamma / lognormal / scaled_logit). For uniform-weight
    fits, returns None.
    """
    if stage not in ("bulk", "tail"):
        return None
    if exposure_mode not in ("weight", "free+weight"):
        return None

    death_rate = data["deaths"].values / data["exposed"].values
    if stage == "bulk":
        mask = (data["deaths"].values >= 1) & (death_rate < threshold_rate)
    else:  # tail
        mask = death_rate >= threshold_rate
    weights = data.loc[mask, "exposed"].values.astype(float)
    if len(weights) == 0:
        return None
    return float(np.mean(weights))


def _stage_draw_from_kit(kit: dict, k: int) -> StageDraw:
    """Pluck the k-th draw out of a kit and pack it into a StageDraw."""
    params_k = kit["params_draws"][k]
    scale_k: float | None = (
        float(kit["scale_draws"][k]) if kit["scale_draws"] is not None else None
    )
    return StageDraw(
        stage=kit["stage"],
        family=kit["family"],
        exposure_mode=kit["exposure_mode"],
        covariate_combo=kit["covariate_combo"],
        params=np.asarray(params_k),
        param_names=list(kit["param_names"]),
        scale=scale_k,
        threshold_rate=kit["threshold_rate"],
        train_weight_mean=kit["train_weight_mean"],
        df_resid=kit["df_resid"],
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_draw_models(models: list[DrawModel], path: str | Path) -> None:
    """Pickle a list of DrawModel objects to ``path``.

    Format: a single pickle file containing the list. The DrawModel /
    StageDraw classes hold only numpy arrays and primitive types — no
    statsmodels objects retained — so the pickle is small (~hundreds of KB
    for 100 draws) and fast to load.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_draw_models(path: str | Path) -> list[DrawModel]:
    """Inverse of save_draw_models."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
