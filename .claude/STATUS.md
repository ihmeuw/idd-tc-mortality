# Project status
Updated: 2026-04-16

## Goals
Build a double-hurdle peaks-over-threshold model for tropical cyclone mortality estimation.
Four-component model: S1 (P(deaths>=1)), S2 (P(rate>=threshold | S1=1)), bulk rate, tail rate.
Pipeline stages: validate → grid → fit → evaluate → select.

## Recent steps
- 2026-04-15: Rewrote 01_prepare_input_data.py — level-filter modes (all/level3/aggregate), default=all; 9,186-row dataset
- 2026-04-16: Added truncated_normal, weibull, log_logistic distributions (tail rate families)
- 2026-04-16: Grid expanded to 831 specs with exposure_mode field (4 modes for rate, offset for count)
- 2026-04-16: Data invalidated — all prior model run output deleted; new data expected 2026-04-17

## Next steps
1. Implement graceful non-convergence in fit_component.py (catch errors, save non-converged result)
2. Ingest new data via 01_prepare_input_data.py → 00-data/20260417/; update current symlink
3. Submit new preliminary run: run-fit-orchestrate --mode preliminary --output-dir .../01-preliminary/20260417
4. run-evaluate on completed run; check pred_obs_ratio ≈ 1.0

## Parking lot
- Uncertainty module (FitResult.cov via hess_inv draws)
- GPD xi uncertainty: joint [beta, xi] draws from hess_inv — documented in DECISIONS.md
- Vignette notebook: load dh_results → ModelQuery → TOPSIS → StagePlotter
- Condorcet/pairwise_dominance_summary as 8th method in model_selection.py
