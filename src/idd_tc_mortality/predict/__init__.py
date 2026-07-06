"""Predict-and-aggregate pipeline for tropical-cyclone direct-risk projections.

Six-tier jobmon DAG, one branch per storm_draw:

    1. setup                 — per storm_draw      (100 tasks)
    2. predict               — per (sd, sc, yb, b) (~14k tasks; inner loop over tc_draws)
    3. aggregate_basin       — per (sd, sc, yb, b) (~14k tasks)
    4. aggregate_year_bin    — per (sd, sc, yb)    (~2k tasks)
    5. aggregate_scenario    — per (sd, sc)        (~400 tasks)
    6. aggregate_storm_draw  — per sd              (100 tasks)

Each tier's CLI entry point is registered in pyproject.toml.
"""
