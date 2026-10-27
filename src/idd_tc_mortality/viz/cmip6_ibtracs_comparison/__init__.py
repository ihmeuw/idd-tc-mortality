"""CMIP6 (tc-risk / CLIMADA) vs IBTrACS comparability report.

Stages (``run-cmip6-ibtracs-comparison``): ``aggregate`` (storm counts by category and
geography, observed deaths) → ``predict`` (shipped mortality model on CMIP6 and on IBTrACS
storms) → ``stats`` (rank-based match measures, raw and detrended, moving-average and
block windows) → ``export`` (JSON + simplified geometry for the interactive artifact).
"""
