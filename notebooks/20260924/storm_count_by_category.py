"""Global TC counts by Saffir-Simpson category, CLIMADA-based max wind.

Mirrors plot_gl_ensemble() in idd-climate-models/notebooks/detect_time_bins.ipynb
(GL basin, 10-90th percentile envelope across model/variant members, thick
multi-model mean, IBTrACS dots) but counts storms from the consolidated CLIMADA
exposure frame (model draws) and the IBTrACS CLIMADA storm files (observed).

Storm definition: a storm with at least one exposed admin0 row in the CLIMADA
frame (land-affecting). Category: per-storm max of CLIMADA max_wind_speed (m/s)
over exposed admin0 units. Thresholds follow idd_climate_models.constants
storm_intensity_dict: Cat 3 >= 50, Cat 4 >= 58, Cat 5 >= 70 m/s.
Model side: count per (member, tc_risk_draw, year), zero-filled, then mean over
the 100 tc_risk_draws per member; envelope/mean are across the 8 members.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLIMADA = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output")
MODEL_FRAME = CLIMADA / "stage4b_v2/_consolidated/storm_exposure_all.parquet"
IBTRACS_ROOT = CLIMADA / "ibtracs_stage4b_pafs_admin0/ibtracs/official/historical"
OUT_DIR = Path("/mnt/team/rapidresponse/pub/tropical-storms/figures/CLIMADA")

YEAR_RANGES = {"historical": (1970, 2014), "ssp126": (2015, 2100),
               "ssp245": (2015, 2100), "ssp585": (2015, 2100)}
CATEGORIES = {"Cat 3 or greater": 50.0, "Cat 4 or greater": 58.0, "Cat 5": 70.0}
SCEN_COLORS = {"historical": "#1f77b4", "ssp126": "#2ca02c",
               "ssp245": "#ff7f0e", "ssp585": "#d62728"}
SCEN_NAMES = {"historical": "Historical", "ssp126": "RCP2.6",
              "ssp245": "RCP4.5", "ssp585": "RCP8.5"}
SVL = "source_id_variant_label"


def model_member_counts() -> pd.DataFrame:
    """Per (category, experiment, member, year): mean over tc_risk_draws of storm count."""
    ds = pads.dataset(str(MODEL_FRAME))
    out = []
    for exp, (y0, y1) in YEAR_RANGES.items():
        t = time.time()
        tb = ds.to_table(columns=[SVL, "tc_risk_draw", "year", "storm_id", "max_wind_speed"],
                         filter=pc.field("experiment_id") == exp)
        g = (tb.group_by([SVL, "tc_risk_draw", "year", "storm_id"])
               .aggregate([("max_wind_speed", "max")]).to_pandas())
        members = g[[SVL, "tc_risk_draw"]].drop_duplicates()
        years = np.arange(y0, y1 + 1)
        full = pd.MultiIndex.from_frame(
            members.merge(pd.DataFrame({"year": years}), how="cross"))
        for cat, thr in CATEGORIES.items():
            cnt = (g[g["max_wind_speed_max"] >= thr]
                   .groupby([SVL, "tc_risk_draw", "year"]).size()
                   .reindex(full, fill_value=0).rename("count").reset_index())
            mem = cnt.groupby([SVL, "year"])["count"].mean().reset_index()
            mem["experiment"], mem["category"] = exp, cat
            out.append(mem)
        print(f"  {exp}: {tb.num_rows:,} rows, {len(g):,} storm-draws, "
              f"{members[SVL].nunique()} members x {members['tc_risk_draw'].nunique()} draws, "
              f"{time.time() - t:.1f}s", flush=True)
    return pd.concat(out, ignore_index=True)


def ibtracs_counts() -> pd.DataFrame:
    """Per (category, year): observed storm count from the IBTrACS CLIMADA files."""
    files = sorted(IBTRACS_ROOT.glob("*/*/storm_paf/storm_*.parquet"))
    years = sorted({int(p.name) for p in IBTRACS_ROOT.iterdir() if p.is_dir()})
    tb = pads.dataset([str(f) for f in files]).to_table(
        columns=["storm_id", "year", "max_wind_speed"])
    g = tb.group_by(["storm_id", "year"]).aggregate([("max_wind_speed", "max")]).to_pandas()
    g["year"] = g["year"].astype(int)
    out = []
    for cat, thr in CATEGORIES.items():
        cnt = (g[g["max_wind_speed_max"] >= thr].groupby("year").size()
               .reindex(years, fill_value=0).rename("count").reset_index())
        cnt["category"] = cat
        out.append(cnt)
    print(f"  IBTrACS: {len(files)} files, {len(g)} storms, years {years[0]}-{years[-1]}", flush=True)
    return pd.concat(out, ignore_index=True)


def plot(model_df: pd.DataFrame, obs_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(len(CATEGORIES), 1, figsize=(14, 5 * len(CATEGORIES)), sharex=True)
    for ax, cat in zip(axes, CATEGORIES):
        m = model_df[model_df["category"] == cat]
        for exp in YEAR_RANGES:
            scen = m[m["experiment"] == exp]
            agg = scen.groupby("year")["count"].agg(
                lo=lambda s: s.quantile(0.10), hi=lambda s: s.quantile(0.90), mean="mean").sort_index()
            ax.fill_between(agg.index, agg["lo"], agg["hi"], color=SCEN_COLORS[exp], alpha=0.20, linewidth=0)
            ax.plot(agg.index, agg["mean"], color=SCEN_COLORS[exp], linewidth=3,
                    label=f"{SCEN_NAMES[exp]} (multi-model mean)")
        o = obs_df[obs_df["category"] == cat]
        ax.scatter(o["year"], o["count"], color="black", s=25, zorder=10, alpha=0.8, label="IBTrACS (observed)")
        ax.set_xlim(1970, 2100)
        ax.set_ylabel("Storm Count", fontsize=12)
        ax.set_title(f"{cat} (CLIMADA max wind >= {CATEGORIES[cat]:.0f} m/s over exposed admin0)",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=10)
    axes[-1].set_xlabel("Year", fontsize=12)
    fig.suptitle("Land-affecting global TC frequency by scenario and Saffir-Simpson category\n"
                 "(members: 8 model/variant realizations, each the mean over 100 CLIMADA draws; "
                 "band = 10th-90th percentile across members)", fontsize=13)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    t0 = time.time()
    print("Model frame ...", flush=True)
    model_df = model_member_counts()
    print("IBTrACS ...", flush=True)
    obs_df = ibtracs_counts()
    fig = plot(model_df, obs_df)
    png = OUT_DIR / "storm_count_by_category.png"
    fig.savefig(png, dpi=150)
    print(f"\nSaved {png}  ({time.time() - t0:.0f}s)")
    summ = (model_df.groupby(["category", "experiment"])["count"].mean().unstack("experiment").round(2))
    print("\nMean annual count per member (model, all years):\n", summ.to_string())
    print("\nIBTrACS mean annual count 1970-2014:\n",
          obs_df.groupby("category")["count"].mean().round(2).to_string())
