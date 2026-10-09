"""Where the late deliveries come from, and when.

    python src/lateness_anatomy.py      # writes report/results/lateness_*.csv

Phase 1 attributed the spikes in the monthly late rate to Black Friday and to
the May 2018 truckers' strike. The Milestone 1 feedback pointed out that the
figure does not support the second claim: the largest peaks sit in February and
March 2018, and May is unremarkable. This script is the careful version of that
analysis, at three resolutions:

    lateness_monthly.csv   by purchase month: volume, late rate, lead time, the
                           promised window, and lead time split into the seller's
                           handling time (purchase -> carrier handoff) and the
                           carrier's transit time (handoff -> customer)
    lateness_weekly.csv    the same by purchase week (Monday to Sunday), which
                           is the resolution at which a ten-day strike can be
                           seen at all
    lateness_strike.csv    the cohorts around the strike of 21 May - 1 June 2018:
                           orders already in transit when the roads closed,
                           orders placed during it, and the May orders it never
                           touched

Nothing here is a model input. The split of lead time uses
order_delivered_carrier_date, a post-outcome column that is forbidden to every
predictor; it is admissible for describing what happened after the fact.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
from data_load import load_all, load_analysis
from features import build_features

RESULTS = config.PROJECT_ROOT / "report" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# The strike: nationwide road blockades from Monday 21 May 2018; most
# demonstrations had ended by 1 June 2018. Correios suspended its time-definite
# Sedex services from 23 May and added five business days to standard Sedex and
# PAC. (Sources are cited in the report's bibliography.)
STRIKE_START = pd.Timestamp("2018-05-21")
STRIKE_END = pd.Timestamp("2018-06-01")
# Other calendar events used for annotation.
BLACK_FRIDAY = pd.Timestamp("2017-11-24")
CARNIVAL_START, CARNIVAL_END = pd.Timestamp("2018-02-09"), pd.Timestamp("2018-02-14")


def _agg(frame: pd.DataFrame, key) -> pd.DataFrame:
    """One row per group with the quantities every table below shares."""
    out = frame.groupby(key).agg(
        n=("order_id", "size"),
        late_pct=("target_is_late", lambda s: 100 * s.mean()),
        mean_lead=("target_delivery_days", "mean"),
        median_lead=("target_delivery_days", "median"),
        promise_days=("estimated_days", "mean"),
        seller_days=("seller_days", "mean"),
        carrier_days=("carrier_days", "mean"),
        mean_delay=("delay_vs_estimate_days", "mean"),
    )
    return out.round(4)


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = df.dropna(subset=["target_delivery_days"]).copy()
    ts = d["order_purchase_timestamp"]
    d["seller_days"] = ((d["order_delivered_carrier_date"] - ts)
                        .dt.total_seconds() / 86400)
    d["carrier_days"] = ((d["order_delivered_customer_date"]
                          - d["order_delivered_carrier_date"])
                         .dt.total_seconds() / 86400)

    # -- monthly, by purchase month -------------------------------------------
    monthly = _agg(d, ts.dt.to_period("M"))
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "purchase_month"
    all_orders = df.groupby(df["order_purchase_timestamp"].dt.to_period("M")).size()
    all_orders.index = all_orders.index.astype(str)
    monthly.insert(0, "all_orders", all_orders.reindex(monthly.index).to_numpy())

    # -- weekly, Monday-to-Sunday weeks labelled by their Monday ---------------
    week = ts.dt.to_period("W-SUN").dt.start_time
    weekly = _agg(d, week)
    weekly.index.name = "week_start"
    weekly = weekly.reset_index()
    weekly["week_start"] = weekly["week_start"].dt.date.astype(str)
    weekly["in_strike"] = ((pd.to_datetime(weekly["week_start"]) >= STRIKE_START - pd.Timedelta(days=6))
                           & (pd.to_datetime(weekly["week_start"]) < STRIKE_END))

    # -- the strike cohorts -----------------------------------------------------
    delivered_at = d["order_delivered_customer_date"]
    cohorts = {
        "May 1-20, delivered before the strike":
            ts.between("2018-05-01", "2018-05-20 23:59:59") & (delivered_at < STRIKE_START),
        "in transit during the strike (bought before 1 Jun, arrived on/after 21 May)":
            (ts < STRIKE_END) & (delivered_at >= STRIKE_START),
        "bought during the strike (21 May - 1 Jun)":
            ts.between(STRIKE_START, STRIKE_END - pd.Timedelta(seconds=1)),
        "bought 2-30 June, after the strike":
            ts.between("2018-06-02", "2018-06-30 23:59:59"),
    }
    strike = pd.DataFrame([
        {"cohort": name, **_agg(d[mask].assign(_g=0), "_g").iloc[0].to_dict()}
        for name, mask in cohorts.items()
    ])

    # -- the February-March 2018 peak against everything else -----------------
    peak = ts.dt.to_period("M").astype(str).isin(["2018-02", "2018-03"])
    anatomy = _agg(d.assign(_g=np.where(peak, "Feb-Mar 2018", "rest of window")), "_g")
    anatomy.index.name = "period"
    by_region = (d[peak].groupby("customer_region")["target_is_late"].mean().mul(100)
                 .rename("peak_late_pct").to_frame()
                 .join(d[~peak].groupby("customer_region")["target_is_late"].mean()
                       .mul(100).rename("rest_late_pct"))
                 .join(d[peak].groupby("customer_region").size().rename("n_peak"))
                 .round(2).reset_index())

    return {"lateness_monthly.csv": monthly.reset_index(),
            "lateness_weekly.csv": weekly,
            "lateness_strike.csv": strike,
            "lateness_peak_anatomy.csv": anatomy.reset_index(),
            "lateness_peak_by_region.csv": by_region}


if __name__ == "__main__":
    print("loading nine tables and building the Phase 1 frame (no history features)")
    frame = build_features(load_analysis(load_all()))
    tables = build_tables(frame)
    pd.set_option("display.width", 200)
    for name, table in tables.items():
        table.to_csv(RESULTS / name, index=False)
        print(f"\n--- {name} ({len(table)} rows)")
        if name == "lateness_weekly.csv":
            w = table.copy()
            print(w[(w["week_start"] >= "2018-01-29") & (w["week_start"] <= "2018-07-09")]
                  [["week_start", "n", "late_pct", "mean_lead", "promise_days",
                    "carrier_days"]].round(2).to_string(index=False))
        else:
            print(table.round(2).to_string(index=False))
