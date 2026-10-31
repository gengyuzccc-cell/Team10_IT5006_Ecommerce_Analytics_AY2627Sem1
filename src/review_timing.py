"""When is the review written: before or after the parcel arrives?

    python src/train_models.py      # first: writes the per-order test predictions
    python src/review_timing.py     # then: writes report/results/review_timing*.csv

Olist emails the satisfaction survey when the parcel is delivered or, if it has
not arrived, when the promised date is reached (review_creation_date is the
date the survey was sent, review_answer_timestamp the moment it was answered).
So for a late parcel the customer is usually asked to review while still
waiting. That matters for Problem 2: the model is framed "at delivery, before
the review is written", which holds for parcels that arrive on time but not for
most late ones - their review is already in when the parcel lands.

This script measures that timing on the analysis window, decomposes the
held-out call list into customers still reachable before they write and
customers who had already written, and restates the expected benefit counting
only the reachable ones. Nothing is refitted; the per-order test predictions are
the ones train_models.py wrote.

    review_timing.csv            share of reviews written before arrival, by group
    review_timing_lateness.csv   1-2 star rate by how early or late the parcel came
    review_timing_capacity.csv   the test-window call list split into reachable /
                                 already-written hits, pooled top-K% and per day
    review_timing_value.csv      expected net benefit with all hits and with the
                                 reachable hits only, by threshold
    story_order.csv              the order used to open the presentation: the top
                                 of the 16 July 2018 replay list
    review_timing_test_orders.csv  one row per held-out test order: was its
                                 review written before the parcel arrived? Read
                                 by the dashboard's Replay tab
    review_timing_subsets.csv    the review-risk score on all, delivered, on-time,
                                 late and never-delivered test orders separately
                                 (does it only restate lateness?)
    review_timing_composition.csv  where the daily top-ten hits come from (never
                                 delivered / late, written before arrival / still
                                 reachable), and the same list on delivered orders
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
STORY_DAY = pd.Timestamp("2018-07-16").date()


def need(name: str, **kw) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `python src/train_models.py` first.")
    return pd.read_csv(path, **kw)


def timing_frame(tables: dict) -> pd.DataFrame:
    """Delivered, reviewed orders of the analysis window with the timing flags."""
    df = build_features(load_analysis(tables))
    d = df.dropna(subset=["delivery_days", "target_low_review",
                         "review_answer_timestamp", "order_delivered_customer_date"]).copy()
    d["late"] = d["is_late"] == 1
    d["low"] = d["target_low_review"] == 1
    d["written_before_arrival"] = d["review_answer_timestamp"] < d["order_delivered_customer_date"]
    d["survey_before_arrival"] = (d["review_creation_date"]
                                  < d["order_delivered_customer_date"].dt.normalize())
    d["survey_on_or_after_promise"] = (d["review_creation_date"]
                                       >= d["order_estimated_delivery_date"])
    d["answer_days_after_delivery"] = ((d["review_answer_timestamp"]
                                        - d["order_delivered_customer_date"])
                                       .dt.total_seconds() / 86400)
    return d


def group_table(d: pd.DataFrame) -> pd.DataFrame:
    groups = [("all", d), ("late", d[d.late]), ("on time", d[~d.late]),
              ("low review", d[d.low]), ("low review, late", d[d.low & d.late]),
              ("low review, on time", d[d.low & ~d.late])]
    rows = []
    for name, g in groups:
        rows.append({"group": name, "n": len(g),
                     "pct_written_before_arrival": 100 * g["written_before_arrival"].mean(),
                     "pct_survey_before_arrival": 100 * g["survey_before_arrival"].mean(),
                     "pct_survey_on_or_after_promise": 100 * g["survey_on_or_after_promise"].mean(),
                     "median_answer_days_after_delivery": g["answer_days_after_delivery"].median(),
                     "pct_low_review": 100 * g["low"].mean()})
    out = pd.DataFrame(rows)
    out["share_of_all_low_reviews_pct"] = np.nan
    lows = d["low"].sum()
    for name, mask in [("low review, late", d.low & d.late), ("low review, on time", d.low & ~d.late)]:
        out.loc[out["group"] == name, "share_of_all_low_reviews_pct"] = 100 * mask.sum() / lows
    return out


def lateness_buckets(d: pd.DataFrame) -> pd.DataFrame:
    """1-2 star rate by how early or late the parcel arrived against its promise."""
    bins = [-np.inf, -10, 0, 3, 7, 14, np.inf]
    labels = ["10+ days early", "0-10 days early", "1-3 days late", "4-7 days late",
              "8-14 days late", "15+ days late"]
    b = pd.cut(d["delay_vs_estimate_days"], bins=bins, labels=labels, right=True)
    return (d.groupby(b, observed=True)["low"]
            .agg(n="size", low_pct=lambda s: 100 * s.mean())
            .rename_axis("bucket").reset_index())


def capacity_split(p: pd.DataFrame) -> pd.DataFrame:
    """The held-out call list, hits split into reachable and already written."""
    rows = []
    n = len(p)
    ranked = p.sort_values("p_low_review", ascending=False, kind="stable").reset_index(drop=True)
    for f in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20):
        top = ranked.head(max(1, int(round(f * n))))
        rows.append({"kind": "pooled", "k": f"{100 * f:g}%", "contacts": len(top),
                     "precision": top["low"].mean(),
                     "reachable_precision": top["reachable"].mean(),
                     "already_written_precision": (top["low"] & ~top["reachable"]).mean(),
                     "late_share": top["late"].mean()})
    p = p.copy()
    p["day"] = p["order_purchase_timestamp"].dt.date
    daily = p.sort_values(["day", "p_low_review"], ascending=[True, False], kind="stable")
    daily["pos"] = daily.groupby("day").cumcount()
    for k in (5, 10, 20):
        top = daily[daily["pos"] < k]
        rows.append({"kind": "daily", "k": f"{k} a day", "contacts": len(top),
                     "precision": top["low"].mean(),
                     "reachable_precision": top["reachable"].mean(),
                     "already_written_precision": (top["low"] & ~top["reachable"]).mean(),
                     "late_share": top["late"].mean()})
    out = pd.DataFrame(rows)
    out["base_rate"] = p["low"].mean()
    out["reachable_base_rate"] = p["reachable"].mean()
    return out


def hit_composition(p: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Where the hits of the daily top-k list come from, and the same list on delivered orders.

    A delivery-day call can reach only a delivered parcel whose review is not yet
    written; orders that never arrived, and late parcels reviewed while waiting, are
    the case for the promised-date contact instead.
    """
    q = p.copy()
    q["day"] = q["order_purchase_timestamp"].dt.date
    rows = []
    for name, frame in [("all reviewed orders", q), ("delivered orders only", q[q["delivered"]])]:
        ranked = frame.sort_values(["day", "p_low_review"], ascending=[True, False], kind="stable")
        top = ranked[ranked.groupby("day").cumcount() < k]
        hit = top[top["low"]]
        n = len(top)
        rows.append({"list": name, "k_per_day": k, "orders": len(frame), "base_rate": frame["low"].mean(),
                     "contacts": n, "hits": len(hit), "precision": len(hit) / n,
                     "never_delivered_per_100": 100 * (~hit["delivered"]).sum() / n,
                     "late_written_before_per_100": 100 * (hit["delivered"] & hit["late"] & hit["written_before_arrival"]).sum() / n,
                     "delivered_reachable_per_100": 100 * (hit["delivered"] & ~hit["written_before_arrival"]).sum() / n,
                     "ontime_written_before_per_100": 100 * (hit["delivered"] & ~hit["late"] & hit["written_before_arrival"]).sum() / n,
                     "contacts_never_delivered_pct": 100 * (~top["delivered"]).mean(),
                     "story_day_hits": int(hit[hit["day"] == STORY_DAY].shape[0])})
    return pd.DataFrame(rows)


def value_table(p: pd.DataFrame, c_action: float, c_miss: float) -> pd.DataFrame:
    """Net benefit by threshold, counting all hits and counting reachable hits only."""
    rows = []
    n = len(p)
    for t in np.arange(0.05, 0.95 + 1e-9, 0.01):
        flag = p["p_low_review"] >= t
        k = int(flag.sum())
        all_tp = int((flag & p["low"]).sum())
        reach_tp = int((flag & p["reachable"]).sum())
        rows.append({"threshold": round(float(t), 2), "flagged": k,
                     "net_all_per_1k": 1000 * (all_tp * c_miss - k * c_action) / n,
                     "net_reachable_per_1k": 1000 * (reach_tp * c_miss - k * c_action) / n})
    return pd.DataFrame(rows)


def story_order(d: pd.DataFrame, p: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """The order at the top of the 16 July 2018 replay list, with its timeline."""
    day = p[p["order_purchase_timestamp"].dt.date == STORY_DAY]
    day = day.sort_values("p_low_review", ascending=False, kind="stable").reset_index(drop=True)
    oid = day.loc[0, "order_id"]
    row = d.set_index("order_id").loc[oid]
    items = tables["order_items"][tables["order_items"]["order_id"] == oid]
    seller = tables["sellers"].set_index("seller_id").loc[items["seller_id"].iloc[0]]
    p1a = need("predictions_p1a_test.csv").set_index("order_id")
    p1b = need("predictions_p1b_test.csv").set_index("order_id")
    top10 = day.head(10)
    return pd.DataFrame([{
        "order_id": oid,
        "purchase": row["order_purchase_timestamp"],
        "promised": row["order_estimated_delivery_date"],
        "survey_sent": row["review_creation_date"],
        "review_answered": row["review_answer_timestamp"],
        "delivered": row["order_delivered_customer_date"],
        "review_score": int(row["review_score"]),
        "customer_city": row["customer_city"], "customer_state": row["customer_state"],
        "seller_city": seller["seller_city"], "seller_state": seller["seller_state"],
        "category": row["lead_category"], "distance_km": row["distance_km"],
        "promised_days": row["estimated_days"], "actual_days": row["delivery_days"],
        "days_late": row["delay_vs_estimate_days"],
        "review_days_before_arrival": (row["order_delivered_customer_date"]
                                       - row["review_answer_timestamp"]).total_seconds() / 86400,
        "pred_days_at_checkout": p1a.loc[oid, "pred_days"],
        "p_late_at_checkout": p1b.loc[oid, "p_late_derived"],
        "p_low_review_at_delivery": day.loc[0, "p_low_review"],
        "rank_on_day": 1, "orders_on_day": len(day),
        "top10_low_reviews": int(top10["low"].sum()),
        "top10_written_before_arrival": int((top10["low"] & ~top10["reachable"]).sum()),
        "low_reviews_on_day": int(day["low"].sum()),
    }])


if __name__ == "__main__":
    tables = load_all()
    d = timing_frame(tables)
    groups = group_table(d)
    groups.to_csv(RESULTS / "review_timing.csv", index=False)
    buckets = lateness_buckets(d)
    buckets.to_csv(RESULTS / "review_timing_lateness.csv", index=False)
    print(buckets.round(1).to_string(index=False))

    p = need("predictions_p2_test.csv", parse_dates=["order_purchase_timestamp"])
    flags = d.set_index("order_id")[["written_before_arrival", "late"]]
    p = p.join(flags, on="order_id")
    p["written_before_arrival"] = p["written_before_arrival"].fillna(False).astype(bool)
    p["late"] = p["late"].fillna(False).astype(bool)
    # Reviewed orders that never arrived (cancelled, or still "shipped" when the
    # extract was taken) carry no delivery fields; keep them visible throughout.
    orders = tables["orders"]
    delivered_ids = set(orders.loc[orders["order_delivered_customer_date"].notna(), "order_id"])
    p["delivered"] = p["order_id"].isin(delivered_ids)
    p["low"] = p["actual_low_review"] == 1
    p["reachable"] = p["low"] & ~p["written_before_arrival"]

    cap = capacity_split(p)
    cap.to_csv(RESULTS / "review_timing_capacity.csv", index=False)
    val = value_table(p, config.COSTS["outreach_cost"], config.COSTS["low_review_cost"])
    val.to_csv(RESULTS / "review_timing_value.csv", index=False)
    story = story_order(d, p, tables)
    story.to_csv(RESULTS / "story_order.csv", index=False)
    (p[["order_id", "written_before_arrival"]]
     .assign(written_before_arrival=lambda x: x["written_before_arrival"].astype(int))
     .to_csv(RESULTS / "review_timing_test_orders.csv", index=False))

    # Is the review score only restating lateness? Rank the on-time parcels alone.
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = []
    for name, g in [("all", p), ("delivered", p[p["delivered"]]),
                    ("on time", p[p["delivered"] & ~p["late"]]), ("late", p[p["late"]]),
                    ("never delivered", p[~p["delivered"]])]:
        top = g.sort_values("p_low_review", ascending=False, kind="stable").head(max(1, int(round(0.05 * len(g)))))
        rows.append({"subset": name, "n": len(g), "base_rate": g["low"].mean(),
                     "roc_auc": roc_auc_score(g["low"], g["p_low_review"]),
                     "pr_auc": average_precision_score(g["low"], g["p_low_review"]),
                     "precision_top5pct": top["low"].mean(),
                     "lift_top5pct": top["low"].mean() / g["low"].mean()})
    subsets = pd.DataFrame(rows)
    subsets.to_csv(RESULTS / "review_timing_subsets.csv", index=False)
    print(subsets.round(3).to_string(index=False))

    composition = hit_composition(p)
    composition.to_csv(RESULTS / "review_timing_composition.csv", index=False)
    print(composition.round(3).to_string(index=False))

    # Daily ten-a-day list, per 1,000 delivered orders, all hits against reachable hits.
    c = config.COSTS
    ten = cap[(cap["kind"] == "daily") & (cap["k"] == "10 a day")].iloc[0]
    per_1k = 1000 * ten["contacts"] / len(p)
    print(f"ten a day: R${per_1k * (ten['precision'] * c['low_review_cost'] - c['outreach_cost']):.0f} per 1,000 (all hits), "
          f"R${per_1k * (ten['reachable_precision'] * c['low_review_cost'] - c['outreach_cost']):.0f} per 1,000 (reachable)")

    pd.set_option("display.width", 220)
    print(groups.round(2).to_string(index=False))
    print()
    print(cap.round(3).to_string(index=False))
    best_all = val.loc[val["net_all_per_1k"].idxmax()]
    best_reach = val.loc[val["net_reachable_per_1k"].idxmax()]
    print(f"\nbest net per 1,000 (all hits):       R${best_all['net_all_per_1k']:.0f} at threshold {best_all['threshold']}")
    print(f"best net per 1,000 (reachable hits): R${best_reach['net_reachable_per_1k']:.0f} at threshold {best_reach['threshold']}")
    print()
    print(story.T.to_string())
    print(f"\nwrote review_timing.csv, review_timing_capacity.csv, review_timing_value.csv, story_order.csv into {RESULTS}")
