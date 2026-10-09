"""Precision@K and Recall@K: evaluation under a fixed outreach capacity.

    python src/train_models.py          # first: writes the per-order test predictions
    python src/capacity_metrics.py      # then: writes report/results/capacity_*.csv

The Milestone 1 feedback made a point the Phase 1 metrics section did not: the
customer-experience team cannot contact every flagged order. It works a ranked
list up to its capacity, so the practical value of the model is the precision
and recall of the top-K, not PR-AUC over the whole ranking. The same holds for
operations and the late-order flag.

Nothing is refitted here. Every number comes from the per-order test
predictions that train_models.py wrote, so the tables below are exactly the
deployed models read a second way.

    capacity_p2.csv           review-risk model, top-K share of the test window
    capacity_p2_daily.csv     review-risk model, top-K per purchase day
    capacity_p1b.csv          late flag, derived route against the best direct
                              classifier, top-K share of the test window
    capacity_operating_points.csv
                              where the threshold-based operating points of the
                              report sit on the same axis (flagged share)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import evaluate as E

RESULTS = config.PROJECT_ROOT / "report" / "results"


def need(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `python src/train_models.py` first.")
    return pd.read_csv(path, parse_dates=["order_purchase_timestamp"])


def with_costs(table: pd.DataFrame, n: int, cost_action: float,
               cost_miss: float) -> pd.DataFrame:
    """Expected net benefit of acting on the top K, same arithmetic as cost_curve."""
    out = table.copy()
    out["break_even_precision"] = cost_action / cost_miss
    out["net_benefit"] = out["tp"] * cost_miss - out["k"] * cost_action
    out["net_benefit_per_1k"] = 1000 * out["net_benefit"] / n
    out["above_break_even"] = out["precision_at_k"] > out["break_even_precision"]
    return out


def operating_point(label: str, y, score, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    flag = np.asarray(score) >= threshold
    tp = int((flag & (y == 1)).sum())
    return {"problem": label, "threshold": float(threshold),
            "flagged_pct": float(100 * flag.mean()),
            "precision": tp / flag.sum() if flag.sum() else np.nan,
            "recall": tp / y.sum() if y.sum() else np.nan}


if __name__ == "__main__":
    meta = json.loads((config.MODELS_DIR / "model_metadata.json").read_text("utf-8"))
    c = config.COSTS

    # -- Problem 2: low-review risk at delivery -------------------------------
    p2 = need("predictions_p2_test.csv")
    y2, s2 = p2["actual_low_review"].to_numpy(), p2["p_low_review"].to_numpy()
    cap2 = with_costs(E.precision_recall_at_k(y2, s2), len(p2),
                      c["outreach_cost"], c["low_review_cost"])
    cap2.insert(0, "problem", "P2 low review")
    cap2.to_csv(RESULTS / "capacity_p2.csv", index=False)

    daily2 = E.capacity_curve(y2, s2, p2["order_purchase_timestamp"])
    daily2.insert(0, "problem", "P2 low review")
    daily2.to_csv(RESULTS / "capacity_p2_daily.csv", index=False)

    # -- Problem 1B: the late flag, both routes --------------------------------
    p1 = need("predictions_p1b_test.csv")
    y1 = p1["actual_late"].to_numpy()
    parts = []
    for route, col in [("derived", "p_late_derived"), ("direct", "p_late_direct")]:
        t = with_costs(E.precision_recall_at_k(y1, p1[col].to_numpy()), len(p1),
                       c["expedite_cost"], c["late_delivery_cost"])
        t.insert(0, "route", route)
        t.insert(0, "model", "Derived: regression margin vs promise" if route == "derived"
                 else str(p1["direct_model"].iloc[0]))
        parts.append(t)
    cap1 = pd.concat(parts, ignore_index=True)
    cap1.to_csv(RESULTS / "capacity_p1b.csv", index=False)

    # -- the report's threshold operating points on the same axis -------------
    ops = pd.DataFrame([
        operating_point("P2 tuned threshold (F1 on validation)", y2, s2,
                        meta["p2"]["threshold"]),
        operating_point("P2 cost-optimal threshold", y2, s2, 0.11),
        operating_point("P1B derived, tuned threshold", y1, p1["p_late_derived"],
                        meta["p1b"]["threshold"]),
    ])
    ops.to_csv(RESULTS / "capacity_operating_points.csv", index=False)

    pd.set_option("display.width", 200)
    print("P2 low review - top-K share of the test window")
    print(cap2[["k_pct", "k", "precision_at_k", "recall_at_k", "lift_at_k",
                "net_benefit_per_1k", "above_break_even"]].round(3).to_string(index=False))
    print("\nP2 low review - top-K per purchase day")
    print(daily2.round(3).to_string(index=False))
    print("\nP1B late flag - derived against direct")
    print(cap1[["route", "k_pct", "k", "precision_at_k", "recall_at_k",
                "above_break_even"]].round(3).to_string(index=False))
    print("\nthreshold operating points")
    print(ops.round(3).to_string(index=False))
    print(f"\nwrote 4 tables into {RESULTS}")
