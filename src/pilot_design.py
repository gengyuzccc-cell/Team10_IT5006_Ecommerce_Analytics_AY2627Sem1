"""Design of the outreach pilot that would measure what Section 5.2 assumes.

    python src/capacity_metrics.py      # first: writes capacity_p2_daily.csv
    python src/pilot_design.py          # then: writes report/results/pilot_*.csv

The cost analysis values a contact at (avoided 1-2 star review) x R$40 minus
R$5 of agent time, which assumes that contacting a customer who would have left
a low review prevents that review. Nothing in the data can test that: Olist
never contacted anyone, so the counterfactual is unobserved. The Phase 2
feedback asked for this to be stated as a limitation and for a pilot that would
measure it. This script sizes that pilot from the numbers the model already
reports, so the recommendation is concrete rather than a gesture.

Design
------
Each day the model ranks the day's deliveries and the customer-experience team
takes the top K. Within that top K, orders are randomised 1:1 - half are
contacted, half are left alone - and the outcome is the 1-2 star review rate in
each arm. Randomising *inside* the list, rather than contacting everyone on it,
is what makes the comparison fair: both arms have the same predicted risk.

The control arm's rate is known already: it is the model's precision at that
daily K on the held-out test window (capacity_p2_daily.csv), 0.572 at ten
contacts a day. The treated arm's rate is what the pilot measures. Sample size
for a two-sided two-proportion z-test at alpha = 0.05 follows the standard
formula; durations assume K contacts a day split evenly across the arms.

The break-even is one line of arithmetic: a contact pays when the reduction in
the low-review probability times the review's cost exceeds the contact's cost,
so the required absolute reduction is c_action / c_miss = 0.125 and the required
relative reduction is 0.125 / precision - about one in five of the low reviews
a contact reaches at K = 10. That is the hypothesis the pilot tests.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm

import config

RESULTS = config.PROJECT_ROOT / "report" / "results"

ALPHA = 0.05
POWERS = (0.80, 0.90)
RELATIVE_REDUCTIONS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
DAILY_K = (5, 10, 20, 40)


def n_per_arm(p0: float, p1: float, alpha: float = ALPHA, power: float = 0.80) -> float:
    """Two-sided two-proportion z-test sample size per arm (normal approximation)."""
    if p0 == p1:
        return np.inf
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    p_bar = (p0 + p1) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar))
           + z_b * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return float(np.ceil(num / (p0 - p1) ** 2))


def design_table(daily: pd.DataFrame, cost_action: float, cost_miss: float) -> pd.DataFrame:
    rows = []
    for _, d in daily.iterrows():
        k, p0 = int(d["k_per_period"]), float(d["precision"])
        for r in RELATIVE_REDUCTIONS:
            p1 = p0 * (1 - r)
            row = {"k_per_day": k, "control_rate": p0, "relative_reduction": r,
                   "treated_rate": p1, "absolute_reduction": p0 - p1,
                   "net_benefit_per_contact": (p0 - p1) * cost_miss - cost_action}
            for power in POWERS:
                n = n_per_arm(p0, p1, power=power)
                tag = f"{int(100 * power)}"
                row[f"n_per_arm_{tag}"] = n
                row[f"days_{tag}"] = float(np.ceil(2 * n / k))
                row[f"weeks_{tag}"] = float(np.ceil(2 * n / k / 7))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["above_break_even"] = out["net_benefit_per_contact"] > 0
    return out


def breakeven_table(daily: pd.DataFrame, cost_action: float, cost_miss: float) -> pd.DataFrame:
    rows = []
    for _, d in daily.iterrows():
        p0 = float(d["precision"])
        abs_req = cost_action / cost_miss
        rows.append({"k_per_day": int(d["k_per_period"]),
                     "flagged_pct": float(d["flagged_pct"]),
                     "control_rate": p0,
                     "breakeven_absolute_reduction": abs_req,
                     "breakeven_relative_reduction": abs_req / p0,
                     "n_per_arm_80_at_breakeven": n_per_arm(p0, p0 - abs_req, power=0.80),
                     "weeks_80_at_breakeven": float(np.ceil(
                         2 * n_per_arm(p0, p0 - abs_req, power=0.80) / d["k_per_period"] / 7))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    daily_path = RESULTS / "capacity_p2_daily.csv"
    if not daily_path.exists():
        raise SystemExit(f"{daily_path} is missing. Run `python src/capacity_metrics.py` first.")
    daily = pd.read_csv(daily_path)
    daily = daily[daily["k_per_period"].isin(DAILY_K)]
    c = config.COSTS

    design = design_table(daily, c["outreach_cost"], c["low_review_cost"])
    design.to_csv(RESULTS / "pilot_design.csv", index=False)
    be = breakeven_table(daily, c["outreach_cost"], c["low_review_cost"])
    be.to_csv(RESULTS / "pilot_breakeven.csv", index=False)

    pd.set_option("display.width", 220)
    print("break-even by daily capacity")
    print(be.round(3).to_string(index=False))
    print("\npilot size by assumed effect (power 0.80 / 0.90, alpha 0.05)")
    print(design[["k_per_day", "control_rate", "relative_reduction", "treated_rate",
                  "n_per_arm_80", "weeks_80", "n_per_arm_90", "weeks_90",
                  "net_benefit_per_contact", "above_break_even"]]
          .round(3).to_string(index=False))
    print(f"\nwrote {RESULTS / 'pilot_design.csv'} and {RESULTS / 'pilot_breakeven.csv'}")
