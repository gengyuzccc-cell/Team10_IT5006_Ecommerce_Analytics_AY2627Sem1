"""Where every number on the presentation slides comes from.

    python src/customer_view.py       # first: writes report/results/customer_*.csv
    python src/slide_provenance.py    # then: slide_provenance.csv and the report's table

The slides were locked on 4 November 2026 and are submitted unchanged. At the
presentation the question was where the numbers on the business-value slide come
from. This script answers it for every number on every slide: each row quotes the
slide's own words, names the kind of number, recomputes it from the repository's
outputs and records whether the slide's figure matches its source to the slide's own
rounding. Where the wording on the slide is wrong or loose, the last field says how
the report reads it.

Kinds
-----
    observed        counted in the Olist tables
    assumed         no data source: an input chosen by the team (config.COSTS)
    derived         arithmetic on observed and assumed numbers; the formula is given
    planned         a pilot design that has not been run
    literature      a claim taken from a cited paper
    no data source  a claim the tables cannot support or refute
    wording         a phrase whose reading needs correcting; no number to check

Inputs: report/results/*.csv (including customer_story.csv and the other files of
src/customer_view.py), report/stats.txt for the Phase 1 leakage demonstration, the
cost assumptions in src/config.py, and the raw tables (through src/data_load.py) for
three numbers no table holds yet: the 2017-18 promise buffer, the carrier's share of
the lead time, and the test-window buffer.

Outputs
-------
    report/results/slide_provenance.csv        one row per number or claim, with the
                                               recomputed value and whether it matches;
                                               `keys` carries the derived values that
                                               src/phase3_stats.py reports (prov_*)
    report/sections/p3_C_provenance_table.tex  the same rows as the report's longtable

The script stops (SystemExit) if any recomputed number no longer matches its slide,
or if a row that names config.COSTS is not marked as an assumption.
"""
from __future__ import annotations

import math
import re
import sys
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import pilot_design as PD
from data_load import build_orders, load_all

RESULTS = config.PROJECT_ROOT / "report" / "results"
TEX_OUT = config.PROJECT_ROOT / "report" / "sections" / "p3_C_provenance_table.tex"
CSV_OUT = RESULTS / "slide_provenance.csv"
KINDS = ("observed", "assumed", "derived", "planned", "literature", "no data source", "wording")
SLIDE_ORDER = [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
SLIDE_TOPIC = {
    1: "title", 2: "one customer's order", 3: "broken promises and ratings",
    4: "when the review is written", 5: "why promises break", 6: "three moments, three tools",
    7: "results at checkout", 8: "results at delivery", 9: "the app and 16 July 2018",
    10: "the recommendations", 11: "business value and the pilot", 12: "lessons", 13: "closing and demo"}

# Windows a row can be counted on. The report's appendix names the test window as the
# default, so the LaTeX table names the window only when it is another one.
W_TEST = "test window: orders placed 30 June to 30 August 2018"
W_TEST_REV = "test window: 12,855 reviewed orders placed 30 June to 30 August 2018"
W_TEST_DEL = "test window: 12,630 delivered orders placed 30 June to 30 August 2018"
W_ANALYSIS_DR = "analysis window: delivered and reviewed orders placed January 2017 to August 2018"
W_ANALYSIS_D = "analysis window: delivered orders placed January 2017 to August 2018"
W_ANALYSIS_R = "analysis window: reviewed orders placed January 2017 to August 2018"
W_MONTHLY = "monthly purchase cohorts, January 2017 to August 2018"
W_STRIKE = "weekly and monthly purchase cohorts, May and June 2018"
W_STORY = "the presentation's order, e1320973"
W_DAY = "orders placed on 16 July 2018"
W_EXTRACT = "the whole extract, September 2016 to October 2018"
W_NONE = ""
DEFAULT_WINDOWS = (W_TEST, W_TEST_REV, W_TEST_DEL)
# The LaTeX table names a window in brackets only when it is not the test window.
SHORT_WINDOW = {W_ANALYSIS_DR: "2017–18, delivered and reviewed", W_ANALYSIS_D: "2017–18, delivered",
                W_ANALYSIS_R: "2017–18, reviewed", W_MONTHLY: "monthly, 2017–18", W_STRIKE: "May–June 2018",
                W_STORY: "order e1320973", W_DAY: "16 July 2018 purchases", W_EXTRACT: "whole extract, 2016–18"}


@dataclass
class Row:
    slide: int
    on_slide: str                 # the slide's own words; fragments of one slide joined by " … "
    kind: str
    source: str
    compute: Callable | None = None   # recomputes the number(s) the slide shows, in its units
    shown: tuple = ()                 # the number(s) as printed on the slide (text compared as text)
    tol: tuple | None = None          # per number; default: half a unit of the slide's last digit
    window: str = W_NONE
    correction: str = ""
    keys: dict = field(default_factory=dict)   # derived values for src/phase3_stats.py (prov_*)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def need(name: str, **kw) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run the scripts listed in README.md first "
                         "(src/customer_view.py last).")
    return pd.read_csv(path, **kw)


def phase1_stat(key: str) -> float:
    path = config.PROJECT_ROOT / "report" / "stats.txt"
    for line in path.read_text("utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == key:
            return float(parts[1])
    raise SystemExit(f"{key} is missing from {path}. Run `python src/report_stats.py` first.")


def raw_numbers() -> dict:
    """The three numbers computed from the raw tables (via data_load): the promise
    buffer and the carrier's share of the lead time on the analysis window's delivered
    orders, and the buffer on the test window's delivered orders."""
    orders = build_orders(load_all())
    win = orders[orders["order_purchase_timestamp"].between(config.ANALYSIS_START, config.ANALYSIS_END)]
    d = win[win["order_delivered_customer_date"].notna()]
    carrier = (d["order_delivered_customer_date"] - d["order_delivered_carrier_date"]).dt.total_seconds() / 86400
    test = d[d["order_purchase_timestamp"] > config.VAL_END]
    return {"n_delivered": len(d), "promise_all": d["estimated_days"].mean(), "lead_all": d["delivery_days"].mean(),
            "carrier_days": carrier.mean(), "carrier_share_pct": 100 * carrier.mean() / d["delivery_days"].mean(),
            "n_test_delivered": len(test), "promise_test": test["estimated_days"].mean(),
            "lead_test": test["delivery_days"].mean(), "n_orders": len(orders),
            "n_window": len(win), "first_year": int(orders["order_purchase_timestamp"].dt.year.min()),
            "last_year": int(orders["order_purchase_timestamp"].dt.year.max())}


def load_inputs() -> SimpleNamespace:
    X = SimpleNamespace()
    X.c = config.COSTS
    X.raw = raw_numbers()
    X.story = need("customer_story.csv", parse_dates=["purchase", "promised", "survey_sent", "review_answered",
                                                     "delivered", "requoted_date", "message_at"]).iloc[0]
    X.j = need("customer_journeys_test.csv", parse_dates=["purchase", "delivered", "answered"])
    X.tim = need("review_timing.csv").set_index("group")
    X.buckets = need("review_timing_lateness.csv").set_index("bucket")["low_pct"]
    X.monthly = need("lateness_monthly.csv").set_index("purchase_month")
    X.weekly = need("lateness_weekly.csv").set_index("week_start")
    X.peak = need("lateness_peak_by_region.csv")
    X.reg = need("reg_p1a.csv").set_index("model")
    X.clf1b = need("clf_p1b.csv").set_index("model")
    cap1b = need("capacity_p1b.csv")
    X.cap1b = cap1b[cap1b["route"] == "derived"].set_index("k_pct")
    X.ops = need("capacity_operating_points.csv")
    X.cap2 = need("capacity_p2.csv").set_index("k_pct")
    X.cap2d = need("capacity_p2_daily.csv").set_index("k_per_period")
    X.tcap = need("review_timing_capacity.csv").set_index("k")
    X.tmix = need("review_timing_composition.csv").set_index("list")
    X.tsub = need("review_timing_subsets.csv").set_index("subset")
    X.tval = need("review_timing_value.csv")
    X.cost2 = need("cost_p2.csv")
    X.perm2 = need("perm_p2.csv").reset_index(drop=True)
    X.clf2 = need("clf_p2.csv").set_index("model")
    X.tvt = need("train_val_test.csv")
    X.pdz = need("pilot_design.csv")
    X.pbe = need("pilot_breakeven.csv").set_index("k_per_day")
    X.p1a = need("predictions_p1a_test.csv", parse_dates=["order_purchase_timestamp"])
    X.p2 = need("predictions_p2_test.csv", parse_dates=["order_purchase_timestamp"])
    X.seg = need("segments_p1a_region.csv").set_index("customer_region")
    X.out = need("customer_outcomes_test.csv").set_index("key")
    X.lists = need("customer_contact_lists.csv").set_index("list")
    X.val = need("customer_value.csv").set_index(["list", "scope"])
    X.dec = need("customer_value_decomposition.csv").set_index("list")
    X.arms = need("customer_pilot_arms.csv").set_index("arm")
    X.rules = need("customer_promise_rules.csv")
    X.survey = need("customer_survey_timing.csv").set_index("window")
    X.lateday = need("customer_lateness_days.csv").set_index("slug")
    X.topics = need("customer_complaint_topics.csv").set_index("group")
    X.repeat = need("customer_repeat.csv")
    X.leak_logreg = phase1_stat("c1b_logreg_roc")
    X.leak_leaked = phase1_stat("c1b_leaked_roc")
    return X


def derived(X: SimpleNamespace) -> SimpleNamespace:
    """Every intermediate the rows below quote, computed once."""
    D = SimpleNamespace()
    tim, raw = X.tim, X.raw
    D.n_rev, D.n_del = len(X.p2), len(X.p1a)
    D.days = int(X.p2["order_purchase_timestamp"].dt.date.nunique())
    D.ontime_low, D.late_low = tim.loc["on time", "pct_low_review"], tim.loc["late", "pct_low_review"]
    D.n_all, D.n_ontime, D.n_late = int(tim.loc["all", "n"]), int(tim.loc["on time", "n"]), int(tim.loc["late", "n"])
    D.late_share = 100 * D.n_late / D.n_all
    D.ontime_share = 100 * D.n_ontime / D.n_all
    D.buffer_all = raw["promise_all"] - raw["lead_all"]
    D.buffer_test = raw["promise_test"] - raw["lead_test"]
    olist = X.reg.loc["Baseline: Olist's own delivery promise", "mae"]
    model = X.reg.loc["VotingRegressor (A+B)", "mae"]
    D.mae_olist, D.mae_model = olist, model
    D.mae_gain = 100 * (1 - model / olist)
    D.p1b = X.ops[X.ops["problem"].str.startswith("P1B")].iloc[0]
    D.expedite_damage = X.c["late_delivery_cost"] * (D.late_low - D.ontime_low) / 100
    D.expedite_be = X.c["expedite_cost"] / D.expedite_damage
    D.ten = X.tmix.loc["all reviewed orders"]
    D.ten_dlv = X.tmix.loc["delivered orders only"]
    D.tcap10 = X.tcap.loc["10 a day"]
    D.ten_prec = 100 * X.cap2d.loc[10, "precision"]
    D.base = 100 * float(X.tcap["base_rate"].iloc[0])
    per_1k = 1000 * D.tcap10["contacts"] / D.n_rev
    c_a, c_m = X.c["outreach_cost"], X.c["low_review_cost"]
    D.v_ten_all = per_1k * (D.tcap10["precision"] * c_m - c_a)
    D.v_ten_reach = per_1k * (D.tcap10["reachable_precision"] * c_m - c_a)
    tv = X.tval.set_index(X.tval["threshold"].round(2))
    D.v011_all, D.v011_reach = tv.loc[0.11, "net_all_per_1k"], tv.loc[0.11, "net_reachable_per_1k"]
    D.v019_all, D.v019_reach = tv.loc[0.19, "net_all_per_1k"], tv.loc[0.19, "net_reachable_per_1k"]
    D.cost_row = X.cost2.loc[X.cost2["net_benefit"].idxmax()]
    D.n_thresholds = len(X.cost2)
    D.dec011, D.dec_ten = X.dec.loc["slides_threshold_011"], X.dec.loc["slides_ten_a_day"]
    D.k10 = X.pdz[(X.pdz["k_per_day"] == 10) & ((X.pdz["relative_reduction"] - 0.25).abs() < 1e-9)].iloc[0]
    D.weeks_weekdays = math.ceil(2 * D.k10["n_per_arm_80"] / 10 / 5)
    D.be10 = X.pbe.loc[10]
    D.call, D.msg = X.lists.loc["delivery_day_k10"], X.lists.loc["promised_date_message"]
    D.arm_b2, D.arm_b3, D.arm_a = X.arms.loc["arm_b_two"], X.arms.loc["arm_b_three"], X.arms.loc["arm_a"]
    rules = X.rules[X.rules["period"] == "test"].set_index("rule")
    D.r_olist = rules.loc["Olist's own promise"]
    D.r_raw = rules.loc["Forecast alone (no buffer)"]
    D.r_r1 = rules.loc["R1 buffer: validation late rate <= Olist's"]
    s = X.story
    D.day_rev = X.j[X.j["purchase"].dt.date == s["purchase"].date()]
    top = D.day_rev[D.day_rev["low_rank_purchase_day"] <= 10]
    D.day_hits = top[top["low"]]
    D.day_base = 100 * D.day_rev["low"].mean()
    d16 = X.p1a[X.p1a["order_purchase_timestamp"].dt.date == s["purchase"].date()]
    D.day_mae_model = (d16["pred_days"] - d16["actual_days"]).abs().mean()
    D.day_mae_olist = (d16["estimated_days"] - d16["actual_days"]).abs().mean()
    D.n_day_delivered = len(d16)
    pooled = X.j.sort_values("p_low_review", ascending=False, kind="stable").head(int(round(0.05 * len(X.j))))
    D.top5_n = len(pooled)
    D.top5_never_share = 100 * (pooled["outcome"] == "never delivered").mean()
    D.top5_written = 100 * (pooled["low"] & pooled["written_before_arrival"]).mean()
    D.top5_reachable_delivered = 100 * (pooled["low"] & pooled["delivery_reach"]).mean()
    rp = X.repeat[(X.repeat["version"].str.startswith("V2")) & (X.repeat["horizon_days"] == 180)]
    D.rep = rp[(rp["group"] == "1-2 star") & (rp["reference"] == "4-5 star")].iloc[0]
    p2rows = X.tvt[X.tvt["problem"] == "P2 low review"]
    D.n_reviewed_window = int(p2rows["n"].sum())
    D.n_low_window = int(X.topics.loc["all", "low_reviews"])
    D.tree_share = 100 * X.clf2.loc["DecisionTreeClassifier", "pr_auc"] / X.clf2.loc["StackingClassifier (A+B)", "pr_auc"]
    D.stack_gap = X.clf2.loc["HistGradientBoostingClassifier", "pr_auc"] - X.clf2.loc["StackingClassifier (A+B)", "pr_auc"]
    perm = X.perm2.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    D.perm_rank = {f: int(i) + 1 for i, f in enumerate(perm["feature"])}
    D.perm_n = len(perm)
    D.perm_imp = perm.set_index("feature")["importance"]
    D.cap1b_band = X.cap1b.loc[[1.0, 2.0, 3.0, 5.0, 7.5], "precision_at_k"] * 100
    return D


# ---------------------------------------------------------------------------
# The rows: slide 11 first, then slides 1-13
# ---------------------------------------------------------------------------
def build_rows(X: SimpleNamespace, D: SimpleNamespace) -> list[Row]:
    s, o, c = X.story, X.out, X.c
    money = lambda v: f"R${v:,.0f}" if v >= 0 else f"−R${-v:,.0f}"
    v = lambda lst, scope, col: X.val.loc[(lst, scope), col]
    m = X.monthly
    integrity = "every customer is still asked; no review is filtered, delayed in publication or rewarded"
    rows: list[Row] = []
    add = rows.append

    # ---------------- slide 11
    add(Row(11, "Worth R$775–R$1,466 / 1k orders", "derived",
            "review_timing_value.csv, threshold 0.11: net_reachable_per_1k, net_all_per_1k",
            lambda: (D.v011_reach, D.v011_all), ("775", "1,466"), window=W_TEST_REV,
            correction=f"per 1,000 reviewed test orders, not orders; ceilings that count every contact as turning "
                       f"its customer around (at one in four: "
                       f"{money(v('slides_threshold_011', 'not written before arrival', 'net_e25'))} and "
                       f"{money(v('slides_threshold_011', 'all hits', 'net_e25'))}, customer_value.csv)"))
    add(Row(11, "R$5 CALL", "assumed", "config.COSTS['outreach_cost']", lambda: c["outreach_cost"], ("5",),
            correction="no data source; varied R$2.5–10 in customer_value_sensitivity.csv"))
    add(Row(11, "R$40 BAD REVIEW", "assumed", "config.COSTS['low_review_cost']", lambda: c["low_review_cost"],
            ("40",),
            correction=f"stands for a 1–2★ rating's effect on later buyers; the customer's own return cannot price it (repeat "
                       f"within 180 days: {D.rep['repeat_pct']:.2f}% after 1–2★, {D.rep['ref_repeat_pct']:.2f}% after "
                       f"4–5★, customer_repeat.csv); varied R$20–80"))
    nw = int(D.dec011["hits_delivered_not_yet_written"] + D.dec011["hits_never_delivered"])
    add(Row(11, "R$775 conservative ceiling on the 35/100 share", "derived",
            f"1000 × ({nw} × 40 − {int(D.dec011['contacts']):,} × 5) / 12,855: the threshold-0.11 list's {nw} hits "
            f"not written before arrival ({int(D.dec011['hits_delivered_not_yet_written'])} delivered + "
            f"{int(D.dec011['hits_never_delivered'])} never delivered)",
            lambda: D.v011_reach, ("775",), window=W_TEST_REV,
            correction=f"belongs to the threshold-0.11 list ({D.dec011['contacts'] / D.days:.0f} contacts a day), "
                       f"not to the 35-in-100 share, which is worth {money(D.v_ten_reach)}"))
    add(Row(11, "R$433 at fixed 10 calls/day", "derived",
            f"1000 × ({int(D.dec_ten['hits_delivered_not_yet_written'] + D.dec_ten['hits_never_delivered'])} × 40 − 614 × 5) / 12,855",
            lambda: D.v_ten_reach, ("433",), window=W_TEST_REV,
            correction=f"of the 35 in 100, {D.ten['never_delivered_per_100']:.1f} never received the parcel; only "
                       f"{D.ten['delivered_reachable_per_100']:.1f} could be reached at delivery (that call alone: "
                       f"{money(D.dec_ten['net_delivery_day_only_per_1k'])})"))
    add(Row(11, "35/100 share", "observed",
            f"review_timing_capacity.csv, 10 a day: "
            f"{int(D.dec_ten['hits_delivered_not_yet_written'] + D.dec_ten['hits_never_delivered'])} of 614 = "
            f"{int(D.dec_ten['hits_never_delivered'])} never delivered + "
            f"{int(D.dec_ten['hits_delivered_not_yet_written'])} delivered, not yet written",
            lambda: 100 * D.tcap10["reachable_precision"], ("35",), window=W_TEST_REV,
            correction="includes the orders that never arrived"))
    add(Row(11, "R$1,466 full ceiling across all hits", "derived",
            f"cost_p2.csv, threshold 0.11: 1000 × ({int(D.cost_row['tp'])} × 40 − {int(D.cost_row['flagged']):,} × 5) "
            f"/ 12,855",
            lambda: D.v011_all, ("1,466",), window=W_TEST_REV,
            correction=f"0.11 is the best of {D.n_thresholds} thresholds read on the test window; the threshold fixed "
                       f"on validation, 0.19, gives {money(D.v019_all)} ({money(D.v019_reach)} not written); "
                       f"{D.dec011['late_never_gross_share_pct']:.1f}% of the gross value is never-delivered orders or late "
                       f"parcels rated before arrival",
            keys={"value_t019_all": D.v019_all, "value_t019_reachable": D.v019_reach}))
    add(Row(11, "50 / 50 for 6 weeks (191 orders per arm)", "planned",
            "pilot_design.csv (k 10, 25%): weeks = ceil(2 × 191 / 10 / 7); n_per_arm_80",
            lambda: (D.k10["weeks_80"], D.k10["n_per_arm_80"]), ("6", "191"),
            correction=f"ten a day on every calendar day ({D.weeks_weekdays} weeks on weekdays); re-sized per contact "
                       f"in customer_pilot_arms.csv: message {int(D.arm_b2['n_per_arm_80'])} per arm, "
                       f"{int(D.arm_b2['weeks_80'])} weeks; delivery-day call {int(D.arm_a['n_per_arm_80'])}, "
                       f"{int(D.arm_a['weeks_80'])} weeks"))
    add(Row(11, "5 RANDOMLY CALLED / DAY … 5 HELD AS CONTROL / DAY", "planned",
            "1:1 randomisation inside the daily top ten", lambda: (10 / 2, 10 / 2), ("5", "5")))
    add(Row(11, "TODAY (NO CALL) / 57.2%", "observed", "capacity_p2_daily.csv (k 10): 351 of 614",
            lambda: D.ten_prec, ("57.2",), window=W_TEST_REV,
            correction=f"a historical rate, not a control group (no pilot ran): "
                       f"{D.ten['never_delivered_per_100']:.1f} points never received the parcel and "
                       f"{D.ten['late_written_before_per_100']:.1f} had rated a late parcel while waiting"))
    add(Row(11, "BREAK-EVEN (~1 IN 5) / ≤ 44.7%", "derived",
            "pilot_breakeven.csv: 0.572 / 0.125; 0.572 − 5/40",
            lambda: (1 / D.be10["breakeven_relative_reduction"],
                     100 * (D.be10["control_rate"] - D.be10["breakeven_absolute_reduction"])),
            ("5", "44.7"), tol=(0.5, None),
            correction=f"{100 * D.be10['breakeven_relative_reduction']:.1f}% (1 in "
                       f"{1 / D.be10['breakeven_relative_reduction']:.1f}); unrelated to the {D.p1b['flagged_pct']:.1f}% "
                       f"of customers the 11% late warning reaches"))
    add(Row(11, "6-WK TARGET (1 IN 4) / 42.9% / 80% power", "planned",
            "pilot_design.csv: 0.572 × (1 − 0.25); power 0.80, alpha 0.05",
            lambda: (100 * D.k10["treated_rate"], 100 * PD.POWERS[0]), ("42.9", "80"),
            correction="one in four is a design choice, not an estimate"))
    add(Row(11, "+R$0.72/call", "derived", "(0.572 − 0.429) × 40 − 5",
            lambda: D.k10["net_benefit_per_contact"], ("0.72",),
            correction=f"only if the one-in-four effect and both costs hold; the ten-a-day list then nets "
                       f"{money(v('slides_ten_a_day', 'all hits', 'net_e25'))} per 1,000 reviewed orders"))
    # Quoted only up to "log": the rest of the sentence is a claim this report does not repeat.
    add(Row(11, "Nobody in the 2016–2018 log", "no data source", "none",
            correction="the slide's statement about calls has no source: the tables record no contact other than the "
                       "survey, so only a randomised pilot can measure what a contact changes"))
    add(Row(11, "PLUS ARM B (PROMISED-DATE OVERDUE ORDERS)", "planned", "customer_pilot_arms.csv",
            correction=f"{D.msg['contacts_per_day']:.1f} customers a day not yet rated when the promised day ends, "
                       f"{100 * D.msg['precision']:.1f}% then rate 1–2★: {int(D.arm_b2['n_per_arm_80'])} per arm, "
                       f"{int(D.arm_b2['weeks_80'])} weeks (two arms) or {int(D.arm_b3['weeks_80'])} (three); "
                       f"outcome: the final rating"))
    add(Row(11, "LIVE VALIDATION … Historical data proves", "wording", "none",
            correction="nothing was validated live: the held-out days show; the pilot would measure"))

    # ---------------- slide 1
    add(Row(1, "100,000 Brazilian e-commerce orders … 99,441 orders (2016–2018)", "observed",
            "orders table: rows, purchase years",
            lambda: (X.raw["n_orders"], X.raw["n_orders"], X.raw["first_year"], X.raw["last_year"]),
            ("100,000", "99,441", "2016", "2018"), tol=(1000, None, None, None), window=W_EXTRACT,
            correction=f"the analysis uses the {X.raw['n_window']:,} orders placed January 2017 to August 2018"))
    add(Row(1, "3. Delivery / Call top-10 risk list", "planned", "customer_contact_lists.csv (delivery_day_k10)",
            lambda: D.call["k_per_day"], ("10",),
            correction=f"worked on the previous day's deliveries not yet rated: {100 * D.call['precision']:.1f} in 100 "
                       f"rate 1–2★ ({D.call['lift']:.1f}× the {100 * D.call['pool_base_rate']:.1f}% base)"))

    # ---------------- slide 2
    fmt_date = lambda t: f"{t.day} {t:%b %Y}"
    day_n = lambda t: (t.normalize() - s["purchase"].normalize()).days
    add(Row(2, "16 Jul 2018 / Day 0 … 7 Aug 2018 / Day 22 … 9 Aug 2018 / Day 24 … 11 Aug 2018 / Day 26 … "
               "17 Aug 2018 / Day 32", "observed",
            "customer_story.csv: purchase, promised, survey_sent, review_answered, delivered",
            lambda: tuple(x for t in (s["purchase"], s["promised"], s["survey_sent"], s["review_answered"],
                                      s["delivered"]) for x in (fmt_date(t), day_n(t))),
            ("16 Jul 2018", "0", "7 Aug 2018", "22", "9 Aug 2018", "24", "11 Aug 2018", "26", "17 Aug 2018", "32"),
            window=W_STORY))
    add(Row(2, "Order #e1320973  ·  Health & beauty item (R$65 + R$28 freight)", "observed",
            "customer_story.csv: order_id, category, price, freight",
            lambda: (s["order_id"][:8], s["category"].replace("_", " & "), s["price"], s["freight"]),
            ("e1320973", "health & beauty", "65", "28"), window=W_STORY,
            correction=f"R${s['price']:.2f} + R${s['freight']:.2f}"))
    add(Row(2, "2,267 km from São Luís (MA) to São Gonçalo (RJ)", "observed",
            "customer_story.csv: distance_km (postcode-prefix centroids), seller and customer city",
            lambda: (s["distance_km"], f"{s['seller_city']} ({s['seller_state']})",
                     f"{s['customer_city']} ({s['customer_state']})"),
            ("2,267", "São Luís (MA)", "São Gonçalo (RJ)"), window=W_STORY))
    add(Row(2, "(~21-day window) … (10.5 days late) … 5.8 days old … took 31.8 days", "observed",
            "customer_story.csv: promised_days, days_late, review_days_before_arrival, actual_days",
            lambda: (s["promised_days"], s["days_late"], s["review_days_before_arrival"], s["actual_days"]),
            ("21", "10.5", "5.8", "31.8"), tol=(0.5, None, None, None), window=W_STORY))
    add(Row(2, "Zero proactive warning sent to the buyer.", "no data source", "none",
            correction="the tables record no message and no contact other than the survey"))
    add(Row(2, "Triggered by the promised calendar date", "no data source",
            "customer_story.csv; customer_survey_timing.csv",
            correction=f"the trigger is not recorded; the survey went out "
                       f"{(s['survey_sent'] - s['promised']).days} days after the promised date, "
                       f"{(s['delivered'] - s['survey_sent']).total_seconds() / 86400:.1f} days before delivery (test "
                       f"window: {X.survey.loc['test window', 'survey_exactly_2_days_after_pct']:.1f}% of late 1–2★ "
                       f"parcels exactly 2 days after)"))
    add(Row(2, "Frustrated and still empty-handed", "wording", "customer_story.csv", window=W_STORY,
            correction=f"rests on the timestamps (answered {s['review_answered'].round('min'):%d %b %H:%M}, delivered "
                       f"{s['delivered'].round('min'):%d %b %H:%M}); the comment ('{s['review_title']}' / '{s['review_message']}') "
                       f"is ambiguous and may mean the packaging"))
    add(Row(2, "the damage on the product page is already done", "no data source", "none",
            correction="the tables do not record what later buyers saw"))

    # ---------------- slide 3
    add(Row(3, "ACROSS 95,561 DELIVERED ORDERS", "observed", "review_timing.csv (all): n",
            lambda: D.n_all, ("95,561",), window=W_ANALYSIS_DR, correction="delivered and reviewed orders"))
    add(Row(3, "DELIVERED ON TIME / 9.2% … 87,902 orders (92.0%)", "observed",
            "review_timing.csv (on time): pct_low_review, n, share",
            lambda: (D.ontime_low, D.n_ontime, D.ontime_share), ("9.2", "87,902", "92.0"), window=W_ANALYSIS_DR))
    add(Row(3, "DELIVERED LATE / 54.0% … 7,659 orders (8.0%)", "observed",
            "review_timing.csv (late): pct_low_review, n, share",
            lambda: (D.late_low, D.n_late, D.late_share), ("54.0", "7,659", "8.0"), window=W_ANALYSIS_DR))
    add(Row(3, "5.9×", "derived", "54.0 / 9.2", lambda: D.late_low / D.ontime_low, ("5.9",), window=W_ANALYSIS_DR))
    add(Row(3, "78.6% … 78.0% … 61.3% … 19.1% … 9.7% … 8.9%", "observed",
            "review_timing_lateness.csv: low_pct by days late or early",
            lambda: tuple(X.buckets[b] for b in ("15+ days late", "8-14 days late", "4-7 days late", "1-3 days late",
                                                 "0-10 days early", "10+ days early")),
            ("78.6", "78.0", "61.3", "19.1", "9.7", "8.9"), window=W_ANALYSIS_DR))
    add(Row(3, "1–3 days late doubles bad reviews (19.1%); 4+ days late jumps past 61%–79%", "derived",
            "19.1 / 9.2; review_timing_lateness.csv",
            lambda: (X.buckets["1-3 days late"] / D.ontime_low, X.buckets["1-3 days late"],
                     X.buckets["4-7 days late"], X.buckets["15+ days late"]),
            ("2", "19.1", "61", "79"), tol=(0.5, None, None, None), window=W_ANALYSIS_DR,
            correction=f"by day: {X.lateday.loc['late_0_1', 'low_pct']:.1f}% during the promised day, "
                       f"{X.lateday.loc['late_1_2', 'low_pct']:.1f}% one to two days late "
                       f"(customer_lateness_days.csv)"))
    add(Row(3, "Only 1 in 12 parcels (8.0%) arrives late … 34% (one-third) of every 1–2★ review", "observed",
            "review_timing.csv: late share; share_of_all_low_reviews_pct",
            lambda: (100 / D.late_share, D.late_share, X.tim.loc["low review, late", "share_of_all_low_reviews_pct"]),
            ("12", "8.0", "34"), tol=(0.5, None, None), window=W_ANALYSIS_DR,
            correction="of the 1–2★ reviews on delivered orders"))
    add(Row(3, "On Olist's own data, bad logistics reviews directly depress a seller's future sales", "literature",
            "refs.bib deshpande2023logistics",
            correction="the study is of another marketplace, not Olist; seller sales were not measured here"))

    # ---------------- slide 4
    add(Row(4, "ON-TIME PARCELS  (11 IN 12) … LATE PARCELS  (1 IN 12)", "observed",
            "review_timing.csv: 12 × each share",
            lambda: (12 * D.ontime_share / 100, 12 * D.late_share / 100), ("11", "1"), tol=(0.5, 0.5),
            window=W_ANALYSIS_DR))
    add(Row(4, "+2.1 days / after delivery scan", "observed",
            "review_timing.csv (low review, on time): median days from delivery to the review",
            lambda: X.tim.loc["low review, on time", "median_answer_days_after_delivery"], ("2.1",),
            window=W_ANALYSIS_DR, correction="on-time parcels rated 1–2★"))
    add(Row(4, "66% of all 1–2★ reviews / Wrong, damaged, or disappointing items … 99.5% review AFTER delivery",
            "observed", "review_timing.csv (low review, on time): share of 1–2★, written after arrival",
            lambda: (X.tim.loc["low review, on time", "share_of_all_low_reviews_pct"],
                     100 - X.tim.loc["low review, on time", "pct_written_before_arrival"]),
            ("66", "99.5"), window=W_ANALYSIS_DR,
            correction=f"not measured on 4 November; since: of on-time 1–2★ comments "
                       f"{X.topics.loc['on time', 'wrong_or_missing_item_pct']:.1f}% mention a wrong or missing item, "
                       f"{X.topics.loc['on time', 'defective_or_damaged_pct']:.1f}% damage or a defect, "
                       f"{X.topics.loc['on time', 'quality_or_not_as_described_pct']:.1f}% quality (keyword topics, "
                       f"customer_complaint_topics.csv)"))
    add(Row(4, "1★ written 4.5d early!", "observed",
            "review_timing.csv (low review, late): median days from the review to delivery",
            lambda: -X.tim.loc["low review, late", "median_answer_days_after_delivery"], ("4.5",),
            window=W_ANALYSIS_DR, correction="all late 1–2★, not only 1★"))
    add(Row(4, "ALREADY WRITTEN / 87.3%", "observed",
            "review_timing.csv (low review, late): pct_written_before_arrival",
            lambda: X.tim.loc["low review, late", "pct_written_before_arrival"], ("87.3",), window=W_ANALYSIS_DR,
            correction=f"{X.survey.loc['test window', 'written_before_arrival_pct']:.1f}% on the test window"))
    add(Row(4, "34% of all 1–2★ reviews / Survey fires at the promised date", "observed",
            "review_timing.csv (low review, late); customer_survey_timing.csv",
            lambda: X.tim.loc["low review, late", "share_of_all_low_reviews_pct"], ("34",), window=W_ANALYSIS_DR,
            correction=f"survey on or after the promised date for "
                       f"{X.survey.loc['analysis window', 'survey_on_or_after_promise_pct']:.1f}%, a median "
                       f"{X.survey.loc['analysis window', 'survey_days_after_promise_median']:.0f} days after; the "
                       f"rule is not recorded"))

    # ---------------- slide 5
    add(Row(5, "sped up 2× … 16.9 d → 7.7 d average delivery", "observed",
            "lateness_monthly.csv mean_lead, Feb and Aug 2018",
            lambda: (m.loc["2018-02", "mean_lead"] / m.loc["2018-08", "mean_lead"], m.loc["2018-02", "mean_lead"],
                     m.loc["2018-08", "mean_lead"]),
            ("2", "16.9", "7.7"), tol=(0.5, None, None), window=W_MONTHLY, correction="delivery time halved"))
    add(Row(5, "75% of total time lives with the carrier", "observed",
            "orders table: mean (delivered − carrier handover) / mean lead time",
            lambda: X.raw["carrier_share_pct"], ("75",), tol=(1.0,), window=W_ANALYSIS_D,
            correction=f"{X.raw['carrier_share_pct']:.1f}% ({X.raw['carrier_days']:.1f} of {X.raw['lead_all']:.1f} "
                       f"days), rounded up",
            keys={"carrier_share_pct": X.raw["carrier_share_pct"], "carrier_days": X.raw["carrier_days"],
                  "lead_all_days": X.raw["lead_all"]}))
    add(Row(5, "across 20 months (Jan 2017 – Aug 2018)", "observed", "lateness_monthly.csv rows",
            lambda: len(m), ("20",), window=W_MONTHLY))
    add(Row(5, "Carrier jam peak 21.4% … Black Friday 14.3% … 16.0% … 10.4%", "observed",
            "lateness_monthly.csv late_pct: Mar 2018, Nov 2017, Feb 2018, Aug 2018",
            lambda: tuple(m.loc[k, "late_pct"] for k in ("2018-03", "2017-11", "2018-02", "2018-08")),
            ("21.4", "14.3", "16.0", "10.4"), window=W_MONTHLY))
    add(Row(5, "~11 extra buffer days on average", "observed", "orders table: mean promised days − mean lead time",
            lambda: D.buffer_all, ("11",), tol=(0.5,), window=W_ANALYSIS_D,
            correction=f"{X.raw['promise_all']:.1f} − {X.raw['lead_all']:.1f}; test window {D.buffer_test:.1f} "
                       f"({X.raw['promise_test']:.2f} − {X.raw['lead_test']:.2f})",
            keys={"buffer_all_days": D.buffer_all, "promise_all_days": X.raw["promise_all"],
                  "buffer_test_days": D.buffer_test, "promise_test_days": X.raw["promise_test"],
                  "lead_test_days": X.raw["lead_test"]}))
    add(Row(5, "(9.8d error)", "observed", "reg_p1a.csv: Olist's own delivery promise, MAE", lambda: D.mae_olist,
            ("9.8",), window=W_TEST_DEL, correction="a test-window figure; the ~11 days are 2017–18"))
    add(Row(5, "missed 1 in 5 orders (21.4%) when regional carrier slowdowns struck", "observed",
            "lateness_monthly.csv (Mar 2018); lateness_peak_by_region.csv",
            lambda: (100 / m.loc["2018-03", "late_pct"], m.loc["2018-03", "late_pct"]), ("5", "21.4"),
            tol=(0.5, None), window=W_MONTHLY,
            correction=f"network-wide: every region {X.peak['peak_late_pct'].min():.1f}–"
                       f"{X.peak['peak_late_pct'].max():.1f}% late in Feb–Mar 2018 "
                       f"({X.peak['rest_late_pct'].min():.1f}–{X.peak['rest_late_pct'].max():.1f}% otherwise)"))
    add(Row(5, "inflates validation AUC from 0.59 to a fake 1.00", "observed",
            "report/stats.txt: c1b_logreg_roc, c1b_leaked_roc",
            lambda: (X.leak_logreg, X.leak_leaked), ("0.59", "1.00"), window=W_TEST,
            correction="a test-window score of the Phase 1 logistic regression, not a validation AUC"))
    add(Row(5, "unseen Jul–Aug 2018 orders", "wording", "config.VAL_END, config.ANALYSIS_END",
            correction=f"orders placed 30 June to 30 August 2018: {int(o.loc['placed', 'orders']):,} "
                       f"({D.n_rev:,} reviewed, {D.n_del:,} delivered)"))
    add(Row(5, "(19.5 → 36.8 days), hiding June lateness (1.4%)", "observed",
            "lateness_weekly.csv promise_days, weeks of 14 and 28 May 2018; lateness_monthly.csv, June 2018",
            lambda: (X.weekly.loc["2018-05-14", "promise_days"], X.weekly.loc["2018-05-28", "promise_days"],
                     m.loc["2018-06", "late_pct"]),
            ("19.5", "36.8", "1.4"), window=W_STRIKE, correction="weekly means; 'hiding' is a reading"))

    # ---------------- slide 6
    add(Row(6, "12.8% / (~1 in 8 delivered orders) … 14.6% across all reviewed orders including undelivered",
            "observed",
            f"review_timing.csv (all): pct_low_review; {D.n_low_window:,} / {D.n_reviewed_window:,} reviewed "
            f"(customer_complaint_topics.csv, train_val_test.csv)",
            lambda: (X.tim.loc["all", "pct_low_review"], 100 / X.tim.loc["all", "pct_low_review"],
                     100 * D.n_low_window / D.n_reviewed_window),
            ("12.8", "8", "14.6"), tol=(None, 0.5, None), window=W_ANALYSIS_R))
    add(Row(6, "a single decision tree captures 96% of the delivery-triage signal", "derived",
            "clf_p2.csv PR-AUC: tree / deployed stack", lambda: D.tree_share, ("96",), window=W_TEST_REV,
            correction=f"review risk, not delivery: {X.clf2.loc['DecisionTreeClassifier', 'pr_auc']:.3f} against "
                       f"{X.clf2.loc['StackingClassifier (A+B)', 'pr_auc']:.3f}"))
    add(Row(6, "Replaces padded 24-day guesses", "observed", "orders table: mean promised days",
            lambda: X.raw["promise_all"], ("24",), tol=(0.5,), window=W_ANALYSIS_D,
            correction=f"{X.raw['promise_all']:.1f} days (2017–18), {D.r_olist['mean_promised_days']:.1f} on test; "
                       f"the recommended date is the forecast + 8 days ({D.r_r1['mean_promised_days']:.1f} on "
                       f"average)"))
    add(Row(6, "pauses the review email before 87.3% of late 1–2★ are written", "observed", "as slide 4",
            lambda: X.tim.loc["low review, late", "pct_written_before_arrival"], ("87.3",), window=W_ANALYSIS_DR,
            correction=f"recommended: ask for the review once the order is complete; {integrity}"))
    add(Row(6, "Triggers automatically the instant a promised date passes", "wording",
            "customer_outcomes_test.csv; customer_lateness_days.csv",
            correction=f"the trigger waits for the end of the promised day: "
                       f"{int(o.loc['late_on_promised_day', 'orders'])} of {int(o.loc['late', 'orders'])} late parcels "
                       f"arrived during it, rated like on-time ones ({X.lateday.loc['late_0_1', 'low_pct']:.1f}% "
                       f"1–2★, against {X.lateday.loc['early_0_1', 'low_pct']:.1f}% a day earlier)"))
    add(Row(6, "~2.1-day post-delivery window", "observed", "as slide 4",
            lambda: X.tim.loc["low review, on time", "median_answer_days_after_delivery"], ("2.1",),
            window=W_ANALYSIS_DR))

    # ---------------- slide 7
    add(Row(7, "Tested on 12,630 unseen future deliveries", "observed", "predictions_p1a_test.csv rows",
            lambda: D.n_del, ("12,630",), window=W_TEST_DEL, correction="delivered test orders"))
    add(Row(7, "cuts date error by 60% … 9.79 days … 3.90 days  (beats GBDT 4.14d & Linear 4.18d)", "derived",
            "reg_p1a.csv MAE; 1 − 3.90 / 9.79",
            lambda: (D.mae_gain, D.mae_olist, D.mae_model, X.reg.loc["HistGradientBoostingRegressor", "mae"],
                     X.reg.loc["LinearRegression", "mae"]),
            ("60", "9.79", "3.90", "4.14", "4.18"), window=W_TEST_DEL))
    add(Row(7, "Costs R$0 to quote", "assumed", "none",
            correction=f"the forecast on its own is exceeded by {D.r_raw['late_pct']:.1f}% of deliveries (the "
                       f"platform's date by {D.r_olist['late_pct']:.1f}%); forecast + 8 days: {D.r_r1['late_pct']:.1f}%, "
                       f"{D.r_r1['mean_promised_days'] - D.r_olist['mean_promised_days']:.1f} days longer "
                       f"(customer_promise_rules.csv)"))
    add(Row(7, "Pads ~11 extra buffer days", "observed", "as slide 5", lambda: D.buffer_all, ("11",), tol=(0.5,),
            window=W_ANALYSIS_D, correction=f"{D.buffer_test:.1f} on the test window"))
    add(Row(7, "lifts PR-AUC 2.9× vs 1.2× (ROC-AUC 0.782 vs 0.584)", "observed",
            "clf_p1b.csv: derived route vs HistGradientBoostingClassifier",
            lambda: (X.clf1b.loc["Derived: regression margin vs promise", "pr_auc_lift"],
                     X.clf1b.loc["HistGradientBoostingClassifier", "pr_auc_lift"],
                     X.clf1b.loc["Derived: regression margin vs promise", "roc_auc"],
                     X.clf1b.loc["HistGradientBoostingClassifier", "roc_auc"]),
            ("2.9", "1.2", "0.782", "0.584"), window=W_TEST_DEL))
    add(Row(7, "35% of late orders (3.5×)", "observed", "capacity_p1b.csv (derived, top 10%): recall, lift",
            lambda: (100 * X.cap1b.loc[10.0, "recall_at_k"], X.cap1b.loc[10.0, "lift_at_k"]), ("35", "3.5"),
            window=W_TEST_DEL,
            correction=f"{100 * X.cap1b.loc[10.0, 'precision_at_k']:.1f}% of that tenth were late"))
    add(Row(7, "#23 of 238 delivered orders on 16 July", "observed", "customer_story.csv (p_late_derived rank)",
            lambda: (s["late_rank_on_day"], s["delivered_orders_on_day"]), ("23", "238"), window=W_DAY,
            keys={"story_late_rank": s["late_rank_on_day"], "story_delivered_on_day": s["delivered_orders_on_day"]}))
    add(Row(7, "≥ 11% risk: Pre-warns 83% (5 in 6)", "observed",
            "capacity_operating_points.csv (P1B, 0.11): recall",
            lambda: (100 * D.p1b["threshold"], 100 * D.p1b["recall"], 6 * D.p1b["recall"]), ("11", "83", "5"),
            tol=(None, None, 0.5), window=W_TEST_DEL,
            correction=f"reaches {D.p1b['flagged_pct']:.1f}% of customers; {100 - 100 * D.p1b['precision']:.1f}% of "
                       f"them then on time ({int(o.loc['prewarn_late_flagged', 'orders'])} late, "
                       f"{int(o.loc['prewarn_ontime_flagged', 'orders']):,} on time)"))
    add(Row(7, "R$12 express upgrade vs. R$40 bad-review cost", "assumed",
            "config.COSTS['expedite_cost'], ['late_delivery_cost']",
            lambda: (c["expedite_cost"], c["late_delivery_cost"]), ("12", "40"),
            correction="R$40 is charged for every late parcel"))
    add(Row(7, "27–28% precision  <  30% break-even", "observed",
            "capacity_p1b.csv precision, top 1–7.5%; 12 / 40",
            lambda: (D.cap1b_band.min(), D.cap1b_band.max(), 100 * 12 / 40), ("27", "28", "30"),
            tol=(0.5, 1.0, None), window=W_TEST_DEL,
            correction=f"{D.cap1b_band.min():.1f}–{D.cap1b_band.max():.1f}% "
                       f"({100 * X.cap1b.loc[10.0, 'precision_at_k']:.1f}% at the top 10%)"))
    add(Row(7, "at least 30% (12 / 40)", "derived", "12 / 40", lambda: 100 * 12 / 40, ("30",),
            correction=f"conservative: with the observed 1–2★ gap ({D.late_low:.1f}% − {D.ontime_low:.1f}%) a late "
                       f"parcel costs R${D.expedite_damage:.1f}, break-even {D.expedite_be:.2f}",
            keys={"expedite_expected_damage": D.expedite_damage, "expedite_breakeven_adjusted": D.expedite_be}))
    add(Row(7, "75% of transit happens after checkout inside the carrier network", "observed", "as slide 5",
            lambda: X.raw["carrier_share_pct"], ("75",), tol=(1.0,), window=W_ANALYSIS_D,
            correction=f"{X.raw['carrier_share_pct']:.1f}%, after the seller's handover"))

    # ---------------- slide 8
    add(Row(8, "Across 62 unseen test days (12,855 reviewed orders)", "observed",
            "predictions_p2_test.csv: purchase days, rows", lambda: (D.days, D.n_rev), ("62", "12,855"),
            window=W_TEST_REV))
    add(Row(8, "turns 11% random hits into 57% … (57.2%) … (11.0% base) … 89 of 100 calls wasted on happy buyers",
            "observed", "capacity_p2_daily.csv (k 10); review_timing_capacity.csv base_rate",
            lambda: (D.base, D.ten_prec, D.ten_prec, D.base, 100 - D.base),
            ("11", "57", "57.2", "11.0", "89"), window=W_TEST_REV,
            correction=f"a historical outcome (no contact recorded besides the survey) of a purchase-day list scored "
                       f"with the realised delivery; deliveries not yet rated: {100 * D.call['precision']:.1f} per 100; "
                       f"3–5★ is not all 'happy'"))
    add(Row(8, "5.2× at 10 calls/day", "derived", "0.572 / 0.110",
            lambda: X.cap2d.loc[10, "precision"] / (D.base / 100), ("5.2",), window=W_TEST_REV))
    add(Row(8, "(68% at top 5%)", "observed", "capacity_p2.csv (top 5%): precision",
            lambda: 100 * X.cap2.loc[5.0, "precision_at_k"], ("68",), window=W_TEST_REV,
            correction=f"{D.top5_never_share:.1f}% of those {D.top5_n} never arrived; {D.top5_written:.1f} in 100 had "
                       f"rated before the parcel arrived; {D.top5_reachable_delivered:.1f} in 100 could be reached at "
                       f"delivery"))
    add(Row(8, "Powered by delay vs. promise, multi-unit baskets, high freight share & category history.",
            "observed", "perm_p2.csv",
            correction=f"delay vs promise {D.perm_imp['delay_vs_estimate_days']:.3f}, days "
                       f"{D.perm_imp['delivery_days']:.3f}, items {D.perm_imp['n_items']:.3f}; freight share and "
                       f"category history rank {D.perm_rank['freight_ratio']} and "
                       f"{D.perm_rank['hist_category_low_review_rate']} of {D.perm_n}"))
    add(Row(8, "still hits 45.1% unhappy buyers (vs 9.6% at random)", "observed",
            "review_timing_composition.csv (delivered only)",
            lambda: (100 * D.ten_dlv["precision"], 100 * D.ten_dlv["base_rate"]), ("45.1", "9.6"),
            window=W_TEST_REV,
            correction=f"{D.ten_dlv['late_written_before_per_100']:.1f} of the 45.1 had rated before their late "
                       f"parcel arrived; "
                       f"{D.ten_dlv['delivered_reachable_per_100']:.1f} per 100 not yet rated at delivery"))
    add(Row(8, "top 5% = 28% (3.5×), where 99.5% wait ~2.1 days after delivery", "observed",
            "review_timing_subsets.csv (on time); review_timing.csv",
            lambda: (100 * X.tsub.loc["on time", "precision_top5pct"], X.tsub.loc["on time", "lift_top5pct"],
                     100 - X.tim.loc["low review, on time", "pct_written_before_arrival"],
                     X.tim.loc["low review, on time", "median_answer_days_after_delivery"]),
            ("28", "3.5", "99.5", "2.1"), window=W_TEST_REV, correction="99.5% and 2.1 days are 2017–18 figures"))
    add(Row(8, "Only 13 hits / 100 needed", "derived", "100 × 5 / 40", lambda: 100 * 5 / 40, ("13",), tol=(0.5,),
            correction="12.5 if every contact works; 12.5 / s if a share s does"))
    add(Row(8, "57 total hits (and 35 non-pre-empted) clear the 12.5% bar", "observed",
            "review_timing_capacity.csv (10 a day)",
            lambda: (100 * D.tcap10["precision"], 100 * D.tcap10["reachable_precision"], 100 * 5 / 40),
            ("57", "35", "12.5"), window=W_TEST_REV,
            correction=f"{D.ten['never_delivered_per_100']:.1f} of the 35 never received the parcel; a call at "
                       f"delivery reaches {D.ten['delivered_reachable_per_100']:.1f}, below 12.5"))
    add(Row(8, "35 not written before arrival … 22 wrote waiting … 43 satisfied (3–5★) … 614 total calls over 62 "
               "days", "observed", "review_timing_capacity.csv (10 a day)",
            lambda: (100 * D.tcap10["reachable_precision"], 100 * D.tcap10["already_written_precision"],
                     100 - 100 * D.tcap10["precision"], D.tcap10["contacts"], D.days),
            ("35", "22", "43", "614", "62"), window=W_TEST_REV,
            correction="list entries, not calls; 3–5★ is not all 'satisfied'"))

    # ---------------- slide 9
    add(Row(9, "8-tab Streamlit app … replay any of 62 test days", "observed",
            "predictions_p2_test.csv purchase days", lambda: D.days, ("62",), window=W_TEST_REV,
            correction="eight tabs when the slides were locked on 4 November 2026"))
    add(Row(9, "16 JULY 2018  (244 REVIEWED ORDERS) … 8 of 10  real 1–2★ … (vs ~1 by random)", "observed",
            "customer_story.csv; 10 × the day's 1–2★ rate",
            lambda: (s["reviewed_orders_on_day"], s["top10_hits"], 10, 10 * D.day_base / 100),
            ("244", "8", "10", "1"), tol=(None, None, None, 0.5), window=W_DAY,
            correction=f"a purchase-day list scored with the realised delivery: all {int(s['top10_hits'])} reachable "
                       f"only when the promised day ended ({int(s['top10_hits_never_delivered'])} never delivered, "
                       f"{int(s['top10_hits_late_written_before'])} rated while waiting); "
                       f"{int(s['top10_hits_reachable_at_delivery'])} at delivery"))
    add(Row(9, "Checkout error: 4.3 d vs Olist's 10.7 d", "observed",
            "predictions_p1a_test.csv, 16 July: MAE of forecast and promise",
            lambda: (D.day_mae_model, D.day_mae_olist), ("4.3", "10.7"), window=W_DAY,
            correction=f"the day's {D.n_day_delivered} delivered orders",
            keys={"day_mae_model": D.day_mae_model, "day_mae_olist": D.day_mae_olist}))
    never = D.day_hits[D.day_hits["outcome"] == "never delivered"]
    waiting = D.day_hits[(D.day_hits["outcome"] != "never delivered") & D.day_hits["written_before_arrival"]]
    status = never["order_status"].value_counts()
    add(Row(9, "4 buyers wrote 1★ while still waiting for a late parcel", "observed",
            "customer_journeys_test.csv, 16 July top ten",
            lambda: (int((waiting["review_score"] == 1).sum()),), ("4",), window=W_DAY))
    add(Row(9, "4 buyers never received their parcel at all (shipped, invoiced or cancelled)", "observed",
            "customer_journeys_test.csv, 16 July top ten: order_status", lambda: (len(never),), ("4",),
            window=W_DAY,
            correction=", ".join(f"{n} {k.replace('canceled', 'cancelled')}" for k, n in status.items())))
    add(Row(9, "57 / 100 average precision", "observed", "capacity_p2_daily.csv (k 10)", lambda: D.ten_prec,
            ("57",), window=W_TEST_REV, correction="351 of 614, pooled"))
    add(Row(9, "(85% RISK) … Flagged #23 of 238 delivered at checkout, and #1 of 244 reviewed on the call list",
            "observed", "customer_story.csv",
            lambda: (100 * s["p_low_at_delivery"], s["late_rank_on_day"], s["delivered_orders_on_day"],
                     s["low_rank_on_day"], s["reviewed_orders_on_day"]),
            ("85", "23", "238", "1", "244"), window=W_DAY,
            correction=f"'that morning's list' was scored once the delivery was known ({s['delivered']:%d %B}); at "
                       f"checkout its late risk was {100 * s['p_late_at_checkout']:.1f}%, above 11%"))

    # ---------------- slide 10
    add(Row(10, "Replace padded promises with our checkout forecast (60% smaller error: 9.79 → 3.90 d)", "derived",
             "reg_p1a.csv MAE", lambda: (D.mae_gain, D.mae_olist, D.mae_model), ("60", "9.79", "3.90"),
             window=W_TEST_DEL,
             correction=f"the forecast on its own is exceeded by {D.r_raw['late_pct']:.1f}% of deliveries (the "
                        f"platform's date by {D.r_olist['late_pct']:.1f}%): recommended, forecast + 8 days "
                        f"({D.r_r1['late_pct']:.1f}%)"))
    add(Row(10, "banner above 11% risk — pre-warning 83% (5 in 6)", "observed", "as slide 7",
             lambda: (100 * D.p1b["threshold"], 100 * D.p1b["recall"], 6 * D.p1b["recall"]), ("11", "83", "5"),
             tol=(None, None, 0.5), window=W_TEST_DEL,
             correction=f"reaches {D.p1b['flagged_pct']:.1f}%, {100 - 100 * D.p1b['precision']:.1f}% then on time; "
                        f"not needed with the forecast-plus-margin date"))
    add(Row(10, "closing the window behind 87.3% of late 1–2★", "observed", "as slide 4",
             lambda: X.tim.loc["low review, late", "pct_written_before_arrival"], ("87.3",), window=W_ANALYSIS_DR,
             correction=f"the effect on ratings is for the pilot's third arm to measure; {integrity}"))
    add(Row(10, "Cost: ~R$0   ·   Immediate trust gain", "assumed", "none",
             correction="an assumed cost; the trust gain is a hypothesis"))
    add(Row(10, "Stops pre-arrival 1★", "wording", "none",
             correction="by construction; the effect on the final rating is unmeasured"))
    add(Row(10, "top ~10 orders each morning (57% precision = 5.2× random; 45% on delivered parcels alone) … "
                "still wait ~2.1 days", "observed", "as slides 4 and 8",
             lambda: (10, D.ten_prec, X.cap2d.loc[10, "precision"] / (D.base / 100), 100 * D.ten_dlv["precision"],
                      X.tim.loc["low review, on time", "median_answer_days_after_delivery"]),
             ("10", "57", "5.2", "45", "2.1"), window=W_TEST_REV,
             correction=f"as a team would work it: {100 * D.call['precision']:.1f} per 100 ({D.call['lift']:.1f}× the "
                        f"{100 * D.call['pool_base_rate']:.1f}% base)"))
    add(Row(10, "Checkout-only precision plateaus at 27–28% across top-K tiers — below the 30% break-even",
             "observed", "as slide 7",
             lambda: (D.cap1b_band.min(), D.cap1b_band.max(), 100 * 12 / 40), ("27", "28", "30"),
             tol=(0.5, 1.0, None), window=W_TEST_DEL,
             correction=f"{D.cap1b_band.min():.1f}–{D.cap1b_band.max():.1f}%; 30% is conservative"))
    add(Row(10, "Cost: ~R$5/call", "assumed", "config.COSTS['outreach_cost']", lambda: c["outreach_cost"], ("5",)))
    add(Row(10, "Refit monthly on a rolling 12-month window (because transit speed halved in 6 months)", "wording",
             "lateness_monthly.csv mean_lead",
             correction=f"the 12-month window was not tested (expanding windows were); delivery time halved "
                        f"({m.loc['2018-02', 'mean_lead']:.1f} → {m.loc['2018-08', 'mean_lead']:.1f} days)"))

    # ---------------- slide 12
    add(Row(12, "squeezing 0.001 AUC from a complex ensemble", "observed", "clf_p2.csv PR-AUC: boosted trees − stack",
             lambda: D.stack_gap, ("0.001",), window=W_TEST_REV,
             correction=f"PR-AUC; the deployed stack is {D.stack_gap:.4f} below"))
    add(Row(12, "a fake 1.00 AUC", "observed", "as slide 5", lambda: X.leak_leaked, ("1.00",), window=W_TEST))
    add(Row(12, "87.3% of late-order bad reviews were written before the parcel even arrived", "observed",
             "as slide 4", lambda: X.tim.loc["low review, late", "pct_written_before_arrival"], ("87.3",),
             window=W_ANALYSIS_DR))
    add(Row(12, "27–28% (< 30% rush break-even)", "observed", "as slide 7",
             lambda: (D.cap1b_band.min(), D.cap1b_band.max(), 100 * 12 / 40), ("27", "28", "30"),
             tol=(0.5, 1.0, None), window=W_TEST_DEL))
    add(Row(12, "how to run the 6-week pilot", "planned", "pilot_design.csv (k 10, 25%)", lambda: D.k10["weeks_80"],
             ("6",),
             correction=f"per contact: {int(D.arm_b2['weeks_80'])} weeks (message, two arms; "
                        f"{int(D.arm_b3['weeks_80'])} with three), {int(D.arm_a['weeks_80'])} (delivery-day call)"))
    add(Row(12, "75% of delivery time happens inside third-party carriers", "observed", "as slide 5",
             lambda: X.raw["carrier_share_pct"], ("75",), tol=(1.0,), window=W_ANALYSIS_D,
             correction=f"{X.raw['carrier_share_pct']:.1f}%"))
    add(Row(12, "Northeast / North routes face structural distance disadvantages.", "observed",
             "segments_p1a_region.csv", window=W_TEST_DEL,
             correction=f"mean delivery {X.seg.loc['North', 'actual_mean']:.1f} (North), "
                        f"{X.seg.loc['Northeast', 'actual_mean']:.1f} (Northeast), "
                        f"{X.seg.loc['Southeast', 'actual_mean']:.1f} days (Southeast); forecast error "
                        f"{X.seg.loc['North', 'mae']:.1f}, {X.seg.loc['Northeast', 'mae']:.1f}, "
                        f"{X.seg.loc['Southeast', 'mae']:.1f}"))

    # ---------------- slide 13
    add(Row(13, "60% closer date … Top-10 calls (5.2×)", "derived", "as slides 7 and 8",
             lambda: (D.mae_gain, X.cap2d.loc[10, "precision"] / (D.base / 100)), ("60", "5.2"), window=W_TEST))
    add(Row(13, "Saved before 1★!", "wording", "none",
             correction=f"means reaching the customer before they rate the order; {integrity}"))
    add(Row(13, "8 of 10 real 1–2★ (#1 = São Gonçalo!)", "observed", "as slide 9",
             lambda: (s["top10_hits"], 10, s["low_rank_on_day"]), ("8", "10", "1"), window=W_DAY))
    add(Row(13, "What-if cost & pilot planner … Test R$5 call vs R$40 review & 6-wk RCT", "assumed",
             "config.COSTS; pilot_design.csv weeks_80",
             lambda: (c["outreach_cost"], c["low_review_cost"], D.k10["weeks_80"]), ("5", "40", "6"),
             correction="both costs are assumptions (customer_value_sensitivity.csv varies them); pilot sizes per "
                        "contact: customer_pilot_arms.csv"))
    return rows


# ---------------------------------------------------------------------------
# Checking and writing
# ---------------------------------------------------------------------------
_NUM = re.compile(r"^[-+]?[\d,]*\.?\d+$")


def _fold(x: str) -> str:
    x = unicodedata.normalize("NFKD", str(x).casefold())
    return re.sub(r"\s+", " ", "".join(ch for ch in x if not unicodedata.combining(ch))).strip()


def check(row: Row) -> tuple[str, str, object, str]:
    """Recompute a row; return (value as text, shown as text, matches, tolerance as text). The
    tolerance is half a unit of the slide's last digit unless the row sets a wider one for a
    figure the slide gives as approximate."""
    shown = "; ".join(row.shown)
    if row.compute is None:
        return "", shown, "", ""
    got = row.compute()
    got = got if isinstance(got, tuple) else (got,)
    if len(got) != len(row.shown):
        raise SystemExit(f"slide {row.slide}: {row.on_slide!r} recomputes {len(got)} values for "
                         f"{len(row.shown)} on the slide")
    tol = row.tol or (None,) * len(got)
    parts, tols, ok = [], [], True
    for g, sh, t in zip(got, row.shown, tol):
        if _NUM.match(sh):
            num = float(sh.replace(",", ""))
            dec = len(sh.split(".")[1]) if "." in sh else 0
            t = 0.5 * 10 ** -dec if t is None else t
            g = float(g)
            ok &= abs(g - num) <= t + 1e-9
            sep = "," if "," in sh else ""
            parts.append(f"{g:{sep}.{dec + 1}f}" if not float(g).is_integer() or dec else f"{g:{sep}.0f}")
            tols.append(f"{t:g}")
        else:
            ok &= _fold(g) == _fold(sh)
            parts.append(str(g))
            tols.append("text")
    return "; ".join(parts), shown, bool(ok), "; ".join(tols)


TEX_MAP = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_\allowbreak{}", "{": r"\{", "}": r"\}", "~": r"$\sim$", "^": r"\^{}",
           "★": r"$\star$", "≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\rightarrow$", "×": r"$\times$",
           "−": r"$-$", "<": r"$<$", ">": r"$>$", "●": r"$\bullet$", "/": r"/\allowbreak{}", "|": r"$|$"}


def tex(s: str) -> str:
    """LaTeX-escape a field. Accented letters stay UTF-8 (the report loads inputenc utf8); an
    opening single quote becomes ` so quoted words read 'like this' in the PDF."""
    s = re.sub(r"(^|(?<=[\s(]))'", "`", str(s))
    return "".join(TEX_MAP.get(ch, ch) for ch in s)


def tex_quote(on_slide: str) -> str:
    return r" \ldots{} ".join(f"``{tex(frag.strip())}''" for frag in on_slide.split(" … "))


def write_tex(prov: pd.DataFrame, path: Path) -> None:
    w = lambda pct: rf"p{{\dimexpr(\textwidth-6\tabcolsep)*{pct}/100\relax}}"
    head = (r"\textbf{On the slide} & \textbf{Kind} & \textbf{Source or formula} & "
            r"\textbf{Reading or correction}\tabularnewline")
    cell = lambda x: r"\raggedright " + x
    lines = [
        "% Generated by src/slide_provenance.py from report/results/slide_provenance.csv.",
        "% Do not edit by hand: change the script and run it again.",
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        rf"\begin{{longtable}}{{@{{}}{w(24)}{w(10)}{w(33)}{w(33)}@{{}}}}",
        r"\caption{Where the numbers on the slides come from: every number or claim, slide 11 first, with its "
        r"kind, its source or formula ($\rightarrow$ the value recomputed from it) and the reading this report "
        r"stands by. Rows count the test window unless the slide's heading or a bracket names another. Generated by "
        r"\code{src/slide\_provenance.py} from \code{report/results/slide\_provenance.csv}.}"
        r"\label{tab:provenance}\tabularnewline",
        r"\toprule", head, r"\midrule", r"\endfirsthead",
        r"\multicolumn{4}{@{}l}{\textit{Table~\ref{tab:provenance}, continued}}\tabularnewline",
        r"\toprule", head, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
    ]
    for i, slide in enumerate(SLIDE_ORDER):
        part = prov[prov["slide"] == slide]
        if part.empty:
            continue
        if i:
            lines.append(r"\addlinespace[3pt]")
        # A slide whose rows mostly count one window other than the test window names it once in
        # its heading; then only the rows counted on another window carry a bracket.
        named = part.loc[part["window"] != "", "window"]
        local = named.map(lambda x: "test" if x in DEFAULT_WINDOWS else x).value_counts()
        default = local.index[0] if len(local) and local.iloc[0] >= 2 else "test"
        heading = rf"\textbf{{Slide {slide}}}: {tex(SLIDE_TOPIC[slide])}"
        if default != "test":
            heading += rf" \textit{{(rows: {tex(SHORT_WINDOW[default])}, unless bracketed)}}"
        # \\* keeps the slide's heading on the page of its first row
        lines.append(rf"\multicolumn{{4}}{{@{{}}p{{\textwidth}}@{{}}}}{{{heading}}}" + r"\\*")
        for _, r in part.iterrows():
            src = r["source"]
            if r["value"]:
                src = f"{src} → {r['value']}"
            win = "test" if r["window"] in DEFAULT_WINDOWS else r["window"]
            if r["window"] and win != default:
                src = f"{src} [{'test window' if win == 'test' else SHORT_WINDOW[win]}]"
            reading = r["correction"] if r["correction"] else "as on the slide"
            lines.append(" & ".join([cell(tex_quote(r["on_slide"])), cell(tex(r["kind"])), cell(tex(src)),
                                     cell(tex(reading))]) + r"\tabularnewline")
    lines += [r"\end{longtable}}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> pd.DataFrame:
    X = load_inputs()
    D = derived(X)
    rows = build_rows(X, D)
    out = []
    for row in rows:
        if row.kind not in KINDS:
            raise SystemExit(f"slide {row.slide}: unknown kind {row.kind!r}")
        value, shown, matches, tolerance = check(row)
        keys = "; ".join(f"{k}={float(v)!r}" for k, v in row.keys.items())
        out.append({"slide": row.slide, "on_slide": row.on_slide, "kind": row.kind, "value": value, "shown": shown,
                    "matches": matches, "source": row.source, "window": row.window, "correction": row.correction,
                    "tolerance": tolerance, "keys": keys})
    prov = pd.DataFrame(out)
    order = {s: i for i, s in enumerate(SLIDE_ORDER)}
    prov = prov.sort_values("slide", key=lambda x: x.map(order), kind="stable").reset_index(drop=True)
    bad = prov[prov["matches"].astype(str) == "False"]
    if len(bad):
        raise SystemExit("these slide numbers no longer match their source:\n"
                         + bad[["slide", "on_slide", "value", "shown"]].to_string(index=False))
    costs = prov[prov["source"].str.contains("config.COSTS", regex=False) & (prov["kind"] != "assumed")]
    if len(costs):
        raise SystemExit("rows naming config.COSTS must be marked 'assumed':\n"
                         + costs[["slide", "on_slide", "kind"]].to_string(index=False))
    prov.to_csv(CSV_OUT, index=False)
    write_tex(prov, TEX_OUT)
    return prov


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p = main()
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(p[["slide", "kind", "on_slide", "value", "shown", "matches"]].to_string(index=False))
    checked = p[p["matches"] != ""]
    print(f"\n{len(p)} rows over {p['slide'].nunique()} slides; {len(checked)} recomputed, all match; "
          f"{(p['kind'] == 'assumed').sum()} assumed")
    print(f"wrote {CSV_OUT} and {TEX_OUT}")
