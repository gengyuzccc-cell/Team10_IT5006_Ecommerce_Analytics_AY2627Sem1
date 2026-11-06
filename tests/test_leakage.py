"""Validation checks for the Phase 2 feature pipeline.

Run directly (no pytest needed):

    python tests/test_leakage.py

The history features in src/features.py are the one place in this project where
a plausible-looking implementation can silently leak the future into the past,
and the damage would be invisible in the metrics - it would just look like a
better model. So they are not merely reviewed, they are checked against a
brute-force recomputation: for a sample of orders, the seller's historical mean
is rebuilt from first principles by filtering every earlier order by hand, and
the two must agree exactly.

The remaining checks cover the contracts the rest of the pipeline relies on:
the join grain, the leakage guard in both regimes, split disjointness, and
backward compatibility with Phase 1.

Sections 8 to 11 read the tables the src/ scripts wrote rather than re-running
them. Section 11, added after the presentation of 5 November 2026, ties the
customer counts, contact lists, reais and pilot sizes of src/customer_view.py
and the slide sources of src/slide_provenance.py to each other and to the
earlier tables, checks that the slides are byte-identical to the locked deck,
lints the documents for wording the revision retired, and checks that the
Replay tab's blind download carries no outcome.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
from data_load import build_orders, load_all, load_analysis
from features import (
    CHECKOUT_FEATURES, DELIVERY_FEATURES, HISTORY_FEATURES, MODEL_FEATURES,
    REVIEW_MODEL_FEATURES, add_history, add_promise, assert_no_leakage,
    build_features, make_splits,
)

FAILURES: list[str] = []


def raises(fn, exc_type) -> bool:
    """True if `fn()` raises `exc_type`."""
    try:
        fn()
    except exc_type:
        return True
    except Exception:
        return False
    return False


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# ===========================================================================
print("Loading source tables ...")
TABLES = load_all()
FULL = build_orders(TABLES)
DF = build_features(load_analysis(TABLES), with_history=True)
SP = make_splits(DF)
RNG = np.random.default_rng(config.SEED)


# ===========================================================================
print("\n1. Join grain and source integrity")
# ---------------------------------------------------------------------------
check("orders table keeps its documented row count",
      len(FULL) == config.EXPECTED_ROWS["orders"], f"{len(FULL):,} rows")
check("order_id is unique after the nine-table join",
      FULL["order_id"].is_unique)
check("analysis window is a strict subset",
      len(DF) < len(FULL), f"{len(DF):,} of {len(FULL):,}")
check("every analysis row sits inside the declared window",
      DF["order_purchase_timestamp"].between(
          config.ANALYSIS_START, config.ANALYSIS_END).all())


# ===========================================================================
print("\n2. History features never see the future (brute-force recomputation)")
# ---------------------------------------------------------------------------
# Rebuild hist_seller_lead_days from first principles for a random sample. An
# order may average in an earlier order only if that earlier parcel had already
# reached its customer at the moment this one was placed.
pool = DF[DF["hist_seller_n"] >= config.MIN_HISTORY].index.to_numpy()
sample = RNG.choice(pool, size=200, replace=False)

src = DF[["lead_seller_id", "order_purchase_timestamp",
          "order_delivered_customer_date", "delivery_days"]]
mismatches, checked = [], 0
for idx in sample:
    row = DF.loc[idx]
    prior = src[(src["lead_seller_id"] == row["lead_seller_id"])
                & (src["order_delivered_customer_date"] < row["order_purchase_timestamp"])]
    prior = prior["delivery_days"].dropna()
    if len(prior) < config.MIN_HISTORY:
        continue
    checked += 1
    if not np.isclose(prior.mean(), row["hist_seller_lead_days"], atol=1e-9):
        mismatches.append((idx, prior.mean(), row["hist_seller_lead_days"]))

check("seller lead-time history matches a hand recomputation",
      not mismatches, f"{checked} orders verified, {len(mismatches)} mismatched")
if mismatches[:3]:
    for idx, expected, got in mismatches[:3]:
        print(f"        idx {idx}: expected {expected:.6f}, got {got:.6f}")

# The strict-boundary rule: nothing may be averaged in at the same instant.
same_instant = DF[DF["order_delivered_customer_date"] == DF["order_purchase_timestamp"]]
check("no order can be its own history",
      len(same_instant) == 0 or same_instant["hist_seller_n"].notna().all(),
      f"{len(same_instant)} same-instant rows")

# A history feature computed over the future would track the *current* month's
# performance too well. The platform trailing mean should LAG the actual mean,
# never anticipate it.
monthly = DF.dropna(subset=["delivery_days"]).groupby(
    DF["order_purchase_timestamp"].dt.to_period("M")).agg(
    actual=("delivery_days", "mean"), tracked=("hist_platform_lead_days", "mean"))
lagged_corr = monthly["tracked"].corr(monthly["actual"].shift(1))
same_corr = monthly["tracked"].corr(monthly["actual"])
check("platform history tracks the previous month more tightly than the current one",
      lagged_corr > same_corr,
      f"corr(lag-1)={lagged_corr:.3f} > corr(same)={same_corr:.3f}")

# Every history value must be finite or NaN - never inf from a zero divisor.
# (np.isfinite(where=...) leaves the masked positions uninitialised, so fill
# first rather than mask.)
_hist = DF[HISTORY_FEATURES].to_numpy(dtype=float)
check("no infinite history values",
      bool(np.isfinite(np.nan_to_num(_hist, nan=0.0)).all()),
      f"{int(np.isinf(_hist).sum())} infinities")

# The first orders in the window cannot have a track record.
early = DF.nsmallest(50, "order_purchase_timestamp")
check("the earliest orders carry little or no seller history",
      early["hist_seller_n"].median() < DF["hist_seller_n"].median(),
      f"median n: {early['hist_seller_n'].median():.0f} vs {DF['hist_seller_n'].median():.0f}")


# ===========================================================================
print("\n3. Leakage guard, both prediction moments")
# ---------------------------------------------------------------------------
for name, feats, regime in [
    ("MODEL_FEATURES @ at_checkout", MODEL_FEATURES, "at_checkout"),
    ("CHECKOUT_FEATURES @ at_checkout", CHECKOUT_FEATURES, "at_checkout"),
    ("REVIEW_MODEL_FEATURES @ at_delivery", REVIEW_MODEL_FEATURES, "at_delivery"),
    ("DELIVERY_FEATURES @ at_delivery", DELIVERY_FEATURES, "at_delivery"),
]:
    try:
        assert_no_leakage(feats, regime)
        check(name, True)
    except ValueError as exc:
        check(name, False, str(exc))

# The guard must actually bite.
for bad, regime in [(["delivery_days"], "at_checkout"),
                    (["review_score"], "at_delivery")]:
    try:
        assert_no_leakage(MODEL_FEATURES + bad, regime)
        check(f"guard rejects {bad[0]} at {regime}", False, "it did not")
    except ValueError:
        check(f"guard rejects {bad[0]} at {regime}", True)

check("an unknown regime is an error, not a silent pass",
      raises(lambda: assert_no_leakage(MODEL_FEATURES, "at_some_point"), KeyError))


# ===========================================================================
print("\n4. Split discipline")
# ---------------------------------------------------------------------------
tr, va, te = SP["train"], SP["val"], SP["test"]
check("train precedes validation precedes test",
      tr["order_purchase_timestamp"].max() <= va["order_purchase_timestamp"].min()
      <= va["order_purchase_timestamp"].max() <= te["order_purchase_timestamp"].min())
check("the splits partition the analysis window",
      len(tr) + len(va) + len(te) == len(DF))
overlap = set(tr["customer_unique_id"]) & set(SP["test_unseen_customers"]["customer_unique_id"])
check("test_unseen_customers shares no buyer with train",
      not overlap, f"{len(overlap)} shared")
check("straddling customers were actually found and removed",
      SP["n_straddling_customers"] > 0, f"{SP['n_straddling_customers']} buyers")


# ===========================================================================
print("\n5. Phase 1 backward compatibility")
# ---------------------------------------------------------------------------
# build_features(with_history=False) must reproduce the Phase 1 frame exactly,
# or every number in the submitted Phase 1 report silently changes.
p1 = build_features(load_analysis(TABLES))
check("with_history=False adds no history columns",
      not (set(HISTORY_FEATURES) & set(p1.columns)))
shared = [c for c in p1.columns if c in DF.columns]
same = all(p1[c].equals(DF[c]) for c in shared if c not in ("route",))
check("every Phase 1 column is byte-identical with and without history",
      same, f"{len(shared)} columns compared")
check("the pinned promise threshold matches the analysis frame",
      np.isclose(DF["estimated_days"].median(), config.PROMISE_MEDIAN_DAYS, atol=1e-9),
      f"{DF['estimated_days'].median():.9f} vs {config.PROMISE_MEDIAN_DAYS:.9f}")
_one_row = DF.head(1)[["estimated_days"]].assign(
    estimated_days=lambda d: d["estimated_days"].astype(float))
check("long_promise is stable on a one-row frame",
      bool(add_promise(_one_row, threshold=config.PROMISE_MEDIAN_DAYS)["long_promise"].iloc[0])
      == bool(DF["long_promise"].iloc[0]),
      "serving path reproduces the training-time flag")
check("Phase 1 feature lists are unchanged",
      len(MODEL_FEATURES) == 25 and len(REVIEW_MODEL_FEATURES) == 28,
      f"{len(MODEL_FEATURES)} / {len(REVIEW_MODEL_FEATURES)}")


# ===========================================================================
print("\n6. Target sanity")
# ---------------------------------------------------------------------------
check("late is defined only for delivered orders",
      DF.loc[DF["target_is_late"].notna(), "order_delivered_customer_date"].notna().all())
check("low-review is defined only where a review exists",
      DF.loc[DF["target_low_review"].notna(), "review_score"].notna().all())
check("no target was imputed to a negative label",
      DF["target_is_late"].isna().sum() > 0 and DF["target_low_review"].isna().sum() > 0)
check("both targets are genuinely imbalanced",
      0.05 < DF["target_is_late"].mean() < 0.15
      and 0.08 < DF["target_low_review"].mean() < 0.25,
      f"late {DF['target_is_late'].mean():.3f}, low review {DF['target_low_review'].mean():.3f}")


# ===========================================================================
print("\n7. The serving path agrees with the training path")
# ---------------------------------------------------------------------------
# The dashboard rebuilds a feature row from raw form fields. If that row drifts
# from what the model was fitted on, the deployed predictions quietly stop
# matching the reported metrics - and the first version of this check would have
# caught a real instance: the platform fallback in models/history_lookup.csv was
# derived by filtering to orders *delivered* in the final 28 days, which selects
# the slowest, most truncated tail of the extract. It read 88 days against a
# training value of 11.7, and the deployed regression over-predicted a typical
# Sao Paulo to Rio order by twenty days while every offline number stayed
# perfect.
try:
    import types as _types
    _st = _types.ModuleType("streamlit")
    # Decorators used at import time, bare (@st.fragment) or called (@st.cache_data(...)).
    _passthrough = lambda f=None, **k: f if callable(f) else (lambda g: g)  # noqa: E731
    _st.cache_resource = _st.cache_data = _st.fragment = _passthrough
    sys.modules.setdefault("streamlit", _st)
    sys.path.insert(0, str(PROJECT_ROOT / "deployment"))
    import predict as PR

    art = PR.load_artifacts()
    lookup_global = art["lookup"]["global"].iloc[0]
    trained_tail = (DF.sort_values("order_purchase_timestamp")["hist_platform_lead_days"]
                    .dropna().iloc[-1])
    check("the shipped platform fallback matches the trained trailing value",
          np.isclose(lookup_global["lead_days"], trained_tail, atol=0.01),
          f"lookup {lookup_global['lead_days']:.2f} vs trained {trained_tail:.2f}")

    te = SP["test"].dropna(subset=["target_delivery_days",
                                   "lead_seller_state", "lead_category"]).head(150)
    raw = pd.DataFrame([{
        "order_purchase_timestamp": o["order_purchase_timestamp"],
        "order_estimated_delivery_date": o["order_estimated_delivery_date"],
        "customer_state": o["customer_state"], "seller_state": o["lead_seller_state"],
        "lead_category": o["lead_category"], "lead_payment_type": o["lead_payment_type"],
        "n_items": o["n_items"], "n_sellers": o["n_sellers"],
        "total_price": o["total_price"], "total_freight": o["total_freight"],
        "total_weight_g": o["total_weight_g"], "max_volume_cm3": o["max_volume_cm3"],
        "total_photos": o["total_photos"], "max_installments": o["max_installments"],
    } for _, o in te.iterrows()])

    served = PR.score(PR.assemble(raw, art["lookup"], PR.route_distances(DF)),
                      art)["predicted_days"].to_numpy()
    trained = art["p1a"]["estimator"].predict(te[CHECKOUT_FEATURES])
    gap = float(np.abs(served - trained).mean())
    # The two cannot be identical: the form supplies a state pair rather than a
    # postcode, and no seller id, so distance and seller history are stand-ins.
    # A few days of mean difference is that approximation; twenty is a bug.
    check("serving predictions track training predictions", gap < 5.0,
          f"mean |serving - training| = {gap:.2f} days over {len(te)} real orders")
    check("serving predictions are in a plausible range",
          3 < served.mean() < 30, f"mean predicted {served.mean():.1f} days")
except Exception as exc:  # noqa: BLE001
    check("serving path is exercisable", False, f"{type(exc).__name__}: {exc}")


# ===========================================================================
print("\n8. Decision layer and EDA tables (added after the Milestone 1 feedback)")
# ---------------------------------------------------------------------------
# Precision@K / Recall@K are computed from the saved test predictions rather
# than refitted, so the check here is that the arithmetic is right and that the
# shipped tables match a recomputation.
import evaluate as E

RESULTS = PROJECT_ROOT / "report" / "results"
# Eight orders, three positives; the ranking puts two positives first, then a
# negative, then the third positive. Top 25% (two orders) is all hits; top 50%
# (four orders) holds three of the three positives.
_y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
_s = np.array([.9, .8, .7, .6, .5, .4, .3, .2])
_full = E.precision_recall_at_k(_y, _s, fractions=(1.0,)).iloc[0]
check("precision@100% is the base rate and recall@100% is one",
      np.isclose(_full["precision_at_k"], _y.mean()) and _full["recall_at_k"] == 1.0)
_top = E.precision_recall_at_k(_y, _s, fractions=(0.25, 0.5)).set_index("k_pct")
check("precision@K counts the hits at the top of the ranking",
      _top.loc[25.0, "precision_at_k"] == 1.0 and _top.loc[25.0, "recall_at_k"] == 2 / 3
      and _top.loc[50.0, "precision_at_k"] == 0.75 and _top.loc[50.0, "recall_at_k"] == 1.0)

try:
    pred = pd.read_csv(RESULTS / "predictions_p2_test.csv",
                       parse_dates=["order_purchase_timestamp"])
    cap = pd.read_csv(RESULTS / "capacity_p2.csv")
    recomputed = E.precision_recall_at_k(pred["actual_low_review"], pred["p_low_review"],
                                         fractions=tuple(cap["k_pct"] / 100))
    check("capacity_p2.csv reproduces from the per-order test predictions",
          np.allclose(recomputed["precision_at_k"], cap["precision_at_k"])
          and np.allclose(recomputed["recall_at_k"], cap["recall_at_k"]))
    check("recall@K never falls as K grows",
          bool(cap["recall_at_k"].is_monotonic_increasing))
    daily = pd.read_csv(RESULTS / "capacity_p2_daily.csv")
    check("a daily capacity of K never flags more than K orders per day",
          bool((daily["flagged"] <= daily["k_per_period"] * daily["periods"]).all()))
except FileNotFoundError as exc:
    check("capacity tables exist (run src/capacity_metrics.py)", False, str(exc))

try:
    weekly = pd.read_csv(RESULTS / "lateness_weekly.csv").set_index("week_start")
    monthly = pd.read_csv(RESULTS / "lateness_monthly.csv")
    strike = pd.read_csv(RESULTS / "lateness_strike.csv").set_index("cohort")
    check("weekly and monthly lateness tables cover the same delivered orders",
          int(weekly["n"].sum()) == int(monthly["n"].sum()),
          f"{int(weekly['n'].sum()):,} = {int(monthly['n'].sum()):,}")
    before = weekly.loc["2018-04-09":"2018-05-07", "late_pct"].mean()
    check("the week before the strike stands out from the weeks before it",
          weekly.loc["2018-05-14", "late_pct"] > 2 * before,
          f"{weekly.loc['2018-05-14', 'late_pct']:.1f}% vs {before:.1f}% mean")
    check("the promised window lengthened during and after the strike",
          weekly.loc["2018-05-28", "promise_days"] > weekly.loc["2018-05-14", "promise_days"] + 10)
    transit = "in transit during the strike (bought before 1 Jun, arrived on/after 21 May)"
    control = "May 1-20, delivered before the strike"
    check("orders in transit during the strike were late far more often than May orders it missed",
          strike.loc[transit, "late_pct"] > 5 * strike.loc[control, "late_pct"])
except (FileNotFoundError, KeyError) as exc:
    check("lateness tables exist (run src/lateness_anatomy.py)", False, repr(exc))


# ===========================================================================
print("\n9. Phase 3: what the report says is deployed is what is pickled (Phase 2 feedback)")
# ---------------------------------------------------------------------------
# The Phase 2 feedback found three places naming three different Problem 2
# models. These checks tie the metadata, the pickles, the derived tables and
# the documentation to one another so the statement cannot drift again.
import json
import modeling as M

try:
    meta = json.loads((PROJECT_ROOT / "models" / "model_metadata.json").read_text("utf-8"))
    bundles = {k: M.load_model(PROJECT_ROOT / "models" / f)
               for k, f in [("p1a", "p1a_leadtime.joblib"), ("p2", "p2_lowreview.joblib")]}
    check("the pickled lead-time model carries the label model_metadata.json announces",
          bundles["p1a"]["label"] == meta["p1a"]["label"],
          f"{bundles['p1a']['label']} vs {meta['p1a']['label']}")
    check("the pickled review model carries the label model_metadata.json announces",
          bundles["p2"]["label"] == meta["p2"]["label"],
          f"{bundles['p2']['label']} vs {meta['p2']['label']}")
    check("the deployed review model is the StackingClassifier the report names",
          type(bundles["p2"]["estimator"]).__name__ == "StackingClassifier"
          and "Stacking" in meta["p2"]["label"])
    check("the deployed lead-time model is the VotingRegressor the report names",
          type(bundles["p1a"]["estimator"]).__name__ == "VotingRegressor"
          and "Voting" in meta["p1a"]["label"])

    # The ensembles must be built from the tuned winners, with the winners'
    # hyperparameters - otherwise Appendix C would describe models nobody fitted.
    hp = pd.read_csv(RESULTS / "hyperparameters.csv", dtype=str).fillna("")
    cand = hp[hp["role"] == "candidate"].set_index(["problem", "estimator", "parameter"])["value"]
    mismatches = []
    for _, row in hp[hp["role"].str.startswith("deployed member")].iterrows():
        key = (row["problem"], row["estimator"], row["parameter"])
        if key in cand.index and cand.loc[key] != row["value"]:
            mismatches.append(f"{key}: {cand.loc[key]} vs {row['value']}")
    check("each deployed ensemble member carries its tuned candidate's hyperparameters",
          not mismatches, "; ".join(mismatches))

    # README and the final report must name the same deployed models.
    docs = [PROJECT_ROOT / "README.md"]
    docs += list((PROJECT_ROOT / "report" / "sections").glob("p3_*deployment*.tex"))
    for doc in docs:
        text = doc.read_text("utf-8")
        check(f"{doc.name} names the deployed review model ({meta['p2']['label']})",
              meta["p2"]["label"].split(" (")[0] in text)
except (FileNotFoundError, KeyError) as exc:
    check("deployment artifacts and hyperparameter table exist", False, repr(exc))

# Fold-to-fold variation: the new table must be the selection tables re-read,
# and its spread must match the cv_std the results tables already carried.
try:
    folds = pd.read_csv(RESULTS / "cv_folds.csv")
    fold_cols = [f"fold{i}" for i in range(1, 6)]
    recomputed_sd = folds[fold_cols].std(axis=1, ddof=0)
    check("cv_folds.csv standard deviations are the population spread of the five folds",
          np.allclose(recomputed_sd, folds["cv_sd"]))
    p2 = folds[folds["problem"] == "P2 low review"].set_index("model")
    clf = pd.read_csv(RESULTS / "clf_p2.csv").set_index("model")
    check("cv_folds.csv agrees with the cv_std column of clf_p2.csv",
          np.allclose(p2.loc[clf.index, "cv_sd"], clf["cv_std"])
          and np.allclose(p2.loc[clf.index, "cv_mean"], clf["cv_pr_auc"]))
    reg = pd.read_csv(RESULTS / "reg_p1a.csv").dropna(subset=["cv_mae"]).set_index("model")
    p1a = folds[folds["problem"] == "P1A lead time"].set_index("model")
    check("regression fold errors are stored as positive MAE and match reg_p1a.csv",
          np.allclose(p1a.loc[reg.index, "cv_mean"], reg["cv_mae"]))
    top = p2.sort_values("cv_mean", ascending=False)
    check("the top two review candidates differ by less than a tenth of the fold-to-fold spread",
          abs(top["cv_mean"].iloc[0] - top["cv_mean"].iloc[1]) < 0.1 * top["cv_sd"].iloc[0])
except FileNotFoundError as exc:
    check("cv_folds.csv exists (run src/selection_tables.py)", False, str(exc))

# The outreach pilot: arithmetic the recommendation rests on.
try:
    import pilot_design as PD

    design = pd.read_csv(RESULTS / "pilot_design.csv")
    be = pd.read_csv(RESULTS / "pilot_breakeven.csv")
    check("a larger assumed effect never needs a larger pilot",
          all(g.sort_values("relative_reduction")["n_per_arm_80"].is_monotonic_decreasing
              for _, g in design.groupby("k_per_day")))
    check("the break-even reduction is the cost ratio divided by the control rate",
          np.allclose(be["breakeven_relative_reduction"] * be["control_rate"],
                      config.COSTS["outreach_cost"] / config.COSTS["low_review_cost"]))
    check("at the break-even effect a contact is worth exactly nothing",
          np.allclose((be["breakeven_absolute_reduction"] * config.COSTS["low_review_cost"]
                       - config.COSTS["outreach_cost"]), 0))
    check("the sample-size formula reproduces a textbook case (0.50 vs 0.40, 80% power: 388 per arm)",
          PD.n_per_arm(0.50, 0.40, power=0.80) == 388)
except FileNotFoundError as exc:
    check("pilot tables exist (run src/pilot_design.py)", False, str(exc))


# ===========================================================================
print("\n10. Phase 3: when the review is written (src/review_timing.py)")
try:
    tcap = pd.read_csv(RESULTS / "review_timing_capacity.csv")
    check("every call-list hit is either still reachable or already written",
          np.allclose(tcap["reachable_precision"] + tcap["already_written_precision"],
                      tcap["precision"]))
    pred = pd.read_csv(RESULTS / "predictions_p2_test.csv",
                       parse_dates=["order_purchase_timestamp"])
    flags = pd.read_csv(RESULTS / "review_timing_test_orders.csv")
    check("the timing flags cover exactly the held-out review predictions",
          len(flags) == len(pred) and set(flags["order_id"]) == set(pred["order_id"]),
          f"{len(flags):,} flags, {len(pred):,} predictions")
    m = pred.merge(flags, on="order_id")
    flagged = m["p_low_review"] >= 0.11
    reach = flagged & (m["actual_low_review"] == 1) & (m["written_before_arrival"] == 0)
    c = config.COSTS
    net = 1000 * (reach.sum() * c["low_review_cost"] - flagged.sum() * c["outreach_cost"]) / len(m)
    val = pd.read_csv(RESULTS / "review_timing_value.csv")
    at = val.loc[val["threshold"].round(2) == 0.11, "net_reachable_per_1k"].iloc[0]
    check("the reachable-only value recomputes from the per-order predictions and flags",
          abs(net - at) < 1e-6, f"R${net:,.0f} per 1,000 reviewed orders")
    story = pd.read_csv(RESULTS / "story_order.csv").iloc[0]
    oid = story["order_id"]
    day = m[m["order_purchase_timestamp"].dt.date == pd.Timestamp("2018-07-16").date()]
    first = day.sort_values("p_low_review", ascending=False, kind="stable").iloc[0]["order_id"]
    delivered = pd.to_datetime(TABLES["orders"].set_index("order_id").loc[oid, "order_delivered_customer_date"])
    reviews = TABLES["order_reviews"]
    answered = pd.to_datetime(reviews.loc[reviews["order_id"] == oid, "review_answer_timestamp"]).max()
    check("the presentation's order tops the 16 July list and was reviewed before it arrived",
          first == oid and answered < delivered,
          f"answered {answered:%d %b %Y}, delivered {delivered:%d %b %Y}")
except FileNotFoundError as exc:
    check("review-timing tables exist (run src/review_timing.py)", False, str(exc))


# ===========================================================================
print("\n11. Customer view and provenance (src/customer_view.py, src/slide_provenance.py)")
# ---------------------------------------------------------------------------
# These checks read the tables the two scripts wrote; they never re-run them.
# They tie the customer counts, the contact lists, the reais, the pilot and the
# sources of the slide numbers to one another and to the earlier tables, so a
# number quoted in the report, the README or the app cannot drift from its source.
import hashlib
import math
import re


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


# The date a customer is quoted: the margin is chosen on validation outcomes only.
try:
    import customer_view as CV

    frontier = _csv("customer_promise_frontier.csv")
    rules = _csv("customer_promise_rules.csv")
    platform = rules["buffer_days"].isna()
    olist_val = float(rules.loc[platform & (rules["period"] == "validation"), "late_pct"].iloc[0])
    r1_buffer = float(rules.loc[rules["rule"].str.startswith("R1"), "buffer_days"].iloc[0])
    grid = frontier.dropna(subset=["buffer_days"])
    smallest = float(grid.loc[grid["val_late_pct"] <= olist_val, "buffer_days"].min())
    check("the quoted margin is the smallest whose validation late rate is no worse than Olist's",
          smallest == r1_buffer == 8.0,
          f"{r1_buffer:.1f} days; Olist's validation late rate {olist_val:.2f}%")
    shuffled = frontier.copy()
    order = np.random.default_rng(config.SEED).permutation(len(shuffled))
    test_cols = ["late_pct", "late_calendar_day_pct", "mean_promised_days"]
    for col in test_cols:
        shuffled[col] = shuffled[col].to_numpy()[order]
    moved = not np.allclose(shuffled[test_cols].to_numpy(), frontier[test_cols].to_numpy(),
                            equal_nan=True)
    chosen = (CV.choose_buffer(shuffled), CV.choose_buffer(shuffled, olist_val))
    check("customer_view.choose_buffer() ignores the test window: shuffled test columns, same margin",
          moved and chosen == (r1_buffer, r1_buffer), f"{chosen[0]:.1f} and {chosen[1]:.1f} days")
except (FileNotFoundError, ImportError, KeyError, IndexError, ValueError) as exc:
    check("promise-margin tables exist (run src/customer_view.py)", False, repr(exc))

# What happened to the customers of the test window, each count with its population.
try:
    out = _csv("customer_outcomes_test.csv").set_index("key")
    n = out["orders"]
    p2_test = _csv("predictions_p2_test.csv")
    p1a_test = _csv("predictions_p1a_test.csv")
    comp = _csv("review_timing_composition.csv").set_index("list").loc["all reviewed orders"]
    rules = _csv("customer_promise_rules.csv")
    olist_test_late = int(rules.loc[rules["buffer_days"].isna() & (rules["period"] == "test"),
                                    "n_late"].iloc[0])
    check("every order placed in the test window arrived on time, arrived late or never arrived",
          n["on_time"] + n["late"] + n["never_delivered"] == n["placed"],
          f"{n['on_time']:,} + {n['late']:,} + {n['never_delivered']:,} = {n['placed']:,}")
    by_den = out.groupby("denominator")["denominator_n"].unique()
    check("every customer count names its population: orders placed or delivered orders",
          set(by_den.index) == {"placed", "delivered"}
          and list(by_den["placed"]) == [n["placed"]]
          and list(by_den["delivered"]) == [len(p1a_test)],
          f"placed {n['placed']:,}; delivered {len(p1a_test):,}")
    check("the slides' daily top ten: 614 contacts, 351 rated 1-2 stars (review_timing_composition.csv)",
          n["top10"] == comp["contacts"] == 614 and n["top10_hit"] == comp["hits"] == 351)
    check("1,418 customers rated the order 1-2 stars, as in the per-order review predictions",
          n["low"] == int(p2_test["actual_low_review"].sum()) == 1418, f"{n['low']:,}")
    reach = ["reach_promise_only", "reach_delivery_only", "reach_either", "reach_neither"]
    check("each customer who rated 1-2 stars falls in exactly one reach class",
          int(n[reach].sum()) == n["low"],
          " + ".join(f"{n[k]:,}" for k in reach) + f" = {n['low']:,}")
    check("every late parcel was either warned at checkout or not: 936, as in the promise rules",
          n["prewarn_late_flagged"] + n["prewarn_late_missed"] == n["late"] == olist_test_late == 936,
          f"{n['prewarn_late_flagged']:,} + {n['prewarn_late_missed']:,} = {n['late']:,}")
except (FileNotFoundError, KeyError, IndexError) as exc:
    check("customer outcome tables exist (run src/customer_view.py)", False, repr(exc))

# The reais: per 1,000 REVIEWED orders, from the assumed costs, reproducing earlier tables.
try:
    c = config.COSTS
    dec = _csv("customer_value_decomposition.csv").set_index("list")
    rtv = _csv("review_timing_value.csv")
    gross = [col for col in dec.columns if col.startswith("gross_per_1k_")]
    parts_add_up = np.allclose(dec[gross].sum(axis=1) - dec["contact_cost_per_1k"],
                               dec["net_all_per_1k"])
    same = []
    for lst, thr in [("slides_threshold_011", 0.11), ("deployed_threshold_019", 0.19)]:
        row = rtv.loc[rtv["threshold"].round(2) == thr].iloc[0]
        same.append(abs(dec.loc[lst, "net_all_per_1k"] - row["net_all_per_1k"]) < 0.01
                    and abs(dec.loc[lst, "net_not_written_per_1k"] - row["net_reachable_per_1k"]) < 0.01)
    t011 = dec.loc["slides_threshold_011"]
    check("the slides' R$1,466 / R$775 decompose by customer and match review_timing_value.csv",
          parts_add_up and all(same)
          and (round(t011["net_all_per_1k"]), round(t011["net_not_written_per_1k"])) == (1466, 775),
          f"R${t011['net_all_per_1k']:,.0f} all hits / R${t011['net_not_written_per_1k']:,.0f} "
          "not yet written (threshold 0.11; 0.19 checked too)")
    comp = _csv("review_timing_composition.csv").set_index("list").loc["all reviewed orders"]
    not_written = comp["contacts"] * (comp["never_delivered_per_100"]
                                      + comp["delivered_reachable_per_100"]) / 100
    expect = {
        "net_all_per_1k": 1000 * (comp["hits"] * c["low_review_cost"]
                                  - comp["contacts"] * c["outreach_cost"]) / comp["orders"],
        "net_not_written_per_1k": 1000 * (not_written * c["low_review_cost"]
                                          - comp["contacts"] * c["outreach_cost"]) / comp["orders"],
    }
    ten = dec.loc["slides_ten_a_day"]
    check("the ten-a-day list is worth R$853 (all hits) / R$433 (not yet written) per 1,000 reviewed orders",
          all(abs(ten[k] - v) < 0.01 for k, v in expect.items())
          and (round(ten["net_all_per_1k"]), round(ten["net_not_written_per_1k"])) == (853, 433),
          f"R${ten['net_all_per_1k']:,.0f} / R${ten['net_not_written_per_1k']:,.0f}")
    val = _csv("customer_value.csv")
    n_reviewed = len(_csv("predictions_p2_test.csv"))
    formula = all(np.allclose(val[f"net_e{int(100 * e)}"],
                              1000 * (e * val["hits"] * c["low_review_cost"]
                                      - val["contacts"] * c["outreach_cost"]) / n_reviewed)
                  for e in (1.0, 0.5, 0.25))
    formula &= np.allclose(val["breakeven_share"],
                           c["outreach_cost"] / (c["low_review_cost"] * val["precision"]))
    check("every reais figure is per 1,000 reviewed orders and uses only the assumed costs",
          formula, f"n = {n_reviewed:,} reviewed orders; R${c['outreach_cost']:.0f} a contact, "
          f"R${c['low_review_cost']:.0f} a 1-2 star rating (config.COSTS)")
except (FileNotFoundError, KeyError, IndexError) as exc:
    check("customer value tables exist (run src/customer_view.py)", False, repr(exc))

# The pilot: one arm per contact, sized with the same formula as before.
try:
    import pilot_design as PD

    arms = _csv("customer_pilot_arms.csv").set_index("arm")
    check("every pilot arm's size per arm comes from pilot_design.n_per_arm",
          all(r[f"n_per_arm_{p}"] == PD.n_per_arm(r["control_rate"], r["treated_rate"], power=p / 100)
              for _, r in arms.iterrows() for p in (80, 90)),
          ", ".join(f"{a} {r['n_per_arm_80']:.0f}" for a, r in arms.iterrows()))
    check("pilot weeks = arms x size per arm / orders a day / 7, rounded up",
          all(r[f"weeks_{p}"] == math.ceil(round(r["arms"] * r[f"n_per_arm_{p}"] / r["per_day"] / 7, 9))
              for _, r in arms.iterrows() for p in (80, 90)))
    design = _csv("pilot_design.csv")
    slide_row = design[(design["k_per_day"] == 10)
                       & np.isclose(design["relative_reduction"], 0.25)].iloc[0]
    s11 = arms.loc["slide_design"]
    check("the slide-11 design equals pilot_design.csv: 191 per arm, six weeks",
          s11["n_per_arm_80"] == slide_row["n_per_arm_80"] == 191
          and s11["weeks_80"] == slide_row["weeks_80"] == 6
          and np.isclose(s11["control_rate"], slide_row["control_rate"]))
except (FileNotFoundError, KeyError, IndexError) as exc:
    check("pilot arm table exists (run src/customer_view.py)", False, repr(exc))

# Box 1: one customer's order, from the story table and from the per-order table.
try:
    story = _csv("customer_story.csv").iloc[0]
    journeys = _csv("customer_journeys_test.csv")
    oid = story["order_id"]
    check("Box 1's order: 23rd of 238 at checkout, 1st of 244 at delivery, 8 of its day's top ten "
          "reachable only on the promised date",
          oid == _csv("story_order.csv").iloc[0]["order_id"] and oid.startswith("e1320973")
          and (story["late_rank_on_day"], story["delivered_orders_on_day"]) == (23, 238)
          and (story["low_rank_on_day"], story["reviewed_orders_on_day"]) == (1, 244)
          and story["top10_hits"] == 8 and story["top10_hits_reachable_on_promise"] == 8
          and story["top10_hits_reachable_at_delivery"] == 0
          and story["requoted_date"] == "2018-08-12"
          and story["reach_class"] == "promised-date message only",
          f"{oid[:8]}, quoted {story['requoted_date']}, {story['reach_class']}")
    row = journeys.set_index("order_id").loc[oid]
    day = journeys[pd.to_datetime(journeys["purchase"]).dt.date
                   == pd.Timestamp(story["purchase"]).date()]
    top = day[day["low_rank_purchase_day"] <= 10]
    hits = top["low"].astype(bool)
    check("the story row agrees with the per-order journey table the application reads",
          row["late_rank_on_day"] == story["late_rank_on_day"]
          and row["delivered_on_day"] == story["delivered_orders_on_day"]
          and row["low_rank_purchase_day"] == story["low_rank_on_day"]
          and row["reviewed_on_purchase_day"] == story["reviewed_orders_on_day"] == len(day)
          and row["reach_class"] == story["reach_class"]
          and str(row["requoted_date"])[:10] == story["requoted_date"]
          and int(hits.sum()) == story["top10_hits"]
          and int((hits & top["promise_reach"].astype(bool)).sum())
          == story["top10_hits_reachable_on_promise"]
          and int((hits & top["delivery_reach"].astype(bool)).sum())
          == story["top10_hits_reachable_at_delivery"],
          f"{len(day)} reviewed orders that day; top ten: {int(hits.sum())} rated 1-2 stars")
except (FileNotFoundError, KeyError, IndexError) as exc:
    check("story tables exist (run src/customer_view.py)", False, repr(exc))

# Where every number on the locked slides comes from (report Appendix P).
try:
    prov = _csv("slide_provenance.csv")
    numeric = prov.loc[prov["matches"].notna(), "matches"]
    numeric = numeric.map(lambda v: v if isinstance(v, (bool, np.bool_))
                          else str(v).strip().lower() == "true")
    check("every checked number on the locked slides matches its recomputation",
          len(numeric) > 0 and bool(numeric.all()),
          f"{int(numeric.sum())} of {len(numeric)} numeric rows match; {len(prov)} rows in all")
    from_costs = prov[prov["source"].fillna("").str.contains("config.COSTS", regex=False)]
    check("every slide number sourced from config.COSTS is labelled an assumption",
          len(from_costs) > 0 and bool((from_costs["kind"] == "assumed").all()),
          f"{len(from_costs)} rows")
    kinds = {"observed", "assumed", "derived", "planned", "literature", "no data source", "wording"}
    check("the provenance table covers all 13 slides, every row of a named kind",
          set(prov["slide"]) == set(range(1, 14)) and set(prov["kind"]) <= kinds,
          ", ".join(f"{k} {v}" for k, v in prov["kind"].value_counts().items()))
except (FileNotFoundError, KeyError) as exc:
    check("slide_provenance.csv exists (run src/slide_provenance.py)", False, repr(exc))

# Wording the revision retired. docs/presentation_script.md is left out on purpose:
# it records the talk as delivered.
_sep = r"[\s~]+"
LINT = {
    "'per 1,000 delivered orders' (values are per 1,000 reviewed orders)":
        rf"per{_sep}(?:1\{{,\}}000|1\\,000|1,000|1000|thousand){_sep}delivered{_sep}orders",
    "'never contacted anyone' / 'was ever contacted' / 'nobody in this history' (no data source)":
        rf"never{_sep}contacted{_sep}anyone|was{_sep}ever{_sep}contacted"
        rf"|nobody{_sep}in{_sep}this{_sep}(?:history|data)",
    "the test window written '2018-07-01 to 2018-08-31'": rf"2018-07-01{_sep}to{_sep}2018-08-31",
    "'would-be'": r"would-be",
    "'Group 10' (the team is Team 10)": rf"Group{_sep}10\b",
}
LINT_FILES = (sorted((PROJECT_ROOT / "report" / "sections").glob("p3_*.tex"))
              + [PROJECT_ROOT / "README.md", PROJECT_ROOT / "build_submission.py"]
              + [PROJECT_ROOT / "docs" / f for f in
                 ("feedback_responses.md", "requirements_checklist.md", "README.md")]
              + sorted((PROJECT_ROOT / "deployment").glob("*.py"))
              + [PROJECT_ROOT / "deployment" / "README.md", PROJECT_ROOT / "dashboard" / "app.py"]
              + sorted((PROJECT_ROOT / "src").glob("*.py")))
# A phrase in the app may be split over adjacent string literals ("... nobody in this "
# / "data ..."): blank out the quotes of such joins, keeping every character position
# (and so every line number) where it was.
_JOIN = re.compile(r"""(["'])(\s*\n\s*)([rRbBfFuU]{0,2})(["'])""")
_texts = {f: _JOIN.sub(lambda m: " " + m.group(2) + " " * (len(m.group(3)) + 1),
                       f.read_text("utf-8", errors="replace"))
          for f in LINT_FILES if f.exists()}
for label, pattern in LINT.items():
    rx = re.compile(pattern, re.IGNORECASE)
    found = [f"{f.relative_to(PROJECT_ROOT).as_posix()}:{text.count(chr(10), 0, m.start()) + 1}"
             for f, text in _texts.items() for m in rx.finditer(text)]
    check(f"retired wording absent from the report, README, docs, app and src: {label}",
          not found, ", ".join(found))

# The slides are submitted exactly as locked on 4 November 2026.
LOCKED_SLIDES = {
    "Team10_Phase3_slides.pptx": "189cd5afafb7d2dd77b5ff8fc9c759fda0b197a2c5a2557dd43ffd736536cfe8",
    "Team10_Phase3_slides.pdf": "f0f159b4fd08f327a1ff5f9c5f2f175c23f125e1467b831366e5f8dde2f77395",
}
for name, digest in LOCKED_SLIDES.items():
    path = PROJECT_ROOT / "docs" / name
    got = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
    check(f"docs/{name} is byte-identical to the deck locked on 4 November", got == digest,
          f"sha256 {got[:12]}...")

# What unhappy customers wrote: the keyword dictionary was checked on a held-out sample.
try:
    prec = _csv("customer_complaint_precision.csv")
    check("the complaint topics were checked on held-out comments: 7 topics x 25, precision 80-96%",
          len(prec) == 7 and bool((prec["read"] == 25).all())
          and bool(prec["precision_pct"].between(80, 96).all()),
          f"{prec['precision_pct'].min():.0f}-{prec['precision_pct'].max():.0f}%")
    labels = _csv("customer_topic_labels.csv")
    per_topic = labels.groupby("topic")["correct"].agg(["size", "sum"])
    p = prec.set_index("topic").loc[per_topic.index]
    check("the precision table recomputes from the 175 committed labels",
          len(labels) == 175 and set(per_topic.index) == set(prec["topic"])
          and (per_topic["size"] == p["read"]).all() and (per_topic["sum"] == p["correct"]).all()
          and np.allclose(100 * per_topic["sum"] / per_topic["size"], p["precision_pct"]))
except (FileNotFoundError, KeyError) as exc:
    check("complaint tables exist (run src/customer_view.py)", False, repr(exc))

# The Replay tab starts blind: its download must not carry what happened.
OUTCOME_COLUMNS = {
    "actual_low_review", "actual_days", "actual_late", "written_before_arrival", "outcome",
    "low", "days_late", "delivered", "answered", "order_status", "review_score",
    "review_title", "review_message", "overdue_eod", "promise_reach", "delivery_reach",
    "reach_class", "review_predates_order",
    # the same outcomes under the labels the revealed table shows
    "actual review", "review written", "actual days", "delivery", "stars",
    "what the customer experienced", "could still be reached by",
}
try:
    # Imported under the streamlit stub of section 7 (which passes st.fragment and the
    # cache decorators through), so nothing renders and nothing is cached.
    import replay as RP

    if not hasattr(RP, "blind_columns"):
        raise AttributeError("deployment/replay.py has no blind_columns() yet")
    frame = None
    try:
        loaded = RP.load_replay()
        loaded = loaded if isinstance(loaded, tuple) else (loaded,)
        frame = next(x for x in loaded if isinstance(x, pd.DataFrame))
    except Exception:  # noqa: BLE001 - fall back to every column the replay can read
        frame = pd.concat([_csv(f).head(50) for f in
                           ("predictions_p2_test.csv", "predictions_p1a_test.csv",
                            "predictions_p1b_test.csv", "customer_journeys_test.csv")], axis=1)
        frame = frame.loc[:, ~frame.columns.duplicated()]
    for arg in ((frame,), (list(frame.columns),), ()):
        try:
            kept = RP.blind_columns(*arg)
            break
        except TypeError:
            continue
    else:
        raise TypeError("blind_columns() accepts neither a frame, a column list nor no argument")
    kept = list(kept.columns) if isinstance(kept, pd.DataFrame) else list(kept)
    leaked = sorted(set(kept) & OUTCOME_COLUMNS)
    check("replay.blind_columns() keeps every outcome out of the blind download",
          bool(kept) and not leaked,
          f"keeps {len(kept)} columns; outcome columns kept: {', '.join(leaked) or 'none'}")
except Exception as exc:  # noqa: BLE001
    check("replay.blind_columns() keeps every outcome out of the blind download", False,
          f"{type(exc).__name__}: {exc}")


# ===========================================================================
print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
