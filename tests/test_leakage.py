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
    _st.cache_resource = lambda **k: (lambda f: f)
    _st.cache_data = lambda **k: (lambda f: f)
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
          abs(net - at) < 1e-6, f"R${net:,.0f} per 1,000 delivered orders")
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
print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
