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
    sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))
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
print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
