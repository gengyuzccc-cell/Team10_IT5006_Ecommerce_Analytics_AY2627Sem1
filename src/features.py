"""
Feature engineering for the Olist order table.

Leakage policy
--------------
Prediction time for every candidate problem is **the moment the order is
placed**. That single decision determines what is admissible:

* `order_delivered_customer_date` and `order_delivered_carrier_date` *define*
  the delivery outcome, so they can never be predictors of it.
* `review_score` and its timestamps are written after the customer has lived
  through the order.
* `order_approved_at` and `shipping_limit_date` are only known once fulfilment
  has started, which is after checkout. The brief calls approval "usually safe";
  we exclude it anyway, because a model meant to run at checkout would not have
  it, and the cost of including it is unmeasurable optimism.
* `order_estimated_delivery_date` **is** admissible: the customer is shown that
  promise at checkout, so it exists before the outcome and is in fact the single
  most useful feature available.

`assert_no_leakage()` is the gate every modelling cell must pass, checked
against `config.POST_OUTCOME_FORBIDDEN`.

The split in `make_splits()` is chronological *and* grouped on
`customer_unique_id`. The brief warns that `customer_id` is regenerated per
order and is therefore not a person; splitting on rows would let the same buyer
appear on both sides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DELIVERY_OUTCOME_FEATURES,
    LEAKAGE_REGIMES,
    LOW_REVIEW_MAX_SCORE,
    MIN_HISTORY,
    REPEAT_HORIZON_DAYS,
    TRAIN_END,
    VAL_END,
)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """LEAKAGE-SAFE. Everything here is known the instant the order is placed."""
    out = df.copy()
    ts = out["order_purchase_timestamp"]
    out["purchase_year"] = ts.dt.year
    out["purchase_month"] = ts.dt.month
    out["purchase_dow"] = ts.dt.dayofweek
    out["purchase_hour"] = ts.dt.hour
    out["purchase_week"] = ts.dt.isocalendar().week.astype(int)
    out["is_weekend"] = out["purchase_dow"].isin([5, 6])
    out["is_business_hours"] = out["purchase_hour"].between(9, 18)
    # Black Friday falls in late November and is the platform's largest spike.
    out["is_nov_peak"] = (out["purchase_month"] == 11) & (ts.dt.day >= 20)
    return out


# ---------------------------------------------------------------------------
# Order composition
# ---------------------------------------------------------------------------
def add_basket(df: pd.DataFrame) -> pd.DataFrame:
    """LEAKAGE-SAFE. What the customer put in the basket and what it cost."""
    out = df.copy()
    price = out["total_price"].replace(0, np.nan)
    out["freight_ratio"] = out["total_freight"] / price
    out["price_per_item"] = out["total_price"] / out["n_items"].replace(0, np.nan)
    out["weight_per_item"] = out["total_weight_g"] / out["n_items"].replace(0, np.nan)
    out["is_multi_seller"] = out["n_sellers"] > 1
    out["log_price"] = np.log1p(out["total_price"])
    out["log_weight"] = np.log1p(out["total_weight_g"])
    out["log_distance"] = np.log1p(out["distance_km"])
    return out


# ---------------------------------------------------------------------------
# The promise made at checkout
# ---------------------------------------------------------------------------
def add_promise(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """LEAKAGE-SAFE. The delivery window quoted to the customer at checkout.

    `estimated_days` is shown on the order confirmation, so it precedes the
    outcome. It also encodes whatever Olist's own routing model knew about the
    shipment, which is why it dominates the feature importance.

    `threshold` is the cut for `long_promise`. Left as None it is the median of
    the frame, which is what Phase 1 did and what the analysis path still does.
    Serving MUST pass `config.PROMISE_MEDIAN_DAYS` instead: the median of a
    one-row frame is that row, so a deployed model would receive a flag that is
    always False - a feature computed one way in training and another in
    production, which is exactly the skew this module exists to prevent.
    """
    out = df.copy()
    out["estimated_days"] = out["estimated_days"].astype(float)
    cut = out["estimated_days"].median() if threshold is None else threshold
    out["long_promise"] = out["estimated_days"] > cut
    return out


# ---------------------------------------------------------------------------
# Historical performance (Phase 2)
# ---------------------------------------------------------------------------
# A seller's track record is the strongest thing a marketplace knows about an
# order at checkout, and it is the classic transactional feature. It is also
# the classic way to leak, so the construction below is deliberate on one point
# that is easy to get wrong:
#
#   "Past" means the earlier order had **closed** before this one was placed -
#   not merely that it was placed earlier.
#
# A parcel bought last week that has not yet arrived tells you nothing today;
# its lead time does not exist yet. Ordering the history by purchase time would
# quietly let each order average in outcomes that had not happened, which is
# Kaufman et al.'s (2012) "illegitimate feature" wearing a convincing disguise.
# Every aggregate here is therefore an as-of join against the moment the earlier
# order's outcome became observable: order_delivered_customer_date for delivery,
# review_creation_date for reviews.
#
# Two further guards:
#   * MIN_HISTORY - a mean over one previous order is noise, not a rate, so
#     below the floor the value is NaN and the fallback chain takes over.
#   * Cold start is a feature, not an error. hist_seller_n carries how many past
#     orders the estimate rests on, so a model can learn to discount a seller it
#     has barely seen instead of being handed a silent imputation.


def _asof_mean(events: pd.DataFrame, queries: pd.DataFrame, key: str,
               value: str, min_history: int = MIN_HISTORY):
    """Expanding mean of `value` per `key`, over events closed before each query.

    `events` needs columns [key, closed_at, value]; `queries` needs [key,
    asked_at] and carries the caller's index. Returns (mean, n) aligned to that
    index. `allow_exact_matches=False` keeps the boundary strict: an outcome
    landing at the very instant a new order is placed is still not knowable.
    """
    ev = events[[key, "closed_at", value]].dropna().sort_values("closed_at")
    grp = ev.groupby(key, sort=False)[value]
    ev = ev.assign(_cum=grp.cumsum(), _n=grp.cumcount() + 1)
    ev = ev[[key, "closed_at", "_cum", "_n"]]

    q = queries[[key, "asked_at"]].reset_index().sort_values("asked_at")
    m = pd.merge_asof(q, ev, left_on="asked_at", right_on="closed_at", by=key,
                      direction="backward", allow_exact_matches=False)
    m = m.set_index("index").reindex(queries.index)

    n = m["_n"].fillna(0)
    mean = (m["_cum"] / m["_n"]).where(n >= min_history)
    return mean, n


def _asof_rolling(events: pd.DataFrame, queries: pd.DataFrame, value: str,
                  window: str, min_history: int = MIN_HISTORY) -> pd.Series:
    """Trailing-window mean of `value` across the whole platform, as of each query.

    An expanding mean would be the wrong instrument here: lead time halves
    across the window, so an average over all history says 13 days while the
    current month says 8. Operations knows its own recent performance, so a
    trailing window is both the more useful feature and the more honest one.
    """
    ev = (events[["closed_at", value]].dropna()
          .sort_values("closed_at").set_index("closed_at"))
    roll = ev[value].rolling(window)
    stat = pd.DataFrame({"_mean": roll.mean(), "_n": roll.count()}).reset_index()

    q = queries[["asked_at"]].reset_index().sort_values("asked_at")
    m = pd.merge_asof(q, stat, left_on="asked_at", right_on="closed_at",
                      direction="backward", allow_exact_matches=False)
    m = m.set_index("index").reindex(queries.index)
    return m["_mean"].where(m["_n"].fillna(0) >= min_history)


def add_history(df: pd.DataFrame, window: str = "28D") -> pd.DataFrame:
    """LEAKAGE-SAFE. Seller, route, category and platform track records.

    Every value is computed from outcomes observable strictly before the order
    was placed, so the whole block is admissible in the `at_checkout` regime.
    Specific estimates fall back to broader ones when a seller is new:

        seller -> route -> seller state -> platform (trailing window)
    """
    out = df.copy()
    out["route"] = (out["lead_seller_state"].astype("string") + ">"
                    + out["customer_state"].astype("string"))

    # -- the two event streams, each stamped with the moment it became known --
    delivery = pd.DataFrame({
        "closed_at": out["order_delivered_customer_date"],
        "delivery_days": out["delivery_days"],
        "is_late": out["is_late"],
        "lead_seller_id": out["lead_seller_id"],
        "route": out["route"],
        "lead_seller_state": out["lead_seller_state"],
        "lead_category": out["lead_category"],
    })
    review = pd.DataFrame({
        "closed_at": out["review_creation_date"],
        "is_low": ((out["review_score"] <= LOW_REVIEW_MAX_SCORE).astype(float)
                   .where(out["review_score"].notna())),
        "lead_seller_id": out["lead_seller_id"],
        "lead_category": out["lead_category"],
    })
    q = pd.DataFrame({"asked_at": out["order_purchase_timestamp"],
                      "lead_seller_id": out["lead_seller_id"],
                      "route": out["route"],
                      "lead_seller_state": out["lead_seller_state"],
                      "lead_category": out["lead_category"]}, index=out.index)

    # -- platform level: the fallback of last resort, and a drift tracker -----
    plat_days = _asof_rolling(delivery, q, "delivery_days", window)
    plat_late = _asof_rolling(delivery, q, "is_late", window)

    # -- seller, route, state, category --------------------------------------
    s_days, s_n = _asof_mean(delivery, q, "lead_seller_id", "delivery_days")
    s_late, _ = _asof_mean(delivery, q, "lead_seller_id", "is_late")
    r_days, _ = _asof_mean(delivery, q, "route", "delivery_days")
    st_days, _ = _asof_mean(delivery, q, "lead_seller_state", "delivery_days")
    st_late, _ = _asof_mean(delivery, q, "lead_seller_state", "is_late")
    s_low, _ = _asof_mean(review, q, "lead_seller_id", "is_low")
    c_low, _ = _asof_mean(review, q, "lead_category", "is_low")

    out["hist_platform_lead_days"] = plat_days
    out["hist_platform_late_rate"] = plat_late
    out["hist_route_lead_days"] = r_days.fillna(st_days).fillna(plat_days)
    out["hist_category_low_review_rate"] = c_low
    out["hist_seller_lead_days"] = s_days.fillna(r_days).fillna(st_days).fillna(plat_days)
    out["hist_seller_late_rate"] = s_late.fillna(st_late).fillna(plat_late)
    out["hist_seller_low_review_rate"] = s_low.fillna(c_low)
    out["hist_seller_n"] = s_n
    return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate 1 (delivery, dual framing), 2 (review) and 3 (repeat purchase).

    Targets are NaN wherever the outcome is genuinely unobservable - an order
    that never arrived, a customer who left no review, or an order too close to
    the end of the data for a 90-day repeat window to have closed. None of these
    is defaulted to a negative label, which would manufacture signal.
    """
    out = df.copy()

    # -- Candidate 1: delivery performance, framed twice
    out["target_delivery_days"] = out["delivery_days"]
    out["target_is_late"] = out["is_late"]

    # -- Candidate 2: low satisfaction
    out["target_low_review"] = np.where(
        out["review_score"].notna(),
        (out["review_score"] <= LOW_REVIEW_MAX_SCORE).astype(float),
        np.nan,
    )

    # -- Candidate 3: does this buyer come back within the horizon?
    ts = out["order_purchase_timestamp"]
    nxt = (
        out.sort_values(["customer_unique_id", "order_purchase_timestamp"])
        .groupby("customer_unique_id")["order_purchase_timestamp"]
        .shift(-1)
        .reindex(out.index)
    )
    gap_days = (nxt - ts).dt.total_seconds() / 86400
    observable = (out["order_purchase_timestamp"].max() - ts).dt.days >= REPEAT_HORIZON_DAYS
    out["days_to_next_order"] = gap_days
    out["repeat_is_censored"] = ~observable
    out["target_repeat"] = np.where(
        observable, (gap_days <= REPEAT_HORIZON_DAYS).fillna(False).astype(float), np.nan
    )
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, with_history: bool = False) -> pd.DataFrame:
    """Run every builder in dependency order.

    `with_history` defaults to False so src/report_stats.py, the Phase 1
    notebooks and the deployed EDA tabs keep reproducing the submitted Phase 1
    numbers column for column. Phase 2 passes True.
    """
    out = add_calendar(df)
    out = add_basket(out)
    out = add_promise(out)
    out = add_targets(out)
    if with_history:
        out = add_history(out)
    return out


# ---------------------------------------------------------------------------
# Feature sets a model is allowed to see
# ---------------------------------------------------------------------------
# Deliberately free of exact duplicates. The first draft carried total_price
# *and* log_price *and* total_payment (which is price + freight), plus
# distance_km beside log_distance and purchase_week beside purchase_month.
# Ordinary least squares on that design matrix is rank-deficient and its
# coefficients explode; keeping one representative of each quantity is what
# makes a plain linear baseline meaningful at all.
NUMERIC_FEATURES = [
    "estimated_days",
    "n_items", "n_sellers",
    "log_price", "price_per_item",
    "total_freight", "freight_ratio",
    "log_weight", "max_volume_cm3", "total_photos",
    "log_distance",
    "max_installments",
    "purchase_month", "purchase_dow", "purchase_hour",
]
BOOLEAN_FEATURES = [
    "is_weekend", "is_business_hours", "is_nov_peak",
    "is_multi_seller", "same_state", "long_promise",
]
CATEGORICAL_FEATURES = [
    "customer_state", "seller_region", "lead_category", "lead_payment_type",
]

# purchase_year is deliberately absent: the split is chronological, so the test
# period contains a year the model never saw during training and a linear term
# on it extrapolates blindly.

MODEL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

# The at-delivery regime may additionally see how the delivery actually went.
REVIEW_MODEL_FEATURES = MODEL_FEATURES + DELIVERY_OUTCOME_FEATURES

# ---------------------------------------------------------------------------
# Phase 2 feature sets
# ---------------------------------------------------------------------------
# Track records, every one computed from outcomes that had closed before the
# order was placed (see add_history), so all of them are admissible at checkout.
HISTORY_FEATURES = [
    "hist_seller_lead_days",
    "hist_seller_late_rate",
    "hist_seller_low_review_rate",
    "hist_seller_n",
    "hist_route_lead_days",
    "hist_category_low_review_rate",
    "hist_platform_lead_days",
    "hist_platform_late_rate",
]

# Phase 2 models read these; the Phase 1 sets above stay frozen so the earlier
# report keeps reproducing.
CHECKOUT_FEATURES = MODEL_FEATURES + HISTORY_FEATURES
DELIVERY_FEATURES = REVIEW_MODEL_FEATURES + HISTORY_FEATURES

NUMERIC_CHECKOUT = NUMERIC_FEATURES + BOOLEAN_FEATURES + HISTORY_FEATURES
NUMERIC_DELIVERY = NUMERIC_CHECKOUT + DELIVERY_OUTCOME_FEATURES


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def assert_no_leakage(feature_names, regime: str = "at_checkout") -> None:
    """Refuse any feature set containing a column unavailable at `regime`.

    `regime` is the moment the model would actually run - see the module
    docstring. Passing the wrong regime is itself the mistake this guard exists
    to catch, so an unknown name is an error rather than a silent pass.
    """
    if regime not in LEAKAGE_REGIMES:
        raise KeyError(
            f"Unknown leakage regime {regime!r}; expected one of "
            f"{sorted(LEAKAGE_REGIMES)}"
        )
    offenders = sorted(set(feature_names) & LEAKAGE_REGIMES[regime])
    if offenders:
        raise ValueError(
            f"At {regime.replace('_', ' ')} these columns are not yet "
            "observable, so they cannot be predictors: " + ", ".join(offenders)
        )


def make_splits(df: pd.DataFrame) -> dict[str, object]:
    """Chronological split, then remove customers who straddle train and test.

    Chronology alone is not enough. 2,997 buyers placed more than one order, so
    a purely time-based cut can still put one of their orders in training and
    another in test; the model would then be graded partly on a person it has
    already seen. This is the concrete form of the brief's customer_id versus
    customer_unique_id warning.
    """
    ts = df["order_purchase_timestamp"]
    train = df[ts <= TRAIN_END]
    val = df[(ts > TRAIN_END) & (ts <= VAL_END)]
    test = df[ts > VAL_END]

    straddlers = set(train["customer_unique_id"]) & set(test["customer_unique_id"])
    test_clean = test[~test["customer_unique_id"].isin(straddlers)]

    return {
        "train": train,
        "val": val,
        "test": test,
        "test_unseen_customers": test_clean,
        "n_straddling_customers": len(straddlers),
    }
