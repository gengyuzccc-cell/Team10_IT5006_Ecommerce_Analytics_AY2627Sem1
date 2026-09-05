"""
Project-wide configuration for the IT5006 Phase 1 Olist analysis.

Everything a notebook, the dashboard or the report might quote as "a decision we
made" lives here, so there is exactly one authoritative copy of each constant.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "olist"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# The nine tables named in the project brief.
TABLES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Row counts asserted on load, so a changed source file fails loudly rather than
# silently altering every number in the report.
EXPECTED_ROWS = {
    "orders": 99_441,
    "order_items": 112_650,
    "order_payments": 103_886,
    "order_reviews": 99_224,
    "customers": 99_441,
    "sellers": 3_095,
    "products": 32_951,
    "geolocation": 1_000_163,
    "category_translation": 71,
}

# --------------------------------------------------------------------------
# Analysis window
# --------------------------------------------------------------------------
# The raw file spans 2016-09 to 2018-10, but the first four months are a
# pilot period with a handful of orders per month, and the final weeks are
# truncated mid-cycle. We model the fully-operational window and report the
# ramp-up separately as a coverage exhibit.
ANALYSIS_START = pd.Timestamp("2017-01-01")
ANALYSIS_END = pd.Timestamp("2018-08-31")

# --------------------------------------------------------------------------
# Chronological split
# --------------------------------------------------------------------------
TRAIN_END = pd.Timestamp("2018-04-30")
VAL_END = pd.Timestamp("2018-06-30")
# Test = 2018-07-01 .. ANALYSIS_END

# --------------------------------------------------------------------------
# Target definitions
# --------------------------------------------------------------------------
# Candidate 1, framing B: an order is late if it reaches the customer after the
# delivery date promised at checkout.
# Candidate 2: a review of 1 or 2 stars counts as low satisfaction.
LOW_REVIEW_MAX_SCORE = 2
# Candidate 3: did the customer place another order within this many days?
REPEAT_HORIZON_DAYS = 30

# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------
# Leakage is not one rule but one rule *per prediction moment*, because what is
# already known depends on when the model runs.
#
#   at_checkout - the order has just been placed. Nothing about fulfilment or
#                 the customer's eventual opinion exists yet. Used by the
#                 delivery candidates and by repeat-purchase.
#   at_delivery - the parcel has arrived and we are predicting the review that
#                 has not been written yet. The delivery outcome is now history
#                 and is admissible; the brief itself lists "delivery
#                 performance" among the features for this problem. Only the
#                 review and its timestamps remain forbidden.
#
# assert_no_leakage(features, regime) enforces the right set.
_REVIEW_COLS = {
    "review_score", "review_creation_date", "review_answer_timestamp",
    "is_low_review", "target_low_review",
}
_DELIVERY_COLS = {
    "order_delivered_customer_date", "order_delivered_carrier_date",
    "delivery_days", "is_late", "delay_vs_estimate_days",
    "target_delivery_days", "target_is_late",
}
_FULFILMENT_COLS = {
    # only known once fulfilment has started, i.e. after checkout
    "order_approved_at", "approval_hours", "shipping_limit_date", "order_status",
}

LEAKAGE_REGIMES = {
    "at_checkout": _REVIEW_COLS | _DELIVERY_COLS | _FULFILMENT_COLS,
    "at_delivery": _REVIEW_COLS | {"order_status"},
}

# Columns describing how delivery actually went. Forbidden at checkout,
# admissible once the parcel has arrived.
DELIVERY_OUTCOME_FEATURES = ["delivery_days", "delay_vs_estimate_days", "is_late"]

# --------------------------------------------------------------------------
# Plot theme
# --------------------------------------------------------------------------
FIG_WIDTH = 7.0          # inches; matches \linewidth at 1in margins on A4
FIG_HEIGHT = 4.0
PALETTE = {
    "primary": "#2F6F9F",
    "accent": "#C1666B",
    "gold": "#C08B2E",
    "green": "#5B8C5A",
    "muted": "#8C8C8C",
    "grid": "#DDDDDD",
    "late": "#C1666B",
    "ontime": "#2F6F9F",
}

# Brazilian regions, used to collapse 27 states into an interpretable grouping.
STATE_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North",
    "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast",
    "PB": "Northeast", "PE": "Northeast", "PI": "Northeast", "RN": "Northeast",
    "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West",
    "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}


# ===========================================================================
# PHASE 2 — modelling, evaluation and deployment
# ===========================================================================
# Phase 1 constants above are frozen: src/report_stats.py must keep reproducing
# the submitted Phase 1 numbers exactly. Everything Phase 2 adds lives below.

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Cross-validation protocol
# --------------------------------------------------------------------------
# The target drifts (lead time halves across the window), so k-fold on shuffled
# rows lets a model learn from the future. TimeSeriesSplit on the training
# period is the selection criterion; StratifiedGroupKFold is computed alongside
# purely as a contrast exhibit, never to choose a model.
CV_SPLITS = 5
SEARCH_ITER = 30           # RandomizedSearchCV draws per model

# How many of the out-of-sample windows the selection rule votes on, counting
# back from deployment. There are six in total - five expanding CV folds plus the
# validation period - and using all of them would give 2017 five sixths of the
# vote on a question about mid-2018.
#
# The monthly series is what fixes the number. Mean lead time runs 12-17 days
# across the 2017 folds, 8-12 days from February 2018, and 8.3 days in the test
# window; performance in early 2017 is simply not evidence about performance
# after the platform doubled its fulfilment speed. Three windows covers roughly
# February to June 2018 - about 32,000 orders - which is recent enough to
# resemble deployment and wide enough that the validation period's two
# opposite-character months (8.2% late in May, 1.4% in June) cannot decide the
# outcome by themselves. Both failure modes are real and both were observed.
SELECTION_WINDOWS = 3
# The search parallelises over candidates; each worker holds its own copy of a
# ~60 MB design matrix plus its fitted model, so the estimators themselves run
# single-threaded. Nested n_jobs=-1 on both levels oversubscribes and exhausts
# memory outright. Six workers leaves headroom for anything else on the machine -
# eight was enough to run out of memory mid-search with the dashboard also open.
N_JOBS = 6
SEARCH_SCORING = {
    "regression": "neg_mean_absolute_error",
    "classification": "average_precision",   # PR-AUC: the right target under imbalance
}

# --------------------------------------------------------------------------
# Imbalance and threshold policy
# --------------------------------------------------------------------------
# Measured in notebook 08: class_weight="balanced" leaves probabilities badly
# miscalibrated (Brier 0.130, mean predicted risk 0.327 against an actual 0.110)
# without improving discrimination. We therefore fit on the observed prior and
# move the decision threshold instead - exactly what van den Goorbergh et al.
# (2022) recommend, and what the Phase 1 literature review promised to test.
CLASS_WEIGHT_POLICY = None          # None = fit on the observed prior
THRESHOLD_OBJECTIVE = "f1"          # tuned on the validation period only
THRESHOLD_GRID = (0.05, 0.95, 0.01)

# --------------------------------------------------------------------------
# Illustrative decision costs
# --------------------------------------------------------------------------
# ASSUMPTIONS, NOT DATA. Olist publishes no cost figures, so these are stated
# openly and every conclusion drawn from them is accompanied by a sensitivity
# sweep. Units are Brazilian reais per order.
COSTS = {
    # Problem 2: contacting a customer who was never going to complain wastes
    # an agent's time; missing one that does costs a 1-2 star review, which
    # Deshpande et al. (2023) trace through to lost seller sales.
    "outreach_cost": 5.0,
    "low_review_cost": 40.0,
    # Problem 1: expediting an order that would have arrived on time wastes
    # freight; letting a late one through costs the same review damage.
    "expedite_cost": 12.0,
    "late_delivery_cost": 40.0,
}

# --------------------------------------------------------------------------
# Pre-registered expectation bands
# --------------------------------------------------------------------------
# Written down BEFORE the Phase 2 models were tuned, from the Phase 1 probes.
# They serve two purposes: they are the report's success criteria, and they are
# a leak alarm. A result far ABOVE its band is not a triumph, it is a bug -
# adding one forbidden column took the Phase 1 late-delivery model from ROC-AUC
# 0.589 to 1.000. check_bands() in src/evaluate.py enforces them.
EXPECTED_BANDS = {
    "p1a_mae":              (3.3, 4.3),    # lead-time regression, days
    "p1b_balanced_accuracy": (0.58, 0.72),  # late flag, best route
    "p1b_roc_auc":          (0.72, 0.85),
    "p2_balanced_accuracy": (0.63, 0.74),
    "p2_roc_auc":           (0.70, 0.79),
    "p2_pr_auc":            (0.34, 0.48),
}

# Which end of a band is the suspicious one. For a score, implausibly GOOD means
# implausibly high; for an error, it means implausibly low. Getting this backwards
# would point the leak alarm at the shortfalls and wave the leaks through.
LOWER_IS_BETTER = {"p1a_mae"}

# --------------------------------------------------------------------------
# History features
# --------------------------------------------------------------------------
# Expanding-window aggregates over a seller's / route's / category's PAST
# orders. Powerful at checkout and completely safe - provided each order sees
# only orders that closed before it. MIN_HISTORY guards against a mean computed
# from one observation; below it the hierarchical fallback takes over.
MIN_HISTORY = 5
HISTORY_FALLBACK_ORDER = ["seller", "route", "seller_state", "category", "global"]

# --------------------------------------------------------------------------
# Serving constants
# --------------------------------------------------------------------------
# `long_promise` is "is this delivery window longer than typical?", and Phase 1
# implemented it as a comparison against the median of whatever frame it was
# handed. That is correct on the analysis frame and silently wrong on a single
# order: the median of one row is that row, so the flag is always False and the
# deployed model sees a feature it never saw in training. Pinning the threshold
# is what makes the training-time and serving-time definitions the same one.
#
# The value is the median of estimated_days over the analysis window; the frame
# median is asserted against it in tests/test_leakage.py, so the constant cannot
# drift away from the data it came from.
PROMISE_MEDIAN_DAYS = 23.222361111111113
