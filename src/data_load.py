"""
Single entry point for the Olist Brazilian e-commerce data.

Every notebook, the dashboard and the report read through this module, so the
numbers quoted in the PDF and the numbers rendered in the dashboard cannot
drift apart.

The nine source tables are normalised around `orders`. Because several of them
are one-to-many against an order, joining naively would multiply rows and
silently reweight every statistic; `build_orders()` aggregates each child table
to one row per order first, then joins.

Pipeline
--------
    load_table(name)   read one CSV, assert its row count
    load_all()         read all nine
    build_orders()     aggregate + join to one row per order
    load_analysis()    build_orders -> restrict to the analysis window
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ANALYSIS_END,
    ANALYSIS_START,
    DATA_RAW,
    EXPECTED_ROWS,
    STATE_REGION,
    TABLES,
)

ORDER_DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# ---------------------------------------------------------------------------
# The cleaning contract. Section 2.2 of the report is generated from this dict,
# so editing a decision here updates the report's audit table too.
# ---------------------------------------------------------------------------
CLEANING_DECISIONS: dict[str, dict[str, str]] = {
    "one_to_many_joins": {
        "issue": "order_items, order_payments and order_reviews are all "
                 "one-to-many against orders. Joining them directly multiplies "
                 "an order into several rows.",
        "evidence": "112,650 items and 103,886 payment records against 99,441 "
                    "orders; a naive merge yields ~117k rows and silently "
                    "over-weights multi-item orders in every average.",
        "decision": "Aggregate each child table to one row per order first "
                    "(counts, sums, and the attributes of the most expensive "
                    "item), then join. The analysis grain is one order.",
    },
    "customer_key": {
        "issue": "customer_id is regenerated for every order, so it identifies "
                 "an order-customer pair rather than a person.",
        "evidence": "99,441 customer_id values for 96,096 customer_unique_id "
                    "values; 2,997 people ordered more than once, one of them "
                    "17 times.",
        "decision": "Carry customer_unique_id and group on it for repeat "
                    "behaviour and for train/test splitting, so the same person "
                    "cannot appear on both sides of the split.",
    },
    "undelivered_orders": {
        "issue": "Delivery targets are undefined for orders that never "
                 "reached the customer.",
        "evidence": "2.98% of orders have no order_delivered_customer_date; "
                    "status is delivered for 97.0% of rows and the remainder "
                    "are shipped, cancelled, unavailable or still processing.",
        "decision": "Delivery targets are computed only for delivered orders "
                    "with a delivery timestamp; the rest are labelled NaN and "
                    "excluded from those models, never imputed.",
    },
    "review_duplicates": {
        "issue": "review_id is not unique, and some orders carry more than one "
                 "review. Neither column is the key it looks like.",
        "evidence": "99,224 review rows hold only 98,410 distinct review_id "
                    "values - 789 ids are reused across unrelated orders - and "
                    "547 orders carry two or three reviews. A further 768 "
                    "orders have no review row at all.",
        "decision": "Key reviews on order_id, not review_id, and keep the "
                    "earliest review per order by creation date. Orders without "
                    "a review get NaN rather than an imputed score.",
    },
    "review_free_text": {
        "issue": "The two free-text review columns are mostly empty, so any "
                 "text-based feature would be defined for a minority of orders.",
        "evidence": "review_comment_title is present for 11.7% of reviews and "
                    "review_comment_message for 41.3%.",
        "decision": "Use the numeric review_score only. No NLP features are "
                    "engineered; the coverage does not support them.",
    },
    "estimate_granularity": {
        "issue": "The promised delivery date is a date, but the actual delivery "
                 "is a full timestamp, so the two are not measured alike.",
        "evidence": "order_estimated_delivery_date takes 459 distinct values "
                    "over 773 days and its time component is always midnight.",
        "decision": "Treat lateness as delivery strictly after the promised "
                    "date, and read delay_vs_estimate_days as accurate to about "
                    "a day rather than to the hour.",
    },
    "orphan_payment": {
        "issue": "One order has no payment record.",
        "evidence": "order bfbd0f9bdef84302105ad712db648a6c (delivered, "
                    "2016-09-15) appears in orders but not in order_payments.",
        "decision": "Left as NaN by the left join. It falls outside the 2017+ "
                    "analysis window and affects nothing, but the left join is "
                    "deliberate so such gaps stay visible instead of dropping "
                    "the order.",
    },
    "geolocation_duplicates": {
        "issue": "The geolocation table holds one row per captured coordinate, "
                 "not one per postcode, and contains points outside Brazil.",
        "evidence": "1,000,163 rows for roughly 19k distinct zip prefixes.",
        "decision": "Collapse to a median latitude/longitude per zip prefix "
                    "(median resists the stray coordinates) and use it to "
                    "compute customer-seller great-circle distance.",
    },
    "product_missing_attrs": {
        "issue": "A small number of products carry no category or dimensions.",
        "evidence": "610 products (1.9%) have a null category name; weight and "
                    "dimensions are null for 2 products.",
        "decision": "Label the category 'unknown' rather than dropping the "
                    "orders, and let the modelling pipeline's imputer handle "
                    "the numeric gaps so the missingness stays visible.",
    },
    "coverage_ramp": {
        "issue": "The platform's first months are a pilot with negligible "
                 "volume, and the final weeks are truncated mid-cycle.",
        "evidence": "Sep 2016 to Dec 2016 contribute 326 orders in total "
                    "against a 2018 run-rate above 6,000 per month; orders "
                    "after 2018-08-31 are cut off before their reviews land.",
        "decision": "Restrict the analysis window to 2017-01-01 .. 2018-08-31 "
                    "and show the ramp-up separately as a coverage exhibit.",
    },
    "portuguese_categories": {
        "issue": "Product categories are in Portuguese.",
        "evidence": "71 categories, joined via "
                    "product_category_name_translation.csv.",
        "decision": "Translate to English for every chart and table; keep the "
                    "original key for traceability.",
    },
}


# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------
# Slim read plan for the deployed dashboard.
#
# Reading all nine tables in full costs 303 MB of RAM, and most of that sits in
# two places the analysis never touches: geolocation_city/state (redundant with
# the customer and seller tables) and the two free-text review columns, which
# are 88% and 59% empty. Selecting columns alone takes geolocation from 152 MB
# to 24 MB, which is what matters on a 1 GB shared container.
#
# Dtypes are deliberately left alone. Narrowing the coordinates to float32 would
# save a further 12 MB but perturbs the derived great-circle distance by up to a
# metre, and an artifact that is only nearly identical to the analysed data is
# not worth 12 MB. As written, load_all(slim=True) reproduces build_orders()
# exactly, column for column.
SLIM_USECOLS = {
    "geolocation": ["geolocation_zip_code_prefix", "geolocation_lat",
                    "geolocation_lng"],
    "order_reviews": ["review_id", "order_id", "review_score",
                      "review_creation_date", "review_answer_timestamp"],
}
SLIM_DTYPES: dict[str, dict[str, str]] = {}


def load_table(name: str, slim: bool = False) -> pd.DataFrame:
    """Read one source table and assert the row count we documented.

    `slim=True` reads only the columns the order table is built from. Use it for
    the dashboard; leave it False for the notebooks and report statistics, which
    profile columns the model never sees.
    """
    if name not in TABLES:
        raise KeyError(f"Unknown table {name!r}; expected one of {sorted(TABLES)}")

    kwargs: dict = {}
    if name == "orders":
        kwargs["parse_dates"] = ORDER_DATE_COLS
    elif name == "order_reviews":
        kwargs["parse_dates"] = ["review_creation_date", "review_answer_timestamp"]
    elif name == "order_items":
        kwargs["parse_dates"] = ["shipping_limit_date"]

    if slim and name in SLIM_USECOLS:
        kwargs["usecols"] = SLIM_USECOLS[name]
        if name in SLIM_DTYPES:
            kwargs["dtype"] = SLIM_DTYPES[name]

    df = pd.read_csv(DATA_RAW / TABLES[name], **kwargs)
    expected = EXPECTED_ROWS[name]
    assert len(df) == expected, (
        f"{name}: expected {expected:,} rows, got {len(df):,}. The source data "
        "has changed and every downstream number is suspect."
    )
    return df


def load_all(slim: bool = False) -> dict[str, pd.DataFrame]:
    """Read all nine tables."""
    return {name: load_table(name, slim=slim) for name in TABLES}


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------
def zip_centroids(geolocation: pd.DataFrame) -> pd.DataFrame:
    """One median coordinate per zip prefix.

    The raw table is one row per captured point, so a mean would be dragged by
    the handful of coordinates that fall outside Brazil entirely.
    """
    g = geolocation[
        geolocation["geolocation_lat"].between(-34, 6)
        & geolocation["geolocation_lng"].between(-74, -34)
    ]
    return (
        g.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in kilometres, vectorised over numpy arrays."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lng2) - np.radians(lng1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Aggregate the one-to-many children
# ---------------------------------------------------------------------------
def _items_per_order(items: pd.DataFrame, products: pd.DataFrame,
                     translation: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    """Collapse order_items (plus product and seller attributes) to one row."""
    trans = translation.rename(columns={translation.columns[0]: "product_category_name"})
    prod = products.merge(trans, on="product_category_name", how="left")
    prod["category"] = (
        prod["product_category_name_english"]
        .fillna(prod["product_category_name"])
        .fillna("unknown")
    )

    it = items.merge(
        prod[["product_id", "category", "product_weight_g", "product_length_cm",
              "product_height_cm", "product_width_cm", "product_photos_qty"]],
        on="product_id", how="left",
    ).merge(
        sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]],
        on="seller_id", how="left",
    )
    it["item_volume_cm3"] = (
        it["product_length_cm"] * it["product_height_cm"] * it["product_width_cm"]
    )

    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_distinct_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        max_item_price=("price", "max"),
        total_weight_g=("product_weight_g", "sum"),
        max_volume_cm3=("item_volume_cm3", "max"),
        total_photos=("product_photos_qty", "sum"),
    )

    # Attributes of the most expensive item: the one that characterises the order.
    # seller_id is carried through for the Phase 2 expanding-window history
    # features (src/features.py); nothing in Phase 1 reads it.
    lead = (
        it.sort_values(["order_id", "price"], ascending=[True, False])
        .groupby("order_id")
        .first()[["category", "seller_id", "seller_state", "seller_zip_code_prefix"]]
        .rename(columns={"category": "lead_category",
                         "seller_id": "lead_seller_id",
                         "seller_state": "lead_seller_state",
                         "seller_zip_code_prefix": "lead_seller_zip"})
    )
    return agg.join(lead)


def _payments_per_order(payments: pd.DataFrame) -> pd.DataFrame:
    """Collapse order_payments to one row per order."""
    agg = payments.groupby("order_id").agg(
        n_payment_records=("payment_sequential", "count"),
        total_payment=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
    )
    # Payment type of the largest single payment on the order.
    lead = (
        payments.sort_values(["order_id", "payment_value"], ascending=[True, False])
        .groupby("order_id")
        .first()[["payment_type"]]
        .rename(columns={"payment_type": "lead_payment_type"})
    )
    return agg.join(lead)


def _reviews_per_order(reviews: pd.DataFrame) -> pd.DataFrame:
    """Earliest review per order; a handful of orders carry more than one."""
    r = reviews.sort_values(["order_id", "review_creation_date"])
    return (
        r.groupby("order_id")
        .first()[["review_score", "review_creation_date", "review_answer_timestamp"]]
    )


# ---------------------------------------------------------------------------
# Build the analysis table
# ---------------------------------------------------------------------------
def build_orders(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Join the nine tables into one tidy row per order.

    Returns every order, including the 3% that were never delivered; the
    delivery targets are NaN for those rather than dropped, so the dashboard can
    still report on order status.
    """
    t = tables or load_all()

    df = t["orders"].merge(
        t["customers"][["customer_id", "customer_unique_id",
                        "customer_state", "customer_city",
                        "customer_zip_code_prefix"]],
        on="customer_id", how="left", validate="one_to_one",
    )

    df = (
        df.merge(_items_per_order(t["order_items"], t["products"],
                                  t["category_translation"], t["sellers"]),
                 left_on="order_id", right_index=True, how="left")
          .merge(_payments_per_order(t["order_payments"]),
                 left_on="order_id", right_index=True, how="left")
          .merge(_reviews_per_order(t["order_reviews"]),
                 left_on="order_id", right_index=True, how="left")
    )
    assert len(df) == EXPECTED_ROWS["orders"], "joins must not change the grain"

    # -- geography -----------------------------------------------------------
    cent = zip_centroids(t["geolocation"])
    df = df.join(cent.rename(columns={"lat": "cust_lat", "lng": "cust_lng"}),
                 on="customer_zip_code_prefix")
    df = df.join(cent.rename(columns={"lat": "sell_lat", "lng": "sell_lng"}),
                 on="lead_seller_zip")
    df["distance_km"] = haversine_km(
        df["cust_lat"], df["cust_lng"], df["sell_lat"], df["sell_lng"]
    )
    df["customer_region"] = df["customer_state"].map(STATE_REGION)
    df["seller_region"] = df["lead_seller_state"].map(STATE_REGION)
    df["same_state"] = (df["customer_state"] == df["lead_seller_state"])

    # -- outcome measures (POST-OUTCOME: never model inputs) ------------------
    delivered = df["order_delivered_customer_date"].notna()
    df["delivery_days"] = np.where(
        delivered,
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"])
        .dt.total_seconds() / 86400,
        np.nan,
    )
    df["delay_vs_estimate_days"] = np.where(
        delivered,
        (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"])
        .dt.total_seconds() / 86400,
        np.nan,
    )
    df["is_late"] = np.where(delivered, (df["delay_vs_estimate_days"] > 0).astype(float), np.nan)
    df["approval_hours"] = (
        (df["order_approved_at"] - df["order_purchase_timestamp"])
        .dt.total_seconds() / 3600
    )

    # -- known at checkout ---------------------------------------------------
    df["estimated_days"] = (
        (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"])
        .dt.total_seconds() / 86400
    )
    df["lead_category"] = df["lead_category"].fillna("unknown")

    return df.sort_values("order_purchase_timestamp").reset_index(drop=True)


def load_analysis(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Build the order table and restrict it to the analysis window."""
    df = build_orders(tables)
    mask = df["order_purchase_timestamp"].between(ANALYSIS_START, ANALYSIS_END)
    return df.loc[mask].reset_index(drop=True)


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Missingness audit for the columns the analysis actually depends on."""
    cols = ["order_delivered_customer_date", "order_approved_at", "review_score",
            "lead_category", "total_price", "total_freight", "total_weight_g",
            "distance_km", "lead_payment_type", "estimated_days"]
    rows = []
    for c in cols:
        s = df[c]
        rows.append({
            "column": c,
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 2),
            "n_unique": int(s.nunique()),
        })
    return pd.DataFrame(rows)
