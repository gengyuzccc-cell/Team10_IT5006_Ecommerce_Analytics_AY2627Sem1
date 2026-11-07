"""Model serving for the Olist dashboard - the Phase 2 proof-of-concept.

Two tabs: a single-order scoring form (with the date the customer would be quoted,
the batch scorer and the triage list), and a model card (what is deployed, where
not to trust it, what the recommendations change for customers, the value and
pilot calculators with both costs as inputs, and where every number on the slides
comes from).

The point of interest here is that nothing reimplements the training-time
feature engineering. A form submission is turned back into a one-row order
frame and pushed through the same `features.py` builders the models were fitted
on, so a feature can never be computed one way in training and another way in
production - the single most common way a deployed model quietly stops matching
its reported metrics.

The one thing the form cannot supply is a seller's track record: a visitor knows
a state and a category, not a 32-character seller id. `models/history_lookup.csv`
carries those aggregates at each level of the same fallback chain
`features.add_history` uses, and `history_features()` walks it in the same order.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PLOT_CFG = {"displayModeBar": False, "responsive": True}

# Columns a batch upload must provide; everything else is derived.
BATCH_COLUMNS = [
    "order_purchase_timestamp", "order_estimated_delivery_date",
    "customer_state", "seller_state", "lead_category", "lead_payment_type",
    "n_items", "n_sellers", "total_price", "total_freight", "total_weight_g",
    "max_volume_cm3", "total_photos", "max_installments",
]
MAX_BATCH_ROWS = 5000


# ===========================================================================
# Artifacts
# ===========================================================================
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts() -> dict:
    """Load the fitted models, the history lookup and the run metadata.

    Cached as a resource rather than data: these are unpickled objects shared
    across sessions, and re-reading them per visitor would waste both the time
    and the memory. Loaded once per server process, on the first run, and shared
    by every later visitor.
    """
    import json

    import modeling as M

    missing = [n for n in ("p1a_leadtime.joblib", "p2_lowreview.joblib",
                           "p1b_margin_to_risk.joblib", "history_lookup.csv",
                           "model_metadata.json")
               if not (MODELS_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(
            "missing model artifacts: " + ", ".join(missing)
            + ". Run `python src/train_models.py` to build them.")

    lookup = pd.read_csv(MODELS_DIR / "history_lookup.csv")
    return {
        "p1a": M.load_model(MODELS_DIR / "p1a_leadtime.joblib"),
        "p2": M.load_model(MODELS_DIR / "p2_lowreview.joblib"),
        "platt": M.load_model(MODELS_DIR / "p1b_margin_to_risk.joblib"),
        "meta": json.loads((MODELS_DIR / "model_metadata.json").read_text("utf-8")),
        "lookup": {lvl: g.set_index("key")
                   for lvl, g in lookup.groupby("level")},
    }


@st.cache_data(show_spinner=False)
def load_capacity() -> pd.DataFrame | None:
    """Precision@K / Recall@K of the review model on the held-out test window.

    Written by src/capacity_metrics.py after the Milestone 1 feedback, which
    pointed out that a team with limited capacity works a ranked list from the
    top, so the metric that describes its day is the precision of the top-K.
    The table is tracked in the repository, so the deployed app can show it
    without recomputing anything.
    """
    path = PROJECT_ROOT / "report" / "results" / "capacity_p2.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def route_distances(_df: pd.DataFrame) -> tuple[dict, float]:
    """Median customer-seller distance per state pair, for the form to fill in.

    Distance is one of the strongest checkout-time features (r = 0.39 with lead
    time), but it comes from postcode centroids the form does not ask for. The
    median for the chosen state pair is the closest available stand-in, and the
    overall median covers a pair the data has never seen.
    """
    d = _df.dropna(subset=["distance_km", "lead_seller_state", "customer_state"])
    med = d.groupby(["lead_seller_state", "customer_state"])["distance_km"].median()
    return med.to_dict(), float(d["distance_km"].median())


# ===========================================================================
# Feature assembly
# ===========================================================================
def history_features(lookup: dict, seller_state: str, customer_state: str,
                     category: str, seller_id: str | None = None) -> dict:
    """Walk the same fallback chain add_history() uses: seller, route, state, global."""
    glob = lookup["global"].iloc[0]

    def get(level: str, key, column):
        table = lookup.get(level)
        if table is None or key is None or key not in table.index:
            return np.nan
        return table.loc[key, column]

    route = f"{seller_state}>{customer_state}"
    seller_days = get("seller", seller_id, "lead_days")
    route_days = get("route", route, "lead_days")
    state_days = get("seller_state", seller_state, "lead_days")
    seller_late = get("seller", seller_id, "late_rate")
    state_late = get("seller_state", seller_state, "late_rate")
    cat_low = get("category", category, "low_review_rate")
    seller_low = get("seller", seller_id, "low_review_rate")

    def first(*values):
        for v in values:
            if v is not None and not pd.isna(v):
                return float(v)
        return np.nan

    return {
        "hist_platform_lead_days": float(glob["lead_days"]),
        "hist_platform_late_rate": float(glob["late_rate"]),
        "hist_route_lead_days": first(route_days, state_days, glob["lead_days"]),
        "hist_category_low_review_rate": first(cat_low, glob["low_review_rate"]),
        "hist_seller_lead_days": first(seller_days, route_days, state_days,
                                       glob["lead_days"]),
        "hist_seller_late_rate": first(seller_late, state_late, glob["late_rate"]),
        "hist_seller_low_review_rate": first(seller_low, cat_low,
                                             glob["low_review_rate"]),
        # No named seller means the seller's order count is UNKNOWN, not zero.
        # Zero would tell the model "brand-new seller", which is a specific and
        # usually pessimistic claim; NaN lets the pipeline's imputer treat it as
        # the missing value it actually is.
        "hist_seller_n": float("nan") if seller_id is None else 0.0,
    }


def assemble(orders: pd.DataFrame, lookup: dict, distances: tuple) -> pd.DataFrame:
    """Turn raw order fields into the exact frame the models were fitted on.

    Runs the training-time builders rather than restating them, so the encoding
    of every derived quantity is identical by construction.
    """
    import config
    from config import STATE_REGION
    from features import add_basket, add_calendar, add_promise

    med_map, med_all = distances
    df = orders.copy()

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_estimated_delivery_date"] = pd.to_datetime(
        df["order_estimated_delivery_date"])
    df["estimated_days"] = ((df["order_estimated_delivery_date"]
                             - df["order_purchase_timestamp"])
                            .dt.total_seconds() / 86400)

    df["lead_seller_state"] = df["seller_state"]
    df["seller_region"] = df["seller_state"].map(STATE_REGION)
    df["customer_region"] = df["customer_state"].map(STATE_REGION)
    df["same_state"] = df["customer_state"] == df["seller_state"]
    df["distance_km"] = [
        med_map.get((s, c), med_all)
        for s, c in zip(df["seller_state"], df["customer_state"])
    ]

    # The pinned threshold, not the frame median - see features.add_promise.
    # A single order has no median to compare itself against.
    df = add_promise(add_basket(add_calendar(df)),
                     threshold=config.PROMISE_MEDIAN_DAYS)

    hist = pd.DataFrame([
        history_features(lookup, r["seller_state"], r["customer_state"],
                         r["lead_category"], r.get("seller_id"))
        for _, r in df.iterrows()
    ], index=df.index)
    return pd.concat([df, hist], axis=1)


def score(frame: pd.DataFrame, art: dict, at_delivery: bool = False) -> pd.DataFrame:
    """Predicted lead time, late risk and - once delivered - low-review risk."""
    p1a, p2, platt = art["p1a"], art["p2"], art["platt"]
    meta = art["meta"]

    out = pd.DataFrame(index=frame.index)
    out["predicted_days"] = p1a["estimator"].predict(frame[p1a["features"]])
    out["promised_days"] = frame["estimated_days"].to_numpy()
    out["margin_days"] = out["predicted_days"] - out["promised_days"]
    out["p_late"] = platt["estimator"].predict_proba(
        out[["margin_days"]].to_numpy())[:, 1]
    out["late_flag"] = (out["p_late"] >= meta["p1b"]["threshold"]).astype(int)

    if at_delivery:
        # The review model runs once the parcel has landed, so it may see how
        # delivery actually went - the at_delivery regime. Here that outcome is
        # the model's own forecast, which is what a pre-arrival triage would
        # have to use; the reported metrics assume the realised values.
        # is_late follows the forecast delay, as it follows the realised delay in
        # training (delivered after the promised date), not the 11% warning flag:
        # the flag would mark an order late while its forecast delay is negative.
        f = frame.copy()
        f["delivery_days"] = out["predicted_days"]
        f["delay_vs_estimate_days"] = out["margin_days"]
        f["is_late"] = (out["margin_days"] > 0).astype(float)
        out["p_low_review"] = p2["estimator"].predict_proba(f[p2["features"]])[:, 1]
        out["low_review_flag"] = (out["p_low_review"] >= meta["p2"]["threshold"]).astype(int)
    return out


# ===========================================================================
# Validation
# ===========================================================================
def validate(row: dict) -> list[str]:
    """Human-readable complaints about one order, empty when it is usable."""
    problems = []
    if row["order_estimated_delivery_date"] <= row["order_purchase_timestamp"]:
        problems.append("The promised delivery date must fall after the purchase date.")
    if row["total_price"] <= 0:
        problems.append("Item total must be greater than zero.")
    if row["total_freight"] < 0:
        problems.append("Freight cannot be negative.")
    if row["n_items"] < 1:
        problems.append("An order needs at least one item.")
    if row["n_sellers"] > row["n_items"]:
        problems.append("An order cannot involve more sellers than items.")
    if row["total_weight_g"] <= 0:
        problems.append("Total weight must be greater than zero.")
    return problems


# ===========================================================================
# UI - single order
# ===========================================================================
def render_predict(df: pd.DataFrame) -> None:
    try:
        art = load_artifacts()
    except FileNotFoundError as exc:
        st.error(f"**Models are not available.** {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"**Could not load the models:** `{type(exc).__name__}: {exc}`")
        return

    for bundle in (art["p1a"], art["p2"], art["platt"]):
        if "version_warning" in bundle:
            st.warning(bundle["version_warning"])

    meta = art["meta"]
    st.markdown(
        theme.section(
            "Score an order",
            "Enter an order as it would look at checkout. You get the lead time the "
            "model expects, the risk of missing the promised date, the date the customer "
            "would be quoted and, under that forecast, the risk of a 1–2 star review."),
        unsafe_allow_html=True)

    states = sorted(df["customer_state"].dropna().unique())
    cats = sorted(df["lead_category"].dropna().unique())
    pays = sorted(df["lead_payment_type"].dropna().unique())

    with st.form("single_order"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Route**")
            seller_state = st.selectbox("Seller state", states,
                                        index=states.index("SP") if "SP" in states else 0)
            customer_state = st.selectbox("Customer state", states,
                                          index=states.index("RJ") if "RJ" in states else 0)
            category = st.selectbox("Product category", cats,
                                    index=cats.index("bed_bath_table")
                                    if "bed_bath_table" in cats else 0)
        with c2:
            st.markdown("**Basket**")
            n_items = st.number_input("Items", 1, 30, 1)
            n_sellers = st.number_input("Distinct sellers", 1, 10, 1)
            total_price = st.number_input("Item total (R$)", 0.01, 50_000.0, 89.90, step=10.0)
            total_freight = st.number_input("Freight (R$)", 0.0, 5_000.0, 17.20, step=1.0)
            weight = st.number_input("Total weight (g)", 1.0, 100_000.0, 900.0, step=100.0)
        with c3:
            st.markdown("**Timing and payment**")
            purchase = st.date_input("Purchase date", pd.Timestamp("2018-07-15"))
            promised = st.date_input("Promised delivery date", pd.Timestamp("2018-08-02"))
            payment = st.selectbox("Payment type", pays,
                                   index=pays.index("credit_card")
                                   if "credit_card" in pays else 0)
            installments = st.number_input("Instalments", 1, 24, 1)
            volume = st.number_input("Largest item volume (cm³)", 1.0, 500_000.0,
                                     8_000.0, step=500.0)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = {
            "order_purchase_timestamp": pd.Timestamp(purchase),
            "order_estimated_delivery_date": pd.Timestamp(promised),
            "customer_state": customer_state, "seller_state": seller_state,
            "lead_category": category, "lead_payment_type": payment,
            "n_items": int(n_items), "n_sellers": int(n_sellers),
            "total_price": float(total_price), "total_freight": float(total_freight),
            "total_weight_g": float(weight), "max_volume_cm3": float(volume),
            "total_photos": 3.0, "max_installments": float(installments),
        }
        problems = validate(row)
        if problems:
            for p in problems:
                st.error(p)
            render_batch(df, art)
            return

        try:
            frame = assemble(pd.DataFrame([row]), art["lookup"], route_distances(df))
            result = score(frame, art, at_delivery=True).iloc[0].astype(object)
        except Exception as exc:  # noqa: BLE001
            st.error(f"**Could not score this order:** `{type(exc).__name__}: {exc}`")
            render_batch(df, art)
            return
        result["purchase"] = row["order_purchase_timestamp"]
        result["promised_date"] = row["order_estimated_delivery_date"]
        # Keep the result across reruns: a sidebar change or a date picked in
        # another tab must not wipe the numbers a visitor is looking at.
        st.session_state["last_prediction"] = result
    elif "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        st.caption("Showing the last order scored in this session; press **Predict** "
                   "to score the form as it stands.")
    else:
        st.info("Fill the form and press **Predict**. The defaults describe a "
                "typical São Paulo → Rio order in July 2018.")
        render_batch(df, art)
        return

    mae = meta["p1a"]["test_mae"]
    late_thr = meta["p1b"]["threshold"]
    low_thr = meta["p2"]["threshold"]
    late_line, low_line = f"{100 * late_thr:.0f}%", f"{100 * low_thr:.0f}%"
    if float(result["predicted_days"]) < 0:
        # The regression extrapolates below zero far outside its training orders (a
        # basket of 30 items, say); the number is shown as the model gives it.
        st.warning(f"**The forecast is below zero ({_dec(result['predicted_days'])} "
                   "days).** A lead time cannot be negative: this order lies outside the "
                   "range of the orders the model was fitted on, so read the numbers below "
                   "as unreliable and check the basket and the dates.")
    # The sub-labels come from the same comparison as the flags (risk >= threshold),
    # so a 10.9% risk reads "below the 11% flag line", never "11% not flagged".
    st.markdown(
        theme.kpi_row([
            ("Predicted lead time", f"{_dec(result['predicted_days'])} d",
             f"±{mae:.1f} d typical error", theme.PRIMARY),
            ("Promised", f"{result['promised_days']:.0f} d",
             f"margin {_signed(float(result['margin_days']), 1)} d", theme.GOLD),
            ("Risk of missing the promise", f"{100 * result['p_late']:.1f}%",
             f"{'at or above' if result['late_flag'] else 'below'} the {late_line} flag line",
             theme.ACCENT),
            ("Risk of a 1–2 star review", f"{100 * result['p_low_review']:.1f}%",
             f"{'at or above' if result['low_review_flag'] else 'below'} the {low_line} flag line",
             theme.ACCENT),
        ]),
        unsafe_allow_html=True)

    quote = render_quote(result)

    # theme.note() emits raw HTML, so this has to be HTML - markdown asterisks
    # would render literally. The advice is the report's first recommendation: quote
    # the forecast plus the margin, and then no blanket warning; if the platform keeps
    # its own date, a warning only for the riskiest tenth of orders, because one at the
    # 11% line would reach almost half of all customers, most of them on time.
    p_late = float(result["p_late"])
    expect = (f"The model expects the parcel {abs(result['margin_days']):.1f} days "
              f"{'after' if result['margin_days'] > 0 else 'before'} the platform's date "
              f"(typical error ±{mae:.1f} days). ")
    if quote is not None:
        quoted, b = quote
        lead = (f"<b>Quote {quoted.day} {quoted:%b %Y}</b>, the forecast plus the {b:g}-day "
                f"margin: with that date no blanket warning is needed. {expect}"
                "If the platform keeps its own date, ")
    else:
        lead = expect + "Against the platform's date, "
    tenth = late_top_tenth()
    if tenth is not None:
        q90, share_late = tenth
        riskiest = (f"the riskiest tenth of held-out orders (late risk {100 * q90:.1f}% or "
                    f"more; {100 * share_late:.1f}% of those parcels were late)")
        advice = (f"this order is in {riskiest}: <b>tell the customer the date may move</b>."
                  if p_late >= q90 else
                  f"a warning belongs only on {riskiest}, and this order's "
                  f"{100 * p_late:.1f}% is below it: <b>no warning</b>.")
    else:
        advice = ("<b>tell the customer the date may move</b>." if result["late_flag"]
                  else "<b>no warning</b>.")
    st.markdown(
        theme.note(
            "What to do with this",
            lead + advice
            + (f"{warning_reach(late_line)}{expedite_text()}<br><br>"
               f"The review risk shown is conditional on that forecast, so it "
               f"inherits the forecast's error. Once the parcel actually lands, "
               f"re-score with the realised delivery date: that is the "
               f"at-delivery moment the model was measured at "
               f"(ROC-AUC {meta['p2']['test_roc_auc']:.2f}, flagging at or above "
               f"{low_line}).")),
        unsafe_allow_html=True)

    render_batch(df, art)


def late_top_tenth() -> tuple[float, float] | None:
    """The riskiest tenth of delivered held-out orders by late risk: its threshold (the
    90th percentile of p_late_derived in predictions_p1b_test.csv, as src/customer_view.py
    defines the list) and the share of those parcels that were late
    (customer_contact_lists.csv, late_top_decile)."""
    p1b = load_table("predictions_p1b_test.csv")
    lists = load_table("customer_contact_lists.csv")
    if p1b is None or lists is None or "late_top_decile" not in set(lists["list"]):
        return None
    share_late = float(lists.set_index("list").loc["late_top_decile", "precision"])
    return float(p1b["p_late_derived"].quantile(0.9)), share_late


def expedite_text() -> str:
    """Why the late flag is not used to pay for a rush (capacity_p1b.csv, derived route)."""
    cap = load_table("capacity_p1b.csv")
    if cap is None:
        return ""
    d = cap[cap["route"] == "derived"]
    return (f" Expediting is not recommended: under the assumed costs a paid rush needs a "
            f"precision of {float(d['break_even_precision'].iloc[0]):.3f}, and the late "
            f"flag's ranked list peaks at {float(d['precision_at_k'].max()):.3f}.")


def warning_reach(late_line: str = "11%") -> str:
    """Who a warning at the deployed flag line would reach on the held-out window
    (customer_outcomes_test.csv)."""
    out = load_table("customer_outcomes_test.csv")
    if out is None:
        return ""
    o = out.set_index("key")
    flagged = float(o.loc["prewarn_flagged", "orders"])
    late_warned = float(o.loc["prewarn_late_flagged", "orders"])
    late_all = late_warned + float(o.loc["prewarn_late_missed", "orders"])
    return (f" A warning at the deployed {late_line} flag line would have reached "
            f"{o.loc['prewarn_flagged', 'per_1000'] / 10:.1f}% of held-out customers, and "
            f"{100 * (flagged - late_warned) / flagged:.1f}% of their parcels then came on "
            f"time; it would have warned {100 * late_warned / late_all:.1f}% of the late ones "
            f"({int(o.loc['prewarn_flagged', 'denominator_n']):,} delivered held-out orders).")


def render_quote(result) -> tuple[pd.Timestamp, float] | None:
    """What the customer would see: the platform's date, and the forecast plus the margin.

    The margin is rule R1 of src/customer_view.py, read from customer_promise_rules.csv,
    and the date is customer_view.requote(): purchase + forecast + margin, rounded up to
    the next midnight, a calendar date like the platform's own. Returns the quoted date
    and the margin, or None when they cannot be computed.
    """
    purchase, promised = result.get("purchase"), result.get("promised_date")
    rules = load_table("customer_promise_rules.csv")
    if purchase is None or promised is None or rules is None:
        return None
    try:
        from customer_view import requote

        b = float(rules.loc[rules["rule"].str.startswith("R1"), "buffer_days"].iloc[0])
        test = rules[rules["period"] == "test"].set_index("rule")
        olist = test.loc["Olist's own promise"]
        r1 = test[test.index.str.startswith("R1")].iloc[0]
        quoted = requote(pd.Series([pd.Timestamp(purchase)]),
                         pd.Series([float(result["predicted_days"])]), b).iloc[0]
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Quoted date unavailable: `{type(exc).__name__}: {exc}`")
        return None
    promised = pd.Timestamp(promised)
    shift = int((quoted - promised).days)
    longer = float(r1["mean_promised_days"] - olist["mean_promised_days"])
    st.markdown(theme.section("What the customer would see",
                              "The delivery date shown at checkout: the platform's own, "
                              "and the one the forecast plus a margin would quote."),
                unsafe_allow_html=True)
    st.markdown(theme.kpi_row([
        ("The platform's date", f"{promised.day} {promised:%b %Y}",
         f"{float(result['promised_days']):.0f} days after the order, as entered", theme.GOLD),
        ("Forecast plus the margin", f"{quoted.day} {quoted:%b %Y}",
         f"forecast {_dec(result['predicted_days'])} d + the {b:g}-day margin, rounded "
         "up to a calendar day", theme.PRIMARY),
        ("Difference", (f"{shift:+d}" if shift else "0") + f" day{'s' if abs(shift) != 1 else ''}",
         "later than the platform's date" if shift > 0 else
         "earlier than the platform's date" if shift < 0 else "the same date", theme.GREEN),
    ]), unsafe_allow_html=True)
    st.caption(
        f"On the {int(r1['n']):,} delivered held-out orders, dates set this way were missed "
        f"for {r1['late_pct']:.1f}% of customers against {olist['late_pct']:.1f}% for the "
        f"platform's date, and were {abs(longer):.1f} days {'longer' if longer > 0 else 'shorter'} "
        "on average. The margin is the smallest whose broken-promise rate on the validation "
        "months is no higher than the platform's own (customer_promise_rules.csv).")
    return quoted, b


# ===========================================================================
# UI - batch
# ===========================================================================
def clean_batch(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalise a batch upload and drop the rows the single-order form would refuse.

    State codes are stripped and upper-cased, dates and numbers are coerced, so a
    value that cannot be parsed marks its row instead of raising deep inside the
    model. Returns the usable rows and one message per problem, with the offending
    row numbers counted as in the file (the header is line 1).
    """
    from config import STATE_REGION

    df = raw.copy()
    missing_cat = df["lead_category"].isna() | df["lead_payment_type"].isna()
    for c in ("customer_state", "seller_state", "lead_category", "lead_payment_type"):
        df[c] = df[c].astype(str).str.strip()
    for c in ("customer_state", "seller_state"):
        df[c] = df[c].str.upper()
    for c in ("n_items", "n_sellers", "total_price", "total_freight", "total_weight_g",
              "max_volume_cm3", "total_photos", "max_installments"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("order_purchase_timestamp", "order_estimated_delivery_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed", dayfirst=True)

    states = list(STATE_REGION)
    checks = [
        ("a date that does not parse",
         df["order_purchase_timestamp"].isna() | df["order_estimated_delivery_date"].isna()),
        ("promised date not after the purchase date",
         df["order_estimated_delivery_date"] <= df["order_purchase_timestamp"]),
        ("unknown state code",
         ~df["customer_state"].isin(states) | ~df["seller_state"].isin(states)),
        ("missing category or payment type", missing_cat),
        ("item total not greater than zero", ~(df["total_price"] > 0)),
        ("negative or missing freight", ~(df["total_freight"] >= 0)),
        ("fewer than one item, or more sellers than items",
         ~(df["n_items"] >= 1) | ~(df["n_sellers"] >= 1) | (df["n_sellers"] > df["n_items"])),
        ("total weight not greater than zero", ~(df["total_weight_g"] > 0)),
        ("volume, photos or instalments not a number",
         df[["max_volume_cm3", "total_photos", "max_installments"]].isna().any(axis=1)),
    ]
    bad = pd.Series(False, index=df.index)
    problems = []
    for label, mask in checks:
        mask = mask.fillna(True).astype(bool)
        if mask.any():
            rows = [str(i + 2) for i in df.index[mask][:6]]
            more = ", …" if int(mask.sum()) > 6 else ""
            plural = "s" if mask.sum() > 1 else ""
            problems.append(f"{label} (row{plural} {', '.join(rows)}{more})")
            bad |= mask
    return df[~bad].reset_index(drop=True), problems


def render_batch(df: pd.DataFrame, art: dict) -> None:
    st.markdown("---")
    st.markdown(
        theme.section("Score a file",
                      "Upload a CSV of orders to get the same three predictions "
                      "for each row."),
        unsafe_allow_html=True)

    template = pd.DataFrame([{
        "order_purchase_timestamp": "2018-07-15", "order_estimated_delivery_date": "2018-08-02",
        "customer_state": "RJ", "seller_state": "SP", "lead_category": "bed_bath_table",
        "lead_payment_type": "credit_card", "n_items": 1, "n_sellers": 1,
        "total_price": 89.90, "total_freight": 17.20, "total_weight_g": 900,
        "max_volume_cm3": 8000, "total_photos": 3, "max_installments": 1,
    }])
    st.download_button("Download the template", template.to_csv(index=False),
                       "order_template.csv", "text/csv")

    upload = st.file_uploader("CSV of orders", type=["csv"],
                              label_visibility="collapsed")
    if upload is None:
        st.caption(f"Required columns: `{'`, `'.join(BATCH_COLUMNS)}`. "
                   f"Up to {MAX_BATCH_ROWS:,} rows.")
        return

    try:
        raw = pd.read_csv(upload)
    except pd.errors.EmptyDataError:
        st.error("**That file is empty.**")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"**Could not read that CSV:** `{type(exc).__name__}: {exc}`")
        return

    if raw.empty:
        st.error("**That file has a header but no rows.**")
        return
    missing = [c for c in BATCH_COLUMNS if c not in raw.columns]
    if missing:
        st.error("**Missing required column(s):** `" + "`, `".join(missing)
                 + "`. Download the template above for the exact format.")
        return
    if len(raw) > MAX_BATCH_ROWS:
        st.warning(f"File has {len(raw):,} rows; scoring the first "
                   f"{MAX_BATCH_ROWS:,}.")
        raw = raw.head(MAX_BATCH_ROWS)

    raw, problems = clean_batch(raw)
    if problems:
        st.warning("**Rows skipped because the form would refuse them:** "
                   + "; ".join(problems[:8]) + (" …" if len(problems) > 8 else "") + ".")
    if raw.empty:
        st.error("**No row in that file can be scored.** Check the dates, the state codes "
                 "and the numeric columns against the template.")
        return

    try:
        frame = assemble(raw, art["lookup"], route_distances(df))
        result = score(frame, art, at_delivery=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"**Could not score that file:** `{type(exc).__name__}: {exc}`. "
                 "Check that dates parse and that the numeric columns hold numbers.")
        return

    scored = pd.concat([raw.reset_index(drop=True),
                        result.reset_index(drop=True).round(4)], axis=1)
    below_zero = int((scored["predicted_days"] < 0).sum())
    if below_zero:
        st.warning(f"**{below_zero} order{'s have' if below_zero > 1 else ' has'} a forecast "
                   "below zero** (a negative `predicted_days` in the download). A lead time "
                   "cannot be negative: such an order lies outside the range of the orders "
                   "the model was fitted on, so check its inputs; the forecast is returned as "
                   "the model gives it.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Orders scored", f"{len(scored):,}")
    c2.metric("Flagged late", f"{100*result['late_flag'].mean():.1f}%")
    c3.metric("Flagged review risk", f"{100*result['low_review_flag'].mean():.1f}%")
    st.dataframe(scored.head(50), use_container_width=True, hide_index=True)

    buf = io.StringIO()
    scored.to_csv(buf, index=False)
    st.download_button("Download scored orders", buf.getvalue(),
                       "scored_orders.csv", "text/csv", type="primary")

    render_triage(scored)


# ===========================================================================
# UI - triage list (capacity-constrained, added after the Milestone 1 feedback)
# ===========================================================================
def render_triage(scored: pd.DataFrame) -> None:
    """The ranked list a customer-experience team actually works.

    A flag at a threshold is not what a team with K agents does: it works the
    day's orders from the riskiest down until capacity runs out. This view ranks
    the scored file by review risk, marks the top-K for a capacity the user
    enters, and shows the precision and recall that capacity achieved on the
    held-out test window (report Section 4.4).
    """
    st.markdown("---")
    st.markdown(
        theme.section("Triage list: work the top of the ranking",
                      "Orders ranked by review risk, then by late risk. Enter the "
                      "number of contacts your team can make and the top-K are marked."),
        unsafe_allow_html=True)

    n = len(scored)
    k = int(st.number_input("Contacts your team can make (capacity K)", min_value=1,
                            max_value=max(1, n), value=min(10, n), step=1))
    ranked = scored.sort_values(["p_low_review", "p_late"], ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, n + 1))
    ranked.insert(1, "in_top_k", ranked["rank"] <= k)
    share = 100 * k / n

    cap = load_capacity()
    if cap is not None and len(cap):
        row = cap.iloc[(cap["k_pct"] - share).abs().argmin()]
        st.caption(
            f"Top {k:,} of {n:,} orders = {share:.1f}% of this file. On the held-out "
            f"test window, the top {row['k_pct']:g}% of reviewed test orders by review risk "
            f"held a customer who went on to leave a 1–2 star review "
            f"{100*row['precision_at_k']:.0f}% of the time (precision@K; historical "
            f"outcome), {100*row['recall_at_k']:.0f}% of all such customers (recall@K) "
            f"and {row['lift_at_k']:.1f}× the rate at random. The review risk here is "
            f"conditional on the forecast delivery; in operation, re-score at delivery "
            f"with the realised dates.")
    else:
        st.caption(f"Top {k:,} of {n:,} orders = {share:.1f}% of this file.")

    cols = ["rank", "in_top_k", "p_low_review", "p_late", "predicted_days",
            "promised_days"] + [c for c in BATCH_COLUMNS if c in ranked.columns]
    st.dataframe(ranked[cols].head(max(k, 10)), use_container_width=True, hide_index=True,
                 column_config={"in_top_k": st.column_config.CheckboxColumn("in top-K")})
    buf = io.StringIO()
    ranked.to_csv(buf, index=False)
    st.download_button("Download the ranked triage list", buf.getvalue(),
                       "triage_list.csv", "text/csv")


# ===========================================================================
# UI - model card
# ===========================================================================
@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame | None:
    """A result table written by the analysis scripts, or None when absent.

    The model card recomputes nothing: every table it shows is one the report
    quotes, read from report/results/ exactly as the scripts wrote it.
    """
    path = PROJECT_ROOT / "report" / "results" / name
    return pd.read_csv(path) if path.exists() else None


def _as_pickled(bundle: dict) -> str:
    """The estimator class and, for an ensemble, its members - read from the artifact."""
    est = bundle["estimator"]
    if not hasattr(est, "estimators"):
        return type(est).__name__
    parts = []
    for _name, member in est.estimators:
        inner = member.steps[-1][1] if hasattr(member, "steps") else member
        parts.append(type(inner).__name__)
    text = f"{type(est).__name__} of " + " + ".join(parts)
    if getattr(est, "weights", None) is not None:
        text += f" · weights {':'.join(str(w) for w in est.weights)}"
    fe = getattr(est, "final_estimator_", None)
    if fe is not None:
        text += f" · meta-learner {type(fe).__name__}"
    return text


def _artifact(name: str) -> str:
    path = MODELS_DIR / name
    return f"{name} · {path.stat().st_size / 1e3:.0f} KB" if path.exists() else name



# The contacts the model card compares: (list in customer_contact_lists.csv, name,
# who it reaches). The value table reads their counts from customer_value.csv.
CONTACTS = [
    ("promised_date_message", "Promised-date message (no model)",
     "customers whose promised day ends without a parcel, not yet rated"),
    ("delivery_day_k10", "Delivery-day call, ten a day",
     "each day's riskiest deliveries whose customer has not yet rated"),
    ("slides_ten_a_day", "The slides' list: ten a day by purchase day",
     "each purchase day's riskiest reviewed orders, scored at delivery"),
]
# Pilot arms: the radio label, its rows in customer_pilot_arms.csv, its contact list.
ARMS = {
    "Promised-date message (Arm B)": ("arm_b", "promised_date_message"),
    "Delivery-day call (Arm A)": ("arm_a", "delivery_day_k10"),
    "Design on slide 11": ("slide_design", "slides_ten_a_day"),
}


def _split_lines(df: pd.DataFrame | None, meta: dict) -> str:
    """The analysis window and the three periods as orders placed, from first to last
    purchase day, counted on the order frame with the masks of features.py (train up
    to config.TRAIN_END, validation to config.VAL_END, test after it). The metadata's
    cut-offs are instants at midnight, so printing them as dates would put 30 June in
    both validation and test; without the frame the cut-offs are printed as such."""
    def span(s: pd.Series, start=None) -> str:
        a, b = (start if start is not None else s.min()), s.max()
        return f"{a.day} {a:%b %Y} – {b.day} {b:%b %Y}"

    try:
        import config

        ts = df["order_purchase_timestamp"]
        train = ts[ts <= config.TRAIN_END]
        val = ts[(ts > config.TRAIN_END) & (ts <= config.VAL_END)]
        test = ts[ts > config.VAL_END]
        lines = (f"- Orders placed **{span(ts, config.ANALYSIS_START)}** ({len(ts):,})\n"
                 f"- Train **{span(train, config.ANALYSIS_START)}** ({len(train):,}), "
                 f"validation **{span(val)}** ({len(val):,}), test **{span(test)}** "
                 f"({len(test):,}) — scored once, for models already chosen\n")
    except Exception:  # noqa: BLE001 - no frame: the configured cut-offs
        return (f"- Window from **{meta['window'][0]}**; train up to "
                f"**{meta['train_end']} 00:00**, validation up to **{meta['val_end']} 00:00**, "
                "test after it — scored once, for models already chosen\n"
                "- Chronological split; a customer-disjoint test subset checks the review "
                "model on buyers it never saw\n")
    # The split itself is by time only; customers are kept apart in a test subset
    # (features.make_splits, report Section 3), not in the test window as a whole.
    try:
        from features import make_splits

        n = int(make_splits(df)["n_straddling_customers"])
        return lines + (f"- Chronological split; a customer-disjoint test subset (the {n} "
                        "buyers with orders in both train and test left out, on "
                        "`customer_unique_id`) checks the review model on people it "
                        "never saw\n")
    except Exception:  # noqa: BLE001 - the subset is context, not a number the card needs
        return lines + "- Chronological split\n"


def render_model_card(df: pd.DataFrame | None = None) -> None:
    try:
        art = load_artifacts()
    except Exception as exc:  # noqa: BLE001
        st.error(f"**Model card unavailable:** {exc}")
        return

    meta = art["meta"]
    st.markdown(
        theme.section("Model card",
                      "What is deployed, on what it was trained, how well it does, how much "
                      "that varies, where it should not be trusted, what it changes for "
                      "customers, and where each number comes from. Tags say what kind of "
                      f"number a card shows: {theme.tag('observed')} counted in the data, "
                      f"{theme.tag('assumed')} no data source, {theme.tag('derived')} computed "
                      f"from the two, {theme.tag('planned')} a design choice."),
        unsafe_allow_html=True)

    observed = theme.tag("observed")
    st.markdown(
        theme.kpi_row([
            ("Lead-time MAE", f"{meta['p1a']['test_mae']:.2f} d",
             f"{observed} held-out test period", theme.PRIMARY),
            ("Late-flag ROC-AUC", f"{meta['p1b']['test_roc_auc']:.3f}",
             f"{observed} PR-AUC {meta['p1b']['test_pr_auc']:.3f} vs "
             f"{meta['p1b']['test_pr_auc_floor']:.3f} floor", theme.GOLD),
            ("Review-risk ROC-AUC", f"{meta['p2']['test_roc_auc']:.3f}",
             f"{observed} balanced acc {meta['p2']['test_balanced_accuracy']:.3f}", theme.GREEN),
            ("Review PR-AUC lift",
             f"{meta['p2']['test_pr_auc'] / meta['p2']['test_pr_auc_floor']:.1f}×",
             f"{observed} over the no-skill floor", theme.ACCENT),
        ]),
        unsafe_allow_html=True)

    rules = load_table("customer_promise_rules.csv")
    margin = (float(rules.loc[rules["rule"].str.startswith("R1"), "buffer_days"].iloc[0])
              if rules is not None else None)
    quote_rule = (f"quote the forecast plus the {margin:g}-day margin, rounded up to a "
                  "calendar day" if margin is not None else "quote with the forecast")

    # -- what is deployed, read from the artifacts themselves ---------------
    st.markdown("#### What is deployed")
    # A wrapping table: the cells are sentences, and a grid would cut them off.
    st.markdown(theme.table(pd.DataFrame([
        {"Problem": "P1A · lead time (regression, at checkout)",
         "Deployed model": meta["p1a"]["label"],
         "As pickled": _as_pickled(art["p1a"]),
         "Artifact": _artifact("p1a_leadtime.joblib"),
         "Decision rule": quote_rule,
         "Test": f"MAE {meta['p1a']['test_mae']:.2f} d · RMSE {meta['p1a']['test_rmse']:.2f} · "
                 f"R² {meta['p1a']['test_r2']:.3f}"},
        {"Problem": "P1B · late vs promise (classification, at checkout)",
         "Deployed model": meta["p1b"]["label"],
         "As pickled": _as_pickled(art["platt"])
                       + " on (predicted − promised days), fitted on validation",
         "Artifact": _artifact("p1b_margin_to_risk.joblib"),
         "Decision rule": f"flag when risk ≥ {meta['p1b']['threshold']:.2f}; a warning to the "
                          "customer only for the riskiest tenth, and only if the platform "
                          "keeps its own date",
         "Test": f"ROC {meta['p1b']['test_roc_auc']:.3f} · PR-AUC {meta['p1b']['test_pr_auc']:.3f} · "
                 f"balanced acc {meta['p1b']['test_balanced_accuracy']:.3f}"},
        {"Problem": "P2 · low review (classification, at delivery)",
         "Deployed model": meta["p2"]["label"],
         "As pickled": _as_pickled(art["p2"]),
         "Artifact": _artifact("p2_lowreview.joblib"),
         "Decision rule": f"flag when risk ≥ {meta['p2']['threshold']:.2f}; rank for the top-K",
         "Test": f"ROC {meta['p2']['test_roc_auc']:.3f} · PR-AUC {meta['p2']['test_pr_auc']:.3f} · "
                 f"F1 {meta['p2']['test_f1']:.3f}"},
    ]), bold=("Problem",)), unsafe_allow_html=True)
    st.caption(
        "The review model that is served is the StackingClassifier the pre-registered "
        "selection rule chose (best mean rank over the three most recent out-of-sample "
        "windows), built from the tuned logistic regression and the tuned "
        "HistGradientBoosting classifier. The single boosted model sits within 0.001 "
        "PR-AUC of it on the test window; the ensemble is kept because switching after "
        "seeing the test window would be selecting on the test set. Both artifacts are "
        "a few hundred kilobytes.")

    never_note = ""
    try:
        import journey

        jr = journey.load_journeys()
        never = jr["outcome"] == "never delivered"
        top10 = jr["low_rank_purchase_day"] <= 10
        never_note = (
            f"- **Some reviewed orders never arrived.** {100 * never.mean():.1f}% of the "
            "reviewed test orders were cancelled, unavailable or not yet delivered when the "
            "extract ends; their empty delivery fields make them easy to rank, and they are "
            f"{100 * never[top10].mean():.0f}% of the slides' daily top-ten picks. A call at "
            "delivery cannot reach these customers; the promised-date message can, and the "
            "Replay tab shows what each of them went through.\n")
    except Exception:  # noqa: BLE001 - the bullet is context, not a number the card needs
        pass

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("#### Training")
        st.markdown(
            _split_lines(df, meta)
            + f"- Seed **{meta['seed']}**, {meta['n_search_iter']} randomised search "
            f"draws per model (six for the random forests), five expanding-window folds\n"
            f"- Thresholds and the margin-to-risk mapping fitted on the validation period")
        hp = load_table("hyperparameters.csv")
        if hp is not None:
            with st.expander("Selected hyperparameters (every searched model and the deployed members)"):
                problem = st.selectbox("Problem", sorted(hp["problem"].unique()), index=2,
                                       key="hp_problem")
                show = hp[hp["problem"] == problem][["model", "role", "parameter", "value"]]
                st.dataframe(show, hide_index=True, use_container_width=True, height=320)
                st.caption("Searched with RandomizedSearchCV, scored by expanding-window "
                           "cross-validation; the deployed ensembles carry exactly the "
                           "winning members' values (asserted in tests/test_leakage.py).")
    with c2:
        st.markdown("#### Where not to trust this")
        st.markdown(
            "- **It is fitted on Brazil, 2017–2018.** Lead time halved across that "
            "window; a model trained on it is off-level on any other period and "
            "needs refitting on a rolling basis.\n"
            "- **Lateness is driven by shocks the model cannot see** — a Black Friday "
            "surge, a carrier-side slowdown in February–March 2018 that coincided "
            "with a tightened promise, and the May 2018 truckers' strike, visible "
            "only at weekly resolution. None is an attribute of an order, so no "
            "checkout-time feature can anticipate them.\n"
            "- **Review risk is scored at delivery.** Asked at checkout, the same "
            "target is far weaker, because the variable that matters most has not "
            "happened yet.\n"
            "- **The value of a contact is assumed, not measured.** The reais below "
            "count a contact as worth the 1–2 star rating it turns around; the tables "
            "record no contact other than the survey, so only a randomised pilot can "
            "measure that effect. The pilot calculator below sizes one.\n"
            "- **A new seller has no track record.** The history features fall back "
            "to route, then state, then platform, and the model is told so.\n"
            + never_note +
            "- **These are decision scores for triage, not entitlements.** They "
            "must not be used to penalise a seller or price a customer.")

    # -- fold-to-fold variation (Phase 2 feedback) ---------------------------
    folds = load_table("cv_folds.csv")
    if folds is not None:
        st.markdown("#### How much the candidates vary from fold to fold")
        st.caption(
            "Every Problem 2 candidate on the five expanding-window folds and the "
            "validation period, with the test score beside them. The fold-to-fold "
            "spread (about ±0.09) is a property of the windows — the mean over candidates "
            "runs from 0.376 on the first fold to 0.630 on the fourth — and it dwarfs "
            "the gap between the "
            "best candidates (under 0.001), which is why selection ranks models "
            "within each window rather than averaging scores across them.")
        p2 = folds[folds["problem"] == "P2 low review"].sort_values("mean_rank_recent")
        fold_cols = [f"fold{i}" for i in range(1, 6)]
        table = p2[["model"] + fold_cols + ["validation", "cv_mean", "cv_sd", "test",
                                            "mean_rank_recent"]].copy()
        table.columns = (["model"] + [f.replace("fold", "fold ") for f in fold_cols]
                         + ["validation", "CV mean", "fold sd", "test", "mean rank (recent 3)"])
        st.dataframe(table.round(3), hide_index=True, use_container_width=True)
        fig = go.Figure()
        windows = fold_cols + ["validation", "test"]
        for _, r in p2.iterrows():
            fig.add_scatter(x=[w.replace("fold", "fold ") for w in windows],
                            y=r[windows].astype(float).to_numpy(), mode="lines+markers",
                            name=r["model"], line=dict(width=3 if r["deployed"] else 1.5),
                            opacity=1.0 if r["deployed"] else 0.7)
        fig.update_layout(height=360, yaxis_title="PR-AUC (out of sample)",
                          title="Problem 2: every candidate on every window")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    # -- capacity ------------------------------------------------------------
    cap = load_capacity()
    if cap is not None and len(cap):
        st.markdown("#### What a ranked list delivers at a fixed capacity")
        st.caption(
            "Precision@K and Recall@K of the review-risk model on the held-out test "
            "window: the top K% of reviewed test orders by review risk, and the share "
            "whose customer went on to rate the order 1–2 stars (historical outcome). A "
            "team works the list from the top, so its capacity, not the model's "
            "threshold, limits what it reaches; whether a contact pays depends on the "
            "two assumed costs and on how many of those customers it can still reach, "
            "below. The Replay-a-day tab shows the list for any single test day.")
        show = (cap[cap["k_pct"].isin([1, 2, 5, 10, 15, 20, 30])]
                [["k_pct", "precision_at_k", "recall_at_k", "lift_at_k"]]
                .rename(columns={"k_pct": "capacity K (% of reviewed test orders)",
                                 "precision_at_k": "precision@K (observed)",
                                 "recall_at_k": "recall@K (observed)",
                                 "lift_at_k": "lift vs random"}))
        st.dataframe(show, hide_index=True, use_container_width=True, column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in show.columns[1:]})

    render_customer_outcomes(margin)
    render_value_and_pilot()
    render_provenance()

    st.markdown("#### Reproducibility")
    st.code("python tests/test_leakage.py       # validation checks, incl. deployed-model consistency\n"
            "python src/train_models.py         # refits everything and rewrites models/\n"
            "python src/capacity_metrics.py     # Precision@K / Recall@K from the saved predictions\n"
            "python src/selection_tables.py     # fold-to-fold variation and selected hyperparameters\n"
            "python src/pilot_design.py         # the pilot as designed on slide 11\n"
            "python src/review_timing.py        # when reviews are written, and who a call can still reach\n"
            "python src/customer_view.py        # the customer tables: journeys, outcomes, contacts, values, pilot arms\n"
            "python src/slide_provenance.py     # where every number on the slides comes from\n"
            "python src/phase3_stats.py         # regenerates every number in the final report",
            language="bash")
    st.caption(f"Artifacts built {art['p1a'].get('saved_at', 'n/a')} on "
               f"scikit-learn {art['p1a'].get('sklearn_version', 'n/a')}.")


# ===========================================================================
# UI - model card: customers, value, pilot, provenance
# ===========================================================================
def render_customer_outcomes(margin: float | None) -> None:
    """What the recommendations change for customers, counted in the data."""
    rules = load_table("customer_promise_rules.csv")
    outc = load_table("customer_outcomes_test.csv")
    lists = load_table("customer_contact_lists.csv")
    if rules is None or outc is None or lists is None:
        return
    test = rules[rules["period"] == "test"].set_index("rule")
    olist = test.loc["Olist's own promise"]
    r1 = test[test.index.str.startswith("R1")].iloc[0]
    o = outc.set_index("key")
    dd = lists.set_index("list").loc["delivery_day_k10"]
    obs, der = theme.tag("observed"), theme.tag("derived")

    st.markdown("#### What the recommendations change for customers")
    st.markdown(theme.kpi_row([
        ("Promised dates broken",
         f"{1000 * olist['n_late'] / olist['n']:.0f} → {1000 * r1['n_late'] / r1['n']:.0f}",
         f"{obs} today → {der} with the forecast plus the {margin:g}-day margin; out of "
         "every 1,000 delivered orders", theme.PRIMARY),
        ("No parcel when the day ended", f"{o.loc['overdue_eod', 'per_1000']:.1f}",
         f"{obs} per 1,000 orders placed: the customers a promised-date message reaches "
         "first; the tables record no message to them", theme.GOLD),
        ("Not yet rated then, later 1–2★", f"{o.loc['eod_low_not_yet_written', 'per_1000']:.1f}",
         f"{obs} per 1,000 orders placed: reached by that message before they rate",
         theme.ACCENT),
        ("Delivery-day top ten", f"{100 * dd['precision']:.0f} in 100",
         f"{obs} went on to rate 1–2★, against {100 * dd['pool_base_rate']:.0f} at random "
         f"({dd['lift']:.1f}×)", theme.GREEN),
    ]), unsafe_allow_html=True)

    tbl = outc[outc["key"] != "placed"]
    show = pd.DataFrame({
        "what the customers went through": tbl["outcome"].str.replace("1-2", "1–2", regex=False),
        "customers": [f"{int(v):,}" for v in tbl["orders"]],
        "out of": [f"{int(n):,} {'orders placed' if d == 'placed' else 'delivered orders'}"
                   for d, n in zip(tbl["denominator"], tbl["denominator_n"])],
        "per 1,000": tbl["per_1000"].round(1),
        "95% interval": [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(tbl["ci95_lo"], tbl["ci95_hi"])],
    })
    # 352 px: the header and nine whole rows (35 px each); the rest scroll.
    st.dataframe(show, hide_index=True, use_container_width=True, height=352,
                 column_config={"what the customers went through":
                                st.column_config.TextColumn(width="large"),
                                "per 1,000": st.column_config.NumberColumn(format="%.1f")})
    st.caption(
        "Counted in the data, with no assumption: every row names its population, the "
        f"{int(o.loc['placed', 'orders']):,} orders placed in the held-out window or the "
        f"{int(o.loc['prewarn_flagged', 'denominator_n']):,} delivered among them; Wilson "
        "95% intervals (customer_outcomes_test.csv). The broken-promise rate with the "
        "margin replays the rule on the same orders and assumes no customer would have "
        "acted differently on a later date (customer_promise_rules.csv).")


@theme.fragment
def render_value_and_pilot() -> None:
    """The two assumed costs as inputs, what each contact is worth under them, and the
    pilot that would measure the one number the data cannot show."""
    import config

    lists = load_table("customer_contact_lists.csv")
    value = load_table("customer_value.csv")
    arms = load_table("customer_pilot_arms.csv")
    if lists is None or value is None or arms is None:
        return
    L = lists.set_index("list")
    V = value.set_index(["list", "scope"])
    A = arms.set_index("arm")
    n_reviewed = int(L.loc["slides_ten_a_day", "pool_n"])

    st.markdown("#### What a contact is worth, and the pilot that would measure it")
    st.markdown(
        f'<div class="bw-sec"><p>{theme.tag("assumed")} Olist publishes no costs, so both '
        "reais figures below are assumptions (src/config.py COSTS); change them here. "
        f"Everything else on this panel is {theme.tag('observed')} on the held-out window "
        f"or {theme.tag('derived')} from the two, per 1,000 reviewed orders "
        f"({n_reviewed:,} reviewed test orders).</p></div>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 1])
    cost = float(c1.number_input("Cost of a contact, R$ (assumed)", min_value=0.0,
                                 max_value=500.0, value=float(config.COSTS["outreach_cost"]),
                                 step=0.5, key="mc_cost"))
    worth = float(c2.number_input("Value of a 1–2 star rating turned around, R$ (assumed)",
                                  min_value=1.0, max_value=5000.0,
                                  value=float(config.COSTS["low_review_cost"]), step=5.0,
                                  key="mc_worth"))

    rows = []
    for name, label, who in CONTACTS:
        allh = V.loc[(name, "all hits")]
        reach = V.loc[(name, "reachable at its moment")]
        contacts = float(reach["contacts"])
        per_1k = 1000 * contacts / n_reviewed
        p_reach = float(reach["precision"])
        be = cost / (worth * p_reach) if p_reach > 0 else np.inf

        def net(e: float) -> float:
            return 1000 * (e * float(reach["hits"]) * worth - contacts * cost) / n_reviewed

        rows.append({
            "contact": label, "who it reaches": who,
            "contacts per 1,000 reviewed orders (observed)": f"{per_1k:.1f}",
            "rated 1–2★, per 100 contacts (observed)": f"{100 * float(allh['precision']):.1f}",
            "of whom not yet rated at that moment, per 100 contacts (observed)":
                f"{100 * p_reach:.1f}",
            "break-even share turned around (derived)":
                f"{100 * be:.1f}%" + (" (cannot pay)" if be > 1 else ""),
            "net R$ if all are turned around (derived)": _signed(net(1.0), 0).lstrip("+"),
            "net R$ if one in two (derived)": _signed(net(0.5), 0).lstrip("+"),
            "net R$ if one in four (derived)": _signed(net(0.25), 0).lstrip("+"),
        })
    value_tbl = pd.DataFrame(rows)
    st.markdown(theme.table(value_tbl, numeric=tuple(value_tbl.columns[2:]), bold=("contact",),
                            widths={"contact": "10rem", "who it reaches": "15rem"}),
                unsafe_allow_html=True)
    st.caption(
        "A contact can only turn around a customer who has not yet rated the order, so "
        "each value counts the customers still reachable at that contact's moment; the "
        "net is per 1,000 reviewed orders, at three shares of them turned around. The "
        "slides' upper value counted every listed customer who rated 1–2 stars; their "
        "lower values left out only those who rated a late parcel before it arrived, so "
        "they still count every such customer whose order never came (customer_value.csv "
        "keeps all three scopes).")

    # -- the pilot ---------------------------------------------------------------
    st.markdown("##### Sizing the pilot")
    try:
        import pilot_design as PD

        label = st.radio("Pilot arm", list(ARMS), horizontal=True, key="pilot_arm")
        arm, list_name = ARMS[label]
        if arm == "arm_b":
            three = st.radio("Arms", ["Two: the message against current practice",
                                      "Three: also separates the survey timing"],
                             horizontal=True, key="pilot_b_arms")
            arm = "arm_b_three" if three.startswith("Three") else "arm_b_two"
        a = A.loc[arm]
        st.caption(f"Who is randomised: {a['population']}. Two-sided two-proportion test "
                   "at alpha = 0.05; the primary outcome is each randomised customer's "
                   "final rating.")
        k1, k2 = st.columns([1.4, 1])
        effect = float(k1.number_input(
            "Effect to detect: fewer 1–2★ ratings among those contacted (%)",
            min_value=1.0, max_value=95.0, value=float(100 * a["target_relative"]), step=1.0,
            format="%.1f", key=f"pilot_effect_{arm}")) / 100
        power = k2.radio("Power", ["80%", "90%"], horizontal=True, key="pilot_power")
        p0 = float(a["control_rate"])
        p1 = p0 * (1 - effect)
        n = PD.n_per_arm(p0, p1, power=0.8 if power == "80%" else 0.9)
        per_day = float(a["per_day"])
        n_arms = int(a["arms"])
        weeks = int(np.ceil(n_arms * n / per_day / 7))
        net_contact = (p0 - p1) * worth - cost
        if abs(net_contact) < 0.005:   # an arm sized at its break-even nets nothing
            net_contact = 0.0
        be_rel = cost / (worth * p0)
        per_1k = float(L.loc[list_name, "contacts_per_1k_reviewed"])
        per_txt = f"{per_day:.0f}" if abs(per_day - round(per_day)) < 1e-9 else f"{per_day:.1f}"
        st.markdown(theme.kpi_row([
            ("Rate without contact", f"{100 * p0:.1f}%",
             f"{theme.tag('observed')} rated 1–2★ on the test window; a historical rate, "
             "not a control group", theme.PRIMARY),
            ("Customers per arm", f"{n:,.0f}",
             f"{theme.tag('planned')} {n_arms} arms; about {weeks} week"
             f"{'s' if weeks != 1 else ''} at {per_txt} listed a day", theme.GOLD),
            ("Net per contact", f"R&#36; {_signed(net_contact, 2)}",
             f"{theme.tag('derived')} at a {100 * effect:.1f}% effect and the assumed costs",
             theme.GREEN if net_contact > 0 else theme.ACCENT),
            ("Break-even effect", f"{100 * be_rel:.1f}%",
             f"{theme.tag('derived')} the reduction a contact needs to pay", theme.ACCENT),
            ("If rolled out", f"R&#36; {_signed(per_1k * net_contact, 0)}",
             f"{theme.tag('derived')} per 1,000 reviewed orders: {per_1k:.1f} contacts × "
             "net per contact", theme.GREEN if net_contact > 0 else theme.ACCENT),
        ]), unsafe_allow_html=True)
        if arm == "arm_a":
            d = L.loc["delivery_day_k10"]
            real_weeks = int(np.ceil(n_arms * n / float(d["contacts_per_day"]) / 7))
            st.caption(
                f"On the test window the delivery-day list held ten customers on "
                f"{int(d['full_days'])} of {int(d['days'])} delivery days "
                f"({float(d['contacts_per_day']):.1f} a day on average); at that rate the "
                f"pilot needs about {real_weeks} weeks.")
        if arm == "slide_design":
            s = L.loc["slides_ten_a_day"]
            ceiling = 100 * (float(s["hits_delivery_only"]) + float(s["hits_either"])) \
                / float(s["contacts"])
            rel = ceiling / (100 * p0)
            st.warning(
                f"On this list a delivery-day call can change at most {ceiling:.1f} of the "
                f"{100 * p0:.1f} points ({100 * rel:.0f}% relative), "
                f"{'below' if rel < be_rel else 'above'} the {100 * be_rel:.1f}% break-even.")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Pilot calculator unavailable: `{type(exc).__name__}: {exc}`")


def _dec(value, decimals: int = 1) -> str:
    """A number with a true minus sign when it is negative ('−1.0', not '-1.0')."""
    return f"{float(value):,.{decimals}f}".replace("-", "−")


def _signed(value: float, decimals: int) -> str:
    """'+0.47' or '−71', with thousands separators; a value that rounds to zero
    prints with no sign ('0.00'), so a break-even arm does not read '+0.00' or '-0.00'."""
    text = f"{abs(value):,.{decimals}f}"
    if float(text.replace(",", "")) == 0:
        return text
    return ("+" if value > 0 else "−") + text


def render_provenance() -> None:
    """Every number on the locked slides, with its kind and source (slide_provenance.csv)."""
    prov = load_table("slide_provenance.csv")
    if prov is None:
        return
    with st.expander("Where the numbers on the slides come from"):
        slides = [11] + sorted(int(s) for s in prov["slide"].unique() if int(s) != 11)
        options = [f"Slide {s}" for s in slides] + ["All slides"]
        pick = st.selectbox("Slide", options, key="prov_slide")
        rows = prov if pick == "All slides" else prov[prov["slide"] == int(pick.split()[-1])]
        show = pd.DataFrame({
            "slide": rows["slide"].astype(int),
            "on the slide": rows["on_slide"],
            "kind": [theme.tag(k) for k in rows["kind"]],
            "exact value": rows["value"].fillna("—"),
            "source or formula": rows["source"].fillna("—"),
            "population and window": rows["window"].fillna("—"),
            "correction or reading": rows["correction"].fillna("—"),
        })
        # Wrapped, not a grid: the sources and corrections are sentences, and a grid
        # would cut every one of them off at the column edge.
        st.markdown(theme.table(show, muted=("slide",), bold=("on the slide",),
                                html_cols=("kind",),
                                widths={"on the slide": "11rem", "exact value": "6rem",
                                        "population and window": "12rem",
                                        "correction or reading": "16rem"},
                                max_height="44rem" if pick == "All slides" else None),
                    unsafe_allow_html=True)
        kinds = prov["kind"].value_counts()
        checked = int(prov["matches"].astype(str).str.lower().eq("true").sum())
        st.caption(
            f"{len(prov)} numbers and phrases on the {prov['slide'].nunique()} slides as "
            "locked on 4 November 2026 ("
            + ", ".join(f"{v} {k}" for k, v in kinds.items())
            + f"); {checked} numeric values were checked against the result files when "
            "src/slide_provenance.py wrote report/results/slide_provenance.csv. The "
            "report's slide-provenance appendix prints the same table.")
