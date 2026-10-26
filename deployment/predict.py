"""Model serving for the Olist dashboard - the Phase 2 proof-of-concept.

Two tabs: a single-order scoring form, and a model card.

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
import streamlit as st

import theme

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

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
    and the memory. Loaded lazily, so a visitor who never opens this tab never
    pays for scikit-learn touching disk.
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

    Distance is one of the strongest checkout-time features (r = 0.40 with lead
    time), but it comes from postcode centroids the form does not ask for. The
    median for the chosen state pair is the honest stand-in, and the overall
    median covers a pair the data has never seen.
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
        f = frame.copy()
        f["delivery_days"] = out["predicted_days"]
        f["delay_vs_estimate_days"] = out["margin_days"]
        f["is_late"] = out["late_flag"].astype(float)
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
            "Enter an order as it would look at checkout. The model returns the "
            "lead time it expects, the risk of missing the promised date, and - "
            "under that forecast - the risk of a 1–2 star review."),
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

    if not submitted:
        st.info("Fill the form and press **Predict**. The defaults describe a "
                "typical São Paulo → Rio order in July 2018.")
        render_batch(df, art)
        return

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
        result = score(frame, art, at_delivery=True).iloc[0]
    except Exception as exc:  # noqa: BLE001
        st.error(f"**Could not score this order:** `{type(exc).__name__}: {exc}`")
        render_batch(df, art)
        return

    mae = meta["p1a"]["test_mae"]
    late_thr = meta["p1b"]["threshold"]
    low_thr = meta["p2"]["threshold"]
    st.markdown(
        theme.kpi_row([
            ("Predicted lead time", f"{result['predicted_days']:.1f} d",
             f"±{mae:.1f} d typical error", theme.PRIMARY),
            ("Promised", f"{result['promised_days']:.0f} d",
             f"margin {result['margin_days']:+.1f} d", theme.GOLD),
            ("Risk of missing the promise", f"{100*result['p_late']:.0f}%",
             "flagged" if result["late_flag"] else "not flagged", theme.ACCENT),
            ("Risk of a 1–2 star review", f"{100*result['p_low_review']:.0f}%",
             "flagged" if result["low_review_flag"] else "not flagged", theme.ACCENT),
        ]),
        unsafe_allow_html=True)

    # theme.note() emits raw HTML, so this has to be HTML - markdown asterisks
    # would render literally.
    verdict = ("<b>Flag for operations.</b> " if result["late_flag"]
               else "<b>No action needed.</b> ")
    st.markdown(
        theme.note(
            "What to do with this",
            verdict
            + (f"The model expects this parcel to arrive "
               f"{abs(result['margin_days']):.1f} days "
               f"{'after' if result['margin_days'] > 0 else 'before'} the promised "
               f"date. Orders are flagged above a {100*late_thr:.0f}% risk, an "
               "operating point chosen on the validation period, not on the test "
               "window.<br><br>"
               f"The review risk shown is conditional on that forecast, so it "
               f"inherits the forecast's error. Once the parcel actually lands, "
               f"re-score with the realised delivery date - that is the "
               f"at-delivery moment the model was measured at "
               f"(ROC-AUC {meta['p2']['test_roc_auc']:.2f}, flagging above "
               f"{100*low_thr:.0f}%).")),
        unsafe_allow_html=True)

    render_batch(df, art)


# ===========================================================================
# UI - batch
# ===========================================================================
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

    try:
        frame = assemble(raw, art["lookup"], route_distances(df))
        result = score(frame, art, at_delivery=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"**Could not score that file:** `{type(exc).__name__}: {exc}`. "
                 "Check that dates parse and that the numeric columns hold numbers.")
        return

    scored = pd.concat([raw.reset_index(drop=True),
                        result.reset_index(drop=True).round(4)], axis=1)
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
            f"test window, acting on the top {row['k_pct']:g}% of deliveries reached a "
            f"customer who would have left a 1–2 star review "
            f"{100*row['precision_at_k']:.0f}% of the time (precision@K) and covered "
            f"{100*row['recall_at_k']:.0f}% of all such reviews (recall@K), "
            f"{row['lift_at_k']:.1f}× better than contacting at random. The review "
            f"risk here is conditional on the forecast delivery; in production, "
            f"re-score at delivery with the realised dates.")
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
def render_model_card() -> None:
    try:
        art = load_artifacts()
    except Exception as exc:  # noqa: BLE001
        st.error(f"**Model card unavailable:** {exc}")
        return

    meta = art["meta"]
    st.markdown(
        theme.section("Model card",
                      "What was trained, on what, how well it does, and where it "
                      "should not be trusted."),
        unsafe_allow_html=True)

    st.markdown(
        theme.kpi_row([
            ("Lead-time MAE", f"{meta['p1a']['test_mae']:.2f} d",
             "held-out test period", theme.PRIMARY),
            ("Late-flag ROC-AUC", f"{meta['p1b']['test_roc_auc']:.3f}",
             f"PR-AUC {meta['p1b']['test_pr_auc']:.3f} vs "
             f"{meta['p1b']['test_pr_auc_floor']:.3f} floor", theme.GOLD),
            ("Review-risk ROC-AUC", f"{meta['p2']['test_roc_auc']:.3f}",
             f"balanced acc {meta['p2']['test_balanced_accuracy']:.3f}", theme.GREEN),
            ("Review PR-AUC lift",
             f"{meta['p2']['test_pr_auc'] / meta['p2']['test_pr_auc_floor']:.1f}×",
             "over the no-skill floor", theme.ACCENT),
        ]),
        unsafe_allow_html=True)

    st.markdown("#### Selected models")
    st.dataframe(pd.DataFrame([
        {"Problem": "P1A · lead time (regression, at checkout)",
         "Model": meta["p1a"]["label"], "Family": meta["p1a"]["family"],
         "Selected on": f"CV MAE {meta['p1a']['cv_mae']:.3f}",
         "Test": f"MAE {meta['p1a']['test_mae']:.2f} d · "
                 f"RMSE {meta['p1a']['test_rmse']:.2f} · R² {meta['p1a']['test_r2']:.3f}"},
        {"Problem": "P1B · late vs promise (classification, at checkout)",
         "Model": meta["p1b"]["label"], "Family": meta["p1b"]["route"],
         "Selected on": "PR-AUC against the direct classifiers",
         "Test": f"ROC {meta['p1b']['test_roc_auc']:.3f} · "
                 f"PR-AUC {meta['p1b']['test_pr_auc']:.3f} · "
                 f"balanced acc {meta['p1b']['test_balanced_accuracy']:.3f}"},
        {"Problem": "P2 · low review (classification, at delivery)",
         "Model": meta["p2"]["label"], "Family": meta["p2"]["family"],
         "Selected on": f"CV PR-AUC {meta['p2']['cv_pr_auc']:.3f}",
         "Test": f"ROC {meta['p2']['test_roc_auc']:.3f} · "
                 f"PR-AUC {meta['p2']['test_pr_auc']:.3f} · "
                 f"F1 {meta['p2']['test_f1']:.3f}"},
    ]), hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Chosen hyperparameters")
        st.json({"P1A " + meta["p1a"]["label"]: meta["p1a"]["params"],
                 "P2 " + meta["p2"]["label"]: meta["p2"]["params"]}, expanded=False)
        st.markdown("#### Training")
        st.markdown(
            f"- Window **{meta['window'][0]} → {meta['window'][1]}**\n"
            f"- Train to **{meta['train_end']}**, validation to **{meta['val_end']}**, "
            f"test thereafter\n"
            f"- Chronological split, disjoint on `customer_unique_id`\n"
            f"- Seed **{meta['seed']}**, {meta['n_search_iter']} randomised search "
            f"draws per model, expanding-window CV")
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
            "- **A new seller has no track record.** The history features fall back "
            "to route, then state, then platform, and the model is told so.\n"
            "- **These are decision scores for triage, not entitlements.** They "
            "must not be used to penalise a seller or price a customer.")

    cap = load_capacity()
    if cap is not None and len(cap):
        st.markdown("#### What a ranked list delivers at a fixed capacity")
        st.caption(
            "Precision@K and Recall@K of the review-risk model on the held-out test "
            "window. A customer-experience team that can contact K% of the day's "
            "deliveries works the list from the top; precision stays above the "
            "break-even of 0.125 at every capacity shown, so the team's capacity, "
            "not the model's threshold, is what limits the value. Added after the "
            "Milestone 1 feedback; see report Section 4.4.")
        show = (cap[cap["k_pct"].isin([1, 2, 5, 10, 15, 20, 30])]
                [["k_pct", "precision_at_k", "recall_at_k", "lift_at_k"]]
                .rename(columns={"k_pct": "capacity K (% of deliveries)",
                                 "precision_at_k": "precision@K",
                                 "recall_at_k": "recall@K",
                                 "lift_at_k": "lift vs random"}))
        st.dataframe(show.round(3), hide_index=True, use_container_width=True)

    st.markdown("#### Reproducibility")
    st.code("python src/train_models.py      # refits everything and rewrites models/\n"
            "python src/capacity_metrics.py  # Precision@K / Recall@K from the saved predictions\n"
            "python tests/test_leakage.py    # validation checks, incl. the history features\n"
            "python src/phase2_stats.py      # regenerates every number in the report",
            language="bash")
    st.caption(f"Artifacts built {art['p1a'].get('saved_at', 'n/a')} on "
               f"scikit-learn {art['p1a'].get('sklearn_version', 'n/a')}.")
