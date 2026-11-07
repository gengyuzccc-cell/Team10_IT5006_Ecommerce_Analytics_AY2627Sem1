"""Replay a held-out test day: the deployed model's list, first blind, then revealed.

The brief notes that, because the Olist extract ends in 2018, a deployed app
predicts on held-out historical records that stand in for new orders. This tab
makes that literal. The test window - orders placed 30 June to 30 August 2018, 62
purchase days - was never used to fit, tune or threshold anything. The per-order
predictions src/train_models.py wrote for it are read back here one day at a time
and ranked by the deployed model's risk that the customer will rate the order
1-2 stars.

Every day opens BLIND: the list shows no outcome, only the scores, the ranks
and each order's details, and its download holds only
those columns (blind_columns()). On the purchase-day list that leaves out the
checkout forecast and late risk as well: they were saved for delivered orders
only, so a blank cell would show which customers never received their parcel.
Changing the day, the capacity or the list turns the reveal off again, so each
day starts blind, as on slides 9 and 13. Switching on "Reveal what actually
happened" then adds each customer's rating, what they went through, which contact
could still have reached them, and the checkout columns, so the list can be
checked against history instead of taken on trust.

The checkout cards of a purchase day (the share flagged at the 11% line, the
mean forecast, and once revealed the lead-time error) cover every delivered
order bought that day, reviewed or not, as the slide-provenance table of the
report does (238 orders on 16 July 2018); on the delivery-day list they cover
the parcels on the list.

Two lists:
  Orders bought that day (as on the slides): every reviewed order of one purchase
      day, scored with its realised delivery.
  Parcels delivered that day, not yet rated (as a team would work it): each
      delivery day's parcels whose customer had not yet rated the order.

Nothing is refitted or re-scored: p_low_review is the deployed stacking model's
score at delivery, p_late the derived late risk of the lead-time regression and
pred_days its forecast (predictions_p2/p1a/p1b_test.csv). Outcomes, ranks and
reach classes come from report/results/customer_journeys_test.csv
(src/customer_view.py), so the tab uses the report's definitions; pooled over the
window, a list's precision is the arithmetic of customer_contact_lists.csv.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import journey
import theme

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "report" / "results"
PLOT_CFG = {"displayModeBar": False, "responsive": True}

# The journey columns the replay needs: outcomes, reach and the lists' own ranks.
JOURNEY_COLUMNS = ["order_id", "customer_city", "order_status", "outcome", "review_score",
                   "days_late", "delivered", "answered", "written_before_arrival",
                   "overdue_eod", "promise_reach", "delivery_reach", "reach_class",
                   "low_rank_purchase_day", "low_rank_delivery_day"]

# What the team would see when the list is drawn up: the blind view and its download
# hold these columns only.
BLIND_COLUMNS = ("rank", "in_top_k", "list_day", "order_id", "order_purchase_timestamp",
                 "customer_city", "customer_state", "customer_region", "lead_category",
                 "estimated_days", "p_low_review", "low_review_flag", "p_late", "late_flag",
                 "pred_days")
# The checkout forecast and late risk were saved for delivered orders only
# (predictions_p1a/p1b_test.csv), so on a list that can hold an order that never
# arrived, a blank in these columns would give the outcome away.
CHECKOUT_COLUMNS = ("p_late", "late_flag", "pred_days")
# What happened after it: never in the blind view or its download.
OUTCOME_COLUMNS = ("actual_low_review", "actual_days", "actual_late", "written_before_arrival",
                   "review_score", "outcome", "days_late", "delivered", "delivered_day",
                   "answered", "order_status", "overdue_eod", "promise_reach",
                   "delivery_reach", "reach_class", "low_rank_delivery_day")

LIST_PURCHASE = "Orders bought that day (as on the slides)"
LIST_DELIVERY = "Parcels delivered that day, not yet rated (as a team would work it)"
# Per list: the day the list belongs to, its rank, and the orders it can hold.
LISTS = {
    LIST_PURCHASE: dict(day="purchase_day", rank="low_rank_purchase_day", pool=None,
                        days="purchase days", slug="bought", day_label="Day the customers ordered"),
    LIST_DELIVERY: dict(day="delivered_day", rank="low_rank_delivery_day",
                        pool="delivery_reach", days="delivery days", slug="delivered",
                        day_label="Day the parcels were delivered"),
}


def blind_columns(columns=None, delivered_only: bool = False) -> list[str]:
    """The columns a blind list may show, in their order, before anything is
    revealed. Every outcome column (OUTCOME_COLUMNS: the rating, the
    delivery, when the review was written, the reach class) is left out, and so are
    the checkout columns (CHECKOUT_COLUMNS) unless every order on the list was
    delivered (delivered_only, the delivery-day list): elsewhere they are blank
    exactly for the orders that never arrived. With no argument, the blind set of
    the purchase-day list."""
    present = set(BLIND_COLUMNS if columns is None else columns)
    hidden = set(OUTCOME_COLUMNS) | (set() if delivered_only else set(CHECKOUT_COLUMNS))
    return [c for c in BLIND_COLUMNS if c in present and c not in hidden]


# ===========================================================================
# Data
# ===========================================================================
@st.cache_data(show_spinner="Reading the held-out test predictions…")
def load_replay() -> tuple[pd.DataFrame, dict]:
    """The three per-order prediction tables of the test window, joined on order,
    with each customer's outcome and reach class from customer_journeys_test.csv."""
    meta = json.loads((PROJECT_ROOT / "models" / "model_metadata.json").read_text("utf-8"))
    p2 = pd.read_csv(RESULTS / "predictions_p2_test.csv",
                     parse_dates=["order_purchase_timestamp"])
    p1a = pd.read_csv(RESULTS / "predictions_p1a_test.csv")[["order_id", "actual_days", "pred_days"]]
    p1b = pd.read_csv(RESULTS / "predictions_p1b_test.csv")[["order_id", "actual_late", "p_late_derived"]]
    jr = journey.load_journeys()[JOURNEY_COLUMNS]
    df = (p2.merge(p1a, on="order_id", how="left", validate="one_to_one")
            .merge(p1b, on="order_id", how="left", validate="one_to_one")
            .merge(jr, on="order_id", how="left", validate="one_to_one")
            .rename(columns={"p_late_derived": "p_late"}))
    df["purchase_day"] = df["order_purchase_timestamp"].dt.date
    df["delivered_day"] = df["delivered"].dt.date
    # Never-delivered orders carry no late risk: their flag is missing, not zero.
    df["late_flag"] = ((df["p_late"] >= meta["p1b"]["threshold"]).astype("Int64")
                       .where(df["p_late"].notna()))
    df["low_review_flag"] = (df["p_low_review"] >= meta["p2"]["threshold"]).astype(int)
    return df, meta


@st.cache_data(show_spinner=False)
def load_checkout() -> pd.DataFrame:
    """Every delivered order of the test window, reviewed or not, with its checkout
    forecast and late risk: what the checkout cards of a purchase day summarise."""
    p1a = pd.read_csv(RESULTS / "predictions_p1a_test.csv",
                      parse_dates=["order_purchase_timestamp"])
    p1b = pd.read_csv(RESULTS / "predictions_p1b_test.csv")[["order_id", "p_late_derived"]]
    co = (p1a.merge(p1b, on="order_id", how="inner", validate="one_to_one")
             .rename(columns={"p_late_derived": "p_late"}))
    co["purchase_day"] = co["order_purchase_timestamp"].dt.date
    return co


def list_pool(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """The orders a list can hold: every reviewed order, or the delivery-day pool."""
    return df if spec["pool"] is None else df[df[spec["pool"]].fillna(False).astype(bool)]


@st.cache_data(show_spinner=False)
def daily_top_k(_df: pd.DataFrame, k: int, list_name: str) -> pd.DataFrame:
    """Contacts, hits and precision of the top-K list on every day of the window, and
    how many of the hits had not yet rated the order when the parcel arrived."""
    spec = LISTS[list_name]
    pool = list_pool(_df, spec)
    rows = []
    for day, g in pool.groupby(spec["day"]):
        top = g[g[spec["rank"]] <= k]
        positives = int(g["actual_low_review"].sum())
        hit = top["actual_low_review"] == 1
        tp = int(hit.sum())
        at_delivery = int((hit & top["delivery_reach"].fillna(False).astype(bool)).sum())
        rows.append({"day": pd.Timestamp(day), "orders": len(g), "low_reviews": positives,
                     "contacted": len(top), "hits": tp, "hits_at_delivery": at_delivery,
                     "precision": tp / len(top) if len(top) else np.nan})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def delivery_day_reference() -> dict | None:
    """How the delivery-day list did over the window (customer_contact_lists.csv)."""
    path = RESULTS / "customer_contact_lists.csv"
    if not path.exists():
        return None
    row = pd.read_csv(path).set_index("list").loc["delivery_day_k10"]
    return {"precision": float(row["precision"]), "lift": float(row["lift"])}


def _hide_outcomes() -> None:
    """A new day, capacity or list starts blind."""
    st.session_state["replay_reveal"] = False


# ===========================================================================
# UI
# ===========================================================================
@theme.fragment
def render_replay() -> None:
    try:
        df, meta = load_replay()
        co = load_checkout()
    except FileNotFoundError as exc:
        st.error(f"**The test-window files are missing.** {exc}. Run "
                 "`python src/train_models.py` for `report/results/predictions_*.csv` and "
                 "`python src/customer_view.py` for `customer_journeys_test.csv`.")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"**Could not load the replay data:** `{type(exc).__name__}: {exc}`")
        return

    late_line = f"{100 * meta['p1b']['threshold']:.0f}%"
    st.markdown(
        theme.section(
            "Replay a day from the held-out test window",
            "Every customer who ordered that day, ranked by the deployed model's risk that "
            "they will rate the order 1–2 stars: first blind, then revealed. Orders placed "
            f"{journey.window_text(df['order_purchase_timestamp'])}; none was used to fit, "
            "tune or threshold anything."),
        unsafe_allow_html=True)

    which = st.radio("Which list?", list(LISTS), horizontal=True, key="replay_list",
                     on_change=_hide_outcomes)
    spec = LISTS[which]
    pool = list_pool(df, spec)
    days = sorted(d for d in pool[spec["day"]].dropna().unique())
    c1, c2, c3 = st.columns([1.2, 1, 1.3])
    with c1:
        default = pd.Timestamp("2018-07-16").date()
        day = st.date_input(spec["day_label"], value=default if default in days else days[0],
                            min_value=days[0], max_value=days[-1],
                            key=f"replay_day_{spec['slug']}", on_change=_hide_outcomes)
    with c2:
        k = int(st.number_input("Customers the team can contact a day (K)", min_value=1,
                                max_value=60, value=10, step=1, key="replay_k",
                                on_change=_hide_outcomes))
    with c3:
        st.write("")
        reveal = st.toggle("Reveal what actually happened", value=False, key="replay_reveal")

    g = (pool[pool[spec["day"]] == day].sort_values(spec["rank"], kind="stable")
         .reset_index(drop=True))
    if g.empty:
        st.warning("**No customer is on this list that day.** Try a weekday in July or the "
                   "first half of August; on the delivery list, Sundays carry no deliveries.")
        return

    n = len(g)
    k_eff = min(k, n)
    g.insert(0, "rank", np.arange(1, n + 1))
    g.insert(1, "in_top_k", g["rank"] <= k_eff)
    g["list_day"] = day
    top = g[g["in_top_k"]]
    hit = top[top["actual_low_review"] == 1]
    h = len(hit)
    positives = int(g["actual_low_review"].sum())
    # The checkout cards: on a purchase day, every delivered order bought that day
    # (reviewed or not: the population of the slide-provenance table); on the
    # delivery-day list, the parcels on the list.
    purchase_list = spec["pool"] is None
    if purchase_list:
        delivered = co[co["purchase_day"] == day]
    else:
        delivered = g[g["p_late"].notna()]
    n_deliv = len(delivered)
    scope = (f"all {n_deliv} delivered orders bought that day"
             if purchase_list else f"the {n_deliv} parcels on this list")

    if purchase_list:
        first = ("Customers who ordered that day", f"{n:,}",
                 f"{int((g['outcome'] == 'never delivered').sum())} never received their parcel",
                 theme.PRIMARY)
    else:
        rated = int(((df["delivered_day"] == day) & ~df["delivery_reach"].fillna(False)
                     .astype(bool)).sum())
        first = ("Parcels delivered, not yet rated", f"{n:,}",
                 (f"{rated} more customer{'s' if rated != 1 else ''} whose parcel came that "
                  f"day had already rated it" if rated else
                  "every customer whose parcel came that day was still to rate it"),
                 theme.PRIMARY)
    cards = [first,
             ("Customers the team can contact (K)", f"{k_eff}",
              f"{100 * k_eff / n:.1f}% of the customers on this list", theme.GOLD)]
    if reveal:
        w = int((hit["outcome"].str.startswith("late") & hit["written_before_arrival"]
                 .fillna(False).astype(bool)).sum())
        nv = int((hit["outcome"] == "never delivered").sum())
        rc = int(hit["delivery_reach"].fillna(False).astype(bool).sum())
        pm = int(hit["promise_reach"].fillna(False).astype(bool).sum())
        cards.append(("Unhappy customers found", f"{h} of {k_eff}",
                      f"{w} wrote while waiting for a late parcel · {nv} never received it · "
                      f"{rc} could still be helped at delivery · {pm} could have been reached "
                      "when their promised day ended", theme.ACCENT))
        if n_deliv:
            mae = float((delivered["pred_days"] - delivered["actual_days"]).abs().mean())
            off = float((delivered["estimated_days"] - delivered["actual_days"]).abs().mean())
            cards.append(("Lead-time error that day", f"{mae:.1f} d",
                          f"mean |predicted − actual| over {scope}; the platform's promise "
                          f"was off by {off:.1f} d", theme.GREEN))
        else:
            cards.append(("Lead-time error that day", "—", "no delivered orders that day",
                          theme.GREEN))
    else:
        if n_deliv:
            flagged = float((delivered["p_late"] >= meta["p1b"]["threshold"]).mean())
            cards += [
                ("Flagged at checkout", f"{100 * flagged:.0f}%",
                 f"late risk at or above the {late_line} flag line, among {scope}"
                 + (" (reviewed or not)" if purchase_list else ""),
                 theme.ACCENT),
                ("Mean predicted lead time", f"{delivered['pred_days'].mean():.1f} d",
                 f"against a promised {delivered['estimated_days'].mean():.0f} d for the "
                 "same orders", theme.GREEN),
            ]
        else:
            cards += [("Flagged at checkout", "—", "no delivered orders that day", theme.ACCENT),
                      ("Mean predicted lead time", "—", "no delivered orders that day",
                       theme.GREEN)]
    st.markdown(theme.kpi_row(cards), unsafe_allow_html=True)

    # -- the ranked list -------------------------------------------------------
    # Orders that never reached the customer carry a review-risk score but no
    # lead-time forecast or outcome; show a dash rather than a misleading number.
    def days_col(col: pd.Series) -> list[str]:
        return [f"{v:.1f}" if pd.notna(v) else "—" for v in col]

    show = pd.DataFrame({
        "rank": g["rank"], "in top-K": g["in_top_k"],
        "review risk": 100 * g["p_low_review"],
        "late risk": 100 * g["p_late"],
        "predicted days": days_col(g["pred_days"]),
        "promised days": g["estimated_days"].round(0),
        "customer": [journey.place(c, s) for c, s in zip(g["customer_city"], g["customer_state"])],
        "region": g["customer_region"], "category": g["lead_category"],
        "order": g["order_id"].str[:8] + "…",
    })
    if purchase_list and not reveal:
        # Blind, a purchase day's list shows no checkout forecast or late risk: they
        # were saved for delivered orders only, so their blanks would mark the
        # customers who never received the parcel (CHECKOUT_COLUMNS).
        show = show.drop(columns=["late risk", "predicted days"])
    cfg = {
        "rank": st.column_config.NumberColumn(width="small"),
        "in top-K": st.column_config.CheckboxColumn("in top-K", width="small"),
        "review risk": st.column_config.ProgressColumn("review risk", format="%.1f%%",
                                                       min_value=0, max_value=100,
                                                       width="medium"),
        "late risk": st.column_config.ProgressColumn("late risk at checkout", format="%.1f%%",
                                                     min_value=0, max_value=100,
                                                     width="small" if reveal else "medium"),
        "promised days": st.column_config.NumberColumn(format="%.0f"),
    }
    if reveal:
        # The grid does not wrap, so the revealed columns use the compact wording, and
        # the customer moves up beside them so each row still says whose order it is.
        show.insert(3, "customer", show.pop("customer"))
        show.insert(4, "stars", g["review_score"])
        show.insert(5, "what the customer experienced",
                    [journey.experienced(r, short=True) for r in g.itertuples()])
        show.insert(6, "could still be reached by",
                    [journey.REACH_PHRASE.get(c, "—") for c in g["reach_class"]])
        show.insert(7, "actual days", days_col(g["actual_days"]))
        show["late risk"] = [f"{100 * v:.1f}%" if pd.notna(v) else "—" for v in g["p_late"]]
        cfg["late risk"] = st.column_config.TextColumn("late risk", width="small")
        cfg["stars"] = st.column_config.NumberColumn("stars", format="%d ★", width="small")
        cfg["what the customer experienced"] = st.column_config.TextColumn(width="large")
        cfg["could still be reached by"] = st.column_config.TextColumn(width="medium")
    st.dataframe(show, use_container_width=True, hide_index=True, height=380,
                 column_config={c: v for c, v in cfg.items() if c in show.columns})
    if reveal:
        st.caption(
            f"Of the {positives} customers on this list who went on to rate the order 1–2 "
            f"stars, the top {k_eff} held {h}"
            + (f" ({100 * h / positives:.0f}%)." if positives else ".")
            + " Reach: whether the promised-date message or a call at delivery would have "
              "come before the customer rated the order.")

    buf = io.StringIO()
    out = g.drop(columns=["purchase_day", "delivered_day"], errors="ignore")
    (out if reveal else out[blind_columns(out.columns, delivered_only=not purchase_list)]
     ).to_csv(buf, index=False)
    st.download_button("Download this day's list" + (" with the outcomes" if reveal
                                                     else " (blind: no outcomes)"),
                       buf.getvalue(),
                       f"replay_{spec['slug']}_{day}{'_revealed' if reveal else '_blind'}.csv",
                       "text/csv")

    # -- where the 1-2 star customers sit, and the same list on every day ---------
    def risk_chart() -> go.Figure:
        fig = go.Figure()
        if reveal:
            colours = np.where(g["actual_low_review"] == 1, theme.ACCENT, theme.PRIMARY)
            fig.add_bar(x=g["rank"], y=100 * g["p_low_review"], marker_color=colours,
                        name="review risk", hovertext=np.where(
                            g["actual_low_review"] == 1, "rated the order 1–2 stars",
                            "rated the order 3–5 stars"))
        else:
            fig.add_bar(x=g["rank"], y=100 * g["p_low_review"], marker_color=theme.PRIMARY,
                        name="review risk")
        fig.add_vrect(x0=0.5, x1=k_eff + 0.5, fillcolor=theme.GOLD, opacity=0.15,
                      line_width=0, annotation_text=f"top {k_eff}",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color=theme.INK))
        fig.add_hline(y=100 * meta["p2"]["threshold"], line=dict(color=theme.MUTED, dash="dot"),
                      annotation_text=f"flag threshold {100 * meta['p2']['threshold']:.0f}%",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color=theme.MUTED))
        # Head-room above the tallest bar, so the 'top K' label does not sit on it.
        top_risk = 100 * float(g["p_low_review"].max())
        fig.update_yaxes(range=[0, max(1.15 * top_risk, 1.3 * 100 * meta["p2"]["threshold"])])
        fig.update_layout(height=340, xaxis_title="rank", yaxis_title="review risk (%)",
                          showlegend=False)
        return fig

    risk_title = theme.section("Where the risk is in the list",
                               "Every customer on the day's list, ranked by review risk.")
    if not reveal:
        st.markdown(risk_title, unsafe_allow_html=True)
        st.plotly_chart(risk_chart(), use_container_width=True, config=PLOT_CFG)
        st.caption("How this day's list turned out, and how the same list did on every "
                   "other day of the window, appear when you switch on **Reveal what "
                   "actually happened**"
                   + (", with each order's checkout forecast and late risk: they were "
                      "saved only for orders that were delivered, so showing them blind "
                      "would mark the customers who never received the parcel."
                      if purchase_list else "."))
    else:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(risk_title, unsafe_allow_html=True)
            st.plotly_chart(risk_chart(), use_container_width=True, config=PLOT_CFG)
            st.caption("Red bars are customers who went on to rate the order 1–2 stars. "
                       "They cluster at the top of the list, which is what makes a ranked "
                       "list worth working from the top.")
        with c2:
            st.markdown(theme.section("The same capacity on every day",
                                      f"Precision of the top-{k} list, day by day across the "
                                      f"whole held-out window ({spec['days']})."),
                        unsafe_allow_html=True)
            daily = daily_top_k(df, k, which)
            pooled = daily["hits"].sum() / daily["contacted"].sum()
            base = float(pool["actual_low_review"].mean())
            fig = go.Figure()
            fig.add_bar(x=daily["day"], y=daily["low_reviews"],
                        name="customers who rated 1–2 stars that day",
                        marker_color=theme.LINE, yaxis="y2")
            fig.add_scatter(x=daily["day"], y=100 * daily["precision"], mode="lines+markers",
                            name=f"precision of the top {k}",
                            line=dict(color=theme.ACCENT, width=2), marker=dict(size=4))
            fig.add_hline(y=100 * pooled, line=dict(color=theme.ACCENT, dash="dash", width=1),
                          annotation_text=f"pooled {100 * pooled:.0f}%",
                          annotation_position="top left",
                          annotation_font=dict(size=10, color=theme.ACCENT))
            fig.add_vline(x=pd.Timestamp(day), line=dict(color=theme.INK, dash="dot", width=1))
            fig.update_layout(
                height=340, yaxis=dict(title="precision (%)", range=[0, 102]),
                yaxis2=dict(title="1–2 star customers", overlaying="y", side="right",
                            showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
            hits, contacted = int(daily["hits"].sum()), int(daily["contacted"].sum())
            if purchase_list:
                # 'Reached' means before the customer rated (the Customer journey tab):
                # on this list most of the 1-2 star customers had rated already, or
                # never received the parcel, by the time it arrived.
                result = (f"the top {k} each day held a customer who went on to rate the "
                          f"order 1–2 stars {100 * pooled:.0f}% of the time ({hits} of "
                          f"{contacted}), against {100 * base:.0f}% at random; "
                          f"{int(daily['hits_at_delivery'].sum())} of those {hits} had not "
                          "yet rated when the parcel arrived, so a call at delivery could "
                          "still have reached them first")
            else:
                result = (f"the top {k} each day reached a customer who had not yet rated "
                          f"the order and went on to rate it 1–2 stars {100 * pooled:.0f}% "
                          f"of the time ({hits} of {contacted}), against {100 * base:.0f}% "
                          "at random")
            st.caption(
                f"Over the {len(daily)} {spec['days']}, {result}. Historical outcomes: the "
                "tables record no contact other than the survey. Days vary because the "
                "number of risky orders varies.")

    ref = delivery_day_reference()
    working = (f" ({100 * ref['precision']:.0f} in 100 went on to rate 1–2 stars, "
               f"{ref['lift']:.1f} times chance)" if ref else "")
    record = (f" Over the window, {100 * ref['precision']:.0f} in 100 customers in its top "
              f"ten a day went on to rate the order 1–2 stars, {ref['lift']:.1f} times "
              "chance." if ref else "")
    if purchase_list:
        body = (
            "These orders were placed after every model, threshold and calibration had been "
            "fixed on earlier data. The list groups customers by the day they ordered so you "
            "can follow one day's customers end to end; each is scored with its realised "
            "delivery. In operation the list is built each morning from the previous day's "
            f"deliveries whose customer has not yet rated the order{working}, and customers "
            "whose promised day ends without a parcel are reached by the promised-date "
            "message instead. Choose <b>" + LIST_DELIVERY.split(" (")[0] + "</b> above to "
            "see that list. For an order that has not shipped yet, the <b>Predict</b> tab "
            "gives the checkout-time view.")
    else:
        body = (
            "This is the list a customer-experience team could work each morning: the "
            "parcels delivered that day whose customer had not yet rated the order, ranked "
            f"by the review-risk score at delivery.{record} Customers who rated while still "
            "waiting for a late parcel, or whose order never came, are not on it: the "
            "promised-date message reaches them when their promised day ends.")
    st.markdown(theme.note("How this list is built", body), unsafe_allow_html=True)
