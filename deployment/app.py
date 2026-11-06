"""
Olist Brazilian E-Commerce - IT5006 Team 10 dashboard.

Run locally:
    streamlit run deployment/app.py     # the application
    streamlit run dashboard/app.py      # the Community Cloud entry point; same thing

Nine tabs, opening on the customer:

  Customer journey  one held-out customer's order from checkout to review: what
                    they went through, what each model said, and which contact
                    could still have reached them (journey.py).
  Phase 1           five exploratory tabs over the joined nine-table order frame.
  Phase 2           a scoring tab that serves the fitted models, with the date
                    the customer would be quoted (predict.py).
  Phase 3           a replay of the held-out test window, one day at a time, blind
                    and then revealed (replay.py), and a model card that states
                    what is deployed, what it changes for customers, what a
                    contact is worth under the assumed costs, how the pilot would
                    be sized, and where each number on the slides comes from.

All of it reads the same src/data_load.py and src/features.py the notebooks and
the reports use, and the result tables the analysis scripts write, so no figure
here can disagree with any PDF; the application refits nothing. The model tabs
load their artifacts once per server process and cache them, so no visitor pays
for unpickling a model twice. deployment/README.md documents usage and the CSV contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT / "src"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:          # Streamlit re-executes this file on every rerun
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import journey
import predict
import replay
import theme

st.set_page_config(
    page_title="Olist E-Commerce Analytics | IT5006",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.register_template()
st.markdown(theme.CSS, unsafe_allow_html=True)

PLOT_CFG = {"displayModeBar": False, "responsive": True}
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "olist"

# Events visible in the delivery series. The February-March 2018 entry was added
# after the Milestone 1 feedback: it is the largest late-rate peak in the data
# (a carrier-side slowdown that coincided with a tightened promised window),
# whereas the May 2018 truckers' strike is visible only at weekly resolution.
EVENTS = [
    (pd.Timestamp("2017-11-24"), "Black Friday"),
    (pd.Timestamp("2018-02-19"), "Carrier slowdown"),
    (pd.Timestamp("2018-05-21"), "Truckers' strike"),
]
STRIKE = (pd.Timestamp("2018-05-21"), pd.Timestamp("2018-06-01"))
SLOWDOWN = (pd.Timestamp("2018-02-19"), pd.Timestamp("2018-04-08"))


# ===========================================================================
# Data
# ===========================================================================
@st.cache_data(show_spinner="Joining the nine Olist tables…")
def get_data() -> pd.DataFrame:
    """Join the nine raw tables and derive the order-level features.

    Runs exactly the pipeline the notebooks and the report use, so no figure
    here can disagree with the PDF. `slim=True` reads only the columns the join
    consumes - it leaves out geolocation_city/state and the two free-text review
    columns - which halves peak memory without changing a single value.

    Cached, so the ~5 s join happens once per container rather than per visit.
    """
    from data_load import load_all, load_analysis
    from features import build_features

    df = build_features(load_analysis(load_all(slim=True)))
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    return df


try:
    DF = get_data()
except (FileNotFoundError, OSError):
    st.error(
        f"**Source data not found.** Expected the nine Olist CSVs in "
        f"`data/raw/olist/`.\n\nClone the full repository and run this app from "
        f"the project root:\n\n```\nstreamlit run deployment/app.py\n```"
    )
    st.stop()
except AssertionError as exc:
    st.error(
        f"**The data failed its integrity contract:** {exc}\n\n"
        "`src/data_load.py` asserts the row count of every source table. A "
        "failure here means the data has changed and no figure on this page can "
        "be trusted."
    )
    st.stop()
except Exception as exc:  # noqa: BLE001 - last-resort guard for a public app
    st.error(f"**Could not start the dashboard:** `{type(exc).__name__}: {exc}`")
    st.stop()

# The analysis window as configured (start) and as the data reach (last order).
from config import ANALYSIS_START as _first  # noqa: E402

_last = DF["order_purchase_timestamp"].max()


# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    st.markdown("### 📦 Olist Explorer")
    st.caption("IT5006 · Team 10 · Phases 1–3")

    with st.expander("ℹ️  How to use this dashboard", expanded=False):
        st.markdown(
            "1. **Customer journey** — one customer's order from checkout to review: "
            "what they went through, what each model said, which contact could still "
            "reach them. Opens on the order from our presentation.\n"
            "2. **Filter** below — four exploratory tabs respond (not Data quality).\n"
            "3. **Overview** — volume, promises kept, and the Black Friday spike.\n"
            "4. **Delivery** — the drift and volatility that shape the modelling.\n"
            "5. **Geography** — where buyers and sellers are, and what that costs.\n"
            "6. **Satisfaction** — how a broken promise turns into a 1–2 star rating, "
            "and what unhappy customers wrote.\n"
            "7. **Data quality** — what is missing and why.\n"
            "8. **Predict** — score one order as it looks at checkout, see the date the "
            "customer would be quoted, or score a whole CSV and get a top-K triage "
            "list. Not affected by the filters.\n"
            "9. **Replay a day** — the deployed model's list for a held-out test day, "
            "blind first, then revealed.\n"
            "10. **Model card** — what is deployed, where not to trust it, what it "
            "changes for customers, the value and pilot calculators, and where every "
            "number on the slides comes from.\n\n"
            "Charts are interactive: drag to zoom, double-click to reset."
        )

    _rated = DF.dropna(subset=["review_score"])
    _kept = 100 * _rated.loc[_rated["is_late"] == 0, "target_low_review"].mean()
    _broken = 100 * _rated.loc[_rated["is_late"] == 1, "target_low_review"].mean()
    with st.expander("🎬  Five-minute tour", expanded=False):
        st.markdown(
            "- **Customer journey** → the São Gonçalo customer, who rated the order "
            "while still waiting for a late parcel.\n"
            f"- **Satisfaction** → 1–2 star ratings rise from {_kept:.0f}% when the "
            f"promise is kept to {_broken:.0f}% when it is broken, and what unhappy "
            "customers wrote.\n"
            "- **Replay a day** → 16 July 2018, ten contacts: read the list blind, "
            "then switch on *Reveal what actually happened*.\n"
            "- **Predict** → score the default São Paulo → Rio order and compare the "
            "platform's date with the date the forecast plus the margin would quote.\n"
            "- **Model card** → change the two assumed costs and pick a pilot arm."
        )

    st.markdown("#### Filters")
    months = sorted(DF["month"].unique())
    d0, d1 = st.select_slider(
        "Purchase month", options=months, value=(months[0], months[-1]),
        format_func=lambda t: pd.Timestamp(t).strftime("%b %Y"),
    )
    regions = st.multiselect(
        "Customer region", sorted(DF["customer_region"].dropna().unique()),
        default=list(sorted(DF["customer_region"].dropna().unique())),
    )
    cats = st.multiselect(
        "Product category (blank = all)",
        sorted(DF["lead_category"].dropna().unique()), default=[],
    )
    delivered_only = st.checkbox("Delivered orders only", value=False)

    mask = DF["month"].between(d0, d1) & DF["customer_region"].isin(regions)
    if cats:
        mask &= DF["lead_category"].isin(cats)
    if delivered_only:
        mask &= DF["order_status"] == "delivered"
    d = DF[mask]

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Orders", f"{len(d):,}")
    c2.metric("Categories", f"{d['lead_category'].nunique():,}")

    with st.expander("⚠️  Data caveats", expanded=False):
        st.markdown(
            "- Three source tables are **one-to-many** against orders; they are "
            "aggregated to one row per order before joining, or every average "
            "would be biased towards large baskets.\n"
            "- `customer_id` is regenerated per order and is **not a person**; "
            "`customer_unique_id` is.\n"
            "- 3.0% of the 99,441 orders never reached the customer, so their delivery "
            "fields are genuinely undefined rather than missing.\n"
            "- Values are in Brazilian reais (R\\$), not inflation-adjusted.\n"
            f"- Window: orders placed from {_first.day} {_first:%B %Y} to {_last.day} "
            f"{_last:%B %Y}; the platform's pilot months and the truncated tail are "
            "excluded."
        )

    st.markdown("---")
    st.caption(
        "Data: Brazilian E-Commerce Public Dataset by Olist, distributed for "
        "IT5006 via Canvas (nine relational tables, 99,441 orders)."
    )

# ===========================================================================
# Masthead
# ===========================================================================
st.markdown(
    theme.masthead(
        "IT5006 · Team 10 · Olist Brazilian E-Commerce, 2017–2018",
        "A delivery date customers can rely on, and help when it slips",
        f"Every Olist order carries a promise to a customer: the delivery date shown at "
        f"checkout. {len(DF):,} orders from <b>nine relational tables</b>, "
        f"{DF['lead_category'].nunique()} product categories, {DF['customer_state'].nunique()} "
        f"states. Nine tabs: <b>Customer journey</b> follows one customer's order from "
        f"checkout to review; five tabs explore the data; <b>Predict</b> scores an order at "
        f"checkout and shows the date the customer would be quoted; <b>Replay a day</b> "
        f"ranks a held-out day's customers blind, then reveals what happened; "
        f"<b>Model card</b> states what is deployed, what it changes for customers, and "
        f"where each number comes from. Five-minute tour: Customer journey → "
        f"Satisfaction → Replay 16 July 2018, blind then revealed → Predict → Model card.",
    ),
    unsafe_allow_html=True,
)

if d.empty:
    st.warning("**No orders match the current filters.** The exploratory tabs show all "
               "orders until the selection is widened; the model tabs are not filtered.")
    d = DF


def _fmt(value, pattern: str) -> str:
    """Format a statistic, or an em dash when the selection has no value for it."""
    return pattern.format(value) if pd.notna(value) else "—"

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Customer journey", "Overview", "Delivery", "Geography", "Satisfaction",
     "Data quality", "🔮 Predict", "⏮ Replay a day", "Model card"]
)


# ===========================================================================
# Tab 0 - Customer journey (not filtered: one held-out order, end to end)
# ===========================================================================
with tab0:
    journey.render_journey()


# ===========================================================================
# Tab 1 - Overview
# ===========================================================================
with tab1:
    _deliv = d.dropna(subset=["delivery_days"])
    _rev = d.dropna(subset=["review_score"])
    _rev_deliv = _rev.dropna(subset=["delivery_days"])
    _asked_early = (_rev_deliv["review_creation_date"]
                    < _rev_deliv["order_delivered_customer_date"].dt.normalize())
    _late_rev = _rev_deliv["is_late"] == 1
    st.markdown(
        theme.kpi_row([
            ("Orders", f"{len(d):,}",
             f"{pd.Timestamp(d0).strftime('%b %Y')} – {pd.Timestamp(d1).strftime('%b %Y')}",
             theme.PRIMARY),
            ("Promise kept", _fmt(100 * (_deliv['is_late'] == 0).mean() if len(_deliv)
                                  else np.nan, "{:.1f}%"),
             f"of {len(_deliv):,} delivered orders arrived by the promised date",
             theme.GREEN),
            ("Median lead time", _fmt(d['delivery_days'].median(), "{:.1f} d"),
             "purchase to arrival", theme.GOLD),
            ("Delivered late", _fmt(100*d['is_late'].mean(), "{:.1f}%"),
             "after the promised date"
             + _fmt(_deliv.loc[_deliv['is_late'] == 1, 'delay_vs_estimate_days'].median(),
                    "; by a median {:.1f} days").replace("—", ""), theme.ACCENT),
        ]),
        unsafe_allow_html=True,
    )
    st.markdown(
        theme.kpi_row([
            ("Never delivered",
             _fmt(100 * d['order_delivered_customer_date'].isna().mean() if len(d)
                  else np.nan, "{:.1f}%"),
             "of orders placed: cancelled, unavailable or not yet delivered when the "
             "extract ends", theme.ACCENT),
            ("Asked to review before the parcel arrived",
             _fmt(100 * _asked_early.mean() if len(_asked_early) else np.nan, "{:.1f}%"),
             "of delivered, reviewed orders: survey sent before the delivery day"
             + (f"; {100 * _asked_early[_late_rev].mean():.0f}% when the parcel was late"
                if _late_rev.any() else ""), theme.GOLD),
            ("Rated 1–2 stars",
             _fmt(100 * _rev['target_low_review'].mean() if len(_rev) else np.nan,
                  "{:.1f}%"),
             f"of {len(_rev):,} reviewed orders", theme.ACCENT),
        ]),
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 1], gap="medium")
    with right:
        metric = st.radio("Series", ["Orders", "Item revenue", "Median order value",
                                     "Median items per order"],
                          label_visibility="collapsed")
    agg = {
        "Orders": ("order_status", "size", "orders", theme.PRIMARY),
        "Item revenue": ("total_price", "sum", "R$", theme.GOLD),
        "Median order value": ("total_price", "median", "R$", theme.GOLD),
        "Median items per order": ("n_items", "median", "items", theme.GREEN),
    }[metric]
    series = d.groupby("month").agg(v=(agg[0], agg[1])).reset_index()

    with left:
        fig = go.Figure()
        fig.add_scatter(x=series["month"], y=series["v"], mode="lines+markers",
                        line=dict(color=agg[3], width=2), marker=dict(size=5),
                        fill="tozeroy", name=metric)
        for date, name in EVENTS:
            if series["month"].min() <= date <= series["month"].max():
                fig.add_shape(type="line", x0=date, x1=date, yref="paper", y0=0, y1=1,
                              line=dict(color=theme.ACCENT, width=1, dash="dot"))
                fig.add_annotation(x=date, yref="paper", y=1.0, yanchor="bottom",
                                   text=name, showarrow=False, xanchor="left", xshift=3,
                                   font=dict(size=10, color=theme.ACCENT))
        fig.update_layout(title=f"{metric} by month", yaxis_title=agg[2], height=380)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    st.markdown(
        theme.note(
            "What to look for",
            "Volume rises about <b>eight-fold</b> across the window — from roughly "
            "800 orders in January 2017 to 6,000–7,500 a month through 2018 — and "
            "<b>November 2017 (Black Friday) is the largest single demand event</b> "
            "at about 1.5&times; the median month. Watch what that spike does to "
            "delivery performance in the next tab: the marketplace absorbs the "
            "orders, but not the parcels."
        ),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(theme.section("Busiest categories", "By order count."),
                    unsafe_allow_html=True)
        top = (d["lead_category"].value_counts().head(10).sort_values()
               .rename_axis("category").reset_index(name="orders"))
        fig = px.bar(top, x="orders", y="category", orientation="h")
        fig.update_traces(marker_color=theme.PRIMARY)
        fig.update_layout(height=330, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    with c2:
        st.markdown(theme.section("Order rhythm", "When Brazilians shop."),
                    unsafe_allow_html=True)
        hr = d.groupby("purchase_hour").size().reset_index(name="orders")
        fig = px.area(hr, x="purchase_hour", y="orders",
                      labels={"purchase_hour": "hour of day"})
        fig.update_traces(line_color=theme.GREEN, fillcolor="rgba(91,140,90,0.25)")
        fig.update_layout(height=330)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)


# ===========================================================================
# Tab 2 - Delivery
# ===========================================================================
with tab2:
    st.markdown(
        theme.section(
            "Delivery performance drifts, and lateness is volatile",
            "This is the finding that shapes the whole modelling strategy."),
        unsafe_allow_html=True,
    )

    dd = d.dropna(subset=["delivery_days"])
    perf = dd.groupby("month").agg(
        mean_days=("delivery_days", "mean"),
        late_pct=("is_late", lambda s: 100 * s.mean()),
        n=("delivery_days", "size"),
    ).reset_index()

    fig = go.Figure()
    fig.add_bar(x=perf["month"], y=perf["late_pct"], name="% delivered late",
                marker_color=theme.ACCENT, opacity=0.45, yaxis="y2")
    fig.add_scatter(x=perf["month"], y=perf["mean_days"], mode="lines+markers",
                    name="mean lead time (days)",
                    line=dict(color=theme.PRIMARY, width=2.5), marker=dict(size=5))
    for date, name in EVENTS:
        if perf["month"].min() <= date <= perf["month"].max():
            fig.add_shape(type="line", x0=date, x1=date, yref="paper", y0=0, y1=1,
                          line=dict(color=theme.INK, width=1, dash="dot"))
            fig.add_annotation(x=date, yref="paper", y=1.0, yanchor="bottom",
                               text=name, showarrow=False, xanchor="left", xshift=3,
                               font=dict(size=10, color=theme.INK))
    fig.update_layout(
        height=400, title="Lead time halves; the late rate spikes instead",
        yaxis=dict(title="mean lead time (days)"),
        yaxis2=dict(title="% late", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    st.markdown(
        theme.note(
            "Why this decides the metric",
            "Mean lead time falls from <b>16.9 days (Feb 2018) to 7.7 days (Aug "
            "2018)</b> — Olist genuinely got better. Across a chronological "
            "train/test split the mean drops <b>5.34 days</b>, so a model fitted on "
            "the past is badly off-level on the present and R² turns negative even "
            "when the model is useful. Mean absolute error is the metric that "
            "survives. Lateness, by contrast, does not trend — it swings between "
            "1.4% and 21.4%. The largest peaks are <b>February–March 2018</b>: "
            "carrier transit lengthened by about three days while the promised "
            "window was tightened by about four. The truckers' strike of 21 May–"
            "1 June 2018 is invisible at monthly resolution — see the weekly chart "
            "below: orders caught in transit were 16.9% late, and the platform then "
            "lengthened its promise from about 19 to 37 days, which is why June is "
            "the quietest month. None of this is observable at checkout, which is "
            "why an order-level late classifier has a low ceiling."
        ),
        unsafe_allow_html=True,
    )

    # Weekly resolution, added after the Milestone 1 feedback: a ten-day strike
    # cannot be seen in a monthly series, and what the monthly series records
    # instead is the platform's response - a much longer promised window.
    st.markdown(
        theme.section("Weekly: the strike shows up, and so does the platform's response",
                      "Late rate by purchase week (bars) against the mean promised "
                      "window (line). Shaded: the carrier slowdown and the strike."),
        unsafe_allow_html=True,
    )
    wk = (dd.groupby(dd["order_purchase_timestamp"].dt.to_period("W-SUN").dt.start_time)
            .agg(late_pct=("is_late", lambda s: 100 * s.mean()),
                 promise_days=("estimated_days", "mean"),
                 n=("delivery_days", "size"))
            .reset_index().rename(columns={"order_purchase_timestamp": "week"}))
    wk = wk[wk["n"] >= 50]
    if len(wk) >= 4:
        fig = go.Figure()
        fig.add_bar(x=wk["week"], y=wk["late_pct"], name="% delivered late, by purchase week",
                    marker_color=theme.ACCENT, opacity=0.5)
        fig.add_scatter(x=wk["week"], y=wk["promise_days"], name="promised window (mean days)",
                        yaxis="y2", mode="lines", line=dict(color=theme.PRIMARY, width=2))
        for (x0, x1), label, colour, pos in [
                (SLOWDOWN, "Feb–Apr 2018: carrier slowdown, promise tightened", theme.GOLD, "top left"),
                (STRIKE, "truckers' strike 21 May–1 Jun", "#777777", "bottom right")]:
            if wk["week"].min() <= x1 and wk["week"].max() >= x0:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=colour, opacity=0.18, line_width=0,
                              annotation_text=label, annotation_position=pos,
                              annotation_font=dict(size=10, color=theme.INK))
        fig.update_layout(
            height=380,
            yaxis=dict(title="% delivered late"),
            yaxis2=dict(title="promised days", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    else:
        st.caption("Widen the filters to see the weekly series (needs at least four weeks "
                   "with 50 or more delivered orders).")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(theme.section("How long delivery takes",
                                  "Purchase to arrival; the red line is the median."),
                    unsafe_allow_html=True)
        fig = px.histogram(dd.assign(v=dd["delivery_days"].clip(upper=60)), x="v",
                           nbins=50, labels={"v": "days (clipped at 60)"})
        fig.update_traces(marker_color=theme.PRIMARY)
        fig.add_vline(x=float(dd["delivery_days"].median()),
                      line=dict(color=theme.ACCENT, width=2))
        fig.update_layout(height=320, yaxis_title="orders")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    with c2:
        st.markdown(theme.section("Promise versus reality",
                                  "Negative means the parcel beat its promised date."),
                    unsafe_allow_html=True)
        v = dd["delay_vs_estimate_days"].clip(-40, 40)
        fig = px.histogram(dd.assign(v=v), x="v", nbins=60,
                           labels={"v": "days late (negative = early)"})
        fig.update_traces(marker_color=theme.GREEN)
        fig.add_vline(x=0, line=dict(color=theme.ACCENT, width=2))
        fig.update_layout(height=320, yaxis_title="orders")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)


# ===========================================================================
# Tab 3 - Geography
# ===========================================================================
with tab3:
    st.markdown(
        theme.section("A lopsided marketplace",
                      "Sellers cluster in São Paulo; customers do not. That gap is "
                      "the engine of the delivery problem."),
        unsafe_allow_html=True,
    )

    cs = (d["customer_state"].value_counts(normalize=True) * 100).head(10)
    ss = (d["lead_seller_state"].value_counts(normalize=True) * 100).reindex(cs.index).fillna(0)
    comp = pd.DataFrame({"state": cs.index, "customers": cs.values, "sellers": ss.values})

    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        fig = go.Figure()
        fig.add_bar(x=comp["state"], y=comp["customers"], name="customers",
                    marker_color=theme.PRIMARY)
        fig.add_bar(x=comp["state"], y=comp["sellers"], name="sellers",
                    marker_color=theme.ACCENT)
        fig.update_layout(height=380, barmode="group", yaxis_title="% of orders",
                          title="Where buyers are, and where sellers are")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    with c2:
        reg = d.groupby("customer_region", observed=True).agg(
            orders=("order_status", "size"),
            median_km=("distance_km", "median"),
            median_days=("delivery_days", "median"),
            late_pct=("is_late", lambda s: 100 * s.mean()),
        ).round(1).sort_values("median_days").reset_index()
        st.dataframe(
            reg.rename(columns={"customer_region": "Region", "orders": "Orders",
                                "median_km": "Median km", "median_days": "Median days",
                                "late_pct": "% late"}),
            hide_index=True, use_container_width=True, height=380,
            column_config={"Orders": st.column_config.NumberColumn(format="%d")},
        )

    st.markdown(
        theme.note(
            "Distance is the strongest checkout-time signal",
            "The median order travels <b>434 km</b> and only 35.7% stay inside the "
            "customer's own state. Distance correlates with lead time at r = 0.39 "
            "(0.40 on the log scale the model uses) — ahead of the platform's own "
            "delivery estimate (r = 0.38) — which is why it is the most valuable "
            "feature available before an order ships."
        ),
        unsafe_allow_html=True,
    )

    # Sample from the rows that survive dropna: a narrowed selection with a few
    # undelivered orders used to ask for more rows than were left.
    geo = d.dropna(subset=["distance_km", "delivery_days"])
    fig = px.scatter(
        geo.sample(min(6000, len(geo)), random_state=42),
        x="distance_km", y="delivery_days", color="customer_region",
        opacity=0.35, labels={"distance_km": "customer–seller distance (km)",
                              "delivery_days": "lead time (days)",
                              "customer_region": ""},
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_yaxes(range=[0, 60])
    fig.update_layout(height=380, title="Farther orders take longer, with a wide spread")
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)


# ===========================================================================
# Tab 4 - Satisfaction
# ===========================================================================
with tab4:
    st.markdown(
        theme.section("When a delivery leaves the customer unhappy",
                      "The strongest relationship in the dataset."),
        unsafe_allow_html=True,
    )

    r = d.dropna(subset=["review_score"])
    _never_r = r[r["order_delivered_customer_date"].isna()]
    st.markdown(
        theme.kpi_row([
            ("Mean review", f"{r['review_score'].mean():.2f} ★",
             f"{100*(r['review_score']>=4).mean():.0f}% are 4–5 stars", theme.GOLD),
            ("1–2 star rate", f"{100*r['target_low_review'].mean():.1f}%",
             "the minority class", theme.ACCENT),
            ("If delivered on time",
             _fmt(100*r.loc[r['is_late']==0,'target_low_review'].mean(), "{:.1f}%"),
             "1–2 star rate", theme.GREEN),
            ("If delivered late",
             _fmt(100*r.loc[r['is_late']==1,'target_low_review'].mean(), "{:.1f}%"),
             "1–2 star rate", theme.ACCENT),
            ("If never received",
             _fmt(100*_never_r['target_low_review'].mean() if len(_never_r) else np.nan,
                  "{:.1f}%"),
             f"1–2 star rate of {len(_never_r):,} reviewed orders never delivered",
             theme.ACCENT),
        ]),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        rs = (r["review_score"].value_counts(normalize=True).sort_index() * 100)
        fig = px.bar(x=rs.index, y=rs.values,
                     labels={"x": "review score", "y": "% of reviews"})
        fig.update_traces(marker_color=theme.PRIMARY,
                          text=[f"{v:.0f}%" for v in rs.values], textposition="outside",
                          cliponaxis=False)
        fig.update_yaxes(range=[0, 1.15 * float(rs.max())] if len(rs) else None)
        fig.update_layout(height=340, title="Reviews skew hard to five stars")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    with c2:
        # The report's bins (src/review_timing.lateness_buckets, review_timing_lateness.csv),
        # computed live on the filtered orders.
        from review_timing import lateness_buckets

        rr = r.dropna(subset=["delay_vs_estimate_days"])
        prof = lateness_buckets(rr.assign(low=rr["target_low_review"] == 1))
        late_side = prof[prof["bucket"].astype(str).isin(
            ["4-7 days late", "8-14 days late", "15+ days late"])]
        title = ("One broken promise is enough:<br>from four days late, most customers are "
                 "unhappy" if len(late_side) == 3 and (late_side["low_pct"] > 50).all()
                 else "1–2 star rate by how early or late the parcel arrived")
        fig = px.bar(prof, x="bucket", y="low_pct",
                     labels={"bucket": "against the promised date",
                             "low_pct": "% rating 1–2 stars"},
                     hover_data={"n": ":,"})
        fig.update_traces(marker_color=theme.ACCENT,
                          text=[f"{v:.0f}%" for v in prof["low_pct"]], textposition="outside",
                          cliponaxis=False)
        # Two-line labels stay horizontal; tilted ones run into the axis title.
        fig.update_xaxes(tickangle=0, tickmode="array", tickvals=list(prof["bucket"].astype(str)),
                         ticktext=[str(b).replace(" days ", " days<br>")
                                   for b in prof["bucket"]])
        fig.update_layout(height=340, title=title, margin=dict(t=70))
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    st.markdown(
        theme.note(
            "Why this changes the modelling design",
            "A late delivery raises the 1–2 star rate from <b>9.2% to 54.0%</b>, a "
            "<b>5.9&times;</b> increase. That single fact is why our review model is "
            "framed <b>after</b> delivery rather than at checkout: predicting the "
            "same target at checkout, without knowing how delivery went, reaches "
            "only PR-AUC 0.212 against 0.402 once the outcome is known. The "
            "prediction moment is a design decision, not a detail."
        ),
        unsafe_allow_html=True,
    )

    # What unhappy customers wrote: src/customer_view.py's topic dictionary over the
    # comments of 1-2 star reviews (analysis window), not the sidebar selection.
    try:
        _topics = journey.load_table("customer_complaint_topics.csv")
        _labels = journey.load_table("customer_complaint_dictionary.csv")
        _prec = journey.load_table("customer_complaint_precision.csv")
    except (FileNotFoundError, OSError):
        _topics = None
    if _topics is not None:
        st.markdown(
            theme.section("In customers' own words",
                          "What 1–2 star reviews with a comment talk about, by what happened "
                          "to the parcel. Analysis window; not affected by the sidebar "
                          "filters."),
            unsafe_allow_html=True)
        _names = dict(zip(_labels["topic"], _labels["label"]))
        _cols = [c for c in _topics.columns
                 if c.endswith("_pct") and c != "with_text_pct"]
        _long = (_topics[_topics["group"] != "all"]
                 .melt(id_vars=["group"], value_vars=_cols, var_name="topic", value_name="pct"))
        _long["topic"] = [(_names.get(c[:-4], "no topic matched") if c != "no_topic_pct"
                           else "no topic matched") for c in _long["topic"]]
        _order = [_names.get(c[:-4], "no topic matched") if c != "no_topic_pct"
                  else "no topic matched" for c in _cols]
        _groups = {"on time": theme.GREEN, "late": theme.ACCENT, "never delivered": theme.GOLD}
        fig = go.Figure()
        for _g, _colour in _groups.items():
            _s = _long[_long["group"] == _g].set_index("topic").reindex(_order)
            _n = _topics.loc[_topics["group"] == _g, "with_text"]
            fig.add_bar(y=_order, x=_s["pct"], orientation="h", marker_color=_colour,
                        name=f"parcel {_g}" + (f" ({int(_n.iloc[0]):,} comments)" if len(_n) else ""),
                        hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(barmode="group", height=470, xaxis_title="% of commented 1–2 star reviews",
                          title="What unhappy customers wrote about")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
        _all = _topics[_topics["group"] == "all"].iloc[0]
        st.caption(
            f"{int(_all['with_text']):,} of the {int(_all['low_reviews']):,} 1–2 star reviews "
            f"in the analysis window carry a comment. Keyword matching, overlapping topics, "
            f"held-out precision {_prec['precision_pct'].min():.0f}–"
            f"{_prec['precision_pct'].max():.0f}% by topic (checked on "
            f"{int(_prec['read'].sum())} labelled held-out comments); shares are lower "
            "bounds. Source: report/results/customer_complaint_topics.csv.")


# ===========================================================================
# Tab 5 - Data quality
# ===========================================================================
with tab5:
    st.markdown(
        theme.section("Nine tables, three of them one-to-many",
                      "The joins are where this dataset does its damage."),
        unsafe_allow_html=True,
    )

    st.markdown(
        theme.kpi_row([
            ("Source tables", "9", "1.55m rows in total", theme.PRIMARY),
            ("Join inflation", "+19%", "if merged naively", theme.ACCENT),
            ("Never delivered", "3.0%", "of all 99,441 orders; delivery fields undefined",
             theme.GOLD),
            ("Repeat buyers", "3.1%", "only via customer_unique_id", theme.GREEN),
        ]),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        miss = (DF[["delivery_days", "review_score", "distance_km", "total_price",
                    "total_weight_g", "estimated_days"]]
                .isna().mean().mul(100).sort_values().reset_index())
        miss.columns = ["column", "pct_missing"]
        fig = px.bar(miss, x="pct_missing", y="column", orientation="h",
                     labels={"pct_missing": "% missing", "column": ""})
        fig.update_traces(marker_color=theme.ACCENT)
        fig.update_layout(height=320, title="Missingness is structural, not random")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    with c2:
        stat = (DF["order_status"].value_counts(normalize=True) * 100).head(6)
        fig = px.bar(x=stat.values, y=stat.index, orientation="h",
                     labels={"x": "% of orders", "y": ""})
        fig.update_traces(marker_color=theme.PRIMARY)
        fig.update_layout(height=320, title="Order status")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    st.markdown(
        theme.note(
            "The two traps",
            "<b>Joins.</b> order_items, order_payments and order_reviews are all "
            "one-to-many against orders. Merging them directly turns 99,441 orders "
            "into 118,434 rows and biases every average towards large baskets, so "
            "each child table is aggregated to one row per order first.<br><br>"
            "<b>Identity.</b> <code>customer_id</code> is regenerated for every "
            "order — 99,441 of them for 96,096 real people. Group on it and you "
            "conclude there are no repeat customers at all; group on "
            "<code>customer_unique_id</code> and 2,997 people ordered again, one of "
            "them 17 times."
        ),
        unsafe_allow_html=True,
    )

    # A third, smaller trap: survey dates that precede the order (live from DF, and
    # the test-window count from customer_journeys_test.csv).
    _pre = DF["review_creation_date"] < DF["order_purchase_timestamp"].dt.normalize()
    try:
        _jr = journey.load_journeys()
        _jpre = _jr[_jr["review_predates_order"]]
        _test_txt = (f" {len(_jpre)} of them are reviewed orders of the held-out test window"
                     + (", all cancelled." if len(_jpre)
                        and _jpre["order_status"].isin(journey.CANCELLED).all() else "."))
    except Exception:  # noqa: BLE001 - the count from DF stands on its own
        _test_txt = ""
    if int(_pre.sum()):
        st.caption(
            f"Dates: {int(_pre.sum())} reviews in the analysis window carry a survey date "
            f"before their order was placed; "
            f"{int(_pre[DF['order_status'] == 'canceled'].sum())} of those orders were "
            f"cancelled.{_test_txt} They are kept as recorded; the Customer journey tab marks "
            "such an order as a data anomaly.")

# ===========================================================================
# Tab 6 - Predict  (Phase 2)
# ===========================================================================
with tab6:
    predict.render_predict(DF)


# ===========================================================================
# Tab 7 - Replay a held-out test day  (Phase 3)
# ===========================================================================
with tab7:
    replay.render_replay()


# ===========================================================================
# Tab 8 - Model card  (Phases 2-3)
# ===========================================================================
with tab8:
    predict.render_model_card(DF)


st.markdown("---")
st.caption(
    "IT5006 Fundamentals of Data Analytics · Group Project, Team 10 · National "
    "University of Singapore. Built with Streamlit and scikit-learn. Source code, "
    "notebooks and the final report: "
    "https://github.com/gengyuzccc-cell/Team10_IT5006_Ecommerce_Analytics_AY2627Sem1"
)
