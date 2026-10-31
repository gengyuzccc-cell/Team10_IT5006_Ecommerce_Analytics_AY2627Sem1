"""
Olist Brazilian E-Commerce - IT5006 Group 10 dashboard.

Run locally:
    streamlit run deployment/app.py     # the application
    streamlit run dashboard/app.py      # the Community Cloud entry point; same thing

Three parts, matching the three project phases:

  Phase 1  five exploratory tabs over the joined nine-table order frame.
  Phase 2  a scoring tab that serves the fitted models, and a model card.
  Phase 3  a replay of the held-out test window, one day at a time, and a model
           card that states what is deployed, how the candidates varied fold to
           fold, and how the outreach pilot would be sized.

All of it reads the same src/data_load.py and src/features.py the notebooks and
the reports use, so no figure here can disagree with any PDF. The model tabs
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


# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    st.markdown("### 📦 Olist Explorer")
    st.caption("IT5006 · Group 10 · Phases 1–3")

    with st.expander("ℹ️  How to use this dashboard", expanded=False):
        st.markdown(
            "1. **Filter** below — the five exploratory tabs respond.\n"
            "2. **Overview** — volume, revenue and the Black Friday spike.\n"
            "3. **Delivery** — the drift and volatility that shape the modelling.\n"
            "4. **Geography** — where buyers and sellers are, and what that costs.\n"
            "5. **Satisfaction** — how lateness turns into bad reviews.\n"
            "6. **Data quality** — what is missing and why.\n"
            "7. **Predict** — score one order or a whole CSV against the deployed "
            "models, and get a top-K triage list. Not affected by the filters.\n"
            "8. **Replay a day** — the deployed models on a held-out test day, "
            "ranked, with the real outcomes revealable.\n"
            "9. **Model card** — what is deployed, how well it does, where it "
            "fails, and how to reproduce it.\n\n"
            "Charts are interactive: drag to zoom, double-click to reset."
        )

    with st.expander("🎬  Five-minute tour", expanded=False):
        st.markdown(
            "- **Delivery** → the weekly chart: lead time halves, lateness spikes.\n"
            "- **Satisfaction** → late delivery turns 9% low reviews into 54%.\n"
            "- **Predict** → score the default São Paulo → Rio order; change the "
            "promised date and watch the late risk move.\n"
            "- **Replay a day** → pick 16 July 2018, capacity 10, reveal outcomes.\n"
            "- **Model card** → the deployed estimators and the fold-to-fold table."
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
            "- 3.0% of orders never reached the customer, so their delivery "
            "fields are genuinely undefined rather than missing.\n"
            "- Values are in Brazilian reais (R$), not inflation-adjusted.\n"
            "- Window restricted to 2017-01-01 – 2018-08-31; the platform's "
            "pilot months and the truncated tail are excluded."
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
        "IT5006 · Group 10 · Exploration, Prediction and Deployment",
        "Olist Brazilian E-Commerce, 2017–2018",
        f"{len(DF):,} orders joined from <b>nine relational tables</b> · one row per "
        f"order · {DF['lead_category'].nunique()} product categories across "
        f"{DF['customer_state'].nunique()} states. The first five tabs explore the "
        f"data; <b>Predict</b> serves the deployed models, <b>Replay a day</b> runs "
        f"them over a held-out test day with the outcomes revealable, and "
        f"<b>Model card</b> states what is deployed and where it should not be "
        f"trusted. Every view reads the same pipeline as the reports, so the "
        f"numbers cannot diverge.",
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Overview", "Delivery", "Geography", "Satisfaction", "Data quality",
     "🔮 Predict", "⏮ Replay a day", "Model card"]
)


# ===========================================================================
# Tab 1 - Overview
# ===========================================================================
with tab1:
    st.markdown(
        theme.kpi_row([
            ("Orders", f"{len(d):,}",
             f"{pd.Timestamp(d0).strftime('%b %Y')} – {pd.Timestamp(d1).strftime('%b %Y')}",
             theme.PRIMARY),
            ("Item revenue", f"R$ {d['total_price'].sum()/1e6:.1f}m",
             f"median R$ {d['total_price'].median():.0f} per order", theme.GOLD),
            ("Median lead time", _fmt(d['delivery_days'].median(), "{:.1f} d"),
             "purchase to arrival", theme.GREEN),
            ("Delivered late", _fmt(100*d['is_late'].mean(), "{:.1f}%"),
             "vs the promised date", theme.ACCENT),
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
        st.markdown(theme.section("How long delivery takes", ""), unsafe_allow_html=True)
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
            "customer's own state. Distance correlates with lead time at r = 0.40 — "
            "ahead of the platform's own delivery estimate (r = 0.38) — which is why "
            "it is the most valuable feature available before an order ships."
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
        theme.section("What turns a delivery into a bad review",
                      "The strongest relationship in the dataset."),
        unsafe_allow_html=True,
    )

    r = d.dropna(subset=["review_score"])
    st.markdown(
        theme.kpi_row([
            ("Mean review", f"{r['review_score'].mean():.2f} ★",
             f"{100*(r['review_score']>=4).mean():.0f}% are 4–5 stars", theme.GOLD),
            ("1–2 star rate", f"{100*r['target_low_review'].mean():.1f}%",
             "the minority class", theme.ACCENT),
            ("If delivered on time",
             f"{100*r.loc[r['is_late']==0,'target_low_review'].mean():.1f}%",
             "1–2 star rate", theme.GREEN),
            ("If delivered late",
             f"{100*r.loc[r['is_late']==1,'target_low_review'].mean():.1f}%",
             "1–2 star rate", theme.ACCENT),
        ]),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        rs = (r["review_score"].value_counts(normalize=True).sort_index() * 100)
        fig = px.bar(x=rs.index, y=rs.values,
                     labels={"x": "review score", "y": "% of reviews"})
        fig.update_traces(marker_color=theme.PRIMARY,
                          text=[f"{v:.0f}%" for v in rs.values], textposition="outside")
        fig.update_layout(height=340, title="Reviews skew hard to five stars")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    with c2:
        rr = r.dropna(subset=["delay_vs_estimate_days"]).copy()
        bins = [-np.inf, -20, -10, -5, 0, 5, 10, 20, np.inf]
        labels = ["20+ early", "10–20 early", "5–10 early", "0–5 early",
                  "0–5 late", "5–10 late", "10–20 late", "20+ late"]
        rr["bucket"] = pd.cut(rr["delay_vs_estimate_days"], bins=bins, labels=labels)
        prof = (rr.groupby("bucket", observed=True)["target_low_review"]
                .mean().mul(100).reset_index())
        fig = px.bar(prof, x="bucket", y="target_low_review",
                     labels={"bucket": "", "target_low_review": "% scoring 1–2 stars"})
        fig.update_traces(marker_color=theme.ACCENT)
        fig.update_layout(height=340, title="Dissatisfaction rises monotonically with lateness")
        fig.update_xaxes(tickangle=-40)
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
            ("Source tables", "9", "1.45m rows in total", theme.PRIMARY),
            ("Join inflation", "+19%", "if merged naively", theme.ACCENT),
            ("Never delivered", "3.0%", "delivery fields undefined", theme.GOLD),
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
    predict.render_model_card()


st.markdown("---")
st.caption(
    "IT5006 Fundamentals of Data Analytics · Group Project, Group 10 · National "
    "University of Singapore. Built with Streamlit and scikit-learn. Source code, "
    "notebooks and the final report: "
    "https://github.com/gengyuzccc-cell/Team10_IT5006_Ecommerce_Analytics_AY2627Sem1"
)
