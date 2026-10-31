"""Replay a held-out test day: the deployed models scoring historical orders as new.

The brief notes that, because the Olist extract ends in 2018, a deployed app
predicts on held-out historical records that stand in for new orders. This tab
makes that literal. The test window - 30 June to 30 August 2018, 62 purchase days - was never used
to fit, tune or threshold anything. The per-order predictions that
src/train_models.py wrote for it are read back here one purchase day at a time,
exactly as the deployed models produced them, and ranked the way the
customer-experience team would work them. Because the outcomes are history, the
tab can also reveal what happened, so a viewer can check the day's triage list
against reality instead of taking the reported metrics on trust.

Nothing is recomputed: `p_low_review` is the deployed stacking model's score at
delivery, `p_late` the derived late-risk of the lead-time regression, and
`pred_days` its lead-time forecast. The daily precision@K the tab reports is the
same arithmetic as src/evaluate.capacity_curve, which is what Section 7 of the
final report tabulates.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "report" / "results"
PLOT_CFG = {"displayModeBar": False, "responsive": True}


# ===========================================================================
# Data
# ===========================================================================
@st.cache_data(show_spinner="Reading the held-out test predictions…")
def load_replay() -> tuple[pd.DataFrame, dict]:
    """The three per-order prediction tables of the test window, joined on order."""
    meta = json.loads((PROJECT_ROOT / "models" / "model_metadata.json").read_text("utf-8"))
    p2 = pd.read_csv(RESULTS / "predictions_p2_test.csv",
                     parse_dates=["order_purchase_timestamp"])
    p1a = pd.read_csv(RESULTS / "predictions_p1a_test.csv")[["order_id", "actual_days", "pred_days"]]
    p1b = pd.read_csv(RESULTS / "predictions_p1b_test.csv")[["order_id", "actual_late", "p_late_derived"]]
    df = (p2.merge(p1a, on="order_id", how="left")
            .merge(p1b, on="order_id", how="left")
            .rename(columns={"p_late_derived": "p_late"}))
    df["day"] = df["order_purchase_timestamp"].dt.date
    df["late_flag"] = (df["p_late"] >= meta["p1b"]["threshold"]).astype("Int64")
    df["low_review_flag"] = (df["p_low_review"] >= meta["p2"]["threshold"]).astype(int)
    # When was the review written? In the data, a late parcel's survey goes out
    # once its promised date has passed (99% of late parcels), so its customer
    # usually reviews while still waiting (src/review_timing.py). Optional: the
    # tab works without the file.
    timing = RESULTS / "review_timing_test_orders.csv"
    if timing.exists():
        df = df.merge(pd.read_csv(timing), on="order_id", how="left")
    else:
        df["written_before_arrival"] = np.nan
    return df, meta


@st.cache_data(show_spinner=False)
def daily_top_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Precision and recall of the top-K list on every purchase day of the window."""
    ranked = df.sort_values(["day", "p_low_review"], ascending=[True, False], kind="stable")
    ranked["pos"] = ranked.groupby("day").cumcount()
    rows = []
    for day, g in ranked.groupby("day"):
        top = g[g["pos"] < k]
        positives = int(g["actual_low_review"].sum())
        tp = int(top["actual_low_review"].sum())
        rows.append({"day": pd.Timestamp(day), "orders": len(g), "low_reviews": positives,
                     "contacted": len(top), "hits": tp,
                     "precision": tp / len(top) if len(top) else np.nan,
                     "recall": tp / positives if positives else np.nan})
    return pd.DataFrame(rows)


# ===========================================================================
# UI
# ===========================================================================
def render_replay() -> None:
    try:
        df, meta = load_replay()
    except FileNotFoundError as exc:
        st.error(f"**The test-window predictions are missing.** {exc}. Run "
                 "`python src/train_models.py` to regenerate `report/results/predictions_*.csv`.")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"**Could not load the replay data:** `{type(exc).__name__}: {exc}`")
        return

    st.markdown(
        theme.section(
            "Replay a day from the held-out test window",
            "Pick a purchase day in the held-out window (30 June – 30 August 2018). The app shows what the deployed "
            "models predicted for every order placed that day — orders never used to "
            "fit, tune or threshold anything — ranked the way the customer-experience "
            "team would work them. Reveal the outcomes to check the list against what "
            "actually happened."),
        unsafe_allow_html=True)

    days = sorted(df["day"].unique())
    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    with c1:
        default = pd.Timestamp("2018-07-16").date()
        day = st.date_input("Purchase day", value=default if default in days else days[0],
                            min_value=days[0], max_value=days[-1])
    with c2:
        k = int(st.number_input("Outreach capacity K (contacts that day)", min_value=1,
                                max_value=60, value=10, step=1))
    with c3:
        reveal = st.toggle("Reveal what actually happened", value=True)

    g = df[df["day"] == day].sort_values("p_low_review", ascending=False).reset_index(drop=True)
    if g.empty:
        st.warning("**No reviewed orders were placed on that day in the "
                   "test window.** Try a weekday in July or the first half of August; "
                   "orders placed in the last days of August had not been delivered "
                   "when the extract was taken.")
        return

    k_eff = min(k, len(g))
    g.insert(0, "rank", np.arange(1, len(g) + 1))
    g.insert(1, "in_top_k", g["rank"] <= k_eff)
    hits = int(g.loc[g["in_top_k"], "actual_low_review"].sum())
    positives = int(g["actual_low_review"].sum())
    precision = hits / k_eff
    recall = hits / positives if positives else np.nan
    mae_day = float((g["pred_days"] - g["actual_days"]).abs().mean())
    early = int((g["in_top_k"] & (g["actual_low_review"] == 1)
                 & (g["written_before_arrival"] == 1)).sum())

    cards = [
        ("Orders placed that day", f"{len(g):,}",
         "reviewed test orders (a few were never delivered)", theme.PRIMARY),
        ("Top-K for outreach", f"{k_eff}",
         f"{100 * k_eff / len(g):.1f}% of the day's orders", theme.GOLD),
    ]
    if reveal:
        cards += [
            ("Hits in the top-K", f"{hits} of {k_eff}",
             f"precision {100 * precision:.0f}% · recall "
             f"{'—' if np.isnan(recall) else f'{100 * recall:.0f}%'} of {positives} low reviews"
             + (f" · {early} written before the parcel arrived" if hits else ""),
             theme.ACCENT),
            ("Lead-time error that day", f"{mae_day:.1f} d",
             f"mean |predicted − actual|; promise off by "
             f"{float((g['estimated_days'] - g['actual_days']).abs().mean()):.1f} d",
             theme.GREEN),
        ]
    else:
        cards += [
            ("Flagged late at checkout", f"{100 * g['late_flag'].fillna(0).mean():.0f}%",
             f"risk ≥ {100 * meta['p1b']['threshold']:.0f}% on the derived flag", theme.ACCENT),
            ("Mean predicted lead time", f"{g['pred_days'].mean():.1f} d",
             f"against a promised {g['estimated_days'].mean():.0f} d", theme.GREEN),
        ]
    st.markdown(theme.kpi_row(cards), unsafe_allow_html=True)

    # -- the ranked list -----------------------------------------------------
    # A handful of reviewed orders in the window never reached the customer, so
    # they carry a review-risk score but no lead-time outcome; show a dash
    # rather than a misleading number for those.
    def days(col: pd.Series) -> list[str]:
        return [f"{v:.1f}" if pd.notna(v) else "—" for v in col]

    show = pd.DataFrame({
        "rank": g["rank"], "in top-K": g["in_top_k"],
        "review risk": 100 * g["p_low_review"],
        "late risk": 100 * g["p_late"],
        "predicted days": days(g["pred_days"]),
        "promised days": g["estimated_days"].round(0),
        "state": g["customer_state"], "region": g["customer_region"],
        "category": g["lead_category"], "order": g["order_id"].str[:8] + "…",
    })
    cfg = {
        "in top-K": st.column_config.CheckboxColumn("in top-K"),
        "review risk": st.column_config.ProgressColumn("review risk", format="%.0f%%",
                                                       min_value=0, max_value=100),
        "late risk": st.column_config.ProgressColumn("late risk at checkout", format="%.0f%%",
                                                     min_value=0, max_value=100),
        "promised days": st.column_config.NumberColumn(format="%.0f"),
    }
    if reveal:
        show.insert(3, "actual review", np.where(g["actual_low_review"] == 1, "1–2 ★", "3–5 ★"))
        show.insert(4, "review written", np.select(
            [g["actual_days"].isna(), g["written_before_arrival"] == 1,
             g["written_before_arrival"] == 0],
            ["—", "before the parcel", "after delivery"], default="—"))
        show.insert(7, "actual days", days(g["actual_days"]))
        show.insert(8, "delivery", np.select(
            [g["actual_late"] == 1, g["actual_late"] == 0], ["late", "on time"],
            default="not delivered"))
    st.dataframe(show, use_container_width=True, hide_index=True, height=380,
                 column_config=cfg)

    buf = io.StringIO()
    g.drop(columns=["day"]).to_csv(buf, index=False)
    st.download_button("Download this day's ranked list", buf.getvalue(),
                       f"replay_{day}.csv", "text/csv")

    # -- where the low reviews sit in the ranking ----------------------------
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(theme.section("Where the risk is in the list",
                                  "Every order of the day, ranked by review risk."),
                    unsafe_allow_html=True)
        fig = go.Figure()
        if reveal:
            colours = np.where(g["actual_low_review"] == 1, theme.ACCENT, theme.PRIMARY)
            fig.add_bar(x=g["rank"], y=100 * g["p_low_review"], marker_color=colours,
                        name="review risk", hovertext=np.where(
                            g["actual_low_review"] == 1, "left a 1–2 star review",
                            "left 3–5 stars"))
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
        fig.update_layout(height=340, xaxis_title="rank", yaxis_title="review risk (%)",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
        if reveal:
            st.caption("Red bars are orders that did leave a 1–2 star review. They "
                       "cluster at the top of the list, which is what makes a ranked "
                       "list worth working from the top.")
    with c2:
        st.markdown(theme.section("The same capacity on every test day",
                                  f"Precision of the top-{k_eff} list, day by day across "
                                  "the whole held-out window."),
                    unsafe_allow_html=True)
        daily = daily_top_k(df, k_eff)
        fig = go.Figure()
        fig.add_bar(x=daily["day"], y=daily["low_reviews"], name="low reviews that day",
                    marker_color=theme.LINE, yaxis="y2")
        fig.add_scatter(x=daily["day"], y=100 * daily["precision"], mode="lines+markers",
                        name=f"precision of the top-{k_eff}",
                        line=dict(color=theme.ACCENT, width=2), marker=dict(size=4))
        mean_p = float(np.nanmean(daily["precision"]))
        fig.add_hline(y=100 * mean_p, line=dict(color=theme.ACCENT, dash="dash", width=1),
                      annotation_text=f"mean {100 * mean_p:.0f}%", annotation_position="top left",
                      annotation_font=dict(size=10, color=theme.ACCENT))
        fig.add_vline(x=pd.Timestamp(day), line=dict(color=theme.INK, dash="dot", width=1))
        fig.update_layout(
            height=340, yaxis=dict(title="precision (%)", range=[0, 102]),
            yaxis2=dict(title="low reviews", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
        st.caption(f"Over the {len(daily)} test days, contacting the top {k_eff} each day "
                   f"reached a would-be 1–2 star reviewer {100 * mean_p:.0f}% of the time "
                   f"on average, against a base rate of "
                   f"{100 * df['actual_low_review'].mean():.0f}%. Days vary because "
                   "the number of risky orders varies; the pooled figures are in the "
                   "model card.")

    st.markdown(
        theme.note(
            "Why this is a fair demonstration",
            "These orders were placed after every model, threshold and calibration had "
            "been fixed on earlier data, so the app is doing here exactly what it would "
            "do on a live feed: scoring orders it has never seen and ranking them for "
            "action. The review risk shown is the score at delivery, with the realised "
            "delivery dates, which is the moment the model was built for. One caveat the "
            "<b>review written</b> column makes visible: when a parcel misses its promised "
            "date, Olist sends the survey once that date has passed, so most customers of late parcels "
            "review while still waiting. For them a call at delivery comes too late, and "
            "the moment to act is the promised date. For an order that has not shipped "
            "yet, the <b>Predict</b> tab gives the checkout-time view instead."),
        unsafe_allow_html=True)
