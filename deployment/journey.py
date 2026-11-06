"""Customer journey: one held-out customer's order, from checkout to review.

The application's first tab. It follows one reviewed order of the held-out test
window (orders placed 30 June to 30 August 2018) through every moment the tables
record: what the customer went through, what each deployed model said at that
moment, and which contact could still have reached the customer before they
rated the order. It opens on the order told in the presentation and in Box 1 of
the final report (order e1320973); any other reviewed test order can be picked
from a day's list or by the first characters of its id.

Every value and every classification is read from the files src/customer_view.py
writes, so this tab and the report share one set of definitions:

  report/results/customer_journeys_test.csv   one row per reviewed test order:
                                              timestamps, the review, predictions,
                                              ranks, outcome and reach class
  report/results/customer_story.csv           the presentation's order (Box 1)
  report/results/customer_story_events.csv    its moments, with their source columns
  report/results/customer_outcomes_test.csv   customer counts with named populations
  report/results/customer_promise_rules.csv   the margin added to the forecast (R1)
  models/model_metadata.json                  the warning and flag thresholds

Nothing is re-scored. The checkout numbers are the saved test-window predictions;
re-scoring a test order through predict.assemble() with its seller id would leak,
because models/history_lookup.csv holds end-of-data seller rates that include the
test window's own outcomes.
"""
from __future__ import annotations

import datetime as _dt
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "report" / "results"
MODELS_DIR = PROJECT_ROOT / "models"
PLOT_CFG = {"displayModeBar": False, "responsive": True}

DATE_COLUMNS = ("purchase", "promised", "shipping_limit", "carrier_handoff", "delivered",
                "survey_sent", "answered", "requoted_date")
MODES = ("The customer from our presentation", "A customer from a test day", "Find an order id")
STORY_DAY = _dt.date(2018, 7, 16)

# The reach classes exactly as src/customer_view.py writes them, and how they read.
REACH_SHORT = {"promised-date message only": "Message only",
               "delivery-day call only": "Call only",
               "either": "Either contact",
               "neither": "Neither"}
REACH_SUB = {"promised-date message only":
             "the promised-date message, when the promised day ended without a parcel",
             "delivery-day call only": "a call when the parcel arrived, before they rated it",
             "either": "the promised-date message or a call at delivery",
             "neither": "they rated the order before either moment"}
REACH_PHRASE = {"promised-date message only": "promised-date message only",
                "delivery-day call only": "call at delivery only",
                "either": "either contact",
                "neither": "neither (rated before both)"}
# Bars of the reach chart, top to bottom: (class, key in customer_outcomes_test.csv).
REACH_BARS = [("delivery-day call only", "reach_delivery_only"),
              ("promised-date message only", "reach_promise_only"),
              ("either", "reach_either"),
              ("neither", "reach_neither")]
STATUS_WORDS = {"canceled": "the order was cancelled",
                "unavailable": "the product was unavailable and the order cancelled",
                "shipped": "still with the carrier when the extract ends",
                "invoiced": "invoiced but never shipped",
                "processing": "never shipped", "approved": "never shipped",
                "created": "never shipped",
                "delivered": "marked delivered, but no delivery date is recorded"}
STATUS_SHORT = {"canceled": "cancelled", "unavailable": "unavailable, cancelled",
                "shipped": "still with the carrier", "invoiced": "never shipped",
                "processing": "never shipped", "approved": "never shipped",
                "created": "never shipped", "delivered": "no delivery date"}
CANCELLED = ("canceled", "unavailable")
LANES = {"Platform and models": 0, "Seller and carrier": 1, "Customer": 2}


# ===========================================================================
# Data - all of it written by src/customer_view.py
# ===========================================================================
@st.cache_data(show_spinner="Reading the held-out customers' journeys…")
def load_journeys() -> pd.DataFrame:
    """One row per reviewed order of the test window, as src/customer_view.py wrote it."""
    j = pd.read_csv(RESULTS / "customer_journeys_test.csv", parse_dates=list(DATE_COLUMNS))
    j["purchase_day"] = j["purchase"].dt.date
    j["delivered_day"] = j["delivered"].dt.date
    return j


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    """A results table, read exactly as the analysis scripts wrote it."""
    return pd.read_csv(RESULTS / name)


@st.cache_data(show_spinner=False)
def thresholds() -> dict:
    """The deployed warning (late risk) and flag (review risk) thresholds."""
    meta = json.loads((MODELS_DIR / "model_metadata.json").read_text("utf-8"))
    return {"late": float(meta["p1b"]["threshold"]), "low": float(meta["p2"]["threshold"])}


def margin_days() -> float:
    """The margin rule R1 adds to the forecast (customer_promise_rules.csv)."""
    rules = load_table("customer_promise_rules.csv")
    return float(rules.loc[rules["rule"].str.startswith("R1"), "buffer_days"].iloc[0])


def story_id(j: pd.DataFrame) -> str:
    """The presentation's order, from customer_story.csv."""
    oid = str(load_table("customer_story.csv")["order_id"].iloc[0])
    return oid if (j["order_id"] == oid).any() else str(j["order_id"].iloc[0])


# ===========================================================================
# Words - shared with the Replay tab
# ===========================================================================
def day_text(ts) -> str:
    """'7 Aug 2018' (no zero padding, portable across platforms)."""
    return f"{ts.day} {ts:%b %Y}" if pd.notna(ts) else "—"


def place(city, state, unknown: str = "unknown") -> str:
    if pd.isna(city):
        return unknown if pd.isna(state) else str(state)
    return f"{str(city).title()} ({state})"


def stars(score) -> str:
    n = int(score)
    return f"{n} star{'s' if n != 1 else ''}"


def window_text(purchases: pd.Series) -> str:
    """The test window as orders placed from its first to its last purchase day."""
    a, b = purchases.min(), purchases.max()
    return f"{a.day} {a:%B} to {b.day} {b:%B %Y}"


def experienced(r, short: bool = False) -> str:
    """What one customer went through, in their terms, from a journeys row.

    short=True is the compact form the Replay grid shows (its cells do not wrap)."""
    rating = f"wrote {int(r.review_score)}★" if short else f"wrote {stars(r.review_score)}"
    days = " d" if short else " days"
    if r.outcome == "never delivered":
        status = (STATUS_SHORT if short else STATUS_WORDS).get(r.order_status, r.order_status)
        return f"Never received {'it' if short else 'the parcel'} ({status}); {rating}"
    when = ""
    if pd.notna(r.answered) and pd.notna(r.delivered):
        gap = (r.delivered - r.answered).total_seconds() / 86400
        verb = "came" if short else "arrived"
        when = (f" {gap:.1f}{days} before it {verb}" if gap > 0
                else f" {-gap:.1f}{days} after it {verb}")
    if r.outcome == "late":
        return f"Waited {r.days_late:.1f}{days} past the promise; {rating}{when}"
    if r.outcome == "late, on the promised day":
        return f"{'Came' if short else 'Parcel came'} during the promised day; {rating}{when}"
    if short:
        return f"Came {-r.days_late:.1f} d early; {rating}{when}"
    return f"Parcel came {-r.days_late:.1f} days before the promised date; {rating}{when}"


def when_rated(r) -> str:
    """Where the customer stood at the moment they rated the order. The summary of the
    whole wait (experienced()) would put the final delay on a day it was not yet known."""
    if r.outcome == "never delivered":
        return f"No parcel: {STATUS_WORDS.get(r.order_status, r.order_status)}"
    if pd.isna(r.answered) or pd.isna(r.delivered):
        return experienced(r)
    gap = (r.delivered - r.answered).total_seconds() / 86400
    if gap <= 0:
        return experienced(r)
    past = (r.answered - r.promised).total_seconds() / 86400
    wait = (f"Still waiting, {past:.1f} days past the promised date" if past > 0 else
            f"Parcel not yet arrived, {-past:.1f} days before the promised date")
    total = f", {r.days_late:.1f} days late in all" if r.outcome == "late" else ""
    return f"{wait}; the parcel came {gap:.1f} days later{total}"


# ===========================================================================
# The order's moments
# ===========================================================================
def moments(r, thr: dict, margin: float) -> pd.DataFrame:
    """Every recorded moment of one order, each with what the customer went through,
    what the models said, and what the report's recommendations would do there."""
    rows: list[dict] = []
    late_line = f"{100 * thr['late']:.0f}%"
    never = r.outcome == "never delivered"
    delivered = pd.notna(r.delivered)

    def add(ts, lane, what, label, customer="", model="", action="", date_only=False,
            model_label=""):
        # label: the short name on the timeline; model_label: a second point in the
        # platform lane when the models spoke at a moment of another lane.
        if pd.notna(ts):
            rows.append(dict(ts=ts, lane=lane, what=what, label=label, customer=customer,
                             model=model, action=action, date_only=date_only,
                             model_label=model_label))

    # -- checkout -------------------------------------------------------------
    if pd.isna(r.p_late_derived):
        model = ("Not scored at checkout: the parcel never arrived, and the test "
                 "predictions cover delivered orders")
        action = ""
    else:
        warn = r.p_late_derived >= thr["late"]
        model = (f"Forecast {r.pred_days:.1f} days; risk of missing the promised date "
                 f"{100 * r.p_late_derived:.1f}%, {'at or above' if warn else 'below'} the "
                 f"{late_line} flag line (#{int(r.late_rank_on_day)} of "
                 f"{int(r.delivered_on_day)} delivered orders bought that day)")
        action = f"Quote {day_text(r.requoted_date)}: the forecast plus the {margin:g}-day margin"
        if delivered and pd.notna(r.requoted_date):
            action += (f" (also missed: the parcel came on {day_text(r.delivered)})"
                       if r.delivered > r.requoted_date
                       else f" (kept: the parcel came on {day_text(r.delivered)})")
    add(r.purchase, "Customer", "Order placed", "Order placed",
        customer=f"Shown a delivery date of {day_text(r.promised)}, "
                 f"{r.promised_days:.0f} days away",
        model=model, action=action,
        model_label="" if pd.isna(r.p_late_derived) else "Checkout forecast")

    # -- seller and carrier ----------------------------------------------------
    add(r.shipping_limit, "Seller and carrier", "Seller's deadline to hand over the parcel",
        "Seller's deadline", customer="Nothing visible to the customer")
    if pd.notna(r.carrier_handoff):
        late_by = (r.carrier_handoff - r.shipping_limit).total_seconds() / 86400 \
            if pd.notna(r.shipping_limit) else np.nan
        add(r.carrier_handoff, "Seller and carrier",
            "Parcel handed to the carrier"
            + (f", {late_by:.1f} days after the seller's deadline" if late_by > 0 else ""),
            "Handed to carrier", customer="Nothing visible to the customer")

    # -- the promised date and the end of that day -----------------------------
    eod = r.promised + pd.Timedelta(days=1)
    if delivered and r.delivered <= r.promised:
        at_promise = "The parcel had already arrived"
    elif delivered and r.delivered < eod:
        at_promise = "No parcel yet; it arrives later that day"
    elif never and r.order_status in CANCELLED:
        at_promise = "No parcel: the order had been cancelled"
    else:
        at_promise = "Still waiting: no parcel"
    add(r.promised, "Customer", "Promised date", "Promised date", customer=at_promise,
        date_only=True)
    if r.overdue_eod:
        add(eod, "Platform and models", "The promised day ends without a parcel",
            "Promised day ends", customer=("Not yet rated the order" if r.promise_reach
                      else "Had already rated the order"),
            model="No model needed: a calendar rule",
            action=("Promised-date message: an apology, the parcel's status and a new "
                    "date, or a refund where the order will not come"
                    if r.promise_reach else
                    "Too late for the message: the customer had already rated the order"),
            date_only=True)

    # -- the survey and the review ----------------------------------------------
    if pd.notna(r.survey_sent):
        before = (not delivered) or r.survey_sent < r.delivered.normalize()
        add(r.survey_sent, "Platform and models", "Satisfaction survey sent", "Survey sent",
            customer=("Asked to rate an order not yet received" if before
                      else "Asked to rate an order already received"),
            action=("Send the survey once the order is complete; every customer is "
                    "still asked, and nothing is filtered, delayed in publication or "
                    "rewarded" if before else ""),
            date_only=True)
    add(r.answered, "Customer", f"Rates the order {stars(r.review_score)}",
        f"Rates it {int(r.review_score)}★", customer=when_rated(r))

    # -- the date the margin would have quoted ----------------------------------
    if pd.notna(r.requoted_date):
        add(r.requoted_date, "Platform and models",
            f"Date the forecast plus the {margin:g}-day margin would have quoted",
            "Date with the margin", customer=("The parcel had already arrived" if delivered
                      and r.delivered <= r.requoted_date else "Still waiting: no parcel"),
            date_only=True)

    # -- delivery ------------------------------------------------------------------
    if delivered:
        if r.outcome == "late":
            came = f"Parcel arrives, {r.days_late:.1f} days after the promised date"
        elif r.outcome == "late, on the promised day":
            came = "Parcel arrives during the promised day"
        else:
            came = f"Parcel arrives, {-r.days_late:.1f} days before the promised date"
        model = (f"Review risk {100 * r.p_low_review:.1f}%: #{int(r.low_rank_purchase_day)} of "
                 f"{int(r.reviewed_on_purchase_day)} reviewed orders bought that day")
        if pd.notna(r.low_rank_delivery_day):
            model += (f"; #{int(r.low_rank_delivery_day)} of {int(r.unrated_delivered_on_day)} "
                      "parcels delivered that day and not yet rated")
            action = "Delivery-day list: the customer has not yet rated the order"
            if r.low_rank_delivery_day <= 10:
                action += ("; inside the day's top ten, so the customer would be offered "
                           "help: a replacement, refund or return")
        else:
            old = (r.delivered - r.answered).total_seconds() / 86400
            action = f"Too late for a call: the review was already {old:.1f} days old"
        add(r.delivered, "Seller and carrier", "Parcel delivered", "Parcel delivered",
            customer=came, model=model, action=action, model_label="Review-risk score")

    e = pd.DataFrame(rows)
    # A dated moment (00:00: the promised date, the survey) on the day of the order
    # cannot precede the order itself - a cancelled order's survey can carry its
    # purchase date - so it sorts at the purchase time, after the order, by insertion.
    # A survey dated before the order (a data anomaly) stays where it was recorded.
    p0, p0_day = r.purchase, r.purchase.normalize()
    e["seq"] = range(len(e))
    e["sort_ts"] = [max(t, p0) if (d and t.normalize() == p0_day) else t
                    for t, d in zip(e["ts"], e["date_only"])]
    e = e.sort_values(["sort_ts", "seq"], kind="stable").reset_index(drop=True)
    e.insert(0, "day", ((e["sort_ts"] - p0).dt.total_seconds() / 86400).round(1))
    return e.drop(columns=["seq", "sort_ts"])


def moments_table(e: pd.DataFrame) -> str:
    """The moments as one wrapping HTML table (a grid would cut the sentences off)."""
    show = pd.DataFrame({
        "day": [f"{d:.1f}" for d in e["day"]],
        "when": [f"{t.day} {t:%b %Y}" + ("" if d else f" {t:%H:%M}")
                 for t, d in zip(e["ts"], e["date_only"])],
        "what happened": e["what"],
        "what the customer experienced": e["customer"],
        "what the models said": e["model"],
        "what the recommendations would do": e["action"],
    })
    return theme.table(show, muted=("day", "when"), bold=("what happened",))


# ===========================================================================
# UI
# ===========================================================================
def pick_order(j: pd.DataFrame, story: str) -> str:
    """The order to follow: the presentation's, one from a day's list, or by id."""
    # An ?order=<id> link (at least 7 characters) opens that order, once per session.
    if not st.session_state.get("jr_link_read"):
        st.session_state["jr_link_read"] = True
        try:
            linked = str(st.query_params.get("order", "")).strip().lower()
        except Exception:  # noqa: BLE001 - a link is a convenience, never a failure
            linked = ""
        if len(linked) >= 7:
            st.session_state["jr_mode"] = MODES[2]
            st.session_state["jr_q"] = linked
    st.session_state.setdefault("jr_q", story[:8])

    mode = st.radio("Whose order?", MODES, horizontal=True, key="jr_mode")
    if mode == MODES[1]:
        days = sorted(j["purchase_day"].unique())
        c1, c2 = st.columns([1, 2.6])
        day = c1.date_input("Day the customer ordered",
                            value=STORY_DAY if STORY_DAY in days else days[0],
                            min_value=days[0], max_value=days[-1], key="jr_day")
        g = j[j["purchase_day"] == day].sort_values("low_rank_purchase_day")
        if g.empty:
            st.warning("No reviewed order of the test window was placed that day.")
            return story
        # Labels are the options (mapped back to the id): AppTest cannot invert a format_func.
        # An order with no item rows has no category ('unknown' in the data): say so.
        labels = {f"#{int(r.low_rank_purchase_day)} · {place(r.customer_city, r.customer_state)}"
                  f" · {'no items recorded' if pd.isna(r.price) else r.lead_category}"
                  f" · review risk {100 * r.p_low_review:.0f}%": r.order_id
                  for r in g.itertuples()}
        choice = c2.selectbox("Customer, ranked as on that day's list (riskiest first)",
                              list(labels), key="jr_pick")
        return labels[choice]
    if mode == MODES[2]:
        q = st.text_input("First 7 or more characters of the order id", key="jr_q")
        q = q.strip().lower()
        if len(q) < 7:
            st.info("Type at least seven characters of the order id; showing the "
                    "presentation's customer meanwhile.")
            return story
        hits = j.loc[j["order_id"].str.startswith(q), "order_id"]
        if hits.empty:
            st.info("No reviewed order of the test window starts with that; showing the "
                    "presentation's customer.")
            return story
        return str(hits.iloc[0])
    return story


def timeline(r, e: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    overdue = r.outcome != "on time"
    end = r.delivered if pd.notna(r.delivered) else e["ts"].max()
    if overdue and pd.notna(end) and end > r.promised:
        fig.add_vrect(x0=r.promised, x1=end, fillcolor=theme.ACCENT, opacity=0.10, line_width=0,
                      annotation_text="waiting past the promise", annotation_position="top left",
                      annotation_font=dict(size=10, color=theme.ACCENT))
    # One point per moment in its lane, plus a point in the platform lane wherever the
    # models spoke at a customer or carrier moment (the checkout forecast, the score at
    # delivery). Labels alternate above and below within a lane so neighbours do not
    # collide; the full sentences are in the hover text and in the table below.
    points = []
    for _, m in e.iterrows():
        points.append((m.ts, m.lane, m.label, m.what, m.customer, m.model, m.action))
        if m.model_label and m.lane != "Platform and models":
            points.append((m.ts, "Platform and models", m.model_label, m.model_label, "",
                           m.model, ""))
    seen: dict[str, int] = {}
    for ts, lane, label, what, customer, model, action in points:
        k = seen.get(lane, 0)
        seen[lane] = k + 1
        if what.startswith("Rates the order"):
            colour = theme.ACCENT
        elif lane == "Seller and carrier":
            colour = theme.GOLD
        elif lane == "Platform and models":
            colour = theme.GREEN
        else:
            colour = theme.PRIMARY
        hover = "<br>".join("<br>".join(textwrap.wrap(t, 70))
                            for t in (what, customer, model, action) if t)
        fig.add_scatter(x=[ts], y=[LANES[lane]], mode="markers+text",
                        text=[label], textfont=dict(size=11, color=theme.INK),
                        textposition="top center" if k % 2 == 0 else "bottom center",
                        marker=dict(size=12, color=colour, line=dict(width=1, color="white")),
                        hovertext=hover, hoverinfo="text", showlegend=False)
    fig.update_yaxes(tickvals=list(LANES.values()), ticktext=list(LANES), range=[-0.75, 2.75],
                     showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=True)
    fig.update_layout(height=300, title=title, margin=dict(l=8, r=30, t=44, b=8))
    return fig


@theme.fragment
def render_journey() -> None:
    try:
        j = load_journeys()
        thr = thresholds()
        margin = margin_days()
        story = story_id(j)
        outcomes = load_table("customer_outcomes_test.csv").set_index("key")
    except FileNotFoundError as exc:
        st.error(f"**The customer journeys are missing.** {exc}. Run "
                 "`python src/customer_view.py` to write `report/results/customer_*.csv`.")
        return
    except Exception as exc:  # noqa: BLE001 - last-resort guard for a public app
        st.error(f"**Could not load the customer journeys:** `{type(exc).__name__}: {exc}`")
        return

    st.markdown(theme.section(
        "One customer's order, from checkout to review",
        f"Every order here comes from the held-out test window (orders placed "
        f"{window_text(j['purchase'])}). See what the customer went through, what each deployed "
        f"model said at that moment, and which contact could still have reached them "
        f"before they rated the order. Not affected by the sidebar filters."),
        unsafe_allow_html=True)

    oid = pick_order(j, story)
    r = next(j[j["order_id"] == oid].itertuples())

    if r.review_predates_order:
        anomalies = j[j["review_predates_order"]]
        all_cancelled = bool(anomalies["order_status"].isin(CANCELLED).all())
        st.warning(
            f"**Data anomaly.** This order's survey is dated {day_text(r.survey_sent)}, "
            f"before the order was placed on {day_text(r.purchase)}. "
            f"{len(anomalies)} reviewed test orders carry such a date"
            f"{', all of them cancelled' if all_cancelled else ''}; the moments below "
            "are shown as recorded.")

    items = ("" if pd.isna(r.n_items) else
             f" · {int(r.n_items)} item{'s' if r.n_items != 1 else ''} from "
             f"{int(r.n_sellers)} seller{'s' if r.n_sellers != 1 else ''}")
    km = "" if pd.isna(r.distance_km) else f" · {r.distance_km:,.0f} km"
    # An order with no item rows (cancelled before any was recorded) has no category,
    # price or seller: say so once rather than print 'unknown'.
    basket = ("no items recorded" if pd.isna(r.price) else
              f"{str(r.lead_category).replace('_', ' ')} · R$ {r.price:,.2f} + "
              f"R$ {0 if pd.isna(r.freight) else r.freight:,.2f} freight")
    # The kicker is set in capitals; the order id keeps its own lower case, as typed.
    st.markdown(theme.card(
        "Order <span style='text-transform:none'>" + theme.esc(r.order_id[:8]) + "…</span> "
        + theme.esc(f"· bought {day_text(r.purchase)}, {r.purchase:%H:%M}"),
        theme.esc(f"{place(r.customer_city, r.customer_state)} ← "
                  f"{place(r.seller_city, r.seller_state, 'seller not recorded')}"),
        theme.esc(f"{basket}{km}{items}")),
        unsafe_allow_html=True)

    never = r.outcome == "never delivered"
    cancelled = never and r.order_status in CANCELLED
    if never:
        took, took_sub = "Never delivered", STATUS_WORDS.get(r.order_status, r.order_status)
    elif r.outcome == "late":
        took, took_sub = f"{r.actual_days:.1f} days", f"{r.days_late:.1f} days late"
    elif r.outcome == "late, on the promised day":
        took, took_sub = f"{r.actual_days:.1f} days", "late: it came during the promised day"
    else:
        took, took_sub = f"{r.actual_days:.1f} days", \
            f"{-r.days_late:.1f} days before the promised date"
    if never:
        written = "written with no parcel ever delivered"
    else:
        gap = (r.delivered - r.answered).total_seconds() / 86400
        written = (f"written {gap:.1f} days before the parcel arrived" if gap > 0
                   else f"written {-gap:.1f} days after the parcel arrived")
    st.markdown(theme.kpi_row([
        ("Promised", day_text(r.promised),
         f"{r.promised_days:.1f} days after the order"
         + ("; the order had been cancelled" if cancelled else ""), theme.PRIMARY),
        ("Took", took, took_sub, theme.GREEN if r.outcome == "on time" else theme.ACCENT),
        ("Review", stars(r.review_score), written, theme.GOLD),
        ("Could still be reached by", REACH_SHORT[r.reach_class], REACH_SUB[r.reach_class],
         theme.GREEN if r.reach_class != "neither" else theme.MUTED),
    ]), unsafe_allow_html=True)

    e = moments(r, thr, margin)
    st.plotly_chart(timeline(r, e, "The order's timeline (hover a point for details)"),
                    use_container_width=True, config=PLOT_CFG)
    st.markdown(moments_table(e), unsafe_allow_html=True)
    if oid == story:
        with st.expander("Where each moment of this order comes from"):
            ev = load_table("customer_story_events.csv")
            st.dataframe(ev[["step", "when", "lane", "what", "source_column"]],
                         hide_index=True, use_container_width=True)
            st.caption("report/results/customer_story_events.csv, written by "
                       "src/customer_view.py; the same moments make Box 1 of the report.")

    title = r.review_title if pd.notna(r.review_title) else ""
    message = r.review_message if pd.notna(r.review_message) else ""
    if title or message:
        words = (f"<b>{theme.esc(title)}</b>" if title else "") + \
                (" &mdash; " if title and message else "") + theme.esc(message)
        gloss = ""
        if oid == story and title == "Bom" and message.startswith("Não veio a embalagem certa"):
            gloss = ("<br><span style='color:#61707D'>Our literal reading, marked ambiguous: "
                     "<i>Bom</i> = good; <i>Não veio a embalagem certa</i> = the right "
                     "packaging did not come. It may refer to the packaging rather than "
                     "the parcel; that the customer was still waiting rests on the "
                     "timestamps alone.</span>")
        st.markdown(theme.note("In the customer's own words (Portuguese, verbatim)",
                               words + gloss), unsafe_allow_html=True)
    else:
        st.caption(f"This customer gave {stars(r.review_score)} without a comment.")

    n_low = int(outcomes.loc["low", "orders"])
    bars = [(REACH_PHRASE[c], int(outcomes.loc[k, "orders"]), c) for c, k in REACH_BARS]
    mine = bool(r.low)
    fig = go.Figure(go.Bar(
        x=[v for _, v, _ in bars], y=[lab for lab, _, _ in bars], orientation="h",
        marker_color=[theme.ACCENT if mine and c == r.reach_class else theme.PRIMARY
                      for _, _, c in bars],
        text=[f"{v:,} ({100 * v / n_low:.0f}%)" for _, v, _ in bars], textposition="outside",
        cliponaxis=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 1.18 * max(v for _, v, _ in bars)])
    fig.update_layout(height=250, xaxis_title="customers",
                      title=f"The {n_low:,} test-window customers who rated 1–2 stars: "
                            f"who could still have been reached?",
                      margin=dict(l=8, r=30, t=48, b=8))
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    st.caption(
        "Reached means the contact would have come before the customer rated the order: "
        "the promised-date message when the promised day ends without a parcel, a call "
        "when the parcel arrives. Counts from customer_outcomes_test.csv. "
        + (f"This customer is in the highlighted bar ({REACH_PHRASE[r.reach_class]})."
           if mine else f"This customer gave {stars(r.review_score)}, so is not in "
                        f"this chart."))
