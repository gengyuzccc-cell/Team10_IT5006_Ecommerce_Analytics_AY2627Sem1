"""
Visual design system for the Olist dashboard.

Kept separate from app.py so the styling decisions are reviewable on their own
and so every chart is guaranteed to use the same template rather than plotly's
defaults, which is what makes a multi-tab dashboard read as one product.

Besides the template and the CSS it holds the small HTML helpers every tab
shares: KPI cards, section headers, callouts, the order card of the Customer
journey tab, and the provenance tags (OBSERVED / ASSUMED / DERIVED / PLANNED)
that say what kind of number a card shows.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
INK = "#1B2733"          # primary text
MUTED = "#61707D"        # secondary text
LINE = "#E3E8ED"         # hairlines and gridlines
SURFACE = "#FFFFFF"
CANVAS = "#F5F7F9"

PRIMARY = "#2F6F9F"      # volume and primary series
ACCENT = "#C1666B"       # lateness, dissatisfaction, thresholds
GOLD = "#C08B2E"         # revenue
GREEN = "#5B8C5A"        # positive outcomes

SEQUENTIAL = [
    "#F2F6F9", "#D6E3ED", "#AFC9DC", "#87AFCB",
    "#5F94B9", "#3F7BA6", "#2F6F9F", "#1E4E73",
]
# Diverging ramp: cool = fast/on-time, warm = slow/late.
DIVERGING = [
    [0.00, "#3D6E9C"], [0.25, "#8FB4CE"], [0.45, "#E4EAEF"],
    [0.60, "#F2D9B8"], [0.80, "#D99A62"], [1.00, "#B4553F"],
]


# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------
def register_template() -> str:
    """Register and activate the house plotly template. Returns its name."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif",
                  size=13, color=INK),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", pad=dict(b=12)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=8, r=8, t=48, b=8),
        colorway=[PRIMARY, ACCENT, GOLD, GREEN, "#7A6C9B", "#3F8E8C"],
        xaxis=dict(gridcolor=LINE, linecolor=LINE, zerolinecolor=LINE,
                   tickfont=dict(size=11, color=MUTED),
                   title=dict(font=dict(size=12, color=MUTED))),
        yaxis=dict(gridcolor=LINE, linecolor=LINE, zerolinecolor=LINE,
                   tickfont=dict(size=11, color=MUTED),
                   title=dict(font=dict(size=12, color=MUTED))),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE,
                        font=dict(size=12, color=INK)),
        colorscale=dict(sequential=[[i / (len(SEQUENTIAL) - 1), c]
                                    for i, c in enumerate(SEQUENTIAL)]),
    )
    pio.templates["olist"] = tpl
    pio.templates.default = "olist"
    return "olist"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
  }}
  .stApp {{ background: {CANVAS}; }}
  /* Streamlit's default 5rem side padding wastes a third of the canvas on a
     laptop screen; 1.8rem keeps the gutter without starving the charts. */
  .block-container {{
      padding: 2.0rem 1.8rem 3rem 1.8rem;
      max-width: 1600px;
  }}
  #MainMenu, footer {{ visibility: hidden; }}

  /* ---- masthead ---- */
  .bw-head {{
      background: {SURFACE};
      border: 1px solid {LINE};
      border-radius: 14px;
      padding: 1.35rem 1.6rem 1.2rem 1.6rem;
      margin-bottom: 1.1rem;
  }}
  .bw-head h1 {{
      font-size: 1.62rem; font-weight: 700; color: {INK};
      margin: 0 0 .3rem 0; letter-spacing: -.02em; line-height: 1.2;
  }}
  .bw-head p {{ font-size: .90rem; color: {MUTED}; margin: 0; line-height: 1.5; }}
  .bw-pill {{
      display: inline-block; font-size: .70rem; font-weight: 600;
      letter-spacing: .06em; text-transform: uppercase;
      color: {PRIMARY}; background: rgba(47,111,159,.10);
      padding: .22rem .6rem; border-radius: 999px; margin-bottom: .55rem;
  }}

  /* ---- KPI cards ----
     auto-fit keeps four across on a wide screen and reflows to two, then one,
     as the container narrows - container width, not viewport width, is what
     actually matters inside a Streamlit column. */
  .bw-kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
      gap: .75rem; margin-bottom: 1.1rem;
  }}
  .bw-kpi {{
      background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px;
      padding: .8rem .55rem .8rem .95rem; position: relative; overflow: hidden;
  }}
  .bw-kpi::before {{
      content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
      background: var(--accent, {PRIMARY});
  }}
  .bw-kpi .lab {{
      font-size: .655rem; font-weight: 600; letter-spacing: .06em;
      text-transform: uppercase; color: {MUTED}; display: block;
      margin-bottom: .28rem; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis;
  }}
  .bw-kpi .val {{
      font-size: 1.42rem; font-weight: 700; color: {INK};
      line-height: 1.12; letter-spacing: -.02em; display: block; white-space: nowrap;
  }}
  .bw-kpi .sub {{
      font-size: .715rem; color: {MUTED}; margin-top: .2rem; display: block;
      line-height: 1.35;
  }}

  /* ---- section headers ---- */
  .bw-sec {{ margin: .3rem 0 .8rem 0; }}
  .bw-sec h3 {{
      font-size: 1.02rem; font-weight: 650; color: {INK};
      margin: 0 0 .15rem 0; letter-spacing: -.01em;
  }}
  .bw-sec p {{ font-size: .84rem; color: {MUTED}; margin: 0; line-height: 1.5; }}

  /* ---- insight callout ---- */
  .bw-note {{
      background: {SURFACE}; border: 1px solid {LINE};
      border-left: 3px solid {ACCENT};
      border-radius: 10px; padding: .85rem 1.05rem;
      font-size: .855rem; color: {INK}; line-height: 1.62; margin: .55rem 0 .3rem 0;
  }}
  .bw-note b {{ color: {INK}; font-weight: 650; }}
  .bw-note .tag {{
      display: block; font-size: .68rem; font-weight: 700; letter-spacing: .08em;
      text-transform: uppercase; color: {ACCENT}; margin-bottom: .3rem;
  }}

  /* ---- chart frame ---- */
  .bw-card {{
      background: {SURFACE}; border: 1px solid {LINE};
      border-radius: 12px; padding: .35rem .55rem .1rem .55rem; margin-bottom: .5rem;
  }}

  /* ---- order card (Customer journey) ---- */
  .bw-order {{
      background: {SURFACE}; border: 1px solid {LINE}; border-left: 3px solid {PRIMARY};
      border-radius: 12px; padding: .8rem 1.1rem .75rem 1.1rem; margin: .4rem 0 .9rem 0;
  }}
  .bw-order .kick {{
      display: block; font-size: .68rem; font-weight: 600; letter-spacing: .06em;
      text-transform: uppercase; color: {MUTED}; margin-bottom: .2rem;
  }}
  .bw-order .ttl {{
      font-size: 1.12rem; font-weight: 650; color: {INK}; line-height: 1.3;
      letter-spacing: -.01em;
  }}
  .bw-order p {{ font-size: .86rem; color: {MUTED}; margin: .2rem 0 0 0; line-height: 1.5; }}

  /* ---- wrapping table (the journey's moments) ---- */
  .bw-tblwrap {{
      overflow-x: auto; margin: .2rem 0 .9rem 0; background: {SURFACE};
      border: 1px solid {LINE}; border-radius: 10px;
  }}
  .bw-tbl {{ width: 100%; border-collapse: collapse; font-size: .80rem; color: {INK}; margin: 0; }}
  .bw-tbl th {{
      text-align: left; font-size: .64rem; font-weight: 650; letter-spacing: .05em;
      text-transform: uppercase; color: {MUTED}; background: {CANVAS};
      padding: .5rem .6rem; border: none; border-bottom: 1px solid {LINE};
  }}
  .bw-tbl td {{
      padding: .45rem .6rem; border: none; border-bottom: 1px solid {LINE};
      vertical-align: top; line-height: 1.42;
  }}
  .bw-tbl tr:last-child td {{ border-bottom: none; }}
  .bw-tbl td.muted {{ white-space: nowrap; color: {MUTED}; }}
  /* Numbers never break; their headers do, or a long header widens its column
     until the sentences beside it are squeezed into a word per line. */
  .bw-tbl td.num {{ text-align: right; white-space: nowrap; }}
  .bw-tbl th.num {{ text-align: right; }}

  /* ---- provenance tags: what kind of number this is ---- */
  .bw-tag {{
      display: inline-block; font-size: .58rem; font-weight: 700; letter-spacing: .07em;
      text-transform: uppercase; padding: .06rem .36rem; border-radius: 4px;
      margin-right: .32rem; vertical-align: 1px; line-height: 1.45; white-space: nowrap;
  }}
  .bw-tag-observed {{ color: #1E4E73; background: rgba(47,111,159,.13); }}
  .bw-tag-assumed  {{ color: #85570F; background: rgba(192,139,46,.18); }}
  .bw-tag-derived  {{ color: #54487A; background: rgba(122,108,155,.17); }}
  .bw-tag-planned  {{ color: #3B6A3A; background: rgba(91,140,90,.17); }}
  .bw-tag-other    {{ color: {MUTED}; background: {LINE}; }}

  /* ---- streamlit widget polish ---- */
  .stTabs [data-baseweb="tab-list"] {{ gap: .35rem; border-bottom: 1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{
      height: 42px; padding: 0 1.05rem; background: transparent;
      border-radius: 8px 8px 0 0; font-size: .90rem; font-weight: 550; color: {MUTED};
  }}
  .stTabs [aria-selected="true"] {{ background: {SURFACE}; color: {PRIMARY}; }}
  section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
  div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 10px; }}
  hr {{ border-color: {LINE}; }}
</style>
"""


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def masthead(pill: str, title: str, subtitle: str) -> str:
    return (f'<div class="bw-head"><span class="bw-pill">{pill}</span>'
            f'<h1>{title}</h1><p>{subtitle}</p></div>')


def kpi_row(cards: list[tuple[str, str, str, str]]) -> str:
    """cards = [(label, value, sub, accent_colour), ...]"""
    inner = "".join(
        f'<div class="bw-kpi" style="--accent:{col}">'
        f'<span class="lab">{lab}</span><span class="val">{val}</span>'
        f'<span class="sub">{sub}</span></div>'
        for lab, val, sub, col in cards
    )
    return f'<div class="bw-kpis">{inner}</div>'


def section(title: str, subtitle: str = "") -> str:
    return f'<div class="bw-sec"><h3>{title}</h3><p>{subtitle}</p></div>'


def note(tag: str, body: str) -> str:
    return f'<div class="bw-note"><span class="tag">{tag}</span>{body}</div>'


def card(kicker: str, title: str, body: str = "") -> str:
    """A framed card for one order: a small kicker line, a title, a line of detail.

    Deliberately not a second masthead: the page keeps one h1.
    """
    return (f'<div class="bw-order"><span class="kick">{kicker}</span>'
            f'<div class="ttl">{title}</div><p>{body}</p></div>')


TAG_KINDS = ("observed", "assumed", "derived", "planned")


def tag(kind: str) -> str:
    """A provenance badge: OBSERVED, ASSUMED, DERIVED or PLANNED.

    The four kinds of report/results/slide_provenance.csv: observed in the data,
    assumed with no data source, derived from the two, or planned (a design
    choice). Any other kind (literature, wording, no data source) gets a grey tag.
    """
    k = str(kind).strip().lower()
    cls = k if k in TAG_KINDS else "other"
    return f'<span class="bw-tag bw-tag-{cls}">{html.escape(k.upper())}</span>'


def esc(text) -> str:
    """Text from the data, made safe to place inside one of the HTML blocks above.

    Escapes markup, keeps the block on one line (a blank line would end the HTML
    block and hand the rest to the Markdown parser) and escapes the dollar sign,
    which Streamlit's Markdown would otherwise read as the start of a formula.
    """
    s = html.escape(str(text))
    s = " / ".join(part.strip() for part in s.splitlines() if part.strip())
    return s.replace("$", "&#36;")


def table(df, numeric=(), muted=(), bold=(), html_cols=(), widths=None,
          max_height: str | None = None) -> str:
    """A small table as wrapping HTML. A scrolling grid cuts sentences off at the
    column edge; a handful of rows that hold sentences read better wrapped.

    numeric columns are right-aligned, muted ones grey and unwrapped, bold ones bold.
    Every cell is escaped with esc() except in html_cols, whose values are markup
    this module wrote (a tag()); a missing value prints as an empty cell. widths
    maps a column to its minimum CSS width; max_height makes a long table scroll.
    """
    cols = list(df.columns)
    widths = widths or {}

    def cls(c) -> str:
        name = "num" if c in numeric else "muted" if c in muted else ""
        return f' class="{name}"' if name else ""

    def cell(c, v) -> str:
        missing = v is None or (isinstance(v, float) and v != v)
        text = "" if missing else (str(v) if c in html_cols else esc(v))
        return f"<td{cls(c)}>" + (f"<b>{text}</b>" if c in bold and text else text) + "</td>"

    def th(c) -> str:
        style = f' style="min-width:{widths[c]}"' if c in widths else ""
        return f"<th{cls(c)}{style}>{esc(c)}</th>"

    head = "".join(th(c) for c in cols)
    body = "".join("<tr>" + "".join(cell(c, v) for c, v in zip(cols, row)) + "</tr>"
                   for row in df.itertuples(index=False))
    scroll = f' style="max-height:{max_height};overflow-y:auto"' if max_height else ""
    return (f'<div class="bw-tblwrap"{scroll}><table class="bw-tbl"><tr>{head}</tr>{body}'
            f'</table></div>')


def fragment(func):
    """st.fragment where Streamlit provides it (1.37+), so a tab's own widgets
    rerun only that tab; otherwise the function unchanged (older Streamlit, or
    the stub module tests/test_leakage.py imports the deployment modules with)."""
    import streamlit as st

    frag = getattr(st, "fragment", None)
    return frag(func) if callable(frag) else func
