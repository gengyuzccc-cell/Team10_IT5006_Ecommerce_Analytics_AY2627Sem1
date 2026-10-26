"""
Visual design system for the Olist dashboard.

Kept separate from app.py so the styling decisions are reviewable on their own
and so every chart is guaranteed to use the same template rather than plotly's
defaults, which is what makes a multi-tab dashboard read as one product.
"""
from __future__ import annotations

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
