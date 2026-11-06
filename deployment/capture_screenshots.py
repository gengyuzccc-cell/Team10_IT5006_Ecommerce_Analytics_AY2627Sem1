"""Capture the application screenshots used in the report and the slides.

    streamlit run deployment/app.py                       # in one terminal
    python deployment/capture_screenshots.py              # local app
    python deployment/capture_screenshots.py https://team10-it5006-olist.streamlit.app

Writes PNGs into docs/screenshots/ and copies the four the report embeds into
report/figures/ (app_journey.png, app_predict.png, app_replay.png, app_card.png).
Needs Playwright with a Chromium build:

    pip install playwright && python -m playwright install chromium

Screenshots are taken at 1440 x 900 CSS pixels with a device scale factor of 2,
so they stay sharp when printed at text width; the full-length model card is
taken at a scale factor of 1. Streamlit's own header (toolbar, Deploy button,
running indicator, sidebar control) is hidden with a style injected into the page
here, not by the application. The Customer journey is the first tab, so every
other tab is opened explicitly, Overview included; the Replay shot is taken after
switching on "Reveal what actually happened", because every day opens blind.

The batch and triage shots upload one day of validation-window orders (bought on
DEMO_DAY, before the test window) as they looked at checkout, built from
data/raw/olist with the same src/data_load.py and src/features.py pipeline the
application runs, in the columns of the Predict tab's template.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "docs" / "screenshots"
FIGURES = PROJECT_ROOT / "report" / "figures"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
# Community Cloud wraps the app in an iframe and a toolbar; the bare app lives
# at <app>/~/+/, which is what a clean screenshot should show.
if "streamlit.app" in URL and "/~/+/" not in URL:
    URL = URL.rstrip("/") + "/~/+/"
READY = ("A delivery date customers can rely on", "Olist Brazilian E-Commerce")
# A validation-window Wednesday before the May 2018 strike: no test-window order, and
# promised windows of the usual length (after the strike they were twice as long).
DEMO_DAY = "2018-05-16"


def demo_batch() -> Path:
    """The orders bought on DEMO_DAY, in the batch format of the Predict tab.

    Built with the application's own pipeline; orders without an item row are left
    out, because an order at checkout always has one."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from data_load import load_all, load_analysis
    from features import build_features

    df = build_features(load_analysis(load_all(slim=True)))
    day = df[(df["order_purchase_timestamp"].dt.strftime("%Y-%m-%d") == DEMO_DAY)
             & (df["n_items"] >= 1)]
    batch = pd.DataFrame({
        "order_purchase_timestamp": day["order_purchase_timestamp"],
        "order_estimated_delivery_date": day["order_estimated_delivery_date"],
        "customer_state": day["customer_state"], "seller_state": day["lead_seller_state"],
        "lead_category": day["lead_category"], "lead_payment_type": day["lead_payment_type"],
        "n_items": day["n_items"], "n_sellers": day["n_sellers"],
        "total_price": day["total_price"], "total_freight": day["total_freight"],
        "total_weight_g": day["total_weight_g"], "max_volume_cm3": day["max_volume_cm3"],
        "total_photos": day["total_photos"], "max_installments": day["max_installments"],
    })
    path = Path(tempfile.mkdtemp()) / f"orders_{DEMO_DAY}.csv"
    batch.to_csv(path, index=False)
    print(f"  demo batch: {len(batch)} orders bought on {DEMO_DAY}")
    return path


def wait_for_app(page, timeout_s: int = 300) -> None:
    """Community Cloud may need to wake the app; local runs take a few seconds."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for frame in page.frames:
            try:
                if frame.locator("text=Yes, get this app back up").count():
                    frame.locator("text=Yes, get this app back up").first.click()
                    time.sleep(20)
                if any(frame.locator(f"text={t}").count() for t in READY):
                    return
            except Exception:  # noqa: BLE001 - a frame may detach while waking
                pass
        time.sleep(2)
    raise SystemExit("the application did not render in time")


def settle(page, seconds: float = 2.5) -> None:
    """Wait for Streamlit to finish rerunning (status widget gone, charts drawn)."""
    time.sleep(seconds)
    try:
        page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=60_000)
    except Exception:  # noqa: BLE001 - absent widget means already settled
        pass
    time.sleep(0.8)


def open_tab(page, name: str) -> None:
    page.get_by_role("tab", name=name).first.click()
    settle(page)


# Streamlit's fixed header (toolbar, Deploy button, running indicator) is chrome, not
# the application: hidden for the screenshots only.
HIDE_CHROME = """
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
"""


def open_app(page) -> None:
    page.goto(URL, wait_until="domcontentloaded")
    wait_for_app(page)
    settle(page, 6)
    for frame in page.frames:     # on Community Cloud the app may sit in a frame
        try:
            frame.add_style_tag(content=HIDE_CHROME)
        except Exception:  # noqa: BLE001 - a cross-origin frame cannot be styled
            pass
    settle(page, 1)


# Streamlit scrolls the main pane (section[data-testid="stMain"]), not the page. With
# its header hidden, an anchor goes this many CSS pixels below the top of the window.
HEADER = 14
_SCROLL_TO = """([needle, off, selector]) => {
    const main = document.querySelector('[data-testid="stMain"]') || document.scrollingElement;
    const visible = el => el && el.offsetParent !== null;
    let target = null;
    if (selector) {
        target = [...main.querySelectorAll(selector)].find(visible);
    } else {
        const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT);
        let node;
        while ((node = walker.nextNode())) {
            if (node.textContent.includes(needle) && visible(node.parentElement)) {
                target = node.parentElement; break;
            }
        }
    }
    if (!target) return false;
    main.scrollTo(0, Math.max(0, target.getBoundingClientRect().top + main.scrollTop - off));
    return true;
}"""


def scroll_to(page, text: str = "", offset: int = HEADER, selector: str = "") -> None:
    """Put the first visible element with this text (or matching this CSS selector)
    `offset` pixels below the top of the window; inactive tabs are hidden, so only the
    open tab counts. Near the end of a tab the pane cannot scroll that far."""
    if not page.evaluate(_SCROLL_TO, [text, offset, selector]):
        print(f"  (could not scroll to {text or selector!r})")
    settle(page, 1.5)


def scroll_top(page) -> None:
    page.evaluate("""() => {
        const main = document.querySelector('[data-testid="stMain"]') || document.scrollingElement;
        main.scrollTo(0, 0);
    }""")
    settle(page, 1.0)


def shot(page, name: str, full_page: bool = False) -> Path:
    """A viewport screenshot; full_page grows the viewport to the main pane's height
    first, because the page itself never scrolls (Streamlit scrolls stMain)."""
    path = OUT / f"{name}.png"
    # Park the pointer in the empty right margin: a hovered grid shows its toolbar.
    size = page.viewport_size
    page.mouse.move(size["width"] - 4, 4)
    time.sleep(0.4)
    if full_page:
        height = page.evaluate("""() => {
            const main = document.querySelector('[data-testid="stMain"]');
            return main ? main.scrollHeight : document.body.scrollHeight;
        }""")
        if int(height) + 40 > 16000:
            print(f"  ({name}: {height} px tall, cut at 16,000)")
        page.set_viewport_size({"width": size["width"], "height": min(int(height) + 40, 16000)})
        settle(page, 4)
        page.screenshot(path=str(path))
        page.set_viewport_size(size)
        settle(page, 2)
    else:
        page.screenshot(path=str(path))
    print(f"  {path.relative_to(PROJECT_ROOT)}")
    return path


def reveal(page) -> None:
    """Switch on the Replay tab's 'Reveal what actually happened' toggle."""
    box = page.get_by_role("checkbox", name="Reveal what actually happened")
    for _ in range(3):
        if box.count() and box.first.is_checked():
            return
        page.locator("[data-testid='stCheckbox']").filter(
            has_text="Reveal what actually happened").first.locator("label").first.click()
        settle(page, 3)
    print("  (the reveal toggle did not switch on)")


def collapse_sidebar(page) -> None:
    """Collapse the sidebar so the model tabs use the full width."""
    try:
        page.hover("[data-testid='stSidebar']")
        page.locator("[data-testid='stSidebarCollapseButton'] button").first.click(
            force=True, timeout=5_000)
        settle(page, 1.5)
    except Exception:  # noqa: BLE001 - the sidebar is cosmetic for these shots
        print("  (sidebar left open)")


TABS = "[data-baseweb='tab-list']"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    batch = demo_batch()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        open_app(page)

        # -- exploratory tabs, with the sidebar (Overview is no longer the first tab)
        open_tab(page, "Overview")
        shot(page, "01_overview")
        open_tab(page, "Delivery")
        scroll_to(page, selector=TABS)
        shot(page, "02_delivery")
        scroll_to(page, "Weekly: the strike shows up")
        shot(page, "03_delivery_weekly")
        scroll_top(page)
        open_tab(page, "Satisfaction")
        scroll_to(page, selector=TABS)
        shot(page, "04_satisfaction")
        scroll_to(page, "In customers' own words")
        shot(page, "04b_satisfaction_words")
        scroll_top(page)

        collapse_sidebar(page)

        # -- Customer journey: the presentation's order -------------------------------
        open_tab(page, "Customer journey")
        scroll_to(page, "One customer's order, from checkout to review")
        app_journey = shot(page, "00_journey")
        scroll_to(page, "what the recommendations would do", offset=HEADER + 10)
        shot(page, "00b_journey_moments")
        scroll_to(page, "In the customer's own words")
        shot(page, "00c_journey_reach")
        scroll_top(page)

        # -- Predict: score the default order, then a day of orders from a file --------
        open_tab(page, "Predict")
        page.get_by_role("button", name="Predict").first.click()
        settle(page, 4)
        scroll_to(page, "Predicted lead time", offset=HEADER + 30)
        app_predict = shot(page, "05_predict")
        page.locator("[data-testid='stFileUploaderDropzoneInput']").first.set_input_files(
            str(batch))
        settle(page, 6)
        scroll_to(page, "Score a file")
        shot(page, "06_predict_batch")
        scroll_to(page, "Triage list: work the top of the ranking")
        shot(page, "06b_predict_triage")
        scroll_top(page)

        # -- Replay a day: blind first, then revealed -----------------------------------
        open_tab(page, "Replay a day")
        settle(page, 3)
        scroll_to(page, selector=TABS)
        shot(page, "07a_replay_blind")
        reveal(page)
        scroll_to(page, selector=TABS)
        app_replay = shot(page, "07_replay")
        scroll_to(page, "customers on this list who went on to rate")
        shot(page, "08_replay_charts")
        scroll_top(page)

        # -- Model card -------------------------------------------------------------
        open_tab(page, "Model card")
        settle(page, 3)
        scroll_to(page, selector=TABS)
        app_card = shot(page, "09_model_card")
        scroll_to(page, "How much the candidates vary from fold to fold")
        shot(page, "10_model_card_folds")
        scroll_to(page, "What the recommendations change for customers")
        shot(page, "10b_model_card_customers")
        scroll_to(page, "What a contact is worth")
        shot(page, "11_model_card_pilot")
        # The pilot as designed on slide 11, with the ceiling a delivery-day call faces.
        page.locator("[data-testid='stRadio']").filter(has_text="Pilot arm").first.get_by_text(
            "Design on slide 11").first.click()
        settle(page, 3)
        scroll_to(page, "Sizing the pilot")
        shot(page, "11a_model_card_slide11_pilot")
        page.locator("[data-testid='stExpander']").filter(
            has_text="Where the numbers on the slides come from").first.locator(
            "summary").first.click()
        settle(page, 2)
        scroll_to(page, "Where the numbers on the slides come from", offset=HEADER + 10)
        shot(page, "11b_model_card_provenance")

        # -- the whole model card, one CSS pixel per pixel (a 2x image would be huge) --
        full = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        open_app(full)
        collapse_sidebar(full)
        open_tab(full, "Model card")
        settle(full, 3)
        shot(full, "12_model_card_full", full_page=True)
        browser.close()
    shutil.rmtree(batch.parent, ignore_errors=True)

    for src, dst in [(app_journey, "app_journey.png"), (app_predict, "app_predict.png"),
                     (app_replay, "app_replay.png"), (app_card, "app_card.png")]:
        shutil.copyfile(src, FIGURES / dst)
        print(f"  report/figures/{dst}")


if __name__ == "__main__":
    main()
