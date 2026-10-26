"""Capture the application screenshots used in the report and the slides.

    streamlit run deployment/app.py                       # in one terminal
    python deployment/capture_screenshots.py              # local app
    python deployment/capture_screenshots.py https://team10-it5006-olist.streamlit.app

Writes PNGs into docs/screenshots/ and copies the three the report embeds into
report/figures/ (app_predict.png, app_replay.png, app_card.png). Needs
Playwright with a Chromium build:

    pip install playwright && python -m playwright install chromium

Screenshots are taken at 1440 x 900 CSS pixels with a device scale factor of 2,
so they stay sharp when printed at text width.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "docs" / "screenshots"
FIGURES = PROJECT_ROOT / "report" / "figures"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
# Community Cloud wraps the app in an iframe and a toolbar; the bare app lives
# at <app>/~/+/, which is what a clean screenshot should show.
if "streamlit.app" in URL and "/~/+/" not in URL:
    URL = URL.rstrip("/") + "/~/+/"


def wait_for_app(page, timeout_s: int = 300) -> None:
    """Community Cloud may need to wake the app; local runs take a few seconds."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for frame in page.frames:
            try:
                if frame.locator("text=Yes, get this app back up").count():
                    frame.locator("text=Yes, get this app back up").first.click()
                    time.sleep(20)
                if frame.locator("text=Olist Brazilian E-Commerce").count():
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


def shot(page, name: str, full_page: bool = False) -> Path:
    path = OUT / f"{name}.png"
    page.screenshot(path=str(path), full_page=full_page)
    print(f"  {path.relative_to(PROJECT_ROOT)}")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page.goto(URL, wait_until="domcontentloaded")
        wait_for_app(page)
        settle(page, 6)

        # -- exploratory tabs, with the sidebar --------------------------------
        shot(page, "01_overview")
        open_tab(page, "Delivery")
        shot(page, "02_delivery")
        page.mouse.wheel(0, 900)
        settle(page, 1.5)
        shot(page, "03_delivery_weekly")
        page.mouse.wheel(0, -3000)
        open_tab(page, "Satisfaction")
        shot(page, "04_satisfaction")

        # -- collapse the sidebar so the model tabs use the full width ---------
        try:
            page.hover("[data-testid='stSidebar']")
            page.locator("[data-testid='stSidebarCollapseButton'] button").first.click(
                force=True, timeout=5_000)
            settle(page, 1.5)
        except Exception:  # noqa: BLE001 - the sidebar is cosmetic for these shots
            print("  (sidebar left open)")

        # -- Predict: score the default order ----------------------------------
        open_tab(page, "Predict")
        page.get_by_role("button", name="Predict").first.click()
        settle(page, 4)
        page.mouse.wheel(0, 760)
        settle(page, 1.5)
        shot(page, "05_predict")
        app_predict = OUT / "05_predict.png"
        page.mouse.wheel(0, 1400)
        settle(page, 1.5)
        shot(page, "06_predict_batch")
        page.mouse.wheel(0, -5000)

        # -- Replay a day ------------------------------------------------------
        open_tab(page, "Replay a day")
        settle(page, 3)
        shot(page, "07_replay")
        app_replay = OUT / "07_replay.png"
        page.mouse.wheel(0, 760)
        settle(page, 1.5)
        shot(page, "08_replay_charts")
        page.mouse.wheel(0, -5000)

        # -- Model card --------------------------------------------------------
        open_tab(page, "Model card")
        settle(page, 3)
        shot(page, "09_model_card")
        app_card = OUT / "09_model_card.png"
        page.mouse.wheel(0, 1500)
        settle(page, 1.5)
        shot(page, "10_model_card_folds")
        page.mouse.wheel(0, 1500)
        settle(page, 1.5)
        shot(page, "11_model_card_pilot")
        shot(page, "12_model_card_full", full_page=True)
        browser.close()

    for src, dst in [(app_predict, "app_predict.png"), (app_replay, "app_replay.png"),
                     (app_card, "app_card.png")]:
        shutil.copyfile(src, FIGURES / dst)
        print(f"  report/figures/{dst}")


if __name__ == "__main__":
    main()
