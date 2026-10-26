"""Assemble the Phase 3 submission ZIP.

    python build_submission.py

Produces Team10_IT5006_Ecommerce_Analytics_AY2627Sem1.zip in the project root,
containing exactly what the brief asks for: the final report PDF, the
presentation slides (PPTX and PDF, identical to the deck locked on 4 November),
and the GitHub link (plus a copy of the README so the archive is
self-describing). The ZIP itself is not tracked in git.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEAM = "Team10"
ZIP = ROOT / f"{TEAM}_IT5006_Ecommerce_Analytics_AY2627Sem1.zip"
REPO = "https://github.com/gengyuzccc-cell/Team10_IT5006_Ecommerce_Analytics_AY2627Sem1"
APP = "https://team10-it5006-olist.streamlit.app"

FILES = {
    f"{TEAM}_Phase3_IT5006_AY2627Sem1.pdf": ROOT / "docs" / f"{TEAM}_Phase3_IT5006_AY2627Sem1.pdf",
    f"{TEAM}_Phase3_slides.pptx": ROOT / "docs" / f"{TEAM}_Phase3_slides.pptx",
    f"{TEAM}_Phase3_slides.pdf": ROOT / "docs" / f"{TEAM}_Phase3_slides.pdf",
    "README.md": ROOT / "README.md",
}

LINKS = f"""IT5006 Group 10 - Phase 3 submission

GitHub repository : {REPO}
Deployed app      : {APP}

Contents of this archive
  {TEAM}_Phase3_IT5006_AY2627Sem1.pdf   final report (application link and GitHub link on the cover)
  {TEAM}_Phase3_slides.pptx / .pdf      presentation slides, identical to the deck submitted on 4 November
  README.md                              repository overview and setup instructions

Team: Pan Pinyou, Li Yudan, Zhu Gengyu
"""


def main() -> int:
    missing = [name for name, path in FILES.items() if not path.exists()]
    if missing:
        print("missing:", ", ".join(missing))
        print("build the report (report/) and the slides (docs/) first")
        return 1
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in FILES.items():
            zf.write(path, name)
            print(f"  + {name}  ({path.stat().st_size / 1e6:.1f} MB)")
        zf.writestr("GITHUB_LINK.txt", LINKS)
        print("  + GITHUB_LINK.txt")
    print(f"\nwrote {ZIP.name}  ({ZIP.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
