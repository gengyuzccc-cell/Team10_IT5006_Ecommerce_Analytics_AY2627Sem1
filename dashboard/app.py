"""Streamlit Community Cloud entry point - the application lives in deployment/.

The deployed app at https://team10-it5006-olist.streamlit.app was created with
`dashboard/app.py` as its main file path, and Community Cloud does not let that
path be changed after deployment. Phase 3 moved the application into the
`deployment/` folder the project brief asks for; this file simply runs it, so
the live deployment keeps working without being torn down and recreated.

    streamlit run deployment/app.py      # the application itself
    streamlit run dashboard/app.py       # identical, via this shim
"""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "deployment" / "app.py"),
               run_name="__main__")
