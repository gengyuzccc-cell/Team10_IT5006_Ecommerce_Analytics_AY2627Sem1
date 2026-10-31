"""Streamlit Community Cloud entry point - the application lives in deployment/.

The deployed app at https://team10-it5006-olist.streamlit.app was created with
`dashboard/app.py` as its main file path, and Community Cloud does not let that
path be changed after deployment. Phase 3 moved the application into the
`deployment/` folder the project brief asks for; this file simply runs it, so
the live deployment keeps working without being torn down and recreated.

    streamlit run deployment/app.py      # the application itself
    streamlit run dashboard/app.py       # identical, via this shim

Community Cloud applies a push by re-running this main script without
restarting the Python process, and it only watches files next to the main
script. deployment/app.py is re-read on every run (runpy), but the modules it
imports - predict, replay, theme - would stay cached in sys.modules at their
old version. Evicting them here makes every run import the current files;
Streamlit's cache_data / cache_resource entries key on the function's module
and source, so the loaded models and tables survive the re-import.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

for _stale in ("predict", "replay", "theme"):
    sys.modules.pop(_stale, None)

runpy.run_path(str(Path(__file__).resolve().parents[1] / "deployment" / "app.py"),
               run_name="__main__")
