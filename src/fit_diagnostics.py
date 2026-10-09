"""Training, validation and test performance of the deployed models, side by side.

    python src/train_models.py       # first: fits and saves the selected models
    python src/fit_diagnostics.py    # then: writes report/results/train_val_test.csv

The results tables report expanding-window cross-validation, the validation
period and the test window. This script adds the in-sample fit on the training
period for the three deployed artefacts, so the generalisation gap can be read
directly: a large drop from training to validation is over-fitting; a drop that
tracks the base rate or the level shift is drift. Nothing is refitted. The saved
estimators are scored on the three periods exactly as they were deployed, with
the thresholds and the margin-to-risk mapping fitted on the validation period.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import evaluate as E
import modeling as M
from data_load import load_all, load_analysis
from features import CHECKOUT_FEATURES, DELIVERY_FEATURES, build_features, make_splits

RESULTS = config.PROJECT_ROOT / "report" / "results"
CLF_KEYS = ("roc_auc", "pr_auc", "pr_auc_floor", "pr_auc_lift", "balanced_accuracy",
            "precision", "recall", "f1", "brier", "threshold")


def main() -> pd.DataFrame:
    meta = json.loads((config.MODELS_DIR / "model_metadata.json").read_text("utf-8"))
    p1a = M.load_model(config.MODELS_DIR / "p1a_leadtime.joblib")["estimator"]
    p2 = M.load_model(config.MODELS_DIR / "p2_lowreview.joblib")["estimator"]
    platt = M.load_model(config.MODELS_DIR / "p1b_margin_to_risk.joblib")["estimator"]

    df = build_features(load_analysis(load_all()), with_history=True)
    sp = make_splits(df)
    rows = []
    for period, part in [("train", sp["train"]), ("validation", sp["val"]),
                         ("test", sp["test"])]:
        d = part.dropna(subset=["target_delivery_days"])
        r = E.regression_row(meta["p1a"]["label"], d["target_delivery_days"],
                             p1a.predict(d[CHECKOUT_FEATURES]))
        rows.append({"problem": "P1A lead time", "model": r["model"], "period": period,
                     "n": len(d), "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]})

        lt = part.dropna(subset=["target_is_late"])
        margin = (p1a.predict(lt[CHECKOUT_FEATURES]) - lt["estimated_days"]).to_numpy()
        p_late = platt.predict_proba(margin.reshape(-1, 1))[:, 1]
        c = E.classification_row(meta["p1b"]["label"], lt["target_is_late"].astype(int),
                                 p_late, meta["p1b"]["threshold"])
        rows.append({"problem": "P1B late flag", "model": c["model"], "period": period,
                     "n": len(lt), "positive_rate": float(lt["target_is_late"].mean()),
                     **{k: c[k] for k in CLF_KEYS}})

        rv = part.dropna(subset=["target_low_review"])
        p_low = p2.predict_proba(rv[DELIVERY_FEATURES])[:, 1]
        c = E.classification_row(meta["p2"]["label"], rv["target_low_review"].astype(int),
                                 p_low, meta["p2"]["threshold"])
        rows.append({"problem": "P2 low review", "model": c["model"], "period": period,
                     "n": len(rv), "positive_rate": float(rv["target_low_review"].mean()),
                     **{k: c[k] for k in CLF_KEYS}})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "train_val_test.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    table = main()
    print(table.round(3).to_string(index=False))
    print(f"\nwrote {RESULTS / 'train_val_test.csv'}")
