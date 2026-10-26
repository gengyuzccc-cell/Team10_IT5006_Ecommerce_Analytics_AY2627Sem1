"""Fold-to-fold variation and selected hyperparameters, read from the saved tables.

    python src/train_models.py          # first: writes report/results/selection_*.csv etc.
    python src/selection_tables.py      # then: writes cv_folds.csv and hyperparameters.csv

Two Phase 2 feedback points are answered here without refitting anything.

1. "Report CV scores with their fold-to-fold variation for the Problem 2
   candidates." The training script already scored every candidate on each of
   the five expanding-window folds and on the validation period and wrote those
   scores to selection_*.csv; this script lays them out as one table per
   problem with the mean, the fold-to-fold standard deviation, the range and
   the test score beside them. The standard deviation uses the population
   formula (ddof=0), which is what scikit-learn's cross-validation reports and
   therefore what the cv_std column of the results tables already holds; the
   test suite asserts the two agree.

2. "The selected hyperparameter values are said to be in Appendix C." They were
   only in the results CSVs. hyperparameters.csv lists every tuned model's
   selected values in long format, and - for the two deployed ensembles, whose
   own `params` are empty because an ensemble has no searched parameters of its
   own - the values carried by their members, read from the deployed pickles so
   that what the appendix prints is what is actually served.
"""
from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import modeling as M

RESULTS = config.PROJECT_ROOT / "report" / "results"
FOLDS = [f"fold{i}" for i in range(1, 6)]

# Which selection table and results table describe which problem, and whether
# the CV score was stored as a negative error (regression) or a score.
PROBLEMS = [
    ("P1A lead time", "selection_p1a.csv", "reg_p1a.csv", "MAE", True, "mae"),
    ("P1B late flag", "selection_p1b.csv", "clf_p1b.csv", "PR-AUC", False, "pr_auc"),
    ("P2 low review", "selection_p2.csv", "clf_p2.csv", "PR-AUC", False, "pr_auc"),
]

# The hyperparameters a reader would want to see for each estimator class.
SHOWN = {
    "LinearRegression": [],
    "LogisticRegression": ["C"],
    "Ridge": ["alpha"],
    "DecisionTreeRegressor": ["max_depth", "min_samples_leaf", "max_features"],
    "DecisionTreeClassifier": ["max_depth", "min_samples_leaf", "max_features"],
    "RandomForestRegressor": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
    "RandomForestClassifier": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
    "HistGradientBoostingRegressor": ["max_iter", "learning_rate", "max_leaf_nodes",
                                      "min_samples_leaf", "l2_regularization"],
    "HistGradientBoostingClassifier": ["max_iter", "learning_rate", "max_leaf_nodes",
                                       "min_samples_leaf", "l2_regularization"],
}


def need(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `python src/train_models.py` first.")
    return pd.read_csv(path)


# ===========================================================================
# 1. Fold-to-fold variation
# ===========================================================================
def cv_fold_table() -> pd.DataFrame:
    meta = json.loads((config.MODELS_DIR / "model_metadata.json").read_text("utf-8"))
    deployed = {"P1A lead time": meta["p1a"]["label"],
                "P1B late flag": "Derived: regression margin vs promise",
                "P2 low review": meta["p2"]["label"]}
    out = []
    for problem, sel_name, res_name, metric, negated, test_col in PROBLEMS:
        sel = need(sel_name)
        res = need(res_name).set_index("model")
        sign = -1.0 if negated else 1.0
        for _, r in sel.iterrows():
            scores = sign * r[[f"score_{f}" for f in FOLDS]].astype(float).to_numpy()
            val = sign * float(r["score_validation"])
            row = {"problem": problem, "model": r["model"], "metric": metric,
                   "family": res.loc[r["model"], "family"] if r["model"] in res.index else "",
                   "rung": res.loc[r["model"], "rung"] if r["model"] in res.index else np.nan}
            row.update({f: float(s) for f, s in zip(FOLDS, scores)})
            row.update({
                "validation": val,
                "cv_mean": float(scores.mean()),
                "cv_sd": float(scores.std(ddof=0)),
                "cv_min": float(scores.min()),
                "cv_max": float(scores.max()),
                "cv_range": float(scores.max() - scores.min()),
                "test": float(res.loc[r["model"], test_col]) if r["model"] in res.index else np.nan,
                "mean_rank_recent": float(r["mean_rank"]),
                "mean_rank_all": float(r["mean_rank_all"]),
                "windows_voting": r["windows_voting"],
                "deployed": r["model"] == deployed[problem],
            })
            out.append(row)
    table = pd.DataFrame(out)
    # Within each problem, the gap between the best two candidates' CV means
    # against the fold-to-fold spread is the number the feedback is really
    # asking about: can cross-validation separate them at all?
    return table


# ===========================================================================
# 2. Selected hyperparameters
# ===========================================================================
def _rows_from_params(problem: str, model: str, family: str, rung, params: dict,
                      estimator_class: str, role: str = "") -> list[dict]:
    rows = []
    shown = SHOWN.get(estimator_class, sorted(params))
    for name in shown:
        if name in params:
            rows.append({"problem": problem, "model": model, "role": role,
                         "family": family, "rung": rung,
                         "estimator": estimator_class, "parameter": name,
                         "value": params[name]})
    if not shown:
        rows.append({"problem": problem, "model": model, "role": role,
                     "family": family, "rung": rung, "estimator": estimator_class,
                     "parameter": "(no free parameters)", "value": ""})
    return rows


def hyperparameter_table() -> pd.DataFrame:
    rows: list[dict] = []
    # -- every searched model, from the results tables ----------------------
    for problem, _, res_name, _, _, _ in PROBLEMS:
        res = need(res_name)
        for _, r in res.iterrows():
            if r["family"] in ("Baseline", "Derived", "Ensemble") or pd.isna(r.get("params")):
                continue
            params = ast.literal_eval(str(r["params"])) if str(r["params"]).strip() else {}
            cls = (str(r["model"]).replace(" (tuned L2)", "")
                   .replace("LogisticRegression", "LogisticRegression"))
            if cls == "Ridge":
                cls = "Ridge"
            rows.extend(_rows_from_params(problem, r["model"], r["family"], r["rung"],
                                          params, cls, role="candidate"))
            if cls.startswith("RandomForest"):
                # n_estimators is fixed at 150 in modeling.py, not searched.
                rows.append({"problem": problem, "model": r["model"], "role": "candidate",
                             "family": r["family"], "rung": r["rung"], "estimator": cls,
                             "parameter": "n_estimators", "value": 150})

    # -- the deployed ensembles, read from the pickles -----------------------
    for problem, path, label_key in [("P1A lead time", "p1a_leadtime.joblib", "p1a"),
                                     ("P2 low review", "p2_lowreview.joblib", "p2")]:
        bundle = M.load_model(config.MODELS_DIR / path)
        est = bundle["estimator"]
        label = bundle["label"]
        weights = getattr(est, "weights", None)
        if weights is not None:
            rows.append({"problem": problem, "model": label, "role": "deployed ensemble",
                         "family": "Ensemble", "rung": "", "estimator": type(est).__name__,
                         "parameter": "weights (linear : tree)",
                         "value": ":".join(str(w) for w in weights)})
        fe = getattr(est, "final_estimator_", None)
        if fe is not None:
            coef = np.ravel(getattr(fe, "coef_", []))
            rows.append({"problem": problem, "model": label, "role": "deployed ensemble",
                         "family": "Ensemble", "rung": "", "estimator": type(est).__name__,
                         "parameter": "meta-learner",
                         "value": f"{type(fe).__name__}(C={fe.get_params().get('C', '')})"})
            if len(coef) == 2:
                rows.append({"problem": problem, "model": label, "role": "deployed ensemble",
                             "family": "Ensemble", "rung": "", "estimator": type(est).__name__,
                             "parameter": "meta-learner coefficients (linear, tree)",
                             "value": f"{coef[0]:.3f}, {coef[1]:.3f}"})
        for member_name, member in est.estimators:
            inner = member.steps[-1][1] if hasattr(member, "steps") else member
            cls = type(inner).__name__
            params = {k: v for k, v in inner.get_params().items() if k in SHOWN.get(cls, [])}
            rows.extend(_rows_from_params(problem, label, "Ensemble", "", params, cls,
                                          role=f"deployed member: {member_name}"))
    table = pd.DataFrame(rows)
    table["value"] = table["value"].map(
        lambda v: f"{v:.4g}" if isinstance(v, float) else str(v))
    return table


# ===========================================================================
if __name__ == "__main__":
    folds = cv_fold_table()
    folds.to_csv(RESULTS / "cv_folds.csv", index=False)
    hp = hyperparameter_table()
    hp.to_csv(RESULTS / "hyperparameters.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    show = ["problem", "model"] + FOLDS + ["validation", "cv_mean", "cv_sd", "test",
                                           "mean_rank_recent", "deployed"]
    print(folds[show].round(3).to_string(index=False))
    print()
    print(hp.to_string(index=False))
    print(f"\nwrote {RESULTS / 'cv_folds.csv'} and {RESULTS / 'hyperparameters.csv'}")
