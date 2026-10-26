"""Regenerate every number quoted in the Phase 3 (final) report.

    python src/phase2_stats.py       # the Phase 2 contract, unchanged
    python src/phase3_stats.py       # that, plus everything Phase 3 added

The final report is the standalone successor of the Phase 2 report, so it quotes
every number the Phase 2 report quoted - those keys are reproduced here verbatim
by importing src/phase2_stats.py - and adds the numbers behind the Phase 2
feedback: the fold-to-fold variation of every candidate, the selected
hyperparameters, the deployed estimators as actually pickled, and the outreach
pilot design. Phase 1's exploratory numbers remain the contract of
src/report_stats.py and report/stats.txt.

Anything neither script can produce is not allowed into the report.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import modeling as M
import phase2_stats as P2   # executes the Phase 2 contract on import

RESULTS = config.PROJECT_ROOT / "report" / "results"
S: dict[str, object] = dict(P2.S)
r = P2.r
need = P2.need


def slug(label: str) -> str:
    s = str(label).lower().replace("derived: ", "derived_")
    s = re.sub(r"[()',+]", "", s)
    s = s.replace(">", "gt").replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", s).strip("_")


# ===========================================================================
# 14. Fold-to-fold variation (Phase 2 feedback, point 1)
# ===========================================================================
folds = need("cv_folds.csv")
FOLD_COLS = [f"fold{i}" for i in range(1, 6)]
for problem, tag in [("P1A lead time", "p1a"), ("P1B late flag", "p1b"),
                     ("P2 low review", "p2")]:
    part = folds[folds["problem"] == problem].sort_values("mean_rank_recent")
    for _, row in part.iterrows():
        k = f"cvf_{tag}_{slug(row['model'])}"
        S[f"{k}_cv_mean"] = r(row["cv_mean"])
        S[f"{k}_cv_sd"] = r(row["cv_sd"])
        S[f"{k}_cv_min"] = r(row["cv_min"])
        S[f"{k}_cv_max"] = r(row["cv_max"])
        S[f"{k}_validation"] = r(row["validation"])
        S[f"{k}_test"] = r(row["test"])
        S[f"{k}_mean_rank"] = r(row["mean_rank_recent"], 2)
        for f in FOLD_COLS:
            S[f"{k}_{f}"] = r(row[f])
    # The question behind the feedback: can CV separate the top candidates?
    top = part.sort_values("cv_mean", ascending=(tag == "p1a")).reset_index(drop=True)
    S[f"cvf_{tag}_best_cv_model"] = str(top.loc[0, "model"])
    S[f"cvf_{tag}_second_cv_model"] = str(top.loc[1, "model"])
    gap = abs(float(top.loc[0, "cv_mean"]) - float(top.loc[1, "cv_mean"]))
    S[f"cvf_{tag}_top2_cv_gap"] = r(gap, 5)
    S[f"cvf_{tag}_median_cv_sd"] = r(part["cv_sd"].median())
    # How many fold-to-fold standard deviations separate the best two? Below one
    # the cross-validated estimate cannot order them.
    S[f"cvf_{tag}_top2_gap_in_sds"] = r(gap / part["cv_sd"].median(), 3)
    # How much of the fold-to-fold spread is the window rather than the model:
    # the spread of fold means across windows against the spread of models
    # within a window.
    window_means = part[FOLD_COLS].astype(float).mean(axis=0)
    S[f"cvf_{tag}_window_mean_min"] = r(window_means.min())
    S[f"cvf_{tag}_window_mean_max"] = r(window_means.max())
    S[f"cvf_{tag}_within_window_sd_median"] = r(part[FOLD_COLS].astype(float).std(axis=0, ddof=0).median())
    S[f"cvf_{tag}_across_window_sd"] = r(window_means.std(ddof=0))
    S[f"cvf_{tag}_deployed_model"] = str(part.loc[part["deployed"], "model"].iloc[0])
    S[f"cvf_{tag}_deployed_mean_rank"] = r(part.loc[part["deployed"], "mean_rank_recent"].iloc[0], 2)

# The deployed review model against the single boosted model, on every view.
_p2 = folds[folds["problem"] == "P2 low review"].set_index("model")
_stack, _hgb = "StackingClassifier (A+B)", "HistGradientBoostingClassifier"
S["cvf_p2_stack_minus_hgb_cv"] = r(_p2.loc[_stack, "cv_mean"] - _p2.loc[_hgb, "cv_mean"], 4)
S["cvf_p2_stack_minus_hgb_val"] = r(_p2.loc[_stack, "validation"] - _p2.loc[_hgb, "validation"], 4)
S["cvf_p2_stack_minus_hgb_test"] = r(_p2.loc[_stack, "test"] - _p2.loc[_hgb, "test"], 4)
S["cvf_p2_stack_mean_rank"] = r(_p2.loc[_stack, "mean_rank_recent"], 2)
S["cvf_p2_hgb_mean_rank"] = r(_p2.loc[_hgb, "mean_rank_recent"], 2)
S["cvf_p2_stack_windows_won"] = int(sum(
    _p2.loc[_stack, w] > _p2.loc[_hgb, w] for w in FOLD_COLS + ["validation"]))

# ===========================================================================
# 15. Selected hyperparameters (Phase 2 feedback, point 3)
# ===========================================================================
hp = need("hyperparameters.csv")
for _, row in hp.iterrows():
    if row["role"] == "candidate":
        key = f"hp_{slug(row['problem'].split()[0])}_{slug(row['model'])}_{slug(row['parameter'])}"
    else:
        member = row["role"].split(":")[-1].strip() if ":" in row["role"] else "ensemble"
        key = f"hp_{slug(row['problem'].split()[0])}_deployed_{member}_{slug(row['parameter'])}"
    S[key] = row["value"]
S["hp_rows"] = len(hp)
S["hp_models_searched"] = int(hp.loc[hp["role"] == "candidate", "model"].nunique())

# ===========================================================================
# 16. The deployed estimators, as pickled (Phase 2 feedback, point 2)
# ===========================================================================
meta = json.loads((config.MODELS_DIR / "model_metadata.json").read_text("utf-8"))
for tag, path in [("p1a", "p1a_leadtime.joblib"), ("p2", "p2_lowreview.joblib"),
                  ("p1b", "p1b_margin_to_risk.joblib")]:
    bundle = M.load_model(config.MODELS_DIR / path)
    est = bundle["estimator"]
    S[f"deployed_{tag}_label"] = bundle["label"]
    S[f"deployed_{tag}_class"] = type(est).__name__
    S[f"deployed_{tag}_artifact_kb"] = r((config.MODELS_DIR / path).stat().st_size / 1e3, 0)
    S[f"deployed_{tag}_sklearn"] = bundle["sklearn_version"]
    if hasattr(est, "estimators"):
        members = []
        for name, member in est.estimators:
            inner = member.steps[-1][1] if hasattr(member, "steps") else member
            members.append(f"{name}={type(inner).__name__}")
        S[f"deployed_{tag}_members"] = "; ".join(members)
    if getattr(est, "weights", None) is not None:
        S[f"deployed_{tag}_weights"] = ":".join(str(w) for w in est.weights)
    fe = getattr(est, "final_estimator_", None)
    if fe is not None:
        coef = np.ravel(fe.coef_)
        S[f"deployed_{tag}_meta_learner"] = type(fe).__name__
        S[f"deployed_{tag}_meta_coef_linear"] = r(coef[0])
        S[f"deployed_{tag}_meta_coef_tree"] = r(coef[1])
        S[f"deployed_{tag}_meta_tree_over_linear"] = r(coef[1] / coef[0], 1)
S["deployed_p1a_matches_metadata"] = S["deployed_p1a_label"] == meta["p1a"]["label"]
S["deployed_p2_matches_metadata"] = S["deployed_p2_label"] == meta["p2"]["label"]
S["deployed_p2_threshold"] = r(meta["p2"]["threshold"], 2)
S["deployed_p1b_threshold"] = r(meta["p1b"]["threshold"], 2)
S["deployed_python_version"] = "3.12"
S["deployed_models_total_kb"] = r(sum(
    (config.MODELS_DIR / p).stat().st_size for p in
    ("p1a_leadtime.joblib", "p2_lowreview.joblib", "p1b_margin_to_risk.joblib")) / 1e3, 0)

# ===========================================================================
# 17. The outreach pilot (Phase 2 feedback, point 4)
# ===========================================================================
design = need("pilot_design.csv")
be = need("pilot_breakeven.csv").set_index("k_per_day")
S["pilot_alpha"] = 0.05
S["pilot_breakeven_absolute_reduction"] = r(be["breakeven_absolute_reduction"].iloc[0])
for k in be.index:
    S[f"pilot_k{k}_control_rate"] = r(be.loc[k, "control_rate"])
    S[f"pilot_k{k}_flagged_pct"] = r(be.loc[k, "flagged_pct"], 1)
    S[f"pilot_k{k}_breakeven_rel_pct"] = r(100 * be.loc[k, "breakeven_relative_reduction"], 1)
    S[f"pilot_k{k}_breakeven_one_in"] = r(1 / be.loc[k, "breakeven_relative_reduction"], 1)
    S[f"pilot_k{k}_weeks_80_at_breakeven"] = int(be.loc[k, "weeks_80_at_breakeven"])
    part = design[design["k_per_day"] == k].set_index("relative_reduction")
    for rr in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        if rr in part.index:
            t = f"r{int(100 * rr)}"
            S[f"pilot_k{k}_{t}_treated_rate"] = r(part.loc[rr, "treated_rate"])
            S[f"pilot_k{k}_{t}_n_per_arm_80"] = int(part.loc[rr, "n_per_arm_80"])
            S[f"pilot_k{k}_{t}_weeks_80"] = int(part.loc[rr, "weeks_80"])
            S[f"pilot_k{k}_{t}_n_per_arm_90"] = int(part.loc[rr, "n_per_arm_90"])
            S[f"pilot_k{k}_{t}_weeks_90"] = int(part.loc[rr, "weeks_90"])
            S[f"pilot_k{k}_{t}_net_per_contact"] = r(part.loc[rr, "net_benefit_per_contact"], 2)

# ===========================================================================
# 18. Figures and tables the final report draws from
# ===========================================================================
S["n_result_tables"] = len(list(RESULTS.glob("*.csv")))
S["n_figures"] = len(list(config.FIGURES_DIR.glob("fig*.pdf")))


# ===========================================================================
if __name__ == "__main__":
    out = config.PROJECT_ROOT / "report" / "stats_phase3.txt"
    lines = [f"{k:52s} {v}" for k, v in S.items()]
    out.write_text("\n".join(lines), encoding="utf-8")
    new = [k for k in S if k not in P2.S]
    print("\n".join(f"{k:52s} {S[k]}" for k in new))
    print(f"\nwrote {out}  ({len(S)} values, {len(new)} new in Phase 3)")
