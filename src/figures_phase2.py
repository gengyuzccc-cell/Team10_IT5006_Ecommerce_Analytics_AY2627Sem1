"""Draw every Phase 2 report figure from the tables train_models.py wrote.

    python src/train_models.py       # fits, writes report/results/
    python src/figures_phase2.py     # draws, writes report/figures/

Separated from the fitting for one reason: a figure should never be able to
disagree with a table. Both come from the same CSVs, so redrawing after a
restyle cannot quietly change a number, and a figure can be iterated on without
a 40-minute refit.

Output is vector PDF at the Phase 1 house size, so the two reports look like one
piece of work.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

import config
from config import PALETTE
from viz import pct, save_fig, use_report_style

RESULTS = config.PROJECT_ROOT / "report" / "results"
P, A, G, GR, M = (PALETTE["primary"], PALETTE["accent"], PALETTE["gold"],
                  PALETTE["green"], PALETTE["muted"])
FAMILY_COLOUR = {"Baseline": M, "Linear": P, "Tree-based": GR,
                 "Ensemble": G, "Derived": A}


def load(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `python src/train_models.py` first.")
    return pd.read_csv(path)


def short(label: str, width: int = 26) -> str:
    label = (label.replace("Baseline: ", "").replace("Regressor", "")
             .replace("Classifier", "").replace("HistGradientBoosting", "HistGB")
             .replace("LogisticRegression", "LogReg")
             .replace("LinearRegression", "OLS").replace("DecisionTree", "Tree")
             .replace("RandomForest", "Forest").strip())
    return label if len(label) <= width else label[: width - 1] + "…"


# ===========================================================================
def fig_regression_ladder() -> str:
    """F1. Does the added complexity pay? One bar per rung, CV error alongside."""
    reg = load("reg_p1a.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 3.5),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    d = reg.sort_values("mae", ascending=False)
    y = np.arange(len(d))
    ax1.barh(y, d["mae"], color=[FAMILY_COLOUR.get(f, M) for f in d["family"]],
             height=0.72)
    ax1.set_yticks(y)
    ax1.set_yticklabels([short(m) for m in d["model"]])
    for i, v in enumerate(d["mae"]):
        ax1.text(v + 0.08, i, f"{v:.2f}", va="center", fontsize=7.5, color="#333")
    ax1.set_xlabel("test MAE (days)")
    ax1.set_title("(a) Lead-time error by model", loc="left")
    ax1.set_xlim(0, d["mae"].max() * 1.16)
    ax1.grid(axis="y", visible=False)

    # The three protocols side by side. Selection uses the middle one, and the
    # point of the panel is that the first does not order the models the way the
    # other two do.
    m = reg.dropna(subset=["cv_mae"]).sort_values("rung").reset_index(drop=True)
    x = np.arange(len(m))
    ax2.errorbar(x, m["cv_mae"], yerr=m["cv_std"], fmt="o-", color=M,
                 capsize=3, label="expanding-window CV", markersize=4, lw=1.1)
    ax2.plot(x, m["val_mae"], "^-", color=P, label="validation (selection)",
             markersize=5)
    ax2.plot(x, m["mae"], "s--", color=A, label="held-out test", markersize=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([short(v, 13) for v in m["model"]], rotation=38,
                        ha="right", fontsize=6.5)
    ax2.set_ylabel("MAE (days)")
    ax2.set_title("(b) Three protocols, same models", loc="left")
    ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return save_fig(fig, "fig10_regression_ladder")


def fig_regression_residuals() -> str:
    """F2. Where the regression is wrong, and whether the drift shows up in it."""
    p = load("predictions_p1a_test.csv")
    p["order_purchase_timestamp"] = pd.to_datetime(p["order_purchase_timestamp"])
    by_month = load("segments_p1a_month.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 3.3))
    lim = (0, min(45, p["actual_days"].quantile(0.995)))
    ax1.hexbin(p["actual_days"], p["pred_days"], gridsize=38, extent=(*lim, *lim),
               cmap="Blues", mincnt=1, linewidths=0)
    ax1.plot(lim, lim, color=A, lw=1.2, ls="--", label="perfect prediction")
    ax1.set_xlim(lim), ax1.set_ylim(lim)
    ax1.set_xlabel("actual lead time (days)")
    ax1.set_ylabel("predicted (days)")
    ax1.set_title("(a) Predicted against actual", loc="left")
    ax1.legend(loc="upper left")

    x = np.arange(len(by_month))
    ax2.bar(x - 0.2, by_month["mae"], 0.4, color=P, label="MAE")
    ax2.bar(x + 0.2, by_month["bias"], 0.4, color=A, label="mean bias")
    ax2.axhline(0, color="#666", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(by_month["_month"], rotation=0, fontsize=8)
    ax2.set_ylabel("days")
    ax2.set_title("(b) Error and bias by test month", loc="left")
    ax2.legend()
    fig.tight_layout()
    return save_fig(fig, "fig11_regression_residuals")


def _curves(ax_pr, ax_roc, series: list[tuple[str, np.ndarray, np.ndarray, str]],
            floor: float) -> None:
    for label, y, s, colour in series:
        pr, rc, _ = precision_recall_curve(y, s)
        ax_pr.plot(rc, pr, color=colour, label=label)
        fp, tp, _ = roc_curve(y, s)
        ax_roc.plot(fp, tp, color=colour, label=label)
    ax_pr.axhline(floor, color=M, ls=":", lw=1.2, label=f"no skill ({floor:.3f})")
    ax_pr.set_xlabel("recall"), ax_pr.set_ylabel("precision")
    ax_roc.plot([0, 1], [0, 1], color=M, ls=":", lw=1.2, label="no skill")
    ax_roc.set_xlabel("false positive rate"), ax_roc.set_ylabel("true positive rate")


def fig_late_routes() -> str:
    """F3. The headline: one model with two outputs beats a purpose-built classifier."""
    p = load("predictions_p1b_test.csv")
    y = p["actual_late"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 3.2))
    _curves(ax1, ax2, [
        (f"direct classifier ({short(p['direct_model'].iloc[0], 18)})",
         y, p["p_late_direct"].to_numpy(), P),
        ("derived from the lead-time regression", y, p["p_late_derived"].to_numpy(), A),
    ], float(y.mean()))
    ax1.set_title("(a) Precision--recall", loc="left")
    ax2.set_title("(b) ROC", loc="left")
    ax1.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return save_fig(fig, "fig12_late_routes")


def fig_low_review_curves() -> str:
    """F4. Problem 2 against its no-skill floor, and the family comparison."""
    p = load("predictions_p2_test.csv")
    low = load("clf_p2.csv")
    y = p["actual_low_review"].to_numpy()
    score = p["p_low_review"].to_numpy()
    floor = float(y.mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 3.2))
    pr, rc, _ = precision_recall_curve(y, score)
    ax1.plot(rc, pr, color=A, label="selected model")
    ax1.axhline(floor, color=M, ls=":", lw=1.2, label=f"no skill ({floor:.3f})")
    ax1.set_xlabel("recall"), ax1.set_ylabel("precision")
    ax1.set_title("(a) Precision--recall against the 1--2 star base rate", loc="left")
    ax1.legend(loc="upper right", fontsize=7.5)

    # A model-by-model strip, so the family comparison sits beside the curve.
    d = low.sort_values("pr_auc")
    yy = np.arange(len(d))
    ax2.barh(yy, d["pr_auc"], color=[FAMILY_COLOUR.get(f, M) for f in d["family"]],
             height=0.7)
    ax2.axvline(d["pr_auc_floor"].iloc[0], color=M, ls=":", lw=1.2)
    ax2.text(d["pr_auc_floor"].iloc[0], len(d) - 0.4,
             f" no skill {d['pr_auc_floor'].iloc[0]:.3f}", fontsize=7, color=M,
             va="top")
    ax2.set_yticks(yy)
    ax2.set_yticklabels([short(m, 22) for m in d["model"]], fontsize=7.5)
    ax2.set_xlabel("test PR-AUC")
    ax2.set_title("(b) PR-AUC by model", loc="left")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    return save_fig(fig, "fig13_low_review")


def fig_calibration() -> str:
    """F5. What weighting the loss does to the probabilities."""
    cal = load("calibration_p2.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 3.2),
                                   gridspec_kw={"width_ratios": [1, 1]})
    ax1.plot([0, cal["mean_predicted"].max()], [0, cal["mean_predicted"].max()],
             ls="--", color=M, lw=1.1, label="perfect calibration")
    ax1.plot(cal["mean_predicted"], cal["observed_rate"], "o-", color=P,
             markersize=4, label="selected model")
    for _, r in cal.iterrows():
        ax1.annotate(f"n={int(r['n']):,}", (r["mean_predicted"], r["observed_rate"]),
                     textcoords="offset points", xytext=(4, -8), fontsize=6, color=M)
    ax1.set_xlabel("mean predicted risk"), ax1.set_ylabel("observed rate")
    ax1.set_title("(a) Reliability, fitted on the observed prior", loc="left")
    ax1.legend(loc="upper left")

    try:
        abl = load("weight_ablation.csv")
        x = np.arange(len(abl))
        ax2.bar(x - 0.19, abl["mean_predicted"], 0.38, color=P, label="mean predicted risk")
        ax2.bar(x + 0.19, abl["actual_rate"], 0.38, color=GR, label="actual rate")
        for i, b in enumerate(abl["brier"]):
            ax2.text(i, max(abl["mean_predicted"].iloc[i], abl["actual_rate"].iloc[i]) + 0.012,
                     f"Brier {b:.3f}", ha="center", fontsize=7.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels([t.replace("class_weight=", "") for t in abl["model"]],
                            fontsize=7.5)
        ax2.set_ylim(0, max(abl["mean_predicted"].max(), abl["actual_rate"].max()) * 1.28)
        ax2.set_title("(b) What class weighting costs", loc="left")
        ax2.legend(loc="upper left", fontsize=7.5)
    except SystemExit:
        ax2.axis("off")
    fig.tight_layout()
    return save_fig(fig, "fig14_calibration")


def fig_threshold_and_cost() -> str:
    """F6. Choosing the operating point, and what it is worth."""
    sweep = load("threshold_sweep_p2.csv")
    cost = load("cost_p2.csv")
    meta = json.loads((config.MODELS_DIR / "model_metadata.json").read_text("utf-8"))
    thr = float(meta["p2"]["threshold"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 3.2))
    ax1.plot(sweep["threshold"], sweep["precision"], color=P, label="precision")
    ax1.plot(sweep["threshold"], sweep["recall"], color=A, label="recall")
    ax1.plot(sweep["threshold"], sweep["f1"], color=GR, lw=1.9, label="F1")
    ax1.axvline(thr, color="#444", ls="--", lw=1.1)
    ax1.text(thr, 0.98, f" tuned on validation: {thr:.2f}", fontsize=7.5,
             color="#444", va="top")
    ax1.set_xlabel("decision threshold"), ax1.set_ylim(0, 1.02)
    ax1.set_title("(a) The precision--recall trade", loc="left")
    ax1.legend(loc="upper right")

    ax2.plot(cost["threshold"], cost["net_benefit_per_1k"], color=P)
    best = cost.loc[cost["net_benefit"].idxmax()]
    ax2.axvline(best["threshold"], color=A, ls="--", lw=1.1)
    ax2.text(best["threshold"], cost["net_benefit_per_1k"].max(),
             f" max at {best['threshold']:.2f}", fontsize=7.5, color=A, va="top")
    ax2.axhline(0, color=M, lw=0.8)
    ax2.set_xlabel("decision threshold")
    ax2.set_ylabel("expected net benefit (R$ per 1,000 orders)")
    ax2.set_title("(b) Under the stated cost assumptions", loc="left")
    fig.tight_layout()
    return save_fig(fig, "fig15_threshold_cost")


def fig_cv_protocol() -> str:
    """F7. What the invalid protocol would have told us, raw and normalised."""
    cvp = load("cv_protocol.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 3.0))
    x = np.arange(len(cvp))
    colours = [P, G, A]
    ax1.bar(x, cvp["mean"], 0.55, yerr=cvp["std"].fillna(0), capsize=4,
            color=colours[: len(cvp)])
    for i, (m, f) in enumerate(zip(cvp["mean"], cvp["floor"])):
        ax1.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)
        ax1.plot([i - 0.28, i + 0.28], [f, f], color=M, ls=":", lw=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["temporal\nCV", "shuffled\nCV", "forward\ntest"], fontsize=8)
    ax1.set_ylabel("PR-AUC")
    ax1.set_title("(a) Raw PR-AUC (dotted: each period's no-skill floor)", loc="left")

    ax2.bar(x, cvp["lift"], 0.55, color=colours[: len(cvp)])
    for i, v in enumerate(cvp["lift"]):
        ax2.text(i, v + 0.05, f"{v:.2f}×", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["temporal\nCV", "shuffled\nCV", "forward\ntest"], fontsize=8)
    ax2.set_ylabel("PR-AUC ÷ no-skill floor")
    ax2.set_ylim(0, cvp["lift"].max() * 1.25)
    ax2.set_title("(b) The same numbers, normalised", loc="left")
    fig.tight_layout()
    return save_fig(fig, "fig16_cv_protocol")


def fig_importance() -> str:
    """F8. What each selected model actually relies on."""
    fig, axes = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 3.4))
    for ax, name, title, colour, unit in [
        (axes[0], "perm_p1a.csv", "(a) Lead time — increase in MAE", P, "days"),
        (axes[1], "perm_p2.csv", "(b) Review risk — drop in PR-AUC", A, "PR-AUC"),
    ]:
        d = load(name).head(10).iloc[::-1]
        y = np.arange(len(d))
        ax.barh(y, d["importance"].abs(), xerr=d["std"], color=colour, height=0.7,
                error_kw={"lw": 0.8, "ecolor": "#888"})
        ax.set_yticks(y)
        ax.set_yticklabels([f.replace("_", " ") for f in d["feature"]], fontsize=7.5)
        ax.set_xlabel(f"permutation importance ({unit})")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save_fig(fig, "fig17_importance")


def fig_segments() -> str:
    """F9. Does it work everywhere, or only where the data is thickest?"""
    reg_seg = load("segments_p1a_region.csv")
    clf_seg = load("segments_p2_region.csv")
    if reg_seg.empty or clf_seg.empty:
        raise RuntimeError("segment tables are empty - every segment fell below min_n")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 3.1))

    d = reg_seg.sort_values("mae")
    x = np.arange(len(d))
    ax1.bar(x, d["mae"], 0.55, color=P)
    for i, (v, n) in enumerate(zip(d["mae"], d["n"])):
        ax1.text(i, v + 0.06, f"n={int(n):,}", ha="center", fontsize=7, color=M)
    ax1.set_xticks(x), ax1.set_xticklabels(d["customer_region"], fontsize=7.5, rotation=20)
    ax1.set_ylabel("MAE (days)")
    ax1.set_ylim(0, d["mae"].max() * 1.2)
    ax1.set_title("(a) Lead-time error by customer region", loc="left")

    d = clf_seg.sort_values("roc_auc")
    x = np.arange(len(d))
    ax2.bar(x, d["roc_auc"], 0.55, color=A)
    ax2.axhline(0.5, color=M, ls=":", lw=1.2)
    ax2.text(len(d) - 0.4, 0.51, "no skill", fontsize=7, color=M, ha="right")
    for i, (v, n) in enumerate(zip(d["roc_auc"], d["n"])):
        ax2.text(i, v + 0.012, f"n={int(n):,}", ha="center", fontsize=7, color=M)
    ax2.set_xticks(x), ax2.set_xticklabels(d["customer_region"], fontsize=7.5, rotation=20)
    ax2.set_ylabel("ROC-AUC")
    ax2.set_ylim(0.4, min(1.0, d["roc_auc"].max() * 1.12))
    ax2.set_title("(b) Review-risk discrimination by region", loc="left")
    fig.tight_layout()
    return save_fig(fig, "fig18_segments")


def fig_extrapolation() -> str:
    """F10. Why blending a linear model with a boosted one helps under drift.

    A tree can only predict values it saw during training: every leaf holds an
    average of training targets, so the whole model is bounded by the training
    range and, more sharply, by the density of that range. Lead times in the test
    window are systematically shorter than anything common in training, so the
    tree cannot follow them down. A linear model has no such bound - it
    extrapolates the level. Averaging the two is not a generic ensembling win; it
    is one member covering the other's specific failure.
    """
    p = load("predictions_p1a_test.csv")
    reg = load("reg_p1a.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 3.1))
    ax1.hist(p["actual_days"].clip(upper=45), bins=45, color=P, alpha=0.55,
             label="actual, test window", density=True)
    ax1.hist(p["pred_days"].clip(upper=45), bins=45, color=A, alpha=0.55,
             label="predicted", density=True)
    ax1.axvline(p["actual_days"].mean(), color=P, lw=1.4, ls="--")
    ax1.axvline(p["pred_days"].mean(), color=A, lw=1.4, ls="--")
    ax1.set_xlabel("lead time (days)"), ax1.set_ylabel("density")
    ax1.set_title("(a) Predicted against actual distribution", loc="left")
    ax1.legend(fontsize=7.5)

    d = reg[reg["family"] != "Baseline"].sort_values("val_mae")
    yy = np.arange(len(d))
    ax2.barh(yy, d["val_mae"], height=0.36, color=P, label="validation MAE")
    ax2.barh(yy + 0.38, d["mae"], height=0.36, color=A, label="test MAE")
    ax2.set_yticks(yy + 0.19)
    ax2.set_yticklabels([short(m, 22) for m in d["model"]], fontsize=7.5)
    ax2.set_xlabel("MAE (days)")
    ax2.set_title("(b) Validation orders the models as the test window does", loc="left")
    ax2.legend(fontsize=7.5, loc="lower right")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    return save_fig(fig, "fig19_extrapolation")


# ===========================================================================
FIGURES = [fig_regression_ladder, fig_regression_residuals, fig_late_routes,
           fig_low_review_curves, fig_calibration, fig_threshold_and_cost,
           fig_cv_protocol, fig_importance, fig_segments, fig_extrapolation]

if __name__ == "__main__":
    use_report_style()
    for fn in FIGURES:
        try:
            print(f"  {Path(fn()).name}")
        except SystemExit as exc:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(FIGURES)} figures into {config.FIGURES_DIR}")
