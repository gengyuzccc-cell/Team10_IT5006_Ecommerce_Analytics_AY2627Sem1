"""Regenerate every number quoted in the Phase 2 report.

    python src/train_models.py     # first: fits everything, writes report/results/
    python src/phase2_stats.py     # then: distils it into report/stats_phase2.txt

The Phase 1 report shipped with src/report_stats.py, which recomputed all 170 of
its numeric claims so the manuscript could be diffed against the code. This is
the same contract for Phase 2. The manuscript is written by hand; not one figure
in it is typed from memory.

Anything this script cannot produce is not allowed into the report.
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
from data_load import load_all, load_analysis
from features import (
    CATEGORICAL_FEATURES, CHECKOUT_FEATURES, DELIVERY_FEATURES, HISTORY_FEATURES,
    MODEL_FEATURES, NUMERIC_CHECKOUT, build_features, make_splits,
)

RESULTS = config.PROJECT_ROOT / "report" / "results"
S: dict[str, object] = {}


def need(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Run `python src/train_models.py` first - this "
            "script distils its output, it does not fit anything itself.")
    return pd.read_csv(path)


def r(x, nd: int = 3):
    """Round for the report, keeping NaN visible rather than silently zeroed."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


# ===========================================================================
# 1. Data, features and the split
# ===========================================================================
df = build_features(load_analysis(load_all()), with_history=True)
sp = make_splits(df)

S["analysis_rows"] = len(df)
S["train_rows"] = len(sp["train"])
S["val_rows"] = len(sp["val"])
S["test_rows"] = len(sp["test"])
S["straddling_customers_removed"] = sp["n_straddling_customers"]
S["n_features_phase1"] = len(MODEL_FEATURES)
S["n_features_checkout"] = len(CHECKOUT_FEATURES)
S["n_features_delivery"] = len(DELIVERY_FEATURES)
S["n_history_features"] = len(HISTORY_FEATURES)
S["n_categorical_features"] = len(CATEGORICAL_FEATURES)
S["n_numeric_checkout"] = len(NUMERIC_CHECKOUT)

# History coverage: how often the fallback chain has to step out to a broader key.
S["pct_orders_with_seller_history"] = r(
    100 * (df["hist_seller_n"] >= config.MIN_HISTORY).mean(), 1)
S["median_seller_history_n"] = int(df["hist_seller_n"].median())
S["pct_history_missing_after_fallback"] = r(
    100 * df[HISTORY_FEATURES].isna().any(axis=1).mean(), 2)
S["hist_platform_lead_min"] = r(df["hist_platform_lead_days"].min(), 1)
S["hist_platform_lead_max"] = r(df["hist_platform_lead_days"].max(), 1)

# Base rates, per split. The drop between train and test is what makes raw
# PR-AUC incomparable across periods.
for tag, part in [("train", sp["train"]), ("val", sp["val"]), ("test", sp["test"])]:
    S[f"late_rate_{tag}_pct"] = r(100 * part["target_is_late"].mean(), 2)
    S[f"low_review_rate_{tag}_pct"] = r(100 * part["target_low_review"].mean(), 2)
S["majority_acc_late_test"] = r(1 - sp["test"]["target_is_late"].mean())
S["majority_acc_low_review_test"] = r(1 - sp["test"]["target_low_review"].mean())

# Drift, restated for the modelling section.
S["train_mean_delivery_days"] = r(sp["train"]["target_delivery_days"].mean(), 2)
S["test_mean_delivery_days"] = r(sp["test"]["target_delivery_days"].mean(), 2)
S["delivery_drift_days"] = r(S["test_mean_delivery_days"] - S["train_mean_delivery_days"], 2)

# ===========================================================================
# 2. Problem 1A - lead-time regression
# ===========================================================================
reg = need("reg_p1a.csv")
meta = json.loads((config.MODELS_DIR / "model_metadata.json").read_text("utf-8"))
S["seed"] = meta["seed"]
S["search_draws"] = meta["n_search_iter"]
S["cv_splits"] = config.CV_SPLITS

for _, row in reg.iterrows():
    key = (str(row["model"]).lower()
           .replace("baseline: ", "").replace("'", "").replace("(", "")
           .replace(")", "").replace("+", "").replace("-", "_")
           .replace(" ", "_").replace("__", "_").strip("_"))
    S[f"p1a_{key}_mae"] = r(row["mae"], 2)
    S[f"p1a_{key}_rmse"] = r(row["rmse"], 2)
    S[f"p1a_{key}_r2"] = r(row["r2"])
    if not pd.isna(row.get("cv_mae")):
        S[f"p1a_{key}_cv_mae"] = r(row["cv_mae"], 2)
        S[f"p1a_{key}_cv_std"] = r(row["cv_std"], 2)

best = reg.loc[reg["family"] != "Baseline"].sort_values("mae").iloc[0]
mean_base = reg.loc[reg["model"].str.contains("training-period mean")].iloc[0]
promise = reg.loc[reg["model"].str.contains("delivery promise")].iloc[0]
S["p1a_selected_model"] = meta["p1a"]["label"]
S["p1a_selected_family"] = meta["p1a"]["family"]
S["p1a_selected_params"] = str(meta["p1a"]["params"])
S["p1a_test_mae"] = r(meta["p1a"]["test_mae"], 2)
S["p1a_test_rmse"] = r(meta["p1a"]["test_rmse"], 2)
S["p1a_test_r2"] = r(meta["p1a"]["test_r2"])
S["p1a_cv_mae"] = r(meta["p1a"]["cv_mae"], 2)
S["p1a_gain_vs_mean_pct"] = r(100 * (1 - best["mae"] / mean_base["mae"]), 1)
S["p1a_gain_vs_promise_pct"] = r(100 * (1 - best["mae"] / promise["mae"]), 1)

# The in-family complexity ladders, which is what "justify the added complexity"
# actually needs: does the next rung beat the one below it?
for fam in ("Linear", "Tree-based", "Ensemble"):
    part = reg[reg["family"] == fam].sort_values("rung")
    if len(part) >= 2:
        S[f"p1a_{fam.lower().replace('-', '_')}_ladder_gain_pct"] = r(
            100 * (1 - part["mae"].iloc[-1] / part["mae"].iloc[0]), 1)

# ===========================================================================
# 3. The Phase 1 re-check
# ===========================================================================
enc = need("encoding_diagnostic.csv")
for _, row in enc.iterrows():
    key = ("ols" if "Linear" in row["encoding"] else "ridge") + (
        "_full" if "full" in row["encoding"] else "_dropped")
    S[f"enc_{key}_rank"] = int(row["rank"])
    S[f"enc_{key}_cols"] = int(row["encoded_columns"])
    S[f"enc_{key}_deficiency"] = int(row["deficiency"])
    S[f"enc_{key}_max_coef"] = f"{row['max_abs_coefficient']:.2e}"
    S[f"enc_{key}_cv_mae"] = f"{row['cv_mae']:.3f}"
    S[f"enc_{key}_test_mae"] = r(row["test_mae"], 3)

# ===========================================================================
# 4. Problem 1B - late vs promise, direct against derived
# ===========================================================================
lat = need("clf_p1b.csv")
for _, row in lat.iterrows():
    key = (str(row["model"]).lower().replace("derived: ", "derived_")
           .replace("(", "").replace(")", "").replace(" ", "_")
           .replace(">", "gt").replace("-", "_").replace("__", "_").strip("_"))
    for m in ("roc_auc", "pr_auc", "pr_auc_lift", "balanced_accuracy",
              "precision", "recall", "f1", "accuracy", "threshold"):
        S[f"p1b_{key}_{m}"] = r(row[m])

direct = lat[lat["route"] == "direct"]
derived = lat[lat["route"] == "derived"]
best_direct = direct.loc[direct["pr_auc"].idxmax()]
best_derived = derived.loc[derived["pr_auc"].idxmax()]
S["p1b_best_direct_model"] = str(best_direct["model"])
S["p1b_best_direct_roc"] = r(best_direct["roc_auc"])
S["p1b_best_direct_prauc"] = r(best_direct["pr_auc"])
S["p1b_best_direct_balacc"] = r(best_direct["balanced_accuracy"])
S["p1b_derived_roc"] = r(best_derived["roc_auc"])
S["p1b_derived_prauc"] = r(best_derived["pr_auc"])
S["p1b_derived_balacc"] = r(best_derived["balanced_accuracy"])
S["p1b_derived_f1"] = r(best_derived["f1"])
S["p1b_prauc_floor"] = r(best_derived["pr_auc_floor"])
S["p1b_derived_over_direct_roc"] = r(best_derived["roc_auc"] - best_direct["roc_auc"])
S["p1b_derived_over_direct_prauc_ratio"] = r(
    best_derived["pr_auc"] / best_direct["pr_auc"], 2)
S["p1b_selected_threshold"] = r(meta["p1b"]["threshold"], 2)

# ===========================================================================
# 5. Problem 2 - low review at delivery
# ===========================================================================
low = need("clf_p2.csv")
for _, row in low.iterrows():
    key = (str(row["model"]).lower().replace("(", "").replace(")", "")
           .replace(" ", "_").replace("+", "").replace("-", "_")
           .replace(",", "").replace("__", "_").strip("_"))
    for m in ("roc_auc", "pr_auc", "pr_auc_lift", "balanced_accuracy",
              "precision", "recall", "f1", "accuracy", "brier", "threshold"):
        S[f"p2_{key}_{m}"] = r(row[m])
    if not pd.isna(row.get("cv_pr_auc")):
        S[f"p2_{key}_cv_prauc"] = r(row["cv_pr_auc"])

S["p2_selected_model"] = meta["p2"]["label"]
S["p2_selected_family"] = meta["p2"]["family"]
S["p2_selected_params"] = str(meta["p2"]["params"])
S["p2_cv_prauc"] = r(meta["p2"]["cv_pr_auc"])
S["p2_test_roc"] = r(meta["p2"]["test_roc_auc"])
S["p2_test_prauc"] = r(meta["p2"]["test_pr_auc"])
S["p2_prauc_floor"] = r(meta["p2"]["test_pr_auc_floor"])
S["p2_prauc_lift"] = r(meta["p2"]["test_pr_auc"] / meta["p2"]["test_pr_auc_floor"], 2)
S["p2_test_balacc"] = r(meta["p2"]["test_balanced_accuracy"])
S["p2_test_f1"] = r(meta["p2"]["test_f1"])
S["p2_test_brier"] = r(meta["p2"]["test_brier"], 4)
S["p2_threshold"] = r(meta["p2"]["threshold"], 2)
S["p2_unseen_customers_prauc"] = r(meta["p2"]["unseen_customers_pr_auc"])
S["p2_unseen_customers_n"] = meta["p2"]["unseen_customers_n"]

best_lin_p2 = low[low["family"] == "Linear"].sort_values("pr_auc").iloc[-1]
S["p2_best_linear_model"] = str(best_lin_p2["model"])
S["p2_best_linear_prauc"] = r(best_lin_p2["pr_auc"])
S["p2_tree_over_linear_prauc_ratio"] = r(
    meta["p2"]["test_pr_auc"] / best_lin_p2["pr_auc"], 2)
dt = low[low["model"].str.contains("DecisionTree")]
if len(dt):
    S["p2_decisiontree_prauc"] = r(dt.iloc[0]["pr_auc"])
    S["p2_decisiontree_balacc"] = r(dt.iloc[0]["balanced_accuracy"])

# ===========================================================================
# 6. Imbalance handling, calibration and the operating point
# ===========================================================================
try:
    abl = need("weight_ablation.csv")
    for _, row in abl.iterrows():
        key = "weighted" if "balanced" in str(row["model"]) else "unweighted"
        S[f"abl_{key}_brier"] = r(row["brier"], 4)
        S[f"abl_{key}_roc"] = r(row["roc_auc"])
        S[f"abl_{key}_prauc"] = r(row["pr_auc"])
        S[f"abl_{key}_mean_predicted"] = r(row["mean_predicted"])
    S["abl_actual_rate"] = r(abl.iloc[0]["actual_rate"])
    S["abl_brier_penalty_pct"] = r(
        100 * (S["abl_weighted_brier"] / S["abl_unweighted_brier"] - 1), 1)
    S["abl_risk_overstatement_x"] = r(
        S["abl_weighted_mean_predicted"] / S["abl_actual_rate"], 1)
except SystemExit:
    pass

cal = need("calibration_p2.csv")
S["cal_max_abs_gap"] = r(cal["gap"].abs().max())
S["cal_bins_reported"] = len(cal)

sweep = need("threshold_sweep_p2.csv")
at_default = sweep.loc[(sweep["threshold"] - 0.50).abs().idxmin()]
at_tuned = sweep.loc[(sweep["threshold"] - meta["p2"]["threshold"]).abs().idxmin()]
for tag, row in [("default", at_default), ("tuned", at_tuned)]:
    for m in ("precision", "recall", "f1", "balanced_accuracy", "flagged_pct"):
        S[f"p2_thr_{tag}_{m}"] = r(row[m])
S["p2_thr_f1_gain"] = r(at_tuned["f1"] - at_default["f1"])

cost = need("cost_p2.csv")
opt = cost.loc[cost["net_benefit"].idxmax()]
S["p2_cost_optimal_threshold"] = r(opt["threshold"], 2)
S["p2_cost_optimal_flagged_pct"] = r(100 * opt["flagged"] / cost["flagged"].max()
                                     if cost["flagged"].max() else np.nan, 1)
S["p2_cost_net_benefit_per_1k"] = r(opt["net_benefit_per_1k"], 1)
S["p2_cost_outreach_assumption"] = config.COSTS["outreach_cost"]
S["p2_cost_low_review_assumption"] = config.COSTS["low_review_cost"]

# ===========================================================================
# 7. The cross-validation protocol exhibit
# ===========================================================================
cvp = need("cv_protocol.csv")
for _, row in cvp.iterrows():
    key = ("temporal" if "TimeSeries" in row["protocol"]
           else "shuffled" if "Group" in row["protocol"] else "forward_test")
    S[f"cv_{key}_prauc"] = r(row["mean"])
    S[f"cv_{key}_std"] = r(row["std"])
    S[f"cv_{key}_floor"] = r(row["floor"])
    S[f"cv_{key}_lift"] = r(row["lift"], 2)
S["cv_std_ratio_shuffled_vs_temporal"] = r(
    S["cv_temporal_std"] / S["cv_shuffled_std"], 1) if S.get("cv_shuffled_std") else None
S["cv_lift_gap_temporal_vs_test"] = r(
    abs(S["cv_temporal_lift"] - S["cv_forward_test_lift"]), 2)

# ===========================================================================
# 8. Interpretability and robustness
# ===========================================================================
for tag, name in [("p1a", "perm_p1a.csv"), ("p2", "perm_p2.csv")]:
    perm = need(name)
    for i, row in perm.head(5).iterrows():
        S[f"perm_{tag}_rank{i+1}_feature"] = row["feature"]
        S[f"perm_{tag}_rank{i+1}_importance"] = r(row["importance"], 4)
    S[f"perm_{tag}_top_share_pct"] = r(
        100 * perm["importance"].head(2).sum()
        / perm.loc[perm["importance"] > 0, "importance"].sum(), 1)

seg = need("segments_p2_region.csv")
S["seg_p2_regions"] = len(seg)
S["seg_p2_roc_min"] = r(seg["roc_auc"].min())
S["seg_p2_roc_max"] = r(seg["roc_auc"].max())
S["seg_p2_roc_spread"] = r(seg["roc_auc"].max() - seg["roc_auc"].min())
S["seg_p2_worst_region"] = str(seg.loc[seg["roc_auc"].idxmin(), "customer_region"])

segm = need("segments_p1a_month.csv")
S["seg_p1a_months"] = len(segm)
S["seg_p1a_mae_min"] = r(segm["mae"].min(), 2)
S["seg_p1a_mae_max"] = r(segm["mae"].max(), 2)
S["seg_p1a_bias_min"] = r(segm["bias"].min(), 2)
S["seg_p1a_bias_max"] = r(segm["bias"].max(), 2)

# ===========================================================================
# 9. The pre-registered bands
# ===========================================================================
bands = need("bands.csv")
S["bands_checked"] = len(bands)
S["bands_in_band"] = int((bands["verdict"] == "in band").sum())
S["bands_suspicious"] = int(bands["verdict"].str.startswith("SUSPICIOUSLY").sum())
S["bands_short"] = int((bands["verdict"] == "short of target").sum())

# ===========================================================================
# 10. Capacity-constrained evaluation (Milestone 1 feedback: Precision@K)
# ===========================================================================
cap2 = need("capacity_p2.csv")
for _, row in cap2.iterrows():
    tag = f"{row['k_pct']:g}".replace(".", "p")
    S[f"cap_p2_k{tag}_precision"] = r(row["precision_at_k"])
    S[f"cap_p2_k{tag}_recall"] = r(row["recall_at_k"])
    S[f"cap_p2_k{tag}_lift"] = r(row["lift_at_k"], 2)
    S[f"cap_p2_k{tag}_net_per_1k"] = r(row["net_benefit_per_1k"], 0)
S["cap_p2_breakeven_precision"] = r(cap2["break_even_precision"].iloc[0])
S["cap_p2_max_k_pct_above_breakeven"] = r(cap2.loc[cap2["above_break_even"], "k_pct"].max(), 1)
S["cap_p2_best_net_k_pct"] = r(cap2.loc[cap2["net_benefit_per_1k"].idxmax(), "k_pct"], 1)

daily = need("capacity_p2_daily.csv")
S["cap_p2_test_days"] = int(daily["periods"].iloc[0])
S["cap_p2_orders_per_day"] = r(len(need("predictions_p2_test.csv")) / daily["periods"].iloc[0], 0)
_base_p2 = float(need("predictions_p2_test.csv")["actual_low_review"].mean())
for _, row in daily.iterrows():
    k = int(row["k_per_period"])
    S[f"cap_p2_daily_k{k}_precision"] = r(row["precision"])
    S[f"cap_p2_daily_k{k}_recall"] = r(row["recall"])
    S[f"cap_p2_daily_k{k}_flagged_pct"] = r(row["flagged_pct"], 1)
    S[f"cap_p2_daily_k{k}_lift"] = r(row["precision"] / _base_p2, 2)

cap1 = need("capacity_p1b.csv")
S["cap_p1b_breakeven_precision"] = r(cap1["break_even_precision"].iloc[0])
for route in ("derived", "direct"):
    part = cap1[cap1["route"] == route]
    for _, row in part.iterrows():
        tag = f"{row['k_pct']:g}".replace(".", "p")
        S[f"cap_p1b_{route}_k{tag}_precision"] = r(row["precision_at_k"])
        S[f"cap_p1b_{route}_k{tag}_recall"] = r(row["recall_at_k"])
        S[f"cap_p1b_{route}_k{tag}_lift"] = r(row["lift_at_k"], 2)
    S[f"cap_p1b_{route}_max_precision"] = r(part["precision_at_k"].max())
S["cap_p1b_derived_over_direct_k10_ratio"] = r(
    S["cap_p1b_derived_k10_precision"] / S["cap_p1b_direct_k10_precision"], 1)
_ratios = (cap1[cap1["route"] == "derived"].set_index("k_pct")["precision_at_k"]
           / cap1[cap1["route"] == "direct"].set_index("k_pct")["precision_at_k"])
S["cap_p1b_derived_over_direct_min_ratio"] = r(_ratios.loc[:30].min(), 1)
S["cap_p1b_derived_over_direct_max_ratio"] = r(_ratios.loc[:30].max(), 1)
S["cap_p1b_derived_breakeven_cost_reais"] = r(
    S["cap_p1b_derived_max_precision"] * config.COSTS["late_delivery_cost"], 0)

ops = need("capacity_operating_points.csv").set_index("problem")
for label, tag in [("P2 tuned threshold (F1 on validation)", "p2_f1_threshold"),
                   ("P2 cost-optimal threshold", "p2_cost_threshold"),
                   ("P1B derived, tuned threshold", "p1b_threshold")]:
    S[f"cap_{tag}_flagged_pct"] = r(ops.loc[label, "flagged_pct"], 1)
    S[f"cap_{tag}_precision"] = r(ops.loc[label, "precision"])
    S[f"cap_{tag}_recall"] = r(ops.loc[label, "recall"])

# ===========================================================================
# 11. Lateness anatomy (Milestone 1 feedback: the Figure 3 attribution)
# ===========================================================================
lm = need("lateness_monthly.csv").set_index("purchase_month")
for ym, tag in [("2017-10", "oct17"), ("2017-11", "nov17"), ("2017-12", "dec17"),
                ("2018-01", "jan18"), ("2018-02", "feb18"), ("2018-03", "mar18"),
                ("2018-04", "apr18"), ("2018-05", "may18"), ("2018-06", "jun18"),
                ("2018-07", "jul18"), ("2018-08", "aug18")]:
    S[f"lat_{tag}_late_pct"] = r(lm.loc[ym, "late_pct"], 1)
    S[f"lat_{tag}_promise_days"] = r(lm.loc[ym, "promise_days"], 1)
    S[f"lat_{tag}_carrier_days"] = r(lm.loc[ym, "carrier_days"], 1)
    S[f"lat_{tag}_seller_days"] = r(lm.loc[ym, "seller_days"], 1)
S["lat_monthly_peak_month"] = str(lm["late_pct"].idxmax())
S["lat_monthly_peak_pct"] = r(lm["late_pct"].max(), 1)
S["lat_monthly_trough_month"] = str(lm["late_pct"].idxmin())
S["lat_monthly_trough_pct"] = r(lm["late_pct"].min(), 2)

pk = need("lateness_peak_anatomy.csv").set_index("period")
for period, tag in [("Feb-Mar 2018", "peak"), ("rest of window", "rest")]:
    S[f"lat_{tag}_late_pct"] = r(pk.loc[period, "late_pct"], 1)
    S[f"lat_{tag}_carrier_days"] = r(pk.loc[period, "carrier_days"], 1)
    S[f"lat_{tag}_seller_days"] = r(pk.loc[period, "seller_days"], 1)
    S[f"lat_{tag}_promise_days"] = r(pk.loc[period, "promise_days"], 1)
    S[f"lat_{tag}_n"] = int(pk.loc[period, "n"])
reg = need("lateness_peak_by_region.csv")
S["lat_peak_regions_elevated"] = int((reg["peak_late_pct"] > reg["rest_late_pct"]).sum())
S["lat_peak_region_min_pct"] = r(reg["peak_late_pct"].min(), 1)
S["lat_peak_region_max_pct"] = r(reg["peak_late_pct"].max(), 1)

wk = need("lateness_weekly.csv").set_index("week_start")
S["lat_week_early_feb_promise_days"] = r(wk.loc["2018-02-05", "promise_days"], 1)
S["lat_week_early_feb_late_pct"] = r(wk.loc["2018-02-05", "late_pct"], 1)
peak_weeks = wk.loc["2018-02-19":"2018-03-19"]
S["lat_week_peak_max_late_pct"] = r(peak_weeks["late_pct"].max(), 1)
S["lat_week_peak_max_week"] = str(peak_weeks["late_pct"].idxmax())
S["lat_week_peak_promise_days"] = r(peak_weeks["promise_days"].mean(), 1)
S["lat_week_peak_carrier_days"] = r(peak_weeks["carrier_days"].max(), 1)
S["lat_week_recovery_week"] = str(wk.loc["2018-03-26":, "late_pct"].lt(5).idxmax())
S["lat_week_pre_strike_late_pct"] = r(wk.loc["2018-05-14", "late_pct"], 1)
S["lat_week_pre_strike_promise_days"] = r(wk.loc["2018-05-14", "promise_days"], 1)
S["lat_week_pre_strike_n"] = int(wk.loc["2018-05-14", "n"])
S["lat_week_strike_late_pct"] = r(wk.loc["2018-05-21", "late_pct"], 1)
S["lat_week_strike_promise_days"] = r(wk.loc["2018-05-21", "promise_days"], 1)
S["lat_week_strike_n"] = int(wk.loc["2018-05-21", "n"])
S["lat_week_post_strike_late_pct"] = r(wk.loc["2018-05-28", "late_pct"], 1)
S["lat_week_post_strike_promise_days"] = r(wk.loc["2018-05-28", "promise_days"], 1)
S["lat_week_apr_may_mean_late_pct"] = r(wk.loc["2018-04-09":"2018-05-07", "late_pct"].mean(), 1)

# The promise was tightened within a single homogeneous route as well, so it is
# a promising decision rather than a change in the mix of routes.
_sp = df[(df["customer_state"] == "SP") & (df["lead_seller_state"] == "SP")
         & df["target_delivery_days"].notna()]
_spw = _sp.groupby(_sp["order_purchase_timestamp"].dt.to_period("W-SUN").dt.start_time).agg(
    promise=("estimated_days", "mean"), late=("target_is_late", "mean"))
_spw.index = _spw.index.strftime("%Y-%m-%d")
S["lat_spsp_promise_late_jan_days"] = r(_spw.loc["2018-01-29", "promise"], 1)
S["lat_spsp_promise_late_mar_days"] = r(_spw.loc["2018-03-19", "promise"], 1)
S["lat_spsp_promise_mid_apr_days"] = r(_spw.loc["2018-04-16", "promise"], 1)
S["lat_spsp_late_late_jan_pct"] = r(100 * _spw.loc["2018-01-29", "late"], 1)
S["lat_spsp_late_peak_pct"] = r(100 * _spw.loc["2018-02-19":"2018-03-19", "late"].max(), 1)

st = need("lateness_strike.csv").set_index("cohort")
_c = {"control": "May 1-20, delivered before the strike",
      "transit": "in transit during the strike (bought before 1 Jun, arrived on/after 21 May)",
      "during": "bought during the strike (21 May - 1 Jun)",
      "after": "bought 2-30 June, after the strike"}
for tag, name in _c.items():
    S[f"lat_strike_{tag}_n"] = int(st.loc[name, "n"])
    S[f"lat_strike_{tag}_late_pct"] = r(st.loc[name, "late_pct"], 1)
    S[f"lat_strike_{tag}_carrier_days"] = r(st.loc[name, "carrier_days"], 1)
    S[f"lat_strike_{tag}_promise_days"] = r(st.loc[name, "promise_days"], 1)

# ===========================================================================
# 12. Deployment footprint
# ===========================================================================
artifacts = sorted(config.MODELS_DIR.glob("*"))
S["n_model_artifacts"] = len(artifacts)
S["models_total_mb"] = r(sum(f.stat().st_size for f in artifacts) / 1e6, 2)
for f in artifacts:
    S[f"artifact_{f.stem}_mb"] = r(f.stat().st_size / 1e6, 3)


# ===========================================================================
if __name__ == "__main__":
    out = config.PROJECT_ROOT / "report" / "stats_phase2.txt"
    lines = [f"{k:44s} {v}" for k, v in S.items()]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}  ({len(S)} values)")
