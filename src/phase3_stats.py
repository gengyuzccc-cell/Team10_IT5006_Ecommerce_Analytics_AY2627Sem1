"""Regenerate every number quoted in the Phase 3 (final) report.

    python src/phase2_stats.py       # the Phase 2 contract, unchanged
    python src/customer_view.py      # first: report/results/customer_*.csv
    python src/slide_provenance.py   # then: report/results/slide_provenance.csv
    python src/phase3_stats.py       # that, plus everything Phase 3 added

The final report is the standalone successor of the Phase 2 report, so it quotes
every number the Phase 2 report quoted - those keys are reproduced here verbatim
by importing src/phase2_stats.py - and adds the numbers behind the Phase 2
feedback: the fold-to-fold variation of every candidate, the selected
hyperparameters, the deployed estimators as actually pickled, and the outreach
pilot design. The revision after the 5 November 2026 presentation adds two
sections. Section 20 (keys cv_*) is the customer's view of the same orders: what
happened to the customers of the test window, with every count's population
named; what 1-2 star reviewers wrote; the promised date the forecast plus a
margin would give; which contact can still reach a customer before the rating;
what a contact is worth under the assumed costs; and the pilot sized per
contact. Section 21 (keys prov_*) is where every number on the slides comes
from. Phase 1's exploratory numbers remain the contract of src/report_stats.py
and report/stats.txt.

Anything these scripts cannot produce is not allowed into the report.
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
# 19. When is the review written? (src/review_timing.py, Appendix "timing")
# ===========================================================================
tim = need("review_timing.csv").set_index("group")
for g, key in [("all", "all"), ("late", "late"), ("on time", "ontime"),
               ("low review, late", "low_late"), ("low review, on time", "low_ontime")]:
    S[f"tim_{key}_n"] = int(tim.loc[g, "n"])
    S[f"tim_{key}_written_before_pct"] = r(tim.loc[g, "pct_written_before_arrival"], 1)
    S[f"tim_{key}_survey_on_after_promise_pct"] = r(tim.loc[g, "pct_survey_on_or_after_promise"], 1)
    S[f"tim_{key}_median_answer_days_after_delivery"] = r(tim.loc[g, "median_answer_days_after_delivery"], 1)
    S[f"tim_{key}_low_pct"] = r(tim.loc[g, "pct_low_review"], 1)
for g, key in [("low review, late", "low_late"), ("low review, on time", "low_ontime")]:
    S[f"tim_{key}_share_of_low_pct"] = r(tim.loc[g, "share_of_all_low_reviews_pct"], 1)
for _, row in need("review_timing_lateness.csv").iterrows():
    S[f"tim_bucket_{slug(row['bucket'])}_low_pct"] = r(row["low_pct"], 1)
tcap = need("review_timing_capacity.csv")
for _, row in tcap.iterrows():
    k = str(row["k"]).replace("%", "pct").replace(" a day", "_daily").replace(".", "p")
    S[f"tim_cap_{k}_precision"] = r(row["precision"])
    S[f"tim_cap_{k}_reachable"] = r(row["reachable_precision"])
    S[f"tim_cap_{k}_written"] = r(row["already_written_precision"])
    S[f"tim_cap_{k}_late_share"] = r(row["late_share"])
S["tim_random_precision"] = r(tcap["base_rate"].iloc[0])
S["tim_random_reachable"] = r(tcap["reachable_base_rate"].iloc[0])
tval = need("review_timing_value.csv")
best_all = tval.loc[tval["net_all_per_1k"].idxmax()]
best_reach = tval.loc[tval["net_reachable_per_1k"].idxmax()]
S["tim_value_all_per_1k"] = int(round(best_all["net_all_per_1k"]))
S["tim_value_all_threshold"] = r(best_all["threshold"], 2)
S["tim_value_reachable_per_1k"] = int(round(best_reach["net_reachable_per_1k"]))
S["tim_value_reachable_threshold"] = r(best_reach["threshold"], 2)
ten = tcap[(tcap["kind"] == "daily") & (tcap["k"] == "10 a day")].iloc[0]
per_1k = 1000 * ten["contacts"] / len(need("review_timing_test_orders.csv"))
c = config.COSTS
S["tim_value_ten_all_per_1k"] = int(round(per_1k * (ten["precision"] * c["low_review_cost"] - c["outreach_cost"])))
S["tim_value_ten_reachable_per_1k"] = int(round(per_1k * (ten["reachable_precision"] * c["low_review_cost"]
                                                          - c["outreach_cost"])))
tsub = need("review_timing_subsets.csv").set_index("subset")
for g, key in [("delivered", "delivered"), ("on time", "ontime"), ("late", "late"),
               ("never delivered", "never_delivered")]:
    S[f"tim_sub_{key}_n"] = int(tsub.loc[g, "n"])
    S[f"tim_sub_{key}_base_rate"] = r(tsub.loc[g, "base_rate"])
    S[f"tim_sub_{key}_roc"] = r(tsub.loc[g, "roc_auc"])
    S[f"tim_sub_{key}_prauc"] = r(tsub.loc[g, "pr_auc"])
    S[f"tim_sub_{key}_top5_precision"] = r(tsub.loc[g, "precision_top5pct"])
    S[f"tim_sub_{key}_top5_lift"] = r(tsub.loc[g, "lift_top5pct"], 1)
tmix = need("review_timing_composition.csv").set_index("list")
for g, key in [("all reviewed orders", "all"), ("delivered orders only", "delivered")]:
    S[f"tim_mix_{key}_orders"] = int(tmix.loc[g, "orders"])
    S[f"tim_mix_{key}_base_rate"] = r(tmix.loc[g, "base_rate"])
    S[f"tim_mix_{key}_contacts"] = int(tmix.loc[g, "contacts"])
    S[f"tim_mix_{key}_hits_per_100"] = r(100 * tmix.loc[g, "precision"], 1)
    S[f"tim_mix_{key}_never_delivered_per_100"] = r(tmix.loc[g, "never_delivered_per_100"], 1)
    S[f"tim_mix_{key}_late_written_before_per_100"] = r(tmix.loc[g, "late_written_before_per_100"], 1)
    S[f"tim_mix_{key}_delivered_reachable_per_100"] = r(tmix.loc[g, "delivered_reachable_per_100"], 1)
    S[f"tim_mix_{key}_contacts_never_delivered_pct"] = r(tmix.loc[g, "contacts_never_delivered_pct"], 1)
    S[f"tim_mix_{key}_story_day_hits"] = int(tmix.loc[g, "story_day_hits"])
story = need("story_order.csv").iloc[0]
S["story_order_id"] = story["order_id"]
S["story_distance_km"] = int(round(story["distance_km"]))
S["story_days_late"] = r(story["days_late"], 1)
S["story_review_days_before_arrival"] = r(story["review_days_before_arrival"], 1)
S["story_p_late_at_checkout"] = r(story["p_late_at_checkout"], 2)
S["story_p_low_at_delivery"] = r(story["p_low_review_at_delivery"], 2)
S["story_orders_on_day"] = int(story["orders_on_day"])
S["story_top10_low_reviews"] = int(story["top10_low_reviews"])
S["story_top10_written_before_arrival"] = int(story["top10_written_before_arrival"])

# ===========================================================================
# 18. Figures and tables the final report draws from
# ===========================================================================
S["n_result_tables"] = len(list(RESULTS.glob("*.csv")))
S["n_figures"] = len(list(config.FIGURES_DIR.glob("fig*.pdf")))


# ===========================================================================
# 20. The customer's view (src/customer_view.py; Sections 1, 4 and 9, Appendix Q)
# ===========================================================================
def need_csv(name: str, **kw) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `python src/customer_view.py` and "
                         "`python src/slide_provenance.py` first.")
    return pd.read_csv(path, **kw)


cv1 = lambda x, n=1: r(x, n)       # the customer tables are read to one decimal unless stated

o = need_csv("customer_outcomes_test.csv").set_index("key")
for k in o.index:
    S[f"cv_out_{k}_n"] = int(o.loc[k, "orders"])
    S[f"cv_out_{k}_per_1k"] = cv1(o.loc[k, "per_1000"])
    S[f"cv_out_{k}_lo_per_1k"] = cv1(o.loc[k, "ci95_lo"])
    S[f"cv_out_{k}_hi_per_1k"] = cv1(o.loc[k, "ci95_hi"])
for den in ("placed", "delivered"):
    S[f"cv_out_denominator_{den}"] = int(o.loc[o["denominator"] == den, "denominator_n"].iloc[0])
t = need_csv("customer_complaint_topics.csv").set_index("group")
tag = {"all": "all", "on time": "ontime", "late": "late", "never delivered": "never"}
for g, k in tag.items():
    S[f"cv_topic_{k}_low_reviews"] = int(t.loc[g, "low_reviews"])
    S[f"cv_topic_{k}_with_text"] = int(t.loc[g, "with_text"])
    S[f"cv_topic_{k}_with_text_pct"] = cv1(t.loc[g, "with_text_pct"])
    for c_ in [c_ for c_ in t.columns if c_.endswith("_pct") and c_ != "with_text_pct"]:
        S[f"cv_topic_{k}_{c_.replace('_pct', '')}_pct"] = cv1(t.loc[g, c_])
b = need_csv("customer_basket.csv").set_index("basket")
for g, k in {"1 item, 1 seller": "single", "2+ items, 1 seller": "multiitem", "2+ sellers": "multiseller",
             "all on-time": "all"}.items():
    S[f"cv_basket_{k}_orders"] = int(b.loc[g, "orders"])
    S[f"cv_basket_{k}_share_orders_pct"] = cv1(b.loc[g, "share_of_on_time_orders_pct"])
    S[f"cv_basket_{k}_low_pct"] = cv1(b.loc[g, "low_rate_pct"])
    S[f"cv_basket_{k}_low_lo_pct"] = cv1(b.loc[g, "ci95_lo_pct"])
    S[f"cv_basket_{k}_low_hi_pct"] = cv1(b.loc[g, "ci95_hi_pct"])
    S[f"cv_basket_{k}_share_low_pct"] = cv1(b.loc[g, "share_of_on_time_low_reviews_pct"])
    S[f"cv_basket_{k}_says_missing_pct"] = cv1(b.loc[g, "says_wrong_missing_or_not_received_pct"])
tm = need_csv("customer_survey_timing.csv").set_index("window")
for g, k in {"analysis window": "window", "test window": "test"}.items():
    for c_ in tm.columns:
        S[f"cv_timing_{k}_{c_}"] = cv1(tm.loc[g, c_], 1) if c_ != "late_low_reviews" else int(tm.loc[g, c_])
p = need_csv("customer_complaint_precision.csv").set_index("topic")
for k in p.index:
    S[f"cv_prec_{k}_pct"] = cv1(p.loc[k, "precision_pct"], 0)
    S[f"cv_prec_{k}_lo_pct"] = cv1(p.loc[k, "ci95_lo_pct"], 0)
    S[f"cv_prec_{k}_hi_pct"] = cv1(p.loc[k, "ci95_hi_pct"], 0)
S["cv_prec_min_pct"] = cv1(p["precision_pct"].min(), 0)
S["cv_prec_max_pct"] = cv1(p["precision_pct"].max(), 0)
S["cv_prec_read_total"] = int(p["read"].sum())
rp = need_csv("customer_repeat.csv")
for v_, vk in [("V1 later calendar day, from purchase", "v1"), ("V2 after delivery, from delivery", "v2")]:
    for h in (30, 90, 180, 365):
        part = rp[(rp.version == v_) & (rp.horizon_days == h)]
        S[f"cv_rep_{vk}_h{h}_cohort_n"] = int(part["cohort_n"].iloc[0])
        S[f"cv_rep_{vk}_h{h}_rate_pct"] = cv1(part["cohort_repeat_pct"].iloc[0], 2)
        for _, row in part.iterrows():
            c_ = (f"cv_rep_{vk}_h{h}_{row['group'].replace(' ', '').replace('-', '')}_vs_"
                  f"{row['reference'].replace(' ', '').replace('-', '')}")
            S[f"{c_}_n"] = int(row["n"])
            S[f"{c_}_rate_pct"] = cv1(row["repeat_pct"], 2)
            S[f"{c_}_ref_rate_pct"] = cv1(row["ref_repeat_pct"], 2)
            S[f"{c_}_diff_pp"] = cv1(row["diff_pp"], 2)
            S[f"{c_}_diff_lo_pp"] = cv1(row["diff_ci95_lo_pp"], 2)
            S[f"{c_}_diff_hi_pp"] = cv1(row["diff_ci95_hi_pp"], 2)
            S[f"{c_}_p"] = cv1(row["p_value"], 3)
            S[f"{c_}_mh_or"] = cv1(row["mh_odds_ratio"], 2)
            S[f"{c_}_mh_lo"] = cv1(row["mh_ci95_lo"], 2)
            S[f"{c_}_mh_hi"] = cv1(row["mh_ci95_hi"], 2)
            S[f"{c_}_mh_p"] = cv1(row["mh_p_value"], 3)
            S[f"{c_}_mde80_pp"] = cv1(row["mde80_pp"], 2)
pr = need_csv("customer_promise_rules.csv")
names = {"Olist's own promise": "olist", "Forecast alone (no buffer)": "raw",
         "R1 buffer: validation late rate <= Olist's": "r1",
         "R2 buffer: same mean promise as Olist (label-free)": "r2",
         "R2-val buffer: validation mean promise = Olist's": "r2val"}
for _, row in pr.iterrows():
    k = f"cv_prom_{names[row['rule']]}_{'val' if row['period'] == 'validation' else 'test'}"
    S[f"{k}_late_pct"] = cv1(row["late_pct"], 2)
    S[f"{k}_late_calday_pct"] = cv1(row["late_calendar_day_pct"], 2)
    S[f"{k}_mean_promise_days"] = cv1(row["mean_promised_days"], 2)
    S[f"{k}_told_minus_waited_days"] = cv1(row["mean_told_minus_waited_days"], 2)
    S[f"{k}_days_late_if_late"] = cv1(row["mean_days_late_if_late"], 2)
    S[f"{k}_n_late"] = int(row["n_late"])
    if pd.notna(row["buffer_days"]):
        S[f"cv_prom_{names[row['rule']]}_buffer_days"] = cv1(row["buffer_days"], 1)
S["cv_prom_test_mean_actual_days"] = cv1(pr.loc[pr.period == "test", "mean_actual_days"].iloc[0], 2)
rg = need_csv("customer_promise_region.csv")
rk = {"North + Northeast": "nne", "Southeast": "se", "South + Central-West": "scw", "All": "all"}
for _, row in rg.iterrows():
    if row["rule"] not in names:
        continue
    k = f"cv_promreg_{rk[row['region_group']]}_{names[row['rule']]}"
    S[f"{k}_n"] = int(row["n"])
    S[f"{k}_late_pct"] = cv1(row["late_pct"], 2)
    S[f"{k}_mean_promise_days"] = cv1(row["mean_promised_days"], 2)
    S[f"{k}_mean_actual_days"] = cv1(row["mean_actual_days"], 2)
    if pd.notna(row.get("pct_earlier_date_than_olist")):
        S[f"{k}_pct_earlier"] = cv1(row["pct_earlier_date_than_olist"], 1)
        S[f"{k}_pct_later"] = cv1(row["pct_later_date_than_olist"], 1)
        S[f"{k}_mean_change_days"] = cv1(row["mean_change_vs_olist_days"], 2)
lt = need_csv("customer_lateness_days.csv")
for _, row in lt.iterrows():
    S[f"cv_lateday_{row['slug']}_n"] = int(row["n"])
    S[f"cv_lateday_{row['slug']}_low_pct"] = cv1(row["low_pct"])
    S[f"cv_lateday_{row['slug']}_lo_pct"] = cv1(row["ci95_lo_pct"])
    S[f"cv_lateday_{row['slug']}_hi_pct"] = cv1(row["ci95_hi_pct"])
cl = need_csv("customer_contact_lists.csv").set_index("list")
for k in cl.index:
    row = cl.loc[k]
    vals = {"contacts": int(row["contacts"]), "hits": int(row["hits"]), "precision": r(row["precision"], 4),
            "lift": r(row["lift"], 2), "base_rate": r(row["pool_base_rate"], 4), "pool_n": int(row["pool_n"]),
            "recall": r(row["recall"], 3), "per_day": cv1(row["contacts_per_day"]), "days": int(row["days"])}
    if pd.notna(row["contacts_per_1k_reviewed"]):
        vals["per_1k"] = cv1(row["contacts_per_1k_reviewed"])
    if pd.notna(row["full_days"]):
        vals["full_days"] = int(row["full_days"])
        vals["precision_full_days"] = r(row["precision_full_days"], 4)
    for cls in ("promise_only", "delivery_only", "either", "neither"):
        if pd.notna(row[f"hits_{cls}"]):
            vals[f"hits_{cls}"] = int(row[f"hits_{cls}"])
    for name, val in vals.items():
        S[f"cv_list_{k}_{name}"] = val
_b1 = need("predictions_p1b_test.csv").dropna(subset=["actual_late"])
S["cv_list_late_top_decile_threshold"] = r(_b1["p_late_derived"].quantile(0.9), 3)
cvv = need_csv("customer_value.csv")
scope_tag = {"all hits": "all", "not written before arrival": "notwritten", "reachable at its moment": "atmoment"}
for _, row in cvv.iterrows():
    k = f"cv_val_{row['list']}_{scope_tag[row['scope']]}"
    S[f"{k}_hits"] = int(row["hits"])
    S[f"{k}_precision"] = r(row["precision"], 4)
    for e in ("e100", "e50", "e25"):
        S[f"{k}_net_{e}"] = int(round(row[f"net_{e}"]))
        S[f"{k}_breakeven_cost_{e}"] = cv1(row[f"breakeven_review_cost_{e}"], 2)
    S[f"{k}_breakeven_pct"] = cv1(100 * row["breakeven_share"])
    S[f"{k}_breakeven_one_in"] = cv1(1 / row["breakeven_share"])
dec = need_csv("customer_value_decomposition.csv").set_index("list")
for k in dec.index:
    for c_ in dec.columns:
        val = dec.loc[k, c_]
        S[f"cv_dec_{k}_{c_}"] = int(val) if c_.startswith(("hits_", "contacts", "written_")) else cv1(val)
S["cv_sens_rows"] = len(need_csv("customer_value_sensitivity.csv"))
arms = need_csv("customer_pilot_arms.csv").set_index("arm")
for k in arms.index:
    row = arms.loc[k]
    S[f"cv_pilot_{k}_control"] = r(row["control_rate"], 4)
    S[f"cv_pilot_{k}_per_day"] = cv1(row["per_day"])
    S[f"cv_pilot_{k}_arms"] = int(row["arms"])
    S[f"cv_pilot_{k}_breakeven_rel_pct"] = cv1(100 * row["breakeven_relative"])
    S[f"cv_pilot_{k}_target_rel_pct"] = cv1(100 * row["target_relative"])
    S[f"cv_pilot_{k}_treated"] = r(row["treated_rate"], 4)
    S[f"cv_pilot_{k}_n_per_arm_80"] = int(row["n_per_arm_80"])
    S[f"cv_pilot_{k}_weeks_80"] = int(row["weeks_80"])
    S[f"cv_pilot_{k}_n_per_arm_90"] = int(row["n_per_arm_90"])
    S[f"cv_pilot_{k}_weeks_90"] = int(row["weeks_90"])
st = need_csv("customer_story.csv", keep_default_na=False).iloc[0]
story_digits = {"promised_days": 1, "price": 2, "freight": 2, "pred_days_at_checkout": 1, "p_late_at_checkout": 3,
                "handover_days_after_limit": 1, "actual_days": 1, "days_late": 1, "review_days_before_arrival": 1,
                "p_low_at_delivery": 3, "message_days_before_review": 1, "call_days_after_review": 1}
for c_, val in st.items():
    if c_ == "distance_km":
        S["cv_story_distance_km"] = int(round(float(val)))
    elif c_ in story_digits:
        S[f"cv_story_{c_}"] = cv1(float(val), story_digits[c_])
    elif str(val).lstrip("-").isdigit():
        S[f"cv_story_{c_}"] = int(val)
    else:
        S[f"cv_story_{c_}"] = str(val)
ev = need_csv("customer_story_events.csv")
S["cv_story_events"] = len(ev)
jr = need_csv("customer_journeys_test.csv", usecols=["order_id", "review_predates_order", "reach_class"])
S["cv_journeys_rows"] = len(jr)
S["cv_journeys_review_predates_order"] = int(jr["review_predates_order"].sum())

# ===========================================================================
# 21. Where every number on the slides comes from (src/slide_provenance.py, Appendix P)
# ===========================================================================
prov = need_csv("slide_provenance.csv", keep_default_na=False)
checked = prov[prov["matches"] != ""]
S["prov_rows"] = len(prov)
S["prov_slides"] = int(prov["slide"].nunique())
S["prov_checked_rows"] = len(checked)
S["prov_all_match"] = bool((checked["matches"].astype(str) == "True").all())
S["prov_assumed_rows"] = int((prov["kind"] == "assumed").sum())
for kind in ("observed", "assumed", "derived", "planned", "literature", "no data source", "wording"):
    S[f"prov_kind_{kind.replace(' ', '_')}"] = int((prov["kind"] == kind).sum())
derived_digits = {"buffer_all_days": 1, "promise_all_days": 1, "lead_all_days": 1, "carrier_days": 1,
                  "carrier_share_pct": 1, "buffer_test_days": 1, "promise_test_days": 1, "lead_test_days": 1,
                  "story_late_rank": 0, "story_delivered_on_day": 0, "expedite_expected_damage": 1,
                  "expedite_breakeven_adjusted": 2, "value_t019_all": 0, "value_t019_reachable": 0,
                  "day_mae_model": 2, "day_mae_olist": 2}
for cell in prov["keys"]:
    for item in filter(None, (x.strip() for x in str(cell).split(";"))):
        name, val = item.split("=")
        nd = derived_digits[name]
        S[f"prov_{name}"] = int(round(float(val))) if nd == 0 else r(float(val), nd)


# ===========================================================================
if __name__ == "__main__":
    out = config.PROJECT_ROOT / "report" / "stats_phase3.txt"
    lines = [f"{k:52s} {v}" for k, v in S.items()]
    out.write_text("\n".join(lines), encoding="utf-8")
    new = [k for k in S if k not in P2.S]
    print("\n".join(f"{k:52s} {S[k]}" for k in new))
    print(f"\nwrote {out}  ({len(S)} values, {len(new)} new in Phase 3)")
