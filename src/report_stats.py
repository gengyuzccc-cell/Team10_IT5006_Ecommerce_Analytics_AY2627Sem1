"""
Regenerate every number quoted in the Phase 1 report.

The report is written by hand, but no figure in it is typed from memory: this
script recomputes each one and writes report/stats.txt. Before submitting, run

    python src/report_stats.py

and diff the output against the manuscript. Any drift is a bug in one or the
other.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from data_load import build_orders, load_all, load_analysis, zip_centroids
from features import (
    CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, BOOLEAN_FEATURES,
    REVIEW_MODEL_FEATURES, assert_no_leakage, build_features, make_splits,
)

S: dict[str, object] = {}
NUM = NUMERIC_FEATURES + BOOLEAN_FEATURES


def pipe(model, numeric):
    """Preprocessing + model, so no fold ever sees test statistics."""
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100,
                                  sparse_output=False), CATEGORICAL_FEATURES),
        ])),
        ("model", model),
    ])


# ===========================================================================
# Dataset shape
# ===========================================================================
tables = load_all()
for name, df_ in tables.items():
    S[f"rows_{name}"] = len(df_)

full = build_orders(tables)
df = build_features(load_analysis(tables))

S["n_tables"] = len(tables)
S["orders_total"] = len(full)
S["analysis_rows"] = len(df)
S["date_min"] = str(full["order_purchase_timestamp"].min().date())
S["date_max"] = str(full["order_purchase_timestamp"].max().date())
S["window_start"] = str(config.ANALYSIS_START.date())
S["window_end"] = str(config.ANALYSIS_END.date())
S["n_customers_unique"] = int(full["customer_unique_id"].nunique())
S["n_customer_ids"] = int(full["customer_id"].nunique())
rep = full.groupby("customer_unique_id").size()
S["n_repeat_customers"] = int((rep > 1).sum())
S["pct_repeat_customers"] = round(100 * (rep > 1).mean(), 1)
S["max_orders_one_customer"] = int(rep.max())
S["n_sellers"] = int(tables["sellers"]["seller_id"].nunique())
S["n_products"] = int(tables["products"]["product_id"].nunique())
S["n_categories"] = int(full["lead_category"].nunique())
S["n_states"] = int(full["customer_state"].nunique())

# order status
st = full["order_status"].value_counts(normalize=True) * 100
S["pct_delivered"] = round(float(st["delivered"]), 1)
S["pct_undelivered"] = round(100 - float(st["delivered"]), 1)
S["pct_no_delivery_date"] = round(
    100 * full["order_delivered_customer_date"].isna().mean(), 2)

# review coverage and key integrity
rev = tables["order_reviews"]
S["rows_reviews_distinct_orders"] = int(rev["order_id"].nunique())
S["n_distinct_review_ids"] = int(rev["review_id"].nunique())
_rid = rev["review_id"].value_counts()
S["n_reused_review_ids"] = int((_rid > 1).sum())
_per_order = rev["order_id"].value_counts()
S["n_orders_multi_review"] = int((_per_order > 1).sum())
S["max_reviews_one_order"] = int(_per_order.max())
S["n_orders_no_review"] = int(len(set(tables["orders"]["order_id"]) - set(rev["order_id"])))
S["pct_review_title_present"] = round(100 * rev["review_comment_title"].notna().mean(), 1)
S["pct_review_message_present"] = round(100 * rev["review_comment_message"].notna().mean(), 1)
S["n_distinct_estimate_dates"] = int(tables["orders"]["order_estimated_delivery_date"].nunique())
S["n_orders_no_payment"] = int(
    len(set(tables["orders"]["order_id"]) - set(tables["order_payments"]["order_id"])))
S["n_products_no_category"] = int(tables["products"]["product_category_name"].isna().sum())
S["pct_orders_with_review"] = round(100 * df["review_score"].notna().mean(), 1)
rs = tables["order_reviews"]["review_score"].value_counts(normalize=True) * 100
S["pct_review_5"] = round(float(rs[5]), 1)
S["pct_review_4_5"] = round(float(rs[4] + rs[5]), 1)
S["pct_review_1"] = round(float(rs[1]), 1)

# geography
cent = zip_centroids(tables["geolocation"])
S["n_zip_prefixes"] = len(cent)
S["median_distance_km"] = round(float(df["distance_km"].median()))
S["pct_same_state"] = round(100 * df["same_state"].mean(), 1)
S["pct_sellers_sp"] = round(
    100 * (tables["sellers"]["seller_state"] == "SP").mean(), 1)
S["pct_customers_sp"] = round(100 * (full["customer_state"] == "SP").mean(), 1)

# money
S["median_order_value"] = round(float(df["total_price"].median()), 2)
S["median_freight"] = round(float(df["total_freight"].median()), 2)
S["median_freight_ratio"] = round(float(df["freight_ratio"].median()), 3)
S["pct_credit_card"] = round(
    100 * (df["lead_payment_type"] == "credit_card").mean(), 1)
S["pct_installments_gt1"] = round(100 * (df["max_installments"] > 1).mean(), 1)

# ===========================================================================
# Delivery behaviour and the drift
# ===========================================================================
d = df.dropna(subset=["target_delivery_days"])
S["delivery_median_days"] = round(float(d["target_delivery_days"].median()), 1)
S["delivery_mean_days"] = round(float(d["target_delivery_days"].mean()), 1)
S["delivery_p90_days"] = round(float(d["target_delivery_days"].quantile(0.9)), 1)
S["late_base_rate_pct"] = round(100 * float(df["target_is_late"].mean()), 2)
S["always_ontime_acc_pct"] = round(100 - S["late_base_rate_pct"], 2)
S["low_review_base_pct"] = round(100 * float(df["target_low_review"].mean()), 2)
S["repeat_base_pct"] = round(100 * float(df["target_repeat"].mean()), 2)
S["repeat_horizon_days"] = config.REPEAT_HORIZON_DAYS

monthly = d.groupby(d["order_purchase_timestamp"].dt.to_period("M")).agg(
    mean_days=("target_delivery_days", "mean"), late=("target_is_late", "mean"))
S["lead_peak_month"] = str(monthly["mean_days"].idxmax())
S["lead_peak_days"] = round(float(monthly["mean_days"].max()), 1)
S["lead_trough_month"] = str(monthly["mean_days"].idxmin())
S["lead_trough_days"] = round(float(monthly["mean_days"].min()), 1)
S["late_peak_month"] = str(monthly["late"].idxmax())
S["late_peak_pct"] = round(100 * float(monthly["late"].max()), 1)
S["late_trough_month"] = str(monthly["late"].idxmin())
S["late_trough_pct"] = round(100 * float(monthly["late"].min()), 2)

# correlations with checkout-time features
for c in ["estimated_days", "log_distance", "total_freight", "log_weight"]:
    S[f"corr_leadtime_{c}"] = round(float(d[c].corr(d["target_delivery_days"])), 3)

# review vs delivery
r = df.dropna(subset=["target_low_review", "target_is_late"])
S["low_review_when_late_pct"] = round(
    100 * float(r.loc[r["target_is_late"] == 1, "target_low_review"].mean()), 1)
S["low_review_when_ontime_pct"] = round(
    100 * float(r.loc[r["target_is_late"] == 0, "target_low_review"].mean()), 1)
S["low_review_late_multiplier"] = round(
    S["low_review_when_late_pct"] / S["low_review_when_ontime_pct"], 1)

# ===========================================================================
# Splits
# ===========================================================================
sp = make_splits(df)
S["train_rows"] = len(sp["train"])
S["val_rows"] = len(sp["val"])
S["test_rows"] = len(sp["test"])
S["straddling_customers"] = sp["n_straddling_customers"]
S["train_end"] = str(config.TRAIN_END.date())
S["val_end"] = str(config.VAL_END.date())
S["n_features_checkout"] = len(MODEL_FEATURES)
S["n_features_delivery"] = len(REVIEW_MODEL_FEATURES)

train_mean = sp["train"]["target_delivery_days"].mean()
te_d = sp["test"].dropna(subset=["target_delivery_days"])
S["train_mean_delivery"] = round(float(train_mean), 2)
S["test_mean_delivery"] = round(float(te_d["target_delivery_days"].mean()), 2)
S["delivery_drift_days"] = round(S["test_mean_delivery"] - S["train_mean_delivery"], 2)

# ===========================================================================
# Candidate probes
# ===========================================================================
def clf_probe(target, feats, numeric, regime, prefix):
    assert_no_leakage(feats, regime)
    tr = sp["train"].dropna(subset=[target])
    te = sp["test"].dropna(subset=[target])
    y = te[target].astype(int)
    S[f"{prefix}_train_rows"] = len(tr)
    S[f"{prefix}_test_rows"] = len(te)
    S[f"{prefix}_test_positive_pct"] = round(100 * float(y.mean()), 2)

    dummy = DummyClassifier(strategy="most_frequent").fit(tr[feats], tr[target])
    S[f"{prefix}_majority_acc"] = round(
        float((dummy.predict(te[feats]) == y).mean()), 3)

    for tag, model in [
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=config.SEED)),
        ("hgb", HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                               random_state=config.SEED)),
    ]:
        m = pipe(model, numeric).fit(tr[feats], tr[target])
        p = m.predict_proba(te[feats])[:, 1]
        pred = (p >= 0.5).astype(int)
        S[f"{prefix}_{tag}_roc"] = round(float(roc_auc_score(y, p)), 3)
        S[f"{prefix}_{tag}_prauc"] = round(float(average_precision_score(y, p)), 3)
        S[f"{prefix}_{tag}_balacc"] = round(float(balanced_accuracy_score(y, pred)), 3)
        S[f"{prefix}_{tag}_recall"] = round(float(recall_score(y, pred)), 3)
        S[f"{prefix}_{tag}_precision"] = round(float(precision_score(y, pred)), 3)
    S[f"{prefix}_prauc_floor"] = round(float(y.mean()), 3)
    S[f"{prefix}_prauc_lift"] = round(S[f"{prefix}_hgb_prauc"] / float(y.mean()), 1)


# C1B - late delivery, at checkout
clf_probe("target_is_late", MODEL_FEATURES, NUM, "at_checkout", "c1b")
# C2 - low review, at delivery
clf_probe("target_low_review", REVIEW_MODEL_FEATURES,
          NUM + config.DELIVERY_OUTCOME_FEATURES, "at_delivery", "c2")
# C2 without the delivery outcome, to size what it contributes
clf_probe("target_low_review", MODEL_FEATURES, NUM, "at_checkout", "c2_checkout")
# C3 - repeat purchase
clf_probe("target_repeat", MODEL_FEATURES, NUM, "at_checkout", "c3")

# C1A - lead time regression
assert_no_leakage(MODEL_FEATURES, "at_checkout")
tr = sp["train"].dropna(subset=["target_delivery_days"])
y = te_d["target_delivery_days"]
S["c1a_train_rows"] = len(tr)
S["c1a_test_rows"] = len(te_d)
preds = {
    "mean": np.full(len(y), tr["target_delivery_days"].mean()),
    "promise": te_d["estimated_days"].values,
}
for tag, model in [("ridge", Ridge(alpha=1.0, random_state=config.SEED)),
                   ("hgb", HistGradientBoostingRegressor(max_iter=200,
                                                         random_state=config.SEED))]:
    preds[tag] = pipe(model, NUM).fit(tr[MODEL_FEATURES],
                                      tr["target_delivery_days"]).predict(te_d[MODEL_FEATURES])
for tag, p in preds.items():
    S[f"c1a_{tag}_mae"] = round(float(mean_absolute_error(y, p)), 2)
    S[f"c1a_{tag}_rmse"] = round(float(np.sqrt(((y - p) ** 2).mean())), 2)
    S[f"c1a_{tag}_r2"] = round(float(r2_score(y, p)), 3)
S["c1a_mae_gain_pct"] = round(100 * (1 - S["c1a_hgb_mae"] / S["c1a_mean_mae"]), 1)

# leakage demonstration: same model, same split, forbidden column added
tr_l = sp["train"].dropna(subset=["target_is_late"])
te_l = sp["test"].dropna(subset=["target_is_late"])
leak_feats = MODEL_FEATURES + ["delivery_days"]
m = pipe(LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=config.SEED), NUM + ["delivery_days"])
m.fit(tr_l[leak_feats], tr_l["target_is_late"])
pl = m.predict_proba(te_l[leak_feats])[:, 1]
S["c1b_leaked_roc"] = round(float(roc_auc_score(te_l["target_is_late"], pl)), 3)
S["c1b_leaked_balacc"] = round(float(balanced_accuracy_score(
    te_l["target_is_late"], (pl >= 0.5).astype(int))), 3)

# unseen-customer generalisation for the primary classifier
un = sp["test_unseen_customers"].dropna(subset=["target_low_review"])
m = pipe(HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                        random_state=config.SEED),
         NUM + config.DELIVERY_OUTCOME_FEATURES)
m.fit(sp["train"].dropna(subset=["target_low_review"])[REVIEW_MODEL_FEATURES],
      sp["train"].dropna(subset=["target_low_review"])["target_low_review"])
pu = m.predict_proba(un[REVIEW_MODEL_FEATURES])[:, 1]
S["c2_unseen_rows"] = len(un)
S["c2_unseen_prauc"] = round(float(average_precision_score(
    un["target_low_review"].astype(int), pu)), 3)


# ===========================================================================
# Added after the Milestone 1 feedback. Everything above is unchanged, so the
# numbers the submitted report quoted still regenerate to the same values; the
# keys below back the corrections the feedback asked for.
# ===========================================================================
# -- growth: "roughly triples" was wrong ---------------------------------------
vol_all = df.groupby(df["order_purchase_timestamp"].dt.to_period("M")).size()
S["growth_first_month"] = str(vol_all.index[0])
S["growth_first_month_orders"] = int(vol_all.iloc[0])
S["growth_peak_month"] = str(vol_all.idxmax())
S["growth_peak_orders"] = int(vol_all.max())
S["growth_peak_ratio"] = round(float(vol_all.max() / vol_all.iloc[0]), 1)
S["growth_2018_mean_orders"] = int(round(vol_all["2018-01":"2018-08"].mean()))
S["growth_2018_mean_ratio"] = round(float(vol_all["2018-01":"2018-08"].mean() / vol_all.iloc[0]), 1)
S["growth_2018_min_orders"] = int(vol_all["2018-01":"2018-08"].min())
S["growth_2018_max_orders"] = int(vol_all["2018-01":"2018-08"].max())
S["black_friday_over_median_x"] = round(float(vol_all.max() / vol_all.median()), 1)

# -- when the late deliveries happened ----------------------------------------
for ym, tag in [("2017-10", "oct2017"), ("2017-11", "nov2017"), ("2018-01", "jan2018"),
                ("2018-02", "feb2018"), ("2018-03", "mar2018"), ("2018-04", "apr2018"),
                ("2018-05", "may2018"), ("2018-06", "jun2018"), ("2018-08", "aug2018")]:
    S[f"late_{tag}_pct"] = round(100 * float(monthly.loc[ym, "late"]), 1)

d2 = d.copy()
d2["carrier_days"] = ((d2["order_delivered_customer_date"] - d2["order_delivered_carrier_date"])
                      .dt.total_seconds() / 86400)
d2["seller_days"] = ((d2["order_delivered_carrier_date"] - d2["order_purchase_timestamp"])
                     .dt.total_seconds() / 86400)
pm = d2["order_purchase_timestamp"].dt.to_period("M").astype(str)
peak = pm.isin(["2018-02", "2018-03"])
S["peak_carrier_days"] = round(float(d2.loc[peak, "carrier_days"].mean()), 1)
S["rest_carrier_days"] = round(float(d2.loc[~peak, "carrier_days"].mean()), 1)
S["peak_seller_days"] = round(float(d2.loc[peak, "seller_days"].mean()), 1)
S["rest_seller_days"] = round(float(d2.loc[~peak, "seller_days"].mean()), 1)
S["promise_jan2018_days"] = round(float(d2.loc[pm == "2018-01", "estimated_days"].mean()), 1)
S["promise_mar2018_days"] = round(float(d2.loc[pm == "2018-03", "estimated_days"].mean()), 1)
S["promise_jun2018_days"] = round(float(d2.loc[pm == "2018-06", "estimated_days"].mean()), 1)

# Monday-to-Sunday purchase weeks around the strike of 21 May - 1 June 2018.
wk = d2.groupby(d2["order_purchase_timestamp"].dt.to_period("W-SUN").dt.start_time).agg(
    n=("order_id", "size"), late=("target_is_late", "mean"), promise=("estimated_days", "mean"))
wk.index = wk.index.strftime("%Y-%m-%d")
S["strike_week_before_late_pct"] = round(100 * float(wk.loc["2018-05-14", "late"]), 1)
S["strike_week_before_promise_days"] = round(float(wk.loc["2018-05-14", "promise"]), 1)
S["strike_week_late_pct"] = round(100 * float(wk.loc["2018-05-21", "late"]), 1)
S["strike_week_promise_days"] = round(float(wk.loc["2018-05-21", "promise"]), 1)
S["strike_week_after_late_pct"] = round(100 * float(wk.loc["2018-05-28", "late"]), 1)
S["strike_week_after_promise_days"] = round(float(wk.loc["2018-05-28", "promise"]), 1)
S["weeks_apr_early_may_mean_late_pct"] = round(
    100 * float(wk.loc["2018-04-09":"2018-05-07", "late"].mean()), 1)
in_transit = ((d2["order_purchase_timestamp"] < pd.Timestamp("2018-06-01"))
              & (d2["order_delivered_customer_date"] >= pd.Timestamp("2018-05-21")))
S["strike_in_transit_n"] = int(in_transit.sum())
S["strike_in_transit_late_pct"] = round(100 * float(d2.loc[in_transit, "target_is_late"].mean()), 1)

# -- the correlation matrix the text never discussed (Figure 8) ----------------
# Computed on the full analysis frame with pairwise deletion, exactly as
# notebook 03 draws the figure, so these match the printed cells.
S["corr_fig8_leadtime_distance"] = round(float(df["target_delivery_days"].corr(df["distance_km"])), 2)
S["corr_fig8_leadtime_promise"] = round(float(df["target_delivery_days"].corr(df["estimated_days"])), 2)
S["corr_fig8_leadtime_freight"] = round(float(df["target_delivery_days"].corr(df["total_freight"])), 2)
S["corr_fig8_leadtime_weight"] = round(float(df["target_delivery_days"].corr(df["total_weight_g"])), 2)
S["corr_fig8_promise_distance"] = round(float(df["estimated_days"].corr(df["distance_km"])), 2)
S["corr_fig8_freight_weight"] = round(float(df["total_freight"].corr(df["total_weight_g"])), 2)
S["corr_fig8_review_leadtime"] = round(float(df["review_score"].corr(df["target_delivery_days"])), 2)
S["corr_fig8_review_promise"] = round(float(df["review_score"].corr(df["estimated_days"])), 2)
S["corr_fig8_leadtime_price"] = round(float(df["total_price"].corr(df["target_delivery_days"])), 2)
S["corr_fig8_leadtime_items"] = round(float(df["n_items"].corr(df["target_delivery_days"])), 2)
S["corr_fig8_review_items"] = round(float(df["review_score"].corr(df["n_items"])), 2)


# ===========================================================================
if __name__ == "__main__":
    out = config.PROJECT_ROOT / "report" / "stats.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k:34s} {v}" for k, v in S.items()]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}  ({len(S)} values)")
