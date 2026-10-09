"""Metrics, thresholds, calibration and interpretability (Phase 2).

One module so that every table in the report, every number in the notebooks and
every figure on the dashboard is computed by the same code path.

Two conventions run through it.

**Accuracy is reported but never led with.** Both targets are heavily
imbalanced - 7.4% late, 11.0% low review in the test period - so a rule that
always predicts the majority class scores 0.93 and 0.89 while catching nothing.
`classification_row` therefore carries `majority_accuracy` beside `accuracy` in
every row, so the two can never be read apart.

**PR-AUC is quoted against its floor.** The no-skill baseline for average
precision is the positive rate itself, which differs between the training period
(15.8% low review) and the test period (11.0%). Raw PR-AUC is consequently not
comparable across periods; `pr_auc_lift` - the ratio to the floor - is, and it
turns out to be the stable quantity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

import config


# ===========================================================================
# Result rows
# ===========================================================================
def regression_row(label: str, y_true, y_pred, **extra) -> dict:
    """MAE, RMSE and R^2 for one predictor.

    MAE leads. Lead time falls 5.34 days between the training and test periods,
    and under a level shift that size R^2 against a variance computed on the
    test period goes negative even for a model that is clearly useful - so R^2
    is reported for completeness and read as a drift diagnostic, not a score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "model": label,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        **extra,
    }


def classification_row(label: str, y_true, y_score, threshold: float = 0.5,
                       **extra) -> dict:
    """Every metric the brief asks for, plus the two that stop it misleading."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_hat = (y_score >= threshold).astype(int)
    floor = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, y_score))
    return {
        "model": label,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": pr_auc,
        "pr_auc_floor": floor,
        "pr_auc_lift": pr_auc / floor if floor else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "majority_accuracy": max(floor, 1 - floor),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0, 1))),
        "threshold": float(threshold),
        **extra,
    }


# ===========================================================================
# Decision threshold
# ===========================================================================
def tune_threshold(y_true, y_score, objective: str = None,
                   grid: tuple = None) -> tuple[float, float]:
    """Pick the operating point on held-out data, never on the test window.

    The alternative - reweighting the loss so 0.5 becomes a sensible cut - is
    what van den Goorbergh et al. (2022) advise against: it does not improve
    ranking and it distorts the probabilities. Moving the threshold changes only
    the decision, leaving the model calibrated.
    """
    objective = objective or config.THRESHOLD_OBJECTIVE
    lo, hi, step = grid or config.THRESHOLD_GRID
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    scorer = {
        "f1": lambda y, p: f1_score(y, p, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score,
    }[objective]

    candidates = np.arange(lo, hi + 1e-9, step)
    scores = [scorer(y_true, (y_score >= t).astype(int)) for t in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def threshold_sweep(y_true, y_score, grid: tuple = None) -> pd.DataFrame:
    """Precision, recall, F1 and flagged volume across the whole threshold range."""
    lo, hi, step = grid or config.THRESHOLD_GRID
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rows = []
    for t in np.arange(lo, hi + 1e-9, step):
        y_hat = (y_score >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_true, y_hat, zero_division=0)),
            "recall": float(recall_score(y_true, y_hat, zero_division=0)),
            "f1": float(f1_score(y_true, y_hat, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
            "flagged_pct": float(100 * y_hat.mean()),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Calibration
# ===========================================================================
def calibration_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Observed rate against predicted risk, in equal-width probability bins.

    A model whose scores are used to rank orders needs only a monotone score; a
    model whose scores feed an expected-cost calculation needs them to mean what
    they say. We do the second, so calibration is reported, not assumed.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_score, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": int(m.sum()),
            "mean_predicted": float(y_score[m].mean()),
            "observed_rate": float(y_true[m].mean()),
        })
    out = pd.DataFrame(rows)
    out["gap"] = out["mean_predicted"] - out["observed_rate"]
    return out


# ===========================================================================
# Interpretability
# ===========================================================================
def permutation_table(estimator, X: pd.DataFrame, y, scoring: str,
                      n_repeats: int = 5, sample: int = 6000,
                      seed: int = config.SEED) -> pd.DataFrame:
    """Permutation importance on held-out data.

    Impurity-based importances are computed on the training set and inflate
    high-cardinality features; permuting a column in the test window and
    measuring the loss answers the question the stakeholder actually asks -
    how much does this model rely on knowing this?
    """
    rng = np.random.default_rng(seed)
    if sample and len(X) > sample:
        take = rng.choice(len(X), size=sample, replace=False)
        X, y = X.iloc[take], np.asarray(y)[take]
    result = permutation_importance(estimator, X, y, scoring=scoring,
                                    n_repeats=n_repeats, random_state=seed,
                                    n_jobs=config.N_JOBS)
    return (pd.DataFrame({"feature": list(X.columns),
                          "importance": result.importances_mean,
                          "std": result.importances_std})
            .sort_values("importance", ascending=False, ignore_index=True))


def linear_coefficients(pipeline, numeric: list[str],
                        categorical: list[str]) -> pd.DataFrame:
    """Standardised coefficients from a fitted linear pipeline.

    Inputs are scaled inside the pipeline, so the coefficients are already on a
    common footing and can be compared directly - one standard deviation of this
    feature moves the prediction by this many days / log-odds.
    """
    prep = pipeline.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    coefs = np.ravel(pipeline.named_steps["model"].coef_)
    return (pd.DataFrame({"feature": names, "coefficient": coefs})
            .assign(magnitude=lambda d: d["coefficient"].abs())
            .sort_values("magnitude", ascending=False, ignore_index=True))


# ===========================================================================
# Business decision layer
# ===========================================================================
def cost_curve(y_true, y_score, cost_per_action: float, cost_per_miss: float,
               grid: tuple = None) -> pd.DataFrame:
    """Expected net benefit of acting on everything above each threshold.

    The costs are assumptions, not measurements - Olist publishes none - so the
    output is always read as a shape rather than a number, and the notebooks
    sweep both parameters to show where the recommended operating point stops
    being the right one.

    Acting on a true positive avoids `cost_per_miss` and spends
    `cost_per_action`; acting on a false positive spends `cost_per_action` for
    nothing; not acting on a positive incurs `cost_per_miss`. Net benefit is
    measured against the do-nothing policy.
    """
    lo, hi, step = grid or config.THRESHOLD_GRID
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rows = []
    for t in np.arange(lo, hi + 1e-9, step):
        y_hat = (y_score >= t).astype(int)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        rows.append({
            "threshold": float(t),
            "flagged": int(y_hat.sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "action_cost": (tp + fp) * cost_per_action,
            "miss_cost": fn * cost_per_miss,
            "net_benefit": tp * cost_per_miss - (tp + fp) * cost_per_action,
        })
    out = pd.DataFrame(rows)
    out["net_benefit_per_1k"] = 1000 * out["net_benefit"] / len(y_true)
    return out


# ===========================================================================
# Capacity-constrained evaluation
# ===========================================================================
# PR-AUC summarises the whole ranking and is the right quantity for choosing
# between models. It is not the quantity a team with limited capacity lives
# with. A customer-experience team that can contact K orders a day works the
# list from the top and never sees the rest, so the metric that describes its
# day is: of the K it reached, how many would really have left a 1-2 star
# review (Precision@K), and what share of all such orders did those K cover
# (Recall@K). The Milestone 1 feedback asked for exactly this alignment between
# the metric and the business action.
def precision_recall_at_k(y_true, y_score,
                          fractions=(0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15,
                                     0.20, 0.25, 0.30, 0.40, 0.50)) -> pd.DataFrame:
    """Precision, recall and lift when only the top-K scored orders are acted on.

    `fractions` are shares of the scored population, so K scales with volume
    and the same table reads correctly on a quiet day and a Black Friday. The
    ranking is by score, ties broken by position, which is what a sorted list
    does. Lift is precision divided by the base rate: how much better than
    picking K orders at random.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    hits = y_true[np.argsort(-y_score, kind="stable")]
    n, positives = len(hits), int(hits.sum())
    base = positives / n if n else np.nan
    rows = []
    for f in fractions:
        k = max(1, int(round(f * n)))
        tp = int(hits[:k].sum())
        precision = tp / k
        rows.append({
            "k_pct": float(100 * f), "k": k, "tp": tp,
            "precision_at_k": precision,
            "recall_at_k": tp / positives if positives else np.nan,
            "lift_at_k": precision / base if base else np.nan,
        })
    return pd.DataFrame(rows)


def capacity_curve(y_true, y_score, timestamps, per_period_k=(5, 10, 20, 40),
                   freq: str = "D") -> pd.DataFrame:
    """Top-K per calendar period: what a fixed daily capacity actually delivers.

    `precision_recall_at_k` ranks the whole window at once, which no team can
    do - it sees today's deliveries today. Here the list is cut per period
    (default: per day of purchase), the K highest-scored orders are taken from
    each, and precision and recall are computed over everything acted on.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=int),
                          "s": np.asarray(y_score, dtype=float)})
    frame["period"] = pd.to_datetime(pd.Series(np.asarray(timestamps))).dt.to_period(freq).to_numpy()
    positives = int(frame["y"].sum())
    ranked = frame.sort_values(["period", "s"], ascending=[True, False], kind="stable")
    ranked["pos"] = ranked.groupby("period").cumcount()
    rows = []
    for k in per_period_k:
        top = ranked[ranked["pos"] < k]
        tp, flagged = int(top["y"].sum()), int(len(top))
        rows.append({"k_per_period": int(k), "freq": freq,
                     "periods": int(frame["period"].nunique()),
                     "flagged": flagged,
                     "flagged_pct": 100 * flagged / len(frame),
                     "tp": tp,
                     "precision": tp / flagged if flagged else np.nan,
                     "recall": tp / positives if positives else np.nan})
    return pd.DataFrame(rows)


def segment_metrics(frame: pd.DataFrame, y_col: str, score_col: str, by: str,
                    threshold: float = 0.5, min_n: int = 200) -> pd.DataFrame:
    """Per-segment performance - the robustness and fairness check.

    An aggregate score can hide a model that works for Sao Paulo and fails
    everywhere else. Segments below `min_n` are dropped rather than reported
    with an unusable confidence interval.
    """
    rows = []
    for name, g in frame.groupby(by, observed=True):
        y = g[y_col].astype(int)
        if len(g) < min_n or y.nunique() < 2:
            continue
        rows.append({by: name, "n": len(g), "positive_rate": float(y.mean()),
                     **{k: v for k, v in
                        classification_row(str(name), y, g[score_col], threshold).items()
                        if k in ("roc_auc", "pr_auc", "pr_auc_lift",
                                 "balanced_accuracy", "precision", "recall", "f1")}})
    if not rows:
        # Every segment fell below min_n or held a single class. An empty table
        # is the honest answer - better than a row computed from 40 orders.
        return pd.DataFrame(columns=[by, "n", "positive_rate", "roc_auc", "pr_auc",
                                     "pr_auc_lift", "balanced_accuracy",
                                     "precision", "recall", "f1"])
    return pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)


def segment_regression(frame: pd.DataFrame, y_col: str, pred_col: str, by: str,
                       min_n: int = 200) -> pd.DataFrame:
    """The same robustness check for the regression."""
    rows = []
    for name, g in frame.groupby(by, observed=True):
        if len(g) < min_n:
            continue
        rows.append({by: name, "n": len(g),
                     "mae": float(mean_absolute_error(g[y_col], g[pred_col])),
                     "bias": float((g[pred_col] - g[y_col]).mean()),
                     "actual_mean": float(g[y_col].mean())})
    if not rows:
        return pd.DataFrame(columns=[by, "n", "mae", "bias", "actual_mean"])
    return pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)


# ===========================================================================
# Model selection
# ===========================================================================
def rank_select(window_scores: dict[str, list[float]],
                window_names: list[str] | None = None,
                recent: int | None = None) -> pd.DataFrame:
    """Rank-averaged rolling-origin selection. Higher scores are better.

    Every candidate is scored on the same set of out-of-sample windows - the
    expanding cross-validation folds inside the training period, plus the
    validation period - and ranked within each window. The winner is the model
    with the best mean rank.

    Two properties matter here, and a single-window criterion has neither.

    *Ranks, not scores.* The windows differ enormously in difficulty: mean lead
    time halves across the analysis period, so early folds carry larger absolute
    errors, and the low-review base rate falls from 15.8% to 11.0%, which moves
    PR-AUC mechanically because its no-skill floor *is* the base rate. Averaging
    raw scores across windows lets the hardest window decide the ordering.
    Averaging ranks asks the only question that transfers: on this stretch of
    data, which model was better?

    *Several windows, not one, and recent ones.* The validation period is two
    months and those two have opposite character - an 8.2% late rate in May 2018
    against 1.4% in June, the lowest in the series - so a criterion computed on
    it alone is noisy enough to reorder candidates. But averaging over all six
    would hand the 2017 folds five sixths of the vote on a question about
    mid-2018, and lead time there runs 12-17 days against 8.3 in the test window.
    `recent` (default `config.SELECTION_WINDOWS`) votes on the last few windows
    only; every window is still scored and returned, so the full picture is
    visible in `selection_*.csv`.

    Ties on mean rank are broken by the last window, the one closest to
    deployment.
    """
    labels = list(window_scores)
    widths = {len(v) for v in window_scores.values()}
    if len(widths) != 1:
        raise ValueError(f"candidates were scored on different windows: {widths}")
    n = widths.pop()
    names = window_names or [f"fold{i + 1}" for i in range(n - 1)] + ["validation"]

    recent = config.SELECTION_WINDOWS if recent is None else recent
    voting = names[-recent:] if recent else names

    frame = pd.DataFrame(window_scores, index=names).T
    ranks = frame.rank(axis=0, ascending=False, method="average")
    out = frame.copy()
    out.columns = [f"score_{c}" for c in names]
    for c in names:
        out[f"rank_{c}"] = ranks[c]
    out["mean_rank_all"] = ranks.mean(axis=1)
    out["mean_rank"] = ranks[voting].mean(axis=1)
    out["windows_voting"] = ", ".join(voting)
    out["last_window"] = frame[names[-1]]
    return (out.sort_values(["mean_rank", "last_window"], ascending=[True, False])
            .reset_index().rename(columns={"index": "model"}))


# ===========================================================================
# The leak alarm
# ===========================================================================
def check_bands(values: dict[str, float], bands: dict = None,
                verbose: bool = True) -> pd.DataFrame:
    """Compare headline results against the bands registered before tuning.

    A result far above its band is not a triumph. Adding one forbidden column to
    the Phase 1 late-delivery model took ROC-AUC from 0.589 to 1.000, and that
    is what an over-band result usually means. Under-band is a different
    conversation - it is a modelling shortfall, not a validity failure - so the
    two are labelled differently.
    """
    bands = bands or config.EXPECTED_BANDS
    rows = []
    for key, value in values.items():
        lo, hi = bands.get(key, (np.nan, np.nan))
        # An error metric is suspicious when it is too LOW; a score when it is
        # too high. Same alarm, opposite ends of the interval.
        too_good = value < lo if key in config.LOWER_IS_BETTER else value > hi
        too_weak = value > hi if key in config.LOWER_IS_BETTER else value < lo
        verdict = ("unregistered" if np.isnan(lo)
                   else "SUSPICIOUSLY GOOD - investigate for leakage" if too_good
                   else "short of target" if too_weak
                   else "in band")
        rows.append({"metric": key, "value": round(float(value), 4),
                     "band": f"[{lo}, {hi}]", "verdict": verdict})
    out = pd.DataFrame(rows)
    if verbose:
        for _, r in out.iterrows():
            mark = {"in band": " ok ", "unregistered": " -- ",
                    "short of target": "MISS"}.get(r["verdict"], "LEAK")
            print(f"  [{mark}] {r['metric']:24s} {r['value']:>8}  {r['band']:>14}  "
                  f"{r['verdict']}")
    return out
