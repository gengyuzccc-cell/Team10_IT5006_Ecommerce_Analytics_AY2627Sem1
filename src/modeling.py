"""Model families, preprocessing and the cross-validation protocol (Phase 2).

Everything a model touches between the feature frame and a fitted estimator
lives here, so the notebooks, the training script and the dashboard cannot
disagree about how a model was built.

Model family budget
-------------------
The brief allows two to three families for the whole project and asks for depth
rather than breadth. We use three, and reuse each across classification and
regression:

    A  Linear      LinearRegression / LogisticRegression   -> Ridge / L2 logistic
    B  Tree-based  DecisionTree -> RandomForest -> HistGradientBoosting
    C  Ensemble    Voting and Stacking assembled from A + B

Within each family the simplest variant is fitted first and every step up the
ladder has to earn its place against it. `rung` records that ordering, so the
results tables can be read as a complexity curve rather than a list.

Preprocessing is family-appropriate, and Phase 2 measured why rather than
assuming it. Phase 1 reported that plain least squares was unusable here -
"the design matrix is rank-deficient, so its coefficients explode" - and the
re-check (`encoding_diagnostic.csv`) confirms the observation while correcting
the conclusion:

    encoding            rank/cols   max |coef|      CV MAE   test MAE
    OLS,   full block     101/105     2.7e+13     3.9e+11      4.181
    OLS,   drop first     101/101        14.7       5.242      4.181
    Ridge, full block     101/105        9.82       5.241      4.182
    Ridge, drop first     101/101        12.5       5.240      4.183

The deficiency is exactly four - one per categorical, because each one-hot block
sums to one and is therefore collinear with the intercept. Least squares still
returns a minimum-norm solution, and its *predictions* are fine: 4.181 days on
the test window, indistinguishable from Ridge. What is not fine is the
coefficient vector, at 10^13, and the cancellation that holds it together is
fragile enough to fail on one of five expanding-window folds. A single
train/test split would never have shown it.

So linear models drop one level per categorical, which restores full rank and
identified coefficients, and are scaled, because a shared penalty is meaningless
across features on different scales. Trees do neither: they are invariant to
monotone rescaling, and the full block lets an unseen level encode as all zeros
- distinct from every known level rather than silently equal to the reference
one. The remaining lesson is about Ridge: with a full-rank design it agrees with
OLS to three decimals. Regularisation buys numerical safety here, not accuracy.
Accuracy comes from dropping linearity altogether.

Splitting and cross-validation
------------------------------
Delivery performance drifts hard (mean lead time halves across the window), so
shuffled k-fold lets a model learn from the future - the invalidity Bergmeir and
Benitez (2012) formalise. `temporal_cv()` is therefore the selection criterion
throughout. `grouped_cv()` exists only to quantify what the invalid protocol
would have told us, and is never used to choose a model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import config

LINEAR, TREE, ENSEMBLE = "Linear", "Tree-based", "Ensemble"


# ===========================================================================
# Preprocessing
# ===========================================================================
def make_preprocessor(numeric: list[str], categorical: list[str],
                      family: str) -> ColumnTransformer:
    """Impute, encode and (for linear models) scale.

    Fitted inside every pipeline, so no fold ever sees statistics computed on
    the data it is scored against.

    `min_frequency=100` folds rare categories into one bucket - 27 of the 73
    product categories in the training period fall below that, and each would
    otherwise contribute a column seen a handful of times.
    `handle_unknown="infrequent_if_exist"` then sends a category absent from
    training into that same bucket.

    Where a column has no rare bucket to send it to - `lead_payment_type` has
    only four levels and all four clear the threshold - an unseen level encodes
    as all zeros and scikit-learn warns. For a tree that is the right answer:
    "none of the levels I was trained on" is precisely what is true, and it is a
    distinct row. For a linear model with `drop="first"` it is a genuine, if
    small, compromise, because an all-zero row is also how the dropped reference
    level is represented; a novel payment type is scored as a credit card. One
    payment level is unseen in the test window, affecting a handful of orders,
    and the alternative - leaving the design rank-deficient - costs more (see
    the module docstring).
    """
    linear = family == LINEAR
    steps: list = [("imp", SimpleImputer(strategy="median"))]
    if linear:
        steps.append(("sc", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(steps), numeric),
        ("cat", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=100,
            drop="first" if linear else None,
            sparse_output=False,
        ), categorical),
    ])


# ===========================================================================
# Model specifications
# ===========================================================================
@dataclass(frozen=True)
class ModelSpec:
    """One rung of one family's complexity ladder."""

    key: str
    label: str
    family: str
    rung: int                       # 0 = the family's simplest variant
    make: Callable[[], BaseEstimator]
    space: dict = field(default_factory=dict)   # RandomizedSearchCV distributions
    # A random forest costs one to two orders of magnitude more per fit than the
    # other rungs - 150 trees over a 100-column design, five times per candidate.
    # It therefore gets its own, smaller draw budget, and a grid trimmed towards
    # the cheap end. That is a documented trade, not a hidden one: spending an
    # extra half hour on more forests, when Phase 1 already showed the forest and
    # the boosted trees landing within 0.01 days of each other, would buy less
    # than spending it on the two problems still unmodelled.
    search_iter: int | None = None

    @property
    def is_baseline(self) -> bool:
        return self.rung == 0

    def pipeline(self, numeric: list[str], categorical: list[str]):
        """The estimator as it is actually fitted, preprocessing included.

        Family C is the exception: its members are complete pipelines already,
        each carrying the encoding its own family needs. Wrapping them in a
        further shared preprocessor would force one encoding on both, and the
        drop-first block that a linear model requires is not the block a tree
        wants.
        """
        if self.family == ENSEMBLE:
            return self.make()
        return Pipeline([("prep", make_preprocessor(numeric, categorical, self.family)),
                         ("model", self.make())])


SEED = config.SEED


def regression_zoo() -> list[ModelSpec]:
    """The two family ladders, for the lead-time regression."""
    return [
        ModelSpec(
            "lin_ols", "LinearRegression", LINEAR, 0,
            lambda: LinearRegression(),
        ),
        ModelSpec(
            "lin_ridge", "Ridge", LINEAR, 1,
            lambda: Ridge(random_state=SEED),
            {"model__alpha": loguniform(1e-3, 1e3)},
        ),
        ModelSpec(
            "tree_dt", "DecisionTreeRegressor", TREE, 0,
            lambda: DecisionTreeRegressor(random_state=SEED),
            {"model__max_depth": randint(3, 20),
             "model__min_samples_leaf": randint(5, 200),
             "model__max_features": uniform(0.4, 0.6)},
        ),
        ModelSpec(
            "tree_rf", "RandomForestRegressor", TREE, 1,
            lambda: RandomForestRegressor(n_estimators=150, n_jobs=1,
                                          random_state=SEED),
            {"model__max_depth": randint(8, 18),
             "model__min_samples_leaf": randint(20, 120),
             "model__max_features": uniform(0.15, 0.25)},
            search_iter=6,
        ),
        ModelSpec(
            "tree_hgb", "HistGradientBoostingRegressor", TREE, 2,
            lambda: HistGradientBoostingRegressor(random_state=SEED),
            {"model__max_iter": randint(120, 500),
             "model__learning_rate": loguniform(0.02, 0.3),
             "model__max_leaf_nodes": randint(15, 64),
             "model__min_samples_leaf": randint(10, 120),
             "model__l2_regularization": loguniform(1e-4, 1e1)},
        ),
    ]


def classification_zoo() -> list[ModelSpec]:
    """The same two ladders, for the binary targets.

    `class_weight` is left at config.CLASS_WEIGHT_POLICY (None) throughout.
    Weighting the loss does not improve discrimination on these targets and
    leaves probabilities badly miscalibrated; the decision threshold is tuned
    on the validation period instead. Notebook 08 measures both arms.
    """
    cw = config.CLASS_WEIGHT_POLICY
    return [
        ModelSpec(
            "lin_logit", "LogisticRegression", LINEAR, 0,
            lambda: LogisticRegression(max_iter=3000, class_weight=cw,
                                       random_state=SEED),
        ),
        ModelSpec(
            "lin_logit_l2", "LogisticRegression (tuned L2)", LINEAR, 1,
            lambda: LogisticRegression(max_iter=3000, penalty="l2",
                                       class_weight=cw, random_state=SEED),
            {"model__C": loguniform(1e-3, 1e2)},
        ),
        ModelSpec(
            "tree_dt", "DecisionTreeClassifier", TREE, 0,
            lambda: DecisionTreeClassifier(class_weight=cw, random_state=SEED),
            {"model__max_depth": randint(3, 16),
             "model__min_samples_leaf": randint(20, 400),
             "model__max_features": uniform(0.4, 0.6)},
        ),
        ModelSpec(
            "tree_rf", "RandomForestClassifier", TREE, 1,
            lambda: RandomForestClassifier(n_estimators=150, n_jobs=1,
                                           class_weight=cw, random_state=SEED),
            {"model__max_depth": randint(6, 16),
             "model__min_samples_leaf": randint(30, 150),
             "model__max_features": uniform(0.15, 0.25)},
            search_iter=6,
        ),
        ModelSpec(
            "tree_hgb", "HistGradientBoostingClassifier", TREE, 2,
            lambda: HistGradientBoostingClassifier(class_weight=cw,
                                                   random_state=SEED),
            {"model__max_iter": randint(120, 500),
             "model__learning_rate": loguniform(0.02, 0.3),
             "model__max_leaf_nodes": randint(15, 64),
             "model__min_samples_leaf": randint(10, 120),
             "model__l2_regularization": loguniform(1e-4, 1e1)},
        ),
    ]


def ensemble_specs(best_linear, best_tree, task: str) -> list[ModelSpec]:
    """Family C, assembled from the winners of families A and B.

    `best_linear` and `best_tree` are the tuned pipelines those searches
    returned. They are cloned - unfitted, but carrying their chosen
    hyperparameters - so the ensemble refits them on whatever fold it is given
    rather than reusing estimators already fitted on the full training set.

    The inner cross-validation is a compromise we could not avoid, and it is
    worth stating plainly. Stacking builds its meta-features with
    `cross_val_predict`, which requires the inner folds to *partition* the
    training set - every row must be predicted exactly once. An expanding window
    does not partition anything: the earliest rows are never in a test fold. So
    the closest valid approximation is `KFold(shuffle=False)`, which at least
    keeps each fold contiguous in time, and the meta-learner is consequently
    fitted on out-of-fold predictions that are in part made by models trained on
    later orders.

    The residual optimism is confined to the meta-learner's two coefficients -
    the base models are refitted on the whole training period afterwards, and
    the outer protocol that scores the finished ensemble is still the expanding
    window. It matters less than it would elsewhere because of how the result
    lands: the ensemble is *flattered* by this arrangement and still does not
    win, which makes the conclusion safer rather than weaker.

    The Voting weights are 1:2 in the tree's favour. That is a stated prior, not
    a tuned parameter: on drifting data the linear member is materially weaker,
    and an equal blend would be a worse model than its better half.
    """
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    best_linear, best_tree = clone(best_linear), clone(best_tree)
    inner = KFold(n_splits=5, shuffle=False)
    if task == "regression":
        return [
            ModelSpec("ens_vote", "VotingRegressor (A+B)", ENSEMBLE, 0,
                      lambda: VotingRegressor(
                          [("linear", best_linear), ("tree", best_tree)],
                          weights=[1, 2])),
            ModelSpec("ens_stack", "StackingRegressor (A+B)", ENSEMBLE, 1,
                      lambda: StackingRegressor(
                          [("linear", best_linear), ("tree", best_tree)],
                          final_estimator=Ridge(alpha=1.0, random_state=SEED),
                          cv=inner, passthrough=False, n_jobs=1)),
        ]
    return [
        ModelSpec("ens_vote", "VotingClassifier (A+B, soft)", ENSEMBLE, 0,
                  lambda: VotingClassifier(
                      [("linear", best_linear), ("tree", best_tree)],
                      voting="soft", weights=[1, 2])),
        ModelSpec("ens_stack", "StackingClassifier (A+B)", ENSEMBLE, 1,
                  lambda: StackingClassifier(
                      [("linear", best_linear), ("tree", best_tree)],
                      final_estimator=LogisticRegression(
                          max_iter=2000, random_state=SEED),
                      cv=inner, passthrough=False, n_jobs=1)),
    ]


# ===========================================================================
# Cross-validation protocol
# ===========================================================================
def temporal_cv(n_splits: int = config.CV_SPLITS) -> TimeSeriesSplit:
    """Expanding-window CV. The caller MUST pass time-sorted rows."""
    return TimeSeriesSplit(n_splits=n_splits)


def grouped_cv(n_splits: int = config.CV_SPLITS) -> StratifiedGroupKFold:
    """Shuffled, customer-disjoint, stratified CV.

    Reported as a contrast only. It is the protocol a reasonable person would
    reach for and it is invalid here: it trains on orders placed after the ones
    it scores. Notebook 09 measures how much it flatters the result.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def sort_for_cv(df: pd.DataFrame) -> pd.DataFrame:
    """TimeSeriesSplit slices positionally, so chronology must be in the order."""
    return df.sort_values("order_purchase_timestamp")


def cv_score(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, numeric: list[str],
             categorical: list[str], scoring: str, cv=None) -> tuple[float, float, np.ndarray]:
    """Mean, standard deviation and per-fold scores under the given protocol."""
    scores = cross_val_score(spec.pipeline(numeric, categorical), X, y,
                             cv=cv or temporal_cv(), scoring=scoring, n_jobs=1)
    return float(scores.mean()), float(scores.std()), scores


# ===========================================================================
# Hyperparameter search
# ===========================================================================
def tune(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, numeric: list[str],
         categorical: list[str], scoring: str, n_iter: int = config.SEARCH_ITER,
         cv=None) -> dict:
    """Randomised search over `spec.space`, scored by expanding-window CV.

    A model with no search space (the family baselines) is simply fitted and
    scored under the same protocol, so its row in the results table is directly
    comparable with the tuned ones rather than being a different kind of number.
    """
    pipe = spec.pipeline(numeric, categorical)
    if not spec.space:
        mean, std, folds = cv_score(spec, X, y, numeric, categorical, scoring, cv)
        pipe.fit(X, y)
        return {"key": spec.key, "label": spec.label, "family": spec.family,
                "rung": spec.rung, "estimator": pipe, "params": {},
                "cv_mean": mean, "cv_std": std, "cv_folds": folds, "n_candidates": 1}

    n_iter = min(n_iter, spec.search_iter or n_iter)
    search = RandomizedSearchCV(
        pipe, spec.space, n_iter=n_iter, scoring=scoring, cv=cv or temporal_cv(),
        random_state=SEED, n_jobs=config.N_JOBS, refit=True, error_score="raise",
    )
    search.fit(X, y)
    i = search.best_index_
    folds = np.array([search.cv_results_[f"split{k}_test_score"][i]
                      for k in range(search.n_splits_)])
    return {"key": spec.key, "label": spec.label, "family": spec.family,
            "rung": spec.rung, "estimator": search.best_estimator_,
            "params": {k.replace("model__", ""): v
                       for k, v in search.best_params_.items()},
            "cv_mean": float(search.best_score_),
            "cv_std": float(search.cv_results_["std_test_score"][i]),
            "cv_folds": folds, "n_candidates": n_iter}


# ===========================================================================
# Artifacts
# ===========================================================================
def save_model(estimator, path: Path, **metadata) -> Path:
    """Persist an estimator alongside the environment that produced it.

    scikit-learn does not guarantee that a pickle written by one version loads
    correctly into another, and a silently mis-loaded model is worse than one
    that refuses to load. The version is recorded here and asserted on the way
    back in.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "estimator": estimator,
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **metadata,
    }, path, compress=3)
    return path


def load_model(path: Path, strict: bool = False) -> dict:
    """Load a bundle written by `save_model`, checking the scikit-learn version."""
    bundle = joblib.load(Path(path))
    trained_on = bundle.get("sklearn_version")
    if trained_on != sklearn.__version__:
        message = (f"{Path(path).name} was trained on scikit-learn {trained_on}, "
                   f"this environment has {sklearn.__version__}. Predictions may "
                   "differ from the reported metrics.")
        if strict:
            raise RuntimeError(message)
        bundle["version_warning"] = message
    return bundle


def write_metadata(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
