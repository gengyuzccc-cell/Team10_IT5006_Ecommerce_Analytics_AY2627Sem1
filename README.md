# IT5006 — Predicting Delivery Performance and Satisfaction Risk

**Group 10** · Pan Pinyou · Li Yudan · Zhu Gengyu
National University of Singapore · IT5006 Fundamentals of Data Analytics · AY2026/27 Semester 1

Two prediction problems on the **Olist Brazilian E-Commerce** dataset — 99,441
orders across nine relational tables — placed at two different moments in an
order's life, carried from exploratory analysis through modelling to a deployed
application.

| | **Problem 1 — delivery performance** | **Problem 2 — satisfaction risk** |
|---|---|---|
| Moment | `at_checkout` | `at_delivery` |
| Stakeholder | Operations / logistics | Customer experience |
| 1A | **Regression** — lead time in days (MAE **3.90 d**, 41% below a constant, 60% below Olist's own promise) | — |
| 1B / 2 | **Classification** — late vs the promised date (ROC-AUC 0.782, **2.9×** the no-skill PR floor) | **Classification** — `review_score ≤ 2` (ROC-AUC 0.749, PR-AUC 0.414 = **3.75×** its floor) |
| Action | Re-quote the promise, pre-warn the buyer; do *not* expedite | Work the day's deliveries from the top of the risk list to capacity |

**Live application:** **[team10-it5006-olist.streamlit.app](https://team10-it5006-olist.streamlit.app)**
(Streamlit Community Cloud, Python 3.12). It scores single orders and CSV files,
returns a top-K triage list, replays any day of the held-out test window with the
real outcomes revealable, and carries a model card. If it is asleep, press
*"Yes, get this app back up!"* and wait a minute.

## Deliverables

| Item | Location |
|---|---|
| **Final report (Phase 3)** | [`docs/Team10_Phase3_IT5006_AY2627Sem1.pdf`](docs/Team10_Phase3_IT5006_AY2627Sem1.pdf) — standalone: literature, data, EDA, modelling, results, deployment, recommendations; answers both rounds of feedback in Appendix I |
| **Presentation slides** | [`docs/Team10_Phase3_slides.pptx`](docs/Team10_Phase3_slides.pptx) · [`docs/Team10_Phase3_slides.pdf`](docs/Team10_Phase3_slides.pdf) — the 10-minute deck, with speaker notes |
| Earlier reports | [`report/Team10_Phase1_IT5006_AY2627Sem1.pdf`](report/Team10_Phase1_IT5006_AY2627Sem1.pdf), [`report/Team10_Phase2_IT5006_AY2627Sem1.pdf`](report/Team10_Phase2_IT5006_AY2627Sem1.pdf) — kept as submitted, with the corrections noted on their covers |
| LaTeX sources | [`report/*.tex`](report/), [`report/sections/`](report/sections/), [`report/refs.bib`](report/refs.bib) — the folder uploads to Overleaf and compiles as-is |
| Notebooks | [`notebooks/`](notebooks/) — ten, executed with outputs (01–04 exploration, 05–09 modelling, 10 the Phase 3 additions) |
| Application | [`deployment/`](deployment/) — the Streamlit app, with its own [README](deployment/README.md) (usage, CSV contract, deployment steps) |
| Reusable code | [`src/`](src/) · validation checks in [`tests/`](tests/) |
| Fitted models | [`models/`](models/) — under 1 MB, with metadata |
| Documentation | [`docs/`](docs/) — requirement checklist, feedback responses, screenshots, presentation script |

## Quick start

The nine Olist tables and the fitted models are committed, so a clone is
self-contained:

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements-dev.txt
```

```bash
python tests/test_leakage.py     # 56 validation checks
streamlit run deployment/app.py  # the application (dashboard/app.py is the Cloud entry point)
```

To regenerate everything from the raw tables, in this order:

```bash
python src/train_models.py       # fits every model (~40 min); writes models/ and report/results/
python src/capacity_metrics.py   # Precision@K / Recall@K from the saved test predictions (no refit)
python src/lateness_anatomy.py   # weekly / monthly / strike-cohort lateness tables (no refit)
python src/fit_diagnostics.py    # deployed models scored on train / validation / test (no refit)
python src/selection_tables.py   # every candidate on every CV window, fold-to-fold spread, selected hyperparameters
python src/pilot_design.py       # sizing of the outreach pilot
python src/figures_phase2.py     # draws every modelling figure (vector PDF)
python src/phase3_stats.py       # regenerates every number quoted in the final report
python src/report_stats.py       # regenerates every exploratory number (unchanged since Phase 1)
python build_submission.py       # assembles the submission ZIP
```

Two dependency files, and they do **not** overlap — `requirements-dev.txt` pulls
the runtime set in with `-r requirements.txt` rather than restating it:

| File | Adds | Installed by |
|---|---|---|
| `requirements.txt` | streamlit, pandas, numpy, plotly, scikit-learn, scipy, joblib | Streamlit Cloud — exactly what the deployed app imports |
| `requirements-dev.txt` | `-r requirements.txt` **+** matplotlib, Jupyter | you, for the notebooks and figures |

scikit-learn is pinned to the **exact** version the pickles were written with,
because scikit-learn makes no cross-version pickle guarantee and a model that
loads but behaves subtly differently is worse than one that refuses to load.
`modeling.load_model()` checks the version and the app surfaces a warning if it
ever drifts.

## Repository layout

```
├── data/raw/olist/                     the nine source tables (126 MB, tracked)
├── notebooks/
│   ├── 01–04                           exploration: audit, temporal, entities, candidates
│   ├── 05–09                           modelling: features, P1A, P1B, P2, selection & ensembles
│   └── 10_phase3_followups.ipynb       fold-to-fold variation, deployed models, pilot, replay
├── src/
│   ├── config.py                       seeds, windows, leakage regimes, bands, costs
│   ├── data_load.py                    nine-table join + CLEANING_DECISIONS
│   ├── features.py                     leakage-annotated builders, history, the guard
│   ├── modeling.py                     family ladders, preprocessing, CV protocol
│   ├── evaluate.py                     metrics, thresholds, calibration, capacity, rank selection
│   ├── train_models.py                 fits everything; the single source of truth
│   ├── capacity_metrics.py             Precision@K / Recall@K from the saved predictions
│   ├── lateness_anatomy.py             weekly, monthly and strike-cohort lateness tables
│   ├── fit_diagnostics.py              deployed models on train / validation / test
│   ├── selection_tables.py             CV folds per candidate; selected hyperparameters
│   ├── pilot_design.py                 the outreach pilot: break-even and sample size
│   ├── figures_phase2.py               every modelling figure, drawn from the tables
│   ├── phase3_stats.py                 regenerates every number in the final report
│   ├── phase2_stats.py / report_stats.py   the Phase 2 and Phase 1 contracts, unchanged
│   └── viz.py                          shared plot style, vector PDF export
├── deployment/
│   ├── app.py                          the Streamlit application (eight tabs)
│   ├── predict.py                      model serving, batch scoring, triage list, model card
│   ├── replay.py                       replay of the held-out test window, day by day
│   ├── theme.py                        plotly template + CSS design system
│   ├── capture_screenshots.py          regenerates docs/screenshots/ with Playwright
│   └── README.md                       usage, CSV contract, deployment and troubleshooting
├── dashboard/app.py                    Community Cloud entry point: runs deployment/app.py
├── models/                             fitted estimators + metadata + history lookup
├── tests/test_leakage.py               56 validation checks
├── report/
│   ├── *.tex, sections/, refs.bib      Overleaf-ready sources for all three reports
│   ├── results/                        every result table, as CSV
│   ├── figures/                        vector PDFs and the application screenshots
│   └── stats*.txt                      every number quoted in each report
├── docs/                               final report, slides, screenshots, checklists
├── build_submission.py                 assembles the submission ZIP
└── requirements.txt / requirements-dev.txt
```

## What is deployed

The served models are the ones the pre-registered selection rule chose, read
back from the pickles (`models/model_metadata.json` is the index; the test suite
asserts the two agree):

| Problem | Artifact | Estimator as pickled | Decision rule |
|---|---|---|---|
| 1A lead time | `p1a_leadtime.joblib` (381 KB) | **VotingRegressor (A+B)**: LinearRegression + HistGradientBoostingRegressor, weights 1:2 | re-quote with the estimate |
| 1B late flag | `p1b_margin_to_risk.joblib` (1 KB) | one-parameter logistic on (predicted − promised days), fitted on validation | flag at risk ≥ 0.11 |
| 2 low review | `p2_lowreview.joblib` (321 KB) | **StackingClassifier (A+B)**: ℓ2 LogisticRegression + HistGradientBoostingClassifier, logistic meta-learner | flag at risk ≥ 0.19; rank for the top-K |

The single HistGradientBoosting classifier sits within 0.001 PR-AUC of the stack
on the test window; the stack is served because the selection rule was fixed
before any test score was seen, and switching afterwards would be selecting on
the test set. Both artifacts are a few hundred kilobytes.

## Results in brief

**Problem 1A — lead time** (test window, July–August 2018; CV is mean ± fold-to-fold sd):

| Predictor | CV MAE | Val MAE | **Test MAE** | RMSE | R² |
|---|---|---|---|---|---|
| Baseline: mean of the training period | — | — | 6.65 | 7.57 | −0.99 |
| Baseline: Olist's own delivery promise | — | — | 9.79 | 12.16 | −4.13 |
| LinearRegression *(family A, rung 0)* | 5.24 ± 1.00 | 4.97 | 4.18 | 5.64 | −0.10 |
| Ridge *(rung 1)* | 5.20 ± 0.95 | 5.16 | 4.31 | 5.78 | −0.16 |
| DecisionTree *(family B, rung 0)* | 5.54 ± 0.91 | 5.93 | 6.12 | 8.66 | −1.61 |
| RandomForest *(rung 1)* | 5.46 ± 0.90 | 5.51 | 4.96 | 6.17 | −0.32 |
| HistGradientBoosting *(rung 2)* | 5.50 ± 0.81 | 4.80 | 4.14 | 5.52 | −0.06 |
| **VotingRegressor (A+B)** *(family C, deployed)* | 5.32 ± 0.87 | **4.78** | **3.90** | **5.34** | **+0.01** |
| StackingRegressor (A+B) | 5.44 ± 0.84 | 4.79 | 4.12 | 5.43 | −0.02 |

**Problem 1B — late vs the promise.** The flag derived from the regression's
margin against the promised date reaches ROC-AUC 0.782 and 2.9× the no-skill
PR floor; the best classifier trained directly on the binary label reaches
0.584 and 1.2×. Its precision never clears the 0.30 break-even a R$12 expedite
would need, so the recommendation is to re-quote and pre-warn, not to expedite.

**Problem 2 — low review at delivery.** ROC-AUC 0.749, PR-AUC 0.414 (3.75× the
0.110 floor), balanced accuracy 0.681, identical on 12,617 customers never seen
in training. Every candidate's score on every CV window, with the fold-to-fold
spread, is in `report/results/cv_folds.csv` (and on the model card): the three
tree-based rungs are within 0.0004 of one another against a spread of ±0.09, so
selection ranks models within windows rather than averaging across them.

**Capacity view.** Contacting the riskiest 5% of deliveries reaches a would-be
1–2 star reviewer 68% of the time and covers 31% of all such reviews; at ten
contacts a day, 57% and 25%. Under illustrative costs of R$5 per contact and
R$40 per low review the policy is worth about R$1,466 per 1,000 delivered orders
— *if* a contact prevents the review it reaches, which the data cannot show.
`src/pilot_design.py` sizes the randomised pilot that would test it: about six
weeks at ten contacts a day to detect a one-in-four reduction, break-even at
roughly one in five.

## What we found about method

1. **Cross-validation inside a drifting training period reorders the
   candidates**, and its fold-to-fold spread is wider than the gaps between
   them. The 2017 folds run 12–17 day lead times against 8.3 in the test window.
   Selection therefore votes on *rank* across the three most recent
   out-of-sample windows.
2. **A two-month validation period is not enough either**: May 2018 is 8.2%
   late, June 1.4%, the lowest in the series.
3. **Weighting the loss for class imbalance is the wrong lever**: ranking is
   unchanged and mean predicted risk inflates threefold. Fit on the observed
   prior and move the threshold.
4. **Raw PR-AUC is not comparable across periods** with different base rates;
   its lift over the floor is.
5. **Capacity, not the threshold, limits the review flag's value**, so
   Precision@K / Recall@K are the metrics that describe the team's day.
6. **Lateness is driven by shocks no order feature can see**: Black Friday, a
   carrier-side slowdown in February–March 2018 that coincided with a tightened
   promise, and the May 2018 truckers' strike, visible only at weekly
   resolution and absorbed into a longer promised window.

## Leakage: one rule per prediction moment

`assert_no_leakage(features, regime)` enforces the right set and is called
before any model is fitted:

| Regime | Situation | Forbidden |
|---|---|---|
| `at_checkout` | order just placed | delivery dates and anything derived from them, review score and its timestamps, `order_approved_at`, `shipping_limit_date`, `order_status` |
| `at_delivery` | parcel arrived, review not yet written | review score and its timestamps only |

Adding one forbidden column to the late-delivery model takes ROC-AUC from
**0.589 to 1.000**. The eight track-record features are as-of aggregates over
orders that had *closed* before the order was placed; `tests/test_leakage.py`
rebuilds one of them from first principles for 200 random orders and requires
exact agreement.

## Responses to the evaluators' feedback

Both rounds of feedback are answered point by point in Appendix I of the final
report and in [`docs/feedback_responses.md`](docs/feedback_responses.md). In
short, after Phase 1: the perceived-quality vs rating distinction, literature
for the model rungs and the features, Precision@K, the corrected lateness
anatomy, the Figure 5a / growth / correlation-matrix corrections, the
stakeholder levers, and the redeployed dashboard. After Phase 2: fold-to-fold
variation for every candidate (Table 5, Appendix G, the model card), one
consistent statement of the deployed Problem 2 model backed by a test, every
figure referenced and the selected hyperparameters printed (Appendix D), and the
outreach-benefit assumption stated as a limitation with a sized pilot.

## Reproducibility

- `SEED = 42` everywhere; both requirements files pin exact versions on
  Python ≤ 3.12, the interpreter every reported number was produced on.
- `load_table()` asserts the row count of all nine source tables, and
  `build_orders()` asserts the grain after joining.
- `python src/phase3_stats.py` regenerates every numeric claim in the final
  report into `report/stats_phase3.txt`; `python src/report_stats.py` still
  reproduces every Phase 1 number byte-for-byte.
- Every figure is generated from the same CSVs the tables come from.
- Model artifacts record the scikit-learn version that wrote them; the loader
  checks it; the test suite checks that the artifacts, the metadata, the
  hyperparameter table and the documentation name the same deployed models.

## Deploying and keeping the app reachable

The app deploys from `main` with main file path `dashboard/app.py` on
**Python 3.12** (*App settings → General*). Keep the Python version at 3.12 or
lower: `numpy==1.26.4` publishes wheels only up to `cp312`; on 3.13+ pip
compiles numpy from source, the build fails, and the app hangs on *"Your app is
in the oven"*. Community Cloud sleeps an app after twelve hours without visitors
and shows a *"Yes, get this app back up!"* button; any visitor can press it.
Open the link once before a grading window or a presentation. Full deployment
notes, the CSV contract and the error-handling inventory are in
[`deployment/README.md`](deployment/README.md).
