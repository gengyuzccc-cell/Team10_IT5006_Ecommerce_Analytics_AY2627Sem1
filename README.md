# IT5006 — Predicting Delivery Performance and Satisfaction Risk

**Group 10** · Pan Pinyou · Li Yudan · Zhu Gengyu
National University of Singapore · IT5006 Fundamentals of Data Analytics

Two prediction problems on the **Olist Brazilian E-Commerce** dataset — 99,441
orders across nine relational tables — placed at two different moments in an
order's life, plus the exploratory analysis they grew out of and a deployed app
that serves them.

**Deliverables**

| Item | Location |
|---|---|
| Phase 2 report (PDF) | [`report/Team10_Phase2_IT5006_AY2627Sem1.pdf`](report/) — modelling, evaluation, deployment; revised after the Milestone 1 feedback (see below) |
| Phase 1 report (PDF) | [`report/Team10_Phase1_IT5006_AY2627Sem1.pdf`](report/) — literature review and EDA; revised October 2026 to correct the facts the Milestone 1 feedback named |
| LaTeX sources | [`report/*.tex`](report/), [`report/sections/`](report/sections/), [`report/refs.bib`](report/refs.bib) — the folder uploads to Overleaf and compiles as-is |
| Notebooks | [`notebooks/*.ipynb`](notebooks/) — nine, executed with outputs (01–04 Phase 1, 05–09 Phase 2) |
| Dashboard | **[it5006-group-project-10.streamlit.app](https://it5006-group-project-10.streamlit.app)** — seven tabs, including live model scoring |
| Reusable code | [`src/`](src/) · validation checks in [`tests/`](tests/) |
| Fitted models | [`models/`](models/) — under 2 MB, with metadata |

---

## Quick start

The nine Olist tables are committed, so a clone is self-contained:

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements-dev.txt
```

```bash
python tests/test_leakage.py     # 41 validation checks
python src/train_models.py       # fits everything, writes models/ and report/results/
python src/capacity_metrics.py   # Precision@K / Recall@K from the saved test predictions (no refit)
python src/lateness_anatomy.py   # weekly / monthly / strike-cohort lateness tables (no refit)
python src/figures_phase2.py     # draws every report figure
python src/phase2_stats.py       # regenerates every number quoted in the Phase 2 report
```

```bash
streamlit run dashboard/app.py
```

Two dependency files, and they do **not** overlap — `requirements-dev.txt` pulls
the runtime set in with `-r requirements.txt` rather than restating it:

| File | Adds | Installed by |
|---|---|---|
| `requirements.txt` | streamlit, pandas, numpy, plotly, **scikit-learn, scipy, joblib** | Streamlit Cloud — exactly what the deployed app imports |
| `requirements-dev.txt` | `-r requirements.txt` **+** matplotlib, Jupyter | you, for the notebooks and figures |

Phase 2 moved scikit-learn into the runtime file: the dashboard no longer only
draws charts, it unpickles `models/*.joblib` and scores orders. It is pinned to
the **exact** version those pickles were written with, because scikit-learn makes
no cross-version pickle guarantee and a model that loads but behaves subtly
differently is worse than one that refuses to load. `modeling.load_model()`
checks the version and the app surfaces a warning if it ever drifts.

---

## The two problems

Phase 1's strongest finding was that a late delivery raises the 1–2 star review
rate from 9.2% to 54.0% — a 5.9× increase, monotone in how late the parcel is.
Phase 2 turns that into two predictions at two prediction moments, forming one
operational chain: operations acts before the parcel ships, customer experience
acts after it arrives and before the customer writes anything.

| | **Problem 1 — delivery performance** | **Problem 2 — satisfaction risk** |
|---|---|---|
| Moment | `at_checkout` | `at_delivery` |
| Stakeholder | Operations / logistics | Customer experience |
| 1A | **Regression** — lead time in days | — |
| 1B | **Classification** — late vs the promised date | **Classification** — `review_score ≤ 2` |
| Action | Re-quote, expedite, warn the buyer | Reach out before the review lands |

### Problem 1A — lead time (regression, at checkout)

| Predictor | CV MAE | Val MAE | **Test MAE** | RMSE | R² |
|---|---|---|---|---|---|
| Baseline: mean of the training period | — | — | 6.65 | 7.57 | −0.99 |
| Baseline: Olist's own delivery promise | — | — | 9.79 | 12.16 | −4.13 |
| LinearRegression *(family A, rung 0)* | 5.24 | 4.97 | 4.18 | 5.64 | −0.10 |
| Ridge *(rung 1)* | 5.20 | 5.16 | 4.31 | 5.78 | −0.16 |
| DecisionTree *(family B, rung 0)* | 5.54 | 5.93 | 6.12 | 8.66 | −1.61 |
| RandomForest *(rung 1)* | 5.46 | 5.51 | 4.96 | 6.17 | −0.32 |
| HistGradientBoosting *(rung 2)* | 5.50 | 4.80 | 4.14 | 5.52 | −0.06 |
| **VotingRegressor (A+B)** *(family C)* | 5.32 | **4.78** | **3.90** | **5.34** | **+0.01** |
| StackingRegressor (A+B) | 5.44 | 4.79 | 4.12 | 5.43 | −0.02 |

**41% below a constant fitted on the past, 60% below the platform's own promise.**
Three things in that table are worth more than the winner:

- **Regularisation does not pay.** Ridge is *worse* than plain least squares on
  every window after cross-validation. Within the linear family, rung 1 fails to
  beat rung 0 and we say so.
- **Complexity does pay inside the tree family**, monotonically: 5.93 → 5.51 →
  4.80 on validation.
- **Only the ensemble has a positive R².** Everything else is negative — the
  signature of a level shift, not of a useless model, which is why MAE leads.

### Problem 1B — late vs the promise (classification, at checkout)

The brief's *dual framing*: one underlying quantity, two outputs. Rather than
train a second model, we threshold the regression's own answer against the
promised date and map that margin to a calibrated probability with a
one-parameter logistic fitted on the validation period.

That derived route beats every purpose-built classifier by a wide margin. The
binary label is 7.4% positive and throws away *how* late an order was; the
continuous target keeps it.

### Problem 2 — low review (classification, at delivery)

Predicted once the parcel has landed and before the review is written. Phase 1
measured what the moment is worth: the same target asked at checkout reaches
PR-AUC 0.212, and at delivery 0.402.

---

## What we found about method

Four results that outlived the models themselves.

1. **Cross-validation inside a drifting training period reorders the
   candidates.** The 2017 folds run 12–17 day lead times against 8.3 in the test
   window, so every model's CV error sits a day high — and the *ordering* does
   not survive. CV ranks Ridge above the boosted trees; every later window ranks
   them the other way round.
2. **A two-month validation period is not enough either.** Its two months pull
   opposite ways — 8.2% late in May 2018, 1.4% in June, the lowest in the series.
   On that window alone the late-order flag would have been a direct classifier
   at ROC-AUC 0.584 in preference to a derived route at 0.782. Selection
   therefore votes on **rank across the three most recent out-of-sample
   windows**.
3. **Weighting the loss for class imbalance is the wrong lever.** It leaves
   ranking essentially unchanged and badly miscalibrates: mean predicted risk
   inflates roughly threefold against the actual rate. We fit on the observed
   prior and move the decision threshold instead — which is what Phase 1's
   literature review predicted and promised to test.
4. **Raw PR-AUC is not comparable across periods**, because its no-skill floor
   *is* the positive rate and that rate falls from 15.8% to 11.0%. Its **lift**
   over the floor is comparable, and turns out to be the stable quantity.
5. **Capacity, not the threshold, is what limits the review flag's value.** A
   customer-experience team works a ranked list up to its capacity, so the
   metric that describes its day is Precision@K / Recall@K, not PR-AUC. At a
   capacity of 5% of deliveries (about ten contacts a day) two contacts in
   three reach a customer who would have left a 1–2 star review, covering 31%
   of all such reviews; precision stays above the R$5 : R$40 break-even at
   every capacity tried. The derived late flag beats the direct classifier at
   every K and still never clears the 0.30 break-even an expensive expedite
   would need. `src/capacity_metrics.py` computes all of this from the saved
   test predictions; nothing is refitted.

---

## Changes after the Milestone 1 feedback

The Milestone 1 feedback is answered point by point in Appendix H of the Phase 2
report. In short:

| Feedback | What changed |
|---|---|
| Survey-based perceived quality and platform ratings treated as one construct | Phase 2 §1 separates them, cites the self-selection literature, and defines the target as the platform rating |
| Model choices cited original algorithm papers, no within-tree-family justification, HistGB unexplained | Phase 2 §3.1 argues the DT → RF → HistGB rungs from the empirical comparisons and from e-commerce applications (including three studies on this dataset), and states why HistGradientBoosting |
| Features not supported by literature | Phase 2 §2.2 cites the support for each feature group and reuses Phase 1's correlation matrix as the empirical screen |
| Evaluate Problem 2 under limited capacity (Precision@K, Recall@K) | New Phase 2 §4.4, Table 6, Figure 11; the operating point in §5.2 is now a capacity; the dashboard's batch scorer returns a top-K triage list |
| Figure 3 attributed the late-rate peaks to the May 2018 strike; the strike needed a source | `src/lateness_anatomy.py`, Phase 2 Appendix G and Figure 12: the peaks are February–March 2018 (a carrier-side slowdown that coincided with a tightened promise); the strike is visible only at weekly resolution and was absorbed into a longer promised window; cited |
| Figure 5a omitted São Paulo; "roughly triples"; Figure 8 never referenced | Phase 1 corrected and rebuilt: the figure had taken the first eight states alphabetically (notebook 03); growth is about eight-fold; the correlation matrix is now discussed |
| Candidate 1 named a stakeholder without saying how the team would use the prediction | Phase 2 Table 2 and §5.3: four operations levers ordered by cost, each with its trigger |
| Dashboard inaccessible | The GitHub account the app deployed from was suspended; redeployed from this repository under a new URL (cover pages, link above) |

Every number added by these changes is regenerated by `src/phase2_stats.py` and
`src/report_stats.py`; the submitted Phase 1 numbers reproduce unchanged.

### A Phase 1 claim, corrected

Phase 1 reported that plain least squares was unusable — "the design matrix is
rank-deficient, so its coefficients explode". Re-running it four ways confirms
the observation and corrects the conclusion:

| Encoding | rank / cols | max \|β\| | CV MAE | test MAE |
|---|---|---|---|---|
| OLS, full one-hot block | 101 / 105 | 2.7×10¹³ | 3.9×10¹¹ | 4.181 |
| OLS, one level dropped | 101 / 101 | 14.7 | 5.242 | 4.181 |
| Ridge, full one-hot block | 101 / 105 | 9.82 | 5.241 | 4.182 |
| Ridge, one level dropped | 101 / 101 | 12.5 | 5.240 | 4.183 |

The deficiency is exactly four — one per categorical, each one-hot block being
collinear with the intercept. The *predictions* were always fine; the
*coefficients* were not, and the cancellation holding them together fails on one
of five folds. A single train/test split would never have shown it. Once the
design is full rank, Ridge and OLS agree to three decimals: regularisation buys
numerical stability here, not accuracy.

---

## Leakage: one rule per prediction moment

`assert_no_leakage(features, regime)` enforces the right set, and is called
before any model is fitted:

| Regime | Situation | Forbidden |
|---|---|---|
| `at_checkout` | order just placed | delivery dates and anything derived from them, review score and its timestamps, `order_approved_at`, `shipping_limit_date`, `order_status` |
| `at_delivery` | parcel arrived, review not yet written | review score and its timestamps only |

Adding one forbidden column to the Phase 1 late-delivery model takes ROC-AUC
from **0.589 to 1.000** — the model reading the answer rather than predicting it.

### The track-record features, and why they are tested rather than reviewed

Phase 2 adds eight features describing a seller's, a route's and the platform's
past performance. They are the strongest thing a marketplace knows at checkout
and the easiest place here to leak, because of one distinction:

> **"Past" means the earlier order had *closed* before this one was placed** —
> not merely that it was placed earlier.

A parcel bought last week that has not yet arrived has no lead time yet. Ordering
the history by *purchase* time would let every order average in outcomes that had
not happened — and it would be invisible in the metrics, because it would simply
look like a better model. Every aggregate is an as-of join against the moment the
earlier outcome became observable.

`tests/test_leakage.py` therefore rebuilds `hist_seller_lead_days` from first
principles for 200 random orders — filtering every earlier order by hand — and
requires exact agreement. A second check exploits the fact that a feature built
from the future would track the *current* month better than the previous one: the
platform tracker correlates 0.80 with the previous month against 0.43 with the
current one.

---

## Repository layout

```
├── data/raw/olist/                     the nine source tables (126 MB, tracked)
├── notebooks/
│   ├── 01–04                           Phase 1: audit, temporal, entities, candidates
│   ├── 05_problem_definition_and_features.ipynb
│   ├── 06_problem1_regression_leadtime.ipynb
│   ├── 07_problem1_classification_late.ipynb
│   ├── 08_problem2_classification_lowreview.ipynb
│   └── 09_selection_ensembles_interpretation.ipynb
├── src/
│   ├── config.py                       seeds, windows, leakage regimes, bands, costs
│   ├── data_load.py                    nine-table join + CLEANING_DECISIONS
│   ├── features.py                     leakage-annotated builders, history, the guard
│   ├── modeling.py                     family ladders, preprocessing, CV protocol
│   ├── evaluate.py                     metrics, thresholds, calibration, rank selection
│   ├── train_models.py                 fits everything; the single source of truth
│   ├── capacity_metrics.py             Precision@K / Recall@K from the saved predictions
│   ├── lateness_anatomy.py             weekly, monthly and strike-cohort lateness tables
│   ├── figures_phase2.py               every report figure, drawn from the tables
│   ├── phase2_stats.py                 regenerates every number in the Phase 2 report
│   ├── report_stats.py                 regenerates every number in the Phase 1 report
│   └── viz.py                          shared plot style, vector PDF export
├── tests/test_leakage.py               41 validation checks
├── dashboard/
│   ├── app.py                          seven tabs
│   ├── predict.py                      model serving, batch scoring, model card
│   └── theme.py                        plotly template + CSS design system
├── models/                             fitted estimators + metadata + history lookup
└── report/
    ├── *.tex, sections/, refs.bib      Overleaf-ready sources
    ├── results/                        every result table, as CSV
    ├── figures/                        vector PDFs, generated
    └── phase2_compliance.md            requirement-by-requirement checklist
```

## Reproducibility

- `SEED = 42` everywhere; both requirements files pin exact versions on
  Python ≤ 3.12, the interpreter every reported number was produced on.
- `load_table()` asserts the row count of all nine source tables, and
  `build_orders()` asserts the grain after joining.
- `python src/report_stats.py` still reproduces **every Phase 1 number
  byte-for-byte** after the Phase 2 additions — asserted in the test suite, since
  `build_features(with_history=False)` is the default for exactly that reason.
- `python src/phase2_stats.py` regenerates every numeric claim in the Phase 2
  report into `report/stats_phase2.txt`. Diff it against the manuscript.
- Every figure is generated as vector PDF from the same CSVs the tables come
  from, so a figure cannot disagree with a table.
- Model artifacts record the scikit-learn version that wrote them; the loader
  checks it.

### Deploying

Live at **[it5006-group-project-10.streamlit.app](https://it5006-group-project-10.streamlit.app)**,
built from `main` / `dashboard/app.py` on **Python 3.12**.

> **Keep the Python version at 3.12 or lower** (*App settings → General*).
> `numpy==1.26.4` publishes wheels only up to `cp312`. On Python 3.13+ pip finds
> no wheel and compiles numpy from source; the build dies and the app then hangs
> on *"Your app is in the oven"* indefinitely, with the real error buried in the
> build log rather than shown in the app.

The deployed estimators are the histogram-boosted variants, whose pickles are a
few hundred kilobytes. A random forest of comparable accuracy pickles to
24–30 MB, which is hostile to a shared container — so the size constraint is
recorded as a deployment decision rather than discovered later. Models are loaded
lazily, so a visitor who only browses the exploratory tabs never pays for
unpickling one.

### Keeping the dashboard reachable

The Milestone 1 dashboard became inaccessible because the GitHub account it was
deployed from was suspended, which took the Streamlit Community Cloud app down
with it. The app now deploys from this repository. Two things keep it reachable:

- **Community Cloud sleeps an app after twelve hours without visitors** and
  shows a "get this app back up" button. Any visitor can press it; the app is
  back in about a minute. Open the link once before a grading window.
- **Python 3.12** must stay selected in the app's settings (see above); on
  3.13 the pinned numpy has no wheel and the build never finishes.
