# Team 10 — A delivery date customers can rely on, and help when it slips

**Team 10** · Pan Pinyou · Li Yudan · Zhu Gengyu
National University of Singapore · IT5006 Fundamentals of Data Analytics · AY2026/27 Semester 1

Every Olist order carries a promise to a customer: the delivery date shown at
checkout. When it is broken, 54% of customers rate the order 1–2 stars (9% when
it is kept), and most of them are asked for that review while still waiting for
the parcel. This project predicts, at checkout, how long an order will take and
whether it will miss its date, and, at delivery, which customers are about to
rate their order 1–2 stars, so that Olist's operations, customer-care and
customer-experience teams can quote dates customers can rely on, tell customers
first when a date is missed, and help while it can still be put right. The data
are the public **Olist Brazilian E-Commerce** tables: 99,441 orders across nine
relational tables, carried from exploratory analysis through modelling to a
deployed application.

| | |
|---|---|
| **Live application** | **[team10-it5006-olist.streamlit.app](https://team10-it5006-olist.streamlit.app)** (Streamlit Community Cloud, Python 3.12). If it is asleep, press *"Yes, get this app back up!"* and wait a minute. |
| **Final report** | [`docs/Team10_Phase3_IT5006_AY2627Sem1.pdf`](docs/Team10_Phase3_IT5006_AY2627Sem1.pdf), revised after the presentation of 5 November 2026 |
| **Slides** | [`docs/Team10_Phase3_slides.pptx`](docs/Team10_Phase3_slides.pptx) · [`.pdf`](docs/Team10_Phase3_slides.pdf), as locked on 4 November 2026 and submitted unchanged; report Appendix P sources every number on them and corrects the slides where they were wrong |
| **What changed after the presentation** | [`docs/feedback_responses.md`](docs/feedback_responses.md) and report Appendix I (Table 22) |

| | **Problem 1 — delivery performance** | **Problem 2 — satisfaction risk** |
|---|---|---|
| For the customer | A date to rely on, and to hear first when it slips | Help while a disappointing order can still be put right |
| Moment | `at_checkout` | `at_delivery` |
| Who acts | Operations / logistics (the date); customer care (the message when the promised day ends, no model) | Customer experience |
| 1A | **Regression** — lead time in days (MAE **3.90 d**, 41% below a constant, 60% below Olist's own promise) | — |
| 1B / 2 | **Classification** — late vs the promised date (ROC-AUC 0.782, **2.9×** the no-skill PR floor) | **Classification** — `review_score ≤ 2` (ROC-AUC 0.749, PR-AUC 0.414 = **3.75×** its floor) |
| Action | Quote the forecast plus an 8-day margin chosen on validation; tell the customer on the promised date; do *not* expedite | Each morning, contact the riskiest deliveries not yet rated, with a fix to offer |

The application opens on one customer's order, from checkout to review
(*Customer journey*); scores single orders and CSV files and shows the date it
would quote (*Predict*); replays any day of the held-out test window, first
blind and then with what actually happened
(*Replay a day*); and carries a model card that labels each number observed,
assumed, derived or planned and takes both assumed costs as inputs.

## What the held-out window shows for customers (30 June – 30 August 2018)

Counted on the test window, on which no model, threshold or margin was fitted
or chosen. Each count names its population: the 12,927 orders placed, the
12,630 delivered, or the 12,855 reviewed.

- **Broken dates: 74 → 25 in every 1,000 delivered orders.** Olist's date was
  missed for 936 of the 12,630 delivered orders. The forecast plus an 8-day
  margin (the smallest margin, in tenths of a day, whose broken-promise rate on the
  validation months is no worse than Olist's own) would have been missed for
  318, for a quoted wait 1.1 days longer on average (18.6 against 17.5 days).
  Quoted bare, the forecast would have been missed for 31.9% of customers.
- **Hearing first when a date slips.** 898 of the 12,927 orders placed (69.5 in
  1,000) still had no parcel when the promised day ended; 297 of them never
  arrived. Of the 804 such customers who rated the order only after that day
  ended, 440 (55%) gave it 1–2 stars: 31% of the 1,418 customers
  who rated 1–2 stars in the window. The calendar finds them with no model, so
  customer care can write that day (an apology, the status, a new date or a
  refund) and ask for the review once the order is complete. Every customer is
  still asked for a review; nothing is filtered, delayed in publication or
  rewarded.
- **Help at delivery.** Of each morning's ten riskiest deliveries whose customer
  has not yet rated the order, 26 in 100 go on to rate it 1–2 stars, against 8
  in 100 at random (3.2×; 546 contacts over 63 delivery days). This is the list
  for customers whose parcel arrived but disappointed.
- **What the data cannot show:** whether a contact changes a rating, because the
  tables record no contact other than the survey. A randomised pilot of five to
  seven weeks, one arm per contact, would measure it (report Section 9.4).

## Deliverables

| Item | Location |
|---|---|
| **Final report (Phase 3)** | [`docs/Team10_Phase3_IT5006_AY2627Sem1.pdf`](docs/Team10_Phase3_IT5006_AY2627Sem1.pdf) — standalone: literature, data, EDA, modelling, results, deployment, recommendations; revised after the presentation of 5 November 2026; answers the Phase 1, Phase 2 and presentation feedback in Appendix I and sources every number on the slides in Appendix P |
| **Presentation slides** | [`docs/Team10_Phase3_slides.pptx`](docs/Team10_Phase3_slides.pptx) · [`docs/Team10_Phase3_slides.pdf`](docs/Team10_Phase3_slides.pdf) — the 10-minute story deck (13 slides) as locked on 4 November 2026 and submitted unchanged, with clickable chapter navigation, timed speaker notes and links and QR codes to the live app; present from the PPTX. Script, demo path and predicted questions as delivered: [`docs/presentation_script.md`](docs/presentation_script.md) |
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
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

```bash
python tests/test_leakage.py     # 89 validation checks
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
python src/review_timing.py      # when the review is written vs the delivery; reachable share; replay flags
python src/customer_view.py      # the customer's view: outcomes per 1,000 orders, contact lists, date margin, complaints, pilot arms, Box 1 (no refit)
python src/slide_provenance.py   # the kind and source of every number on the locked slides (report Appendix P)
python src/figures_phase2.py     # draws every modelling figure (vector PDF)
python src/phase3_stats.py       # regenerates every number quoted in the final report
python src/report_stats.py       # regenerates every exploratory number (unchanged since Phase 1)
python tests/test_leakage.py     # the validation checks
python build_submission.py       # assembles the submission ZIP, only if every submission guard passes
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
│   ├── config.py                       seeds, windows, leakage regimes, bands, assumed costs
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
│   ├── review_timing.py                review written before or after the parcel arrived
│   ├── customer_view.py                what happened to the test window's customers, who each contact reaches, the date margin
│   ├── slide_provenance.py             the kind and source of every number on the locked slides
│   ├── figures_phase2.py               every modelling figure, drawn from the tables
│   ├── phase3_stats.py                 regenerates every number in the final report
│   ├── phase2_stats.py / report_stats.py   the Phase 2 and Phase 1 contracts, unchanged
│   └── viz.py                          shared plot style, vector PDF export
├── deployment/
│   ├── app.py                          the Streamlit application (nine tabs)
│   ├── journey.py                      Customer journey: one order from checkout to review
│   ├── predict.py                      model serving, batch scoring, triage list, model card
│   ├── replay.py                       replay of the held-out test window, day by day, blind then revealed
│   ├── theme.py                        plotly template + CSS design system
│   ├── capture_screenshots.py          regenerates docs/screenshots/ with Playwright
│   └── README.md                       usage, CSV contract, deployment and troubleshooting
├── dashboard/app.py                    Community Cloud entry point: runs deployment/app.py
├── models/                             fitted estimators + metadata + history lookup
├── tests/test_leakage.py               89 validation checks
├── report/
│   ├── *.tex, sections/, refs.bib      Overleaf-ready sources for all three reports
│   ├── results/                        every result table, as CSV
│   ├── figures/                        vector PDFs and the application screenshots
│   └── stats*.txt                      every number quoted in each report
├── docs/                               final report, slides, screenshots, checklists
├── build_submission.py                 assembles the submission ZIP behind its guards
└── requirements.txt / requirements-dev.txt
```

## What is deployed

The served lead-time and review models are the ones the pre-registered selection
rule chose (the late flag is the regression's own margin, kept although the rule
ranked a direct classifier first: report Section 7.2), read back from the pickles (`models/model_metadata.json` is the index; the test suite
asserts the two agree):

| Problem | Artifact | Estimator as pickled | Decision rule |
|---|---|---|---|
| 1A lead time | `p1a_leadtime.joblib` (381 KB) | **VotingRegressor (A+B)**: LinearRegression + HistGradientBoostingRegressor, weights 1:2 | quote the estimate plus an 8-day margin chosen on validation, rounded up to a calendar day |
| 1B late flag | `p1b_margin_to_risk.joblib` (1 KB) | two-parameter logistic on (predicted − promised days), fitted on validation | flag at risk ≥ 0.11 |
| 2 low review | `p2_lowreview.joblib` (321 KB) | **StackingClassifier (A+B)**: ℓ2 LogisticRegression + HistGradientBoostingClassifier, logistic meta-learner | flag at risk ≥ 0.19; rank for the top-K |

The single HistGradientBoosting classifier sits within 0.001 PR-AUC of the stack
on the test window; the stack is served because the selection rule was fixed
before any test score was seen, and switching afterwards would be selecting on
the test set. Both artifacts are a few hundred kilobytes. Nothing was refitted
after the presentation: the margin and every customer count come from the saved
models and their saved predictions.

## Results in brief

**Problem 1A — lead time** (test window: orders placed 30 June to 30 August 2018; CV is mean ± fold-to-fold sd):

| Predictor | CV MAE | Val MAE | **Test MAE** | RMSE | R² |
|---|---|---|---|---|---|
| Baseline: mean of the training period | — | — | 6.65 | 7.57 | −0.99 |
| Baseline: Olist's own delivery promise | — | — | 9.79 | 12.16 | −4.13 |
| LinearRegression *(family A, rung 0)* | 5.24 ± 1.00 | 4.96 | 4.18 | 5.64 | −0.10 |
| Ridge *(rung 1)* | 5.20 ± 0.95 | 5.16 | 4.31 | 5.77 | −0.16 |
| DecisionTree *(family B, rung 0)* | 5.54 ± 0.91 | 5.93 | 6.12 | 8.66 | −1.61 |
| RandomForest *(rung 1)* | 5.46 ± 0.90 | 5.51 | 4.96 | 6.17 | −0.32 |
| HistGradientBoosting *(rung 2)* | 5.49 ± 0.81 | 4.80 | 4.14 | 5.52 | −0.06 |
| **VotingRegressor (A+B)** *(family C, deployed)* | 5.31 ± 0.87 | **4.78** | **3.90** | **5.34** | **+0.01** |
| StackingRegressor (A+B) | 5.44 ± 0.84 | 4.79 | 4.12 | 5.43 | −0.02 |

The forecast estimates the typical delivery, so the recommendation quotes the
forecast plus the 8-day margin above, not the forecast alone.

**Problem 1B — late vs the promise.** The flag derived from the regression's
margin against the promised date reaches ROC-AUC 0.782 and 2.9× the no-skill
PR floor; the best classifier trained directly on the binary label reaches
0.584 and 1.2×. The derived flag's precision never clears the 0.30 break-even a
R$12 expedite would need, so the recommendation is a better date, not expediting. A warning at
the 11% threshold would reach 5,647 of the 12,630 delivered test orders, 4,874
of them on time, so if Olist keeps its own date a warning belongs only on the
riskiest tenth, of whom one in four (0.258) was late.

**Problem 2 — low review at delivery.** ROC-AUC 0.749, PR-AUC 0.414 (3.75× the
0.110 floor), balanced accuracy 0.681, PR-AUC 0.415 on the 12,617 test orders of customers never seen
in training. On the 12,571 delivered parcels alone, ROC-AUC 0.711 and PR-AUC
0.333 against a 0.096 floor; the other 284 reviewed orders never arrived. Every
candidate's score on every CV window, with the fold-to-fold spread, is in
`report/results/cv_folds.csv` (and on the model card): the forest, the boosted
model and the stack are within 0.0004 of one another against a spread of ±0.09,
so selection ranks models within windows rather than averaging across them.

**Who the review-risk list reaches.** Ranked over all reviewed test orders, the
riskiest 5% hold a customer who went on to rate the order 1–2 stars 68% of the
time. The slides' list, each purchase day's top ten, is right 57.2% of the time,
but per 100 listed customers 26.4 rated an order that never arrived and 22.0 had
already written while waiting for a late parcel; only 8.8 could still be reached
when the parcel came. Built as a team would work it
(each morning, the previous day's deliveries whose customer has not yet rated
the order), the top ten a day reach 26 in 100 (3.2× chance). Costs are
assumptions (R$5 a contact, R$40 for a 1–2 star rating a contact turns around;
Olist publishes no costs): the delivery-day call pays only if it turns around
about one in two of those customers, the promised-date message about one in
four. The R$775 on the slides belongs to the threshold-0.11 list; the
ten-a-day list is worth R$433 per 1,000 reviewed orders leaving out those who
rated a late parcel before it arrived (R$853 counting all), and only if every
contact works.
`report/results/customer_value_sensitivity.csv` varies both costs and the share
a contact turns around.

**When the review is written.** Olist's review table records when the survey was
sent and when it was answered. For a late parcel the survey goes out once the
promised date has passed, and 87% of its 1–2 star reviews are written before the
parcel arrives (a median 4.5 days before); on-time parcels, which produce two
thirds of all low reviews, are reviewed a median 2.1 days after delivery
(delivered, reviewed orders of the analysis window). On the slides' ten-a-day
list, 57 of every 100 customers listed went on to rate 1–2 stars: 26 whose
order never arrived, 22 who had already written while waiting for a late parcel,
and about 9 a delivery-day call could still reach. Customers let down by a late
or missing parcel are reached when their promised day ends instead
(`src/review_timing.py`, `src/customer_view.py`; report Appendices M and Q).

## Where the numbers come from

The numbers in this project are of four kinds, and Section 9 of the report and
the application's model card label each figure by its kind where it is used:

- **Observed**: counted in the nine Olist tables, on the held-out test window
  unless another window is named (rates, counts, precision).
- **Assumed**: no data source. Olist publishes no costs, so R$5 a contact, R$40
  for a 1–2 star rating a contact turns around and R$12 to expedite are inputs
  we chose (`src/config.py`, `COSTS`); `report/results/customer_value_sensitivity.csv`
  shows what changes if they are wrong.
- **Derived**: arithmetic on the other two (break-even shares, reais per 1,000
  reviewed orders), with the formula stated.
- **Planned**: the pilot that would measure what the data cannot (sizes per arm,
  weeks).

`python src/phase3_stats.py` writes every number the final report quotes into
`report/stats_phase3.txt`, read from the tables in `report/results/*.csv`. The
slides were locked on 4 November 2026 and are submitted unchanged:
`src/slide_provenance.py` lists every number on all 13 slides in
`report/results/slide_provenance.csv` (105 rows: 65 observed, 15 derived, 7
assumed, 6 planned, 4 with no data source, 7 about wording, 1 from the
literature; all 88 numeric rows recompute and match), and report Appendix P
prints it with a correction wherever a slide's number or wording was wrong.

| Number | Kind | File (`report/results/` unless stated) |
|---|---|---|
| 54% of customers rate the order 1–2 stars when the date is broken, 9% when it is kept | observed (delivered, reviewed orders, 2017–2018) | `review_timing.csv`; `report/stats_phase3.txt` (`tim_late_low_pct`, `tim_ontime_low_pct`) |
| Dates missed for 936 → 318 of 12,630 delivered orders with an 8-day margin chosen on validation | observed | `customer_promise_rules.csv` |
| 898 of 12,927 orders placed had no parcel when the promised day ended | observed | `customer_outcomes_test.csv` |
| 26 in 100 of each morning's top ten went on to rate 1–2 stars (8 at random) | observed | `customer_contact_lists.csv` |
| R$5 a contact; R$40 for a 1–2 star rating a contact turns around | assumed | `src/config.py` (`COSTS`) |
| A contact pays if it turns around 1 in 4.4 (promised-date message) or 1 in 2.1 (delivery-day call) | derived: 5 / (40 × precision) | `customer_value.csv` |
| R$433 to R$1,466 per 1,000 reviewed orders on the slides | derived, every contact assumed to work | `customer_value.csv`, `customer_value_decomposition.csv` |
| 209 customers per arm for the promised-date message, 159 for the delivery-day call; five to seven weeks | planned | `customer_pilot_arms.csv` |

## What we found about method

1. **Cross-validation inside a drifting training period reorders the
   candidates**, and its fold-to-fold spread is wider than the gaps between
   them. The CV folds run 11.5–16.5 day lead times against 8.3 in the test window.
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
7. **When the customer reviews decided more than which model to use.** Most
   customers let down by a late or missing parcel rated the order before it came
   (77% of late parcels later rated 1–2 stars in the test window, 87% in the
   analysis window), so they are reached when the promised day ends, without a
   model, and the model's list serves the customers whose parcel arrived but
   disappointed.

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

All three rounds of feedback are answered point by point in Appendix I of the
final report and in [`docs/feedback_responses.md`](docs/feedback_responses.md).
In short, after Phase 1: the perceived-quality vs rating distinction, literature
for the model rungs and the features, Precision@K, the corrected lateness
anatomy, the Figure 5a / growth / correlation-matrix corrections, the
stakeholder levers, and the redeployed dashboard. After Phase 2: fold-to-fold
variation for every candidate (Table 5, Appendix G, the model card), one
consistent statement of the deployed Problem 2 model backed by a test, every
figure referenced and the selected hyperparameters printed (Appendix D), and the
outreach-benefit assumption stated as a limitation with a sized pilot. After the
presentation of 5 November 2026 (Appendix I, Table 22, and Appendix P): the
report now starts from the customer, with one customer's order opening Section 1
(Box 1), each problem stated with what that customer needs, and the customers
each step reaches counted from the data (Table 9) before any reais; the figures behind
each recommendation are labelled observed, assumed or derived
(Table 8); every number on the unchanged slides is sourced in Appendix P; and
the pilot is sized on the customers each contact can still reach.

## Reproducibility

- `SEED = 42` everywhere. Numbers produced on Python 3.10.9 with these pins; the
  app runs on 3.12.
- `load_table()` asserts the row count of all nine source tables, and
  `build_orders()` asserts the grain after joining.
- `python src/phase3_stats.py` regenerates every numeric claim in the final
  report into `report/stats_phase3.txt`; `python src/report_stats.py` still
  reproduces every Phase 1 number byte-for-byte.
- `python src/customer_view.py` stops before writing anything if it no longer
  reproduces the numbers it builds on (the rescored test predictions, the
  slides' 614 / 351 and 1,927 / 712, R$1,466 / 775 / 853 / 433);
  `python src/slide_provenance.py` stops if any slide number no longer matches
  its source, or if a number taken from `config.COSTS` is not labelled assumed.
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
