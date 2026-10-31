# Deployment — the Streamlit application

**Live:** https://team10-it5006-olist.streamlit.app
**Source of truth:** this folder. `dashboard/app.py` at the repository root is a
two-line shim that runs `deployment/app.py`; it exists because the Community Cloud
deployment was created with that main-file path and Community Cloud does not let
the path be changed afterwards.

| File | Role |
|---|---|
| `app.py` | Page layout, sidebar filters, the five exploratory tabs, and the wiring of the three model tabs |
| `predict.py` | Model serving: single-order form, batch CSV scoring, top-K triage list, model card (deployed estimators, fold-to-fold variation, hyperparameters, Precision@K, pilot calculator) |
| `replay.py` | *Replay a day*: scores of the held-out July–August 2018 orders, one purchase day at a time, with the outcomes revealable |
| `theme.py` | Plotly template and the CSS design system, so every tab reads as one product |

The application imports the same `src/data_load.py`, `src/features.py` and
`src/modeling.py` the notebooks and the report use, so no figure here can disagree
with either PDF, and no feature is computed one way in training and another way
in production.

## Run it locally

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt          # the runtime set only
streamlit run deployment/app.py          # or: streamlit run dashboard/app.py
```

The nine Olist tables are committed under `data/raw/olist/`, and the fitted models
under `models/`, so a fresh clone runs without a training step. `python
src/train_models.py` refits everything if you want to.

## How to use it

1. **Overview · Delivery · Geography · Satisfaction · Data quality** — the
   exploratory analysis. The sidebar filters (purchase month, customer region,
   category, delivered-only) apply to the first four of these tabs; *Data quality*
   describes the raw tables and is not filtered.
2. **Predict** — enter one order as it looks at checkout and press *Predict*.
   You get the expected lead time (±3.9 days typical error), the risk of missing
   the promised date against the tuned 11 % threshold, and — conditional on that
   forecast — the risk of a 1–2 star review. Below the form, upload a CSV to score
   a whole file; the template button gives the exact columns. The scored file comes
   back with the three predictions appended and a **triage list**: rows ranked by
   review risk with the top-K marked for the capacity you enter, and the precision
   and recall that capacity achieved on the held-out window beside it.
3. **Replay a day** — pick a purchase day in the held-out test window (30 June –
   30 August 2018, its 62 purchase days) and a daily capacity K. The tab shows what the deployed models
   predicted for every order placed that day, ranked, and — with *Reveal what
   actually happened* on — which of them really left a 1–2 star review and how
   long delivery really took. This is the brief's "held-out historical records
   simulating new orders", made inspectable.
4. **Model card** — what is deployed (the estimators as pickled, with their
   members and hyperparameters), the test metrics, the fold-to-fold variation of
   every candidate, Precision@K / Recall@K, where the models should not be
   trusted, a calculator for the outreach pilot, and the commands that reproduce
   every number.

### Batch CSV contract

There are no HTTP API endpoints: the proof of concept is the Streamlit app, and
the batch scorer is its programmatic surface. A CSV must carry these columns
(download the template from the Predict tab for an example row):

| Column | Type | Example |
|---|---|---|
| `order_purchase_timestamp` | date or datetime | `2018-07-15` |
| `order_estimated_delivery_date` | date | `2018-08-02` |
| `customer_state`, `seller_state` | two-letter Brazilian state | `RJ`, `SP` |
| `lead_category` | Olist English category name | `bed_bath_table` |
| `lead_payment_type` | `credit_card`, `boleto`, `voucher`, `debit_card` | `credit_card` |
| `n_items`, `n_sellers` | integers ≥ 1, sellers ≤ items | `1`, `1` |
| `total_price`, `total_freight` | R$, price > 0, freight ≥ 0 | `89.90`, `17.20` |
| `total_weight_g`, `max_volume_cm3`, `total_photos`, `max_installments` | numbers | `900`, `8000`, `3`, `1` |

Up to 5,000 rows per upload. The response is the same file with
`predicted_days`, `promised_days`, `margin_days`, `p_late`, `late_flag`,
`p_low_review` and `low_review_flag` appended, plus the ranked triage list.

### Error handling

Every failure a visitor can cause produces a specific message rather than a
traceback: a promised date before the purchase date, a non-positive price,
negative freight, more sellers than items, an empty or malformed CSV, a missing
column, an oversized file, an unknown category (scored through the encoder's
rare-level bucket), an unknown state code (the row is refused), missing model artifacts, and a scikit-learn version
that differs from the one the models were pickled with (a visible warning, not a
silent mismatch). The data loader asserts the row count of all nine source tables
and stops the app with an explanation if they change. In a batch upload, state codes
are normalised (`sp` becomes `SP`), dates and numbers are parsed defensively, and
rows that the form would refuse are skipped and listed by row number; the
remaining rows are scored.

## Deploying on Streamlit Community Cloud

1. *New app* → repository `gengyuzccc-cell/Team10_IT5006_Ecommerce_Analytics_AY2627Sem1`,
   branch `main`, main file path `dashboard/app.py` (or `deployment/app.py` for a
   fresh deployment).
2. *Advanced settings* → **Python 3.12**. `numpy==1.26.4` publishes wheels only
   up to CPython 3.12; on 3.13 pip compiles numpy from source, the build fails,
   and the app hangs on *"Your app is in the oven"* with the real error buried in
   the build log. `requirements.txt` carries environment markers that select
   wheel-backed versions on 3.13+, but the pinned, tested configuration is 3.12.
3. Deploy. Every push to `main` redeploys automatically; the models are loaded
   once per server process and cached, so no visitor pays for unpickling them
   twice.

**Staying reachable.** Community Cloud puts an app to sleep after twelve hours
without traffic and shows a *"Yes, get this app back up!"* button; anyone with the
link can press it, and the app is back in about a minute. Open the link once
before a grading window or a presentation. No credentials are used anywhere: the
app reads only files committed to this repository.

## Where the numbers come from

Everything the model tabs display is read from files that the analysis scripts
write — `models/model_metadata.json`, `models/*.joblib`, and the tables in
`report/results/` (`capacity_p2.csv`, `cv_folds.csv`, `hyperparameters.csv`,
`pilot_breakeven.csv`, `predictions_*_test.csv`, `review_timing_test_orders.csv`). The app recomputes nothing, so a
number on screen is a number in the report.
