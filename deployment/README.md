# Deployment — the Streamlit application

**Live:** https://team10-it5006-olist.streamlit.app
**Source of truth:** this folder. `dashboard/app.py` at the repository root is a
short shim that runs `deployment/app.py`; it exists because the Community Cloud
deployment was created with that main-file path and Community Cloud does not let
the path be changed afterwards. The shim also drops the deployment modules
(`journey`, `predict`, `replay`, `theme`) from `sys.modules` before each run, so a
push takes effect without a reboot; a change under `src/` still needs one.

| File | Role |
|---|---|
| `app.py` | Page layout, sidebar filters, the five exploratory tabs, and the wiring of the model tabs |
| `journey.py` | *Customer journey*: one held-out customer's order from checkout to review, read from `report/results/customer_*.csv` |
| `predict.py` | Model serving: single-order form with the date the customer would be quoted, batch CSV scoring, top-K triage list, and the model card (deployed estimators, fold-to-fold variation, hyperparameters, Precision@K, what the recommendations change for customers, the value and pilot calculators, where every number on the slides comes from) |
| `replay.py` | *Replay a day*: the held-out test window (orders placed 30 June – 30 August 2018), one day at a time, blind first and then revealed |
| `theme.py` | Plotly template, the CSS design system and the small HTML helpers (KPI cards, the order card, wrapping tables, the provenance tags) |
| `capture_screenshots.py` | Takes the screenshots in `docs/screenshots/` and the four the report embeds |

The application imports the same `src/data_load.py`, `src/features.py` and
`src/modeling.py` the notebooks and the report use, so no feature is computed one
way in training and another way in production. It refits nothing.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # the runtime set only
streamlit run deployment/app.py          # or: streamlit run dashboard/app.py
```

The nine Olist tables are committed under `data/raw/olist/`, the fitted models
under `models/`, and the result tables under `report/results/`, so a fresh clone
runs without a training step.

## How to use it

The application has nine tabs and opens on the customer.

1. **Customer journey** — one reviewed order of the held-out test window followed
   from checkout to review: the order card, the promise, how long the parcel
   took, when and how the customer rated it, and which contact could still have
   reached them first. A timeline in three lanes (customer; seller and carrier;
   platform and models) and a table of moments say, for each moment, what the
   customer experienced, what the deployed models said and what the report's
   recommendations would do there; the customer's own words and the chart of
   who could still be reached among all 1–2 star raters of the window follow.
   It opens on the order from the presentation and Box 1 of the report
   (`e1320973…`). *Whose order?* switches to any customer of a test day's list
   (ranked as that day's list ranks them) or to an order id (first seven or more
   characters; a link with `?order=<id>` opens it directly). Not affected by the
   sidebar filters.
2. **Overview · Delivery · Geography · Satisfaction · Data quality** — the
   exploratory analysis. The sidebar filters (purchase month, customer region,
   category, delivered-only) apply to the first four; *Data quality* describes
   the raw tables and is not filtered, and the *In customers' own words* panel of
   *Satisfaction* covers the whole analysis window.
3. **Predict** — enter one order as it looks at checkout and press *Predict*.
   You get the expected lead time with its typical error (±3.9 days), the risk of
   missing the promised date against the deployed 11 % flag line, and — conditional
   on that forecast — the risk of a 1–2 star review against the 19 % flag line. *What
   the customer would see* sets the platform's date beside the date the forecast
   plus the 8-day margin would quote (the margin is rule R1 of
   `customer_promise_rules.csv`, chosen on the validation months). *What to do with
   this* follows the report's first recommendation: quote that date, and no blanket
   warning is needed; if the platform keeps its own date, a warning goes only to the
   riskiest tenth of orders (late risk of 26.1 % or more on the held-out window,
   where one parcel in four was late), because the 11 % line would reach 44.7 % of
   customers, most of them on time. Below the
   form, upload a CSV to score a whole file; the template button gives the exact
   columns. The scored file comes back with the predictions appended and a
   **triage list**: rows ranked by review risk with the top K marked for the
   capacity you enter, and the precision and recall that capacity reached on the
   held-out window beside it.
4. **Replay a day** — pick a day of the held-out test window and a daily
   capacity K. Every day opens **blind**: the list shows the scores, the ranks and
   each order's details but no outcome (a purchase day's orders are scored with
   their realised delivery, as on the slides) — and its
   download holds only those columns (`replay.blind_columns()`, checked in
   `tests/test_leakage.py`). On the purchase-day list the checkout forecast and late
   risk wait for the reveal too: they were saved for delivered orders only, so a
   blank would mark the customers who never received the parcel. The checkout cards
   of a purchase day cover every delivered order bought that day, reviewed or not
   (238 on 16 July 2018, as in the report's slide-provenance appendix). Switch on *Reveal what actually happened* to add each
   customer's rating, what they went through, and which contact could still have
   reached them; changing the day, K or the list turns the reveal off again.
   *Which list?* chooses between the orders bought that day (as on the slides)
   and the parcels delivered that day whose customer had not yet rated the order
   (as a team would work it). With the reveal on, the tab also shows the same
   list's precision on every day of the window.
5. **Model card** — what is deployed (the estimators as pickled, with their
   members and hyperparameters), the test metrics, the fold-to-fold variation of
   every candidate, Precision@K / Recall@K, where the models should not be
   trusted, what the recommendations change for customers (counts with named
   populations and 95 % intervals), what a contact is worth with **both costs as
   inputs** (the R$5 contact and the R$40 rating are assumptions, `src/config.py`
   COSTS), a pilot calculator per contact (promised-date message, delivery-day
   call, and the design on slide 11), *Where the numbers on the slides come from*,
   and the commands that reproduce every number. Every card is tagged OBSERVED,
   ASSUMED, DERIVED or PLANNED.

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

An optional `seller_id` column switches on that seller's own track record from
`models/history_lookup.csv`. Those aggregates run to the end of the data
(August 2018), so use the column only for orders placed after it: for a
historical order the seller's record would already include that order's own
outcome. Without the column, the history features fall back to route, state and
platform, as the form does.

### Error handling

Every failure a visitor can cause produces a specific message rather than a
traceback: a promised date before the purchase date, a non-positive price,
negative freight, more sellers than items, an empty or malformed CSV, a missing
column, an oversized file, an unknown category (scored through the encoder's
rare-level bucket), an unknown state code (the row is refused), missing model
artifacts or result tables, and a scikit-learn version that differs from the one
the models were pickled with (a visible warning, not a silent mismatch). A forecast
below zero, which the lead-time regression gives only far outside the orders it was
fitted on (thirty items in one basket, for instance), is flagged rather than hidden. The data
loader asserts the row count of all nine source tables and stops the app with an
explanation if they change. In a batch upload, state codes are normalised (`sp`
becomes `SP`), dates and numbers are parsed defensively, and rows that the form
would refuse are skipped and listed by row number; the remaining rows are scored.

## Deploying on Streamlit Community Cloud

1. *New app* → repository `gengyuzccc-cell/Team10_IT5006_Ecommerce_Analytics_AY2627Sem1`,
   branch `main`, main file path `dashboard/app.py` (or `deployment/app.py` for a
   fresh deployment).
2. *Advanced settings* → **Python 3.12**. `numpy==1.26.4` publishes wheels only
   up to CPython 3.12; on 3.13 pip compiles numpy from source, the build fails,
   and the app hangs on *"Your app is in the oven"* with the real error buried in
   the build log. `requirements.txt` carries environment markers that select
   wheel-backed versions on 3.13+, but the pinned, tested configuration is 3.12.
   Streamlit is pinned at 1.39.
3. Deploy. Every push to `main` redeploys automatically; the models are loaded
   once per server process and cached, so no visitor pays for unpickling them
   twice. A change to a module under `src/` is picked up only after *Reboot app*
   in the Cloud console.

**Staying reachable.** Community Cloud puts an app to sleep after twelve hours
without traffic and shows a *"Yes, get this app back up!"* button; anyone with the
link can press it, and the app is back in about a minute. Open the link once
before a grading window or a presentation. No credentials are used anywhere: the
app reads only files committed to this repository.

## Where the numbers come from

Everything the model tabs display is read from files the analysis scripts write;
the app recomputes no result, so a number on screen is a number in the report.

| Shown in | Read from |
|---|---|
| Customer journey | `report/results/customer_journeys_test.csv` (one row per reviewed test order: timestamps, review, saved predictions, ranks, outcome, reach class), `customer_story.csv`, `customer_story_events.csv`, `customer_outcomes_test.csv`, `customer_promise_rules.csv`, `models/model_metadata.json` |
| Satisfaction, *In customers' own words* | `customer_complaint_topics.csv`, `customer_complaint_dictionary.csv`, `customer_complaint_precision.csv` |
| Predict | `models/*.joblib`, `models/history_lookup.csv`, `models/model_metadata.json`, `customer_promise_rules.csv`, `customer_outcomes_test.csv`, `customer_contact_lists.csv` and `predictions_p1b_test.csv` (the riskiest tenth by late risk), `capacity_p1b.csv`, `capacity_p2.csv` |
| Replay a day | `predictions_p2_test.csv`, `predictions_p1a_test.csv`, `predictions_p1b_test.csv`, `customer_journeys_test.csv`, `customer_contact_lists.csv`, `models/model_metadata.json` |
| Model card | `models/*`, `cv_folds.csv`, `hyperparameters.csv`, `capacity_p2.csv`, `customer_promise_rules.csv`, `customer_outcomes_test.csv`, `customer_contact_lists.csv`, `customer_value.csv`, `customer_pilot_arms.csv`, `slide_provenance.csv` |

The `customer_*.csv` tables are written by `python src/customer_view.py`, and
`slide_provenance.csv` by `python src/slide_provenance.py`; the report's
slide-provenance appendix prints the same table. Each number is one of four
kinds, and the model card tags it so: **observed** (counted in the data, e.g. the
57.2% of the slides' daily top ten who rated 1–2 stars), **assumed** (no data
source: the R$5 contact and the R$40 rating), **derived** (computed from the two,
e.g. a break-even share), or **planned** (a design choice, e.g. a pilot's size).
The exploratory tabs compute their figures live from the same joined order frame
the notebooks use. Beyond that, the app computes only the predictions for an
order entered in Predict, the calculators' arithmetic on the two assumed costs,
and counts, shares and time gaps over the rows it reads; at ten a day, the Replay
tab's totals over the window equal `customer_contact_lists.csv` (614 contacts and
351 customers who rated 1–2 stars on the purchase-day list, 546 and 142 on the
delivery-day list).
