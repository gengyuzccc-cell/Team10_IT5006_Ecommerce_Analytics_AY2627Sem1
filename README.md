# IT5006 Phase 1 — Predicting Delivery Performance and Customer Satisfaction

**Group 10** · Pan Pinyou · Li Yudan · Zhu Gengyu
National University of Singapore · IT5006 Fundamentals of Data Analytics

Literature review, exploratory data analysis, and an interactive dashboard over
the **Olist Brazilian E-Commerce** dataset — 99,441 orders across nine
relational tables.

**Deliverables**

| Item | Location |
|---|---|
| Report (PDF) | [`report/Team10_Phase1_IT5006_AY2627Sem1.pdf`](report/Team10_Phase1_IT5006_AY2627Sem1.pdf) — 5 body pages + references + appendix |
| Notebooks | [`notebooks/*.ipynb`](notebooks/) — four, executed with outputs |
| Dashboard | **[it5006-group-project-10.streamlit.app](https://it5006-group-project-10.streamlit.app)** — source in [`dashboard/`](dashboard/) |
| Reusable code | [`src/`](src/) |

---

## Quick start

The nine Olist tables are committed, so a clone is self-contained:

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements-dev.txt
```

```bash
python src/report_stats.py
```

```bash
streamlit run dashboard/app.py
```

Two dependency files, and they do **not** overlap — `requirements-dev.txt` pulls
the runtime set in with `-r requirements.txt` rather than restating it:

| File | Adds | Installed by |
|---|---|---|
| `requirements.txt` | streamlit, pandas, numpy, plotly | Streamlit Cloud — exactly what `dashboard/app.py` imports |
| `requirements-dev.txt` | `-r requirements.txt` **+** scikit-learn, scipy, matplotlib, Jupyter | you, for the notebooks and figures |

The split exists because Streamlit Cloud installs `requirements.txt` and nothing
else; merging the two would put the whole analysis stack back into the
deployment, which is what broke it the first time.

---

## Data

The **Brazilian E-Commerce Public Dataset by Olist**: nine related CSV tables,
about 1.45 million rows, covering orders placed between September 2016 and
October 2018. Distributed for IT5006 via Canvas — use that copy, not a fresh
Kaggle download, so results stay comparable across teams.

All nine are **committed to this repository** (126 MB), so cloning is enough to
reproduce the notebooks, the figures and the dashboard. The largest file,
geolocation at 61 MB, is under GitHub's 100 MB per-file limit. The 108 MB SQLite
build of the same data is redundant with the CSVs and exceeds that limit, so it
is excluded.

```
data/raw/olist/          (126 MB, tracked)
├── olist_orders_dataset.csv              99,441   the analysis spine
├── olist_order_items_dataset.csv        112,650   one-to-many
├── olist_order_payments_dataset.csv     103,886   one-to-many
├── olist_order_reviews_dataset.csv       99,224   one-to-many
├── olist_customers_dataset.csv           99,441
├── olist_sellers_dataset.csv              3,095
├── olist_products_dataset.csv            32,951
├── olist_geolocation_dataset.csv      1,000,163
└── product_category_name_translation.csv     71
```

The dashboard joins these tables at startup rather than reading a pre-built
artifact, so it can never drift from the report. Reading all nine in full costs
303 MB of RAM; `load_all(slim=True)` selects only the columns the join consumes
— it omits `geolocation_city/state` and the two free-text review columns — which
takes the load to 164 MB and the peak to 268 MB, comfortably inside Streamlit
Cloud's ~1 GB. Dtypes are left untouched, so the slim path reproduces the full
one **column for column**; narrowing coordinates to `float32` would save another
12 MB but shifts computed distances by up to a metre, which is not a trade worth
making.

---

## What the analysis found

Six findings shape everything downstream.

1. **Three of the nine tables are one-to-many against orders.** Joining them
   directly turns 99,441 orders into 118,437 rows — a 19% inflation — and biases
   every average towards large baskets. Each child table is aggregated to one
   row per order *before* joining.
2. **`customer_id` is not a customer.** It is regenerated per order: 99,441 of
   them for 96,096 real people. Group on it and you conclude there are no repeat
   buyers; group on `customer_unique_id` and 2,997 people (3.1%) ordered again,
   one of them 17 times.
3. **Delivery performance drifts hard.** Mean lead time falls from 16.9 days
   (Feb 2018) to 7.7 days (Aug 2018) — and by **5.34 days** between the training
   and test periods. Under a level shift that size, R² against a baseline fitted
   on the past goes negative even for a useful model, so **MAE** is the honest
   regression metric.
4. **Lateness is volatile, not predictable.** It swings between 1.4% and 21.4%
   monthly, spiking at Black Friday 2017 and during the May 2018 truckers'
   strike — exogenous shocks no order-level feature can anticipate.
5. **A late delivery raises the 1–2 star rate from 9.2% to 54.0%**, a 5.9×
   increase, monotone in how late the parcel is. This is the strongest
   relationship in the data.
6. **The prediction moment is a design decision, not a detail** — see below.

### Leakage: one rule per prediction moment

Leakage here is not one rule, because *what is already known depends on when the
model runs*. `assert_no_leakage(features, regime)` enforces the right set:

| Regime | Situation | Forbidden |
|---|---|---|
| `at_checkout` | order just placed | delivery dates, review, `order_approved_at`, status |
| `at_delivery` | parcel arrived, review not yet written | review score and its timestamps only |

Moving the review model between the two nearly **doubles** achievable signal
(PR-AUC 0.212 → 0.402). Moving it the wrong way is worse: adding one forbidden
column to the late-delivery model takes ROC-AUC from **0.589 to 1.000** — the
model is reading the answer, not predicting it.

---

## Candidate problems

Three were explored against the brief's scoping checklist. Two model families
only: a linear baseline and one tree ensemble, reused across both task types.

**Candidate 1 — delivery performance, framed twice (primary).** Regression on
lead time and classification of late vs on-time, from one coherent question.

| Predictor | MAE (days) | R² |
|---|---|---|
| Mean of training period | 6.65 | −0.99 |
| Olist's own delivery promise | 9.79 | −4.13 |
| Ridge | 4.61 | −0.23 |
| **HistGradientBoosting** | **3.89** | **0.07** |

A 41.5% MAE reduction. Note the platform's own promise is a *worse* point
forecast than a constant — Olist pads its estimates deliberately, since a
promise the customer beats is good service.

The classification is weak and honestly so: balanced accuracy 0.580, PR-AUC
0.121 against a 0.074 floor. **Always predicting "on time" scores 0.926
accuracy** while catching zero late orders — exactly the trap the brief warns
about.

**Candidate 2 — low review score, predicted at delivery (secondary).** The
strongest classifier found: PR-AUC **0.402** vs a 0.110 floor (3.6×), balanced
accuracy **0.692**, ROC-AUC 0.730 — and identical (0.403) on customers never
seen in training, so it is not memorising buyers.

**Candidate 3 — repeat purchase within 30 days (rejected).** Base rate 1.8%,
balanced accuracy 0.517 — chance. Separately, right-censoring destroys the
evaluation: at the natural 90-day horizon *no* labelled orders remain in the
test window at all.

We expect a tuned Phase 2 Candidate 2 classifier in the **high 60s** balanced
accuracy and Candidate 1's regression at **MAE 3.5–4 days**. A late-delivery
classifier reporting 95% accuracy would be the majority-class rule or a leak.

---

## Repository layout

```
├── data/raw/olist/                    the nine source tables (126 MB, tracked)
├── notebooks/
│   ├── 01_data_audit.ipynb            nine tables, joins, keys, quality
│   ├── 02_eda_temporal.ipynb          growth, Black Friday, delivery drift
│   ├── 03_eda_entities.ipynb          geography, products, money, reviews
│   └── 04_candidate_problems.ipynb    leakage regimes, three candidates
├── src/
│   ├── config.py                      seeds, paths, windows, leakage regimes
│   ├── data_load.py                   nine-table join + CLEANING_DECISIONS
│   ├── features.py                    leakage-annotated builders + the guard
│   ├── viz.py                         shared plot style, vector PDF export
│   └── report_stats.py                regenerates every number in the report
├── dashboard/
│   ├── app.py                         Streamlit, five linked views
│   └── theme.py                       plotly template + CSS design system
├── requirements.txt                   dashboard runtime (what Cloud installs)
├── requirements-dev.txt               + analysis stack for the notebooks
└── report/
    ├── Team10_Phase1_IT5006_AY2627Sem1.pdf   the submitted report
    └── figures/                       9 vector PDFs, generated by the notebooks
```

## Reproducibility

- `SEED = 42` everywhere; both requirements files pin exact versions on
  Python ≤ 3.12, the interpreter every reported number was produced on.
- `load_table()` asserts the row count of all nine source tables, and
  `build_orders()` asserts the grain after joining. A changed source file fails
  loudly rather than silently altering the report.
- Every figure is generated into `report/figures/` as vector PDF by the
  notebooks — none is pasted in by hand.
- `python src/report_stats.py` recomputes all 170 numeric claims in the report.
- All 20 bibliography entries were verified against publisher records (DOI,
  volume, issue, pages) rather than quoted from memory.

### Deploying

Live at **[it5006-group-project-10.streamlit.app](https://it5006-group-project-10.streamlit.app)**,
built from `main` / `dashboard/app.py` on **Python 3.12**.

> **Keep the Python version at 3.12 or lower** (*App settings → General*).
> `numpy==1.26.4` publishes wheels only up to `cp312`. On Python 3.13+ pip finds
> no wheel and compiles numpy from source; the build dies and the app then hangs
> on *"Your app is in the oven"* indefinitely, with the real error buried in the
> build log rather than shown in the app.
