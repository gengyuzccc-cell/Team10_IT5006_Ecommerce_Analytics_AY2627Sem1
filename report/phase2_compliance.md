# Phase 2 compliance matrix

Every requirement in `phase 2 requirements.docx`, mapped to where it is satisfied.
This is a working checklist for the team, not report filler — the report itself is
6–8 body pages and does not contain this table.

Legend: **§** = section of `report/Team10_Phase2_IT5006_AY2627Sem1.pdf`.

---

## Problem scoping (required)

| Requirement | Detail | Where |
|---|---|---|
| Number of problems | 1 or 2 (not more) | **2** — delivery performance, satisfaction risk. §1, Table 1 |
| Problem types | Must cover both classification and regression | Regression: P1A lead time. Classification: P1B late flag *and* P2 low review. §1 |
| Scope | Achievable with Olist data and course techniques | Both targets derived from the nine tables; no external data. §1 |
| Justification | Business stakeholder, target definition, success criteria | Operations and customer experience; target definitions and pre-registered success bands in §1, ranges in Appendix B |

## Modelling requirements — quality over quantity

| Requirement | Where |
|---|---|
| Model families: 2–3 total across the project | **3** — Linear, Tree-based, Ensemble. §3.1 |
| Families reused across classification and regression | Same three families serve P1A, P1B and P2. §3.1 |
| Baseline: simplest variant in each family first | Rung 0 of each ladder; every rung above compared against it. §3.1, §4.1, §4.3 |
| Every more complex model shown to beat its baseline | Ladder tables report the delta at each rung. §4.1, §4.3 |
| Train–test discipline, no leakage | Three-period chronological split, customer-disjoint (§2.1, Table 2); guard §2.4 |
| Model selection compared systematically, choice explained | Rank-averaged over the three most recent out-of-sample windows; the rule and the two alternatives it replaced are argued in §3.2, applied in §4.1–§4.3, and its one failure reported in §4.2; the rungs themselves are argued from the literature in §3.1 (added after the Milestone 1 feedback) |
| Cross-validation for robust estimates and tuning | `TimeSeriesSplit(5)` inside train for every hyperparameter search. §3.3 |
| Reproducibility: seeds, preprocessing in pipelines | `SEED = 42` throughout; all preprocessing inside `sklearn.Pipeline`. §3.2, §3.5, Appendix A |

## Deliverables

| Deliverable | Where |
|---|---|
| Problem statement(s) and stakeholder context | §1 |
| Data preprocessing and cleaning methodology | §2.1, and the Phase 1 cleaning contract it inherits |
| Feature engineering strategy | §2.2, §2.3 (track-record features) |
| Model descriptions and training process | §3.1, §3.2 |
| Hyperparameter tuning methodology | §3.3, Appendix C (search spaces and selected values) |
| Classification metrics: Precision, Recall, F1, AUC-ROC / PR-AUC | §4.2, §4.3 — all reported in Tables 4 and 5 (precision and recall columns added after the Milestone 1 feedback), PR-AUC always against its no-skill floor, accuracy beside the majority-class rule; §4.4 adds Precision@K / Recall@K for the capacity-constrained action |
| Regression metrics: MAE, RMSE, R² | §4.1 — all three, with MAE leading and R² read as a drift diagnostic |
| Cross-validation results | Every results table carries the CV column beside validation and test; Appendix F quantifies what the protocol was telling us; Appendix J adds the in-sample training fit so train / validation / test can be compared directly |
| Feature engineering strategy, with a rationale per feature | §2.2 (groups, literature), Appendix K (all 36 features: definition, reason, moment knowable) |
| Feature importance / interpretability | §5.1 — permutation importance on the test window, both problems, with the correlated-predictor caveat stated |
| Model comparison and final selection with justification | §3.3 (the rule), §4.1, §4.2, §4.3 |
| Ensemble or stacking results | §4.5 — Voting and Stacking from families A + B; earns its place on 1A, does not on 2, both reported |
| Actionable insights for the stakeholder | Table 2 (the levers each team pulls, with trigger and cost), §5.2, §5.3 — decision policy, capacity-based operating point, cost sensitivity, per-stakeholder actions |
| Discussion of limitations and constraints | §6, and the honest protocol failure in §4.2 |
| Response to the Milestone 1 feedback | Appendix H maps each feedback point to the section that acts on it; Appendix G is the corrected lateness analysis with the strike cited |

## Submission format

| Requirement | Where |
|---|---|
| Technical report as PDF with GitHub repository link | `report/Team10_Phase2_IT5006_AY2627Sem1.pdf`, repository link on the cover |
| GitHub repository with Jupyter notebooks and Python scripts | `notebooks/05–09`, `src/`, `tests/`, `dashboard/` |
| Model performance summary tables in the report | §4 — one summary table per problem, plus the protocol table; complete tables in `report/results/*.csv` |
| Report length 6–8 pages excluding cover, references, appendices | Verified at build time — see "Page count" below |

## Assessment criteria

| Criterion | How it is addressed |
|---|---|
| Problem framing and scoping | Two problems at two prediction moments forming one operational chain; scoping checklist closed out in notebook 05 |
| Data preparation and feature engineering, incl. leakage avoidance | Regime-specific leakage guard; the as-of history construction verified against a brute-force recomputation (`tests/test_leakage.py`) |
| Modelling discipline | Three periods with one job each; test scored once; seeds and pipelines throughout; 41 automated validation checks |
| Quality of model family comparison | Three families as complexity ladders, each rung compared on the same protocol, with the honest finding that regularisation buys nothing here |
| Evaluation rigor | PR-AUC against its floor, majority accuracy printed beside accuracy, calibration reported, CV-protocol contrast quantified |
| Ensemble / stacking justified and explained | §4.5, including the inner-CV compromise stacking forces and what it means for the conclusion |
| Business interpretation | §5 — who acts, on what, at which threshold, and how the answer moves with the cost ratio |

## Technical requirements

| Requirement | Where |
|---|---|
| Python: pandas, scikit-learn, matplotlib/seaborn, numpy | `requirements.txt`, `requirements-dev.txt` |
| Dashboard: Streamlit | `dashboard/app.py` — seven tabs |
| Deployment: Streamlit Cloud | https://team10-it5006-olist.streamlit.app |
| Version control: GitHub | https://github.com/gengyuzccc-cell/Team10_IT5006_Ecommerce_Analytics_AY2627Sem1 (the Milestone 1 account was suspended; everything was re-pushed here) |
| Documentation: Jupyter notebooks with markdown | `notebooks/05–09`, executed with outputs |

## Modelling guidelines

| Guideline | How |
|---|---|
| Use scikit-learn Pipelines to prevent leakage | Every model is a `Pipeline`; preprocessing fitted inside each fold |
| Simplest variant first, complexity justified by evidence | Rung 0 → 1 → 2 in each family, deltas reported |
| 2–3 families, reused across task types | Exactly 3, reused |
| Stratified splits for imbalanced classification | `StratifiedGroupKFold` used as the contrast protocol; the primary split is temporal, because a shuffled split is invalid under drift — argued in §3.3 rather than skipped |
| Seeds set and documented | `config.SEED = 42`, referenced in every estimator |
| Ensembles when they add demonstrable value | Built and measured; the verdict is reported, not assumed |
| Watch for post-outcome leakage | `assert_no_leakage(features, regime)`, tested in both directions |
| Watch for imbalance; do not rely on accuracy | `majority_accuracy` in every classification row; PR-AUC lift as the headline |

## Coding guidelines

| Guideline | How |
|---|---|
| Clean, well-commented code, meaningful names | `src/` modules carry the reasoning for each decision, not just its implementation |
| Reproducible, seeds set | `SEED = 42`; `src/phase2_stats.py` regenerates every reported number |
| Error handling and data validation | `tests/test_leakage.py` (41 checks); row-count assertions on load; typed errors in the dashboard |
| Separation of preparation, modelling, evaluation, deployment | `data_load.py` / `features.py` → `modeling.py` → `evaluate.py` → `dashboard/predict.py` |
| All code in GitHub with descriptive commits | See the repository history |
| Comprehensive README | `README.md` — setup, execution, results, deployment notes |

## Data usage guidelines

| Guideline | How |
|---|---|
| Olist as the primary and only source | No external data used |
| Handle missing values and outliers | Structural missingness left as `NaN` and excluded rather than imputed; in-pipeline median imputation for model inputs; geolocation outliers dropped by a Brazil bounding box |
| Document transformations and assumptions | `CLEANING_DECISIONS` in `data_load.py`; cost assumptions stated in `config.COSTS` and swept in §5.3 |
| Ensure reproducibility | Appendix A |

## Deployment guidelines

| Guideline | How |
|---|---|
| Accessible via live URL | https://team10-it5006-olist.streamlit.app |
| Error handling for edge cases | Invalid dates, non-positive amounts, more sellers than items, empty/malformed/oversized CSV, missing columns, unknown category or state, missing model artifacts, scikit-learn version drift |
| Clear usage instructions | Sidebar guide; form defaults; downloadable CSV template |
| Document any API endpoints with examples | No API endpoints — the proof of concept is the Streamlit app. The batch CSV contract is documented in the Predict tab and in Appendix E |
| No hardcoded credentials | None anywhere; nothing in the app reads a secret |
| Test deployment before submission | Local run plus the deployed URL exercised across all seven tabs, including the batch scorer's triage list; the app is opened once before each grading window because Community Cloud sleeps it after twelve idle hours |

---

## Page count

The body is §1–§6 plus the conclusion. Cover page, references and appendices are
excluded by the brief. Check before submitting:

```
cd report && pdflatex Team10_Phase2_IT5006_AY2627Sem1
```

and confirm the reference list begins on the page after §6.
