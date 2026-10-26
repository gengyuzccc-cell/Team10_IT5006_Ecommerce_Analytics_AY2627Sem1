# Presentation script — 10 minutes, 13 slides

The same text is in the speaker notes of `Team10_Phase3_slides.pptx`. Timings
assume a normal speaking pace (about 140 words a minute); the whole script is
roughly 1,350 words, which leaves 20–30 seconds of slack for a live click
through the application on slide 9. The deck is locked on Wednesday 4 November
2026 23:59; the ZIP (report + the identical deck + GitHub link) is due Sunday
8 November 2026 23:59.

| # | Slide | Time | Cumulative |
|---|---|---|---|
| 1 | Title | 0:20 | 0:20 |
| 2 | One number drove the whole project | 0:45 | 1:05 |
| 3 | Nine tables, one row per order — and two traps | 0:55 | 2:00 |
| 4 | What the data told us: drift, volatility and shocks | 0:55 | 2:55 |
| 5 | Three families as ladders | 0:55 | 3:50 |
| 6 | Problem 1 — lead time and the late flag | 1:00 | 4:50 |
| 7 | Problem 2 — low-review risk | 1:05 | 5:55 |
| 8 | Evaluate the way the team works: Precision@K | 0:55 | 6:50 |
| 9 | Deployed application | 0:55 | 7:45 |
| 10 | Replay a day, and the model card | 0:40 | 8:25 |
| 11 | Recommendations, impact, the pilot | 1:00 | 9:25 |
| 12 | Limitations and method lessons | 0:25 | 9:50 |
| 13 | Thank you / questions | 0:10 | 10:00 |

**Suggested split (three speakers):** slides 1–5 (framing, data, design), slides
6–8 (results), slides 9–13 (deployment, recommendations, close). Each block is
about three minutes.

## Live demo path (slide 9, optional — keep under 40 seconds)

Open https://team10-it5006-olist.streamlit.app *before* the session so it is
awake. Have the **Predict** tab open with the default São Paulo → Rio order
already scored. Change the promised delivery date from 2 August to 25 July and
press Predict: the margin flips and the late risk rises. If time is short, skip
the click and point at the screenshot. For questions, **Replay a day** on
16 July 2018 with capacity 10 and the outcomes revealed is the most convincing
single screen.

## Script

**1 — Title (0:20).** Good evening. We are Group 10. Our project asks two
questions of the Olist marketplace data, at two different moments in an order's
life, and turns the answers into decisions that two teams can act on — and we
have deployed the result as a live application that you can open right now.

**2 — One number (0:45).** The data is Olist, a Brazilian marketplace: 99,441
orders in nine relational tables. One finding from our exploratory phase stood
above everything else. If a parcel arrives on time, about nine per cent of
customers leave a one- or two-star review. If it arrives late, fifty-four per
cent do. That is a six-fold increase, and it grows with every day of delay. So
we framed two problems at two moments. At checkout, before anything ships: how
long will this take and will it miss the promised date — that is for
operations. At delivery, before the review is written: will this customer leave
a bad review — that is for customer experience. They form one chain: the order
flagged at checkout is the one that arrives late, and the one that arrives late
is the one worth calling.

**3 — Data and traps (0:55).** Nine tables, one row per order. Two things in
this dataset bite anyone who is not careful. Three tables are one-to-many
against orders, so a naive join inflates the data by nineteen per cent and
quietly reweights every average. And customer_id is regenerated per order — it
is not a person. We split the data by time, with each period doing exactly one
job: fit on the past, choose and calibrate on a two-month validation window, and
score the July–August 2018 test window once, for models already chosen. And we
define leakage per prediction moment: delivery dates are forbidden at checkout
and allowed at delivery. One forbidden column takes ROC-AUC from 0.59 to a
perfect 1.0 — so the guard is an assertion that runs before any model is fitted.

**4 — What the data told us (0:55).** Three facts shaped everything. First,
drift: Olist got much faster during the window — mean lead time fell from
seventeen days to under eight. A model trained on the slower past is off-level
on the present, so R-squared goes negative even for a useful model, we lead with
mean absolute error, and we plan for monthly refits. Second, lateness does not
trend, it spikes — from one per cent to twenty-one per cent month to month. Our
Phase 1 report blamed the May 2018 truckers' strike; the evaluators asked us to
look again, and the real peak is a February–March carrier slowdown that
coincided with Olist tightening its promise. Third, the strike is real but only
visible at weekly resolution, because Olist responded by stretching the promised
window to thirty-seven days. None of these shocks is a property of an order,
which puts a ceiling on any late classifier — and we said so before modelling.

**5 — Modelling design (0:55).** The brief allows two or three model families;
we use three, and we run each as a ladder, starting from the simplest variant.
Plain least squares before Ridge; one tree before a forest before gradient
boosting; and an ensemble built only from the winners of the first two. That way
"the complexity was worth it" is a finding, not an assumption. Three design
rules. Time is respected: hyperparameters are tuned with expanding-window
cross-validation. Models are selected by rank across the three most recent
out-of-sample windows, because the windows differ in difficulty far more than the
models do. And the metrics match the action: PR-AUC against its no-skill floor
to choose a model, precision at K for what a team's day actually looks like, and
calibrated probabilities for the cost layer.

**6 — Problem 1 (1:00).** Problem 1, the regression. The deployed model is a
voting blend of a linear model and gradient-boosted trees, with mean absolute
error of three point nine days — forty-one per cent better than a constant, and
sixty per cent better than the date Olist itself shows the customer, because a
promise is a commercial decision, padded, not a forecast. Inside the tree family
every rung paid off; inside the linear family it did not — Ridge is worse than
plain least squares, and we say so. The blend wins for a specific reason: trees
cannot extrapolate a falling level, the linear member can, and averaging covers
that exact failure. For the late flag we did not need a second model at all:
comparing the regression's forecast with the promised date gives two point nine
times the no-skill floor, against one point two for the best classifier trained
directly on the binary label.

**7 — Problem 2 (1:05).** Problem 2, the review model. ROC-AUC of 0.75 and
PR-AUC of 0.41 — three and three-quarter times the no-skill floor, on a target
where always predicting "satisfied" scores eighty-nine per cent accuracy and
catches nothing. It scores the same on customers it has never seen. The Phase 2
evaluators asked us to show the fold-to-fold variation, and it is instructive:
the spread across folds is about plus or minus 0.09, while the best three tree
models are within a thousandth of each other. Cross-validation cannot separate
them — which is exactly why we rank within windows. What is deployed is the
stacking classifier that our pre-registered rule chose; the single boosted model
is a tie, and we say so in one place, consistently, with a test that checks it.
And the ablation we promised in Phase 1: weighting the loss for imbalance buys
no discrimination and triples every predicted risk, so we move the threshold
instead.

**8 — Precision@K (0:55).** PR-AUC is the right number for choosing a model, but
nobody acts on a whole ranking. A customer-experience team contacts the K
riskiest deliveries it can reach in a day, so the number that describes its day
is the precision of that top K. At five per cent of deliveries — about ten
contacts a day — two in three contacts reach a customer who really would have
left a one- or two-star review, covering thirty-one per cent of them. Under an
illustrative five-reais contact and forty-reais review, break-even precision is
0.125, and the list clears it at every capacity we tried: the constraint is the
team's capacity, not the model's threshold. The late flag tells the opposite
story: it never reaches the 0.30 that a twelve-reais expedite would need. A
genuinely informative model that is still not worth paying to act on — so
operations gets the free levers, not the expensive one.

**9 — Deployed application (0:55).** Everything is deployed. The application
serves all three models: enter one order as it would look at checkout and you
get the expected lead time, the risk of missing the promise against the tuned
threshold, and the review risk. Upload a CSV and you get the same three
predictions for every row, plus the triage list: enter your team's capacity and
the top-K are marked, with the precision that capacity achieved on the held-out
window shown beside them. Nothing in the app reimplements the feature
engineering — a form submission runs through exactly the same code the models
were trained on, which is the most common way a deployed model quietly stops
matching its reported metrics. *(Optional live click: change the promised date,
watch the late risk move.)*

**10 — Replay a day (0:40).** New in this phase is Replay a day. Because the data
ends in 2018, the app predicts on held-out historical orders that stand in for
new ones — so we made that literal. Pick any day in the July–August test window,
pick a capacity, and the app ranks that day's orders exactly as the team would
work them. Then flip the toggle and it reveals what actually happened. On the
sixteenth of July, eight of the ten riskiest orders really did leave a one- or
two-star review. The model card states exactly what is deployed, read from the
artifacts themselves, with every hyperparameter, the fold-to-fold table, and
where the models should not be trusted.

**11 — Recommendations and the pilot (1:00).** What should each team do?
Operations should re-quote every promise with the model's estimate and pre-warn
buyers above an eleven per cent late risk — both nearly free, and the warning
reaches five in six of the orders that will actually miss. It should not
expedite on the flag; the precision is never there. Customer experience should
work each day's deliveries from the top of the risk list to its capacity. Under
our stated costs that policy is worth about fourteen hundred reais per thousand
orders. But that figure rests on one assumption the data cannot test: that a
call prevents the review. So our final recommendation is an experiment, not a
rollout: randomise inside the daily top-K, contact half, leave half. The
control rate is already known — fifty-seven per cent — break-even needs one
averted review in five, and six weeks at ten contacts a day is enough to detect
a one-in-four effect. The model card has the calculator.

**12 — Limitations and method (0:25).** The limitations were stated before the
numbers were. The models are fitted on a moving target, so they need monthly
refits. The biggest driver of lateness is shocks no order can see. The benefit of
outreach is assumed until the pilot runs. And a rating is not satisfaction. Four
method lessons travel further: select by rank on recent windows, because
cross-validation under drift reorders models and its spread swamps the
differences; move the threshold rather than weight the loss; compare PR-AUC by
its lift over the floor, not raw; and value the negative results.

**13 — Close (0:10).** Thank you. The app is live at the link, the repository
holds every notebook, the tests and the final report, and we are happy to take
questions.

## Likely questions, and where the answer is

| Question | Answer in one line | Report |
|---|---|---|
| Why not deploy the simpler boosted model for Problem 2? | It is a statistical tie (0.001 PR-AUC); the pre-registered rank rule chose the stack, and switching after seeing the test window would be selecting on the test set. | §7.5, Table 7 |
| Why is R² negative / near zero? | Lead time fell 5.3 days between training and test; under a level shift R² against the test variance is negative even for a useful model. MAE is the honest metric. | §4.1, §7.1 |
| How do you know the track-record features do not leak? | They are as-of joins on the moment the earlier outcome became observable; a test rebuilds one from first principles for 200 random orders and requires exact agreement. | §5.2, Appendix A |
| Why a chronological split rather than stratified? | The target drifts; a random split trains on the future. Per-period positive rates are reported, and the stratified, customer-grouped protocol is shown as a contrast exhibit (its spread is 15× narrower and no closer to the truth). | §3.2, Appendix G |
| Where do the R$5 / R$40 costs come from? | Stated assumptions — Olist publishes none. The decision hinges on break-even precision (0.125 and 0.300), which is reported and swept. | §9.2 |
| What would change the "do not expedite" verdict? | Expediting cheaper than about R$11 an order, or a flag with precision above 0.30 at useful volumes. | §9.3 |
| How does the pilot avoid annoying customers? | The hold-out arm inside the top-K measures that direction too; the primary outcome is the 1–2 star rate in each arm. | §9.4 |
