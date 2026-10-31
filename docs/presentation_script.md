# Phase 3 presentation: script, demo path and predicted questions

Group 10 · IT5006 · 10 minutes of presentation + 5 minutes of questions · deck: [`Team10_Phase3_slides.pptx`](Team10_Phase3_slides.pptx) (13 slides; present from the .pptx — the [PDF](Team10_Phase3_slides.pdf) is a static copy of the same 13 slides).

The English text below is word-for-word the speaker notes in the deck (Presenter View, Alt+F5), including the time stamp at the start of each note.

## The story in one paragraph

The course asks what we used to solve which problem, not which models we tried. So the talk follows one real held-out order whose 1-star review was written 5.8 days *before* the parcel arrived, shows that this is the pattern (a broken promise turns into a bad review, and for late parcels the review usually comes first), and builds the solution around **when** Olist can still act: at checkout (an honest date and a free warning), on the promised date (speak first, hold the survey until the parcel arrives) and at delivery (ten calls a day from a ranked list). Models appear in one line on slide 6; the evidence is told in business units (days, calls, reais), the live app replays the very day the story order was bought, and the close admits what the data cannot know and proposes the experiment that would.

**Arc:** hook (one order) → the pattern → the timing discovery → why promises break → three tools for three moments → proof at checkout → proof at delivery → the live app → Monday playbook → what it is worth and how to prove it → lessons → callback and live demo.

## Timing and speakers

| # | Slide | Time (as in the notes) | Speaker | Words | Implied pace |
|---|---|---|---|---|---|
| 1 | Catch the bad review before it's written | 0:00–0:25 | Pan Pinyou | 49 | 118 wpm |
| 2 | This review arrived before the parcel did | 0:25–1:10 | Pan Pinyou | 84 | 112 wpm |
| 3 | A broken promise turns into a bad review | 1:10–1:50 | Pan Pinyou | 72 | 108 wpm |
| 4 | For a late parcel, the review comes first | 1:50–2:35 | Pan Pinyou | 105 | 140 wpm |
| 5 | Why promises break: a moving, shock-hit network | 2:35–3:15 | Li Yudan | 92 | 138 wpm |
| 6 | Three moments before the review — three simple tools | 3:15–4:00 | Li Yudan | 107 | 143 wpm |
| 7 | Honest dates at checkout — and why rushing loses money | 4:00–4:45 | Li Yudan | 116 | 155 wpm |
| 8 | Ten calls a day find 5× more unhappy customers | 4:45–5:30 | Li Yudan | 139 | 185 wpm |
| 9 | Replay 16 July 2018 blind — then reveal reality | 5:30–6:15 | Zhu Gengyu | 97 | 129 wpm |
| 10 | Four changes Olist can launch on Monday | 6:15–6:50 | Zhu Gengyu | 76 | 130 wpm |
| 11 | Worth R$775–R$1,466 per 1k orders — test before scaling | 6:50–7:25 | Zhu Gengyu | 112 | 192 wpm |
| 12 | Three lessons we would tell the next analytics team | 7:25–7:50 | Zhu Gengyu | 58 | 139 wpm |
| 13 | The review arrived before the parcel. Olist can get there first. | 7:50–8:00, then the 2-minute live demo | Zhu Gengyu | 36 | — |

Total: **1,143 words** of speech. At a normal 135 words a minute that is about **8:30**, so the time stamps in the notes (8:00) are optimistic: slides 7, 8 and 11 carry more words than their slots allow (155, 185 and 192 wpm). Either trim one sentence on each, or keep the live demo to about 1:30, so that the whole talk ends inside the ten minutes. The speaker split is the one used in rehearsal; any member can take any question.

**Clicks.** Every slide builds by itself when it opens (all animations are "with previous"; there are no mid-slide clicks). Press → once per slide. The footer pills (01 The Problem … 05 Takeaways, Q&A) are clickable and jump to the first slide of each chapter; on slide 13 the four "↺" pills jump back to slides 2, 4, 8 and 11 for questions, the three "Next up" cards and the app card open the live app, and the GitHub card opens the repository.

## Slide by slide

### 1. Catch the bad review before it's written
*Opening · 0:00–0:25 · Pan Pinyou*

**On screen.** Night-lights map of Olist's customers and sellers; the four moments of an order's life (checkout → promised date → delivery → review); the team; a QR code that opens the live app.

> 0:00–0:25 | Good morning/afternoon Professor and everyone. We are Team 10: Pan Pinyou, Li Yudan and Zhu Gengyu. Our project is not a story about chasing the fanciest algorithm — it is a detective story about timing: how to catch a 1-star e-commerce review BEFORE the customer has already written it.

**中文要点**

- 开场：我们是第10组（Pan Pinyou、Li Yudan、Zhu Gengyu）。
- 定调：这不是"比谁的算法更花哨"，而是一个关于**时机**的侦探故事——如何在顾客写下一星差评**之前**拦住它。
- 本页动画自动播放，不需要点击。

### 2. This review arrived before the parcel did
*01 The problem · 0:25–1:10 · Pan Pinyou*

**On screen.** The 2,267 km route from São Luís (MA) to São Gonçalo (RJ) on the map; five dated rows (16 Jul order → 7 Aug promised date passes → 9 Aug survey → 11 Aug 1-star review → 17 Aug parcel arrives); the punchline card: the review was already 5.8 days old.

> 0:25–1:10 | Let's start with one real customer in São Gonçalo. On 16 July 2018 the buyer ordered a R$65 health & beauty item from 2,267 km away. Olist promised delivery by 7 August. That day came and went — silence. Two days later, Olist's automated system emailed the buyer asking 'How was your order?' Frustrated and empty-handed, the buyer gave 1 star on 11 August. When the parcel finally arrived 10.5 days late on 17 August, the 1-star review was already nearly 6 days old.

**中文要点**

- 一个真实的测试集订单：7月16日下单（R$65 美妆个护商品，距离 2,267 公里），承诺 8 月 7 日送达。
- 承诺日过了没动静；8月9日 Olist 自动发出评价问卷；8月11日顾客还没收到货就打了**一星**。
- 8月17日包裹才到（迟到 10.5 天），此时差评已经写好 5.8 天了。

### 3. A broken promise turns into a bad review
*01 The problem · 1:10–1:50 · Pan Pinyou*

**On screen.** Two stat cards: delivered on time 9.2% leave a 1–2★ review (87,902 orders, 92.0%); delivered late 54.0% (7,659 orders, 8.0%); the 5.9× badge; the 1–2★ rate by arrival-vs-promise bucket (8.9% more than 10 days early, 9.7% 0–10 days early, 19.1% 1–3 days late, 61.3% 4–7 days late, 78.0% 8–14 days late, 78.6% more than 14 days late).

> 1:10–1:50 | Was this buyer just unlucky? We checked all 95,561 delivered orders. When a parcel arrives on time, only 9.2% of buyers leave a 1- or 2-star review. Once it crosses the promised date, bad reviews jump nearly six times to 54% — and past 4 days late, over 61% to 79% are unhappy. Only 1 in 12 parcels arrives late, yet they cause a full third of all bad reviews on Olist.

**中文要点**

- 不是个例：95,561 个已送达且有评价的订单里，准时到货差评率 9.2%，迟到后 54.0%，接近 6 倍。
- 迟到 1–3 天差评率翻倍到 19.1%，迟到 4 天以上 61%–79%。
- 迟到只占 1/12（8.0%）的包裹，却贡献了三分之一（34%）的差评。

### 4. For a late parcel, the review comes first
*01 The problem · 1:50–2:35 · Pan Pinyou*

**On screen.** Two lanes. On-time parcels (11 in 12; 66% of all 1–2★ reviews): 99.5% review after delivery, median 2.1 days after the delivery scan — a window to call. Late parcels (1 in 12; 34% of all 1–2★ reviews): the survey fires at the promised date, the 1★ is written a median 4.5 days before arrival, 87.3% of late 1–2★ reviews are already written when the parcel lands.

> 1:50–2:35 | Here is the twist that changed our entire project. Why not just call every risky order right after delivery? We checked WHEN reviews are actually written. For on-time orders — which still account for 66% of all bad reviews — buyers wait about 2.1 days after delivery before reviewing. That gives support a real 2-day window to call. But for late orders, Olist fires the survey on the promised date — so 87.3% of unhappy late buyers have ALREADY submitted their 1- or 2-star review before the box even lands! So the real question isn't just WHO is unhappy, it's WHEN Olist can still act.

**中文要点**

- 项目的转折点：我们对比了评价时间戳和签收时间。
- 准时订单（占差评的 66%）：顾客到货后约 2.1 天才写评价，客服有 2 天窗口可以打电话。
- 迟到订单：问卷在承诺日就发出，87.3% 的迟到差评在包裹到达前已经写好。问题不只是"谁会不满意"，而是"Olist 什么时候还来得及行动"。

### 5. Why promises break: a moving, shock-hit network
*02 Why & tools · 2:35–3:15 · Li Yudan*

**On screen.** Monthly late rate over 20 months (Black Friday 14.3%, February 2018 16.0%, March 2018 21.4%, June 2018 1.4%, August 2018 10.4%) with the unseen July–August 2018 test window marked; three cards: lead time 16.9 → 7.7 days and 75% of the time with the carrier; ~11 buffer days in the promise (9.8-day error) yet 21.4% late in March 2018; leaking a post-event column turns ROC-AUC 0.59 into a fake 1.00.

> 2:35–3:15 | Why couldn't Olist's existing calendar catch this? Because the ground underneath them kept moving. Average delivery time actually halved from 17 days to under 8 days — yet sudden shocks like the Feb–Mar 2018 carrier jam pushed lateness past 21%, and the May 2018 national truckers' strike forced Olist to pad promises up to 37 days. And if an analytics team accidentally leaks future timestamps or mixes dates randomly, you get a fake 1.00 AUC that collapses in production. So we tested everything strictly forward in time on unseen July–August 2018 orders.

**中文要点**

- 地面在动：平均配送时间从 17 天（2018年2月）降到不到 8 天（2018年8月）；75% 的时间在承运商手里。
- 冲击不可预见：2018 年 2–3 月的承运商拥堵把迟到率推到 21.4%；2018 年 5 月全国卡车司机罢工后 Olist 把承诺时窗从 19.5 天拉长到 36.8 天，6 月迟到率只有 1.4%。
- 防泄漏：用了事后才知道的字段，AUC 会从 0.59 变成假的 1.00；我们所有结果都只在未见过的 2018 年 7–8 月订单上、严格按时间向前测试。

### 6. Three moments before the review — three simple tools
*02 Why & tools · 3:15–4:00 · Li Yudan*

**On screen.** Three tool cards: at checkout (Operations) a delivery-time forecast; on the promised date (Customer Care) a calendar rule, no ML; at delivery (Customer Experience) a daily review-risk list of about ten buyers. The "too late" card: 12.8% of delivered orders (14.6% of all reviewed orders) end in a 1–2★. Footer: even a single decision tree captures 96% of the delivery-triage signal; everything ships in one scikit-learn + Streamlit app.

> 3:15–4:00 | That diagnosis gave us our solution map: three moments before the review, and one simple tool for each. First, at checkout, Operations uses a delivery-time forecast to quote honest dates and flag risky routes. Second, on the promised date, Customer Care doesn't need machine learning at all — just a calendar rule that messages the buyer and pauses the review survey the moment an order goes overdue. Third, at delivery, Customer Experience gets a daily top-10 call list for buyers most likely to leave 1–2 stars. Notice: even a single tree captures 96% of the signal. Choosing the right moment mattered far more than a fancy model.

**中文要点**

- 三个时刻、三个简单工具：下单时的配送天数预测（运营）；承诺日的日历规则，不需要机器学习（客服）；签收时的每日前 10 名差评风险名单（客户体验）。
- 单棵决策树就能拿到 96% 的分诊信号——选对时机比换更复杂的模型重要得多。
- 全部部署在一个 scikit-learn + Streamlit 应用里。

### 7. Honest dates at checkout — and why rushing loses money
*03 Proof & app · 4:00–4:45 · Li Yudan*

**On screen.** 12,630 unseen deliveries: Olist's promise is off by 9.79 days, our forecast by 3.90 (60% closer; GBDT 4.14, linear 4.18). What works: deriving the late risk from the day forecast lifts PR-AUC 2.9× vs 1.2× for a direct classifier (ROC-AUC 0.782 vs 0.584); the riskiest 10% hold 35% of the late parcels (3.5×), the slide-2 order ranked #23 of 238 on 16 July; a banner above 11% risk pre-warns 83% (5 in 6) of late parcels. What does not: at R$12 per expedite against R$40 per bad review, 30% precision is the break-even, and checkout flags plateau at 27–28%.

> 4:00–4:45 | Let's check the proof on unseen July and August 2018 orders. First, at checkout: Olist's padded promises miss actual arrival by nearly 10 days. Our forecast cuts that error by 60% down to 3.9 days — for free. It also catches 35% of late orders in the top 10%, including our opening São Gonçalo buyer at #23 that day. Just as important is what NOT to do: if express shipping costs R$12 against a R$40 bad review, you need 30% precision to break even — and checkout-only flags plateau at 27 to 28% because 75% of transit happens later with the carrier. So quote honest dates and send free warnings, don't burn cash rushing parcels blind.

**中文要点**

- 下单时：Olist 的承诺平均差 9.79 天，我们的预测差 3.90 天（缩小 60%），而且免费。
- 风险最高的 10% 订单包含 35% 的迟到包裹；开场那单当天排第 23/238。
- 不该做的事：加急 R$12、差评损失 R$40，需要 30% 的命中率才回本，而下单时的预警最多 27–28%——所以报真实日期、发免费提醒，不要盲目花钱加急。

### 8. Ten calls a day find 5× more unhappy customers
*03 Proof & app · 4:45–5:30 · Li Yudan*

**On screen.** 62 unseen test days, 12,855 reviewed orders. Daily top-10 list: 57 of 100 calls reach a real 1–2★ customer (57.2%; 35 not yet written, 22 already written while waiting; 614 calls over 62 days) against 11 of 100 at random (11.0% base rate); 5.2× at ten calls a day (68% precision at the top 5%); delivered parcels only: without the never-delivered orders the daily top-10 still hits 45.1% (base rate 9.6%), and on strictly on-time parcels the top 5% are 28% unhappy (3.5×); break-even at R$5 per call vs R$40 per review is 13 hits in 100.

> 4:45–5:30 | Now look at Moment 3 — at delivery. If a support team calls 10 orders at random, only 11 out of 100 calls reach an unhappy customer. With our daily top-10 list across all 62 unseen test days, 57 out of 100 calls reach a real 1- or 2-star buyer — over 5 times random! And here's the crucial check: some of those hits are orders that never arrived, and the promised-date rule is what reaches them. Drop every never-delivered order and the daily top-10 still hits 45 unhappy customers per 100 calls, against 10 at random; on strictly on-time parcels — where 99.5% of buyers haven't written yet and wait 2.1 days — the top 5% are 28% unhappy, three and a half times random. That catches broken, wrong, or high-freight items while we can still fix them.

**中文要点**

- 签收时：随机打 100 通电话只有 11 通找到差评顾客；按我们的每日前 10 名名单打，100 通里 57 通命中（62 个测试日、12,855 个有评价的订单），是随机的 5 倍多。
- 关键检验：57 个命中里有 26 个是从未送达的订单（由承诺日规则来处理）。去掉所有未送达订单，每日前 10 名仍有 45/100 命中（随机约 10）；只看严格准时到货的订单（99.5% 的人还没写评价、平均等 2.1 天），前 5% 有 28% 会差评，是基准的 3.5 倍。
- 盈亏平衡只需要 100 通里命中 13 通。

### 9. Replay 16 July 2018 blind — then reveal reality
*03 Proof & app · 5:30–6:15 · Zhu Gengyu*

**On screen.** Screenshot of the deployed app's *Replay a day* tab for 16 July 2018 (244 reviewed orders): 8 of 10 flagged orders really left 1–2★ (4 wrote while waiting for a late parcel, 4 never received the parcel: shipped, invoiced or cancelled); row 1 at 85% risk is the slide-2 order; across all 62 test days the top-10 list averages 57 hits per 100 calls. The ▶ Open Live App button and the screenshot open the app.

> 5:30–6:15 | Everything we just showed lives inside our deployed Streamlit web app, which we'll demo live in two minutes. In the 'Replay a day' tab, you pick any unseen day in July or August 2018 — here 16 July with 244 reviewed orders — get the blind morning call list first, and then click 'Reveal what actually happened'. 8 out of 10 flagged orders actually left 1–2 stars: 4 wrote while waiting for a late box, and 4 never received their parcel at all. And #1 on that list at 85% risk is our opening São Gonçalo buyer!

**中文要点**

- 前面所有内容都在我们部署好的 Streamlit 应用里，两分钟后现场演示。
- "Replay a day"：选 7 月 16 日（244 个有评价的订单），先看盲打的早晨名单，再点"Reveal"：前 10 名里 8 个真的差评——4 个在等迟到包裹时写的，4 个根本没收到货。
- 名单第 1 名（85% 风险）就是开场那位 São Gonçalo 的顾客。

### 10. Four changes Olist can launch on Monday
*04 Monday plan · 6:15–6:50 · Zhu Gengyu*

**On screen.** Four action cards: at checkout quote the honest date and warn above 11% risk (9.79 → 3.90 days; pre-warns 83%); on the promised date speak first and hold the review survey until arrival (closes the window behind 87.3% of late 1–2★); at delivery call the daily top-10 list (57% precision = 5.2× random; 45% on delivered parcels alone); the guardrail: do not pay R$12 to rush on a checkout flag (27–28% < 30% break-even). Footer: refit monthly on a rolling 12-month window and monitor the weekly late rate.

> 6:15–6:50 | So what should Olist actually launch on Monday? Four concrete moves: One, at checkout, quote our honest forecast and show a free warning banner above 11% risk. Two, on the promised date, automatically apologise and pause the review survey whenever an order goes overdue. Three, at delivery, have support call the daily top-10 risk list. And four, do NOT pay R$12 to rush orders off a checkout flag alone — wait for a live carrier scan.

**中文要点**

- 周一就能做的四件事：下单时报真实预测日期并在风险 ≥ 11% 时免费提醒；承诺日一过就自动道歉并暂停评价问卷；签收时客服打每日前 10 名名单。
- 第四件是"不做"：不要只凭下单时的预警就花 R$12 加急，等承运商的实时扫描停滞再说。
- 上线后每月用滚动 12 个月窗口重训，每周监控迟到率。

### 11. Worth R$775–R$1,466 per 1k orders — test before scaling
*04 Monday plan · 6:50–7:25 · Zhu Gengyu*

**On screen.** Net value ceiling at R$5 per call vs R$40 per bad review: R$775 per 1,000 delivered orders counting only the 35/100 hits not yet written (R$433 at ten calls a day), R$1,466 across all hits once the promised-date survey hold keeps late orders reachable. The honest catch: nobody in the 2016–2018 log was ever called. The 6-week randomised pilot: each morning's top 10 split 5 called / 5 control, 191 orders per arm; control 57.2%, break-even 44.7% (about 1 in 5), target 42.9% (1 in 4) at 80% power, +R$0.72 per call; Arm B randomises the promised-date apology and survey hold.

> 6:50–7:25 | Under a R$5 call against a R$40 bad-review cost, daily triage is worth R$775 to R$1,466 per thousand orders. Now here is our honest catch: in the historical dataset, nobody was ever called — so we know we find the right unhappy customers, not how often a call changes their mind. Instead of guessing, Olist can run a 6-week A/B pilot tomorrow using our app: take each morning's top 10, randomly call 5 and hold 5 as control. Converting just 1 in 5 breaks even at 44.7%, and in 6 weeks we have 80% statistical power to prove a 1-in-4 drop to 42.9% — plus Arm B testing the promised-date survey pause.

**中文要点**

- 价值：按每通电话 R$5、每个差评 R$40，每 1,000 单值 R$775（只算还没写差评的顾客）到 R$1,466（承诺日暂停问卷后所有命中都可触达）。
- 诚实的局限：历史数据里没有人被打过电话，所以我们知道"找得准"，不知道"电话能否改变主意"。
- 6 周随机试点：每天前 10 名随机打 5 个、留 5 个对照，每组 191 单；差评率从 57.2% 降到 44.7% 即回本，降到 42.9%（减少四分之一）可以在 80% 统计功效下被证明；B 组检验承诺日的道歉 + 暂停问卷。

### 12. Three lessons we would tell the next analytics team
*05 Takeaways · 7:25–7:50 · Zhu Gengyu*

**On screen.** Three lessons: timing beats fancy models (87.3% of late-order bad reviews were written before arrival); judge strictly on the future (fake 1.00 AUC vs 0.59 honest; refit monthly); design for blind spots (checkout flags plateau at 27–28%; offline logs cannot prove persuasion). Guardrail: risk scores for proactive care and honest dates only — never to penalise sellers (75% of delivery time is inside third-party carriers).

> 7:25–7:50 | Stepping back, three takeaways stood out. First, timing beats fancy models: checking WHEN reviews are written unlocked far more value than switching algorithms. Second, judge strictly on unseen future data and refit monthly as the network moves. Third, design honestly around blind spots — and never use delay scores to punish sellers for carrier delays beyond their control.

**中文要点**

- 三个教训：时机胜过花哨的模型；只用未见过的未来数据评判，并随网络变化每月重训；坦诚面对盲区。
- 治理红线：风险分数只用于主动客服和真实日期，绝不用来惩罚卖家——75% 的配送时间在第三方承运商手里。

### 13. The review arrived before the parcel. Olist can get there first.
*Close · 7:50–8:00, then the live demo · Zhu Gengyu*

**On screen.** Callback line; the four moments with their results (60% closer date; speak first + hold the survey; top-10 calls, 5.2×; saved before the 1★); QR codes and links to the live app and the GitHub repository; the three "Next up" demo cards (Replay a day, Predict live, Model card) and four "↺" pills that jump back to slides 2, 4, 8 and 11 for questions.

> 7:50–8:00 (+ 2:00 Live App Demo) | The review arrived before the parcel — with three simple interventions, Olist can get there first. Let's switch over to our live Streamlit app now to replay 16 July 2018 and score a live checkout order!

**中文要点**

- 收尾呼应开场：差评比包裹先到——用三个简单的干预，Olist 可以抢在前面。
- 切换到线上应用做 2 分钟演示（见下文路径）；本页的四个"↺"按钮可在问答时跳回第 2、4、8、11 页。

## Live demo path (about two minutes, after slide 13)

1. **Ten minutes before the talk** open <https://team10-it5006-olist.streamlit.app>. If Streamlit shows "This app has gone to sleep", press **Yes, get this app back up!** and wait about a minute. Open **⏮ Replay a day**: the defaults are purchase day 2018/07/16, capacity 10 and **Reveal what actually happened** switched on — switch Reveal **off** so the list can be shown blind first.
2. **Replay a day (about 50 seconds).** Click ▶ Open Live App on slide 9 or 13 (or Alt+Tab to the browser). Show the blind list ranked by review risk, then switch Reveal on: the KPI cards read 244 orders, 8 of 10 hits, 4 written before the parcel arrived, lead-time error 4.3 days against a promise off by 10.7 days; point at the **review written** column ("before the parcel" / "after delivery") and at row 1 — 85% risk, São Gonçalo (RJ), health_beauty: the order from slide 2.
3. **Predict (about 40 seconds).** Score the slide-2 order as it looked at checkout: seller state MA, customer state RJ, category health_beauty, 1 item, 1 seller, item total 64.99, freight 27.74, weight 400 g, largest item volume 2,964 cm³, purchase date 2018/07/16, promised date 2018/08/07, payment voucher (its larger payment), 1 instalment. The app returns its expected lead time, the risk of missing the promised date (on the test window this order scored 18% at checkout — #23 of 238 that day, above the 11% warning line) and, under that forecast, the review risk. Then move the promised date two weeks later and press Predict again: the risk of missing the promise falls — the checkout story in one move.
4. **Model card (about 20 seconds).** Scroll to *Sizing the outreach pilot*: capacity 10, assumed reduction 25%, power 80% → control-arm rate 57%, 191 orders per arm (about six weeks), +R$0.72 per contact, break-even effect 22%. Change the capacity or the effect to show the calculator move.
5. **If the network fails**, stay on slide 9: the screenshot is the same Replay screen with the same numbers, and the Predict and Model card numbers above are in the speaker's notes.

## Rehearsal checklist

- Present from the .pptx in PowerPoint (2016 or later is enough: the deck uses fade transitions and automatic builds only). The typeface is Segoe UI, which is installed on every Windows machine; on macOS it is substituted, so present from a Windows laptop or fall back to the PDF.
- Target 130–140 words a minute. Slides 7 and 11 are the dense ones: rehearse them with a timer, and keep the live demo under two minutes.
- Pan Pinyou: slides 1–4 · Li Yudan: slides 5–8 · Zhu Gengyu: slides 9–13 and the live demo. Any member can take any question below.
- Wake the app before the session (step 1 of the demo path); keep a second browser tab with the GitHub repository open.
- Slides must be uploaded by Wed 4 Nov 2026 23:59 (locked after that); the ZIP by Sun 8 Nov 2026 23:59; peer evaluation on Canvas by Sun 15 Nov 2026 23:59.

## Predicted questions

The professor grades the story (what was used to solve which problem), so expect questions about the insight, the decision and the evidence more than about algorithms. Each answer is about 20 seconds; the slide named on the right is the one to jump to (the "↺" pills on slide 13 reach slides 2, 4, 8 and 11 in one click; the footer pills reach the start of each chapter).

**Q1. In one sentence: what problem did you solve, and how?**  
*Go to: Slide 4*

> Bad reviews on Olist mostly follow broken delivery promises, and for late parcels the review is written before the parcel even arrives. We predict which promise will break and which customer is about to complain, and we put each prediction at the moment a team can still act: checkout, the promised date, or delivery.

中文要点：一句话：差评主要来自"承诺没兑现"，而且迟到包裹往往在到货前就被差评；我们预测哪些承诺会破、哪些顾客要差评，并把预测放到团队还能行动的三个时刻。

**Q2. How do you know the review was written before the parcel arrived?**  
*Go to: Slide 4*

> Olist's review table has two timestamps: when the survey was sent and when the customer answered it. Comparing the answer with the delivery timestamp, 87% of 1–2★ reviews on late parcels were answered before delivery, typically 4.5 days before. For 99% of late parcels the survey went out on or after the promised date. Even if a carrier logged delivery a day late, a 4.5-day gap would survive.

中文要点：评价表有"问卷发送时间"和"回答时间"两个时间戳；和签收时间比：迟到包裹的1–2星差评 87% 在到货前写好（中位数提前4.5天）；99% 的迟到包裹在承诺日当天或之后收到问卷。即使签收时间记晚一天，结论也不变。

**Q3. Did you cherry-pick the São Gonçalo order?**  
*Go to: Slide 9 / live app*

> It is the first name on the list for 16 July, the default day of our Replay tab, so anyone can reproduce it. 16 July is a good day, eight hits out of ten, and we say so on the slide. Across all 62 test days the top ten average 57 hits per 100 calls, and 15 of the 62 days reach eight or more.

中文要点：不是挑的：它是 Replay 默认日期（7月16日）名单上的第1名，任何人都能复现；那天是"好日子"（10中8），我们在幻灯片上说明了；62个测试日平均每100通57个命中，其中15天达到8个及以上。

**Q4. Sending the survey after the parcel arrives: isn't that gaming the ratings?**  
*Go to: Slides 10–11*

> No review is suppressed or filtered: every customer is still asked, only after the parcel arrives, so the rating describes the whole experience instead of a parcel still on the road. The contact on the promised date is an apology with a new date, which is ordinary good service. Because it could still shift ratings, we would test it as its own arm of the pilot rather than roll it out.

中文要点：不删除、不筛选评价，每位顾客照样被邀请评价，只是等到货后再问，评价反映完整体验；承诺日当天的联系是道歉+新日期，属于正常服务。因为仍可能影响评分，所以作为试点的独立一组来检验，而不是直接推行。

**Q5. Why does a bad review matter to Olist if the parcel is already late?**  
*Go to: Slides 3–4*

> Bad ratings cost sellers future sales (Deshpande & Pendem, 2023), and Olist earns from its sellers. And two thirds of bad reviews come from parcels that arrived on time; those customers write about two days after delivery, so they can still be reached.

中文要点：差评会减少卖家未来的销量（Deshpande & Pendem, 2023），而 Olist 靠卖家赚钱；另外三分之二的差评来自准时到达的包裹，到货后约2天才写，仍然来得及联系。

**Q6. Which models did you use, and why those?**  
*Go to: Slide 6*

> Three families, each run from its simplest version: linear models; trees up to histogram gradient boosting; and voting and stacking ensembles of the two, all in scikit-learn, chosen on the most recent months before the test. Deployed: a voting blend for delivery days, a late flag derived from the forecast against the promise, and a stacked model for review risk. A single decision tree already gets 96% of the way, which is why we say the model was not the bottleneck: the moment and the action were.

中文要点：三类模型（线性；决策树→随机森林→HistGradientBoosting；投票/堆叠集成），全部用 scikit-learn，按测试前最近几个月的表现来选。部署：交付天数用 Voting，迟到预警由"预测 vs 承诺"推导，差评风险用 Stacking。单棵决策树就达到最佳效果的96%——瓶颈不在模型，而在时机和行动。

**Q7. How accurate is it, in business terms?**  
*Go to: Slides 7–8*

> At checkout our delivery date is off by 3.9 days against 9.8 for Olist's promise, and the riskiest 10% of orders hold 35% of the late parcels. At delivery, ten calls a day reach 57 unhappy customers per 100 calls, 35 of them before they write, against 11 and 9 when calling at random.

中文要点：下单时：交付日期误差3.9天，承诺是9.8天；风险最高10%的订单包含35%的迟到包裹。签收时：每天10通电话，每100通命中57个将差评的顾客（35个还没写），随机只有11个（9个）。

**Q8. Why not simply expedite the risky parcels?**  
*Go to: Slide 7*

> Because it loses money. An expedite costs about R$12 against R$40 for a bad review, so it pays only if at least 30% of flagged parcels would really be late. Our flags never get there; at best they are right 28% of the time. Warning the customer and quoting an honest date cost almost nothing, so that is what the flag is for. If expediting ever cost less than about R$11, we would revisit.

中文要点：会亏钱：加急约R$12、差评损失R$40，需要至少30%的预警包裹真的迟到才回本；我们最好也只有28%。预警顾客、给真实日期几乎零成本，所以预警用在这里；如果加急成本低于约R$11再考虑。

**Q9. Where do R$5 and R$40 come from?**  
*Go to: Slide 11*

> They are stated assumptions: Olist publishes no costs. So we show the arithmetic and the break-even instead: a call pays when at least 13 of 100 calls reach a would-be bad reviewer. We are at 35 counting only customers who have not yet written, so the conclusion survives either cost being off by a factor of two.

中文要点：这是明示的假设（Olist 没公开成本），所以我们给出盈亏平衡点：每100通电话至少命中13个就划算；只算还没写差评的我们也有35个，任何一个成本估错一倍，结论仍然成立。

**Q10. Why ten calls a day?**  
*Go to: Slide 8*

> It is a capacity assumption, roughly one agent's spare time, and the app lets you change it. The list is worked from the top, so precision falls as you call more: 63% at five a day, 57% at ten, 44% at twenty. All of these stay far above the 13% break-even.

中文要点：这是产能假设（大约一名客服的空闲时间），App 里可以调整；名单从风险最高往下打，打得越多命中率越低：每天5通63%、10通57%、20通44%，都远高于13%的盈亏线。

**Q11. What if another truckers' strike happens?**  
*Go to: Slide 5*

> No order attribute can see it coming, so the models cannot either, and we say so. That is why the plan has a weekly late-rate monitor and monthly refits. After the 2018 strike Olist lengthened its promised dates, and June 2018 had the lowest late rate in the data, 1.4%.

中文要点：单个订单的特征看不到罢工，模型也预测不了，我们明确说明这个局限；所以方案包括每周监控迟到率、每月重训。2018年罢工后 Olist 延长了承诺日期，6月迟到率是全数据最低的1.4%。

**Q12. How would Olist use this day to day?**  
*Go to: Slides 6 and 10; live app*

> Operations shows the forecast date at checkout and warns the riskiest buyers. Customer care runs a calendar rule every day: parcels past their promised date get an apology and a new date. The customer-experience team opens the app each morning and calls the top ten. The app scores one order or a whole file, and replays any test day.

中文要点：运营：下单页显示预测日期并预警高风险买家；客服：每天按日历规则，过了承诺日还没到的包裹先道歉、给新日期；客户体验团队：每天早上打开 App，打名单前10个。App 可以给单个订单或整个文件打分，也能回放任意测试日。

**Q13. How do you know the test was fair, with no leakage?**  
*Go to: Slide 5*

> We split by time: trained on January 2017 to April 2018, chose models on May and June, and scored July and August once. A guard removes every column that is only known after the prediction moment; one such column on its own pushes ROC-AUC from 0.59 to 1.00. Customers who appear on both sides are checked separately, and 60 automated checks rerun all of this.

中文要点：按时间切分：2017年1月–2018年4月训练，5–6月选模型，7–8月只测一次；防泄漏机制去掉预测时刻之后才知道的列（只漏一列 ROC 就从0.59变成1.00）；两边都出现的顾客单独检查；60个自动化检查。

**Q14. Isn't the review model just predicting lateness?**  
*Go to: Slide 8*

> Lateness is its strongest signal, and late or never-delivered parcels fill the top of the list. But it is not only lateness: without the orders that never arrived, the daily top-10 on delivered parcels still reaches 45 unhappy customers per 100 calls against a 9.6% base rate, and on strictly on-time deliveries the top 5% are 28% unhappy, 3.5 times their base rate (ROC-AUC 0.65). Those are the customers the delivery-day calls are for. Because it leans on lateness, we use it for triage, never to penalise a seller for a carrier's delay.

中文要点：迟到确实是最强信号，迟到和未送达的包裹排在名单最前面；但不只是迟到：去掉从未送达的订单，已送达包裹上的每日前 10 名仍有 45/100 命中（基准 9.6%）；只看严格准时到货的订单，前 5% 有 28% 会差评（基准的 3.5 倍，ROC 0.65）——签收时打电话正是针对这群人。也因为它依赖迟到信息，只用于分诊，绝不用来惩罚卖家。

**Q15. How much is it worth, and how would you prove it?**  
*Go to: Slide 11*

> Up to R$775 per 1,000 delivered orders at the best call volume, or R$433 at ten calls a day, counting only customers we can still reach, and only if a call actually prevents the bad review. The data cannot tell us that, because nobody was ever called. So: six weeks; each morning take the ten riskiest deliveries, call five at random and leave five. If calls cut the 1–2★ rate from 57% to about 45% they pay for themselves, and 191 customers per arm detect a one-in-four drop with 80% power.

中文要点：最多每1000单 R$775（最佳通话量），每天10通是 R$433，只算还能联系上的顾客，而且前提是电话真的能阻止差评。数据里没人被打过电话，所以要做实验：6周，每天前10名随机打5个、不打5个；差评率从57%降到约45%就回本；每组191人可以以80%的把握检测出降低四分之一。

**Q16. Some of the orders on your top-10 list were never delivered. How is that an "at delivery" call?**  
*Go to: Slides 8–9; report Appendix M; model card*

> Correct, and we show it on slide 9: of the eight hits on 16 July, four were orders that never arrived. Across the test window 284 of the 12,855 reviewed orders, 2.2%, were cancelled or still in transit when the data was extracted; the model reads their missing delivery fields as risk, and they are 37% of the daily top ten — 26 of the 57 hits per 100 calls, with another 22 hits written while waiting for a late parcel. A delivery-day call cannot reach those two groups; the promised-date contact can, because an order that never arrives still passes its promised date. That is why the two interventions are a pair. Restricted to delivered parcels the list still reaches 45 per 100, 16 of them before the review is written, and on strictly on-time parcels the top 5% are 28% unhappy. The decomposition is in Appendix M and on the model card.

中文要点：是的，第 9 页也写了：7 月 16 日的 8 个命中里 4 个是从未送达的订单。测试集 12,855 个有评价的订单里有 284 个（2.2%）被取消或提取数据时仍在途中；模型把它们缺失的配送字段读成风险，它们占每日前 10 名的 37%——每 100 通里 57 个命中中的 26 个，另有 22 个是等迟到包裹时写的。签收日的电话够不到这两类人，但承诺日的联系够得到（没送到的订单同样会过承诺日），所以两项干预是一对。只看已送达包裹，名单仍有 45/100 命中（16 个在写评价之前）；严格准时的订单前 5% 有 28% 会差评。分解表在报告附录 M 和模型卡上。

### Quick answers

- **What was the hardest part?** Choosing the moment. Our first plan was to call customers at delivery, before they review. Checking the two review timestamps showed that for late parcels the review usually comes first, which is why the promised date became its own moment.  
  中文：最难的是选时机：最初计划是签收时打电话；检查两个时间戳后发现迟到包裹的差评往往先写，所以加了"承诺日"这个时刻。
- **Why does Olist pad its promised date?** A promised date is a commercial decision: Cui et al. (2024) show that promising faster raises sales but increases returns and hurts retention. Our forecast gives Olist an honest second number to use alongside it.  
  中文：承诺日期是商业决策：Cui 等（2024）发现承诺更快能提高销量，但会增加退货、降低留存；我们的预测给 Olist 一个诚实的参考数字。
- **What would you do with more time?** Run the two-arm pilot, test sending the survey after arrival, add carrier tracking data, and refit weekly during peaks such as Black Friday.  
  中文：做两组试点实验、测试到货后再发问卷、加入承运商物流追踪数据、在黑五等高峰期每周重训。

## Where every number on the slides comes from

| Number on the slides | Source (`report/results/` unless stated) |
|---|---|
| Story order: dates, 2,267 km, R$64.99 + R$27.74 (rounded to R$65 + R$28), 10.5 days late, review 5.8 days before arrival, 85% risk, #1 of 244, #23 of 238 | `story_order.csv`, `predictions_p1b_test.csv`, `review_timing_test_orders.csv`, the raw `olist_order_items_dataset.csv` (`src/review_timing.py`) |
| 9.2% / 54.0% 1–2★ rate (5.9×); 87,902 on-time and 7,659 late of 95,561 delivered, reviewed orders; 34% of bad reviews from late parcels; 66% / 34% split; 87.3% written before arrival; 99.5% of on-time 1–2★ written after delivery; 4.5 days before / 2.1 days after; 12.8% of delivered orders and 14.6% of all reviewed orders end 1–2★ | `review_timing.csv`; `report/stats_phase3.txt` (`tim_*`) |
| 1–2★ rate by arrival bucket (8.9% more than 10 days early, 9.7% 0–10 days early, 19.1% 1–3 days late, 61.3% 4–7, 78.0% 8–14, 78.6% more than 14) | `review_timing_lateness.csv` |
| Monthly late rate (14.3% Nov 2017, 16.0% Feb 2018, 21.4% Mar 2018, 1.4% Jun 2018, 10.4% Aug 2018); lead time 16.9 → 7.7 days; promised window 19.5 → 36.8 days around the strike; ~11 buffer days; 75% of delivery time with the carrier | `lateness_monthly.csv`, `lateness_weekly.csv`, `lateness_strike.csv` |
| ROC-AUC 0.589 → 1.000 with one leaked column | `report/stats.txt` (Phase 1 leakage demonstration) |
| Promise MAE 9.79 vs forecast 3.90 (60% closer); GBDT 4.14, linear 4.18; 12,630 delivered test orders | `reg_p1a.csv`, `predictions_p1a_test.csv` |
| Late flag: ROC-AUC 0.782 vs 0.584, PR-AUC lift 2.9× vs 1.2×; riskiest 10% hold 35% of late parcels (3.5×); 83% (5 in 6) recall at the 11% threshold; precision plateau 27–28% below the 30% break-even (R$12 / R$40) | `clf_p1b.csv`, `capacity_p1b.csv`, `cost_p1b.csv`, `capacity_operating_points.csv`; costs in `src/config.py` |
| Ten calls a day: 57 hits per 100 (57.2%), 35 reachable / 22 already written, 614 calls over 62 days; 11 per 100 at random (11.0% base); 13 per 100 to break even; 5.2× at ten a day, 68% at the top 5% | `capacity_p2_daily.csv`, `capacity_p2.csv`, `review_timing_capacity.csv`; R$5 / R$40 in `src/config.py` |
| 12,855 reviewed test orders; 8 of 10 on 16 July; 57% average over the 62 days | `predictions_p2_test.csv` with `review_timing_test_orders.csv` |
| Delivered parcels only: daily top-10 45.1 hits per 100 (base rate 9.6%, 16 reachable, 29 already written); strictly on-time parcels: top 5% 28.0% unhappy (3.5×, ROC-AUC 0.651); 284 never-delivered orders = 26 of the 57 hits per 100 | `review_timing_subsets.csv`, `review_timing_composition.csv` |
| Single decision tree = 96% of the signal (PR-AUC 0.396 vs 0.414) | `clf_p2.csv` |
| R$775 (reachable) / R$1,466 (all) per 1,000 delivered orders; R$433 at ten calls a day | `review_timing_value.csv` |
| Pilot: control 57.2%, break-even 44.7% (a 21.9% relative reduction, about 1 in 5), target 42.9% (1 in 4), 191 per arm, 6 weeks, 80% power, +R$0.72 per call | `pilot_design.csv`, `report/stats_phase3.txt` (`pilot_k10_*`) |
