# Phase 3 presentation: script, demo path and predicted questions

Group 10 · IT5006 · 10 minutes · deck: [`Team10_Phase3_slides.pptx`](Team10_Phase3_slides.pptx) (present from the .pptx; the [PDF](Team10_Phase3_slides.pdf) is a static copy that also contains the 8 backup slides).

The English text below is word-for-word the speaker notes in the deck (Presenter View, Alt+F5). Both are generated from one file, so they cannot drift apart.

## The story in one paragraph

The course asks what we used to solve which problem, not which models we tried. So the talk follows one real held-out order whose 1-star review was written six days *before* the parcel arrived, shows that this is the pattern (a broken promise turns into a bad review, and for late parcels the review usually comes first), and builds the solution around **when** Olist can still act: at checkout (honest dates, early warnings), on the promised date (speak first, survey after arrival) and at delivery (ten calls a day). Models appear in one line on slide 6; the evidence is told in business units (days, calls, reais), the live app replays the very day the story order was bought, and the close admits what the data cannot know and proposes the experiment that would.

**Arc:** hook (one order) → pattern → idea (three moments) → why it is hard → what we built → does it work (checkout, delivery, live) → what Olist should do → what it is worth and how to prove it → lessons → callback to the hook.

## Timing and speakers

| # | Slide | Time | Speaker | Words | Pace |
|---|---|---|---|---|---|
| 1 | Catch the bad review before it's written | 0:00–0:23 | Pan Pinyou | 49 | 128 wpm |
| 2 | This review arrived before the parcel did | 0:23–1:18 | Pan Pinyou | 114 | 124 wpm |
| 3 | A broken promise turns into a bad review | 1:18–2:07 | Pan Pinyou | 106 | 130 wpm |
| 4 | Step in before the review — at three moments | 2:07–2:57 | Pan Pinyou | 112 | 134 wpm |
| 5 | The ground kept moving under the data | 2:57–3:49 | Li Yudan | 107 | 123 wpm |
| 6 | What we built: an early-warning system | 3:49–4:33 | Li Yudan | 97 | 132 wpm |
| 7 | Honest dates, and an early-warning list | 4:33–5:22 | Li Yudan | 107 | 131 wpm |
| 8 | Ten calls a day find the unhappy customers | 5:22–6:08 | Li Yudan | 103 | 134 wpm |
| 9 | Replay a day the models never saw | 6:08–7:08 | Zhu Gengyu | 116 | 116 wpm |
| 10 | Four changes Olist can make on Monday | 7:08–7:55 | Zhu Gengyu | 104 | 133 wpm |
| 11 | Up to R$775 per 1,000 orders — if a call helps | 7:55–8:49 | Zhu Gengyu | 120 | 133 wpm |
| 12 | What we learned | 8:49–9:23 | Zhu Gengyu | 73 | 129 wpm |
| 13 | The review arrived before the parcel (close and questions) | 9:23–9:43 | Zhu Gengyu | 44 | 132 wpm |

Total: **1252 words**, about 9.3 minutes of speech at a normal 135 words a minute. The schedule adds a few seconds for each of the two clicks, the Morph transitions and the two handovers (slide 4 → 5 and slide 8 → 9), and ends at **9:43**, leaving a short buffer inside the ten minutes.

**Clicks.** Every slide builds by itself after the transition; press → once to move on. Only two slides wait for a click mid-slide: slide 2 (the punchline) and slide 9 (“8 of 10”). The footer chapter names are clickable, and on slide 13 the eight question buttons jump to the backup slides (each has “← Back to questions”).

## Slide by slide

### 1. Catch the bad review before it's written  
*Opening · 0:00–0:23 · Pan Pinyou*

**On screen.** Night-lights map of Olist's customers (blue) and sellers (gold); the four-stage journey checkout → promised date → delivery → review. Builds automatically.

> Good evening, we are Group 10. Our data: about a hundred thousand orders from Olist, a Brazilian online marketplace. Our project is about something every online shopper knows: a late parcel, and the bad review that follows. Our question: can Olist catch that bad review before it is written?

**中文要点**

- 开场：我们是第10组，数据是 Olist（巴西电商平台）约10万个订单。
- 抛出问题：能不能在差评写出来**之前**就拦住它？
- 本页动画全部自动出现，不需要点击。

### 2. This review arrived before the parcel did  
*01 The problem · 0:23–1:18 · Pan Pinyou*

**On screen.** The journey icons glide into a vertical timeline (Morph) — and the star lands above the truck. Map of the 2,267 km route. Five dated rows build automatically; **one click** reveals the punchline.

> Let me start with one real order from our test data. On the sixteenth of July 2018, a customer in São Gonçalo, near Rio, bought a health-and-beauty product from a seller in São Luís, more than two thousand kilometres away. Olist promised delivery by the seventh of August. That date passed; the parcel was still on the road. Two days later Olist emailed the review survey, as it does for almost every parcel past its promised date. On the eleventh the customer answered, still waiting: one star. **[CLICK]** And the parcel? It arrived on the seventeenth, ten days late. By then the review was almost six days old. The review arrived before the parcel did.

**中文要点**

- 一个真实测试订单：7月16日下单，承诺8月7日到；承诺日过了，包裹还在路上。
- 8月9日 Olist 发出评价问卷 → 8月11日顾客打了**一星**（还没收到货）。
- 【点击】8月17日才到，晚了10天——差评已经写了快6天。
- 金句：**差评比包裹先到**（The review arrived before the parcel did）。

### 3. A broken promise turns into a bad review  
*01 The problem · 1:18–2:07 · Pan Pinyou*

**On screen.** 9% vs 54%, bars growing early → late, tiles: 87% written before arrival, 1 in 12 parcels late.

> This is not one unlucky customer. Across ninety-five thousand delivered orders, about nine per cent of customers give one or two stars when the parcel is on time. When it is late, fifty-four per cent do, and the later, the angrier: past a week late, almost four in five. One parcel in twelve is late, yet those parcels produce a third of all bad reviews. And the timestamps showed us something new: eighty-seven per cent of those bad reviews are written before the parcel arrives, because the survey goes out once the promised date has passed. By the time the parcel lands, the damage is done.

**中文要点**

- 不是个例：准时到达 9% 差评，迟到 54%；晚一周以上，接近五分之四。
- 每12个包裹有1个迟到，却贡献了三分之一的差评。
- 时间戳的发现：迟到包裹的差评 **87% 在到货前就写好了**，因为承诺日一过问卷就发出。
- 为什么重要：差评会伤害卖家未来销量（Deshpande & Pendem, 2023）。

### 4. Step in before the review — at three moments  
*02 Our approach · 2:07–2:57 · Pan Pinyou*

**On screen.** The journey grows into three big moments; three team cards (Operations, Customer care, Customer experience).

> That told us where to act: before the review, at three moments. At checkout, operations can see which promises are likely to break, and quote an honest date or warn the buyer early. On the promised date, if the parcel isn't there, customer care can reach the customer first, and ask for the review after it arrives: no model needed, just the calendar. And at delivery: two thirds of bad reviews come from on-time parcels, written about two days later, so there is still time to call the riskiest. Today Olist finds out only at the last step; our job was to move that knowledge earlier. Li Yudan will show you how.

**中文要点**

- 所以要在差评**之前**、在三个时刻行动：
- 下单时（运营）：哪些承诺会破？给真实日期、提前预警。
- 承诺日（客服）：包裹没到就先联系道歉、给新日期，到货后再请评价——**不需要模型，只需要日历**。
- 签收时（客户体验）：三分之二差评来自准时包裹，到货后约2天才写，还来得及打电话。
- 交接：“Li Yudan will show you how.”

### 5. The ground kept moving under the data  
*02 Our approach · 2:57–3:49 · Li Yudan*

**On screen.** Monthly late-rate chart sweeping left → right with the shocks coloured; three reasons it is hard.

> Thank you. Why is this hard? Because the ground kept moving. Olist halved its delivery time within six months, from seventeen days to eight, so last year's patterns are wrong this year. Lateness jumps with events no single order reveals: Black Friday, a carrier slowdown in February and March, a national truckers' strike in May. And it is easy to fool yourself: give a model one column from the future, like the actual delivery date, and it scores a perfect one-point-zero, and is useless. So we trained on 2017 to April 2018, and judged everything on July and August 2018, two months the models had never seen.

**中文要点**

- 难点一：半年内交付时间减半（17天→8天），去年的规律今年就不对了。
- 难点二：突发事件单个订单看不出来——黑五、2–3月承运商变慢、5月卡车司机罢工。
- 难点三：很容易骗自己——给模型一个“未来列”ROC 就是1.00，但没用。
- 所以：2017年–2018年4月训练，2018年7–8月（模型没见过的两个月）评估。

### 6. What we built: an early-warning system  
*02 Our approach · 3:49–4:33 · Li Yudan*

**On screen.** Input panel → three components → “Under the hood” bar (the only place models appear).

> So what did we build? An early-warning system in three pieces. First, a delivery forecast that runs at checkout: how many days the parcel will take, and how likely the promise is to break. Second, a review-risk score that runs at delivery: how likely this customer is to give one or two stars. Third, a live web app that anyone on the team can open. Under the hood these are standard tree-ensemble models from scikit-learn. Every team has good models; the hard part, and the part that made the difference, was choosing the moment and the action.

**中文要点**

- 我们做了一个三件套的预警系统：下单时的交付预测、签收时的差评风险分、团队都能用的网页App。
- 模型只用一句话带过：scikit-learn 的树集成模型。
- 核心观点：**大家都有好模型，真正难的是选对时机和行动**（呼应教授的要求）。

### 7. Honest dates, and an early-warning list  
*03 Does it work? · 4:33–5:22 · Li Yudan*

**On screen.** Two bars (9.8 vs 3.9 days) wipe in, “60% closer”, then the 35% and < 30% tiles.

> Does it work? At checkout, first: honest dates. Olist's promise is off by almost ten days on average, because it is padded on purpose. Our forecast is off by under four: sixty per cent closer. Second: an early-warning list. The riskiest ten per cent of orders hold thirty-five per cent of all late parcels, three and a half times better than chance; our São Gonçalo order was already number twenty-three that day. But we also learned what not to do: even the riskiest parcels are late less than thirty per cent of the time, so paying twelve reais to rush them would lose money. Warn, don't expedite.

**中文要点**

- 下单时效果一：真实日期——Olist 承诺平均差将近10天（故意留余量），我们差不到4天，准60%。
- 效果二：预警名单——风险最高的10%订单包含35%的迟到包裹（随机的3.5倍）；开场订单当天排第23/238。
- 反直觉的发现：别花钱加急——最高风险的包裹迟到率也不到30%，R$12加急要30%才回本。**预警，不加急**。

### 8. Ten calls a day find the unhappy customers  
*03 Does it work? · 5:22–6:08 · Li Yudan*

**On screen.** Two 10×10 waffles: our list (dark = reachable, pale = already wrote) vs random calls.

> At delivery: imagine the team can make ten calls a day. Calling at random, eleven of every hundred calls reach someone about to give one or two stars. With our list it is fifty-seven: five times better. But remember the timestamps: twenty-two of those fifty-seven had already written while waiting for a late parcel; thirty-five can still be reached. That is why the promised date matters for late parcels, while the call list catches the on-time ones. At five reais a call against forty per bad review, calling pays from thirteen hits per hundred. We reach thirty-five. Zhu Gengyu will show it live.

**中文要点**

- 签收时：假设每天能打10通电话。随机打：每100通只有11个将差评的顾客；用我们的名单：57个（5倍）。
- 但其中22个在等迟到包裹时已经写了差评，35个还来得及——这就是承诺日那个时刻重要的原因。
- 成本：R$5一通电话 vs R$40一个差评，每100通命中13个就回本，我们有35个。
- 交接：“Zhu Gengyu will show it live.”

### 9. Replay a day the models never saw  
*03 Does it work? · 6:08–7:08 · Zhu Gengyu*

**On screen.** Screenshot of the live app's Replay tab; **one click** reveals “8 of 10”; button opens the live app.

> Here it is, live: the Replay tab. Because the data ends in 2018, it replays any day from the two test months as the team would have seen it, with the outcomes hidden until you reveal them. Let's take the sixteenth of July, ten calls. **[CLICK]** Eight of the ten customers on the list really did give one or two stars. Four could still have been reached at delivery; four had already written while waiting for a late parcel. And number one on the list, at eighty-five per cent risk, is the order we started with. To be fair, this was a good day: across all sixty-two test days, the list averages fifty-seven out of a hundred.

**中文要点**

- 现场演示 Replay 页：回放测试期任意一天，结果先隐藏，再揭晓。
- 7月16日、10通电话 →【点击】10个里8个真的给了差评：4个还来得及，4个已经写了。
- 名单第1名（风险85%）就是开场的那个订单。
- 主动说明：这是“好日子”，62个测试日平均是57%。

### 10. Four changes Olist can make on Monday  
*04 What Olist should do · 7:08–7:55 · Zhu Gengyu*

**On screen.** Four action cards.

> So what should Olist do on Monday? Four things. One: show customers our forecast, not the padded promise. It is free, and sixty per cent closer to reality. Two: when a promise breaks, speak first: apologise with a new date, and send the survey after the parcel arrives. Three: every morning, call the ten riskiest deliveries, at about five reais a call. And four, the one we didn't expect: don't pay to expedite on our flag; at twelve reais a parcel it never pays back. Behind all four: refit the models monthly and watch the weekly late rate. The next strike won't announce itself.

**中文要点**

- 周一就能做的四件事：
- ① 下单页显示我们的预测日期，不用留余量的承诺（免费，准60%）。
- ② 承诺破了先开口：道歉+新日期，到货后再发评价问卷。
- ③ 每天早上打风险最高的10个签收订单（约R$5一通）。
- ④ 意外的结论：不要按预警花钱加急（R$12永远不回本）。背后：每月重训，每周监控迟到率。

### 11. Up to R$775 per 1,000 orders — if a call helps  
*04 What Olist should do · 7:55–8:49 · Zhu Gengyu*

**On screen.** R$775 tile, “Nobody was ever called”, then the three-step experiment diagram.

> What is it worth? With our stated assumptions, five reais per call and forty per bad review, the call list is worth up to seven hundred and seventy-five reais per thousand delivered orders, counting only customers we can still reach. But here is the honest catch: in the data, nobody was ever called, so history cannot tell us whether a call changes a review. So we recommend an experiment, not a rollout. Every morning, take the ten riskiest deliveries; call five at random, leave five. After six weeks, if the called group's one- and two-star rate drops from fifty-seven to about forty-five per cent, the calls pay for themselves. Run the same split for late parcels on their promised date.

**中文要点**

- 价值：按明示假设（R$5/通，R$40/差评），每1000单最多 R$775（只算还能联系上的顾客）；每天10通是 R$433。
- 坦白：数据里从来没人被打过电话，历史数据无法证明电话能改变评价。
- 所以建议是**实验而不是直接推行**：每天前10名随机打5个、不打5个，6周后比较差评率；57%降到约45%就回本。
- 第二组实验：迟到包裹在承诺日当天联系。

### 12. What we learned  
*04 What Olist should do · 8:49–9:23 · Zhu Gengyu*

**On screen.** Three lessons and the limits strip.

> Three things we learned. First: timing beats modelling. Every team can fit a good model; our biggest gain was discovering when the review gets written. Second: judge on the future. Delivery got twice as fast, and models chosen on old months would have misled us. Third: say what the data can't know. Strikes, slowdowns, and whether a call actually helps. That is why our plan ends with monitoring, monthly refits and an experiment.

**中文要点**

- 三点收获：① 时机胜过建模；② 用“未来”来评判（交付快了一倍，老数据选的模型会误导）；③ 说清数据不知道的（罢工、电话是否有效）→ 监控、重训、实验。

### 13. The review arrived before the parcel (close and questions)  
*Close · 9:23–9:43 · Zhu Gengyu*

**On screen.** Callback line, QR codes, and the clickable question navigator (8 buttons → backup slides).

> So: the review arrived before the parcel. With honest dates, a word before the survey and ten calls a day, Olist can get there first. Thank you. The app and the code are one scan away, and we are happy to take your questions.

**中文要点**

- 呼应开头：差评比包裹先到；有了真实日期、问卷前的一句话、每天10通电话，Olist 可以抢先一步。
- 谢谢，欢迎提问；提问时点右侧按钮跳到对应备份页，备份页有“Back to questions”按钮返回。

## Live demo path (slide 9)

1. **Ten minutes before the talk** open <https://team10-it5006-olist.streamlit.app>. If Streamlit shows “This app has gone to sleep”, press the wake-up button and wait about a minute. Open **Replay a day**: the defaults are purchase day 2018/07/16, capacity 10 and **Reveal what actually happened** switched on. To reveal live, switch Reveal off before the talk and on again at the click.
2. On slide 9, click **▶ Open the live app** (or Alt+Tab to the browser). Point at the KPI card “8 of 10”, the **review written** column (“before the parcel” / “after delivery”), and row 1: 85% risk, the order from slide 2.
3. Optional, only if ahead of time: **Predict** tab, move the **Promised delivery date** later and press Predict; the risk of missing the promise falls, which is the checkout story in one move.
4. Alt+Tab back to PowerPoint and click once to reveal “8 of 10” (or just continue).
5. **If the network fails**, stay on the slide: the screenshot is the same screen, and the click still reveals the result.

## Rehearsal checklist

- Present from the .pptx in PowerPoint 2019 or Microsoft 365 (Morph needs it; older versions fall back to fades). Fonts are embedded.
- Target a normal pace of 130–140 words a minute; mark the two clicks and the two handovers in rehearsal.
- Pan Pinyou: slides 1–4 · Li Yudan: slides 5–8 · Zhu Gengyu: slides 9–13 and the live demo. Any member can take any question below.
- Wake the app before the session; keep a second browser tab with the GitHub repository open.
- Slides must be uploaded by Wed 4 Nov 2026 23:59 (locked after that); the ZIP by Sun 8 Nov 2026 23:59; peer evaluation on Canvas by 15 Nov.

## Predicted questions

The professor grades the story (what was used to solve which problem), so expect questions about the insight, the decision and the evidence more than about algorithms. Each answer is about 20 seconds; the backup slide named on the right is one click away from slide 13.

**Q1. In one sentence: what problem did you solve, and how?**  
*Go to: Slide 4*

> Bad reviews on Olist mostly follow broken delivery promises, and for late parcels the review is written before the parcel even arrives. We predict which promise will break and which customer is about to complain, and we put each prediction at the moment a team can still act: checkout, the promised date, or delivery.

中文要点：一句话：差评主要来自“承诺没兑现”，而且迟到包裹往往在到货前就被差评；我们预测哪些承诺会破、哪些顾客要差评，并把预测放到团队还能行动的三个时刻。

**Q2. How do you know the review was written before the parcel arrived?**  
*Go to: Backup: When reviews are written*

> Olist's review table has two timestamps: when the survey was sent and when the customer answered it. Comparing the answer with the delivery timestamp, 87% of 1–2★ reviews on late parcels were answered before delivery, typically 4.5 days before. For 99% of late parcels the survey went out on or after the promised date. Even if a carrier logged delivery a day late, a 4.5-day gap would survive.

中文要点：评价表有“问卷发送时间”和“回答时间”两个时间戳；和签收时间比：迟到包裹的1–2星差评 87% 在到货前写好（中位数提前4.5天）；99% 的迟到包裹在承诺日当天或之后收到问卷。即使签收时间记晚一天，结论也不变。

**Q3. Did you cherry-pick the São Gonçalo order?**  
*Go to: Slide 9 / live app*

> It is the first name on the list for 16 July, the default day of our Replay tab, so anyone can reproduce it. 16 July is a good day, eight hits out of ten, and we say so on the slide. Across all 62 test days the top ten average 57 hits per 100 calls, and 15 of the 62 days reach eight or more.

中文要点：不是挑的：它是 Replay 默认日期（7月16日）名单上的第1名，任何人都能复现；那天是“好日子”（10中8），我们在幻灯片上说明了；62个测试日平均每100通57个命中，其中15天达到8个及以上。

**Q4. Sending the survey after the parcel arrives: isn't that gaming the ratings?**  
*Go to: Backup: Limits and next steps*

> No review is suppressed or filtered: every customer is still asked, only after the parcel arrives, so the rating describes the whole experience instead of a parcel still on the road. The contact on the promised date is an apology with a new date, which is ordinary good service. Because it could still shift ratings, we would test it as its own arm of the pilot rather than roll it out.

中文要点：不删除、不筛选评价，每位顾客照样被邀请评价，只是等到货后再问，评价反映完整体验；承诺日当天的联系是道歉+新日期，属于正常服务。因为仍可能影响评分，所以作为试点的独立一组来检验，而不是直接推行。

**Q5. Why does a bad review matter to Olist if the parcel is already late?**  
*Go to: Slides 3–4*

> Bad ratings cost sellers future sales (Deshpande & Pendem, 2023), and Olist earns from its sellers. And two thirds of bad reviews come from parcels that arrived on time; those customers write about two days after delivery, so they can still be reached.

中文要点：差评会减少卖家未来的销量（Deshpande & Pendem, 2023），而 Olist 靠卖家赚钱；另外三分之二的差评来自准时到达的包裹，到货后约2天才写，仍然来得及联系。

**Q6. Which models did you use, and why those?**  
*Go to: Backup: The models under the hood*

> Three families, each run from its simplest version: linear models; trees up to histogram gradient boosting; and voting and stacking ensembles of the two, all in scikit-learn, chosen on the most recent months before the test. Deployed: a voting blend for delivery days, a late flag derived from the forecast against the promise, and a stacked model for review risk. A single decision tree already gets 96% of the way, which is why we say the model was not the bottleneck: the moment and the action were.

中文要点：三类模型（线性；决策树→随机森林→HistGradientBoosting；投票/堆叠集成），全部用 scikit-learn，按测试前最近几个月的表现来选。部署：交付天数用 Voting，迟到预警由“预测 vs 承诺”推导，差评风险用 Stacking。单棵决策树就达到最佳效果的96%——瓶颈不在模型，而在时机和行动。

**Q7. How accurate is it, in business terms?**  
*Go to: Backup: Checkout numbers / Delivery numbers*

> At checkout our delivery date is off by 3.9 days against 9.8 for Olist's promise, and the riskiest 10% of orders hold 35% of the late parcels. At delivery, ten calls a day reach 57 unhappy customers per 100 calls, 35 of them before they write, against 11 and 9 when calling at random.

中文要点：下单时：交付日期误差3.9天，承诺是9.8天；风险最高10%的订单包含35%的迟到包裹。签收时：每天10通电话，每100通命中57个将差评的顾客（35个还没写），随机只有11个（9个）。

**Q8. Why not simply expedite the risky parcels?**  
*Go to: Backup: Checkout numbers / Where the money comes from*

> Because it loses money. An expedite costs about R$12 against R$40 for a bad review, so it pays only if at least 30% of flagged parcels would really be late. Our flags never get there; at best they are right 29% of the time. Warning the customer and quoting an honest date cost almost nothing, so that is what the flag is for. If expediting ever cost less than about R$11, we would revisit.

中文要点：会亏钱：加急约R$12、差评损失R$40，需要至少30%的预警包裹真的迟到才回本；我们最好也只有29%。预警顾客、给真实日期几乎零成本，所以预警用在这里；如果加急成本低于约R$11再考虑。

**Q9. Where do R$5 and R$40 come from?**  
*Go to: Backup: Where the money comes from*

> They are stated assumptions: Olist publishes no costs. So we show the arithmetic and the break-even instead: a call pays when at least 13 of 100 calls reach a would-be bad reviewer. We are at 35 counting only customers who have not yet written, so the conclusion survives either cost being off by a factor of two.

中文要点：这是明示的假设（Olist 没公开成本），所以我们给出盈亏平衡点：每100通电话至少命中13个就划算；只算还没写差评的我们也有35个，任何一个成本估错一倍，结论仍然成立。

**Q10. Why ten calls a day?**  
*Go to: Backup: Delivery numbers*

> It is a capacity assumption, roughly one agent's spare time, and the app lets you change it. The list is worked from the top, so precision falls as you call more: 63% at five a day, 57% at ten, 44% at twenty. All of these stay far above the 13% break-even.

中文要点：这是产能假设（大约一名客服的空闲时间），App 里可以调整；名单从风险最高往下打，打得越多命中率越低：每天5通63%、10通57%、20通44%，都远高于13%的盈亏线。

**Q11. What if another truckers' strike happens?**  
*Go to: Slide 5; Backup: Limits and next steps*

> No order attribute can see it coming, so the models cannot either, and we say so. That is why the plan has a weekly late-rate monitor and monthly refits. After the 2018 strike Olist lengthened its promised dates, and June 2018 had the lowest late rate in the data, 1.4%.

中文要点：单个订单的特征看不到罢工，模型也预测不了，我们明确说明这个局限；所以方案包括每周监控迟到率、每月重训。2018年罢工后 Olist 延长了承诺日期，6月迟到率是全数据最低的1.4%。

**Q12. How would Olist use this day to day?**  
*Go to: Slides 6 and 10; Backup: The app, tab by tab*

> Operations shows the forecast date at checkout and warns the riskiest buyers. Customer care runs a calendar rule every day: parcels past their promised date get an apology and a new date. The customer-experience team opens the app each morning and calls the top ten. The app scores one order or a whole file, and replays any test day.

中文要点：运营：下单页显示预测日期并预警高风险买家；客服：每天按日历规则，过了承诺日还没到的包裹先道歉、给新日期；客户体验团队：每天早上打开 App，打名单前10个。App 可以给单个订单或整个文件打分，也能回放任意测试日。

**Q13. How do you know the test was fair, with no leakage?**  
*Go to: Backup: How we tested*

> We split by time: trained on January 2017 to April 2018, chose models on May and June, and scored July and August once. A guard removes every column that is only known after the prediction moment; one such column on its own pushes ROC-AUC from 0.59 to 1.00. Customers who appear on both sides are checked separately, and 60 automated checks rerun all of this.

中文要点：按时间切分：2017年1月–2018年4月训练，5–6月选模型，7–8月只测一次；防泄漏机制去掉预测时刻之后才知道的列（只漏一列 ROC 就从0.59变成1.00）；两边都出现的顾客单独检查；56个自动化检查。

**Q14. Isn't the review model just predicting lateness?**  
*Go to: Backup: Delivery numbers / Limits*

> Lateness is its strongest signal, and late parcels fill the top of the list. But on on-time parcels alone it still ranks well: ROC-AUC 0.71, and the top 5% of on-time deliveries are 54% unhappy, 5.7 times their base rate. Those are exactly the customers the delivery-day calls are for. Because it leans on lateness, we use it for triage, never to penalise a seller for a carrier's delay.

中文要点：迟到确实是最强信号，迟到包裹排在名单最前面；但只看准时到达的包裹它依然有效：ROC 0.71，前5%里有54%会差评（是基准的5.7倍）——签收时打电话正是针对这群人。也因为它依赖迟到信息，只用于分诊，绝不用来惩罚卖家。

**Q15. How much is it worth, and how would you prove it?**  
*Go to: Slide 11; Backup: Where the money comes from*

> Up to R$775 per 1,000 delivered orders at the best call volume, or R$433 at ten calls a day, counting only customers we can still reach, and only if a call actually prevents the bad review. The data cannot tell us that, because nobody was ever called. So: six weeks; each morning take the ten riskiest deliveries, call five at random and leave five. If calls cut the 1–2★ rate from 57% to about 45% they pay for themselves, and 191 customers per arm detect a one-in-four drop with 80% power.

中文要点：最多每1000单 R$775（最佳通话量），每天10通是 R$433，只算还能联系上的顾客，而且前提是电话真的能阻止差评。数据里没人被打过电话，所以要做实验：6周，每天前10名随机打5个、不打5个；差评率从57%降到约45%就回本；每组191人可以以80%的把握检测出降低四分之一。

### Quick answers

- **What was the hardest part?** Choosing the moment. Our first plan was to call customers at delivery, before they review. Checking the two review timestamps showed that for late parcels the review usually comes first, which is why the promised date became its own moment.  
  中文：最难的是选时机：最初计划是签收时打电话；检查两个时间戳后发现迟到包裹的差评往往先写，所以加了“承诺日”这个时刻。
- **Why does Olist pad its promised date?** A promised date is a commercial decision: Cui et al. (2024) show that promising faster raises sales but increases returns and hurts retention. Our forecast gives Olist an honest second number to use alongside it.  
  中文：承诺日期是商业决策：Cui 等（2024）发现承诺更快能提高销量，但会增加退货、降低留存；我们的预测给 Olist 一个诚实的参考数字。
- **What would you do with more time?** Run the two-arm pilot, test sending the survey after arrival, add carrier tracking data, and refit weekly during peaks such as Black Friday.  
  中文：做两组试点实验、测试到货后再发问卷、加入承运商物流追踪数据、在黑五等高峰期每周重训。

## Where every number on the slides comes from

| Number on the slides | Source (report/results unless stated) |
|---|---|
| Story order: dates, 2,267 km, 10.5 days late, review 5.8 days before arrival, 85% risk, #1 of 244, #23 of 238 | `story_order.csv`, `predictions_p1b_test.csv` (from `src/review_timing.py`) |
| 9% / 54% 1–2★ rate; 1 in 12 late; 34% of bad reviews; 87% written before arrival; 99% surveyed on/after the promised date; 4.5 days before / 2.1 days after | `review_timing.csv` |
| 1–2★ rate by lateness bucket (9%, 10%, 19%, 61%, 78%, 79%) | `review_timing_lateness.csv` |
| Monthly late rate; 17 → 8 days lead time | `lateness_monthly.csv` |
| ROC-AUC 0.589 → 1.000 with one leaked column | `report/stats.txt` (Phase 1 leakage demonstration) |
| Promise MAE 9.79 vs forecast 3.90 (60% closer) | `reg_p1a.csv` |
| Riskiest 10%: 35% of late parcels (3.5×); late-flag precision ≤ 0.29 < 0.30 break-even | `capacity_p1b.csv`, `cost_p1b.csv` |
| Ten a day: 57 hits / 35 reachable / 22 already written; random 11 / 9; 13 to break even | `review_timing_capacity.csv`; R$5/R$40 in `src/config.py` |
| 57% average over 62 test days; 8 of 10 on 16 July | `predictions_p2_test.csv` with `review_timing_test_orders.csv` |
| R$775 (reachable) / R$1,466 (all) per 1,000; R$433 / R$853 at ten a day | `review_timing_value.csv`; printed by `src/review_timing.py` |
| Pilot: 57% → 45%, 191 per arm, 6 weeks, 80% power | `pilot_design.csv`, `report/stats_phase3.txt` (`pilot_k10_*`) |
| On-time parcels alone: ROC-AUC 0.71, top 5% 54% unhappy (5.7×) | `review_timing_subsets.csv` |
