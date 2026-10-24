---
type: Note
title: Where every number and every step comes from
description: Each claim in project 02 marked as published practice, convention, or Claude's own choice, with what is still missing. Written 2026-10-19.
status: open
---

# Where every number and every step comes from

Every figure and every design decision in this project falls into one
of three boxes. The third box is the honest one.

| Box | Meaning |
|---|---|
| **Published** | a named source states it |
| **Convention** | widely used in industry, no derivation behind it |
| **Chosen** | Claude picked it; nothing supports the exact value |

Nothing here is settled. Items in the third box are candidates for
removal or for grounding, on Rafael's order.

---

## 1. The loop itself

The six steps are not one methodology. They are two established
patterns joined together, and the joins are mine.

| Step | Pattern | Box |
|---|---|---|
| flag, decide, label | the investigator feedback loop of card-fraud detection | Published |
| watch | performance and input monitoring from credit-risk practice | Convention |
| retrain, promote | champion and challenger from credit scoring | Convention |

**Flag, decide, label.** In card fraud, a model ranks transactions, a
fixed number go to human investigators, and their verdicts come back as
labels for the next training round. This is the setting Carcillo, Le
Borgne, Caelen and Bontempi describe and measure in
[Streaming active learning strategies for real-life credit card fraud
detection](https://arxiv.org/pdf/1804.07481). They score with Top-100
precision, for the same reason this project scores the weekly top five:
the review budget, not accuracy, is what a desk actually has.

The companion handbook,
[Reproducible Machine Learning for Credit Card Fraud Detection](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook)
(Le Borgne, Siblini, Lebichot and Bontempi, ULB, 2022), is the source
for validating forward through time and for labels that arrive late.

**Champion and challenger.** The incumbent model keeps serving while a
candidate is measured against it, and a winner is promoted. Described
by [Oracle](https://docs.oracle.com/en/industries/financial-services/ofs-analytical-applications/credit-risk-analytics/81210/emoug/champion-challenger.html),
[IBM](https://ibm.com/docs/en/SS69YH_8.2.1/cads_manager_ddita/model_management/thick/champion_challenger.html)
and [SAS](https://support.sas.com/documentation/cdl/en/edmug/67015/HTML/default/p060qw8wi2u6mfn18ca2bvpl5hz8.htm).

Two caveats, both against this project. Those are vendor documents, not
regulatory ones; a formal account of model governance would start from
supervisory guidance such as the Federal Reserve's SR 11-7, which this
project has not read. And in the usual champion-challenger setup a
share of live traffic goes to the challenger. Here nothing is split:
both models score the same weeks offline. That is a weaker test and the
documents do not say it is acceptable.

**What no source covers.** The order of the six steps, and that drift
alone may trigger a retraining. Those are mine.

---

## 2. The error rates the simulation plants

These come from project 01, which plants the errors this project hunts.

| Claim | Figure | Box | Source |
|---|---|---|---|
| duplicate or erroneous payments | 1.5% of disbursements | Published, second-hand | APQC, measure 105905, as quoted by [Corpay](https://www.corpay.com/resources/blog/duplicate-payment-detection) and [CFO.com](https://www.cfo.com/news/metric-of-the-month-detect-and-prevent-duplicate-or-erroneous-payments/656852/) |
| the same, best and worst quartile | 0.8% and 2.0% | Published, second-hand | [Corpay](https://www.corpay.com/resources/blog/duplicate-payment) |
| freight invoices carrying an error | 5% to 10% | Claimed in the plan | Tompkins Ventures, **not verified in this pass** |

Two problems with the first row. Nobody has read APQC's own page; every
figure above is quoted by a vendor. And APQC's number is a share of
*disbursements*, while this project applies it per *invoice*. Those are
not the same denominator.

---

## 3. The numbers that govern the loop

| Name | Value | Box | What it rests on |
|---|---|---|---|
| `K`, invoices ranked first each week | 5 | Chosen | about as many as the simulation plants per week, so a perfect model scores 1 |
| `DESK`, invoices checked by hand | 25 | Chosen | nothing |
| `LINE`, the drift threshold | 0.25 | Convention | the usual band: under 0.1 stable, 0.1 to 0.25 minor, over 0.25 a real shift ([R `scorecard`](https://www.rdocumentation.org/packages/scorecard/versions/0.4.4/topics/perf_psi)) |
| `RUN`, weeks under the floor | 3 | Chosen | nothing |
| `CHALLENGE_WEEKS` | 4 | Chosen | nothing |
| `DRIFT_WEEKS` | 8 | Chosen | nothing |
| the floor | measured | Published method | the tenth percentile of the champion's own weekly precision |

**On 0.25.** It is a rule of thumb, not a derived threshold, and it is
criticised in the literature. Potgieter, Fourie and colleagues show the
constant ignores sample size: it fires needlessly on small samples and
stays quiet on real shifts in large ones, and they propose a
chi-squared alternative instead
([arXiv 2307.11878](https://arxiv.org/pdf/2307.11878v4)). This project
uses the rule of thumb anyway, with a week against eight weeks, and
says so here rather than pretending the number is principled.

**On 5, 25, 3, 4 and 8.** None of these is measured or sourced. Each
was chosen because it sounded like a working week. The floor is the one
number in this table that is derived from data.

---

## 4. Claims that must be removed or grounded

Tracked in [removals.md](removals.md).

1. "Two people" at the desk. No headcount was decided or measured.
   Out of the README since 2026-10-19.
2. "5,600 invoices a year, about 200 wrong." Out of the README since
   2026-10-19. The invented years it described were dropped; the run
   now read holds 1,003 invoices and 35 errors.

---

## 5. What this project does not claim

- It is not validated against any real accounts-payable data. Every
  invoice is invented by project 01.
- The analyst is played by the answer key, so the feedback loop is
  measured against a perfect reviewer who never errs and never tires.
- No supervisory or audit standard has been read, and none is met.

## Bibliography

- Carcillo, F., Le Borgne, Y.-A., Caelen, O., Bontempi, G. *Streaming
  active learning strategies for real-life credit card fraud
  detection: assessment and visualization.* arXiv:1804.07481.
  https://arxiv.org/pdf/1804.07481
- Le Borgne, Y.-A., Siblini, W., Lebichot, B., Bontempi, G.
  *Reproducible Machine Learning for Credit Card Fraud Detection: a
  Practical Handbook.* Université Libre de Bruxelles, 2022.
  https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
- Potgieter, C. et al. *A statistical approach to population stability
  measurement.* arXiv:2307.11878. https://arxiv.org/pdf/2307.11878v4
- APQC, Open Standards Benchmarking, measure 105905, duplicate or
  erroneous payments, as quoted by Corpay and CFO.com. The primary page
  has not been read.
- Oracle Financial Services, *Champion Challenger.*
  https://docs.oracle.com/en/industries/financial-services/ofs-analytical-applications/credit-risk-analytics/81210/emoug/champion-challenger.html
- SAS, *Champion and challenger models in Enterprise Decision Manager.*
  https://support.sas.com/documentation/cdl/en/edmug/67015/HTML/default/p060qw8wi2u6mfn18ca2bvpl5hz8.htm
- IBM, *Champion challenger overview.*
  https://ibm.com/docs/en/SS69YH_8.2.1/cads_manager_ddita/model_management/thick/champion_challenger.html

### Not yet read, and needed

- APQC's own benchmark page, for the duplicate-payment figure and its
  denominator.
- Tompkins Ventures, for the freight invoice error rate.
- Siddiqi, N. *Credit Risk Scorecards*, the PSI section, for where the
  0.25 band actually comes from.
- Federal Reserve SR 11-7, *Guidance on Model Risk Management*, for
  what promoting a model is supposed to require.
