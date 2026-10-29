# Upwork portfolio item

**Title:** Invoice Anomaly Ranking, and the Honest Finding That the
Clerk's Checks Beat the Models

**Role:** Data scientist and ML engineer

**Skills:** Python, Machine Learning, MLOps, scikit-learn, LightGBM,
MLflow, Model Evaluation

**Media:** `panels.png`, `monitoring.png`

**Description:**

An importer's cost department had 1,003 filed invoices and 35 of them
billed wrong. Nobody reads a thousand pages looking for 35, so the
invoices are ranked worst first and the desk checks the top few each
week.

The work follows CRISP-DM end to end: the business problem, the data,
one row per invoice built only from what was knowable on its own date,
the clerk's twelve checks as a baseline, then models trained on the
errors already found, then a time split that trains on earlier weeks
and tests on later ones.

**What it found is not the flattering answer.** The clerk's hand checks
came out ahead of every model:

| Model | Average precision | 95% interval |
|---|---|---|
| Clerk's checks | 0.53 | 0.23 to 0.84 |
| LightGBM | 0.03 | 0.02 to 0.05 |
| Isolation forest | 0.13 | 0.02 to 0.42 |
| ECOD | 0.14 | 0.02 to 0.44 |

And the intervals overlap, so even that lead is reported as not proven.
Ten test errors cannot separate two models, and saying so is part of
the work. The published result is the measured one, not the one that
would look better on a slide.

What the ranking is still worth: of the 32 errors in the 786 invoices
it scored week by week, 23 reached a weekly top five. Read from the
top, the first 25 invoices carry 7 of the 10 errors, against 1 expected
by chance.

Also built, and also reported honestly: the retraining loop runs, logs
and retrains on drift, but on 24 invoices a week it never promoted a
challenger. The project makes two separate claims and keeps them apart,
that the system operates and that the model finds errors.

- 212 automated tests
- Bootstrap intervals on every figure
- MLflow tracking, Prometheus gauges, a weekly walk-forward loop

Built for teams that would rather know what their model cannot do.
