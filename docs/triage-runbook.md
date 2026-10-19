---
type: Runbook
title: Closing the triage loop
description: Every step left in project 02, test first, for someone who has never seen this code.
status: stable
---

# Closing the triage loop

You are finishing project 02. Read this whole page before you touch a
file. Do the steps in order. Do not skip a step because it looks easy.

## What this project is

Project 01 invents an importer's invoices, mails them, reads them back
from PDF and writes an answer key. Project 02 reads that output and
scores every invoice for how likely it is to be an error. The point of
the work left is not a better score. It is a **loop**: the model flags,
an analyst decides, the decision becomes a new label, the model is
retrained, and a better model takes over. Grafana shows all of it.

## Rules you must follow

1. **Test first, always.** Write the test. Run it. See it fail for the
   right reason. Then write the code. Then run the whole suite.
2. **Never change a test to make it pass.** If a test is wrong, say so
   and stop.
3. **No chart files, ever.** Every chart of projects 01 and 02 is drawn
   in Grafana, from Prometheus gauges. You write gauges, not pictures.
4. **Nothing reads the future.** A feature or a label computed at week 9
   may use week 8 and earlier only.
5. **Seed 5 is sacred.** `year-seed5` is the held-out year. Nothing is
   tuned, chosen or inspected on it. You may score it once per step.
6. **Run the whole suite between steps**, not just the new test.
7. **Nothing you run may take longer than ten minutes.** If it does,
   stop it, write down how long it was going to take, and carry on.
8. Comments explain **why**, never what. Match the voice of the files
   around you: plain words, short sentences.

## Where things are

```text
02-invoice-anomalies/
├── run.py                      features, walk, compare, score, metrics, ui
├── mlflow.db                   the local MLflow store (SQLite)
├── data/features/*.csv         cached feature tables, safe to delete
└── src/
    ├── data/load.py            reads one dataset from project 01
    ├── features/build.py       COLUMNS: the feature names, in order
    ├── evaluation/
    │   ├── metrics.py          precision_at_k, recall_at, recall_by_kind,
    │   │                       twin_alarms, average_precision; K = 5
    │   ├── reading.py          perfect data against what the reader got
    │   └── splits.py           week_of, known_labels, prequential
    └── models/
        ├── rules.py            RULES: (name, test, weight)
        ├── train.py            MODELS, report, compare, table_for
        ├── lifecycle.py        walk_forward, register_if_better
        ├── scoring.py          rows(): the dashboard table
        └── exporter.py         publish(): the Prometheus gauges
```

Tests mirror `src/` under `tests/`.

## First, make it run

```bash
cd ../01-invoice-pipeline
python run.py dataset year 1 2 3 4 5
python run.py dataset drift 7
cd ../02-invoice-anomalies
python -m pip install -r requirements.txt
python -m pytest -q
```

The suite must be green before you start. If it is not, fix that first
and nothing else.

---

# Step 1. Put the customs errors on the scoreboard

**Why.** Project 01 now plants two more kinds, `duty_rate_wrong` and
`broker_fee_over`. The numbers printed in `README.md` were measured
before they existed, so they are wrong today.

**Test first.** `tests/evaluation/test_metrics.py`, a new test:

- Build a small frame with `week`, `score`, `is_anomaly`, `anomaly`,
  `lookalike_of`, holding one `duty_rate_wrong`, one `broker_fee_over`
  and one `fee_amended` twin.
- Assert `recall_by_kind` returns a key for each of the two kinds.
- Assert `twin_alarms` returns a key for `fee_amended`.

**Then.** Delete `data/features/*.csv` so the features are built again.
Run:

```bash
python run.py features
python run.py walk
python run.py compare
```

**Done when.** `compare` prints four models, and the MLflow run holds a
`recall_duty_rate_wrong` and a `recall_broker_fee_over` metric. Write
the new table into `README.md`, replacing the one dated 2026-10-12.
Keep the date line honest: change it to the day you ran it.

---

# Step 2. Prove the score is real

**Why.** A number nobody can break is not evidence. Two tests break it
on purpose.

**Test first.** New file `tests/models/test_honesty.py`:

1. **The leak test.** Copy a feature table, add a column that is the
   label itself, train LightGBM on it, score the held-out weeks.
   Assert average precision is above 0.99. A model that cannot cheat
   when handed the answer is not learning at all.
2. **The shuffle test.** Shuffle `is_anomaly` within the table, train,
   score. Assert average precision lands near the base rate, under
   0.15. A model that still scores well on nonsense is reading
   something it should not.

Use a small table: one seed, the first twelve weeks. Both tests must
run in under a minute each.

**Done when.** Both pass, and the suite is green.

---

# Step 3. The analyst's verdict becomes a label

**Why.** Today `scoring.rows()` invents the analyst's decision from the
answer key, for display. Nothing feeds it back. That feedback is the
whole project.

**New file** `src/models/feedback.py`.

**Test first.** New file `tests/models/test_feedback.py`:

1. `labels_from(decisions, as_of)` takes the scored table and a date.
   - A row decided `confirmed` on or before `as_of` becomes label 1.
   - A row decided `dismissed` on or before `as_of` becomes label 0.
   - A row still `pending` is **dropped**, not labelled 0.
   - A row never flagged is label 0: nobody looked, and it is the
     normal case.
2. `labels_from` must refuse a decision dated after `as_of`. Assert it
   raises, with a message naming the row.
3. A table with no decisions at all returns the labels
   `splits.known_labels` would give. The loop must degrade to today's
   behaviour, not break.

**Then** write the module. Keep it pure: a frame in, a frame out, no
files, no MLflow.

**Done when.** The three tests pass and `splits.known_labels` is still
used by `train.with_known_labels`. Do not rewrite that path yet.

---

# Step 4. Watch the precision week by week

**Why.** You cannot retrain on a feeling. You need a number per week
that says the model is slipping.

**New file** `src/evaluation/watch.py`.

**Test first.** New file `tests/evaluation/test_watch.py`:

1. `weekly(scored)` returns one row per week: the week, the number
   flagged, the number confirmed, and `precision` as confirmed divided
   by the decided flags. Pending rows count in neither.
2. A week where every flag is still pending has `precision` as null,
   not zero. Assert it, and assert the row is still present.
3. `slipping(weekly, floor, run)` returns True only when `run`
   consecutive finished weeks sit below `floor`. One bad week is noise.
   Test: floor 0.5, run 3, a series of 0.8, 0.4, 0.9, 0.3, 0.3, 0.3
   returns True; the same series without the last 0.3 returns False.

Use `floor = 0.60` and `run = 3` as the module defaults, and say in the
docstring why: the desk checks five a week, so three straight weeks
below three-in-five is a real fall, not a quiet fortnight.

---

# Step 5. Notice the world changing

**Why.** Precision falls after the damage. Drift in the features shows
up first.

**Add to** `src/evaluation/watch.py`.

**Test first**, in `tests/evaluation/test_watch.py`:

1. `drift(reference, recent)` returns one number per feature in
   `features.build.COLUMNS`: the population stability index between the
   two frames.
2. Identical frames give 0 for every feature. Assert it exactly, to
   four decimal places.
3. A feature shifted hard gives above 0.25, the usual line for "this
   has moved". Build the shifted frame in the test; do not read a file.
4. `drifted(scores, line=0.25)` returns the names over the line, in
   order, worst first.

**Done when.** The four tests pass, and nothing in `watch.py` imports
MLflow or writes a file.

---

# Step 6. Retrain, and only promote a winner

**Why.** This is the step a client cares about. A model that is replaced
on evidence is worth more than a model with a better score.

**Add to** `src/models/lifecycle.py`.

**Test first.** New file `tests/models/test_retraining.py`:

1. `retrain(table, as_of, feedback)` trains LightGBM on the weeks up to
   `as_of`, using the labels from step 3, and returns the fitted model
   and the rows it saw. Assert it never saw a row after `as_of`.
2. `challenge(model, held_out)` scores the held-out year and returns the
   same dictionary `train.report` returns. Assert the keys match
   `train.report` exactly, so the two can be compared.
3. `register_if_better` already exists and already refuses a model that
   does not beat the champion. Add one test: a challenger that scores
   **equal** to the champion stays challenger. A tie is not a win.

**Then.** Add a command to `run.py`:

```bash
python run.py retrain [DATASET]
```

It must print, in plain words, four lines: the weeks it trained on, the
champion's score, the challenger's score, and which one serves now.

**Done when.** Running it twice in a row promotes nothing the second
time, and says so.

---

# Step 7. The whole story, on one command

**Why.** A demo nobody can run is not a demo.

**Add to** `run.py` a command `loop`, over the drift dataset
(`drift-seed7`, 26 weeks, where the anomaly rates double at week 13).

For each week in order:

1. Score the week with the current champion.
2. Build the analyst's decisions, as of that week's end.
3. Turn the decisions into labels (step 3).
4. Append the week to the weekly watch (step 4) and the drift
   measure (step 5).
5. If `slipping` or `drifted` fires, retrain and challenge (step 6).
6. Write the whole state out, every week, as the scored CSV the
   exporter reads.

**Test first.** New file `tests/runs/test_loop.py`, on a cut-down table
of eight weeks:

1. The loop visits every week once, in order.
2. No retraining happens before the floor is breached.
3. After a week where the labels are made deliberately bad, a
   retraining is recorded.
4. The run is reproducible: the same seed twice gives the same
   sequence of events.

**Ten-minute rule.** If 26 weeks takes longer than ten minutes, cut the
loop to the first 16 and write down what the full run would cost.

---

# Step 8. Put it on the screen

**Why.** Everything above is invisible until Grafana draws it.

**Add to** `src/models/exporter.py`, test first in
`tests/models/test_exporter.py`. Four new gauges:

| Gauge | Labels | Holds |
|---|---|---|
| `anomaly_weekly_precision` | none | the last finished week's precision |
| `anomaly_feature_drift` | `feature` | the stability index, per feature |
| `anomaly_retrainings` | none | how many retrainings have happened |
| `anomaly_champion_version` | none | the registry version serving now |

Keep the four that exist. Each new gauge needs a test that publishes a
small frame and reads the value back out of the registry.

**Then** add one Grafana panel per gauge to project 01's dashboard file,
`01-invoice-pipeline/monitoring/grafana/dashboards/pipeline.json`, in a
seventh block titled `7. The model on duty`. Match the style of the six
blocks already there: same colours, same legend placement, no new
colours. The dashboard is the home dashboard, served from the file, so
reload `http://localhost:3000` and look at it before you say it works.

**Done when.** You have looked at the panels with your own eyes, with
the loop running, and the retraining count went up on screen.

---

# Step 9. Write it down and publish

1. Rewrite `README.md` around the loop, not the score table. Order:
   what the desk's day looks like, the loop, the numbers, how to run it.
   The numbers stay, but they stop being the headline.
2. Take one screenshot of the Grafana block from step 8 into
   `docs/monitoring.png`. No other image.
3. Add `docs/reading-order.md`, the same shape as project 01's.
4. Publish the repository as `invoice-anomalies`. `git` is not on the
   PATH on this machine; the GitHub CLI is installed and signed in, so
   create and push with `gh`.
5. Add an entry to the wiki `log.md` saying what was built and when.

## Done, for the whole project

- The suite is green and every step above has its tests.
- `python run.py loop` runs end to end and ends with a promotion.
- Grafana shows the fall, the retraining and the recovery.
- The README leads with the loop.
- The repository is public.
