---
type: Runbook
title: Closing the triage loop
description: Steps 1 to 9 are done. Three faults remain, each with its test. Rewritten 2026-10-19.
status: stable
---

# Closing the triage loop

The first nine steps are done and published. This page is now the
second pass: three faults, found by review, each with the test that
proves it fixed.

Read this whole page before touching a file.

## Rules you must follow

1. **Test first, always.** The test comes from Claude, already written
   and already failing. You make it pass. You do not write it.
2. **Never change a test.** If a test looks wrong, stop and say so.
3. **Never touch anything on the protected list below.**
4. **No chart files.** Charts live in Grafana, drawn from Prometheus
   gauges.
5. **Nothing reads the future.** A feature or label at week 9 may use
   week 8 and earlier only.
6. **Seed 5 is sacred.** `year-seed5` is scored, never tuned on.
7. **Run the whole suite between steps.** Not just the new test.
8. **Nothing runs longer than ten minutes.** If it would, stop it, write
   down the cost, and carry on.
9. Comments say **why**, never what. Short sentences, plain words.

## Protected: working code, do not rewrite

These are correct. Change them only where a step below names the
function by name, and change nothing else in the file.

| File | Why it is protected |
|---|---|
| `src/data/load.py` | the number-matched join; it was broken once already |
| `src/features/build.py` | 24 features, none reading the future |
| `src/evaluation/splits.py` | the walk-forward split and the late labels |
| `src/evaluation/metrics.py` | every published number comes from here |
| `src/models/train.py` | the four models and the MLflow comparison |
| `src/models/feedback.py` | the verdict becomes a label |
| `src/models/exporter.py` | the eight gauges Grafana reads |
| `src/runs/loop.py` | the week-by-week walk, including the maturing decisions |
| every file under `tests/` | written by Claude, not yours to edit |
| `monitoring/grafana/dashboards/pipeline.json` | ten panels, the grid is right |

If a fix seems to need one of these changed, that is a finding. Write
it down and stop.

## The state of the thing

- 105 tests pass. The loop runs end to end on `drift-seed7`.
- LightGBM is `invoice-anomaly@champion`, 0.86 average precision on the
  held-out year.
- Grafana is alive: ten panels, the four model gauges among them.
- Three faults remain. They are below, worst first.

---

# Fault 1. The rules baseline collapsed

**What is wrong.** The README now shows the desk's rules at 0.20
average precision, below Isolation Forest. That breaks the project's
argument: the rules are the floor a model has to clear.

**Why it happens.** `rules.score` is a weighted share of the rules that
fire, and the weights were set by how costly a miss is, not by how
often the rule is right. Measured on `year-seed1`, 5330 invoices:

| Rule | Fires on | Of those, real |
|---|---|---|
| account not on file | 0.2% | 100% |
| sender not on file | 0.2% | 100% |
| price above contract | 0.8% | 100% |
| quantity above receipt | 0.3% | 100% |
| duty at a rate not in force | 0.1% | 100% |
| broker fee above contract | 0.2% | 100% |
| round total, not a known fee | 0.3% | 65% |
| young vendor | 0.2% | 75% |
| charge not on the rate card | 2.5% | 23% |
| same amount again | 3.8% | 17% |
| **order split under the limit** | **8.1%** | **6.5%** |
| **no order behind a goods bill** | **12.7%** | **1.9%** |

The last two fire on a fifth of every invoice and are almost always
wrong. Each carries weight 2, the same as "account not on file", which
is never wrong. So about a thousand invoices a year tie at the same
score as the handful that are certain, and the weekly top five is
filled by whichever row sorted first. That is the collapse.

**The fix.** Weight a rule by how much it raises the odds, not by how
sad the miss would be. Rule weights must be learnt from the training
weeks only, never from the test year, and never from a week the desk
has not decided yet.

**Tests, already written** (`tests/models/test_rule_weights.py`):

1. A rule that fires on 1% of rows and is right every time outranks a
   rule that fires on 20% and is right 2% of the time.
2. Weights are fitted on the training rows passed in, and the function
   raises if asked to fit on rows after the cutoff.
3. The twelve rule names and their order are unchanged. `scoring.STAGE`
   must keep matching them.
4. On a table where every rule is equally right, the score matches
   today's equal-weight score, to four decimal places.

**Done when.** `python run.py compare` puts the rules above both
unsupervised models on average precision, and the README table is
rewritten with the new numbers and the date you ran them.

---

# Fault 2. The loop retrains forever and promotes nothing

**What is wrong.** On `drift-seed7` the weekly precision sits under the
0.60 floor almost every week. The loop retrains 25 times in 26 weeks
and no challenger ever beats the 0.86 champion, so the dashboard shows
a flat line and a counter going up. The mechanics are right; nothing
tells a story.

**Why it happens.** Three things at once.

1. The floor is a guess. Nothing measured it.
2. The champion was trained on four full years. A challenger trained on
   a handful of weeks cannot beat it on the held-out year, ever.
3. A retraining fires on drift **or** a slip, and drift fires almost
   every week, because the comparison is one week against the previous
   week and a week is small.

**The fix, in this order.**

- The floor comes from the training years: the tenth percentile of the
  weekly precision a healthy champion reaches. Not a guess.
- A challenger is judged on **the weeks since it was trained**, not on
  the held-out year. The held-out year stays the one-off final number,
  reported once, as it is today.
- Drift fires on a week against the **eight weeks before it**, so a
  small week cannot ring the bell alone.

**Tests, already written** (`tests/evaluation/test_floor.py`,
`tests/runs/test_loop_story.py`):

1. `watch.floor_from(weekly)` returns the tenth percentile of the
   finished weeks, and ignores weeks with nothing decided.
2. Drift against an eight-week window does not fire on a healthy
   stretch, and does fire when the rates double halfway through.
3. Over the drift dataset the loop retrains **between one and four
   times**, not every week.
4. At least one of those retrainings is promoted.
5. The promoted model's week-since-training precision is higher than
   the model it replaced, on the same weeks.

**Done when.** `python run.py loop` prints a fall, a retraining, a
promotion and a recovery, in that order, and the Grafana panel
`7. The model on duty: weekly precision` shows the same shape.

---

# Fault 3. The README still sells the score

**What is wrong.** The README leads with a table of four models. The
project's point is the loop.

**The fix.** Rewrite the README in this order, keeping every number:

1. The desk's day: 5,300 invoices a year, about 190 wrong, two people.
2. The loop, in six lines: flag, decide, label, watch, retrain, promote.
3. One screenshot, `docs/monitoring.png`, of block 7.
4. The numbers, including the new rules baseline.
5. How to run it.

**Test** (`tests/test_readme.py`): the README names every command in
`run.py --help` and every number it prints matches the latest MLflow
run. No test may be written by you.

**Done when.** The README leads with the loop, the repository is
pushed, and `log.md` has an entry.

---

## Order of work

Fault 1, then 2, then 3. Fault 2's numbers change once Fault 1 lands,
so do not measure the floor before the weights are fixed.

## When you are stuck

Stop and write what you tried. Do not edit a test, do not weaken a
threshold to make a check pass, and do not touch the protected list.
A step that cannot be done as written is a fault in the step, and the
step gets rewritten, not the code.
