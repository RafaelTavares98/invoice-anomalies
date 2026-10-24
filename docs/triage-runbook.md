---
type: Runbook
title: Closing project 02
description: The closing steps of project 02, all done by 2026-10-22; closed 2026-10-24.
status: done
---

# Closing project 02

**Closed on 2026-10-24.** Steps 13 to 15 were done on 21 and 22
October. The README, tutorial and reference were brought up to date
on 24 October; what the run cannot show is written in all three. The
page below is kept as the record of how the project was closed.

Steps 1 to 12 are done. What follows is what is left, and nothing
else. Read the whole page before touching a file.

## Three changes made on 21 October, with the suite not run after them

I changed three production files to get the ranking working, and I did
not run the whole suite afterwards. Step 13 is where that gets
checked, and it is the first thing you do.

| File | Change | Why |
|---|---|---|
| `src/features/build.py` | `cut` ranks the values before correlating | an amount runs to millions with a few huge outliers; a plain correlation read those as noise and the cut kept three weak flags while dropping every feature that marks an error |
| `src/runs/loop.py` | `_ranking` falls back to the clerk's checks when the model returns fewer than three distinct scores in a week | the model answered the same number for all 786 invoices, so the weekly top five was whichever rows sorted first |
| `src/runs/loop.py` | `_redecided` re-decides every week already walked, as of the current week | a verdict matures; deciding each week once left every real error open for ever and the precision watch read zero on all 33 weeks |
| `run.py` | `_state_for_gauges` feeds the report and the watch state into `exporter.publish` | four of the eight gauges were never set, so the dashboard's model panels read empty |

What the loop does after those changes: 33 weeks, 23 of 32 errors
found, 140 false positives, precision 0.14, recall 0.72. Before them:
8 of 32 found and a constant score.

## What changed since you last worked here

Two defects were found and fixed on 20 October, after your last
report. Both are in `src/runs/loop.py` and `run.py`. Your summary of
the project's state is out of date; this page is current.

1. **Verdicts never matured.** Each week was decided once and never
   revisited, so a real error stayed "open" for ever and the weekly
   precision read zero on all 33 weeks. `_redecided` now re-decides
   every week already walked, as of the current week. The loop now
   finds 4 weeks with a confirmed error, precision between 0.2 and
   0.6, over 13 retrainings.
2. **The dashboard file was contaminated.** `watch` merged its rows
   into an `anomalies.csv` left over from a dataset that no longer
   exists: 3,424 rows, most of them invoice numbers Moty never
   issued. The file was deleted and rebuilt.

The test conflict you reported is also fixed: `test_retraining.py`
now names the feature it fits.

## You may not touch

| | Why |
|---|---|
| `README.md` | mine |
| anything under `docs/` | mine, including this page |
| `../01-invoice-pipeline/monitoring/` | the Grafana dashboard is mine |
| `../03-payables-panel/` anything but `data/anomalies.csv` | not this project |
| any file under `tests/` | written by the reviewer |

Report what you find in those files. Do not change them.

---

# Step 13. Prove the suite is green

```bash
python -m pytest -q
```

It takes about twelve minutes. Run it once, at the start, and once at
the end. Report the number that passed and the time both times.

The four changes listed at the top of this page have not been through
the suite. Expect failures there first, in particular:

- `tests/features/test_feature_cut.py`, since `cut` now ranks before
  correlating.
- `tests/runs/test_watch_command.py` and `tests/runs/test_commands.py`,
  since the loop's scores and verdicts both moved.

If a test fails, fix the code, never the test. If a test looks wrong,
stop and write down why: the step gets rewritten, not the test.

---

# Step 14. One writer for the dashboard file

**What is wrong.** `watch` and `score` both write
`../03-payables-panel/data/anomalies.csv`, and they disagree:

| Command | Rows | Why |
|---|---|---|
| `score` | 1,003 | every invoice, scored by one model fitted on the whole run |
| `watch` | 786 | only the weeks the loop walked; the first eight have no model yet |

Whichever ran last wins, and the dashboard has no way to tell which
it is reading.

**What to do.** `watch` becomes the only writer of that file.
`score` writes `data/scored.csv` inside this project instead.

Then make the file say what it is. Every row `watch` writes carries
two more columns:

| Column | Holds |
|---|---|
| `scored_on` | the date of the week the model ranked it in |
| `model_weeks` | how many weeks of data the model on duty had been trained on |

An invoice from before the first trainable week is not in the file at
all. That is correct: no model had seen anything yet, and a score
there would be invented.

**Tests:** `tests/runs/test_commands.py` and
`tests/runs/test_watch_command.py` both assert on the old paths.
**Do not edit them.** Read them first: if they contradict this step,
stop and say so, and the step gets rewritten rather than the test.

---

# Step 15. Say what the run cannot show

**What is wrong.** 8 of the run's real errors are only discovered
after 14 October, the last day of the run. `found_after_days` puts
their discovery past the end of the data. The loop can therefore
never confirm them, whatever the model does, and that is why all 13
retrainings fired on drift rather than on a fall in precision.

**What to do.** Nothing to the loop. Measure it and write it down:

1. Count, from `labels.csv`, how many errors are discovered after the
   last invoice date. Count how many are discovered within the run.
2. Append the two counts to `docs/removals.md`, as a new section
   titled "What the run cannot show", with the date you measured them
   and the one-line reason.
3. Add nothing to the README. Tell me the numbers in your report and
   I will write the page.

This is the project's sharpest limitation and it belongs in writing,
measured, not described from memory.

---

## Order

13, then 14, then 15, then 13 again.

## When you are done

Report, in plain words:

- the suite before and after: how many passed, how long
- what `watch` did on its real run: weeks, retrainings, promotions,
  and how many weeks had a confirmed error
- the two counts from step 15
- every file you touched
- anything this page did not cover
