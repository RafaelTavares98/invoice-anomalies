---
type: Note
title: Invented claims to remove
description: Facts Claude put into the project that nobody decided and nothing measures. Nothing here is removed until Rafael says so.
status: open
---

# Invented claims to remove

Items 1 and 2 are out of `README.md` as of 2026-10-19, and out of the
runbook since it was rewritten on 2026-10-21. They are still in
`pdf_README.pdf` and `pdf_triage-runbook.pdf`, which have not been
rebuilt. Item 3 is closed as of 2026-10-24.

Things written into this project that Rafael never decided and no code,
dataset or measurement supports. The list is filled first and acted on
later, in one pass, on his order.

Nothing here is removed yet.

## 1. "Two people" at the payables desk

**Where it is**

| File | Line |
|---|---|
| `README.md` | "Two people check the five the model ranks first every week" |
| `docs/triage-runbook.md` | Fault 3: "5,300 invoices a year, about 190 wrong, two people" |
| `docs/pdf_README.pdf` | the printed copy of the same line |
| `docs/pdf_triage-runbook.pdf` | the printed copy of the same line |

**Why it is wrong.** No headcount was ever decided or measured. The
code holds two numbers only: `metrics.K` is 5, the invoices ranked
first each week, and `metrics.DESK` is 25, what a desk checks by hand.

**Why it was written.** To make the number 5 sound motivated. It
already was: 5 is about how many errors a week the simulation plants,
so a perfect model scores 1 and every false alarm shows. A measured
reason was replaced with an invented scene.

**What replaces it.** Nothing, unless Rafael says otherwise. The line
states what the code does and stops there.

## 2. "About 5,600 invoices a year, about 200 of them wrong"

**Where it is**

| File | Line |
|---|---|
| `README.md` | "About 5,600 invoices a year, about 200 of them wrong" |
| `docs/pdf_README.pdf` | the printed copy |

**What is actually measured.** On `year-seed1`: 5,330 invoices, 191 of
them anomalies. That is one seed, counted in this session. The plan
says the year profile aims at about 5,300 invoices and about 250
anomalies, which does not match either.

**Where 5,600 and 200 come from.** Unknown. The runbook asked for
"5,300 a year, about 190 wrong", the measured figures. DeepSeek wrote
other numbers into the README and nobody checked them.

**What it needs.** Count the invoices and the anomalies across the five
year seeds, write the real figures, and say which seeds they come from.

## 3. References to the invented runs still standing

Found on 2026-10-19, in the pass that moved the project onto the live
run. Step 7 removed the old commands from `run.py`. Step 9 deleted the
cached feature files and moved the two tests that read the dropped
sets to `tests/_retired/`. What is left is on the protected list.

| File | Line | What stands |
|---|---|---|
| ~~`run.py`~~ | 15 | help text: the loop runs on `drift-seed7` by default — **removed in step 7** |
| ~~`run.py`~~ | 44 | `TRAIN_SEEDS` holds `year-seed1` to `year-seed4` — **removed in step 7** |
| ~~`run.py`~~ | 45 | `HELD_OUT` holds `year-seed5` — **removed in step 7** |
| ~~`run.py`~~ | 130 | `retrain` defaults to `drift-seed7` — **removed in step 7** |
| ~~`run.py`~~ | 175 | `loop` defaults to `drift-seed7` — **removed in step 7** |
| ~~`data/features/year-seed1.csv` .. `year-seed5.csv`~~ | — | cached features — **deleted in step 9** |
| ~~`tests/models/test_honesty.py`~~ | 17 | reads `year-seed1` — **moved to `tests/_retired/` in step 9** |
| ~~`tests/runs/test_loop_story.py`~~ | 32, 33 | reads and names `drift-seed7` — **moved to `tests/_retired/` in step 9** |
| ~~`src/runs/loop.py`~~ | 73 | `training_floor` defaulted to the four year seeds — **rewritten in step 10** |
| ~~`src/data/load.py`~~ | 22 | `DATASETS` points at the invented datasets folder — **removed 2026-10-24** |
| ~~`src/data/load.py`~~ | 102 | `available()` globs `year-seed*` — **removed 2026-10-24**, with `read()` and the test that used them |
| ~~`docs/triage-runbook.md`~~ | 53 | says the five year folders are no longer used — **gone in the 2026-10-21 rewrite** |
| ~~`docs/triage-runbook.md`~~ | 169 | step 6 names the `year-seed*` folders — **gone in the 2026-10-21 rewrite** |
| ~~`docs/triage-runbook.md`~~ | 176 | step 6 asks for this list — **gone in the 2026-10-21 rewrite** |
| `docs/removals.md` | 51 | this page's item 2 names `year-seed1` |

Nothing new calls `load.read()`; `live()` reads the real run instead.
`run.py` no longer names an invented run. The files that still do are
protected, so they stay.

One note: two active tests also name the dropped runs while policing
them, `tests/test_readme.py` (a docstring) and
`tests/runs/test_commands.py` (asserting `run.py` holds no such name).
They are tests, so they are not edited. The sources this pass controls,
`run.py`, `README.md` and everything under `src/` except the protected
`loop.py` rewrite, name neither.

## What the run cannot show

Measured on 2026-10-21, from `labels.csv`, over the 35 distinct invoice
numbers: **29 errors are discovered within the run**, on or before
2026-10-14, the last invoice date, and **6 are discovered after it**.

The reason, in one line: `found_after_days` puts six discoveries past
the end of the data, so the loop can never confirm them, whatever the
model does.

