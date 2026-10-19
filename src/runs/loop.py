"""The loop: score, decide, watch, retrain, promote, one week at a time.

The model flags, an analyst decides, the decision becomes a label, the
model is retrained, and a better model takes over. This module walks
that loop week by week over one dataset: score the week with the serving
model, play the analyst as of the week's end, turn the verdicts into
labels, watch the precision and the drift, and retrain when either
fires.

Three things keep the story honest.

- The floor comes from the training years: the tenth percentile of the
  weekly precision a healthy champion reaches. Nothing is guessed.
- Drift compares a week to the eight weeks before it, so one small week
  cannot ring the bell alone.
- A challenger is judged on the weeks since it was trained, never on
  the held-out year. The held-out year stays the one-off final number.
  The verdict is offline: each week's top five, from each model, is
  checked against the answer key, because the desk's own verdicts
  mature too slowly to judge four weeks. While a challenger waits for
  its verdict, no new retraining starts.

Decisions mature: a flag takes days to decide, so every week the
verdicts are rebuilt as of that week's end, and a week's precision grows
with the evidence. A retraining whose labels hold one class keeps the
serving model: there is nothing to learn, and the trigger is recorded.
"""

import numpy as np
import pandas as pd

from data import load
from evaluation import splits, watch
from models import feedback, lifecycle, scoring, train

#: How many weeks judge a challenger before the desk decides.
CHALLENGE_WEEKS = 4

#: How many weeks of history the drift comparison looks back on.
DRIFT_WEEKS = 8

#: Features that move with the calendar, not with the world: a vendor
#: ages a day every day, and the thirty-day count ramps as history
#: fills. They would ring forever and say nothing.
CLOCKS = ('vendor_age_days', 'invoices_30d')


def _slice(data, numbers):
    """The dataset rows of one week, the same shape as the whole."""
    invoices = data.invoices[data.invoices['number'].isin(numbers)]
    lines = data.lines[data.lines['number'].isin(numbers)]
    labels = data.labels[data.labels['number'].isin(numbers)]
    return load.Dataset.from_frames(data.name, invoices, lines, data.erp,
                                    labels)


def _precision(rows):
    """Confirmed over decided among one week's flags; NaN when undecided."""
    allowed = rows['decision'].isin(('confirmed', 'dismissed'))
    decided = int(allowed.sum())
    if not decided:
        return float('nan')
    return float((rows.loc[allowed, 'decision'] == 'confirmed').mean())


def _top_precision(table, scores, k=5):
    """Share of one week's top k that are real errors, from the key."""
    top = (table.assign(score=scores)
           .sort_values('score', ascending=False, kind='stable').head(k))
    return float(top['is_anomaly'].mean())


def training_floor(model, names=('year-seed1', 'year-seed2', 'year-seed3',
                                 'year-seed4')):
    """The tenth percentile of the champion's weekly precision, on the
    training years with every decision matured."""
    frames = []
    for name in names:
        data = load.read(name)
        built = train.table_for(data, name)
        built = built.assign(week=splits.week_of(built))
        scores = model.predict_proba(train._matrix(built))[:, 1]
        scored = scoring.rows(built, data, scores,
                              as_of=built['issue_date'].max())
        frames.append(watch.weekly(scored))
    return watch.floor_from(pd.concat(frames, ignore_index=True))


def run(table, data, held_out, model, promote, seed=0, k=5, floor=watch.FLOOR,
        run_length=watch.RUN, line=watch.LINE, challenge_weeks=CHALLENGE_WEEKS):
    """Walk the loop over every week, in order.

    `table` carries a `week` column. `model` is the serving model, and
    `promote(model, verdict)` is called once a challenger wins its
    weeks, to register it. `held_out` is kept for the final report and
    is not used to judge a challenger. Returns the events, the weekly
    watch as of the last week, the last drift per feature, and every
    scored row, in week order.
    """
    events = []
    drift_by_feature = {}
    weeks = sorted(table['week'].unique())
    slices = []
    scored = None
    pending = None
    cooling = 0
    for week in weeks:
        this_week = table[table['week'] == week].reset_index(drop=True)
        as_of = this_week['issue_date'].max() + pd.Timedelta(days=6)
        scores = model.predict_proba(train._matrix(this_week))[:, 1]
        data_week = _slice(data, set(this_week['number']))
        slices.append((this_week, data_week, scores))
        rebuilt = [scoring.rows(t, d, s, k=k, as_of=as_of)
                   for t, d, s in slices]
        scored = pd.concat(rebuilt, ignore_index=True)
        serving_precision = _precision(rebuilt[-1])
        seen = pd.concat([t for t, _, _ in slices], ignore_index=True)
        joined = seen.merge(
            scored[['number', 'flagged', 'decision', 'decided_on']],
            on='number', how='left')
        labels = feedback.labels_from(joined, as_of)

        over = []
        if len(slices) > DRIFT_WEEKS:
            window = [t for t, _, _ in slices[-1 - DRIFT_WEEKS:-1]]
            held = watch.drift(pd.concat(window, ignore_index=True), this_week)
            drift_by_feature = {name: score for name, score in held.items()
                                if name not in CLOCKS}
            over = watch.drifted(drift_by_feature, line)

        current_weekly = watch.weekly(scored)
        row_for_week = current_weekly.iloc[-1]
        slipping = watch.slipping(current_weekly, floor, run_length)

        promoted = False
        challenger_precision = None
        verdict_served = None
        if pending is not None:
            challenger_scores = pending['model'].predict_proba(
                train._matrix(this_week))[:, 1]
            pending['weeks'].append((this_week, challenger_scores, scores))
            if len(pending['weeks']) >= challenge_weeks:
                served = float(np.mean(
                    [_top_precision(t, old, k)
                     for (t, _, old) in pending['weeks']]))
                offered = float(np.mean(
                    [_top_precision(t, new, k)
                     for (t, new, _) in pending['weeks']]))
                if offered > served:
                    model = pending['model']
                    promoted = True
                    challenger_precision = offered
                    verdict_served = served
                    promote(model, {'challenger_precision': offered,
                                    'serving_precision': served})
                pending = None

        retrained = False
        if cooling > 0:
            cooling -= 1
        elif pending is None and (slipping or over):
            retrained = True
            try:
                challenger, _ = lifecycle.retrain(seen, as_of, labels,
                                                  seed=seed)
                pending = {'model': challenger, 'weeks': []}
                cooling = challenge_weeks
            except ValueError:
                pass

        events.append({
            'week': week,
            'flagged': int(rows_flagged(this_week, scored)),
            'confirmed': int(row_for_week['confirmed']),
            'decided': int(row_for_week['decided']),
            'precision': (float(row_for_week['precision'])
                          if pd.notna(row_for_week['precision'])
                          else float('nan')),
            'top_precision': _top_precision(this_week, scores, k),
            'serving_precision': serving_precision,
            'challenger_precision': challenger_precision,
            'verdict_served': verdict_served,
            'drifted': ','.join(over),
            'retrained': retrained,
            'promoted': promoted,
        })
    weekly = watch.weekly(scored)
    return events, weekly, drift_by_feature, scored


def rows_flagged(this_week, scored):
    """How many rows of the newest week were flagged in it."""
    numbers = set(this_week['number'])
    return int(scored[scored['number'].isin(numbers)]['flagged'].sum())
