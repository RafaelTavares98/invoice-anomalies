"""The loop: score, decide, watch, retrain, one week at a time.

The model flags, an analyst decides, the decision becomes a label, the
model is retrained, and a better model takes over. This module walks
that loop week by week over one dataset: score the week with the serving
model, play the analyst as of the week's end, turn the verdicts into
labels, watch the precision and the drift, and retrain when either
fires. Every week's scored rows come out in order, with the decisions
the desk has by the last week, so the exporter can read them as they
stand.

Two things make the watch honest. Decisions mature: a flag takes days
to decide, so every week the verdicts are rebuilt as of that week's end,
and a week's precision grows with the evidence. Drift compares a week to
the week before it: the world changing shows up as a jump, and features
that simply grow with time do not ring forever.

A retraining whose labels hold one class keeps the serving model: there
is nothing to learn, and the trigger is still recorded.
"""

import pandas as pd

from data import load
from evaluation import watch
from models import feedback, lifecycle, scoring, train


def _slice(data, numbers):
    """The dataset rows of one week, the same shape as the whole."""
    invoices = data.invoices[data.invoices['number'].isin(numbers)]
    lines = data.lines[data.lines['number'].isin(numbers)]
    labels = data.labels[data.labels['number'].isin(numbers)]
    return load.Dataset.from_frames(data.name, invoices, lines, data.erp,
                                    labels)


def run(table, data, held_out, model, promote, seed=0, k=5,
        floor=watch.FLOOR, run_length=watch.RUN, line=watch.LINE):
    """Walk the loop over every week, in order.

    `table` carries a `week` column. `model` is the serving model, and
    `promote(model, report)` registers a retrained one and returns
    whether it won. Returns the events, the weekly watch as of the last
    week, the last drift per feature, and every scored row, in week
    order, decided as of the last week's end.
    """
    events = []
    drift_by_feature = {}
    weeks = sorted(table['week'].unique())
    slices = []
    scored = None
    for week in weeks:
        this_week = table[table['week'] == week].reset_index(drop=True)
        as_of = this_week['issue_date'].max() + pd.Timedelta(days=6)
        scores = model.predict_proba(train._matrix(this_week))[:, 1]
        slices.append((this_week, _slice(data, set(this_week['number'])),
                       scores))
        rebuilt = [scoring.rows(t, d, s, k=k, as_of=as_of)
                   for t, d, s in slices]
        scored = pd.concat(rebuilt, ignore_index=True)
        seen = pd.concat([t for t, _, _ in slices], ignore_index=True)
        joined = seen.merge(
            scored[['number', 'flagged', 'decision', 'decided_on']],
            on='number', how='left')
        labels = feedback.labels_from(joined, as_of)
        current_weekly = watch.weekly(scored)
        row_for_week = current_weekly.iloc[-1]
        over = []
        if len(slices) > 1:
            previous = slices[-2][0]
            drift_by_feature = watch.drift(previous, this_week)
            over = watch.drifted(drift_by_feature, line)
        slipping = watch.slipping(current_weekly, floor, run_length)
        retrained = promoted = False
        if slipping or over:
            try:
                model, _ = lifecycle.retrain(seen, as_of, labels, seed=seed)
                report = lifecycle.challenge(model, held_out)
                promoted = promote(model, report)
            except ValueError:
                promoted = False
            retrained = True
        events.append({
            'week': week,
            'flagged': int(this_week_flags(slices[-1], scored, k)),
            'confirmed': int(row_for_week['confirmed']),
            'decided': int(row_for_week['decided']),
            'precision': (float(row_for_week['precision'])
                          if pd.notna(row_for_week['precision'])
                          else float('nan')),
            'drifted': ','.join(over),
            'retrained': retrained,
            'promoted': promoted,
        })
    weekly = watch.weekly(scored)
    return events, weekly, drift_by_feature, scored


def this_week_flags(last_slice, scored, k):
    """How many rows of the newest week were flagged in it."""
    table, _, _ = last_slice
    numbers = set(table['number'])
    return int(scored[scored['number'].isin(numbers)]['flagged'].sum())
