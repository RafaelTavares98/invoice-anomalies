"""The week-by-week walk on the live run.

The serving model scores a week it never saw; the analyst's verdicts
become labels; a fall or a drift starts a challenger; and the challenger
takes over only when the intervals of the two are separated. With 35
errors that will rarely happen, and a promotion on overlapping
intervals is the mistake this project exists to avoid.

The verdict reads recall: a desk buys a model for the errors it finds.
Precision is reported beside it, because a recall won by flagging
everything is no use.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import load
from evaluation import intervals, metrics, splits, watch as watch_mod
from models import feedback, lifecycle, rules, scoring
from models import train as train_module

#: Weeks a challenger is judged on before the desk decides.
CHALLENGE_WEEKS = 4

#: Weeks of history the drift comparison looks back on.
DRIFT_WEEKS = 8

#: Features that move with the calendar, not with the world.
CLOCKS = ('vendor_age_days', 'invoices_30d')

#: Invoices the desk checks first each week.
CHECKED = metrics.K


def _matrix(table, features):
    """The columns the serving model reads: the cut, or all of them
    when a caller handed a model of its own."""
    if features:
        return table[list(features)].fillna(0)
    return train_module._matrix(table)


def _week_start(dates):
    return (dates.dt.normalize()
            - pd.to_timedelta(dates.dt.weekday, unit='D'))


def _week_text(week):
    if hasattr(week, 'strftime'):
        return week.strftime('%Y-%m-%d')
    return int(week)


#: Below this many distinct scores in a week, the model is not ranking.
TELLING = 3


def _ranking(model, week, features):
    """The week's scores, from the model when it ranks at all.

    Trained on weeks where almost no error has come to light yet, the
    model answers the same number for every invoice. A constant is not
    a ranking: the week's top five would then be whichever rows sorted
    first. When that happens the clerk's own checks rank the week, and
    they are the baseline the model has to beat anyway.
    """
    scores = model.predict_proba(_matrix(week, features))[:, 1]
    if len(np.unique(np.round(scores, 6))) >= TELLING:
        return scores
    return rules.score(week).to_numpy()


def _redecided(frame, data, as_of):
    """The same scored rows, with the verdicts as they stand today.

    Only the verdict moves. The score, the flag and the checks were
    decided by the model on duty that week and are history.
    """
    decided, decided_on = scoring._decisions(frame, data, as_of)
    return frame.assign(decision=decided, decided_on=decided_on)


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


def _called(scores, k=CHECKED):
    """The top k of one week, by score: what the desk checks."""
    order = np.argsort(-np.asarray(scores), kind='stable')
    called = np.zeros(len(order), dtype=int)
    called[order[:k]] = 1
    return called


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    if hasattr(value, 'item'):
        return value.item()
    return value


def _log(path, entry):
    """One JSON line per week, with a timestamp that carries its zone."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {'at': datetime.now(timezone.utc).astimezone().isoformat(),
            **entry}
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(_clean(line)) + '\n')


def _first_trainable(table, starts):
    """The first week whose past weeks carry both a real error and a
    clean invoice by the week's end. Before that there is nothing to
    learn from, so nothing is scored."""
    for position, week in enumerate(starts):
        past = table[table['week'] < week]
        if not len(past):
            continue
        as_of = week + pd.Timedelta(days=6)
        if splits.known_labels(past, as_of).nunique() >= 2:
            return position
    return len(starts)


def training_floor(model, table, data, features=None, share=0.7):
    """The tenth percentile of the model's weekly precision on the
    early weeks: what a healthy model reaches on this run."""
    train_part, _ = splits.by_time(table, share=share)
    as_of = train_part['issue_date'].max()
    scores = model.predict_proba(_matrix(train_part, features))[:, 1]
    scored = scoring.rows(train_part,
                          _slice(data, set(train_part['number'])),
                          scores, as_of=as_of)
    return watch_mod.floor_from(watch_mod.weekly(scored))


def run(table, data, held_out=None, model=None, promote=None, seed=0,
        floor=watch_mod.FLOOR, run_length=watch_mod.RUN, line=watch_mod.LINE,
        challenge_weeks=CHALLENGE_WEEKS, out=None, log=None, state=None,
        weeks=None, features=None):
    """Walk the run week by week.

    `model` is the serving model; with none, the walk starts at the
    first week whose past can train one, and trains it there. Returns
    one event per scored week, in order.
    """
    live = model is None
    table = table.copy()
    if 'week' not in table:
        table['week'] = _week_start(table['issue_date'])
    starts = sorted(table['week'].unique())
    if model is None:
        starts = starts[_first_trainable(table, starts):]
    # What earlier runs left behind: a run stopped and started again
    # carries its counts and its weekly history on, not from zero.
    earlier = {'retrainings': 0, 'promotions': 0, 'weekly': []}
    if state is not None and Path(state).exists():
        saved = json.loads(Path(state).read_text(encoding='utf-8'))
        earlier = {'retrainings': int(saved.get('retrainings', 0)),
                   'promotions': int(saved.get('promotions', 0)),
                   'weekly': list(saved.get('weekly') or [])}
        last = saved.get('last_week')
        if last:
            starts = [week for week in starts if _week_text(week) > last]
    if weeks is not None:
        starts = starts[:weeks]

    existing = None
    if out is not None and Path(out).exists():
        existing = pd.read_csv(out, keep_default_na=False)
        # Only this command's rows may stay: a row without `scored_on`
        # was written by something else, and merging it would put an
        # invoice nobody walked into the dashboard's file.
        if 'scored_on' in existing:
            existing = existing[existing['scored_on'].astype(str) != '']
        else:
            existing = None
    done = []
    events = []
    trained_to = None
    weeks_trained = 0
    pending = None
    cooling = 0
    drift_by = {}
    for week in starts:
        this = table[table['week'] == week].reset_index(drop=True)
        as_of = this['issue_date'].max() + pd.Timedelta(days=6)
        if model is None:
            past = table[table['week'] < week]
            labels = splits.known_labels(past, as_of)
            model, _ = lifecycle.retrain(past, as_of, labels, seed=seed,
                                         features=features)
            trained_to = past['issue_date'].max()
            weeks_trained = int(past['week'].nunique())
            floor = training_floor(model, table, data, features)

        scores = _ranking(model, this, features)
        data_week = _slice(data, set(this['number']))
        rows = scoring.rows(this, data_week, scores, k=CHECKED, as_of=as_of)
        rows = rows.assign(scored_on=_week_text(week),
                           model_weeks=weeks_trained)
        done.append(rows)
        # A verdict matures: an error takes days or months to come to
        # light, so a flag decided this week is still open next week.
        # Deciding each week once and never again left every real error
        # marked open for ever, and the precision watch read zero on a
        # run where the ranking was catching errors.
        done = [_redecided(frame, data, as_of) for frame in done]

        promoted = False
        separated = False
        challenger_precision = None
        verdict_served = None
        if pending is not None:
            challenger_scores = pending['model'].predict_proba(
                _matrix(this, features))[:, 1]
            pending['weeks'].append((
                this['is_anomaly'].to_numpy(dtype=int),
                _called(scores), _called(challenger_scores)))
            if len(pending['weeks']) >= challenge_weeks:
                truth = np.concatenate([w[0] for w in pending['weeks']])
                served_calls = np.concatenate([w[1] for w in pending['weeks']])
                challenger_calls = np.concatenate([w[2] for w in pending['weeks']])
                served_range = intervals.around(metrics.recall, truth,
                                                served_calls)
                challenger_range = intervals.around(metrics.recall, truth,
                                                    challenger_calls)
                separated = intervals.separated(challenger_range, served_range)
                if (separated
                        and metrics.recall(truth, challenger_calls)
                        > metrics.recall(truth, served_calls)):
                    model = pending['model']
                    trained_to = pending['trained_to']
                    weeks_trained = pending['weeks_trained']
                    promoted = True
                    verdict = {
                        'challenger_precision': metrics.precision(
                            truth, challenger_calls),
                        'serving_precision': metrics.precision(
                            truth, served_calls)}
                    challenger_precision = verdict['challenger_precision']
                    verdict_served = verdict['serving_precision']
                    if promote is not None:
                        promote(model, verdict)
                pending = None

        weekly = watch_mod.weekly(pd.concat(done, ignore_index=True))
        over = []
        previous = [start for start in sorted(table['week'].unique())
                    if start < week][-DRIFT_WEEKS:]
        if len(previous) >= DRIFT_WEEKS:
            window = table[table['week'].isin(previous)]
            held = watch_mod.drift(window, this)
            drift_by = {name: value for name, value in held.items()
                        if name not in CLOCKS}
            over = watch_mod.drifted(drift_by, line)
        slipping = (floor is not None
                    and watch_mod.slipping(weekly, floor, run_length))
        trigger = 'precision' if slipping else ('drift' if over else None)
        retrained = False
        if cooling > 0:
            cooling -= 1
        elif pending is None and trigger is not None:
            retrained = True
            seen = table[table['week'] <= week].reset_index(drop=True)
            merged = seen.merge(
                pd.concat(done, ignore_index=True)[
                    ['number', 'flagged', 'decision', 'decided_on']],
                on='number', how='left')
            labels = feedback.labels_from(merged, as_of)
            try:
                if live:
                    challenger, _ = lifecycle.retrain(
                        seen, as_of, labels, seed=seed, features=features)
                else:
                    challenger, _ = lifecycle.retrain(
                        seen, as_of, labels, seed=seed)
                pending = {'model': challenger, 'weeks': [],
                           'trained_to': seen['issue_date'].max(),
                           'weeks_trained': int(seen['week'].nunique())}
                cooling = challenge_weeks + 1
            except ValueError:
                pass

        event = {
            'week': _week_text(week),
            'checked': int(rows['flagged'].sum()),
            'confirmed': int((rows['decision'] == 'confirmed').sum()),
            'dismissed': int((rows['decision'] == 'dismissed').sum()),
            'trained_to': _week_text(trained_to) if trained_to is not None
            else None,
            'retrained': retrained,
            'because': trigger if retrained else None,
            'promoted': promoted,
            'separated': separated,
            'challenger_precision': challenger_precision,
            'verdict_served': verdict_served,
        }
        events.append(event)
        if out is not None:
            frames = ([existing] if existing is not None else []) + done
            whole = (pd.concat(frames, ignore_index=True)
                     .drop_duplicates('number', keep='last'))
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            whole.to_csv(out, index=False)
        if log is not None:
            _log(log, {'command': 'watch', **event})
        if state is not None:
            # The gauges are served from this file between runs, so it
            # carries what the week left behind, not only the place.
            promotions = (earlier['promotions']
                          + sum(1 for e in events if e['promoted']))
            now = watch_mod.weekly(
                pd.concat(done, ignore_index=True)).to_dict('records')
            walked = {str(row['week']) for row in now}
            history = [row for row in earlier['weekly']
                       if str(row['week']) not in walked] + now
            Path(state).write_text(json.dumps({
                'last_week': _week_text(week),
                'retrainings': (earlier['retrainings']
                                + sum(1 for e in events if e['retrained'])),
                'promotions': promotions,
                'champion_version': promotions + 1,
                'drift': {name: float(value)
                          for name, value in drift_by.items()},
                'weekly': history,
            }, default=str), encoding='utf-8')
    weekly = watch_mod.weekly(pd.concat(done, ignore_index=True)) if done else None
    return events, weekly, drift_by if events else {}, done
