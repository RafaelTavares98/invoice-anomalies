"""The commands that keep the project running on Moty's own run.

    python run.py prepare [--out FILE]   load the run, build and cut the
                                         features, write the table
    python run.py train [--table FILE]   split by time, fit every model
                                         and the clerk's checks, report
    python run.py score [--table FILE] [--out FILE]
                                         one model over the whole run,
                                         into data/scored.csv
    python run.py watch [--table FILE] [--out FILE] [--state FILE]
                                         walk the run week by week, and
                                         keep going; the only writer of
                                         the dashboard's file
    python run.py serve [--port N]       the gauges, kept up

Every command appends one JSON line to data/runs.log, so days of
running stay readable.
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / 'src'))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from data import load                                # noqa: E402
from evaluation import intervals, metrics, splits    # noqa: E402
from features import build                           # noqa: E402
from models import exporter, rules, scoring, train as train_module  # noqa: E402

#: The one run this project reads.
LIVE = HERE.parent / '01-invoice-pipeline' / 'Invoices' / 'live'

DATA = HERE / 'data'
TABLE = DATA / 'features.csv'
REPORT = DATA / 'model-report.json'
LOG = DATA / 'runs.log'

#: The scored rows, written here and nowhere else. Only `watch` writes
#: it, so the file always says which model produced it. Project 03's
#: `refresh_live_data.py` copies it into that project's `data-live/`.
#: It used to be written straight into project 03's sample folder, which
#: the dashboard does not read, so a fresh run never reached the page.
SCORED = DATA / 'anomalies.csv'

#: Where `score` writes its one-model pass, kept inside this project.
SCORE_FILE = DATA / 'scored.csv'

#: Where watch keeps its place between runs.
WATCH_STATE = DATA / 'watch-state.json'

#: The share of the run that trains; the rest is tested, later in time.
SHARE = 0.7

#: Invoices the desk checks first each week, the ranking's own budget.
CHECKED = metrics.K

#: Points on the caught-as-more-are-checked curve.
STEP = 25


def _log(path, entry):
    """One JSON line per command, with a timestamp that carries its zone."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {'at': datetime.now(timezone.utc).astimezone().isoformat(),
            **entry}
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(_clean(line)) + '\n')


def _clean(value):
    """What json can carry: no NaN, no numpy, dates as text."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if hasattr(value, 'item'):
        return value.item()
    return value


def _option(words, name, default):
    if name in words:
        place = words.index(name) + 1
        if place < len(words):
            return Path(words[place])
    return Path(default)


def _load_table(path):
    table = pd.read_csv(path, keep_default_na=False)
    table['issue_date'] = pd.to_datetime(table['issue_date'])
    for column in build.COLUMNS:
        table[column] = pd.to_numeric(table[column], errors='coerce')
    table['is_anomaly'] = pd.to_numeric(table['is_anomaly']).astype(int)
    table['found_after_days'] = pd.to_numeric(table['found_after_days'],
                                              errors='coerce')
    return table


def _week_of(issue_dates):
    return (issue_dates.dt.normalize()
            - pd.to_timedelta(issue_dates.dt.weekday, unit='D'))


def _top_called(table, scores, k=CHECKED):
    """The share the desk checks: the top k of each week, by score."""
    ranked = pd.Series(scores, index=table.index)
    order = ranked.sort_values(ascending=False, kind='stable').index
    weeks = _week_of(table['issue_date'])
    called = pd.Series(0, index=table.index)
    for week, places in weeks.groupby(weeks).groups.items():
        candidates = [index for index in order if index in places][:k]
        called.loc[candidates] = 1
    return called.astype(int)


def _average_precision_interval(truth, scores, seed=0, draws=2000):
    """A figure read off the ranking needs its own resampling."""
    truth = np.asarray(truth, dtype=int)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        picked = rng.integers(0, len(truth), len(truth))
        value = metrics.pr_curve(truth[picked], scores[picked])['average_precision']
        if not math.isnan(value):
            values.append(value)
    if not values:
        return [float('nan'), float('nan')]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _caught_by_checked(truth, scores, step=STEP):
    """Errors caught as the ranking is read from the top down."""
    ranked = np.asarray(truth, dtype=int)[np.argsort(-np.asarray(scores),
                                                        kind='stable')]
    caught = np.cumsum(ranked)
    points = [[0, 0]]
    for checked in range(step, len(ranked) + 1, step):
        points.append([checked, int(caught[checked - 1])])
    return points


def prepare(out=TABLE, log=LOG):
    """Load the run, build the features, cut them, write the table."""
    data = load.live(LIVE)
    table = build.features(data)
    answers = data.labels[['number', 'anomaly', 'lookalike_of',
                           'found_after_days', 'is_anomaly']]
    table = table.merge(answers, on='number', how='left')
    kept = build.cut(table)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ['number', 'issue_date', 'anomaly', 'lookalike_of',
               'found_after_days', 'is_anomaly', *build.COLUMNS]
    table[columns].to_csv(out, index=False)
    out.with_suffix('.meta.json').write_text(
        json.dumps({'kept': kept, 'order': list(build.COLUMNS)}),
        encoding='utf-8')
    report = {
        'rows': len(table),
        'errors': int(table['is_anomaly'].sum()),
        'features_before': len(build.COLUMNS),
        'features_after': len(kept),
        'kept': kept,
        'order': list(build.COLUMNS),
    }
    _log(log, {'command': 'prepare', 'table': str(out), **report})
    return report


def _fit_scores(train_part, test_part, kept, cutoff):
    """Every model's ranking of the test invoices, biggest first."""
    scores = {}
    truth_label = splits.known_labels(train_part, cutoff)
    scores['rules'] = rules.score(
        test_part, rules.learn_weights(train_part, cutoff)).to_numpy()

    from lightgbm import LGBMClassifier
    booster = LGBMClassifier(**train_module.Boosting.params)
    booster.fit(train_part[kept].fillna(0), truth_label)
    scores['lightgbm'] = booster.predict_proba(test_part[kept].fillna(0))[:, 1]

    from pyod.models.ecod import ECOD
    from pyod.models.iforest import IForest
    matrix_train = train_part[kept].fillna(0).to_numpy(dtype=float)
    matrix_test = test_part[kept].fillna(0).to_numpy(dtype=float)
    forest = IForest(**train_module.Isolation.params)
    forest.fit(matrix_train)
    scores['isolation_forest'] = forest.decision_function(matrix_test)
    ecod = ECOD()
    ecod.fit(matrix_train)
    scores['ecod'] = ecod.decision_function(matrix_test)
    return scores


def train(table=TABLE, log=LOG, report_path=REPORT):
    """Split by time, fit every model, report every figure with its interval."""
    table = _load_table(table)
    train_part, test_part = splits.by_time(table, share=SHARE)
    cutoff = train_part['issue_date'].max()
    detail = splits.describe(table, share=SHARE)
    kept = build.cut(train_part, cutoff=cutoff)
    truth = test_part['is_anomaly'].to_numpy(dtype=int)
    kinds = test_part['anomaly'].fillna('').to_numpy()

    models = {}
    for name, scores in _fit_scores(train_part, test_part, kept, cutoff).items():
        called = _top_called(test_part, scores)
        p = metrics.precision(truth, called)
        r = metrics.recall(truth, called)
        f = metrics.f1(truth, called)
        ap = metrics.pr_curve(truth, scores)['average_precision']
        models[name] = {
            'confusion': metrics.confusion(truth, called),
            'precision': {'value': p,
                          'interval': intervals.around(metrics.precision,
                                                       truth, called)},
            'recall': {'value': r,
                       'interval': intervals.around(metrics.recall,
                                                    truth, called)},
            'f1': {'value': f,
                   'interval': intervals.around(metrics.f1,
                                                truth, called)},
            'average_precision': {
                'value': ap,
                'interval': _average_precision_interval(truth, scores)},
            'recall_by_kind': metrics.recall_by_kind(truth, called, kinds),
            'caught_by_checked': _caught_by_checked(truth, scores),
        }

    sides = {
        side: {'first': detail[side]['first'].strftime('%Y-%m-%d'),
               'last': detail[side]['last'].strftime('%Y-%m-%d'),
               'errors': detail[side]['errors']}
        for side in ('train', 'test')
    }
    report = {
        'split': {'cutoff': cutoff.strftime('%Y-%m-%d'), **sides},
        **sides,
        'baseline': 'rules',
        'models': models,
    }
    exporter.publish_report(report, report_path)
    _log(log, {'command': 'train',
               'train_errors': report['split']['train']['errors'],
               'test_errors': report['split']['test']['errors'],
               'features': len(kept)})
    return report


def score(table=TABLE, out=SCORE_FILE, log=LOG):
    """Write one row per invoice, from one model fitted on the whole run."""
    table = _load_table(table)
    data = load.live(LIVE)
    kept = build.cut(table)
    from lightgbm import LGBMClassifier
    booster = LGBMClassifier(**train_module.Boosting.params)
    booster.fit(table[kept].fillna(0), table['is_anomaly'])
    scores = booster.predict_proba(table[kept].fillna(0))[:, 1]
    rows = scoring.rows(table, data, scores)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out, index=False)
    _log(log, {'command': 'score', 'rows': len(rows), 'out': str(out)})
    return rows


def watch(table=TABLE, out=SCORED, log=LOG, state=None, weeks=None):
    """Walk the run week by week, and keep going. Safe to stop.

    The state is used only when asked for: a command run by hand starts
    from the beginning unless it is given one. The feature cut is the
    one `prepare` wrote beside the table, so the serving model and
    every challenger read the same columns.
    """
    from runs import loop
    table_frame = _load_table(table)
    data = load.live(LIVE)
    events, _, _, _ = loop.run(table_frame, data, out=out, log=log,
                               state=state, weeks=weeks,
                               features=_kept_for(table_frame, table))
    return events


def _kept_for(table_frame, table):
    """The cut prepare wrote, else the cut this table allows."""
    meta = Path(table).with_suffix('.meta.json')
    if meta.exists():
        return json.loads(meta.read_text(encoding='utf-8'))['kept']
    return build.cut(table_frame)


def _state_for_gauges():
    """What the loop and the last measurement left behind.

    Serving the scored file alone leaves four of the eight gauges unset,
    and the dashboard's model panels then read empty while everything
    else works. These three files hold the rest.
    """
    found = {'average_precision': None, 'weekly': None, 'drift': None,
             'retrainings': 0, 'champion_version': None}
    if REPORT.exists():
        report = json.loads(REPORT.read_text(encoding='utf-8'))
        baseline = report.get('baseline')
        models = report.get('models') or {}
        serving = models.get(baseline) or next(iter(models.values()), None)
        if serving:
            found['average_precision'] = serving['average_precision']['value']
    if WATCH_STATE.exists():
        state = json.loads(WATCH_STATE.read_text(encoding='utf-8'))
        found['retrainings'] = state.get('retrainings', 0)
        found['champion_version'] = state.get('champion_version')
        found['drift'] = state.get('drift')
        weekly = state.get('weekly')
        if weekly:
            found['weekly'] = pd.DataFrame(weekly)
    return found


def serve(port=None, log=LOG):
    """Keep the gauges up for the dashboard to scrape."""
    import time
    from prometheus_client import CollectorRegistry, start_http_server

    port = int(port or exporter.PORT)
    holder = {'registry': CollectorRegistry()}

    def refresh():
        registry = CollectorRegistry()
        scored = pd.read_csv(SCORED, keep_default_na=False)
        exporter.publish(registry, scored, **_state_for_gauges())
        holder['registry'] = registry

    class Live:
        def collect(self):
            return holder['registry'].collect()

    try:
        refresh()
    except FileNotFoundError:
        pass
    start_http_server(port, registry=Live())
    _log(log, {'command': 'serve', 'port': port})
    print(f'gauges on http://localhost:{port}/metrics', flush=True)
    while True:
        time.sleep(30)
        try:
            refresh()
        except FileNotFoundError:
            pass


def main():
    words = sys.argv[1:]
    if not words or '--help' in words:
        print(__doc__)
        return
    mode = words[0]
    if mode == 'prepare':
        report = prepare(out=_option(words, '--out', TABLE),
                         log=_option(words, '--log', LOG))
        print(json.dumps(_clean(report), indent=2))
    elif mode == 'train':
        report = train(table=_option(words, '--table', TABLE),
                       log=_option(words, '--log', LOG))
        print(json.dumps(_clean(report['split']), indent=2))
    elif mode == 'score':
        score(table=_option(words, '--table', TABLE),
              out=_option(words, '--out', SCORE_FILE),
              log=_option(words, '--log', LOG))
    elif mode == 'watch':
        limit = None
        if '--weeks' in words:
            place = words.index('--weeks') + 1
            if place < len(words):
                limit = int(words[place])
        watch(table=_option(words, '--table', TABLE),
              out=_option(words, '--out', SCORED),
              log=_option(words, '--log', LOG),
              state=_option(words, '--state', WATCH_STATE),
              weeks=limit)
    elif mode == 'serve':
        port = None
        if '--port' in words:
            place = words.index('--port') + 1
            if place < len(words):
                port = int(words[place])
        serve(port=port, log=_option(words, '--log', LOG))
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()


