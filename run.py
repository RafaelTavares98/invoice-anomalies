"""Find anomalies in the invoices project 01 simulates.

    python run.py features        build and cache the feature tables
    python run.py walk            week by week over seeds 1 to 4
    python run.py compare         train on seeds 1 to 4, test on seed 5,
                                  every model recorded in MLflow; LightGBM
                                  becomes champion only if it beats the last
    python run.py score [DATASET] [FILE]
                                  the champion scores one dataset for the
                                  dashboard (busy-seed2026, into project 03)
    python run.py metrics [PORT]  the scored file as Prometheus gauges (9109)
    python run.py ui              open the MLflow screen on port 5000

Seed 5 is held out: no model, setting or rule is tuned on it. Make the
datasets first, in project 01: python run.py dataset year 1 2 3 4 5
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / 'src'))

import pandas as pd                                  # noqa: E402

from data import load                                # noqa: E402
from evaluation import splits                        # noqa: E402
from models import feedback, lifecycle, scoring, train  # noqa: E402

CACHE = HERE / 'data' / 'features'
TRACKING = f'sqlite:///{(HERE / "mlflow.db").as_posix()}'
TRAIN_SEEDS = ('year-seed1', 'year-seed2', 'year-seed3', 'year-seed4')
HELD_OUT = 'year-seed5'


def table(name):
    """One dataset's features and answers, cached as a CSV."""
    path = CACHE / f'{name}.csv'
    if path.exists():
        return pd.read_csv(path, parse_dates=['issue_date'], keep_default_na=False,
                           na_values=[''])
    CACHE.mkdir(parents=True, exist_ok=True)
    built = train.table_for(load.read(name), name)
    built.to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=['issue_date'], keep_default_na=False,
                       na_values=[''])


def _text(frame):
    for column in ('anomaly', 'lookalike_of'):
        frame[column] = frame[column].fillna('')
    return frame


def compare():
    seen = pd.concat([table(n) for n in TRAIN_SEEDS], ignore_index=True)
    held = table(HELD_OUT)
    results = train.compare(_text(seen), _text(held), tracking_uri=TRACKING,
                            log_models=True,
                            description='Train on seeds 1 to 4, test on held-out seed 5.')
    width = max(map(len, results))
    print(f'{"model":<{width}}  top 5  recall@25  avg precision')
    for name, found in results.items():
        print(f'{name:<{width}}  {found["precision_at_k"]:>5.2f}  '
              f'{found["recall_at_25"]:>9.2f}  {found["average_precision"]:>13.2f}')
    best = results['lightgbm']
    promoted = lifecycle.register_if_better(
        f'runs:/{best["run_id"]}/model', round(best['average_precision'], 4),
        tracking_uri=TRACKING, features=train.COLUMNS)
    print('lightgbm', 'is the new champion' if promoted else 'stays challenger')


def walk():
    """Week by week over the four training years, each fold in MLflow."""
    import mlflow
    seen = _text(pd.concat([table(n) for n in TRAIN_SEEDS], ignore_index=True))
    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment(train.EXPERIMENT)
    with mlflow.start_run(run_name='walk-forward'):
        for name in ('rules', 'lightgbm'):
            with mlflow.start_run(run_name=f'walk-{name}', nested=True):
                folds = lifecycle.walk_forward(seen, name)
                for fold in folds:
                    mlflow.log_metrics({k: v for k, v in fold.items() if k != 'week'},
                                       step=fold['week'])
                mean = sum(f['average_precision'] for f in folds) / len(folds)
                worst = min(f['average_precision'] for f in folds)
                mlflow.log_metrics({'mean_average_precision': mean,
                                    'worst_average_precision': worst})
                print(f'{name:<9} {len(folds)} folds, average precision '
                      f'mean {mean:.2f}, worst {worst:.2f}')


#: The corpus the README of project 01 describes, and the dashboard shows.
DASHBOARD = 'busy-seed2026'
DASHBOARD_FILE = HERE.parent / '03-payables-panel' / 'data' / 'anomalies.csv'

#: What the loop wrote last: precision, drift, retrainings, champion.
LOOP_STATE = HERE.parent / '03-payables-panel' / 'data' / 'loop-state.json'


def score(name=DASHBOARD, out=DASHBOARD_FILE):
    """Score one dataset with the champion, for the dashboard."""
    import mlflow
    from models import scoring
    mlflow.set_tracking_uri(TRACKING)
    champion = mlflow.lightgbm.load_model(f'models:/{lifecycle.NAME}@champion')
    data = load.read(name)
    built = train.table_for(data, name)
    scores = champion.predict_proba(train._matrix(built))[:, 1]
    found = scoring.rows(built, data, scores)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    found.to_csv(out, index=False)
    print(f'{len(found)} invoices scored, {int(found.flagged.sum())} flagged, '
          f'into {out}')


def retrain(name='drift-seed7'):
    """Score a dataset, retrain on the analyst's verdicts, challenge on
    the held-out year, and register the winner. Prints four lines: the
    weeks trained on, the champion's score, the challenger's score, and
    which one serves now."""
    import mlflow
    from mlflow.models import infer_signature
    mlflow.set_tracking_uri(TRACKING)
    client = mlflow.MlflowClient()
    data = load.read(name)
    held = table(HELD_OUT)
    built = train.table_for(data, name)
    champion = mlflow.lightgbm.load_model(f'models:/{lifecycle.NAME}@champion')
    scores = champion.predict_proba(train._matrix(built))[:, 1]
    as_of = built['issue_date'].max()
    rows = scoring.rows(built, data, scores, as_of=as_of)
    joined = built.merge(rows[['number', 'flagged', 'decision',
                               'decided_on']], on='number', how='left')
    labels = feedback.labels_from(joined, as_of)
    model, seen = lifecycle.retrain(built, as_of, labels)
    found = lifecycle.challenge(model, held)
    before = client.get_model_version_by_alias(lifecycle.NAME, 'champion')
    champion_score = float(before.tags.get('average_precision', '-inf'))
    with mlflow.start_run(run_name=f'retrain-{name}'):
        mlflow.log_metrics(found)
        sample = train._matrix(seen).head(5)
        mlflow.lightgbm.log_model(
            model, name='model',
            signature=infer_signature(sample, model.predict_proba(sample)[:, 1]),
            input_example=sample)
        promoted = lifecycle.register_if_better(
            f'runs:/{mlflow.active_run().info.run_id}/model',
            round(found['average_precision'], 4), tracking_uri=TRACKING,
            features=train.COLUMNS)
    serving = client.get_model_version_by_alias(lifecycle.NAME, 'champion')
    print(f'trained on {splits.week_of(seen).nunique()} weeks, '
          f'{seen["issue_date"].min().date()} to {seen["issue_date"].max().date()}')
    print(f'champion: average precision {champion_score:.4f}')
    print(f'challenger: average precision {found["average_precision"]:.4f}')
    print(('challenger wins: version '
           f'{serving.version} serves now') if promoted else
          (f'challenger stays challenger: version '
           f'{serving.version} still serves'))


def loop(name='drift-seed7', out=DASHBOARD_FILE):
    """The whole story on one command: score, decide, watch, retrain.

    Over the drift dataset, week by week: the serving model scores, the
    analyst decides, the decisions become labels, the precision and the
    drift are watched, and a fall retrains and challenges the champion.
    Every week's scored rows are written to the dashboard file."""
    import mlflow
    from mlflow.models import infer_signature
    from runs import loop as loop_mod
    mlflow.set_tracking_uri(TRACKING)
    data = load.read(name)
    built = train.table_for(data, name)
    built = built.assign(week=splits.week_of(built))
    held = table(HELD_OUT)
    champion = mlflow.lightgbm.load_model(f'models:/{lifecycle.NAME}@champion')

    def promote(model, report):
        with mlflow.start_run(run_name='loop-retrain'):
            mlflow.log_metrics(report)
            sample = train._matrix(built).head(5)
            mlflow.lightgbm.log_model(
                model, name='model',
                signature=infer_signature(sample,
                                          model.predict_proba(sample)[:, 1]),
                input_example=sample)
            return lifecycle.register_if_better(
                f'runs:/{mlflow.active_run().info.run_id}/model',
                round(report['average_precision'], 4), tracking_uri=TRACKING,
                features=train.COLUMNS)

    events, weekly, drifts, scored = loop_mod.run(
        built, data, held, champion, promote)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    retrainings = sum(1 for event in events if event['retrained'])
    import json
    serving = mlflow.MlflowClient().get_model_version_by_alias(
        lifecycle.NAME, 'champion')
    LOOP_STATE.write_text(json.dumps({
        'last_precision': (float(weekly['precision'].dropna().iloc[-1])
                           if weekly['precision'].notna().any() else None),
        'drift': drifts,
        'retrainings': retrainings,
        'champion_version': str(serving.version),
    }, indent=2), encoding='utf-8')
    print(f'{len(events)} weeks, {retrainings} retraining(s), '
          f'{int(scored["flagged"].sum())} flags, into {out}')
    for event in events:
        if event['retrained']:
            print(f'  week {event["week"]}: precision {event["precision"]:.2f}, '
                  f'drift [{event["drifted"]}], retrained, '
                  f'{"promoted" if event["promoted"] else "stays challenger"}')


def metrics(port=None):
    """The scored file as Prometheus gauges, for project 01's Grafana.
    Serves until stopped; the file is read again every 30 seconds."""
    import time
    import mlflow
    import pandas as pd
    from prometheus_client import CollectorRegistry, start_http_server
    from models import exporter
    mlflow.set_tracking_uri(TRACKING)
    client = mlflow.MlflowClient()
    precision = float(client.get_model_version_by_alias(
        lifecycle.NAME, 'champion').tags['average_precision'])
    holder = {'registry': CollectorRegistry()}
    port = int(port or exporter.PORT)

    def refresh():
        import json
        registry = CollectorRegistry()
        state = {}
        if LOOP_STATE.exists():
            state = json.loads(LOOP_STATE.read_text(encoding='utf-8'))
        weekly = None
        if state.get('last_precision') is not None:
            weekly = pd.DataFrame([{'precision': state['last_precision']}])
        exporter.publish(registry, pd.read_csv(DASHBOARD_FILE,
                                               keep_default_na=False),
                         precision, weekly=weekly,
                         drift=state.get('drift') or {},
                         retrainings=state.get('retrainings', 0),
                         champion_version=state.get('champion_version'))
        holder['registry'] = registry

    refresh()

    class Live:
        def collect(self):
            return holder['registry'].collect()

    start_http_server(port, registry=Live())
    print(f'anomaly metrics on http://localhost:{port}/metrics', flush=True)
    while True:
        time.sleep(30)
        refresh()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'compare'
    if mode == 'features':
        for name in (*TRAIN_SEEDS, HELD_OUT):
            print(name, len(table(name)), 'invoices')
    elif mode == 'compare':
        compare()
    elif mode == 'walk':
        walk()
    elif mode == 'score':
        score(*sys.argv[2:4])
    elif mode == 'retrain':
        retrain(*sys.argv[2:3])
    elif mode == 'loop':
        loop(*sys.argv[2:4])
    elif mode == 'metrics':
        metrics(*sys.argv[2:3])
    elif mode == 'ui':
        subprocess.run([sys.executable, '-m', 'mlflow', 'ui',
                        '--backend-store-uri', TRACKING, '--port', '5000'])
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
