"""Walking forward week by week, and promoting a model only when it wins."""

import numpy as np
import pandas as pd
import pytest

from features.build import COLUMNS
from models import lifecycle


@pytest.fixture
def history():
    """Twenty weeks of two parallel businesses; anomalies carry a flag."""
    rng = np.random.default_rng(2)
    frames = []
    for seed in ('a', 'b'):
        n = 400
        t = pd.DataFrame({c: rng.random(n) for c in COLUMNS})
        t['issue_date'] = pd.Timestamp('2026-01-05') + pd.to_timedelta(
            np.arange(n) // 20 * 7, unit='D')
        t['is_anomaly'] = (rng.random(n) < 0.1).astype(int)
        t['bank_not_on_file'] = t['is_anomaly']
        t['anomaly'] = np.where(t['is_anomaly'] == 1, 'account_mismatch', '')
        t['lookalike_of'] = ''
        t['found_after_days'] = 7.0
        t['seed'] = seed
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def test_each_fold_trains_on_the_past_and_tests_the_next_weeks(history):
    folds = lifecycle.walk_forward(history, 'lightgbm', first_week=8, step=4, k=2)

    assert [f['week'] for f in folds] == [8, 12, 16]
    assert all(0 <= f['precision_at_k'] <= 1 for f in folds)


def test_a_model_is_promoted_only_when_it_beats_the_champion(tmp_path):
    import mlflow
    uri = f'sqlite:///{(tmp_path / "runs.db").as_posix()}'
    mlflow.set_tracking_uri(uri)

    first = lifecycle.register_if_better(_logged_model(uri), score=0.80, tracking_uri=uri)
    worse = lifecycle.register_if_better(_logged_model(uri), score=0.70, tracking_uri=uri)
    better = lifecycle.register_if_better(_logged_model(uri), score=0.90, tracking_uri=uri)

    client = mlflow.MlflowClient()
    champion = client.get_model_version_by_alias(lifecycle.NAME, 'champion')
    assert (first, worse, better) == (True, False, True)
    assert champion.tags['average_precision'] == '0.9'


def _logged_model(uri):
    import mlflow
    from sklearn.linear_model import LogisticRegression
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment('test')
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, name='model')
    return f'runs:/{run.info.run_id}/model'


def test_a_champion_built_on_other_features_gives_way(tmp_path):
    """A champion that cannot read today's features cannot score today's
    invoices, however high its old score; the new model takes over."""
    import mlflow
    uri = f'sqlite:///{(tmp_path / "runs.db").as_posix()}'
    mlflow.set_tracking_uri(uri)

    lifecycle.register_if_better(_logged_model(uri), score=0.95,
                                 tracking_uri=uri, features=('a', 'b'))
    took_over = lifecycle.register_if_better(_logged_model(uri), score=0.80,
                                             tracking_uri=uri,
                                             features=('a', 'b', 'c'))

    champion = mlflow.MlflowClient().get_model_version_by_alias(
        lifecycle.NAME, 'champion')
    assert took_over is True
    assert champion.tags['features'] == 'a,b,c'
