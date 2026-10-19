"""Retraining on the analyst's labels, and promoting only a winner."""

import numpy as np
import pandas as pd
import pytest

from features.build import COLUMNS
from models import lifecycle, train


@pytest.fixture
def business():
    """Twelve weeks of one business; anomalies carry a matching feature."""
    rng = np.random.default_rng(4)
    n = 240
    table = pd.DataFrame({c: rng.random(n) for c in COLUMNS})
    table['issue_date'] = pd.Timestamp('2026-01-05') + pd.to_timedelta(
        np.arange(n) // 20 * 7, unit='D')
    table['is_anomaly'] = (rng.random(n) < 0.1).astype(int)
    table['bank_not_on_file'] = table['is_anomaly']
    table['anomaly'] = np.where(table['is_anomaly'] == 1, 'account_mismatch', '')
    table['lookalike_of'] = ''
    table['found_after_days'] = 7.0
    return table


def test_retrain_never_sees_a_row_after_the_cutoff(business):
    as_of = pd.Timestamp('2026-02-15')
    labels = pd.Series(business['is_anomaly'].to_numpy(), index=business.index)

    model, rows = lifecycle.retrain(business, as_of, labels)

    assert rows['issue_date'].max() <= as_of
    assert model is not None


def test_challenge_reports_the_same_keys_as_train_report(business):
    split = int(len(business) * 0.7)
    seen, held = business.iloc[:split], business.iloc[split:]
    fitted = train.MODELS['lightgbm']()
    scores = fitted.fit_score(seen, held)

    found = lifecycle.challenge(fitted.model, held)
    expected = train.report(held, scores)

    assert set(found) == set(expected)


def test_a_tie_does_not_take_the_crown(tmp_path):
    import mlflow
    uri = f'sqlite:///{(tmp_path / "runs.db").as_posix()}'

    first = lifecycle.register_if_better(_logged_model(uri), score=0.80,
                                         tracking_uri=uri)
    tied = lifecycle.register_if_better(_logged_model(uri), score=0.80,
                                        tracking_uri=uri)

    champion = mlflow.MlflowClient().get_model_version_by_alias(
        lifecycle.NAME, 'champion')
    assert (first, tied) == (True, False)
    assert champion.tags['average_precision'] == '0.8'


def _logged_model(uri):
    import mlflow
    from sklearn.linear_model import LogisticRegression
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment('test')
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, name='model')
    return f'runs:/{run.info.run_id}/model'
