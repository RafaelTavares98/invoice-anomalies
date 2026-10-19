"""Training and comparing models, every run recorded in MLflow."""

import numpy as np
import pandas as pd
import pytest

from features.build import COLUMNS
from models import train


@pytest.fixture
def synthetic():
    """Four hundred invoices; the anomalies carry an account not on file."""
    rng = np.random.default_rng(1)
    n = 400
    table = pd.DataFrame({c: rng.random(n) for c in COLUMNS})
    table['issue_date'] = pd.Timestamp('2026-01-05') + pd.to_timedelta(
        np.arange(n) // 20 * 7, unit='D')
    table['is_anomaly'] = (rng.random(n) < 0.08).astype(int)
    table['bank_not_on_file'] = table['is_anomaly']
    table['anomaly'] = np.where(table['is_anomaly'] == 1, 'account_mismatch', '')
    table['lookalike_of'] = ''
    table['found_after_days'] = 5.0
    table['seed'] = np.where(np.arange(n) < 300, 'train', 'test')
    return table


@pytest.mark.parametrize('name', list(train.MODELS))
def test_each_model_scores_every_test_invoice(synthetic, name):
    fitted = train.MODELS[name]()
    scores = fitted.fit_score(synthetic[synthetic.seed == 'train'],
                              synthetic[synthetic.seed == 'test'])

    assert len(scores) == 100
    assert np.isfinite(scores).all()


def test_the_supervised_model_finds_the_planted_signal(synthetic):
    scores = train.MODELS['lightgbm']().fit_score(
        synthetic[synthetic.seed == 'train'], synthetic[synthetic.seed == 'test'])
    report = train.report(synthetic[synthetic.seed == 'test'], scores, k=5)

    assert report['average_precision'] > 0.9


def test_a_comparison_is_recorded_in_mlflow(synthetic, tmp_path):
    import mlflow
    uri = f'sqlite:///{(tmp_path / "runs.db").as_posix()}'

    results = train.compare(synthetic[synthetic.seed == 'train'],
                            synthetic[synthetic.seed == 'test'],
                            tracking_uri=uri, k=5)

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[train.EXPERIMENT])
    assert set(results) == set(train.MODELS)
    assert len(runs) == len(train.MODELS) + 1          # one parent, one child each
    assert 'metrics.precision_at_k' in runs.columns
