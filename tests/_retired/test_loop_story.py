# Retired 2026-10-19: it told the loop's story on drift-seed7, and that set is gone.
"""The loop tells a story on the drift dataset: a fall, a retrain or
two, a promotion, and a recovery. Never a retrain every week."""

import numpy as np
import pandas as pd
import pytest

from data import load
from evaluation import splits
from features.build import COLUMNS
from runs import loop as loop_mod


@pytest.fixture(scope='module')
def story():
    """The drift dataset with the champion as delivered, version 4.

    A version never changes, so the test does not depend on the mutable
    champion alias, and never touches the registry itself."""
    import mlflow
    from pathlib import Path
    from models import train
    root = Path(__file__).resolve().parents[2]
    db = root / 'mlflow.db'
    if not db.exists():
        pytest.skip('no local MLflow store')
    mlflow.set_tracking_uri(f'sqlite:///{db.as_posix()}')
    try:
        champion = mlflow.lightgbm.load_model('models:/invoice-anomaly/4')
    except Exception:
        pytest.skip('model version 4 is not in the local store')
    data = load.read('drift-seed7')
    table = train.table_for(data, 'drift-seed7')
    table = table.assign(week=splits.week_of(table))
    floor = loop_mod.training_floor(champion)
    events, weekly, _, _ = loop_mod.run(
        table, data, None, champion,
        promote=lambda model, verdict: True, floor=floor)
    return events, weekly


def test_the_loop_retrains_between_one_and_four_times(story):
    events, _ = story

    retrains = [event for event in events if event['retrained']]
    assert 1 <= len(retrains) <= 4


def test_at_least_one_challenger_is_promoted(story):
    events, _ = story

    assert any(event['promoted'] for event in events)


def test_a_promoted_model_beats_the_one_it_replaced(story):
    events, _ = story

    for event in (e for e in events if e['promoted']):
        assert event['challenger_precision'] > event['verdict_served']


class FixedScorer:
    def predict_proba(self, matrix):
        scores = (matrix.index % 20) / 20
        return np.column_stack([1 - scores, scores])


def _doubling_business():
    """Twelve weeks; one feature sits still, then doubles halfway."""
    rng = np.random.default_rng(9)
    rows, labels = [], []
    for week in range(12):
        mean = 1.5 if week >= 6 else 0.0
        for slot in range(20):
            number = f'W{week}-{slot}'
            rows.append(dict(
                number=number,
                issue_date=str(pd.Timestamp('2026-01-05')
                               + pd.Timedelta(weeks=week)),
                issuer='ACM', kind='goods', grand_total='100.00',
                order_number='PO-1', shipment_id='S-1',
                pay_to='', sender='', revises='',
                z=rng.normal(mean, 1.0)))
            labels.append(dict(number=number, anomaly='', lookalike_of='',
                               found_after_days=0))
    invoices = pd.DataFrame(rows)
    from data.load import Dataset
    lines = invoices[['number']].assign(
        position=1, code='P1', description='Cap', quantity='10',
        unit_price='10.00', net='100.00')
    erp = {'vendors': pd.DataFrame({
        'code': ['ACM'], 'onboarded_on': ['2019-01-01'],
        'bank_account': ['ACME BANK 1'],
        'email': ['accounts@acm-billing.example']})}
    data = Dataset.from_frames('doubling', invoices, lines, erp,
                               pd.DataFrame(labels))
    table = invoices[['number']].copy()
    table['issue_date'] = pd.to_datetime(invoices['issue_date'])
    for column in COLUMNS:
        table[column] = 0.0
    table['amount_zscore'] = invoices['z']
    table['week'] = invoices['number'].str.extract(r'W(\d+)-')[0].astype(int)
    table['is_anomaly'] = 0
    table['anomaly'] = ''
    table['lookalike_of'] = ''
    table['found_after_days'] = 0.0
    return table, data


def test_drift_ignores_a_healthy_stretch_and_fires_on_a_double():
    table, data = _doubling_business()

    events, _, _, _ = loop_mod.run(
        table, data, None, FixedScorer(),
        promote=lambda model, verdict: True, floor=0.0)

    healthy = [event for event in events if 1 <= event['week'] <= 5]
    doubled = [event for event in events if 6 <= event['week'] <= 11]
    assert all(event['drifted'] == '' for event in healthy)
    assert any(event['drifted'] for event in doubled)
