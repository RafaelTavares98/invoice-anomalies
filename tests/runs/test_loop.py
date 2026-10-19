"""The loop: score, decide, watch, retrain, one week at a time."""

import numpy as np
import pandas as pd
import pytest

from data.load import Dataset
from features.build import COLUMNS
from runs import loop as loop_mod


class FixedScorer:
    """A model whose score is the row's slot: the top five are known."""

    def predict_proba(self, matrix):
        scores = (matrix.index % 20) / 20
        return np.column_stack([1 - scores, scores])


def _invoices():
    rows = []
    for week in range(8):
        for slot in range(20):
            rows.append(dict(
                number=f'W{week}-{slot}',
                issue_date=str(pd.Timestamp('2026-01-05')
                               + pd.Timedelta(weeks=week)),
                issuer='ACM', kind='goods', grand_total='100.00',
                order_number='PO-1', shipment_id='S-1',
                pay_to='', sender='', revises=''))
    return pd.DataFrame(rows)


def _labels(invoices, anomalies):
    honest = np.zeros(len(invoices), dtype=bool)
    week = invoices['number'].str.extract(r'W(\d)-')[0].astype(int)
    planted = week.isin([0, 1, 2, 5]) & invoices['number'].str.endswith('-0')
    return pd.DataFrame({
        'number': invoices['number'],
        'anomaly': np.where(planted if anomalies else honest, 'price_drift', ''),
        'lookalike_of': '',
        'found_after_days': 0,
    })


@pytest.fixture
def business():
    """Eight weeks, twenty invoices a week, features of zeros."""
    invoices = _invoices()
    labels = _labels(invoices, anomalies=True)
    lines = invoices[['number']].assign(
        position=1, code='P1', description='Cap', quantity='10',
        unit_price='10.00', net='100.00')
    erp = {
        'vendors': pd.DataFrame({
            'code': ['ACM'], 'onboarded_on': ['2019-01-01'],
            'bank_account': ['ACME BANK 1'],
            'email': ['accounts@acm-billing.example']}),
        'contracts': pd.DataFrame({
            'order_number': ['PO-1'], 'code': ['P1'], 'version': ['1'],
            'price': ['10.00'], 'effective_on': ['2025-12-01']}),
    }
    data = Dataset.from_frames('tiny-loop', invoices, lines, erp, labels)
    table = invoices[['number']].copy()
    table['issue_date'] = pd.to_datetime(invoices['issue_date'])
    for column in COLUMNS:
        table[column] = 0.0
    table['week'] = invoices['number'].str.extract(r'W(\d)-')[0].astype(int)
    table['is_anomaly'] = (labels['anomaly'] != '').astype(int)
    table['anomaly'] = labels['anomaly']
    table['lookalike_of'] = ''
    table['found_after_days'] = 0.0
    return table, data


@pytest.fixture
def held():
    return None


def test_the_loop_visits_every_week_once_in_order(business):
    table, data = business

    events, _, _, _ = loop_mod.run(
        table, data, table.iloc[:40], FixedScorer(),
        promote=lambda model, report: False)

    assert [event['week'] for event in events] == list(range(8))


def test_no_retraining_before_the_floor_is_breached(business):
    """Three bad weeks in a row are needed; the first two pass quietly."""
    table, data = business

    events, _, _, _ = loop_mod.run(
        table, data, table.iloc[:40], FixedScorer(),
        promote=lambda model, report: False)

    assert [event['retrained'] for event in events[:2]] == [False, False]


def test_deliberately_bad_labels_bring_a_retraining(business):
    """The analyst dismisses everything: precision falls to zero, the
    floor breaks, and a retraining is recorded."""
    table, data = business
    data.labels['is_anomaly'] = 0
    data.labels['anomaly'] = ''

    events, _, _, _ = loop_mod.run(
        table, data, table.iloc[:40], FixedScorer(),
        promote=lambda model, report: False)

    assert any(event['retrained'] for event in events)


def test_the_same_seed_gives_the_same_events(business):
    table, data = business

    first, _, _, _ = loop_mod.run(
        table, data, table.iloc[:40], FixedScorer(),
        promote=lambda model, report: False, seed=11)
    second, _, _, _ = loop_mod.run(
        table, data, table.iloc[:40], FixedScorer(),
        promote=lambda model, report: False, seed=11)

    assert [event['retrained'] for event in first] == \
        [event['retrained'] for event in second]
