"""Moty's own run is the only data this project reads.

The five invented training years are gone. What replaces them is the
run the earlier pipeline wrote into `Invoices/live/`: one operation,
one history, 1,044 invoices. These tests pin the loader's contract on
a run written by hand, and then check the real one against the four
numbers the README publishes.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from data import load

#: Where the earlier pipeline writes its live run.
LIVE = (Path(__file__).resolve().parents[3]
        / '01-invoice-pipeline' / 'Invoices' / 'live')

#: The four numbers the README publishes. If the run changes, these
#: change with it, and so does the README.
INVOICES = 1003
ERRORS = 35
FALSE_POSITIVES = 46
FIRST_DAY = '2026-01-02'
LAST_DAY = '2026-10-14'


def _write_run(root):
    """A run of six invoices, written by hand, in the live layout."""
    root.mkdir(parents=True, exist_ok=True)
    (root / 'erp').mkdir(exist_ok=True)

    invoices = pd.DataFrame([
        dict(number='A-1', issuer='ACM', kind='goods', issue_date='2026-01-05',
             due_date='2026-02-05', grand_total='1200.00', order_number='PO-1',
             shipment_id='S-1', pay_to='', sender='', revises=''),
        dict(number='A-2', issuer='ACM', kind='goods', issue_date='2026-02-10',
             due_date='2026-03-10', grand_total='1200.00', order_number='PO-1',
             shipment_id='S-1', pay_to='', sender='', revises=''),
        dict(number='F-1', issuer='CAR', kind='freight', issue_date='2026-03-02',
             due_date='2026-03-17', grand_total='2100.00', order_number='',
             shipment_id='S-1', pay_to='', sender='', revises=''),
        dict(number='F-2', issuer='CAR', kind='freight', issue_date='2026-04-02',
             due_date='2026-04-17', grand_total='2000.00', order_number='',
             shipment_id='S-1', pay_to='', sender='', revises=''),
        dict(number='B-1', issuer='SCA', kind='broker', issue_date='2026-05-06',
             due_date='2026-06-06', grand_total='800.00', order_number='',
             shipment_id='S-1', pay_to='', sender='', revises=''),
        dict(number='G-1', issuer='SHL', kind='goods', issue_date='2026-06-01',
             due_date='2026-07-01', grand_total='5000.00', order_number='',
             shipment_id='', pay_to='', sender='', revises=''),
    ])
    lines = pd.DataFrame([
        dict(number=n, position=1, code='P1', description='Cap',
             quantity='10', unit_price='120.00', net='1200.00')
        for n in invoices['number']
    ])
    labels = pd.DataFrame([
        dict(number='A-1', anomaly='', lookalike_of='', found_after_days=''),
        dict(number='A-2', anomaly='near_duplicate', lookalike_of='',
             found_after_days='10'),
        dict(number='F-1', anomaly='freight_overbill', lookalike_of='',
             found_after_days='20'),
        dict(number='F-2', anomaly='', lookalike_of='freight_overbill',
             found_after_days=''),
        dict(number='B-1', anomaly='broker_fee_over', lookalike_of='',
             found_after_days='15'),
        dict(number='G-1', anomaly='', lookalike_of='unmatched_vendor',
             found_after_days=''),
    ])
    vendors = pd.DataFrame([
        dict(code='ACM', onboarded_on='2019-01-01', bank_account='ACME BANK 1',
             email='accounts@acm.example'),
        dict(code='CAR', onboarded_on='2019-01-01', bank_account='CAR BANK',
             email='accounts@car.example'),
        dict(code='SCA', onboarded_on='2019-01-01', bank_account='SCA BANK',
             email='accounts@sca.example'),
        dict(code='SHL', onboarded_on='2026-05-20', bank_account='SHL BANK',
             email='accounts@shl.example'),
    ])

    invoices.to_csv(root / 'invoices.csv', index=False)
    lines.to_csv(root / 'invoice_lines.csv', index=False)
    labels.to_csv(root / 'labels.csv', index=False)
    vendors.to_csv(root / 'erp' / 'vendors.csv', index=False)
    (root / 'ledger.json').write_text(json.dumps({'invoices': 6}),
                                      encoding='utf-8')
    return root


@pytest.fixture
def run(tmp_path):
    return _write_run(tmp_path / 'live')


def test_the_run_loads_as_one_set(run):
    data = load.live(run)

    assert len(data.invoices) == 6
    assert len(data.labels) == 6
    assert set(data.erp) >= {'vendors'}


def test_every_invoice_has_its_answer(run):
    data = load.live(run)

    assert set(data.invoices['number']) == set(data.labels['number'])
    assert data.labels['number'].is_unique


def test_an_error_and_a_false_positive_are_told_apart(run):
    data = load.live(run)
    labels = data.labels.set_index('number')

    assert labels.at['F-1', 'is_anomaly'] == 1
    assert labels.at['F-2', 'is_anomaly'] == 0
    assert labels.at['F-2', 'lookalike_of'] == 'freight_overbill'


def test_the_delay_until_an_error_is_found_survives_loading(run):
    data = load.live(run)
    labels = data.labels.set_index('number')

    assert labels.at['A-2', 'found_after_days'] == 10
    assert pd.isna(labels.at['A-1', 'found_after_days'])


def test_dates_come_back_as_dates(run):
    data = load.live(run)

    assert data.invoices['issue_date'].dtype.kind == 'M'
    assert data.invoices['issue_date'].min() == pd.Timestamp('2026-01-05')


def test_a_run_missing_its_answers_is_refused(run, tmp_path):
    (run / 'labels.csv').unlink()

    with pytest.raises(FileNotFoundError, match='labels'):
        load.live(run)


# --------------------------------------------------- the real run

def test_the_real_run_matches_the_readme():
    """The four numbers on the README's front page.

    This needs the live run on disk. It is not skipped when the folder
    is missing: a suite that quietly proves nothing is worse than one
    that fails.

    The key holds 1,044 rows and 1,003 numbers: 41 were issued twice,
    revised or resent. One invoice is one number, so the loader keeps
    one row per number and the counts below are over numbers.
    """
    assert LIVE.exists(), f'the live run is not at {LIVE}'
    data = load.live(LIVE)

    assert data.labels['number'].is_unique
    assert len(data.invoices) == INVOICES
    assert int(data.labels['is_anomaly'].sum()) == ERRORS
    assert int((data.labels['lookalike_of'] != '').sum()) == FALSE_POSITIVES
    assert str(data.invoices['issue_date'].min().date()) == FIRST_DAY
    assert str(data.invoices['issue_date'].max().date()) == LAST_DAY


def test_the_three_kinds_that_never_happened():
    """Three error kinds are absent from this run, and two of them show
    up only as false positives. A model is asked about them without
    having seen the real thing, so the loader must not hide them."""
    assert LIVE.exists(), f'the live run is not at {LIVE}'
    data = load.live(LIVE)
    errors = set(data.labels.loc[data.labels['anomaly'] != '', 'anomaly'])
    twins = set(data.labels.loc[data.labels['lookalike_of'] != '',
                                'lookalike_of'])

    assert {'account_mismatch', 'round_total', 'split_billing'} & errors == set()
    assert {'account_mismatch', 'round_total'} <= twins
