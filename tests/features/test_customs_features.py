"""The two customs checks: a duty against the tariff table, a broker fee
against the contract.

A rate that changed by a published resolution is not an error; a fee the
contract was amended to allow is not an error. Both look like one.
"""

import pandas as pd
import pytest

from data.load import Dataset
from features import build


def frame(rows):
    return pd.DataFrame(rows)


def _doc(number, kind, day, total):
    return dict(number=number, issuer='SCA', kind=kind, issue_date=day,
                due_date=day, grand_total=total, order_number='',
                shipment_id='B/L X1', pay_to='', sender='', revises='')


@pytest.fixture
def table():
    invoices = frame([
        _doc('SCA-1', 'customs', '2026-04-10', '1000.00'),   # right rate
        _doc('SCA-2', 'customs', '2026-04-11', '1080.00'),   # 18% keyed for 16%
        _doc('SCA-3', 'customs', '2026-06-01', '700.00'),    # after the cut, new rate
        _doc('SCA-4', 'broker', '2026-04-10', '1450.00'),    # at contract
        _doc('SCA-5', 'broker', '2026-04-11', '2030.00'),    # 40% over
        _doc('SCA-6', 'broker', '2026-04-12', '1595.00'),    # amended, 10% over
    ])
    lines = frame([
        dict(number='SCA-1', position=1, code='', description='II 7612.10.00',
             quantity='6250.00', unit_price='0.16', net='1000.00'),
        dict(number='SCA-2', position=1, code='', description='II 7612.10.00',
             quantity='6000.00', unit_price='0.18', net='1080.00'),
        dict(number='SCA-3', position=1, code='', description='II 7612.10.00',
             quantity='6250.00', unit_price='0.112', net='700.00'),
        dict(number='SCA-4', position=1, code='', description='Customs clearance fee',
             quantity='1', unit_price='1450.00', net='1450.00'),
        dict(number='SCA-5', position=1, code='', description='Customs clearance fee',
             quantity='1', unit_price='2030.00', net='2030.00'),
        dict(number='SCA-6', position=1, code='',
             description='Customs clearance fee (amended)',
             quantity='1', unit_price='1595.00', net='1595.00'),
    ])
    erp = {
        'vendors': frame([dict(code='SCA', onboarded_on='2019-01-01',
                               bank_account='', email='x')]),
        'tariffs': frame([
            dict(ncm='7612.10.00', ii='0.16', effective_on='2019-01-01'),
            dict(ncm='7612.10.00', ii='0.112', effective_on='2026-05-15'),
        ]),
        'rate_card': frame([dict(kind='fixed_fee', issuer='SCA',
                                 description='Customs clearance fee',
                                 amount='1450.00')]),
        'approval_policy': frame([dict(limit='25000.00', approver='x')]),
    }
    labels = frame([dict(number=n, anomaly='', lookalike_of='',
                         found_after_days='') for n in invoices['number']])
    data = Dataset.from_frames('customs', invoices, lines, erp, labels)
    return build.features(data).set_index('number')


def test_a_duty_at_the_rate_in_force_is_clean(table):
    assert table.loc['SCA-1', 'duty_rate_off'] == 0


def test_a_duty_keyed_at_another_rate_stands_out(table):
    assert table.loc['SCA-2', 'duty_rate_off'] == pytest.approx(0.02)


def test_a_published_cut_is_not_an_error(table):
    assert table.loc['SCA-3', 'duty_rate_off'] == 0


def test_a_fee_at_contract_is_clean(table):
    assert table.loc['SCA-4', 'fee_over_contract'] == 0


def test_a_fee_over_contract_stands_out(table):
    assert table.loc['SCA-5', 'fee_over_contract'] == pytest.approx(0.40)


def test_an_amended_fee_says_so(table):
    assert table.loc['SCA-6', 'fee_amended'] == 1
    assert table.loc['SCA-5', 'fee_amended'] == 0


def test_other_documents_carry_zeros(tiny):
    found = build.features(tiny)

    assert (found['duty_rate_off'] == 0).all()
    assert (found['fee_over_contract'] == 0).all()
