"""A tiny business, written by hand, so every feature can be checked by eye.

Two vendors. ACME has a contract, an order, a receipt and an account on
file. SHL is onboarded late and has no order. The invoices are chosen to
trip, or not trip, one feature each.
"""

import pandas as pd
import pytest

from data.load import Dataset


def frame(rows):
    return pd.DataFrame(rows)


@pytest.fixture
def tiny():
    invoices = frame([
        # number, issuer, kind, issued, due, total, order, shipment, pay_to, sender
        dict(number='A-1', issuer='ACM', kind='goods', issue_date='2026-01-05',
             due_date='2026-02-05', grand_total='1200.00', order_number='PO-1',
             shipment_id='S-1', pay_to='', sender='', revises=''),
        dict(number='A-2', issuer='ACM', kind='goods', issue_date='2026-01-20',
             due_date='2026-02-20', grand_total='1200.00', order_number='PO-1',
             shipment_id='S-1', pay_to='', sender='', revises=''),
        dict(number='A-3', issuer='ACM', kind='goods', issue_date='2026-02-01',
             due_date='2026-03-01', grand_total='3000.00', order_number='PO-1',
             shipment_id='S-1', pay_to='Wise IBAN 9', sender='accounts@acm-agent.example',
             revises=''),
        dict(number='S-1', issuer='SHL', kind='goods', issue_date='2026-02-10',
             due_date='2026-03-10', grand_total='5000.00', order_number='',
             shipment_id='', pay_to='', sender='accounts@shl-billing.example', revises=''),
        dict(number='F-1', issuer='CAR', kind='freight', issue_date='2026-02-11',
             due_date='2026-02-26', grand_total='2100.00', order_number='',
             shipment_id='S-1', pay_to='', sender='', revises=''),
    ])
    lines = frame([
        dict(number='A-1', position=1, code='P1', description='Cap', quantity='100',
             unit_price='12.00', net='1200.00'),
        dict(number='A-2', position=1, code='P1', description='Cap', quantity='100',
             unit_price='12.00', net='1200.00'),
        dict(number='A-3', position=1, code='P1', description='Cap', quantity='200',
             unit_price='15.00', net='3000.00'),
        dict(number='S-1', position=1, code='', description='Consultancy', quantity='1',
             unit_price='5000.00', net='5000.00'),
        dict(number='F-1', position=1, code='', description='Ocean freight Ningbo to Santos',
             quantity='1', unit_price='1900.00', net='1900.00'),
        dict(number='F-1', position=2, code='', description='Additional handling, terminal',
             quantity='1', unit_price='200.00', net='200.00'),
    ])
    erp = {
        'vendors': frame([
            dict(code='ACM', onboarded_on='2019-01-01', bank_account='ACME BANK 1',
                 email='accounts@acm-billing.example'),
            dict(code='SHL', onboarded_on='2026-02-01', bank_account='SHL BANK',
                 email='accounts@shl-billing.example'),
            dict(code='CAR', onboarded_on='2019-01-01', bank_account='',
                 email='accounts@car-billing.example'),
        ]),
        'vendor_changes': frame([], ),
        'purchase_orders': frame([dict(number='PO-1', vendor='ACM', raised_on='2025-12-01')]),
        'contracts': frame([
            dict(order_number='PO-1', code='P1', version='1', price='12.00',
                 effective_on='2025-12-01'),
        ]),
        'receipts': frame([
            dict(order_number='PO-1', code='P1', shipment_id='S-1', quantity='150',
                 received_on='2026-02-15'),
        ]),
        'rate_card': frame([
            dict(kind='freight', issuer='CAR', description='Ocean freight, per container',
                 amount='1850.00'),
            dict(kind='surcharge', issuer='CAR', description='Fuel surcharge',
                 amount='per tariff'),
        ]),
        'approval_policy': frame([dict(limit='25000.00', approver='x')]),
    }
    labels = frame([
        dict(number='A-1', anomaly='', lookalike_of='', found_after_days=''),
        dict(number='A-2', anomaly='near_duplicate', lookalike_of='', found_after_days='10'),
        dict(number='A-3', anomaly='account_mismatch', lookalike_of='', found_after_days='30'),
        dict(number='S-1', anomaly='unmatched_vendor', lookalike_of='', found_after_days='90'),
        dict(number='F-1', anomaly='freight_overbill', lookalike_of='', found_after_days='20'),
    ])
    return Dataset.from_frames('tiny', invoices, lines, erp, labels)
