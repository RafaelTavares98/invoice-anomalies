"""Each feature on the tiny business, and the rule that none reads the future.

An invoice is reviewed on its due date, before it is paid, so a feature
may use any record dated on or before the due date, and no other. Its
history features look only at earlier invoices of the same issuer.
"""

import math

import pandas as pd
import pytest

from data.load import Dataset
from features import build


@pytest.fixture
def table(tiny):
    return build.features(tiny).set_index('number')


@pytest.fixture
def split(tiny):
    """Two bills for the same order and the same box, two days apart."""
    extra = tiny.raw_invoices.iloc[[0, 0]].copy()
    extra['number'] = ['B-1', 'B-2']
    extra['issue_date'] = ['2026-03-02', '2026-03-04']
    extra['shipment_id'] = 'S-9'
    labels = tiny.raw_labels.iloc[[0, 0]].copy()
    labels['number'] = ['B-1', 'B-2']
    data = Dataset.from_frames(
        'split', pd.concat([tiny.raw_invoices, extra]), tiny.raw_lines,
        tiny.raw_erp, pd.concat([tiny.raw_labels, labels]))
    return build.features(data).set_index('number')


class TestAgainstTheContract:
    def test_a_price_above_the_contract(self, table):
        assert table.loc['A-3', 'price_over_contract'] == pytest.approx(0.25)
        assert table.loc['A-1', 'price_over_contract'] == 0

    def test_a_quantity_above_the_receipt(self, table):
        assert table.loc['A-3', 'quantity_over_receipt'] == pytest.approx(1 / 3)

    def test_a_receipt_not_yet_signed_is_unknown(self, table):
        assert table.loc['A-1', 'receipt_known'] == 0
        assert math.isnan(table.loc['A-1', 'quantity_over_receipt'])

    def test_a_charge_missing_from_the_rate_card(self, table):
        assert table.loc['F-1', 'unknown_charges'] == 1
        assert table.loc['A-1', 'unknown_charges'] == 0


class TestAgainstTheVendorMaster:
    def test_an_account_not_on_file(self, table):
        assert table.loc['A-3', 'bank_not_on_file'] == 1
        assert table.loc['A-1', 'bank_not_on_file'] == 0

    def test_a_sender_not_on_file(self, table):
        assert table.loc['A-3', 'sender_not_on_file'] == 1
        assert table.loc['S-1', 'sender_not_on_file'] == 0

    def test_a_young_vendor_without_an_order(self, table):
        assert table.loc['S-1', 'vendor_age_days'] == 9
        assert table.loc['S-1', 'has_order'] == 0
        assert table.loc['A-1', 'has_order'] == 1


class TestAgainstTheIssuersHistory:
    def test_the_same_amount_again(self, table):
        assert table.loc['A-2', 'same_amount_30d'] == 1
        assert table.loc['A-1', 'same_amount_30d'] == 0

    def test_a_second_bill_for_the_same_box_within_a_week(self, split):
        assert split.loc['B-2', 'same_shipment_7d'] == 1
        assert split.loc['B-1', 'same_shipment_7d'] == 0

    def test_days_since_the_last_invoice(self, table):
        assert table.loc['A-2', 'days_since_last'] == 15
        assert math.isnan(table.loc['A-1', 'days_since_last'])


class TestSplitOrPartial:
    def test_bills_for_one_order_on_one_shipment_are_counted(self, table):
        """A split cuts one delivery into pieces: same order, same box. A
        partial delivery bills each box apart."""
        assert table.loc['A-2', 'same_shipment_7d'] == 0
        assert table.loc['A-1', 'same_shipment_7d'] == 0


class TestDigits:
    def test_a_round_total(self, table):
        assert table.loc['S-1', 'round_thousands'] == 1
        assert table.loc['F-1', 'round_thousands'] == 0


class TestNoFuture:
    def test_later_invoices_change_nothing_earlier(self, tiny):
        """Cut the business after A-2; A-1 and A-2 must not move."""
        cut = Dataset.from_frames(
            'cut', tiny.raw_invoices.iloc[:2], tiny.raw_lines, tiny.raw_erp,
            tiny.raw_labels.iloc[:2])
        full = build.features(tiny).set_index('number').loc[['A-1', 'A-2']]
        early = build.features(cut).set_index('number')

        pd.testing.assert_frame_equal(full, early, check_dtype=False)

    def test_no_answer_is_a_feature(self, tiny):
        columns = set(build.features(tiny).columns)

        assert not columns & {'anomaly', 'lookalike_of', 'found_after_days',
                              'is_anomaly'}
