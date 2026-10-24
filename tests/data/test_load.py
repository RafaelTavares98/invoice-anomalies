"""Matching invoices to their answers, on a business written by hand."""

import pandas as pd

from data import load


class TestTheTinyBusiness:
    def test_dates_and_amounts_are_typed(self, tiny):
        assert pd.api.types.is_datetime64_any_dtype(tiny.invoices['issue_date'])
        assert tiny.invoices['grand_total'].dtype == float

    def test_labels_line_up_with_invoices(self, tiny):
        assert list(tiny.labels['number']) == list(tiny.invoices['number'])

    def test_an_anomaly_is_a_one(self, tiny):
        assert list(tiny.labels['is_anomaly']) == [0, 1, 1, 1, 1]


class TestResends:
    def test_an_exact_resend_is_dropped_with_its_label(self, tiny):
        """The reader's hash drops a resend; the model never sees it."""
        doubled = load.Dataset.from_frames(
            'doubled',
            pd.concat([tiny.raw_invoices, tiny.raw_invoices.iloc[[0]]]),
            tiny.raw_lines, tiny.raw_erp,
            pd.concat([tiny.raw_labels, tiny.raw_labels.iloc[[0]]]))

        assert len(doubled.invoices) == len(tiny.invoices)
        assert len(doubled.labels) == len(tiny.labels)


class TestLabelsThatDoNotLineUp:
    """A dataset of what was read holds only the invoices somebody drew
    and the reader kept, while the answers cover the whole run. Pairing
    them off by position puts the wrong answer on every row."""

    def test_the_answers_are_matched_by_number(self, tiny):
        more = pd.concat([tiny.raw_labels,
                          tiny.raw_labels.assign(number='ZZZ-9999')])

        data = load.Dataset.from_frames(
            'partial', tiny.raw_invoices, tiny.raw_lines, tiny.raw_erp,
            more)

        assert list(data.labels['number']) == list(data.invoices['number'])

    def test_invoices_the_answers_do_not_cover_are_dropped(self, tiny):
        """A row with no answer cannot be scored, and keeping it would
        count as a clean invoice something nobody checked."""
        fewer = tiny.raw_labels.iloc[1:]

        data = load.Dataset.from_frames(
            'partial', tiny.raw_invoices, tiny.raw_lines, tiny.raw_erp,
            fewer)

        assert len(data.invoices) == len(data.labels) == len(fewer)

    def test_the_answers_still_say_what_they_said(self, tiny):
        shuffled = tiny.raw_labels.iloc[::-1].reset_index(drop=True)

        data = load.Dataset.from_frames(
            'shuffled', tiny.raw_invoices, tiny.raw_lines, tiny.raw_erp,
            shuffled)

        assert list(data.labels['is_anomaly']) == [0, 1, 1, 1, 1]
