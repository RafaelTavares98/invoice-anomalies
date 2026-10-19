"""What the reader's mistakes cost the model.

Every figure the project has reported so far came off the simulation's
own numbers, which are always right. A desk is handed what somebody
read off a page, and the reader gets about one invoice in ten wrong.
A score that has never met a misread total is a score nobody should
quote.

These tests build both tables by hand so they run in a second. The real
pair of datasets is measured once and written into the runbook.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation import reading


def table(rows=120, noise=0.0, seed=0):
    """A small scored table: one clean signal, with noise on top."""
    rng = np.random.default_rng(seed)
    days = pd.to_datetime('2026-04-06') + pd.to_timedelta(
        np.arange(rows) * 2, unit='D')
    is_anomaly = (np.arange(rows) % 7 == 0).astype(int)
    signal = is_anomaly + rng.normal(0, noise, rows)
    return pd.DataFrame({
        # The year sits in the number as it does on a real invoice, so
        # a date read wrong by a century can be caught against it.
        'number': [f'NCL-2026-{n:04d}' for n in range(rows)],
        'issue_date': days,
        'grand_total': 1000 + signal * 500,
        'is_anomaly': is_anomaly,
        'anomaly': np.where(is_anomaly == 1, 'price_above_contract', ''),
        'lookalike_of': '',
        'found_after_days': 10.0,
    })


def by_amount(train, test):
    """A stand-in model: the bigger the bill, the more suspicious.

    These tests are about the comparison, not about the champion. A
    scorer with no settings keeps them to a second and makes the
    expected order of the two figures obvious.
    """
    return test['grand_total'].to_numpy()


class TestTheComparison:
    def test_it_reports_a_figure_for_each_table(self):
        found = reading.beside(table(), table(noise=0.6),
                               fit_score=by_amount)

        assert set(found) == {'perfect', 'read', 'cost', 'invoices_perfect',
                              'invoices_read', 'undated_perfect',
                              'undated_read', 'misdated_perfect',
                              'misdated_read', 'only_perfect',
                              'only_read'}

    def test_perfect_data_scores_higher_than_read_data(self):
        """The whole point: reading mistakes cost something, and the
        number says how much."""
        found = reading.beside(table(), table(noise=0.9),
                               fit_score=by_amount)

        assert found['perfect'] > found['read']

    def test_the_cost_is_the_difference(self):
        found = reading.beside(table(), table(noise=0.9),
                               fit_score=by_amount)

        assert found['cost'] == pytest.approx(
            found['perfect'] - found['read'])

    def test_identical_tables_cost_nothing(self):
        found = reading.beside(table(), table(), fit_score=by_amount)

        assert found['cost'] == pytest.approx(0.0, abs=1e-9)

    def test_it_scores_the_same_invoices_on_both_sides(self):
        """A smaller table scores worse whatever the reader did, so the
        two sides are cut to the invoices they share. Otherwise the
        figure measures the sample size, not the reading."""
        found = reading.beside(table(rows=120), table(rows=90),
                               fit_score=by_amount)

        assert found['invoices_perfect'] == found['invoices_read'] == 90

    def test_the_invoices_only_one_side_has_are_counted(self):
        found = reading.beside(table(rows=120), table(rows=90),
                               fit_score=by_amount)

        assert found['only_perfect'] == 30
        assert found['only_read'] == 0

    def test_a_table_too_small_to_split_is_refused(self):
        """Fewer weeks than the warm-up leaves no test week at all, and
        a score over no invoices is not a score."""
        with pytest.raises(reading.NotEnoughWeeks):
            reading.beside(table(rows=6), table(rows=6),
                           fit_score=by_amount)


class TestAnInvoiceWithNoDate:
    """The reader misses a date now and then. Nothing can be done with
    such a row: every feature is measured on the review day, and the
    split walks forward a week at a time. It is dropped, and the drop
    is counted, because it is itself part of what reading costs."""

    def test_it_is_not_scored(self):
        broken = table()
        broken.loc[3, 'issue_date'] = pd.NaT

        found = reading.beside(table(), broken, fit_score=by_amount)

        assert found['invoices_read'] == len(broken) - 1

    def test_the_drop_is_reported(self):
        broken = table()
        broken.loc[[3, 9], 'issue_date'] = pd.NaT

        found = reading.beside(table(), broken, fit_score=by_amount)

        assert found['undated_read'] == 2
        assert found['undated_perfect'] == 0


class TestADateReadWrongByCenturies:
    """A misread date put one invoice in the year 2176. The split walks
    forward a week at a time, so the table went from eight weeks to two
    and a half thousand, almost all of them empty, and nothing could be
    scored.

    The invoice number carries its own year, and the number was read off
    the same page. A date whose year disagrees with it was read wrong.
    """

    def test_a_year_that_fights_the_number_is_dropped(self):
        broken = table()
        broken.loc[4, 'issue_date'] = pd.Timestamp('2176-07-01')

        kept = reading.plausible(broken, year=2026)

        assert len(kept) == len(broken) - 1

    def test_a_year_that_agrees_is_kept(self):
        assert len(reading.plausible(table(), year=2026)) == len(table())

    def test_the_year_is_read_off_the_numbers(self):
        """Nothing here is told what year the run was: it is the year
        most of the invoice numbers carry."""
        assert reading.run_year(table()) == 2026

    def test_the_drop_is_reported(self):
        broken = table()
        broken.loc[4, 'issue_date'] = pd.Timestamp('2176-07-01')

        found = reading.beside(table(), broken, fit_score=by_amount)

        assert found['misdated_read'] == 1


class TestItNeverLooksAhead:
    def test_the_score_uses_only_earlier_weeks(self, monkeypatch):
        """The no-future rule holds here too: a model is fitted on the
        weeks before the one it is scored on, never on it."""
        seen = []

        def remember(train, test):
            seen.append((train['issue_date'].max(),
                         test['issue_date'].min()))
            return np.zeros(len(test))

        reading.beside(table(), table(), fit_score=remember)

        for last_trained, first_tested in seen:
            assert last_trained < first_tested
