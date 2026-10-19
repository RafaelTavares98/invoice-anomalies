"""Scores turned into rows a dashboard can read.

Each invoice gets the champion's score, whether it lands in its week's
top five, and, from the desk's rules, the stage where the error most
likely happened and the check that points at it.
"""

import pandas as pd
import pytest

from features import build
from models import scoring


@pytest.fixture
def rows(tiny):
    table = build.features(tiny)
    scores = [0.10, 0.80, 0.95, 0.70, 0.60]
    return scoring.rows(table, tiny, scores, k=2)


def test_one_row_per_invoice(rows, tiny):
    assert list(rows['number']) == list(tiny.invoices['number'])


def test_without_names_on_file_the_supplier_is_its_code(rows):
    assert list(rows['supplier']) == ['ACM', 'ACM', 'ACM', 'SHL', 'CAR']


def test_with_names_on_file_the_supplier_is_named(tiny):
    tiny.erp['vendors']['name'] = ['Acme Caps', 'Shl Services', 'Carrier Co']
    named = scoring.rows(build.features(tiny), tiny, [0.1] * 5, k=2)

    assert named['supplier'].iloc[0] == 'Acme Caps'


def test_the_top_k_of_each_week_is_flagged(rows):
    flagged = rows[rows['flagged'] == 1]

    assert flagged.groupby('week').size().max() <= 2
    assert rows.loc[rows['number'] == 'A-3', 'flagged'].item() == 1


def test_the_stage_comes_from_the_heaviest_rule_that_fired(rows):
    stage = dict(zip(rows['number'], rows['stage']))
    check = dict(zip(rows['number'], rows['check']))

    assert stage['A-3'] == 'vendor master'
    assert check['A-3'] == 'account not on file'
    assert stage['F-1'] == 'freight billing'
    assert stage['S-1'] == 'purchasing'


def test_an_invoice_no_rule_explains_says_so(rows):
    assert rows.loc[rows['number'] == 'A-1', 'stage'].item() == 'model only'


def test_every_rule_has_a_stage():
    from models import rules
    assert {name for name, _, _ in rules.RULES} == set(scoring.STAGE)


def test_no_answer_leaks_into_the_rows(rows):
    assert not set(rows.columns) & {'anomaly', 'lookalike_of', 'is_anomaly'}


class TestWhatTheAnalystDecided:
    """A flag is confirmed once its error has been found, dismissed after
    a short review when it was honest, and pending until then."""

    def test_each_flag_has_a_decision_and_an_origin(self, tiny):
        table = build.features(tiny)
        found = scoring.rows(table, tiny, [0.1, 0.8, 0.95, 0.7, 0.6], k=5,
                             as_of=pd.Timestamp('2026-06-30'))
        flagged = found[found['flagged'] == 1]

        assert set(flagged['decision']) <= {'confirmed', 'dismissed', 'pending'}
        assert set(flagged['origin']) == {'client'}

    def test_an_error_found_by_the_day_is_confirmed(self, tiny):
        table = build.features(tiny)
        found = scoring.rows(table, tiny, [0.1, 0.8, 0.95, 0.7, 0.6], k=5,
                             as_of=pd.Timestamp('2026-06-30'))

        decision = dict(zip(found['number'], found['decision']))
        assert decision['A-2'] == 'confirmed'      # found 10 days after 20 Jan

    def test_an_error_not_yet_found_is_pending(self, tiny):
        table = build.features(tiny)
        found = scoring.rows(table, tiny, [0.1, 0.8, 0.95, 0.7, 0.6], k=5,
                             as_of=pd.Timestamp('2026-02-15'))

        decision = dict(zip(found['number'], found['decision']))
        assert decision['S-1'] == 'pending'        # takes 90 days

    def test_an_honest_flag_is_dismissed_after_review(self, tiny):
        table = build.features(tiny)
        found = scoring.rows(table, tiny, [0.9, 0.1, 0.1, 0.1, 0.1], k=1,
                             as_of=pd.Timestamp('2026-06-30'))

        decision = dict(zip(found['number'], found['decision']))
        assert decision['A-1'] == 'dismissed'

    def test_an_unflagged_invoice_has_no_decision(self, tiny):
        """One flag a week: S-1 and F-1 share a week, so one is left out."""
        found = scoring.rows(build.features(tiny), tiny,
                             [0.1, 0.8, 0.95, 0.7, 0.6], k=1)

        assert set(found.loc[found['flagged'] == 0, 'decision']) == {''}
        assert set(found.loc[found['flagged'] == 0, 'origin']) == {''}

    def test_a_decided_flag_carries_its_date(self, tiny):
        """A-2 was issued on 20 Jan and found 10 days later. A-1 is
        honest, so it is dismissed after the three-day review."""
        found = scoring.rows(build.features(tiny), tiny,
                             [0.1, 0.8, 0.95, 0.7, 0.6], k=5,
                             as_of=pd.Timestamp('2026-06-30'))

        dates = dict(zip(found['number'], found['decided_on']))
        assert dates['A-2'] == '2026-01-30'
        assert dates['A-1'] == '2026-01-08'
