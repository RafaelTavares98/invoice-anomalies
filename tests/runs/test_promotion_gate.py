"""The gate that decides a promotion, on a business written by hand.

The story tests beside this one need the local MLflow store and skip
without it, so on a fresh clone they prove nothing. These do not: the
corpus, the serving model and the challenger are all built here, and
the only thing under test is the loop's own rule. A challenger takes
over when it beats the serving model on the weeks since it was trained,
and a tie is not a win.
"""

import pandas as pd
import pytest

from data.load import Dataset
from features.build import COLUMNS
from runs import loop as loop_mod

#: Rows a week, and which of them are real errors.
PER_WEEK = 10
WRONG = (0, 1, 2, 3)
WEEKS = 12

#: The column each stub model reads its ranking from. Both are real
#: feature names, so the matrix the loop hands the model is untouched.
SERVED_FROM = 'amount_zscore'
OFFERED_FROM = 'days_since_last'

#: Rankings, best first. The serving model catches two of the four
#: errors; the orders below are what each challenger would catch.
SERVED_ORDER = (0, 1, 6, 7, 8)
BETTER_ORDER = (0, 1, 2, 3, 4)
EQUAL_ORDER = (0, 1, 7, 8, 9)
WORSE_ORDER = (5, 6, 7, 8, 9)


class Ranker:
    """Scores a week by one column, highest first."""

    def __init__(self, column):
        self.column = column

    def predict_proba(self, matrix):
        scores = matrix[self.column].to_numpy(dtype=float)
        return pd.DataFrame({0: 1 - scores, 1: scores}).to_numpy()


def _ranked(order, slots=PER_WEEK):
    """A score per slot: the slots named in `order` rank highest, in
    order, and every other slot sits below them."""
    scores = [0.0] * slots
    for place, slot in enumerate(order):
        scores[slot] = 1.0 - place * 0.01
    return scores


def _corpus(offered_order):
    """Twelve weeks of one vendor, four errors a week.

    Each row carries two rankings: the one the serving model reads and
    the one the challenger reads. Everything else is zero, so nothing
    but the ranking can move a result.
    """
    served = _ranked(SERVED_ORDER)
    offered = _ranked(offered_order)
    rows, labels, lines = [], [], []
    for week in range(WEEKS):
        monday = pd.Timestamp('2026-01-05') + pd.Timedelta(weeks=week)
        for slot in range(PER_WEEK):
            number = f'W{week:02d}-{slot}'
            wrong = slot in WRONG
            rows.append(dict(number=number, issuer='ACM', kind='goods',
                             issue_date=str(monday.date()),
                             due_date=str((monday + pd.Timedelta(days=30)).date()),
                             grand_total='1000.00', order_number='PO-1',
                             shipment_id='S-1', pay_to='', sender='',
                             revises=''))
            labels.append(dict(number=number,
                               anomaly='price_drift' if wrong else '',
                               lookalike_of='', found_after_days='0'))
            lines.append(dict(number=number, position=1, code='P1',
                              description='Cap', quantity='10',
                              unit_price='100.00', net='1000.00'))

    invoices = pd.DataFrame(rows)
    erp = {'vendors': pd.DataFrame([dict(code='ACM', onboarded_on='2019-01-01',
                                         bank_account='ACME BANK 1',
                                         email='accounts@acm.example')])}
    data = Dataset.from_frames('gate', invoices, pd.DataFrame(lines), erp,
                               pd.DataFrame(labels))

    table = pd.DataFrame({'number': invoices['number']})
    table['issue_date'] = pd.to_datetime(invoices['issue_date'])
    for column in COLUMNS:
        table[column] = 0.0
    slots = invoices['number'].str.slice(-1).astype(int)
    table[SERVED_FROM] = [served[slot] for slot in slots]
    table[OFFERED_FROM] = [offered[slot] for slot in slots]
    table['week'] = invoices['number'].str.slice(1, 3).astype(int)
    table['is_anomaly'] = [int(slot in WRONG) for slot in slots]
    table['anomaly'] = ['price_drift' if slot in WRONG else '' for slot in slots]
    table['lookalike_of'] = ''
    table['found_after_days'] = 0.0
    return table, data


def _walk(monkeypatch, offered_order):
    """The loop over the corpus, with the challenger handed to it.

    The retraining is replaced so the challenger is known, not drawn
    from LightGBM. Everything the gate does is still the loop's own
    code. The floor is 1.0 so the desk always counts as slipping, which
    is the only way to reach the gate in a test this small.
    """
    table, data = _corpus(offered_order)
    challenger = Ranker(OFFERED_FROM)
    monkeypatch.setattr(loop_mod.lifecycle, 'retrain',
                        lambda seen, as_of, labels, seed=0: (challenger, seen))
    registered = []
    events, _, _, _ = loop_mod.run(
        table, data, None, Ranker(SERVED_FROM),
        promote=lambda model, verdict: registered.append(verdict),
        floor=1.0)
    return events, registered


def test_a_better_challenger_takes_over(monkeypatch):
    events, registered = _walk(monkeypatch, BETTER_ORDER)

    promoted = [event for event in events if event['promoted']]
    assert len(promoted) == 1
    assert promoted[0]['challenger_precision'] > promoted[0]['verdict_served']
    assert len(registered) == 1


def test_a_tie_is_not_a_win(monkeypatch):
    events, registered = _walk(monkeypatch, EQUAL_ORDER)

    assert not any(event['promoted'] for event in events)
    assert registered == []


def test_a_worse_challenger_is_refused(monkeypatch):
    events, registered = _walk(monkeypatch, WORSE_ORDER)

    assert not any(event['promoted'] for event in events)
    assert registered == []


def test_the_challenger_is_judged_on_four_weeks(monkeypatch):
    """The verdict lands four weeks after the retraining, not before."""
    events, _ = _walk(monkeypatch, BETTER_ORDER)

    retrained = next(event['week'] for event in events if event['retrained'])
    promoted = next(event['week'] for event in events if event['promoted'])

    assert promoted - retrained == loop_mod.CHALLENGE_WEEKS


def test_no_new_challenger_while_one_is_waiting(monkeypatch):
    """One retraining at a time: the desk judges before it trains again."""
    events, _ = _walk(monkeypatch, WORSE_ORDER)

    weeks = [event['week'] for event in events if event['retrained']]
    gaps = [b - a for a, b in zip(weeks, weeks[1:])]

    assert all(gap > loop_mod.CHALLENGE_WEEKS for gap in gaps), weeks


def test_the_verdict_reports_both_numbers(monkeypatch):
    """What is registered says what beat what."""
    _, registered = _walk(monkeypatch, BETTER_ORDER)

    assert set(registered[0]) == {'challenger_precision', 'serving_precision'}
    assert registered[0]['challenger_precision'] > registered[0]['serving_precision']


@pytest.mark.parametrize('order,expected', [
    (BETTER_ORDER, 0.8),
    (SERVED_ORDER, 0.4),
    (WORSE_ORDER, 0.0),
])
def test_the_corpus_scores_what_the_test_claims(order, expected):
    """The fixture itself, checked by eye: of the top five, how many are
    real errors. If this drifts, every test above is measuring something
    other than what it says."""
    caught = sum(1 for slot in order if slot in WRONG)

    assert caught / len(order) == pytest.approx(expected)
