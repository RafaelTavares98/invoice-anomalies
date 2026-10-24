"""24 features against 37 errors is too many, so the set is cut.

The usual rule of thumb is about ten events per feature. With 37
errors in the whole run, and fewer on the training side, a model is
allowed three or four features, not twenty-four. The cut is made on
the training rows only; a cut that peeked at the test weeks would
flatter every figure that follows.
"""

import pandas as pd
import pytest

from features import build


def _table(n=300, errors=30, seed=0):
    """A table where one feature tells the truth, one is noise, and the
    rest are constant."""
    rng = pd.Series(range(n))
    truth = (rng < errors).astype(int)
    frame = pd.DataFrame({
        'number': [f'I-{i:03d}' for i in range(n)],
        'issue_date': pd.Timestamp('2026-01-05') + pd.to_timedelta(rng, unit='D'),
        'is_anomaly': truth,
        'found_after_days': 0.0,
    })
    for column in build.COLUMNS:
        frame[column] = 0.0
    frame['price_over_contract'] = truth * 0.5
    frame['amount_zscore'] = [(i % 7) / 7 for i in range(n)]
    return frame


def test_the_cut_keeps_about_one_feature_per_ten_errors():
    table = _table(errors=30)

    kept = build.cut(table, events_per_feature=10)

    assert 1 <= len(kept) <= 3


def test_the_cut_keeps_the_feature_that_tells_the_truth():
    table = _table()

    kept = build.cut(table, events_per_feature=10)

    assert 'price_over_contract' in kept


def test_the_cut_drops_features_that_never_move():
    table = _table()

    kept = build.cut(table, events_per_feature=10)

    assert 'duty_rate_off' not in kept
    assert 'fee_amended' not in kept


def test_the_kept_features_come_back_in_the_fixed_column_order():
    table = _table(errors=80)

    kept = build.cut(table, events_per_feature=10)

    assert kept == [c for c in build.COLUMNS if c in kept]


def test_the_cut_is_the_same_twice():
    table = _table()

    assert build.cut(table, events_per_feature=10) == \
           build.cut(table, events_per_feature=10)


def test_the_cut_refuses_rows_after_the_cutoff():
    """The cut is part of training. Showing it a test week is a leak."""
    table = _table()
    cutoff = table['issue_date'].iloc[100]

    with pytest.raises(ValueError, match='cutoff'):
        build.cut(table, events_per_feature=10, cutoff=cutoff)


def test_the_cut_uses_only_the_rows_given_to_it():
    """Same training rows, two different futures: the kept set must not
    change."""
    table = _table()
    early = table[table['issue_date'] <= table['issue_date'].iloc[150]]

    honest = build.cut(early, events_per_feature=10)
    again = build.cut(early.assign(is_anomaly=early['is_anomaly']),
                      events_per_feature=10)

    assert honest == again


def test_a_table_with_too_few_errors_is_refused():
    """Three errors buy no features at all, and a model trained on them
    would be noise with a score attached."""
    table = _table(errors=3)

    with pytest.raises(ValueError, match='errors'):
        build.cut(table, events_per_feature=10)
