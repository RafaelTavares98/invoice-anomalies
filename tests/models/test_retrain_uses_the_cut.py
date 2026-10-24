"""A retraining obeys the same feature cut as the first training.

35 errors buy about three features. `prepare` cuts the set on that
rule, and `train` honours it. A retraining inside `watch` that quietly
fits all 24 undoes the whole argument: the model on duty would be the
overfitted one, and every figure beside it would be measuring
something else.
"""

import pandas as pd
import pytest

from features import build
from models import lifecycle


def _table(n=300, errors=30):
    rows = pd.Series(range(n))
    truth = (rows < errors).astype(int)
    frame = pd.DataFrame({
        'number': [f'I-{i:03d}' for i in range(n)],
        'issue_date': pd.Timestamp('2026-01-05') + pd.to_timedelta(rows, unit='D'),
        'is_anomaly': truth,
        'found_after_days': 0.0,
    })
    for column in build.COLUMNS:
        frame[column] = 0.0
    frame['price_over_contract'] = truth * 0.5
    frame['amount_zscore'] = [(i % 7) / 7 for i in range(n)]
    return frame


def _labels(table):
    return table['is_anomaly'].astype(int)


def test_retrain_fits_only_the_features_it_is_given():
    table = _table()
    kept = ['price_over_contract', 'amount_zscore']

    model, _ = lifecycle.retrain(table, table['issue_date'].max(),
                                 _labels(table), features=kept)

    assert list(model.feature_name_) == kept


def test_retrain_without_a_cut_is_refused():
    """Passing no feature list is how all 24 crept back in. It is now an
    error, not a default."""
    table = _table()

    with pytest.raises(ValueError, match='features'):
        lifecycle.retrain(table, table['issue_date'].max(), _labels(table))


def test_a_cut_naming_a_column_the_table_lacks_is_refused():
    table = _table().drop(columns=['duty_rate_off'])

    with pytest.raises(KeyError, match='duty_rate_off'):
        lifecycle.retrain(table, table['issue_date'].max(), _labels(table),
                          features=['price_over_contract', 'duty_rate_off'])


def test_the_challenger_is_scored_on_the_same_features():
    table = _table()
    kept = ['price_over_contract', 'amount_zscore']
    model, _ = lifecycle.retrain(table, table['issue_date'].max(),
                                 _labels(table), features=kept)

    found = lifecycle.challenge(model, table, features=kept)

    assert set(found) >= {'precision', 'recall'}


def test_scoring_a_challenger_on_other_features_is_refused():
    """A model fitted on three columns cannot read twenty-four. Saying
    so out loud is cheaper than a silent wrong number."""
    table = _table()
    model, _ = lifecycle.retrain(table, table['issue_date'].max(),
                                 _labels(table),
                                 features=['price_over_contract'])

    with pytest.raises(ValueError, match='features'):
        lifecycle.challenge(model, table,
                            features=['price_over_contract', 'amount_zscore'])


def test_the_cut_travels_with_the_model_through_the_week_walk():
    """What `watch` records each week has to name the features the
    serving model actually reads, so a reader can tell which model
    produced a score."""
    table = _table()
    kept = ['price_over_contract']
    model, rows = lifecycle.retrain(table, table['issue_date'].max(),
                                    _labels(table), features=kept)

    assert lifecycle.features_of(model) == kept
