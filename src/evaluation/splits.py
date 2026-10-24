"""Splitting by time, with labels that arrive late.

Shuffled folds let a model learn from next month to predict last month,
and its score means nothing. Here the past trains and the next week
tests, one week at a time (prequential validation, Le Borgne and
Bontempi; month-grouped folds, IEEE-CIS winner). A label counts only if
its anomaly had been found by the cutoff.
"""

import pandas as pd


def week_of(table):
    """Weeks since the first invoice, from zero."""
    first = table['issue_date'].min()
    return ((table['issue_date'] - first).dt.days // 7).astype(int)


def known_labels(table, cutoff):
    """1 only for anomalies found by the cutoff; the rest read as normal."""
    found = table['issue_date'] + pd.to_timedelta(
        table['found_after_days'].fillna(0), unit='D')
    return ((table['is_anomaly'] == 1) & (found <= cutoff)).astype(int)


def prequential(table, first_week=8, test_weeks=1):
    """(train index, test index) pairs, walking forward a week at a time."""
    weeks = week_of(table)
    for start in range(first_week, weeks.max() + 1, test_weeks):
        train = table.index[weeks < start]
        test = table.index[(weeks >= start) & (weeks < start + test_weeks)]
        if len(test):
            yield train, test


def by_time(table, share=None, cutoff=None):
    """The past trains, the future tests, never the other way round.

    One of `share` and `cutoff` decides where the line falls; both or
    neither is a mistake. A test side with no errors cannot measure
    anything, so it is refused instead of scored.
    """
    if (share is None) == (cutoff is None):
        raise ValueError('give exactly one of share and cutoff')
    dated = table.sort_values('issue_date', kind='stable')
    if cutoff is None:
        if not 0 < share < 1:
            raise ValueError('share must be between 0 and 1')
        position = max(1, min(len(dated) - 1, int(round(len(dated) * share))))
        cut = pd.Timestamp(dated['issue_date'].iloc[position - 1])
    else:
        cut = pd.Timestamp(cutoff)
    train = dated[dated['issue_date'] <= cut]
    test = dated[dated['issue_date'] > cut]
    if not len(test) or not test['is_anomaly'].sum():
        raise ValueError('the test side has no errors')
    return train, test


def describe(table, share=None, cutoff=None):
    """The dates and the error count on each side, and where the line fell."""
    train, test = by_time(table, share=share, cutoff=cutoff)

    def side(part):
        return {'first': part['issue_date'].min(),
                'last': part['issue_date'].max(),
                'errors': int(part['is_anomaly'].sum())}

    return {'cutoff': train['issue_date'].max(),
            'train': side(train), 'test': side(test)}
