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
