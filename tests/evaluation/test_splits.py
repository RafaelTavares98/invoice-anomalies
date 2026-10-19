"""Splitting by time, with labels that arrive late.

A model trained on a Monday knows only the anomalies found by then. An
error from last week is still labelled normal; that is the noise a real
desk lives with (Le Borgne and Bontempi, prequential validation).
"""

import pandas as pd

from evaluation import splits


def table():
    days = pd.to_datetime(['2026-01-05', '2026-01-12', '2026-01-19',
                           '2026-01-26', '2026-02-02', '2026-02-09'])
    return pd.DataFrame({
        'issue_date': days,
        'is_anomaly': [1, 0, 1, 0, 1, 0],
        'found_after_days': [5.0, None, 60.0, None, 3.0, None],
    })


def test_a_label_found_after_the_cutoff_reads_as_normal():
    """Jan 5 is found Jan 10, in time. Jan 19 waits 60 days, and Feb 2 is
    found Feb 5: both after a Feb 1 cutoff."""
    known = splits.known_labels(table(), cutoff=pd.Timestamp('2026-02-01'))

    assert list(known) == [1, 0, 0, 0, 0, 0]


def test_folds_never_test_on_the_past():
    for train, test in splits.prequential(table(), first_week=2, test_weeks=1):
        assert table().loc[train, 'issue_date'].max() < table().loc[test, 'issue_date'].min()


def test_folds_walk_forward_one_week_at_a_time():
    folds = list(splits.prequential(table(), first_week=2, test_weeks=1))

    assert len(folds) == 4
    assert [len(train) for train, _ in folds] == [2, 3, 4, 5]


def test_week_numbers_start_at_zero():
    assert list(splits.week_of(table())) == [0, 1, 2, 3, 4, 5]
