"""One split, by time, with its dates and its error counts stated.

A model tested on weeks it trained on scores well and means nothing.
This split is the guard, and it reports enough that a reader can see
how little is on the test side: about a dozen errors out of 37.
"""

import pandas as pd
import pytest

from evaluation import splits


def _weeks(n=40, errors=(3, 9, 15, 21, 27, 33, 38)):
    """One invoice a week for n weeks, errors on the weeks named."""
    rows = []
    for week in range(n):
        day = pd.Timestamp('2026-01-05') + pd.Timedelta(weeks=week)
        rows.append(dict(
            number=f'W{week:02d}',
            issue_date=day,
            is_anomaly=int(week in errors),
            found_after_days=20.0 if week in errors else 0.0))
    return pd.DataFrame(rows)


def test_the_split_puts_every_training_row_before_every_test_row():
    table = _weeks()

    train, test = splits.by_time(table, share=0.7)

    assert train['issue_date'].max() < test['issue_date'].min()
    assert len(train) + len(test) == len(table)


def test_the_split_reports_its_dates_and_its_errors():
    table = _weeks()

    report = splits.describe(table, share=0.7)

    assert report['train']['first'] == pd.Timestamp('2026-01-05')
    assert report['train']['errors'] + report['test']['errors'] == 7
    assert report['test']['errors'] >= 1
    assert report['cutoff'] == report['train']['last']


def test_a_split_with_no_errors_on_the_test_side_is_refused():
    """A test side with nothing to find cannot measure anything."""
    table = _weeks(errors=(1, 2, 3))

    with pytest.raises(ValueError, match='no errors'):
        splits.by_time(table, share=0.7)


def test_labels_known_at_the_cutoff_exclude_what_was_not_found_yet():
    table = _weeks(n=10, errors=(2, 8))
    cutoff = pd.Timestamp('2026-01-05') + pd.Timedelta(weeks=8, days=1)

    known = splits.known_labels(table, cutoff)

    # Week 2's error was found 20 days later, well before the cutoff.
    assert known.iloc[2] == 1
    # Week 8's error is one day old at the cutoff; nobody knows yet.
    assert known.iloc[8] == 0


def test_the_cut_date_can_be_given_instead_of_a_share():
    table = _weeks()
    cutoff = pd.Timestamp('2026-06-01')

    train, test = splits.by_time(table, cutoff=cutoff)

    assert train['issue_date'].max() <= cutoff
    assert test['issue_date'].min() > cutoff


def test_a_share_and_a_date_together_are_refused():
    table = _weeks()

    with pytest.raises(ValueError, match='share|cutoff'):
        splits.by_time(table, share=0.7, cutoff=pd.Timestamp('2026-06-01'))
