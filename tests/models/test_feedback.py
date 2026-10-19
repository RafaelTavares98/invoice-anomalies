"""The analyst's verdict becomes the label the next training run learns."""

import pandas as pd
import pytest

from evaluation import splits
from models import feedback


@pytest.fixture
def decided():
    return pd.DataFrame({
        'number': ['A-1', 'A-2', 'A-3', 'A-4', 'A-5'],
        'flagged': [1, 1, 1, 0, 0],
        'decision': ['confirmed', 'dismissed', 'pending', '', ''],
        'decided_on': ['2026-03-01', '2026-03-02', '', '', ''],
        'issue_date': ['2026-01-05', '2026-01-20', '2026-02-01',
                       '2026-02-10', '2026-02-11'],
        'is_anomaly': [1, 0, 1, 0, 0],
        'found_after_days': [10, 0, 30, 0, 0],
    })


def test_verdicts_become_labels(decided):
    labels = feedback.labels_from(decided, '2026-03-10')

    assert labels.loc[0] == 1          # confirmed before the cutoff
    assert labels.loc[1] == 0          # dismissed before the cutoff
    assert 2 not in labels.index       # pending is dropped, not labelled
    assert labels.loc[3] == 0          # never flagged: the normal case
    assert labels.loc[4] == 0


def test_a_verdict_after_the_cutoff_is_refused(decided):
    with pytest.raises(ValueError, match='A-1'):
        feedback.labels_from(decided, '2026-02-15')


def test_no_verdicts_degrades_to_known_labels(decided):
    blank = decided.assign(decision='', decided_on='')
    labels = feedback.labels_from(blank, '2026-03-10')

    expected = splits.known_labels(
        blank.assign(issue_date=pd.to_datetime(blank['issue_date'])),
        pd.Timestamp('2026-03-10'))
    assert labels.equals(expected)
