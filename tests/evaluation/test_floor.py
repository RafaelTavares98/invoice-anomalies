"""The retrain floor, measured from the training years, not guessed."""

import pandas as pd
import pytest

from evaluation import watch


def test_floor_is_the_tenth_percentile_of_finished_weeks():
    weekly = pd.DataFrame({
        'week': [1, 2, 3, 4, 5],
        'precision': [0.2, None, 0.8, 0.6, 0.4],
    })

    assert watch.floor_from(weekly) == pytest.approx(0.26)


def test_weeks_with_nothing_decided_are_ignored():
    weekly = pd.DataFrame({
        'week': [1, 2, 3],
        'precision': [None, None, None],
    })

    assert watch.floor_from(weekly) is None
