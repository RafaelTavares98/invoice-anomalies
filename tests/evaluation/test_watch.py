"""The desk's precision, week by week, and when it slips."""

import numpy as np
import pandas as pd
import pytest

from evaluation import watch
from features.build import COLUMNS


@pytest.fixture
def scored():
    return pd.DataFrame({
        'week': ['2026-01-05', '2026-01-05', '2026-01-05',
                 '2026-01-12', '2026-01-12'],
        'flagged': [1, 1, 1, 1, 1],
        'decision': ['confirmed', 'dismissed', 'pending', 'pending', ''],
    })


def test_weekly_counts_flags_and_decisions(scored):
    weekly = watch.weekly(scored)

    assert list(weekly.columns) == ['week', 'flagged', 'confirmed',
                                    'decided', 'precision']
    first = weekly[weekly['week'] == '2026-01-05'].iloc[0]
    assert first['flagged'] == 3
    assert first['confirmed'] == 1
    assert first['precision'] == 0.5


def test_a_week_with_every_flag_pending_keeps_its_row(scored):
    weekly = watch.weekly(scored)

    second = weekly[weekly['week'] == '2026-01-12'].iloc[0]
    assert len(weekly) == 2
    assert pd.isna(second['precision'])


def test_slipping_needs_three_straight_bad_weeks():
    table = pd.DataFrame({
        'week': [1, 2, 3, 4, 5, 6],
        'precision': [0.8, 0.4, 0.9, 0.3, 0.3, 0.3],
    })

    assert watch.slipping(table, floor=0.5, run=3)
    assert not watch.slipping(table.iloc[:5], floor=0.5, run=3)


def test_pending_weeks_do_not_break_the_run():
    """A pending week neither counts nor breaks the streak."""
    table = pd.DataFrame({
        'week': [1, 2, 3, 4, 5],
        'precision': [0.2, None, 0.2, 0.2, 0.9],
    })

    assert watch.slipping(table, floor=0.5, run=3)


@pytest.fixture
def random_frames():
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({c: rng.random(1000) for c in COLUMNS})
    return reference, reference.copy()


def test_identical_frames_give_zero_for_every_feature(random_frames):
    reference, recent = random_frames

    found = watch.drift(reference, recent)

    assert set(found) == set(COLUMNS)
    for value in found.values():
        assert value == pytest.approx(0, abs=1e-4)


def test_a_hard_shift_passes_the_line(random_frames):
    reference, recent = random_frames
    recent[COLUMNS[0]] = recent[COLUMNS[0]] + 0.5

    found = watch.drift(reference, recent)

    assert found[COLUMNS[0]] > 0.25


def test_drifted_names_the_worst_first():
    scores = {'a': 0.1, 'b': 0.3, 'c': 0.5}

    assert watch.drifted(scores, line=0.25) == ['c', 'b']
