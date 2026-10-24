"""Every figure carries an interval, because a dozen errors is a dozen.

A time split over this run leaves about twelve errors to be measured
on. A precision of 0.58 computed on twelve events is not meaningfully
different from 0.40. These tests pin the interval, and pin the rule
that two overlapping intervals are reported as not separated.
"""

import numpy as np
import pytest

from evaluation import intervals, metrics


def _truth_and_scores(n=1000, errors=40, seed=0):
    """A table where the score and the truth agree most of the time."""
    rng = np.random.default_rng(seed)
    truth = np.zeros(n, dtype=int)
    truth[:errors] = 1
    scores = np.where(truth == 1,
                      rng.normal(0.7, 0.2, n),
                      rng.normal(0.3, 0.2, n)).clip(0, 1)
    return truth, scores


def test_an_interval_contains_the_figure_it_surrounds():
    truth, scores = _truth_and_scores()
    called = (scores > 0.5).astype(int)
    point = metrics.precision(truth, called)

    low, high = intervals.around(metrics.precision, truth, called, seed=0)

    assert low <= point <= high
    assert 0.0 <= low < high <= 1.0


def test_fewer_events_give_a_wider_interval():
    """The whole reason this module exists."""
    many_truth, many_scores = _truth_and_scores(n=1000, errors=200)
    few_truth, few_scores = _truth_and_scores(n=1000, errors=12)

    wide = intervals.around(metrics.recall, few_truth,
                            (few_scores > 0.5).astype(int), seed=0)
    narrow = intervals.around(metrics.recall, many_truth,
                              (many_scores > 0.5).astype(int), seed=0)

    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_the_same_seed_gives_the_same_interval():
    truth, scores = _truth_and_scores()
    called = (scores > 0.5).astype(int)

    first = intervals.around(metrics.precision, truth, called, seed=7)
    again = intervals.around(metrics.precision, truth, called, seed=7)

    assert first == again


def test_overlapping_intervals_are_not_separated():
    assert intervals.separated((0.40, 0.70), (0.55, 0.85)) is False
    assert intervals.separated((0.55, 0.85), (0.40, 0.70)) is False


def test_intervals_that_do_not_touch_are_separated():
    assert intervals.separated((0.10, 0.30), (0.45, 0.80)) is True


def test_intervals_that_meet_at_a_point_are_not_separated():
    assert intervals.separated((0.10, 0.50), (0.50, 0.90)) is False


def test_a_comparison_says_which_is_higher_only_when_they_are_separated():
    clear = intervals.compare('model', (0.60, 0.80), 'rules', (0.20, 0.40))
    muddy = intervals.compare('model', (0.40, 0.80), 'rules', (0.35, 0.60))

    assert clear['winner'] == 'model'
    assert clear['separated'] is True
    assert muddy['winner'] is None
    assert muddy['separated'] is False


def test_the_interval_width_is_reported_in_the_comparison():
    found = intervals.compare('model', (0.60, 0.80), 'rules', (0.20, 0.40))

    assert found['model'] == (0.60, 0.80)
    assert found['rules'] == (0.20, 0.40)


def test_a_figure_that_cannot_be_computed_comes_back_empty():
    """No errors at all in a resample: there is no recall to report, and
    the interval must say so rather than invent a zero."""
    truth = np.zeros(50, dtype=int)
    called = np.zeros(50, dtype=int)

    low, high = intervals.around(metrics.recall, truth, called, seed=0)

    assert np.isnan(low) and np.isnan(high)
