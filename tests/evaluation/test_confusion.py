"""The ordinary measures, worked out by hand first.

Confusion matrix, precision, recall, F1. Nothing clever: these are the
figures anybody reviewing the project will look for, and they are the
ones the panels show.
"""

import numpy as np
import pytest

from evaluation import metrics

#: 10 invoices. 4 are real errors. The model calls 5 of them wrong and
#: gets 3 of those right, so: caught 3, missed 1, false alarm 2,
#: left alone 4.
TRUTH = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
CALLED = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])


def test_the_confusion_matrix_counts_four_cells():
    found = metrics.confusion(TRUTH, CALLED)

    assert found['caught'] == 3
    assert found['missed'] == 1
    assert found['false_alarm'] == 2
    assert found['left_alone'] == 4
    assert sum(found.values()) == len(TRUTH)


def test_precision_is_caught_over_everything_called_wrong():
    assert metrics.precision(TRUTH, CALLED) == pytest.approx(3 / 5)


def test_recall_is_caught_over_every_real_error():
    assert metrics.recall(TRUTH, CALLED) == pytest.approx(3 / 4)


def test_f1_is_the_harmonic_mean():
    expected = 2 * (0.6 * 0.75) / (0.6 + 0.75)

    assert metrics.f1(TRUTH, CALLED) == pytest.approx(expected)


def test_calling_nothing_wrong_gives_no_precision_rather_than_a_crash():
    nothing = np.zeros(10, dtype=int)

    assert np.isnan(metrics.precision(TRUTH, nothing))
    assert metrics.recall(TRUTH, nothing) == 0.0


def test_a_run_with_no_errors_at_all_gives_no_recall():
    clean = np.zeros(10, dtype=int)

    assert np.isnan(metrics.recall(clean, CALLED))


def test_recall_by_kind_names_the_kinds_that_are_never_found():
    kinds = np.array(['dup', 'dup', 'freight', 'customs',
                      '', '', '', '', '', ''])

    found = metrics.recall_by_kind(TRUTH, CALLED, kinds)

    assert found['dup'] == pytest.approx(1.0)
    assert found['freight'] == pytest.approx(1.0)
    assert found['customs'] == pytest.approx(0.0)
    assert '' not in found


def test_the_curve_walks_every_threshold():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.75, 0.6, 0.1, 0.05, 0.0, 0.0])

    curve = metrics.pr_curve(TRUTH, scores)

    assert len(curve['precision']) == len(curve['recall'])
    assert curve['recall'][0] <= curve['recall'][-1]
    assert 0.0 <= curve['average_precision'] <= 1.0


def test_average_precision_is_one_when_every_error_outranks_every_clean_row():
    scores = np.where(TRUTH == 1, 0.9, 0.1)

    curve = metrics.pr_curve(TRUTH, scores)

    assert curve['average_precision'] == pytest.approx(1.0)
