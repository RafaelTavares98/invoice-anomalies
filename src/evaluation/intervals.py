"""Every figure carries an interval, because a dozen errors is a dozen.

A precision of 0.58 measured on twelve events is not meaningfully
different from 0.40. This module says so rather than letting the number
be read as more than it is. It decides nothing by itself.
"""

import numpy as np

#: Resamples drawn for one interval. Two thousand settles the edges well
#: enough for a figure whose top is already the sample size.
DRAWS = 2000

#: The two-sided share a figure is reported with.
LEVEL = 0.95


def around(figure, truth, called, seed=0, level=LEVEL, draws=DRAWS):
    """Resample the rows and return the interval around one figure.

    The same seed gives the same interval. A figure that cannot be
    computed on the whole sample has no interval either.
    """
    truth = np.asarray(truth, dtype=int)
    called = np.asarray(called, dtype=int)
    if np.isnan(figure(truth, called)):
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    size = len(truth)
    values = []
    for _ in range(draws):
        picked = rng.integers(0, size, size)
        value = figure(truth[picked], called[picked])
        if not np.isnan(value):
            values.append(value)
    if not values:
        return float('nan'), float('nan')
    low = float(np.quantile(values, (1 - level) / 2))
    high = float(np.quantile(values, 1 - (1 - level) / 2))
    return low, high


def separated(a, b):
    """True only when the two intervals do not overlap at all.

    Touching at a point is not separated: at that point they agree.
    """
    return a[1] < b[0] or b[1] < a[0]


def compare(name_a, a, name_b, b):
    """Both intervals, whether they are apart, and the winner if so."""
    apart = separated(a, b)
    winner = None
    if apart:
        winner = name_a if a[0] > b[0] else name_b
    return {name_a: a, name_b: b, 'separated': apart, 'winner': winner}
