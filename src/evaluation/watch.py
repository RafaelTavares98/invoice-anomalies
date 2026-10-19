"""The desk's precision, week by week, and when it slips.

The score itself says nothing about how the model behaves in front of a
person. Once an analyst has decided on the flags, each finished week has
a precision: of the flags that were decided, how many were real. That
number, watched week by week, is the earliest honest signal that the
model is slipping.
"""

import numpy as np
import pandas as pd

from features.build import COLUMNS

#: A week that falls below this three times in a row calls for a retrain.
FLOOR = 0.60

#: The desk checks five a week, so three straight weeks below
#: three-in-five is a real fall, not a quiet fortnight.
RUN = 3


def floor_from(weekly_table, quantile=0.10):
    """The tenth percentile of the finished weeks' precision.

    The floor is measured, not guessed: it is where a healthy champion's
    worst tenth of weeks sits. A week with nothing decided is not
    finished and teaches nothing.
    """
    finished = weekly_table['precision'].dropna()
    if not len(finished):
        return None
    return float(finished.quantile(quantile))

#: The verdicts that close a case.
DECIDED = ('confirmed', 'dismissed')


def weekly(scored):
    """One row per week: flags, confirmed, decided, precision.

    Precision is confirmed divided by decided. Pending rows count in
    neither, so a week where nothing is decided yet keeps its row with a
    null precision rather than a fake zero.
    """
    rows = scored.copy()
    rows['confirmed'] = rows['decision'] == 'confirmed'
    rows['decided'] = rows['decision'].isin(DECIDED)
    grouped = rows.groupby('week', sort=False).agg(
        flagged=('flagged', 'sum'),
        confirmed=('confirmed', 'sum'),
        decided=('decided', 'sum'))
    grouped['precision'] = grouped['confirmed'] / grouped['decided'].mask(
        grouped['decided'] == 0)
    return grouped.reset_index()


def slipping(weekly_table, floor=FLOOR, run=RUN):
    """True when `run` consecutive finished weeks sit below the floor.

    One bad week is noise. A week with no decisions yet is not finished
    and neither counts nor breaks the run.
    """
    streak = 0
    for precision in weekly_table['precision'].dropna():
        streak = streak + 1 if precision < floor else 0
        if streak >= run:
            return True
    return False


#: The usual line for "this has moved".
LINE = 0.25


def _psi(reference, recent, bins=10):
    """The population stability index of one feature.

    Bins are the deciles of the reference, so a frame scored against
    itself gives zero exactly, and a shifted frame gives a large number.
    Missing values count as zero in both frames.
    """
    ref = pd.Series(reference).fillna(0).astype(float).to_numpy()
    rec = pd.Series(recent).fillna(0).astype(float).to_numpy()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges = np.concatenate(([-np.inf], edges, [np.inf]))
    share = lambda values: np.histogram(values, bins=edges)[0] / len(values)
    a, b = share(ref), share(rec)
    a = np.where(a == 0, 1e-6, a)
    b = np.where(b == 0, 1e-6, b)
    return float(np.sum((b - a) * np.log(b / a)))


def drift(reference, recent):
    """One population stability index per feature in COLUMNS."""
    return {column: _psi(reference[column], recent[column])
            for column in COLUMNS}


def drifted(scores, line=LINE):
    """The features over the line, worst first."""
    over = [name for name, value in scores.items() if value > line]
    return sorted(over, key=lambda name: -scores[name])
