"""What the reader's mistakes cost the model.

Every figure this project has reported so far came off the simulation's
own numbers. Those are always right: the total on the page is the total
the generator chose. A payables desk is handed what somebody read off a
page instead, and the reader gets roughly one invoice in ten wrong.

So the same model is fitted and scored twice, once on each table, and
the two average precisions are reported side by side. The difference is
what reading costs, and it is the only honest version of the figure.

Nothing here looks ahead: each test week is scored by a model fitted on
the weeks before it, exactly as `splits.prequential` orders.
"""

import numpy as np
import pandas as pd

from evaluation import metrics, splits


class NotEnoughWeeks(Exception):
    """Fewer weeks than the warm-up, so there is no week left to score."""


def _default_fit_score(train, test):
    """The champion model, fitted on the past and scored on the week."""
    from models.train import Boosting, with_known_labels
    return Boosting().fit_score(with_known_labels(train), test)


def walk(table, fit_score=None, first_week=8):
    """Every test week scored by a model that never saw it.

    Returns the whole scored table, so one average precision covers the
    year rather than one week of it.
    """
    fit_score = fit_score or _default_fit_score
    pieces = []
    for train, test in splits.prequential(table, first_week=first_week):
        if not len(train):
            continue
        rows = table.loc[test]
        pieces.append(rows.assign(score=fit_score(table.loc[train], rows)))
    if not pieces:
        raise NotEnoughWeeks(
            f'no week after the first {first_week}; nothing to score')
    return pd.concat(pieces)


def dated(table):
    """The rows that can be placed in time.

    The reader misses a date now and then. Every feature is measured on
    the review day and the split walks forward a week at a time, so a
    row with no date cannot be used at all. Dropping it is honest;
    guessing one would not be.
    """
    return table[table['issue_date'].notna()]


def run_year(table):
    """The year most of the invoice numbers carry.

    An invoice number looks like NCL-2026-3001, and it was read off the
    same page as the date. Nothing here is told which year the run was:
    it is whichever year the numbers agree on.
    """
    years = (table['number'].astype(str).str.extract(r'-((?:19|20)\d{2})-')[0]
             .dropna().astype(int))
    return int(years.mode().iloc[0]) if len(years) else None


def plausible(table, year=None):
    """The rows whose date agrees with the year on their own number.

    One date read as 2612 took the table from eight weeks to two and a
    half thousand, almost all empty, and nothing could be scored. The
    page carries its own year twice; when the two disagree, the date is
    the one that was read wrong.
    """
    year = year if year is not None else run_year(table)
    if year is None:
        return table
    dated_rows = table['issue_date'].notna()
    sane = (~dated_rows) | table['issue_date'].dt.year.between(year - 1,
                                                               year + 1)
    return table[sane]


def _average_precision(table, fit_score, first_week):
    scored = walk(table, fit_score, first_week)
    scored = scored.assign(week=splits.week_of(scored))
    if scored['is_anomaly'].nunique() < 2:
        return 0.0
    return metrics.average_precision(scored)


def beside(perfect, read, fit_score=None, first_week=8):
    """The same model on both tables, and what the difference costs.

    `perfect` is built from the simulation's figures, `read` from what
    the reader made of the pages. Both are scored the same way, and the
    invoices the reader left undated are counted as part of the cost.
    """
    usable_perfect, usable_read = dated(perfect), dated(read)
    clean_perfect = plausible(usable_perfect)
    clean_read = plausible(usable_read)

    # A smaller table scores worse whatever the reader did, so both
    # sides are cut to the invoices they share. Without this the figure
    # measures the size of the sample, not the quality of the reading.
    shared = (set(clean_perfect['number'].astype(str))
              & set(clean_read['number'].astype(str)))
    both_perfect = clean_perfect[clean_perfect['number'].astype(str)
                                 .isin(shared)]
    both_read = clean_read[clean_read['number'].astype(str).isin(shared)]

    on_perfect = _average_precision(both_perfect, fit_score, first_week)
    on_read = _average_precision(both_read, fit_score, first_week)
    return {
        'perfect': on_perfect,
        'read': on_read,
        'cost': on_perfect - on_read,
        'invoices_perfect': len(both_perfect),
        'invoices_read': len(both_read),
        'undated_perfect': len(perfect) - len(usable_perfect),
        'undated_read': len(read) - len(usable_read),
        'misdated_perfect': len(usable_perfect) - len(clean_perfect),
        'misdated_read': len(usable_read) - len(clean_read),
        'only_perfect': len(clean_perfect) - len(both_perfect),
        'only_read': len(clean_read) - len(both_read),
    }
