"""How a model is scored: as a payables desk that checks k invoices a week.

Accuracy says nothing when 5% of invoices are anomalies. What a desk
needs is how many of the invoices it has time to check are real
(precision in the top k, Le Borgne and Bontempi), which kinds of error
get through, and how often an honest look-alike wastes a check.

Each function takes a frame with `week`, `score`, `is_anomaly`,
`anomaly` and `lookalike_of`.

The confusion matrix and the measures beside it work on plain arrays of
truth and calls: on one real run there are no weeks to lean on.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

#: Invoices checked first each week: about as many as there are anomalies,
#: so a perfect model scores 1 and every false alarm shows.
K = 5

#: Invoices a desk can check by hand in a week, for recall.
DESK = 25


def recall_at(scored, k=DESK):
    """The share of all anomalies that landed in a week's top k."""
    caught = _top(scored, k)
    total = scored['is_anomaly'].sum()
    return float(caught['is_anomaly'].sum() / total) if total else 0.0


def _top(scored, k):
    return (scored.sort_values('score', ascending=False, kind='stable')
            .groupby('week', group_keys=False).head(k))


def precision_at_k(scored, k=K):
    """Mean over weeks of the share of the top k that are anomalies."""
    top = _top(scored, k)
    return float(top.groupby('week')['is_anomaly'].mean().mean())


def recall_by_kind(scored, k=K, kinds=None):
    """For each anomaly kind, the share that was caught.

    Two callers, one name. The week-by-week path passes a scored frame
    and reads its own columns. The single-run path passes the truth,
    the calls and the kinds as arrays, because one run has no weeks.
    """
    if isinstance(scored, pd.DataFrame):
        caught = _top(scored, k).index
        found = {}
        for kind, group in scored[scored['anomaly'] != ''].groupby('anomaly'):
            found[kind] = float(group.index.isin(caught).mean())
        return found
    truth = np.asarray(scored, dtype=int)
    called = np.asarray(k, dtype=int)
    labels = np.asarray(kinds)
    found = {}
    for kind in pd.unique(labels):
        if kind == '':
            continue
        mask = labels == kind
        real = int(truth[mask].sum())
        found[kind] = (float(((called[mask] == 1) & (truth[mask] == 1)).sum()
                             / real) if real else float('nan'))
    return found


def twin_alarms(scored, k=K):
    """For each kind of honest twin, the share flagged in a top k."""
    caught = _top(scored, k).index
    alarms = {}
    for kind, group in scored[scored['lookalike_of'] != ''].groupby('lookalike_of'):
        alarms[kind] = float(group.index.isin(caught).mean())
    return alarms


def average_precision(scored):
    """The area under the precision-recall curve, over every invoice."""
    return float(average_precision_score(scored['is_anomaly'], scored['score']))


def confusion(truth, called):
    """The four cells: caught, missed, false alarm, left alone."""
    truth = np.asarray(truth, dtype=int)
    called = np.asarray(called, dtype=int)
    return {
        'caught': int(((truth == 1) & (called == 1)).sum()),
        'missed': int(((truth == 1) & (called == 0)).sum()),
        'false_alarm': int(((truth == 0) & (called == 1)).sum()),
        'left_alone': int(((truth == 0) & (called == 0)).sum()),
    }


def precision(truth, called):
    """Of everything called wrong, the share that was wrong.

    Nothing called wrong means no precision to report, not zero: the
    model made no claim to be right about.
    """
    cells = confusion(truth, called)
    claimed = cells['caught'] + cells['false_alarm']
    return float(cells['caught'] / claimed) if claimed else float('nan')


def recall(truth, called):
    """Of every real error, the share that was caught.

    No errors at all means no recall to report, not zero.
    """
    cells = confusion(truth, called)
    real = cells['caught'] + cells['missed']
    return float(cells['caught'] / real) if real else float('nan')


def f1(truth, called):
    """The harmonic mean of precision and recall."""
    p = precision(truth, called)
    r = recall(truth, called)
    if np.isnan(p) or np.isnan(r) or (p + r) == 0:
        return float('nan')
    return float(2 * p * r / (p + r))


def pr_curve(truth, scores):
    """Every threshold walked once, from the top score down.

    Recall climbs as the walk goes; the average precision is the area
    under that climb, weighted by the errors each step adds.
    """
    truth = np.asarray(truth, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind='stable')
    ranked = truth[order]
    caught = np.cumsum(ranked)
    called = np.cumsum(np.ones(len(ranked), dtype=int))
    precision_curve = caught / called
    total = int(truth.sum())
    recall_curve = (caught / total if total
                    else np.full(len(ranked), np.nan))
    return {'precision': precision_curve, 'recall': recall_curve,
            'average_precision': float(
                average_precision_score(truth, scores))}
