"""How a model is scored: as a payables desk that checks k invoices a week.

Accuracy says nothing when 5% of invoices are anomalies. What a desk
needs is how many of the invoices it has time to check are real
(precision in the top k, Le Borgne and Bontempi), which kinds of error
get through, and how often an honest look-alike wastes a check.

Each function takes a frame with `week`, `score`, `is_anomaly`,
`anomaly` and `lookalike_of`.
"""

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


def recall_by_kind(scored, k=K):
    """For each anomaly kind, the share that landed in a week's top k."""
    caught = _top(scored, k).index
    found = {}
    for kind, group in scored[scored['anomaly'] != ''].groupby('anomaly'):
        found[kind] = float(group.index.isin(caught).mean())
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
