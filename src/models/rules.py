"""The checks a payables desk writes by hand, as one score.

Each rule is a red flag a clerk already looks for. The score is the
weighted share of flags raised. Every model in this project must beat it
on the same weeks; a model that cannot is not worth its upkeep.

The static weights below are the desk's own guess. `learn_weights`
replaces the guess with what the training weeks measure: a rule earns
its weight by how much it raises the odds of a real error, not by how
sad a miss would be. A rare rule that is always right outranks a common
one that is almost always wrong.
"""

import numpy as np
import pandas as pd

from evaluation import splits

#: Flag name, test, weight. Heavier weights are the costlier misses.
RULES = (
    ('account not on file', lambda t: t['bank_not_on_file'] == 1, 3),
    ('sender not on file', lambda t: t['sender_not_on_file'] == 1, 2),
    ('no order behind a goods bill', lambda t: (t['has_order'] == 0) & (t['is_freight'] == 0)
     & (t['is_customs'] == 0), 2),
    ('price above contract', lambda t: t['price_over_contract'] > 0.02, 2),
    ('quantity above receipt', lambda t: t['quantity_over_receipt'].fillna(0) > 0.02, 2),
    ('charge not on the rate card', lambda t: t['unknown_charges'] > 0, 2),
    ('same amount again', lambda t: t['same_amount_30d'] == 1, 1),
    ('order split under the limit', lambda t: t['under_limit_but_order_over'] == 1, 1),
    ('round total, not a known fee', lambda t: (t['round_thousands'] == 1) & (t['known_fixed_fee'] == 0), 1),
    ('young vendor', lambda t: t['vendor_age_days'] < 90, 1),
    ('duty at a rate not in force', lambda t: t['duty_rate_off'] > 0.001, 2),
    ('broker fee above contract', lambda t: (t['fee_over_contract'] > 0.02)
     & (t['fee_amended'] == 0), 2),
)

TOTAL = sum(weight for _, _, weight in RULES)


def flags(table):
    """One column per rule, 1 where it fires."""
    return pd.DataFrame({name: test(table).astype(int)
                         for name, test, _ in RULES}, index=table.index)


def learn_weights(table, cutoff):
    """Each rule's weight, from how much it raises the odds.

    The odds come from the rows passed in, and the rows must all be
    dated on or before the cutoff: a week the desk has not decided yet
    teaches nothing. The label column wins when the caller already
    computed it; otherwise the labels are the ones known by the cutoff.
    A rule that never fires has nothing to say and weighs nothing.
    """
    cutoff = pd.Timestamp(cutoff)
    dates = pd.to_datetime(table['issue_date'])
    if len(dates) and dates.max() > cutoff:
        raise ValueError(f'a row after the cutoff {cutoff.date()} was passed in')
    if 'label' in table:
        labels = table['label'].astype(int)
    else:
        labels = splits.known_labels(table, cutoff).astype(int)
    base = float(labels.mean()) if len(labels) else 0.0
    weights = {}
    for name, test, _ in RULES:
        fired = test(table).astype(bool)
        fires = int(fired.sum())
        if not fires or base <= 0:
            weights[name] = 0.0
            continue
        hits = int(labels[fired].sum())
        precision = (hits + 1) / (fires + 2)
        weights[name] = precision / base
    return weights


def score(table, weights=None):
    """The weighted share of rules that fire, from 0 to 1."""
    raised = flags(table)
    if weights is None:
        values = np.array([weight for _, _, weight in RULES], dtype=float)
    else:
        values = np.array([weights[name] for name, _, _ in RULES],
                          dtype=float)
    total = values.sum()
    if total <= 0:
        return pd.Series(0.0, index=table.index)
    return pd.Series(raised.values @ values / total, index=table.index)
