"""The checks a payables desk writes by hand, as one score.

Each rule is a red flag a clerk already looks for. The score is the
weighted share of flags raised. Every model in this project must beat it
on the same weeks; a model that cannot is not worth its upkeep.
"""

import numpy as np
import pandas as pd

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


def score(table):
    """The weighted share of rules that fire, from 0 to 1."""
    raised = flags(table)
    weights = np.array([weight for _, _, weight in RULES])
    return pd.Series(raised.values @ weights / TOTAL, index=table.index)
