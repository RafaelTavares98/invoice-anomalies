"""Rule weights learnt from the training weeks, not guessed from cost.

A rule earns its weight by how much it raises the odds of a real error.
A rare rule that is always right must outrank a common one that is
almost always wrong, whatever a miss would cost.
"""

import numpy as np
import pandas as pd
import pytest

from models import rules, scoring

NAMES = [name for name, _, _ in rules.RULES]


def _blank(n=1000):
    """Every rule silent, thirty days old vendors, dates inside January."""
    return pd.DataFrame({
        'issue_date': pd.Timestamp('2026-01-05'),
        'is_anomaly': 0,
        'bank_not_on_file': 0,
        'sender_not_on_file': 0,
        'has_order': 1,
        'is_freight': 0,
        'is_customs': 0,
        'price_over_contract': 0.0,
        'quantity_over_receipt': 0.0,
        'unknown_charges': 0,
        'same_amount_30d': 0,
        'under_limit_but_order_over': 0,
        'round_thousands': 0,
        'known_fixed_fee': 1,
        'vendor_age_days': 365,
        'duty_rate_off': 0.0,
        'fee_over_contract': 0.0,
        'fee_amended': 0,
        'found_after_days': 0.0,
    }, index=range(n))


def _rare_and_common():
    """A rule right every time on 1% of rows, a rule right 2% of the
    time on 20% of rows, the rest silent."""
    table = _blank()
    table.loc[0:9, 'sender_not_on_file'] = 1
    table.loc[0:9, 'is_anomaly'] = 1
    table.loc[10:209, 'under_limit_but_order_over'] = 1
    table.loc[10:13, 'is_anomaly'] = 1
    return table


def test_a_rare_and_right_rule_outranks_a_common_and_wrong_one():
    table = _rare_and_common()

    weights = rules.learn_weights(table, '2026-01-31')

    assert weights['sender not on file'] > weights['order split under the limit']


def test_weights_are_fitted_on_the_rows_passed_in():
    table = _rare_and_common()
    honest = table.drop(columns='is_anomaly')
    honest['label'] = 0

    right = rules.learn_weights(table.assign(label=table['is_anomaly']),
                                '2026-01-31')
    wrong = rules.learn_weights(honest, '2026-01-31')

    assert right['sender not on file'] != wrong['sender not on file']


def test_a_row_after_the_cutoff_is_refused():
    table = _rare_and_common()
    table.loc[999, 'issue_date'] = pd.Timestamp('2026-03-01')

    with pytest.raises(ValueError, match='cutoff'):
        rules.learn_weights(table, '2026-01-31')


def test_the_twelve_rule_names_and_their_order_are_unchanged():
    assert NAMES == [
        'account not on file',
        'sender not on file',
        'no order behind a goods bill',
        'price above contract',
        'quantity above receipt',
        'charge not on the rate card',
        'same amount again',
        'order split under the limit',
        'round total, not a known fee',
        'young vendor',
        'duty at a rate not in force',
        'broker fee above contract',
    ]
    assert set(scoring.STAGE) == set(NAMES)


def test_equally_right_rules_keep_the_equal_weight_score():
    """Every rule fires together and is right: the learnt score is the
    plain share of rules that fired, to four decimal places."""
    table = _blank(120)
    rows = pd.Index(range(12))
    table.loc[rows, 'bank_not_on_file'] = 1
    table.loc[rows, 'sender_not_on_file'] = 1
    table.loc[rows, 'has_order'] = 0
    table.loc[rows, 'price_over_contract'] = 0.5
    table.loc[rows, 'quantity_over_receipt'] = 0.5
    table.loc[rows, 'unknown_charges'] = 1
    table.loc[rows, 'same_amount_30d'] = 1
    table.loc[rows, 'under_limit_but_order_over'] = 1
    table.loc[rows, 'round_thousands'] = 1
    table.loc[rows, 'known_fixed_fee'] = 0
    table.loc[rows, 'vendor_age_days'] = 30
    table.loc[rows, 'duty_rate_off'] = 1.0
    table.loc[rows, 'fee_over_contract'] = 1.0
    table.loc[rows, 'is_anomaly'] = 1

    raised = rules.flags(table)
    weights = rules.learn_weights(table, '2026-01-31')
    learnt = rules.score(table, weights)

    for value in weights.values():
        assert value == pytest.approx(weights[NAMES[0]], abs=1e-9)
    assert np.allclose(learnt, raised.mean(axis=1), atol=1e-4)
