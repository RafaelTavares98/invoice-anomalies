"""Rule weights are earned on the training weeks, and nowhere else.

The weights decide the ranking a desk sees, so a weight that learnt
anything from the test weeks would flatter every number this project
publishes. These tests pin the arithmetic to a hand-worked figure and
then prove the test weeks cannot move it.
"""

import pandas as pd
import pytest

from models import rules, train

CUTOFF = '2026-01-31'


def _blank(n):
    """Every rule silent, every date inside January."""
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


def test_a_weight_is_the_lift_worked_out_by_hand():
    """One rule, fired on ten rows of a hundred, right on eight of them.

    Ten in a hundred are errors, so the base is 0.10. The rule's own
    precision is smoothed, because a rule that fires three times and is
    right three times is not proof of anything: (8 + 1) / (10 + 2),
    which is 0.75. The weight is that over the base, so 7.5.
    """
    table = _blank(100)
    table.loc[0:9, 'bank_not_on_file'] = 1
    table.loc[0:7, 'is_anomaly'] = 1
    table.loc[90:91, 'is_anomaly'] = 1

    weights = rules.learn_weights(table, CUTOFF)

    assert weights['account not on file'] == pytest.approx(7.5)


def test_a_rule_that_never_fires_weighs_nothing():
    table = _blank(100)
    table.loc[0:9, 'is_anomaly'] = 1

    weights = rules.learn_weights(table, CUTOFF)

    assert weights['duty at a rate not in force'] == 0.0


def test_the_test_weeks_cannot_move_a_weight():
    """The same training rows, two different futures.

    The later weeks are built to argue the opposite: there the rule is
    always wrong. If a single weight changes, something read ahead.
    """
    past = _blank(100)
    past.loc[0:9, 'bank_not_on_file'] = 1
    past.loc[0:7, 'is_anomaly'] = 1

    honest = rules.learn_weights(past, CUTOFF)

    future = _blank(100)
    future['issue_date'] = pd.Timestamp('2026-03-05')
    future.loc[0:49, 'bank_not_on_file'] = 1
    future.loc[0:49, 'is_anomaly'] = 0

    scores_before = rules.score(future, honest)
    scores_after = rules.score(future, rules.learn_weights(past, CUTOFF))

    assert honest == rules.learn_weights(past, CUTOFF)
    assert scores_before.equals(scores_after)


def test_the_model_fits_its_weights_on_the_training_rows_only():
    """`Rules.fit_score` is handed a train frame and a test frame. The
    ranking it gives the test frame must not change when the test
    frame's own answers are destroyed."""
    past = _blank(60)
    past.loc[0:5, 'bank_not_on_file'] = 1
    past.loc[0:5, 'is_anomaly'] = 1
    past.loc[6:11, 'under_limit_but_order_over'] = 1

    later = _blank(40)
    later['issue_date'] = pd.Timestamp('2026-02-05')
    later.loc[0:9, 'bank_not_on_file'] = 1
    later.loc[10:19, 'under_limit_but_order_over'] = 1
    later.loc[0:19, 'is_anomaly'] = 1

    truthful = train.Rules().fit_score(past, later)
    lying = train.Rules().fit_score(past, later.assign(is_anomaly=0))

    assert list(truthful) == list(lying)


def test_a_rare_and_right_rule_outranks_a_common_and_wrong_one():
    """The fault this replaced: two rules of the same hand-set weight,
    one certain and one noise, scored the same."""
    table = _blank(1000)
    table.loc[0:9, 'bank_not_on_file'] = 1
    table.loc[0:9, 'is_anomaly'] = 1
    table.loc[10:209, 'under_limit_but_order_over'] = 1
    table.loc[10:13, 'is_anomaly'] = 1

    weights = rules.learn_weights(table, CUTOFF)
    certain = _blank(1).assign(bank_not_on_file=1)
    noisy = _blank(1).assign(under_limit_but_order_over=1)

    assert (rules.score(certain, weights) > rules.score(noisy, weights)).all()
