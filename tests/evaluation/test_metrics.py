"""How a model is scored: as a desk that checks k invoices a week."""

import pandas as pd
import pytest

from evaluation import metrics


@pytest.fixture
def scored():
    return pd.DataFrame({
        'week': [1, 1, 1, 1, 2, 2, 2, 2],
        'score': [0.9, 0.8, 0.1, 0.0, 0.7, 0.6, 0.5, 0.4],
        'is_anomaly': [1, 0, 1, 0, 0, 0, 0, 1],
        'anomaly': ['account_mismatch', '', 'unmatched_vendor', '', '', '', '', 'split_billing'],
        'lookalike_of': ['', 'account_mismatch', '', '', 'price_drift', '', '', ''],
    })


def test_precision_in_the_top_k_per_week(scored):
    """Week 1: top 2 hold 1 anomaly. Week 2: top 2 hold none. Mean 0.25."""
    assert metrics.precision_at_k(scored, k=2) == pytest.approx(0.25)


def test_recall_by_kind_counts_what_the_top_k_caught(scored):
    found = metrics.recall_by_kind(scored, k=2)

    assert found == {'account_mismatch': 1.0, 'unmatched_vendor': 0.0, 'split_billing': 0.0}


def test_false_alarms_on_twins(scored):
    """Two twins; both landed in a top 2, both are false alarms."""
    alarms = metrics.twin_alarms(scored, k=2)

    assert alarms == {'account_mismatch': 1.0, 'price_drift': 1.0}


def test_recall_at_a_desks_weekly_capacity(scored):
    """Top 2 per week catch the account mismatch only: 1 of 3."""
    assert metrics.recall_at(scored, k=2) == pytest.approx(1 / 3)


def test_average_precision_over_everything(scored):
    assert 0 < metrics.average_precision(scored) <= 1


@pytest.fixture
def customs_scored():
    return pd.DataFrame({
        'week': [1, 1, 1, 1, 1],
        'score': [0.9, 0.8, 0.7, 0.1, 0.0],
        'is_anomaly': [1, 1, 0, 0, 0],
        'anomaly': ['duty_rate_wrong', 'broker_fee_over', '', '', ''],
        'lookalike_of': ['', '', 'fee_amended', '', ''],
    })


def test_customs_kinds_land_on_the_scoreboard(customs_scored):
    """The two customs kinds each get their own recall line."""
    found = metrics.recall_by_kind(customs_scored, k=2)

    assert 'duty_rate_wrong' in found
    assert 'broker_fee_over' in found


def test_amended_fee_twin_gets_its_alarm_line(customs_scored):
    alarms = metrics.twin_alarms(customs_scored, k=2)

    assert 'fee_amended' in alarms
