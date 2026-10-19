"""The anomaly figures, on the same monitoring stack as project 01.

Charts live in Grafana only, so the scored file is turned into gauges a
Prometheus can scrape: flags by stage, what the analyst decided, and the
champion's held-out average precision.
"""

import pandas as pd
from prometheus_client import CollectorRegistry

from models import exporter


def scored():
    return pd.DataFrame({
        'number': ['A', 'B', 'C', 'D'],
        'stage': ['billing', 'customs', 'customs', 'model only'],
        'flagged': [1, 1, 1, 0],
        'decision': ['confirmed', 'dismissed', 'pending', ''],
        'grand_total': [100.0, 200.0, 300.0, 400.0],
    })


def value(registry, name, **labels):
    return registry.get_sample_value(name, labels)


def test_flags_by_stage():
    registry = CollectorRegistry()
    exporter.publish(registry, scored(), average_precision=0.89)

    assert value(registry, 'anomaly_flags', stage='customs') == 2
    assert value(registry, 'anomaly_flags', stage='billing') == 1


def test_what_the_analyst_decided():
    registry = CollectorRegistry()
    exporter.publish(registry, scored(), average_precision=0.89)

    assert value(registry, 'anomaly_decisions', decision='confirmed') == 1
    assert value(registry, 'anomaly_decisions', decision='pending') == 1


def test_the_money_flagged_and_the_champions_score():
    registry = CollectorRegistry()
    exporter.publish(registry, scored(), average_precision=0.89)

    assert value(registry, 'anomaly_flagged_amount') == 600.0
    assert value(registry, 'anomaly_champion_average_precision') == 0.89


def weekly():
    return pd.DataFrame({
        'week': ['2026-04-06', '2026-04-13'],
        'flagged': [5, 5],
        'confirmed': [3, 1],
        'decided': [5, 5],
        'precision': [0.6, 0.2],
    })


def test_the_last_finished_weeks_precision():
    registry = CollectorRegistry()
    exporter.publish(registry, scored(), weekly=weekly())

    assert value(registry, 'anomaly_weekly_precision') == 0.2


def test_the_drift_per_feature():
    registry = CollectorRegistry()
    exporter.publish(registry, scored(),
                     drift={'vendor_age_days': 0.4, 'invoices_30d': 0.1})

    assert value(registry, 'anomaly_feature_drift', feature='vendor_age_days') == 0.4
    assert value(registry, 'anomaly_feature_drift', feature='invoices_30d') == 0.1


def test_how_many_retrainings_happened():
    registry = CollectorRegistry()
    exporter.publish(registry, scored(), retrainings=7)

    assert value(registry, 'anomaly_retrainings') == 7


def test_which_version_serves_now():
    registry = CollectorRegistry()
    exporter.publish(registry, scored(), champion_version='4')

    assert value(registry, 'anomaly_champion_version') == 4
