"""The anomaly figures as Prometheus gauges.

Project 01's monitoring stack scrapes them beside the pipeline's own,
so every chart of project 02 is drawn in the same Grafana. Nothing is
drawn here.
"""

from prometheus_client import Gauge

PORT = 9109


def publish(registry, scored, average_precision=None, weekly=None,
            drift=None, retrainings=0, champion_version=None):
    """Set every gauge from one scored table and the loop's state."""
    flags = Gauge('anomaly_flags', 'Invoices flagged, by process stage.',
                  ['stage'], registry=registry)
    decisions = Gauge('anomaly_decisions', 'Flags by what the analyst '
                      'decided: confirmed, dismissed, pending.', ['decision'],
                      registry=registry)
    amount = Gauge('anomaly_flagged_amount', 'Money on the flagged invoices, '
                   'in each document\'s own currency.', registry=registry)
    champion = Gauge('anomaly_champion_average_precision', 'The champion '
                     'model\'s average precision on the held-out year.',
                     registry=registry)
    weekly_precision = Gauge('anomaly_weekly_precision', 'The precision of '
                             'the last finished week.', registry=registry)
    drift_gauge = Gauge('anomaly_feature_drift', 'The population stability '
                        'index of each feature, week over week.',
                        ['feature'], registry=registry)
    retrainings_gauge = Gauge('anomaly_retrainings', 'How many times the '
                              'loop retrained.', registry=registry)
    version_gauge = Gauge('anomaly_champion_version', 'The registry version '
                          'serving now.', registry=registry)
    flagged = scored[scored['flagged'] == 1]
    for stage, count in flagged['stage'].value_counts().items():
        flags.labels(stage).set(count)
    for decision, count in flagged['decision'].value_counts().items():
        if decision:
            decisions.labels(decision).set(count)
    amount.set(float(flagged['grand_total'].sum()))
    if average_precision is not None:
        champion.set(average_precision)
    if weekly is not None and len(weekly):
        finished = weekly[weekly['precision'].notna()]
        if len(finished):
            weekly_precision.set(float(finished['precision'].iloc[-1]))
    if drift:
        for feature, score in drift.items():
            drift_gauge.labels(feature).set(score)
    retrainings_gauge.set(retrainings)
    if champion_version is not None:
        version_gauge.set(int(champion_version))
    return registry
