"""What the dashboard is given to draw.

The four panels are a confusion matrix, a precision-recall curve, the
errors caught as more invoices are checked, and the recall of each
kind of error. None of those can be drawn from the gauges this project
used to publish, so the exporter publishes the figures themselves.

These tests pin what is published, not how it is drawn.
"""

import json

import pandas as pd
import pytest

from models import exporter


@pytest.fixture
def report():
    """One finished evaluation, as `train` hands it over."""
    return {
        'split': {'cutoff': '2026-07-31',
                  'train': {'first': '2026-01-02', 'last': '2026-07-31',
                            'errors': 23},
                  'test': {'first': '2026-08-03', 'last': '2026-10-14',
                           'errors': 12}},
        'baseline': 'rules',
        'models': {
            'rules': {
                'confusion': {'caught': 5, 'missed': 7,
                              'false_alarm': 9, 'left_alone': 240},
                'precision': {'value': 0.36, 'interval': (0.18, 0.56)},
                'recall': {'value': 0.42, 'interval': (0.20, 0.65)},
                'f1': {'value': 0.38, 'interval': (0.20, 0.58)},
                'average_precision': {'value': 0.31,
                                      'interval': (0.15, 0.50)},
                'recall_by_kind': {'freight_overbill': 0.5,
                                   'near_duplicate': 0.0},
                'caught_by_checked': [[0, 0], [25, 3], [50, 5]],
            },
            'lightgbm': {
                'confusion': {'caught': 7, 'missed': 5,
                              'false_alarm': 11, 'left_alone': 238},
                'precision': {'value': 0.39, 'interval': (0.21, 0.58)},
                'recall': {'value': 0.58, 'interval': (0.32, 0.82)},
                'f1': {'value': 0.47, 'interval': (0.27, 0.66)},
                'average_precision': {'value': 0.40,
                                      'interval': (0.20, 0.61)},
                'recall_by_kind': {'freight_overbill': 0.75,
                                   'near_duplicate': 0.33},
                'caught_by_checked': [[0, 0], [25, 5], [50, 7]],
            },
        },
    }


def test_the_report_is_written_where_the_dashboard_reads_it(report, tmp_path):
    out = tmp_path / 'model-report.json'

    exporter.publish_report(report, out)
    written = json.loads(out.read_text(encoding='utf-8'))

    assert written['models'].keys() == report['models'].keys()
    assert written['baseline'] == 'rules'


def test_every_figure_keeps_its_interval(report, tmp_path):
    out = tmp_path / 'model-report.json'

    exporter.publish_report(report, out)
    written = json.loads(out.read_text(encoding='utf-8'))

    for model in written['models'].values():
        for name in ('precision', 'recall', 'f1', 'average_precision'):
            assert len(model[name]['interval']) == 2


def test_the_report_says_whether_the_models_are_separated(report, tmp_path):
    """Two overlapping intervals mean the comparison is not decided, and
    the file has to say so rather than leave a reader to eyeball it."""
    out = tmp_path / 'model-report.json'

    exporter.publish_report(report, out)
    written = json.loads(out.read_text(encoding='utf-8'))

    assert written['verdict']['separated'] is False
    assert written['verdict']['winner'] is None


def test_a_clear_win_is_reported_as_one(report, tmp_path):
    report['models']['lightgbm']['average_precision'] = {
        'value': 0.80, 'interval': (0.70, 0.90)}
    out = tmp_path / 'model-report.json'

    exporter.publish_report(report, out)
    written = json.loads(out.read_text(encoding='utf-8'))

    assert written['verdict']['separated'] is True
    assert written['verdict']['winner'] == 'lightgbm'


def test_the_split_dates_travel_with_the_figures(report, tmp_path):
    """A figure without its split is unreadable: the panel has to be
    able to say which weeks it was measured on."""
    out = tmp_path / 'model-report.json'

    exporter.publish_report(report, out)
    written = json.loads(out.read_text(encoding='utf-8'))

    assert written['split']['test']['first'] == '2026-08-03'
    assert written['split']['test']['errors'] == 12


def test_the_report_carries_the_time_it_was_written(report, tmp_path):
    out = tmp_path / 'model-report.json'

    exporter.publish_report(report, out)
    written = json.loads(out.read_text(encoding='utf-8'))

    assert pd.Timestamp(written['written_at']).tz is not None


def test_the_gauges_still_publish_the_running_figures(tmp_path):
    """The dashboard draws the panels from the report; Prometheus keeps
    the running count. Both go on being published."""
    from prometheus_client import CollectorRegistry

    scored = pd.DataFrame({
        'stage': ['billing', 'customs'],
        'decision': ['confirmed', 'pending'],
        'grand_total': [1200.0, 800.0],
        'flagged': [1, 1],
    })

    registry = exporter.publish(CollectorRegistry(), scored)
    names = {metric.name for metric in registry.collect()}

    assert 'anomaly_flags' in names
    assert 'anomaly_decisions' in names
