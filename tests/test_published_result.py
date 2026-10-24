"""Once a result exists, the README publishes it, whoever won.

The clerk's twelve checks are the baseline. If they beat every model,
that is the finding and it goes on the front page as it stands. A
README that keeps saying "None yet" while a report sits on disk is
the same dishonesty as a made-up score, pointing the other way.
"""

import json
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
README = HERE / 'README.md'
REPORT = HERE / 'data' / 'model-report.json'


@pytest.fixture(scope='module')
def text():
    return README.read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def report():
    assert REPORT.exists(), f'no report at {REPORT}; run train first'
    return json.loads(REPORT.read_text(encoding='utf-8'))


def _results(text):
    section = text.split('## Results', 1)
    assert len(section) == 2, 'the README has no Results section'
    return section[1].split('\n## ', 1)[0]


def test_the_results_section_no_longer_says_none_yet(text):
    assert 'None yet' not in _results(text)


def test_every_model_in_the_report_is_on_the_page(text, report):
    section = _results(text)

    for name in report['models']:
        assert name.replace('_', ' ') in section.lower(), name


def test_each_figure_is_printed_with_its_interval(text, report):
    """A point estimate alone is the thing this project refuses to
    publish. Every model's line carries two more numbers."""
    section = _results(text)

    for name, found in report['models'].items():
        line = next(l for l in section.splitlines()
                    if name.replace('_', ' ') in l.lower())
        numbers = re.findall(r'0\.\d\d', line)
        assert len(numbers) >= 3, (name, line)


def test_each_figure_is_the_one_in_the_report(text, report):
    """The page once kept last week's figures after the report moved.
    Each model's line must carry its precision, recall and average
    precision as the report holds them, to two places."""
    section = _results(text)

    for name, found in report['models'].items():
        line = next(l for l in section.splitlines()
                    if l.startswith('|') and name.replace('_', ' ')
                    in l.lower())
        for figure in ('precision', 'recall', 'average_precision'):
            printed = f"{found[figure]['value']:.2f}"
            assert printed in line, (name, figure, printed, line)


def test_the_verdict_is_stated_in_words(text, report):
    section = _results(text).lower()

    if report['verdict']['separated']:
        assert str(report['verdict']['winner']).replace('_', ' ') in section
    else:
        assert 'overlap' in section or 'not separated' in section


def test_the_baseline_result_is_not_buried(text, report):
    """If the clerk's checks win, the page says so plainly, near the
    top of the section."""
    section = _results(text)
    models = report['models']
    best = max(models, key=lambda n: models[n]['average_precision']['value'])

    if best == report['baseline']:
        head = '\n'.join(section.strip().splitlines()[:6]).lower()
        assert 'checks' in head or 'clerk' in head


def test_the_size_of_the_test_side_is_stated(text, report):
    """Ten errors is why the intervals are wide. The reader is told."""
    section = _results(text)

    assert str(report['split']['test']['errors']) in section


def test_the_dates_the_result_was_measured_on_are_stated(text, report):
    section = _results(text)
    first = report['split']['test']['first']
    last = report['split']['test']['last']

    assert first in section or first[:7] in section
    assert last in section or last[:7] in section
