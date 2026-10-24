"""The README tells the truth about the one run this project reads.

The old version of this file policed a results table and a list of
commands. Both described five invented training years that have been
dropped, so both are gone. What is policed now is narrower and
harder: the numbers on the front page must match the answer key, and
the page must not claim a result while none has been measured.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
README = HERE / 'README.md'
LIVE = HERE.parent / '01-invoice-pipeline' / 'Invoices' / 'live'


@pytest.fixture(scope='module')
def text():
    return README.read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def answers():
    path = LIVE / 'labels.csv'
    assert path.exists(), f'the answer key is not at {path}'
    return pd.read_csv(path, keep_default_na=False)


def _numbers(text):
    """Every number on the page, commas stripped."""
    return {int(found.replace(',', ''))
            for found in re.findall(r'\b\d[\d,]*\b', text)}


def test_the_invoice_count_matches_the_answer_key(text, answers):
    assert len(answers) in _numbers(text)


def test_the_error_count_matches_the_answer_key(text, answers):
    errors = int((answers['anomaly'] != '').sum())

    assert errors in _numbers(text)


def test_the_false_positive_count_matches_the_answer_key(text, answers):
    twins = int((answers['lookalike_of'] != '').sum())

    assert twins in _numbers(text)


def test_the_error_table_adds_up(text, answers):
    """The table of error kinds carries its own total; that total has to
    be the number of errors in the key."""
    errors = int((answers['anomaly'] != '').sum())
    twins = int((answers['lookalike_of'] != '').sum())
    total = next(line for line in text.splitlines()
                 if line.startswith('| **total**'))
    cells = [cell.strip().strip('*') for cell in total.strip('|').split('|')]

    assert cells[1:3] == [str(errors), str(twins)]


def test_a_score_is_printed_only_when_a_report_backs_it(text):
    """The page may print a result when one has been measured, and not
    otherwise. Which of the two it is depends on whether the report is
    on disk, so this test reads the same thing the reader would."""
    results = text.split('## Results', 1)
    assert len(results) == 2, 'the README has no Results section'
    section = results[1].split('\n## ', 1)[0]
    measured = (HERE / 'data' / 'model-report.json').exists()

    if not measured:
        assert 'None yet' in section
        assert not re.search(r'\b0\.\d\d\b', section), \
            'a score is printed with no report behind it'
    else:
        assert 'None yet' not in section
        assert re.search(r'\b0\.\d\d\b', section), \
            'a report exists and the page prints no figure from it'


def test_the_invented_years_are_not_advertised(text):
    """`year-seed` and `drift-seed` are the dropped runs. The README may
    say they were dropped; it may not tell anyone to build them."""
    for line in text.splitlines():
        if 'run.py dataset' in line:
            pytest.fail(f'the README still tells the reader to build a '
                        f'dataset: {line.strip()}')


def test_every_command_the_readme_shows_exists(text):
    """Only commands `run.py` actually dispatches may appear in the
    page's shell blocks."""
    import ast

    source = (HERE / 'run.py').read_text(encoding='utf-8')
    known = set(re.findall(r"mode == '(\w+)'", source))
    shown = set(re.findall(r'python run\.py (\w+)', text))

    assert shown <= known, f'the README shows {shown - known}'
    assert ast.parse(source) is not None


def test_the_panels_shown_are_the_measured_ones(text):
    """The page once showed a sketch with invented numbers. It now
    shows the panels measured on the run, says so beside the image, and
    never points at the sketch."""
    assert 'panels-sketch' not in text
    line = next(i for i, l in enumerate(text.splitlines())
                if 'docs/panels.png' in l)
    nearby = '\n'.join(text.splitlines()[line:line + 4]).lower()

    assert 'measured' in nearby
    assert (HERE / 'docs' / 'panels.png').exists()


def test_the_page_carries_no_broken_characters(text):
    """A file saved in the wrong encoding shows SÃ£o for São. That is
    the first thing a visitor sees, so it is checked."""
    for broken in ('Ã', 'Â', 'â”'):
        assert broken not in text, broken
