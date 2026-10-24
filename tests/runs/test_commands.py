"""The commands that have to work before anything can be left running.

Nothing here trains or scores for real. These tests pin the shape of
the four commands the project needs on Moty's own run: that they
exist, that they read the live run and nothing else, that they write
what the dashboard reads, and that running twice does not corrupt
what the first run wrote.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[2]

#: What the dashboard of project 03 reads.
SCORED = HERE.parent / '03-payables-panel' / 'data' / 'anomalies.csv'

#: The columns that file must carry, whatever else is added.
COLUMNS = ('number', 'supplier', 'kind', 'issue_date', 'week',
           'grand_total', 'score', 'flagged', 'check', 'stage', 'checks',
           'decision', 'origin')


@pytest.fixture(scope='module')
def commands():
    import run
    return run


def test_the_four_commands_exist(commands):
    for name in ('prepare', 'train', 'score', 'serve'):
        assert hasattr(commands, name), name


def test_no_command_reads_an_invented_dataset(commands):
    """`year-seed` and `drift-seed` are the dropped runs. No command may
    name one any more."""
    source = Path(commands.__file__).read_text(encoding='utf-8')

    assert 'year-seed' not in source
    assert 'drift-seed' not in source


def test_prepare_writes_a_table_and_says_what_it_cut(commands, tmp_path):
    """`prepare` loads the run, builds the features, and cuts them to
    the number the error count allows. It reports both sizes."""
    report = commands.prepare(out=tmp_path / 'features.csv', log=tmp_path / 'runs.log')

    assert (tmp_path / 'features.csv').exists()
    assert report['rows'] == 1003
    assert report['errors'] == 35
    assert report['features_before'] > report['features_after'] >= 1
    assert report['kept'] == sorted(report['kept'], key=report['order'].index)


def test_train_splits_by_time_and_reports_both_sides(commands, tmp_path):
    """`train` never scores a week it trained on, and says where the
    line falls."""
    commands.prepare(out=tmp_path / 'features.csv', log=tmp_path / 'runs.log')

    report = commands.train(table=tmp_path / 'features.csv', log=tmp_path / 'runs.log',
                            report_path=tmp_path / 'model-report.json')

    assert report['train']['last'] < report['test']['first']
    assert report['test']['errors'] >= 1
    assert report['train']['errors'] + report['test']['errors'] == 35


def test_train_reports_every_figure_with_an_interval(commands, tmp_path):
    commands.prepare(out=tmp_path / 'features.csv', log=tmp_path / 'runs.log')

    report = commands.train(table=tmp_path / 'features.csv', log=tmp_path / 'runs.log',
                            report_path=tmp_path / 'model-report.json')

    for model, found in report['models'].items():
        for name in ('precision', 'recall', 'f1'):
            low, high = found[name]['interval']
            assert low <= found[name]['value'] <= high, (model, name)


def test_train_keeps_the_clerk_checks_as_the_baseline(commands, tmp_path):
    """The clerk's own checks are measured the same way as any model,
    and the result stands whichever wins."""
    commands.prepare(out=tmp_path / 'features.csv', log=tmp_path / 'runs.log')

    report = commands.train(table=tmp_path / 'features.csv', log=tmp_path / 'runs.log',
                            report_path=tmp_path / 'model-report.json')

    assert 'rules' in report['models']
    assert report['baseline'] == 'rules'


def test_score_writes_the_file_the_dashboard_reads(commands, tmp_path):
    out = tmp_path / 'anomalies.csv'
    commands.prepare(out=tmp_path / 'features.csv', log=tmp_path / 'runs.log')
    commands.train(table=tmp_path / 'features.csv', log=tmp_path / 'runs.log',
                            report_path=tmp_path / 'model-report.json')

    commands.score(table=tmp_path / 'features.csv', out=out, log=tmp_path / 'runs.log')
    written = pd.read_csv(out, keep_default_na=False)

    assert list(COLUMNS) == [c for c in COLUMNS if c in written.columns]
    assert len(written) == 1003
    assert written['flagged'].isin((0, 1)).all()


def test_score_is_safe_to_run_twice(commands, tmp_path):
    """It will run on a schedule for days. The second run replaces the
    first cleanly, with no duplicated rows."""
    out = tmp_path / 'anomalies.csv'
    commands.prepare(out=tmp_path / 'features.csv', log=tmp_path / 'runs.log')
    commands.train(table=tmp_path / 'features.csv', log=tmp_path / 'runs.log',
                            report_path=tmp_path / 'model-report.json')

    commands.score(table=tmp_path / 'features.csv', out=out, log=tmp_path / 'runs.log')
    first = pd.read_csv(out, keep_default_na=False)
    commands.score(table=tmp_path / 'features.csv', out=out, log=tmp_path / 'runs.log')
    again = pd.read_csv(out, keep_default_na=False)

    assert len(first) == len(again)
    assert again['number'].is_unique


def test_every_run_leaves_a_dated_line_in_a_log(commands, tmp_path):
    """Left running for days, the only way to know what happened is the
    log. Each command appends one line, with a real timestamp."""
    log = tmp_path / 'runs.log'
    commands.prepare(out=tmp_path / 'features.csv', log=log)

    lines = [json.loads(line) for line in
             log.read_text(encoding='utf-8').splitlines() if line.strip()]

    assert lines
    last = lines[-1]
    assert last['command'] == 'prepare'
    assert pd.Timestamp(last['at']).tz is not None
    assert 'rows' in last
