"""The command that is left running for days.

`prepare`, `train` and `score` each do one thing once. None of them
keeps the project alive, and none of them exercises the part a client
is actually buying: that a verdict becomes a label, a fall starts a
challenger, and a better model takes over.

`watch` is that command. It walks the run week by week, and it is the
only thing that has to survive being left alone.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def commands():
    import run
    return run


@pytest.fixture
def prepared(commands, tmp_path):
    commands.prepare(out=tmp_path / 'features.csv', log=tmp_path / 'runs.log')
    return tmp_path


def test_the_command_exists(commands):
    assert hasattr(commands, 'watch')


def test_it_walks_every_week_in_order(commands, prepared):
    events = commands.watch(table=prepared / 'features.csv',
                            log=prepared / 'runs.log',
                            out=prepared / 'anomalies.csv')

    weeks = [event['week'] for event in events]
    assert weeks == sorted(weeks)
    assert len(weeks) == len(set(weeks))
    assert len(weeks) >= 30


def test_each_week_records_what_was_checked_and_what_was_decided(
        commands, prepared):
    events = commands.watch(table=prepared / 'features.csv',
                            log=prepared / 'runs.log',
                            out=prepared / 'anomalies.csv')

    for event in events:
        assert event['checked'] >= 0
        assert event['confirmed'] + event['dismissed'] <= event['checked']


def test_no_week_is_scored_by_a_model_that_saw_it(commands, prepared):
    """The whole point of walking week by week. Each event names the
    last date its serving model was trained on; that date is always
    before the week being scored."""
    events = commands.watch(table=prepared / 'features.csv',
                            log=prepared / 'runs.log',
                            out=prepared / 'anomalies.csv')

    for event in events:
        if event['trained_to'] is None:
            continue
        assert pd.Timestamp(event['trained_to']) < pd.Timestamp(event['week'])


def test_a_retraining_is_recorded_with_what_triggered_it(commands, prepared):
    events = commands.watch(table=prepared / 'features.csv',
                            log=prepared / 'runs.log',
                            out=prepared / 'anomalies.csv')

    for event in (e for e in events if e['retrained']):
        assert event['because'] in ('precision', 'drift')


def test_a_challenger_only_takes_over_on_a_separated_verdict(
        commands, prepared):
    """35 errors cannot usually separate two models. A promotion on
    overlapping intervals is the error this project exists to avoid."""
    events = commands.watch(table=prepared / 'features.csv',
                            log=prepared / 'runs.log',
                            out=prepared / 'anomalies.csv')

    for event in (e for e in events if e['promoted']):
        assert event['separated'] is True


def test_it_writes_the_scored_file_as_it_goes(commands, prepared):
    out = prepared / 'anomalies.csv'

    commands.watch(table=prepared / 'features.csv',
                   log=prepared / 'runs.log', out=out)
    written = pd.read_csv(out, keep_default_na=False)

    assert written['number'].is_unique
    assert len(written) <= 1003


def test_every_week_leaves_a_line_in_the_log(commands, prepared):
    log = prepared / 'runs.log'

    events = commands.watch(table=prepared / 'features.csv',
                            log=log, out=prepared / 'anomalies.csv')
    lines = [json.loads(line) for line in
             log.read_text(encoding='utf-8').splitlines() if line.strip()]
    weeks = [line for line in lines if line['command'] == 'watch']

    assert len(weeks) == len(events)
    assert all(pd.Timestamp(line['at']).tz is not None for line in weeks)


def test_stopping_and_starting_again_carries_on(commands, prepared):
    """It will be stopped. When it starts again it must not redo the
    weeks it already decided, and must not lose them either."""
    out = prepared / 'anomalies.csv'
    state = prepared / 'watch-state.json'

    first = commands.watch(table=prepared / 'features.csv', out=out,
                           log=prepared / 'runs.log', state=state, weeks=10)
    second = commands.watch(table=prepared / 'features.csv', out=out,
                            log=prepared / 'runs.log', state=state)

    assert len(first) == 10
    assert second[0]['week'] > first[-1]['week']
    assert pd.read_csv(out, keep_default_na=False)['number'].is_unique


def test_a_restart_keeps_the_counts_and_the_weeks(commands, prepared):
    """The gauges read the state file. A restart that zeroed the
    retrainings and dropped the weeks already walked would show a loop
    that had barely run."""
    out = prepared / 'anomalies.csv'
    state = prepared / 'watch-state.json'

    first = commands.watch(table=prepared / 'features.csv', out=out,
                           log=prepared / 'runs.log', state=state, weeks=10)
    second = commands.watch(table=prepared / 'features.csv', out=out,
                            log=prepared / 'runs.log', state=state)
    saved = json.loads(state.read_text(encoding='utf-8'))

    retrained = sum(1 for event in first + second if event['retrained'])
    assert saved['retrainings'] == retrained
    assert len(saved['weekly']) == len(first) + len(second)
