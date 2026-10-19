"""The README tells the truth: every command, and the latest numbers."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
README = HERE / 'README.md'

ROWS = {
    'rules': 'Rules a desk writes by hand',
    'isolation_forest': 'Isolation Forest, no labels',
    'ecod': 'ECOD, no labels',
    'lightgbm': 'LightGBM',
}


def test_the_readme_names_every_command_in_help():
    result = subprocess.run([sys.executable, 'run.py', '--help'], cwd=HERE,
                            capture_output=True, text=True)
    commands = set(re.findall(r'python run\.py (\w+)', result.stdout))
    text = README.read_text(encoding='utf-8')

    assert commands
    for command in commands:
        assert f'run.py {command}' in text, command


def test_the_table_matches_the_latest_comparison():
    import mlflow
    db = HERE / 'mlflow.db'
    if not db.exists():
        pytest.skip('no local MLflow store')
    mlflow.set_tracking_uri(f'sqlite:///{db.as_posix()}')
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name('invoice-anomalies')
    parents = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'compare'",
        order_by=['start_time DESC'], max_results=1)
    parent = parents[0]
    children = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
    text = README.read_text(encoding='utf-8')

    assert children
    for run in children:
        name = run.data.tags['mlflow.runName']
        metrics = run.data.metrics
        line = next(line for line in text.splitlines()
                    if line.startswith('|') and ROWS[name] in line)
        cells = [cell.strip().strip('*') for cell in line.strip('|').split('|')]
        assert cells[1:4] == [
            f"{metrics['precision_at_k']:.2f}",
            f"{metrics['recall_at_25']:.2f}",
            f"{metrics['average_precision']:.2f}",
        ], name
