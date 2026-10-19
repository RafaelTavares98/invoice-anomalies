"""Train the models, score them the same way, record every run in MLflow.

From simplest to strongest, as the plan orders them:

1. rules       the checks a desk writes by hand; the bar to beat
2. isolation   PyOD Isolation Forest, no labels needed
3. ecod        PyOD ECOD, no labels, no settings
4. lightgbm    gradient boosting on the labels known at the cutoff

Every model sees the same training invoices and is scored on the same
test weeks. One MLflow parent run holds the comparison; each model is a
child run with its parameters, its metrics and, for the trained ones,
the model itself.
"""

import numpy as np
import pandas as pd

from evaluation import metrics, splits
from features.build import COLUMNS
from models import rules

EXPERIMENT = 'invoice-anomalies'


def _matrix(table):
    return table[list(COLUMNS)].astype(float)


def _labels(table):
    return table['label'] if 'label' in table else table['is_anomaly']


class Rules:
    params = {}

    def fit_score(self, train, test):
        weights = rules.learn_weights(train, train['issue_date'].max())
        return rules.score(test, weights).to_numpy()


class _PyOD:
    params = {}

    def fit_score(self, train, test):
        detector = self.make()
        detector.fit(_matrix(train).fillna(0).to_numpy())
        return detector.decision_function(_matrix(test).fillna(0).to_numpy())


class Isolation(_PyOD):
    params = {'n_estimators': 300, 'random_state': 0}

    def make(self):
        from pyod.models.iforest import IForest
        return IForest(**self.params)


class Ecod(_PyOD):
    def make(self):
        from pyod.models.ecod import ECOD
        return ECOD()


class Boosting:
    params = {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 15,
              'min_child_samples': 10, 'class_weight': 'balanced',
              'random_state': 0, 'verbose': -1}

    def __init__(self):
        self.model = None

    def fit_score(self, train, test):
        from lightgbm import LGBMClassifier
        self.model = LGBMClassifier(**self.params)
        self.model.fit(_matrix(train), _labels(train))
        return self.model.predict_proba(_matrix(test))[:, 1]


MODELS = {'rules': Rules, 'isolation_forest': Isolation, 'ecod': Ecod,
          'lightgbm': Boosting}


def report(test, scores, k=metrics.K):
    """Every number the README reports for one model on one test set."""
    scored = test.assign(score=scores, week=splits.week_of(test))
    found = {
        'precision_at_k': metrics.precision_at_k(scored, k),
        'average_precision': metrics.average_precision(scored),
        'recall_at_25': metrics.recall_at(scored, metrics.DESK),
    }
    found |= {f'recall_{kind}': value
              for kind, value in metrics.recall_by_kind(scored, k).items()}
    found |= {f'twin_alarm_{kind}': value
              for kind, value in metrics.twin_alarms(scored, k).items()}
    return found


def with_known_labels(train):
    """The labels a desk would have had on the last training day."""
    return train.assign(label=splits.known_labels(train, train['issue_date'].max()))


def compare(train, test, tracking_uri=None, k=metrics.K, log_models=False,
            description=''):
    """Fit and score every model; one MLflow parent run, one child each."""
    import mlflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    train = with_known_labels(train)
    results = {}
    with mlflow.start_run(run_name='compare', description=description):
        mlflow.log_input(mlflow.data.from_pandas(
            _matrix(train).assign(label=train['label']), name='train'),
            context='training')
        mlflow.log_input(mlflow.data.from_pandas(
            _matrix(test).assign(is_anomaly=test['is_anomaly']), name='test'),
            context='testing')
        mlflow.log_params({'k': k, 'train_rows': len(train), 'test_rows': len(test),
                           'train_labels_known': int(train['label'].sum()),
                           'train_anomalies_true': int(train['is_anomaly'].sum())})
        for name, kind in MODELS.items():
            with mlflow.start_run(run_name=name, nested=True) as run:
                model = kind()
                scores = model.fit_score(train, test)
                found = report(test, scores, k)
                mlflow.log_params({'model': name, **model.params})
                mlflow.log_metrics(found)
                if log_models and name == 'lightgbm':
                    _log_boosting(model, train)
                results[name] = {**found, 'run_id': run.info.run_id}
    return results


def _log_boosting(model, train):
    import mlflow
    from mlflow.models import infer_signature
    sample = _matrix(train).head(5)
    mlflow.lightgbm.log_model(
        model.model, name='model',
        signature=infer_signature(sample, model.model.predict_proba(sample)[:, 1]),
        input_example=sample)


def table_for(data, seed_name):
    """Features and answers for one dataset, side by side."""
    from features import build
    table = build.features(data)
    labels = data.labels.drop(columns='number')
    return pd.concat([table, labels], axis=1).assign(seed=seed_name)
