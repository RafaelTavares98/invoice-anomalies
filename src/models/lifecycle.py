"""A model's life after the first comparison.

Walk forward: train on every week before a cutoff, with the labels known
by then, and test the weeks after it, as the year goes by. One held-out
score can be luck; a score that holds week after week is not.

Retrain: the analyst's verdicts become labels, a new model trains on
them, and it is challenged on the held-out year.

Promote: a model is registered in MLflow, and it becomes `champion` only
when it beats the champion before it on the same held-out data. A tie is
not a win. The service always loads
`models:/invoice-anomaly@champion`.
"""

import numpy as np
import pandas as pd

from evaluation import metrics, splits
from models import train

NAME = 'invoice-anomaly'


def walk_forward(table, model_name, first_week=8, step=4, k=5):
    """Score per fold: train on weeks before the cutoff, test the next
    `step` weeks. Seeds are parallel years, so week numbers are shared."""
    weeks = splits.week_of(table)
    folds = []
    for start in range(first_week, int(weeks.max()) + 1, step):
        past = table[weeks < start]
        future = table[(weeks >= start) & (weeks < start + step)]
        if not len(future) or not future['is_anomaly'].any():
            continue
        cutoff = past['issue_date'].max()
        past = past.assign(label=splits.known_labels(past, cutoff))
        if past['label'].nunique() < 2:
            continue
        scores = train.MODELS[model_name]().fit_score(past, future)
        found = train.report(future, scores, k)
        folds.append({'week': start, **found})
    return folds


def retrain(table, as_of, feedback, seed=0, features=None):
    """A new LightGBM on the analyst's labels, up to `as_of`.

    `feedback` is the label Series from `models.feedback.labels_from`,
    aligned on the table's index. Rows whose verdict is still pending
    carry no label and are left out. Only the named features are fitted:
    35 errors buy about three, and a silent default is how twenty-four
    crept back in. Returns the fitted model and the rows it saw.
    """
    if features is None:
        raise ValueError('name the features to fit; there is no default')
    from lightgbm import LGBMClassifier
    seen = table[table['issue_date'] <= pd.Timestamp(as_of)]
    labels = feedback.reindex(seen.index).dropna().astype(int)
    if labels.nunique() < 2:
        raise ValueError('the labels hold one class; nothing to learn')
    rows = seen.loc[labels.index]
    columns = list(features)
    params = dict(train.Boosting.params, random_state=seed)
    model = LGBMClassifier(**params)
    model.fit(rows[columns].fillna(0), labels.to_numpy())
    return model, rows


def features_of(model):
    """The columns the model was fitted on."""
    names = getattr(model, 'feature_name_', None)
    if names is None:
        raise ValueError('the model does not name its features')
    return list(names)


def challenge(model, held_out, features=None):
    """Score a model on the features it was fitted on.

    With no list, the caller gets the week-by-week report it always got.
    With one, the list must be the model's own, and the answer carries
    the ordinary measures beside it: precision, recall and F1 over the
    week's top k, and the average precision of the ranking.
    """
    if features is None:
        scores = model.predict_proba(train._matrix(held_out))[:, 1]
        return train.report(held_out, scores)
    columns = list(features)
    if columns != features_of(model):
        raise ValueError('the features differ from the model\'s own')
    scores = model.predict_proba(held_out[columns].fillna(0))[:, 1]
    scored = held_out.assign(score=scores, week=splits.week_of(held_out))
    top = metrics._top(scored, metrics.K).index
    called = np.asarray(scored.index.isin(top), dtype=int)
    truth = scored['is_anomaly'].to_numpy(dtype=int)
    return {'precision': metrics.precision(truth, called),
            'recall': metrics.recall(truth, called),
            'f1': metrics.f1(truth, called),
            'average_precision': metrics.pr_curve(truth,
                                                  scores)['average_precision']}


def register(model_uri, tracking_uri=None, score=None, features=None):
    """Register a model and make it champion, on the loop's verdict.

    A challenger that wins its weeks since training has already earned
    the alias; the held-out score is only the number reported beside it.
    """
    import mlflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    version = mlflow.register_model(model_uri, NAME)
    if score is not None:
        client.set_model_version_tag(NAME, version.version,
                                     'average_precision', str(score))
    if features:
        client.set_model_version_tag(NAME, version.version, 'features',
                                     ','.join(features))
    client.set_registered_model_alias(NAME, 'champion', version.version)
    return version.version


def register_if_better(model_uri, score, tracking_uri=None, metric='average_precision',
                       features=None):
    """Register the model; make it champion if it beats the current one."""
    import mlflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    columns = ','.join(features) if features else ''
    try:
        current = client.get_model_version_by_alias(NAME, 'champion')
        best = float(current.tags.get(metric, '-inf'))
        # A champion that reads other features cannot score today's
        # invoices, so its old score is no bar to clear.
        if columns and current.tags.get('features', '') != columns:
            best = float('-inf')
    except Exception:
        best = float('-inf')
    version = mlflow.register_model(model_uri, NAME)
    client.set_model_version_tag(NAME, version.version, metric, str(score))
    if columns:
        client.set_model_version_tag(NAME, version.version, 'features', columns)
    if score <= best:
        client.set_registered_model_alias(NAME, 'challenger', version.version)
        return False
    client.set_registered_model_alias(NAME, 'champion', version.version)
    return True
