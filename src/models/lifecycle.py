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

import pandas as pd

from evaluation import splits
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


def retrain(table, as_of, feedback, seed=0):
    """A new LightGBM trained on the analyst's labels, up to `as_of`.

    `feedback` is the label Series from `models.feedback.labels_from`,
    aligned on the table's index. Rows whose verdict is still pending
    carry no label and are left out. Returns the fitted model and the
    rows it saw, nothing more. Labels of one class carry nothing to
    learn, so they are refused.
    """
    from lightgbm import LGBMClassifier
    seen = table[table['issue_date'] <= pd.Timestamp(as_of)]
    labels = feedback.reindex(seen.index).dropna().astype(int)
    if labels.nunique() < 2:
        raise ValueError('the labels hold one class; nothing to learn')
    rows = seen.loc[labels.index]
    params = dict(train.Boosting.params, random_state=seed)
    model = LGBMClassifier(**params)
    model.fit(train._matrix(rows), labels.to_numpy())
    return model, rows


def challenge(model, held_out):
    """The held-out year, scored and reported as `train.report` would."""
    scores = model.predict_proba(train._matrix(held_out))[:, 1]
    return train.report(held_out, scores)


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
