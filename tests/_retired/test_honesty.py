# Retired 2026-10-19: it measured leakage on the five invented years, and those sets are gone.
"""Two ways to break the score on purpose. A model that cannot cheat
when handed the answer is not learning, and a model that scores well on
shuffled labels is reading something it should not."""

import numpy as np
import pandas as pd
import pytest

from data import load
from evaluation import splits
from models import train


@pytest.fixture
def twelve_weeks():
    """The first twelve weeks of one seed, the smallest real table."""
    table = train.table_for(load.read('year-seed1'), 'year-seed1')
    table = table.assign(week=splits.week_of(table))
    return table[table['week'] < 12].reset_index(drop=True)


def _split(table):
    """The first prequential fold: eight weeks train, the ninth tests."""
    seen, held = next(iter(splits.prequential(table, first_week=8)))
    return table.loc[seen], table.loc[held]


def _fit_scores(seen, held):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(**train.Boosting.params)
    model.fit(train._matrix(seen), seen['is_anomaly'])
    return model.predict_proba(train._matrix(held))[:, 1]


def test_a_model_handed_the_answer_cheats(twelve_weeks):
    """With the label as a feature, average precision jumps near 1."""
    from sklearn.metrics import average_precision_score
    seen, held = _split(twelve_weeks)
    leak_seen = train._matrix(seen).assign(the_answer=seen['is_anomaly'])
    leak_held = train._matrix(held).assign(the_answer=held['is_anomaly'])

    from lightgbm import LGBMClassifier
    model = LGBMClassifier(**train.Boosting.params)
    model.fit(leak_seen, seen['is_anomaly'])
    scores = model.predict_proba(leak_held)[:, 1]

    assert average_precision_score(held['is_anomaly'], scores) > 0.99


def test_a_model_on_shuffled_labels_scores_the_base_rate(twelve_weeks):
    """Nonsense labels carry no signal: the score falls to the base rate."""
    from sklearn.metrics import average_precision_score
    shuffled = twelve_weeks.copy()
    shuffled['is_anomaly'] = np.random.default_rng(7).permutation(
        shuffled['is_anomaly'].to_numpy())
    seen, held = _split(shuffled)
    scores = _fit_scores(seen, held)

    assert average_precision_score(held['is_anomaly'], scores) < 0.15
