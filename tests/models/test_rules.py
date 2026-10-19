"""The rules a desk would write by hand: the bar every model must beat."""

from features import build
from models import rules


def test_every_planted_anomaly_outranks_the_clean_invoice(tiny):
    table = build.features(tiny).set_index('number')
    scores = rules.score(table)

    for number in ('A-2', 'A-3', 'S-1', 'F-1'):
        assert scores[number] > scores['A-1']


def test_a_score_is_between_zero_and_one(tiny):
    scores = rules.score(build.features(tiny))

    assert scores.between(0, 1).all()
