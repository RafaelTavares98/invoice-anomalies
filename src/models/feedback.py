"""The analyst's verdict becomes the label the next training run learns.

The loop is: the model flags, an analyst decides, the decision becomes a
label, the model is retrained. This module is the third step, and it is
pure: a frame in, a frame out, no files, no MLflow.

A confirmed row is a real error and trains as label 1. A dismissed row
is an honest invoice and trains as label 0. A pending row teaches
nothing yet, so it drops rather than passing for clean. A row nobody
ever flagged is the normal case and trains as 0.
"""

import pandas as pd

from evaluation import splits

#: The verdicts that close a case.
DECIDED = ('confirmed', 'dismissed')


def labels_from(decisions, as_of):
    """One label per surviving row, aligned on the frame's index.

    `decisions` carries the scored rows joined with the answers: at
    least `number`, `flagged`, `decision`, `decided_on`, `issue_date`,
    `is_anomaly` and `found_after_days`. With no verdict at all, the
    labels are what `splits.known_labels` would give, so the loop
    degrades to today's behaviour instead of breaking.
    """
    frame = decisions.copy()
    as_of = pd.Timestamp(as_of)
    if 'decided_on' not in frame or frame['decision'].eq('').all():
        dated = frame.assign(issue_date=pd.to_datetime(frame['issue_date']))
        return splits.known_labels(dated, as_of)

    decided_on = pd.to_datetime(frame['decided_on'], errors='coerce')
    late = (decided_on > as_of) & frame['decision'].isin(DECIDED)
    if late.any():
        name = frame.loc[late, 'number'].iloc[0]
        raise ValueError(f'{name} was decided after the cutoff')

    flagged = frame['flagged'].astype(bool)
    confirmed = (frame['decision'] == 'confirmed') & (decided_on <= as_of)
    dismissed = (frame['decision'] == 'dismissed') & (decided_on <= as_of)
    pending = flagged & ~confirmed & ~dismissed

    labels = pd.Series(0, index=frame.index, name='label', dtype=int)
    labels[confirmed] = 1
    return labels[~pending]
