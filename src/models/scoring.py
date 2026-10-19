"""Scores turned into rows a dashboard can read.

The model says how unusual an invoice is; it does not say why. The desk's
rules do, so each row carries the heaviest rule that fired, as the check
to make, and the stage of the process that rule belongs to. An invoice no
rule explains is still listed: the model saw something the rules did not.
"""

import pandas as pd

from evaluation import metrics
from models import rules

#: The stage of the process each rule points at.
STAGE = {
    'account not on file': 'vendor master',
    'sender not on file': 'vendor master',
    'young vendor': 'vendor master',
    'no order behind a goods bill': 'purchasing',
    'quantity above receipt': 'receiving',
    'price above contract': 'billing',
    'same amount again': 'billing',
    'order split under the limit': 'billing',
    'round total, not a known fee': 'billing',
    'charge not on the rate card': 'freight billing',
    'duty at a rate not in force': 'customs',
    'broker fee above contract': 'customs',
}

#: Said of an invoice the model flags and no rule explains.
NO_RULE = 'model only'

#: Heaviest first, so the first that fires is the one to check first.
_BY_WEIGHT = [name for name, _, _ in sorted(rules.RULES, key=lambda r: -r[2])]


def _names(data):
    vendors = data.erp.get('vendors', pd.DataFrame())
    if 'name' not in vendors:
        return {}
    return dict(zip(vendors['code'], vendors['name']))


#: Days an analyst takes to look at an honest flag and dismiss it.
REVIEW_DAYS = 3


def _decisions(out, data, as_of):
    """What the analyst decided about each flag, as of a day, and when.

    The simulation knows which flags were real and when each error would
    have come to light, so it plays the analyst: a real error is confirmed
    once it has been found, an honest invoice is dismissed after a short
    review, and anything younger than that is still pending."""
    labels = data.labels.set_index('number')
    issued = pd.to_datetime(out['issue_date'])
    decided = []
    decided_on = []
    for index, row in out.iterrows():
        if not row['flagged']:
            decided.append('')
            decided_on.append('')
            continue
        label = labels.loc[row['number']] if row['number'] in labels.index else None
        if isinstance(label, pd.DataFrame):
            label = label.iloc[0]
        real = label is not None and label['is_anomaly'] == 1
        delay = (label['found_after_days'] if real else REVIEW_DAYS)
        ready = issued[index] + pd.Timedelta(days=float(delay or 0)) <= as_of
        if not ready:
            decided.append('pending')
            decided_on.append('')
        else:
            decided.append('confirmed' if real else 'dismissed')
            decided_on.append(
                str((issued[index] + pd.Timedelta(days=float(delay or 0))).date()))
    return decided, decided_on


def rows(table, data, scores, k=metrics.K, as_of=None):
    """One row per invoice: who, when, how much, the score, the flag, the
    stage and check the rules point at, and what the analyst decided by
    `as_of` (the last invoice date when none is given). Every flag the
    model raises is a client error until the analyst says otherwise: the
    supplier's or the carrier's, never the reader's."""
    raised = rules.flags(table)
    names = _names(data)
    inv = data.invoices
    week = (inv['issue_date']
            - pd.to_timedelta(inv['issue_date'].dt.weekday, unit='D'))
    out = pd.DataFrame({
        'number': inv['number'],
        'supplier': inv['issuer'].map(lambda code: names.get(code, code)),
        'kind': inv['kind'],
        'issue_date': inv['issue_date'].dt.date.astype(str),
        'week': week.dt.date.astype(str),
        'grand_total': inv['grand_total'].round(2),
        'score': pd.Series(list(scores), index=inv.index).round(4),
    })
    fired = [[name for name in _BY_WEIGHT if raised.at[index, name]]
             for index in inv.index]
    out['check'] = [found[0] if found else NO_RULE for found in fired]
    out['stage'] = [STAGE[found[0]] if found else NO_RULE for found in fired]
    out['checks'] = ['; '.join(found) for found in fired]
    ranked = out.sort_values('score', ascending=False, kind='stable')
    top = ranked.groupby('week', group_keys=False).head(k).index
    out['flagged'] = out.index.isin(top).astype(int)
    out['origin'] = out['flagged'].map({1: 'client', 0: ''})
    as_of = as_of if as_of is not None else inv['issue_date'].max()
    out['decision'], out['decided_on'] = _decisions(out, data, as_of)
    return out
