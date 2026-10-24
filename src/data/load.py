"""The live run written by project 01, as typed tables.

`live()` reads the one real run. Its invoices come from the ledger when
no CSV sits at the root, because the reader keeps 1,003 of the 1,044
documents and drops the exact resends.

An exact resend arrives twice with the same bytes. The reader's hash
keeps one, so the model sees one, and its label goes with it.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATES = ('issue_date', 'due_date', 'onboarded_on', 'changed_on', 'raised_on',
         'required_by', 'effective_on', 'received_on', 'departed_on',
         'arrived_on')
AMOUNTS = ('grand_total', 'subtotal', 'discount', 'quantity', 'unit_price',
           'net', 'price', 'agreed_price', 'value', 'limit')


def _typed(frame):
    frame = frame.copy()
    for column in frame.columns:
        if column in DATES:
            frame[column] = pd.to_datetime(frame[column], errors='coerce')
        elif column in AMOUNTS:
            frame[column] = pd.to_numeric(frame[column], errors='coerce').astype(float)
        elif frame[column].dtype == object:
            frame[column] = frame[column].fillna('')
    return frame


@dataclass
class Dataset:
    name: str
    invoices: pd.DataFrame
    lines: pd.DataFrame
    erp: dict
    labels: pd.DataFrame
    raw_invoices: pd.DataFrame = field(repr=False, default=None)
    raw_lines: pd.DataFrame = field(repr=False, default=None)
    raw_erp: dict = field(repr=False, default=None)
    raw_labels: pd.DataFrame = field(repr=False, default=None)

    @classmethod
    def from_frames(cls, name, invoices, lines, erp, labels):
        """The invoices and their answers, matched by invoice number.

        They used to be paired off by position, which held only while
        every invoice in the run was in the file. A dataset of what was
        read holds the invoices somebody drew and the reader kept, and
        the answers cover the whole run, so position puts the wrong
        answer on every row.
        """
        invoices = invoices.reset_index(drop=True).fillna('')
        labels = labels.reset_index(drop=True).fillna('')
        keep = ~invoices.astype(str).duplicated()
        kept_invoices = _typed(invoices[keep]).reset_index(drop=True)

        answers = labels.drop_duplicates('number').set_index('number')
        known = kept_invoices['number'].isin(answers.index)
        # An invoice with no answer cannot be scored, and keeping it
        # would count as clean something nobody ever checked.
        kept_invoices = kept_invoices[known].reset_index(drop=True)
        kept_labels = (answers.reindex(kept_invoices['number'])
                       .reset_index())
        kept_labels = kept_labels[['number', 'anomaly', 'lookalike_of',
                                   'found_after_days']].copy()
        kept_labels['is_anomaly'] = (kept_labels['anomaly'] != '').astype(int)
        kept_labels['found_after_days'] = pd.to_numeric(
            kept_labels['found_after_days'], errors='coerce')
        typed_lines = _typed(lines.fillna('').astype(str).drop_duplicates())
        return cls(name, kept_invoices, typed_lines,
                   {k: _typed(v) for k, v in erp.items()}, kept_labels,
                   invoices, lines, erp, labels)


def _csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def live(root):
    """The one real run, as the typed tables the rest of the code wants.

    The folder is the pipeline's own: labels and erp at the root, the
    invoices either at the root or in the ledger the pipeline writes.
    The key holds 1,044 rows for 1,003 numbers: 41 were issued twice,
    revised or resent. One invoice is one number, so one row is kept.
    """
    root = Path(root)
    labels_path = root / 'labels.csv'
    if not labels_path.exists():
        raise FileNotFoundError(f'no labels.csv in {root}')
    raw_invoices = _one_per_number(_live_invoices(root))
    raw_lines = _live_lines(root)
    raw_labels = _one_per_number(_csv(labels_path))
    erp = {p.stem: _csv(p) for p in (root / 'erp').glob('*.csv')}
    return Dataset(
        root.name,
        _typed(raw_invoices),
        _typed(raw_lines),
        {name: _typed(frame) for name, frame in erp.items()},
        _live_labels(raw_labels),
        raw_invoices, raw_lines, erp, raw_labels)


def _one_per_number(frame):
    """The pipeline issues a document again under the same number on a
    revision or a resend; the desk files one invoice per number."""
    if 'number' not in frame:
        return frame
    return frame.drop_duplicates('number', keep='first').reset_index(drop=True)


def _live_invoices(root):
    """The invoices: the root CSV, else the ledger, else the data copy."""
    path = root / 'invoices.csv'
    if path.exists():
        return _csv(path)
    ledger = root / 'ledger.json'
    if ledger.exists():
        loaded = json.loads(ledger.read_text(encoding='utf-8'))
        invoices = loaded.get('invoices') if isinstance(loaded, dict) else None
        if isinstance(invoices, list) and invoices:
            frame = pd.DataFrame(invoices)
            return frame.drop(columns=[name for name in ('anomaly',
                                                         'lookalike_of')
                                       if name in frame])
    fallback = root / 'data' / 'invoices.csv'
    if fallback.exists():
        return _csv(fallback)
    raise FileNotFoundError(f'no invoices or ledger.json in {root}')


def _live_lines(root):
    """The lines: the root CSV, else the reader's copy, else nothing.

    The reader names its columns after its own work: `invoice_number`
    and `row` are the invoice and the line's place on the page. The
    feature code asks for `number` and `position`, so they are renamed
    on the way in.
    """
    path = root / 'invoice_lines.csv'
    if not path.exists():
        fallback = root / 'data' / 'invoice_lines.csv'
        if not fallback.exists():
            return pd.DataFrame()
        path = fallback
    lines = _csv(path)
    renames = {'invoice_number': 'number', 'row': 'position'}
    return lines.rename(columns={old: new for old, new in renames.items()
                                 if old in lines and new not in lines})


def _live_labels(frame):
    """The answer key, with the columns every caller already expects."""
    labels = frame.copy()
    for name in ('anomaly', 'lookalike_of'):
        labels[name] = labels[name].fillna('') if name in labels else ''
    labels['found_after_days'] = pd.to_numeric(
        labels['found_after_days'], errors='coerce')
    labels['is_anomaly'] = (labels['anomaly'] != '').astype(int)
    return labels
