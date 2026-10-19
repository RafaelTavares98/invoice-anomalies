"""Features for one invoice, as a payables desk would see it on review day.

An invoice is reviewed on its due date, before it is paid. Each feature
uses records dated on or before that day and earlier invoices of the
same issuer, never anything later. The families follow the plan:

- against the contract: price and quantity above what was agreed, charges
  missing from the rate card (three-way match);
- against the vendor master: an account or a sender not on file, a young
  vendor, no order behind the bill (vendor master checks);
- against the issuer's history: the same amount again, the time since
  the last bill, the amount against the issuer's usual (IEEE-CIS winner:
  score against the entity);
- digits: round totals (audit round-number tests).
"""

import numpy as np
import pandas as pd

#: Freight lines that are the move itself, not a charge on top of it.
MAIN_FREIGHT = ('Air freight', 'Ocean freight', 'Road haulage', 'Port storage')

#: The columns a model may use, in a fixed order.
COLUMNS = (
    'price_over_contract', 'quantity_over_receipt', 'receipt_known',
    'unknown_charges', 'known_fixed_fee',
    'bank_not_on_file', 'sender_not_on_file', 'vendor_age_days', 'has_order',
    'same_amount_30d', 'days_since_last', 'invoices_30d', 'amount_zscore',
    'same_order_7d_total', 'under_limit_but_order_over', 'same_shipment_7d',
    'round_thousands', 'cents_zero', 'is_freight', 'grand_total',
    'duty_rate_off', 'fee_over_contract', 'fee_amended', 'is_customs',
)


def features(data):
    """One row per invoice: its number, issue date, and COLUMNS."""
    inv = data.invoices
    out = pd.DataFrame({'number': inv['number'], 'issue_date': inv['issue_date']})
    out = out.join(_contract(data))
    out = out.join(_vendor_master(data))
    out = out.join(_history(data))
    out = out.join(_customs(data))
    total = inv['grand_total']
    out['round_thousands'] = ((total % 1000 == 0) & (total > 0)).astype(int)
    out['cents_zero'] = (total.round(2) % 1 == 0).astype(int)
    out['is_freight'] = (inv['kind'] == 'freight').astype(int)
    out['is_customs'] = inv['kind'].isin(('customs', 'broker')).astype(int)
    out['grand_total'] = total
    return out[['number', 'issue_date', *COLUMNS]]


# ------------------------------------------------------------- the customs


def _customs(data):
    """The duty statement against the tariff table, and the broker's fees
    against the contract. A rate cut by a published resolution is in the
    table from its date, so a statement that applies it reads clean. A
    fee raised by an amended contract says so on the line."""
    inv, lines, erp = data.invoices, data.lines, data.erp
    tariffs = erp.get('tariffs', pd.DataFrame())
    rates = {}
    if len(tariffs):
        for row in tariffs.itertuples(index=False):
            rates.setdefault(row.ncm, []).append(
                (row.effective_on, float(row.ii)))
    card = erp.get('rate_card', pd.DataFrame())
    fees = {}
    if len(card):
        fixed = card[card['kind'] == 'fixed_fee']
        fees = {(i, d): float(a) for i, d, a in
                zip(fixed['issuer'], fixed['description'], fixed['amount'])
                if _is_number(a)}
    by_number = _index(lines, ('number',), ('description', 'unit_price'))

    rows = []
    for invoice in inv.itertuples(index=False):
        mine = by_number.get((invoice.number,), [])
        duty = fee = amended = 0.0
        if invoice.kind == 'customs':
            for description, rate in mine:
                tax, _, ncm = description.partition(' ')
                if tax != 'II' or ncm not in rates:
                    continue
                in_force = [(day, r) for day, r in rates[ncm]
                            if day <= invoice.issue_date]
                if in_force:
                    duty = max(duty, abs(rate - max(in_force)[1]))
        if invoice.kind == 'broker':
            for description, price in mine:
                base = description.replace(' (amended)', '')
                agreed = fees.get((invoice.issuer, base))
                if not agreed:
                    continue
                if base != description:
                    amended = 1.0
                fee = max(fee, price / agreed - 1)
        rows.append({'duty_rate_off': round(duty, 6),
                     'fee_over_contract': round(max(fee, 0.0), 6),
                     'fee_amended': amended})
    return pd.DataFrame(rows, index=inv.index)


def _is_number(text):
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------ the contract


def _contract(data):
    """Three-way match. Records are indexed once, then looked up per line;
    filtering whole tables per invoice took a minute for a year."""
    inv, erp = data.invoices, data.erp
    card = erp.get('rate_card', pd.DataFrame())
    known = set(card['description']) if len(card) else set()
    fixed = (set(zip(card.loc[card['kind'] == 'fixed_fee', 'issuer'],
                     card.loc[card['kind'] == 'fixed_fee', 'description']))
             if len(card) else set())
    terms = _index(erp.get('contracts'), ('order_number', 'code'),
                   ('effective_on', 'version', 'price'))
    signed = _index(erp.get('receipts'), ('order_number', 'shipment_id'),
                    ('code', 'quantity', 'received_on'))
    lines = _index(data.lines, ('number',),
                   ('code', 'description', 'quantity', 'unit_price'))

    rows = []
    for invoice in inv.itertuples(index=False):
        mine = lines.get((invoice.number,), [])
        review = invoice.due_date
        rows.append({
            'price_over_contract': _price_over(invoice, mine, terms, review),
            **_quantity_over(invoice, mine, signed, review),
            'unknown_charges': (sum(1 for _, d, _, _ in mine
                                    if not d.startswith(MAIN_FREIGHT) and d not in known)
                                if invoice.kind == 'freight' else 0),
            'known_fixed_fee': int(any((invoice.issuer, d) in fixed
                                       for _, d, _, _ in mine)),
        })
    return pd.DataFrame(rows, index=inv.index)


def _index(frame, keys, values):
    """{key tuple: [value tuples]} for a table, empty when it is missing."""
    found = {}
    if frame is None or not len(frame):
        return found
    for row in frame[list(keys) + list(values)].itertuples(index=False):
        found.setdefault(tuple(row[:len(keys)]), []).append(tuple(row[len(keys):]))
    return found


def _price_over(invoice, mine, terms, review):
    if not invoice.order_number:
        return 0.0
    worst = 0.0
    for code, _, _, price in mine:
        if not code:
            continue
        valid = [t for t in terms.get((invoice.order_number, code), [])
                 if t[0] <= review]
        if not valid:
            continue
        agreed = max(valid, key=lambda t: (t[0], str(t[1])))[2]
        if agreed:
            worst = max(worst, price / agreed - 1)
    return worst


def _quantity_over(invoice, mine, signed, review):
    unknown = {'quantity_over_receipt': np.nan, 'receipt_known': 0}
    if not invoice.order_number:
        return unknown
    got = {}
    for code, quantity, received in signed.get(
            (invoice.order_number, invoice.shipment_id), []):
        if received <= review:
            got[code] = got.get(code, 0) + quantity
    coded = [(code, quantity) for code, _, quantity, _ in mine if code]
    if not got or not coded:
        return unknown
    worst = 0.0
    for code, quantity in coded:
        if got.get(code):
            worst = max(worst, quantity / got[code] - 1)
    return {'quantity_over_receipt': worst, 'receipt_known': 1}


# ------------------------------------------------------- the vendor master


def _vendor_master(data):
    inv, erp = data.invoices, data.erp
    vendors = {v.code: v for v in erp['vendors'].itertuples(index=False)}
    changes = _index(erp.get('vendor_changes'), ('code',),
                     ('changed_on', 'new_value'))
    rows = []
    for invoice in inv.itertuples(index=False):
        vendor = vendors.get(invoice.issuer)
        accounts = {vendor.bank_account} if vendor is not None else set()
        accounts |= {new for day, new in changes.get((invoice.issuer,), [])
                     if day <= invoice.due_date}
        onboarded = vendor.onboarded_on if vendor is not None else invoice.issue_date
        rows.append({
            'bank_not_on_file': int(bool(invoice.pay_to) and _plain(invoice.pay_to)
                                    not in {_plain(a) for a in accounts}),
            'sender_not_on_file': int(bool(invoice.sender) and (
                vendor is None or invoice.sender != vendor.email)),
            'vendor_age_days': (invoice.issue_date - onboarded).days,
            'has_order': int(bool(invoice.order_number)),
        })
    return pd.DataFrame(rows, index=inv.index)

def _plain(text):
    return ' '.join(str(text).split())


# ------------------------------------------------------- the issuer's past


def _history(data):
    inv = data.invoices
    limit = _limit(data)
    rows = {}
    for _, group in inv.groupby('issuer'):
        # A stable sort keeps same-day invoices in arrival order, so
        # everything already seen arrived before this one.
        group = group.sort_values('issue_date', kind='stable')
        seen = []
        for index, invoice in group.iterrows():
            day, total = invoice['issue_date'], invoice['grand_total']
            before = seen
            month = [s for s in before if (day - s['issue_date']).days <= 30]
            week_order = [s for s in before if invoice['order_number']
                          and s['order_number'] == invoice['order_number']
                          and (day - s['issue_date']).days <= 7]
            past = [s['grand_total'] for s in before]
            spread = np.std(past) if len(past) >= 3 else 0
            order_total = total + sum(s['grand_total'] for s in week_order)
            rows[index] = {
                'same_amount_30d': int(any(abs(s['grand_total'] - total) < 0.005
                                           for s in month)),
                'days_since_last': ((day - before[-1]['issue_date']).days
                                    if before else np.nan),
                'invoices_30d': len(month),
                'amount_zscore': ((total - np.mean(past)) / spread
                                  if spread else 0.0),
                'same_order_7d_total': order_total,
                'under_limit_but_order_over': int(bool(week_order)
                                                  and total < limit < order_total),
                # A split cuts one delivery into pieces; a partial delivery
                # bills each box on its own shipment.
                'same_shipment_7d': sum(1 for s in week_order
                                        if invoice['shipment_id']
                                        and s['shipment_id'] == invoice['shipment_id']),
            }
            seen.append({'issue_date': day, 'grand_total': total,
                         'order_number': invoice['order_number'],
                         'shipment_id': invoice['shipment_id']})
    return pd.DataFrame.from_dict(rows, orient='index').reindex(inv.index)


def _limit(data):
    policy = data.erp.get('approval_policy', pd.DataFrame())
    numbers = pd.to_numeric(policy.get('limit', pd.Series(dtype=float)),
                            errors='coerce').dropna()
    return float(numbers.min()) if len(numbers) else 25000.0
