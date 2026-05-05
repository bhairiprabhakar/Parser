"""
schema_engine.py — Generic Column-Schema Inference Engine
==========================================================
Works on ANY tabular Marg ERP / pharma distribution document.
No new parser needed when a new document format arrives —
just add synonyms to ROLE_SYNONYMS if a new column name appears.
"""

import re

# 🚀 INJECTED PHASE 3 NLP CLEANERS
try:
    from transformers.cleaners import clean_number
    from transformers.entity_scrubber import parse_store_and_location
except ImportError:
    # Safe fallback if run completely standalone
    def clean_number(v, is_int=False): return float(re.sub(r'[^\d.\-]', '', str(v)) or '0') if not is_int else int(float(re.sub(r'[^\d.\-]', '', str(v)) or '0'))
    def parse_store_and_location(n): return n.strip(), ""


# ══════════════════════════════════════════════════════════════════════════
#  SEMANTIC ROLE REGISTRY  — add synonyms here, never write a new parser
# ══════════════════════════════════════════════════════════════════════════
ROLE_SYNONYMS = {
    "store": [
        "PARTY NAME", "STORE NAME", "STORENAME", "CUSTOMER NAME", "CUSTOMER",
        "CLIENT NAME", "PARTY", "M/S", "RETAILER", "CHEMIST", "PHARMACIST",
        "DEALER", "BUYER", "CONSIGNEE", "SHOP NAME", "FIRM NAME",
    ],
    "amount": [
        "BILL AMT", "BILL AMT.", "BILL AMOUNT", "AMOUNT", "NET AMOUNT",
        "NET SALE", "NET SALES", "SALE AMOUNT", "VALUE", "TOTAL",
        "NET VALUE", "SALE VALUE", "INVOICE AMOUNT", "INV AMT",
    ],
    "date": [
        "DATE", "BILL DATE", "INVOICE DATE", "INV DATE", "SALE DATE",
        "TRANS DATE", "VOUCHER DATE", "TXN DATE",
    ],
    "bill_no": [
        "BILL NO", "BILL NO.", "BILL NUMBER", "INVOICE NO", "INVOICE NO.",
        "INV NO", "VOUCHER NO", "DOC NO", "REF NO",
    ],
    "item": [
        "ITEM", "ITEM NAME", "PRODUCT", "PRODUCT NAME", "DESCRIPTION",
        "DESC", "BRAND", "BRAND NAME", "DRUG NAME", "MEDICINE",
    ],
    "qty":      ["QTY", "QUANTITY", "UNITS", "PCS", "PIECES", "NOS"],
    "free":     ["FREE", "FREE QTY", "FREE UNITS", "BONUS"],
    "rate":     ["RATE", "SALE PR", "SALE PRICE", "PRICE", "MRP", "UNIT PRICE"],
    "taxable":  ["TAXABLE", "TAXABLE AMT", "TAXABLE AMOUNT", "TAXABLE VALUE", "BASIC"],
    "tax":      ["TAX", "GST", "CGST", "SGST", "IGST", "TAX AMT", "TAX AMOUNT"],
    "sur_tax":  ["SUR TAX", "SUR. TAX", "SURTAX", "ADDITIONAL TAX", "CESS"],
    "location": ["PLACE", "LOCATION", "CITY", "AREA", "ADDRESS", "STATION", "DISTRICT"],
    "exempted": ["EXEMPTED", "EXEMPT", "NIL RATED", "NON TAXABLE"],
    "round_off":["R.OFF", "ROUND OFF", "ROUND", "ROUNDING"],
    "hsn":      ["HSN", "HSN CODE", "HSN NO", "SAC CODE"],
    "batch":    ["BATCH", "BATCH NO", "BATCH NO.", "LOT NO", "MFG BATCH"],
    "expiry":   ["EXP", "EXP DATE", "EXPIRY", "EXPIRY DATE", "EXPIRE"],
    "discount": ["DISC", "DISCOUNT", "DISC AMT", "DISC %", "REBATE", "CD"],
    "serial":   ["SL", "SR", "SR NO", "SR.", "S.NO", "SL NO", "NO."],
}

_ROLE_LOOKUP = {}
for _role, _syns in ROLE_SYNONYMS.items():
    for _s in _syns:
        _ROLE_LOOKUP[re.sub(r'[^A-Z0-9]', '', _s.upper())] = _role

_MULTIWORD_COLS = sorted(
    [s for syns in ROLE_SYNONYMS.values() for s in syns if ' ' in s],
    key=len, reverse=True
)

def _role_of(text: str) -> str:
    key = re.sub(r'[^A-Z0-9]', '', text.upper())
    if key in _ROLE_LOOKUP:
        return _ROLE_LOOKUP[key]
    for k, r in _ROLE_LOOKUP.items():
        if k and (k in key or key in k) and len(key) >= 2:
            return r
    return 'unknown'

def detect_column_schema(header_line: str) -> list:
    line = header_line
    claimed = [False] * (len(line) + 1)
    cols = []

    for mw in _MULTIWORD_COLS:
        pattern = re.compile(r'\b' + re.escape(mw) + r'\b', re.IGNORECASE)
        m = pattern.search(line)
        if m and not any(claimed[m.start():m.end()]):
            for i in range(m.start(), m.end()): claimed[i] = True
            cols.append((m.start(), m.end(), mw.upper()))

    for m in re.finditer(r'\S+', line):
        if not any(claimed[m.start():m.end()]):
            cols.append((m.start(), m.end(), m.group().upper()))

    cols.sort(key=lambda c: c[0])
    schema = []
    for i, (x_start, x_end, name) in enumerate(cols):
        zone_end = cols[i + 1][0] if i + 1 < len(cols) else 9999
        role = _role_of(name)
        if role == 'serial': continue
        schema.append({'col_name': name, 'role': role, 'x_start': x_start, 'x_end': zone_end})
    return schema

def parse_row_by_schema(data_line: str, schema: list) -> dict:
    result = {}
    for col in schema:
        raw = data_line[col['x_start']:col['x_end']].strip()
        role = col['role']
        if not raw or role == 'unknown': continue

        numeric_roles = {'amount','qty','free','rate','taxable','tax','sur_tax','exempted','round_off','discount'}
        if role in numeric_roles:
            # 🚀 UPGRADED to use the V1 math sanitizer
            result[role] = clean_number(raw)
        else:
            result[role] = raw
    return result

def _is_separator(line: str) -> bool:
    return bool(re.match(r'^[\-\*=_\s]+$', line.strip()))

def _is_header_line(line: str) -> bool:
    s = line.strip()
    if not s: return False
    if re.search(r'[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]', s.upper()): return False
    if re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', s): return False
    if re.match(r'^[-*=_\s]+$', s): return False
    if re.search(r'\d+\.\d{2}', s): return False
    clean = re.sub(r'[^A-Z0-9]', '', s.upper())
    hits = sum(1 for k in _ROLE_LOOKUP if k and k in clean and len(k) >= 2)
    words = max(len(re.findall(r'\S+', s)), 1)
    return hits >= 3 and (hits / words) >= 0.55

def _is_data_row(line: str) -> bool:
    return bool(re.search(r'\d+\.\d{2}', line))

def _extract_agency_header(lines: list) -> tuple:
    ag = {'Name': '', 'GSTIN': '', 'Phone': '', 'Email': '', 'Address': ''}
    rd = {'FromDate': '', 'ToDate': '', 'DetectedFormat': 'SCHEMA_INFERRED', 'ParserMode': 'SCHEMA_INFERRED'}
    addr = []
    for line in lines:
        s = line.strip()
        if not s or _is_separator(s): continue
        gm = re.search(r'\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b', s)
        if gm: ag['GSTIN'] = gm.group(1)
        dates = re.findall(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', s)
        if len(dates) >= 2 and not rd['FromDate']: rd['FromDate'], rd['ToDate'] = dates[0], dates[-1]
        pm = re.search(r'Phone\s*[:#]?\s*([\d,\s]+)', s, re.IGNORECASE)
        em = re.search(r'E-?Mail\s*[:#]?\s*([\w.\-+]+@[\w.\-]+)', s, re.IGNORECASE)
        if pm: ag['Phone'] = pm.group(1).strip()
        if em: ag['Email'] = em.group(1).strip()
        if not ag['GSTIN']: addr.append(s)

    for l in addr:
        if len(l) > 4 and not re.match(r'^\d', l):
            ag['Name'] = l
            break
    ag['Address'] = ', '.join(addr)
    return ag, rd

_NUMERIC_ROLES = {'amount','taxable','tax','sur_tax','free_amt','exempted','round_off','qty','free','rate','discount'}

def _peel_numerics(line: str, count: int) -> tuple:
    tokens = line.split()
    nums, text = [], list(tokens)
    for token in reversed(tokens):
        if re.match(r'^-?[\d,]+(\.[\d]{1,2})?$', token):
            nums.insert(0, clean_number(token))
            text.pop()
            if len(nums) == count: break
        else: break
    return ' '.join(text), nums

def parse_row_hybrid(line: str, schema: list) -> dict:
    num_roles = [c['role'] for c in schema if c['role'] in _NUMERIC_ROLES]
    left, nums = _peel_numerics(line.strip(), len(num_roles))
    result = {}
    for i, role in enumerate(num_roles):
        if i < len(nums): result[role] = nums[i]
    left_s = left.strip()
    m = re.match(r'^(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+([A-Z]{1,4}\d{3,})\s+(.+)$', left_s, re.IGNORECASE)
    if m:
        result['date'], result['bill_no'] = m.group(1), m.group(2)
        result['store'] = re.sub(r'\s{2,}', ' ', m.group(3)).strip()
    else:
        m2 = re.match(r'^(\d+)\s+(.+)$', left_s)
        result['store'] = re.sub(r'\s{2,}', ' ', m2.group(2)).strip() if m2 else left_s
    return result

def parse_by_schema_inference(raw_text: str, source_file: str = '') -> dict:
    lines = raw_text.replace('\x0c', '\n').split('\n')
    header_idx, header_line, preamble = None, '', []
    for i, line in enumerate(lines):
        if _is_header_line(line):
            header_idx, header_line, preamble = i, line, lines[:i]
            break

    if header_idx is None:
        return {'AgencyDetails': {}, 'ReportDetails': {'DetectedFormat': 'UNKNOWN', 'ParserMode': 'SCHEMA_INFERRED', 'ColumnsDetected': []}, 'Areas': []}

    schema = detect_column_schema(header_line)
    ag, rd = _extract_agency_header(preamble)
    rd['ColumnsDetected'] = [f"{c['col_name']}→{c['role']}" for c in schema]
    area = {'AreaName': 'UNKNOWN AREA', 'Stores': []}
    stores = []
    schema_roles = {col['role'] for col in schema}
    has_store_col = 'store' in schema_roles
    has_item_col = 'item' in schema_roles
    current_store_name = ag.get('Name', 'UNKNOWN STORE')
    current_bill_no = ''
    skip_keywords = {'GRAND TOTAL', 'TOTAL :', 'TOTAL', 'NET TOTAL', 'PAGE TOTAL', '*** END', 'SUB TOTAL', 'PAGE SUB'}

    for line in lines[header_idx + 1:]:
        if not line.strip() or _is_separator(line): continue
        if any(kw in line.upper().strip() for kw in skip_keywords): continue

        if not has_store_col and has_item_col and not _is_data_row(line):
            s = line.strip()
            if (len(s) >= 4 and not re.match(r'^\d{1,2}[-/]\d{1,2}', s) and not _is_header_line(s) and re.search(r'[A-Za-z]{3,}', s)):
                s = re.sub(r'(?i)^M/S\.?\s*', '', s).strip()
                if s: current_store_name = s.upper()
            continue

        if not _is_data_row(line): continue
        row = parse_row_hybrid(line, schema) if has_store_col else parse_row_by_schema(line, schema)
        if not row or row.get('amount', 0.0) == 0.0: continue

        if has_store_col:
            raw_store_name = str(row.get('store', '')).strip()
            if not raw_store_name: continue
        else:
            bill_no = str(row.get('bill_no', '')).strip()
            if bill_no and bill_no != current_bill_no: current_bill_no = bill_no
            raw_store_name = current_store_name

        # 🚀 UPGRADED to use the V1 Entity NLP Scrubber to remove PVT LTD / Brands!
        s_name, s_loc = parse_store_and_location(raw_store_name)
        if not s_loc and has_store_col:
            s_loc = str(row.get('location', ''))

        stores.append({
            'StoreName':     s_name.upper(),
            'StoreLocation': s_loc,
            'Items': [{
                'Description': str(row.get('item', 'BILL AMOUNT')),
                'Qty':         row.get('qty', 0),
                'Free':        row.get('free', 0),
                'Rate':        row.get('rate', 0.0),
                'Amount':      row.get('amount', 0.0),
                'Percent':     row.get('discount', 0.0),
                'Bill_Date':   str(row.get('date', '')),
                'Bill_No':     str(row.get('bill_no', '')),
                'Taxable':     row.get('taxable', ''),
                'Tax':         row.get('tax', ''),
                'Sur_Tax':     row.get('sur_tax', ''),
                'Free_Amt':    row.get('free', ''),
                'Exempted':    row.get('exempted', ''),
                'Round_Off':   row.get('round_off', ''),
                'Batch':       str(row.get('batch', '')),
                'HSN':         str(row.get('hsn', '')),
                'Expiry':      str(row.get('expiry', '')),
            }]
        })

    area['Stores'] = stores
    return {'AgencyDetails': ag, 'ReportDetails': rd, 'Areas': [area]}