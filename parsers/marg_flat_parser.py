import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging
logger = logging.getLogger(__name__)


MARG_HEADER_PATTERNS = [
    re.compile(r'Party\s+Name\s+Product\s+Qty\s+Free\s+Rate\s+Amount', re.IGNORECASE),
    re.compile(r'Area\s+Party\s+Name\s+Product', re.IGNORECASE),
    re.compile(r'Product\s+Qty\s+Free\s+Rate\s+Amount', re.IGNORECASE),
    re.compile(r'Particulars?\s+Qty\s+Rate\s+Amount', re.IGNORECASE),
]

MARG_STORE_PATTERN = re.compile(
    r'^(?:Dr\.?\s*)?'
    r'([A-Z][A-Za-z\s]+(?:Store|Medical|Pharmacy|Agency|Trading|Corporation|Distributor|Surgicals|Mart|Enterprises|Stores)?)'
    r'(?:\s*,\s*|\s{2,})'
    r'([A-Za-z\s]+?(?:Road|Street|Nagar|Colony|Market|Complex|Bazaar|Chowk|Circle)?)'
    r'(?:\s*\(?\s*([A-Za-z\s]+)\)?)?',
    re.IGNORECASE
)

MARG_PRODUCT_PATTERN = re.compile(
    r'^(.+?)\s+'
    r'(\d+[xX]?\d*\.?\d*)\s+'  # Qty (e.g., 10, 5x2)
    r'(\d+\.?\d*)\s+'           # Free
    r'(\d+\.?\d*)\s+'           # Rate
    r'(\d+\.?\d*)'              # Amount
    r'(?:\s+(\d+\.?\d*))?'      # Optional Disc%
    r'(?:\s+(\d+\.?\d*))?'     # Optional Tax%
)

MARG_AMOUNT_LINE = re.compile(
    r'^(?:Total|Grand\s+Total|Net\s+Amount|Amount)\s*:?\s*'
    r'([\d,]+\.?\d*)',
    re.IGNORECASE
)


@dataclass
class MargStoreRow:
    """A single store entry within a MARG flat table."""
    area: str = ""
    store_name: str = ""
    store_location: str = ""
    product: str = ""
    qty: str = ""
    free: str = ""
    rate: str = ""
    amount: str = ""
    disc_percent: str = ""
    tax_amount: str = ""


class MargFlatParser:
    """Parses MARG ERP flat-table format: one row per store with sales data."""

    def __init__(self):
        self.current_area = ""
        self.current_store = None

    def parse(self, text: str) -> Tuple[List[MargStoreRow], List[str]]:
        lines = text.strip().split('\n')
        rows = []
        errors = []
        i = 0
        header_found = False
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if not header_found:
                for pat in MARG_HEADER_PATTERNS:
                    if pat.search(line):
                        header_found = True
                        break
                area_match = re.match(r'^Area\s*:\s*(.+)$', line, re.IGNORECASE)
                if area_match:
                    self.current_area = area_match.group(1).strip()
                    i += 1
                    continue
                i += 1
                continue
            amount_match = MARG_AMOUNT_LINE.match(line)
            if amount_match:
                i += 1
                continue
            store_match = self._match_store_line(line)
            if store_match:
                self.current_store = store_match
                rows.append(store_match)
                i += 1
                continue
            product_match = self._match_product_line(line)
            if product_match and self.current_store:
                product_match.area = self.current_store.area or self.current_area
                product_match.store_name = self.current_store.store_name
                product_match.store_location = self.current_store.store_location
                rows.append(product_match)
                i += 1
                continue
            if self.current_store:
                product_match = self._match_product_line(line)
                if product_match:
                    product_match.area = self.current_store.area or self.current_area
                    product_match.store_name = self.current_store.store_name
                    product_match.store_location = self.current_store.store_location
                    rows.append(product_match)
                    i += 1
                    continue
            i += 1
        return rows, errors

    def _match_store_line(self, line: str) -> Optional[MargStoreRow]:
        m = MARG_STORE_PATTERN.match(line)
        if m:
            store_name = m.group(1).strip()
            location = (m.group(2) or "").strip()
            area = (m.group(3) or self.current_area).strip()
            corrected = self._fix_store_name(store_name)
            return MargStoreRow(
                area=area,
                store_name=corrected,
                store_location=location
            )
        return None

    def _match_product_line(self, line: str) -> Optional[MargStoreRow]:
        m = MARG_PRODUCT_PATTERN.match(line)
        if m:
            return MargStoreRow(
                product=m.group(1).strip(),
                qty=m.group(2),
                free=m.group(3),
                rate=m.group(4),
                amount=m.group(5),
                disc_percent=m.group(6) or "",
                tax_amount=m.group(7) or ""
            )
        return None

    def _fix_store_name(self, name: str) -> str:
        name = re.sub(r'\s+', ' ', name).strip()
        corrections = {
            r'\bDR\b': 'Dr.',
            r'\bMED\b': 'Medical',
            r'\bPHAR\b': 'Pharmacy',
            r'\bPH\b': 'Pharmacy',
            r'\bAGCY\b': 'Agency',
            r'\bDIST\b': 'Distributor',
            r'\bCORP\b': 'Corporation',
            r'\bENT\b': 'Enterprises',
            r'\bPVT\b': 'Pvt.',
            r'\bLTD\b': 'Ltd.',
        }
        for pat, repl in corrections.items():
            name = re.sub(pat, repl, name)
        return name

    def extract_summary(self, rows: List[MargStoreRow]) -> Dict[str, Any]:
        summary = {
            'total_rows': len(rows),
            'unique_stores': len(set(r.store_name for r in rows if r.store_name)),
            'total_amount': 0.0,
            'total_qty': 0,
            'areas': list(set(r.area for r in rows if r.area)),
            'store_list': list(set(
                (r.store_name, r.store_location)
                for r in rows if r.store_name
            ))
        }
        total_amount = 0.0
        total_qty = 0
        for r in rows:
            try:
                total_amount += float(r.amount) if r.amount else 0.0
            except ValueError:
                pass
            try:
                total_qty += int(float(r.qty)) if r.qty else 0
            except ValueError:
                pass
        summary['total_amount'] = round(total_amount, 2)
        summary['total_qty'] = total_qty
        return summary

    def to_dict_rows(self, rows: List[MargStoreRow]) -> List[Dict[str, str]]:
        output = []
        for r in rows:
            output.append({
                'Area': r.area,
                'Store Name': r.store_name,
                'Store Location': r.store_location,
                'Item Description': r.product,
                'Qty': r.qty,
                'Free': r.free,
                'Rate': r.rate,
                'Amount': r.amount,
                'Disc%': r.disc_percent,
                'Tax Amount': r.tax_amount,
            })
        return output
