import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import logging
logger = logging.getLogger(__name__)


class ValidationEngine:
    """Post-OCR validation and correction engine."""

    def __init__(self):
        self.common_corrections = {
            'O': '0', 'l': '1', 'I': '1', 'S': '5',
            'B': '8', 'Z': '2', 's': '5', 'g': '9',
        }
        self.business_terms = {
            'Qty': ['Qty', 'Quantity', 'QTY', 'qty'],
            'Rate': ['Rate', 'rate', 'RATE', 'Rs', 'RS', 'rs'],
            'Amount': ['Amount', 'amount', 'AMOUNT', 'Amt', 'amt'],
            'Free': ['Free', 'free', 'FREE'],
            'Disc': ['Disc', 'disc', 'DISCOUNT', 'Discount', 'discount'],
            'GST': ['GST', 'gst', 'CGST', 'SGST', 'IGST'],
            'Total': ['Total', 'total', 'TOTAL', 'Grand Total', 'GRAND TOTAL'],
            'Party': ['Party', 'party', 'PARTY', 'Name', 'NAME'],
        }

    def find_grand_total(self, lines: List[str]) -> Optional[float]:
        total = None
        for line in lines:
            match = re.search(
                r'(?:grand\s*total|total|net\s*amount|net\s*value)'
                r'\s*:?\s*([\d,]+\.?\d*)',
                line, re.IGNORECASE
            )
            if match:
                val_str = match.group(1).replace(',', '')
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                if val > 999999:
                    continue
                if '.' in val_str:
                    total = val
                elif total is None:
                    total = val
        return total

    def normalize_ocr_text(self, text: str) -> str:
        corrected = []
        for word in text.split():
            if word[0].isupper() and len(word) > 1 and word[1:].islower():
                corrected.append(word)
                continue
            cleaned = ''.join(self.common_corrections.get(c, c)
                              for c in word)
            corrected.append(cleaned)
        return ' '.join(corrected)

    def validate_line_structure(self, line: str,
                                expected_parts: int = 8) -> Dict[str, Any]:
        parts = line.split()
        result = {
            'original': line,
            'parts_count': len(parts),
            'is_valid': len(parts) >= expected_parts * 0.5,
            'issues': []
        }
        if not line.strip():
            result['issues'].append('EMPTY_LINE')
        elif len(parts) < 3:
            result['issues'].append('TOO_FEW_PARTS')
        digit_count = sum(1 for p in parts if re.match(r'^[\d,.]+$', p))
        alpha_count = sum(1 for p in parts if re.match(r'^[a-zA-Z]+$', p))
        if digit_count > 0 and alpha_count == 0:
            result['issues'].append('ALL_DIGITS_NO_ALPHA')
        elif alpha_count > 0 and digit_count == 0:
            result['issues'].append('ALL_ALPHA_NO_DIGITS')
        return result

    def validate_table_data(self, rows: List[Dict]) -> List[Dict]:
        validated = []
        for row in rows:
            row['validation_flags'] = []
            if not row.get('Item Description', '').strip():
                row['validation_flags'].append('MISSING_ITEM')
            for field in ['Qty', 'Rate', 'Amount']:
                val = row.get(field, '')
                if val:
                    try:
                        num = float(str(val).replace(',', ''))
                        if num < 0:
                            row['validation_flags'].append(
                                f'NEGATIVE_{field.upper()}'
                            )
                    except ValueError:
                        row['validation_flags'].append(
                            f'INVALID_{field.upper()}'
                        )
            validated.append(row)
        return validated

    def detect_duplicate_lines(self, lines: List[str]) -> List[int]:
        line_stripped = [l.strip() for l in lines]
        duplicates = []
        seen = {}
        for i, line in enumerate(line_stripped):
            if line in seen:
                duplicates.append(i)
            else:
                seen[line] = i
        return duplicates
