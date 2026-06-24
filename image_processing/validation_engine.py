import re
from typing import List, Dict, Any, Optional, Tuple
import logging
logger = logging.getLogger(__name__)


class ValidationEngine:
    _TOTAL_KEYWORDS = frozenset([
        "grand total", "total value", "net sales", "value in rs",
        "net amount", "total:", "total :", "invoice value", "net payable",
    ])

    def find_grand_total(self, text: str) -> float:
        best_decimal = 0.0
        best_integer = 0.0
        lines = text.split("\n")

        def _candidates(ln: str):
            decimals = re.findall(r'\d[\d,]*\.\d{1,2}', ln.replace(",", ""))
            integers = re.findall(r'\b\d{4,9}\b', ln)
            results  = []
            for d in decimals:
                try:
                    v = float(d)
                    if 100 <= v < 1_000_000:
                        results.append(("decimal", v))
                except ValueError:
                    pass
            for i in integers:
                try:
                    v = float(i)
                    if 100 <= v < 1_000_000:
                        results.append(("integer", v))
                except ValueError:
                    pass
            return results

        for i, line in enumerate(lines):
            ll = line.lower()
            if not any(kw in ll for kw in self._TOTAL_KEYWORDS):
                continue

            search_lines = [line]
            if i > 0:
                search_lines.append(lines[i - 1])
            if i < len(lines) - 1:
                search_lines.append(lines[i + 1])

            for sl in search_lines:
                for kind, val in _candidates(sl):
                    if kind == "decimal":
                        if val > best_decimal:
                            best_decimal = val
                    else:
                        if val > best_integer:
                            best_integer = val

        return best_decimal if best_decimal >= 100 else best_integer

    def validate(self, extracted_total: float, grand_total: float,
                 tolerance: float = 1.0) -> bool:
        if grand_total <= 0:
            return True
        diff = abs(extracted_total - grand_total)
        ok   = diff < tolerance
        if ok:
            logger.info("Validation passed diff=%.2f", diff)
        else:
            logger.warning("Validation gap extracted=%.2f grand=%.2f diff=%.2f",
                           extracted_total, grand_total, diff)
        return ok

    def normalize_ocr_text(self, text: str) -> str:
        corrected = []
        for word in text.split():
            if word[0].isupper() and len(word) > 1 and word[1:].islower():
                corrected.append(word)
                continue
            cleaned = ''.join({'O': '0', 'l': '1', 'I': '1', 'S': '5',
                               'B': '8', 'Z': '2', 's': '5', 'g': '9'}.get(c, c)
                              for c in word)
            corrected.append(cleaned)
        return ' '.join(corrected)
