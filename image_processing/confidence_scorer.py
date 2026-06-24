import re
from typing import List, Dict, Any, Optional
import logging
logger = logging.getLogger(__name__)


class ConfidenceScorer:
    @staticmethod
    def score_result(rapid_result) -> float:
        if not rapid_result:
            return 0.0
        total_weight = 0.0
        weighted_sum = 0.0
        for _, text, score in rapid_result:
            conf = float(score[0] if isinstance(score, (list, tuple)) else score)
            w    = max(1, len(str(text)))
            weighted_sum  += conf * w
            total_weight  += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def score_text(text: str) -> float:
        if not text or not text.strip():
            return 0.0
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            return 0.0

        total_chars = max(len(text), 1)
        digit_count = sum(c.isdigit() for c in text)
        alpha_count = sum(c.isalpha() for c in text)
        digit_ratio = digit_count / total_chars
        tab_ratio   = sum(1 for l in lines if "\t" in l) / len(lines)
        fill_ratio  = len(lines) / max(text.count("\n") + 1, 1)

        if digit_ratio > 0.85:
            return 0.25

        if alpha_count < 5:
            return 0.10

        base = 0.4 * digit_ratio + 0.3 * tab_ratio + 0.3 * fill_ratio

        has_decimal_amounts = bool(re.search(r'\d{2,}[\.,]\d{2}', text))
        has_product_names   = alpha_count / total_chars > 0.15

        bonus = 0.0
        if has_decimal_amounts:
            bonus += 0.05
        if has_product_names:
            bonus += 0.05

        return min(base + bonus, 1.0)
