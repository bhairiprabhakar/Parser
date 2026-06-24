import re
from typing import List, Dict, Any, Optional
import logging
logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Evaluates OCR output confidence using multiple heuristics."""

    def __init__(self):
        self.min_char_confidence = 0.3
        self.min_word_confidence = 0.4
        self.min_line_confidence = 0.5

    def score_text(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        text = text.strip()
        score = 1.0
        char_count = len(text)
        digit_count = sum(1 for c in text if c.isdigit())
        alpha_count = sum(1 for c in text if c.isalpha())
        if char_count == 0:
            return 0.0
        if alpha_count == 0 and digit_count > 0:
            score *= 0.3
        if digit_count / max(char_count, 1) > 0.8:
            score *= 0.5
        weird_set = "@#$%^&*_=+[]{}|;:',.?!"
        weird_chars = sum(1 for c in text if c in weird_set)
        weird_ratio = weird_chars / max(char_count, 1)
        if weird_ratio > 0.3:
            score *= 0.4
        upper_ratio = sum(1 for c in text if c.isupper()) / max(alpha_count, 1)
        if upper_ratio > 0.9 and alpha_count > 3:
            score *= 0.8
        if re.search(r'(.)\1{3,}', text):
            score *= 0.3
        avg_word_len = char_count / max(len(text.split()), 1)
        if avg_word_len < 2 and len(text.split()) > 1:
            score *= 0.5
        elif avg_word_len > 30:
            score *= 0.6
        return max(0.0, min(1.0, score))

    def score_batch(self, texts: List[str]) -> List[float]:
        return [self.score_text(t) for t in texts]

    def get_quality_flags(self, text: str) -> List[str]:
        flags = []
        score = self.score_text(text)
        if score < 0.3:
            flags.append("VERY_LOW_CONFIDENCE")
        elif score < 0.5:
            flags.append("LOW_CONFIDENCE")
        if score > 0.9:
            flags.append("HIGH_CONFIDENCE")
        if re.search(r'(.)\1{4,}', text):
            flags.append("REPETITIVE_CHARS")
        words = text.split()
        if len(words) > 0:
            avg_len = sum(len(w) for w in words) / len(words)
            if avg_len > 25:
                flags.append("UNUSUALLY_LONG_WORDS")
        return flags
