import re
import csv
import logging
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None


class PageDewarper:
    """Corrects page curl/warp in scanned documents."""

    def __init__(self):
        pass

    def dewarp(self, image: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=100, maxLineGap=10)
        if lines is None or len(lines) < 5:
            return image
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        median_angle = np.median(angles)
        if abs(median_angle) > 2:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        return image





def _detect_format_with_regex(text: str) -> str:
    """Detect format type from text content."""
    patterns = {
        "LUPIN": [
            r'\bLUPIN\b', r'\bLUP\s*\d{4}\b',
            r'Lupin\s+(Park|Campus|Center|House)',
        ],
        "RELIABO": [
            r'\bRELIABO\b', r'\breliabo\b',
            r'Reliabo\s+Pharma(?:ceuticals)?',
        ],
        "APEX": [
            r'\bAPEX\b', r'\bApex\b',
            r'Apex\s+(Laboratories|Pharma|Drugs)',
        ],
        "SKY": [
            r'\bSKY\b', r'\bSky\b',
            r'Sky\s+(Pharma|Health|Life)',
        ],
        "ENCURE": [
            r'\bENCURE\b', r'\bEncure\b',
            r'Encure\s+Pharmaceuticals',
        ],
        "MARG": [
            r'\bMARG\b', r'\bMarg\b',
            r'Marg\s+(ERP|Shop|Retail)',
            r'Party\s+Name\s+Product\s+Qty',
            r'Area\s+Party',
        ],
        "GENERIC": [
            r'(?:TAX\s+)?INVOICE',
            r'Bill\s+No',
            r'GST\s+(?:IN|No)',
        ]
    }
    for fmt, pats in patterns.items():
        for pat in pats:
            if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
                return fmt
    return "UNKNOWN"


def enhanced_write_csv(csv_path: str, rows: List[Dict], fieldnames: List[str],
                       qa_report_path: Optional[str] = None):
    """Write CSV with QA flags column."""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            qa_flags = row.get('qa_flags', [])
            if isinstance(qa_flags, list):
                qa_flags = '; '.join(qa_flags)
            row['QA Flags'] = qa_flags if qa_flags else ''
            writer.writerow(row)
    if qa_report_path:
        flagged = [r for r in rows if r.get('qa_flags')]
        if flagged:
            with open(qa_report_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction='ignore')
                writer.writeheader()
                for row in flagged:
                    row['QA Flags'] = ('; '.join(row['qa_flags'])
                                       if isinstance(row.get('qa_flags'), list)
                                       else row.get('qa_flags', ''))
                    writer.writerow(row)
