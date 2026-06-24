import cv2
import numpy as np
import re
from typing import List, Tuple, Optional
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)


class SpatialClusterer:
    _MIN_CONF      = 0.45
    _MIN_TOLERANCE = 15.0

    COLUMN_ROLES = [
        "item_code", "city", "party_name", "product_name", "packing",
        "quantity", "free", "avg_rate", "amount", "company",
    ]

    _ITEM_CODE_RE   = re.compile(r'^\d{3,7}$')
    _CITY_RE        = re.compile(r'^[A-Z][A-Z ]{2,20}$', re.I)
    _PACKING_RE     = re.compile(
        r'^\d{1,4}\s*[*xX]\s*\d+$'
        r'|^\d+\s*(?:ML|L|GM|MG|MCG)$'
        r'|^\d+\s*(?:TAB|CAP|SYP|INJ|DRY)$'
        r'|^1\s*[xX]\s*\d+\s*(?:GM|ML|TAB)?$'
        r'|^\d+\s*CM\s*[xX]\s*\d+\s*MT?$',
        re.I
    )
    _QTY_FREE_RE    = re.compile(r'^\d{1,4}$')
    _RATE_RE        = re.compile(r'^\d{1,3}\.\d{2,3}$')
    _AMOUNT_RE      = re.compile(r'^\d{1,7}\.\d{2}$')
    _COMPANY_RE     = re.compile(r'^(?:LUPIN|LOPIN|LUP1N|LOP1N|LUPIM)$', re.I)

    _OCR_FIXES = [
        (re.compile(r'^[:\)](\d{4,6})$'),          r'\1'),
        (re.compile(r'^L(\d{5})$'),                 r'1\1'),
        (re.compile(r'\bSLUPIN\b', re.I),           'LUPIN'),
        (re.compile(r'\b2LUPIN\b', re.I),           'LUPIN'),
        (re.compile(r'\bLUP1N\b',  re.I),           'LUPIN'),
        (re.compile(r'\bLOP1N\b',  re.I),           'LUPIN'),
        (re.compile(r'\bLOPIN\b',  re.I),           'LUPIN'),
        (re.compile(r'\bLUPIM\b',  re.I),           'LUPIN'),
        (re.compile(r'(\d{2,3}):(\d{3})\b'),        r'\1.\2'),
        (re.compile(r'\b([A-Z])(\d{5})\b'),         r'1\2'),
    ]

    _WATERMARK_RE = re.compile(
        r'scanned?\s+with|oken\s*scan|oken\s*scanner', re.IGNORECASE)

    _HEADER_DROP_RE = re.compile(
        r'^(sale\b|party\s*&\s*product|party\s*name|product\s*name|'
        r'packing\b|item\s*code|iten\s*code|city\b|cizy\b|quantity\b|'
        r'free\b|avg\.?\s*rate|amount\s*company|grand\s*total)',
        re.I)

    def extract_blocks(self, rapid_result) -> list:
        blocks = []
        for box, text, score in rapid_result:
            score_val = float(score[0] if isinstance(score, (list, tuple)) else score)
            if score_val < self._MIN_CONF:
                continue
            text = str(text).strip()
            if not text:
                continue
            if self._WATERMARK_RE.search(text):
                continue
            for pattern, replacement in self._OCR_FIXES:
                text = pattern.sub(replacement, text)
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            y_min, y_max = min(ys), max(ys)
            x_min, x_max = min(xs), max(xs)
            h = y_max - y_min
            blocks.append({
                "text":       text,
                "center_y":   (y_min + y_max) / 2.0,
                "y_min":      y_min,  "y_max": y_max,
                "x_min":      x_min,  "x_max": x_max,
                "height":     h,
                "char_width": (x_max - x_min) / max(1, len(text)),
                "confidence": score_val,
            })
        return blocks

    def cluster_rows(self, blocks: list) -> list:
        if not blocks:
            return []
        heights   = [b["height"] for b in blocks]
        median_h  = float(np.median(heights)) if heights else 20.0
        tolerance = max(median_h * 1.0, self._MIN_TOLERANCE)
        sorted_blocks = sorted(blocks, key=lambda b: b["center_y"])
        row_anchors: list = []
        row_buckets: list = []
        for blk in sorted_blocks:
            cy = blk["center_y"]
            best_idx, best_dist = -1, float("inf")
            for i, anchor in enumerate(row_anchors):
                d = abs(cy - anchor)
                if d <= tolerance and d < best_dist:
                    best_dist, best_idx = d, i
            if best_idx >= 0:
                row_buckets[best_idx].append(blk)
            else:
                row_anchors.append(cy)
                row_buckets.append([blk])
        return [sorted(bucket, key=lambda b: b["x_min"]) for bucket in row_buckets]

    def detect_column_boundaries(self, all_blocks: list) -> list:
        if not all_blocks:
            return []
        x_centres = [(b["x_min"] + b["x_max"]) / 2.0 for b in all_blocks]
        if not x_centres:
            return []
        x_min_all = min(x_centres)
        x_max_all = max(x_centres)
        page_w    = x_max_all - x_min_all
        if page_w < 10:
            return []
        buckets   = 200
        hist      = [0] * buckets
        for xc in x_centres:
            idx = min(int((xc - x_min_all) / page_w * buckets), buckets - 1)
            hist[idx] += 1
        smoothed = [0.0] * buckets
        for i in range(buckets):
            smoothed[i] = sum(hist[max(0,i-1):i+2]) / 3.0
        threshold = max(smoothed) * 0.20
        in_gap    = False
        gap_starts = []
        for i, v in enumerate(smoothed):
            if v < threshold and not in_gap:
                in_gap = True
                gap_starts.append(i)
            elif v >= threshold:
                in_gap = False
        col_starts = sorted(set(
            x_min_all + (gs / buckets) * page_w for gs in gap_starts))
        min_sep = page_w * 0.02
        filtered = [col_starts[0]] if col_starts else []
        for x in col_starts[1:]:
            if x - filtered[-1] >= min_sep:
                filtered.append(x)
        logger.debug("Detected %d column boundaries: %s",
                     len(filtered), [f"{x:.0f}" for x in filtered[:12]])
        return filtered

    def assign_column_slot(self, block: dict, col_boundaries: list) -> int:
        if not col_boundaries:
            return 0
        tok_w    = max(block["x_max"] - block["x_min"], 1.0)
        bounds   = col_boundaries + [col_boundaries[-1] + tok_w * 3]
        best_col, best_overlap = 0, -1.0
        for i in range(len(bounds) - 1):
            col_x0, col_x1 = bounds[i], bounds[i + 1]
            overlap = (min(block["x_max"], col_x1) -
                       max(block["x_min"], col_x0))
            if overlap > best_overlap:
                best_overlap = overlap
                best_col     = i
        return best_col

    def classify_token_role(self, text: str) -> str:
        t = text.strip()
        if self._COMPANY_RE.match(t):        return "company"
        if self._AMOUNT_RE.match(t):         return "amount"
        if self._RATE_RE.match(t):           return "avg_rate"
        if self._QTY_FREE_RE.match(t) and int(t) < 10000:
            return "quantity_or_free"
        if self._PACKING_RE.match(t):        return "packing"
        if self._ITEM_CODE_RE.match(t):      return "item_code"
        return "text"

    def row_to_text(self, row: list, col_boundaries: list = None) -> str:
        if not row:
            return ""
        if col_boundaries and len(col_boundaries) >= 5:
            slots: dict = {i: [] for i in range(len(col_boundaries))}
            for blk in row:
                slot = self.assign_column_slot(blk, col_boundaries)
                slots[slot].append(blk["text"])
            parts = [" ".join(slots[i]).strip() for i in range(len(col_boundaries))]
            while parts and not parts[-1]:
                parts.pop()
            return "\t".join(parts)
        else:
            avg_cw = float(np.mean([b["char_width"] for b in row])) or 8.0
            parts: list = []
            for i, blk in enumerate(row):
                gap = (blk["x_min"] - row[i-1]["x_max"]) if i > 0 else 0.0
                if gap < 0:
                    sep = " "
                elif gap <= 1.0 * avg_cw:
                    sep = ""
                elif gap <= 3.5 * avg_cw:
                    sep = " "
                else:
                    sep = "\t"
                if parts:
                    parts.append(sep + blk["text"])
                else:
                    parts.append(blk["text"])
            line = "".join(parts)
            tokens = line.split("\t")
            if tokens:
                last = tokens[-1]
                m = re.match(r'^([\d.,]+)(LUPIN|LOPIN|LUP1N|LUPIM)$', last, re.I)
                if m:
                    tokens[-1] = m.group(1)
                    tokens.append(m.group(2))
            return "\t".join(tokens)

    def find_header_row(self, rows: list) -> int:
        header_kw = {"item", "code", "city", "party", "product",
                     "packing", "quantity", "free", "rate", "amount"}
        for i, row in enumerate(rows):
            row_text = " ".join(b["text"].lower() for b in row)
            matches  = sum(1 for kw in header_kw if kw in row_text)
            if matches >= 3:
                logger.debug("Header row found at row index %d", i)
                return i
        return -1

    def find_footer_start(self, rows: list) -> int:
        footer_kw = re.compile(r'grand\s*total|end\s*of\s*report', re.I)
        for i, row in enumerate(rows):
            row_text = " ".join(b["text"] for b in row)
            if footer_kw.search(row_text):
                logger.debug("Footer starts at row index %d", i)
                return i
        return len(rows)

    def blocks_to_text(self, blocks: list) -> str:
        rows = self.cluster_rows(blocks)
        if not rows:
            return ""
        col_boundaries = self.detect_column_boundaries(blocks)
        header_idx     = self.find_header_row(rows)
        footer_idx     = self.find_footer_start(rows)
        start_row = max(0, header_idx) if header_idx >= 0 else 0
        data_rows  = rows[start_row:footer_idx + 1]
        raw_lines = [self.row_to_text(r, col_boundaries) for r in data_rows if r]
        cleaned   = self._stitch_and_clean_text(raw_lines)
        return "\n".join(cleaned)

    _ORPHAN_RE     = re.compile(
        r'^[\d\s\t.,+*xX/:-]*(lupin|lopin|lup1n|lop1n)?[\d\s\t.,+*xX/:-]*$',
        re.I)
    _AMOUNT_RE_SC  = re.compile(r'\d[\d,]*\.\d{2}')

    def _stitch_and_clean_text(self, raw_lines: list) -> list:
        lines = [l for l in raw_lines
                 if not self._HEADER_DROP_RE.match(l.strip())]
        stitched: list = []
        i = 0
        while i < len(lines):
            cur = lines[i]
            if (i + 1 < len(lines)
                    and not self._AMOUNT_RE_SC.search(cur)
                    and self._ORPHAN_RE.match(lines[i + 1].strip())):
                stitched.append(cur.rstrip() + "\t" + lines[i + 1].strip())
                i += 2
            else:
                stitched.append(cur)
                i += 1
        cleaned: list = []
        for line in stitched:
            line = re.sub(
                r'Scanned\s+with\s+OKEN\s+Scanner?|OKEN\s+Scanner?',
                '', line, flags=re.I).strip()
            if line:
                cleaned.append(line)
        return cleaned
