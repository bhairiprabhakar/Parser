import re
from typing import List, Dict, Any, Optional, Tuple
import logging
logger = logging.getLogger(__name__)


class MultiPageContinuityPass:
    """Reconciles text across page boundaries for multi-page documents."""

    def __init__(self):
        self.split_word_pattern = re.compile(r'(\w+)-$')
        self.page_number_pattern = re.compile(r'^-?\d+\s*-?\s*$')
        self.header_footer_patterns = [
            re.compile(r'page\s+\d+', re.IGNORECASE),
            re.compile(r'^\s*\d+\s*$'),
            re.compile(r'continued\.\.\.?', re.IGNORECASE),
            re.compile(r'contd\.?', re.IGNORECASE),
        ]

    def process_pages(self, pages_text: List[str]) -> str:
        if not pages_text:
            return ""
        if len(pages_text) == 1:
            return pages_text[0]
        cleaned_pages = []
        for i, text in enumerate(pages_text):
            lines = text.split('\n')
            filtered = self._filter_page_artifacts(lines, i, len(pages_text))
            cleaned_pages.append('\n'.join(filtered))
        merged = self._merge_across_pages(cleaned_pages)
        return merged

    def _filter_page_artifacts(self, lines: List[str], page_idx: int,
                               total_pages: int) -> List[str]:
        filtered = []
        for line in lines:
            if self.page_number_pattern.match(line.strip()):
                continue
            if any(p.match(line.strip()) for p in self.header_footer_patterns):
                continue
            if not line.strip():
                continue
            filtered.append(line)
        return filtered

    def _merge_across_pages(self, pages: List[str]) -> str:
        merged = []
        for i, page_text in enumerate(pages):
            lines = page_text.split('\n')
            if i > 0 and lines:
                first_line = lines[0]
                match = self.split_word_pattern.match(first_line)
                if match:
                    prefix = match.group(1)
                    if merged:
                        last_line = merged[-1]
                        merged[-1] = last_line + prefix
                    lines = lines[1:]
            merged.extend(lines)
        return '\n'.join(merged)

    def detect_split_rows(self, text: str) -> List[Tuple[str, str]]:
        lines = text.split('\n')
        splits = []
        for i in range(len(lines) - 1):
            current = lines[i].strip()
            next_line = lines[i + 1].strip()
            if not current or not next_line:
                continue
            if current.endswith('-') or current.endswith(','):
                splits.append((current, next_line))
            current_parts = current.split()
            next_parts = next_line.split()
            if (len(current_parts) >= 3 and len(next_parts) >= 2 and
                    current_parts[-1].replace('.', '').isdigit() and
                    next_parts[0].replace('.', '').isdigit()):
                splits.append((current, next_line))
        return splits

    def reconstruct_table_rows(self, text: str,
                               max_line_distance: int = 3) -> List[str]:
        lines = text.split('\n')
        if not lines:
            return []
        rows = []
        i = 0
        while i < len(lines):
            current = lines[i].strip()
            if not current:
                i += 1
                continue
            parts = current.split()
            if len(parts) < 4:
                rows.append(current)
                i += 1
                continue
            combined = current
            j = 1
            while (i + j < len(lines) and j <= max_line_distance):
                next_line = lines[i + j].strip()
                if not next_line:
                    j += 1
                    continue
                next_parts = next_line.split()
                if (len(parts) >= 4 and len(next_parts) >= 2 and
                        len(next_parts) < len(parts) and
                        any(p.replace('.', '').isdigit() for p in next_parts[:2])):
                    combined += ' ' + next_line
                    j += 1
                else:
                    break
            rows.append(combined)
            i += j
        return rows
