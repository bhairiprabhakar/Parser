import re
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)


@dataclass
class ERPColumnMap:
    """Maps column headers to normalized field names."""
    display_name: str
    field_name: str
    data_type: str = "string"
    required: bool = False
    aliases: List[str] = field(default_factory=list)
    position: Optional[int] = None


class ERPTemplateEngine:
    """Matches invoice columns to ERP field names using learned mappings."""

    def __init__(self, schemas_path: Optional[str] = None):
        self.schemas: Dict[str, List[ERPColumnMap]] = {}
        self.default_mappings: Dict[str, str] = {
            'qty': 'Qty', 'quantity': 'Qty',
            'rate': 'Rate', 'price': 'Rate', 'rs': 'Rate',
            'amount': 'Amount', 'total': 'Amount', 'amt': 'Amount',
            'free': 'Free',
            'disc': 'Disc%', 'discount': 'Disc%', 'disc%': 'Disc%',
            'gst': 'GST', 'cgst': 'CGST', 'sgst': 'SGST', 'igst': 'IGST',
            'item': 'Item Description', 'product': 'Item Description',
            'description': 'Item Description', 'particulars': 'Item Description',
            'brand': 'Brand Name', 'brand name': 'Brand Name',
            'dosage': 'Dosage', 'dos': 'Dosage',
            'pack': 'Packing', 'packing': 'Packing',
            'party': 'Party Name', 'customer': 'Party Name',
            'name': 'Party Name', 'store': 'Store Name',
            'area': 'Area', 'location': 'Store Location',
        }
        if schemas_path and os.path.exists(schemas_path):
            self._load_schemas(schemas_path)

    def _load_schemas(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for fmt, columns in data.items():
                cols = []
                for col_data in columns:
                    if isinstance(col_data, dict):
                        cols.append(ERPColumnMap(**col_data))
                    elif isinstance(col_data, str):
                        field = self.default_mappings.get(col_data.lower(), col_data)
                        cols.append(ERPColumnMap(
                            display_name=col_data,
                            field_name=field
                        ))
                self.schemas[fmt] = cols
        except Exception as e:
            logger.warning(f"Failed to load schemas from {path}: {e}")

    def save_schemas(self, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            serializable = {}
            for fmt, columns in self.schemas.items():
                serializable[fmt] = [
                    {
                        'display_name': c.display_name,
                        'field_name': c.field_name,
                        'data_type': c.data_type,
                        'required': c.required,
                        'aliases': c.aliases,
                        'position': c.position,
                    }
                    for c in columns
                ]
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save schemas: {e}")

    def detect_format_columns(self, headers: List[str]) -> List[ERPColumnMap]:
        mapping = []
        for h in headers:
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', h).strip().lower()
            field = self.default_mappings.get(cleaned, h)
            mapping.append(ERPColumnMap(
                display_name=h.strip(),
                field_name=field
            ))
        return mapping

    def upsert_schema(self, format_name: str, columns: List[ERPColumnMap]):
        existing = self.schemas.get(format_name, [])
        existing_names = {c.display_name for c in existing}
        for col in columns:
            if col.display_name not in existing_names:
                existing.append(col)
        self.schemas[format_name] = existing

    def get_schema(self, format_name: str) -> List[ERPColumnMap]:
        return self.schemas.get(format_name, [])

    def match_columns(self, format_name: str,
                      headers: List[str]) -> List[Optional[str]]:
        schema = self.get_schema(format_name)
        if not schema:
            mapping = self.detect_format_columns(headers)
            self.upsert_schema(format_name, mapping)
            schema = mapping
        result = []
        for h in headers:
            cleaned = h.strip().lower()
            matched = None
            for col in schema:
                if col.display_name.lower() == cleaned:
                    matched = col.field_name
                    break
                if cleaned in [a.lower() for a in col.aliases]:
                    matched = col.field_name
                    break
            if matched is None:
                field = self.default_mappings.get(cleaned, h.strip())
                result.append(field)
            else:
                result.append(matched)
        return result


class AIFormatInferenceEngine:
    """Infers document format by analyzing text patterns."""

    def __init__(self, schemas_path: Optional[str] = None,
                 min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        self.format_patterns: Dict[str, List[Dict]] = {}
        self.schemas_path = schemas_path
        if schemas_path and os.path.exists(schemas_path):
            self._load_patterns(schemas_path)

    def _load_patterns(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for fmt, columns in data.items():
                if isinstance(columns, list):
                    patterns = []
                    for col in columns:
                        if isinstance(col, dict) and 'display_name' in col:
                            patterns.append({
                                'pattern': re.escape(col['display_name']),
                                'weight': 1.0
                            })
                    self.format_patterns[fmt] = patterns
        except Exception as e:
            logger.warning(f"Failed to load format patterns: {e}")

    def infer_format(self, text: str) -> Tuple[str, float]:
        if not text:
            return "UNKNOWN", 0.0
        scores = defaultdict(float)
        for fmt, patterns in self.format_patterns.items():
            for p in patterns:
                matches = re.findall(p['pattern'], text, re.IGNORECASE)
                scores[fmt] += len(matches) * p.get('weight', 1.0)
        company_keywords = {
            "LUPIN": [r'\bLUPIN\b', r'Lupin\s+Limited', r'LUP\s*\d{4}'],
            "RELIABO": [r'\bRELIABO\b', r'Reliabo\s+Pharma'],
            "APEX": [r'\bAPEX\b', r'Apex\s+Laboratories'],
            "SKY": [r'\bSKY\b', r'Sky\s+Pharma'],
            "ENCURE": [r'\bENCURE\b', r'Encure\s+Pharma'],
            "MARG": [
                r'Party\s+Name\s+Product\s+Qty',
                r'Area\s+Party\s+Name',
                r'Product\s+Qty\s+Free\s+Rate',
            ],
        }
        for fmt, patterns in company_keywords.items():
            for p in patterns:
                if re.search(p, text, re.IGNORECASE | re.MULTILINE):
                    scores[fmt] += 5.0
        if not scores:
            return "UNKNOWN", 0.0
        best_format = max(scores, key=scores.get)
        best_score = scores[best_format]
        if best_score < self.min_confidence:
            return "UNKNOWN", best_score
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.0
        return best_format, min(1.0, confidence)

    def learn_from_example(self, format_name: str, text: str):
        if format_name not in self.format_patterns:
            self.format_patterns[format_name] = []
        lines = text.strip().split('\n')
        header_line = None
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                header_line = line
                break
        if header_line:
            words = header_line.split()
            for w in words:
                clean = re.sub(r'[^a-zA-Z0-9]', '', w)
                if clean and len(clean) > 2:
                    pattern = {
                        'pattern': re.escape(w),
                        'weight': 0.5
                    }
                    existing = [p for p in self.format_patterns[format_name]
                                if p['pattern'] == pattern['pattern']]
                    if not existing:
                        self.format_patterns[format_name].append(pattern)
        if self.schemas_path:
            try:
                with open(self.schemas_path, 'w', encoding='utf-8') as f:
                    json.dump(self.format_patterns, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to persist learned patterns: {e}")
