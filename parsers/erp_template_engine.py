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
    item_code:    int = -1
    area:         int = -1
    party_name:   int = -1
    product_name: int = -1
    packing:      int = -1
    qty:          int = -1
    free_qty:     int = -1
    rate:         int = -1
    amount:       int = -1
    company:      int = -1
    gstin:        int = -1
    hsn:          int = -1
    batch:        int = -1
    expiry:       int = -1
    discount:     int = -1
    tax_pct:      int = -1


@dataclass
class ERPSchema:
    erp_name:         str
    format_id:        str
    header_keywords:  list
    column_map:       ERPColumnMap = field(default_factory=ERPColumnMap)
    tab_sep:          str = "\t"
    date_formats:     list = field(
        default_factory=lambda: ["%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y"]
    )
    subtotal_re:      Optional[re.Pattern] = None
    grand_total_re:   Optional[re.Pattern] = field(
        default_factory=lambda: re.compile(
            r'(?i)\b(grand\s*total|net\s*payable|total\s*value)\b'
        )
    )
    header_end_re:    Optional[re.Pattern] = None
    notes:            str = ""


@dataclass
class FormatInferenceResult:
    format_id:     str
    erp_name:      str
    confidence:    float
    tier:          int
    column_map:    dict
    parsing_hints: dict
    source:        str


class ERPTemplateEngine:
    _instance: Optional["ERPTemplateEngine"] = None

    def __init__(self):
        self._schemas: list = []
        self._register_builtin_schemas()

    @classmethod
    def instance(cls) -> "ERPTemplateEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, schema: ERPSchema) -> None:
        self._schemas.insert(0, schema)

    def detect(self, header_lines: list) -> Optional[ERPSchema]:
        joined = " ".join(header_lines[:30]).upper()
        for schema in self._schemas:
            if all(kw.upper() in joined for kw in schema.header_keywords):
                return schema
        return None

    def extract_rows(self, text_lines: list, schema: ERPSchema,
                     skip_header_lines: int = 0) -> list:
        cm   = schema.column_map
        sep  = schema.tab_sep
        rows = []

        for raw_line in text_lines[skip_header_lines:]:
            line = raw_line.strip()
            if not line:
                continue
            if schema.grand_total_re and schema.grand_total_re.search(line):
                row_type = "grand_total"
            elif schema.subtotal_re and schema.subtotal_re.search(line):
                row_type = "subtotal"
            else:
                row_type = "data"

            cols = line.split(sep)

            def _get(idx: int) -> str:
                if idx < 0 or idx >= len(cols):
                    return ""
                return cols[idx].strip()

            row = {
                "_row_type":    row_type,
                "_schema":      schema.format_id,
                "item_code":    _get(cm.item_code),
                "area":         _get(cm.area),
                "party_name":   _get(cm.party_name),
                "product_name": _get(cm.product_name),
                "packing":      _get(cm.packing),
                "qty":          _get(cm.qty),
                "free_qty":     _get(cm.free_qty),
                "rate":         _get(cm.rate),
                "amount":       _get(cm.amount),
                "company":      _get(cm.company),
                "gstin":        _get(cm.gstin),
                "hsn":          _get(cm.hsn),
                "batch":        _get(cm.batch),
                "expiry":       _get(cm.expiry),
                "discount":     _get(cm.discount),
                "tax_pct":      _get(cm.tax_pct),
            }
            rows.append(row)
        return rows

    def _register_builtin_schemas(self) -> None:
        self.register(ERPSchema(
            erp_name="BUSY",
            format_id="BUSY_PARTY_PRODUCT_WISE",
            header_keywords=["BUSY", "PARTY WISE", "PRODUCT WISE"],
            column_map=ERPColumnMap(
                party_name=0, product_name=1, packing=2,
                qty=3, free_qty=4, rate=5, amount=6,
                discount=7, tax_pct=8,
            ),
            subtotal_re=re.compile(r'(?i)\bsub\s*total\b'),
        ))

        self.register(ERPSchema(
            erp_name="BUSY",
            format_id="BUSY_SALE_REGISTER",
            header_keywords=["BUSY", "SALE REGISTER"],
            column_map=ERPColumnMap(
                party_name=0, product_name=2, qty=3,
                rate=4, amount=5, gstin=6, hsn=7, tax_pct=8,
            ),
            subtotal_re=re.compile(r'(?i)\bsub\s*total\b'),
        ))

        self.register(ERPSchema(
            erp_name="GOFRUGAL",
            format_id="GOFRUGAL_DIST_SALES",
            header_keywords=["GOFRUGAL"],
            column_map=ERPColumnMap(
                product_name=0, hsn=1, batch=2, expiry=3,
                qty=4, free_qty=5, rate=6, amount=7, tax_pct=8,
            ),
            subtotal_re=re.compile(r'(?i)\b(subtotal|sub\s*total)\b'),
        ))

        self.register(ERPSchema(
            erp_name="GOFRUGAL",
            format_id="GOFRUGAL_PARTY_WISE",
            header_keywords=["GOFRUGAL", "PARTY WISE"],
            column_map=ERPColumnMap(
                party_name=0, area=1, qty=2, free_qty=3,
                rate=4, amount=5, discount=6, tax_pct=7,
            ),
            subtotal_re=re.compile(r'(?i)\bsub\s*total\b'),
        ))

        self.register(ERPSchema(
            erp_name="MEDICIN",
            format_id="MEDICIN_STOCK_STATEMENT",
            header_keywords=["MEDICIN"],
            column_map=ERPColumnMap(
                item_code=0, product_name=1, company=2, packing=3,
                qty=4, rate=5, amount=6, batch=7, expiry=8,
            ),
            subtotal_re=re.compile(r'(?i)\btotal\b'),
        ))

        logger.info("ERPTemplateEngine: %d built-in schemas registered", len(self._schemas))


class AIFormatInferenceEngine:
    _instance: Optional["AIFormatInferenceEngine"] = None
    _LEARN_FILE = "format_learned_schemas.json"

    def __init__(self):
        self._learned: dict = {}
        self._load_learned_store()

    @classmethod
    def instance(cls) -> "AIFormatInferenceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def infer(self, lines: list, raw_text: str = "") -> FormatInferenceResult:
        features = self._extract_features(lines)
        tier1    = self._structural_classify(features)

        if tier1.confidence >= 0.75:
            return tier1

        tier2 = self._column_pattern_probe(lines, features)
        if tier2 and tier2.confidence >= 0.60:
            self._persist_learned_schema(tier2)
            self._register_inferred_schema(tier2)
            return tier2

        cached = self._load_learned_schema(self._cache_key(features))
        if cached and cached.confidence >= 0.55:
            return cached

        return tier1

    def _extract_features(self, lines: list) -> dict:
        if not lines:
            return {}

        data_lines = [l for l in lines if l.strip() and not l.strip().startswith('---')]
        if not data_lines:
            return {}

        tab_counts    = [l.count('\t') for l in data_lines]
        col_counts    = [c + 1 for c in tab_counts]
        total_chars   = sum(len(l) for l in data_lines)
        digit_chars   = sum(sum(c.isdigit() for c in l) for l in data_lines)
        alpha_chars   = sum(sum(c.isalpha() for c in l) for l in data_lines)

        amt_lines  = sum(1 for l in data_lines if re.search(r'\b\d{2,7}\.\d{2}\b', l))
        lupin_lines = sum(1 for l in data_lines if re.search(r'\bLUPIN\b', l, re.I))
        gstin_lines = sum(1 for l in data_lines if re.search(
            r'\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]\b', l))
        date_lines  = sum(1 for l in data_lines if re.search(
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', l))
        bill_lines  = sum(1 for l in data_lines if re.search(
            r'\b(bill|invoice|vch|voucher)\s*(no|#|num)', l, re.I))
        hsn_lines   = sum(1 for l in data_lines if re.search(r'\bhsn\b', l, re.I))
        batch_lines = sum(1 for l in data_lines if re.search(
            r'\b(batch|b\.no|exp\.?|expiry)\b', l, re.I))
        subtotal_lines = sum(1 for l in data_lines if re.search(
            r'\b(sub\s*total|subtotal)\b', l, re.I))
        party_lines = sum(1 for l in data_lines if re.search(
            r'\b(party\s*(name|wise)|customer|store\s*name)\b', l, re.I))
        drug_lines  = sum(1 for l in data_lines if re.search(
            r'\b(?:HEPP|LUPI|ONECLAV|DEFENAC|BILALUP|REVEAL|CEFP|MEGARICH|'
            r'PANTOLUP|AZILUP|CIPROVA|LUPICEF|XIMECEF|FLUCALUP)\w*', l, re.I))

        n = max(len(data_lines), 1)
        modal_cols = max(set(col_counts), key=col_counts.count) if col_counts else 0

        _hsn_col_re    = re.compile(r'^\d{4,8}$')
        _expiry_re     = re.compile(r'\b\d{1,2}[-/]\d{4}\b')
        _batch_col_re  = re.compile(r'^[A-Z]\d{2,6}[A-Z]?\d*$', re.I)
        _lead_int_re   = re.compile(r'^\d{3,8}$')

        hsn_col_lines    = 0
        expiry_lines_ext = 0
        batch_col_lines  = 0
        leading_int_lines= 0
        for dl in data_lines:
            parts = dl.split('\t')
            if len(parts) > 1 and _hsn_col_re.match(parts[1].strip()):
                hsn_col_lines += 1
            if len(parts) > 2 and _batch_col_re.match(parts[2].strip()):
                batch_col_lines += 1
            if _expiry_re.search(dl):
                expiry_lines_ext += 1
            if parts and _lead_int_re.match(parts[0].strip()):
                leading_int_lines += 1

        return {
            "n_lines":            n,
            "mean_cols":          float(sum(col_counts)) / n,
            "modal_cols":         modal_cols,
            "max_cols":           max(col_counts) if col_counts else 0,
            "digit_ratio":        digit_chars / max(total_chars, 1),
            "alpha_ratio":        alpha_chars / max(total_chars, 1),
            "amt_line_frac":      amt_lines       / n,
            "lupin_frac":         lupin_lines      / n,
            "gstin_frac":         gstin_lines      / n,
            "date_frac":          date_lines       / n,
            "bill_frac":          bill_lines       / n,
            "hsn_frac":           hsn_lines        / n,
            "hsn_col_frac":       hsn_col_lines    / n,
            "batch_frac":         batch_lines      / n,
            "batch_col_frac":     batch_col_lines  / n,
            "expiry_frac":        expiry_lines_ext / n,
            "subtotal_frac":      subtotal_lines   / n,
            "party_header_frac":  party_lines      / n,
            "drug_line_frac":     drug_lines       / n,
            "leading_int_frac":   leading_int_lines/ n,
            "tab_std":            float(
                (sum((c - sum(tab_counts)/n)**2 for c in tab_counts) / n)**0.5
            ) if n > 1 else 0.0,
        }

    def _structural_classify(self, f: dict) -> FormatInferenceResult:
        if not f:
            return self._unknown(0.0)

        drug_heavy    = f.get("drug_line_frac", 0)    > 0.3
        lupin_heavy   = f.get("lupin_frac", 0)         > 0.3
        amt_dense     = f.get("amt_line_frac", 0)      > 0.6
        wide_table    = f.get("modal_cols", 0)         >= 8
        narrow_tbl    = f.get("modal_cols", 0)         <= 4
        has_gstin     = f.get("gstin_frac", 0)         > 0.05
        has_batch     = f.get("batch_frac", 0)         > 0.05
        has_hsn       = f.get("hsn_frac", 0)           > 0.05
        has_bills     = f.get("bill_frac", 0)          > 0.05
        has_dates     = f.get("date_frac", 0)          > 0.1
        has_subtotals = f.get("subtotal_frac", 0)      > 0.02
        alpha_rich    = f.get("alpha_ratio", 0)        > 0.35
        digit_rich    = f.get("digit_ratio", 0)        > 0.30
        stable_cols   = f.get("tab_std", 999)          < 2.0
        has_hsn_col   = f.get("hsn_col_frac", 0)      > 0.4
        has_batch_col = f.get("batch_col_frac", 0)    > 0.3
        has_expiry    = f.get("expiry_frac", 0)        > 0.3
        leading_int   = f.get("leading_int_frac", 0)  > 0.5

        if drug_heavy and lupin_heavy and amt_dense and wide_table and stable_cols:
            return FormatInferenceResult(
                format_id="MARG_PARTY_PRODUCT_FLAT", erp_name="MARG",
                confidence=0.92, tier=1,
                column_map={"city": 0, "store": "1+2", "product": 3,
                            "packing": 4, "qty": 5, "free": 6,
                            "rate": 7, "amount": 8, "company": 9},
                parsing_hints={"multi_col_store": True, "lupin_tag_in_last_col": True},
                source="structural_classifier",
            )

        if lupin_heavy and not drug_heavy and narrow_tbl and amt_dense:
            return FormatInferenceResult(
                format_id="MARG_PARTY_WISE_SUMMARY", erp_name="MARG",
                confidence=0.80, tier=1,
                column_map={"store": 0, "qty": 1, "amount": 2},
                parsing_hints={"summary_mode": True},
                source="structural_classifier",
            )

        if amt_dense and (has_hsn_col or has_hsn) and (has_batch_col or has_batch) and has_expiry:
            return FormatInferenceResult(
                format_id="GOFRUGAL_DIST_SALES", erp_name="GOFRUGAL",
                confidence=0.88, tier=1,
                column_map={"product_name": 0, "hsn": 1, "batch": 2,
                            "expiry": 3, "qty": 4, "free": 5,
                            "rate": 6, "amount": 7},
                parsing_hints={"has_batch_expiry": True},
                source="structural_classifier",
            )

        if has_gstin and has_dates and amt_dense:
            conf = 0.88 if leading_int else 0.80
            col_map = {"gstin": 3, "amount": -1}
            if leading_int:
                col_map.update({"bill_no": 0, "date": 1, "store": 2})
            else:
                col_map.update({"store": 0, "date": 1})
            return FormatInferenceResult(
                format_id="PAGED_STORE_INVOICE", erp_name="MARG",
                confidence=conf, tier=1,
                column_map=col_map,
                parsing_hints={"invoice_mode": True, "bill_no_in_col0": leading_int},
                source="structural_classifier",
            )

        if has_subtotals and amt_dense and wide_table and not lupin_heavy:
            conf = 0.82 if has_gstin else 0.74
            return FormatInferenceResult(
                format_id="BUSY_PARTY_PRODUCT_WISE", erp_name="BUSY",
                confidence=conf, tier=1,
                column_map={"store": 0, "product": 1, "packing": 2,
                            "qty": 3, "free": 4, "rate": 5, "amount": 6},
                parsing_hints={"has_subtotals": True},
                source="structural_classifier",
            )

        if (has_bills or leading_int) and has_dates and alpha_rich and digit_rich:
            return FormatInferenceResult(
                format_id="SALES_BOOK_REGISTER", erp_name="MARG",
                confidence=0.78, tier=1,
                column_map={"bill_no": 0, "date": 1, "store": 2,
                            "amount": -1, "taxable": -2},
                parsing_hints={"bill_per_row": True},
                source="structural_classifier",
            )

        if amt_dense and wide_table:
            return FormatInferenceResult(
                format_id="GENERIC_TABULAR", erp_name="UNKNOWN",
                confidence=0.45, tier=1,
                column_map={"amount": -1, "rate": -2},
                parsing_hints={},
                source="structural_classifier",
            )

        return self._unknown(0.20)

    @staticmethod
    def _unknown(conf: float) -> FormatInferenceResult:
        return FormatInferenceResult(
            format_id="UNKNOWN", erp_name="UNKNOWN",
            confidence=conf, tier=1, column_map={},
            parsing_hints={}, source="structural_classifier",
        )

    _SCHEMA_TEMPLATES = {
        "MARG_PARTY_PRODUCT_FLAT": {
            0: ("text",        1.0),
            1: ("text",        0.8),
            2: ("text",        0.5),
            3: ("drug",        1.5),
            4: ("packing",     1.0),
            5: ("integer",     0.8),
            6: ("integer",     0.6),
            7: ("amount",      1.0),
            8: ("amount",      1.5),
            9: ("company_tag", 1.2),
        },
        "MARG_PARTY_WISE_SUMMARY": {
            0: ("text",    1.0),
            1: ("integer", 1.0),
            2: ("amount",  1.5),
        },
        "BUSY_PARTY_PRODUCT_WISE": {
            0: ("text",    1.0),
            1: ("drug",    1.2),
            2: ("packing", 0.8),
            3: ("integer", 0.8),
            4: ("integer", 0.5),
            5: ("amount",  1.0),
            6: ("amount",  1.5),
            7: ("gstin",   1.2),
            8: ("integer", 0.5),
        },
        "GOFRUGAL_DIST_SALES": {
            0: ("drug",    1.5),
            1: ("hsn",     1.2),
            2: ("text",    0.8),
            3: ("date",    1.0),
            4: ("integer", 0.8),
            5: ("integer", 0.5),
            6: ("amount",  1.0),
            7: ("amount",  1.5),
            8: ("integer", 0.5),
        },
        "PAGED_STORE_INVOICE": {
            0: ("integer", 0.8),
            1: ("date",    1.2),
            2: ("text",    1.0),
            3: ("gstin",   1.5),
            4: ("amount",  1.2),
            5: ("amount",  1.0),
            6: ("amount",  0.8),
            7: ("amount",  0.8),
        },
        "SALES_BOOK_REGISTER": {
            0: ("integer", 0.8),
            1: ("date",    1.2),
            2: ("text",    1.0),
            3: ("amount",  1.5),
            4: ("amount",  1.0),
        },
        "MEDICIN_STOCK_STATEMENT": {
            0: ("text",    0.8),
            1: ("drug",    1.5),
            2: ("text",    0.8),
            3: ("packing", 0.8),
            4: ("integer", 0.8),
            5: ("amount",  1.0),
            6: ("amount",  1.5),
            7: ("text",    0.8),
            8: ("date",    0.8),
        },
    }

    _PROBE_AMOUNT      = re.compile(r'^-?\d{1,7}\.\d{2}$')
    _PROBE_INTEGER     = re.compile(r'^\d{1,6}$')
    _PROBE_DATE        = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')
    _PROBE_GSTIN       = re.compile(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]\b')
    _PROBE_HSN         = re.compile(r'^\d{4,8}$')
    _PROBE_DRUG        = re.compile(
        r'\b(?:HEPP|LUPI|ONECLAV|DEFENAC|BILALUP|REVEAL|CEFP|MEGARICH|'
        r'PANTOLUP|AZILUP|CIPROVA|LUPICEF|XIMECEF|FLUCALUP|AMOX|CEFO|'
        r'METFO|ATORV|AZITHRO|PANTO|OMEP|CETIRIZ|PARACET|IBUPROF)\w*', re.I)
    _PROBE_PACKING     = re.compile(
        r'\b\d{1,3}\s*[*xX]\s*\d+\b|\b\d+\s*(?:ML|MG|GM|TAB|CAP|SYP|INJ)\b', re.I)
    _PROBE_COMPANY_TAG = re.compile(r'\b(LUPIN|CIPLA|SUNPHARMA|ALKEM|MACLEODS|'
                                    r'ZYDUS|PFIZER|ABBOTT|GSK|MANKIND|DRL)\b', re.I)
    _PROBE_TEXT        = re.compile(r'[A-Za-z]{3,}')

    def _probe_col_type(self, values: list) -> dict:
        n = max(len(values), 1)
        counts = {t: 0 for t in ["amount","integer","date","gstin","hsn",
                                  "drug","packing","company_tag","text","empty"]}
        for v in values:
            v = v.strip()
            if not v:
                counts["empty"] += 1; continue
            if self._PROBE_AMOUNT.match(v):      counts["amount"]      += 1
            elif self._PROBE_GSTIN.search(v):    counts["gstin"]       += 1
            elif self._PROBE_DATE.search(v):     counts["date"]        += 1
            elif self._PROBE_COMPANY_TAG.search(v): counts["company_tag"] += 1
            elif self._PROBE_DRUG.search(v):     counts["drug"]        += 1
            elif self._PROBE_PACKING.search(v):  counts["packing"]     += 1
            elif self._PROBE_HSN.match(v):       counts["hsn"]         += 1
            elif self._PROBE_INTEGER.match(v):   counts["integer"]     += 1
            elif self._PROBE_TEXT.search(v):     counts["text"]        += 1
        return {t: c / n for t, c in counts.items()}

    def _column_pattern_probe(self, lines: list, features: dict) -> Optional[FormatInferenceResult]:
        data_rows = [l.split('\t') for l in lines
                     if l.strip() and not re.search(
                         r'\b(grand\s*total|sub\s*total|page\s*no)\b', l, re.I)][:30]
        if len(data_rows) < 3:
            return None

        col_counts = [len(r) for r in data_rows]
        modal_n    = max(set(col_counts), key=col_counts.count)

        data_rows = [r for r in data_rows if abs(len(r) - modal_n) <= 1]
        if len(data_rows) < 3:
            return None

        data_rows = [(r + [''] * modal_n)[:modal_n] for r in data_rows]

        col_profiles: list = []
        for ci in range(modal_n):
            vals = [row[ci] for row in data_rows]
            col_profiles.append(self._probe_col_type(vals))

        best_score   = 0.0
        best_fmt_id  = None
        best_col_map = {}

        for fmt_id, template in self._SCHEMA_TEMPLATES.items():
            score, col_map = self._score_template(col_profiles, template, modal_n)
            if score > best_score:
                best_score   = score
                best_fmt_id  = fmt_id
                best_col_map = col_map

        if best_score >= 0.50 and best_fmt_id:
            erp = best_fmt_id.split('_')[0]
            return FormatInferenceResult(
                format_id     = best_fmt_id,
                erp_name      = erp,
                confidence    = min(best_score, 0.95),
                tier          = 2,
                column_map    = best_col_map,
                parsing_hints = {"probed_col_count": modal_n,
                                 "probe_score": round(best_score, 3)},
                source        = "column_probe",
            )

        return self._auto_discover_columns(col_profiles, modal_n, features)

    def _score_template(self, col_profiles: list, template: dict,
                        modal_n: int) -> tuple:
        total_weight = sum(w for _, (_, w) in template.items())
        earned       = 0.0
        col_map      = {}

        _type_to_field = {
            "drug":        "product_name",
            "company_tag": "company",
            "gstin":       "gstin",
            "date":        "date",
            "hsn":         "hsn",
        }

        for col_idx, (expected_type, weight) in template.items():
            if col_idx >= len(col_profiles):
                continue
            match_frac = col_profiles[col_idx].get(expected_type, 0.0)
            earned    += weight * match_frac

            if expected_type == "amount":
                col_map.setdefault("rate",   col_idx)
                col_map["amount"] = col_idx

            semantic = _type_to_field.get(expected_type)
            if semantic:
                col_map[semantic] = col_idx
            elif expected_type == "text" and col_idx <= 2:
                col_map.setdefault("store_name", col_idx)
            elif expected_type == "integer" and col_idx <= 6:
                col_map.setdefault("qty", col_idx)

        score = earned / max(total_weight, 1e-9)
        return score, col_map

    def _auto_discover_columns(self, col_profiles: list, modal_n: int,
                                features: dict) -> Optional[FormatInferenceResult]:
        if not col_profiles:
            return None

        def _best_col(type_key: str) -> int:
            scores = [(i, p.get(type_key, 0)) for i, p in enumerate(col_profiles)]
            best   = max(scores, key=lambda x: x[1])
            return best[0] if best[1] > 0.15 else -1

        amount_cols = sorted(
            [(i, p.get("amount", 0)) for i, p in enumerate(col_profiles)],
            key=lambda x: -x[1]
        )
        amount_cols = [(i, s) for i, s in amount_cols if s > 0.15]
        if not amount_cols:
            return None

        amount_idx = amount_cols[0][0]
        rate_idx   = amount_cols[1][0] if len(amount_cols) >= 2 else -1

        drug_idx   = _best_col("drug")
        gstin_idx  = _best_col("gstin")
        qty_idx    = _best_col("integer")
        date_idx   = _best_col("date")
        hsn_idx    = _best_col("hsn")
        pack_idx   = _best_col("packing")
        company_idx= _best_col("company_tag")

        text_cols  = sorted(
            [(i, p.get("text", 0)) for i, p in enumerate(col_profiles)
             if i not in {amount_idx, rate_idx, drug_idx, gstin_idx,
                           qty_idx, date_idx, hsn_idx, pack_idx, company_idx}],
            key=lambda x: -x[1]
        )
        store_idx  = text_cols[0][0] if text_cols else -1
        area_idx   = text_cols[1][0] if len(text_cols) >= 2 else -1

        col_map = {
            "amount":       amount_idx,
            "rate":         rate_idx,
            "product_name": drug_idx,
            "store_name":   store_idx,
            "area":         area_idx,
            "qty":          qty_idx,
            "gstin":        gstin_idx,
            "date":         date_idx,
            "hsn":          hsn_idx,
            "packing":      pack_idx,
            "company":      company_idx,
        }
        col_map = {k: v for k, v in col_map.items() if v >= 0}

        key_fields_found = sum(1 for f in ["amount","product_name","store_name","qty"]
                               if col_map.get(f, -1) >= 0)
        confidence = 0.40 + key_fields_found * 0.10

        erp_guess  = "MARG" if features.get("lupin_frac", 0) > 0.1 else "GENERIC"
        fmt_id     = f"{erp_guess}_AUTO_{modal_n}COL"

        return FormatInferenceResult(
            format_id     = fmt_id,
            erp_name      = erp_guess,
            confidence    = confidence,
            tier          = 2,
            column_map    = col_map,
            parsing_hints = {"auto_discovered": True, "modal_n": modal_n,
                             "key_fields_resolved": key_fields_found},
            source        = "column_probe",
        )

    def _register_inferred_schema(self, result: FormatInferenceResult) -> None:
        try:
            cm_raw = result.column_map
            def _ci(key):
                v = cm_raw.get(key, -1)
                return int(v) if isinstance(v, (int, float)) and v != -1 else -1

            new_schema = ERPSchema(
                erp_name        = result.erp_name,
                format_id       = result.format_id,
                header_keywords = [],
                column_map      = ERPColumnMap(
                    item_code    = _ci("item_code"),
                    area         = _ci("area"),
                    party_name   = _ci("store_name"),
                    product_name = _ci("product_name"),
                    packing      = _ci("packing"),
                    qty          = _ci("qty"),
                    free_qty     = _ci("free"),
                    rate         = _ci("rate"),
                    amount       = _ci("amount"),
                    company      = _ci("company"),
                    gstin        = _ci("gstin"),
                    hsn          = _ci("hsn"),
                    batch        = _ci("batch"),
                    expiry       = _ci("expiry"),
                ),
                notes = (f"[Probed tier={result.tier} conf={result.confidence:.2f}]  "
                         + result.parsing_hints.get("notes", "")),
            )
            _ERP_TEMPLATE_ENGINE.register(new_schema)
        except Exception as exc:
            logger.warning("Could not register probed schema: %s", exc)

    def _persist_learned_schema(self, result: FormatInferenceResult) -> None:
        try:
            store = self._load_all_learned()
            key   = self._cache_key_from_result(result)
            store[key] = {
                "format_id":     result.format_id,
                "erp_name":      result.erp_name,
                "confidence":    result.confidence,
                "tier":          3,
                "column_map":    result.column_map,
                "parsing_hints": result.parsing_hints,
                "source":        "learned_store",
            }
            with open(self._LEARN_FILE, 'w') as f:
                json.dump(store, f, indent=2)
        except Exception as exc:
            logger.warning("Could not persist schema: %s", exc)

    def _load_learned_schema(self, cache_key: str) -> Optional[FormatInferenceResult]:
        store = self._load_all_learned()
        d = store.get(cache_key)
        if d:
            return FormatInferenceResult(
                format_id     = d["format_id"],
                erp_name      = d["erp_name"],
                confidence    = float(d["confidence"]),
                tier          = int(d.get("tier", 3)),
                column_map    = d.get("column_map", {}),
                parsing_hints = d.get("parsing_hints", {}),
                source        = "learned_store",
            )
        return None

    def _load_all_learned(self) -> dict:
        try:
            if os.path.exists(self._LEARN_FILE):
                with open(self._LEARN_FILE) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _load_learned_store(self) -> None:
        store = self._load_all_learned()
        if store:
            logger.info("Learning store: %d schemas loaded", len(store))
            for entry in store.values():
                result = FormatInferenceResult(
                    format_id     = entry["format_id"],
                    erp_name      = entry["erp_name"],
                    confidence    = float(entry.get("confidence", 0.6)),
                    tier          = 3,
                    column_map    = entry.get("column_map", {}),
                    parsing_hints = entry.get("parsing_hints", {}),
                    source        = "learned_store",
                )
                self._register_inferred_schema(result)

    @staticmethod
    def _cache_key(features: dict) -> str:
        modal = features.get("modal_cols", 0)
        drug  = "Y" if features.get("drug_line_frac", 0) > 0.2 else "N"
        lupin = "Y" if features.get("lupin_frac", 0) > 0.2 else "N"
        amt   = "Y" if features.get("amt_line_frac", 0) > 0.5 else "N"
        return f"cols{modal}_drug{drug}_lup{lupin}_amt{amt}"

    @staticmethod
    def _cache_key_from_result(result: FormatInferenceResult) -> str:
        cm = result.column_map
        ncols = max(cm.values()) + 1 if cm else 0
        f_id  = result.format_id[:20]
        return f"auto_{f_id}_{ncols}col"


_ERP_TEMPLATE_ENGINE = ERPTemplateEngine.instance()
_AI_FORMAT_ENGINE = AIFormatInferenceEngine.instance()
