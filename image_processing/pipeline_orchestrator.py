import os
import re
import csv
import json
import uuid
import datetime
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

import numpy as np

from .quality_analyzer import DocumentQualityAnalyzer, DocumentQualityProfile
from .preprocessor import AdaptivePreprocessor
from .spatial_clusterer import SpatialClusterer
from .topological_aligner import TopologicalAligner
from .confidence_scorer import ConfidenceScorer
from .multipass_ocr import MultiPassOCR
from .validation_engine import ValidationEngine
from .table_layout_detector import AITableDetector, AILayoutDetector, TableRegion
from .continuity_pass import MultiPageContinuityPass

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

try:
    from PIL import Image
except ImportError:
    Image = None

# RapidOCR lazy engine singleton
_RAPID_ENGINE = None
_RAPIDOCR_AVAILABLE = False
try:
    from rapidocr_onnxruntime import RapidOCR
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    pass


def get_rapid_engine():
    global _RAPID_ENGINE
    if _RAPID_ENGINE is None:
        if not _RAPIDOCR_AVAILABLE:
            logger.error("RapidOCR not found.")
            return None
        logger.info("Initialising RapidOCR ONNX Engine ...")
        _RAPID_ENGINE = RapidOCR()
    return _RAPID_ENGINE


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


class EnterpriseOCRPipeline:
    """Main orchestrator for the 12-layer enterprise OCR pipeline."""

    def __init__(self):
        self._analyzer = DocumentQualityAnalyzer()
        self._preprocessor = AdaptivePreprocessor()
        self._clusterer = SpatialClusterer()
        self._scorer = ConfidenceScorer()
        self._validator = ValidationEngine()
        self._multipass = MultiPassOCR()
        self._table_detector = AITableDetector()
        self._layout_detector = AILayoutDetector()
        self.continuity_pass = MultiPageContinuityPass()
        self.dewarper = PageDewarper()
        self.quality_analyzer = self._analyzer
        self.preprocessor = self._preprocessor
        self.spatial_clusterer = self._clusterer
        self.topological_aligner = TopologicalAligner()
        self.confidence_scorer = self._scorer
        self.validation_engine = self._validator
        self.ocr_engine = self._multipass

    def process_document(self, pdf_path: str, dpi: int = 300,
                         high_res_pass: bool = True) -> Dict[str, Any]:
        result = {
            'status': 'success',
            'pages': [],
            'full_text': '',
            'metadata': {},
            'quality_profile': None,
            'tables': [],
            'errors': []
        }
        pages = self._load_pages(pdf_path, dpi)
        if not pages:
            result['status'] = 'error'
            result['errors'].append('No pages could be loaded')
            return result
        all_page_texts = []
        all_ocr_data = []
        combined_tables = []
        global_quality = None
        for page_num, page_img in enumerate(pages):
            try:
                page_result = self._process_page(page_img, page_num, high_res_pass)
                all_page_texts.append(page_result['text'])
                all_ocr_data.extend(page_result['ocr_data'])
                combined_tables.extend(page_result['tables'])
                if global_quality is None:
                    global_quality = page_result['quality']
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                result['errors'].append(f"Page {page_num}: {str(e)}")
        merged_text = self.continuity_pass.process_pages(all_page_texts)
        result['pages'] = all_page_texts
        result['full_text'] = merged_text
        result['metadata'] = {
            'total_pages': len(pages),
            'total_ocr_regions': len(all_ocr_data),
            'total_tables': len(combined_tables),
            'processing_time': datetime.datetime.now().isoformat()
        }
        result['quality_profile'] = global_quality
        result['tables'] = combined_tables
        return result

    def _load_pages(self, pdf_path: str, dpi: int) -> List[np.ndarray]:
        pages = []
        ext = os.path.splitext(pdf_path)[1].lower()
        if ext == '.pdf':
            if convert_from_path is None:
                logger.error("pdf2image not installed")
                return []
            try:
                pil_pages = convert_from_path(pdf_path, dpi=dpi)
                for pil_page in pil_pages:
                    img = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
                    pages.append(img)
            except Exception as e:
                logger.error(f"Failed to convert PDF to images: {e}")
        elif ext in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp'):
            if cv2 is None:
                logger.error("cv2 not installed")
                return []
            img = cv2.imread(pdf_path)
            if img is not None:
                pages.append(img)
        return pages

    def _process_page(self, image: np.ndarray, page_num: int,
                      high_res_pass: bool) -> Dict[str, Any]:
        result = {
            'page_num': page_num,
            'text': '',
            'ocr_data': [],
            'tables': [],
            'quality': None
        }
        quality = self._analyzer.analyze(image)
        result['quality'] = quality
        preprocessed = self._preprocessor.process(image, quality)
        engine = get_rapid_engine()
        if engine is None:
            return result
        best_text, best_conf = self._multipass.run(preprocessed, engine)
        result['ocr_data'] = []
        result['text'] = best_text
        detected_tables = self._table_detector.detect(preprocessed)
        result['tables'] = detected_tables
        return result

    def extract_text(self, pil_image):
        """Public entry point — 12-layer pipeline on a PIL Image, returns text string."""
        engine = get_rapid_engine()
        if engine is None:
            return ""
        image_cv = np.array(pil_image)
        if len(image_cv.shape) == 3:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        profile = self._analyzer.analyze(image_cv)
        preprocessed = self._preprocessor.process(image_cv, profile)
        deskew_applied = abs(profile.skew_angle) > 0.5
        if profile.has_fold and not deskew_applied:
            preprocessed = self.dewarper.dewarp(preprocessed)
        best_text, best_conf = self._multipass.run(preprocessed, engine)
        try:
            layout_regions = self._layout_detector.detect(preprocessed, best_text)
            filtered_text = AILayoutDetector.filter_text_to_table_region(
                best_text, layout_regions
            )
            if filtered_text.strip() and len(filtered_text) >= len(best_text) * 0.3:
                best_text = filtered_text
        except Exception:
            pass
        if best_conf < 0.45:
            logger.warning(
                "LOW OCR CONFIDENCE (%.3f). Manual verification recommended.", best_conf
            )
        return best_text

    def extract_text_with_validation(self, pil_image):
        """Extended entry point returning (text, grand_total_found, confidence)."""
        engine = get_rapid_engine()
        if engine is None:
            return "", 0.0, 0.0
        image_cv = np.array(pil_image)
        if len(image_cv.shape) == 3:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        profile = self._analyzer.analyze(image_cv)
        preprocessed = self._preprocessor.process(image_cv, profile)
        deskew_applied = abs(profile.skew_angle) > 0.5
        if profile.has_fold and not deskew_applied:
            preprocessed = self.dewarper.dewarp(preprocessed)
        best_text, best_conf = self._multipass.run(preprocessed, engine)
        try:
            layout_regions = self._layout_detector.detect(preprocessed, best_text)
            filtered_text = AILayoutDetector.filter_text_to_table_region(
                best_text, layout_regions
            )
            if filtered_text.strip() and len(filtered_text) >= len(best_text) * 0.3:
                best_text = filtered_text
        except Exception:
            pass
        grand_total = self._validator.find_grand_total(best_text)
        return best_text, grand_total, best_conf


# ── Module-level singleton ────────────────────────────────────────────────────
_ENTERPRISE_PIPELINE = EnterpriseOCRPipeline()


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


def process_and_convert(pdf_path: str, output_dir: str,
                         report_type: str = "AUTO",
                         company_detect: bool = True,
                         use_erp_engine: bool = True,
                         extractor_params: Optional[Dict] = None) -> Dict[str, Any]:
    """Main entry point: OCR a document and produce CSV/JSON output."""
    if extractor_params is None:
        extractor_params = {}
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pipeline = _ENTERPRISE_PIPELINE
    ocr_result = pipeline.process_document(
        pdf_path,
        dpi=extractor_params.get('dpi', 300)
    )
    full_text = ocr_result['full_text']
    detected_format = "UNKNOWN"
    if company_detect:
        detected_format = _detect_format_with_regex(full_text)
    table_data = []
    if ocr_result['tables']:
        for t in ocr_result['tables']:
            table_data.append({
                'region': f"{t.x},{t.y},{t.w},{t.h}",
                'cells_count': len(t.cells) if hasattr(t, 'cells') else 0,
                'confidence': t.confidence
            })
    qp = ocr_result.get('quality_profile')
    if qp is not None:
        quality_score = max(0.0, min(100.0, (
            100.0
            - max(0.0, qp.blur_score / 2.0 if qp.is_blurry else 0.0)
            - (10.0 if qp.has_shadow else 0.0)
            - (15.0 if qp.is_dull else 0.0)
            - (15.0 if qp.has_dark_background else 0.0)
            - qp.noise_level * 30.0
            - (5.0 if qp.needs_upscale else 0.0)
        )))
        quality_flags = []
        if qp.is_blurry:
            quality_flags.append("blurry")
        if qp.has_shadow:
            quality_flags.append("shadow")
        if qp.is_dull:
            quality_flags.append("dull")
        if qp.has_dark_background:
            quality_flags.append("dark_background")
        if qp.noise_level > 0.3:
            quality_flags.append("noisy")
        if qp.is_low_contrast:
            quality_flags.append("low_contrast")
    else:
        quality_score = 0.0
        quality_flags = []
    output = {
        'status': ocr_result['status'],
        'file_name': base_name,
        'detected_format': detected_format,
        'full_text': full_text,
        'pages': ocr_result['pages'],
        'tables': table_data,
        'quality_score': quality_score,
        'quality_flags': quality_flags,
        'metadata': ocr_result['metadata'],
        'errors': ocr_result['errors']
    }
    txt_path = os.path.join(output_dir, f"{base_name}_ocr_output.txt")
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Detected Format: {detected_format}\n")
            f.write(f"File: {base_name}\n")
            f.write(f"Quality Score: {output['quality_score']:.1f}\n")
            f.write(f"Quality Flags: {', '.join(output['quality_flags'])}\n")
            f.write("=" * 60 + "\n")
            f.write(full_text)
    except Exception as e:
        logger.error(f"Failed to write OCR text: {e}")
    json_path = os.path.join(output_dir, f"{base_name}_ocr_result.json")
    try:
        serializable = {
            'status': output['status'],
            'file_name': output['file_name'],
            'detected_format': output['detected_format'],
            'quality_score': output['quality_score'],
            'quality_flags': output['quality_flags'],
            'metadata': output['metadata'],
            'errors': output['errors'],
            'full_text_length': len(output['full_text']),
            'page_count': len(output['pages']),
            'table_count': len(output['tables'])
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write JSON result: {e}")
    return output


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
