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
from .spatial_clusterer import SpatialClusterer, TextRegion
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

    def __init__(self, use_rapid: bool = True, use_paddle: bool = True,
                 use_table_transformer: bool = True,
                 use_layout_parser: bool = True,
                 lang: str = 'en', gpu: bool = False):
        self.quality_analyzer = DocumentQualityAnalyzer()
        self.preprocessor = AdaptivePreprocessor()
        self.spatial_clusterer = SpatialClusterer()
        self.topological_aligner = TopologicalAligner()
        self.confidence_scorer = ConfidenceScorer()
        self.ocr_engine = MultiPassOCR(use_rapid=use_rapid, use_paddle=use_paddle,
                                       lang=lang, gpu=gpu)
        self.validation_engine = ValidationEngine()
        self.table_detector = AITableDetector(
            use_table_transformer=use_table_transformer,
            use_layout_parser=use_layout_parser,
            device='cuda' if gpu else 'cpu'
        )
        self.layout_detector = AILayoutDetector()
        self.continuity_pass = MultiPageContinuityPass()
        self.dewarper = PageDewarper()

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
        quality = self.quality_analyzer.analyze(image)
        result['quality'] = quality
        processed = self.preprocessor.preprocess(image, quality)
        aligned = self.topological_aligner.align(processed)
        dewarped = self.dewarper.dewarp(aligned.aligned_image)
        ocr_data, avg_conf = self.ocr_engine.ocr(dewarped,
                                                   high_res_for_low_conf=high_res_pass)
        result['ocr_data'] = ocr_data
        regions = []
        for data in ocr_data:
            bbox = data['bbox']
            confidence = self.confidence_scorer.score_text(data['text'])
            region = TextRegion(
                x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3],
                text=data['text'], confidence=confidence
            )
            regions.append(region)
        clustered = self.spatial_clusterer.cluster(regions, image.shape[:2])
        detected_tables, method = self.table_detector.detect_tables(dewarped)
        result['tables'] = detected_tables
        lines = []
        for cluster in clustered:
            line_text = ' '.join(r.text for r in cluster)
            lines.append(line_text)
        result['text'] = '\n'.join(lines)
        return result


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
    pipeline = EnterpriseOCRPipeline(
        use_rapid=extractor_params.get('use_rapid', True),
        use_paddle=extractor_params.get('use_paddle', True),
        lang=extractor_params.get('lang', 'en'),
        gpu=extractor_params.get('gpu', False)
    )
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
    output = {
        'status': ocr_result['status'],
        'file_name': base_name,
        'detected_format': detected_format,
        'full_text': full_text,
        'pages': ocr_result['pages'],
        'tables': table_data,
        'quality_score': (ocr_result['quality_profile'].overall_score
                          if ocr_result['quality_profile'] else 0.0),
        'quality_flags': (ocr_result['quality_profile'].flags
                          if ocr_result['quality_profile'] else []),
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
