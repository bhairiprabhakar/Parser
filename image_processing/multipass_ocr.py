import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
logger = logging.getLogger(__name__)

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_OCR_AVAILABLE = True
except ImportError:
    RAPID_OCR_AVAILABLE = False
    logger.warning("rapidocr-onnxruntime not available, falling back")

try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    logger.warning("paddleocr not available")


class MultiPassOCR:
    """Multi-pass OCR with confidence-based fallback strategies."""

    def __init__(self, use_rapid: bool = True, use_paddle: bool = True,
                 lang: str = 'en', gpu: bool = False):
        self.lang = lang
        self.gpu = gpu
        self.rapid_ocr = None
        self.paddle_ocr = None
        if use_rapid and RAPID_OCR_AVAILABLE:
            try:
                self.rapid_ocr = RapidOCR()
                logger.info("Initialized RapidOCR")
            except Exception as e:
                logger.warning(f"Failed to init RapidOCR: {e}")
        if use_paddle and PADDLE_OCR_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang,
                                            show_log=False, use_gpu=gpu)
                logger.info("Initialized PaddleOCR")
            except Exception as e:
                logger.warning(f"Failed to init PaddleOCR: {e}")
        if self.rapid_ocr is None and self.paddle_ocr is None:
            logger.warning("No OCR engine available!")

    def _ocr_with_rapid(self, image: np.ndarray) -> List[Dict]:
        if self.rapid_ocr is None:
            return []
        result, elapse = self.rapid_ocr(image)
        if result is None:
            return []
        boxes, texts, scores = result
        regions = []
        for box, text, score in zip(boxes, texts, scores):
            x_coords = [int(p[0]) for p in box]
            y_coords = [int(p[1]) for p in box]
            x, y = min(x_coords), min(y_coords)
            w = max(x_coords) - x
            h = max(y_coords) - y
            regions.append({
                'bbox': [x, y, w, h],
                'text': text,
                'confidence': float(score)
            })
        return regions

    def _ocr_with_paddle(self, image: np.ndarray) -> List[Dict]:
        if self.paddle_ocr is None:
            return []
        result = self.paddle_ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return []
        regions = []
        for line in result[0]:
            box = line[0]
            text, score = line[1]
            x_coords = [int(p[0]) for p in box]
            y_coords = [int(p[1]) for p in box]
            x, y = min(x_coords), min(y_coords)
            w = max(x_coords) - x
            h = max(y_coords) - y
            regions.append({
                'bbox': [x, y, w, h],
                'text': text,
                'confidence': float(score)
            })
        return regions

    def ocr(self, image: np.ndarray, high_res_for_low_conf: bool = True
            ) -> Tuple[List[Dict], float]:
        if image is None or image.size == 0:
            return [], 0.0
        regions1 = self._ocr_with_rapid(image) if self.rapid_ocr else []
        regions2 = self._ocr_with_paddle(image) if self.paddle_ocr else []
        if not regions1 and not regions2:
            return [], 0.0
        merged = self._merge_results(regions1, regions2)
        if high_res_for_low_conf and merged:
            low_conf_regions = [r for r in merged
                                if r['confidence'] < 0.5]
            if low_conf_regions and len(low_conf_regions) > len(merged) * 0.3:
                h, w = image.shape[:2]
                if max(h, w) < 2000:
                    scale = 2.0
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    try:
                        hi_res = cv2.resize(image, (new_w, new_h),
                                            interpolation=cv2.INTER_CUBIC)
                    except Exception:
                        hi_res = image
                    hi_res_regions1 = (self._ocr_with_rapid(hi_res)
                                       if self.rapid_ocr else [])
                    hi_res_regions2 = (self._ocr_with_paddle(hi_res)
                                       if self.paddle_ocr else [])
                    hi_res_merged = self._merge_results(
                        hi_res_regions1, hi_res_regions2)
                    if hi_res_merged:
                        for i, r in enumerate(merged):
                            if i < len(hi_res_merged) and r['confidence'] < 0.5:
                                if hi_res_merged[i]['confidence'] > r['confidence']:
                                    merged[i] = hi_res_merged[i]
        avg_conf = (np.mean([r['confidence'] for r in merged])
                    if merged else 0.0)
        return merged, float(avg_conf)

    def _merge_results(self, regions1: List[Dict],
                       regions2: List[Dict]) -> List[Dict]:
        if not regions1:
            return regions2
        if not regions2:
            return regions1
        merged = []
        used = set()
        for r1 in regions1:
            best_match = None
            best_overlap = 0
            for j, r2 in enumerate(regions2):
                if j in used:
                    continue
                overlap = self._bbox_overlap(r1['bbox'], r2['bbox'])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (j, r2)
            if best_match and best_overlap > 0.3:
                j, r2 = best_match
                used.add(j)
                chosen = (r1 if r1['confidence'] >= r2['confidence']
                          else r2)
                merged.append(chosen)
            else:
                merged.append(r1)
        for j, r2 in enumerate(regions2):
            if j not in used:
                merged.append(r2)
        return merged

    def _bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xi = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        yi = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = xi * yi
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / max(union, 1)


import cv2
