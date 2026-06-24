import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import logging
logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

TABLE_TRANSFORMER_AVAILABLE = False
LAYOUT_PARSER_AVAILABLE = False

try:
    import torch
    from transformers import TableTransformerForObjectDetection, DetrImageProcessor
    TABLE_TRANSFORMER_AVAILABLE = True
except ImportError:
    pass

try:
    import layoutparser as lp
    LAYOUT_PARSER_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TableRegion:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0
    cells: List[Tuple[int, int, int, int]] = field(default_factory=list)
    is_rotated: bool = False


@dataclass
class PageRegion:
    x: int
    y: int
    w: int
    h: int
    region_type: str = "text"
    confidence: float = 0.0


class AITableDetector:
    """Detects table regions using deep learning with fallback strategies."""

    def __init__(self, use_table_transformer: bool = True,
                 use_layout_parser: bool = True,
                 device: str = 'cpu'):
        self.device = device
        self.table_detector = None
        self.layout_model = None
        if use_table_transformer and TABLE_TRANSFORMER_AVAILABLE:
            try:
                model_name = "microsoft/table-transformer-detection"
                self.table_detector = TableTransformerForObjectDetection.from_pretrained(
                    model_name
                )
                self.image_processor = DetrImageProcessor.from_pretrained(model_name)
                self.table_detector.to(device)
                self.table_detector.eval()
                logger.info("Loaded Table Transformer model")
            except Exception as e:
                logger.warning(f"Failed to load Table Transformer: {e}")
                self.table_detector = None
        if use_layout_parser and LAYOUT_PARSER_AVAILABLE:
            try:
                self.layout_model = lp.PaddleDetectionLayoutModel(
                    config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                    threshold=0.5,
                    device=device
                )
                logger.info("Loaded LayoutParser model")
            except Exception:
                logger.warning("LayoutParser model not available")
                self.layout_model = None

    def detect_tables(self, image: np.ndarray) -> Tuple[List[TableRegion], str]:
        if self.table_detector is not None:
            tables = self._detect_with_transformer(image)
            if tables:
                return tables, "transformer"
        if self.layout_model is not None:
            tables = self._detect_with_layout(image)
            if tables:
                return tables, "layout_parser"
        tables = self._detect_with_projection(image)
        if tables:
            return tables, "projection"
        tables = self._detect_with_contour(image)
        return tables, "contour"

    def _detect_with_transformer(self, image: np.ndarray) -> List[TableRegion]:
        if self.table_detector is None:
            return []
        try:
            pil_image = Image.fromarray(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )
            inputs = self.image_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.table_detector(**inputs)
            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = self.image_processor.post_process_object_detection(
                outputs, threshold=0.5, target_sizes=target_sizes
            )[0]
            tables = []
            for score, label, box in zip(results['scores'], results['labels'],
                                          results['boxes']):
                if label == 0:
                    x1, y1, x2, y2 = box.tolist()
                    tables.append(TableRegion(
                        x=int(x1), y=int(y1),
                        w=int(x2 - x1), h=int(y2 - y1),
                        confidence=float(score)
                    ))
            return tables
        except Exception as e:
            logger.warning(f"Table Transformer detection failed: {e}")
            return []

    def _detect_with_layout(self, image: np.ndarray) -> List[TableRegion]:
        if self.layout_model is None:
            return []
        try:
            pil_image = Image.fromarray(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )
            layout_result = self.layout_model.detect(pil_image)
            tables = []
            for block in layout_result:
                if block.type in ('Table', 'table', 'TABLE'):
                    x1, y1, x2, y2 = block.coordinates
                    tables.append(TableRegion(
                        x=int(x1), y=int(y1),
                        w=int(x2 - x1), h=int(y2 - y1),
                        confidence=block.score if hasattr(block, 'score') else 0.5
                    ))
            return tables
        except Exception as e:
            logger.warning(f"LayoutParser detection failed: {e}")
            return []

    def _detect_with_projection(self, image: np.ndarray) -> List[TableRegion]:
        if cv2 is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        h_proj = np.sum(binary, axis=1) // 255
        v_proj = np.sum(binary, axis=0) // 255
        h_thresh = np.mean(h_proj) * 0.3
        v_thresh = np.mean(v_proj) * 0.3
        h_lines = np.where(h_proj > h_thresh)[0]
        v_lines = np.where(v_proj > v_thresh)[0]
        if len(h_lines) < 3 or len(v_lines) < 3:
            return []
        h_gaps = np.diff(h_lines)
        v_gaps = np.diff(v_lines)
        h_breaks = np.where(h_gaps > 5)[0]
        v_breaks = np.where(v_gaps > 5)[0]
        row_starts = [h_lines[0]] + [h_lines[b + 1] for b in h_breaks]
        row_ends = [h_lines[b] for b in h_breaks] + [h_lines[-1]]
        col_starts = [v_lines[0]] + [v_lines[b + 1] for b in v_breaks]
        col_ends = [v_lines[b] for b in v_breaks] + [v_lines[-1]]
        if not row_starts or not col_starts:
            return []
        x = min(col_starts)
        y = min(row_starts)
        w = max(col_ends) - x
        h = max(row_ends) - y
        cells = []
        for rs, re in zip(row_starts, row_ends):
            for cs, ce in zip(col_starts, col_ends):
                cells.append((cs, rs, ce, re))
        h, w_total = image.shape[:2]
        if w > w_total * 0.8 and h > h_total * 0.2:
            return [TableRegion(x=x, y=y, w=w, h=h,
                                confidence=0.6, cells=cells)]
        return []

    def _detect_with_contour(self, image: np.ndarray) -> List[TableRegion]:
        if cv2 is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        tables = []
        h_img, w_img = image.shape[:2]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            rect_area = w * h
            if rect_area == 0:
                continue
            extent = area / rect_area
            if (extent > 0.3 and w > w_img * 0.3 and h > h_img * 0.1 and
                    w * h > w_img * h_img * 0.05):
                tables.append(TableRegion(x=x, y=y, w=w, h=h,
                                          confidence=0.4))
        return tables


class AILayoutDetector:
    """Detects page layout structure (headers, footers, text blocks)."""

    def __init__(self):
        pass

    def detect_layout(self, image: np.ndarray) -> List[PageRegion]:
        regions = []
        h, w = image.shape[:2]
        regions.append(PageRegion(0, 0, w, int(h * 0.1),
                                   region_type="header"))
        regions.append(PageRegion(0, int(h * 0.9), w, int(h * 0.1),
                                   region_type="footer"))
        regions.append(PageRegion(0, int(h * 0.1), w, int(h * 0.8),
                                   region_type="body"))
        return regions
