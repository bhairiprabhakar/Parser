import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    aligned_image: np.ndarray
    skew_angle: float
    confidence: float
    fold_regions: List[Tuple[int, int, int, int]]


class TopologicalAligner:
    """Corrects skew, slant, and detects fold regions in document images."""

    def __init__(self, min_confidence: float = 0.5,
                 fold_gradient_threshold: float = 50.0):
        self.min_confidence = min_confidence
        self.fold_gradient_threshold = fold_gradient_threshold

    def align(self, image: np.ndarray) -> AlignmentResult:
        if image is None or image.size == 0:
            return AlignmentResult(image, 0.0, 0.0, [])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        skew_angle = self._detect_skew(gray)
        aligned = self._rotate(image, skew_angle)
        fold_regions = self._detect_folds(gray)
        return AlignmentResult(
            aligned_image=aligned,
            skew_angle=skew_angle,
            confidence=self._compute_confidence(gray, skew_angle),
            fold_regions=fold_regions
        )

    def _detect_skew(self, gray: np.ndarray) -> float:
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10:
            return 0.0
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        return angle

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        if abs(angle) < 0.5:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, matrix, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _detect_folds(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = gray.shape
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        horizontal_proj = np.sum(grad_mag, axis=1)
        vertical_proj = np.sum(grad_mag, axis=0)
        fold_regions = []
        mean_h = np.mean(horizontal_proj)
        for i in range(1, len(horizontal_proj) - 1):
            if (horizontal_proj[i] > self.fold_gradient_threshold * mean_h and
                    horizontal_proj[i] > horizontal_proj[i - 1] * 2 and
                    horizontal_proj[i] > horizontal_proj[i + 1] * 2):
                y_start = max(0, i - 10)
                y_end = min(h, i + 10)
                fold_regions.append((0, y_start, w, y_end - y_start))
        mean_v = np.mean(vertical_proj)
        for j in range(1, len(vertical_proj) - 1):
            if (vertical_proj[j] > self.fold_gradient_threshold * mean_v and
                    vertical_proj[j] > vertical_proj[j - 1] * 2 and
                    vertical_proj[j] > vertical_proj[j + 1] * 2):
                x_start = max(0, j - 10)
                x_end = min(w, j + 10)
                fold_regions.append((x_start, 0, x_end - x_start, h))
        return fold_regions

    def _compute_confidence(self, gray: np.ndarray,
                            skew_angle: float) -> float:
        if abs(skew_angle) > 15:
            return 0.3
        if abs(skew_angle) > 5:
            return 0.7
        return 0.95
