import cv2
import numpy as np
from typing import Optional, Tuple
import logging
logger = logging.getLogger(__name__)


class AdaptivePreprocessor:
    """Adaptive image preprocessing for OCR based on quality profile."""

    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi

    def preprocess(self, image: np.ndarray, quality_profile=None) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        img = self._resize_to_target(image)
        if quality_profile:
            if "BLURRY" in (quality_profile.flags or []):
                img = self._sharpen(img)
            if "LOW_CONTRAST" in (quality_profile.flags or []):
                img = self._enhance_contrast(img)
            if "TOO_DARK" in (quality_profile.flags or []):
                img = self._adjust_brightness(img, 1.2, 30)
            if "TOO_BRIGHT" in (quality_profile.flags or []):
                img = self._adjust_brightness(img, 0.8, -30)
            if "NOISY" in (quality_profile.flags or []):
                img = self._denoise(img)
            if "FOLD_DETECTED" in (quality_profile.flags or []):
                img = self._correct_fold(img)
        else:
            img = self._auto_enhance(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def _resize_to_target(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        current_dpi = max(h, w) / 8.27
        if current_dpi < self.target_dpi * 0.8:
            scale = self.target_dpi / current_dpi
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif current_dpi > self.target_dpi * 1.5:
            scale = self.target_dpi / current_dpi
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _adjust_brightness(self, image: np.ndarray, alpha: float = 1.0,
                           beta: int = 0) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def _correct_fold(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=100, maxLineGap=50)
        if lines is not None:
            mask = np.ones_like(gray) * 255
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask, (x1, y1), (x2, y2), 0, 5)
            result = cv2.inpaint(image, 255 - mask, 3, cv2.INPAINT_TELEA)
            return result
        return image

    def _auto_enhance(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 15:
            image = self._sharpen(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return image
