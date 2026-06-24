import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging
logger = logging.getLogger(__name__)


@dataclass
class DocumentQualityProfile:
    blur_ratio: float = 0.0
    skew_angle: float = 0.0
    contrast: float = 0.0
    brightness: float = 0.0
    noise_level: float = 0.0
    resolution: int = 0
    fold_present: bool = False
    overall_score: float = 0.0
    flags: list = field(default_factory=list)


class DocumentQualityAnalyzer:
    """Pre-OCR quality assessment for document images."""

    def __init__(self, blur_threshold: float = 15.0,
                 contrast_min: float = 20.0,
                 brightness_min: float = 30.0,
                 brightness_max: float = 230.0,
                 noise_threshold: float = 50.0,
                 resolution_min: int = 150):
        self.blur_threshold = blur_threshold
        self.contrast_min = contrast_min
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.noise_threshold = noise_threshold
        self.resolution_min = resolution_min

    def analyze(self, image: np.ndarray, filename: str = "") -> DocumentQualityProfile:
        profile = DocumentQualityProfile()
        if image is None or image.size == 0:
            profile.flags.append("EMPTY_IMAGE")
            profile.overall_score = 0.0
            return profile
        profile.blur_ratio = self._detect_blur(image)
        profile.skew_angle = self._detect_skew(image)
        profile.contrast = self._compute_contrast(image)
        profile.brightness = self._compute_brightness(image)
        profile.noise_level = self._estimate_noise(image)
        profile.resolution = max(image.shape[0], image.shape[1])
        profile.fold_present = self._detect_fold(image)
        profile.flags = self._generate_flags(profile)
        profile.overall_score = self._compute_overall(profile)
        return profile

    def _detect_blur(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _detect_skew(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10:
            return 0.0
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        return angle

    def _compute_contrast(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return gray.std()

    def _compute_brightness(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return gray.mean()

    def _estimate_noise(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).std()

    def _detect_fold(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
        if lines is None:
            return False
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        fold_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            dist = np.sqrt((mid_x - center_x) ** 2 + (mid_y - center_y) ** 2)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist < min(h, w) * 0.4 and length > min(h, w) * 0.3:
                fold_count += 1
        return fold_count >= 2

    def _generate_flags(self, profile: DocumentQualityProfile) -> list:
        flags = []
        if profile.blur_ratio < self.blur_threshold:
            flags.append("BLURRY")
        if abs(profile.skew_angle) > 5:
            flags.append(f"SKEWED_{profile.skew_angle:.1f}deg")
        if profile.contrast < self.contrast_min:
            flags.append("LOW_CONTRAST")
        if profile.brightness < self.brightness_min:
            flags.append("TOO_DARK")
        if profile.brightness > self.brightness_max:
            flags.append("TOO_BRIGHT")
        if profile.noise_level > self.noise_threshold:
            flags.append("NOISY")
        if profile.resolution < self.resolution_min:
            flags.append("LOW_RESOLUTION")
        if profile.fold_present:
            flags.append("FOLD_DETECTED")
        return flags

    def _compute_overall(self, profile: DocumentQualityProfile) -> float:
        score = 100.0
        if profile.blur_ratio < self.blur_threshold:
            score -= 20
        if abs(profile.skew_angle) > 5:
            score -= 10
        if profile.contrast < self.contrast_min:
            score -= 15
        if profile.brightness < self.brightness_min:
            score -= 10
        if profile.brightness > self.brightness_max:
            score -= 10
        if profile.noise_level > self.noise_threshold:
            score -= 15
        if profile.resolution < self.resolution_min:
            score -= 10
        if profile.fold_present:
            score -= 10
        return max(0.0, score)
