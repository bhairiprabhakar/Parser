from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentQualityProfile:
    """Immutable result produced by DocumentQualityAnalyzer."""
    __slots__ = (
        "blur_score", "skew_angle", "contrast_score",
        "has_shadow", "has_fold", "estimated_dpi",
        "is_low_contrast", "is_blurry", "needs_upscale",
        "width", "height",
        "rotation_degrees",
        "is_dull",
        "noise_level",
        "has_dark_background",
        "text_density",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        return (
            f"<QualityProfile blur={self.blur_score:.1f} skew={self.skew_angle:.2f} "
            f"contrast={self.contrast_score:.1f} shadow={self.has_shadow} "
            f"fold={self.has_fold} dpi={self.estimated_dpi} "
            f"rot={self.rotation_degrees} dull={self.is_dull} "
            f"noise={self.noise_level:.2f} dark_bg={self.has_dark_background}>"
        )


class DocumentQualityAnalyzer:
    _BLUR_THRESHOLD = 80.0
    _CONTRAST_THRESHOLD = 40.0
    _SHADOW_RATIO = 1.6
    _MIN_OCR_WIDTH = 1800

    _DULL_MEAN_HIGH = 185
    _DULL_MEAN_LOW = 80
    _DULL_STD_THRESHOLD = 28.0
    _NOISE_LAPLACIAN_HIGH = 600.0
    _DARK_BG_MEAN = 100

    def analyze(self, image_cv: np.ndarray) -> DocumentQualityProfile:
        h, w = image_cv.shape[:2]
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if len(image_cv.shape) == 3 else image_cv.copy()

        blur     = self._blur_score(gray)
        skew     = self._detect_skew(gray)
        contrast = self._contrast_score(gray)
        shadow   = self._detect_shadow(gray)
        fold     = self._detect_fold(gray)
        dpi      = self._estimate_dpi(w, h)

        rotation = self._detect_rotation(gray)
        dull     = self._detect_dull(gray)
        noise    = self._noise_level(gray, blur)
        dark_bg  = self._detect_dark_background(gray)
        text_den = self._text_density(gray)

        profile = DocumentQualityProfile(
            blur_score          = blur,
            skew_angle          = skew,
            contrast_score      = contrast,
            has_shadow          = shadow,
            has_fold            = fold,
            estimated_dpi       = dpi,
            is_blurry           = blur < self._BLUR_THRESHOLD,
            is_low_contrast     = contrast < self._CONTRAST_THRESHOLD,
            needs_upscale       = w < self._MIN_OCR_WIDTH,
            width=w, height=h,
            rotation_degrees    = rotation,
            is_dull             = dull,
            noise_level         = noise,
            has_dark_background = dark_bg,
            text_density        = text_den,
        )
        logger.info("Quality: %s", profile)
        return profile

    @staticmethod
    def _blur_score(gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _detect_skew(gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None:
            return 0.0
        angles = []
        for line in lines:
            rho, theta = line[0]
            a = np.degrees(theta) - 90
            if abs(a) < 45:
                angles.append(a)
        if not angles:
            return 0.0
        return float(np.median(angles))

    @staticmethod
    def _contrast_score(gray: np.ndarray) -> float:
        return float(gray.std())

    @staticmethod
    def _detect_shadow(gray: np.ndarray) -> bool:
        h, w = gray.shape
        h2, w2 = h // 2, w // 2
        tl = gray[:h2, :w2].mean()
        br = gray[h2:, w2:].mean()
        if br == 0:
            return False
        return (tl / br) > 1.6

    @staticmethod
    def _detect_fold(gray: np.ndarray) -> bool:
        h, w = gray.shape
        strip_w = max(w // 10, 20)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        center = sobelx[:, w // 2 - strip_w // 2 : w // 2 + strip_w // 2]
        left  = sobelx[:, :strip_w]
        right = sobelx[:, -strip_w:]
        center_var = center.var()
        edge_var   = np.mean([left.var(), right.var()])
        if edge_var == 0:
            return False
        return (center_var / edge_var) > 1.5

    @staticmethod
    def _estimate_dpi(w: int, h: int) -> int:
        short = min(w, h)
        return int(short / 8.27)

    @staticmethod
    def _detect_rotation(gray: np.ndarray) -> int:
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None:
            return 90 if w > h * 1.3 else 0
        near_h = 0
        near_v = 0
        for line in lines:
            rho, theta = line[0]
            a = np.degrees(theta)
            if abs(a - 90) < 20:
                near_h += 1
            if abs(a) < 20 or abs(a - 180) < 20:
                near_v += 1
        if w > h * 1.2:
            return 90 if near_v >= near_h else 270
        if near_h > 0 and near_v == 0:
            h_half = h // 2
            top_mean = np.mean(gray[:h_half])
            bot_mean = np.mean(gray[h_half:])
            return 180 if bot_mean > top_mean else 0
        return 0

    @staticmethod
    def _detect_dull(gray: np.ndarray) -> bool:
        m = gray.mean()
        s = gray.std()
        return (m > 185 and s < 28) or (m < 80 and s < 28)

    @staticmethod
    def _noise_level(gray: np.ndarray, blur_score: float) -> float:
        blur_comp = min(blur_score / 600.0, 1.0)
        residual = np.abs(gray.astype(np.float32) - np.median(gray)).mean() / 255.0
        return float(np.clip((blur_comp + residual) / 2.0, 0.0, 1.0))

    @staticmethod
    def _detect_dark_background(gray: np.ndarray) -> bool:
        return float(np.median(gray)) < 100

    @staticmethod
    def _text_density(gray: np.ndarray) -> float:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return float(np.count_nonzero(binary)) / float(binary.size)
