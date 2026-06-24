import cv2
import numpy as np
from typing import Optional, Tuple
import logging
logger = logging.getLogger(__name__)


class AdaptivePreprocessor:
    _ROTATE_MAP = {
        90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }

    def process(self, image_cv: np.ndarray,
                profile) -> np.ndarray:
        h_orig, w_orig = image_cv.shape[:2]

        rot = getattr(profile, 'rotation_degrees', 0)
        if rot in self._ROTATE_MAP:
            image_cv = cv2.rotate(image_cv, self._ROTATE_MAP[rot])
            logger.info("Rotated %d deg to upright orientation", rot)

        if profile.needs_upscale:
            scale = 1800.0 / w_orig
            image_cv = cv2.resize(image_cv, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            logger.info("Upscaled %.1fx, width=%dpx, dpi=%d",
                        scale, int(w_orig * scale), profile.estimated_dpi)

        angle = profile.skew_angle
        if 0.5 < abs(angle) < 45:
            image_cv = self._deskew(image_cv, angle)
            logger.info("Deskewed %.2f deg", angle)

        gray = (cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                if len(image_cv.shape) == 3 else image_cv.copy())

        if getattr(profile, 'has_dark_background', False):
            gray = cv2.bitwise_not(gray)
            logger.info("Dark-background inversion applied")

        if profile.has_shadow:
            gray = self._normalize_illumination(gray)
            logger.info("Shadow normalisation applied")

        if getattr(profile, 'is_dull', False):
            gray = self._enhance_dull(gray)
            logger.info("Dull-page enhancement applied")

        if profile.is_low_contrast or profile.has_shadow:
            clip = max(1.5, min(4.0, 80.0 / max(profile.contrast_score, 1.0)))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        noise = getattr(profile, 'noise_level', 0.0)
        if noise > 0.4:
            d = 7 if noise > 0.7 else 5
            gray = cv2.bilateralFilter(gray, d=d,
                                       sigmaColor=int(30 + noise * 40),
                                       sigmaSpace=int(30 + noise * 40))
            logger.info("Bilateral noise filter applied noise=%.2f d=%d", noise, d)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if profile.is_blurry or profile.needs_upscale:
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
            gray = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)
            logger.debug("Unsharp-mask sharpening applied")

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _deskew(image_cv: np.ndarray, angle: float) -> np.ndarray:
        (h, w) = image_cv.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
        return cv2.warpAffine(image_cv, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

    @staticmethod
    def _normalize_illumination(gray: np.ndarray) -> np.ndarray:
        gray_f = gray.astype(np.float32) + 1.0
        ksize = max(gray.shape[0] // 8, gray.shape[1] // 8, 31)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        background = cv2.GaussianBlur(gray_f, (ksize, ksize), 0)
        normalised = (gray_f / background) * 128.0
        normalised = np.clip(normalised, 0, 255).astype(np.uint8)
        return normalised

    @staticmethod
    def _enhance_dull(gray: np.ndarray) -> np.ndarray:
        lut = np.array([
            min(255, int(255.0 * (i / 255.0) ** 0.5))
            for i in range(256)
        ], dtype=np.uint8)
        gray = cv2.LUT(gray, lut)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        return clahe.apply(gray)
