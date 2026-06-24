import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
logger = logging.getLogger(__name__)


class MultiPassOCR:
    def __init__(self):
        self._scorer = None
        self._clusterer = None
        self._topo = None

    def _lazy_init(self):
        if self._scorer is None:
            from .confidence_scorer import ConfidenceScorer
            from .spatial_clusterer import SpatialClusterer
            from .topological_aligner import TopologicalAligner
            self._scorer = ConfidenceScorer()
            self._clusterer = SpatialClusterer()
            self._topo = TopologicalAligner()

    _EARLY_EXIT_CONFIDENCE = 0.72

    def run(self, image_cv: np.ndarray, engine) -> tuple:
        self._lazy_init()
        passes = self._generate_passes(image_cv)
        best_text, best_conf = "", 0.0

        for idx, (name, variant) in enumerate(passes):
            try:
                result, _ = engine(variant)
                if not result:
                    continue

                pass_conf = self._scorer.score_result(result)

                try:
                    blocks = self._clusterer.extract_blocks(result)
                    text_sc = self._clusterer.blocks_to_text(blocks)
                    combined_sc = 0.6 * pass_conf + 0.4 * self._scorer.score_text(text_sc)
                except Exception:
                    text_sc, combined_sc = "", 0.0

                try:
                    text_ta = self._topo.to_text(result)
                    combined_ta = 0.6 * pass_conf + 0.4 * self._scorer.score_text(text_ta)
                except Exception:
                    text_ta, combined_ta = "", 0.0

                if combined_ta > combined_sc:
                    text_best, combined_best = text_ta, combined_ta
                else:
                    text_best, combined_best = text_sc, combined_sc

                if combined_best > best_conf:
                    best_text, best_conf = text_best, combined_best

                if idx == 0 and best_conf >= self._EARLY_EXIT_CONFIDENCE:
                    logger.debug("MultiPass: early exit after pass-1 conf=%.3f", best_conf)
                    break

            except Exception as e:
                logger.debug("MultiPass [%s] failed: %s", name, e)

        return best_text, best_conf

    @staticmethod
    def _generate_passes(base: np.ndarray) -> list:
        gray_bgr = (cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY),
                                  cv2.COLOR_GRAY2BGR)
                    if len(base.shape) == 3 else
                    cv2.cvtColor(base, cv2.COLOR_GRAY2BGR))
        gray     = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2GRAY)
        passes   = [("adaptive", base)]

        try:
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
            sharp   = cv2.addWeighted(gray, 3.0, blurred, -2.0, 0)
            passes.append(("sharpened", cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

        try:
            clahe  = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            boosted = clahe.apply(gray)
            passes.append(("clahe_high", cv2.cvtColor(boosted, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

        try:
            lut    = np.array([min(255, int(((i / 255.0) ** 0.6) * 255))
                               for i in range(256)], dtype=np.uint8)
            gamma  = cv2.LUT(gray, lut)
            passes.append(("gamma", cv2.cvtColor(gamma, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

        try:
            bilateral = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
            passes.append(("bilateral", cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

        try:
            lut_dull = np.array([
                min(255, int(255.0 * (i / 255.0) ** 0.55))
                for i in range(256)
            ], dtype=np.uint8)
            dull_gamma = cv2.LUT(gray, lut_dull)
            clahe_hi   = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
            dull_enh   = clahe_hi.apply(dull_gamma)
            passes.append(("dull_enhance", cv2.cvtColor(dull_enh, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

        return passes
