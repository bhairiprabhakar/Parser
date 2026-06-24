import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    confidence: float = 0.0
    line_num: int = 0
    is_table: bool = False
    is_header: bool = False
    is_footer: bool = False


class SpatialClusterer:
    """Groups OCR text regions into logical clusters using spatial analysis."""

    def __init__(self, x_tolerance: int = 30, y_tolerance: int = 10,
                 min_cluster_size: int = 2,
                 header_region_ratio: float = 0.15,
                 footer_region_ratio: float = 0.1):
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance
        self.min_cluster_size = min_cluster_size
        self.header_region_ratio = header_region_ratio
        self.footer_region_ratio = footer_region_ratio

    def cluster(self, regions: List[TextRegion],
                image_shape: Tuple[int, int]) -> List[List[TextRegion]]:
        if not regions:
            return []
        sorted_regions = sorted(regions, key=lambda r: (r.y, r.x))
        clusters = []
        current_cluster = [sorted_regions[0]]
        for region in sorted_regions[1:]:
            prev = current_cluster[-1]
            if (region.y - (prev.y + prev.h)) <= self.y_tolerance:
                current_cluster.append(region)
            else:
                if len(current_cluster) >= self.min_cluster_size:
                    clusters.append(current_cluster)
                current_cluster = [region]
        if len(current_cluster) >= self.min_cluster_size:
            clusters.append(current_cluster)
        return self._refine_clusters(clusters, image_shape)

    def _refine_clusters(self, clusters: List[List[TextRegion]],
                         image_shape: Tuple[int, int]) -> List[List[TextRegion]]:
        h, w = image_shape[:2]
        result = []
        for cluster in clusters:
            cluster_x = min(r.x for r in cluster)
            cluster_y = min(r.y for r in cluster)
            cluster_w = max(r.x + r.w for r in cluster) - cluster_x
            cluster_h = max(r.y + r.h for r in cluster) - cluster_y
            if cluster_y < h * self.header_region_ratio:
                for r in cluster:
                    r.is_header = True
            elif cluster_y > h * (1 - self.footer_region_ratio):
                for r in cluster:
                    r.is_footer = True
            result.append(cluster)
        return result

    def detect_table_structure(self, regions: List[TextRegion],
                               image_shape: Tuple[int, int]) -> List[TextRegion]:
        if not regions:
            return regions
        h, w = image_shape[:2]
        rows = defaultdict(list)
        for region in sorted(regions, key=lambda r: (r.y, r.x)):
            row_key = round(region.y / self.y_tolerance)
            rows[row_key].append(region)
        table_regions = []
        for row_key, row_regions in rows.items():
            if len(row_regions) >= 3:
                row_regions.sort(key=lambda r: r.x)
                gaps = []
                for i in range(len(row_regions) - 1):
                    gap = row_regions[i + 1].x - (row_regions[i].x + row_regions[i].w)
                    gaps.append(gap)
                mean_gap = np.mean(gaps) if gaps else 0
                if mean_gap > self.x_tolerance * 0.5:
                    for r in row_regions:
                        r.is_table = True
                        table_regions.append(r)
        if table_regions:
            for r in regions:
                if r in table_regions:
                    r.is_table = True
        return regions
