import cv2
import numpy as np
import re
from typing import List, Tuple, Optional
import logging
logger = logging.getLogger(__name__)


class TopologicalAligner:
    _HEADER_PASSTHROUGH_KEYWORDS = [
        "DURGA DRUGS", "Party & Product", "From 01",
        "From Date", "To Date", "Sale Report",
    ]

    def align(self, rapid_result: list) -> list:
        if not rapid_result:
            return []

        nodes = []
        for item in rapid_result:
            box, text, confidence = item
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            nodes.append({
                'x_min':    x_min,
                'x_max':    x_max,
                'y_min':    y_min,
                'y_max':    y_max,
                'y_center': (y_min + y_max) / 2.0,
                'height':   y_max - y_min,
                'text':     str(text).strip(),
            })

        if not nodes:
            return []

        median_height      = float(np.median([n['height'] for n in nodes]))
        vertical_tolerance = median_height * 0.45

        nodes.sort(key=lambda n: (n['y_center'], n['x_min']))
        structured_rows: list = []

        for node in nodes:
            placed = False
            for row in structured_rows:
                closest = min(row, key=lambda r: abs(r['x_min'] - node['x_min']))
                if abs(node['y_center'] - closest['y_center']) <= vertical_tolerance:
                    row.append(node)
                    placed = True
                    break
            if not placed:
                structured_rows.append([node])

        char_width_px = max(1.0, median_height * 0.4)

        final_lines: list = []
        for row in structured_rows:
            row.sort(key=lambda n: n['x_min'])
            combined_text = " ".join(n['text'] for n in row)

            if any(kw in combined_text for kw in self._HEADER_PASSTHROUGH_KEYWORDS):
                final_lines.append(combined_text)
                continue

            constructed_row   = ""
            current_char_pos  = 0

            for node in row:
                target_char_pos = int(node['x_min'] / char_width_px)
                if target_char_pos > current_char_pos:
                    constructed_row += " " * (target_char_pos - current_char_pos)
                elif current_char_pos > 0:
                    constructed_row += " "
                constructed_row  += node['text']
                current_char_pos  = len(constructed_row)

            final_lines.append(constructed_row.rstrip())

        return final_lines

    def to_text(self, rapid_result: list) -> str:
        lines = self.align(rapid_result)
        tab_lines = [re.sub(r'   +', '\t', line) for line in lines]
        return "\n".join(tab_lines)
