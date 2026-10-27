"""Small geometry helpers:
  smart_resize   -- image size rule of the Qwen2.5-VL processor (multiples of `factor`, area within [min, max])
  xywh_to_xyxy   -- box format conversion
  scale_box      -- scale a box to a resized image
  parse_box      -- parse the first box [x1, y1, x2, y2] from a model's text output
"""
import math
import re


def smart_resize(h, w, factor, min_pixels, max_pixels):
    """Same rule as the Qwen2.5-VL / Qwen3-VL processor: side lengths are multiples of factor, area within [min, max]."""
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def scale_box(box, sx, sy):
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


_NUM = r"-?\d+(?:\.\d+)?"


def parse_box(text):
    """First box [x1, y1, x2, y2] in a model output: "bbox_2d": [..] first, then any bracketed 4 numbers,
    then (x1,y1),(x2,y2). Returns None if nothing is found. Guarantees x1 <= x2 and y1 <= y2."""
    if not text:
        return None
    pats = [r'"bbox_2d"\s*:\s*\[\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*\]',
            r'[\[\(]\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*[\]\)]',
            r'\(\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*\)\s*,?\s*\(\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*\)']
    for p in pats:
        m = re.search(p, text)
        if m:
            x1, y1, x2, y2 = (float(g) for g in m.groups())
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return None
