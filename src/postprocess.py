"""Turn a probability map into separate filament instances."""
import cv2
import numpy as np


def postprocess(prob, thr, min_area, close_k):
    """Threshold, optionally close small gaps, split into connected components,
    and drop components smaller than `min_area` pixels."""
    b = (prob > thr).astype(np.uint8)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    return [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
