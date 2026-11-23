"""Panoptic-Quality-style metric: sum(IoU of matches) / (TP + 0.5 FP + 0.5 FN)."""


def match(gts, preds, iou_thr=0.5):
    """Greedy one-to-one matching. Returns (iou_sum, TP, FP, FN)."""
    used, iou_sum, tp = set(), 0.0, 0
    for g in gts:
        best, best_j = 0.0, -1
        for j, p in enumerate(preds):
            if j in used:
                continue
            inter = (g & p).sum()
            if inter == 0:
                continue
            iou = inter / (g | p).sum()
            if iou > best:
                best, best_j = iou, j
        if best > iou_thr:
            used.add(best_j)
            iou_sum += best
            tp += 1
    return iou_sum, tp, len(preds) - tp, len(gts) - tp


def panoptic_quality(all_gts, all_preds):
    s = tp = fp = fn = 0
    for gts, preds in zip(all_gts, all_preds):
        a, b, c, d = match(gts, preds)
        s, tp, fp, fn = s + a, tp + b, fp + c, fn + d
    return s / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)
