"""Bounding-box IoU and greedy one-to-one detection matching."""

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match(predictions, truths, minimum_iou):
    candidates = sorted(((iou(p, t), pi, ti) for pi, p in enumerate(predictions)
                         for ti, t in enumerate(truths)), reverse=True)
    used_pred, used_truth = set(), set()
    for overlap, pi, ti in candidates:
        if overlap < minimum_iou:
            break
        if pi not in used_pred and ti not in used_truth:
            used_pred.add(pi)
            used_truth.add(ti)
    return len(used_pred), len(predictions)-len(used_pred), len(truths)-len(used_truth)

