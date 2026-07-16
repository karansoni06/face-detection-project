"""Evaluate detectors with CSV ground truth: image,x1,y1,x2,y2 (pixel coordinates)."""
import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path
import cv2
from detectors import create_detector


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['haar', 'torch'], required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--neighbors', type=int, default=5)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, default=Path('results/evaluation.csv'))
    args = parser.parse_args()
    annotations = defaultdict(list)
    with args.annotations.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            name = row['image']
            if name.startswith('/') or '..' in Path(name).parts:
                raise ValueError('Invalid relative image path')
            annotations[name].append(tuple(int(row[k]) for k in ('x1','y1','x2','y2')))
    if not annotations:
        raise ValueError('No annotations found')
    detector = create_detector(args.model, args.threshold, args.neighbors, args.device)
    tp = fp = fn = 0
    latencies = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image','ground_truth','predictions','tp','fp','fn','inference_ms'])
        for name, truths in annotations.items():
            frame = cv2.imread(str(args.images / name))
            if frame is None:
                raise FileNotFoundError(args.images / name)
            start = time.perf_counter()
            preds = detector.detect(frame)
            latency = 1000 * (time.perf_counter() - start)
            matches, false_pos, false_neg = match(preds, truths, args.iou)
            tp += matches
            fp += false_pos
            fn += false_neg
            latencies.append(latency)
            writer.writerow([name, len(truths), len(preds), matches,
                             false_pos, false_neg, round(latency, 3)])
    precision = tp / (tp+fp) if tp+fp else 0
    recall = tp / (tp+fn) if tp+fn else 0
    f1 = 2*precision*recall / (precision+recall) if precision+recall else 0
    print(f'Model: {args.model} | Images: {len(annotations)} | TP={tp} FP={fp} FN={fn}')
    print(f'Precision={precision:.3%} Recall={recall:.3%} F1={f1:.3%}')
    print(f'Mean inference latency={statistics.mean(latencies):.2f} ms')
    print(f'Detailed results: {args.output}')

if __name__ == '__main__':
    main()
