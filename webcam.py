"""Live detection and CSV-based runtime benchmarking. Press q to exit."""
import argparse
import csv
import time
from pathlib import Path
import cv2
from detectors import create_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['haar', 'torch'], default='haar')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--neighbors', type=int, default=5)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--log', default='results/webcam.csv')
    args = parser.parse_args()
    detector = create_detector(args.model, args.threshold, args.neighbors, args.device)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if __import__('sys').platform == 'win32' else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam; check privacy permissions or --camera')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'width', 'height', 'detections', 'inference_ms', 'loop_fps'])
            last_time = time.perf_counter()
            index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                start = time.perf_counter()
                boxes = detector.detect(frame)
                inference_ms = (time.perf_counter() - start) * 1000
                now = time.perf_counter()
                fps = 1.0 / max(now - last_time, 1e-9)
                last_time = now
                for x1, y1, x2, y2, confidence in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence:.2f}', (x1, max(15, y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f'{args.model} | {fps:.1f} loop FPS | {inference_ms:.1f} ms',
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
                cv2.imshow('Face Detection - q to quit', frame)
                writer.writerow([index, frame.shape[1], frame.shape[0], len(boxes),
                                 round(inference_ms, 3), round(fps, 3)])
                index += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f'Saved runtime measurements to {log_path}')

if __name__ == '__main__':
    main()
