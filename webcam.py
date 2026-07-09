"""Live Haar and MTCNN face detection. Press q to exit."""
import argparse
import sys
import cv2
from detectors import create_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['haar', 'torch'], default='haar')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    detector = create_detector(args.model, device=args.device)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam; check privacy permissions or --camera')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for x1, y1, x2, y2, confidence in detector.detect(frame):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('Face Detection - q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
