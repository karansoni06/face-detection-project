"""Two interchangeable face detectors returning (x1,y1,x2,y2,score)."""
import cv2
import numpy as np


class HaarDetector:
    def __init__(self, scale=1.1, neighbors=5):
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.model = cv2.CascadeClassifier(path)
        if self.model.empty():
            raise RuntimeError('Could not load Haar cascade: ' + path)
        self.scale = scale
        self.neighbors = neighbors

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rectangles = self.model.detectMultiScale(
            gray, scaleFactor=self.scale, minNeighbors=self.neighbors,
            minSize=(30, 30))
        return [(int(x), int(y), int(x + w), int(y + h), 1.0)
                for (x, y, w, h) in rectangles]


class TorchDetector:
    def __init__(self, threshold=0.85, device='auto'):
        import torch
        from facenet_pytorch import MTCNN
        self.torch = torch
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model = MTCNN(keep_all=True, device=self.device)
        self.threshold = threshold

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode():
            boxes, probabilities = self.model.detect(rgb)
        if boxes is None:
            return []
        height, width = frame.shape[:2]
        results = []
        for box, score in zip(boxes, probabilities):
            if score is None or not np.isfinite(score) or score < self.threshold:
                continue
            x1, y1, x2, y2 = box
            results.append((max(0, int(x1)), max(0, int(y1)),
                            min(width, int(x2)), min(height, int(y2)), float(score)))
        return results


def create_detector(name, threshold=0.85, neighbors=5, device='auto'):
    if name == 'haar':
        return HaarDetector(neighbors=neighbors)
    if name == 'torch':
        return TorchDetector(threshold=threshold, device=device)
    raise ValueError('Unknown detector: ' + name)
