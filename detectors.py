"""Two interchangeable face detectors returning (x1,y1,x2,y2,score)."""
import cv2


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


def create_detector(name, threshold=0.85, neighbors=5, device='auto'):
    if name == 'haar':
        return HaarDetector(neighbors=neighbors)
    raise ValueError('Unknown detector: ' + name)
