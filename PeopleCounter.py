import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


class PeopleCounter:
    def __init__(self, model_path="../YoloWeights/yolov8n.pt", conf_threshold=0.3,
                 max_age=30, min_hits=3, iou_threshold=0.2):
        '''
        Args:
            model_path (str): Path to YOLO model weights
            conf_threshold (float): Confidence threshold for detection
            max_age (int): Maximum frames to keep track of objects
            min_hits (int): Minimum hits to start tracking
            iou_threshold (float): IOU threshold for tracking
        '''
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.class_names = self._init_class_names()
        self.person_states = {}
        self.count_upward = []
        self.count_downward = []

    def _init_class_names(self):
        return ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
                "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    def detect_people(self, img):
        results = self.model(img, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = math.ceil((box.conf[0] * 100))

                if self.class_names[cls] == 'person' and conf > self.conf_threshold:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        return detections

    def track_and_count(self, img, counting_line):
        # Detect and track people
        detections = self.detect_people(img)
        resultsTracker = self.tracker.update(detections)

        # Draw counting line
        cv2.line(img, (counting_line[0], counting_line[1]),
                 (counting_line[2], counting_line[3]), (0, 0, 255), 2)

        # Process each tracked person
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            w, h = x2 - x1, y2 - y1

            # Draw bounding box and ID
            cvzone.cornerRect(img, (x1, y1, w, h), rt=2)
            cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(40, y1)),
                               scale=1, thickness=1, offset=10)

            # Center point
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

            # Initialize state for new IDs
            if id not in self.person_states:
                self.person_states[id] = {
                    "last_y": cy,
                    "counted": False
                }

            # Count people crossing the line
            self._process_line_crossing(img, id, cy, counting_line)

            # Update last position
            self.person_states[id]["last_y"] = cy

        # Draw counts on image
        cvzone.putTextRect(img, f'Up: {len(self.count_upward)}. Down: {len(self.count_downward)}',
                           (5, 20), scale=1, thickness=2, colorR=(0, 255, 0))

        return img, len(self.count_upward), len(self.count_downward)

    def _process_line_crossing(self, img, id, cy, counting_line):
        if abs(cy - counting_line[1]) < 10 and not self.person_states[id]["counted"]:
            if cy < self.person_states[id]["last_y"]:
                self.count_upward.append(id)
                self._highlight_crossing_line(img, counting_line)
            elif cy > self.person_states[id]["last_y"]:
                self.count_downward.append(id)
                self._highlight_crossing_line(img, counting_line)
            self.person_states[id]["counted"] = True

        # Reset counting state if person moves far from line
        if abs(cy - counting_line[1]) > 30:
            self.person_states[id]["counted"] = False

    def _highlight_crossing_line(self, img, counting_line):
        cv2.line(img, (counting_line[0], counting_line[1]),
                 (counting_line[2], counting_line[3]), (0, 255, 0), 2)

    def reset_counts(self):
        self.count_upward = []
        self.count_downward = []
        self.person_states = {}


def main():
    # Initialize counter
    counter = PeopleCounter()

    # Set up video capture
    cap = cv2.VideoCapture('video.mp4')
    counting_line = [-50, 110, 500, 110] # x1, y1, x2, y2

    while True:
        success, img = cap.read()
        if not success:
            break

        # Process frame
        img, up_count, down_count = counter.track_and_count(img, counting_line)

        # Display results
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()