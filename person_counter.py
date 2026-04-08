from ultralytics import YOLO
import cv2
from text_helper import put_kazakh_text

class PersonCounter:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        print("✅ Адам санау моделі жүктелді")

    def count(self, frame):
        results = self.model(frame, classes=[0], verbose=False)
        count = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Жасыл төртбұрыш
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # ✅ Қазақша мәтін
                frame = put_kazakh_text(frame,
                                        f"Адам {conf:.0%}",
                                        (x1, max(y1 - 35, 0)),
                                        font_size=25,
                                        color=(0, 255, 0))

        return count, frame