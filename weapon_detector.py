from ultralytics import YOLO
import cv2
from text_helper import put_kazakh_text

class WeaponDetector:
    def __init__(self):
        # ❗ Өз моделіңді қос
        self.model = YOLO("weapon_model.pt")

        # ❗ Егер өзің train жасаған болсаң — класстарды өзгерт
        # Мысалы: 0 = weapon
        self.weapon_classes = [0]  

        print("✅ Қару тану моделі жүктелді")

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        weapon_found = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])

                if cls in self.weapon_classes:
                    weapon_found = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # Қызыл төртбұрыш
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # Қазақша текст
                    frame = put_kazakh_text(
                        frame,
                        f"ҚАРУ! {conf:.0%}",
                        (x1, max(y1 - 35, 0)),
                        font_size=25,
                        color=(0, 0, 255)
                    )

        return weapon_found, frame