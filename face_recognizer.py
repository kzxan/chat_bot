import face_recognition
import cv2
import os
import numpy as np
from text_helper import put_kazakh_text

class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces"):
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, directory):
        """known_faces папкасындағы фотоларды жүктеу"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("⚠ known_faces папкасы жасалды — фотоларды салыңыз!")
            return

        for filename in os.listdir(directory):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(directory, filename)
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    self.known_encodings.append(encodings[0])
                    # Файл атынан есім алу (adam1.jpg → adam1)
                    name = os.path.splitext(filename)[0]
                    self.known_names.append(name)
                    print(f"✅ Жүктелді: {name}")
                else:
                    print(f"⚠ Бет табылмады: {filename}")

        print(f"✅ Барлығы {len(self.known_names)} адам жүктелді")

    def recognize(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        unknown_found = False  # ✅ Жаңа

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 2; right *= 2; bottom *= 2; left *= 2

            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.5)
            name = None

            if True in matches:
                import numpy as np
                distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match = np.argmin(distances)
                if matches[best_match]:
                    name = self.known_names[best_match]

            if name:
                color = (0, 255, 0)
                label = f"{name}"
            else:
                color = (180, 105, 255)
                label = "Белгісіз!"
                unknown_found = True  # ✅

                # Скрин сақтау
                face_crop = frame[max(0, top):bottom, max(0, left):right]
                if face_crop.size > 0:
                    self.save_unknown(frame, face_crop)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom), (right, bottom + 40), color, -1)
            frame = put_kazakh_text(frame, label,
                                    (left + 5, bottom + 5),
                                    font_size=22, color=(255, 255, 255))

        return frame, unknown_found  # ✅ Екі мән қайтарады