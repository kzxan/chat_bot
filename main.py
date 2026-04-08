import cv2
import os
from dotenv import load_dotenv  # .env файлын оқу үшін
from person_counter import PersonCounter
from weapon_detector import WeaponDetector
from face_recognizer import FaceRecognizer
from whatsapp_alert import WhatsAppAlert
from text_helper import put_kazakh_text

# .env файлынан мәндерді жүктеу — барлық нәрседен бұрын
load_dotenv()

# Камера мекенжайы мен шекті санды .env-тен алу,
# егер .env-те жоқ болса, default мән қолданылады
SOURCE      = os.getenv("CAMERA_SOURCE", "http://192.168.43.1:8080/video")
MAX_PERSONS = int(os.getenv("MAX_PERSONS", 10))

def main():
    cap = cv2.VideoCapture(SOURCE)

    counter    = PersonCounter()
    weapon_det = WeaponDetector()
    face_rec   = FaceRecognizer("known_faces")
    alert      = WhatsAppAlert()  # WhatsApp кілттерін .env-тен алады

    print("✅ Жүйе іске қосылды...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Камерадан сурет келмеді!")
            break

        # 1. Адам санау
        person_count, frame = counter.count(frame)

        # 2. Қару тану
        weapon_found, frame = weapon_det.detect(frame)

        # 3. Бет тану
        frame, unknown_found = face_rec.recognize(frame)

        # 4. Қажет болса WhatsApp дабыл жіберу
        if weapon_found:
            alert.send("ҚАРУ", "Қару-жарақ анықталды!", frame)

        if unknown_found:
            alert.send("БЕЛГІСІЗ АДАМ", "Танылмаған адам анықталды!", frame)

        if person_count > MAX_PERSONS:
            alert.send("АДАМ КӨП", f"Адам саны: {person_count}/{MAX_PERSONS}", frame)

        # 5. Экранға мәтін шығару
        color = (0, 0, 255) if person_count > MAX_PERSONS else (0, 255, 0)
        frame = put_kazakh_text(frame,
                                f"Адам саны: {person_count}/{MAX_PERSONS}",
                                (10, 10), font_size=32, color=color)

        if weapon_found:
            frame = put_kazakh_text(frame, "ҚАРУ АНЫҚТАЛДЫ!",
                                    (10, 55), font_size=32, color=(0, 0, 255))

        if unknown_found:
            frame = put_kazakh_text(frame, "БЕЛГІСІЗ АДАМ!",
                                    (10, 100), font_size=32, color=(180, 105, 255))

        cv2.imshow("Қауіпсіздік жүйесі", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()