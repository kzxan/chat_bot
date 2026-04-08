from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import anthropic
import requests as req
import datetime
import os
import base64
import numpy as np
import cv2
from ultralytics import YOLO
import face_recognition

app = Flask(__name__)

# =============================
# БАПТАУЛАР
# =============================
ACCOUNT_SID    = ""
AUTH_TOKEN     = ""
FROM_NUMBER    = "whatsapp:+"  # Twilio номері
TO_NUMBER      = "whatsapp:+"
ANTHROPIC_KEY  = ""  # anthropic.com-дан алыңыз

twilio_client  = Client(ACCOUNT_SID, AUTH_TOKEN)
claude_client  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# Модельдер
weapon_model = YOLO("weapon_model.pt")
yolo_model   = YOLO("yolov8n.pt")

# Сөйлесу тарихы (әр нөмір бойынша)
conversation_history = {}

# Known faces
known_encodings = []
known_names = []

def load_known_faces():
    directory = "known_faces"
    if not os.path.exists(directory):
        return
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    print(f"✅ {len(known_names)} адам жүктелді")

load_known_faces()


# =============================
# YOLO АНАЛИЗІ
# =============================
def yolo_analyze(image_url):
    """YOLO арқылы суретті анализ жасау"""
    try:
        response = req.get(image_url, auth=(ACCOUNT_SID, AUTH_TOKEN))
        img_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return {}, None

        findings = {
            "person_count": 0,
            "weapon_found": False,
            "known_people": [],
            "unknown_count": 0,
        }

        # Адам санау
        p_results = yolo_model(frame, classes=[0], verbose=False, conf=0.5)
        findings["person_count"] = sum(len(r.boxes) for r in p_results)

        # Қару тану
        w_results = weapon_model(frame, verbose=False, conf=0.35)
        k_results = yolo_model(frame, classes=[43], verbose=False, conf=0.35)
        if sum(len(r.boxes) for r in w_results) + \
           sum(len(r.boxes) for r in k_results) > 0:
            findings["weapon_found"] = True

        # Бет тану
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb)
        encs  = face_recognition.face_encodings(rgb, locs)

        import numpy as np_lib
        for enc in encs:
            if known_encodings:
                matches   = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
                distances = face_recognition.face_distance(known_encodings, enc)
                best      = np_lib.argmin(distances)
                if matches[best]:
                    findings["known_people"].append(known_names[best])
                else:
                    findings["unknown_count"] += 1
            else:
                findings["unknown_count"] += 1

        return findings, frame

    except Exception as e:
        print(f"YOLO қате: {e}")
        return {}, None


# =============================
# CLAUDE AI — АҚЫЛДЫ ЖАУАП
# =============================
def ask_claude(user_number, user_message, image_url=None, yolo_findings=None):
    """Claude AI-дан жауап алу"""

    # Сөйлесу тарихын алу немесе жасау
    if user_number not in conversation_history:
        conversation_history[user_number] = []

    history = conversation_history[user_number]

    # Жүйелік нұсқау
    system_prompt = """Сен қауіпсіздік жүйесінің ақылды көмекшісісің.
Сенің міндетің:
1. Камерадан келген суреттерді анализ жасау
2. Қауіпсіздік мәселелері туралы кеңес беру
3. Пайдаланушының сұрақтарына жауап беру

Жауаптарыңда:
- Қысқа және нақты бол
- Қазақша жауап бер
- Қауіп табылса, нақты ескерт
- Эмодзи қолдан

YOLO нәтижелері бар болса, оларды да ескер."""

    # Хабарлама мазмұны
    content = []

    # Сурет бар болса
    if image_url:
        try:
            img_response = req.get(image_url,
                                  auth=(ACCOUNT_SID, AUTH_TOKEN))
            img_base64 = base64.b64encode(
                img_response.content).decode("utf-8")

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_base64
                }
            })

            # YOLO нәтижелерін қосу
            if yolo_findings:
                yolo_text = f"""
YOLO анализі нәтижесі:
- Адам саны: {yolo_findings.get('person_count', 0)}
- Қару: {'ИӘ ⚠️' if yolo_findings.get('weapon_found') else 'Жоқ ✅'}
- Танылған адамдар: {', '.join(yolo_findings.get('known_people', [])) or 'Жоқ'}
- Белгісіз адам: {yolo_findings.get('unknown_count', 0)}

Осы суретті толық анализ жасап, қауіпсіздік тұрғысынан бағала.
"""
                content.append({"type": "text", "text": yolo_text})
            else:
                content.append({
                    "type": "text",
                    "text": "Осы суретті қауіпсіздік тұрғысынан анализ жаса."
                })

        except Exception as e:
            content.append({"type": "text", "text": user_message})
    else:
        content.append({"type": "text", "text": user_message})

    # Тарихқа қосу
    history.append({"role": "user", "content": content})

    # Тарихты 10 хабармен шектеу (жад)
    if len(history) > 10:
        history = history[-10:]
        conversation_history[user_number] = history

    # Claude-ға сұрау
    try:
        response = claude_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1000,
            system=system_prompt,
            messages=history
        )

        assistant_reply = response.content[0].text

        # Тарихқа assistant жауабын қосу
        history.append({
            "role": "assistant",
            "content": assistant_reply
        })

        return assistant_reply

    except Exception as e:
        return f"❌ Claude қате: {str(e)}"


# =============================
# WEBHOOK
# =============================
@app.route("/bot", methods=["POST"])
def bot():
    incoming_msg = request.values.get("Body", "").strip()
    num_media    = int(request.values.get("NumMedia", 0))
    user_number  = request.values.get("From", "")

    resp = MessagingResponse()
    msg  = resp.message()

    # Фото жіберілсе
    if num_media > 0:
        media_url = request.values.get("MediaUrl0", "")

        # 1. YOLO анализі
        msg.body("🔍 Анализ жасалуда...")
        findings, frame = yolo_analyze(media_url)

        # 2. Claude AI толық анализі
        claude_response = ask_claude(
            user_number,
            incoming_msg or "Суретті анализ жаса",
            image_url=media_url,
            yolo_findings=findings
        )

        msg.body(claude_response)

    # Мәтін жіберілсе — Claude жауап береді
    else:
        if incoming_msg:
            claude_response = ask_claude(user_number, incoming_msg)
            msg.body(claude_response)
        else:
            msg.body("Сәлем! Сұрақ жазыңыз немесе фото жіберіңіз 📸")

    return str(resp)


# =============================
# КАМЕРАДАН СИГНАЛ
# =============================
def send_camera_alert(alert_type, message, frame=None):
    """main.py-ден шақырылады"""
    try:
        now = datetime.datetime.now().strftime('%H:%M:%S')

        # Claude арқылы ақылды хабарлама жасау
        smart_msg = ask_claude(
            "camera_system",
            f"Камера дабылы: {alert_type}. {message}. "
            f"Уақыт: {now}. "
            f"Қысқа ескерту хабарламасы жаз."
        )

        twilio_client.messages.create(
            from_=FROM_NUMBER,
            to=TO_NUMBER,
            body=f"🚨 {smart_msg}"
        )
        print(f"✅ Ақылды дабыл жіберілді")

    except Exception as e:
        print(f"❌ Қате: {e}")


if __name__ == "__main__":
    from pyngrok import ngrok
    public_url = ngrok.connect(5000)
    print(f"\n✅ Webhook URL: {public_url}/bot")
    print("👆 Twilio-ға осыны енгізіңіз!\n")
    app.run(port=5000, debug=False)