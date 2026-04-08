from twilio.rest import Client
import cv2
import datetime
import os
from dotenv import load_dotenv  # .env файлын оқу үшін

# .env файлынан кілттерді жүктеу
load_dotenv()

class WhatsAppAlert:
    def __init__(self):
        # Енді тырнақша ішіне ештеңе жазбаймыз — .env-тен автоматты алады
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token  = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self.to_number   = os.getenv("TWILIO_TO_NUMBER")

        self.client = Client(self.account_sid, self.auth_token)
        self.last_sent = {}
        self.cooldown = 30  # секунд — жиі жібермеу үшін

        os.makedirs("alerts/белгісіз_адам", exist_ok=True)
        os.makedirs("alerts/қару", exist_ok=True)
        print("✅ WhatsApp сигнал жүйесі дайын")

    def send(self, alert_type, message, frame):
        now = datetime.datetime.now()

        # Cooldown тексеру — бір минутта бір рет жіберу
        if alert_type in self.last_sent:
            diff = (now - self.last_sent[alert_type]).seconds
            if diff < self.cooldown:
                return

        self.last_sent[alert_type] = now
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        time_str  = now.strftime('%H:%M:%S')

        # Скриншот сақтау
        if alert_type == "ҚАРУ":
            path = f"alerts/қару/{timestamp}.jpg"
        else:
            path = f"alerts/белгісіз_адам/{timestamp}.jpg"

        cv2.imwrite(path, frame)
        print(f"📸 Сақталды: {path}")

        # WhatsApp хабарлама жіберу
        try:
            full_message = (
                f"🚨 ҚАУІПСІЗДІК ДАБЫЛЫ\n"
                f"━━━━━━━━━━━━━━━\n"
                f"⚠ Түрі: {alert_type}\n"
                f"📋 Ақпарат: {message}\n"
                f"🕐 Уақыт: {time_str}\n"
                f"📁 Скрин сақталды: {path}"
            )

            self.client.messages.create(
                from_=self.from_number,
                to=self.to_number,
                body=full_message
            )
            print(f"✅ WhatsApp жіберілді: {alert_type}")

        except Exception as e:
            print(f"❌ WhatsApp қате: {e}")