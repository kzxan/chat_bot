import cv2
import datetime
import os

class AlertSystem:
    def __init__(self):
        self.last_alert = {}
        self.cooldown = 5  # секунд — қайталамау үшін
        os.makedirs("alerts", exist_ok=True)
        print("✅ Сигнал жүйесі дайын")

    def trigger(self, alert_type, message, frame):
        now = datetime.datetime.now()

        # Cooldown тексеру
        if alert_type in self.last_alert:
            diff = (now - self.last_alert[alert_type]).seconds
            if diff < self.cooldown:
                return

        self.last_alert[alert_type] = now

        # 1. Консольға хабар
        print(f"\n🚨 СИГНАЛ [{now.strftime('%H:%M:%S')}]: {message}")

        # 2. Скриншот сақтау
        filename = f"alerts/{alert_type}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Скриншот сақталды: {filename}")