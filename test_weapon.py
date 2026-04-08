from ultralytics import YOLO

model = YOLO("weapon_model.pt")

print("✅ Модель жүктелді")
print("Класс саны:", len(model.names))
print("Класстар:", model.names)
print("\n🎉 Модель дайын — main.py іске қосуға болады!")