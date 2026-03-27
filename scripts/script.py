import cv2
import datetime
import os

# Путь к скачанному файлу каскада для номеров
cascade_path = r"C:\MyPythonProjects\AV\yolo-coco\haarcascade_russian_plate_number.xml"

# Проверяем, существует ли файл
if not os.path.exists(cascade_path):
    print(f"❌ Файл каскада не найден: {cascade_path}")
    exit()

# Загружаем каскад
plate_cascade = cv2.CascadeClassifier(cascade_path)

if plate_cascade.empty():
    print("❌ Не удалось загрузить каскад. Проверь файл.")
    exit()

print("✅ Каскад для номеров загружен!")

cap = cv2.VideoCapture(0)
recording = False
out = None

print("🔍 Слежу за номерами... Нажми 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Детекция номеров
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    plate_detected = len(plates) > 0

    # Рисуем рамки вокруг номеров
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "PLATE", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Запись при обнаружении номера
    if plate_detected:
        if not recording:
            filename = f"C:/Users/Денис/Downloads/plate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
            recording = True
            print(f"🎬 НОМЕР! Запись: {filename}")
    else:
        if recording:
            recording = False
            out.release()
            print("⏹️ Запись остановлена")

    if recording:
        out.write(frame)
        cv2.putText(frame, "REC", (frame.shape[1] - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Number Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
