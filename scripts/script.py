import cv2
import datetime
import os

# Загрузка готовой модели
cascade_path = r"C:\MyPythonProjects\AV\yolo-coco\coches.xml"
car_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
recording = False
out = None

print("🔍 Слежу за машинами (Haar каскад)... Нажми 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    car_detected = len(cars) > 0

    # Рисуем рамки
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Запись при обнаружении
    if car_detected:
        if not recording:
            filename = f"C:/Users/Денис/Downloads/car_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
            recording = True
            print(f"🎬 МАШИНА! Запись: {filename}")
    else:
        if recording:
            recording = False
            out.release()
            print("⏹️ Запись остановлена")

    if recording:
        out.write(frame)
        cv2.putText(frame, "REC", (frame.shape[1] - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Car Detection (Haar)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
