import cv2
import easyocr
import numpy as np

# Инициализируем OCR (русский + английский)
reader = easyocr.Reader(['ru', 'en'])

def find_license_plate(image):
    """Находит прямоугольник, похожий на номер"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Бинаризация
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # слишком маленькие области
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        # Номера обычно имеют ширину больше высоты
        if 2 < aspect_ratio < 5:
            return (x, y, w, h)
    return None

def process_plate(frame):
    """Основная функция: находит номер и распознает текст"""
    plate_rect = find_license_plate(frame)
    if plate_rect is None:
        return None, None
    
    x, y, w, h = plate_rect
    plate_img = frame[y:y+h, x:x+w]
    
    # Распознаем текст
    result = reader.readtext(plate_img)
    if result:
        text = result[0][1]
        return plate_rect, text
    return plate_rect, None

# Пример использования с камерой
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    plate_rect, plate_text = process_plate(frame)
    
    if plate_rect:
        x, y, w, h = plate_rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if plate_text:
            cv2.putText(frame, plate_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("License Plate", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
