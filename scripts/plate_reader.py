import cv2
import easyocr
import pandas as pd
from ultralytics import YOLO

class PlateReader:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.reader = easyocr.Reader(['ru', 'en'], gpu=False)
        
        # Загружаем базу разрешённых номеров
        try:
            self.allowed_df = pd.read_csv("data/allowed_plates.csv")
            self.allowed_plates = set(self.allowed_df['plate'].astype(str).str.upper().values)
            print(f"Загружено разрешённых номеров: {len(self.allowed_plates)}")
        except FileNotFoundError:
            self.allowed_plates = set()
            print("Файл data/allowed_plates.csv не найден")
    
    def detect_plate(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, "Не удалось загрузить изображение"
        
        # Уменьшаем если большое
        height, width = img.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        results = self.model(img)
        
        cars = [box for box in results[0].boxes if int(box.cls[0]) == 2]
        if len(cars) == 0:
            return None, "Машина не найдена"
        
        box = cars[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Вырезаем нижнюю часть (номер)
        plate_y1 = y1 + int((y2 - y1) * 0.7)
        plate_y2 = y2
        plate_x1 = x1 + int((x2 - x1) * 0.25)
        plate_x2 = x2 - int((x2 - x1) * 0.25)
        
        plate_y1 = max(0, plate_y1)
        plate_y2 = min(img.shape[0], plate_y2)
        plate_x1 = max(0, plate_x1)
        plate_x2 = min(img.shape[1], plate_x2)
        
        if plate_y2 <= plate_y1 or plate_x2 <= plate_x1:
            return None, "Область номера слишком мала"
        
        plate_crop = img[plate_y1:plate_y2, plate_x1:plate_x2]
        plate_crop = cv2.convertScaleAbs(plate_crop, alpha=1.5, beta=0)
        
        try:
            ocr_result = self.reader.readtext(plate_crop)
            if len(ocr_result) == 0:
                return plate_crop, "Текст не распознан"
            plate_text = ocr_result[0][1].upper().replace(" ", "").replace("-", "")
            return plate_crop, plate_text
        except Exception as e:
            return plate_crop, f"Ошибка OCR"
    
    def process(self, image_path):
        crop, text = self.detect_plate(image_path)
        is_allowed = text in self.allowed_plates if text not in ["Машина не найдена", "Текст не распознан", "Не удалось загрузить изображение"] else False
        
        return {
            "plate_text": text,
            "is_allowed": is_allowed,
            "message": "✅ ДОСТУП РАЗРЕШЁН" if is_allowed else "❌ ДОСТУП ЗАПРЕЩЁН",
            "success": text not in ["Машина не найдена", "Текст не распознан"],
            "cropped_image": crop
        }


if __name__ == "__main__":
    reader = PlateReader()
    result = reader.process(r"C:\MyPythonProjects\AV\yolo-coco\Ав\Сканирование_20260328-1918-07.jpg")
    
    print(f"Распознанный номер: {result['plate_text']}")
    print(f"Результат: {result['message']}")
    print(f"В базе: {result['is_allowed']}")
    
    if result['cropped_image'] is not None:
        cv2.imwrite("cropped_plate.jpg", result['cropped_image'])
        print("Вырезанный номер сохранён в cropped_plate.jpg")
