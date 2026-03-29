import cv2
import easyocr
from ultralytics import YOLO

class PlateReader:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        # Инициализируем EasyOCR только один раз
        self.reader = easyocr.Reader(['ru', 'en'], gpu=False)
    
    def detect_plate(self, image_path):
        # Уменьшаем изображение перед обработкой, если оно большое
        img = cv2.imread(image_path)
        if img is None:
            return None, "Не удалось загрузить изображение"
        
        # Ограничиваем размер изображения (максимум 1280 по ширине)
        height, width = img.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        results = self.model(img)
        
        # Ищем машину (класс 2)
        cars = [box for box in results[0].boxes if int(box.cls[0]) == 2]
        if len(cars) == 0:
            return None, "Машина не найдена"
        
        # Берём первую найденную машину
        box = cars[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Вырезаем нижнюю часть (где номер)
        plate_y1 = y1 + int((y2 - y1) * 0.7)
        plate_y2 = y2
        plate_x1 = x1 + int((x2 - x1) * 0.25)
        plate_x2 = x2 - int((x2 - x1) * 0.25)
        
        # Проверяем границы
        plate_y1 = max(0, plate_y1)
        plate_y2 = min(img.shape[0], plate_y2)
        plate_x1 = max(0, plate_x1)
        plate_x2 = min(img.shape[1], plate_x2)
        
        # Убеждаемся, что область не пустая
        if plate_y2 <= plate_y1 or plate_x2 <= plate_x1:
            return None, "Область номера слишком мала"
        
        plate_crop = img[plate_y1:plate_y2, plate_x1:plate_x2]
        
        # Увеличиваем контраст для лучшего распознавания
        plate_crop = cv2.convertScaleAbs(plate_crop, alpha=1.5, beta=0)
        
        # Уменьшаем вырезанную область для EasyOCR (если слишком большая)
        crop_h, crop_w = plate_crop.shape[:2]
        if crop_w > 800:
            scale = 800 / crop_w
            new_w = 800
            new_h = int(crop_h * scale)
            plate_crop = cv2.resize(plate_crop, (new_w, new_h))
        
        # Распознаём текст
        try:
            ocr_result = self.reader.readtext(plate_crop)
            if len(ocr_result) == 0:
                return plate_crop, "Текст не распознан"
            plate_text = ocr_result[0][1].upper().replace(" ", "").replace("-", "")
            return plate_crop, plate_text
        except Exception as e:
            return plate_crop, f"Ошибка OCR: {str(e)[:50]}"
    
    def process(self, image_path):
        crop, text = self.detect_plate(image_path)
        return {
            "plate_text": text,
            "success": text not in ["Машина не найдена", "Текст не распознан", "Не удалось загрузить изображение"] and not text.startswith("Ошибка OCR"),
            "cropped_image": crop
        }

if __name__ == "__main__":
    reader = PlateReader()
    
    # Используй правильный путь к своему фото
    result = reader.process(r"C:\MyPythonProjects\AV\yolo-coco\Ав\Сканирование_20260328-1918-07.jpg")
    
    print(f"Распознанный номер: {result['plate_text']}")
    print(f"Успех: {result['success']}")
    
    if result['cropped_image'] is not None:
        cv2.imwrite("cropped_plate.jpg", result['cropped_image'])
        print("Вырезанный номер сохранён в cropped_plate.jpg")
