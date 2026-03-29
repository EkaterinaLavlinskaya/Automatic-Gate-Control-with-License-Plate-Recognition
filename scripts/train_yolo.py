# ============================================
# ДООБУЧЕНИЕ YOLOv8 НА ДАТАСЕТЕ Av.rar
# ============================================

import os
import zipfile
import shutil
from ultralytics import YOLO

# ---------------------------
# 1. Установка и подготовка
# ---------------------------
print("🚀 Установка unrar...")
!apt-get install -y unrar > /dev/null 2>&1

print("📦 Загрузка датасета с Google Drive...")
!gdown "1h2j5hZnGbT8YZjc6elJwZ3Lizeo6J-iJ" -O Av.rar --quiet

# ---------------------------
# 2. Распаковка
# ---------------------------
print("📂 Распаковка архива...")
os.makedirs("/content/data", exist_ok=True)
!unrar x Av.rar /content/data/ > /dev/null 2>&1

# ---------------------------
# 3. Копирование всех JPG и TXT в одну папку
# ---------------------------
print("🖼️ Копирование файлов в единую папку...")
os.makedirs("/content/images", exist_ok=True)

jpg_files = []
txt_files = []

for root, dirs, files in os.walk("/content/data"):
    for file in files:
        if file.lower().endswith('.jpg'):
            jpg_files.append(os.path.join(root, file))
        elif file.lower().endswith('.txt'):
            txt_files.append(os.path.join(root, file))

for src in jpg_files + txt_files:
    shutil.copy(src, "/content/images/")

print(f"   ✅ JPG: {len(jpg_files)}")
print(f"   ✅ TXT: {len(txt_files)}")

# ---------------------------
# 4. Создание dataset.yaml
# ---------------------------
yaml_content = """
path: /content/images
train: .
val: .
nc: 1
names: ['car']
"""

with open('/content/dataset.yaml', 'w') as f:
    f.write(yaml_content)

print("   ✅ dataset.yaml создан")

# ---------------------------
# 5. Запуск обучения
# ---------------------------
print("\n🔥 ЗАПУСК ОБУЧЕНИЯ (200 эпох, ~30-60 минут)...")
print("=" * 50)

model = YOLO("yolov8n.pt")

results = model.train(
    data="/content/dataset.yaml",
    epochs=200,           # 200 эпох
    imgsz=416,            # размер изображения
    batch=8,              # батч (уменьши до 4 при ошибке памяти)
    device=0,             # GPU
    workers=2,            
    patience=50,          # остановка при отсутствии улучшений
    save_period=25,       # сохранять каждые 25 эпох
    verbose=True
)

# ---------------------------
# 6. Тестирование модели
# ---------------------------
print("\n🧪 Тестирование модели на первом изображении...")
test_image = "/content/images/" + [f for f in os.listdir("/content/images") if f.endswith('.jpg')][0]
results_test = model(test_image)

from google.colab.patches import cv2_imshow
cv2_imshow(results_test[0].plot())

# ---------------------------
# 7. Скачивание модели
# ---------------------------
print("\n📥 Скачивание best.pt на компьютер...")
from google.colab import files
files.download('/content/runs/detect/train/weights/best.pt')

print("\n✅ ВСЁ ГОТОВО! Модель обучена и скачана.")
print("📁 Путь к модели: /content/runs/detect/train/weights/best.pt")
