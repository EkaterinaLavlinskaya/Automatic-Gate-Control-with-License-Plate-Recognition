import os
import shutil

os.chdir('/content')

# Проверяем, что папка Ав существует
if os.path.exists('Ав'):
    print("Папка 'Ав' найдена. Копируем файлы...")
    
    # Очищаем папку darknet/data/custom
    !rm -rf /content/darknet/data/custom
    !mkdir -p /content/darknet/data/custom
    
    # Копируем файлы из папки Ав
    !cp Ав/*.jpg /content/darknet/data/custom/
    !cp Ав/*.txt /content/darknet/data/custom/
    
    # Проверяем количество
    os.chdir('/content/darknet')
    print("\n=== Проверка данных ===")
    !echo "Фото (JPG):" && ls data/custom/*.jpg | wc -l
    !echo "Разметка (TXT):" && ls data/custom/*.txt | wc -l
    
    # Пересоздаем train.txt
    !find data/custom -name "*.jpg" > train.txt
    
    # Проверяем, что darknet53.conv.74 на месте
    if not os.path.exists("darknet53.conv.74"):
        !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/darknet53.conv.74
    
    # Запускаем обучение
    print("\n✅ Начинаем обучение (20-40 минут)...")
    print("Следи за loss — он должен уменьшаться\n")
    !./darknet detector train data/custom.data cfg/yolov3_custom.cfg darknet53.conv.74 -dont_show -map
    
else:
    print("Папка 'Ав' не найдена. Сначала выполни скачивание и распаковку архива.")
  
import os
os.chdir('/content/darknet')

# Список всех jpg
jpg_files = set([f.replace('.jpg', '') for f in os.listdir('data/custom') if f.endswith('.jpg')])
# Список всех txt
txt_files = set([f.replace('.txt', '') for f in os.listdir('data/custom') if f.endswith('.txt')])

print("Файлы без разметки (есть jpg, нет txt):", jpg_files - txt_files)
print("Файлы без фото (есть txt, нет jpg):", txt_files - jpg_files)
# -*- coding: utf-8 -*-
"""Дообучение YOLOv3 """

import os

# 1. Скачиваем твой архив (45 фото, 45 txt)
!wget "https://drive.usercontent.google.com/download?id=1h2j5hZnGbT8YZjc6elJwZ3Lizeo6J-iJ&export=download&confirm=t" -O Av.rar

# 2. Распаковываем
!apt-get install -y unrar
!unrar x Av.rar

# 3. Клонируем darknet
if not os.path.exists('darknet'):
    !git clone https://github.com/pjreddie/darknet.git

%cd darknet

# 4. Компилируем
!make

# 5. Очищаем папку для данных
!rm -rf data/custom
!mkdir -p data/custom

# 6. Копируем файлы из папки Ав
!cp ../Ав/*.jpg data/custom/
!cp ../Ав/*.txt data/custom/

# 7. Проверяем количество
print("\n=== Проверка данных ===")
print("Фото:")
!ls data/custom/*.jpg | wc -l
print("Разметка:")
!ls data/custom/*.txt | wc -l

# 8. Создаем train.txt
!find data/custom -name "*.jpg" > train.txt

# 9. Создаем classes.names
!echo "car" > data/custom/classes.names

# 10. Создаем custom.data
with open('data/custom.data', 'w') as f:
    f.write("classes = 1\n")
    f.write("train = /content/darknet/train.txt\n")
    f.write("valid = /content/darknet/train.txt\n")
    f.write("names = data/custom/classes.names\n")
    f.write("backup = /content/darknet/backup\n")

# 11. Настраиваем конфиг YOLO
!cp cfg/yolov3.cfg cfg/yolov3_custom.cfg
!sed -i 's/batch=64/batch=16/' cfg/yolov3_custom.cfg
!sed -i 's/subdivisions=16/subdivisions=8/' cfg/yolov3_custom.cfg
!sed -i 's/classes=80/classes=1/' cfg/yolov3_custom.cfg
!sed -i 's/filters=255/filters=18/' cfg/yolov3_custom.cfg

# 12. Скачиваем предобученные веса
if not os.path.exists("darknet53.conv.74"):
    !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/darknet53.conv.74

# 13. Запускаем обучение
print("\n✅ Начинаем обучение (20-40 минут)...")
print("Следи за loss — он должен уменьшаться\n")
!./darknet detector train data/custom.data cfg/yolov3_custom.cfg darknet53.conv.74 -dont_show -map

# 14. Скачиваем результат
print("\n✅ Обучение завершено. Скачиваем веса...")
from google.colab import files
files.download('/content/darknet/backup/yolov3_custom_last.weights')
