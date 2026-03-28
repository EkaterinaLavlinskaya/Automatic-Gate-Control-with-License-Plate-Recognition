#Разметака и конвертация разметки 

import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_file, output_dir, class_map):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_map:
            continue
        class_id = class_map[class_name]

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # YOLO format: class_id x_center y_center width height (normalized)
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save to .txt file
    txt_filename = os.path.splitext(os.path.basename(xml_file))[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, 'w') as f:
        f.write("\n".join(yolo_lines))

# ===== НАСТРОЙКИ =====
xml_folder = r"C:\MyPythonProjects\AV\yolo-coco\labels"          # ← Укажи папку, где лежат XML
output_folder = r"C:\MyPythonProjects\AV\yolo-coco"   # ← Куда сохранить TXT
class_map = {"car": 0}                         # car → class_id 0

os.makedirs(output_folder, exist_ok=True)

for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        convert_voc_to_yolo(os.path.join(xml_folder, xml_file), output_folder, class_map)

print("✅ Конвертация завершена!")
