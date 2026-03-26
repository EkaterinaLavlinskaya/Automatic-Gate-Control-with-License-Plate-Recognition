import cv2
import datetime

# Открываем камеру
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit()

# Получаем параметры кадра
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20.0

# Создаем имя файла с датой и временем
filename = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"

# Настройка сохранения видео
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

print(f"Запись начата. Файл: {filename}")
print("Нажми 'q' для остановки")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break
    
    # ===== ДОБАВЛЯЕМ ТАЙМСТАМП =====
    # Получаем текущее время
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Добавляем текст на кадр
    cv2.putText(
        frame, 
        timestamp, 
        (10, 30),                    # позиция (x, y) - левый верхний угол
        cv2.FONT_HERSHEY_SIMPLEX,    # шрифт
        0.7,                         # размер
        (0, 255, 0),                 # цвет (зеленый)
        2,                           # толщина
        cv2.LINE_AA                  # сглаживание
    )
    # =================================
    
    # Показываем кадр на экране
    cv2.imshow('Camera', frame)
    
    # Сохраняем кадр в видео
    out.write(frame)
    
    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Запись завершена. Видео сохранено как {filename}")
