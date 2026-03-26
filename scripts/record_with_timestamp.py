import cv2
import datetime

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Камера не найдена")
    exit()

filename = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))

print(f"Запись: {filename}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Camera', frame)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Готово!")
