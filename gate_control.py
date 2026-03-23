import serial
import time

# Подключение к Arduino (порт может отличаться)
# В Arduino IDE: Инструменты → Порт → посмотри, какой порт выбран
arduino = serial.Serial('COM3', 9600)   # на Windows
# arduino = serial.Serial('/dev/ttyACM0', 9600)  # на Linux / Mac
time.sleep(2)   # ждём, пока Arduino инициализируется

# Отправляем команду на включение светодиода
arduino.write(b'1')
print("Светодиод включён")

time.sleep(3)   # светодиод горит 3 секунды

# Отправляем команду на выключение
arduino.write(b'0')
print("Светодиод выключен")

arduino.close()
