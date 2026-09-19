import serial
import time

arduino = serial.Serial("COM7", 9600, timeout=1)

time.sleep(2)

print("Arduino connected!")

while True:

    command = input(
        "\nMasukkan command "
        "(ORGANIK / ANORGANIK / RESIDU / EXIT): "
    ).upper()

    if command == "EXIT":
        break

    if command in ["ORGANIK", "ANORGANIK", "RESIDU"]:
        arduino.write((command + "\n").encode())

        print("Mengirim:", command)

    else:
        print("Command tidak dikenal.")

arduino.close()