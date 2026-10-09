"""
arduino_controller.py
=====================
Modul koneksi dan komunikasi Arduino untuk SEGRATRUK.

Diekstrak dari hierarki_klasifikasi.ipynb:
  - Koneksi serial ke Arduino
  - Daftar serial port
  - Kirim hasil klasifikasi (ORGANIK / ANORGANIK / RESIDU) ke Arduino
  - Baca event sensor HC-SR04 (OBJECT_DETECTED / OBJECT_CLEARED)

Contoh pemakaian di notebook / script lain:

    from arduino_controller import ArduinoController

    ctrl = ArduinoController(port="COM8")
    ctrl.connect()

    msg = ctrl.read_sensor_message()
    if msg == ArduinoController.MSG_OBJECT_DETECTED:
        ctrl.send_label("Organik")

    ctrl.close()
"""

import sys
import time

import serial
import serial.tools.list_ports
import cv2

# Pastikan output console di Windows tidak error karena karakter emoji / non-ASCII
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ============================================================
# KONFIGURASI DEFAULT
# ============================================================

ARDUINO_PORT = "COM8"
ARDUINO_BAUDRATE = 9600

# Label hasil klasifikasi -> perintah serial untuk Arduino
LABEL_TO_COMMAND = {
    "Organik": "ORGANIK",       # Servo D9
    "Anorganik": "ANORGANIK",   # Servo D10
    "Residu": "RESIDU",         # Servo D11
}


# ============================================================
# DAFTAR SERIAL PORT
# ============================================================

def list_serial_ports():
    """Cetak dan kembalikan daftar serial port yang terdeteksi."""

    print("=== DAFTAR SERIAL PORT ===")

    ports = list(serial.tools.list_ports.comports())

    if len(ports) == 0:
        print("❌ Tidak ada serial port yang terdeteksi.")
    else:
        for port in ports:
            print(f"{port.device} | {port.description}")

    return ports


def find_arduino_port():
    """Cari port Arduino otomatis dari deskripsi serial port."""
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "ch341" in desc or "usb-serial" in desc:
            return port.device
    return None


# ============================================================
# CONTROLLER
# ============================================================

class ArduinoController:
    """Mengelola koneksi serial dan komunikasi dengan Arduino."""

    # Pesan dari sensor (Arduino -> Python)
    MSG_OBJECT_DETECTED = "OBJECT_DETECTED"
    MSG_OBJECT_CLEARED = "OBJECT_CLEARED"

    def __init__(self, port=None, baudrate=ARDUINO_BAUDRATE, timeout=1):
        if port is None or port == "auto":
            auto_port = find_arduino_port()
            self.port = auto_port if auto_port else ARDUINO_PORT
        else:
            self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.arduino = None

    # --------------------------------------------------------
    # Koneksi
    # --------------------------------------------------------

    @property
    def is_connected(self):
        return self.arduino is not None and self.arduino.is_open

    def close(self):
        """Tutup koneksi Arduino jika masih aktif."""

        try:
            if self.is_connected:
                print("🔌 Menutup koneksi Arduino...")

                self.arduino.close()
                

                time.sleep(0.5)

                print("✅ Koneksi Arduino berhasil ditutup.")

        except Exception as e:
            print(f"⚠️ Tidak dapat menutup koneksi: {e}")

    def connect(self):
        """
        Buat koneksi baru ke Arduino.
        Koneksi lama (jika ada) ditutup lebih dulu.

        Return:
            True jika berhasil, False jika gagal.
        """

        if self.is_connected:
            print("⚠️ Koneksi Arduino lama ditemukan.")

        self.close()

        try:
            print(f"\n🔌 Menghubungkan ke Arduino pada {self.port}...")

            self.arduino = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )

            # Arduino biasanya reset ketika serial connection dibuka
            time.sleep(2)

            print("✅ Arduino berhasil terhubung!")
            print(f"Port : {self.port}")
            print(f"Baud : {self.baudrate}")

            return True

        except serial.SerialException as e:
            print("❌ Gagal menghubungkan Arduino.")
            print(f"Error: {e}")

            self.arduino = None

            return False

    # --------------------------------------------------------
    # Kirim hasil klasifikasi
    # --------------------------------------------------------

    def send_label(self, label):
        """
        Kirim hasil klasifikasi ke Arduino.

        Args:
            label: "Organik", "Anorganik", atau "Residu"

        Return:
            True jika terkirim, False jika gagal.
        """

        command = LABEL_TO_COMMAND.get(label)

        if command is None:
            print(f"⚠️ Label tidak dikenal: {label}")
            return False

        if not self.is_connected:
            print("❌ Arduino tidak terhubung. Perintah tidak dikirim.")
            return False

        self.arduino.write((command + "\n").encode())

        print(f"📤 Mengirim ke Arduino: {command}")

        return True

    def send_command(self, cmd):
        """Kirim perintah teks langsung ke Arduino."""
        if not self.is_connected:
            return False
        try:
            self.arduino.write((cmd.strip() + "\n").encode("utf-8"))
            return True
        except Exception as e:
            print(f"❌ Gagal mengirim serial ke Arduino: {e}")
            return False

    def motor_forward(self, speed=255):
        """Jalankan motor konveyor maju dengan kecepatan (0-255)."""
        spd = max(0, min(255, int(speed)))
        return self.send_command(f"MOTOR_FORWARD:{spd}")

    def motor_stop(self):
        """Hentikan motor konveyor (Speed = 0)."""
        return self.send_command("MOTOR_STOP")

    def reset_servos(self):
        """Kembalikan semua servo ke posisi netral (0 derajat)."""
        return self.send_command("RESET")

    # --------------------------------------------------------
    # Baca event sensor
    # --------------------------------------------------------

    def read_sensor_message(self):
        """
        Baca satu pesan dari Arduino (non-blocking).

        Return:
            String pesan (mis. "OBJECT_DETECTED"), atau None jika tidak ada.
        """

        if not self.is_connected:
            return None

        if self.arduino.in_waiting > 0:

            message = self.arduino.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if message:
                print(f"📡 Arduino: {message}")

            return message

        return None

    # --------------------------------------------------------
    # Context manager
    # --------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# ============================================================
# WEBCAM & PREVIEW
# ============================================================

def list_cameras(max_index=5):
    """Cari index webcam yang bisa dibuka. Return list of int."""
    found = []
    print("=== DAFTAR KAMERA ===")
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  📷 Kamera index {index} : {w}x{h}")
                found.append(index)
        cap.release()

    if not found:
        print("  ❌ Tidak ada kamera yang terdeteksi.")
    return found


def preview_camera(camera_index=1, controller=None, window_name="SEGRATRUK - Preview & Controller", focus=None, autofocus=False):
    """
    Tampilkan preview webcam dengan overlay status Arduino dan kontrol fokus.

    Args:
        camera_index: index webcam (default: 1)
        controller: instance ArduinoController (opsional)
        window_name: nama window cv2
        focus: nilai fokus manual awal (0 - 1000)
        autofocus: jika True, aktifkan autofocus otomatis

    Kontrol Keyboard:
        [1]            : Kirim perintah ORGANIK (Servo D9)
        [2]            : Kirim perintah ANORGANIK (Servo D10)
        [3]            : Kirim perintah RESIDU (Servo D11)
        [A]            : Toggle Auto Focus (ON / OFF)
        [ [ ] / [ ] ]  : Kurangi / Tambah fokus manual (-10 / +10)
        [-] / [+]      : Kurangi / Tambah fokus cepat (-50 / +50)
        [Q]            : Keluar dari preview
    """
    print(f"\n🎥 Membuka kamera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"⚠️ Gagal dengan CAP_DSHOW, mencoba backend default...")
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Kamera index {camera_index} tidak dapat dibuka.")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Inisialisasi kontrol fokus
    initial_af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    initial_foc = cap.get(cv2.CAP_PROP_FOCUS)
    supports_focus = not (initial_af < 0 and initial_foc < 0)

    cur_autofocus = False
    cur_focus = int(initial_foc) if initial_foc >= 0 else 350

    if supports_focus:
        if autofocus:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cur_autofocus = True
        elif focus is not None:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, focus)
            cur_autofocus = False
            cur_focus = focus
        else:
            cur_autofocus = (initial_af > 0)
            if not cur_autofocus:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_FOCUS, cur_focus)

        def on_focus_trackbar(val):
            nonlocal cur_focus, cur_autofocus
            if cur_autofocus:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cur_autofocus = False
                cv2.setTrackbarPos("Auto Focus (0/1)", window_name, 0)
            cur_focus = val
            cap.set(cv2.CAP_PROP_FOCUS, val)

        def on_autofocus_trackbar(val):
            nonlocal cur_autofocus
            cur_autofocus = bool(val)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if cur_autofocus else 0)
            if not cur_autofocus:
                cap.set(cv2.CAP_PROP_FOCUS, cur_focus)

        cv2.createTrackbar("Focus", window_name, max(0, min(1000, cur_focus)), 1000, on_focus_trackbar)
        cv2.createTrackbar("Auto Focus (0/1)", window_name, 1 if cur_autofocus else 0, 1, on_autofocus_trackbar)

    print(f"✅ Kamera siap! Window preview aktif.")
    print("⌨️  Tekan Q untuk keluar, atau tekan 1/2/3 untuk tes servo.")

    last_sensor_msg = "-"
    last_sent_action = "-"

    key_actions = {
        ord('1'): "Organik",
        ord('2'): "Anorganik",
        ord('3'): "Residu"
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Gagal membaca frame dari kamera.")
                break

            # Baca sensor jika Arduino terhubung
            if controller is not None and controller.is_connected:
                msg = controller.read_sensor_message()
                if msg:
                    last_sensor_msg = msg

            # Overlay informasi di frame
            cv2.rectangle(frame, (10, 10), (630, 115), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 10), (630, 115), (100, 100, 100), 1)

            # Status Arduino
            if controller is not None and controller.is_connected:
                ard_text = f"Arduino: CONNECTED ({controller.port})"
                ard_color = (0, 255, 0)
            else:
                port_name = controller.port if controller else "-"
                ard_text = f"Arduino: NOT CONNECTED ({port_name})"
                ard_color = (0, 0, 255)

            cv2.putText(frame, ard_text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ard_color, 2)

            # Status Sensor & Fokus
            sensor_color = (0, 255, 255) if last_sensor_msg == ArduinoController.MSG_OBJECT_DETECTED else (200, 200, 200)
            focus_str = ("Focus: AUTO" if cur_autofocus else f"Focus: {cur_focus}") if supports_focus else "Focus: FIXED"
            cv2.putText(frame, f"Sensor: {last_sensor_msg} | {focus_str}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, sensor_color, 2)

            # Status Kirim Terakhir
            cv2.putText(frame, f"Kirim Terakhir  : {last_sent_action}", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1)

            # Petunjuk tombol di bawah
            help_text = "[1/2/3] Servo | [A] AutoFocus | [ [ / ] ] Fokus | [Q] Keluar"
            cv2.putText(frame, help_text, (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q atau Esc
                print("🛑 Preview dihentikan.")
                break

            if key in key_actions:
                label_target = key_actions[key]
                last_sent_action = label_target
                if controller is not None and controller.is_connected:
                    controller.send_label(label_target)
                else:
                    print(f"⚠️ Mencoba kirim '{label_target}', tapi Arduino tidak terhubung.")

            # Autofocus toggle [A]
            elif key == ord('a') and supports_focus:
                cur_autofocus = not cur_autofocus
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if cur_autofocus else 0)
                cv2.setTrackbarPos("Auto Focus (0/1)", window_name, 1 if cur_autofocus else 0)
                if not cur_autofocus:
                    cap.set(cv2.CAP_PROP_FOCUS, cur_focus)
                print(f"📷 Autofocus: {'ON' if cur_autofocus else 'OFF'}")

            # Fokus bertambah [ ] ] / [+]
            elif (key == ord(']') or key == ord('+') or key == ord('=')) and supports_focus:
                cur_autofocus = False
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cv2.setTrackbarPos("Auto Focus (0/1)", window_name, 0)
                step = 50 if key in [ord('+'), ord('=')] else 10
                cur_focus = min(1000, cur_focus + step)
                cap.set(cv2.CAP_PROP_FOCUS, cur_focus)
                cv2.setTrackbarPos("Focus", window_name, cur_focus)
                print(f"🔍 Fokus: {cur_focus}")

            # Fokus berkurang [ [ ] / [-]
            elif (key == ord('[') or key == ord('-')) and supports_focus:
                cur_autofocus = False
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cv2.setTrackbarPos("Auto Focus (0/1)", window_name, 0)
                step = 50 if key == ord('-') else 10
                cur_focus = max(0, cur_focus - step)
                cap.set(cv2.CAP_PROP_FOCUS, cur_focus)
                cv2.setTrackbarPos("Focus", window_name, cur_focus)
                print(f"🔍 Fokus: {cur_focus}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# CLI & TEST UTAMA
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SEGRATRUK - Arduino & Camera Preview Utility")
    parser.add_argument("--port", type=str, default=None,
                        help="Port serial Arduino (misal: COM9). Default: otomatis deteksi.")
    parser.add_argument("--camera", type=int, default=None,
                        help="Index kamera webcam (misal: 1). Default: otomatis deteksi.")
    parser.add_argument("--focus", type=int, default=None,
                        help="Nilai fokus kamera (0 - 1000).")
    parser.add_argument("--autofocus", action="store_true",
                        help="Aktifkan autofocus kamera otomatis.")
    parser.add_argument("--scan", action="store_true",
                        help="Hanya pindai port dan kamera tanpa membuka preview.")
    parser.add_argument("--no-arduino", action="store_true",
                        help="Jalankan preview kamera saja tanpa menghubungkan ke Arduino.")
    args = parser.parse_args()

    # Pindai port serial & kamera
    list_serial_ports()
    cameras = list_cameras()

    if args.scan:
        raise SystemExit(0)

    # Tentukan index kamera
    cam_idx = args.camera
    if cam_idx is None:
        if 1 in cameras:
            cam_idx = 1
        elif len(cameras) > 0:
            cam_idx = cameras[0]
        else:
            cam_idx = 0

    if args.no_arduino:
        print("\n⚡ Mode kamera saja (--no-arduino aktif).")
        preview_camera(camera_index=cam_idx, focus=args.focus, autofocus=args.autofocus)
    else:
        target_port = args.port or find_arduino_port() or ARDUINO_PORT
        print(f"\n🎯 Menggunakan port Arduino: {target_port}")

        ctrl = ArduinoController(port=target_port)
        connected = ctrl.connect()
        if not connected:
            print("⚠️ Lanjut preview kamera tanpa koneksi Arduino...")

        try:
            preview_camera(camera_index=cam_idx, controller=ctrl, focus=args.focus, autofocus=args.autofocus)
        finally:
            ctrl.close()
