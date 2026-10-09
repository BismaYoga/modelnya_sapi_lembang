"""
main.py
=======
SEGRATRUK - Sistem Klasifikasi Sampah Hierarkis Real-Time
(Organik, Anorganik, Residu) terintegrasi dengan Arduino dan Kamera.

Arsitektur Model:
  1. Model 1 (model_organik_anorganik.h5): Organik vs Anorganik
  2. Model 2 (binary_best_model.keras): Non-Recyclable (Residu) vs Recyclable (Anorganik)

Hardware:
  - Arduino Uno (Auto-detect / COM9)
  - Sensor Ultrasonik HC-SR04 (Trig D13, Echo D12) -> Kirim event: OBJECT_DETECTED / OBJECT_CLEARED
  - Motor Driver L298N (Konveyor):
      - ENA (PWM Kecepatan) -> Pin D5
      - IN1 (Direction 1)   -> Pin D4
      - IN2 (Direction 2)   -> Pin D7
      - Logika: Stop 3 detik saat detect untuk scan kamera, lalu jalan Speed 255 agar lewat.
  - Servo Aktuator:
      - Organik   -> Servo D9
      - Anorganik -> Servo D10
      - Residu    -> Servo D11
  - Kamera Webcam (Auto-detect / Index 1 atau 0)

Cara Menjalankan:
  py main.py                     (Menjalankan sistem real-time lengkap)
  py main.py --camera 0          (Gunakan webcam index 0)
  py main.py --no-arduino        (Mode simulasi AI kamera tanpa Arduino)
  py main.py --continuous        (Klasifikasi terus-menerus tanpa menunggu sensor)
  py main.py --image "tes.jpg"   (Klasifikasi satu foto)
  py main.py --scan              (Pindai port Arduino dan kamera yang tersedia)
"""

import os
import sys
import time
import warnings
from collections import Counter, deque
from pathlib import Path

# Supress TensorFlow & Protobuf warning log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="google.protobuf")

# Pastikan console Windows utf-8
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
# CEK DEPENDENCIES
# ============================================================

try:
    import cv2
    import numpy as np
    from PIL import Image
    import serial
    import serial.tools.list_ports
except ImportError as e:
    print(f"❌ Dependency belum lengkap: {e}")
    print("Silakan install: pip install opencv-python pillow pyserial")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    print("❌ TensorFlow tidak ditemukan pada environment Python saat ini.")
    print("💡 Catatan: TensorFlow terpasang di Python 3.13.")
    print("👉 Silakan jalankan dengan perintah: py main.py")
    sys.exit(1)


# ============================================================
# ⚙️ PUSAT KONFIGURASI SISTEM SEGRATRUK (UBAH SEMUA NILAI DI SINI)
# ============================================================

# ------------------------------------------------------------
# 1. BATAS JARAK SENSOR ULTRASONIK (HC-SR04: Trig D13, Echo D12)
# ------------------------------------------------------------
# Batas maksimal jarak (cm) untuk mendeteksi sampah.
# Jika jarak benda <= nilai ini, sistem menganggap ada sampah terdeteksi.
# Nilai default dinaikkan ke 35.0 cm agar sensor mudah dan responsif mendeteksi.
SENSOR_MAX_DISTANCE_CM = 8.0   # <-- UBAH BATAS MAKSIMAL DETEKSI (misal: 30.0, 35.0, 40.0 cm)
SENSOR_MIN_DISTANCE_CM = 0.0    # Batas minimal (cm) untuk mengabaikan noise

# ------------------------------------------------------------
# 2. KONFIGURASI MOTOR KONVEYOR (L298N: ENA D5, IN1 D4, IN2 D7)
# ------------------------------------------------------------
CONVEYOR_STANDBY_SPEED = 255   # Kecepatan konveyor normal saat menunggu sampah (0 - 255)
CONVEYOR_PASS_SPEED    = 255   # Kecepatan PENUH konveyor saat melewatkan sampah setelah scan (0 - 255)
SCAN_STOP_DURATION     = 3.0   # Berapa detik konveyor BERHENTI saat sampah terdeteksi untuk scan AI (detik)
PASS_MIN_DURATION      = 2.0   # Durasi minimal konveyor jalan speed 255 sebelum cek cleared (detik)

# ------------------------------------------------------------
# 3. ARAH / SUDUT SERVO (SERVO ORGANIK D9, ANORGANIK D10, RESIDU D11)
# ------------------------------------------------------------
# Sudut servo saat posisi standby / netral (derajat 0 - 180)
SERVO_NEUTRAL_ANGLES = {
    "Organik": 0,       # Posisi netral Servo Organik (D9)
    "Anorganik": 0,     # Posisi netral Servo Anorganik (D10)
    "Residu": 0,        # Posisi netral Servo Residu (D11)
}

# Sudut servo saat AKTIF / membuka gerbang pemilah (derajat 0 - 180)
# (Sesuaikan arah gerak servo Anda: misal 90, 120, atau 180)
SERVO_ACTIVE_ANGLES = {
    "Organik": 90,      # Arah sudut buka Servo Organik (D9)
    "Anorganik": 90,    # Arah sudut buka Servo Anorganik (D10)
    "Residu": 90,       # Arah sudut buka Servo Residu (D11)
}

# ------------------------------------------------------------
# 4. DURASI SERVO TERBUKA SETELAH MOTOR JALAN LAGI (TIAP SERVO BEDA-BEDA)
# ------------------------------------------------------------
# Berapa detik servo tetap terbuka SETELAH motor mulai jalan lagi dengan speed 255.
# Karena posisi masing-masing bak/gate di konveyor berbeda-beda, durasi bisa disesuaikan per kategori:
SERVO_HOLD_DURATIONS = {
    "Organik": 3.0,     # Servo Organik tetap terbuka selama 3.0 detik setelah motor jalan
    "Anorganik": 4.5,   # Servo Anorganik tetap terbuka selama 4.5 detik setelah motor jalan
    "Residu": 6.0,      # Servo Residu tetap terbuka selama 6.0 detik setelah motor jalan
}

# ------------------------------------------------------------
# 5. ATURAN AMBANG BATAS CONFIDENCE / AKURASI (PERSEN %)
# ------------------------------------------------------------
# Jika confidence/akurasi prediksi di bawah nilai ini, MAKA SERVO TIDAK AKAN BERGERAK.
MIN_CONFIDENCE_THRESHOLD = 60.0

CONFIDENCE_THRESHOLDS = {
    "Organik": 30.0,     # Minimal akurasi % agar Servo Organik bergerak
    "Anorganik": 30.0,   # Minimal akurasi % agar Servo Anorganik bergerak
    "Residu": 70.0,      # Minimal akurasi % agar Servo Residu bergerak
}

# Ambang batas probabilitas Sigmoid Model 2 (0.0 sampai 1.0)
MODEL2_PROB_THRESHOLD = 0.3

# ------------------------------------------------------------
# 6. MODEL & SISTEM PARAMETER
# ------------------------------------------------------------
DEFAULT_MODEL1_PATH = "model_organik_anorganik.h5"
DEFAULT_MODEL2_PATH = "binary_best_model.keras"
IMG_SIZE = (224, 224)

# Mapping kelas Model 1 (Softmax / Categorical)
MODEL1_CLASSES = {0: "Anorganik", 1: "Organik"}

# Mapping kelas Model 2 (Binary Sigmoid: <0.5 Residu, >=0.5 Anorganik)
MODEL2_CLASSES = {0: "Residu", 1: "Anorganik"}

# Warna visualisasi BGR untuk OpenCV
LABEL_COLORS_BGR = {
    "Organik": (50, 205, 50),     # Hijau
    "Anorganik": (255, 140, 0),   # Biru muda / cyan
    "Residu": (0, 0, 230),        # Merah
    "LOW_CONF": (0, 165, 255),    # Oranye / peringatan
    "NO OBJECT": (150, 150, 150)  # Abu-abu
}

DEFAULT_STABLE_FRAMES = 5
DEFAULT_RESET_DELAY = 12.0     # Timeout keselamatan maksimal (detik)
DEFAULT_DIFF_THRESHOLD = 30
DEFAULT_MIN_OBJECT_AREA = 1500


def is_confidence_sufficient(label, confidence):
    """
    Cek apakah nilai confidence memenuhi syarat ambang batas minimal.
    Return: (is_passed: bool, required_threshold: float)
    """
    required = CONFIDENCE_THRESHOLDS.get(label, MIN_CONFIDENCE_THRESHOLD)
    return confidence >= required, required


# ============================================================
# DETEKSI HARDWARE
# ============================================================

def list_serial_ports():
    """Tampilkan dan kembalikan port serial yang aktif."""
    print("\n=== DAFTAR SERIAL PORT ===")
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("  ❌ Tidak ada serial port yang terdeteksi.")
    else:
        for p in ports:
            print(f"  🔌 {p.device} | {p.description}")
    return ports


def find_arduino_port():
    """Deteksi otomatis port Arduino dari deskripsi COM port."""
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "ch341" in desc or "usb-serial" in desc:
            return p.device
    return None


def list_cameras(max_index=4):
    """Cari index kamera yang bisa dibuka."""
    print("\n=== DAFTAR KAMERA ===")
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  📷 Kamera Index {idx} : {w}x{h}")
                found.append(idx)
        cap.release()
    if not found:
        print("  ❌ Tidak ada kamera yang terdeteksi.")
    return found


# ============================================================
# KONTROLER ARDUINO
# ============================================================

class ArduinoController:
    """Mengelola komunikasi serial ke Arduino Nano / Uno."""

    MSG_OBJECT_DETECTED = "OBJECT_DETECTED"
    MSG_OBJECT_CLEARED = "OBJECT_CLEARED"

    def __init__(self, port=None, baudrate=9600, timeout=1):
        if port is None or port == "auto":
            auto_port = find_arduino_port()
            self.port = auto_port if auto_port else "COM9"
        else:
            self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.arduino = None

    @property
    def is_connected(self):
        return self.arduino is not None and self.arduino.is_open

    def connect(self):
        self.close()
        try:
            print(f"🔌 Menghubungkan ke Arduino pada {self.port} (baud {self.baudrate})...")
            self.arduino = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)  # Menunggu Arduino reset setelah DTR aktif
            print(f"✅ Arduino berhasil terhubung pada {self.port}!")
            return True
        except serial.SerialException as e:
            print(f"❌ Gagal menghubungkan Arduino pada {self.port}: {e}")
            self.arduino = None
            return False

    def close(self):
        if self.is_connected:
            try:
                self.arduino.close()
                time.sleep(0.3)
                print("🔌 Koneksi Arduino ditutup.")
            except Exception as e:
                print(f"⚠️ Gagal menutup port: {e}")

    def send_label(self, label, angle=None):
        """Kirim perintah aktuator berdasarkan label dan sudut (opsional)."""
        if angle is not None:
            return self.set_servo_angle(label, angle)
        return self.send_command(label.upper())

    def set_servo_angle(self, label, angle):
        """Kirim perintah sudut servo spesifik ke Arduino (misal SERVO:Organik:90)."""
        ang = max(0, min(180, int(angle)))
        print(f"⚙️ [SERVO] {label} -> Sudut {ang}°")
        return self.send_command(f"SERVO:{label}:{ang}")

    def set_max_distance(self, distance_cm):
        """Atur batas jarak deteksi maksimal di Arduino (cm)."""
        dist = max(5, int(distance_cm))
        print(f"📡 [SENSOR] Mengatur batas deteksi maksimal: {dist} cm")
        return self.send_command(f"SET_MAX_DIST:{dist}")

    def send_command(self, cmd):
        """Kirim perintah teks serial generik ke Arduino."""
        if not self.is_connected:
            return False
        try:
            self.arduino.write((cmd.strip() + "\n").encode("utf-8"))
            return True
        except Exception as e:
            print(f"❌ Gagal mengirim serial ke Arduino: {e}")
            return False

    def motor_forward(self, speed=255):
        """Jalankan motor konveyor maju dengan kecepatan tertentu (0-255)."""
        spd = max(0, min(255, int(speed)))
        print(f"⚙️ [MOTOR] Maju kecepatan: {spd}")
        return self.send_command(f"MOTOR_FORWARD:{spd}")

    def motor_stop(self):
        """Hentikan motor konveyor (Speed = 0)."""
        print("🛑 [MOTOR] Berhenti (Speed 0)")
        return self.send_command("MOTOR_STOP")

    def reset_servos(self):
        """Kembalikan semua servo ke posisi netral (0 derajat)."""
        print("🔄 [SERVO] Reset ke posisi netral (0°)")
        return self.send_command("RESET")

    def read_sensor_message(self):
        """Membaca pesan dari Arduino secara non-blocking."""
        if not self.is_connected:
            return None

        try:
            if self.arduino.in_waiting > 0:
                msg = self.arduino.readline().decode("utf-8", errors="ignore").strip()
                if msg:
                    print(f"📡 [ARDUINO] {msg}")
                return msg
        except Exception as e:
            print(f"⚠️ Error membaca serial: {e}")
        return None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
# MODEL KLASIFIKASI HIERARKIS
# ============================================================

class HierarchicalWasteClassifier:
    """Mengelola Model 1 & Model 2 dan logika klasifikasi hierarkis."""

    def __init__(self, model1_path=DEFAULT_MODEL1_PATH, model2_path=DEFAULT_MODEL2_PATH):
        self.model1_path = Path(model1_path)
        self.model2_path = Path(model2_path)
        self.model1 = None
        self.model2 = None
        self.loaded = False

    def load(self):
        """Muat kedua model ke memory."""
        print("\n⏳ Memuat model deep learning...")

        if not self.model1_path.exists():
            raise FileNotFoundError(f"File model 1 tidak ditemukan: {self.model1_path.resolve()}")
        if not self.model2_path.exists():
            raise FileNotFoundError(f"File model 2 tidak ditemukan: {self.model2_path.resolve()}")

        print(f"  • Model 1 (Organik vs Anorganik) : {self.model1_path.name}")
        self.model1 = load_model(str(self.model1_path))

        print(f"  • Model 2 (Residu vs Recyclable)  : {self.model2_path.name}")
        self.model2 = load_model(str(self.model2_path))

        # Warm-up inference pertama agar tidak lambat saat realtime
        dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        _ = self.model1.predict(dummy, verbose=0)
        _ = self.model2.predict(dummy, verbose=0)

        self.loaded = True
        print("✅ Kedua model berhasil dimuat dan siap digunakan!")

    def preprocess_frame(self, frame_bgr):
        """Konversi BGR OpenCV ke format input model (RGB, 224x224, normalized 0..1)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_resized = img_pil.resize(IMG_SIZE)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def predict_batch(self, img_batch):
        """
        Eksekusi inferensi hierarkis dari tensor batch (1, 224, 224, 3):
          Langkah 1: Model 1 -> Organik vs Anorganik
          Langkah 2: Jika Anorganik -> Model 2 -> Residu vs Anorganik
        """
        if not self.loaded:
            raise RuntimeError("Model belum dimuat! Panggil load() terlebih dahulu.")

        # --- STEP 1: Model 1 ---
        pred1 = self.model1.predict(img_batch, verbose=0)
        pred1_idx = int(np.argmax(pred1[0]))
        pred1_label = MODEL1_CLASSES[pred1_idx]
        pred1_conf = float(pred1[0][pred1_idx] * 100)

        # Jika Organik, langsung selesai
        if pred1_label == "Organik":
            return {
                "final_label": "Organik",
                "confidence": pred1_conf,
                "model1_label": pred1_label,
                "model1_conf": pred1_conf,
                "model2_label": None,
                "model2_conf": None
            }

        # --- STEP 2: Model 2 (Jika Anorganik) ---
        pred2 = self.model2.predict(img_batch, verbose=0)
        prob2 = float(pred2[0][0])  # Sigmoid

        if prob2 >= MODEL2_PROB_THRESHOLD:
            pred2_label = "Anorganik"  # Recyclable
            pred2_conf = prob2 * 100
        else:
            pred2_label = "Residu"     # Non-Recyclable
            pred2_conf = (1.0 - prob2) * 100

        return {
            "final_label": pred2_label,
            "confidence": pred2_conf,
            "model1_label": pred1_label,
            "model1_conf": pred1_conf,
            "model2_label": pred2_label,
            "model2_conf": pred2_conf
        }

    def predict_frame(self, frame_bgr):
        """Prediksi langsung dari satu frame OpenCV BGR."""
        img_batch = self.preprocess_frame(frame_bgr)
        res = self.predict_batch(img_batch)
        return res["final_label"], res["confidence"]

    def predict_image_file(self, image_path):
        """Prediksi dari file gambar di disk."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Gambar tidak ditemukan: {path.resolve()}")
        frame_bgr = cv2.imread(str(path))
        if frame_bgr is None:
            raise ValueError(f"Gagal membaca file gambar: {path.resolve()}")
        img_batch = self.preprocess_frame(frame_bgr)
        return self.predict_batch(img_batch)


# ============================================================
# SISTEM KLASIFIKASI REAL-TIME
# ============================================================

def run_realtime_system(
    classifier: HierarchicalWasteClassifier,
    controller: ArduinoController,
    camera_index: int = 1,
    mode_sensor: bool = True,
    stable_frames: int = DEFAULT_STABLE_FRAMES,
    reset_delay: float = DEFAULT_RESET_DELAY,
    focus: int = None,
    autofocus: bool = False
):
    """
    Loop utama sistem real-time SEGRATRUK.

    Alur Sensor Mode (Default):
      1. State: WAITING_SENSOR (Tunggu Arduino sensor ultrasonik lapor OBJECT_DETECTED)
      2. State: CLASSIFYING (Kamera membaca frame berulang kali sampai prediksi stabil)
      3. State: CONFIRMED (Kirim perintah servo ke Arduino: ORGANIK / ANORGANIK / RESIDU)
      4. State: WAITING_OBJECT_CLEAR (Tunggu sensor lapor OBJECT_CLEARED sebelum sampah berikutnya)

    Fitur Keyboard:
      - [Q] / [Esc]  : Keluar program
      - [Space] / [C]: Klasifikasi instan frame saat ini (manual trigger)
      - [M]          : Toggle mode antara Sensor Ultrasonik vs Continuous Otomatis
      - [R]          : Reset sistem ke status awal
      - [1 / 2 / 3]  : Manual tes aktuator servo Arduino (Organik / Anorganik / Residu)
      - [A]          : Toggle Auto Focus kamera (ON / OFF)
      - [ [ ] / [ ] ]: Kurangi / Tambah nilai fokus kamera (-10 / +10)
      - [-] / [+]    : Kurangi / Tambah nilai fokus cepat (-50 / +50)
    """

    print(f"\n🎥 Membuka kamera index {camera_index} (DirectShow)...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("⚠️ Gagal dengan CAP_DSHOW, mencoba backend default...")
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Tidak dapat membuka webcam pada index {camera_index}!")
        return

    # Atur resolusi kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "SEGRATRUK - Waste Classification & Hardware System"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # ============================================================
    # PENGATURAN FOKUS KAMERA
    # ============================================================
    initial_af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    initial_foc = cap.get(cv2.CAP_PROP_FOCUS)
    supports_focus = not (initial_af < 0 and initial_foc < 0)

    cur_autofocus = False
    cur_focus = int(initial_foc) if initial_foc >= 0 else 350

    if supports_focus:
        if autofocus:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cur_autofocus = True
            print("📷 Kamera Autofocus: ON")
        elif focus is not None:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, focus)
            cur_autofocus = False
            cur_focus = focus
            print(f"📷 Fokus kamera diatur manual: {cur_focus}")
        else:
            cur_autofocus = (initial_af > 0)
            if not cur_autofocus:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_FOCUS, cur_focus)
            print(f"📷 Fokus kamera: {cur_focus} (Auto: {cur_autofocus})")

        # Trackbar Callbacks
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
    else:
        print("ℹ️ Kamera ini memiliki fokus tetap (Fixed Focus / tidak mendukung software slider).")

    # State Machine
    system_state = "WAITING_SENSOR" if mode_sensor else "CONTINUOUS"
    prediction_history = []  # Menyimpan tuple (label, conf) selama pemindaian 3 detik

    confirmed_label = None
    confirmed_conf = 0.0
    scan_start_time = 0.0
    pass_start_time = 0.0
    last_sent_time = 0.0
    last_sensor_msg = "-"
    last_action_note = "-"

    # Pelacak Sensor & Servo Per-Kategori
    live_distance = 999.0
    current_active_servo = None
    servo_closed = True
    servo_close_timer = 0.0

    # FPS counter
    prev_time = time.time()
    fps = 0.0

    print("\n" + "=" * 60)
    print("🚀 SEGRATRUK SYSTEM RUNNING (PUSAT KONTROL PYTHON)")
    print("=" * 60)
    print(f"  • Model 1 & 2    : READY")
    print(f"  • Arduino        : {'CONNECTED (' + controller.port + ')' if controller and controller.is_connected else 'OFFLINE / SIMULASI'}")
    print(f"  • Sensor Jarak   : Batas Maksimal {SENSOR_MAX_DISTANCE_CM:.0f} cm")
    print(f"  • Konveyor Motor : Standby {CONVEYOR_STANDBY_SPEED} | Lewat {CONVEYOR_PASS_SPEED} | Stop Scan {SCAN_STOP_DURATION:.0f}s")
    print(f"  • Sudut Servo    : Organik {SERVO_ACTIVE_ANGLES['Organik']}° | Anorganik {SERVO_ACTIVE_ANGLES['Anorganik']}° | Residu {SERVO_ACTIVE_ANGLES['Residu']}°")
    print(f"  • Durasi Buka    : Organik {SERVO_HOLD_DURATIONS['Organik']}s | Anorganik {SERVO_HOLD_DURATIONS['Anorganik']}s | Residu {SERVO_HOLD_DURATIONS['Residu']}s")
    print("------------------------------------------------------------")
    print("  [Space] Pemicu Scan 3s | [4] Motor 255 | [0] Motor Stop")
    print("  [1/2/3] Tes Servo      | [M] Ganti Mode | [Q] Keluar")
    print("=" * 60 + "\n")

    # Inisialisasi awal ke Arduino
    if controller and controller.is_connected:
        # Kirim konfigurasi jarak deteksi ke Arduino
        controller.set_max_distance(SENSOR_MAX_DISTANCE_CM)
        # Reset semua servo ke posisi netral
        for lbl, ang in SERVO_NEUTRAL_ANGLES.items():
            controller.set_servo_angle(lbl, ang)
        # Jalankan konveyor standby jika dalam mode sensor
        if mode_sensor:
            print(f"🚀 Memulai konveyor awal (Speed {CONVEYOR_STANDBY_SPEED})...")
            controller.motor_forward(CONVEYOR_STANDBY_SPEED)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Frame gagal dibaca dari webcam.")
                time.sleep(0.05)
                continue

            # Hitung FPS
            cur_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, cur_time - prev_time))
            prev_time = cur_time

            # ----------------------------------------------------
            # BACA PESAN SERIAL ARDUINO & UPDATE JARAK SENSOR
            # ----------------------------------------------------
            msg = None
            if controller and controller.is_connected:
                msg = controller.read_sensor_message()
                if msg:
                    last_sensor_msg = msg
                    if msg.startswith("DIST:"):
                        try:
                            live_distance = float(msg.split(":")[1])
                        except (ValueError, IndexError):
                            pass

            # ----------------------------------------------------
            # STATE MACHINE (100% DIKONTROL DARI PYTHON)
            # ----------------------------------------------------
            display_pred = "NO OBJECT"
            display_conf = 0.0

            # --- MODE 1: SENSOR EVENT (LOGIKA PENUH PYTHON) ---
            if system_state == "WAITING_SENSOR":
                display_pred = "NO OBJECT"
                display_conf = 0.0
                status_text = f"KONVEYOR JALAN ({CONVEYOR_STANDBY_SPEED}) - MENUNGGU SAMPAH..."

                # Cek apakah ada objek terdeteksi (baik dari pesan event Arduino atau live_distance <= batas)
                is_detected = (msg == ArduinoController.MSG_OBJECT_DETECTED) or \
                              (SENSOR_MIN_DISTANCE_CM <= live_distance <= SENSOR_MAX_DISTANCE_CM)

                if is_detected:
                    print(f"\n📦 [EVENT] Sampah terdeteksi (Jarak: {live_distance:.1f} cm <= {SENSOR_MAX_DISTANCE_CM:.0f} cm)!")
                    print(f"🛑 [MOTOR] Menghentikan konveyor seketika (Speed 0). Memulai scan AI ({SCAN_STOP_DURATION:.0f} detik)...")
                    if controller and controller.is_connected:
                        controller.motor_stop()

                    prediction_history.clear()
                    scan_start_time = time.time()
                    system_state = "CLASSIFYING"

            elif system_state == "CLASSIFYING":
                # Konveyor dalam posisi BERHENTI (Speed 0).
                # Lakukan inferensi kamera pada setiap frame selama SCAN_STOP_DURATION detik.
                label, conf = classifier.predict_frame(frame)
                prediction_history.append((label, conf))

                display_pred = label
                display_conf = conf

                elapsed_scan = time.time() - scan_start_time
                remaining_scan = max(0.0, SCAN_STOP_DURATION - elapsed_scan)
                status_text = f"SCANNING AI ({remaining_scan:.1f}s) - {len(prediction_history)} frame"

                # Cek apakah durasi berhenti untuk scan (3 detik) sudah selesai
                if elapsed_scan >= SCAN_STOP_DURATION:
                    # Ambil hasil mayoritas selama 3 detik pemindaian
                    if prediction_history:
                        labels = [item[0] for item in prediction_history]
                        most_common_label = Counter(labels).most_common(1)[0][0]
                        matching_confs = [item[1] for item in prediction_history if item[0] == most_common_label]
                        confirmed_label = most_common_label
                        confirmed_conf = float(np.mean(matching_confs)) if matching_confs else conf
                    else:
                        confirmed_label = label
                        confirmed_conf = conf

                    # Cek aturan ambang batas confidence
                    is_passed, req_threshold = is_confidence_sufficient(confirmed_label, confirmed_conf)

                    print("\n" + "=" * 55)
                    print(f"✅ [SCAN {SCAN_STOP_DURATION:.0f} DETIK SELESAI - OBJEK TERKONFIRMASI]")
                    print(f"  Kategori    : {confirmed_label.upper()}")
                    print(f"  Confidence  : {confirmed_conf:.1f}% (Ambang batas minimal: {req_threshold:.1f}%)")

                    if is_passed:
                        active_angle = SERVO_ACTIVE_ANGLES.get(confirmed_label, 90)
                        hold_sec = SERVO_HOLD_DURATIONS.get(confirmed_label, 4.0)
                        print(f"  Status Servo: 🟢 BUKA KE {active_angle}° (Tetap buka {hold_sec}s setelah motor jalan)")
                        if controller and controller.is_connected:
                            controller.set_servo_angle(confirmed_label, active_angle)

                        current_active_servo = confirmed_label
                        servo_close_timer = hold_sec
                        servo_closed = False
                        last_action_note = f"KIRIM -> {confirmed_label.upper()} ({confirmed_conf:.1f}%)"
                    else:
                        print(f"  Status Servo: ⛔ DIBAWAH AMBANG BATAS ({confirmed_conf:.1f}% < {req_threshold:.1f}%)")
                        print("                ⚠️ ATURAN: TIDAK ADA SERVO YANG BERGERAK.")
                        current_active_servo = None
                        servo_closed = True
                        servo_close_timer = 2.0
                        last_action_note = f"DIABAIKAN (<{req_threshold:.0f}%)"

                    print(f"🚀 [MOTOR] Menjalankan konveyor MAJU KECEPATAN PENUH (Speed {CONVEYOR_PASS_SPEED}) agar sampah lewat!")
                    print("=" * 55)

                    # Nyalakan konveyor dengan speed 255 agar sampah lewat
                    if controller and controller.is_connected:
                        controller.motor_forward(CONVEYOR_PASS_SPEED)

                    pass_start_time = time.time()
                    last_sent_time = time.time()
                    prediction_history.clear()
                    system_state = "WAITING_OBJECT_CLEAR"

            elif system_state == "WAITING_OBJECT_CLEAR":
                display_pred = confirmed_label or "PASSING"
                display_conf = confirmed_conf

                elapsed_pass = time.time() - pass_start_time
                status_text = f"MELEWATKAN SAMPAH (Speed {CONVEYOR_PASS_SPEED}) - {elapsed_pass:.1f}s..."

                # ----------------------------------------------------
                # TUTUP SERVO SETELAH DURASI MASING-MASING TERCAPAI
                # ----------------------------------------------------
                if not servo_closed and current_active_servo and elapsed_pass >= servo_close_timer:
                    neutral_angle = SERVO_NEUTRAL_ANGLES.get(current_active_servo, 0)
                    if controller and controller.is_connected:
                        controller.set_servo_angle(current_active_servo, neutral_angle)
                    servo_closed = True
                    print(f"🔄 [SERVO] Durasi buka {current_active_servo} ({servo_close_timer}s) selesai -> Servo ditutup ke {neutral_angle}°")

                # ----------------------------------------------------
                # SELESAIKAN SIKLUS MELEWATKAN SAMPAH
                # ----------------------------------------------------
                min_pass_duration = max(PASS_MIN_DURATION, servo_close_timer + 1.0)
                is_cleared = (msg == ArduinoController.MSG_OBJECT_CLEARED) or \
                             (live_distance > SENSOR_MAX_DISTANCE_CM) or \
                             (elapsed_pass >= DEFAULT_RESET_DELAY)

                if elapsed_pass >= min_pass_duration and is_cleared:
                    print("\n🔄 [READY] Area bersih / sampah telah lewat.")
                    print(f"⚙️ [MOTOR] Konveyor kembali ke kecepatan normal (Speed {CONVEYOR_STANDBY_SPEED}).")
                    if controller and controller.is_connected:
                        for lbl, ang in SERVO_NEUTRAL_ANGLES.items():
                            controller.set_servo_angle(lbl, ang)
                        controller.motor_forward(CONVEYOR_STANDBY_SPEED)

                    confirmed_label = None
                    confirmed_conf = 0.0
                    current_active_servo = None
                    servo_closed = True
                    prediction_history.clear()
                    system_state = "WAITING_SENSOR"

            # --- MODE 2: CONTINUOUS VISION ---
            elif system_state == "CONTINUOUS":
                label, conf = classifier.predict_frame(frame)
                display_pred = label
                display_conf = conf
                status_text = "CONTINUOUS LIVE"

                elapsed = time.time() - last_sent_time
                if (confirmed_label != label or elapsed >= DEFAULT_RESET_DELAY):
                    confirmed_label = label
                    confirmed_conf = conf
                    last_sent_time = time.time()

                    is_passed, req_threshold = is_confidence_sufficient(label, conf)
                    if is_passed:
                        active_angle = SERVO_ACTIVE_ANGLES.get(label, 90)
                        if controller and controller.is_connected:
                            controller.set_servo_angle(label, active_angle)
                        last_action_note = f"KIRIM -> {label.upper()}"
                    else:
                        last_action_note = f"DIABAIKAN (<{req_threshold:.0f}%)"

            # ----------------------------------------------------
            # VISUALISASI HUD & GUI
            # ----------------------------------------------------
            display = frame.copy()
            h, w = display.shape[:2]

            # 1. Background Panel Atas
            cv2.rectangle(display, (10, 10), (w - 10, 125), (25, 25, 25), -1)
            cv2.rectangle(display, (10, 10), (w - 10, 125), (80, 80, 80), 1)

            # 2. Text Prediksi Utama & Warna Kategori
            if display_pred == "NO OBJECT":
                pred_color = LABEL_COLORS_BGR.get("NO OBJECT", (150, 150, 150))
                pred_text = "STATUS: NO OBJECT"
            elif display_pred == "PASSING":
                pred_color = (0, 255, 255)
                pred_text = f"LEWAT (SPEED {CONVEYOR_PASS_SPEED})"
            else:
                is_passed, req_threshold = is_confidence_sufficient(display_pred, display_conf)
                if is_passed:
                    pred_color = LABEL_COLORS_BGR.get(display_pred, (0, 255, 0))
                    pred_text = f"{display_pred.upper()} ({display_conf:.1f}%)"
                else:
                    pred_color = LABEL_COLORS_BGR["LOW_CONF"]
                    pred_text = f"{display_pred.upper()} ({display_conf:.1f}%) [DIBAWAH {req_threshold:.0f}%]"

            cv2.putText(display, pred_text, (25, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.82, pred_color, 2)

            # 3. Status Subtitle
            cv2.putText(display, f"Alur   : {status_text}", (25, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            # 4. Info Hardware & Jarak Sensor Real-Time
            ard_status = f"CONNECTED ({controller.port})" if (controller and controller.is_connected) else "OFFLINE"
            ard_col = (50, 205, 50) if (controller and controller.is_connected) else (80, 80, 220)
            dist_str = f"{live_distance:.0f}cm" if live_distance < 900 else "--"
            dist_col = (50, 205, 50) if live_distance <= SENSOR_MAX_DISTANCE_CM else (200, 200, 200)

            cv2.putText(display, f"Arduino : {ard_status} | Jarak: {dist_str} (Maks: {SENSOR_MAX_DISTANCE_CM:.0f}cm)", (25, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, dist_col, 1)

            focus_txt = ("Focus: AUTO" if cur_autofocus else f"Focus: {cur_focus}") if supports_focus else "Focus: FIXED"
            cv2.putText(display, f"Aksi    : {last_action_note} | FPS: {fps:.1f} | {focus_txt}", (25, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # 5. Panel Bawah (Bantuan Tombol)
            cv2.rectangle(display, (10, h - 35), (w - 10, h - 8), (20, 20, 20), -1)
            help_msg = "[Spasi] Scan 3s | [4] Motor 255 | [0] Motor Stop | [1/2/3] Servo | [M] Mode | [Q] Keluar"
            cv2.putText(display, help_msg, (20, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            # Tampilkan Window
            cv2.imshow(window_name, display)

            # ----------------------------------------------------
            # RESPON TOMBOL KEYBOARD
            # ----------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("\n🛑 Sistem dihentikan oleh user.")
                break

            # Manual Trigger Klasifikasi Saat Ini [Space / C]
            elif key == ord(' ') or key == ord('c'):
                print("\n⚡ [MANUAL TRIGGER] Menghentikan konveyor, memulai scan manual 3 detik...")
                if controller and controller.is_connected:
                    controller.motor_stop()
                prediction_history.clear()
                scan_start_time = time.time()
                system_state = "CLASSIFYING"

            # Ganti Mode [M]
            elif key == ord('m'):
                if system_state in ["WAITING_SENSOR", "CLASSIFYING", "WAITING_OBJECT_CLEAR"]:
                    system_state = "CONTINUOUS"
                    if controller and controller.is_connected:
                        controller.motor_stop()
                    print("\n🔀 Mode dialihkan ke: CONTINUOUS (Klasifikasi live)")
                else:
                    system_state = "WAITING_SENSOR"
                    if controller and controller.is_connected:
                        controller.motor_forward(CONVEYOR_STANDBY_SPEED)
                    print("\n🔀 Mode dialihkan ke: SENSOR ULTRASONIK (Konveyor aktif)")
                prediction_history.clear()

            # Reset Status [R]
            elif key == ord('r'):
                prediction_history.clear()
                confirmed_label = None
                current_active_servo = None
                servo_closed = True
                system_state = "WAITING_SENSOR" if mode_sensor else "CONTINUOUS"
                last_action_note = "RESET STATUS"
                if controller and controller.is_connected:
                    for lbl, ang in SERVO_NEUTRAL_ANGLES.items():
                        controller.set_servo_angle(lbl, ang)
                    if mode_sensor:
                        controller.motor_forward(CONVEYOR_STANDBY_SPEED)
                    else:
                        controller.motor_stop()
                print("\n🔄 Sistem direset ke posisi awal.")

            # Tes Manual Motor Speed 255 [4]
            elif key == ord('4'):
                print(f"🧪 Tes manual: MOTOR MAJU SPEED {CONVEYOR_PASS_SPEED}")
                if controller and controller.is_connected:
                    controller.motor_forward(CONVEYOR_PASS_SPEED)
                last_action_note = f"TES -> MOTOR {CONVEYOR_PASS_SPEED}"

            # Tes Manual Motor Standby [5]
            elif key == ord('5'):
                print(f"🧪 Tes manual: MOTOR STANDBY SPEED {CONVEYOR_STANDBY_SPEED}")
                if controller and controller.is_connected:
                    controller.motor_forward(CONVEYOR_STANDBY_SPEED)
                last_action_note = f"TES -> MOTOR {CONVEYOR_STANDBY_SPEED}"

            # Tes Manual Motor Stop [0 / S]
            elif key == ord('0') or key == ord('s'):
                print("🧪 Tes manual: MOTOR STOP (0)")
                if controller and controller.is_connected:
                    controller.motor_stop()
                last_action_note = "TES -> MOTOR STOP"

            # Toggle Autofocus [A]
            elif key == ord('a'):
                if supports_focus:
                    cur_autofocus = not cur_autofocus
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if cur_autofocus else 0)
                    cv2.setTrackbarPos("Auto Focus (0/1)", window_name, 1 if cur_autofocus else 0)
                    if not cur_autofocus:
                        cap.set(cv2.CAP_PROP_FOCUS, cur_focus)
                    last_action_note = f"AUTOFOCUS -> {'ON' if cur_autofocus else 'OFF'}"
                    print(f"📷 Autofocus: {'ON' if cur_autofocus else 'OFF (Manual)'}")

            # Atur Fokus Bertambah [ ] ] atau [+]
            elif key == ord(']') or key == ord('+') or key == ord('='):
                if supports_focus:
                    cur_autofocus = False
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    cv2.setTrackbarPos("Auto Focus (0/1)", window_name, 0)
                    step = 50 if key in [ord('+'), ord('=')] else 10
                    cur_focus = min(1000, cur_focus + step)
                    cap.set(cv2.CAP_PROP_FOCUS, cur_focus)
                    cv2.setTrackbarPos("Focus", window_name, cur_focus)
                    last_action_note = f"FOKUS -> {cur_focus}"
                    print(f"🔍 Fokus Kamera: {cur_focus}")

            # Atur Fokus Berkurang [ [ ] atau [-]
            elif key == ord('[') or key == ord('-'):
                if supports_focus:
                    cur_autofocus = False
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    cv2.setTrackbarPos("Auto Focus (0/1)", window_name, 0)
                    step = 50 if key == ord('-') else 10
                    cur_focus = max(0, cur_focus - step)
                    cap.set(cv2.CAP_PROP_FOCUS, cur_focus)
                    cv2.setTrackbarPos("Focus", window_name, cur_focus)
                    last_action_note = f"FOKUS -> {cur_focus}"
                    print(f"🔍 Fokus Kamera: {cur_focus}")

            # Tes Servo Manual [1 / 2 / 3]
            elif key == ord('1'):
                ang = SERVO_ACTIVE_ANGLES.get("Organik", 90)
                print(f"🧪 Tes manual: ORGANIK -> Servo D9 ke {ang}°")
                if controller and controller.is_connected:
                    controller.set_servo_angle("Organik", ang)
                last_action_note = f"TES -> ORGANIK {ang}°"
            elif key == ord('2'):
                ang = SERVO_ACTIVE_ANGLES.get("Anorganik", 90)
                print(f"🧪 Tes manual: ANORGANIK -> Servo D10 ke {ang}°")
                if controller and controller.is_connected:
                    controller.set_servo_angle("Anorganik", ang)
                last_action_note = f"TES -> ANORGANIK {ang}°"
            elif key == ord('3'):
                ang = SERVO_ACTIVE_ANGLES.get("Residu", 90)
                print(f"🧪 Tes manual: RESIDU -> Servo D11 ke {ang}°")
                if controller and controller.is_connected:
                    controller.set_servo_angle("Residu", ang)
                last_action_note = f"TES -> RESIDU {ang}°"

    finally:
        if controller and controller.is_connected:
            print("🛑 Mematikan motor konveyor...")
            controller.motor_stop()
            for lbl, ang in SERVO_NEUTRAL_ANGLES.items():
                controller.set_servo_angle(lbl, ang)
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SEGRATRUK - Sistem Klasifikasi Sampah Hierarkis Real-Time"
    )
    parser.add_argument("--camera", type=int, default=None,
                        help="Index kamera webcam (misal: 1 atau 0). Default: otomatis deteksi.")
    parser.add_argument("--port", type=str, default=None,
                        help="Port serial Arduino (misal: COM9). Default: otomatis deteksi.")
    parser.add_argument("--model1", type=str, default=DEFAULT_MODEL1_PATH,
                        help="Path file Model 1 (Organik vs Anorganik)")
    parser.add_argument("--model2", type=str, default=DEFAULT_MODEL2_PATH,
                        help="Path file Model 2 (Residu vs Anorganik)")
    parser.add_argument("--no-arduino", action="store_true",
                        help="Jalankan tanpa menghubungkan ke Arduino (mode simulasi kamera saja)")
    parser.add_argument("--continuous", action="store_true",
                        help="Mulai langsung dalam mode klasifikasi terus-menerus tanpa menunggu sensor ultrasonik")
    parser.add_argument("--image", type=str, default=None,
                        help="Prediksi satu file gambar statis lalu cetak hasilnya (tanpa kamera live)")
    parser.add_argument("--focus", type=int, default=None,
                        help="Atur nilai fokus kamera (0 - 1000). Mengaktifkan mode fokus manual.")
    parser.add_argument("--autofocus", action="store_true",
                        help="Aktifkan autofocus kamera secara otomatis.")
    parser.add_argument("--scan", action="store_true",
                        help="Pindai port Arduino dan kamera webcam yang terdeteksi lalu keluar")

    args = parser.parse_args()

    # Mode Scan
    if args.scan:
        list_serial_ports()
        list_cameras()
        return

    # Inisialisasi Classifier
    classifier = HierarchicalWasteClassifier(
        model1_path=args.model1,
        model2_path=args.model2
    )

    # Mode Single Image
    if args.image:
        classifier.load()
        print(f"\n🖼️ Memproses gambar: {args.image}")
        res = classifier.predict_image_file(args.image)
        print("\n" + "=" * 45)
        print("HASIL PREDIKSI HIERARKIS")
        print("=" * 45)
        print(f"Hasil Akhir : {res['final_label'].upper()}")
        print(f"Confidence  : {res['confidence']:.2f}%")
        is_passed, req_thresh = is_confidence_sufficient(res['final_label'], res['confidence'])
        print(f"Status Servo: {'🟢 AKAN BERGERAK' if is_passed else f'⛔ DIAM (Confidence < {req_thresh:.0f}%)'}")
        print(f"Model 1     : {res['model1_label']} ({res['model1_conf']:.2f}%)")
        if res['model2_label']:
            print(f"Model 2     : {res['model2_label']} ({res['model2_conf']:.2f}%)")
        print("=" * 45)
        return

    # Muat Model untuk Realtime
    classifier.load()

    # Inisialisasi Arduino
    controller = None
    if not args.no_arduino:
        port = args.port or find_arduino_port() or "COM9"
        controller = ArduinoController(port=port)
        ok = controller.connect()
        if not ok:
            print("⚠️ Melanjutkan sistem dalam mode kamera tanpa Arduino...")
    else:
        print("\n⚡ Mode tanpa Arduino (--no-arduino aktif).")

    # Tentukan Kamera
    cam_idx = args.camera
    if cam_idx is None:
        cameras = list_cameras()
        if 1 in cameras:
            cam_idx = 1
        elif cameras:
            cam_idx = cameras[0]
        else:
            cam_idx = 0

    try:
        run_realtime_system(
            classifier=classifier,
            controller=controller,
            camera_index=cam_idx,
            mode_sensor=not args.continuous,
            focus=args.focus,
            autofocus=args.autofocus
        )
    finally:
        if controller:
            if controller.is_connected:
                controller.motor_stop()
                controller.reset_servos()
            controller.close()


if __name__ == "__main__":
    main()
