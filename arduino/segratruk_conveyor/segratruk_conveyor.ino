/*
 * segratruk_conveyor.ino
 * =====================================================================
 * Firmware Jembatan Hardware (Hardware Bridge) SEGRATRUK
 * Seluruh logika otomasi, sudut servo, timer per-kategori, dan jarak
 * DIATUR DAN DIKONTROL 100% OLEH SCRIPT PYTHON (main.py).
 * 
 * Hardware & Pinout:
 * 1. Sensor Ultrasonik HC-SR04:
 *    - Trig : Pin D13
 *    - Echo : Pin D12
 * 
 * 2. Motor Driver L298N (Konveyor):
 *    - ENA  : Pin D5 (PWM Kecepatan)
 *    - IN1  : Pin D4 (Arah Putar 1)
 *    - IN2  : Pin D7 (Arah Putar 2)
 * 
 * 3. Servo Pemilah:
 *    - Organik   : Pin D9
 *    - Anorganik : Pin D10
 *    - Residu    : Pin D11
 * 
 * Protokol Serial (9600 baud):
 * Python -> Arduino:
 *   - "SET_MAX_DIST:<cm>"     -> Mengatur batas jarak deteksi maksimal (default 35cm)
 *   - "MOTOR_FORWARD:<speed>" -> Motor maju dengan PWM speed (0-255)
 *   - "MOTOR_STOP"           -> Motor berhenti seketika
 *   - "SERVO:<label>:<angle>"-> Set servo tertentu ke sudut derajat (0-180)
 *   - "ORGANIK" / "ANORGANIK" / "RESIDU" -> Trigger servo ke sudut aktif default
 *   - "RESET"                -> Semua servo kembali ke sudut 0°
 * 
 * Arduino -> Python:
 *   - "DIST:<cm>"            -> Data jarak sensor real-time
 *   - "OBJECT_DETECTED"      -> Dikirim saat sensor mendeteksi objek (jarak <= threshold)
 *   - "OBJECT_CLEARED"       -> Dikirim saat sensor bersih dari objek
 * =====================================================================
 */

#include <Servo.h>

// Definisi Pin Ultrasonik
const int PIN_TRIG = 13;
const int PIN_ECHO = 12;

// Definisi Pin Motor L298N
const int PIN_ENA  = 5;   // PWM Speed
const int PIN_IN1  = 4;   // Direction 1
const int PIN_IN2  = 7;   // Direction 2

// Definisi Pin Servo
const int PIN_SERVO_ORGANIK   = 9;
const int PIN_SERVO_ANORGANIK = 10;
const int PIN_SERVO_RESIDU    = 11;

// Objek Servo
Servo servoOrganik;
Servo servoAnorganik;
Servo servoResidu;

// Ambang batas jarak (bisa diubah dari Python lewat SET_MAX_DIST:<cm>)
int distanceThreshold = 35;  // Default dinaikkan ke 35 cm agar mudah mendeteksi sampah
bool objectCurrentlyDetected = false;
unsigned long lastDistReportTime = 0;

// =====================================================================
// FUNGSI MOTOR
// =====================================================================
void setMotor(int speed) {
  if (speed <= 0) {
    digitalWrite(PIN_IN1, LOW);
    digitalWrite(PIN_IN2, LOW);
    analogWrite(PIN_ENA, 0);
  } else {
    digitalWrite(PIN_IN1, HIGH);
    digitalWrite(PIN_IN2, LOW);
    analogWrite(PIN_ENA, constrain(speed, 0, 255));
  }
}

// =====================================================================
// FUNGSI SERVO
// =====================================================================
void setServoAngle(String label, int angle) {
  angle = constrain(angle, 0, 180);
  if (label == "Organik" || label == "ORGANIK") {
    servoOrganik.write(angle);
  } else if (label == "Anorganik" || label == "ANORGANIK") {
    servoAnorganik.write(angle);
  } else if (label == "Residu" || label == "RESIDU") {
    servoResidu.write(angle);
  }
}

void resetAllServos() {
  servoOrganik.write(0);
  servoAnorganik.write(0);
  servoResidu.write(0);
}

// =====================================================================
// FUNGSI SENSOR ULTRASONIK
// =====================================================================
int readDistance() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  long duration = pulseIn(PIN_ECHO, HIGH, 30000); // timeout 30ms (~5 meter)
  if (duration == 0) return 999;
  return duration * 0.034 / 2;
}

// =====================================================================
// SETUP
// =====================================================================
void setup() {
  Serial.begin(9600);

  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);

  pinMode(PIN_ENA, OUTPUT);
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);

  servoOrganik.attach(PIN_SERVO_ORGANIK);
  servoAnorganik.attach(PIN_SERVO_ANORGANIK);
  servoResidu.attach(PIN_SERVO_RESIDU);

  resetAllServos();
  setMotor(0);

  Serial.println("ARDUINO_BRIDGE_READY");
}

// =====================================================================
// LOOP UTAMA
// =====================================================================
void loop() {
  // 1. Terima Perintah dari Python
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("MOTOR_FORWARD:")) {
      int speed = cmd.substring(14).toInt();
      setMotor(speed);
    } else if (cmd == "MOTOR_STOP") {
      setMotor(0);
    } else if (cmd.startsWith("SET_MAX_DIST:")) {
      distanceThreshold = cmd.substring(13).toInt();
    } else if (cmd.startsWith("SERVO:")) {
      // Format: SERVO:<label>:<angle>  misal: SERVO:Organik:90
      int firstColon = cmd.indexOf(':');
      int secondColon = cmd.indexOf(':', firstColon + 1);
      if (secondColon > 0) {
        String label = cmd.substring(firstColon + 1, secondColon);
        int angle = cmd.substring(secondColon + 1).toInt();
        setServoAngle(label, angle);
      }
    } else if (cmd == "ORGANIK") {
      servoOrganik.write(90);
    } else if (cmd == "ANORGANIK") {
      servoAnorganik.write(90);
    } else if (cmd == "RESIDU") {
      servoResidu.write(90);
    } else if (cmd == "RESET" || cmd == "SERVO_RESET") {
      resetAllServos();
    }
  }

  // 2. Baca Sensor Jarak
  int dist = readDistance();
  unsigned long now = millis();

  // Kirim data jarak live ke Python setiap 150ms agar bisa tampil di HUD
  if (now - lastDistReportTime >= 150) {
    lastDistReportTime = now;
    if (dist < 999) {
      Serial.print("DIST:");
      Serial.println(dist);
    }
  }

  // Deteksi Perubahan Status Objek
  if (dist > 2 && dist <= distanceThreshold) {
    if (!objectCurrentlyDetected) {
      objectCurrentlyDetected = true;
      Serial.println("OBJECT_DETECTED");
    }
  } else if (dist > (distanceThreshold + 4) || dist >= 999) { // Histeresis 4cm
    if (objectCurrentlyDetected) {
      objectCurrentlyDetected = false;
      Serial.println("OBJECT_CLEARED");
    }
  }

  delay(20);
}
