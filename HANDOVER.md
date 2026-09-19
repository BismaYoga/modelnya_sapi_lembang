# 📦 DOKUMEN HANDOVER PROYEK SEGRATRUK
> **Sistem Pengelolaan Sampah Cerdas & Terintegrasi Berbasis AI**  
> *Tanggal Handover*: 20 September 2026  
> *Penulis / Pengembang*: Pair Programming AI Assistant & Bisma Yoga

---

## 📑 DAFTAR ISI
1. [Ringkasan Proyek & Arsitektur](#1-ringkasan-proyek--arsitektur)
2. [Akses Server VPS & Konfigurasi Domain](#2-akses-server-vps--konfigurasi-domain)
3. [Aplikasi Web Frontend (SEGRATRUK)](#3-aplikasi-web-frontend-segratruk)
4. [Model AI & Deep Learning (Klasifikasi Sampah)](#4-model-ai--deep-learning-klasifikasi-sampah)
5. [Proyek Animasi Iklan SaaS Video (Remotion)](#5-proyek-animasi-iklan-saas-video-remotion)
6. [Struktur File & Direktori Proyek](#6-struktur-file--direktori-proyek)
7. [Panduan Operasional & Maintenance (SOP)](#7-panduan-operasional--maintenance-sop)

---

## 1. RINGKASAN PROYEK & ARSITEKTUR

**SEGRATRUK** adalah platform Smart Waste Management yang mengintegrasikan pelacakan armada truk sampah real-time (GPS fleet command) dengan sistem pemilahan sampah otomatis menggunakan computer vision deep learning.

### 🌐 Repositori GitHub
| Komponen | Repositori GitHub | Branch | Path Lokal |
| :--- | :--- | :--- | :--- |
| **AI Models & Training** | `https://github.com/BismaYoga/modelnya_sapi_lembang.git` | `main` | `C:\Users\Bisma\Downloads\Documents\tanin` |
| **Website Frontend** | `https://github.com/BismaYoga/segratruk.git` | `main` | `C:\Users\Bisma\Downloads\Documents\tanin\website` |

---

## 2. AKSES SERVER VPS & KONFIGURASI DOMAIN

### A. Informasi Server
- **IP Server**: `202.10.47.34`
- **Domain Wildcard**: `http://segratruk.202.10.47.34.nip.io`
- **Port Operasional**: `80` (HTTP Standar)
- **Status Layanan**: Online & Aktif

### B. Konfigurasi Layanan di Server
Server menjalankan web server static file untuk aplikasi frontend `segratruk` pada port 80.
- **Direktori Aplikasi di Server**: `/root/segratruk` (atau folder deploy git clone)
- **Command Menjalankan Server di Background**:
  ```bash
  nohup python3 -m http.server 80 --directory /root/segratruk > /var/log/segratruk.log 2>&1 &
  ```
- **Memeriksa Status Proses**:
  ```bash
  ps aux | grep "http.server"
  netstat -tuln | grep 80
  ```

---

## 3. APLIKASI WEB FRONTEND (SEGRATRUK)

Frontend dibangun menggunakan arsitektur Vanilla HTML5, CSS3 modern, dan JavaScript modern tanpa framework berat, memastikan performa instan dan konsumsi resource minimal.

### A. Aset Brand Resmi
Terletak di `website/` (dan disalin ke `video/public/`):
- `logo.png` — Logo resmi SEGRATRUK (ikon daur ulang hijau modern)
- `hero-landing.png` — Gambar truk armada dengan latar siluet kota modern
- `hero-next-lanjut.png` — Ilustrasi armada pendukung
- `gambartruk-home.png` — Miniatur truk armada untuk indikator dashboard

### B. Routing & Mobile First Navigation
- Terdapat logika auto-routing: Saat website pertama kali dibuka (terutama pada perangkat mobile / Android), pengguna akan diarahkan ke **Landing Page** terlebih dahulu untuk memahami sistem sebelum masuk ke Command Center.
- Navigasi mulus antar view (*Landing Page* ↔ *Live Dashboard Fleet*).

---

## 4. MODEL AI & DEEP LEARNING (KLASIFIKASI SAMPAH)

### A. Model 3 Kelas: Organik, Anorganik, Residu
- **Arsitektur**: MobileNetV2 (Transfer Learning + Fine-Tuning)
- **Dataset**:
  - *Organik*: `arthurwaruwu/datasetcapstonefixx` (Kaggle)
  - *Anorganik*: `phenomsg/waste-classification` (Folder `Recyclable`)
  - *Residu*: `phenomsg/waste-classification` (Folder `Non-Recyclable`)
- **File Model**:
  - `waste_classifier_organik_anorganik_residu.keras` (Model Keras v3)
  - `waste_classifier.tflite` (Model Edge/Mobile TFLite ~3.3 MB)
  - `class_info.json` (Label mapping & parameter normalisasi)
  - `waste_classification.py` / `waste_classification.ipynb` (Pipeline training)

### B. Model Biner: Recyclable vs Non-Recyclable
- **Arsitektur**: MobileNetV2 Binary Classifier
- **File Model**:
  - `binary_model_final.keras`
  - `binary_model.tflite` (~2.6 MB)
  - `binary_class_info.json`
  - `binary_classification.py` / `binary_classification.ipynb`

---

## 5. PROYEK ANIMASI IKLAN SAAS VIDEO (REMOTION)

Proyek video berformat **Landscape (1920x1080 @ 30fps)** berlokasi di direktori mandiri `video/` dan diabaikan dari git utama (`.gitignore`).

### A. Lokasi File Video Hasil Render
- **File MP4 Utama**:  
  👉 [`video/out/segratruk-commercial.mp4`](file:///C:/Users/Bisma/Downloads/Documents/tanin/video/out/segratruk-commercial.mp4)  
  *(Resolusi: 1920x1080 | Format: MP4 | 30 FPS | Durasi: 24 detik / 720 frames | Ukuran: ~18.4 MB)*

### B. Karakteristik Desain & Storyboard Video
1. **Background Aesthetic**: Putih cerah (*Light SaaS Studio*) dengan aksen hijau emerald, pendaran aurora orb lembut, dan cyber grid lines halus.
2. **Dynamic 3D Camera Tilts (Tidak Kaku)**: Menggunakan transformasi ruang 3D (`perspective: 1400px`, `rotateX`, `rotateY`, `rotateZ`), sudut isometrik melayang, serta camera pan & swoop.
3. **Virtual Animated Cursor**: Pointer kursor halus yang mengarahkan pandangan audiens, lengkap dengan interaksi hover dan animasi klik dengan **gelombang lingkaran klik hijau (*emerald click ripple*)**.
4. **Alur Interaksi Antar Halaman (Flow)**:
   - *Act 1 (0s–5.5s)*: Sudut 3D miring pada Landing Page resmi -> kursor meluncur dan mengklik tombol *"Masuk Dashboard"*.
   - *Act 2 (5.5s–11.5s)*: Layar bertransisi ke Dashboard -> kamera **zoom in tajam** ke panel peta GPS rute armada -> kursor mengklik armada `TRK-01`.
   - *Act 3 (11.5s–17.5s)*: Kamera meluncur (*glide*) ke panel kanan menyorot **Klasifikasi AI MobileNetV2** -> kursor mengklik switch toggle **"OTOMATIS: AKTIF"** -> counter volume sampah bertambah live.
   - *Act 4 (17.5s–21s)*: **Grand Zoom Out** dramatis ke sudut isometrik floating window dengan pop-up badge notifikasi 3D: *"Pemilahan Otomatis Berhasil! Efisiensi Rute +42%"*.
   - *Act 5 (21s–24s)*: Outro elegan dengan logo resmi SEGRATRUK dan tombol call to action `segratruk.202.10.47.34.nip.io`.
5. **Bahasa & Teks**: Sepenuhnya menggunakan Bahasa Indonesia dengan gaya mikro-copy minimalis (tanpa teks penjelasan panjang).

### C. Komponen Utama Remotion (`video/src/`)
- `Root.tsx`: Registrasi komposisi video 1920x1080 30fps.
- `SegratrukCommercial.tsx`: Orchestrator utama timeline, 3D camera rig, kursor, dan transisi halaman.
- `components/BrowserMockup.tsx`: Bingkai browser frosted glass putih dengan traffic lights macOS dan address bar SSL.
- `components/GlowBackground.tsx`: Latar belakang putih cerah dengan pendaran hijau emerald.
- `components/VirtualCursor.tsx`: Kursor virtual animasi dengan gelombang ripple klik.
- `components/views/LandingPageView.tsx`: Komponen antarmuka Landing Page.
- `components/views/DashboardView.tsx`: Komponen antarmuka Fleet Command & Klasifikasi AI.

---

## 6. STRUKTUR FILE & DIREKTORI PROYEK

```text
C:\Users\Bisma\Downloads\Documents\tanin\
├── HANDOVER.md                                      # Dokumen handover utama (file ini)
├── SESSION_LOG.md                                   # Log riwayat percakapan & pengembangan
├── .gitignore                                       # Mengabaikan video/ dan file besar
│
├── [MODEL & TRAINING ARTIFACTS]
│   ├── waste_classification.py / .ipynb             # Pipeline 3-kelas
│   ├── waste_classifier_organik_anorganik_residu.keras
│   ├── waste_classifier.tflite
│   ├── class_info.json
│   ├── binary_classification.py / .ipynb            # Pipeline biner
│   ├── binary_model_final.keras
│   ├── binary_model.tflite
│   └── binary_class_info.json
│
├── website/                                         # [REPO GITHUB: segratruk]
│   ├── index.html                                   # Halaman utama aplikasi web
│   ├── style.css                                    # Desain visual & layout responsif
│   ├── app.js                                       # Logika aplikasi, tracking & routing
│   ├── logo.png                                     # Logo resmi
│   ├── hero-landing.png                             # Asset hero landing page
│   ├── hero-next-lanjut.png                         # Asset hero pendukung
│   └── gambartruk-home.png                          # Asset miniatur armada truk
│
└── video/                                           # [REMOTION SAAS VIDEO PROJECT]
    ├── package.json                                 # Dependency Remotion & React
    ├── remotion.config.ts                           # Konfigurasi Chromium headless
    ├── public/                                      # Asset logo & gambar video
    ├── src/
    │   ├── index.ts                                 # Entry point Remotion
    │   ├── Root.tsx                                 # Konfigurasi resolusi & durasi
    │   ├── SegratrukCommercial.tsx                  # Master orchestrator iklan 3D
    │   └── components/
    │       ├── BrowserMockup.tsx
    │       ├── GlowBackground.tsx
    │       ├── VirtualCursor.tsx
    │       └── views/
    │           ├── LandingPageView.tsx
    │           └── DashboardView.tsx
    └── out/
        └── segratruk-commercial.mp4                 # Video MP4 final hasil render (18.4 MB)
```

---

## 7. PANDUAN OPERASIONAL & MAINTENANCE (SOP)

### A. Menjalankan Preview Video Remotion Studio
```powershell
cd C:\Users\Bisma\Downloads\Documents\tanin\video
npm start
```
Buka browser di `http://localhost:3000` untuk memutar timeline secara interaktif.

### B. Merender Ulang Video MP4
```powershell
cd C:\Users\Bisma\Downloads\Documents\tanin\video
npx remotion render src/index.ts SegratrukPromo out/segratruk-commercial.mp4
```

### C. Deploy Perubahan Website ke VPS
1. Lakukan commit dan push dari folder `website`:
   ```powershell
   cd C:\Users\Bisma\Downloads\Documents\tanin\website
   git add .
   git commit -m "update website"
   git push origin main
   ```
2. Login ke VPS dan tarik update terbaru:
   ```bash
   ssh root@202.10.47.34
   cd /root/segratruk
   git pull origin main
   ```
   *(Karena menggunakan HTTP server statis, perubahan akan langsung aktif seketika tanpa perlu restart service).*

### D. Restart Service Web di VPS (Jika Diperlukan)
```bash
# Matikan proses http server lama
pkill -f "python3 -m http.server 80"

# Jalankan kembali di background
nohup python3 -m http.server 80 --directory /root/segratruk > /var/log/segratruk.log 2>&1 &
```

---
*Dokumen ini disusun untuk memudahkan handover teknis dan operasional sistem SEGRATRUK kepada tim developer, penguji, maupun stakeholder terkait.*
