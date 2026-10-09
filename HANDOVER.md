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
  *(Resolusi: 1920x1080 | Format: MP4 | 30 FPS | Durasi: 30 detik / 900 frames | Ukuran: ~13.7 MB)*

### B. Arsitektur 9 Scene Berpacing Cepat & Tata Suara (Audio Synced & Spring Physics)
Setiap scene dibungkus dalam Remotion `<Sequence>` dengan normalisasi frame lokal (`0 to duration`), memanfaatkan `spring()` physics dan kurva `Easing` untuk gerakan yang organik, elastis, dan bebas dari kesan kaku:
1. **Scene 1 (0.0s–2.5s) | Opening Title Card**:
   - Tipografi elegan high-contrast SEGRATRUK (94px) dengan tracking expand dan dual-node SVG (tanpa emoji).
   - Dua kartu telemetri tech melayang di samping untuk mengisi whitespace.
   - SFX: Ambient tech riser & scene whoosh.
2. **Scene 2 (2.5s–5.5s) | Truk Armada sebagai Sumber Data Primer**:
   - Menghubungkan storytelling fisik: armada truk mengumpulkan sampah dan data di jalanan.
   - Grid 1540px lebih rapat & besar; 3 Callout sensor dengan spring pop-in dan continuous floating.
   - SFX: 3x pop audio teratur untuk tiap sensor.
3. **Scene 3 (5.5s–8.5s) | Mobile App Splash & Interactive Enter**:
   - Smartphone mockup front-facing dengan idle float lembut & subtle 3D tilt.
   - Kursor meluncur halus dengan *cubic deceleration* mengklik tombol *"Mulai Jelajahi"* dengan elastic bounce.
   - SFX: Click UI + whoosh transition.
4. **Scene 4 (8.5s–12.0s) | Mobile Dashboard — Live Counter Growth**:
   - Status armada `TRK-01`, live counter berputar cepat naik dari 0 ke 4.200 kg (Organik $2.350\text{ kg}$, Anorganik $1.250\text{ kg}$, Residu $600\text{ kg}$).
   - Grafik kurva analitik SVG bertumbuh (*scale-Y growth*) dari baseline (+12.4% vs kemarin). Side pills meluncur spring dari kiri-kanan.
5. **Scene 5 (12.0s–16.0s) | GPS Fleet Tracking & Animated Route Drawing**:
   - Peta vektor denah kota Denpasar (framing rapat).
   - **Animated Path Drawing**: Garis rute hijau terang digambar melengkung otomatis di jalan raya (`strokeDashoffset`).
   - Marker truk melaju di jalurnya, speedometer fluktuatif ($22\text{--}28\text{ km/jam}$), kapasitas tangki terisi ($85\%$).
6. **Scene 6 (16.0s–19.5s) | AI Sort Toggle & Category Cascade**:
   - Kursor mengklik switch toggle AI $\rightarrow$ pendaran hijau menyala *"OTOMATIS: AKTIF"*.
   - 3 kartu pemilahan (Organik, Anorganik, Residu) meluncur masuk bertingkat pasca-klik (*staggered cascade*).
   - SFX: UI switch click + 3x pop audio.
7. **Scene 7 (19.5s–23.0s) | 3D Checkmark Celebration & Confetti**:
   - Pop-up **3D Floating Checkmark Badge** berputar dan meletup dengan 22 partikel konfeti digital multi-warna & multi-bentuk menyebar ke seluruh kanvas dengan gravitasi lembut.
   - Ticker angka metrics (+42%, 99.2%, 0%).
   - SFX: Dual bell chime / celebration ding.
8. **Scene 8 (23.0s–26.5s) | Rute & Jadwal TPS Timeline**:
   - Isometric 3D tilt (-2.5 deg) dengan timeline penjemputan TPS. Status TPS Sesetan bertransformasi menjadi *"Selesai ✓"* dengan badge pop-in scale & garis konektor SVG beranimasi.
9. **Scene 9 (26.5s–30.0s) | Grand Brand Outro & CTA**:
   - Hero card brand resmi SEGRATRUK (940px) dengan logo neon emerald, tracking text expand, 3 floating feature pills.
   - Tombol CTA interaktif `segratruk.202.10.47.34.nip.io` dengan efek breathing, elastic press, dan expanding ripple ring.
   - SFX: Click audio + musical outro resolution.

### C. Komponen Utama Remotion (`video/src/`)
- `Root.tsx`: Registrasi komposisi video 1920x1080 30fps (900 frames / 30 detik).
- `SegratrukCommercial.tsx`: Master timeline orchestrator 9 scene dengan sinkronisasi audio BGM & SFX.
- `generate_audio.py`: Script synthesizer audio stereo 44.1kHz (`bgm.wav`, `whoosh.wav`, `click.wav`, `pop.wav`, `chime.wav`).
- `scenes/`:
  - `Scene1TitleCard.tsx`: Title card pembuka tipografi elegan.
  - `Scene2TruckSource.tsx`: Truk armada sumber data dengan 3 callout sensor.
  - `Scene3AppSplash.tsx`: Splash screen mobile framing rapat.
  - `Scene4DashboardCounters.tsx`: Dashboard live counter 0 -> 4.200 kg & grafik tumbuh.
  - `Scene5GPSTracking.tsx`: Peta vektor denpasar, animated path drawing, telemetri HUD.
  - `Scene6AISortToggle.tsx`: AI sort switch click & staggered cards.
  - `Scene7Celebration.tsx`: 3D checkmark pop-up & konfeti digital.
  - `Scene8RouteTimeline.tsx`: Timeline penjemputan TPS & status pop-in.
  - `Scene9BrandOutro.tsx`: Grand outro brand hero card & tombol CTA.
- `components/`:
  - `PhoneMockup.tsx`: Hardware mockup smartphone 3D (titanium bezel, Dynamic Island, status bar).
  - `GlowBackground.tsx`: Latar studio cerah dengan aurora emerald dan grid cyber.
  - `VirtualCursor.tsx`: Kursor pointer virtual dengan efek ripple klik hijau.
  - `mobile/`: Komponen layar mobile SEGRATRUK individual.



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
