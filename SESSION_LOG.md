# 📋 Session Log — Proyek Klasifikasi Sampah (Tanin)

> **Tujuan Proyek**: Membuat model klasifikasi gambar sampah ke dalam 3 kategori: **Organik**, **Anorganik**, dan **Residu** menggunakan Transfer Learning (MobileNetV2).
>
> **Session ID**: `da403f51-fe35-48bb-80bf-f8d29e650aab`  
> **Tanggal Mulai**: 2026-08-23

---

## ⚠️ Aturan Proyek

1. **SESSION_LOG.md WAJIB di-update setiap prompt** — tanpa terkecuali. Setiap interaksi harus tercatat di log ini.

---

## Sesi 1 — 2026-08-23 (10:21 - 10:26)

### Prompt 1 (10:21) — Request Pembuatan File Klasifikasi
**User Request:**
- Buat file `.py` (bukan notebook langsung) untuk klasifikasi sampah 3 kategori:
  - **Organik** → dari dataset [arthurwaruwu/datasetcapstonefixx](https://www.kaggle.com/datasets/arthurwaruwu/datasetcapstonefixx)
  - **Anorganik** → folder `Recyclable` dari dataset [phenomsg/waste-classification](https://www.kaggle.com/datasets/phenomsg/waste-classification)
  - **Residu** → folder `Non-Recyclable` dari dataset [phenomsg/waste-classification](https://www.kaggle.com/datasets/phenomsg/waste-classification)
- Harus ada **preprocessing** agar jumlah data tidak jomplang dan input seragam.

**Apa yang dilakukan:**
1. Riset struktur kedua dataset dari Kaggle:
   - Dataset `phenomsg/waste-classification` punya 4 folder: Hazardous, Non-Recyclable, Organic, Recyclable. Masing-masing punya subfolder.
     - Recyclable: Paper, Glass, Plastic-Bottles
     - Non-Recyclable: Plastic-Wrappers, Styrofoam, Food-Cups
   - Dataset `arthurwaruwu/datasetcapstonefixx` → dataset organik (~1.3GB zip)
2. Membuat file [`waste_classification.py`](file:///C:/Users/Bisma/Downloads/Documents/tanin/waste_classification.py)

**Isi file `waste_classification.py`:**

| # | Bagian | Deskripsi |
|---|--------|-----------|
| 1 | Install & Import | TensorFlow, PIL, sklearn, matplotlib, seaborn |
| 2 | Download Dataset | Via Kaggle API (`kaggle datasets download`) |
| 3 | Extract Dataset | Unzip kedua dataset |
| 4 | Organisasi Data | Mapping ke 3 kategori (Organik/Anorganik/Residu) |
| 5 | Preprocessing | Validasi gambar, buang corrupt |
| 6 | Balancing | Undersampling ke jumlah kelas terkecil |
| 7 | Resize & Split | 224×224, split 70/15/15 (train/val/test) |
| 8 | Augmentasi | Rotation, flip, zoom, brightness (train only) |
| 9 | Model | MobileNetV2 Transfer Learning |
| 10 | Training Fase 1 | Feature extraction (base frozen), 15 epochs |
| 11 | Training Fase 2 | Fine-tuning (30 layer terakhir), 15 epochs |
| 12 | Evaluasi | Classification report, confusion matrix |
| 13 | Save Model | `.keras` + `.tflite` + `class_info.json` |
| 14 | Fungsi Prediksi | `predict_waste()` untuk gambar baru |

**Format file:** Percent (`# %%`) cell separator — kompatibel Jupyter, VS Code, Colab.

**Status:** ✅ File berhasil dibuat.

---

### Prompt 2 (10:26) — Request Session Log
**User Request:**
- Buat file `SESSION_LOG.md` untuk mencatat semua percakapan.
- Harus selalu di-update setiap prompt.

**Apa yang dilakukan:**
- Membuat file `SESSION_LOG.md` ini.

**Status:** ✅ File berhasil dibuat.

---

### Prompt 3 (10:27) — Request Konversi ke Notebook
**User Request:**
- Jadikan file `.py` menjadi notebook (`.ipynb`).

**Apa yang dilakukan:**
1. Install `jupytext` via pip.
2. Jalankan `jupytext --to notebook waste_classification.py`.
3. Berhasil menghasilkan file `waste_classification.ipynb`.

**Status:** ✅ Notebook berhasil dibuat.

---

## Sesi 2 — 2026-08-24 (01:33 - ...)

### Prompt 4 (01:33) — Melanjutkan Proyek + Aturan Log
**User Request:**
- Lanjutkan proyek.
- Setiap prompt **wajib** update `SESSION_LOG.md`.
- Catat aturan ini di log.

**Apa yang dilakukan:**
1. Menambahkan section **⚠️ Aturan Proyek** di `SESSION_LOG.md`.
2. Mencatat prompt ini sebagai Sesi 2 Prompt 4.

**Status:** ✅ Selesai. Menunggu instruksi lanjutan dari user.

---

### Prompt 5 (01:35) — Perbaikan Model (Akurasi Rendah 78%)
**User Request:**
- Hasil model kurang bagus (accuracy 78%, f1-score ~0.77)
- Penyebab utama: data sangat imbalance (Organik: 10381, Anorganik: 662, Residu: 647)
- Undersampling ke 647 membuang 93% data Organik
- Perbaiki `.py` dan buat ulang `.ipynb`

**Analisis Masalah:**
- Total data setelah undersample hanya ~1941 gambar → terlalu sedikit
- Model MobileNetV2 kurang optimal untuk dataset kecil

**Perbaikan yang Dilakukan:**
1. **Data Strategy**: Hybrid balancing — Organik downsample ke ~2000, Anorganik & Residu tetap semua, gunakan `class_weight` untuk handle imbalance sisanya
2. **Model**: Ganti ke **EfficientNetB0** (lebih akurat dari MobileNetV2)
3. **Classification Head**: Lebih robust — tambah BatchNormalization, Dense lebih besar (512→256→3)
4. **Label Smoothing**: 0.1 untuk mengurangi overconfidence
5. **Augmentasi**: Jauh lebih agresif (rotation 40°, vertical flip, brightness range lebih lebar, channel shift)
6. **Training**: Lebih lama (20 + 30 epoch), CosineDecay LR, patience lebih tinggi
7. **Evaluasi**: Tambah Test-Time Augmentation (TTA) dan per-class accuracy

**Status:** ✅ Selesai. File `.py` dan `.ipynb` berhasil dibuat ulang.

---

### Prompt 6 (01:44) — Notebook Belum Ada, Ulangi Konversi
**User Request:**
- Notebook `.ipynb` belum ada/tidak ditemukan. Ulangi pembuatan.

**Apa yang dilakukan:**
1. Cek file — ditemukan `waste_classification1.ipynb` (nama bergeser, bukan `waste_classification.ipynb`).
2. Konversi ulang dengan `jupytext --to notebook -o waste_classification.ipynb`.
3. File `waste_classification.ipynb` berhasil dibuat.

**Status:** ✅ Selesai.

---

### Prompt 7 (01:45) — Skip Download Jika Data Sudah Ada
**User Request:**
- Data sudah ter-download, jangan download ulang biar tidak lama.

**Apa yang dilakukan:**
1. Cek folder `datasets/` — data sudah lengkap (zip + folder extract).
2. Perbaiki logika download: sekarang cek **folder extract** dulu, bukan hanya zip.
   - Jika folder extract sudah ada & berisi data → skip download & extract.
   - Jika folder extract kosong tapi zip ada → skip download, lanjut extract.
   - Jika keduanya tidak ada → baru download.
3. Konversi ulang `.py` → `.ipynb`.

**Status:** ✅ Selesai.

---

### Prompt 8 (01:48) — Perbarui .ipynb
**User Request:**
- Perbarui notebook `.ipynb` agar sinkron dengan `.py` terbaru.

**Apa yang dilakukan:**
- Konversi ulang `waste_classification.py` → `waste_classification.ipynb` via `jupytext`.

**Status:** ✅ Selesai.

---

### Prompt 9 (01:50) — Hapus & Buat Ulang Notebook
**User Request:**
- Hapus notebook lama, buat ulang dari `.py` terbaru.

**Apa yang dilakukan:**
1. Hapus `waste_classification.ipynb` dan `waste_classification1.ipynb`.
2. Buat ulang `waste_classification.ipynb` dari `waste_classification.py` via `jupytext`.

**Status:** ✅ Selesai.

---

### Prompt 10 (01:54) — Fix Error EfficientNetB0 + Pertanyaan Data Jomplang
**User Request:**
- Error `ValueError: Shape mismatch` saat build EfficientNetB0 (input_tensor issue di Keras 3).
- Tanya apakah data memang dibuat jomplang.

**Apa yang dilakukan:**
1. Fix: ganti `input_tensor=Input(shape=...)` → `input_shape=(224,224,3)` (kompatibel Keras 3).
2. Hapus import `Input` yang tidak dipakai lagi.
3. Jelaskan strategi hybrid balancing: data sengaja tidak 100% rata, sisa imbalance ditangani `class_weight`.
4. Regenerate `.ipynb`.

**Status:** ✅ Selesai.

---

### Prompt 11 (01:58) — Error Masih Muncul, Ganti ke MobileNetV2
**User Request:**
- Error `ValueError: Shape mismatch` masih muncul di EfficientNetB0 (bug Keras + Python 3.13).

**Apa yang dilakukan:**
1. Ganti EfficientNetB0 → **MobileNetV2** (sudah terbukti jalan di v1).
2. Semua improvement lainnya tetap dipertahankan (hybrid balancing, MixUp, class weight, label smoothing, TTA, augmentasi agresif, BatchNorm head).
3. Update semua referensi di kode (import, class_info, ringkasan).
4. Hapus & buat ulang `.ipynb`.

**Status:** ✅ Selesai.

---

### Prompt 12 (02:02) — Fix class_weight Error dengan Generator
**User Request:**
- Error `class_weight is not supported for Python generator inputs`.

**Apa yang dilakukan:**
1. `class_weight` tidak bisa dipakai di `model.fit()` kalau pakai Python generator.
2. Fix: integrasikan class weights langsung ke dalam `mixup_generator()` sebagai `sample_weight` (yield 3 elemen: X, y, weights).
3. Hapus `class_weight=class_weights_dict` dari kedua `model.fit()`.
4. Regenerate `.ipynb`.

**Status:** ✅ Selesai.

---

### Prompt 13 (02:26) — Fix TypeError: CosineDecay vs ReduceLROnPlateau Konflik
**User Request:**
- Error `TypeError: This optimizer was created with a LearningRateSchedule object` saat epoch 5.
- `ReduceLROnPlateau` mencoba set LR, tapi `CosineDecay` membuat LR read-only.

**Apa yang dilakukan:**
1. Hapus `ReduceLROnPlateau` dari `callbacks_phase1` (line 599-604).
2. Hapus `ReduceLROnPlateau` dari `callbacks_phase2` (line 674-679).
3. Hapus import `ReduceLROnPlateau` (line 45).
4. `CosineDecay` sudah cukup mengatur penurunan LR secara smooth — `ReduceLROnPlateau` tidak diperlukan.
5. Regenerate `.ipynb`.

**Status:** ✅ Selesai.

---

### Prompt 14 (06:53) — Buat Klasifikasi Biner (Non-Recyclable vs Recyclable)
**User Request:**
- Buat file `.py` dan `.ipynb` baru untuk klasifikasi **biner** antara `Non-Recyclable` dan `Recyclable`.
- Dataset dari folder lokal yang sudah ada:
  - `datasets/waste_raw/Non-Recyclable/Non-Recyclable/` (647 gambar: ceramic, diapers, plastic bags, sanitary napkin, styrofoam)
  - `datasets/waste_raw/Recyclable/Recyclable/` (665 gambar: cans, glass, paper, plastic bottles)

**Analisis Data:**
- Data sudah cukup balance (647 vs 665) → tidak perlu balancing khusus.
- Total: ~1312 gambar.

**Apa yang dilakukan:**
1. Buat `binary_classification.py` — pipeline lengkap:
   - Scan & validasi gambar dari folder lokal (tanpa download)
   - Visualisasi distribusi & sample
   - Split 70/15/15 (stratified)
   - Augmentasi agresif + MixUp
   - MobileNetV2 Transfer Learning (binary output: Dense(1, sigmoid))
   - `BinaryCrossentropy` (bukan `CategoricalCrossentropy`)
   - CosineDecay LR (tanpa ReduceLROnPlateau)
   - Fase 1: Feature Extraction (20 epoch) + Fase 2: Fine-Tuning (30 epoch)
   - TTA evaluation
   - Confusion matrix
   - Save `.keras`, `.tflite`, `class_info.json`
   - Fungsi `predict_binary()` untuk inferensi
2. Konversi ke `binary_classification.ipynb` via jupytext.

**Output files (setelah training):**
- `binary_model_final.keras`
- `binary_best_model_phase1.keras`
- `binary_best_model_phase2.keras`
- `binary_model.tflite`
- `binary_class_info.json`
- `binary_distribusi_data.png`
- `binary_sample_gambar.png`
- `binary_training_history.png`
- `binary_confusion_matrix.png`
- `binary_prediksi_sample.png`

**Status:** ✅ Selesai.

---

### Prompt 15 (07:00) — Buat Ulang Binary Classification (Lebih Simpel, Tanpa Freeze/Unfreeze)
**User Request:**
- Gunakan algoritma paling sesuai untuk kasus ini.
- Tidak perlu freeze/unfreeze (2-phase).
- Buat ulang `.py` dan `.ipynb`.

**Analisis & Keputusan:**
- Dataset kecil (~1300 gambar), balanced → **Transfer Learning single-phase** paling cocok.
- Base model MobileNetV2 di-freeze (feature extractor), hanya classification head yang dilatih.
- Tanpa 2-phase (freeze→unfreeze) karena dataset terlalu kecil — unfreeze berisiko overfitting.
- LR fixed + `ReduceLROnPlateau` (bukan CosineDecay — lebih simpel, tidak ada konflik).
- Hapus MixUp (over-engineering untuk dataset kecil balanced).
- Label smoothing dikurangi (0.05).

**Perbedaan vs Versi Sebelumnya:**

| Aspek | Sebelumnya | Sekarang |
|-------|-----------|----------|
| Training | 2 fase (freeze → unfreeze) | Single phase |
| LR Schedule | CosineDecay | Fixed + ReduceLROnPlateau |
| Head | GAP→BN→256→D(0.4)→BN→128→D(0.3)→1 | GAP→BN→128→D(0.5)→1 |
| MixUp | Ya (alpha=0.2) | Tidak |
| Label Smoothing | 0.1 | 0.05 |
| EarlyStopping | patience=7 | patience=10 |
| Max Epochs | 20+30=50 | 50 (single) |

**Status:** ✅ Selesai.

---

## File yang Sudah Dibuat

| File | Deskripsi | Status |
|------|-----------|--------|
| `waste_classification.py` | Script klasifikasi sampah **v2 (Improved)** — MobileNetV2, Hybrid Balancing, MixUp, TTA | ✅ Selesai |
| `waste_classification.ipynb` | Notebook (konversi dari .py v2) | ✅ Selesai |
| `binary_classification.py` | Script klasifikasi **biner** — Non-Recyclable vs Recyclable | ✅ Selesai |
| `binary_classification.ipynb` | Notebook (konversi dari .py) | ✅ Selesai |
| `SESSION_LOG.md` | Log percakapan ini | ✅ Aktif |

## Dataset yang Digunakan

### Proyek 1 — Klasifikasi 3 Kelas (Organik/Anorganik/Residu)

| Kategori | Sumber Dataset | URL |
|----------|---------------|-----|
| Organik | arthurwaruwu/datasetcapstonefixx | [Link](https://www.kaggle.com/datasets/arthurwaruwu/datasetcapstonefixx) |
| Anorganik | phenomsg/waste-classification → `Recyclable/` | [Link](https://www.kaggle.com/datasets/phenomsg/waste-classification) |
| Residu | phenomsg/waste-classification → `Non-Recyclable/` | [Link](https://www.kaggle.com/datasets/phenomsg/waste-classification) |

### Proyek 2 — Klasifikasi Biner (Non-Recyclable vs Recyclable)

| Kategori | Folder Lokal | Jumlah |
|----------|-------------|--------|
| Non-Recyclable | `datasets/waste_raw/Non-Recyclable/Non-Recyclable/` | 647 |
| Recyclable | `datasets/waste_raw/Recyclable/Recyclable/` | 665 |

---

*Log ini akan di-update setiap kali ada prompt baru.*

