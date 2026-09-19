# %% [markdown]
# # Klasifikasi Sampah: Organik, Anorganik, Residu (Improved v2)
#
# Notebook ini melakukan klasifikasi gambar sampah ke dalam 3 kategori:
# - **Organik**: Sampah yang dapat terurai secara alami (dari dataset arthurwaruwu/datasetcapstonefixx)
# - **Anorganik**: Sampah yang dapat didaur ulang / Recyclable (dari dataset phenomsg/waste-classification)
# - **Residu**: Sampah yang tidak dapat didaur ulang / Non-Recyclable (dari dataset phenomsg/waste-classification)
#
# **Perbaikan dari v1:**
# - Hybrid Balancing (bukan pure undersampling) → lebih banyak data training
# - EfficientNetB0 (menggantikan MobileNetV2) → akurasi lebih tinggi
# - MixUp Augmentation → regularisasi lebih baik
# - Label Smoothing → mengurangi overconfidence
# - Class Weights → menangani sisa imbalance
# - Test-Time Augmentation (TTA) → evaluasi lebih robust
# - Augmentasi lebih agresif → generalisasi lebih baik

# %% [markdown]
# ## 1. Install & Import Libraries

# %%
# Install dependencies (uncomment jika belum terinstall)
# !pip install kaggle tensorflow matplotlib scikit-learn seaborn Pillow pandas

# %%
import os
import shutil
import zipfile
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# ## 2. Download Dataset dari Kaggle
#
# Pastikan file `kaggle.json` sudah dikonfigurasi:
# - Letakkan di `~/.kaggle/kaggle.json` (Linux/Mac) atau `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
# - Dapatkan dari https://www.kaggle.com/settings -> "Create New Token"

# %%
# Konfigurasi path
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "datasets"
ORGANIK_ZIP = DATA_DIR / "organik.zip"
WASTE_CLASS_ZIP = DATA_DIR / "waste_classification.zip"
ORGANIK_EXTRACT = DATA_DIR / "organik_raw"
WASTE_EXTRACT = DATA_DIR / "waste_raw"

# Buat direktori
DATA_DIR.mkdir(parents=True, exist_ok=True)
ORGANIK_EXTRACT.mkdir(parents=True, exist_ok=True)
WASTE_EXTRACT.mkdir(parents=True, exist_ok=True)

# %%
# Download dataset Organik (arthurwaruwu/datasetcapstonefixx)
# Skip jika folder extract sudah ada ATAU zip sudah ada
if ORGANIK_EXTRACT.exists() and any(ORGANIK_EXTRACT.iterdir()):
    print("Dataset Organik sudah ter-extract, skip download & extract.")
elif not ORGANIK_ZIP.exists():
    print("Downloading dataset Organik...")
    os.system(f'kaggle datasets download -d arthurwaruwu/datasetcapstonefixx -p "{DATA_DIR}" --force')
    # Rename file hasil download
    downloaded = list(DATA_DIR.glob("datasetcapstonefixx*"))
    if downloaded:
        downloaded[0].rename(ORGANIK_ZIP)
    print("Download Organik selesai!")
else:
    print("ZIP Organik sudah ada, skip download.")

# %%
# Download dataset Waste Classification (phenomsg/waste-classification)
# Skip jika folder extract sudah ada ATAU zip sudah ada
if WASTE_EXTRACT.exists() and any(WASTE_EXTRACT.iterdir()):
    print("Dataset Waste Classification sudah ter-extract, skip download & extract.")
elif not WASTE_CLASS_ZIP.exists():
    print("Downloading dataset Waste Classification...")
    os.system(f'kaggle datasets download -d phenomsg/waste-classification -p "{DATA_DIR}" --force')
    # Rename file hasil download
    downloaded = list(DATA_DIR.glob("waste-classification*"))
    if downloaded:
        downloaded[0].rename(WASTE_CLASS_ZIP)
    print("Download Waste Classification selesai!")
else:
    print("ZIP Waste Classification sudah ada, skip download.")

# %% [markdown]
# ## 3. Extract Dataset

# %%
def extract_zip(zip_path, extract_to):
    """Extract zip file ke folder tujuan."""
    if not any(extract_to.iterdir()):
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction selesai: {extract_to}")
    else:
        print(f"Folder {extract_to} sudah berisi data, skip extraction.")

extract_zip(ORGANIK_ZIP, ORGANIK_EXTRACT)
extract_zip(WASTE_CLASS_ZIP, WASTE_EXTRACT)

# %%
# Lihat struktur folder hasil extract
def show_tree(path, prefix="", max_depth=3, current_depth=0):
    """Tampilkan struktur folder."""
    if current_depth >= max_depth:
        return
    entries = sorted(Path(path).iterdir())
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]

    if files:
        print(f"{prefix}[{len(files)} files]")
    for d in dirs:
        print(f"{prefix}📁 {d.name}/")
        show_tree(d, prefix + "  ", max_depth, current_depth + 1)

print("=== Struktur Dataset Organik ===")
show_tree(ORGANIK_EXTRACT)
print("\n=== Struktur Dataset Waste Classification ===")
show_tree(WASTE_EXTRACT)

# %% [markdown]
# ## 4. Pengumpulan & Validasi Gambar
#
# Mapping:
# - **Organik** ← Semua gambar dari dataset arthurwaruwu/datasetcapstonefixx
# - **Anorganik** ← Folder `Recyclable/` dari phenomsg/waste-classification
# - **Residu** ← Folder `Non-Recyclable/` dari phenomsg/waste-classification

# %%
# Cari folder sumber secara otomatis
def find_image_dirs(base_path, target_names):
    """Cari folder berdasarkan nama di dalam base_path."""
    found = {}
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            d_lower = d.lower().replace("-", "").replace("_", "").replace(" ", "")
            for target in target_names:
                t_lower = target.lower().replace("-", "").replace("_", "").replace(" ", "")
                if d_lower == t_lower:
                    found[target] = Path(root) / d
    return found

# Cari folder Recyclable dan Non-Recyclable
waste_dirs = find_image_dirs(WASTE_EXTRACT, ["Recyclable", "Non-Recyclable"])
print("Folder ditemukan di waste dataset:")
for k, v in waste_dirs.items():
    print(f"  {k}: {v}")

# %%
# Fungsi untuk mengumpulkan path gambar dari folder (termasuk subfolder)
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

def collect_image_paths(folder_path):
    """Kumpulkan semua path gambar valid dari folder dan subfoldernya."""
    paths = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if Path(f).suffix.lower() in VALID_EXTENSIONS:
                paths.append(str(Path(root) / f))
    return paths

def validate_image(img_path):
    """Validasi apakah file gambar bisa dibuka dan tidak corrupt."""
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def validate_images_batch(image_list, label=""):
    """Validasi batch gambar dan kembalikan hanya yang valid."""
    valid = []
    invalid_count = 0
    for i, img_path in enumerate(image_list):
        if validate_image(img_path):
            valid.append(img_path)
        else:
            invalid_count += 1
        if (i + 1) % 1000 == 0:
            print(f"  [{label}] Validasi {i+1}/{len(image_list)}...")
    if invalid_count > 0:
        print(f"  [{label}] {invalid_count} gambar corrupt/invalid dibuang.")
    else:
        print(f"  [{label}] Semua {len(valid)} gambar valid.")
    return valid

# %%
# Kumpulkan gambar per kategori
organik_images = collect_image_paths(ORGANIK_EXTRACT)
print(f"Jumlah gambar Organik (raw): {len(organik_images)}")

anorganik_images = []
if "Recyclable" in waste_dirs:
    anorganik_images = collect_image_paths(waste_dirs["Recyclable"])
print(f"Jumlah gambar Anorganik (raw): {len(anorganik_images)}")

residu_images = []
if "Non-Recyclable" in waste_dirs:
    residu_images = collect_image_paths(waste_dirs["Non-Recyclable"])
print(f"Jumlah gambar Residu (raw): {len(residu_images)}")

# %%
# Validasi gambar — buang yang corrupt
print("\nValidasi gambar...")
organik_valid = validate_images_batch(organik_images, "Organik")
anorganik_valid = validate_images_batch(anorganik_images, "Anorganik")
residu_valid = validate_images_batch(residu_images, "Residu")

print(f"\nSetelah validasi:")
print(f"  Organik:   {len(organik_valid)}")
print(f"  Anorganik: {len(anorganik_valid)}")
print(f"  Residu:    {len(residu_valid)}")

# %% [markdown]
# ## 5. Visualisasi Distribusi Awal

# %%
categories = ['Organik', 'Anorganik', 'Residu']
counts_raw = [len(organik_valid), len(anorganik_valid), len(residu_valid)]

plt.figure(figsize=(8, 5))
bars = plt.bar(categories, counts_raw, color=['#2ecc71', '#3498db', '#e74c3c'])
for bar, count in zip(bars, counts_raw):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
             f'{count}', ha='center', va='bottom', fontweight='bold')
plt.title('Distribusi Data SEBELUM Balancing', fontsize=14)
plt.ylabel('Jumlah Gambar')
plt.tight_layout()
plt.savefig('distribusi_sebelum_balancing.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Hybrid Balancing
#
# **Strategi baru (v2):**
# - Organik di-downsample ke ~2000 (bukan ke 647 seperti v1)
# - Anorganik & Residu **tetap semua** (662 & 647)
# - Sisa imbalance ditangani oleh `class_weight` saat training
# - Total data: ~3300 (vs ~1941 di v1) → 70% lebih banyak data training

# %%
# Hybrid Balancing: downsample Organik ke 2000, keep all Anorganik & Residu
TARGET_ORGANIK = 2000

random.seed(SEED)
if len(organik_valid) > TARGET_ORGANIK:
    organik_balanced = random.sample(organik_valid, TARGET_ORGANIK)
    print(f"Organik di-downsample dari {len(organik_valid)} → {TARGET_ORGANIK}")
else:
    organik_balanced = organik_valid
    print(f"Organik tetap {len(organik_valid)} (sudah <= {TARGET_ORGANIK})")

anorganik_balanced = anorganik_valid  # Keep ALL
residu_balanced = residu_valid        # Keep ALL

print(f"\nSetelah Hybrid Balancing:")
print(f"  Organik:   {len(organik_balanced)}")
print(f"  Anorganik: {len(anorganik_balanced)}")
print(f"  Residu:    {len(residu_balanced)}")
total = len(organik_balanced) + len(anorganik_balanced) + len(residu_balanced)
print(f"  TOTAL:     {total}")

# %%
# Visualisasi distribusi setelah balancing
counts_balanced = [len(organik_balanced), len(anorganik_balanced), len(residu_balanced)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(categories, counts_raw, color=['#2ecc71', '#3498db', '#e74c3c'])
for i, count in enumerate(counts_raw):
    ax1.text(i, count + 50, f'{count}', ha='center', va='bottom', fontweight='bold')
ax1.set_title('SEBELUM Balancing', fontsize=14)
ax1.set_ylabel('Jumlah Gambar')

ax2.bar(categories, counts_balanced, color=['#2ecc71', '#3498db', '#e74c3c'])
for i, count in enumerate(counts_balanced):
    ax2.text(i, count + 20, f'{count}', ha='center', va='bottom', fontweight='bold')
ax2.set_title('SETELAH Hybrid Balancing', fontsize=14)
ax2.set_ylabel('Jumlah Gambar')

plt.suptitle('Perbandingan Distribusi Data', fontsize=16)
plt.tight_layout()
plt.savefig('distribusi_setelah_balancing.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Membuat DataFrame & Split Data (70/15/15)

# %%
# Buat DataFrame dari semua gambar yang sudah di-balance
df_all = pd.DataFrame({
    'filename': organik_balanced + anorganik_balanced + residu_balanced,
    'class': (['Organik'] * len(organik_balanced) +
              ['Anorganik'] * len(anorganik_balanced) +
              ['Residu'] * len(residu_balanced))
})

# Shuffle
df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"Total gambar: {len(df_all)}")
print(df_all['class'].value_counts())

# %%
# Split: Train 70%, Val 15%, Test 15%
train_df, temp_df = train_test_split(
    df_all, test_size=0.30, stratify=df_all['class'], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['class'], random_state=SEED
)

print(f"\nJumlah Train: {len(train_df)}")
print(train_df['class'].value_counts())
print(f"\nJumlah Validation: {len(val_df)}")
print(val_df['class'].value_counts())
print(f"\nJumlah Test: {len(test_df)}")
print(test_df['class'].value_counts())

# %% [markdown]
# ## 8. Visualisasi Sample Gambar

# %%
def show_sample_images(dataframe, n_samples=5):
    """Tampilkan sample gambar dari setiap kategori."""
    fig, axes = plt.subplots(3, n_samples, figsize=(15, 9))

    for row, category in enumerate(categories):
        cat_images = dataframe[dataframe['class'] == category]['filename'].values[:n_samples]

        for col, img_path in enumerate(cat_images):
            try:
                img = Image.open(img_path).convert('RGB')
                axes[row, col].imshow(img)
            except Exception:
                axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center')
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(category, fontsize=14, fontweight='bold')

    plt.suptitle('Sample Gambar per Kategori', fontsize=16)
    plt.tight_layout()
    plt.savefig('sample_gambar.png', dpi=150, bbox_inches='tight')
    plt.show()

show_sample_images(train_df)

# %% [markdown]
# ## 9. Data Generator dengan Augmentasi Agresif

# %%
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Augmentasi AGRESIF untuk training (v2 — jauh lebih kuat dari v1)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,          # v1: 30
    width_shift_range=0.3,      # v1: 0.2
    height_shift_range=0.3,     # v1: 0.2
    shear_range=0.3,            # v1: 0.2
    zoom_range=0.3,             # v1: 0.2
    horizontal_flip=True,
    vertical_flip=True,         # BARU: sampah bisa posisi apa aja
    brightness_range=[0.6, 1.4],  # v1: [0.8, 1.2]
    channel_shift_range=30.0,   # BARU: variasi warna
    fill_mode='nearest'
)

# Validasi & Test hanya rescale (tanpa augmentasi)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# %%
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=categories,
    shuffle=True,
    seed=SEED
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=categories,
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=categories,
    shuffle=False
)

# %%
# Tampilkan class mapping
print("Class indices:", train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())
print("Class names:", class_names)
print(f"\nTrain samples: {train_generator.samples}")
print(f"Val samples:   {val_generator.samples}")
print(f"Test samples:  {test_generator.samples}")

# %% [markdown]
# ## 10. Hitung Class Weights
#
# Karena data masih imbalance (Organik ~2000 vs Anorganik ~662 vs Residu ~647),
# kita gunakan `class_weight` agar model tidak bias ke kelas mayoritas.

# %%
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.array(categories),
    y=train_df['class'].values
)

class_weights_dict = {}
for cls_name, weight in zip(categories, class_weights_array):
    idx = train_generator.class_indices[cls_name]
    class_weights_dict[idx] = weight

print("Class Weights:")
for cls_name, weight in zip(categories, class_weights_array):
    print(f"  {cls_name}: {weight:.4f}")

# %% [markdown]
# ## 11. MixUp Generator
#
# MixUp: mencampurkan 2 gambar dan label-nya secara linear.
# Ini membantu model belajar decision boundary yang lebih smooth dan mengurangi overfitting.

# %%
def mixup_generator(generator, alpha=0.2, class_weights=None):
    """Wrapper generator yang mengaplikasikan MixUp augmentation.
    
    class_weights: dict {class_idx: weight} — diintegrasikan sebagai sample_weight
    karena Keras tidak mendukung class_weight dengan Python generator.
    """
    # Buat array weights berdasarkan jumlah kelas
    if class_weights is not None:
        num_classes = len(class_weights)
        weight_array = np.array([class_weights[i] for i in range(num_classes)])
    
    while True:
        X1, y1 = next(generator)
        X2, y2 = next(generator)

        # Samakan ukuran batch
        batch_size = min(X1.shape[0], X2.shape[0])
        if batch_size == 0:
            continue

        X1, y1 = X1[:batch_size], y1[:batch_size]
        X2, y2 = X2[:batch_size], y2[:batch_size]

        # Lambda dari distribusi Beta
        lam = np.random.beta(alpha, alpha, batch_size)
        X_lam = lam.reshape(batch_size, 1, 1, 1)
        y_lam = lam.reshape(batch_size, 1)

        # Campurkan gambar dan label
        X_mix = X1 * X_lam + X2 * (1 - X_lam)
        y_mix = y1 * y_lam + y2 * (1 - y_lam)

        if class_weights is not None:
            # Hitung sample_weight: weighted sum berdasarkan proporsi label campuran
            sample_weights = np.dot(y_mix, weight_array)
            yield X_mix, y_mix, sample_weights
        else:
            yield X_mix, y_mix

# Bungkus train generator dengan MixUp + class weights
train_mixup_gen = mixup_generator(train_generator, alpha=0.2, class_weights=class_weights_dict)

# Hitung steps
steps_per_epoch = max(1, len(train_df) // BATCH_SIZE)
validation_steps = max(1, len(val_df) // BATCH_SIZE)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# %% [markdown]
# ## 12. Build Model (Transfer Learning - MobileNetV2)
#
# **Perbaikan dari v1:**
# - Classification head lebih kuat: BatchNormalization di setiap Dense layer
# - Dense layer lebih besar (512→256 vs 256→128)
# - Label smoothing: 0.1 → mengurangi overconfidence pada prediksi

# %%
def build_model(num_classes=3, img_size=IMG_SIZE):
    """Build model dengan MobileNetV2 sebagai base (transfer learning)."""
    # Load MobileNetV2 pretrained pada ImageNet, tanpa top layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )

    # Freeze base model (awal training hanya custom layers yang di-train)
    base_model.trainable = False

    # Classification head yang diperkuat (v2)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model

model, base_model = build_model(num_classes=3)
model.summary()

# %% [markdown]
# ## 13. Training — Fase 1: Feature Extraction (Base Frozen)
#
# Hanya melatih classification head (base model tetap frozen).
# Menggunakan CosineDecay learning rate schedule.

# %%
EPOCHS_PHASE1 = 20

# CosineDecay: learning rate mulai 1e-3 lalu turun halus ke ~0
lr_schedule_p1 = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=EPOCHS_PHASE1 * steps_per_epoch
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule_p1),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Callbacks
callbacks_phase1 = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model_phase1.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# %%
print("=" * 60)
print("FASE 1: Feature Extraction (Base Model Frozen)")
print(f"  Epochs: {EPOCHS_PHASE1}")
print(f"  LR: CosineDecay dari 1e-3")
print(f"  Label Smoothing: 0.1")
print(f"  MixUp Alpha: 0.2")
print("=" * 60)

history_phase1 = model.fit(
    train_mixup_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_phase1,
    verbose=1
)

# %% [markdown]
# ## 14. Training — Fase 2: Fine-Tuning (Unfreeze 50 Layer Terakhir)
#
# Unfreeze 50 layer terakhir dari EfficientNetB0 dan training ulang
# dengan learning rate sangat kecil (1e-5) agar tidak merusak fitur yang sudah dipelajari.

# %%
# Unfreeze 50 layer terakhir dari base model
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Hitung jumlah trainable vs non-trainable params
trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
print(f"Trainable params:     {trainable_count:,}")
print(f"Non-trainable params: {non_trainable_count:,}")

# %%
EPOCHS_PHASE2 = 30

# CosineDecay: learning rate mulai 1e-5 (sangat kecil untuk fine-tuning)
lr_schedule_p2 = CosineDecay(
    initial_learning_rate=1e-5,
    decay_steps=EPOCHS_PHASE2 * steps_per_epoch
)

# Compile ulang dengan learning rate lebih kecil
model.compile(
    optimizer=Adam(learning_rate=lr_schedule_p2),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model_phase2.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# %%
print("=" * 60)
print("FASE 2: Fine-Tuning (50 Layer Terakhir Base Model)")
print(f"  Epochs: {EPOCHS_PHASE2}")
print(f"  LR: CosineDecay dari 1e-5")
print(f"  Trainable params: {trainable_count:,}")
print("=" * 60)

# Buat MixUp generator baru karena generator sebelumnya mungkin sudah terpakai
train_generator_p2 = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=categories,
    shuffle=True,
    seed=SEED
)
train_mixup_gen_p2 = mixup_generator(train_generator_p2, alpha=0.2, class_weights=class_weights_dict)

# Buat val generator baru juga
val_generator_p2 = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=categories,
    shuffle=False
)

history_phase2 = model.fit(
    train_mixup_gen_p2,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator_p2,
    validation_steps=validation_steps,
    callbacks=callbacks_phase2,
    verbose=1
)

# %% [markdown]
# ## 15. Visualisasi Training History

# %%
def plot_training_history(history1, history2, save_path='training_history.png'):
    """Plot gabungan training history dari 2 fase."""
    # Gabungkan history
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    epochs_range = range(1, len(acc) + 1)
    phase1_end = len(history1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Fine-tuning Start')
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax2.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Fine-tuning Start')
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Training History (Fase 1: Feature Extraction → Fase 2: Fine-Tuning)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

plot_training_history(history_phase1, history_phase2)

# %% [markdown]
# ## 16. Evaluasi pada Test Set

# %%
print("=" * 60)
print("EVALUASI PADA TEST SET")
print("=" * 60)

# Buat test generator baru untuk evaluasi
test_gen_eval = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=categories,
    shuffle=False
)

# Evaluasi keseluruhan
test_loss, test_accuracy = model.evaluate(test_gen_eval, verbose=1)
print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# %%
# Prediksi untuk classification report & confusion matrix
test_gen_eval.reset()
y_pred_proba = model.predict(test_gen_eval, verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = test_gen_eval.classes

# Classification Report
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# Per-class accuracy
print("=== Akurasi Per Kelas ===")
cm = confusion_matrix(y_true, y_pred)
for i, name in enumerate(class_names):
    cls_total = cm[i].sum()
    cls_correct = cm[i, i]
    cls_acc = cls_correct / cls_total if cls_total > 0 else 0
    print(f"  {name}: {cls_acc*100:.2f}% ({cls_correct}/{cls_total})")

# %%
# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Normalized Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 17. Evaluasi dengan Test-Time Augmentation (TTA)
#
# TTA: prediksi gambar test sebanyak beberapa kali dengan augmentasi ringan,
# lalu rata-ratakan hasilnya. Biasanya meningkatkan akurasi 1-3%.

# %%
print("=" * 60)
print("EVALUASI DENGAN TEST-TIME AUGMENTATION (TTA)")
print("=" * 60)

# Augmentasi ringan untuk TTA
tta_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

NUM_TTA = 5

def evaluate_with_tta(model, dataframe, num_tta=NUM_TTA):
    """Evaluasi model dengan Test-Time Augmentation."""
    all_preds = []

    # Prediksi 1: gambar asli (tanpa augmentasi)
    print("TTA 0/{num_tta}: Prediksi gambar asli...")
    gen_original = val_test_datagen.flow_from_dataframe(
        dataframe=dataframe, x_col='filename', y_col='class',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', classes=categories, shuffle=False
    )
    preds = model.predict(gen_original, verbose=1)
    all_preds.append(preds)

    # Prediksi 2-6: gambar ter-augmentasi
    for i in range(num_tta):
        print(f"TTA {i+1}/{num_tta}: Prediksi gambar ter-augmentasi...")
        gen_tta = tta_datagen.flow_from_dataframe(
            dataframe=dataframe, x_col='filename', y_col='class',
            target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', classes=categories, shuffle=False
        )
        preds = model.predict(gen_tta, verbose=1)
        all_preds.append(preds)

    # Rata-rata dari semua prediksi
    mean_preds = np.mean(all_preds, axis=0)
    return np.argmax(mean_preds, axis=1)

y_pred_tta = evaluate_with_tta(model, test_df)

# %%
# Hasil TTA
print("\n=== Classification Report (dengan TTA) ===")
print(classification_report(y_true, y_pred_tta, target_names=class_names))

tta_accuracy = accuracy_score(y_true, y_pred_tta)
print(f"Overall TTA Accuracy: {tta_accuracy*100:.2f}%")
print(f"Peningkatan dari standar: {(tta_accuracy - test_accuracy)*100:+.2f}%")

# Per-class accuracy dengan TTA
print("\n=== Akurasi Per Kelas (dengan TTA) ===")
cm_tta = confusion_matrix(y_true, y_pred_tta)
for i, name in enumerate(class_names):
    cls_total = cm_tta[i].sum()
    cls_correct = cm_tta[i, i]
    cls_acc = cls_correct / cls_total if cls_total > 0 else 0
    print(f"  {name}: {cls_acc*100:.2f}% ({cls_correct}/{cls_total})")

# %%
# Confusion Matrix TTA
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tta, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (dengan TTA)', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_tta.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 18. Visualisasi Prediksi

# %%
def show_predictions(dataframe, model, class_names, n_samples=12):
    """Tampilkan gambar dengan prediksi dan label asli."""
    sample = dataframe.sample(n=min(n_samples, len(dataframe)), random_state=SEED)

    cols = 4
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for i, (_, row) in enumerate(sample.iterrows()):
        ax = axes[i]
        try:
            img = load_img(row['filename'], target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_batch, verbose=0)
            pred_idx = np.argmax(predictions[0])
            pred_label = class_names[pred_idx]
            true_label = row['class']
            confidence = predictions[0][pred_idx] * 100

            ax.imshow(img)
            color = 'green' if pred_label == true_label else 'red'
            ax.set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)",
                         color=color, fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
        ax.axis('off')

    # Hide unused axes
    for i in range(len(sample), len(axes)):
        axes[i].axis('off')

    plt.suptitle('Prediksi Model pada Test Set', fontsize=14)
    plt.tight_layout()
    plt.savefig('prediksi_sample.png', dpi=150, bbox_inches='tight')
    plt.show()

show_predictions(test_df, model, class_names)

# %% [markdown]
# ## 19. Save Model Final

# %%
# Save model dalam format Keras
model.save('waste_classifier_organik_anorganik_residu.keras')
print("Model tersimpan: waste_classifier_organik_anorganik_residu.keras")

# Save juga dalam format TFLite (untuk deployment mobile)
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = 'waste_classifier.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model TFLite tersimpan: {tflite_path} ({len(tflite_model) / 1024 / 1024:.1f} MB)")
except Exception as e:
    print(f"Gagal mengonversi ke TFLite: {e}")

# Save class names
class_info = {
    'class_names': class_names,
    'class_indices': train_generator.class_indices,
    'img_size': list(IMG_SIZE),
    'model_architecture': 'MobileNetV2 (Transfer Learning v2)',
    'improvements': [
        'Hybrid Balancing (Organik downsample ke 2000)',
        'MobileNetV2 + Enhanced Head (BatchNorm, Dense 512→256)',
        'MixUp Augmentation (alpha=0.2)',
        'Label Smoothing (0.1)',
        'Class Weights (balanced)',
        'Augmentasi agresif (rotation 40, vertical flip, channel shift)',
        'CosineDecay LR schedule',
        'Test-Time Augmentation (TTA)'
    ]
}
with open('class_info.json', 'w') as f:
    json.dump(class_info, f, indent=2)
print("Class info tersimpan: class_info.json")

# %% [markdown]
# ## 20. Fungsi Prediksi untuk Gambar Baru

# %%
def predict_waste(image_path, model=model, class_names=class_names, img_size=IMG_SIZE):
    """Prediksi kategori sampah dari path gambar."""
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(img_size, Image.LANCZOS)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx] * 100

    # Tampilkan hasil
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(img)
    ax1.set_title(f"Prediksi: {class_names[pred_idx]}\n({confidence:.1f}%)",
                  fontsize=12, fontweight='bold')
    ax1.axis('off')

    colors = ['#2ecc71' if i == pred_idx else '#bdc3c7' for i in range(len(class_names))]
    bars = ax2.barh(class_names, predictions[0] * 100, color=colors)
    ax2.set_xlabel('Confidence (%)')
    ax2.set_title('Probabilitas per Kelas')
    ax2.set_xlim(0, 100)

    for bar, val in zip(bars, predictions[0] * 100):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center')

    plt.tight_layout()
    plt.show()

    return class_names[pred_idx], confidence

# Contoh penggunaan (uncomment dan ganti path):
# result, conf = predict_waste("path/ke/gambar/sampah.jpg")
# print(f"Hasil: {result} ({conf:.1f}%)")

# %% [markdown]
# ## Ringkasan Perbaikan (v1 → v2)
#
# | Aspek | v1 (78% accuracy) | v2 (improved) |
# |-------|-------------------|---------------|
# | **Data** | Undersample semua ke 647 (~1941 total) | Hybrid: Organik→2000, keep all lainnya (~3300) + class_weight |
# | **Base Model** | MobileNetV2 | MobileNetV2 + **Enhanced Head** |
# | **Head** | Dense 256→128→3 | Dense 512→256→3 + **BatchNormalization** |
# | **Loss** | CrossEntropy | CrossEntropy + Label Smoothing 0.1 |
# | **Augmentasi** | Rotation 30°, brightness 0.8-1.2 | Rotation 40°, brightness 0.6-1.4, vertical flip, channel shift |
# | **MixUp** | ❌ | ✅ (alpha=0.2) |
# | **Class Weight** | ❌ | ✅ (balanced) |
# | **Training** | 15+15 epoch, fixed LR | 20+30 epoch, CosineDecay LR |
# | **Fine-tune** | 30 layers | 50 layers |
# | **TTA** | ❌ | ✅ (5 augmented predictions) |
