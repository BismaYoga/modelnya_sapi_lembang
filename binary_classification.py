# %% [markdown]
# # Klasifikasi Biner: Non-Recyclable vs Recyclable
#
# Notebook ini melakukan klasifikasi gambar sampah ke dalam 2 kategori:
# - **Non-Recyclable**: Sampah yang tidak dapat didaur ulang (ceramic, diapers, plastic bags/wrappers, sanitary napkin, styrofoam)
# - **Recyclable**: Sampah yang dapat didaur ulang (cans, glass containers, paper products, plastic bottles)
#
# **Pendekatan:**
# - Transfer Learning — MobileNetV2 (pretrained ImageNet)
# - Base model di-freeze, hanya classification head yang dilatih (single-phase)
# - Cocok untuk dataset kecil (~1300 gambar) agar tidak overfitting
# - Augmentasi agresif + Test-Time Augmentation (TTA)

# %% [markdown]
# ## 1. Import Libraries

# %%
import os
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# ## 2. Konfigurasi

# %%
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50  # Single phase, EarlyStopping akan berhenti lebih awal
LEARNING_RATE = 1e-3

# Path dataset
NON_RECYCLABLE_DIR = r"C:\Users\Bisma\Downloads\Documents\tanin\datasets\waste_raw\Non-Recyclable\Non-Recyclable"
RECYCLABLE_DIR = r"C:\Users\Bisma\Downloads\Documents\tanin\datasets\waste_raw\Recyclable\Recyclable"

categories = ['Non-Recyclable', 'Recyclable']

print("Konfigurasi:")
print(f"  Image Size : {IMG_SIZE}")
print(f"  Batch Size : {BATCH_SIZE}")
print(f"  Max Epochs : {EPOCHS}")
print(f"  LR         : {LEARNING_RATE}")
print(f"  Categories : {categories}")

# %% [markdown]
# ## 3. Kumpulkan & Validasi Data

# %%
def validate_image(filepath):
    """Cek apakah file gambar valid dan tidak corrupt."""
    try:
        img = Image.open(filepath)
        img.verify()
        return True
    except Exception:
        return False

def collect_images(base_dir, label):
    """Kumpulkan semua path gambar dari folder dan subfolder."""
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    data = []
    for img_path in Path(base_dir).rglob('*'):
        if img_path.is_file() and img_path.suffix.lower() in valid_ext:
            data.append({'filename': str(img_path), 'class': label})
    return data

# Kumpulkan data
print("Mengumpulkan data gambar...")
all_data = []
all_data.extend(collect_images(NON_RECYCLABLE_DIR, 'Non-Recyclable'))
all_data.extend(collect_images(RECYCLABLE_DIR, 'Recyclable'))
df = pd.DataFrame(all_data)

print(f"Total gambar ditemukan: {len(df)}")
print(df['class'].value_counts())

# %%
# Validasi — buang gambar corrupt
print("\nMemvalidasi gambar...")
valid_mask = df['filename'].apply(validate_image)
corrupt_count = (~valid_mask).sum()
df = df[valid_mask].reset_index(drop=True)

print(f"Gambar corrupt: {corrupt_count}")
print(f"Gambar valid  : {len(df)}")
print(df['class'].value_counts())

# %% [markdown]
# ## 4. Visualisasi Distribusi Data

# %%
fig, ax = plt.subplots(figsize=(8, 5))
counts = df['class'].value_counts()
colors = ['#e74c3c', '#2ecc71']
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black')

for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=14)

ax.set_title('Distribusi Data per Kelas', fontsize=16, fontweight='bold')
ax.set_xlabel('Kelas', fontsize=12)
ax.set_ylabel('Jumlah Gambar', fontsize=12)
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig('binary_distribusi_data.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Sample Gambar

# %%
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for row, cat in enumerate(categories):
    samples = df[df['class'] == cat].sample(n=5, random_state=SEED)
    for col, (_, s) in enumerate(samples.iterrows()):
        ax = axes[row, col]
        try:
            ax.imshow(load_img(s['filename'], target_size=IMG_SIZE))
        except Exception:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(cat, fontsize=10, fontweight='bold')
        ax.axis('off')

plt.suptitle('Sample Gambar per Kelas', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('binary_sample_gambar.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Split Data — 70% Train / 15% Val / 15% Test

# %%
train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=SEED, stratify=df['class']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['class']
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    print(f"{name:5}: {len(d)} gambar — "
          f"NR: {len(d[d['class']=='Non-Recyclable'])}, "
          f"R: {len(d[d['class']=='Recyclable'])}")

# %% [markdown]
# ## 7. Data Augmentation & Generators

# %%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# %%
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df, x_col='filename', y_col='class',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', classes=categories,
    shuffle=True, seed=SEED
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df, x_col='filename', y_col='class',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', classes=categories,
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df, x_col='filename', y_col='class',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', classes=categories,
    shuffle=False
)

print(f"\nClass indices: {train_generator.class_indices}")

# %% [markdown]
# ## 8. Bangun Model — MobileNetV2 Transfer Learning
#
# - Base model MobileNetV2 pretrained ImageNet — **seluruh base di-freeze**
# - Hanya classification head yang dilatih
# - Cocok untuk dataset kecil (~1300 gambar) — menghindari overfitting

# %%
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze seluruh base model

# Classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)

model.summary()

# %% [markdown]
# ## 9. Compile & Training
#
# - **Single phase** — tanpa freeze/unfreeze
# - LR fixed 1e-3 + ReduceLROnPlateau (turunkan LR saat val_loss stagnan)
# - EarlyStopping patience 10 (cukup sabar karena single phase)

# %%
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=BinaryCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'binary_best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# %%
print("=" * 60)
print("TRAINING: Transfer Learning — MobileNetV2")
print(f"  Base Model    : Frozen (feature extractor)")
print(f"  Trainable head: GAP → BN → Dense(128) → Dropout(0.5) → Sigmoid")
print(f"  Max Epochs    : {EPOCHS}")
print(f"  LR            : {LEARNING_RATE} (+ ReduceLROnPlateau)")
print(f"  EarlyStopping : patience=10")
print("=" * 60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# %% [markdown]
# ## 10. Training History

# %%
def plot_history(history, save_path='binary_training_history.png'):
    """Plot training & validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history.history['accuracy']) + 1)

    ax1.plot(epochs_range, history.history['accuracy'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs_range, history.history['val_accuracy'], 'r-', label='Val', linewidth=2)
    ax1.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history.history['loss'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs_range, history.history['val_loss'], 'r-', label='Val', linewidth=2)
    ax2.set_title('Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Training History — Non-Recyclable vs Recyclable', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Disimpan: {save_path}")

plot_history(history)

# %% [markdown]
# ## 11. Evaluasi pada Test Set (dengan TTA)

# %%
def evaluate_with_tta(model, test_df, categories, img_size=(224, 224),
                      n_augments=5, batch_size=32):
    """Evaluasi dengan Test-Time Augmentation."""
    orig_gen = ImageDataGenerator(rescale=1./255)
    tta_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1]
    )

    # Prediksi original
    orig_flow = orig_gen.flow_from_dataframe(
        dataframe=test_df, x_col='filename', y_col='class',
        target_size=img_size, batch_size=batch_size,
        class_mode='binary', classes=categories, shuffle=False
    )
    preds_sum = model.predict(orig_flow, steps=len(orig_flow), verbose=0).flatten()

    # Prediksi augmented
    for _ in range(n_augments):
        aug_flow = tta_gen.flow_from_dataframe(
            dataframe=test_df, x_col='filename', y_col='class',
            target_size=img_size, batch_size=batch_size,
            class_mode='binary', classes=categories, shuffle=False
        )
        preds_sum += model.predict(aug_flow, steps=len(aug_flow), verbose=0).flatten()

    preds_avg = preds_sum / (1 + n_augments)
    y_pred = (preds_avg >= 0.5).astype(int)
    y_true = orig_flow.classes
    return y_true, y_pred, preds_avg

# %%
print("Evaluasi dengan Test-Time Augmentation (5x)...")
y_true, y_pred, y_proba = evaluate_with_tta(model, test_df, categories, IMG_SIZE)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (TTA)")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=categories))

acc = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy (TTA): {acc:.4f} ({acc*100:.2f}%)")

for i, cat in enumerate(categories):
    mask = y_true == i
    cat_acc = (y_pred[mask] == y_true[mask]).mean()
    print(f"  {cat}: {cat_acc:.4f} ({cat_acc*100:.2f}%)")

# %% [markdown]
# ## 12. Confusion Matrix

# %%
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories, ax=ax1)
ax1.set_title('Confusion Matrix (Absolute)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=categories, yticklabels=categories, ax=ax2)
ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.suptitle('Confusion Matrix — Non-Recyclable vs Recyclable', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('binary_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 13. Sample Prediksi

# %%
samples = test_df.sample(n=10, random_state=SEED)
fig, axes = plt.subplots(2, 5, figsize=(18, 8))

for idx, (_, row) in enumerate(samples.iterrows()):
    ax = axes[idx // 5, idx % 5]
    try:
        img = load_img(row['filename'], target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        prob = model.predict(np.expand_dims(img_array, 0), verbose=0)[0][0]
        pred = categories[int(prob >= 0.5)]
        true = row['class']
        conf = prob if prob >= 0.5 else 1 - prob

        ax.imshow(img)
        ok = pred == true
        ax.set_title(f"{'✓' if ok else '✗'} {pred}\n(True: {true})\nConf: {conf:.1%}",
                     fontsize=9, color='green' if ok else 'red', fontweight='bold')
    except Exception:
        ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')

plt.suptitle('Sample Prediksi — Test Set', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('binary_prediksi_sample.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 14. Save Model

# %%
# Keras format
model.save('binary_model_final.keras')
print("Disimpan: binary_model_final.keras")

# TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('binary_model.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"Disimpan: binary_model.tflite ({os.path.getsize('binary_model.tflite')/1024/1024:.2f} MB)")

# Class info
class_info = {
    'task': 'binary_classification',
    'categories': categories,
    'class_indices': {'Non-Recyclable': 0, 'Recyclable': 1},
    'img_size': list(IMG_SIZE),
    'model': 'MobileNetV2 (Transfer Learning, base frozen)',
    'training': 'Single phase, ReduceLROnPlateau',
    'description': 'Klasifikasi biner: Non-Recyclable vs Recyclable'
}
with open('binary_class_info.json', 'w') as f:
    json.dump(class_info, f, indent=2)
print("Disimpan: binary_class_info.json")

# %% [markdown]
# ## 15. Fungsi Prediksi

# %%
def predict_binary(image_path, model=None, model_path='binary_model_final.keras',
                   img_size=(224, 224)):
    """
    Prediksi klasifikasi biner untuk gambar baru.

    Args:
        image_path: Path ke gambar
        model: Model Keras (opsional)
        model_path: Path ke file .keras
        img_size: Ukuran target

    Returns:
        dict hasil prediksi
    """
    cats = ['Non-Recyclable', 'Recyclable']
    if model is None:
        model = load_model(model_path)

    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    prob = model.predict(np.expand_dims(img_array, 0), verbose=0)[0][0]
    pred_idx = int(prob >= 0.5)
    pred = cats[pred_idx]
    conf = prob if pred_idx == 1 else 1 - prob

    print(f"Prediksi  : {pred}")
    print(f"Confidence: {conf:.2%}")
    print(f"  Non-Recyclable: {1-prob:.4f}")
    print(f"  Recyclable    : {prob:.4f}")

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"{pred} ({conf:.1%})", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    return {'prediction': pred, 'confidence': float(conf),
            'probabilities': {'Non-Recyclable': float(1-prob), 'Recyclable': float(prob)}}

# %% [markdown]
# ## 16. Ringkasan

# %%
print("=" * 60)
print("RINGKASAN — KLASIFIKASI BINER SAMPAH")
print("=" * 60)
print(f"""
Model       : MobileNetV2 Transfer Learning (base frozen)
Training    : Single phase, {EPOCHS} epoch max
LR          : {LEARNING_RATE} + ReduceLROnPlateau
Data        : {len(df)} gambar ({len(df[df['class']=='Non-Recyclable'])} NR / {len(df[df['class']=='Recyclable'])} R)
Split       : {len(train_df)} train / {len(val_df)} val / {len(test_df)} test
Accuracy    : {acc:.4f} ({acc*100:.2f}%)

Output:
  - binary_model_final.keras
  - binary_model.tflite
  - binary_class_info.json
""")
print("=" * 60)
