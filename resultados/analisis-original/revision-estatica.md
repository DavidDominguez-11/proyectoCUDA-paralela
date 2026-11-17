# Evidencia de lectura estática del original

Fecha UTC: 2026-11-17T16:43:42.379833+00:00.

Tipo: fuentes y salidas históricas extraídas; sin ejecutar celdas.

Notebook SHA-256: `ce8824e860ae7e02684bf702fd6865b7d87cc9eb71917f56bfb0ee8d9ccb8e8a`.

PDF SHA-256: `2dd8efc1ffb36d1b0908fbb400808a713b99e1c1322f71843e3e22198f72eb6d`.

Celdas: 68; de código: 33.

Índices de celda desde cero. El contador guardado no identifica una corrida nueva.

Comando desde la raíz (usar otro destino para repetir sin sobrescribir):

```powershell
python -X utf8 scripts/inspeccionar_original.py --destino resultados/analisis-original/revision-estatica.md
```

## Épocas visibles en salidas guardadas

| Celda | Modelo | Etiquetas Epoch encontradas | Última época visible | Máximo anunciado |
|---|---|---:|---:|---:|
| 32 | CNN-A | 15 | 15 | 15 |
| 34 | CNN-B | 11 | 11 | 15 |
| 41 | FC-A | 15 | 15 | 15 |
| 42 | FC-B | 15 | 15 | 15 |
| 52 | CNN seleccionada + augmentation | 13 | 13 | 15 |

Estas etiquetas pertenecen a logs persistidos. No prueban una ejecución actual ni tiempos comparables.

## Código completo de las celdas originales

### Celda 2

SHA-256 del texto UTF-8 de la celda: `18e9c4ce338626bd593fe02336efbf047e630e93738c7d739da6f21c40ebf65d`.

```python
import os
import json
import random
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

import kagglehub

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
```

### Celda 4

SHA-256 del texto UTF-8 de la celda: `40040d96ff9fd748f2c944d9a738390c7be08ea155de4faa4e43f08f87913ef6`.

```python
# Download latest version
path = kagglehub.dataset_download("grassknoted/asl-alphabet")

print("Path to dataset files:", path)

DATASET_DIR = Path(path)
TRAIN_DIR = DATASET_DIR / "asl_alphabet_train" / "asl_alphabet_train"
TEST_DIR = DATASET_DIR / "asl_alphabet_test" / "asl_alphabet_test"

assert TRAIN_DIR.exists(), f"No se encontró la carpeta de entrenamiento en {TRAIN_DIR}"
assert TEST_DIR.exists(), f"No se encontró la carpeta de prueba oficial en {TEST_DIR}"

CLASSES = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
print(f"Número de clases: {len(CLASSES)}")
print(CLASSES)
```

### Celda 6

SHA-256 del texto UTF-8 de la celda: `18679fe5ba769b89f96ea2f79cbac0505e28378b57c9fa9a2af19a0b39136f0e`.

```python
counts = {c: len(list((TRAIN_DIR / c).glob("*.jpg"))) for c in CLASSES}
df_counts = pd.DataFrame({"clase": list(counts.keys()), "n_imagenes": list(counts.values())})
df_counts = df_counts.sort_values("clase").reset_index(drop=True)

test_files = list(TEST_DIR.glob("*.jpg"))

print(f"Total de imágenes de entrenamiento disponibles: {df_counts['n_imagenes'].sum()}")
print(f"Número de clases: {df_counts.shape[0]}")
print(f"Imágenes en el set de prueba OFICIAL de Kaggle: {len(test_files)}")
df_counts
```

### Celda 9

SHA-256 del texto UTF-8 de la celda: `7d71d67d18543b8e9e1d288fe1c05d5860e27075253c7a217a2c7e5cc8f53412`.

```python
def show_class_samples(classes, n_samples=5, seed=SEED):
    rng = random.Random(seed)
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(2.2 * n_samples, 2.2 * len(classes)))
    for i, c in enumerate(classes):
        files = list((TRAIN_DIR / c).glob("*.jpg"))
        sample_files = rng.sample(files, n_samples)
        for j, f in enumerate(sample_files):
            img = Image.open(f)
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(c, fontsize=12)
                ax.text(-0.15, 0.5, c, fontsize=14, fontweight="bold",
                        transform=ax.transAxes, va="center", ha="right")
    plt.suptitle("Variabilidad intra-clase: 5 muestras por letra", fontsize=14, y=1.0)
    plt.tight_layout()
    plt.show()

show_class_samples(["A", "B", "L", "M", "N", "S", "space"], n_samples=5)
```

### Celda 12

SHA-256 del texto UTF-8 de la celda: `c741695fecc4cf4f58f93a99110774a95b2aa2a8fa3808cd58e16ba8ed17e3aa`.

```python
rng = random.Random(SEED)
sample_files = []
for c in rng.sample(CLASSES, 10):
    files = list((TRAIN_DIR / c).glob("*.jpg"))
    sample_files.extend(rng.sample(files, 5))

props = []
for f in sample_files:
    img = Image.open(f)
    props.append({"archivo": f.name, "clase": f.parent.name, "tamaño": img.size,
                  "modo": img.mode, "formato": img.format})

df_props = pd.DataFrame(props)
print(df_props["tamaño"].value_counts())
print(df_props["modo"].value_counts())
print(df_props["formato"].value_counts())
```

### Celda 15

SHA-256 del texto UTF-8 de la celda: `295841faa3b0f9531d195fbe850e607a8fb01eb74cce745c8aad3341465ffa94`.

```python
plt.figure(figsize=(14, 6))
sns.barplot(data=df_counts, x="clase", y="n_imagenes", color="#4C72B0")
plt.xticks(rotation=90)
plt.title("Número de imágenes por clase (set de entrenamiento original)")
plt.ylabel("Número de imágenes")
plt.xlabel("Clase")
plt.tight_layout()
plt.show()

print(df_counts["n_imagenes"].describe())
```

### Celda 18

SHA-256 del texto UTF-8 de la celda: `462030509877fdb979b4e4336220826e11b81d839871a7813c476d80ca650269`.

```python
def show_confusable_groups(groups, n_samples=4):
    n_rows = sum(len(g) for g in groups)
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(2.2 * n_samples, 2.2 * n_rows))
    rng = random.Random(SEED)
    row = 0
    for group in groups:
        for c in group:
            files = list((TRAIN_DIR / c).glob("*.jpg"))
            sample_files = rng.sample(files, n_samples)
            for j, f in enumerate(sample_files):
                img = Image.open(f)
                ax = axes[row, j]
                ax.imshow(img)
                ax.axis("off")
                if j == 0:
                    ax.text(-0.15, 0.5, c, fontsize=14, fontweight="bold",
                            transform=ax.transAxes, va="center", ha="right")
            row += 1
    plt.suptitle("Grupos de letras visualmente similares", fontsize=14, y=1.0)
    plt.tight_layout()
    plt.show()

show_confusable_groups([["M", "N", "S"], ["U", "V", "R"]], n_samples=4)
```

### Celda 22

SHA-256 del texto UTF-8 de la celda: `829b6baa3531b54e7efa235495bb6777343c475fd6faaced077ab04ae233cddb`.

```python
SAMPLES_PER_CLASS = 600
IMG_SIZE = 64

def build_sampled_file_list(train_dir, classes, samples_per_class, seed=SEED):
    rng = random.Random(seed)
    records = []
    for c in classes:
        files = list((train_dir / c).glob("*.jpg"))
        chosen = rng.sample(files, min(samples_per_class, len(files)))
        for f in chosen:
            records.append({"filepath": str(f), "label": c})
    return pd.DataFrame(records)

df_sample = build_sampled_file_list(TRAIN_DIR, CLASSES, SAMPLES_PER_CLASS)
print(f"Total de imágenes submuestreadas: {len(df_sample)}")
df_sample["label"].value_counts().sort_index()
```

### Celda 23

SHA-256 del texto UTF-8 de la celda: `cba34d5910e7e98d0100808ecf552ef076bb8860bab8e40bc3ba8bc7036d9375`.

```python
# División estratificada 70/15/15 (train / val / test) sobre la submuestra
df_train, df_temp = train_test_split(
    df_sample, test_size=0.30, stratify=df_sample["label"], random_state=SEED
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, stratify=df_temp["label"], random_state=SEED
)

print(f"Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")
print("\nProporción por clase en train (primeras 5):")
print(df_train["label"].value_counts(normalize=True).sort_index().head())
```

### Celda 24

SHA-256 del texto UTF-8 de la celda: `9a1366af2077b6a42697fb6f81b7e75c62e73b501787412ed07520a0de0344be`.

```python
# Codificación de etiquetas: nombre de clase -> índice entero
label_to_idx = {c: i for i, c in enumerate(CLASSES)}
idx_to_label = {i: c for c, i in label_to_idx.items()}

def load_images(df, img_size=IMG_SIZE):
    X = np.zeros((len(df), img_size, img_size, 3), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.int64)
    for i, row in enumerate(df.itertuples()):
        img = Image.open(row.filepath).convert("RGB").resize((img_size, img_size))
        X[i] = np.asarray(img, dtype=np.float32) / 255.0
        y[i] = label_to_idx[row.label]
    return X, y

X_train, y_train = load_images(df_train)
X_val, y_val = load_images(df_val)
X_test, y_test = load_images(df_test)

print("X_train:", X_train.shape, X_train.dtype, "min/max:", X_train.min(), X_train.max())
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)
```

### Celda 26

SHA-256 del texto UTF-8 de la celda: `ecbc04f5af04689ef61204fc3684a4699baba394c1d05c3a5d629c24c40bdaac`.

```python
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

np.save(PROCESSED_DIR / "X_train.npy", X_train)
np.save(PROCESSED_DIR / "y_train.npy", y_train)
np.save(PROCESSED_DIR / "X_val.npy", X_val)
np.save(PROCESSED_DIR / "y_val.npy", y_val)
np.save(PROCESSED_DIR / "X_test.npy", X_test)
np.save(PROCESSED_DIR / "y_test.npy", y_test)

with open(PROCESSED_DIR / "label_mapping.json", "w") as f:
    json.dump(idx_to_label, f, indent=2)

print("Arreglos preprocesados guardados en:", PROCESSED_DIR.resolve())
```

### Celda 28

SHA-256 del texto UTF-8 de la celda: `feb426150ed04c5eee84c8da58a0dfd638feb7600cab4df7e8eae348980fb5b3`.

```python
example_row = df_train.iloc[0]
original = Image.open(example_row["filepath"]).convert("RGB")
processed = (X_train[0] * 255).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(original)
axes[0].set_title(f"Original 200x200 — clase '{example_row['label']}'")
axes[0].axis("off")
axes[1].imshow(processed)
axes[1].set_title(f"Preprocesada {IMG_SIZE}x{IMG_SIZE}, normalizada")
axes[1].axis("off")
plt.tight_layout()
plt.show()
```

### Celda 31

SHA-256 del texto UTF-8 de la celda: `d825be3c27643d79059bd3a53a43813635411d090458e64c2dbb9695a05003fb`.

```python
from tensorflow import keras
from tensorflow.keras import layers, callbacks

NUM_CLASSES = len(CLASSES)
EPOCHS = 15
BATCH_SIZE = 128

def make_early_stop():
    return callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

def build_cnn_a():
    model = keras.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="CNN_A_baseline")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

cnn_a = build_cnn_a()
cnn_a.summary()
```

### Celda 32

SHA-256 del texto UTF-8 de la celda: `c21db28fb45418c380c72d7df98e0b56a9c6e4bc4e3dd83125e7880363e05a90`.

```python
history_a = cnn_a.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[make_early_stop()], verbose=2,
)
```

### Celda 33

SHA-256 del texto UTF-8 de la celda: `0a45e6f0d29af5682c2d13dc2f348a4fa60ee7bfca21546a3c1ddbb6e1149bac`.

```python
def build_cnn_b():
    # momentum=0.9 (en vez del 0.99 por defecto) para que las estadísticas de
    # BatchNormalization converjan más rápido dado el número de pasos por época;
    # con el valor por defecto la validación se queda "atascada" en accuracy
    # aleatoria durante varias épocas hasta que el promedio móvil se estabiliza.
    def bn():
        return layers.BatchNormalization(momentum=0.9)

    model = keras.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        bn(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        bn(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        bn(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        bn(),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="CNN_B_regularizada")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

cnn_b = build_cnn_b()
cnn_b.summary()
```

### Celda 34

SHA-256 del texto UTF-8 de la celda: `556369d45be19a29bd21c350f893f7db4d26a9fecf7a8d105e2f5081b9ef43cd`.

```python
history_b = cnn_b.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[make_early_stop()], verbose=2,
)
```

### Celda 35

SHA-256 del texto UTF-8 de la celda: `5b9b6626008ed7ab8ad5e5c91b553088ce141b04b18d567a7c3b4458a5924d35`.

```python
def plot_history(histories, labels, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for h, lbl in zip(histories, labels):
        axes[0].plot(h.history["accuracy"], label=f"{lbl} train")
        axes[0].plot(h.history["val_accuracy"], linestyle="--", label=f"{lbl} val")
        axes[1].plot(h.history["loss"], label=f"{lbl} train")
        axes[1].plot(h.history["val_loss"], linestyle="--", label=f"{lbl} val")
    axes[0].set_title(f"Accuracy {title}"); axes[0].set_xlabel("Época"); axes[0].legend(fontsize=8)
    axes[1].set_title(f"Loss {title}"); axes[1].set_xlabel("Época"); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

plot_history([history_a, history_b], ["CNN-A", "CNN-B"], title="(CNN-A vs CNN-B)")
```

### Celda 36

SHA-256 del texto UTF-8 de la celda: `29242238a8bae1ee42529a484e2dd297d4d3bb261ab6662326de7463dc945ec5`.

```python
test_loss_a, test_acc_a = cnn_a.evaluate(X_test, y_test, verbose=0)
test_loss_b, test_acc_b = cnn_b.evaluate(X_test, y_test, verbose=0)

print(f"CNN-A -> test accuracy: {test_acc_a:.4f}, test loss: {test_loss_a:.4f}")
print(f"CNN-B -> test accuracy: {test_acc_b:.4f}, test loss: {test_loss_b:.4f}")

best_cnn_name = "CNN-A" if test_acc_a >= test_acc_b else "CNN-B"
best_cnn_model = cnn_a if test_acc_a >= test_acc_b else cnn_b
best_cnn_test_acc = max(test_acc_a, test_acc_b)
print("\nMejor CNN:", best_cnn_name, f"(test accuracy = {best_cnn_test_acc:.4f})")
```

### Celda 38

SHA-256 del texto UTF-8 de la celda: `3cd5654922ac4ced0c6a2b827609acecf8afa5ff9608632ee5a06b650c0f67b4`.

```python
from sklearn.metrics import confusion_matrix, classification_report

y_pred_probs_cnn = best_cnn_model.predict(X_test, verbose=0)
y_pred_cnn = np.argmax(y_pred_probs_cnn, axis=1)

cm = confusion_matrix(y_test, y_pred_cnn)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f"Matriz de confusión — {best_cnn_name} (set de prueba)")
plt.xlabel("Predicción"); plt.ylabel("Real")
plt.tight_layout()
plt.show()

print(classification_report(y_test, y_pred_cnn, target_names=CLASSES, zero_division=0))
```

### Celda 41

SHA-256 del texto UTF-8 de la celda: `9780dcc56f19bddfed9eb7584b0692d27ebb0240d48c0b3ccb3c576770631b2f`.

```python
def build_fc(hidden_layers, dropout=0.3, name="FC"):
    # BatchNormalization justo después de Flatten: sin ella, la red fully-connected
    # no lograba entrenar (el optimizador la llevaba a una salida uniforme -log(1/29)-
    # desde la primera época, un caso de "dead ReLU" típico con entradas de muy alta
    # dimensión sin normalizar por lote).
    model = keras.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Flatten(),
        layers.BatchNormalization(),
    ], name=name)
    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

fc_a = build_fc([128, 64], dropout=0.3, name="FC_A_pequena")
fc_a.summary()
history_fc_a = fc_a.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[make_early_stop()], verbose=2,
)
```

### Celda 42

SHA-256 del texto UTF-8 de la celda: `18e7027a65b7be01b3820e608ed8090ef69f06273fef38c93158c7d2b6bc9fe3`.

```python
fc_b = build_fc([512, 256, 128], dropout=0.4, name="FC_B_grande")
fc_b.summary()
history_fc_b = fc_b.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[make_early_stop()], verbose=2,
)
```

### Celda 43

SHA-256 del texto UTF-8 de la celda: `7aba88f32c73d939769308946ab294731ce1741322d29da07b809365e307379d`.

```python
plot_history([history_fc_a, history_fc_b], ["FC-A", "FC-B"], title="(FC-A vs FC-B)")

test_loss_fc_a, test_acc_fc_a = fc_a.evaluate(X_test, y_test, verbose=0)
test_loss_fc_b, test_acc_fc_b = fc_b.evaluate(X_test, y_test, verbose=0)

print(f"FC-A -> test accuracy: {test_acc_fc_a:.4f}")
print(f"FC-B -> test accuracy: {test_acc_fc_b:.4f}")

best_fc_name = "FC-A" if test_acc_fc_a >= test_acc_fc_b else "FC-B"
best_fc_model = fc_a if test_acc_fc_a >= test_acc_fc_b else fc_b
best_fc_test_acc = max(test_acc_fc_a, test_acc_fc_b)
print("\nMejor red fully-connected:", best_fc_name, f"(test accuracy = {best_fc_test_acc:.4f})")
```

### Celda 46

SHA-256 del texto UTF-8 de la celda: `d9e1d35f54f3d2c4648184e9db5494e8fbbf3704c7d33ba8bef89bff6059e54e`.

```python
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

N_COMPONENTS = 150
pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_flat)
X_val_pca = pca.transform(X_val_flat)
X_test_pca = pca.transform(X_test_flat)

print(f"Varianza explicada acumulada con {N_COMPONENTS} componentes: {pca.explained_variance_ratio_.sum():.3f}")
```

### Celda 47

SHA-256 del texto UTF-8 de la celda: `23ca9b392957c62277d9c449310500a750ee90ccb3936b09ab72975cf2136e7e`.

```python
rf_configs = {
    "RF-A (100 árboles, sin límite de profundidad)": RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=SEED, n_jobs=-1
    ),
    "RF-B (300 árboles, profundidad máx. 30)": RandomForestClassifier(
        n_estimators=300, max_depth=30, random_state=SEED, n_jobs=-1
    ),
}

rf_results = {}
for name, clf in rf_configs.items():
    clf.fit(X_train_pca, y_train)
    val_acc = accuracy_score(y_val, clf.predict(X_val_pca))
    rf_results[name] = {"model": clf, "val_accuracy": val_acc}
    print(f"{name} -> val accuracy: {val_acc:.4f}")

best_rf_name = max(rf_results, key=lambda k: rf_results[k]["val_accuracy"])
best_rf_model = rf_results[best_rf_name]["model"]
test_acc_rf = accuracy_score(y_test, best_rf_model.predict(X_test_pca))

print(f"\nMejor Random Forest: {best_rf_name}")
print(f"Test accuracy: {test_acc_rf:.4f}")
```

### Celda 50

SHA-256 del texto UTF-8 de la celda: `e2c9f7889abac49d1140573dc1d4ae27aaf0b97c78d1ba91cc083b874ef6fcfb`.

```python
comparison = pd.DataFrame({
    "modelo": ["CNN-A", "CNN-B", "FC-A", "FC-B", best_rf_name],
    "tipo": ["CNN", "CNN", "Fully-connected", "Fully-connected", "Random Forest"],
    "test_accuracy": [test_acc_a, test_acc_b, test_acc_fc_a, test_acc_fc_b, test_acc_rf],
}).sort_values("test_accuracy", ascending=False).reset_index(drop=True)

plt.figure(figsize=(9, 5))
sns.barplot(data=comparison, x="modelo", y="test_accuracy", hue="tipo", dodge=False)
plt.ylim(0, 1)
plt.title("Comparación de accuracy en el set de prueba")
plt.ylabel("Test accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

comparison
```

### Celda 52

SHA-256 del texto UTF-8 de la celda: `b531f7b1d9e6e108d7d24b4d611dc6a7ef730d14400d30d719dfc09816abb83a`.

```python
augmentation = keras.Sequential([
    layers.RandomRotation(0.04),        # ~ ±15 grados
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

build_best_cnn = build_cnn_a if best_cnn_name == "CNN-A" else build_cnn_b

inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
x = augmentation(inputs)
outputs = build_best_cnn()(x)
cnn_aug = keras.Model(inputs, outputs, name=f"{best_cnn_name.replace('-', '_')}_con_augmentation")
cnn_aug.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history_aug = cnn_aug.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[make_early_stop()], verbose=2,
)
```

### Celda 53

SHA-256 del texto UTF-8 de la celda: `b4b72405819699add360f5ff739e6709a5a411dc9fcd29f414b952993c0d43d8`.

```python
plot_history([history_aug], [f"{best_cnn_name}+aug"], title="(con augmentation)")

test_loss_aug, test_acc_aug = cnn_aug.evaluate(X_test, y_test, verbose=0)
print(f"{best_cnn_name} sin augmentation -> test accuracy: {best_cnn_test_acc:.4f}")
print(f"{best_cnn_name} con augmentation  -> test accuracy: {test_acc_aug:.4f}")
```

### Celda 56

SHA-256 del texto UTF-8 de la celda: `fcff58a3bea3439678266f6daa88ab82080b668410b2e323655565c6cf56ae6b`.

```python
final_comparison = pd.concat([
    comparison,
    pd.DataFrame([{
        "modelo": f"{best_cnn_name}+aug",
        "tipo": "CNN + augmentation",
        "test_accuracy": test_acc_aug,
    }])
], ignore_index=True).sort_values("test_accuracy", ascending=False).reset_index(drop=True)

display(final_comparison)

model_registry = {
    "CNN-A": cnn_a,
    "CNN-B": cnn_b,
    "FC-A": fc_a,
    "FC-B": fc_b,
    best_rf_name: best_rf_model,
    f"{best_cnn_name}+aug": cnn_aug,
}

final_model_name = final_comparison.iloc[0]["modelo"]
final_model = model_registry[final_model_name]
final_test_acc = final_comparison.iloc[0]["test_accuracy"]

print(f"\nModelo final seleccionado: {final_model_name} (test accuracy = {final_test_acc:.4f})")
```

### Celda 58

SHA-256 del texto UTF-8 de la celda: `5c2fbc5ec23f23fdafcde43aa445e269cac463afb1a6e425279df43b6ef06723`.

```python
import joblib

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

for name, obj in {"cnn_a": cnn_a, "cnn_b": cnn_b, "fc_a": fc_a, "fc_b": fc_b, "cnn_aug": cnn_aug}.items():
    obj.save(MODELS_DIR / f"{name}.keras")

joblib.dump(best_rf_model, MODELS_DIR / "best_rf_model.joblib")
joblib.dump(pca, MODELS_DIR / "pca.joblib")
joblib.dump({"best_cnn_name": best_cnn_name, "best_rf_name": best_rf_name,
             "final_model_name": final_model_name}, MODELS_DIR / "metadata.joblib")

print("Modelos guardados en:", MODELS_DIR.resolve())
```

### Celda 61

SHA-256 del texto UTF-8 de la celda: `b3cfc3c1c6e32c36cdfef50b9b46567fab73763667432bb8327540996c94326e`.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import json
import joblib
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MODELS_DIR = Path("data/models")
PROCESSED_DIR = Path("data/processed")

IMG_SIZE = 64
idx_to_label = {int(k): v for k, v in json.load(open(PROCESSED_DIR / "label_mapping.json")).items()}
CLASSES = [idx_to_label[i] for i in range(len(idx_to_label))]

X_test = np.load(PROCESSED_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")

cnn_a = keras.models.load_model(MODELS_DIR / "cnn_a.keras")
cnn_b = keras.models.load_model(MODELS_DIR / "cnn_b.keras")
fc_a = keras.models.load_model(MODELS_DIR / "fc_a.keras")
fc_b = keras.models.load_model(MODELS_DIR / "fc_b.keras")
cnn_aug = keras.models.load_model(MODELS_DIR / "cnn_aug.keras")
best_rf_model = joblib.load(MODELS_DIR / "best_rf_model.joblib")
pca = joblib.load(MODELS_DIR / "pca.joblib")

metadata = joblib.load(MODELS_DIR / "metadata.joblib")
best_cnn_name = metadata["best_cnn_name"]
best_rf_name = metadata["best_rf_name"]

# Recalculamos el accuracy de cada modelo sobre el mismo test set guardado
# (solo inferencia, toma segundos) para reconstruir la tabla comparativa.
X_test_pca = pca.transform(X_test.reshape(len(X_test), -1))

_, test_acc_a = cnn_a.evaluate(X_test, y_test, verbose=0)
_, test_acc_b = cnn_b.evaluate(X_test, y_test, verbose=0)
_, test_acc_fc_a = fc_a.evaluate(X_test, y_test, verbose=0)
_, test_acc_fc_b = fc_b.evaluate(X_test, y_test, verbose=0)
_, test_acc_aug = cnn_aug.evaluate(X_test, y_test, verbose=0)
test_acc_rf = accuracy_score(y_test, best_rf_model.predict(X_test_pca))

comparison = pd.DataFrame({
    "modelo": ["CNN-A", "CNN-B", "FC-A", "FC-B", best_rf_name],
    "tipo": ["CNN", "CNN", "Fully-connected", "Fully-connected", "Random Forest"],
    "test_accuracy": [test_acc_a, test_acc_b, test_acc_fc_a, test_acc_fc_b, test_acc_rf],
}).sort_values("test_accuracy", ascending=False).reset_index(drop=True)

final_comparison = pd.concat([
    comparison,
    pd.DataFrame([{"modelo": f"{best_cnn_name}+aug", "tipo": "CNN + augmentation", "test_accuracy": test_acc_aug}]),
], ignore_index=True).sort_values("test_accuracy", ascending=False).reset_index(drop=True)

model_registry = {
    "CNN-A": cnn_a, "CNN-B": cnn_b, "FC-A": fc_a, "FC-B": fc_b,
    best_rf_name: best_rf_model, f"{best_cnn_name}+aug": cnn_aug,
}
final_model_name = final_comparison.iloc[0]["modelo"]
final_model = model_registry[final_model_name]

print("Modelos y datos recargados desde disco.")
print(f"Modelo final: {final_model_name} (test accuracy = {final_comparison.iloc[0]['test_accuracy']:.4f})")
```

### Celda 62

SHA-256 del texto UTF-8 de la celda: `1bef33e91eb1db4f0faf1b7c89c4b725d6434f5961698eef47f88a8d8b5d7ee0`.

```python
IMAGENES_GRUPO_DIR = Path("imagenes_grupo")

def preprocess_single_image(filepath, img_size=IMG_SIZE):
    img = Image.open(filepath).convert("RGB").resize((img_size, img_size))
    return np.asarray(img, dtype=np.float32) / 255.0

def predict_with_final_model(arr):
    if isinstance(final_model, RandomForestClassifier):
        flat_pca = pca.transform(arr.reshape(1, -1))
        pred_idx = final_model.predict(flat_pca)[0]
    else:
        pred_idx = int(np.argmax(final_model.predict(arr[np.newaxis, ...], verbose=0)[0]))
    return idx_to_label[pred_idx]

own_files = sorted(IMAGENES_GRUPO_DIR.glob("*.*"))
assert len(own_files) >= 5, "Se requieren al menos 5 imágenes en imagenes_grupo/"
print(f"Imágenes encontradas en {IMAGENES_GRUPO_DIR}/: {len(own_files)}")

own_results = []
for f in own_files:
    true_label = f.stem.split("_")[0]
    arr = preprocess_single_image(f)
    pred_label = predict_with_final_model(arr)
    own_results.append({
        "archivo": f.name, "real": true_label, "prediccion": pred_label,
        "correcto": true_label.upper() == pred_label.upper(),
    })

df_own = pd.DataFrame(own_results)
df_own
```

### Celda 63

SHA-256 del texto UTF-8 de la celda: `1dc36568b048a7c547b9321fc823bddb9d4c1c6af91bef4afb30bf8c1739f65a`.

```python
n = len(own_files)
cols = min(n, 5)
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
axes = np.array(axes).reshape(-1)

for ax, f, row in zip(axes, own_files, df_own.itertuples()):
    img = Image.open(f).convert("RGB")
    ax.imshow(img)
    color = "green" if row.correcto else "red"
    ax.set_title(f"real: {row.real} | pred: {row.prediccion}", color=color, fontsize=10)
    ax.axis("off")
for ax in axes[len(own_files):]:
    ax.axis("off")

plt.suptitle(f"Predicciones del modelo final ({final_model_name}) sobre imagenes_grupo/", y=1.02)
plt.tight_layout()
plt.show()

accuracy_own = df_own["correcto"].mean()
print(f"Accuracy sobre imágenes propias del equipo: {accuracy_own:.2%} ({df_own['correcto'].sum()}/{len(df_own)})")
```

### Celda 64

SHA-256 del texto UTF-8 de la celda: `51fd0fc36ea1aec102a42a166420d2421fb5ed600b78890b8e084e8f9c92b7ab`.

```python
# Chequeo adicional: ¿el modelo con mejor accuracy interno (test set propio) es
# también el que mejor generaliza a estas imágenes independientes? Evaluamos TODOS
# los modelos entrenados sobre imagenes_grupo/, no solo el seleccionado como "final".

def predict_label_generic(model_obj, arr):
    if isinstance(model_obj, RandomForestClassifier):
        pred_idx = model_obj.predict(pca.transform(arr.reshape(1, -1)))[0]
    else:
        pred_idx = int(np.argmax(model_obj.predict(arr[np.newaxis, ...], verbose=0)[0]))
    return idx_to_label[pred_idx]

own_arrays = [preprocess_single_image(f) for f in own_files]
own_true = [f.stem.split("_")[0] for f in own_files]

generalization_rows = []
for name, model_obj in model_registry.items():
    preds = [predict_label_generic(model_obj, arr) for arr in own_arrays]
    acc_placeholder = np.mean([p.upper() == t.upper() for p, t in zip(preds, own_true)])
    generalization_rows.append({"modelo": name, "test_accuracy_interno": final_comparison.set_index("modelo").loc[name, "test_accuracy"] if name in final_comparison["modelo"].values else np.nan,
                                 "accuracy_imagenes_grupo": acc_placeholder})

df_generalization = pd.DataFrame(generalization_rows).sort_values("accuracy_imagenes_grupo", ascending=False).reset_index(drop=True)
df_generalization
```

## Salidas históricas seleccionadas

### Salida de celda 6

```text
Total de imágenes de entrenamiento disponibles: 87000
Número de clases: 29
Imágenes en el set de prueba OFICIAL de Kaggle: 28
      clase  n_imagenes
0         A        3000
1         B        3000
2         C        3000
3         D        3000
4         E        3000
5         F        3000
6         G        3000
7         H        3000
8         I        3000
9         J        3000
10        K        3000
11        L        3000
12        M        3000
13        N        3000
14        O        3000
15        P        3000
16        Q        3000
17        R        3000
18        S        3000
19        T        3000
20        U        3000
21        V        3000
22        W        3000
23        X        3000
24        Y        3000
25        Z        3000
26      del        3000
27  nothing        3000
28    space        3000
```

### Salida de celda 23

```text
Train: 12180  Val: 2610  Test: 2610

Proporción por clase en train (primeras 5):
label
A    0.034483
B    0.034483
C    0.034483
D    0.034483
E    0.034483
Name: proportion, dtype: float64
```

### Salida de celda 24

```text
X_train: (12180, 64, 64, 3) float32 min/max: 0.0 1.0
X_val:   (2610, 64, 64, 3)
X_test:  (2610, 64, 64, 3)
```

### Salida de celda 36

```text
CNN-A -> test accuracy: 0.9272, test loss: 0.2450
CNN-B -> test accuracy: 0.9552, test loss: 0.3566

Mejor CNN: CNN-B (test accuracy = 0.9552)
```

### Salida de celda 43

```text
<Figure size 1200x400 with 2 Axes>
FC-A -> test accuracy: 0.5307
FC-B -> test accuracy: 0.4203

Mejor red fully-connected: FC-A (test accuracy = 0.5307)
```

### Salida de celda 47

```text
RF-A (100 árboles, sin límite de profundidad) -> val accuracy: 0.8962
RF-B (300 árboles, profundidad máx. 30) -> val accuracy: 0.9103

Mejor Random Forest: RF-B (300 árboles, profundidad máx. 30)
Test accuracy: 0.8973
```

### Salida de celda 53

```text
<Figure size 1200x400 with 2 Axes>
CNN-B sin augmentation -> test accuracy: 0.9552
CNN-B con augmentation  -> test accuracy: 0.9559
```

### Salida de celda 56

```text
                                    modelo                tipo  test_accuracy
0                                CNN-B+aug  CNN + augmentation       0.955939
1                                    CNN-B                 CNN       0.955172
2                                    CNN-A                 CNN       0.927203
3  RF-B (300 árboles, profundidad máx. 30)       Random Forest       0.897318
4                                     FC-A     Fully-connected       0.530651
5                                     FC-B     Fully-connected       0.420307

Modelo final seleccionado: CNN-B+aug (test accuracy = 0.9559)
```

### Salida de celda 62

```text
Imágenes encontradas en imagenes_grupo/: 15
             archivo   real prediccion  correcto
0        A_luis.jpeg      A          M     False
1        B_luis.jpeg      B          U     False
2        C_luis.jpeg      C          C      True
3        D_luis.jpeg      D          Z     False
4        G_luis.jpeg      G          P     False
5        K_bran.jpeg      K          K      True
6        L_bran.jpeg      L          L      True
7        M_bran.jpeg      M          J     False
8        N_bran.jpeg      N          P     False
9        R_bran.jpeg      R          L     False
10      S_david.jpeg      S          E     False
11  space_david.jpeg  space          P     False
12      U_david.jpeg      U          E     False
13      V_david.jpeg      V          E     False
14      W_david.jpeg      W          E     False
```

### Salida de celda 64

```text
                                    modelo  test_accuracy_interno  \
0                                CNN-B+aug               0.955939   
1                                    CNN-A               0.927203   
2                                    CNN-B               0.955172   
3                                     FC-A               0.530651   
4                                     FC-B               0.420307   
5  RF-B (300 árboles, profundidad máx. 30)               0.897318   

   accuracy_imagenes_grupo  
0                 0.200000  
1                 0.133333  
2                 0.133333  
3                 0.066667  
4                 0.066667  
5                 0.000000
```
