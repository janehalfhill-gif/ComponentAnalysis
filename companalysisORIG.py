# ================================
# Full Ash Grain Componentry Pipeline
# Semi-Automated Labeling + Fine-Tuned Xception
# ================================

# ----------------
# 1. Install dependencies
# ----------------
# !pip install tensorflow opencv-python tifffile patchify numpy pandas scikit-learn tqdm umap-learn matplotlib seaborn

# ----------------
# 2. Import libraries
# ----------------
import tifffile as tiff
import cv2
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.cluster import KMeans
import umap

# ----------------
# 3. Settings
# ----------------
input_folder = "GreenTuffsTrial"      # raw microscopy images
patch_folder = "grains_patches"         # folder to save grain patches
embedding_folder = "embeddings"         # folder to save embeddings
output_folder = "results"               # folder to save predictions & componentry

os.makedirs(patch_folder, exist_ok=True)
os.makedirs(embedding_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

patch_size = 299       # Xception input size
num_clusters = 20      # clustering for semi-automated labeling
num_classes = 4        # ash, pumice, crystalline, lithic (for final classification)
batch_size = 32
epochs = 10

# ----------------
# 4. Function: Extract grains from one image
# ----------------
def process_image(image_path):
    try:
        img_name = os.path.basename(image_path).split(".")[0]

        # --- Read image ---
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return []

        # --- Convert to RGB (handle grayscale/RGBA) ---
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Threshold and clean mask ---
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)

        # --- Contours ---
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        padding = 20
        grain_crops = []

        for cnt in contours:
            try:
                x, y, w, h = cv2.boundingRect(cnt)

                # --- Apply padding ---
                x1 = max(x - padding, 0)
                y1 = max(y - padding, 0)
                x2 = min(x + w + padding, img.shape[1])
                y2 = min(y + h + padding, img.shape[0])

                bw = x2 - x1
                bh = y2 - y1
                side = max(bw, bh)

                # --- square crop center ---
                cx = x1 + bw // 2
                cy = y1 + bh // 2

                x1s = max(cx - side // 2, 0)
                y1s = max(cy - side // 2, 0)
                x2s = min(cx + side // 2, img.shape[1])
                y2s = min(cy + side // 2, img.shape[0])

                # --- Extract patch ---
                patch = img[y1s:y2s, x1s:x2s]

                # Skip if patch is empty or invalid
                if patch is None or patch.size == 0 or patch.shape[0] < 10 or patch.shape[1] < 10:
                    continue

                # --- Resize to Xception input ---
                patch = cv2.resize(patch, (299, 299))

                # Store grain
                grain_crops.append({
                    "image": patch,
                    "parent_image": img_name,
                    "bbox": (x1s, y1s, x2s, y2s)
                })

            except Exception as e:
                print(f"[WARN] In contour processing ({image_path}): {e}")
                continue

        return grain_crops

    except Exception as e:
        print(f"[FATAL] process_image crashed for {image_path}: {e}")
        return []

# ----------------
# 5. Extract grains from all images in parallel
# ----------------
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
               if f.lower().endswith(('.tif','.tiff','.png','.jpg'))]

num_workers = min(cpu_count(), len(image_files))
all_grains = []

with Pool(num_workers) as p:
    for grains in tqdm(p.imap(process_image, image_files), total=len(image_files)):
        all_grains.extend(grains)

print(f"Total grains extracted: {len(all_grains)}")

# ----------------
# 6. Load Xception model and extract embeddings
# ----------------
base_model = Xception(weights='imagenet', include_top=False, pooling='avg')

embeddings = []
for grain in tqdm(all_grains):
    img = grain["image"].astype("float32")  # extract the patch
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    emb = base_model.predict(x, verbose=0)
    embeddings.append(emb.flatten())

embeddings = np.array(embeddings)
np.save(os.path.join(embedding_folder, "grain_embeddings.npy"), embeddings)
print("Embeddings saved:", embeddings.shape)

# ----------------
# 7. Cluster grains for semi-automated labeling
# ----------------
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# ----------------
# 8. Visualize cluster representatives (optional)
# ----------------
for cluster_id in range(num_clusters):
    indices = np.where(cluster_labels == cluster_id)[0]
    if len(indices) == 0:
        continue
    rep_idx = random.choice(indices)

    plt.figure(figsize=(2,2))
    plt.imshow(all_grains[rep_idx]["image"])
    plt.title(f"Cluster {cluster_id}")
    plt.axis('off')
    plt.show()

# At this point, manually assign a label to each cluster:
# Example: cluster_to_class = {0:'ash', 1:'pumice', ...}
# Replace with your own cluster labeling
cluster_to_class = {i: 'unknown' for i in range(num_clusters)}  # <-- manually assign

semi_labels = [cluster_to_class[cl] for cl in cluster_labels]

# Save semi-automated labels
df_semi = pd.DataFrame({'cluster': cluster_labels, 'class': semi_labels})
df_semi.to_csv(os.path.join(output_folder, "semi_auto_labels.csv"), index=False)

# ----------------
# 9. Fine-tune Xception using labeled patches
# ----------------
# Organize grains_patches folder in subfolders by class:
# grains_patches/
# ├── ash/
# ├── pumice/
# ├── crystalline/
# └── lithic/

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = datagen.flow_from_directory(
    patch_folder,
    target_size=(patch_size, patch_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    patch_folder,
    target_size=(patch_size, patch_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Optional: unfreeze last layers
for layer in base_model.layers[-30:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=5)

# ----------------
# 10. Classify all grains and compute componentry
# ----------------
grain_imgs = np.array([cv2.resize(g, (patch_size, patch_size)) for g in all_grains])
grain_imgs = preprocess_input(grain_imgs)

pred_probs = model.predict(grain_imgs, batch_size=batch_size)
pred_classes = np.argmax(pred_probs, axis=1)
class_labels = {v:k for k,v in train_generator.class_indices.items()}
pred_labels = [class_labels[i] for i in pred_classes]

df_pred = pd.DataFrame({'class': pred_labels})
percentages = df_pred['class'].value_counts(normalize=True) * 100

df_pred.to_csv(os.path.join(output_folder, "all_grain_predictions.csv"), index=False)
percentages.to_csv(os.path.join(output_folder, "componentry_percentages.csv"))
np.save(os.path.join(embedding_folder, "fine_tuned_embeddings.npy"), embeddings)

print("Pipeline complete! Componentry percentages:")
print(percentages)



