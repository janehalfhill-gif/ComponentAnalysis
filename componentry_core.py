# componentry_core.py

import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.cluster import KMeans

# Cached model instance to avoid repeated loads per run.
_XCEPTION_MODEL = None


def _get_xception_model():
    global _XCEPTION_MODEL
    if _XCEPTION_MODEL is None:
        _XCEPTION_MODEL = Xception(weights="imagenet", include_top=False, pooling="avg")
    return _XCEPTION_MODEL

# -------------------------
# Grain extraction
# -------------------------

def extract_grains_from_image(
    image_path,
    padding=150,
    scale=1.6,
    parent_name=None,
    parent_hash=None,
):
    img_name = parent_name or os.path.basename(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return []

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Watershed to split touching grains.
    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(clean, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), markers)

    contours = []
    min_area = max(60, int(0.0001 * img.shape[0] * img.shape[1]))
    for label_id in np.unique(markers):
        if label_id <= 1:
            continue
        mask = np.uint8(markers == label_id) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area:
                continue
            contours.append(cnt)

    if not contours:
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    grains = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, img.shape[1])
        y2 = min(y + h + padding, img.shape[0])

        bw = x2 - x1
        bh = y2 - y1
        side = int(max(bw, bh) * scale)

        cx = x1 + bw // 2
        cy = y1 + bh // 2

        x1s = max(cx - side // 2, 0)
        y1s = max(cy - side // 2, 0)
        x2s = min(cx + side // 2, img.shape[1])
        y2s = min(cy + side // 2, img.shape[0])

        patch = img[y1s:y2s, x1s:x2s]

        if patch is None or patch.size == 0:
            continue

        if patch.shape[0] != 299 or patch.shape[1] != 299:
            patch = cv2.resize(patch, (299, 299))

        grains.append({
            "image": patch,
            "parent_image": img_name,
            "parent_hash": parent_hash,
            "bbox": (x1s, y1s, x2s, y2s)
        })

    return grains


def extract_grains(
    image_paths,
    padding=150,
    scale=1.6,
    use_parallel=False,
    max_workers=None,
    image_meta=None,
    error_log=None,
):
    all_grains = []

    def _process(path):
        meta = image_meta.get(path, {}) if image_meta else {}
        return extract_grains_from_image(
            path,
            padding=padding,
            scale=scale,
            parent_name=meta.get("name"),
            parent_hash=meta.get("hash"),
        )

    if use_parallel and len(image_paths) > 1:
        workers = max_workers or min(32, (os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process, p): p for p in image_paths}
            for future in tqdm(as_completed(futures), total=len(futures)):
                path = futures[future]
                try:
                    all_grains.extend(future.result())
                except Exception as exc:
                    if error_log is not None:
                        error_log.append({"image": path, "error": str(exc)})
    else:
        for p in tqdm(image_paths):
            try:
                all_grains.extend(_process(p))
            except Exception as exc:
                if error_log is not None:
                    error_log.append({"image": p, "error": str(exc)})
    return all_grains


# -------------------------
# Xception embeddings
# -------------------------

def compute_embeddings(grains, batch_size=32):
    model = _get_xception_model()

    if not grains:
        return np.array([])

    embeddings_chunks = []
    total = len(grains)
    for i in range(0, total, batch_size):
        batch = grains[i:i + batch_size]
        imgs = np.array([g["image"] for g in batch], dtype="float32")
        imgs = preprocess_input(imgs)
        batch_embeddings = model.predict(imgs, batch_size=len(batch), verbose=0)
        embeddings_chunks.append(batch_embeddings.reshape(batch_embeddings.shape[0], -1))

    return np.vstack(embeddings_chunks)


# -------------------------
# Clustering
# -------------------------

def cluster_grains(embeddings, n_clusters=20, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    return kmeans.fit_predict(embeddings)

