# componentry_core.py

import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.cluster import KMeans

# Import our grain segmenter (uses Cellpose like ImageGrains)
from grain_segmenter import (
    segment_and_extract,
    create_overlay_imagegrains_style,
    get_default_model_path,
)

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
    padding=20,
    scale=1.0,
    min_size=15,
    edge_filter=0.1,
    min_axis_px=8,
    diameter=None,
    model_path=None,
    use_gpu=False,
    parent_name=None,
    parent_hash=None,
    stage_cb=None,
):
    """
    Extract grains from an image using Cellpose (like ImageGrains).
    """
    img_name = parent_name or os.path.basename(image_path)
    
    if stage_cb:
        stage_cb("loading", 0, "Loading image...")
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return []

    # Convert to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get model path
    if model_path is None:
        model_path = get_default_model_path()

    try:
        # Run Cellpose segmentation + filtering + extraction
        grains, labels, num_grains = segment_and_extract(
            img,
            model_path=model_path,
            diameter=diameter,
            min_size=int(min_size),
            edge_filter=float(edge_filter),
            min_axis_px=int(min_axis_px),
            padding=int(padding),
            scale=float(scale),
            use_gpu=use_gpu,
            stage_cb=stage_cb,
        )
    except Exception as e:
        print(f"[componentry_core] Segmentation error: {e}")
        return []

    # Add parent info to each grain
    for grain in grains:
        grain["parent_image"] = img_name
        grain["parent_hash"] = parent_hash

    if stage_cb:
        stage_cb("done", 95, f"Extracted {len(grains)} grains.")

    return grains


def extract_grains(
    image_paths,
    padding=20,
    scale=1.0,
    min_size=15,
    edge_filter=0.1,
    min_axis_px=8,
    diameter=None,
    model_path=None,
    use_gpu=False,
    use_parallel=False,
    max_workers=None,
    image_meta=None,
    error_log=None,
    progress_cb=None,
    stage_cb=None,
):
    """
    Extract grains from multiple images using Cellpose.
    """
    all_grains = []
    total = len(image_paths)
    completed = 0

    def _process(path, img_stage_cb=None):
        meta = image_meta.get(path, {}) if image_meta else {}
        return extract_grains_from_image(
            path,
            padding=padding,
            scale=scale,
            min_size=min_size,
            edge_filter=edge_filter,
            min_axis_px=min_axis_px,
            diameter=diameter,
            model_path=model_path,
            use_gpu=use_gpu,
            parent_name=meta.get("name"),
            parent_hash=meta.get("hash"),
            stage_cb=img_stage_cb,
        )

    # Don't use parallel for Cellpose - it manages GPU internally
    def _make_stage_cb(img_idx):
        def _img_stage_cb(stage, pct, msg):
            if stage_cb:
                base = int((img_idx / total) * 85) + 5
                img_pct = int((pct / 100) * (85 / total))
                overall_pct = min(90, base + img_pct)
                stage_cb(stage, overall_pct, msg)
        return _img_stage_cb

    for img_idx, p in enumerate(tqdm(image_paths)):
        try:
            all_grains.extend(_process(p, _make_stage_cb(img_idx)))
        except Exception as exc:
            print(f"[componentry_core] Error processing {p}: {exc}")
            if error_log is not None:
                error_log.append({"image": p, "error": str(exc)})
        completed += 1
        if progress_cb:
            progress_cb(completed, total, p)

    return all_grains


def generate_overlay_preview(image_path, min_size=15, edge_filter=0.1, min_axis_px=8,
                              diameter=None, model_path=None, use_gpu=False):
    """
    Generate an ImageGrains-style overlay preview showing detected grains.
    """
    from grain_segmenter import segment_with_cellpose, filter_grains
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, 0

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if model_path is None:
        model_path = get_default_model_path()

    labels, num_raw = segment_with_cellpose(
        img, model_path=model_path, diameter=diameter, min_size=min_size, use_gpu=use_gpu
    )
    
    labels, num_grains = filter_grains(labels, edge_filter=edge_filter, min_axis_px=min_axis_px)
    
    overlay = create_overlay_imagegrains_style(img, labels)
    
    return overlay, num_grains


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
