import json
import os
import sys
import traceback
import hashlib
import shutil
import glob
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

def _ensure_skimage():
    try:
        import skimage  # noqa: F401
        return True
    except Exception:
        try:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "scikit-image>=0.20.0",
            ])
            import skimage  # noqa: F401
            return True
        except Exception:
            return False


if not _ensure_skimage():
    raise RuntimeError(
        "Missing dependency: scikit-image. "
        "Install it with 'python -m pip install scikit-image>=0.20.0'."
    )


# Cellpose is optional - we have a fallback to thresholding
def _check_cellpose():
    try:
        import cellpose  # noqa: F401
        return True
    except Exception:
        print("[backend] Cellpose not installed - will use threshold segmentation as fallback.")
        return False

_CELLPOSE_AVAILABLE = _check_cellpose()

# Ensure repo module path is available (componentry_core.py lives in ComponentryAnalysis/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from componentry_core import extract_grains, cluster_grains, _get_xception_model
from tensorflow.keras.applications.xception import preprocess_input
import tensorflow as tf


STATE_DIR = os.path.join("results", "ui_state")
SAMPLES_DIR = os.path.join(STATE_DIR, "grain_samples")
GRAINS_PATH = os.path.join(STATE_DIR, "grains.npy")
EMBEDDINGS_PATH = os.path.join(STATE_DIR, "embeddings.npy")
EMBEDDING_INDEX_PATH = os.path.join(STATE_DIR, "embedding_indices.npy")
CLUSTERS_PATH = os.path.join(STATE_DIR, "clusters.npy")
META_PATH = os.path.join(STATE_DIR, "meta.json")
EXPORTS_DIR = os.path.join("results", "exports")
MODEL_PATH = os.path.join(STATE_DIR, "trained_model.keras")


def _ensure_state_dir():
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)


def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _hash_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _emit_progress(step, percent, message=None):
    payload = {"step": step, "percent": percent}
    if message:
        payload["message"] = message
    print(f"PROGRESS {json.dumps(payload)}", flush=True)


def _load_embedding_indices(expected_len=None):
    if os.path.exists(EMBEDDING_INDEX_PATH):
        indices = np.load(EMBEDDING_INDEX_PATH)
        if expected_len is None or len(indices) == int(expected_len):
            return indices.astype(int)
    if expected_len is None:
        return None
    return np.arange(int(expected_len), dtype=int)


def _parse_cluster_assignments(payload):
    provided = payload or {}
    parsed = {}
    if not isinstance(provided, dict):
        return parsed
    for k, v in provided.items():
        try:
            parsed[int(k)] = v
        except Exception:
            continue
    return parsed


def _relabel_contiguous(labels):
    unique = sorted(set(int(x) for x in labels.tolist()))
    mapping = {old: new for new, old in enumerate(unique)}
    relabeled = np.array([mapping[int(x)] for x in labels], dtype=int)
    return relabeled, mapping


def _build_cluster_samples(labels, seed=42):
    cluster_samples = []
    if not (os.path.exists(GRAINS_PATH) and cv2 is not None):
        return cluster_samples
    grains = np.load(GRAINS_PATH, allow_pickle=True).tolist()
    indices = _load_embedding_indices(expected_len=len(labels))
    rng = np.random.default_rng(int(seed))
    for cluster_id in sorted(set(int(x) for x in labels.tolist())):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        rep = int(rng.choice(idx))
        grain_idx = int(indices[rep]) if indices is not None and rep < len(indices) else rep
        grain = grains[grain_idx] if 0 <= grain_idx < len(grains) else None
        if grain is None:
            continue
        img = grain.get("preview")
        if img is None:
            img = grain.get("image")
        if img is None:
            continue
        path = os.path.join(SAMPLES_DIR, f"cluster_{cluster_id}.png")
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        cluster_samples.append({"cluster_id": int(cluster_id), "path": os.path.abspath(path)})
    return cluster_samples


def _ensure_exports_dir():
    os.makedirs(EXPORTS_DIR, exist_ok=True)


def _export_copy_file(src_path, dest_dir):
    if not src_path or not os.path.exists(src_path):
        return None
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dest_path)
    return dest_path


def _export_copy_glob(pattern, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    matches = []
    for path in sorted([p for p in glob.glob(pattern) if os.path.exists(p)]):
        dest_path = os.path.join(dest_dir, os.path.basename(path))
        shutil.copy2(path, dest_path)
        matches.append(dest_path)
    return matches


def _compute_componentry_from_labels(labels, assignments, classes):
    cluster_to_class = {}
    for cluster in sorted(set(labels.tolist())):
        if cluster in assignments:
            cluster_to_class[cluster] = assignments[cluster]
        else:
            cluster_to_class[cluster] = classes[int(cluster) % len(classes)]
    grain_labels = [cluster_to_class[c] for c in labels]
    counts = {}
    for label in grain_labels:
        counts[label] = counts.get(label, 0) + 1
    total = len(grain_labels) or 1
    return {k: round((v / total) * 100, 2) for k, v in counts.items()}


def _prepare_training_data(settings):
    if not os.path.exists(CLUSTERS_PATH):
        raise RuntimeError("No clusters found. Run clustering first.")
    if not os.path.exists(GRAINS_PATH):
        raise RuntimeError("No grains found. Run extraction first.")

    labels = np.load(CLUSTERS_PATH).astype(int)
    grains = np.load(GRAINS_PATH, allow_pickle=True).tolist()
    indices = _load_embedding_indices(expected_len=len(labels))
    if indices is None:
        indices = np.arange(len(labels), dtype=int)

    assignments = _parse_cluster_assignments(settings.get("clusterAssignments") or {})
    classes = settings.get("classes") or ["ash", "pumice", "crystalline", "lithic", "unknown"]
    classes = [c for c in classes if c]

    # Ensure any assignment labels are included in class list
    for value in assignments.values():
        if value not in classes:
            classes.append(value)
    if not classes:
        classes = ["unknown"]

    images = []
    targets = []
    for idx, cluster_id in enumerate(labels.tolist()):
        cluster_id = int(cluster_id)
        if cluster_id not in assignments:
            continue
        label_name = assignments[cluster_id]
        if label_name not in classes:
            continue
        grain_idx = int(indices[idx]) if idx < len(indices) else idx
        if grain_idx < 0 or grain_idx >= len(grains):
            continue
        img = grains[grain_idx].get("image")
        if img is None:
            continue
        images.append(img)
        targets.append(label_name)

    if not images:
        raise RuntimeError("No labeled grains found for training. Assign cluster labels first.")

    label_to_idx = {name: i for i, name in enumerate(classes)}
    y = np.array([label_to_idx[name] for name in targets], dtype=int)
    x = np.array(images, dtype="float32")
    x = preprocess_input(x)
    return x, y, classes


def _run_train(settings):
    _emit_progress("train", 5, "Preparing training data...")
    x, y, classes = _prepare_training_data(settings)
    num_classes = len(classes)
    seed = int(settings.get("seed", 42))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    split = max(1, int(len(x) * 0.8))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(x) else indices[:1]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    batch_size = int(settings.get("trainBatchSize", 32))
    epochs = int(settings.get("trainEpochs", 5))
    if epochs < 1:
        epochs = 1

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        base_model = tf.keras.applications.Xception(
            weights="imagenet", include_top=False, pooling="avg"
        )
        base_model.trainable = False
        inputs = tf.keras.Input(shape=(299, 299, 3))
        x_feat = base_model(inputs, training=False)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x_feat)
        model = tf.keras.Model(inputs, outputs)

    if model.output_shape[-1] != num_classes:
        # Rebuild head to match the requested classes
        base_model = tf.keras.applications.Xception(
            weights="imagenet", include_top=False, pooling="avg"
        )
        base_model.trainable = False
        inputs = tf.keras.Input(shape=(299, 299, 3))
        x_feat = base_model(inputs, training=False)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x_feat)
        model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Phase 1: train classifier head
    _emit_progress("train", 40, "Training classifier head...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=max(1, int(epochs * 0.6)),
        batch_size=batch_size,
        verbose=0,
    )

    # Phase 2: fine-tune last layers
    for layer in model.layers[-30:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    _emit_progress("train", 75, "Fine-tuning model...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=max(1, epochs - int(epochs * 0.6)),
        batch_size=batch_size,
        verbose=0,
    )

    _ensure_state_dir()
    model.save(MODEL_PATH)
    _emit_progress("train", 95, "Saving trained model...")

    meta = _load_json(META_PATH)
    trained_at = datetime.utcnow().isoformat() + "Z"
    meta["trained"] = {
        "trained_at": trained_at,
        "classes": classes,
        "samples": int(len(x)),
        "model_path": os.path.abspath(MODEL_PATH),
    }
    _save_json(META_PATH, meta)
    _emit_progress("train", 100, "Training complete.")
    return {
        "trained": True,
        "trained_at": trained_at,
        "class_labels": classes,
        "samples": int(len(x)),
        "model_path": os.path.abspath(MODEL_PATH),
    }


def _run_predict(settings):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("No trained model found. Run training first.")
    if not os.path.exists(GRAINS_PATH):
        raise RuntimeError("No grains found. Run extraction first.")

    _emit_progress("predict", 5, "Loading trained model...")
    meta = _load_json(META_PATH)
    trained = meta.get("trained") or {}
    classes = trained.get("classes") or settings.get("classes") or ["unknown"]
    classes = [c for c in classes if c] or ["unknown"]

    labels = np.load(CLUSTERS_PATH).astype(int) if os.path.exists(CLUSTERS_PATH) else None
    grains = np.load(GRAINS_PATH, allow_pickle=True).tolist()
    indices = None
    if labels is not None:
        indices = _load_embedding_indices(expected_len=len(labels))
    if indices is None:
        indices = np.arange(len(grains), dtype=int)

    images = []
    for idx in indices:
        if idx < 0 or idx >= len(grains):
            continue
        img = grains[int(idx)].get("image")
        if img is None:
            continue
        images.append(img)
    if not images:
        raise RuntimeError("No grains available for prediction.")

    x = np.array(images, dtype="float32")
    x = preprocess_input(x)
    model = tf.keras.models.load_model(MODEL_PATH)
    _emit_progress("predict", 60, "Running inference...")
    preds = model.predict(x, batch_size=int(settings.get("trainBatchSize", 32)), verbose=0)
    pred_idx = np.argmax(preds, axis=1)
    counts = {}
    for idx in pred_idx:
        label = classes[int(idx) % len(classes)]
        counts[label] = counts.get(label, 0) + 1
    total = len(pred_idx) or 1
    percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}
    _emit_progress("predict", 100, "Prediction complete.")
    return {"classes": percentages, "predicted_grains": len(pred_idx)}


def _run_extract(images, settings):
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for grain extraction.")
    if not images:
        raise RuntimeError("No images provided.")

    validation = settings.get("validation", {})
    max_files = int(validation.get("maxFiles", 200))
    max_file_size_mb = int(validation.get("maxFileSizeMb", 50))
    min_dim = int(validation.get("minDim", 200))
    min_focus = float(validation.get("minFocus", 30))
    max_total_pixels_mp = int(validation.get("maxTotalPixelsMp", 2000))

    image_meta = {}
    valid_paths = []
    total_pixels = 0
    max_total_pixels = max_total_pixels_mp * 1_000_000
    max_bytes = max_file_size_mb * 1024 * 1024
    for path in images[:max_files]:
        if not os.path.exists(path):
            continue
        if os.path.getsize(path) > max_bytes:
            continue
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        if total_pixels + (h * w) > max_total_pixels:
            continue
        if min(h, w) < min_dim:
            continue
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        focus = cv2.Laplacian(gray, cv2.CV_64F).var()
        if focus < min_focus:
            continue
        total_pixels += h * w
        file_hash = _hash_file(path)
        image_meta[path] = {"name": os.path.basename(path), "hash": file_hash}
        valid_paths.append(path)

    if not valid_paths:
        raise RuntimeError("No valid image paths found.")

    _emit_progress("extract", 5, "Preparing grain extraction...")

    def _progress(done, total, path):
        if total <= 0:
            return
        percent = 5 + int((done / total) * 80)
        name = os.path.basename(path) if path else f"{done}/{total}"
        _emit_progress("extract", percent, f"Extracting grains ({done}/{total}): {name}")

    def _stage_cb(stage, pct, msg):
        _emit_progress("extract", pct, msg)

    # Get model path for Cellpose
    model_path = settings.get("modelPath") or None

    grains = extract_grains(
        valid_paths,
        padding=int(settings.get("padding", 20)),
        scale=float(settings.get("scale", 1.4)),
        min_size=int(settings.get("minSize", 15)),
        edge_filter=float(settings.get("edgeFilter", 0.1)),
        min_axis_px=int(settings.get("minAxisPx", 8)),
        diameter=settings.get("diameter"),  # None = auto-detect
        model_path=model_path,
        use_gpu=bool(settings.get("useGpu", False)),
        use_parallel=False,  # Don't parallelize Cellpose - it manages GPU internally
        max_workers=int(settings.get("maxWorkers", 8)),
        image_meta=image_meta,
        error_log=[],
        progress_cb=_progress,
        stage_cb=_stage_cb,
    )
    _emit_progress("extract", 90, "Saving grain samples...")

    _ensure_state_dir()
    # Save a few grain previews
    sample_paths = []
    if cv2 is not None:
        for idx, grain in enumerate(grains[:12]):
            path = os.path.join(SAMPLES_DIR, f"grain_{idx}.png")
            img = grain.get("preview")
            if img is None:
                img = grain.get("image")
            if img is None:
                continue
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
            sample_paths.append(os.path.abspath(path))

    np.save(GRAINS_PATH, np.array(grains, dtype=object), allow_pickle=True)
    _save_json(
        META_PATH,
        {
            "images": valid_paths,
            "settings": settings,
            "grains": len(grains),
        },
    )
    _emit_progress("extract", 100, "Extraction complete.")
    return {"images": len(valid_paths), "grains": len(grains), "samples": sample_paths}


def _run_embeddings(settings):
    if not os.path.exists(GRAINS_PATH):
        raise RuntimeError("No grains found. Run extraction first.")
    grains_all = np.load(GRAINS_PATH, allow_pickle=True).tolist()
    batch_size = int(settings.get("batchSize", 32))
    max_grains = int(settings.get("maxGrains", 0) or 0)
    seed = int(settings.get("seed", 42))

    meta = _load_json(META_PATH)
    embed_meta = meta.get("embeddings")
    if embed_meta and os.path.exists(EMBEDDINGS_PATH):
        if (
            embed_meta.get("batch_size") == batch_size
            and embed_meta.get("max_grains") == max_grains
            and embed_meta.get("seed") == seed
        ):
            # Ensure index mapping exists; otherwise, recompute when we can't safely reconstruct it.
            if os.path.exists(EMBEDDING_INDEX_PATH):
                _emit_progress("embed", 100, "Loaded cached embeddings.")
                return {
                    "grains": embed_meta.get("grains", len(grains_all)),
                    "embeddings": embed_meta.get("embeddings", 0),
                    "cached": True,
                }
            if max_grains <= 0:
                _ensure_state_dir()
                np.save(EMBEDDING_INDEX_PATH, np.arange(len(grains_all), dtype=int))
                _emit_progress("embed", 100, "Loaded cached embeddings.")
                return {
                    "grains": embed_meta.get("grains", len(grains_all)),
                    "embeddings": embed_meta.get("embeddings", 0),
                    "cached": True,
                }
            # Cached subset embeddings without an index map are unsafe for cluster previews; recompute.

    indices = np.arange(len(grains_all), dtype=int)
    grains = grains_all
    if max_grains > 0 and len(grains_all) > max_grains:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(grains_all), size=max_grains, replace=False)
        grains = [grains_all[int(i)] for i in indices]

    model = _get_xception_model()

    embeddings_chunks = []
    total = len(grains)
    if total == 0:
        raise RuntimeError("No grains available for embeddings.")

    for i in range(0, total, batch_size):
        batch = grains[i : i + batch_size]
        imgs = np.array([g["image"] for g in batch], dtype="float32")
        imgs = preprocess_input(imgs)
        batch_embeddings = model.predict(imgs, batch_size=len(batch), verbose=0)
        embeddings_chunks.append(batch_embeddings.reshape(batch_embeddings.shape[0], -1))
        percent = int(((i + len(batch)) / total) * 100)
        _emit_progress("embed", percent, f"Embedding batch {i // batch_size + 1} / {int(np.ceil(total / batch_size))}")

    embeddings = np.vstack(embeddings_chunks)
    _ensure_state_dir()
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(EMBEDDING_INDEX_PATH, np.array(indices, dtype=int))
    meta["embeddings"] = {
        "batch_size": batch_size,
        "max_grains": max_grains,
        "seed": seed,
        "grains": len(grains_all),
        "embeddings": int(embeddings.shape[0]),
    }
    _save_json(META_PATH, meta)
    _emit_progress("embed", 100, "Embeddings complete.")
    return {"grains": len(grains), "embeddings": int(embeddings.shape[0])}


def _run_cluster(settings):
    if not os.path.exists(EMBEDDINGS_PATH):
        raise RuntimeError("No embeddings found. Run embeddings first.")
    embeddings = np.load(EMBEDDINGS_PATH)
    num_clusters = int(settings.get("numClusters", 20))
    seed = int(settings.get("seed", 42))
    _emit_progress("cluster", 20, "Clustering grains...")
    labels = cluster_grains(embeddings, n_clusters=num_clusters, random_state=seed)
    _ensure_state_dir()
    np.save(CLUSTERS_PATH, labels)
    cluster_samples = _build_cluster_samples(labels, seed=seed)
    class_labels = settings.get("classes") or ["ash", "pumice", "crystalline", "lithic", "unknown"]
    class_labels = [c for c in class_labels if c]
    if not class_labels:
        class_labels = ["unknown"]
    target_categories = int(settings.get("targetCategories") or 0)
    if target_categories > 0 and target_categories < len(class_labels):
        class_labels = class_labels[:target_categories]
    cluster_assignments = {int(cluster_id): "to choose" for cluster_id in range(num_clusters)}
    _emit_progress("cluster", 100, "Clustering complete.")
    return {
        "clusters": num_clusters,
        "labels": len(labels),
        "cluster_samples": cluster_samples,
        "cluster_assignments": cluster_assignments,
        "class_labels": class_labels,
    }


def _run_componentry(settings):
    if not os.path.exists(CLUSTERS_PATH):
        raise RuntimeError("No clusters found. Run clustering first.")
    labels = np.load(CLUSTERS_PATH)
    classes = settings.get("classes") or ["unknown"]
    classes = [c for c in classes if c]
    if not classes:
        classes = ["unknown"]
    target_categories = int(settings.get("targetCategories") or 0)
    if target_categories > 0 and target_categories < len(classes):
        classes = classes[:target_categories]
    background_class = str(settings.get("backgroundClass") or "background").strip().lower()
    ignore_background = bool(settings.get("ignoreBackground", True))

    _emit_progress("componentry", 30, "Assigning cluster classes...")
    cluster_to_class = {}
    provided = settings.get("clusterAssignments") or {}
    for cluster in sorted(set(labels.tolist())):
        if str(cluster) in provided:
            cluster_to_class[cluster] = provided[str(cluster)]
        elif cluster in provided:
            cluster_to_class[cluster] = provided[cluster]
        else:
            cluster_to_class[cluster] = "to choose"

    grain_labels = [cluster_to_class[c] for c in labels]
    counts = {}
    for label in grain_labels:
        normalized = str(label).strip()
        if ignore_background and normalized.lower() == background_class:
            continue
        if ignore_background and normalized.lower() == "to choose":
            continue
        counts[normalized] = counts.get(normalized, 0) + 1

    total = sum(counts.values()) or 1
    percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}
    _emit_progress("componentry", 100, "Componentry complete.")
    return {"classes": percentages, "total_grains": len(grain_labels)}


def _run_export_page(payload, settings):
    page = payload.get("page")
    if not page:
        raise RuntimeError("export_page requires a page id.")
    page = str(page).lower()
    if page not in {"grains", "embeddings", "clusters", "results"}:
        raise RuntimeError(f"Unknown export page: {page}")

    export_base_dir = payload.get("exportBaseDir") or payload.get("export_base_dir") or None
    if export_base_dir:
        export_base_dir = os.path.abspath(export_base_dir)
    else:
        _ensure_exports_dir()
        export_base_dir = EXPORTS_DIR

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(export_base_dir, f"{page}_{timestamp}")
    os.makedirs(export_dir, exist_ok=True)

    class_labels = settings.get("classes") or ["ash", "pumice", "crystalline", "lithic", "unknown"]
    class_labels = [c for c in class_labels if c] or ["unknown"]
    assignments = _parse_cluster_assignments(settings.get("clusterAssignments") or {})

    copied = {}
    if os.path.exists(META_PATH):
        copied["meta"] = _export_copy_file(META_PATH, export_dir)

    if page in {"grains", "embeddings"}:
        copied["grain_samples"] = _export_copy_glob(
            os.path.join(SAMPLES_DIR, "grain_*.png"),
            os.path.join(export_dir, "grain_samples"),
        )

    if page == "grains":
        copied["grains"] = _export_copy_file(GRAINS_PATH, export_dir)

    if page == "embeddings":
        copied["embeddings"] = _export_copy_file(EMBEDDINGS_PATH, export_dir)
        copied["embedding_indices"] = _export_copy_file(EMBEDDING_INDEX_PATH, export_dir)

    if page in {"clusters", "results"}:
        copied["cluster_samples"] = _export_copy_glob(
            os.path.join(SAMPLES_DIR, "cluster_*.png"),
            os.path.join(export_dir, "cluster_samples"),
        )
        copied["clusters"] = _export_copy_file(CLUSTERS_PATH, export_dir)

    export_meta = {
        "page": page,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "class_labels": class_labels,
        "cluster_assignments": assignments,
        "settings": settings,
    }

    if page == "results" and os.path.exists(CLUSTERS_PATH):
        labels = np.load(CLUSTERS_PATH)
        export_meta["componentry"] = _compute_componentry_from_labels(labels, assignments, class_labels)

    _save_json(os.path.join(export_dir, "export_meta.json"), export_meta)
    return {"page": page, "path": os.path.abspath(export_dir), "files": copied}


def _run_merge(payload, settings):
    if not os.path.exists(CLUSTERS_PATH):
        raise RuntimeError("No clusters found. Run clustering first.")
    to_merge = payload.get("clustersToMerge") or payload.get("clusters_to_merge") or []
    if not isinstance(to_merge, list):
        raise RuntimeError("clustersToMerge must be a list of cluster ids.")
    to_merge = sorted({int(x) for x in to_merge})
    if len(to_merge) < 2:
        raise RuntimeError("Select at least two clusters to merge.")

    labels = np.load(CLUSTERS_PATH).astype(int)
    present = set(int(x) for x in labels.tolist())
    missing = [c for c in to_merge if c not in present]
    if missing:
        raise RuntimeError(f"Clusters not found: {missing}")

    target = int(min(to_merge))
    _emit_progress("merge", 30, f"Merging clusters into {target}...")
    merged = labels.copy()
    for cid in to_merge:
        merged[labels == cid] = target

    merged, relabel_map = _relabel_contiguous(merged)
    np.save(CLUSTERS_PATH, merged)

    class_labels = settings.get("classes") or ["ash", "pumice", "crystalline", "lithic", "unknown"]
    class_labels = [c for c in class_labels if c] or ["unknown"]

    provided = _parse_cluster_assignments(settings.get("clusterAssignments") or {})
    new_assignments = {}
    # Invert relabel_map: old_id -> new_id
    for old_id, new_id in relabel_map.items():
        if old_id in to_merge:
            # Use the target cluster's label if possible, otherwise first merged cluster label, otherwise fallback.
            chosen = (
                provided.get(target)
                or provided.get(to_merge[0])
                or class_labels[int(new_id) % len(class_labels)]
            )
            new_assignments[int(new_id)] = chosen
        else:
            new_assignments[int(new_id)] = provided.get(old_id) or class_labels[int(new_id) % len(class_labels)]

    cluster_samples = _build_cluster_samples(merged, seed=int(settings.get("seed", 42)))
    new_cluster_ids = []
    if target in relabel_map:
        new_cluster_ids = [int(relabel_map[target])]
    _emit_progress("merge", 100, "Merge complete.")
    return {
        "clusters": int(len(set(int(x) for x in merged.tolist()))),
        "labels": int(len(merged)),
        "cluster_samples": cluster_samples,
        "cluster_assignments": new_assignments,
        "class_labels": class_labels,
        "new_cluster_ids": new_cluster_ids,
    }


def _run_split(payload, settings):
    if not os.path.exists(CLUSTERS_PATH):
        raise RuntimeError("No clusters found. Run clustering first.")
    if not os.path.exists(EMBEDDINGS_PATH):
        raise RuntimeError("No embeddings found. Run embeddings first.")

    cluster_id = payload.get("clusterToSplit") if "clusterToSplit" in payload else payload.get("cluster_to_split")
    if cluster_id is None:
        raise RuntimeError("clusterToSplit is required.")
    cluster_id = int(cluster_id)

    k = payload.get("numSplits") if "numSplits" in payload else payload.get("num_splits")
    k = int(k or 2)
    if k < 2 or k > 10:
        raise RuntimeError("numSplits must be between 2 and 10.")

    seed = int(settings.get("seed", 42))
    labels = np.load(CLUSTERS_PATH).astype(int)
    embeddings = np.load(EMBEDDINGS_PATH)
    if len(labels) != len(embeddings):
        raise RuntimeError("Embeddings/cluster labels length mismatch. Re-run clustering.")

    idx = np.where(labels == cluster_id)[0]
    if len(idx) < k:
        raise RuntimeError(f"Cluster {cluster_id} has only {len(idx)} items; cannot split into {k}.")

    _emit_progress("split", 30, f"Splitting cluster {cluster_id} into {k}...")
    sub_labels = cluster_grains(embeddings[idx], n_clusters=k, random_state=seed)

    # Assign new cluster ids for split parts
    max_id = int(max(int(x) for x in labels.tolist()))
    updated = labels.copy()
    for part in range(k):
        new_id = cluster_id if part == 0 else max_id + part
        updated[idx[sub_labels == part]] = int(new_id)

    updated, relabel_map = _relabel_contiguous(updated)
    np.save(CLUSTERS_PATH, updated)

    class_labels = settings.get("classes") or ["ash", "pumice", "crystalline", "lithic", "unknown"]
    class_labels = [c for c in class_labels if c] or ["unknown"]

    provided = _parse_cluster_assignments(settings.get("clusterAssignments") or {})
    old_label_for_split = provided.get(cluster_id)
    new_assignments = {}
    for old_id, new_id in relabel_map.items():
        if old_id == cluster_id or old_id > max_id:
            # All split-derived clusters inherit original class label (if any) by default.
            new_assignments[int(new_id)] = old_label_for_split or class_labels[int(new_id) % len(class_labels)]
        else:
            new_assignments[int(new_id)] = provided.get(old_id) or class_labels[int(new_id) % len(class_labels)]

    cluster_samples = _build_cluster_samples(updated, seed=seed)
    split_old_ids = [cluster_id] + [max_id + part for part in range(1, k)]
    new_cluster_ids = [int(relabel_map[old_id]) for old_id in split_old_ids if old_id in relabel_map]
    _emit_progress("split", 100, "Split complete.")
    return {
        "clusters": int(len(set(int(x) for x in updated.tolist()))),
        "labels": int(len(updated)),
        "cluster_samples": cluster_samples,
        "cluster_assignments": new_assignments,
        "class_labels": class_labels,
        "new_cluster_ids": new_cluster_ids,
    }


def main():
    try:
        payload = json.load(sys.stdin)
        step = payload.get("step")
        settings = payload.get("settings", {})
        if "clusterAssignments" in payload:
            settings["clusterAssignments"] = payload.get("clusterAssignments")
        images = payload.get("images", [])

        if step == "extract":
            data = _run_extract(images, settings)
        elif step == "embed":
            data = _run_embeddings(settings)
        elif step == "cluster":
            data = _run_cluster(settings)
        elif step == "merge":
            data = _run_merge(payload, settings)
        elif step == "split":
            data = _run_split(payload, settings)
        elif step == "train":
            data = _run_train(settings)
        elif step == "predict":
            data = _run_predict(settings)
        elif step == "export_page":
            data = _run_export_page(payload, settings)
        elif step == "componentry":
            data = _run_componentry(settings)
        elif step == "reset":
            if os.path.exists(STATE_DIR):
                for root, _dirs, files in os.walk(STATE_DIR):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except OSError:
                            pass
            data = {"reset": True}
        else:
            raise RuntimeError("Unknown step requested.")

        print(json.dumps({"ok": True, "data": data}))
    except Exception as exc:
        error = f"{exc}"
        details = traceback.format_exc()
        print(json.dumps({"ok": False, "error": error, "details": details}))


if __name__ == "__main__":
    main()
