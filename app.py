"""Streamlit app for ash grain componentry analysis."""

import hashlib
import json
import os
import tempfile
import time

import streamlit as st
import numpy as np
import pandas as pd
try:
    import cv2
except ImportError:
    cv2 = None
from componentry_core import extract_grains, compute_embeddings, cluster_grains

st.set_page_config(page_title="Ash Grain Componentry Analyzer", layout="wide")

st.title("🌋 Ash Grain Componentry Analyzer")


def _compute_run_id(image_hashes_value, padding_value, scale_value, num_clusters_value, seed_value):
    hasher = hashlib.sha256()
    for name, digest in sorted(image_hashes_value.items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(digest.encode("utf-8"))
    hasher.update(
        f"p{padding_value}_s{scale_value}_c{num_clusters_value}_seed{seed_value}".encode("utf-8")
    )
    return hasher.hexdigest()[:16]


def _log_event(run_id_value, stage, payload):
    os.makedirs("results", exist_ok=True)
    record = {"run_id": run_id_value, "stage": stage, **payload}
    with open(os.path.join("results", "run_log.jsonl"), "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")

# Sidebar controls
st.sidebar.header("Settings")
padding = st.sidebar.slider("Padding (crop margin)", 50, 400, 150)
scale = st.sidebar.slider("Crop scale factor", 1.0, 3.0, 1.6)
num_clusters = st.sidebar.slider("Number of clusters", 5, 50, 20)
seed = st.sidebar.number_input("Random seed", 0, 2**31 - 1, 42)
batch_size = st.sidebar.selectbox("Embedding batch size", [16, 32, 64, 128], index=1)
use_parallel = st.sidebar.checkbox("Parallel extraction", value=True)
max_workers_default = min(8, (os.cpu_count() or 1))
max_workers = st.sidebar.number_input("Max extraction workers", 1, 64, max_workers_default)
st.sidebar.subheader("Input validation")
max_files = st.sidebar.number_input("Max files per run", 1, 1000, 200)
max_file_size_mb = st.sidebar.number_input("Max file size (MB)", 1, 500, 50)
min_dim = st.sidebar.number_input("Min image dimension (px)", 64, 5000, 200)
min_focus = st.sidebar.number_input("Min focus (Laplacian var)", 0, 500, 30)
max_total_pixels_mp = st.sidebar.number_input("Max total pixels (MP)", 10, 100000, 2000)

uploaded_files = st.file_uploader(
    "Upload microscopy images (.tif, .png, .jpg)",
    accept_multiple_files=True,
    type=["tif", "tiff", "png", "jpg"]
)

if uploaded_files:
    if cv2 is None:
        st.error("OpenCV (cv2) is required to read images. Please install it and retry.")
        st.stop()
    if len(uploaded_files) > max_files:
        st.warning(f"Limiting to first {max_files} files.")
        uploaded_files = uploaded_files[:max_files]

    st.success(f"{len(uploaded_files)} images uploaded.")

    # Save files temporarily
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    skipped = []
    image_hashes = {}
    image_meta = {}
    total_pixels = 0
    max_total_pixels = int(max_total_pixels_mp) * 1_000_000

    max_bytes = int(max_file_size_mb) * 1024 * 1024
    for f in uploaded_files:
        data = f.getbuffer()
        if not data or len(data) == 0:
            skipped.append((f.name, "empty file"))
            continue
        if len(data) > max_bytes:
            skipped.append((f.name, f"file too large (> {max_file_size_mb} MB)"))
            continue

        img = cv2.imdecode(  # pylint: disable=no-member
            np.frombuffer(data, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if img is None:
            skipped.append((f.name, "unreadable image"))
            continue

        h, w = img.shape[:2]
        if total_pixels + (h * w) > max_total_pixels:
            skipped.append((f.name, "total pixel cap exceeded"))
            continue
        if min(h, w) < int(min_dim):
            skipped.append((f.name, f"too small (< {min_dim}px)"))
            continue

        gray = img if len(img.shape) == 2 else cv2.cvtColor(  # pylint: disable=no-member
            img,
            cv2.COLOR_BGR2GRAY,
        )
        focus = cv2.Laplacian(gray, cv2.CV_64F).var()  # pylint: disable=no-member
        if focus < float(min_focus):
            skipped.append((f.name, "low focus/blur"))
            continue

        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as out:
            out.write(data.tobytes())
        file_hash = hashlib.sha256(data).hexdigest()
        image_paths.append(path)
        total_pixels += h * w
        image_hashes[f.name] = file_hash
        image_meta[path] = {"name": f.name, "hash": file_hash}

    if skipped:
        st.warning("Some files were skipped due to validation:")
        st.write(pd.DataFrame(skipped, columns=["file", "reason"]))

    if not image_paths:
        st.error("No valid images to process after validation.")
        st.stop()

    if st.button("🔍 Extract grains"):
        np.random.seed(int(seed))
        run_id = _compute_run_id(image_hashes, padding, scale, num_clusters, seed)
        t0 = time.perf_counter()
        errors = []
        with st.spinner("Extracting grains..."):
            grains = extract_grains(
                image_paths,
                padding=padding,
                scale=scale,
                use_parallel=use_parallel,
                max_workers=int(max_workers),
                image_meta=image_meta,
                error_log=errors,
            )
        duration = time.perf_counter() - t0

        _log_event(run_id, "extract_grains", {
            "images": len(image_paths),
            "grains": len(grains),
            "duration_sec": round(duration, 4),
            "errors": len(errors),
        })

        st.session_state["grains"] = grains
        st.session_state["image_paths"] = image_paths
        st.session_state["run_id"] = run_id
        st.session_state["image_hashes"] = image_hashes
        st.session_state["image_meta"] = image_meta
        st.write(f"Total grains extracted: {len(grains)}")
        if errors:
            st.warning("Some images failed during extraction:")
            st.write(pd.DataFrame(errors))

        # Show sample grains
        sample = [g.get("preview", g["image"]) for g in grains[:20]]
        st.image(sample, width=120, caption=[f"Grain {i}" for i in range(len(sample))])

if "grains" in st.session_state and st.button("🧠 Compute embeddings"):
    os.makedirs("embeddings", exist_ok=True)
    run_id = st.session_state.get("run_id")
    cache_path = os.path.join("embeddings", f"embeddings_{run_id}.npy") if run_id else None
    by_image_dir = os.path.join("embeddings", "by_image")
    os.makedirs(by_image_dir, exist_ok=True)

    if cache_path and os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        st.info("Loaded cached embeddings.")
    else:
        grains = st.session_state["grains"]
        index_map = {i: g for i, g in enumerate(grains)}
        by_hash = {}
        for idx, grain in index_map.items():
            key = grain.get("parent_hash")
            if key:
                by_hash.setdefault(key, []).append(idx)
        embeddings_by_index = [None] * len(grains)

        t0 = time.perf_counter()
        cache_hits = 0
        cache_misses = 0
        with st.spinner("Running Xception..."):
            for key, indices in by_hash.items():
                cache_key = f"{key}_p{padding}_s{scale}.npy"
                cache_file = os.path.join(by_image_dir, cache_key)
                if os.path.exists(cache_file):
                    cached = np.load(cache_file)
                    if cached.shape[0] == len(indices):
                        for local_idx, grain_idx in enumerate(indices):
                            embeddings_by_index[grain_idx] = cached[local_idx]
                        cache_hits += 1
                        continue

                subset = [grains[i] for i in indices]
                subset_embeddings = compute_embeddings(subset, batch_size=int(batch_size))
                for local_idx, grain_idx in enumerate(indices):
                    embeddings_by_index[grain_idx] = subset_embeddings[local_idx]
                np.save(cache_file, subset_embeddings)
                cache_misses += 1

            # Fallback for grains missing a hash
            missing = [i for i, v in enumerate(embeddings_by_index) if v is None]
            if missing:
                subset = [grains[i] for i in missing]
                subset_embeddings = compute_embeddings(subset, batch_size=int(batch_size))
                for local_idx, grain_idx in enumerate(missing):
                    embeddings_by_index[grain_idx] = subset_embeddings[local_idx]

        embeddings = np.stack(embeddings_by_index, axis=0)
        duration = time.perf_counter() - t0
        _log_event(run_id, "compute_embeddings", {
            "grains": len(grains),
            "duration_sec": round(duration, 4),
            "batch_size": int(batch_size),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        })

        if cache_path:
            np.save(cache_path, embeddings)
            st.info("Embeddings cached for reuse.")

    st.session_state["embeddings"] = embeddings
    st.success(f"Embeddings computed: {embeddings.shape}")

if "embeddings" in st.session_state and st.button("🧩 Cluster grains"):
    np.random.seed(int(seed))
    t0 = time.perf_counter()
    labels = cluster_grains(st.session_state["embeddings"], num_clusters, random_state=int(seed))
    duration = time.perf_counter() - t0
    st.session_state["clusters"] = labels
    st.success("Clustering complete!")
    _log_event(st.session_state.get("run_id"), "cluster_grains", {
        "clusters": int(num_clusters),
        "duration_sec": round(duration, 4),
    })

    # Show representative grains per cluster
    st.subheader("Cluster representatives")
    rng = np.random.default_rng(int(seed))
    for c in range(num_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        rep = rng.choice(idx)
        st.image(st.session_state["grains"][rep]["image"], width=120)
        st.write(f"Cluster {c}")

if "clusters" in st.session_state:
    st.subheader("Assign labels to clusters")

    cluster_to_class = {}
    classes = ["ash", "pumice", "crystalline", "lithic", "unknown"]

    for c in range(num_clusters):
        cluster_to_class[c] = st.selectbox(f"Cluster {c}", classes, key=f"cluster_{c}")

    if st.button("📊 Compute componentry"):
        t0 = time.perf_counter()
        labels = st.session_state["clusters"]
        grain_labels = [cluster_to_class[c] for c in labels]

        df = pd.DataFrame({"class": grain_labels})
        percentages = df["class"].value_counts(normalize=True) * 100

        st.subheader("Componentry results (%)")
        st.dataframe(percentages)

        st.download_button(
            "Download results (CSV)",
            df.to_csv(index=False),
            "grain_predictions.csv",
            "text/csv"
        )

        duration = time.perf_counter() - t0
        _log_event(st.session_state.get("run_id"), "compute_componentry", {
            "grains": len(grain_labels),
            "duration_sec": round(duration, 4),
        })

