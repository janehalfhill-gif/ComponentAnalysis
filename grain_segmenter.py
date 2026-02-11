# grain_segmenter.py
# Grain segmentation using Cellpose (like ImageGrains)
# Creates the same visualization style with blue boundaries

import cv2
import numpy as np
import os

# Try to import Cellpose
_CELLPOSE_AVAILABLE = False
_cellpose_model = None

try:
    from cellpose import models
    _CELLPOSE_AVAILABLE = True
except ImportError:
    pass

# Try to import skimage for visualization
try:
    from skimage.segmentation import mark_boundaries
    from skimage.color import label2rgb
    from skimage.measure import label, regionprops, regionprops_table
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


def get_default_model_path():
    """Get path to the ImageGrains model if available."""
    # Check common locations
    possible_paths = [
        os.path.expanduser("~/imagegrains/models/IG2_full_set_cp_SAM"),
        os.path.expanduser("~/imagegrains/models/IG2_full_set.200525"),
        os.path.expanduser("~/Downloads/imagegrains-main/imagegrains-main/models/full_set_1.170223"),
        "C:/Users/itsvi/Downloads/imagegrains-main/imagegrains-main/models/full_set_1.170223",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def segment_with_cellpose(img, model_path=None, diameter=None, min_size=15, use_gpu=False):
    """
    Segment grains using Cellpose (like ImageGrains).
    Falls back to simple thresholding if Cellpose fails.
    """
    global _cellpose_model
    
    # Try Cellpose first
    if _CELLPOSE_AVAILABLE:
        try:
            if model_path is None:
                model_path = get_default_model_path()
            
            if model_path and os.path.exists(model_path):
                # Load model (cache it)
                if _cellpose_model is None or getattr(_cellpose_model, '_model_path', None) != model_path:
                    print(f"[grain_segmenter] Loading Cellpose model from: {model_path}")
                    _cellpose_model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
                    _cellpose_model._model_path = model_path
                
                # Run segmentation
                masks, flows, styles = _cellpose_model.eval(
                    [img],
                    diameter=diameter,
                    min_size=int(min_size),
                    channels=None,
                )
                
                if masks and masks[0] is not None:
                    labels = masks[0]
                    num_grains = len(np.unique(labels)) - 1
                    if num_grains > 0:
                        print(f"[grain_segmenter] Cellpose found {num_grains} grains")
                        return labels.astype(np.int32), num_grains
        except Exception as e:
            print(f"[grain_segmenter] Cellpose failed: {e}, falling back to thresholding...")
    
    # Fallback: Simple thresholding (works well for grains on white/light background)
    print("[grain_segmenter] Using simple thresholding segmentation...")
    return segment_with_threshold(img, min_size=min_size)


def segment_with_threshold(img, min_size=15):
    """
    Simple but effective segmentation using Otsu thresholding.
    Works well for grains on uniform background.
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu thresholding - automatically finds optimal threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Watershed to split touching grains
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Need BGR for watershed
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)
    
    # Create clean labels
    labels = np.zeros_like(markers, dtype=np.int32)
    label_id = 1
    for region_id in np.unique(markers):
        if region_id <= 1:  # Skip background and boundaries
            continue
        mask = (markers == region_id)
        area = np.sum(mask)
        if area >= min_size:
            labels[mask] = label_id
            label_id += 1
    
    num_grains = label_id - 1
    print(f"[grain_segmenter] Threshold segmentation found {num_grains} grains")
    return labels, num_grains


def filter_grains(labels, edge_filter=0.1, min_axis_px=8):
    """
    Filter grains by edge location and minimum size (like ImageGrains).
    
    Parameters:
    -----------
    labels : ndarray
        Labeled mask
    edge_filter : float
        Fraction of image edge to exclude (0.0-0.5)
    min_axis_px : int
        Minimum minor axis length in pixels
        
    Returns:
    --------
    filtered_labels : ndarray
        Filtered labeled mask
    num_grains : int
        Number of grains after filtering
    """
    if not _SKIMAGE_AVAILABLE:
        return labels, len(np.unique(labels)) - 1
    
    h, w = labels.shape
    edge_y = int(h * edge_filter)
    edge_x = int(w * edge_filter)
    
    # Get region properties
    props = regionprops(labels)
    
    # Create filtered mask
    filtered = np.zeros_like(labels)
    new_label = 1
    
    for prop in props:
        # Check centroid location (edge filter)
        cy, cx = prop.centroid
        if edge_filter > 0:
            if cy < edge_y or cy > (h - edge_y):
                continue
            if cx < edge_x or cx > (w - edge_x):
                continue
        
        # Check minimum axis length
        if prop.minor_axis_length < min_axis_px:
            continue
        
        # Keep this grain
        filtered[labels == prop.label] = new_label
        new_label += 1
    
    return filtered, new_label - 1


def create_overlay_imagegrains_style(img, labels, alpha=0.5, boundary_color=(0, 0, 1)):
    """
    Create an overlay exactly like ImageGrains does.
    Uses skimage.segmentation.mark_boundaries and skimage.color.label2rgb.
    
    Parameters:
    -----------
    img : ndarray
        Original image (RGB)
    labels : ndarray
        Labeled mask
    alpha : float
        Transparency of the colored overlay
    boundary_color : tuple
        RGB color for boundaries (default: blue)
        
    Returns:
    --------
    overlay : ndarray
        RGB overlay image (uint8)
    """
    if not _SKIMAGE_AVAILABLE:
        # Fallback: simple boundary drawing
        return create_overlay_simple(img, labels)
    
    # Generate random colors for each grain (like ImageGrains mask_cmap)
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    
    # Create colormap
    rng = np.random.default_rng(42)
    colors = []
    cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO)
    for i in range(num_labels):
        idx = int((i / max(1, num_labels - 1)) * 255) if num_labels > 1 else 128
        c = cmap[idx, 0] / 255.0  # Normalize to 0-1
        colors.append((c[2], c[1], c[0]))  # BGR to RGB
    rng.shuffle(colors)
    
    # Create colored label overlay
    colored = label2rgb(labels, image=img, colors=colors, bg_label=0, alpha=alpha, bg_color='black')
    
    # Add boundaries
    overlay = mark_boundaries(colored, labels, color=boundary_color, mode='thick')
    
    # Convert to uint8
    overlay = (overlay * 255).astype(np.uint8)
    
    return overlay


def create_overlay_simple(img, labels, boundary_color=(255, 100, 0)):
    """
    Simple overlay with boundaries and centroids (fallback if skimage not available).
    """
    if len(img.shape) == 2:
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        overlay = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        
        mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, boundary_color, 1)
        
        # Draw centroid
        ys, xs = np.where(labels == label_id)
        if len(ys) > 0:
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            cv2.circle(overlay, (cx, cy), 2, (255, 255, 255), -1)
    
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def extract_grain_patches(img, labels, padding=20, scale=1.0, target_size=(299, 299)):
    """
    Extract individual grain patches from the image based on labels.
    """
    grains = []
    
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        
        ys, xs = np.where(labels == label_id)
        if len(ys) == 0:
            continue
        
        area = len(ys)
        
        # Bounding box
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        h = y_max - y_min + 1
        w = x_max - x_min + 1
        
        # Add padding
        x1 = max(x_min - padding, 0)
        y1 = max(y_min - padding, 0)
        x2 = min(x_max + padding, img.shape[1])
        y2 = min(y_max + padding, img.shape[0])
        
        # Scale
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
        
        if patch is None or patch.size == 0 or patch.shape[0] < 10 or patch.shape[1] < 10:
            continue
        
        patch = cv2.resize(patch, target_size)
        
        # Create preview with boundary
        preview = patch.copy()
        try:
            mask_patch = (labels[y1s:y2s, x1s:x2s] == label_id).astype(np.uint8) * 255
            mask_patch = cv2.resize(mask_patch, target_size, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw green outline
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            cv2.drawContours(preview_bgr, contours, -1, (0, 255, 100), 2)
            preview = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            preview = patch
        
        grains.append({
            "image": patch,
            "preview": preview,
            "bbox": (x1s, y1s, x2s, y2s),
            "area": area,
            "label_id": int(label_id),
        })
    
    return grains


def segment_and_extract(img, model_path=None, diameter=None, min_size=15, 
                        edge_filter=0.1, min_axis_px=8, padding=20, scale=1.0,
                        use_gpu=False, stage_cb=None):
    """
    Complete pipeline: segment with Cellpose, filter, and extract patches.
    
    Returns:
    --------
    grains : list
        List of grain dicts with 'image', 'preview', etc.
    labels : ndarray
        Labeled mask
    num_grains : int
        Number of grains
    """
    if stage_cb:
        stage_cb("segment", 10, "Running Cellpose segmentation...")
    
    labels, num_raw = segment_with_cellpose(
        img, 
        model_path=model_path, 
        diameter=diameter, 
        min_size=min_size, 
        use_gpu=use_gpu
    )
    
    if stage_cb:
        stage_cb("filter", 40, f"Filtering {num_raw} grains...")
    
    labels, num_grains = filter_grains(labels, edge_filter=edge_filter, min_axis_px=min_axis_px)
    
    if stage_cb:
        stage_cb("extract", 60, f"Extracting {num_grains} grain patches...")
    
    grains = extract_grain_patches(img, labels, padding=padding, scale=scale)
    
    if stage_cb:
        stage_cb("done", 95, f"Done: {len(grains)} grains")
    
    return grains, labels, len(grains)


def save_overlay(img, labels, output_path, title=None):
    """
    Save the ImageGrains-style overlay to a file.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    overlay = create_overlay_imagegrains_style(img, labels)
    num_grains = len(np.unique(labels)) - 1
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(overlay)
    ax.axis('off')
    
    if title:
        ax.set_title(f"{title} (n={num_grains})", fontsize=14)
    else:
        ax.set_title(f"n={num_grains}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
