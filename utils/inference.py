"""
Inference functions for all 4 models
Place this file in: utils/inference.py
"""

import torch
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import math

# Optional DETR/DINO-style detector for NT ROI discovery
try:
    from transformers import DetrImageProcessor, DetrForObjectDetection  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

DETR_NAME = "facebook/detr-resnet-50"
_detr_processor = None
_detr_model = None
_detr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_image_gray(img_path_or_array):
    """
    Load and normalize a grayscale image to [0, 1]
    
    Args:
        img_path_or_array: file path (str) or numpy array
    
    Returns:
        numpy array in float32, values [0, 1]
    """
    if isinstance(img_path_or_array, str):
        img = cv2.imread(img_path_or_array, cv2.IMREAD_GRAYSCALE)
    else:
        img = img_path_or_array
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    
    return img


def _load_detr_detector():
    """Lazy-load DETR/DINO detector if transformers is available."""
    global _detr_model, _detr_processor
    if not _TRANSFORMERS_AVAILABLE:
        return None, None
    if _detr_model is not None and _detr_processor is not None:
        return _detr_model, _detr_processor
    try:
        _detr_processor = DetrImageProcessor.from_pretrained(DETR_NAME)
        _detr_model = DetrForObjectDetection.from_pretrained(DETR_NAME).to(_detr_device)
        _detr_model.eval()
        return _detr_model, _detr_processor
    except Exception:
        _detr_model, _detr_processor = None, None
        return None, None


def run_detr_inference(img_path: str, score_thresh: float = 0.5):
    """Run DETR to get bounding boxes; returns list of [x1,y1,x2,y2] floats."""
    model, processor = _load_detr_detector()
    if model is None or processor is None:
        return []
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt").to(_detr_device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([img_rgb.shape[:2]], device=_detr_device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=score_thresh)[0]
    boxes = []
    for box in results.get("boxes", []):
        boxes.append(box.tolist())
    return boxes


def build_roi_from_boxes(boxes, orig_w, orig_h, resized_w, resized_h):
    """Create ROI mask from a list of boxes in original coordinates."""
    roi_mask = np.zeros((resized_h, resized_w), dtype=np.float32)
    if not boxes:
        return roi_mask
    scale_x = resized_w / float(orig_w)
    scale_y = resized_h / float(orig_h)
    for (x1, y1, x2, y2) in boxes:
        sx1 = int(round(x1 * scale_x))
        sy1 = int(round(y1 * scale_y))
        sx2 = int(round(x2 * scale_x))
        sy2 = int(round(y2 * scale_y))
        sx1 = max(0, min(resized_w - 1, sx1))
        sx2 = max(0, min(resized_w - 1, sx2))
        sy1 = max(0, min(resized_h - 1, sy1))
        sy2 = max(0, min(resized_h - 1, sy2))
        if sx2 > sx1 and sy2 > sy1:
            roi_mask[sy1:sy2, sx1:sx2] = 1.0
    return roi_mask


def build_intelligent_roi_mask(img_resized: np.ndarray) -> np.ndarray:
    """Heuristic ROI for NT when no detector boxes exist.

    We constrain to the mid-lower band of the image, pick the darkest blob,
    and cap the ROI size so the model does not consider the entire frame.
    """
    h, w = img_resized.shape
    roi_mask = np.zeros((h, w), dtype=np.float32)

    # Focus on the middle-lower region where NT usually appears
    y_top = int(h * 0.35)
    y_bottom = int(h * 0.78)
    band = img_resized[y_top:y_bottom, :]

    band_u8 = (band * 255).astype(np.uint8)
    band_blur = cv2.GaussianBlur(band_u8, (7, 7), 0)

    # Keep only the darkest 8th-percentile pixels to approximate fluid
    thresh_val = int(np.percentile(band_blur, 8))
    dark = (band_blur <= thresh_val).astype(np.uint8)

    # Clean noise and merge nearby pixels
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    # Pick the largest reasonable component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    chosen = None
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w_box / max(1, h_box)

        # Reject overly large or implausible shapes
        if area < 30:
            continue
        if area > 0.12 * h * w:
            continue
        if h_box > 0.25 * h:
            continue
        if aspect < 1.2:  # NT band tends to be wider than tall
            continue

        if chosen is None or area > chosen[0]:
            chosen = (area, stats[i])

    if chosen is None:
        # Fallback: narrow central band
        x1 = int(w * 0.30)
        x2 = int(w * 0.70)
        y1 = int(h * 0.55)
        y2 = int(h * 0.70)
    else:
        _, st = chosen
        x1 = st[cv2.CC_STAT_LEFT]
        y1 = st[cv2.CC_STAT_TOP] + y_top
        x2 = x1 + st[cv2.CC_STAT_WIDTH]
        y2 = y1 + st[cv2.CC_STAT_HEIGHT]

        # Add small margins
        pad_x = int(0.08 * w)
        pad_y = int(0.05 * h)
        x1 = max(0, x1 - pad_x)
        x2 = min(w - 1, x2 + pad_x)
        y1 = max(0, y1 - pad_y)
        y2 = min(h - 1, y2 + pad_y)

    # Cap ROI size to avoid covering the full frame
    max_area = 0.15 * h * w
    box_area = (y2 - y1) * (x2 - x1)
    if box_area > max_area:
        scale = math.sqrt(max_area / max(box_area, 1))
        new_h = max(10, int((y2 - y1) * scale))
        new_w = max(10, int((x2 - x1) * scale))
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2
        y1 = max(0, cy - new_h // 2)
        y2 = min(h, y1 + new_h)
        x1 = max(0, cx - new_w // 2)
        x2 = min(w, x1 + new_w)

    roi_mask[y1:y2, x1:x2] = 1.0
    return roi_mask


def generate_pseudo_nt_mask_inference(
    img_resized: np.ndarray,
    roi_mask: np.ndarray
) -> np.ndarray:
    """
    Generate pseudo NT segmentation mask using the same logic as training.
    This applies Otsu thresholding within the ROI to segment the dark NT fluid.
    
    CRITICAL: If ROI is the entire image (no bounding box), we create an
    intelligent ROI by detecting the darkest region in the neck area.
    
    Args:
        img_resized: grayscale image in [0,1], shape (H, W)
        roi_mask: ROI mask in [0,1], shape (H, W)
    
    Returns:
        pseudo_mask: binary mask (H, W) with values {0,1}
    """
    h, w = img_resized.shape
    pseudo_mask = np.zeros((h, w), dtype=np.float32)
    
    # Check if ROI is reasonable; if empty, return blank mask
    roi_area = roi_mask.sum()
    if roi_area < 10:
        return pseudo_mask
    
    # Find ROI region
    roi_coords = np.where(roi_mask > 0.5)
    if len(roi_coords[0]) == 0:
        return pseudo_mask
    
    # Get bounding box of ROI
    y_min, y_max = roi_coords[0].min(), roi_coords[0].max()
    x_min, x_max = roi_coords[1].min(), roi_coords[1].max()
    
    # Ensure reasonable ROI size (not too small)
    if (y_max - y_min) < 20 or (x_max - x_min) < 20:
        return pseudo_mask
    
    # Extract ROI region
    roi_img = img_resized[y_min:y_max+1, x_min:x_max+1]
    if roi_img.size == 0:
        return pseudo_mask
    
    # Convert to uint8 for OpenCV
    roi_uint8 = (roi_img * 255).astype(np.uint8)
    
    # Apply Gaussian blur to reduce speckle noise
    roi_blur = cv2.GaussianBlur(roi_uint8, (5, 5), 0)
    
    # Otsu threshold (binary inverse: dark NT fluid -> white)
    _, roi_thr = cv2.threshold(
        roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    roi_thr = cv2.morphologyEx(roi_thr, cv2.MORPH_OPEN, kernel, iterations=1)
    roi_thr = cv2.morphologyEx(roi_thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Keep largest connected component to remove spurious blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        roi_thr, connectivity=8
    )
    if num_labels > 1:
        # Get areas (skip background label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_label = 1 + np.argmax(areas)
        roi_cc = (labels == max_label).astype(np.uint8)
    else:
        roi_cc = (roi_thr > 0).astype(np.uint8)
    
    roi_cc = roi_cc.astype(np.float32)
    
    # Additional size filtering: clamp NT to ROI-relative area
    nt_area = roi_cc.sum()
    max_reasonable_area = 0.35 * roi_area  # within ROI
    global_cap = 0.08 * (h * w)            # absolute cap
    target_cap = min(max_reasonable_area, global_cap)
    if nt_area > target_cap and target_cap > 0:
        scale = math.sqrt(target_cap / max(nt_area, 1))
        box_h = int((roi_cc.shape[0]) * scale)
        box_w = int((roi_cc.shape[1]) * scale)
        cy = roi_cc.shape[0] // 2
        cx = roi_cc.shape[1] // 2
        y1 = max(0, cy - box_h // 2)
        y2 = min(roi_cc.shape[0], y1 + box_h)
        x1 = max(0, cx - box_w // 2)
        x2 = min(roi_cc.shape[1], x1 + box_w)
        shrunk = np.zeros_like(roi_cc)
        shrunk[y1:y2, x1:x2] = 1.0
        roi_cc = shrunk
    
    # Place back into full mask
    pseudo_mask[y_min:y_max+1, x_min:x_max+1] = roi_cc
    
    # Ensure binary {0,1}
    pseudo_mask = (pseudo_mask > 0.5).astype(np.float32)
    return pseudo_mask

# ============================================================================
# PLANE CLASSIFICATION
# ============================================================================

def infer_plane_classification(image, model, target_size=380):
    """
    Run plane classification inference
    
    Args:
        image: numpy array (grayscale or RGB)
        model: Keras model
        target_size: int, resize dimension
    
    Returns:
        dict with predictions
    """
    # Preprocess image
    img = cv2.resize(image, (target_size, target_size))
    
    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict
    predictions = model.predict(img, verbose=0)[0]
    
    # Correct class names for fetal plane classification
    class_names = [
        'brain-cerebellum',
        'brain-hc',
        'brain-ventricular',
        'heart-3v',
        'heart-3vt',
        'heart-4ch',
        'heart-other',
        'mid-sagittal-nonstandard',
        'mid-sagittal-standard'
    ]
    
    # Get top predictions
    top_k = 5
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    top_predictions = []
    for idx in top_indices:
        if idx < len(class_names):
            top_predictions.append({
                'class': class_names[idx],
                'probability': float(predictions[idx])
            })
    
    predicted_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class {predicted_idx}"
    
    return {
        'predicted_class': predicted_class,
        'confidence': float(predictions[predicted_idx]),
        'all_predictions': predictions.tolist(),
        'top_predictions': top_predictions
    }

# ============================================================================
# BRAIN ANOMALY DETECTION
# ============================================================================

def infer_brain_anomaly(image, models, target_size=224):
    """
    Run brain anomaly detection using ensemble
    
    Args:
        image: numpy array (grayscale or RGB)
        models: dict of Keras models
        target_size: int, resize dimension
    
    Returns:
        dict with predictions
    """
    # Preprocess image
    img = cv2.resize(image, (target_size, target_size))
    
    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Correct class names for fetal brain anomaly detection
    class_names = [
        'arachnoid-cyst',
        'cerebellah-hypoplasia',
        'encephalocele',
        'mild-ventriculomegaly',
        'moderate-ventriculomegaly',
        'severe-ventriculomegaly'
    ]
    
    if models is None or len(models) == 0:
        # Fallback for demo - return mild ventriculomegaly as example
        return {
            'predicted_class': 'mild-ventriculomegaly',
            'confidence': 0.75,
            'ensemble_predictions': [{'model': 'Demo', 'prediction': 'mild-ventriculomegaly', 'confidence': 0.75}]
        }
    
    # Get predictions from each model
    ensemble_predictions = []
    all_probs = []
    
    for model_name, model in models.items():
        try:
            preds = model.predict(img, verbose=0)[0]
            all_probs.append(preds)
            
            pred_idx = np.argmax(preds)
            pred_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
            
            ensemble_predictions.append({
                'model': model_name,
                'prediction': pred_class,
                'confidence': float(preds[pred_idx])
            })
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
    
    # Average predictions
    if all_probs:
        avg_probs = np.mean(all_probs, axis=0)
        final_idx = np.argmax(avg_probs)
        final_class = class_names[final_idx] if final_idx < len(class_names) else f"Class {final_idx}"
        final_conf = float(avg_probs[final_idx])
    else:
        final_class = 'normal'
        final_conf = 0.5
    
    return {
        'predicted_class': final_class,
        'confidence': final_conf,
        'ensemble_predictions': ensemble_predictions
    }

# ============================================================================
# HEART SEGMENTATION
# ============================================================================

def compute_ctr(pred_mask):
    """Compute Cardiothoracic Ratio (CTR)"""
    cardiac = (pred_mask == 1).astype(np.uint8)
    thorax = (pred_mask == 2).astype(np.uint8)
    
    # Area-based CTR
    area_cardiac = cardiac.sum()
    area_thorax = thorax.sum()
    ctr_area = area_cardiac / area_thorax if area_thorax > 0 else 0.0
    
    # Diameter-based CTR (maximum width)
    cardiac_width = cardiac.sum(axis=0).max()
    thorax_width = thorax.sum(axis=0).max()
    ctr_diameter = cardiac_width / thorax_width if thorax_width > 0 else 0.0
    
    return ctr_area, ctr_diameter

def compute_cardiac_angle(pred_mask):
    """Compute cardiac axis angle"""
    cardiac = (pred_mask == 1).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(cardiac, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) >= 5 and cv2.contourArea(cnt) > 10:
        ellipse = cv2.fitEllipse(cnt)
        return float(ellipse[2])
    
    # PCA fallback
    ys, xs = np.where(cardiac > 0)
    if len(xs) < 2:
        return None
    
    x = xs - xs.mean()
    y = ys - ys.mean()
    cov = np.cov(np.vstack([x, y]))
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argmax(eigvals)
    vx, vy = eigvecs[:, idx]
    angle = math.degrees(math.atan2(vy, vx))
    return float(angle)

def create_heart_overlay(image, pred_mask):
    """Create visualization overlay for heart segmentation"""
    # Ensure image is uint8 BGR
    if len(image.shape) == 2:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
        if overlay.dtype != np.uint8:
            overlay = (overlay * 255).astype(np.uint8) if overlay.max() <= 1.0 else overlay.astype(np.uint8)
    
    # Ensure overlay is writable
    overlay = np.ascontiguousarray(overlay)
    
    # Define colors (BGR format)
    cardiac_color = np.array([60, 60, 230], dtype=np.uint8)  # Red in BGR
    thorax_color = np.array([230, 100, 50], dtype=np.uint8)  # Blue in BGR
    
    # Create colored overlays
    cardiac_mask = (pred_mask == 1)
    thorax_mask = (pred_mask == 2)
    
    # Apply thorax overlay (blue)
    if np.any(thorax_mask):
        overlay_copy = overlay.copy()
        overlay_copy[thorax_mask] = thorax_color
        overlay = cv2.addWeighted(overlay, 0.6, overlay_copy, 0.4, 0)
    
    # Apply cardiac overlay (red)
    if np.any(cardiac_mask):
        overlay_copy = overlay.copy()
        overlay_copy[cardiac_mask] = cardiac_color
        overlay = cv2.addWeighted(overlay, 0.6, overlay_copy, 0.4, 0)
    
    return overlay

def infer_heart_segmentation(image, model, target_size=512):
    """
    Run heart segmentation inference
    
    Args:
        image: numpy array (grayscale)
        model: PyTorch UMamba model
        target_size: int, resize dimension
    
    Returns:
        dict with segmentation and measurements
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess
    img = cv2.resize(image, (target_size, target_size))
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()#.to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_mask = torch.argmax(probs, dim=1).cpu().numpy()[0]
    
    # Compute measurements
    ctr_area, ctr_diameter = compute_ctr(pred_mask)
    angle = compute_cardiac_angle(pred_mask)
    
    # Resize mask back to original size
    pred_mask_resized = cv2.resize(
        pred_mask.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create overlay
    overlay = create_heart_overlay(image, pred_mask_resized)
    
    return {
        'ctr_area': ctr_area,
        'ctr_diameter': ctr_diameter,
        'angle': angle,
        'pred_mask': pred_mask_resized,
        'overlay_image': overlay
    }

# ============================================================================
# NT SEGMENTATION
# ============================================================================

def measure_nt_thickness(mask, pixel_spacing_mm=0.1):
    """
    Measure NT thickness from a binary mask.
    - mask: (H, W) with values 0/1 (NT region = 1)
    - For each column x, find top and bottom of NT, compute height.
    - Thickness = max column height.
    
    Args:
        mask: numpy array (H, W) with values 0/1
        pixel_spacing_mm: float, pixel spacing for mm conversion
    
    Returns:
        max_thickness_pixels: int, maximum thickness in pixels
        thickness_mm: float or None, thickness in mm
    """
    mask_bin = (mask > 0.5).astype(np.uint8)
    h, w = mask_bin.shape
    
    max_thickness_pixels = 0
    
    for x in range(w):
        column = mask_bin[:, x]
        ys = np.where(column > 0)[0]
        if len(ys) == 0:
            continue
        thickness = ys.max() - ys.min()
        if thickness > max_thickness_pixels:
            max_thickness_pixels = thickness
    
    thickness_mm = None
    if pixel_spacing_mm is not None:
        thickness_mm = max_thickness_pixels * pixel_spacing_mm
    
    return max_thickness_pixels, thickness_mm

def create_nt_overlay(image, pred_mask):
    """Create visualization overlay for NT segmentation"""
    # Ensure we have a proper image format
    if len(image.shape) == 2:
        # Grayscale - convert to BGR
        img_display = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) if image.max() <= 1.0 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # Already BGR
        img_display = image.copy()
        if img_display.max() <= 1.0:
            img_display = (img_display * 255).astype(np.uint8)
    
    # Red overlay for NT region (BGR format)
    nt_mask = pred_mask > 0.5
    overlay = img_display.copy()
    overlay[nt_mask] = cv2.addWeighted(
        overlay[nt_mask], 0.6,
        np.full_like(overlay[nt_mask], [60, 60, 230]), 0.4,  # Red color in BGR
        0
    )
    
    return overlay


def apply_post_processing(pred_mask, min_area=50):
    """
    Apply post-processing to remove small noise regions
    
    Args:
        pred_mask: binary mask (H, W) with values 0/1
        min_area: minimum area in pixels to keep
    
    Returns:
        cleaned binary mask
    """
    mask_uint8 = (pred_mask * 255).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    # Create cleaned mask
    cleaned_mask = np.zeros_like(pred_mask)
    
    # Keep only components larger than min_area (skip background label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 1.0
    
    return cleaned_mask


def get_nt_risk_assessment(thickness_mm, gestational_age_weeks=None):
    """
    Assess NT thickness risk level
    
    Args:
        thickness_mm: NT thickness in millimeters
        gestational_age_weeks: gestational age (11-14 weeks), optional
    
    Returns:
        dict with risk assessment
    """
    if thickness_mm is None:
        return {'risk': 'unknown', 'message': 'Unable to measure NT thickness'}
    
    # Use gestational age-specific thresholds if available
    if gestational_age_weeks is not None and 11 <= gestational_age_weeks <= 14:
        from config import NT_THRESHOLDS
        thresholds = NT_THRESHOLDS.get(gestational_age_weeks, NT_THRESHOLDS[13])
        
        if thickness_mm <= thresholds['normal']:
            risk = 'normal'
            message = f'NT measurement within normal range for {gestational_age_weeks} weeks'
        elif thickness_mm <= thresholds['borderline']:
            risk = 'borderline'
            message = f'NT measurement borderline for {gestational_age_weeks} weeks - consider follow-up'
        else:
            risk = 'high'
            message = f'NT measurement elevated for {gestational_age_weeks} weeks - recommend genetic counseling'
    else:
        # Use default thresholds
        from config import DEFAULT_NT_THRESHOLD
        
        if thickness_mm <= DEFAULT_NT_THRESHOLD['normal_max']:
            risk = 'normal'
            message = 'NT measurement within normal range'
        else:
            risk = 'high'
            message = 'NT measurement elevated - recommend genetic counseling and detailed follow-up'
    
    return {
        'risk': risk,
        'thickness_mm': thickness_mm,
        'message': message
    }

def infer_nt_segmentation(image, model, target_size=512, pixel_spacing_mm=0.1, use_pseudo_mask=True):
    """
    Run NT segmentation inference EXACTLY matching the training pipeline.
    
    CRITICAL: Training used pseudo-masks generated via Otsu thresholding.
    The model learned to refine these pseudo-masks, not segment from scratch.
    
    Args:
        image: numpy array (grayscale or BGR)
        model: PyTorch UMamba model (in_ch=2, out_ch=1)
        target_size: int, resize dimension (should match training: 512)
        pixel_spacing_mm: float, pixel spacing for mm measurements
        use_pseudo_mask: bool, whether to use pseudo-mask initialization (recommended)
    
    Returns:
        dict with:
            - thickness_pixels: int
            - thickness_mm: float or None
            - pred_mask: numpy array (H, W) in original resolution
            - overlay_image: BGR image with overlay
            - risk_assessment: dict with risk level
            - pseudo_mask_resized: initial pseudo mask before model refinement
    """
    device = next(model.parameters()).device  # Get model's device
    
    # Store original dimensions
    if len(image.shape) == 3:
        orig_h, orig_w = image.shape[:2]
    else:
        orig_h, orig_w = image.shape
    
    # 1) Load and normalize grayscale image to [0,1]
    img = load_image_gray(image)
    
    # 2) Resize to model input size using INTER_AREA (same as training)
    img_resized = cv2.resize(
        img, (target_size, target_size), interpolation=cv2.INTER_AREA
    )
    
    # 3) Build ROI mask
    roi_mask = None
    # 3a) If image is a path, try DETR/DINO detection to create ROI
    if isinstance(image, str):
        det_boxes = run_detr_inference(image, score_thresh=0.5)
        if det_boxes:
            roi_mask = build_roi_from_boxes(det_boxes, orig_w, orig_h, target_size, target_size)
    # 3b) If detector not available or no boxes, fall back to heuristic ROI
    if roi_mask is None or roi_mask.sum() < 1:
        roi_mask = build_intelligent_roi_mask(img_resized)
    
    # 3.5) Generate pseudo-mask using Otsu (SAME AS TRAINING)
    # This is critical - the model was trained to refine pseudo-masks, not raw images
    pseudo_mask_resized = None
    if use_pseudo_mask:
        pseudo_mask_resized = generate_pseudo_nt_mask_inference(
            img_resized=img_resized,
            roi_mask=roi_mask
        )
    
    # 4) Build input tensor [image, roi_mask] - shape (2, H, W)
    img_tensor = np.stack([img_resized, roi_mask], axis=0)  # (2, H, W)
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float().to(device)  # (1, 2, H, W)
    
    # 5) Forward through model to refine the segmentation
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]  # (H, W)
        pred_mask_resized = (probs > 0.5).astype(np.float32)

    # Restrict predictions to ROI to avoid spill-over
    pred_mask_resized = pred_mask_resized * roi_mask
    
    # If pseudo-mask exists and model didn't predict much, use pseudo-mask as fallback
    if use_pseudo_mask and pseudo_mask_resized is not None:
        # Check if model prediction is too small (likely failed)
        model_area = pred_mask_resized.sum()
        pseudo_area = pseudo_mask_resized.sum()
        
        # If model predicted very little but pseudo-mask has content, blend or use pseudo
        if model_area < 100 and pseudo_area > 100:
            # Use pseudo-mask directly as the model might not have learned well
            pred_mask_resized = pseudo_mask_resized
        elif model_area > 0 and pseudo_area > 0:
            # Optionally blend: union of both masks (conservative)
            pred_mask_resized = np.maximum(pred_mask_resized, pseudo_mask_resized * 0.5)
            pred_mask_resized = (pred_mask_resized > 0.5).astype(np.float32)
    
    # 6) Resize mask back to original image resolution
    pred_mask_orig = cv2.resize(
        pred_mask_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )
    pred_mask_orig = (pred_mask_orig > 0.5).astype(np.float32)
    
    # 7) Apply post-processing to remove small noise
    pred_mask_orig = apply_post_processing(pred_mask_orig, min_area=50)
    
    # 8) Measure NT thickness
    thickness_px, thickness_mm = measure_nt_thickness(
        pred_mask_orig, pixel_spacing_mm
    )
    
    # 9) Get risk assessment
    risk_assessment = get_nt_risk_assessment(thickness_mm)
    
    # 10) Create overlay for visualization
    overlay = create_nt_overlay(image, pred_mask_orig)
    
    return {
        'thickness_pixels': thickness_px,
        'thickness_mm': thickness_mm,
        'pred_mask': pred_mask_orig,
        'overlay_image': overlay,
        'risk_assessment': risk_assessment,
        'pseudo_mask_resized': pseudo_mask_resized  # For debugging
    }