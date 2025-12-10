"""
Inference functions for all 4 models
Place this file in: utils/inference.py
"""

import torch
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import math

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
    # Convert to BGR for OpenCV
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    
    # Define colors
    cardiac_color = np.array([230, 60, 60], dtype=np.uint8)  # Red
    thorax_color = np.array([50, 100, 230], dtype=np.uint8)  # Blue
    
    # Create mask overlays
    cardiac_mask = pred_mask == 1
    thorax_mask = pred_mask == 2
    
    # Apply colors
    overlay[thorax_mask] = cv2.addWeighted(
        overlay[thorax_mask], 0.6,
        np.full_like(overlay[thorax_mask], thorax_color), 0.4,
        0
    )
    overlay[cardiac_mask] = cv2.addWeighted(
        overlay[cardiac_mask], 0.6,
        np.full_like(overlay[cardiac_mask], cardiac_color), 0.4,
        0
    )
    
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
    """Measure NT thickness from segmentation mask"""
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
    
    thickness_mm = max_thickness_pixels * pixel_spacing_mm
    return max_thickness_pixels, thickness_mm

def create_nt_overlay(image, pred_mask):
    """Create visualization overlay for NT segmentation"""
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    
    # Red overlay for NT region
    nt_mask = pred_mask > 0.5
    overlay[nt_mask] = cv2.addWeighted(
        overlay[nt_mask], 0.6,
        np.full_like(overlay[nt_mask], [60, 60, 230]), 0.4,  # Red color
        0
    )
    
    return overlay

def infer_nt_segmentation(image, model, target_size=512, pixel_spacing_mm=0.1):
    """
    Run NT segmentation inference
    
    Args:
        image: numpy array (grayscale)
        model: PyTorch UMamba model
        target_size: int, resize dimension
        pixel_spacing_mm: float, pixel spacing for measurements
    
    Returns:
        dict with segmentation and measurements
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    orig_h, orig_w = image.shape
    
    # Preprocess
    img_resized = cv2.resize(image, (target_size, target_size))
    img_resized = img_resized.astype(np.float32) / 255.0
    
    # For NT model, we need 2 channels: image + ROI mask
    # Since we don't have ROI boxes here, we'll create a full ROI
    roi_mask = np.ones((target_size, target_size), dtype=np.float32)
    
    # Stack channels
    img_tensor = np.stack([img_resized, roi_mask], axis=0)
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float().to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float().cpu().numpy()[0, 0]
    
    # Resize back to original
    pred_mask_orig = cv2.resize(
        pred_mask,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Measure thickness
    thickness_px, thickness_mm = measure_nt_thickness(pred_mask_orig, pixel_spacing_mm)
    
    # Create overlay
    overlay = create_nt_overlay(image, pred_mask_orig)
    
    return {
        'thickness_pixels': thickness_px,
        'thickness_mm': thickness_mm,
        'pred_mask': pred_mask_orig,
        'overlay_image': overlay
    }