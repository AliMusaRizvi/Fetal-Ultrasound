# Fetal Ultrasound Analysis System

Fetal Ultrasound Analysis System is a Streamlit app that bundles four deep-learning tasks for prenatal ultrasound frames:

- **Fetal plane classification** (9 classes) using ResNet50 + PAICS attention.
- **Brain anomaly detection** ensemble (MobileNetV2, EfficientNetB3, Xception, DenseNet121).
- **Heart segmentation** (UMamba U-Net) with CTR and cardiac angle metrics.
- **Nuchal translucency (NT) segmentation and thickness measurement** (UMamba U-Net) with heuristic ROI and pseudo-mask guidance.

The app is research-grade only; clinical decisions must remain with qualified professionals.

## Repository Structure

```
app.py                  # Streamlit UI
config.py               # Paths, class lists, thresholds, hyperparameters
requirements.txt        # Python dependencies
models/                 # Model weights (not tracked)
  ├─ best_paics_model.keras
  ├─ best_heart_model.pth
  ├─ umamba_nt_seg_best.pth
  └─ brain_anomaly/
       ├─ DenseNet121_final.keras
       ├─ EfficientNetB3_final.keras
       ├─ MobileNetV2_final.keras
       └─ Xception_final.keras
utils/
  ├─ inference.py       # Inference pipelines (plane, brain, heart, NT)
  ├─ model_loader.py    # Model definitions + loaders
  └─ __init__.py
```

## Prerequisites

- Python 3.9+ (tested with CPU; GPU works if PyTorch/TensorFlow builds match your CUDA stack).
- Windows PowerShell (commands below assume PowerShell).
- Model weight files placed under `models/` as shown above.

## Setup

```powershell
# 1) Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt
```

If you use GPU, install the CUDA-matched wheels for PyTorch/TensorFlow instead of the CPU defaults.

## Running the App

```powershell
streamlit run app.py
```

Then open the provided local URL (default http://localhost:8501). Select an analysis type in the sidebar, upload a single ultrasound frame, and click **Run Analysis**.

### What Each Mode Does

- **Fetal Plane Classification**: predicts one of 9 standard planes.
- **Brain Anomaly Detection**: ensemble vote across four CNNs; returns top prediction and confidences.
- **Heart Segmentation (CTR)**: segments cardiac vs thorax, reports area and diameter CTR, plus cardiac axis angle.
- **NT Measurement**: segments the NT region, measures max thickness (pixels and mm using `pixel_spacing_mm` in `config.py`), and assigns a risk band using configurable thresholds.

## NT Segmentation Notes

- Input to the model is 2-channel: grayscale + ROI guidance.
- At inference, `utils/inference.py` builds a heuristic ROI (mid-lower dark band) when no detector boxes are available, generates a pseudo-mask with Otsu inside the ROI, and constrains model predictions to that ROI to avoid full-frame false positives.
- You can adjust ROI and thresholds in `generate_pseudo_nt_mask_inference` and `build_intelligent_roi_mask` if your data distribution differs.

## Configuration

Edit `config.py` to tweak:

- `MODEL_PATHS`: override weight locations.
- `NT_CONFIG.pixel_spacing_mm`: mm-per-pixel for NT measurement.
- `NT_THRESHOLDS` / `DEFAULT_NT_THRESHOLD`: risk cutoffs.
- Class lists and display metadata.

## Troubleshooting

- **Models fail to load**: verify filenames/paths match `MODEL_PATHS` and that weights are compatible with the defined architectures.
- **Black/blank outputs**: ensure uploads are valid grayscale/RGB frames; very large images are resized internally to 512 for segmentation.
- **Slow startup**: first load of TensorFlow/PyTorch can be slow; weights are cached after loading.
- **ROI too wide for NT**: tighten percentile/area caps in `build_intelligent_roi_mask` or integrate actual detector bounding boxes if available.

## License

This project is provided for research use only. Ensure compliance with local regulations and obtain appropriate approvals before any clinical deployment.
