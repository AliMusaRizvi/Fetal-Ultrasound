"""
Configuration file for Fetal Ultrasound Analysis System
Place this file in the root directory
"""


MODEL_PATHS = {
    'plane_classification': 'models/best_paics_model.keras',
    'nt_segmentation': 'models/umamba_nt_seg_best.pth',
    'heart_segmentation': 'models/best_heart_model.pth',
    'brain_anomaly_dir': 'models/brain_anomaly/'
}

 
# Fetal Plane Classification Classes (9 classes)
PLANE_CLASSES = [
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

# Brain Anomaly Detection Classes (6 classes)
BRAIN_ANOMALY_CLASSES = [
    'arachnoid-cyst',
    'cerebellah-hypoplasia',
    'encephalocele',
    'mild-ventriculomegaly',
    'moderate-ventriculomegaly',
    'severe-ventriculomegaly'
]

# MODEL HYPERPARAMETERS

# Plane Classification
PLANE_CONFIG = {
    'input_size': 380,
    'num_classes': len(PLANE_CLASSES),
    'batch_size': 16
}

# Brain Anomaly Detection
BRAIN_CONFIG = {
    'input_size': 224,
    'num_classes': len(BRAIN_ANOMALY_CLASSES),
    'ensemble_models': ['MobileNetV2', 'EfficientNetB3', 'Xception', 'DenseNet121']
}

# Heart Segmentation
HEART_CONFIG = {
    'input_size': 512,
    'num_classes': 3,  # background, cardiac, thorax
    'in_channels': 1,
    'base_channels': 64
}

# NT Segmentation
NT_CONFIG = {
    'input_size': 512,
    'in_channels': 2,  # image + ROI mask
    'out_channels': 1,
    'base_channels': 32,
    'pixel_spacing_mm': 0.1  # Default pixel spacing
}


# Cardiothoracic Ratio (CTR) thresholds
CTR_THRESHOLDS = {
    'normal_max': 0.50,
    'borderline_max': 0.55,
    'abnormal_min': 0.55
}

# NT Thickness thresholds (in mm) by gestational age
NT_THRESHOLDS = {
    11: {'normal': 2.5, 'borderline': 3.5},
    12: {'normal': 3.0, 'borderline': 4.0},
    13: {'normal': 3.5, 'borderline': 4.5},
    14: {'normal': 4.0, 'borderline': 5.0}
}

# Default NT threshold (if gestational age unknown)
DEFAULT_NT_THRESHOLD = {
    'normal_max': 3.5,
    'high_risk_min': 3.5
}


PLANE_CLINICAL_INFO = {
    'brain-cerebellum': {
        'description': 'Transverse cerebellar plane',
        'measurements': ['Transcerebellar diameter', 'Cisterna magna', 'Nuchal fold'],
        'purpose': 'Evaluates posterior fossa structures and cerebellar development'
    },
    'brain-hc': {
        'description': 'Transthalamic plane at level of cavum septum pellucidum',
        'measurements': ['Head circumference (HC)', 'Biparietal diameter (BPD)', 'Occipitofrontal diameter (OFD)'],
        'purpose': 'Standard plane for head biometry and dating'
    },
    'brain-ventricular': {
        'description': 'Transventricular plane',
        'measurements': ['Lateral ventricle width', 'Atrial diameter'],
        'purpose': 'Assessment of ventricular size and detection of ventriculomegaly'
    },
    'heart-3v': {
        'description': 'Three-vessel view',
        'measurements': ['Great vessel size and alignment'],
        'purpose': 'Evaluation of aorta, pulmonary artery, and SVC'
    },
    'heart-3vt': {
        'description': 'Three-vessel and trachea view',
        'measurements': ['Vessel alignment with trachea'],
        'purpose': 'Assessment of outflow tracts and aortic arch'
    },
    'heart-4ch': {
        'description': 'Four-chamber view',
        'measurements': ['Cardiac axis', 'Chamber size ratio', 'CTR'],
        'purpose': 'Standard cardiac screening for structural abnormalities'
    },
    'heart-other': {
        'description': 'Alternative cardiac view',
        'measurements': ['Variable'],
        'purpose': 'May require repositioning for standard views'
    },
    'mid-sagittal-nonstandard': {
        'description': 'Non-standard mid-sagittal view',
        'measurements': ['May not be accurate'],
        'purpose': 'Requires adjustment for proper NT measurement'
    },
    'mid-sagittal-standard': {
        'description': 'Standard mid-sagittal view',
        'measurements': ['Nuchal translucency (NT)', 'Nasal bone', 'Profile'],
        'purpose': 'First-trimester screening for chromosomal abnormalities'
    }
}

BRAIN_ANOMALY_INFO = {
    'arachnoid-cyst': {
        'severity': 'Moderate',
        'description': 'Fluid-filled sac between brain and arachnoid membrane',
        'prognosis': 'Usually benign but requires monitoring',
        'management': 'Serial imaging, possible neurosurgical intervention after birth'
    },
    'cerebellah-hypoplasia': {
        'severity': 'Severe',
        'description': 'Underdevelopment of the cerebellum',
        'prognosis': 'Variable, depends on extent and associated findings',
        'management': 'Genetic counseling, multidisciplinary evaluation'
    },
    'encephalocele': {
        'severity': 'Severe',
        'description': 'Neural tube defect with brain tissue herniation',
        'prognosis': 'Variable, depends on location and amount of tissue involved',
        'management': 'Immediate neurosurgical referral, high-risk pregnancy management'
    },
    'mild-ventriculomegaly': {
        'severity': 'Mild',
        'description': 'Lateral ventricle width 10-12mm',
        'prognosis': 'Often resolves, generally favorable if isolated',
        'management': 'Close monitoring with follow-up scans'
    },
    'moderate-ventriculomegaly': {
        'severity': 'Moderate',
        'description': 'Lateral ventricle width 12-15mm',
        'prognosis': 'May be associated with chromosomal abnormalities',
        'management': 'Detailed anatomical survey, consider amniocentesis'
    },
    'severe-ventriculomegaly': {
        'severity': 'Severe',
        'description': 'Lateral ventricle width >15mm',
        'prognosis': 'High risk for developmental delay',
        'management': 'Comprehensive genetic evaluation, multidisciplinary care planning'
    }
}


# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Confidence thresholds for color coding
CONFIDENCE_THRESHOLDS = {
    'high': 0.80,
    'moderate': 0.60,
    'low': 0.00
}

# Image upload settings
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']
MAX_FILE_SIZE_MB = 10

# DEVICE SETTINGS

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_MIXED_PRECISION = True if DEVICE == 'cuda' else False

# LOGGING

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'