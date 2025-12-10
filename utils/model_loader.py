"""
Model loading utilities for all 4 models
Place this file in: utils/model_loader.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import streamlit as st
from tensorflow import keras
import numpy as np

# ============================================================================
# PYTORCH MODELS (Heart & NT Segmentation)
# ============================================================================

class MambaBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        identity = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.act(out)
        gate = torch.sigmoid(self.gate_conv(out))
        out = self.pwconv(out) * gate
        out = out + identity
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.mamba = MambaBlock(out_ch)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.mamba(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class UMambaUNet(nn.Module):
    """UMamba U-Net for segmentation"""
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)
        self.up1 = UpBlock(base_ch * 16, base_ch * 8)
        self.up2 = UpBlock(base_ch * 8, base_ch * 4)
        self.up3 = UpBlock(base_ch * 4, base_ch * 2)
        self.up4 = UpBlock(base_ch * 2, base_ch)
        self.head = nn.Conv2d(base_ch, out_ch, kernel_size=1)
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.bottleneck(self.pool4(x4))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.head(x)
        return logits

# ============================================================================
# KERAS MODELS (Plane Classification)
# ============================================================================

def PAICS_block(x):
    """Plane-Attention block for plane classification"""
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply
    
    channels = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(channels // 8, activation="relu")(se)
    se = Dense(channels, activation="sigmoid")(se)
    se = Reshape((1, 1, channels))(se)
    return Multiply()([x, se])

def build_plane_classification_model(num_classes=9):
    """
    Build ResNet50 + PAICS model for plane classification
    9 classes: brain-cerebellum, brain-hc, brain-ventricular, heart-3v, 
               heart-3vt, heart-4ch, heart-other, mid-sagittal-nonstandard,
               mid-sagittal-standard
    """
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    
    base = ResNet50(
        weights=None,  # We'll load our trained weights
        include_top=False,
        input_shape=(380, 380, 3)
    )
    
    x = base.output
    x = PAICS_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base.input, outputs=output)
    return model

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_plane_model(model_path='models/best_paics_model.keras'):
    """Load plane classification model"""
    try:
        # 9 classes for fetal plane classification
        num_classes = 9
        
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading plane model: {e}")
        return None

def load_brain_models(model_dir='models/brain_anomaly/'):
    """Load ensemble of brain anomaly detection models"""
    try:
        models = {}
        model_names = ['MobileNetV2', 'EfficientNetB3', 'Xception', 'DenseNet121']
        
        for name in model_names:
            model_path = Path(model_dir) / f'{name}_final.keras'
            if model_path.exists():
                models[name] = keras.models.load_model(str(model_path))
        
        if not models:
            st.warning("No brain anomaly models found. Using placeholder.")
            return None
        
        return models
    except Exception as e:
        st.error(f"Error loading brain models: {e}")
        return None

def load_heart_model(model_path='models/best_heart_model.pth'):
    """Load heart segmentation model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Heart model has 3 output classes: bg, cardiac, thorax
        model = UMambaUNet(in_ch=1, out_ch=3, base_ch=32)
        
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            st.warning(f"Heart model not found at {model_path}. Using untrained model.")
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading heart model: {e}")
        return None

def load_nt_model(model_path='models/umamba_nt_seg_best.pth'):
    """Load NT segmentation model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # NT model has 2 input channels and 1 output channel
        model = UMambaUNet(in_ch=2, out_ch=1, base_ch=32)
        
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            st.warning(f"NT model not found at {model_path}. Using untrained model.")
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading NT model: {e}")
        return None

def load_all_models():
    """Load all models and return as dictionary"""
    models = {}
    
    with st.spinner("Loading Plane Classification Model..."):
        models['plane_model'] = load_plane_model()
    
    with st.spinner("Loading Brain Anomaly Models (Ensemble)..."):
        models['brain_models'] = load_brain_models()
    
    with st.spinner("Loading Heart Segmentation Model..."):
        models['heart_model'] = load_heart_model()
    
    with st.spinner("Loading NT Segmentation Model..."):
        models['nt_model'] = load_nt_model()
    
    # Check if all models loaded
    loaded = sum(1 for v in models.values() if v is not None)
    st.info(f"✅ Successfully loaded {loaded}/4 models")
    
    return models