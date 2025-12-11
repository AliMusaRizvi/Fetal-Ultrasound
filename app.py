import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Import custom modules (we'll create these)
from utils.model_loader import load_all_models
from utils.inference import (
    infer_plane_classification,
    infer_brain_anomaly,
    infer_nt_segmentation,
    infer_heart_segmentation
)


# Page config
st.set_page_config(
    page_title="Fetal Ultrasound Analysis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Global settings */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #0f4c75; /* Deep medical blue */
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        font-family: 'Segoe UI', sans-serif;
    }
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #3282b8;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5rem;
    }
    
    /* Cards */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .metric-card h2 {
        font-size: 1.8rem;
        color: #0f4c75;
        margin: 0;
        font-weight: 700;
    }
    
    /* Alert boxes */
    .info-box, .warning-box, .success-box, .error-box {
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .info-box { 
        background-color: #e3f2fd; 
        border-left: 4px solid #2196f3; 
        color: #0d47a1; 
    }
    .warning-box { 
        background-color: #fff3e0; 
        border-left: 4px solid #ff9800; 
        color: #e65100; 
    }
    .success-box { 
        background-color: #e8f5e9; 
        border-left: 4px solid #4caf50; 
        color: #1b5e20; 
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        color: #b71c1c;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #0f4c75;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #3282b8;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.models = {}

@st.cache_resource
def initialize_models():
    """Load all models once and cache them"""
    with st.spinner("🔄 Loading AI models... This may take a minute..."):
        models = load_all_models()
    return models

def main():
    # Header
    st.markdown('<h1 class="main-header">Fetal Ultrasound Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Analysis Mode")
        
        analysis_type = st.radio(
            "Select Task",
            [
                "Fetal Plane Classification",
                "Brain Anomaly Detection",
                "Heart Segmentation (CTR)",
                "NT Measurement"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        with st.expander("ℹ️ Model Information"):
            st.markdown("""
            **System Capabilities:**
            - **Plane Classification**: ResNet50 + PAICS
            - **Brain Anomaly**: Multi-model Ensemble
            - **Heart Analysis**: UMamba Segmentation
            - **NT Measurement**: UMamba + ROI Guidance
            """)
        
        st.markdown("---")
        st.caption("Research Prototype v1.0")
        st.caption("© 2025 Fetal Analysis Group")
    
    # Load models
    if not st.session_state.models_loaded:
        try:
            st.session_state.models = initialize_models()
            st.session_state.models_loaded = True
        except Exception as e:
            st.error(f"System Error: Failed to initialize models. {str(e)}")
            st.stop()
    
    # Main content area
    # Use a container for the upload section to center it or make it prominent
    with st.container():
        col1, col2 = st.columns([1, 1.5], gap="large")
        
        with col1:
            st.markdown('<h2 class="sub-header">Image Acquisition</h2>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload Ultrasound Image",
                type=['png', 'jpg', 'jpeg'],
                help="Supported formats: PNG, JPG, JPEG"
            )
            
            if uploaded_file:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Source Image", use_container_width=True)
                
                # Image info
                st.markdown(f"""
                <div class="info-box" style="margin-top: 1rem;">
                    <strong>Image Properties</strong><br>
                    <small>Resolution: {image.size[0]} x {image.size[1]} px | Mode: {image.mode}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if uploaded_file:
                st.markdown('<h2 class="sub-header">Clinical Analysis</h2>', unsafe_allow_html=True)
                
                # Convert image for processing
                image_array = np.array(image)
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                else:
                    image_gray = image_array
                
                # Run inference button
                if st.button("Start Analysis", type="primary", use_container_width=True):
                    with st.spinner("Processing image..."):
                        try:
                            # Route to appropriate model
                            if "Plane Classification" in analysis_type:
                                results = infer_plane_classification(
                                    image_gray,
                                    st.session_state.models['plane_model']
                                )
                                display_plane_results(results, image_array)
                            
                            elif "Brain Anomaly" in analysis_type:
                                results = infer_brain_anomaly(
                                    image_gray,
                                    st.session_state.models['brain_models']
                                )
                                display_brain_results(results, image_array)
                            
                            elif "Heart Segmentation" in analysis_type:
                                results = infer_heart_segmentation(
                                    image_gray,
                                    st.session_state.models['heart_model']
                                )
                                display_heart_results(results, image_array)
                            
                            elif "NT Measurement" in analysis_type:
                                results = infer_nt_segmentation(
                                    image_gray,
                                    st.session_state.models['nt_model']
                                )
                                display_nt_results(results, image_array)
                        
                        except Exception as e:
                            st.markdown(f"""
                            <div class="error-box">
                                <strong>Analysis Error</strong><br>
                                {str(e)}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                # Placeholder when no image is uploaded
                st.info("Please upload an ultrasound image to begin analysis.")
                st.markdown("""
                <div style="text-align: center; color: #aaa; padding: 2rem;">
                    <p>Select an analysis mode from the sidebar and upload an image.</p>
                </div>
                """, unsafe_allow_html=True)

def display_plane_results(results, image):
    """Display plane classification results"""
    st.markdown("### Classification Results")
    
    # Main prediction
    pred_class = results['predicted_class']
    confidence = results['confidence']
    
    # Format class name for display
    formatted_class = pred_class.replace('-', ' ').title()
    
    # Color coding based on confidence
    if confidence > 0.8:
        color = "#e8f5e9" # Light green bg
        border_color = "#4caf50"
        quality = "High Confidence"
    elif confidence > 0.6:
        color = "#fff3e0" # Light orange bg
        border_color = "#ff9800"
        quality = "Moderate Confidence"
    else:
        color = "#ffebee" # Light red bg
        border_color = "#f44336"
        quality = "Low Confidence"
    
    st.markdown(f"""
    <div class="metric-card" style="background-color: {color}; border-left: 5px solid {border_color};">
        <h3>Predicted Plane</h3>
        <h2 style="color: #333;">{formatted_class}</h2>
        <p style="margin-top: 10px; font-size: 16px; color: #666;">
            Confidence: <strong>{confidence:.1%}</strong> ({quality})
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical relevance
    st.markdown("### Clinical Relevance")
    
    clinical_relevance = {
        'brain-cerebellum': 'Essential for evaluating posterior fossa and cerebellar development',
        'brain-hc': 'Critical for head circumference measurement and brain biometry',
        'brain-ventricular': 'Important for assessing ventricular size and detecting ventriculomegaly',
        'heart-3v': 'Three-vessel view - evaluates great vessels alignment',
        'heart-3vt': 'Three-vessel and trachea view - assesses cardiac outflow tracts',
        'heart-4ch': 'Four-chamber view - standard cardiac screening plane',
        'heart-other': 'Alternative cardiac view - may require repositioning',
        'mid-sagittal-nonstandard': 'Non-standard sagittal view - may need adjustment for NT measurement',
        'mid-sagittal-standard': 'Standard sagittal view - optimal for NT measurement'
    }
    
    st.markdown(f"""
    <div class="info-box">
        <strong>Clinical Context:</strong><br>
        {clinical_relevance.get(pred_class, "Clinical information not available.")}
    </div>
    """, unsafe_allow_html=True)
    
    # Acquisition recommendations
    if 'nonstandard' in pred_class or 'other' in pred_class:
        st.markdown("""
        <div class="warning-box">
            <strong>Acquisition Recommendation:</strong><br>
            This plane may not be optimal for standard measurements. 
            Consider repositioning the probe to obtain a standard view.
        </div>
        """, unsafe_allow_html=True)
    
    # Top predictions
    st.markdown("### Prediction Probabilities")
    
    # Format predictions for display
    formatted_preds = []
    for pred in results['top_predictions']:
        formatted_preds.append({
            'Plane': pred['class'].replace('-', ' ').title(),
            'Probability': f"{pred['probability']:.1%}",
            'Status': 'High' if pred['probability'] > 0.8 else 'Moderate' if pred['probability'] > 0.6 else 'Low'
        })
    
    top_preds_df = pd.DataFrame(formatted_preds)
    st.dataframe(top_preds_df, use_container_width=True, hide_index=True)
    
    # Probability bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = [p['class'].replace('-', ' ').title() for p in results['top_predictions']]
    probs = [p['probability'] for p in results['top_predictions']]
    
    # Professional colors
    colors = ['#2196f3' if p > 0.8 else '#90caf9' for p in probs]
    
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, probs, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Probability')
    ax.set_title('Class Probabilities')
    ax.set_xlim([0, 1.1])
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add value labels
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=9, color='#555')
    
    plt.tight_layout()
    st.pyplot(fig)

def display_brain_results(results, image):
    """Display brain anomaly detection results"""
    st.markdown("### Anomaly Detection Results")
    
    pred_class = results['predicted_class']
    confidence = results['confidence']
    
    # Anomaly severity mapping
    severity_map = {
        'arachnoid-cyst': 'Moderate',
        'cerebellah-hypoplasia': 'Severe',
        'encephalocele': 'Severe',
        'mild-ventriculomegaly': 'Mild',
        'moderate-ventriculomegaly': 'Moderate',
        'severe-ventriculomegaly': 'Severe'
    }
    
    severity = severity_map.get(pred_class, 'Unknown')
    
    # Display formatted anomaly name
    formatted_name = pred_class.replace('-', ' ').title()
    
    # Determine box style based on severity
    if severity == 'Severe':
        box_class = "error-box"
    elif severity == 'Moderate':
        box_class = "warning-box"
    else:
        box_class = "info-box"
    
    st.markdown(f"""
    <div class="{box_class}">
        <h3 style="margin-top:0;">Anomaly Detected: {formatted_name}</h3>
        <p><strong>Severity:</strong> {severity}</p>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
        <hr style="margin: 10px 0; border-top: 1px solid rgba(0,0,0,0.1);">
        <p><strong>Recommendation:</strong> Immediate consultation with a fetal medicine specialist is advised for detailed assessment and counseling.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical information based on anomaly type
    st.markdown("### Clinical Information")
    
    clinical_info = {
        'arachnoid-cyst': """
        **Arachnoid Cyst:**
        - Fluid-filled sac between brain and arachnoid membrane
        - Usually benign but requires monitoring
        - Prognosis depends on size and location
        - May require neurosurgical intervention after birth
        """,
        'cerebellah-hypoplasia': """
        **Cerebellar Hypoplasia:**
        - Underdevelopment of the cerebellum
        - Can affect motor coordination and balance
        - Associated with various genetic syndromes
        - Requires comprehensive genetic counseling
        """,
        'encephalocele': """
        **Encephalocele:**
        - Neural tube defect with brain tissue protruding through skull
        - Requires immediate surgical correction after birth
        - Prognosis varies based on location and extent
        - High-risk pregnancy requiring specialized care
        """,
        'mild-ventriculomegaly': """
        **Mild Ventriculomegaly:**
        - Lateral ventricle width: 10-12mm
        - Often resolves spontaneously
        - Requires close monitoring with follow-up scans
        - Generally favorable prognosis if isolated
        """,
        'moderate-ventriculomegaly': """
        **Moderate Ventriculomegaly:**
        - Lateral ventricle width: 12-15mm
        - Requires detailed anatomical survey
        - Consider amniocentesis for genetic testing
        - May be associated with chromosomal abnormalities
        """,
        'severe-ventriculomegaly': """
        **Severe Ventriculomegaly:**
        - Lateral ventricle width: >15mm
        - High risk for developmental delay
        - Comprehensive genetic and anatomical evaluation needed
        - Requires multidisciplinary care planning
        """
    }
    
    st.markdown(f"""
    <div class="info-box">
        {clinical_info.get(pred_class, "Clinical information not available.")}
    </div>
    """, unsafe_allow_html=True)
    
    # Ensemble predictions
    st.markdown("### Model Ensemble Predictions")
    ensemble_df = pd.DataFrame(results['ensemble_predictions'])
    st.dataframe(ensemble_df, use_container_width=True, hide_index=True)

def display_heart_results(results, image):
    """Display heart segmentation and CTR results"""
    st.markdown("### Cardiac Analysis Results")
    
    # CTR measurements
    ctr_area = results['ctr_area']
    ctr_diameter = results['ctr_diameter']
    
    # Normal CTR range is typically < 0.5
    ctr_status = "Normal" if ctr_area < 0.5 else "Enlarged"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CTR (Area)", f"{ctr_area:.3f}", delta="Normal" if ctr_area < 0.5 else "High", delta_color="inverse")
    with col2:
        st.metric("CTR (Diameter)", f"{ctr_diameter:.3f}")
    with col3:
        st.metric("Cardiac Axis", f"{results['angle']:.1f}°" if results['angle'] else "N/A")
    
    # Interpretation
    box_class = "success-box" if ctr_area < 0.5 else "warning-box"
    
    st.markdown(f"""
    <div class="{box_class}">
        <h4>Clinical Interpretation</h4>
        <p><strong>Status:</strong> {ctr_status}</p>
        <p><strong>CTR Value:</strong> {ctr_area:.3f} (Normal range: < 0.50)</p>
        <p><strong>Note:</strong> {"Within normal limits" if ctr_area < 0.5 else "May require further evaluation"}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display segmentation
    st.markdown("### Segmentation Visualization")
    overlay = results['overlay_image']
    st.image(overlay, caption="Heart Segmentation Overlay", use_container_width=True)

def display_nt_results(results, image):
    """Display NT measurement results"""
    st.markdown("### NT Measurement Results")
    
    thickness_mm = results['thickness_mm']
    thickness_px = results['thickness_pixels']
    risk_info = results.get('risk_assessment', {})
    
    # Get risk level from assessment or fallback to simple threshold
    if risk_info and 'risk' in risk_info:
        risk_level = risk_info['risk'].title()
        risk_message = risk_info.get('message', '')
        
        # Color coding based on risk
        if risk_info['risk'] == 'normal':
            color = "#e8f5e9"
            border_color = "#4caf50"
            icon = "✅"
        elif risk_info['risk'] == 'borderline':
            color = "#fff3e0"
            border_color = "#ff9800"
            icon = "⚠️"
        else:  # high or unknown
            color = "#ffebee"
            border_color = "#f44336"
            icon = "❗"
    else:
        # Fallback to simple threshold
        risk_level = "Low Risk" if thickness_mm and thickness_mm < 3.5 else "Increased Risk"
        risk_message = "Standard threshold applied (3.5mm)"
        color = "#e8f5e9" if thickness_mm and thickness_mm < 3.5 else "#ffebee"
        border_color = "#4caf50" if thickness_mm and thickness_mm < 3.5 else "#f44336"
        icon = "✅" if thickness_mm and thickness_mm < 3.5 else "❗"
    
    # Display main metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("NT Thickness (mm)", f"{thickness_mm:.2f}" if thickness_mm else "N/A")
    with col2:
        st.metric("NT Thickness (pixels)", f"{thickness_px}")
    
    # Risk assessment card
    st.markdown(f"""
    <div class="metric-card" style="background-color: {color}; border-left: 5px solid {border_color};">
        <h3>NT Measurement Analysis</h3>
        <h2 style="color: #333;">{thickness_mm:.2f} mm</h2>
        <h4 style="color: #555; margin-top: 0.5rem;">Risk Level: {icon} {risk_level}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical interpretation
    st.markdown(f"""
    <div class="info-box">
        <h4>Clinical Interpretation</h4>
        <p><strong>Measurement:</strong> {thickness_mm:.2f} mm ({thickness_px} pixels)</p>
        <p><strong>Assessment:</strong> {risk_message}</p>
        <p><strong>Note:</strong> NT measurements should be performed between 11-14 weeks gestation. 
        This automated measurement should be confirmed by a qualified sonographer.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display segmentation
    st.markdown("### NT Segmentation")
    overlay = results['overlay_image']
    st.image(overlay, caption="NT Segmentation Overlay", use_container_width=True)
    
    # Gestational age reference
    st.markdown("### NT Reference Values by Gestational Age")
    reference_data = {
        'Gestational Age (weeks)': ['11', '12', '13', '14'],
        'Normal Range (mm)': ['< 2.5', '< 3.0', '< 3.5', '< 4.0'],
        'Borderline (mm)': ['2.5-3.5', '3.0-4.0', '3.5-4.5', '4.0-5.0']
    }
    st.table(pd.DataFrame(reference_data))

if __name__ == "__main__":
    main()