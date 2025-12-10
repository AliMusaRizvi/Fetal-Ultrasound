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
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
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
    st.markdown('<h1 class="main-header">🏥 Fetal Ultrasound Analysis System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/ultrasound.png", width=100)
        st.title("Navigation")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "🔍 Fetal Plane Classification",
                "🧠 Brain Anomaly Detection",
                "💙 Heart Segmentation (CTR)",
                "📏 NT Measurement (Down Syndrome)"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Model Information")
        st.info("""
        **Available Models:**
        - Plane Classification (ResNet50+PAICS)
        - Brain Anomaly (Ensemble)
        - Heart Segmentation (UMamba)
        - NT Segmentation (UMamba)
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.warning("""
        This is a research tool. 
        Always consult a medical 
        professional for diagnosis.
        """)
    
    # Load models
    if not st.session_state.models_loaded:
        try:
            st.session_state.models = initialize_models()
            st.session_state.models_loaded = True
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.info("Please ensure all model files are in the 'models/' directory")
            st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an ultrasound image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a fetal ultrasound image for analysis"
        )
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.markdown(f"""
            <div class="info-box">
                <strong>Image Details:</strong><br>
                Size: {image.size[0]} x {image.size[1]} pixels<br>
                Mode: {image.mode}<br>
                Format: {uploaded_file.type}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file:
            st.markdown('<h2 class="sub-header">🔬 Analysis Results</h2>', unsafe_allow_html=True)
            
            # Convert image for processing
            image_array = np.array(image)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image_array
            
            # Run inference button
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                with st.spinner("🔬 Analyzing image..."):
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
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.exception(e)

def display_plane_results(results, image):
    """Display plane classification results"""
    st.markdown("### 🎯 Classification Results")
    
    # Main prediction
    pred_class = results['predicted_class']
    confidence = results['confidence']
    
    # Format class name for display
    formatted_class = pred_class.replace('-', ' ').title()
    
    # Color coding based on confidence
    if confidence > 0.8:
        color = "#28a745"
        icon = "✅"
        quality = "High Confidence"
    elif confidence > 0.6:
        color = "#ffc107"
        icon = "⚠️"
        quality = "Moderate Confidence"
    else:
        color = "#dc3545"
        icon = "❌"
        quality = "Low Confidence"
    
    st.markdown(f"""
    <div class="metric-card" style="background: {color};">
        <h3>{icon} Predicted Plane: {formatted_class}</h3>
        <h2>Confidence: {confidence:.1%}</h2>
        <p style="margin-top: 10px; font-size: 16px;">{quality}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical relevance
    st.markdown("### 📋 Clinical Relevance")
    
    clinical_relevance = {
        'brain-cerebellum': '🧠 Essential for evaluating posterior fossa and cerebellar development',
        'brain-hc': '🧠 Critical for head circumference measurement and brain biometry',
        'brain-ventricular': '🧠 Important for assessing ventricular size and detecting ventriculomegaly',
        'heart-3v': '💙 Three-vessel view - evaluates great vessels alignment',
        'heart-3vt': '💙 Three-vessel and trachea view - assesses cardiac outflow tracts',
        'heart-4ch': '💙 Four-chamber view - standard cardiac screening plane',
        'heart-other': '💙 Alternative cardiac view - may require repositioning',
        'mid-sagittal-nonstandard': '📏 Non-standard sagittal view - may need adjustment for NT measurement',
        'mid-sagittal-standard': '📏 Standard sagittal view - optimal for NT measurement'
    }
    
    st.info(clinical_relevance.get(pred_class, "Clinical information not available."))
    
    # Acquisition recommendations
    if 'nonstandard' in pred_class or 'other' in pred_class:
        st.warning("""
        **⚠️ Acquisition Recommendation:**
        This plane may not be optimal for standard measurements. 
        Consider repositioning the probe to obtain a standard view.
        """)
    
    # Top predictions
    st.markdown("### 📊 Top Predictions")
    
    # Format predictions for display
    formatted_preds = []
    for pred in results['top_predictions']:
        formatted_preds.append({
            'Plane': pred['class'].replace('-', ' ').title(),
            'Probability': f"{pred['probability']:.1%}",
            'Confidence': '🟢' if pred['probability'] > 0.8 else '🟡' if pred['probability'] > 0.6 else '🔴'
        })
    
    top_preds_df = pd.DataFrame(formatted_preds)
    st.dataframe(top_preds_df, use_container_width=True, hide_index=True)
    
    # Probability bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = [p['class'].replace('-', ' ').title() for p in results['top_predictions']]
    probs = [p['probability'] for p in results['top_predictions']]
    
    colors = ['#28a745' if p > 0.8 else '#ffc107' if p > 0.6 else '#dc3545' for p in probs]
    ax.barh(classes, probs, color=colors)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Classification Probabilities by Plane', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    
    # Add value labels on bars
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)

def display_brain_results(results, image):
    """Display brain anomaly detection results"""
    st.markdown("### 🧠 Anomaly Detection Results")
    
    pred_class = results['predicted_class']
    confidence = results['confidence']
    
    # Anomaly severity mapping
    severity_map = {
        'arachnoid-cyst': '🟡 Moderate',
        'cerebellah-hypoplasia': '🔴 Severe',
        'encephalocele': '🔴 Severe',
        'mild-ventriculomegaly': '🟢 Mild',
        'moderate-ventriculomegaly': '🟡 Moderate',
        'severe-ventriculomegaly': '🔴 Severe'
    }
    
    severity = severity_map.get(pred_class, '⚪ Unknown')
    
    # Display formatted anomaly name
    formatted_name = pred_class.replace('-', ' ').title()
    
    st.markdown(f"""
    <div class="warning-box">
        <h3>⚠️ Brain Anomaly Detected</h3>
        <p><strong>Type:</strong> {formatted_name}</p>
        <p><strong>Severity:</strong> {severity}</p>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
        <p><strong>Recommendation:</strong> Immediate consultation with a fetal medicine specialist is advised for detailed assessment and counseling.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical information based on anomaly type
    st.markdown("### 📚 Clinical Information")
    
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
    
    st.info(clinical_info.get(pred_class, "Clinical information not available."))
    
    # Ensemble predictions
    st.markdown("### 🤖 Model Ensemble Predictions")
    ensemble_df = pd.DataFrame(results['ensemble_predictions'])
    st.dataframe(ensemble_df, use_container_width=True)

def display_heart_results(results, image):
    """Display heart segmentation and CTR results"""
    st.markdown("### 💙 Cardiac Analysis Results")
    
    # CTR measurements
    ctr_area = results['ctr_area']
    ctr_diameter = results['ctr_diameter']
    
    # Normal CTR range is typically < 0.5
    ctr_status = "Normal" if ctr_area < 0.5 else "Enlarged"
    color = "#28a745" if ctr_area < 0.5 else "#dc3545"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CTR (Area)", f"{ctr_area:.3f}", delta=ctr_status)
    with col2:
        st.metric("CTR (Diameter)", f"{ctr_diameter:.3f}")
    with col3:
        st.metric("Cardiac Axis", f"{results['angle']:.1f}°" if results['angle'] else "N/A")
    
    # Interpretation
    st.markdown(f"""
    <div class="info-box" style="border-color: {color};">
        <h4>📋 Clinical Interpretation</h4>
        <p><strong>Status:</strong> {ctr_status}</p>
        <p><strong>CTR Value:</strong> {ctr_area:.3f} (Normal range: < 0.50)</p>
        <p><strong>Note:</strong> {"Within normal limits" if ctr_area < 0.5 else "May require further evaluation"}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display segmentation
    st.markdown("### 🎨 Segmentation Visualization")
    overlay = results['overlay_image']
    st.image(overlay, caption="Heart Segmentation Overlay", use_container_width=True)

def display_nt_results(results, image):
    """Display NT measurement results"""
    st.markdown("### 📏 NT Measurement Results")
    
    thickness_mm = results['thickness_mm']
    thickness_px = results['thickness_pixels']
    risk_info = results.get('risk_assessment', {})
    
    # Get risk level from assessment or fallback to simple threshold
    if risk_info and 'risk' in risk_info:
        risk_level = risk_info['risk'].title()
        risk_message = risk_info.get('message', '')
        
        # Color coding based on risk
        if risk_info['risk'] == 'normal':
            color = "#28a745"
            icon = "✅"
        elif risk_info['risk'] == 'borderline':
            color = "#ffc107"
            icon = "⚠️"
        else:  # high or unknown
            color = "#dc3545"
            icon = "❌"
    else:
        # Fallback to simple threshold
        risk_level = "Low Risk" if thickness_mm and thickness_mm < 3.5 else "Increased Risk"
        risk_message = "Standard threshold applied (3.5mm)"
        color = "#28a745" if thickness_mm and thickness_mm < 3.5 else "#dc3545"
        icon = "✅" if thickness_mm and thickness_mm < 3.5 else "⚠️"
    
    # Display main metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("NT Thickness (mm)", f"{thickness_mm:.2f}" if thickness_mm else "N/A")
    with col2:
        st.metric("NT Thickness (pixels)", f"{thickness_px}")
    
    # Risk assessment card
    st.markdown(f"""
    <div class="metric-card" style="background: {color};">
        <h3>{icon} NT Measurement Analysis</h3>
        <h2>{thickness_mm:.2f} mm</h2>
        <h4>Risk Level: {risk_level}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical interpretation
    st.markdown(f"""
    <div class="info-box">
        <h4>📋 Clinical Interpretation</h4>
        <p><strong>Measurement:</strong> {thickness_mm:.2f} mm ({thickness_px} pixels)</p>
        <p><strong>Assessment:</strong> {risk_message}</p>
        <p><strong>Note:</strong> NT measurements should be performed between 11-14 weeks gestation. 
        This automated measurement should be confirmed by a qualified sonographer.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display segmentation
    st.markdown("### 🎨 NT Segmentation")
    overlay = results['overlay_image']
    st.image(overlay, caption="NT Segmentation Overlay", use_container_width=True)
    
    # Gestational age reference
    st.markdown("### 📊 NT Reference Values by Gestational Age")
    reference_data = {
        'Gestational Age (weeks)': ['11', '12', '13', '14'],
        'Normal Range (mm)': ['< 2.5', '< 3.0', '< 3.5', '< 4.0'],
        'Borderline (mm)': ['2.5-3.5', '3.0-4.0', '3.5-4.5', '4.0-5.0']
    }
    st.table(pd.DataFrame(reference_data))

if __name__ == "__main__":
    main()