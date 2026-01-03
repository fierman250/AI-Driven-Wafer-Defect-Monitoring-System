"""
Main Streamlit App for Wafer Defect Monitoring System
Landing Page
"""
import streamlit as st
import sys
from pathlib import Path
from PIL import Image

# Add Repository to path for imports
sys.path.insert(0, str(Path(__file__).parent / "Repository"))

# Page configuration
st.set_page_config(
    page_title="Wafer Defect Monitoring",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get background image path
bg_image_path = Path(__file__).parent / "Pages" / "LPBackgroung.png"

# Load and encode background image if it exists
bg_image_base64 = ""
if bg_image_path.exists():
    import base64
    with open(bg_image_path, "rb") as img_file:
        bg_image_base64 = base64.b64encode(img_file.read()).decode()

# Custom CSS for better styling with background image
st.markdown(f"""
<style>
    html, body {{
        margin: 0;
        padding: 0;
    }}
    
    /* Apply background to the main app container */
    .stApp {{
        background-image: url('data:image/png;base64,{bg_image_base64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        min-height: 100vh;
    }}
    
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.40);
        z-index: 0;
        pointer-events: none;
    }}
    
    /* Ensure all Streamlit content is visible above background */
    .main,
    [data-testid="stHeader"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarNav"],
    .main .block-container,
    .element-container,
    section[data-testid="stSidebar"],
    div[data-testid="stVerticalBlock"] {{
        position: relative !important;
        z-index: 10 !important;
        background: transparent;
    }}
    
    /* Overlay for better text readability */
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px;
        padding: 2rem;
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Ensure sidebar is visible */
    section[data-testid="stSidebar"] {{
        background-color: rgba(255, 255, 255, 0.98) !important;
    }}
    
    /* Style sidebar title - larger and centered, positioned at top */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] [class*="stMarkdown"] h1 {{
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        margin: 0 0 1.5rem 0 !important;
        padding: 0 !important;
        color: #1E88E5 !important;
        order: -1 !important;
    }}
    
    /* Style Streamlit's page navigation - larger text and centered */
    nav[data-testid="stSidebarNav"] {{
        margin-top: 1rem;
    }}
    
    /* Style navigation links - larger text, centered, and better spacing */
    nav[data-testid="stSidebarNav"] ul {{
        list-style: none;
        padding: 0;
        margin: 0;
    }}
    
    nav[data-testid="stSidebarNav"] li {{
        margin: 0.5rem 0;
    }}
    
    nav[data-testid="stSidebarNav"] a {{
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        text-align: center !important;
        display: block !important;
        padding: 0.75rem 1rem !important;
        width: 90% !important;
        margin: 0 auto !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
    }}
    
    nav[data-testid="stSidebarNav"] a:hover {{
        background-color: rgba(30, 136, 229, 0.1) !important;
        transform: translateX(5px);
    }}
    
    /* Ensure sidebar content order - title first, above navigation */
    [data-testid="stSidebar"] > div {{
        display: flex !important;
        flex-direction: column !important;
    }}
    
    /* Move title to top using CSS order */
    [data-testid="stSidebar"] .sidebar-title-container {{
        order: -10 !important;
    }}
    
    /* Move navigation to appear after title */
    nav[data-testid="stSidebarNav"] {{
        order: 0 !important;
    }}
    
    .main-header {{
        font-size: 3rem !important;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 0.5rem 0 !important;
        margin-top: -5rem !important;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }}
    
    .subtitle {{
        font-size: 1.5rem !important;
        color: #666;
        text-align: center;
        margin-bottom: 18rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }}
    
    .feature-card {{
        background: linear-gradient(135deg, #1bbabf 0%, #764ba2 100%);
        font-size: 1.3rem;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        opacity: 0.60;
    }}
    
    
    .metric-card {{
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }}
    
    .status-running {{
        color: #43A047;
        font-weight: bold;
    }}
    
    .status-stopped {{
        color: #E53935;
        font-weight: bold;
    }}
    
    .landing-button {{
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 8px;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
        font-weight: bold;
        transition: transform 0.2s;
    }}
    
    .landing-button:hover {{
        transform: scale(1.05);
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar title - using custom HTML to ensure it appears at top
st.sidebar.markdown("""
<div class="sidebar-title-container" style="text-align: center; font-size: 1.8rem; font-weight: 700; color: #1E88E5; margin-bottom: 1.5rem; padding: 0;">
    🔬 Wafer Defect Monitoring
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Landing Page Content
st.markdown('<h1 class="main-header">🔬 AI-Driven Wafer Defect Monitoring Framework using ML and LLM-POWERED AI AGENT</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A comprehensive semiconductor manufacturing monitoring system combining ML-based defect detection with LLM-powered intelligent analysis</p>', unsafe_allow_html=True)
# st.markdown("---")

# Features Section
st.markdown("## ✨ Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🏭 Manufacturing Simulation</h3>
        <ul>
            <li>Multi-Machine Simulation</li>
            <li>Real-time Processing</li>
            <li>Parallel Processing</li>
            <li>Comprehensive Logging</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 ML Defect Detection</h3>
        <ul>
            <li>ResNet18 CNN Model</li>
            <li>9 Defect Classes</li>
            <li>HSV-based Analysis</li>
            <li>Confidence Scoring</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>🧠 LLM-Powered AI</h3>
        <ul>
            <li>Natural Language Queries</li>
            <li>Multi-Physics Analysis</li>
            <li>Root Cause Explanations</li>
            <li>Automated Reports</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")
# Quick Navigation
st.markdown("## 🚀 Get Started")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Go to Dashboard", type="primary", use_container_width=True):
        try:
            st.switch_page("Pages/1_Dashboard.py")
        except:
            st.switch_page("pages/1_Dashboard.py")

with col2:
    if st.button("🤖 Defect Analytics", use_container_width=True):
        try:
            st.switch_page("Pages/2_DEFECT ANALYTICS.py")
        except:
            st.switch_page("pages/2_DEFECT ANALYTICS.py")

with col3:
    if st.button("📈 Defect Analytics", use_container_width=True):
        try:
            st.switch_page("Pages/3_AI_Assistant.py")
        except:
            st.switch_page("pages/3_AI_Assistant.py")

st.markdown("---")

# Additional Info
st.markdown("## 📋 System Capabilities")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("""
    **Defect Classes Detected:**
    - WM-811K Datasets (9 Classes)
    - Center, Donut, Edge-Loc, Edge-Ring, Local, Near-Full, Normal, Random, Scratch
    
    **Process Types:**
    - By Mechanical (Dicing, Grinding, Polishing)
    - By Electrical (Probe Testing, Parametric Testing)
    - By Thermal (Annealing, Stress Relief, Burn-in)
    """)

with info_col2:
    st.markdown("""
    **AI Features:**
    - Natural language query answering
    - Multi-physics root cause analysis
    - Automated daily summaries
    - PDF report generation
    - Real-time monitoring dashboard
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; font-size: 1.2rem;">
    <p><strong>This work was done for the Semiconductor Manufacturing Intelligence System Course Project</strong></p>
    <p>By: Iska (P86137210) , Firman (M38147023) , Indah Ayu (M38137028) </p>
    <p><strong> NATIONAL CHENG KUNG UNIVERSITY </strong></p>
    <p><strong> 2025 </strong></p>
</div>
""", unsafe_allow_html=True)

