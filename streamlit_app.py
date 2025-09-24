# Create and activate virtual environment: python -m venv venv 
# Install Dependencies: pip install -r requirements.txt
# Run app: streamlit run streamlit_app.py

# streamlit_app.py
import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO
import datetime
import random

import torch
from torch.serialization import add_safe_globals
import ultralytics.nn.tasks
add_safe_globals([ultralytics.nn.tasks.DetectionModel])

from streamlit_option_menu import option_menu
# streamlit_app.py
import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO
import datetime
import random
from streamlit_option_menu import option_menu # Import the library

import torch
from torch.serialization import add_safe_globals
import ultralytics.nn.tasks
add_safe_globals([ultralytics.nn.tasks.DetectionModel])


# --------------------------
# Setup Page
# --------------------------
st.set_page_config(
    page_title="EcoClassify - Recycling Assistant",
    page_icon="♻️",
    layout="centered"
)

# --------------------------
# Custom CSS for styling
# --------------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom right, #f7f9fc, #eaf5f2);
            font-family: "Segoe UI", sans-serif;
        }
        .card {
            background: #ffffff;
            border-radius: 10px;
            padding: 2rem;
            margin: 2rem auto;
            max-width: 800px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .card:hover {
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }
        .title {
            font-size: 2rem;
            font-weight: bold;
            color: #2e7d32;
        }
        .subtitle {
            font-size: 1rem;
            color: #555;
        }
        .result-success {
            border-left: 6px solid #4caf50;
            padding-left: 1rem;
        }
        .result-error {
            border-left: 6px solid #f44336;
            padding-left: 1rem;
        }
        .result-warning {
            border-left: 6px solid #ff9800;
            padding-left: 1rem;
        }
        .history-item {
            background: #fafafa;
            padding: 0.8rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Apply animated gradient to full page */
html, body, [data-testid="stAppViewContainer"] {
    height: 100%;
    margin: 0;
    padding: 0;
    animation: gradientShift 20s ease infinite;
    background: linear-gradient(-45deg, #e8f5e9, #f1f8e9, #c8e6c9, #dcedc8);
    background-size: 400% 400%;
}

/* Animate the gradient */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Optional: make cards and containers pop */
[data-testid="stAppViewBlockContainer"] {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Animated Gradient Background
# --------------------------
st.markdown("""
<style>
body {
    animation: gradientShift 15s ease infinite;
    background: linear-gradient(-45deg, #e8f5e9, #f1f8e9, #c8e6c9, #dcedc8);
    background-size: 400% 400%;
    font-family: "Segoe UI", sans-serif;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load YOLO Model
# --------------------------
@st.cache_resource
def load_model():
    try:
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.warning("⚠️ Model failed to load. Placeholder mode activated. You can still explore the UI.")
        return None
    
model = load_model()

# --------------------------
# Session state for history
# --------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# --------------------------
# Recycling logic (YOLO only)
# --------------------------
def determine_recyclability(results, model):
    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id].lower()
        confidence = float(box.conf[0].item())

        recyclable_keywords = ["plastic", "glass", "metal", "paper", "cardboard"]
        non_recyclable_keywords = ["food", "battery", "electronics", "ceramic"]

        if any(k in label for k in recyclable_keywords):
            result = "recyclable"
        elif any(k in label for k in non_recyclable_keywords):
            result = "non-recyclable"
        else:
            result = "uncertain"

        return {"result": result, "category": label, "confidence": confidence}
    else:
        return {"result": "uncertain", "category": "none", "confidence": 0.0}

# --------------------------
# Styled Title Header
# --------------------------
st.markdown("""
<div style="
    background: linear-gradient(to right, #e8f5e9, #f1f8e9);
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
">
    <h1 style="color: #2e7d32; font-size: 2.5rem; margin-bottom: 0.5rem;"> EcoClassify</h1>
    <p style="color: #4e944f; font-size: 1.2rem;">AI-Powered Recycling Assistant</p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Navigation Bar
# --------------------------
view = option_menu(
    menu_title=None,
    options=["Classify", "History"],
    icons=["camera", "list-check"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa", "border-radius": "10px"},
        "icon": {"color": "green", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "padding": "10px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#c8e6c9"},
    }
)

# --------------------------
# Classify View
# --------------------------
if view == "Classify":
    st.markdown("""
    <div class="card" style="text-align:center;">
        <h2>Is it recyclable?</h2>
        <p>Upload a photo of any item and our AI will instantly tell you if it's recyclable, 
        along with helpful recycling tips.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        # Run YOLO prediction
        if model:
            with st.spinner("Analyzing image..."):
                results = model.predict(image, conf=0.25)
                parsed = determine_recyclability(results, model)
        else:
            # Fallback if model failed to load
            parsed = {
                "result": "uncertain",
                "category": "placeholder",
                "confidence": 0.0
            }


        # Result card styling
        if parsed["result"] == "recyclable":
            css_class = "result-success"
            message = "Great! This item can be recycled. Make sure to clean it properly before placing it in your recycling bin."
        elif parsed["result"] == "non-recyclable":
            css_class = "result-error"
            message = "This item cannot be recycled through regular curbside recycling. Consider alternative disposal methods."
        else:
            css_class = "result-warning"
            message = "We're not certain about this item. When in doubt, check with your local recycling guidelines."

        st.markdown(f"""
        <div class="card {css_class}">
            <h3>Prediction: {parsed['result'].capitalize()} ({parsed['category']})</h3>
            <p>Confidence: <b>{parsed['confidence']*100:.2f}%</b></p>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)

        # Save to history
        st.session_state["history"].insert(0, {
            "filename": uploaded_file.name,
            "prediction": parsed['result'],
            "category": parsed['category'],
            "confidence": f"{parsed['confidence']*100:.2f}",
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# --------------------------
# History View
# --------------------------
elif view == "History":
    st.markdown('<div class="card"><h2>Classification History</h2></div>', unsafe_allow_html=True)

    if len(st.session_state["history"]) == 0:
        st.info("No classifications yet. Upload an image in the 'Classify' tab.")
    else:
        for item in st.session_state["history"]:
            st.markdown(f"""
            <div class="history-item">
                <b>Prediction:</b> {item['prediction'].capitalize()} ({item['category']})<br>
                <b>Time:</b> {item['time']}<br>
                <b>Confidence:</b> {item['confidence']}%<br>
                <b>File:</b> {item['filename']}
            </div>
            """, unsafe_allow_html=True)
