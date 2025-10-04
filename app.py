# app.py
import streamlit as st
from PIL import Image
import time
from datetime import datetime
import base64
import io

# Import model functions from models.py
from models import (
    load_resnet34,
    load_efficientnet_b0,
    load_lenet,
    load_yolo,
    load_mobilenet,
    predict,
    predict_yolo
)

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(page_title="EcoClassify AI", layout="wide")

# --------------------------
# Title
# --------------------------
st.markdown("""
    <h1 style="text-align: center; font-size: 3rem; color: #388E3C;">EcoClassify AI</h1>
    <h3 style="text-align: center; font-size: 1.5rem; color: #616161;">Classifying recyclable materials using AI models</h3>
""", unsafe_allow_html=True)

# --------------------------
# Custom CSS
# --------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(-45deg, #e8f5e9, #f1f8e9, #e0f7fa, #f3e5f5);
  background-size: 400% 400%;
  animation: gradientShift 20s ease infinite;
}
@keyframes gradientShift {
  0% {background-position:0% 50%;}
  50% {background-position:100% 50%;}
  100% {background-position:0% 50%;}
}
.pred-card {
  background: white;
  border-radius: 1rem;
  padding: 1rem;
  box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  margin-bottom: 1rem;
  transition: all 0.3s ease;
}
.pred-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}
div[data-baseweb="tab-list"] {
    display: flex;
    justify-content: center;
    gap: 2rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load Models
# --------------------------
resnet34_model = load_resnet34()
efficientnet_model = load_efficientnet_b0()
lenet_model = load_lenet()
yolo_model = load_yolo()
mobilenet_model = load_mobilenet()

# --------------------------
# Constants
# --------------------------
class_names = ["glass", "metal", "paper", "plastic"]

class_colors = {
    "glass":   {"bg": "#e3f2fd", "text": "#1565c0"},
    "metal":   {"bg": "#ede7f6", "text": "#4527a0"},
    "paper":   {"bg": "#fff3e0", "text": "#e65100"},
    "plastic": {"bg": "#fce4ec", "text": "#ad1457"},
}

model_weights = {
    "ResNet-34 Model": "models/best_resnet34.pth",
    "EfficientNet-B0 Model": "models/best_efficientnet_b0.pth",
    "LeNet Model": "models/best_lenet.pth",
    "YOLOv8 Model": "models/best.pt",
    "MobileNet-V2 Model": "models/best_mobilenet.pth"
}

# --------------------------
# Session State
# --------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------
# Tabs
# --------------------------
tabs = st.tabs(["Classify", "History"])

# --------------------------
# CLASSIFY TAB
# --------------------------
with tabs[0]:
    uploaded = st.file_uploader("Upload a recyclable item image", type=["jpg", "jpeg", "png"])
    
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        # Image Preview
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; align-items:center; flex-direction:column; margin:20px 0;">
                <div style="background:white; border-radius:1rem; padding:12px; box-shadow:0 4px 15px rgba(0,0,0,0.1);">
                    <img src="data:image/png;base64,{img_b64}" width="300" style="border-radius:12px;"/>
                </div>
                <p style="text-align:center; font-size:0.9rem; color:gray; margin-top:8px;">Uploaded Image</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Make Predictions
        with st.spinner("Analyzing with models..."):
            time.sleep(2)
            resnet_pred, resnet_conf = predict(image, resnet34_model, class_names)
            effnet_pred, effnet_conf = predict(image, efficientnet_model, class_names)
            lenet_pred, lenet_conf = predict(image, lenet_model, class_names)
            mobilenet_pred, mobilenet_conf = predict(image, mobilenet_model, class_names)
            
            # Only predict with YOLO if it loaded successfully
            if yolo_model is not None:
                yolo_pred, yolo_conf = predict_yolo(image, yolo_model)
                all_preds = [
                    ("ResNet-34 Model", resnet_pred, resnet_conf),
                    ("EfficientNet-B0 Model", effnet_pred, effnet_conf),
                    ("LeNet Model", lenet_pred, lenet_conf),
                    ("YOLOv8 Model", yolo_pred, yolo_conf),
                    ("MobileNet-V2 Model", mobilenet_pred, mobilenet_conf),
                ]
            else:
                all_preds = [
                    ("ResNet-34 Model", resnet_pred, resnet_conf),
                    ("EfficientNet-B0 Model", effnet_pred, effnet_conf),
                    ("LeNet Model", lenet_pred, lenet_conf),
                    ("MobileNet-V2 Model", mobilenet_pred, mobilenet_conf),
                ]

        # Display Model Predictions
        st.subheader("Model Predictions")
        cols = st.columns(2)
        
        for i, (mname, mpred, mconf) in enumerate(all_preds):
            bar_color = "#34a853" if mconf >= 80 else "#fbbc04" if mconf >= 50 else "#ea4335"
            colors = class_colors.get(mpred.lower(), {"bg": "#e6f4ea", "text": "#137333"})

            with cols[i % 2]:
                with st.container():
                    st.markdown(f"""
                    <div class="pred-card">
                      <h4>{mname}</h4>
                      <p style="margin-bottom:4px;"><b>Prediction:</b> 
                        <span style="background:{colors['bg']}; color:{colors['text']}; 
                            padding:2px 6px; border-radius:6px;">{mpred}</span>
                      </p>
                      <p style="margin-bottom:6px;"><b>Confidence:</b> {mconf:.1f}%</p>
                      <div style="background:#eee; border-radius:8px; height:10px; width:100%; margin-bottom:8px;">
                        <div style="background:{bar_color}; height:10px; border-radius:8px; width:{mconf:.1f}%;"></div>
                      </div>
                    """, unsafe_allow_html=True)

                    # Download button for model weights
                    weight_path = model_weights.get(mname)
                    if weight_path:
                        try:
                            with open(weight_path, "rb") as f:
                                btn_bytes = f.read()
                            st.download_button(
                                label=f"↑ Download {mname} Weights",
                                data=btn_bytes,
                                file_name=weight_path.split("/")[-1],
                                mime="application/octet-stream",
                                key=f"download-{mname}"
                            )
                        except FileNotFoundError:
                            pass
                    
                    st.markdown("</div>", unsafe_allow_html=True)

        # Save to history
        st.session_state.history.insert(0, {
            "filename": uploaded.name,
            "timestamp": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
            "results": all_preds
        })

# --------------------------
# HISTORY TAB
# --------------------------
with tabs[1]:
    st.header("Classification History")
    st.caption("View your past classifications and model predictions")
    
    if not st.session_state.history:
        st.info("No classifications yet. Upload an image to get started!")
    else:
        for entry in st.session_state.history:
            st.markdown(f"""
            <div class="pred-card" style="padding:1.5rem; margin-bottom:1.5rem;">
              <b style="font-size:1.1rem;">{entry['filename']}</b><br>
              <small style="color:gray;"> {entry['timestamp']}</small><br><br>
            """, unsafe_allow_html=True)

            for m, p, c in entry["results"]:
                colors = class_colors.get(p.lower(), {"bg": "#e6f4ea", "text": "#137333"})
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                    background:#f9f9f9; border-radius:8px; padding:10px 14px; margin-bottom:8px;
                    box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);">
                    <span style="font-weight:500;">{m}</span>
                    <span>
                        <span style="background:{colors['bg']}; color:{colors['text']};
                            padding:3px 10px; border-radius:6px; margin-right:8px;">{p}</span>
                        <b>{c:.0f}%</b>
                    </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)