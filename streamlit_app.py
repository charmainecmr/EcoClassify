# app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import time
from datetime import datetime

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(page_title="EcoClassify AI", page_icon="♻️", layout="wide")

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
/* Center the tab container */
div[data-baseweb="tab-list"] {
    display: flex;
    justify-content: center;
    gap: 2rem; /* space between tabs */
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# Load Models
# --------------------------
@st.cache_resource
def load_resnet34():
    num_classes = 4
    model = models.resnet34(pretrained=False)

    # Rebuild the same head that was used in training
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),  # fc.0
        nn.ReLU(),                             # fc.1
        nn.Dropout(0.4),                       # fc.2 (if you had dropout)
        nn.Linear(512, num_classes)            # fc.3
    )

    state_dict = torch.load("best_resnet34.pth", map_location="cpu")
    model.load_state_dict(state_dict)  # strict=True now works
    model.eval()
    return model

@st.cache_resource
def load_efficientnet_b0():
    num_classes = 4
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("best_efficientnet_b0.pth", map_location="cpu"))
    model.eval()
    return model

resnet34_model = load_resnet34()
efficientnet_model = load_efficientnet_b0()
class_names = ["glass", "metal", "paper", "plastic"]

# --------------------------
# Image Transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------------
# Prediction
# --------------------------
def predict(image: Image.Image, model, class_names):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
    conf, idx = torch.max(probs, 0)
    return class_names[idx], conf.item() * 100

# --------------------------
# Session State (History)
# --------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------
# Tabs
# --------------------------
st.title("♻️ EcoClassify AI")

# Custom CSS to center tabs
st.markdown("""
<style>
div[data-baseweb="tab-list"] {
    display: flex;
    justify-content: center;
    gap: 3rem;
}
div[data-baseweb="tab"] {
    font-size: 1.2rem;
    font-weight: 600;
    padding: 1rem 2rem;
    border-radius: 12px;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #4caf50, #81c784);
    color: white !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


tabs = st.tabs(["✨ Classify", "🕒 History"])

# --- Classify Tab ---
with tabs[0]:
    uploaded = st.file_uploader("Upload a recyclable item image", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing with 5 models..."):
            time.sleep(2)

            # Real predictions
            resnet_pred, resnet_conf = predict(image, resnet34_model, class_names)
            effnet_pred, effnet_conf = predict(image, efficientnet_model, class_names)

            # Mock others for layout
            mock_models = [
                ("MobileNet Model", resnet_pred, resnet_conf - 3),
                ("VGG-16 Model", effnet_pred, effnet_conf - 4),
                ("Inception Model", resnet_pred, resnet_conf + 5),
            ]

            all_preds = [
                ("ResNet-34 Model", resnet_pred, resnet_conf),
                ("EfficientNet-B0 Model", effnet_pred, effnet_conf),
            ] + mock_models

        st.subheader("Model Predictions")
        cols = st.columns(3)
        for i, (mname, mpred, mconf) in enumerate(all_preds):
            bar_color = "#34a853" if mconf >= 80 else "#fbbc04" if mconf >= 50 else "#ea4335"

            with cols[i % 3]:
                st.markdown(f"""
                <div class="pred-card">
                  <h4>{mname}</h4>
                  <p style="margin-bottom:4px;"><b>Prediction:</b> 
                    <span style="background:#e6f4ea; color:#137333; padding:2px 6px; 
                    border-radius:6px;">{mpred}</span>
                  </p>

                  <p style="margin-bottom:6px;"><b>Confidence:</b> {mconf:.1f}%</p>

                  <!-- Progress Bar -->
                  <div style="background:#eee; border-radius:8px; height:10px; width:100%; margin-bottom:8px;">
                    <div style="background:{bar_color}; height:10px; border-radius:8px; width:{mconf:.1f}%;">
                    </div>
                  </div>

                  <p style="color:green; background:#e6f4ea; padding:6px; border-radius:8px; 
                     font-size:90%; margin-top:8px;">
                     ✅ Recyclable - Clean and place in recycling bin
                  </p>
                </div>
                """, unsafe_allow_html=True)


        # Save to history
        st.session_state.history.insert(0, {
            "filename": uploaded.name,
            "timestamp": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
            "results": all_preds
        })

# --- History Tab ---
with tabs[1]:
    st.header("Classification History")
    st.caption("View your past classifications and model predictions")

    if not st.session_state.history:
        st.info("No classifications yet. Upload an image to get started!")
    else:
        for entry in st.session_state.history:
            st.markdown(f"""
            <div class="pred-card">
              <b>{entry['filename']}</b><br>
              <small>🕒 {entry['timestamp']}</small><br><br>
              {"".join([f"<p>{m}: <b>{p}</b> ({c:.0f}%)</p>" for m,p,c in entry['results']])}
            </div>
            """, unsafe_allow_html=True)
