import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import time
from datetime import datetime
import base64, io

# --------------------------
# Configure PyTorch for safe loading
# --------------------------
torch.serialization.add_safe_globals([
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.activation.Hardswish,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.upsampling.Upsample,
    torch.nn.Sequential,
    torch.nn.ModuleList,
])

# Page Setup
st.set_page_config(page_title="EcoClassify AI", layout="wide")

# Title
st.markdown("""
    <h1 style="text-align: center; font-size: 3rem; color: #388E3C;">EcoClassify AI</h1>
    <h3 style="text-align: center; font-size: 1.5rem; color: #616161;">Classifying recyclable materials using AI models</h3>
""", unsafe_allow_html=True)

# Custom CSS
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
@st.cache_resource
def load_resnet34():
    num_classes = 4
    model = models.resnet34(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    state_dict = torch.load("models/best_resnet34.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_resource
def load_efficientnet_b0():
    num_classes = 4
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("models/best_efficientnet_b0.pth", map_location="cpu", weights_only=False))
    model.eval()
    return model

@st.cache_resource
def load_lenet():
    num_classes = 4
    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 5)
            self.fc1 = nn.Linear(32 * 53 * 53, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = LeNet()
    model.load_state_dict(torch.load("models/best_lenet.pth", map_location="cpu", weights_only=False))
    model.eval()
    return model

@st.cache_resource
def load_yolo():
    try:
        # Register YOLO-specific classes as safe globals
        try:
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
        except ImportError:
            pass
        
        # Try to register common ultralytics modules
        try:
            import ultralytics.nn.modules as nn_modules
            module_classes = []
            for name in dir(nn_modules):
                attr = getattr(nn_modules, name)
                if isinstance(attr, type):
                    module_classes.append(attr)
            if module_classes:
                torch.serialization.add_safe_globals(module_classes)
        except Exception:
            pass

        import warnings
        from ultralytics import YOLO
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Load with weights_only=False to bypass the safe globals requirement
            model = YOLO("models/best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        st.info("YOLO model will be skipped. Other models will continue to work.")
        return None

@st.cache_resource
def load_mobilenet():
    num_classes = 4
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state_dict = torch.load("models/best_mobilenet.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Load all models
resnet34_model = load_resnet34()
efficientnet_model = load_efficientnet_b0()
lenet_model = load_lenet()
yolo_model = load_yolo()
mobilenet_model = load_mobilenet()

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
# Predictions
# --------------------------
def predict(image: Image.Image, model, class_names):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
    conf, idx = torch.max(probs, 0)
    return class_names[idx], conf.item() * 100

def predict_yolo(image: Image.Image):
    if yolo_model is None:
        return "YOLO model not available", 0
    try:
        results = yolo_model.predict(image, conf=0.25, imgsz=640, verbose=False)
        if len(results[0].boxes) > 0:
            class_id = int(results[0].boxes.cls[0].item())
            conf = float(results[0].boxes.conf[0].item()) * 100
            label = results[0].names[class_id]
            return label, conf
        else:
            return "No object detected", 0
    except Exception as e:
        return f"Error: {str(e)}", 0

# --------------------------
# Session State
# --------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------
# Tabs
# --------------------------
tabs = st.tabs(["Classify", "History"])

# Class colors
class_colors = {
    "glass":   {"bg": "#e3f2fd", "text": "#1565c0"},
    "metal":   {"bg": "#ede7f6", "text": "#4527a0"},
    "paper":   {"bg": "#fff3e0", "text": "#e65100"},
    "plastic": {"bg": "#fce4ec", "text": "#ad1457"},
}

# --------------------------
# Model → Weights mapping
# --------------------------
model_weights = {
    "ResNet-34 Model": "models/best_resnet34.pth",
    "EfficientNet-B0 Model": "models/best_efficientnet_b0.pth",
    "LeNet Model": "models/best_lenet.pth",
    "YOLOv8 Model": "models/best.pt",
    "MobileNet-V2 Model": "models/best_mobilenet.pth"
}

with tabs[0]:
    uploaded = st.file_uploader("Upload a recyclable item image", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        # Preview
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

        with st.spinner("Analyzing with models..."):
            time.sleep(2)
            resnet_pred, resnet_conf   = predict(image, resnet34_model, class_names)
            effnet_pred, effnet_conf   = predict(image, efficientnet_model, class_names)
            lenet_pred, lenet_conf     = predict(image, lenet_model, class_names)
            mobilenet_pred, mobilenet_conf = predict(image, mobilenet_model, class_names)
            
            # Only predict with YOLO if it loaded successfully
            if yolo_model is not None:
                yolo_pred, yolo_conf = predict_yolo(image)
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

        # --------------------------
        # Show Model Predictions
        # --------------------------
        st.subheader("Model Predictions")
        cols = st.columns(2)
        for i, (mname, mpred, mconf) in enumerate(all_preds):
            bar_color = "#34a853" if mconf >= 80 else "#fbbc04" if mconf >= 50 else "#ea4335"
            colors = class_colors.get(mpred.lower(), {"bg": "#e6f4ea", "text": "#137333"})

            with cols[i % 2]:
                with st.container():
                    # Open card
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

                    # Download button inside card
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

        # Save history
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