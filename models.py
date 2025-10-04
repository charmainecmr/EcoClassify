# models.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import streamlit as st

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

# --------------------------
# LeNet Architecture
# --------------------------
class LeNet(nn.Module):
    def __init__(self, num_classes=4):
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

# --------------------------
# Model Loading Functions
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
    model = LeNet(num_classes)
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
# Prediction Functions
# --------------------------
def predict(image: Image.Image, model, class_names):
    """Make prediction using PyTorch models"""
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
    conf, idx = torch.max(probs, 0)
    return class_names[idx], conf.item() * 100

def predict_yolo(image: Image.Image, yolo_model):
    """Make prediction using YOLO model"""
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