import json
import os

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
from torchvision.models import resnet18

import face_utils

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
# ResNet18 v2, trained with emotions_resnet18_colab_v2.ipynb (FER+ + fine-tuning on facial_expressions)
MODEL_PATH = os.path.join(MODELS_DIR, "resnet18_emotions.pth")
META_PATH = os.path.join(MODELS_DIR, "resnet18_emotions.json")

FR = {
    "happiness": "Joie", "sadness": "Tristesse", "anger": "Colère", "surprise": "Surprise",
    "fear": "Peur", "disgust": "Dégoût", "neutral": "Neutre", "contempt": "Mépris",
}


class ResNetEmotion(nn.Module):
    """48x48 gray face -> ResNet18 at img_size, ImageNet-normalized, averaged with the mirrored face."""

    def __init__(self, num_classes, img_size):
        super().__init__()
        self.net = resnet18(weights=None)
        self.net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.net.fc.in_features, num_classes))
        self.img_size = img_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        return (self.net(x).softmax(1) + self.net(x.flip(-1)).softmax(1)) / 2


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])


@st.cache_resource
def load_model():
    with open(META_PATH) as f:
        meta = json.load(f)
    model = ResNetEmotion(len(meta["labels"]), meta["img_size"])
    model.net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, meta["labels"]


@st.cache_resource
def load_face_detector():
    return face_utils.FaceDetector()


def predict(face_img, model, labels):
    """Return {emotion: probability}, sorted from most to least likely."""
    with torch.no_grad():
        probs = model(transform(face_img).unsqueeze(0))[0].numpy()
    return dict(sorted(zip(labels, probs.astype(float)), key=lambda kv: -kv[1]))


def find_faces(img, max_faces=8):
    """Detect faces with YuNet and align each one like FER2013 faces. Returns [(box, aligned PIL image)]."""
    rgb = np.asarray(img.convert("RGB"))
    faces = load_face_detector().detect(np.ascontiguousarray(rgb[:, :, ::-1]))[:max_faces]
    return [(f.box, Image.fromarray(face_utils.align_face(rgb, f))) for f in faces]


def show_result(face_img, results, caption):
    top = next(iter(results))
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.image(face_img, caption=caption, width="stretch")
    with col2:
        st.markdown(f"## {FR[top]}")


# ---------------- UI ----------------
st.set_page_config(page_title="Détection d'émotions", layout="centered")
st.title("Détection d'émotions faciales")
model, labels = load_model()
emotions = [FR[e] for e in labels]
st.subheader(f"Votre visage exprime-t-il : {', '.join(emotions[:-1])} ou {emotions[-1]} ?")

source = st.radio("Source de l'image", ["Importer une photo", "Photo webcam"], horizontal=True)

img = None
if source == "Importer une photo":
    f = st.file_uploader("Choisissez une photo", type=["jpg", "jpeg", "png", "webp"])
    if f:
        img = Image.open(f)
else:
    f = st.camera_input("Prenez une photo")
    if f:
        img = Image.open(f)

if img is not None:
    img = ImageOps.exif_transpose(img).convert("RGB")  # phone photos: apply EXIF rotation
    faces = find_faces(img)
    if not faces:
        st.warning("Aucun visage détecté : l'image entière est analysée (résultat peu fiable).")
        show_result(img, predict(img, model, labels), "Image entière")
    else:
        results = [predict(face, model, labels) for _, face in faces]
        if len(faces) > 1:
            # Photo with numbered boxes
            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)
            lw = max(2, img.width // 250)
            for i, (x, y, w, h) in enumerate([b for b, _ in faces], start=1):
                draw.rectangle([x, y, x + w, y + h], outline=(0, 200, 0), width=lw)
                draw.text((x + lw + 2, y + lw), str(i), fill=(255, 255, 0), font_size=max(14, int(h / 5)))
            st.image(annotated, caption=f"{len(faces)} visages détectés", width="stretch")
        show_result(faces[0][1], results[0], "Visage 1" if len(faces) > 1 else "Visage analysé")
        if len(faces) > 1:
            st.subheader("Autres visages")
            cols = st.columns(min(4, len(faces) - 1))
            for i, ((_, face), res) in enumerate(zip(faces[1:], results[1:]), start=2):
                top = next(iter(res))
                with cols[(i - 2) % len(cols)]:
                    st.image(face, width="stretch")
                    st.markdown(f"**{i}.** {FR[top]}")
