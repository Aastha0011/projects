import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="COVID-19 Detection", page_icon="🩻")

st.title("🩻 COVID-19 Detection from Chest X-ray")

# ------------------------
# Debug Information
# ------------------------
st.write("Current Working Directory:")
st.code(os.getcwd())

st.write("Files in Current Directory:")
st.write(os.listdir("."))

# ------------------------
# Locate model.keras
# ------------------------
MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

IMG_SIZE = (299, 299)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    prob = float(prediction[0][0])

    if prob > 0.5:
        st.error(f"🦠 COVID-19 Detected\n\nConfidence: {prob*100:.2f}%")
    else:
        st.success(f"✅ Normal\n\nConfidence: {(1-prob)*100:.2f}%")
