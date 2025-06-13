import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from mapping_info import soil_info
import os

# Load model
model = tf.keras.models.load_model("model_terrascan.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

# UI
st.title("🌍 TerraScan – Klasifikasi Jenis Tanah")
st.write("Upload foto tanah, dan sistem akan memprediksi jenis tanah, status kesuburannya, serta memberikan penjelasan.")

# Tambahkan informasi jenis tanah yang didukung
st.info(
    "📌 **Jenis tanah yang dapat dipindai oleh sistem ini:**\n\n"
    "- Aluvial\n"
    "- Gambut\n"
    "- Humus\n"
    "- Kapur\n"
    "- Latosol\n"
    "- Mergel\n"
    "- Podsol\n"
    "- Vulkanik"
)

uploaded_file = st.file_uploader("Upload Gambar Tanah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Preprocessing
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)  # Tambah batch dimensi
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Prediction
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    confidence = predictions[0][pred_index]
    predicted_label = labels[pred_index]

    # Get mapping info
    info = soil_info.get(predicted_label, {})
    status = "🌱 Subur" if info.get("subur") else "🚫 Tidak Subur"
    description = info.get("deskripsi", "-")

    # Display result
    st.markdown(f"### Jenis Tanah: **{predicted_label}**")
    st.markdown(f"### Status: **{status}**")
    st.markdown(f"**Deskripsi:** {description}")
    st.markdown(f"📊 Confidence: `{confidence:.2%}`")