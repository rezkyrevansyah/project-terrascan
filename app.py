import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from mapping_info import soil_info
import os

# Nonaktifkan logging TensorFlow yang tidak perlu
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Nonaktifkan log CUDA/CPU jika tidak menggunakan GPU

# Load model dengan caching untuk performa lebih baik
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model_terrascan.keras")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

# Load labels
@st.cache_data
def load_labels():
    try:
        with open("labels.txt", "r") as f:
            return f.read().splitlines()
    except Exception as e:
        st.error(f"Gagal memuat labels: {str(e)}")
        return []

model = load_model()
labels = load_labels()

# UI
st.title("🌍 TerraScan – Klasifikasi Jenis Tanah")
st.write("Upload foto tanah, dan sistem akan memprediksi jenis tanah, status kesuburannya, serta memberikan penjelasan.")

uploaded_file = st.file_uploader("Upload Gambar Tanah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Baca dan tampilkan gambar yang diunggah
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

        # Preprocessing Gambar
        img_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Prediksi
        if model is not None:
            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_index])  # Konversi ke float native Python
            predicted_label = labels[pred_index] if labels else "Label tidak ditemukan"

            # Ambil informasi dari soil_info
            info = soil_info.get(predicted_label, {})
            status = "🌱 Subur" if info.get("subur", False) else "🚫 Tidak Subur"
            description = info.get("deskripsi", "Tidak ada deskripsi tersedia.")

            # Tampilkan hasil
            st.markdown(f"### Jenis Tanah: **{predicted_label}**")
            st.markdown(f"### Status: **{status}**")
            st.markdown(f"**Deskripsi:** {description}")
            st.markdown(f"📊 Confidence: `{confidence:.2%}`")
        else:
            st.error("Model tidak berhasil dimuat. Silakan coba lagi atau periksa log.")
    except Exception as e:
        st.error(f"Terjadi error saat memproses gambar: {str(e)}")