import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from mapping_info import soil_info
import os

# Nonaktifkan logging TensorFlow yang tidak perlu
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

# Daftar jenis tanah yang didukung
SUPPORTED_SOILS = [
    "Aluvial", "Gambut", "Humus", 
    "Kapur", "Latosol", "Mergel",
    "Podsol", "Vulkanik"
]

# UI
st.title("🌍 TerraScan – Klasifikasi Jenis Tanah")
st.write("Upload foto tanah, dan sistem akan memprediksi jenis tanah, status kesuburannya, serta memberikan penjelasan. *app masih proses improvement")

# Tampilkan informasi jenis tanah yang didukung
with st.expander("📋 Jenis Tanah yang Dapat Diprediksi", expanded=True):
    st.write("Aplikasi ini dapat mengenali jenis tanah berikut:")
    cols = st.columns(2)
    for i, soil_type in enumerate(SUPPORTED_SOILS):
        cols[i % 2].markdown(f"- {soil_type}")
    st.info("""Pastikan:
    - Gambar tanah difoto secara close-up
    - Pencahayaan cukup
    - Background netral/tidak ramai""")

uploaded_file = st.file_uploader("Upload Gambar Tanah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Baca dan tampilkan gambar yang diunggah
        image = Image.open(uploaded_file).convert("RGB")
        
        # Perbaikan: Gunakan parameter width bukan use_container_width
        st.image(
            image, 
            caption="Gambar yang diunggah",
            width=400  # Atur lebar gambar (bisa disesuaikan)
        )

        # Preprocessing Gambar
        img_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Prediksi
        if model is not None:
            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_index])
            predicted_label = labels[pred_index] if labels else "Label tidak ditemukan"

            # Validasi hasil prediksi
            if predicted_label not in SUPPORTED_SOILS:
                st.warning(f"⚠️ Hasil prediksi '{predicted_label}' tidak termasuk dalam daftar yang didukung")

            # Ambil informasi dari soil_info
            info = soil_info.get(predicted_label, {})
            status = "🌱 Subur" if info.get("subur", False) else "🚫 Tidak Subur"
            description = info.get("deskripsi", "Tidak ada deskripsi tersedia.")

            # Tampilkan hasil
            st.success("✅ Analisis Berhasil!")
            st.markdown(f"### Jenis Tanah: **{predicted_label}**")
            st.markdown(f"### Status: **{status}**")
            st.markdown(f"**Deskripsi:** {description}")
            st.markdown(f"**Tingkat Akurasi:** `{confidence:.2%}`")
            
            # Tambahkan visualisasi confidence
            st.progress(float(confidence))
            
        else:
            st.error("Model tidak berhasil dimuat. Silakan coba lagi atau periksa log.")
    except Exception as e:
        st.error(f"Terjadi error saat memproses gambar: {str(e)}")

# Tambahkan footer
st.markdown("---")
st.caption("© 2024 TerraScan - Aplikasi Klasifikasi Jenis Tanah")