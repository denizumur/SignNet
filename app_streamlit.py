import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from PIL import Image

# Sınıf etiketleri
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh > 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Veh > 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}

# Sayfa ayarı
st.set_page_config(page_title="🚦 Trafik İşareti Tanıma", page_icon="🚗", layout="centered")
st.markdown("<h1 style='text-align:center;color:#ff4b4b;'>🚦 Trafik İşareti Sınıflandırma</h1>", unsafe_allow_html=True)
st.markdown("Trafik işareti fotoğrafı yükleyin, model ne olduğunu tahmin etsin!")

# Modeli yükle
@st.cache_resource
def load_trained_model():
    return load_model("traffic_sign_model.keras")

model = load_trained_model()

# Görsel yükle
uploaded_file = st.file_uploader("Bir trafik işareti fotoğrafı yükleyin", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Tahmin görselini hazırla
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (50, 50))
    img_input = np.expand_dims(img_resized, axis=0) / 255.0

    predictions = model.predict(img_input)
    class_id = np.argmax(predictions)
    probability = np.max(predictions)

    st.markdown(f"### **Tahmin:** `{classes[class_id]}`")
    st.markdown(f"### **Thmin Güven Skoru:** `{probability * 100:.2f}%`")
