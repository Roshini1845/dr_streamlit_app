import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

# Download model from Google Drive if not already present
model_path = 'dr_model.h5'
file_id = '1Zr9GlvsuogIM6_r5dswTJDwyeCahDylC'
if not os.path.exists(model_path):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# Class labels
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

st.title("👁 Diabetic Retinopathy Detection")
st.write("Upload a retina image to detect DR severity.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    image = image.resize((128, 128))  # Ensure this matches your model input size
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(image_array)
    result = class_names[np.argmax(prediction)]

    st.markdown(f"### 🧠 Prediction: **{result}**")
