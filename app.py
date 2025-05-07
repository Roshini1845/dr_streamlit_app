import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model('dr_model.h5')

# Class labels
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

st.title("👁 Diabetic Retinopathy Detection")
st.write("Upload a retina image to detect DR severity.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    image = image.resize((128, 128))  # Update if your model uses another size
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(image_array)
    result = class_names[np.argmax(prediction)]

    st.markdown(f"### 🧠 Prediction: **{result}**")
