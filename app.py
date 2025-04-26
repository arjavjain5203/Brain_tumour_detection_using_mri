import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("brain_tumor_model.h5")
classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI scan image to detect the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    # Preprocess
    img = image.resize((128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    pred = model.predict(img)
    prediction = {classes[i]: float(pred[0][i]) for i in range(4)}
    
    st.subheader("Prediction:")
    st.json(prediction)
