import gradio as gr
import numpy as np
from PIL import Image

# Load model
from tensorflow.keras.models import load_model
model = load_model("brain_tumor_model.h5")

# Class labels
classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def predict(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(1, 128, 128, 3)
    pred = model.predict(img)
    return {classes[i]: float(pred[0][i]) for i in range(4)}

# Create Gradio interface
interface = gr.Interface(fn=predict,
                         inputs=gr.Image(type="pil"),
                         outputs=gr.Label(num_top_classes=4),
                         title="Brain Tumor MRI Classifier",
                         description="Upload an MRI scan to detect brain tumor type.")

interface.launch()
