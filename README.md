# 🧠 AI Powered Brain Tumor Detection from MRI

This project leverages deep learning to detect and classify brain tumors from MRI scans. Using a Convolutional Neural Network (CNN) trained on publicly available medical imaging datasets, the model can identify four different conditions: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**.

---

## 🩻 Problem Statement

Brain tumors are a serious health issue and often require early diagnosis for effective treatment. Manual interpretation of MRI scans by radiologists can be time-consuming, expensive, and prone to human error—especially in under-resourced areas.

---

## 💡 Solution

This project offers an AI-driven web application that:
- Accepts MRI scan images as input.
- Preprocesses the image for optimal model input.
- Predicts the tumor type (or absence of one) using a trained CNN model.
- Delivers results instantly via an interactive user interface (Gradio/Streamlit).

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow** / **Keras** – Model building and training
- **OpenCV** – Image processing
- **NumPy** – Numerical operations
- **Gradio** / **Streamlit** – Web deployment
- **Pillow** – Image handling

---

## 🗂 Dataset

- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes**: 
  - Glioma Tumor  
  - Meningioma Tumor  
  - Pituitary Tumor  
  - No Tumor  
- The dataset includes thousands of labeled MRI images distributed across these four categories.

---

## 📈 Model Architecture

A Convolutional Neural Network (CNN) with the following layers:

- 3 Convolution + MaxPooling layers
- Flattening Layer
- Dense Layer with ReLU activation
- Dropout for regularization
- Final Dense Layer with Softmax activation (for classification into 4 categories)

---

## 🔍 How It Works

1. The user uploads an MRI image via a simple UI.
2. The image is resized and normalized to match the model’s input expectations.
3. The model predicts the tumor category.
4. The prediction is displayed back to the user with the label.

---

## 🚀 Deployment Options

### Gradio (Recommended for Live Demo)

```bash
pip install -r requirements.txt
python app_gradio.py
