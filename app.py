import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras import backend as K

# Custom Metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

# Load the model with custom objects
@st.cache_resource
def load_model_custom():
    custom_objects = {
        "f1_m": f1_m,
        "precision_m": precision_m,
        "recall_m": recall_m,
        "dsc": dsc
    }
    return load_model(r"C:\Users\parid\Desktop\domain\flood_save.keras", custom_objects=custom_objects)

# Set up the Streamlit app
st.set_page_config(
    page_title="Flood Segmentation",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #e6f7ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stFileUploader>div>label {
        background-color: #2196F3;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.image(
    "https://media.gettyimages.com/id/135891916/photo/flood-in-bangkok.jpg?s=612x612&w=0&k=20&c=K4VBPxsEg3t1LnNLQMEz5Lcp05snzcQUhIoYOI7iBJs=",
    use_container_width=True  # Updated parameter
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<h3 style='text-align: center; color: #33679a;'><b>Get in Touch</b></h3>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f"<p style='text-align: center; font-size: 14px;'>"
    f"📈 <a href='https://www.linkedin.com/in/birendra-parida-121a71329'>LinkedIn</a> | "
    f"📧 <a href='mailto:paridabirendra890@gmail.com.com'>Email</a></p>",
    unsafe_allow_html=True
)

# Main content
st.title("🌊 Flood Area Segmentation App")
st.markdown(
    """
    <p style='font-size: 18px; font-style: italic;'>
        Detect high-risk flood areas with AI - faster and more accurately!
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# File uploader with emoji
uploaded_file = st.file_uploader(
    ":file_folder: Choose an image...", 
    type=["jpg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    st.image(img, caption="🌊 Uploaded Image", use_container_width=True  # Updated parameter
)

    # Preprocess the image
    img_resized = cv2.resize(img, (512, 512))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict button
    if st.button("✨ Predict Flood Areas"):
        with st.spinner("Detecting flood areas... ⏳"):
            try:
                model = load_model_custom()
                pred = model.predict(img_input)
                pred_binary = np.where(pred > 0.5, 1, 0).squeeze()  # Convert to binary mask

                # Display results
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(img)
                axes[0].set_title("Original Image", fontsize=14)
                axes[0].axis("off")

                axes[1].imshow(pred_binary, cmap="Blues")
                axes[1].set_title("Flood Segmentation", fontsize=14)
                axes[1].axis("off")

                st.pyplot(fig)

                # Additional statistics
                st.info(f"⚡ Model Output Shape: {pred.shape}")
                st.info(f"💧 Binary Mask Shape: {pred_binary.shape}")

            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")

st.markdown("---")
st.markdown(
    """
    ### 📌 How it works
    1. Upload an aerial image or satellite map
    2. Click the **Predict Flood Areas** button
    3. See the flood risk areas highlighted in blue
    """
)

# Footer
st.markdown(
    """
    <footer style='text-align: center; padding: 10px;'>
        <p style='color: #33679a;'>
            <small>
                Made with ❤️ by Birendra kumar parida | 2025 | 
            </small>
        </p>
    </footer>
    """,
    unsafe_allow_html=True
)