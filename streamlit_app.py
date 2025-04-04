import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load YOLO model (cached to avoid reloads)
@st.cache_resource
def load_model():
    # Ensure the correct path to the model
    return YOLO("assets/80Yolov8.pt")

model = load_model()

# Streamlit UI
st.title("🚧 Pothole Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Convert PIL image to numpy array (YOLO expects numpy arrays, not PIL images)
    image_np = np.array(image)

    # Run YOLO detection
    results = model(image_np)

    # Check if potholes were detected
    if len(results[0].boxes) > 0:
        # Get annotated image (convert from BGR to RGB for Streamlit display)
        annotated_img = results[0].plot()[:, :, ::-1]  # Convert BGR to RGB
        st.image(annotated_img, caption="Detected Potholes", use_column_width=True)

        # Show detection details
        st.subheader("Detection Results")
        for box in results[0].boxes:
            conf = box.conf.item()  # Confidence score
            cls_id = box.cls.item()  # Class ID (usually 0 for potholes)
            st.write(f"- Pothole detected with confidence: {conf:.2f}")
    else:
        st.warning("No potholes detected. Try another image.")
