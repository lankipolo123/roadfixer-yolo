import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load YOLO model (cached to avoid reloads)
@st.cache_resource
def load_model():
    return YOLO("assets/80Yolov8.pt")  # Ensure path is correct!

model = load_model()

# Streamlit UI
st.title("🚧 Pothole Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Convert PIL to numpy (YOLO expects numpy)
    image_np = np.array(image)

    # Run YOLO detection
    results = model(image_np)

    # Check if potholes detected
    if len(results[0].boxes) > 0:
        # Get annotated image (convert BGR to RGB for Streamlit)
        annotated_img = results[0].plot()[:, :, ::-1]
        st.image(annotated_img, caption="Detected Potholes", use_column_width=True)

        # Show detection details
        st.subheader("Detection Results")
        for box in results[0].boxes:
            conf = box.conf.item()  # Confidence score
            cls_id = box.cls.item()  # Class ID (0 if only 'pothole' class)
            st.write(f"- Pothole detected with confidence: {conf:.2f}")
    else:
        st.warning("No potholes detected. Try another image.")