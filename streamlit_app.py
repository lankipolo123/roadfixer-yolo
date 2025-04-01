from ultralytics import YOLO
import streamlit as st
from PIL import Image

# Load the YOLO model
@st.cache_resource
def load_model():
    model = YOLO("assets/80Yolov8.pt")
    return model

model = load_model()

# Streamlit title and file uploader
st.title("Pothole Detection Web App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run the YOLO model
    results = model(image)

    # Render the result
    if results:  # Check if results are not empty
        # Rendering the image with bounding boxes
        img_with_boxes = results[0].plot()  # Assuming the first result is the one you need
        st.image(img_with_boxes, caption="Detected Potholes", use_container_width=True)
    else:
        st.write("No potholes detected.")
