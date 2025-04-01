import base64
import io
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

# Function to convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

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

        # Convert the result image to base64
        img_base64 = image_to_base64(img_with_boxes)

        # Send the base64 string to be consumed by Flutter
        st.write("Base64 Encoded Image:")
        st.text(img_base64)  # You can provide this string to your Flutter app
    else:
        st.write("No potholes detected.")
