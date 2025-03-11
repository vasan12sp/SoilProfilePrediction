import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import gaussian_filter1d

# Load TFLite Models
@st.cache_resource
def load_tflite_models():
    soil_type_interpreter = tf.lite.Interpreter(model_path="soil_type_model.tflite")
    soil_density_interpreter = tf.lite.Interpreter(model_path="soil_density_model.tflite")
    
    # Allocate tensors
    soil_type_interpreter.allocate_tensors()
    soil_density_interpreter.allocate_tensors()
    
    return soil_type_interpreter, soil_density_interpreter

soil_type_interpreter, soil_density_interpreter = load_tflite_models()

# Function to resize and normalize the image
def preprocess_image(image):
    IMG_SIZE = (224, 224)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0).astype(np.float32)

# Function to apply gamma correction
def apply_gamma_correction(image, gamma=1.4):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function to segment image into soil layers
def segment_image(image, slice_height=50, smoothing_sigma=2):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    num_slices = height // slice_height

    mean_intensity = np.array([np.mean(gray[i * slice_height:(i + 1) * slice_height, :]) for i in range(num_slices)])
    smooth_intensity = gaussian_filter1d(mean_intensity, sigma=smoothing_sigma)
    centerline = np.mean(smooth_intensity)

    crossing_points = []
    for i in range(1, len(smooth_intensity)):
        if (smooth_intensity[i - 1] < centerline and smooth_intensity[i] > centerline) or \
           (smooth_intensity[i - 1] > centerline and smooth_intensity[i] < centerline):
            crossing_points.append(i * slice_height)

    min_layer_gap = 80
    filtered_boundaries = []
    prev_y = -min_layer_gap

    for y in crossing_points:
        if y - prev_y > min_layer_gap:
            filtered_boundaries.append(y)
            prev_y = y

    return filtered_boundaries

# Function to overlay segmentation lines
def draw_segmentation_lines(image, boundaries):
    output_image = image.copy()
    for y in boundaries:
        cv2.line(output_image, (0, y), (image.shape[1], y), (255, 0, 0), 2)  # Red line
    return output_image

# Function to run inference on TFLite models
def run_tflite_model(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Function to predict soil type and density using TFLite models
def predict_soil(image):
    image = preprocess_image(image)
    
    # Predict soil type (Classification)
    soil_type_pred = run_tflite_model(soil_type_interpreter, image)
    soil_types = ["Alluvial soil", "Black soil", "Clay soil", "Red soil"]
    predicted_class = np.argmax(soil_type_pred)
    predicted_soil_name = soil_types[predicted_class]

    # Predict soil density (Regression)
    soil_density_pred = run_tflite_model(soil_density_interpreter, image)
    predicted_density = soil_density_pred[0][0]

    return predicted_soil_name, predicted_density

# Streamlit UI
st.title("🌱 Soil Type & Density Detection")
st.write("Upload a soil profile image to analyze its soil type and density.")

st.warning("**Note:** The input image is assumed to contain no shadows or foreign objects, as these may affect the model’s accuracy. The predictions for soil type and density are approximate and should not be considered as precise scientific measurements.")


# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
image = None

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if st.button("Predict"):
        st.subheader("Segmentation Results")

        # Segment the image
        segmented_boundaries = segment_image(image)
        segmented_image = draw_segmentation_lines(image, segmented_boundaries)

        # Show original and segmented images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(segmented_image, caption="Segmented Image", use_container_width=True)

        # Extract segments based on detected boundaries
        segments = []
        height = image.shape[0]
        segment_starts = [0] + segmented_boundaries + [height]

        for i in range(len(segment_starts) - 1):
            y_start = segment_starts[i]
            y_end = segment_starts[i + 1]
            segments.append(image[y_start:y_end, :])

        st.subheader("📌 Segment Predictions")
        for idx, segment in enumerate(segments):
            gamma_corrected = apply_gamma_correction(segment)
            soil_type, soil_density = predict_soil(gamma_corrected)

            st.image(segment, caption=f"Segment {idx + 1}", use_container_width=True)
            st.write(f"✅ *Segment {idx + 1} - Soil Type:* {soil_type}")
            st.write(f"📏 *Segment {idx + 1} - Soil Density:* {soil_density:.2f} g/cm³")
