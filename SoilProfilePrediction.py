import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import gaussian_filter1d

# Load TFLite Models
@st.cache_resource
def load_tflite_models():
    check_soil_interpreter = tf.lite.Interpreter(model_path="check_soil_model.tflite")
    soil_type_interpreter = tf.lite.Interpreter(model_path="soil_type_model.tflite")
    soil_density_interpreter = tf.lite.Interpreter(model_path="soil_density_model.tflite")
    
    # Allocate tensors
    check_soil_interpreter.allocate_tensors()
    soil_type_interpreter.allocate_tensors()
    soil_density_interpreter.allocate_tensors()
    
    return (check_soil_interpreter, soil_type_interpreter, soil_density_interpreter)

(check_soil_interpreter, soil_type_interpreter, soil_density_interpreter) = load_tflite_models()

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
def segment_image(image, smoothing_sigma=2, gradient_threshold=0.7):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    # 1. Dynamic slice height
    slice_height = max(50, height // 40)
    num_slices = height // slice_height

    # 2. Compute mean intensity profile
    mean_intensity = np.array([
        np.mean(gray[i * slice_height:(i + 1) * slice_height, :]) for i in range(num_slices)
    ])
    smooth_intensity = gaussian_filter1d(mean_intensity, sigma=smoothing_sigma)

    # 3. Compute derivative of smoothed intensity
    gradient = np.gradient(smooth_intensity)
    peak_indices = np.where(np.abs(gradient) > gradient_threshold)[0]

    # 4. Convert to pixel coordinates
    crossing_points = [i * slice_height for i in peak_indices]

    # 5. Filter out close boundaries
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

# Function to check if the image is a soil profile
def is_soil_image(image):
    image = preprocess_image(image)
    prediction = run_tflite_model(check_soil_interpreter, image)
    return prediction[0][0] > 0.65  # Assuming the model outputs probability, threshold at 0.65

# Function to predict soil type and density
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

# Function to merge adjacent segments with same predicted soil type
def merge_segments_by_soil_type(segments, predicted_types, min_layer_gap=50):
    merged_segments = []
    merged_types = []
    merged_indices = []

    current_segment = segments[0]
    current_type = predicted_types[0]
    current_start_idx = 0

    for i in range(1, len(segments)):
        if predicted_types[i] == current_type:
            # Merge vertically (stack rows)
            current_segment = np.vstack((current_segment, segments[i]))
        else:
            # Save current merged segment
            merged_segments.append(current_segment)
            merged_types.append(current_type)
            merged_indices.append((current_start_idx, i))  # index range in original segments

            # Start new segment
            current_segment = segments[i]
            current_type = predicted_types[i]
            current_start_idx = i

    # Append last merged segment
    merged_segments.append(current_segment)
    merged_types.append(current_type)
    merged_indices.append((current_start_idx, len(segments)))

    return merged_segments, merged_types, merged_indices

# Streamlit UI
st.title("🌱 Soil Type & Density Detection with Layer Merging")
st.write("Upload a soil profile image to analyze its soil type and density.")

st.warning("**Note:** The input image is assumed to contain no shadows or foreign objects, as these may affect the model’s accuracy. The predictions for soil type and density are approximate and should not be considered as precise scientific measurements.")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
image = None

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Check if the image is a soil profile
    if is_soil_image(image):
        if st.button("Predict"):
            st.subheader("Segmentation & Prediction Results")

            # 1. Initial segmentation
            segmented_boundaries = segment_image(image)
            height = image.shape[0]
            segment_starts = [0] + segmented_boundaries + [height]

            # 2. Extract initial segments
            segments = [image[segment_starts[i]:segment_starts[i+1], :] for i in range(len(segment_starts) - 1)]

            # 3. Predict soil types for each segment
            predicted_types = []
            for segment in segments:
                gamma_corrected = apply_gamma_correction(segment)
                soil_type, _ = predict_soil(gamma_corrected)
                predicted_types.append(soil_type)

            # 4. Merge segments by soil type
            merged_segments, merged_types, merged_indices = merge_segments_by_soil_type(segments, predicted_types)

            # 5. Re-run prediction on merged segments for soil density
            final_predictions = []
            for merged_segment in merged_segments:
                gamma_corrected = apply_gamma_correction(merged_segment)
                soil_type, soil_density = predict_soil(gamma_corrected)
                final_predictions.append((soil_type, soil_density))

            # 6. Prepare final segmentation lines after merging
            final_boundaries = []
            for idx_range in merged_indices[:-1]:  # skip last boundary since it's image bottom
                # upper boundary of next merged segment
                y = segment_starts[idx_range[1]]
                final_boundaries.append(y)

            # 7. Draw final segmentation lines
            segmented_image = draw_segmentation_lines(image, final_boundaries)

            # 8. Display original and segmented image
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(segmented_image, caption="Segmented Image with Merged Layers", use_container_width=True)

            # 9. Display merged segments and predictions
            st.subheader("📌 Merged Segment Predictions")
            for idx, (merged_segment, (soil_type, soil_density)) in enumerate(zip(merged_segments, final_predictions)):
                st.image(merged_segment, caption=f"Merged Segment {idx + 1}", use_container_width=True)
                st.write(f"✅ *Soil Type:* {soil_type}")
                st.write(f"📏 *Soil Density:* {soil_density:.2f} g/cm³")
    else:
        st.error("🚨 The uploaded image is **not a soil profile**. Please upload a valid soil profile image.")
