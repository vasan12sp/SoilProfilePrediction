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

    check_soil_interpreter.allocate_tensors()
    soil_type_interpreter.allocate_tensors()
    soil_density_interpreter.allocate_tensors()

    return (check_soil_interpreter, soil_type_interpreter, soil_density_interpreter)

(check_soil_interpreter, soil_type_interpreter, soil_density_interpreter) = load_tflite_models()

def preprocess_image(image):
    IMG_SIZE = (224, 224)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0).astype(np.float32)

def apply_gamma_correction(image, gamma=1.4):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def segment_image(image, smoothing_sigma=2, gradient_threshold=0.7):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    slice_height = max(50, height // 40)
    num_slices = height // slice_height

    mean_intensity = np.array([
        np.mean(gray[i * slice_height:(i + 1) * slice_height, :]) for i in range(num_slices)
    ])
    smooth_intensity = gaussian_filter1d(mean_intensity, sigma=smoothing_sigma)
    gradient = np.gradient(smooth_intensity)
    peak_indices = np.where(np.abs(gradient) > gradient_threshold)[0]

    crossing_points = [i * slice_height for i in peak_indices]

    min_layer_gap = 80
    filtered_boundaries = []
    prev_y = -min_layer_gap

    for y in crossing_points:
        if y - prev_y > min_layer_gap:
            filtered_boundaries.append(y)
            prev_y = y

    return filtered_boundaries

def draw_segmentation_lines(image, boundaries):
    output_image = image.copy()
    for y in boundaries:
        cv2.line(output_image, (0, y), (image.shape[1], y), (255, 0, 0), 2)
    return output_image

def run_tflite_model(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def is_soil_image(image):
    image = preprocess_image(image)
    prediction = run_tflite_model(check_soil_interpreter, image)
    return prediction[0][0] > 0.65

def predict_soil(image):
    image = preprocess_image(image)

    soil_type_pred = run_tflite_model(soil_type_interpreter, image)
    soil_types = ["Alluvial soil", "Black soil", "Clay soil", "Red soil"]
    predicted_class = np.argmax(soil_type_pred)
    predicted_soil_name = soil_types[predicted_class]

    soil_density_pred = run_tflite_model(soil_density_interpreter, image)
    predicted_density = soil_density_pred[0][0]

    return predicted_soil_name, predicted_density

# Streamlit UI
st.title("🌱 Soil Type & Density Detection")
st.write("Upload a soil profile image to analyze its soil type and density.")
st.warning("**Note:** Input images should be clear and free from shadows or foreign objects.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
image = None

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if is_soil_image(image):
        if st.button("Predict"):
            st.subheader("Segmentation Results")

            # Step 1: Initial segmentation
            boundaries = segment_image(image)

            # Step 2: Split into segments
            segment_starts = [0] + boundaries + [image.shape[0]]
            segments = [image[segment_starts[i]:segment_starts[i+1], :] for i in range(len(segment_starts) - 1)]

            # Step 3: Predict both soil types and densities for each segment
            initial_predictions = []
            for seg in segments:
                gamma_corrected = apply_gamma_correction(seg)
                soil_type, soil_density = predict_soil(gamma_corrected)
                initial_predictions.append((soil_type, soil_density))

            # Step 4: Merge consecutive segments if type is same and density difference is small
            merged_segments = []
            merged_boundaries = []
            current_segment = segments[0]
            current_soil_type, current_density = initial_predictions[0]
            current_start = segment_starts[0]

            density_threshold = 0.2  # you can fine-tune this

            for i in range(1, len(segments)):
                next_soil_type, next_density = initial_predictions[i]
                density_diff = abs(next_density - current_density)

                if next_soil_type == current_soil_type and density_diff <= density_threshold:
                    current_segment = np.vstack((current_segment, segments[i]))
                else:
                    merged_segments.append(current_segment)
                    merged_boundaries.append(segment_starts[i])
                    current_segment = segments[i]
                    current_soil_type = next_soil_type
                    current_density = next_density

            merged_segments.append(current_segment)

            # Step 5: Draw segmentation lines after merging
            segmented_image_after_merge = draw_segmentation_lines(image, merged_boundaries)

            # Step 6: Display original and merged segmented image
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(segmented_image_after_merge, caption="Segmented Image", use_container_width=True)

            # Step 7: Show merged segment predictions
            st.subheader("📌 Segment Predictions")
            for idx, segment in enumerate(merged_segments):
                gamma_corrected = apply_gamma_correction(segment)
                soil_type, soil_density = predict_soil(gamma_corrected)

                st.image(segment, caption=f"Segment {idx + 1}", use_container_width=True)
                st.write(f"✅ *Segment {idx + 1} - Soil Type:* {soil_type}")
                st.write(f"📏 *Segment {idx + 1} - Soil Density:* {soil_density:.2f} g/cm³")
    else:
        st.error("🚨 The uploaded image is **not a soil profile**. Please upload a valid soil profile image.")
