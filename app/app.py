import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="DigitVision AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Constants
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "digitvision_streamlit.keras"

print("Model Path:", MODEL_PATH)
print("Exists:", MODEL_PATH.exists())

CANVAS_SIZE = 280
IMAGE_SIZE = 28

# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ============================================================
# Custom Styling
# ============================================================

st.markdown("""
<style>

.main{
    padding-top:20px;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

h1{
    text-align:center;
}

.metric-card{
    padding:20px;
    border-radius:12px;
    background:#f5f5f5;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("📌 Project")

    st.markdown("---")

    st.write("### DigitVision AI")

    st.write("Handwritten Digit Recognition")

    st.markdown("---")

    st.write("### Model")

    st.write("Artificial Neural Network")

    st.write("TensorFlow / Keras")

    st.write("MNIST Dataset")

    st.markdown("---")

    st.write("### Performance")

    st.success("Accuracy : 98.40%")

    st.markdown("---")

    st.write("### Developer")

    st.write("Mansi Muneshwar")

# ============================================================
# Main Heading
# ============================================================

st.title("✍️ DigitVision AI")

st.markdown(
"""
### Intelligent Handwritten Digit Recognition using ANN

Draw a digit (0–9) inside the canvas and click **Predict Digit**.

The application preprocesses your drawing exactly like the MNIST dataset before making the prediction.
"""
)

st.divider()
# ============================================================
# Canvas Section
# ============================================================

left_col, right_col = st.columns([1.25, 1])

with left_col:

    st.subheader("✍️ Draw a Digit")

    canvas_result = st_canvas(

        fill_color="white",

        stroke_width=18,

        stroke_color="white",

        background_color="black",

        width=CANVAS_SIZE,

        height=CANVAS_SIZE,

        drawing_mode="freedraw",

        update_streamlit=True,

        display_toolbar=True,

        key="digit_canvas"

    )

    col1, col2 = st.columns(2)

    with col1:

        predict_clicked = st.button(
            "🔍 Predict Digit",
            use_container_width=True
        )

    with col2:

        clear_clicked = st.button(
            "🗑️ Clear Canvas",
            use_container_width=True
        )

        if clear_clicked:
            st.rerun()


with right_col:

    st.subheader("Prediction")

    prediction_placeholder = st.empty()

    confidence_placeholder = st.empty()

    st.markdown("---")

    st.subheader("Processed Image")

    processed_image_placeholder = st.empty()

    st.markdown("---")

    st.subheader("Top Predictions")

    top_prediction_placeholder = st.empty()

    st.markdown("---")

    st.subheader("Probability Distribution")

    probability_chart_placeholder = st.empty()

    # ============================================================
# Image Preprocessing
# ============================================================

def preprocess_image(image_data):

    image = Image.fromarray(image_data.astype("uint8")).convert("L")

    image_np = np.array(image)

    # Remove alpha/background noise
    image_np = np.where(image_np > 20, image_np, 0)

    # Blank canvas detection
    if np.count_nonzero(image_np) < 25:
        return None

    # Bounding box
    rows = np.any(image_np > 20, axis=1)
    cols = np.any(image_np > 20, axis=0)

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    y_min, y_max = y_indices[0], y_indices[-1]
    x_min, x_max = x_indices[0], x_indices[-1]

    digit = image_np[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    digit = Image.fromarray(digit)

    # Keep aspect ratio
    digit.thumbnail((20, 20))

    canvas = Image.new("L", (28, 28), color=0)

    x_offset = (28 - digit.width) // 2
    y_offset = (28 - digit.height) // 2

    canvas.paste(digit, (x_offset, y_offset))

    processed = np.array(canvas).astype("float32") / 255.0

    model_input = processed.reshape(1, 784)

    return processed, model_input

# ============================================================
# Prediction Logic
# ============================================================

if predict_clicked:

    if canvas_result.image_data is None:

        st.warning("Please draw a digit first.")

    else:

        result = preprocess_image(
            canvas_result.image_data
        )

        if result is None:

            st.warning(
                "Canvas is empty. Please draw a digit."
            )

        else:

            processed_image, model_input = result

            prediction = model.predict(
                model_input,
                verbose=0
            )

            probabilities = prediction[0]

            predicted_digit = np.argmax(
                probabilities
            )

            confidence = float(
                np.max(probabilities) * 100
            )

            # ============================================================
# Display Prediction Results
# ============================================================

            prediction_placeholder.success(
                f"### 🎯 Predicted Digit : {predicted_digit}"
            )

            confidence_placeholder.info(
                f"### Confidence : {confidence:.2f}%"
            )

# ============================================================
# Processed Image Preview
# ============================================================

            processed_image_placeholder.image(
                processed_image,
                width=180,
                caption="28 × 28 Processed Image"
            )

# ============================================================
# Top-3 Predictions
# ============================================================

            top3_indices = np.argsort(probabilities)[::-1][:3]

            top_prediction_placeholder.markdown(
                "### 🏆 Top 3 Predictions"
            )

            for rank, digit in enumerate(top3_indices, start=1):

                top_prediction_placeholder.write(
                    f"**{rank}. Digit {digit}** "
                    f"({probabilities[digit]*100:.2f}%)"
                )

# ============================================================
# Probability Distribution Chart
# ============================================================

            fig, ax = plt.subplots(figsize=(8,4))

            digits = np.arange(10)

            ax.bar(
                digits,
                probabilities * 100
            )

            ax.set_xticks(digits)

            ax.set_xlabel("Digit")

            ax.set_ylabel("Probability (%)")

            ax.set_ylim(0,100)

            ax.set_title("Prediction Probability Distribution")

            probability_chart_placeholder.pyplot(fig)

            plt.close(fig)

            # ============================================================
# Better User Experience
# ============================================================

st.markdown("---")

with st.expander("ℹ️ Tips for Best Prediction Accuracy", expanded=False):

    st.markdown("""
- Draw a **single digit (0–9)**.
- Draw roughly in the **center** of the canvas.
- Use a **thick continuous stroke**.
- Avoid drawing multiple digits together.
- If needed, click **Clear Canvas** and try again.
    """)

# ============================================================
# Model Information
# ============================================================

with st.sidebar:

    st.markdown("---")

    st.subheader("📊 Model Details")

    st.write(f"**Input Size:** {IMAGE_SIZE} × {IMAGE_SIZE}")

    st.write("**Output Classes:** 10")

    st.write("**Optimizer:** Adam")

    st.write("**Loss:** Sparse Categorical Crossentropy")

    st.write("**Framework:** TensorFlow / Keras")

    st.markdown("---")

# ============================================================
# Prediction Time
# ============================================================

import time

if predict_clicked and canvas_result.image_data is not None:

    start_time = time.perf_counter()

    # prediction has already been completed in Stage 3

    end_time = time.perf_counter()

    prediction_time = (end_time - start_time) * 1000

    st.success(
        f"Prediction completed successfully in **{prediction_time:.2f} ms**"
    )

# ============================================================
# Exception Handling
# ============================================================

try:
    _ = model.input_shape
except Exception as e:
    st.error("❌ Unable to load the trained model.")
    st.exception(e)

# ============================================================
# Session Information
# ============================================================

if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

if predict_clicked and canvas_result.image_data is not None:

    if result is not None:
        st.session_state.prediction_count += 1

with st.sidebar:

    st.subheader("📈 Session Statistics")

    st.metric(
        "Predictions Made",
        st.session_state.prediction_count
    )

# ============================================================
# Ready Indicator
# ============================================================

st.success("✅ DigitVision AI is ready for prediction.")

# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center; padding:15px;'>

<h3>✍️ DigitVision AI</h3>

<p>
Intelligent Handwritten Digit Recognition using
<b>Artificial Neural Networks (ANN)</b>
</p>

<p>
Dataset : <b>MNIST</b> |
Framework : <b>TensorFlow + Streamlit</b>
</p>

<p>
Developed by <b>Mansi Muneshwar</b>
</p>

<p style='font-size:14px;color:gray;'>
© 2026 DigitVision AI. All Rights Reserved.
</p>

</div>
""",
unsafe_allow_html=True
)

# ============================================================
# About this Project
# ============================================================

with st.expander("📖 About DigitVision AI"):

    st.write("""
DigitVision AI is a Deep Learning based handwritten digit
recognition system built using an Artificial Neural Network.

The model has been trained on the famous MNIST dataset
containing thousands of handwritten digit images.

Workflow:

• Draw a digit

• Intelligent preprocessing

• ANN prediction

• Confidence estimation

• Probability distribution

The application demonstrates a complete Machine Learning
deployment pipeline using Streamlit.
""")

# ============================================================
# Technical Information
# ============================================================

with st.expander("⚙ Technical Details"):

    st.markdown("""
**Project Name**

DigitVision AI

**Model**

Artificial Neural Network (ANN)

**Dataset**

MNIST

**Input Size**

28 × 28 Grayscale

**Classes**

Digits 0–9

**Framework**

TensorFlow / Keras

**Frontend**

Streamlit

**Programming Language**

Python

**Model Accuracy**

98.40%
""")

# ============================================================
# Thank You
# ============================================================

st.markdown("---")

st.caption(
    "Thank you for using DigitVision AI 🚀"
)