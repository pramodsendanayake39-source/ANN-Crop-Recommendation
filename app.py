import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SMALL UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main page width */
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    h1 {
        color: #1b5e20 !important;
        font-weight: 800 !important;
    }

    /* Section headings */
    h2, h3 {
        color: #2e7d32 !important;
    }

    /* Input labels */
    [data-testid="stWidgetLabel"] p {
        font-weight: 700 !important;
    }

    /* Primary button */
    button[kind="primary"] {
        min-height: 52px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        border: 1px solid rgba(46, 125, 50, 0.35);
        border-radius: 12px;
        padding: 16px;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div > div > div > div {
        background-color: #43a047 !important;
    }

    /* Remove excessive spacing */
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = "final_crop_recommendation_ann.keras"
SCALER_FILE = "crop_scaler.pkl"
ENCODER_FILE = "crop_label_encoder.pkl"


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    ENCODER_FILE
]

missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error(
        "Missing required model files: "
        + ", ".join(missing_files)
    )

    st.stop()


# ============================================================
# LOAD TRAINED ANN SYSTEM
# ============================================================

@st.cache_resource
def load_system():

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    label_encoder = joblib.load(
        ENCODER_FILE
    )

    return model, scaler, label_encoder


model, scaler, label_encoder = load_system()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌾 Smart Crop Recommendation System"
)

st.markdown(
    """
    ### Artificial Neural Network Based Agricultural Decision Support System
    """
)

st.caption(
    "Soil Nutrients • Climate • ANN Prediction • 22 Crop Classes"
)


# ============================================================
# INTRODUCTION
# ============================================================

st.success(
    """
    🌱 **Find a suitable crop for the entered environmental conditions**

    Enter the soil nutrient values and climatic conditions below.
    The trained Artificial Neural Network analyses all seven
    parameters and predicts the most suitable crop.
    """
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header(
    "📋 Environmental Conditions"
)


with st.form(
    "crop_form",
    border=True
):

    # ========================================================
    # NUTRIENTS
    # ========================================================

    st.subheader(
        "🧪 Soil Nutrient Parameters"
    )

    st.caption(
        "Enter the nutrient values within the ranges used "
        "during ANN training."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=140.0,
            value=70.0,
            step=1.0,
            help="Training range: 0–140"
        )

        st.caption(
            "📌 Range: 0–140"
        )


    with col2:

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=5.0,
            max_value=145.0,
            value=50.0,
            step=1.0,
            help="Training range: 5–145"
        )

        st.caption(
            "📌 Range: 5–145"
        )


    with col3:

        potassium = st.number_input(
            "Potassium (K)",
            min_value=5.0,
            max_value=205.0,
            value=50.0,
            step=1.0,
            help="Training range: 5–205"
        )

        st.caption(
            "📌 Range: 5–205"
        )


    st.divider()


    # ========================================================
    # CLIMATE
    # ========================================================

    st.subheader(
        "🌦️ Climatic Conditions"
    )


    col4, col5 = st.columns(2)


    with col4:

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=8.83,
            max_value=43.68,
            value=25.00,
            step=0.10,
            format="%.2f",
            help="Training range: 8.83–43.68 °C"
        )

        st.caption(
            "🌡️ Range: 8.83–43.68 °C"
        )


    with col5:

        humidity = st.number_input(
            "Relative Humidity (%)",
            min_value=14.26,
            max_value=99.98,
            value=70.00,
            step=0.10,
            format="%.2f",
            help="Training range: 14.26–99.98%"
        )

        st.caption(
            "💧 Range: 14.26–99.98 %"
        )


    st.divider()


    # ========================================================
    # SOIL & RAINFALL
    # ========================================================

    st.subheader(
        "🌍 Soil & Rainfall Conditions"
    )


    col6, col7 = st.columns(2)


    with col6:

        ph = st.number_input(
            "Soil pH",
            min_value=3.50,
            max_value=9.94,
            value=6.50,
            step=0.01,
            format="%.2f",
            help="Training range: 3.50–9.94"
        )

        st.caption(
            "🧪 Range: 3.50–9.94 | Unitless"
        )


    with col7:

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=20.21,
            max_value=298.56,
            value=100.00,
            step=0.10,
            format="%.2f",
            help="Training range: 20.21–298.56 mm"
        )

        st.caption(
            "🌧️ Range: 20.21–298.56 mm"
        )


    st.write("")


    submitted = st.form_submit_button(
        "🌾 Analyse Conditions & Recommend Crop",
        type="primary",
        width="stretch"
    )


# ============================================================
# INPUT GUIDANCE
# ============================================================

st.info(
    """
    **Input guidance:**  
    All values should remain within the displayed training ranges.
    The ANN has not been validated for conditions outside these limits.
    """
)


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # CREATE INPUT RECORD
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "N": [nitrogen],
            "P": [phosphorus],
            "K": [potassium],
            "temperature": [temperature],
            "humidity": [humidity],
            "ph": [ph],
            "rainfall": [rainfall]
        }
    )


    # --------------------------------------------------------
    # SCALE INPUT
    # --------------------------------------------------------

    input_scaled = scaler.transform(
        input_data
    )


    # --------------------------------------------------------
    # ANN PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict(
        input_scaled,
        verbose=0
    )[0]


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    predicted_crop = label_encoder.inverse_transform(
        [predicted_index]
    )[0]


    confidence = float(
        probabilities[
            predicted_index
        ] * 100
    )


    # ========================================================
    # MAIN RESULT
    # ========================================================

    st.divider()

    st.header(
        "🎯 Recommendation"
    )


    with st.container(
        border=True
    ):

        result_left, result_right = st.columns(
            [2, 1]
        )


        with result_left:

            st.caption(
                "🌱 ANN RECOMMENDED CROP"
            )

            st.title(
                predicted_crop.title()
            )

            st.write(
                "Based on the soil nutrient and climatic "
                "conditions entered above."
            )


        with result_right:

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )


        if confidence >= 90:

            st.success(
                "✅ Very High Model Confidence"
            )

        elif confidence >= 70:

            st.success(
                "✅ High Model Confidence"
            )

        elif confidence >= 50:

            st.warning(
                "⚠️ Moderate Model Confidence"
            )

        else:

            st.error(
                "⚠️ Low Model Confidence"
            )


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    st.subheader(
        "📊 Top 3 ANN Predictions"
    )


    top3_indices = np.argsort(
        probabilities
    )[-3:][::-1]


    for rank, index in enumerate(
        top3_indices,
        start=1
    ):

        crop_name = label_encoder.inverse_transform(
            [int(index)]
        )[0]


        probability = float(
            probabilities[index]
        )


        probability_percent = (
            probability * 100
        )


        with st.container(
            border=True
        ):

            crop_col, probability_col = st.columns(
                [4, 1]
            )


            with crop_col:

                if rank == 1:

                    st.markdown(
                        f"### 🥇 {crop_name.title()}"
                    )

                elif rank == 2:

                    st.markdown(
                        f"### 🥈 {crop_name.title()}"
                    )

                else:

                    st.markdown(
                        f"### 🥉 {crop_name.title()}"
                    )


            with probability_col:

                st.metric(
                    "Probability",
                    f"{probability_percent:.2f}%"
                )


            st.progress(
                min(
                    max(
                        probability,
                        0.0
                    ),
                    1.0
                )
            )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.subheader(
        "📋 Submitted Conditions"
    )


    summary = pd.DataFrame(
        {
            "Parameter": [
                "Nitrogen (N)",
                "Phosphorus (P)",
                "Potassium (K)",
                "Temperature",
                "Relative Humidity",
                "Soil pH",
                "Rainfall"
            ],

            "Entered Value": [
                f"{nitrogen:.0f}",
                f"{phosphorus:.0f}",
                f"{potassium:.0f}",
                f"{temperature:.2f}",
                f"{humidity:.2f}",
                f"{ph:.2f}",
                f"{rainfall:.2f}"
            ],

            "Unit": [
                "Dataset nutrient units",
                "Dataset nutrient units",
                "Dataset nutrient units",
                "°C",
                "%",
                "Unitless",
                "mm"
            ],

            "Training Range": [
                "0–140",
                "5–145",
                "5–205",
                "8.83–43.68",
                "14.26–99.98",
                "3.50–9.94",
                "20.21–298.56"
            ]
        }
    )


    st.dataframe(
        summary,
        hide_index=True,
        width="stretch"
    )


# ============================================================
# ABOUT THE MODEL
# ============================================================

st.divider()


with st.expander(
    "🧠 About the ANN Model"
):

    st.markdown(
        """
        #### Network Architecture

        **Input layer**
        - 7 environmental variables

        **Hidden layer 1**
        - 32 neurons
        - ReLU activation

        **Hidden layer 2**
        - 16 neurons
        - ReLU activation

        **Output layer**
        - 22 neurons
        - Softmax activation

        #### Model Inputs

        - Nitrogen
        - Phosphorus
        - Potassium
        - Temperature
        - Relative Humidity
        - Soil pH
        - Rainfall

        #### Model Output

        The crop class with the highest ANN output probability.
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    """
    ⚠️ **Academic Prototype**

    This application was developed as part of an Artificial
    Neural Network academic project.

    The model prediction is based on patterns learned from the
    selected crop recommendation dataset. It should not replace
    professional agricultural advice, laboratory soil testing,
    or site-specific field investigation.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌾 Artificial Neural Network Crop Recommendation Project"
)

st.caption(
    "7 Inputs → 32 Neurons → 16 Neurons → 22 Crop Classes"
)
