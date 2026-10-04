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
    page_title="Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# COMPACT UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Hide unnecessary Streamlit chrome */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Compact main page */
    .block-container {
        max-width: 1250px;
        padding-top: 0.7rem;
        padding-bottom: 0.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* Reduce vertical spacing */
    div[data-testid="stVerticalBlock"] {
        gap: 0.45rem;
    }

    /* Main title */
    h1 {
        color: #2e7d32 !important;
        font-size: 2rem !important;
        margin-top: 0 !important;
        margin-bottom: 0.1rem !important;
    }

    /* Section headings */
    h2 {
        font-size: 1.25rem !important;
        color: #43a047 !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }

    h3 {
        font-size: 1.05rem !important;
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }

    /* Normal text */
    p {
        margin-top: 0.1rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* Input labels */
    [data-testid="stWidgetLabel"] p {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
    }

    /* Number inputs */
    [data-testid="stNumberInput"] {
        margin-bottom: -0.2rem !important;
    }

    /* Input fields slightly smaller */
    [data-baseweb="input"] input {
        min-height: 34px !important;
        font-size: 0.85rem !important;
    }

    /* Compact form */
    [data-testid="stForm"] {
        padding: 0.8rem !important;
        border-radius: 12px !important;
    }

    /* Prediction button */
    button[kind="primary"] {
        min-height: 42px !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        border: 1px solid rgba(76, 175, 80, 0.35);
        border-radius: 10px;
        padding: 0.55rem 0.7rem !important;
    }

    [data-testid="stMetricLabel"] p {
        font-size: 0.75rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
    }

    /* Progress bars */
    [data-testid="stProgress"] {
        margin-top: -0.25rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* Dividers */
    hr {
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        padding: 0.55rem 0.8rem !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        margin-top: 0.3rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = "final_crop_recommendation_ann.keras"
SCALER_FILE = "crop_scaler.pkl"
ENCODER_FILE = "crop_label_encoder.pkl"


# ============================================================
# CHECK FILES
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
        "Missing required files: "
        + ", ".join(missing_files)
    )

    st.stop()


# ============================================================
# LOAD MODEL
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
# SESSION STATE
# ============================================================

if "prediction_made" not in st.session_state:

    st.session_state.prediction_made = False


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [3.2, 1]
)


with header_left:

    st.title(
        "🌾 Smart Crop Recommendation System"
    )

    st.caption(
        "Artificial Neural Network Based Agricultural "
        "Decision Support System"
    )


with header_right:

    st.caption("ANN Architecture")

    st.write(
        "**7 → 32 → 16 → 22**"
    )


st.divider()


# ============================================================
# MAIN TWO-COLUMN LAYOUT
# ============================================================

input_column, result_column = st.columns(
    [1.15, 0.85],
    gap="large"
)


# ============================================================
# LEFT SIDE — USER INPUTS
# ============================================================

with input_column:

    st.subheader(
        "📋 Environmental Conditions"
    )


    with st.form(
        "crop_form",
        border=True
    ):

        # ----------------------------------------------------
        # SOIL NUTRIENTS
        # ----------------------------------------------------

        st.markdown(
            "**🧪 Soil Nutrients**"
        )


        n_col, p_col, k_col = st.columns(3)


        with n_col:

            nitrogen = st.number_input(
                "N | 0–140 kg/ha",
                min_value=0.0,
                max_value=140.0,
                value=70.0,
                step=1.0
            )


        with p_col:

            phosphorus = st.number_input(
                "P | 5–145 kg/ha",
                min_value=5.0,
                max_value=145.0,
                value=50.0,
                step=1.0
            )


        with k_col:

            potassium = st.number_input(
                "K | 5–205 kg/ha",
                min_value=5.0,
                max_value=205.0,
                value=50.0,
                step=1.0
            )


        # ----------------------------------------------------
        # CLIMATE
        # ----------------------------------------------------

        st.markdown(
            "**🌦️ Climate**"
        )


        temp_col, humidity_col = st.columns(2)


        with temp_col:

            temperature = st.number_input(
                "Temperature | 8.83–43.68 °C",
                min_value=8.83,
                max_value=43.68,
                value=25.00,
                step=0.10,
                format="%.2f"
            )


        with humidity_col:

            humidity = st.number_input(
                "Humidity | 14.26–99.98 %",
                min_value=14.26,
                max_value=99.98,
                value=70.00,
                step=0.10,
                format="%.2f"
            )


        # ----------------------------------------------------
        # SOIL / RAINFALL
        # ----------------------------------------------------

        st.markdown(
            "**🌍 Soil & Rainfall**"
        )


        ph_col, rainfall_col = st.columns(2)


        with ph_col:

            ph = st.number_input(
                "Soil pH | 3.50–9.94",
                min_value=3.50,
                max_value=9.94,
                value=6.50,
                step=0.01,
                format="%.2f"
            )


        with rainfall_col:

            rainfall = st.number_input(
                "Rainfall | 20.21–298.56 mm",
                min_value=20.21,
                max_value=298.56,
                value=100.00,
                step=0.10,
                format="%.2f"
            )


        submitted = st.form_submit_button(
            "🌾 Analyse & Recommend Crop",
            type="primary",
            width="stretch"
        )


    st.caption(
        "ℹ️ Use values only within the displayed "
        "training ranges."
    )


# ============================================================
# RUN ANN WHEN BUTTON IS PRESSED
# ============================================================

if submitted:

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


    input_scaled = scaler.transform(
        input_data
    )


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


    top3_indices = np.argsort(
        probabilities
    )[-3:][::-1]


    # Save prediction to session state
    st.session_state.prediction_made = True

    st.session_state.predicted_crop = (
        predicted_crop
    )

    st.session_state.confidence = (
        confidence
    )

    st.session_state.probabilities = (
        probabilities
    )

    st.session_state.top3_indices = (
        top3_indices
    )


# ============================================================
# RIGHT SIDE — RESULTS
# ============================================================

with result_column:

    st.subheader(
        "🎯 Recommendation"
    )


    if not st.session_state.prediction_made:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🌱 Ready for Prediction"
            )

            st.write(
                "Enter the environmental conditions "
                "on the left and click **Analyse & "
                "Recommend Crop**."
            )

            st.info(
                "The ANN predicts one of 22 crop classes."
            )


    else:

        predicted_crop = (
            st.session_state.predicted_crop
        )

        confidence = (
            st.session_state.confidence
        )

        probabilities = (
            st.session_state.probabilities
        )

        top3_indices = (
            st.session_state.top3_indices
        )


        # ----------------------------------------------------
        # MAIN RECOMMENDATION CARD
        # ----------------------------------------------------

        with st.container(
            border=True
        ):

            recommendation_col, confidence_col = st.columns(
                [1.5, 1]
            )


            with recommendation_col:

                st.caption(
                    "🌱 RECOMMENDED CROP"
                )

                st.header(
                    predicted_crop.title()
                )


            with confidence_col:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


            if confidence >= 90:

                st.success(
                    "✅ Very High Confidence"
                )

            elif confidence >= 70:

                st.success(
                    "✅ High Confidence"
                )

            elif confidence >= 50:

                st.warning(
                    "⚠️ Moderate Confidence"
                )

            else:

                st.error(
                    "⚠️ Low Confidence"
                )


        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        st.markdown(
            "**📊 Top 3 Predictions**"
        )


        medal_icons = [
            "🥇",
            "🥈",
            "🥉"
        ]


        for rank, index in enumerate(
            top3_indices
        ):

            crop_name = (
                label_encoder.inverse_transform(
                    [int(index)]
                )[0]
            )


            probability = float(
                probabilities[index]
            )


            percentage = (
                probability * 100
            )


            name_col, value_col = st.columns(
                [3, 1]
            )


            with name_col:

                st.write(
                    f"{medal_icons[rank]} "
                    f"**{crop_name.title()}**"
                )


            with value_col:

                st.write(
                    f"**{percentage:.2f}%**"
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


        # ----------------------------------------------------
        # COMPACT INPUT SUMMARY
        # ----------------------------------------------------

        with st.expander(
            "📋 Input Summary"
        ):

            st.write(
                f"""
                **N:** {nitrogen:.0f} |
                **P:** {phosphorus:.0f} |
                **K:** {potassium:.0f}

                **Temperature:** {temperature:.2f} °C |
                **Humidity:** {humidity:.2f} %

                **pH:** {ph:.2f} |
                **Rainfall:** {rainfall:.2f} mm
                """
            )


# ============================================================
# BOTTOM INFORMATION
# ============================================================

st.divider()


bottom_left, bottom_right = st.columns(
    [2, 1]
)


with bottom_left:

    st.caption(
        "⚠️ Academic ANN prototype. Predictions should "
        "not replace professional agricultural advice."
    )


with bottom_right:

    with st.expander(
        "🧠 Model Info"
    ):

        st.write(
            """
            **Architecture:** 7 → 32 → 16 → 22

            **Hidden activation:** ReLU

            **Output activation:** Softmax

            **Output:** 22 crop classes
            """
        )
