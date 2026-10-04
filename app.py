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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Hide Streamlit default footer/menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main page background */
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(76, 175, 80, 0.10), transparent 25%),
            radial-gradient(circle at bottom left, rgba(46, 125, 50, 0.12), transparent 22%),
            linear-gradient(180deg, #0B1220 0%, #0E1117 55%, #111827 100%);
    }

    /* Main content width and top spacing */
    .block-container {
        max-width: 1320px;
        padding-top: 1.8rem;
        padding-bottom: 1rem;
        padding-left: 1.4rem;
        padding-right: 1.4rem;
    }

    /* Reduce some vertical gaps */
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }

    /* Headings */
    h1, h2, h3 {
        color: #F8FAFC !important;
    }

    /* Widget labels */
    [data-testid="stWidgetLabel"] p {
        font-size: 0.90rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    /* Inputs */
    [data-baseweb="input"] input {
        font-size: 0.95rem !important;
        min-height: 38px !important;
    }

    /* Forms */
    [data-testid="stForm"] {
        border-radius: 16px !important;
        padding: 1rem !important;
        background: rgba(255,255,255,0.02);
    }

    /* Buttons */
    button[kind="primary"] {
        min-height: 46px !important;
        border-radius: 10px !important;
        font-size: 0.96rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #43A047 0%, #66BB6A 100%) !important;
        border: none !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        border: 1px solid rgba(102, 187, 106, 0.28);
        border-radius: 14px;
        padding: 0.7rem 0.9rem !important;
        background: rgba(255,255,255,0.02);
    }

    /* Progress bars */
    [data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #43A047, #66BB6A) !important;
    }

    /* Divider */
    hr {
        margin-top: 0.6rem !important;
        margin-bottom: 0.6rem !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        border-radius: 12px !important;
    }

    /* Hero section */
    .hero-banner {
        background:
            linear-gradient(135deg, rgba(67,160,71,0.20), rgba(38,50,56,0.18)),
            rgba(255,255,255,0.03);
        border: 1px solid rgba(102, 187, 106, 0.22);
        border-radius: 22px;
        padding: 1.2rem 1.4rem 1.1rem 1.4rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.20);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.3fr 0.9fr;
        gap: 1rem;
        align-items: center;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.08;
        color: #F8FAFC;
        margin-bottom: 0.45rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #D1D5DB;
        margin-bottom: 0.8rem;
        line-height: 1.45;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-bottom: 0.6rem;
    }

    .hero-badge {
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        background: rgba(102, 187, 106, 0.14);
        color: #E8F5E9;
        border: 1px solid rgba(102, 187, 106, 0.20);
    }

    .hero-note {
        color: #A7F3D0;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .hero-visual {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .hero-card {
        width: 100%;
        max-width: 360px;
        border-radius: 18px;
        padding: 0.8rem;
        background: rgba(8, 15, 24, 0.65);
        border: 1px solid rgba(102, 187, 106, 0.16);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .hero-mini-top {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.6rem;
    }

    .hero-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 4px;
    }

    .hero-mini-title {
        color: #F8FAFC;
        font-size: 0.85rem;
        font-weight: 700;
    }

    .hero-mini-sub {
        color: #9CA3AF;
        font-size: 0.74rem;
    }

    .hero-bars {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
    }

    .hero-bar {
        background: rgba(255,255,255,0.06);
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
    }

    .hero-fill-1 {
        width: 78%;
        height: 100%;
        background: linear-gradient(90deg, #66BB6A, #A5D6A7);
    }

    .hero-fill-2 {
        width: 52%;
        height: 100%;
        background: linear-gradient(90deg, #42A5F5, #90CAF9);
    }

    .hero-fill-3 {
        width: 64%;
        height: 100%;
        background: linear-gradient(90deg, #FFB74D, #FFE082);
    }

    .hero-bottom {
        margin-top: 0.75rem;
        padding: 0.6rem;
        border-radius: 12px;
        background: rgba(102, 187, 106, 0.10);
        color: #E8F5E9;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Small screen adjustment */
    @media (max-width: 900px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 1.7rem;
        }
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
    file for file in required_files
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
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-grid">
            <div>
                <div class="hero-title">🌾 Smart Crop Recommendation System</div>
                <div class="hero-subtitle">
                    Artificial Neural Network Based Agricultural Decision Support System.
                    Enter soil nutrients and climatic conditions to predict the most suitable crop from 22 crop classes.
                </div>

                <div class="hero-badges">
                    <div class="hero-badge">7 Input Features</div>
                    <div class="hero-badge">ANN Prediction</div>
                    <div class="hero-badge">22 Crop Classes</div>
                    <div class="hero-badge">Interactive UI</div>
                </div>

                <div class="hero-note">
                    Designed for an academic crop recommendation prototype 🌱
                </div>
            </div>

            <div class="hero-visual">
                <div class="hero-card">
                    <div class="hero-mini-top">
                        <div>
                            <div class="hero-mini-title">Prediction Dashboard</div>
                            <div class="hero-mini-sub">Live ANN-based recommendation preview</div>
                        </div>
                        <div>
                            <span class="hero-dot" style="background:#66BB6A;"></span>
                            <span class="hero-dot" style="background:#42A5F5;"></span>
                            <span class="hero-dot" style="background:#FFB74D;"></span>
                        </div>
                    </div>

                    <div class="hero-bars">
                        <div class="hero-bar"><div class="hero-fill-1"></div></div>
                        <div class="hero-bar"><div class="hero-fill-2"></div></div>
                        <div class="hero-bar"><div class="hero-fill-3"></div></div>
                    </div>

                    <div class="hero-bottom">
                        🌿 Clean interface • compact layout • readable predictions
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN LAYOUT
# ============================================================

input_column, result_column = st.columns(
    [1.15, 0.85],
    gap="large"
)


# ============================================================
# LEFT COLUMN — INPUTS
# ============================================================

with input_column:

    st.subheader("📋 Environmental Conditions")

    with st.form(
        "crop_form",
        border=True
    ):

        # SOIL NUTRIENTS
        st.markdown("### 🧪 Soil Nutrients")

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

        # CLIMATE
        st.markdown("### 🌦️ Climate")

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

        # SOIL / RAINFALL
        st.markdown("### 🌍 Soil & Rainfall")

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
        "ℹ️ Use values only within the displayed training ranges."
    )


# ============================================================
# PREDICTION
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

    input_scaled = scaler.transform(input_data)

    probabilities = model.predict(
        input_scaled,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_crop = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    top3_indices = np.argsort(probabilities)[-3:][::-1]

    st.session_state.prediction_made = True
    st.session_state.predicted_crop = predicted_crop
    st.session_state.confidence = confidence
    st.session_state.probabilities = probabilities
    st.session_state.top3_indices = top3_indices
    st.session_state.last_inputs = {
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }


# ============================================================
# RIGHT COLUMN — RESULTS
# ============================================================

with result_column:

    st.subheader("🎯 Recommendation")

    if not st.session_state.prediction_made:

        with st.container(border=True):
            st.markdown("### 🌱 Ready for Prediction")
            st.write(
                "Enter the environmental conditions on the left and click "
                "**Analyse & Recommend Crop**."
            )
            st.info("The ANN predicts one of 22 crop classes.")

    else:

        predicted_crop = st.session_state.predicted_crop
        confidence = st.session_state.confidence
        probabilities = st.session_state.probabilities
        top3_indices = st.session_state.top3_indices

        last_inputs = st.session_state.last_inputs
        nitrogen = last_inputs["nitrogen"]
        phosphorus = last_inputs["phosphorus"]
        potassium = last_inputs["potassium"]
        temperature = last_inputs["temperature"]
        humidity = last_inputs["humidity"]
        ph = last_inputs["ph"]
        rainfall = last_inputs["rainfall"]

        # MAIN RESULT
        with st.container(border=True):

            recommendation_col, confidence_col = st.columns(
                [1.45, 1]
            )

            with recommendation_col:
                st.caption("🌱 RECOMMENDED CROP")
                st.header(predicted_crop.title())

            with confidence_col:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            if confidence >= 90:
                st.success("✅ Very High Confidence")
            elif confidence >= 70:
                st.success("✅ High Confidence")
            elif confidence >= 50:
                st.warning("⚠️ Moderate Confidence")
            else:
                st.error("⚠️ Low Confidence")

        # TOP 3
        st.markdown("### 📊 Top 3 Predictions")

        medal_icons = ["🥇", "🥈", "🥉"]

        for rank, index in enumerate(top3_indices):

            crop_name = label_encoder.inverse_transform(
                [int(index)]
            )[0]

            probability = float(probabilities[index])
            percentage = probability * 100

            name_col, value_col = st.columns([3, 1])

            with name_col:
                st.write(
                    f"{medal_icons[rank]} **{crop_name.title()}**"
                )

            with value_col:
                st.write(f"**{percentage:.2f}%**")

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

        # INPUT SUMMARY
        with st.expander("📋 Input Summary"):
            st.write(
                f"""
                **N:** {nitrogen:.0f} kg/ha | **P:** {phosphorus:.0f} kg/ha | **K:** {potassium:.0f} kg/ha

                **Temperature:** {temperature:.2f} °C | **Humidity:** {humidity:.2f} %

                **Soil pH:** {ph:.2f} | **Rainfall:** {rainfall:.2f} mm
                """
            )


# ============================================================
# FOOTER / MODEL INFO
# ============================================================

st.divider()

bottom_left, bottom_right = st.columns([2, 1])

with bottom_left:
    st.caption(
        "⚠️ Academic ANN prototype. Predictions should not replace "
        "professional agricultural advice."
    )

with bottom_right:
    with st.expander("🧠 Model Info"):
        st.write(
            """
            **Architecture:** 7 → 32 → 16 → 22

            **Hidden activation:** ReLU

            **Output activation:** Softmax

            **Output:** 22 crop classes
            """
        )
