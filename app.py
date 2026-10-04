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
# CUSTOM STYLE
# ============================================================

st.html(
    """
<style>

/* ==========================================================
   PAGE
========================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 92% 8%,
            rgba(76,175,80,0.13),
            transparent 22%
        ),
        linear-gradient(
            145deg,
            #071019 0%,
            #0B1118 50%,
            #101721 100%
        );
}


/* More top room prevents clipping */

.block-container {
    max-width: 1320px;

    padding-top: 18px !important;
    padding-bottom: 12px !important;

    padding-left: 18px !important;
    padding-right: 18px !important;
}


/* Moderate spacing */

div[data-testid="stVerticalBlock"] {
    gap: 0.48rem;
}


/* ==========================================================
   TEXT
========================================================== */

h1,
h2,
h3,
h4 {
    color: #F8FAFC !important;

    line-height: 1.30 !important;

    padding-top: 2px !important;
    padding-bottom: 2px !important;
}


p {
    color: #E5E7EB;

    line-height: 1.45 !important;
}


/* Input labels */

[data-testid="stWidgetLabel"] {
    overflow: visible !important;
}


[data-testid="stWidgetLabel"] p {
    color: #F8FAFC !important;

    font-weight: 700 !important;
    font-size: 0.84rem !important;

    line-height: 1.35 !important;

    padding-top: 2px !important;
    padding-bottom: 2px !important;

    overflow: visible !important;
}


/* Caption */

[data-testid="stCaptionContainer"] p {
    color: #AEB8C4 !important;

    line-height: 1.35 !important;
}


/* ==========================================================
   HERO
========================================================== */

.hero-banner {

    width: 100%;

    box-sizing: border-box;

    overflow: visible;

    background:
        linear-gradient(
            125deg,
            rgba(27,94,32,0.42),
            rgba(13,25,34,0.72)
        );

    border:
        1px solid
        rgba(102,187,106,0.30);

    border-radius: 18px;

    padding:
        18px 22px
        18px 22px;

    margin-top: 4px;
    margin-bottom: 16px;

    box-shadow:
        0 12px 35px
        rgba(0,0,0,0.20);
}


.hero-grid {

    display: grid;

    grid-template-columns:
        1.45fr 0.65fr;

    gap: 25px;

    align-items: center;

    overflow: visible;
}


.hero-title {

    color: #F8FFF9;

    font-size:
        clamp(
            26px,
            2.45vw,
            37px
        );

    line-height: 1.25;

    font-weight: 850;

    margin:
        2px 0
        8px 0;

    padding-top: 4px;

    overflow: visible;
}


.hero-subtitle {

    color: #D7E0E6;

    font-size: 14px;

    line-height: 1.55;

    margin-bottom: 11px;
}


.hero-badges {

    display: flex;

    flex-wrap: wrap;

    gap: 7px;

    padding-bottom: 2px;
}


.hero-badge {

    display: inline-flex;

    align-items: center;

    min-height: 26px;

    background:
        rgba(76,175,80,0.15);

    border:
        1px solid
        rgba(102,187,106,0.30);

    color: #E8F5E9;

    padding:
        5px 10px;

    border-radius: 999px;

    font-size: 11px;

    font-weight: 750;

    line-height: 1.3;
}


/* ==========================================================
   HERO DASHBOARD
========================================================== */

.hero-visual {

    display: flex;

    justify-content: center;

    align-items: center;

    padding-top: 8px;

    padding-bottom: 4px;

    overflow: visible;
}


.dashboard-card {

    width: 100%;

    max-width: 310px;

    box-sizing: border-box;

    background:
        rgba(4,11,17,0.62);

    border:
        1px solid
        rgba(102,187,106,0.22);

    border-radius: 15px;

    padding:
        14px 14px
        12px 14px;

    overflow: visible;
}


.dashboard-title {

    display: block;

    color: #F8FAFC;

    font-weight: 750;

    font-size: 12px;

    line-height: 1.4;

    margin-top: 1px;

    margin-bottom: 4px;

    padding-top: 2px;

    overflow: visible;
}


.dashboard-subtitle {

    color: #9AABBB;

    font-size: 10px;

    line-height: 1.4;

    margin-bottom: 10px;
}


.bar-row {

    display: grid;

    grid-template-columns:
        50px 1fr;

    align-items: center;

    gap: 8px;

    margin: 7px 0;
}


.bar-name {

    color: #CFD8DC;

    font-size: 10px;

    line-height: 1.4;
}


.bar-bg {

    height: 7px;

    background:
        rgba(255,255,255,0.07);

    border-radius: 10px;

    overflow: hidden;
}


.bar-green {

    width: 82%;

    height: 100%;

    background:
        linear-gradient(
            90deg,
            #43A047,
            #81C784
        );
}


.bar-blue {

    width: 63%;

    height: 100%;

    background:
        linear-gradient(
            90deg,
            #1E88E5,
            #64B5F6
        );
}


.bar-orange {

    width: 47%;

    height: 100%;

    background:
        linear-gradient(
            90deg,
            #FB8C00,
            #FFCC80
        );
}


.dashboard-note {

    margin-top: 10px;

    padding-top: 2px;

    color: #A7F3D0;

    font-size: 10px;

    font-weight: 650;

    line-height: 1.4;

    text-align: center;
}


/* ==========================================================
   INPUT FORM
========================================================== */

[data-testid="stForm"] {

    background:
        rgba(255,255,255,0.018);

    border:
        1px solid
        rgba(148,163,184,0.28)
        !important;

    border-radius:
        15px !important;

    padding:
        0.85rem 0.95rem
        !important;

    overflow: visible !important;
}


/* Inputs */

[data-baseweb="input"] {

    background-color:
        #151D27 !important;

    overflow: visible !important;
}


[data-baseweb="input"] input {

    min-height:
        37px !important;

    color:
        #FFFFFF !important;

    font-size:
        0.88rem !important;

    font-weight:
        650 !important;

    line-height:
        1.4 !important;
}


/* Number input buttons */

[data-testid="stNumberInput"] button {

    min-height:
        37px !important;
}


/* ==========================================================
   PRIMARY BUTTON
========================================================== */

button[kind="primary"] {

    min-height:
        44px !important;

    border-radius:
        9px !important;

    border:
        none !important;

    color:
        #FFFFFF !important;

    font-size:
        0.90rem !important;

    font-weight:
        750 !important;

    line-height:
        1.35 !important;

    background:
        linear-gradient(
            90deg,
            #388E3C,
            #66BB6A
        ) !important;
}


button[kind="primary"]:hover {

    background:
        linear-gradient(
            90deg,
            #2E7D32,
            #4CAF50
        ) !important;
}


/* ==========================================================
   CONTAINERS
========================================================== */

[data-testid="stVerticalBlockBorderWrapper"] {

    border-radius:
        14px !important;

    overflow:
        visible !important;
}


/* ==========================================================
   METRICS
========================================================== */

[data-testid="stMetric"] {

    background:
        rgba(255,255,255,0.02);

    border:
        1px solid
        rgba(76,175,80,0.25);

    border-radius:
        12px;

    padding:
        0.65rem 0.75rem
        !important;

    overflow:
        visible !important;
}


[data-testid="stMetricLabel"] p {

    color:
        #CBD5E1 !important;

    line-height:
        1.35 !important;
}


[data-testid="stMetricValue"] {

    color:
        #FFFFFF !important;

    font-size:
        1.65rem !important;

    line-height:
        1.3 !important;

    overflow:
        visible !important;
}


/* ==========================================================
   PROGRESS BARS
========================================================== */

[data-testid="stProgress"] {

    margin-top:
        -2px !important;

    margin-bottom:
        3px !important;
}


[data-testid="stProgress"] > div > div > div > div {

    background:
        linear-gradient(
            90deg,
            #43A047,
            #81C784
        ) !important;
}


/* ==========================================================
   ALERTS
========================================================== */

[data-testid="stAlert"] {

    border-radius:
        10px !important;

    padding:
        0.60rem 0.75rem
        !important;

    overflow:
        visible !important;
}


[data-testid="stAlert"] p {

    line-height:
        1.4 !important;
}


/* ==========================================================
   DIVIDERS
========================================================== */

hr {

    margin-top:
        0.55rem !important;

    margin-bottom:
        0.55rem !important;

    border-color:
        rgba(
            148,
            163,
            184,
            0.20
        ) !important;
}


/* ==========================================================
   EXPANDERS
========================================================== */

[data-testid="stExpander"] {

    border-radius:
        10px !important;

    overflow:
        visible !important;
}


/* ==========================================================
   RESPONSIVE
========================================================== */

@media (max-width: 900px) {

    .hero-grid {

        grid-template-columns:
            1fr;
    }


    .hero-visual {

        display:
            none;
    }


    .hero-title {

        font-size:
            25px;
    }

}

</style>
"""
)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = (
    "final_crop_recommendation_ann.keras"
)

SCALER_FILE = (
    "crop_scaler.pkl"
)

ENCODER_FILE = (
    "crop_label_encoder.pkl"
)


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
    if not os.path.exists(
        file
    )
]


if missing_files:

    st.error(
        "Missing required files: "
        + ", ".join(
            missing_files
        )
    )

    st.stop()


# ============================================================
# LOAD SYSTEM
# ============================================================

@st.cache_resource
def load_system():

    model = (
        tf.keras.models.load_model(
            MODEL_FILE,
            compile=False
        )
    )


    scaler = (
        joblib.load(
            SCALER_FILE
        )
    )


    label_encoder = (
        joblib.load(
            ENCODER_FILE
        )
    )


    return (
        model,
        scaler,
        label_encoder
    )


model, scaler, label_encoder = (
    load_system()
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_made" not in st.session_state:

    st.session_state.prediction_made = (
        False
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
<div class="hero-banner">

    <div class="hero-grid">

        <div>

            <div class="hero-title">
                🌾 Smart Crop Recommendation System
            </div>

            <div class="hero-subtitle">
                Artificial Neural Network based agricultural
                decision-support system for recommending a
                suitable crop using soil nutrients and climatic
                conditions.
            </div>

            <div class="hero-badges">

                <div class="hero-badge">
                    🌱 7 Input Features
                </div>

                <div class="hero-badge">
                    🧠 ANN Model
                </div>

                <div class="hero-badge">
                    🌾 22 Crop Classes
                </div>

                <div class="hero-badge">
                    ⚡ Interactive Prediction
                </div>

            </div>

        </div>


        <div class="hero-visual">

            <div class="dashboard-card">

                <div class="dashboard-title">
                    🌱 ANN Prediction Dashboard
                </div>

                <div class="dashboard-subtitle">
                    Environmental suitability overview
                </div>


                <div class="bar-row">

                    <div class="bar-name">
                        Soil
                    </div>

                    <div class="bar-bg">
                        <div class="bar-green"></div>
                    </div>

                </div>


                <div class="bar-row">

                    <div class="bar-name">
                        Climate
                    </div>

                    <div class="bar-bg">
                        <div class="bar-blue"></div>
                    </div>

                </div>


                <div class="bar-row">

                    <div class="bar-name">
                        Rainfall
                    </div>

                    <div class="bar-bg">
                        <div class="bar-orange"></div>
                    </div>

                </div>


                <div class="dashboard-note">
                    ANN-based crop suitability analysis
                </div>

            </div>

        </div>

    </div>

</div>
"""
)


# ============================================================
# MAIN LAYOUT
# ============================================================

input_column, result_column = (
    st.columns(
        [
            1.15,
            0.85
        ],
        gap="large"
    )
)


# ============================================================
# LEFT COLUMN
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
            "### 🧪 Soil Nutrients"
        )


        n_col, p_col, k_col = (
            st.columns(
                3
            )
        )


        with n_col:

            nitrogen = (
                st.number_input(
                    "N | 0–140 kg/ha",
                    min_value=0.0,
                    max_value=140.0,
                    value=70.0,
                    step=1.0
                )
            )


        with p_col:

            phosphorus = (
                st.number_input(
                    "P | 5–145 kg/ha",
                    min_value=5.0,
                    max_value=145.0,
                    value=50.0,
                    step=1.0
                )
            )


        with k_col:

            potassium = (
                st.number_input(
                    "K | 5–205 kg/ha",
                    min_value=5.0,
                    max_value=205.0,
                    value=50.0,
                    step=1.0
                )
            )


        # ----------------------------------------------------
        # CLIMATE
        # ----------------------------------------------------

        st.markdown(
            "### 🌦️ Climate"
        )


        temp_col, humidity_col = (
            st.columns(
                2
            )
        )


        with temp_col:

            temperature = (
                st.number_input(
                    "Temperature | 8.83–43.68 °C",
                    min_value=8.83,
                    max_value=43.68,
                    value=25.00,
                    step=0.10,
                    format="%.2f"
                )
            )


        with humidity_col:

            humidity = (
                st.number_input(
                    "Humidity | 14.26–99.98 %",
                    min_value=14.26,
                    max_value=99.98,
                    value=70.00,
                    step=0.10,
                    format="%.2f"
                )
            )


        # ----------------------------------------------------
        # SOIL AND RAINFALL
        # ----------------------------------------------------

        st.markdown(
            "### 🌍 Soil & Rainfall"
        )


        ph_col, rainfall_col = (
            st.columns(
                2
            )
        )


        with ph_col:

            ph = (
                st.number_input(
                    "Soil pH | 3.50–9.94",
                    min_value=3.50,
                    max_value=9.94,
                    value=6.50,
                    step=0.01,
                    format="%.2f"
                )
            )


        with rainfall_col:

            rainfall = (
                st.number_input(
                    "Rainfall | 20.21–298.56 mm",
                    min_value=20.21,
                    max_value=298.56,
                    value=100.00,
                    step=0.10,
                    format="%.2f"
                )
            )


        submitted = (
            st.form_submit_button(
                "🌾 Analyse & Recommend Crop",
                type="primary",
                width="stretch"
            )
        )


    st.caption(
        "ℹ️ Use values only within the "
        "displayed training ranges."
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    input_data = (
        pd.DataFrame(
            {
                "N": [
                    nitrogen
                ],

                "P": [
                    phosphorus
                ],

                "K": [
                    potassium
                ],

                "temperature": [
                    temperature
                ],

                "humidity": [
                    humidity
                ],

                "ph": [
                    ph
                ],

                "rainfall": [
                    rainfall
                ]
            }
        )
    )


    input_scaled = (
        scaler.transform(
            input_data
        )
    )


    probabilities = (
        model.predict(
            input_scaled,
            verbose=0
        )[0]
    )


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    predicted_crop = (
        label_encoder.inverse_transform(
            [
                predicted_index
            ]
        )[0]
    )


    confidence = float(
        probabilities[
            predicted_index
        ] * 100
    )


    top3_indices = (
        np.argsort(
            probabilities
        )[-3:][::-1]
    )


    st.session_state.prediction_made = (
        True
    )


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
# RIGHT COLUMN
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
                "on the left and select "
                "**Analyse & Recommend Crop**."
            )


            st.info(
                "The trained ANN predicts one "
                "of 22 crop classes."
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


        last_inputs = (
            st.session_state.last_inputs
        )


        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        with st.container(
            border=True
        ):

            crop_col, confidence_col = (
                st.columns(
                    [
                        1.5,
                        1
                    ]
                )
            )


            with crop_col:

                st.caption(
                    "🌱 RECOMMENDED CROP"
                )

                st.header(
                    predicted_crop.title()
                )


            with confidence_col:

                st.metric(
                    "Model Confidence",
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
            "### 📊 Top 3 Predictions"
        )


        medals = [
            "🥇",
            "🥈",
            "🥉"
        ]


        for rank, index in enumerate(
            top3_indices
        ):

            crop_name = (
                label_encoder.inverse_transform(
                    [
                        int(
                            index
                        )
                    ]
                )[0]
            )


            probability = float(
                probabilities[
                    index
                ]
            )


            percentage = (
                probability * 100
            )


            name_col, percent_col = (
                st.columns(
                    [
                        3,
                        1
                    ]
                )
            )


            with name_col:

                st.write(
                    f"{medals[rank]} "
                    f"**{crop_name.title()}**"
                )


            with percent_col:

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
        # INPUT SUMMARY
        # ----------------------------------------------------

        with st.expander(
            "📋 Input Summary"
        ):

            st.write(
                f"""
**Nitrogen:** {last_inputs["nitrogen"]:.0f} kg/ha  
**Phosphorus:** {last_inputs["phosphorus"]:.0f} kg/ha  
**Potassium:** {last_inputs["potassium"]:.0f} kg/ha  

**Temperature:** {last_inputs["temperature"]:.2f} °C  
**Humidity:** {last_inputs["humidity"]:.2f} %  

**Soil pH:** {last_inputs["ph"]:.2f}  
**Rainfall:** {last_inputs["rainfall"]:.2f} mm
"""
            )


# ============================================================
# BOTTOM
# ============================================================

st.divider()


bottom_left, bottom_right = (
    st.columns(
        [
            2,
            1
        ]
    )
)


with bottom_left:

    st.caption(
        "⚠️ Academic ANN prototype. "
        "Predictions should not replace "
        "professional agricultural advice "
        "or field assessment."
    )


with bottom_right:

    with st.expander(
        "🧠 Model Info"
    ):

        st.write(
            """
**ANN Architecture**

7 Inputs → 32 Neurons → 16 Neurons → 22 Outputs

**Hidden Activation:** ReLU

**Output Activation:** Softmax

**Output:** Recommended crop from 22 crop classes
"""
        )
