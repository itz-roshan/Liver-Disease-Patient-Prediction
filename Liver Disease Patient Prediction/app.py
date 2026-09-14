import streamlit as st
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LiverCare AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    return joblib.load(model_path)


try:

    model = load_model()

except Exception as e:

    st.error("❌ Unable to load model.pkl")

    st.warning(
        "Make sure model.pkl is present in the same "
        "folder as app.py."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🩺 LiverCare AI")

st.subheader(
    "Liver Disease Prediction System"
)

st.write(
    "Enter the patient's information and liver function "
    "test values to generate a prediction."
)

st.divider()


# ============================================================
# PROJECT INFORMATION
# ============================================================

info1, info2, info3 = st.columns(3)


with info1:

    st.info(
        """
        👤 **Patient Information**

        Enter age and gender.
        """
    )


with info2:

    st.info(
        """
        🧪 **Laboratory Tests**

        Enter liver function test values.
        """
    )


with info3:

    st.info(
        """
        🤖 **AI Prediction**

        Get prediction from the trained ANN model.
        """
    )


st.write("")


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("👤 Patient Information")

st.caption(
    "Enter the patient's basic demographic information."
)


patient_col1, patient_col2 = st.columns(2)


with patient_col1:

    age = st.number_input(
        "Age",
        min_value=4,
        max_value=90,
        value=45,
        step=1
    )


with patient_col2:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )


st.divider()


# ============================================================
# LIVER FUNCTION TEST
# ============================================================

st.header("🧪 Liver Function Test")

st.caption(
    "Enter all laboratory values used by the trained model."
)


# ------------------------------------------------------------
# ROW 1
# ------------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    total_bilirubin = st.number_input(
        "Total Bilirubin",
        min_value=0.4,
        max_value=75.0,
        value=1.00,
        step=0.10,
        format="%.2f"
    )


with col2:

    direct_bilirubin = st.number_input(
        "Direct Bilirubin",
        min_value=0.1,
        max_value=19.7,
        value=0.30,
        step=0.10,
        format="%.2f"
    )


# ------------------------------------------------------------
# ROW 2
# ------------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    alkaline_phosphotase = st.number_input(
        "Alkaline Phosphotase",
        min_value=63,
        max_value=2110,
        value=208,
        step=1
    )


with col2:

    alamine = st.number_input(
        "Alamine Aminotransferase",
        min_value=10,
        max_value=2000,
        value=35,
        step=1
    )


# ------------------------------------------------------------
# ROW 3
# ------------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    aspartate = st.number_input(
        "Aspartate Aminotransferase",
        min_value=10,
        max_value=4929,
        value=42,
        step=1
    )


with col2:

    total_proteins = st.number_input(
        "Total Proteins",
        min_value=2.7,
        max_value=9.6,
        value=6.60,
        step=0.10,
        format="%.2f"
    )


# ------------------------------------------------------------
# ROW 4
# ------------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    albumin = st.number_input(
        "Albumin",
        min_value=0.9,
        max_value=5.5,
        value=3.10,
        step=0.10,
        format="%.2f"
    )


with col2:

    ag_ratio = st.number_input(
        "Albumin and Globulin Ratio",
        min_value=0.3,
        max_value=2.8,
        value=0.93,
        step=0.01,
        format="%.2f"
    )


st.write("")


# ============================================================
# BUTTONS
# ============================================================

button_col1, button_col2, button_col3 = st.columns(
    [1, 2, 1]
)


with button_col2:

    predict_button = st.button(
        "🔍  Analyze Patient",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # GENDER ENCODING
    #
    # Notebook:
    # Male   = 0
    # Female = 1
    # --------------------------------------------------------

    if gender == "Male":

        gender_value = 0

    else:

        gender_value = 1


    # --------------------------------------------------------
    # CREATE MODEL INPUT
    #
    # EXACT FEATURE ORDER
    # --------------------------------------------------------

    input_data = np.array(
        [[
            age,
            gender_value,
            total_bilirubin,
            direct_bilirubin,
            alkaline_phosphotase,
            alamine,
            aspartate,
            total_proteins,
            albumin,
            ag_ratio
        ]],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # SHOW INPUT SHAPE
    # --------------------------------------------------------

    if input_data.shape != (1, 10):

        st.error(
            "❌ Input format error. "
            "Model requires exactly 10 features."
        )

        st.stop()


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        with st.spinner(
            "🤖 Analyzing patient data..."
        ):

            prediction = model.predict(
                input_data,
                verbose=0
            )


        # Convert prediction to scalar

        model_output = float(
            np.asarray(prediction).reshape(-1)[0]
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(str(e))

        st.stop()


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header("📊 Prediction Result")


    # --------------------------------------------------------
    # DECISION
    #
    # Existing notebook model uses linear output.
    # For this application we use 0.50 as decision threshold.
    # --------------------------------------------------------

    threshold = 0.50


    if model_output >= threshold:

        disease_detected = True

    else:

        disease_detected = False


    # --------------------------------------------------------
    # RESULT MESSAGE
    # --------------------------------------------------------

    if disease_detected:

        st.error(
            """
            ## ⚠️ Liver Disease Indicated

            The model output is above the 0.50
            decision threshold.
            """
        )

    else:

        st.success(
            """
            ## ✅ Lower Risk Indicated

            The model output is below the 0.50
            decision threshold.
            """
        )


    st.write("")


    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    st.subheader("Prediction Summary")


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Model Output",
            f"{model_output:.3f}"
        )


    with result_col2:

        st.metric(
            "Decision Threshold",
            f"{threshold:.3f}"
        )


    with result_col3:

        if disease_detected:

            st.metric(
                "Prediction",
                "Disease Indicated"
            )

        else:

            st.metric(
                "Prediction",
                "Lower Risk"
            )


    # ========================================================
    # PATIENT SUMMARY
    # ========================================================

    st.write("")

    with st.expander(
        "📋 View Patient Information"
    ):

        st.subheader("Patient")

        p1, p2 = st.columns(2)


        with p1:

            st.write(
                f"**Age:** {age}"
            )


        with p2:

            st.write(
                f"**Gender:** {gender}"
            )


        st.divider()

        st.subheader("Laboratory Values")


        l1, l2 = st.columns(2)


        with l1:

            st.write(
                f"**Total Bilirubin:** "
                f"{total_bilirubin:.2f}"
            )

            st.write(
                f"**Alkaline Phosphotase:** "
                f"{alkaline_phosphotase}"
            )

            st.write(
                f"**Aspartate Aminotransferase:** "
                f"{aspartate}"
            )

            st.write(
                f"**Albumin:** "
                f"{albumin:.2f}"
            )


        with l2:

            st.write(
                f"**Direct Bilirubin:** "
                f"{direct_bilirubin:.2f}"
            )

            st.write(
                f"**Alamine Aminotransferase:** "
                f"{alamine}"
            )

            st.write(
                f"**Total Proteins:** "
                f"{total_proteins:.2f}"
            )

            st.write(
                f"**Albumin and Globulin Ratio:** "
                f"{ag_ratio:.2f}"
            )


    # ========================================================
    # MODEL INPUT PREVIEW
    # ========================================================

    with st.expander(
        "🔧 View Model Input"
    ):

        st.write(
            "The following 10 values are sent to the model:"
        )

        st.write(input_data)

        st.caption(
            "Feature order: Age, Gender, Total Bilirubin, "
            "Direct Bilirubin, Alkaline Phosphotase, "
            "Alamine Aminotransferase, "
            "Aspartate Aminotransferase, Total Proteins, "
            "Albumin, Albumin and Globulin Ratio."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🩺 LiverCare AI • Artificial Neural Network Project"
)

st.caption(
    "For educational and demonstration purposes only. "
    "This application is not a medical diagnosis tool."
)

# streamlit run app.py
