import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("flight_delay_model.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Delay Prediction Dashboard")
st.write("Enter flight details to estimate the likelihood of a delay.")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Model Information")
    st.write("**Model:** Logistic Regression")
    st.write("**Dataset:** Synthetic (200 flights)")
    st.write("**Purpose:** Educational demo")
    st.write("**Note:** Predictions are probabilistic, not guarantees.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Feature Importance", "Responsible AI"])

# -----------------------------
# Tab 1: Prediction
# -----------------------------
with tab1:

    st.subheader("Flight Details")

    airlines = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"]
    airline = st.selectbox("Airline", airlines)

    airports = ["JFK", "LAX", "ORD", "ATL", "DFW", "MIA", "SFO", "SEA"]
    origin = st.selectbox("Origin Airport", airports)
    destination = st.selectbox("Destination Airport", airports)

    departure_time = st.number_input("Departure Time (0–2359)", min_value=0, max_value=2359, value=900)
    distance = st.number_input("Distance (miles)", min_value=1, max_value=5000, value=800)

    if st.button("Predict Delay"):
        features = np.array([[departure_time, distance]])
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][prediction]

        # Display prediction
        if prediction == 1:
            st.error(f"Prediction: **Delayed**")
        else:
            st.success(f"Prediction: **On Time**")

        # Confidence metric
        st.metric("Prediction Confidence", f"{confidence*100:.1f}%")

        # Confidence explanation
        if confidence < 0.55:
            st.warning("Low confidence — the model is uncertain due to mixed signals.")
        elif confidence < 0.75:
            st.info("Moderate confidence — the model sees some delay patterns.")
        else:
            st.success("High confidence — the model strongly recognizes this pattern.")

        # Explanation text
        st.write("---")
        st.write("### Why this prediction?")
        st.write("""
        - **Departure Time** and **Distance** influence delay likelihood  
        - Evening flights and long-distance flights tend to have higher delay risk  
        - The model uses logistic regression to estimate probability  
        """)

# -----------------------------
# Tab 2: Feature Importance
# -----------------------------
with tab2:
    st.subheader("Feature Importance (Logistic Regression Coefficients)")

    coef = model.coef_[0]
    features = ["Departure Time", "Distance"]

    fig, ax = plt.subplots()
    ax.bar(features, coef, color=["#4A90E2", "#50E3C2"])
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Feature Influence on Delay Probability")
    st.pyplot(fig)

# -----------------------------
# Tab 3: Responsible AI
# -----------------------------
with tab3:
    st.subheader("Responsible AI Notes")
    st.write("""
    - This model is trained on synthetic data for demonstration purposes  
    - Predictions should not be used for real-world flight decisions  
    - Real delays depend on weather, maintenance, staffing, and air traffic  
    - Always consider real-time airline and airport information  
    """)
