import streamlit as st
import joblib
import numpy as np

#  Loading model
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# setting Page Configuration
st.set_page_config(
    page_title="Breast Cancer Diagnosis",
    page_icon="cancer.png",
    layout="wide"
)

# building ui for streamlit
st.image("cancer.png", width=100)
st.title("Breast Cancer Diagnosis 🩺")

st.write(
    "Predict whether a breast tumor is Benign or Malignant using Machine Learning."
)

st.metric("Model Accuracy", "97.37%")

with st.expander("About Dataset"):
    st.write("""
    The Wisconsin Breast Cancer Dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 
    It describes characteristics of the cell nuclei present in the image to help determine whether a tumor is malignant or benign.
    """)

with st.expander("About Features"):
    st.write("""
    The 30 features include mean, standard error (SE), and worst (largest) values for:
    - **Radius**: Distances from center to points on the perimeter
    - **Texture**: Standard deviation of gray-scale values
    - **Perimeter**: Perimeter of the core tumor
    - **Area**: Area of the core tumor
    - **Smoothness**: Local variation in radius lengths
    - **Compactness**: Perimeter^2 / area - 1.0
    - **Concavity**: Severity of concave portions of the contour
    - **Concave Points**: Number of concave portions of the contour
    - **Symmetry**: Symmetry of the cell nuclei
    - **Fractal Dimension**: "Coastline approximation" - 1
    """)

# Step 4: Sidebar
st.sidebar.title("About")
st.sidebar.info("""
**Dataset:**
Wisconsin Breast Cancer Dataset

**Model:**
Support Vector Machine

**Author:**
Suraj Bhatt
""")

# Step 5: User Input using 3 Columns
st.subheader("Tumor Features Input")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Mean Features**")
    radius_mean = st.number_input("Radius Mean", value=10.0)
    texture_mean = st.number_input("Texture Mean", value=15.0)
    perimeter_mean = st.number_input("Perimeter Mean", value=65.0)
    area_mean = st.number_input("Area Mean", value=300.0)
    smoothness_mean = st.number_input("Smoothness Mean", value=0.1)
    compactness_mean = st.number_input("Compactness Mean", value=0.1)
    concavity_mean = st.number_input("Concavity Mean", value=0.1)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.05)
    symmetry_mean = st.number_input("Symmetry Mean", value=0.2)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.06)

with col2:
    st.markdown("**Standard Error Features**")
    radius_se = st.number_input("Radius SE", value=0.5)
    texture_se = st.number_input("Texture SE", value=1.0)
    perimeter_se = st.number_input("Perimeter SE", value=3.0)
    area_se = st.number_input("Area SE", value=25.0)
    smoothness_se = st.number_input("Smoothness SE", value=0.01)
    compactness_se = st.number_input("Compactness SE", value=0.02)
    concavity_se = st.number_input("Concavity SE", value=0.03)
    concave_points_se = st.number_input("Concave Points SE", value=0.01)
    symmetry_se = st.number_input("Symmetry SE", value=0.02)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.003)

with col3:
    st.markdown("**Worst Features**")
    radius_worst = st.number_input("Radius Worst", value=15.0)
    texture_worst = st.number_input("Texture Worst", value=20.0)
    perimeter_worst = st.number_input("Perimeter Worst", value=100.0)
    area_worst = st.number_input("Area Worst", value=700.0)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.15)
    compactness_worst = st.number_input("Compactness Worst", value=0.25)
    concavity_worst = st.number_input("Concavity Worst", value=0.3)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.1)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.3)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.08)


# Step 6: Prediction Button
if st.button("Predict", type="primary"):

    # Exactly 30 features, in the same order as the training dataset
    features = np.array([[
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]])

    # Scaling
    features = scaler.transform(features)

    # prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features).max()

    st.divider()
    
    # Displaying Result
    st.subheader("Diagnosis Result")
    if prediction == 1:
        st.error("🟥 Malignant")
    else:
        st.success("🟩 Benign")

    # Confidence
    st.write(f"**Confidence:** {probability*100:.2f}%")
