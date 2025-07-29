import streamlit as st
import pickle
import os
import numpy as np

# Build path to classifier.pkl alongside this script
model_fp = os.path.join(os.path.dirname(__file__), "classifier.pkl")

# Load the classifier
with open(model_fp, "rb") as f:
    model = pickle.load(f)

st.title("Breast Cancer Classifier")
st.write("Enter the cell nucleus measurements to classify the cancer")

mean_radius            = st.slider("Mean Radius",               7.0,    28.0,    14.0,    step=0.01)
mean_texture           = st.slider("Mean Texture",              10.0,   39.0,    19.0,    step=0.01)
mean_perimeter         = st.slider("Mean Perimeter",            44.0,   188.0,   100.0,   step=0.1)
mean_area              = st.slider("Mean Area",                 144,    2501,    1000,    step=1)
mean_smoothness        = st.slider("Mean Smoothness",           0.05,   0.16,    0.10,    step=0.0001)
mean_compactness       = st.slider("Mean Compactness",          0.02,   0.35,    0.10,    step=0.001)
mean_concavity         = st.slider("Mean Concavity",            0.00,   0.43,    0.10,    step=0.001)
mean_concave_points    = st.slider("Mean Concave Points",       0.00,   0.20,    0.05,    step=0.001)
mean_symmetry          = st.slider("Mean Symmetry",             0.11,   0.30,    0.20,    step=0.001)
mean_fractal_dimension = st.slider("Mean Fractal Dimension",    0.05,   0.10,    0.06,    step=0.0001)

se_radius            = st.slider("Radius SE",                  0.11,   2.87,    1.00,    step=0.001)
se_texture           = st.slider("Texture SE",                 0.36,   4.88,    2.00,    step=0.01)
se_perimeter         = st.slider("Perimeter SE",               0.76,   21.98,   10.00,   step=0.1)
se_area              = st.slider("Area SE",                    6.80,   542.20,  200.00,  step=0.1)
se_smoothness        = st.slider("Smoothness SE",              0.002,  0.031,   0.01,    step=0.0001)
se_compactness       = st.slider("Compactness SE",             0.002,  0.14,    0.05,    step=0.001)
se_concavity         = st.slider("Concavity SE",               0.00,   0.40,    0.10,    step=0.001)
se_concave_points    = st.slider("Concave Points SE",          0.00,   0.05,    0.01,    step=0.001)
se_symmetry          = st.slider("Symmetry SE",                0.01,   0.08,    0.04,    step=0.001)
se_fractal_dimension = st.slider("Fractal Dimension SE",       0.001,  0.03,    0.01,    step=0.0001)

worst_radius            = st.slider("Worst Radius",               8.0,    36.0,    20.0,    step=0.01)
worst_texture           = st.slider("Worst Texture",              12.0,   50.0,    25.0,    step=0.01)
worst_perimeter         = st.slider("Worst Perimeter",            50.0,   251.0,   125.0,   step=0.1)
worst_area              = st.slider("Worst Area",                 185,    4254,    2000,    step=1)
worst_smoothness        = st.slider("Worst Smoothness",           0.07,   0.22,    0.15,    step=0.0001)
worst_compactness       = st.slider("Worst Compactness",          0.03,   1.06,    0.50,    step=0.001)
worst_concavity         = st.slider("Worst Concavity",            0.00,   1.25,    0.50,    step=0.001)
worst_concave_points    = st.slider("Worst Concave Points",       0.00,   0.29,    0.15,    step=0.001)
worst_symmetry          = st.slider("Worst Symmetry",             0.16,   0.66,    0.40,    step=0.001)
worst_fractal_dimension = st.slider("Worst Fractal Dimension",    0.06,   0.21,    0.12,    step=0.0001)

if st.button("Predict"):
    features = np.array([[  mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                           mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
                           mean_fractal_dimension, se_radius, se_texture, se_perimeter, se_area,
                           se_smoothness, se_compactness, se_concavity, se_concave_points,
                           se_symmetry, se_fractal_dimension, worst_radius, worst_texture,
                           worst_perimeter, worst_area, worst_smoothness, worst_compactness,
                           worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension]] )
    prediction = model.predict(features)[0]

    if prediction == "M":
        st.write("Malignant")
    elif prediction == "B":
        st.write("Benign")

