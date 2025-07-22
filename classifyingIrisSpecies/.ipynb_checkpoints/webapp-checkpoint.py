import streamlit as st
import pickle
import numpy as np
import os

HERE = os.path.dirname(__file__)  
MODEL_FP = os.path.join(HERE, "classifier.pkl")

with open(MODEL_FP, "rb") as model_file:
    model = pickle.load(model_file)

st.title("Iris Species Classifier")
st.write("Enter the flower measurements to classify the species.")

sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, step=0.1)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    if (prediction[0]==0):
        st.write(f"Predicted Iris Species: {"Iris-setosa"}")
    elif(prediction[0]==1):
        st.write(f"Predicted Iris Species: {"Iris-versicolor"}")
    elif(prediction[0]==2):
        st.write(f"Predicted Iris Species: {"Iris-verginica"}")