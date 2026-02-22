import streamlit as st
import joblib
import pandas as pd

model = joblib.load("dtc_model.pkl")

st.title("wine type prediction !")

fixed_acidity = st.number_input("fixed_acidity")
volatile_acidity = st.number_input("volatile_acidity")
citric_acid = st.number_input("citric_acid")
residual_sugar = st.number_input("residual_sugar")
chlorides = st.number_input("chlorides")
free_sulfur_dioxide = st.number_input("free_sulfur_dioxide")
total_sulfur_dioxide = st.number_input("total_sulfur_dioxide")
density = st.number_input("density")
pH = st.number_input("pH")
sulphates = st.number_input("sulphates")
alcohol = st.number_input("alcohol")
quality = st.number_input("quality")

input_data = pd.DataFrame([[
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    pH,
    sulphates,
    alcohol,
    quality
]],columns=model.features_names_in_)

if st.button("predict"):
  prediction = model.predict(input_data)
  if prediction == "white":
    st.success("The wine is white wine")
  else:
    st.error("The wine is red wine")
