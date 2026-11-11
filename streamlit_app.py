from pathlib import Path

import mlflow
import mlflow.sklearn
import streamlit as st


MODEL_PATH = Path(__file__).parent / "model"


@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model(str(MODEL_PATH))


st.set_page_config(
    page_title="California Housing Estimator",
    page_icon="🏡",
    layout="centered",
)

st.title("California Housing Estimator")
st.write(
    "Enter details for a California district to estimate its median house value."
)
st.info(
    "This model estimates a district-level median, not the value of an individual home."
)

with st.sidebar:
    st.header("About this model")
    st.write("The app uses the packaged champion Linear Regression model.")
    st.write(
        "Predictions are shown in US dollars. The model was trained to predict "
        "values in units of $100,000."
    )

with st.form("housing_features"):
    st.subheader("District details")
    left, right = st.columns(2)

    with left:
        median_income = st.number_input(
            "Median income",
            min_value=0.0,
            value=5.0,
            step=0.1,
            help="Median household income, measured in tens of thousands of US dollars.",
        )
        house_age = st.number_input(
            "Median house age (years)",
            min_value=0.0,
            value=20.0,
            step=1.0,
            help="Median age of houses in the district.",
        )
        average_rooms = st.number_input(
            "Average rooms per household",
            min_value=0.0,
            value=5.0,
            step=0.1,
        )
        average_bedrooms = st.number_input(
            "Average bedrooms per household",
            min_value=0.0,
            value=1.0,
            step=0.1,
        )

    with right:
        population = st.number_input(
            "District population",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            help="Total population of the district.",
        )
        average_occupancy = st.number_input(
            "Average occupants per household",
            min_value=0.0,
            value=3.0,
            step=0.1,
        )
        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=36.0,
            step=0.01,
            format="%.2f",
        )
        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-120.0,
            step=0.01,
            format="%.2f",
        )

    submitted = st.form_submit_button("Estimate median value", type="primary")

if submitted:
    model = load_model()
    features = [
        median_income,
        house_age,
        average_rooms,
        average_bedrooms,
        population,
        average_occupancy,
        latitude,
        longitude,
    ]
    predicted_value = float(model.predict([features])[0])
    value_in_dollars = predicted_value * 100_000

    st.subheader("Estimated median house value")
    st.metric("Model estimate", f"${value_in_dollars:,.0f}")
    st.caption(
        f"Model output: {predicted_value:.3f} units of $100,000. "
        "This is an estimate based on district-level data."
    )
