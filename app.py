
import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Student Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Pass/Fail Prediction")
st.write("Predict whether a student will pass or fail.")

st.divider()

# Load model and dataset
@st.cache_resource
def load_model():
    return joblib.load("student_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("student_learning.csv")

model = load_model()
df = load_data()

# Select features
features = df.drop(
    columns=["passed", "final_score"],
    errors="ignore"
)

# Student input form
st.subheader("Enter Student Details")

student_data = {}

with st.form("student_form"):

    for column in features.columns:

        if pd.api.types.is_numeric_dtype(features[column]):

            student_data[column] = st.number_input(
                column.replace("_", " ").title(),
                value=float(features[column].median())
            )

        else:
            options = features[column].dropna().unique().tolist()

            if options:
                student_data[column] = st.selectbox(
                    column.replace("_", " ").title(),
                    options
                )
            else:
                student_data[column] = st.text_input(column)

    submit = st.form_submit_button("Predict Result")

# Prediction
if submit:

    input_data = pd.DataFrame(
        [student_data],
        columns=features.columns
    )

    prediction = model.predict(input_data)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🎉 Student is Predicted to PASS!")
    else:
        st.error("Student is Predicted to FAIL!")

    # Show probability
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        st.write("Prediction Probability:")

        for label, probability in zip(
            model.classes_, probabilities
        ):
            st.write(f"Class {label}: {probability:.2%}")

st.divider()

st.caption(
    "This project uses synthetic classroom data. "
    "Predictions are for educational purposes only."
)
